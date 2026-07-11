"""Problem 3: inter-satellite topology, routing, and traffic engineering."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from starlink_modeling.problem2_grid import make_regular_grid  # noqa: E402
from starlink_modeling.problem2_orbit import (  # noqa: E402
    R_EARTH_KM,
    eci_to_ecef,
    generate_walker_constellation,
    orbital_period_seconds,
)
from starlink_modeling.problem3_io import (  # noqa: E402
    CONSTELLATION_SIGNATURE,
    ConstellationSelectionError,
    SelectedConstellation,
    discover_problem2_constellation,
    load_problem3_config,
    write_selected_constellation,
)
from starlink_modeling.problem3_reporting import write_json, write_method_notes  # noqa: E402
from starlink_modeling.problem3_routing import (  # noqa: E402
    evaluate_routes,
    minimum_delay_ground_route,
    visibility_matrix,
)
from starlink_modeling.problem3_topology import (  # noqa: E402
    LIGHT_SPEED_KM_S,
    TAU_PROCESSING_MS,
    TopologySnapshot,
    build_adjacency,
    build_topology_series,
    build_topology_snapshot,
    summarize_topology_series,
)
from starlink_modeling.problem3_traffic import (  # noqa: E402
    C_ACCESS_GBPS,
    PEAK_TO_MEAN_RATIO,
    TRAFFIC_DENSITY_MBPS_PER_KM2,
    AccessAllocation,
    assignment_probabilities,
    build_traffic_timeseries,
    fair_access_allocation,
    grid_cell_areas_km2,
    load_region_traffic_density,
    load_traffic_data,
    make_synthetic_demo_traffic,
    nearest_visible_baseline,
    region_traffic_metrics,
    spatial_ground_demand,
    summarize_traffic_timeseries,
)
from starlink_modeling.problem3_visualization import (  # noqa: E402
    plot_routing_figures,
    plot_topology_figures,
    plot_traffic_figures,
    set_problem3_figure_context,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "problem3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三：星间链路与通信路由优化")
    parser.add_argument("--mode", choices=("quick", "standard", "full"), default="standard")
    parser.add_argument("--stage", choices=("topology", "routing", "traffic", "all"), default="all")
    parser.add_argument("--constellation-file", type=str, default=None)
    parser.add_argument("--traffic-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-demo-traffic", action="store_true")
    parser.add_argument("--pair-mode", choices=("sampled", "exact"), default="sampled")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("problem3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        output_dir / "problem3_run.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def validate_config(config: dict[str, Any]) -> None:
    for section in ("constellation", "region", "constants", "topology", "routing", "traffic"):
        if section not in config:
            raise ValueError(f"问题三配置缺少 {section} 段。")
    positive = [
        config["topology"]["duration_orbits"],
        config["topology"]["time_step_seconds"],
        config["routing"]["duration_hours"],
        config["routing"]["time_step_seconds"],
        config["routing"]["ground_grid_step_deg"],
        config["routing"]["route_pair_count"],
        config["traffic"]["duration_hours"],
        config["traffic"]["time_step_seconds"],
    ]
    if any(float(value) <= 0.0 for value in positive):
        raise ValueError("问题三持续时间、步长、网格和样本数必须为正。")
    if config["topology"]["crosslink_matching_method"] not in {
        "cyclic_shift",
        "hungarian",
    }:
        raise ValueError("crosslink_matching_method 必须为 cyclic_shift 或 hungarian。")
    expected_constellation = config["constellation"]
    if (
        expected_constellation.get("constellation_scenario") != "double"
        or int(expected_constellation.get("M", -1)) != 40
        or int(expected_constellation.get("N", -1)) != 46
        or int(expected_constellation.get("total_satellites", -1)) != 1840
        or expected_constellation.get("constellation_signature")
        != CONSTELLATION_SIGNATURE
    ):
        raise ValueError("问题三配置必须锁定 standard_double_M40_N46_S1840。")
    region = config["region"]
    expected_region = (4.0, 53.0, 73.0, 135.0, TRAFFIC_DENSITY_MBPS_PER_KM2)
    actual_region = (
        float(region["region_lat_min_deg"]),
        float(region["region_lat_max_deg"]),
        float(region["region_lon_min_deg"]),
        float(region["region_lon_max_deg"]),
        float(region["traffic_density_mbps_per_km2"]),
    )
    if not np.allclose(actual_region, expected_region, atol=1e-12, rtol=0.0):
        raise ValueError(f"问题三目标区域或流量密度配置错误：{actual_region}")
    computed_region = region_traffic_metrics(actual_region[-1])
    for key in (
        "region_area_km2",
        "mean_traffic_mbps",
        "mean_traffic_gbps",
        "mean_traffic_tbps",
        "peak_traffic_tbps",
    ):
        if not math.isclose(
            float(region[key]), computed_region[key], rel_tol=1e-12, abs_tol=1e-9
        ):
            raise ValueError(f"配置中的 {key} 与公式计算结果不一致。")
    if not math.isclose(
        float(region["peak_to_mean_ratio"]), PEAK_TO_MEAN_RATIO, abs_tol=1e-12
    ):
        raise ValueError("峰均比必须为 1.5。")


def constellation_from_selection(selected: SelectedConstellation):
    return generate_walker_constellation(
        selected.M,
        selected.N,
        selected.inclination_deg,
        selected.phase_factor_F,
        selected.Omega0_deg,
        selected.u0_deg,
        selected.altitude_km,
    )


def print_selected(selected: SelectedConstellation) -> None:
    print(f"问题三采用的问题二 standard 星座文件：{selected.source_file}")
    print(
        "读取到的星座参数："
        f"M={selected.M}, N={selected.N}, S={selected.total_satellites}, "
        f"i={selected.inclination_deg:.8f}°, F={selected.phase_factor_F}, "
        f"Omega0={selected.Omega0_deg:.8f}°, u0={selected.u0_deg:.8f}°, "
        f"h={selected.altitude_km:.1f} km, theta={selected.theta_deg:.8f}°, "
        f"scenario={selected.scenario}, signature={selected.constellation_signature}, "
        f"validation_status={selected.validation_status}"
    )


def invalidate_stale_constellation_outputs(
    output_dir: Path,
    selected: SelectedConstellation,
    logger: logging.Logger,
) -> bool:
    """Remove cached outputs when the Problem 2 constellation input changes."""

    selected_path = output_dir / "selected_problem2_constellation.json"
    if not selected_path.exists():
        return False
    try:
        previous = json.loads(selected_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        previous = {}
    previous_key = previous.get("constellation_cache_key")
    previous_signature = previous.get("constellation_signature")
    unchanged = (
        previous_key == selected.constellation_cache_key
        and previous_signature == selected.constellation_signature
    )
    if unchanged:
        return False
    logger.warning(
        "检测到星座缓存键变化（%s -> %s），清除旧 topology/routing/traffic 输出。",
        previous_signature or "legacy_or_unknown",
        selected.constellation_signature,
    )
    for path in output_dir.iterdir():
        if path.is_file() and path.name != "problem3_run.log":
            path.unlink()
    return True


def resume_outputs_match(
    output_dir: Path,
    selected: SelectedConstellation,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    metadata_path = output_dir / "problem3_run_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("completed")
        and metadata.get("constellation_cache_key")
        == selected.constellation_cache_key
        and metadata.get("mode") == args.mode
        and metadata.get("pair_mode") == args.pair_mode
        and metadata.get("config") == config
    )


def write_satellite_mapping(constellation, output_dir: Path) -> None:
    frame = constellation.to_dataframe()
    frame.insert(0, "satellite_id", np.arange(constellation.total_satellites, dtype=int))
    frame.to_csv(output_dir / "satellite_id_mapping.csv", index=False)


def _progress_logger(logger: logging.Logger, stage: str):
    start = time.perf_counter()
    last_bucket = {-1}

    def report(completed: int, total: int) -> None:
        bucket = int(completed * 10 / max(total, 1))
        if bucket in last_bucket and completed != total:
            return
        last_bucket.clear()
        last_bucket.add(bucket)
        elapsed = time.perf_counter() - start
        remaining = elapsed / max(completed, 1) * max(0, total - completed)
        logger.info(
            "%s 进度 %d/%d（%.1f%%），预计剩余 %.1f min",
            stage,
            completed,
            total,
            100.0 * completed / max(total, 1),
            remaining / 60.0,
        )

    return report


def topology_times(config: dict[str, Any], altitude_km: float) -> np.ndarray:
    period = orbital_period_seconds(altitude_km=altitude_km)
    duration = float(config["topology"]["duration_orbits"]) * period
    step = float(config["topology"]["time_step_seconds"])
    return np.arange(0.0, duration, step)


def run_topology_stage(
    constellation,
    config: dict[str, Any],
    output_dir: Path,
    workers: int,
    logger: logging.Logger,
) -> dict[str, object]:
    times = topology_times(config, constellation.altitude_km)
    logger.info("拓扑阶段：%d 个快照，步长 %.1f s", len(times), config["topology"]["time_step_seconds"])
    snapshots = build_topology_series(
        constellation,
        times,
        max_distance_km=float(config["constants"]["D_ISL_MAX_KM"]),
        matching_method=config["topology"]["crosslink_matching_method"],
        workers=workers,
        progress=_progress_logger(logger, "拓扑阶段"),
    )
    (
        summary,
        topology_frame,
        link_frame,
        availability,
        shifts,
        degrees,
        snapshot_edges,
    ) = summarize_topology_series(constellation, snapshots)
    summary.update(
        {
            "mode": config["mode"],
            "matching_method": config["topology"]["crosslink_matching_method"],
            "D_ISL_MAX_KM": float(config["constants"]["D_ISL_MAX_KM"]),
            "orbital_period_minutes": orbital_period_seconds(
                altitude_km=constellation.altitude_km
            )
            / 60.0,
        }
    )
    write_json(output_dir / "topology_summary.json", summary)
    topology_frame.to_csv(output_dir / "topology_timeseries.csv", index=False)
    link_frame.to_csv(output_dir / "link_timeseries.csv", index=False)
    availability.to_csv(output_dir / "link_availability.csv", index=False)
    shifts.to_csv(output_dir / "crosslink_shift_timeseries.csv", index=False)
    degrees.to_csv(output_dir / "satellite_degree_timeseries.csv", index=False)
    snapshot_edges.to_csv(output_dir / "topology_snapshot_edges.csv", index=False)
    plot_topology_figures(
        constellation,
        snapshots,
        summary,
        topology_frame,
        link_frame,
        availability,
        shifts,
        output_dir,
    )
    logger.info(
        "拓扑完成：连通时间比例=%.6f，最大度=%d，平均边数=%.1f",
        summary["network_connected_time_ratio"],
        summary["maximum_degree"],
        summary["mean_edge_count"],
    )
    return summary


def _routing_snapshots(
    constellation,
    times: np.ndarray,
    config: dict[str, Any],
    workers: int,
    logger: logging.Logger,
) -> tuple[list[TopologySnapshot], list[np.ndarray]]:
    snapshots = build_topology_series(
        constellation,
        times,
        max_distance_km=float(config["constants"]["D_ISL_MAX_KM"]),
        matching_method=config["topology"]["crosslink_matching_method"],
        workers=workers,
        progress=_progress_logger(logger, "路由拓扑快照"),
    )
    positions_ecef = [
        eci_to_ecef(snapshot.positions_eci_km, snapshot.time_seconds)[0]
        for snapshot in snapshots
    ]
    return snapshots, positions_ecef


def _delay_heatmap(
    grid,
    snapshot: TopologySnapshot,
    positions_ecef: np.ndarray,
    theta_rad: float,
    maximum_destinations: int,
) -> pd.DataFrame:
    center_lat = (float(grid.latitude_deg.min()) + float(grid.latitude_deg.max())) / 2.0
    center_lon = (float(grid.longitude_deg.min()) + float(grid.longitude_deg.max())) / 2.0
    source = int(
        np.argmin(
            (grid.latitude_deg - center_lat) ** 2
            + (grid.longitude_deg - center_lon) ** 2
        )
    )
    if grid.size <= maximum_destinations:
        destinations = np.arange(grid.size, dtype=int)
    else:
        destinations = np.linspace(0, grid.size - 1, maximum_destinations, dtype=int)
    adjacency = build_adjacency(len(positions_ecef), snapshot.edges)
    rows = []
    for destination in destinations:
        if destination == source:
            continue
        result = minimum_delay_ground_route(
            adjacency,
            positions_ecef,
            grid.unit_vectors[source],
            grid.unit_vectors[destination],
            theta_rad,
            time_seconds=snapshot.time_seconds,
            source_lat=float(grid.latitude_deg[source]),
            source_lon=float(grid.longitude_deg[source]),
            destination_lat=float(grid.latitude_deg[destination]),
            destination_lon=float(grid.longitude_deg[destination]),
        )
        if result["reachable"]:
            rows.append(result)
    return pd.DataFrame(rows)


def run_routing_stage(
    constellation,
    selected: SelectedConstellation,
    config: dict[str, Any],
    output_dir: Path,
    pair_mode: str,
    seed: int,
    workers: int,
    logger: logging.Logger,
) -> dict[str, object]:
    routing = config["routing"]
    grid = make_regular_grid(float(routing["ground_grid_step_deg"]))
    times = np.arange(
        0.0,
        float(routing["duration_hours"]) * 3600.0,
        float(routing["time_step_seconds"]),
    )
    logger.info(
        "路由阶段：%d 个时间快照、%d 个地面点、%s 模式",
        len(times),
        grid.size,
        pair_mode,
    )
    snapshots, positions_ecef = _routing_snapshots(
        constellation, times, config, workers, logger
    )
    summary, time_metrics, routes, worst, distribution, access = evaluate_routes(
        grid,
        snapshots,
        positions_ecef,
        math.radians(selected.theta_deg),
        int(routing["route_pair_count"]),
        seed,
        pair_mode,
        int(routing["exact_block_size"]),
    )
    summary.update(
        {
            "mode": config["mode"],
            "ground_grid_step_deg": float(routing["ground_grid_step_deg"]),
            "ground_grid_point_count": grid.size,
            "routing_snapshot_count": len(times),
            "theta_deg": selected.theta_deg,
            "processing_delay_ms_per_satellite": TAU_PROCESSING_MS,
            "delay_threshold_ms": 30.0,
        }
    )
    write_json(output_dir / "routing_summary.json", summary)
    time_metrics.to_csv(output_dir / "routing_time_metrics.csv", index=False)
    routes.to_csv(output_dir / "route_pair_samples.csv", index=False)
    write_json(output_dir / "worst_delay_route.json", worst)
    distribution.to_csv(output_dir / "delay_distribution.csv", index=False)
    access.to_csv(output_dir / "ground_point_access_summary.csv", index=False)

    heatmap = _delay_heatmap(
        grid,
        snapshots[0],
        positions_ecef[0],
        math.radians(selected.theta_deg),
        int(routing["heatmap_max_destinations"]),
    )
    worst_time = float(worst.get("time_seconds", snapshots[0].time_seconds))
    worst_index = int(np.argmin(np.abs(times - worst_time)))
    plot_routing_figures(
        routes,
        time_metrics,
        worst,
        snapshots[worst_index],
        positions_ecef[worst_index],
        heatmap,
        output_dir,
    )
    logger.info(
        "路由完成：平均=%.4f ms，p95=%.4f ms，最大=%.4f ms，30ms比例=%.4f",
        summary.get("mean_delay_ms", math.nan),
        summary.get("p95_delay_ms", math.nan),
        summary.get("max_delay_ms", math.nan),
        summary.get("ratio_below_30ms", math.nan),
    )
    return summary


def _expected_assignment_cost_ms(
    snapshot: TopologySnapshot,
    visibility: np.ndarray,
    slant_km: np.ndarray,
    ground_weights: np.ndarray,
    seed: int,
    destination_count: int = 12,
) -> np.ndarray:
    satellite_count = visibility.shape[1]
    source = snapshot.edges["source_satellite"].to_numpy(int)
    target = snapshot.edges["target_satellite"].to_numpy(int)
    edge_delay = (
        snapshot.edges["distance_km"].to_numpy(float) / LIGHT_SPEED_KM_S * 1000.0
        + TAU_PROCESSING_MS
    )
    rows = np.concatenate((source, target))
    columns = np.concatenate((target, source))
    data = np.concatenate((edge_delay, edge_delay))
    graph = coo_matrix(
        (data, (rows, columns)), shape=(satellite_count, satellite_count)
    ).tocsr()
    satellite_delay = sparse_dijkstra(graph, directed=False, return_predecessors=False)
    rng = np.random.default_rng(int(seed))
    destination_count = min(int(destination_count), visibility.shape[0])
    destinations = rng.choice(
        visibility.shape[0],
        size=destination_count,
        replace=False,
        p=ground_weights / ground_weights.sum(),
    )
    expected_from_satellite = np.zeros(satellite_count, dtype=float)
    used = 0
    for ground in destinations:
        egress = np.flatnonzero(visibility[ground])
        if egress.size == 0:
            continue
        downlink = slant_km[ground, egress] / LIGHT_SPEED_KM_S * 1000.0
        expected_from_satellite += np.min(
            satellite_delay[:, egress] + downlink[None, :], axis=1
        )
        used += 1
    if used:
        expected_from_satellite /= used
    else:
        expected_from_satellite[:] = 1.0e6
    return slant_km / LIGHT_SPEED_KM_S * 1000.0 + TAU_PROCESSING_MS + expected_from_satellite[None, :]


def _allocation_mean_cost(allocation: AccessAllocation) -> float:
    if allocation.assignments.empty or allocation.throughput_gbps <= 0.0:
        return math.nan
    return float(
        np.average(
            allocation.assignments["assignment_cost_ms"],
            weights=allocation.assignments["assigned_gbps"],
        )
    )


def _allocation_cost_statistic(
    allocation: AccessAllocation, statistic: str
) -> float:
    if allocation.assignments.empty:
        return math.nan
    costs = allocation.assignments["assignment_cost_ms"].to_numpy(float)
    weights = allocation.assignments["assigned_gbps"].to_numpy(float)
    if statistic == "maximum":
        return float(np.max(costs))
    if statistic != "p95":
        raise ValueError(f"Unknown allocation cost statistic: {statistic}")
    order = np.argsort(costs)
    cumulative = np.cumsum(weights[order])
    threshold = 0.95 * cumulative[-1]
    return float(costs[order[np.searchsorted(cumulative, threshold, side="left")]])


def _dominant_assignments(
    allocation: AccessAllocation, ground_count: int
) -> np.ndarray:
    dominant = np.full(ground_count, -1, dtype=int)
    if allocation.assignments.empty:
        return dominant
    frame = allocation.assignments.sort_values(
        ["ground_index", "assigned_gbps", "satellite_id"],
        ascending=[True, False, True],
    ).drop_duplicates("ground_index")
    dominant[frame["ground_index"].to_numpy(int)] = frame["satellite_id"].to_numpy(int)
    return dominant


def _write_empty_traffic_outputs(output_dir: Path, reason: str) -> dict[str, object]:
    summary = {
        "status": "skipped_missing_real_data",
        "reason": reason,
        "isl_capacity_modeled": False,
        "access_capacity_gbps_per_satellite": C_ACCESS_GBPS,
    }
    write_json(output_dir / "traffic_summary.json", summary)
    empty_outputs = {
        "traffic_data_audit.csv": ["source_file", "traffic_value", "traffic_unit", "converted_average_gbps"],
        "traffic_timeseries.csv": ["time_seconds", "satellite_demand_gbps", "optimized_throughput_gbps"],
        "satellite_load_timeseries.csv": ["time_seconds", "satellite_id", "optimized_load_gbps", "utilization"],
        "grid_service_timeseries.csv": ["time_seconds", "ground_index", "demand_gbps", "optimized_served_gbps"],
        "optimized_assignment_sample.csv": ["time_seconds", "ground_index", "satellite_id", "assigned_gbps"],
        "baseline_assignment_sample.csv": ["time_seconds", "ground_index", "satellite_id", "assigned_gbps"],
        "traffic_sensitivity.csv": ["eta_sat", "service_ratio", "maximum_utilization"],
        "flow_engineering_comparison.csv": ["strategy", "mean_throughput_gbps", "blocked_ratio", "maximum_utilization"],
    }
    for name, columns in empty_outputs.items():
        pd.DataFrame(columns=columns).to_csv(output_dir / name, index=False)
    for figure in output_dir.glob("fig_problem3_*traffic*.png"):
        figure.unlink(missing_ok=True)
    for name in (
        "fig_problem3_throughput_and_demand.png",
        "fig_problem3_service_ratio_time.png",
        "fig_problem3_satellite_utilization_heatmap.png",
        "fig_problem3_max_utilization_time.png",
        "fig_problem3_baseline_vs_optimized.png",
    ):
        (output_dir / name).unlink(missing_ok=True)
    return summary


def run_traffic_stage(
    constellation,
    selected: SelectedConstellation,
    config: dict[str, Any],
    output_dir: Path,
    traffic_file: str | None,
    allow_demo: bool,
    seed: int,
    logger: logging.Logger,
) -> dict[str, object]:
    if traffic_file:
        dataset = load_traffic_data(traffic_file)
    elif allow_demo:
        if config["mode"] != "quick":
            raise ValueError("--allow-demo-traffic 只能与 --mode quick 一起使用。")
        dataset = make_synthetic_demo_traffic(
            float(config["traffic"]["demo_mean_traffic_gbps"])
        )
        logger.warning("流量阶段使用 synthetic_demo，仅用于代码验收。")
    else:
        density_file = (
            PROJECT_ROOT / "data" / "external" / "problem3_region_traffic_density.csv"
        )
        dataset = load_region_traffic_density(density_file)
        logger.info(
            "采用用户给定区域流量密度 %.5f Mbps/km^2，目标区域面积 %.4f km^2。",
            dataset.metadata["traffic_density_mbps_per_km2"],
            dataset.metadata["region_area_km2"],
        )

    traffic_config = config["traffic"]
    routing_config = config["routing"]
    grid = make_regular_grid(float(routing_config["ground_grid_step_deg"]))
    region_metrics = region_traffic_metrics(
        float(config["region"]["traffic_density_mbps_per_km2"])
    )
    cell_areas_km2 = grid_cell_areas_km2(
        grid.weights, region_metrics["region_area_km2"]
    )
    grid_mean_demands_gbps = (
        cell_areas_km2 * TRAFFIC_DENSITY_MBPS_PER_KM2 / 1000.0
    )
    if not math.isclose(
        float(grid_mean_demands_gbps.sum()),
        region_metrics["mean_traffic_gbps"],
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise RuntimeError("网格平均需求之和与目标区域平均总流量不一致。")
    times = np.arange(
        0.0,
        float(traffic_config["duration_hours"]) * 3600.0,
        float(traffic_config["time_step_seconds"]),
    )
    traffic_profile = build_traffic_timeseries(
        dataset,
        times,
        float(traffic_config["eta_sat"]),
        traffic_config["traffic_profile_mode"],
    )
    traffic_rows: list[dict[str, object]] = []
    load_rows: list[dict[str, object]] = []
    service_rows: list[dict[str, object]] = []
    optimized_samples: list[pd.DataFrame] = []
    baseline_samples: list[pd.DataFrame] = []
    optimized_dominant: list[np.ndarray] = []
    baseline_dominant: list[np.ndarray] = []
    peak_context: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    max_sample_rows = int(traffic_config["assignment_sample_max_rows"])
    progress = _progress_logger(logger, "流量阶段")

    for time_index, row in traffic_profile.iterrows():
        snapshot = build_topology_snapshot(
            constellation,
            float(row["time_seconds"]),
            max_distance_km=float(config["constants"]["D_ISL_MAX_KM"]),
            matching_method=config["topology"]["crosslink_matching_method"],
        )
        positions_ecef = eci_to_ecef(
            snapshot.positions_eci_km, snapshot.time_seconds
        )[0]
        visibility, slant = visibility_matrix(
            grid.unit_vectors, positions_ecef, math.radians(selected.theta_deg)
        )
        demands = spatial_ground_demand(
            float(row["satellite_demand_gbps"]), grid.weights
        )
        if not math.isclose(
            float(demands.sum()),
            float(row["satellite_demand_gbps"]),
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise RuntimeError("网格瞬时需求之和与区域瞬时总需求不一致。")
        costs = _expected_assignment_cost_ms(
            snapshot, visibility, slant, grid.weights, seed + time_index
        )
        optimized = fair_access_allocation(
            demands,
            visibility,
            capacity_gbps=float(config["constants"]["C_ACCESS_GBPS"]),
            assignment_cost_ms=costs,
        )
        baseline = nearest_visible_baseline(
            demands,
            visibility,
            slant,
            capacity_gbps=float(config["constants"]["C_ACCESS_GBPS"]),
            seed=seed,
        )
        if not baseline.assignments.empty:
            baseline_ground = baseline.assignments["ground_index"].to_numpy(int)
            baseline_satellite = baseline.assignments["satellite_id"].to_numpy(int)
            baseline.assignments["assignment_cost_ms"] = costs[
                baseline_ground, baseline_satellite
            ]
        optimized_dominant.append(_dominant_assignments(optimized, grid.size))
        baseline_dominant.append(_dominant_assignments(baseline, grid.size))
        visible_satellite_count = int(np.count_nonzero(np.any(visibility, axis=0)))
        active_access_satellite_count = int(
            np.count_nonzero(optimized.satellite_loads_gbps > 1e-10)
        )
        visible_access_capacity_gbps = (
            C_ACCESS_GBPS * active_access_satellite_count
        )
        traffic_rows.append(
            {
                **row.to_dict(),
                "visible_satellite_count": visible_satellite_count,
                "active_access_satellite_count": active_access_satellite_count,
                "visible_access_capacity_gbps": visible_access_capacity_gbps,
                "demand_capacity_ratio": float(row["satellite_demand_gbps"])
                / max(visible_access_capacity_gbps, 1e-12),
                "optimized_service_ratio": optimized.service_ratio,
                "optimized_throughput_gbps": optimized.throughput_gbps,
                "optimized_blocked_gbps": optimized.blocked_gbps,
                "optimized_maximum_utilization": optimized.maximum_utilization,
                "optimized_mean_utilization": float(
                    optimized.satellite_loads_gbps.mean() / C_ACCESS_GBPS
                ),
                "optimized_utilization_std": optimized.utilization_standard_deviation,
                "optimized_mean_delay_ms": _allocation_mean_cost(optimized),
                "optimized_p95_delay_ms": _allocation_cost_statistic(optimized, "p95"),
                "optimized_maximum_delay_ms": _allocation_cost_statistic(optimized, "maximum"),
                "baseline_minimum_service_ratio": baseline.service_ratio,
                "baseline_throughput_gbps": baseline.throughput_gbps,
                "baseline_blocked_gbps": baseline.blocked_gbps,
                "baseline_maximum_utilization": baseline.maximum_utilization,
                "baseline_utilization_std": baseline.utilization_standard_deviation,
                "baseline_mean_delay_ms": _allocation_mean_cost(baseline),
                "baseline_p95_delay_ms": _allocation_cost_statistic(baseline, "p95"),
                "baseline_maximum_delay_ms": _allocation_cost_statistic(baseline, "maximum"),
            }
        )
        load_rows.extend(
            {
                "time_seconds": float(row["time_seconds"]),
                "satellite_id": satellite,
                "optimized_load_gbps": optimized.satellite_loads_gbps[satellite],
                "baseline_load_gbps": baseline.satellite_loads_gbps[satellite],
                "utilization": optimized.satellite_loads_gbps[satellite] / C_ACCESS_GBPS,
            }
            for satellite in range(constellation.total_satellites)
        )
        service_rows.extend(
            {
                "time_seconds": float(row["time_seconds"]),
                "ground_index": ground,
                "latitude_deg": float(grid.latitude_deg[ground]),
                "longitude_deg": float(grid.longitude_deg[ground]),
                "grid_area_km2": cell_areas_km2[ground],
                "mean_demand_gbps": grid_mean_demands_gbps[ground],
                "demand_gbps": demands[ground],
                "optimized_served_gbps": optimized.ground_served_gbps[ground],
                "baseline_served_gbps": baseline.ground_served_gbps[ground],
                "visible_satellite_count": int(np.count_nonzero(visibility[ground])),
            }
            for ground in range(grid.size)
        )
        for allocation, collection in (
            (optimized, optimized_samples),
            (baseline, baseline_samples),
        ):
            sample = allocation.assignments.head(
                max(1, max_sample_rows // max(len(times), 1))
            ).copy()
            sample.insert(0, "time_seconds", float(row["time_seconds"]))
            collection.append(sample)
        if peak_context is None or float(row["satellite_demand_gbps"]) >= float(
            traffic_profile["satellite_demand_gbps"].max()
        ) - 1e-9:
            peak_context = (visibility, slant, costs, np.asarray(grid.weights))
        progress(time_index + 1, len(times))

    traffic_frame = pd.DataFrame(traffic_rows)
    load_frame = pd.DataFrame(load_rows)
    service_frame = pd.DataFrame(service_rows)
    optimized_sample = pd.concat(optimized_samples, ignore_index=True)
    baseline_sample = pd.concat(baseline_samples, ignore_index=True)
    optimized_switches = int(
        np.count_nonzero(np.diff(np.stack(optimized_dominant), axis=0))
    ) if len(optimized_dominant) > 1 else 0
    baseline_switches = int(
        np.count_nonzero(np.diff(np.stack(baseline_dominant), axis=0))
    ) if len(baseline_dominant) > 1 else 0

    sensitivity_rows = []
    if peak_context is not None:
        visibility, _, costs, weights = peak_context
        peak_actual = float(traffic_frame["actual_region_demand_gbps"].max())
        for eta in traffic_config["eta_sat_sensitivity"]:
            demand = spatial_ground_demand(peak_actual * float(eta), weights)
            result = fair_access_allocation(
                demand, visibility, assignment_cost_ms=costs
            )
            sensitivity_rows.append(
                {
                    "eta_sat": float(eta),
                    "demand_gbps": float(demand.sum()),
                    "service_ratio": result.service_ratio,
                    "throughput_gbps": result.throughput_gbps,
                    "blocked_gbps": result.blocked_gbps,
                    "maximum_utilization": result.maximum_utilization,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    comparison = pd.DataFrame(
        [
            {
                "strategy": "nearest_visible_satellite",
                "strategy_cn": "最近卫星基准",
                "mean_throughput_gbps": float(traffic_frame["baseline_throughput_gbps"].mean()),
                "blocked_ratio": float(traffic_frame["baseline_blocked_gbps"].sum() / max(traffic_frame["satellite_demand_gbps"].sum(), 1e-12)),
                "minimum_service_ratio": float(traffic_frame["baseline_minimum_service_ratio"].min()),
                "maximum_utilization": float(traffic_frame["baseline_maximum_utilization"].max()),
                "utilization_standard_deviation": float(traffic_frame["baseline_utilization_std"].mean()),
                "mean_delay_ms": float(traffic_frame["baseline_mean_delay_ms"].mean()),
                "p95_delay_ms": float(traffic_frame["baseline_p95_delay_ms"].max()),
                "maximum_delay_ms": float(traffic_frame["baseline_maximum_delay_ms"].max()),
                "handover_count": baseline_switches,
            },
            {
                "strategy": "three_stage_optimized",
                "strategy_cn": "三级优化方案",
                "mean_throughput_gbps": float(traffic_frame["optimized_throughput_gbps"].mean()),
                "blocked_ratio": float(traffic_frame["optimized_blocked_gbps"].sum() / max(traffic_frame["satellite_demand_gbps"].sum(), 1e-12)),
                "minimum_service_ratio": float(traffic_frame["optimized_service_ratio"].min()),
                "maximum_utilization": float(traffic_frame["optimized_maximum_utilization"].max()),
                "utilization_standard_deviation": float(traffic_frame["optimized_utilization_std"].mean()),
                "mean_delay_ms": float(traffic_frame["optimized_mean_delay_ms"].mean()),
                "p95_delay_ms": float(traffic_frame["optimized_p95_delay_ms"].max()),
                "maximum_delay_ms": float(traffic_frame["optimized_maximum_delay_ms"].max()),
                "handover_count": optimized_switches,
            },
        ]
    )
    summary = summarize_traffic_timeseries(traffic_frame, dataset)
    peak_index = int(traffic_frame["satellite_demand_gbps"].idxmax())
    minimum_index = int(traffic_frame["satellite_demand_gbps"].idxmin())
    summary.update(
        {
            "mode": config["mode"],
            "eta_sat": float(traffic_config["eta_sat"]),
            "eta_sat_label": traffic_config["eta_sat_label"],
            "traffic_profile_mode": traffic_config["traffic_profile_mode"],
            "optimized_mean_delay_ms": float(traffic_frame["optimized_mean_delay_ms"].mean()),
            "baseline_mean_delay_ms": float(traffic_frame["baseline_mean_delay_ms"].mean()),
            "optimized_handover_count": optimized_switches,
            "baseline_handover_count": baseline_switches,
            "constellation_signature": selected.constellation_signature,
            "constellation_scenario": selected.scenario,
            "M": selected.M,
            "N": selected.N,
            "total_satellites": selected.total_satellites,
            "theoretical_total_access_capacity_gbps": selected.total_satellites
            * C_ACCESS_GBPS,
            "theoretical_mean_service_ratio_upper_bound": selected.total_satellites
            * C_ACCESS_GBPS
            / region_metrics["mean_traffic_gbps"],
            "theoretical_peak_service_ratio_upper_bound": selected.total_satellites
            * C_ACCESS_GBPS
            / (region_metrics["peak_traffic_tbps"] * 1000.0),
            "mean_service_ratio": float(
                traffic_frame["optimized_service_ratio"].mean()
            ),
            "peak_demand_service_ratio": float(
                traffic_frame.loc[peak_index, "optimized_service_ratio"]
            ),
            "minimum_demand_service_ratio": float(
                traffic_frame.loc[minimum_index, "optimized_service_ratio"]
            ),
            "mean_blocked_gbps": float(
                traffic_frame["optimized_blocked_gbps"].mean()
            ),
            "peak_demand_blocked_gbps": float(
                traffic_frame.loc[peak_index, "optimized_blocked_gbps"]
            ),
            "mean_visible_satellite_count": float(
                traffic_frame["visible_satellite_count"].mean()
            ),
            "mean_active_access_satellite_count": float(
                traffic_frame["active_access_satellite_count"].mean()
            ),
            "isl_capacity_constraint": False,
        }
    )
    dataset.audit.to_csv(output_dir / "traffic_data_audit.csv", index=False)
    write_json(output_dir / "traffic_summary.json", summary)
    traffic_frame.to_csv(output_dir / "traffic_timeseries.csv", index=False)
    load_frame.to_csv(output_dir / "satellite_load_timeseries.csv", index=False)
    service_frame.to_csv(output_dir / "grid_service_timeseries.csv", index=False)
    optimized_sample.to_csv(output_dir / "optimized_assignment_sample.csv", index=False)
    baseline_sample.to_csv(output_dir / "baseline_assignment_sample.csv", index=False)
    sensitivity.to_csv(output_dir / "traffic_sensitivity.csv", index=False)
    comparison.to_csv(output_dir / "flow_engineering_comparison.csv", index=False)
    plot_traffic_figures(
        traffic_frame, load_frame, comparison, sensitivity, output_dir
    )
    logger.info(
        "流量完成：吞吐量均值=%.3f Gbps，阻塞比例=%.6f，最大利用率=%.6f",
        summary["optimized_mean_throughput_gbps"],
        summary["optimized_blocked_ratio"],
        summary["maximum_satellite_utilization"],
    )
    return summary


def write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    selected: SelectedConstellation | None,
    started: datetime,
    elapsed_seconds: float,
    completed: bool,
    stage_status: dict[str, str],
) -> None:
    traffic_metrics = region_traffic_metrics()
    metadata = {
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "completed": completed,
        "mode": args.mode,
        "stage": args.stage,
        "pair_mode": args.pair_mode,
        "seed": args.seed,
        "workers": args.workers,
        "resume": args.resume,
        "allow_demo_traffic": args.allow_demo_traffic,
        "traffic_file": args.traffic_file,
        "selected_constellation": selected.to_dict() if selected else None,
        "constellation_scenario": selected.scenario if selected else None,
        "M": selected.M if selected else None,
        "N": selected.N if selected else None,
        "total_satellites": selected.total_satellites if selected else None,
        "constellation_signature": (
            selected.constellation_signature if selected else None
        ),
        "constellation_cache_key": (
            selected.constellation_cache_key if selected else None
        ),
        **traffic_metrics,
        "access_capacity_gbps_per_satellite": C_ACCESS_GBPS,
        "isl_capacity_constraint": False,
        "stage_status": stage_status,
        "config": config,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "disclaimer": "Quick 仅用于代码验收；最终论文数值结论必须使用 Standard/Full。默认流量密度来自用户给定值。",
    }
    write_json(output_dir / "problem3_run_metadata.json", metadata)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(OUTPUT_DIR)
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    selected: SelectedConstellation | None = None
    config: dict[str, Any] = {}
    completed = False
    stage_status: dict[str, str] = {}
    try:
        config = load_problem3_config(PROJECT_ROOT, args.mode)
        validate_config(config)
        selected = discover_problem2_constellation(
            PROJECT_ROOT, args.constellation_file
        )
        print_selected(selected)
        invalidate_stale_constellation_outputs(OUTPUT_DIR, selected, logger)
        resume_cache_valid = resume_outputs_match(
            OUTPUT_DIR, selected, config, args
        )
        write_selected_constellation(selected, OUTPUT_DIR)
        constellation = constellation_from_selection(selected)
        set_problem3_figure_context(selected)
        write_satellite_mapping(constellation, OUTPUT_DIR)

        requested = (
            ["topology", "routing", "traffic"]
            if args.stage == "all"
            else [args.stage]
        )
        for stage in requested:
            summary_file = OUTPUT_DIR / f"{stage}_summary.json"
            if args.resume and summary_file.exists() and resume_cache_valid:
                logger.info("--resume：检测到 %s，跳过 %s 阶段。", summary_file, stage)
                stage_status[stage] = "resumed_existing_output"
                continue
            if args.resume and summary_file.exists() and not resume_cache_valid:
                logger.warning(
                    "--resume：现有 %s 的星座、模式或配置不匹配，将重新计算 %s。",
                    summary_file,
                    stage,
                )
            if stage == "topology":
                run_topology_stage(
                    constellation, config, OUTPUT_DIR, args.workers, logger
                )
                stage_status[stage] = "completed"
            elif stage == "routing":
                run_routing_stage(
                    constellation,
                    selected,
                    config,
                    OUTPUT_DIR,
                    args.pair_mode,
                    args.seed,
                    args.workers,
                    logger,
                )
                stage_status[stage] = "completed"
            else:
                summary = run_traffic_stage(
                    constellation,
                    selected,
                    config,
                    OUTPUT_DIR,
                    args.traffic_file,
                    args.allow_demo_traffic,
                    args.seed,
                    logger,
                )
                stage_status[stage] = str(summary.get("status", "completed"))
        write_method_notes(PROJECT_ROOT, OUTPUT_DIR, selected, config, args.mode)
        completed = True
    except KeyboardInterrupt:
        logger.warning("收到中断信号；已完成阶段的 CSV/JSON 保留，可使用 --resume 跳过。")
        raise
    except (ConstellationSelectionError, FileNotFoundError, ValueError) as exc:
        logger.error("问题三输入或配置错误：%s", exc)
        raise
    finally:
        elapsed = time.perf_counter() - clock
        write_run_metadata(
            OUTPUT_DIR,
            args,
            config,
            selected,
            started,
            elapsed,
            completed,
            stage_status,
        )

    print("\n问题三运行摘要")
    print(f"mode={args.mode}, stage={args.stage}, elapsed={elapsed / 60.0:.2f} min")
    print(f"stage_status={stage_status}")
    if args.mode == "quick":
        print("注意：Quick 结果仅用于代码验收，不能替代 Standard 正式结果。")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
