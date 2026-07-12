"""Traffic ingestion, fair admission, and access-load balancing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from .problem2_orbit import R_EARTH_KM


C_ACCESS_GBPS = 20.0
TRAFFIC_DENSITY_MBPS_PER_KM2 = 7.02139
REGION_LAT_MIN_DEG = 4.0
REGION_LAT_MAX_DEG = 53.0
REGION_LON_MIN_DEG = 73.0
REGION_LON_MAX_DEG = 135.0
PEAK_TO_MEAN_RATIO = 1.5


@dataclass
class TrafficDataset:
    status: str
    data_type: str
    average_actual_gbps: float
    hourly_profile: pd.DataFrame | None
    audit: pd.DataFrame
    source_names: list[str]
    source_urls: list[str]
    is_synthetic_demo: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessAllocation:
    service_ratio: float
    throughput_gbps: float
    blocked_gbps: float
    maximum_utilization: float
    utilization_standard_deviation: float
    assignments: pd.DataFrame
    satellite_loads_gbps: np.ndarray
    ground_served_gbps: np.ndarray
    optimization_status: str


RATE_FACTORS_TO_GBPS = {"gbps": 1.0, "tbps": 1000.0}
CUMULATIVE_BYTES = {
    "gb": 1.0e9,
    "tb": 1.0e12,
    "pb": 1.0e15,
    "eb": 1.0e18,
}


def spherical_rectangle_area_km2(
    lat_min_deg: float = REGION_LAT_MIN_DEG,
    lat_max_deg: float = REGION_LAT_MAX_DEG,
    lon_min_deg: float = REGION_LON_MIN_DEG,
    lon_max_deg: float = REGION_LON_MAX_DEG,
    earth_radius_km: float = R_EARTH_KM,
) -> float:
    """Exact spherical area of the target latitude-longitude rectangle."""

    if not (-90.0 <= lat_min_deg < lat_max_deg <= 90.0):
        raise ValueError("纬度边界必须满足 -90 <= min < max <= 90。")
    if not lon_min_deg < lon_max_deg:
        raise ValueError("经度上边界必须大于下边界。")
    delta_lambda = math.radians(lon_max_deg - lon_min_deg)
    sine_span = math.sin(math.radians(lat_max_deg)) - math.sin(
        math.radians(lat_min_deg)
    )
    return float(earth_radius_km**2 * delta_lambda * sine_span)


def region_traffic_metrics(
    traffic_density_mbps_per_km2: float = TRAFFIC_DENSITY_MBPS_PER_KM2,
) -> dict[str, float]:
    area = spherical_rectangle_area_km2()
    mean_mbps = float(traffic_density_mbps_per_km2) * area
    return {
        "region_lat_min_deg": REGION_LAT_MIN_DEG,
        "region_lat_max_deg": REGION_LAT_MAX_DEG,
        "region_lon_min_deg": REGION_LON_MIN_DEG,
        "region_lon_max_deg": REGION_LON_MAX_DEG,
        "traffic_density_mbps_per_km2": float(traffic_density_mbps_per_km2),
        "earth_radius_km": R_EARTH_KM,
        "region_area_km2": area,
        "mean_traffic_mbps": mean_mbps,
        "mean_traffic_gbps": mean_mbps / 1.0e3,
        "mean_traffic_tbps": mean_mbps / 1.0e6,
        "peak_to_mean_ratio": PEAK_TO_MEAN_RATIO,
        "peak_traffic_tbps": PEAK_TO_MEAN_RATIO * mean_mbps / 1.0e6,
        "minimum_traffic_tbps": 0.5 * mean_mbps / 1.0e6,
    }


def load_region_traffic_density(path: str | Path) -> TrafficDataset:
    """Load the user-supplied uniform density and derive total regional traffic."""

    path = Path(path)
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError("区域流量密度文件必须且只能包含一行。")
    row = frame.iloc[0]
    density = float(row["traffic_density_value"])
    normalized_unit = str(row["traffic_density_normalized_unit"]).strip().lower()
    if normalized_unit not in {"mbps/km²", "mbps/km2"}:
        raise ValueError("区域流量密度必须归一化为 Mbps/km²。")
    bounds = (
        float(row["lat_min_deg"]),
        float(row["lat_max_deg"]),
        float(row["lon_min_deg"]),
        float(row["lon_max_deg"]),
    )
    expected_bounds = (
        REGION_LAT_MIN_DEG,
        REGION_LAT_MAX_DEG,
        REGION_LON_MIN_DEG,
        REGION_LON_MAX_DEG,
    )
    if not np.allclose(bounds, expected_bounds, atol=1e-12, rtol=0.0):
        raise ValueError(f"区域边界必须为 {expected_bounds}，实际为 {bounds}。")
    if not math.isclose(
        density, TRAFFIC_DENSITY_MBPS_PER_KM2, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ValueError(
            f"流量密度必须为 {TRAFFIC_DENSITY_MBPS_PER_KM2} Mbps/km²。"
        )
    metrics = region_traffic_metrics(density)
    audit = frame.copy()
    audit.insert(0, "source_file", str(path.resolve()))
    for key, value in metrics.items():
        audit[key] = value
    return TrafficDataset(
        status="user_given_region_density",
        data_type="uniform_area_density",
        average_actual_gbps=metrics["mean_traffic_gbps"],
        hourly_profile=None,
        audit=audit,
        source_names=_clean_sources(frame, "source_name"),
        source_urls=_clean_sources(frame, "source_url"),
        metadata=metrics,
    )


def _clean_sources(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    return sorted(
        {
            str(value).strip()
            for value in frame[column].dropna()
            if str(value).strip()
        }
    )


def load_traffic_data(path: str | Path) -> TrafficDataset:
    """Parse rate, cumulative-volume, or timestamped traffic data."""

    path = Path(path)
    frame = pd.read_csv(path)
    required = {"traffic_value", "traffic_unit"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"流量文件缺少字段：{', '.join(missing)}")
    if frame.empty:
        raise ValueError("流量文件没有数据行。")
    values = pd.to_numeric(frame["traffic_value"], errors="raise").to_numpy(float)
    if np.any(values < 0.0):
        raise ValueError("traffic_value 不能为负数。")
    units = frame["traffic_unit"].astype(str).str.strip().str.lower()
    audit = frame.copy()

    has_timestamps = "timestamp" in frame and frame["timestamp"].notna().all()
    if has_timestamps:
        unsupported = sorted(set(units) - set(RATE_FACTORS_TO_GBPS))
        if unsupported:
            raise ValueError(f"小时曲线仅支持 Gbps/Tbps，发现：{unsupported}")
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        converted = np.asarray(
            [value * RATE_FACTORS_TO_GBPS[unit] for value, unit in zip(values, units)]
        )
        hourly = pd.DataFrame(
            {"timestamp": timestamps, "traffic_gbps": converted}
        ).sort_values("timestamp")
        audit["converted_average_gbps"] = converted
        data_type = "hourly_rate_curve"
        average = float(np.mean(converted))
    elif set(units).issubset(RATE_FACTORS_TO_GBPS):
        converted = np.asarray(
            [value * RATE_FACTORS_TO_GBPS[unit] for value, unit in zip(values, units)]
        )
        audit["converted_average_gbps"] = converted
        hourly = None
        data_type = "average_rate"
        average = float(np.sum(converted))
    elif set(units).issubset(CUMULATIVE_BYTES):
        required_period = {"period_start", "period_end"}
        missing_period = sorted(required_period - set(frame.columns))
        if missing_period:
            raise ValueError(
                "累计流量数据需要 period_start 和 period_end："
                + ", ".join(missing_period)
            )
        starts = pd.to_datetime(frame["period_start"], utc=True, errors="raise")
        ends = pd.to_datetime(frame["period_end"], utc=True, errors="raise")
        seconds = (ends - starts).dt.total_seconds().to_numpy(float)
        if np.any(seconds <= 0.0):
            raise ValueError("累计流量统计周期必须为正。")
        converted = np.asarray(
            [
                value * CUMULATIVE_BYTES[unit] * 8.0 / duration / 1.0e9
                for value, unit, duration in zip(values, units, seconds)
            ]
        )
        audit["period_seconds"] = seconds
        audit["converted_average_gbps"] = converted
        hourly = None
        data_type = "cumulative_volume"
        average = float(np.sum(converted))
    else:
        raise ValueError(
            "同一流量文件不能混合速率单位与累计单位，或包含不支持的 traffic_unit。"
        )

    audit.insert(0, "source_file", str(path.resolve()))
    return TrafficDataset(
        status="real_data_loaded",
        data_type=data_type,
        average_actual_gbps=average,
        hourly_profile=hourly,
        audit=audit,
        source_names=_clean_sources(frame, "source_name"),
        source_urls=_clean_sources(frame, "source_url"),
    )


def make_synthetic_demo_traffic(mean_gbps: float) -> TrafficDataset:
    audit = pd.DataFrame(
        [
            {
                "source_file": "synthetic_demo",
                "region_name": "测试区域",
                "traffic_value": float(mean_gbps),
                "traffic_unit": "Gbps",
                "converted_average_gbps": float(mean_gbps),
                "source_name": "synthetic_demo",
                "source_url": "",
                "notes": "仅用于 quick 代码验收，不是实际流量数据或论文结论。",
            }
        ]
    )
    return TrafficDataset(
        status="synthetic_demo",
        data_type="average_rate",
        average_actual_gbps=float(mean_gbps),
        hourly_profile=None,
        audit=audit,
        source_names=["synthetic_demo"],
        source_urls=[],
        is_synthetic_demo=True,
    )


def periodic_problem_spec_profile(times_seconds: np.ndarray) -> np.ndarray:
    """Daily smooth profile with global mean 1 and peak-to-mean ratio 1.5."""

    times = np.asarray(times_seconds, dtype=float)
    # Put the peak at t=0 so a short Quick run still exercises the 1.5 peak.
    peak_seconds = 0.0
    return 1.0 + 0.5 * np.cos(2.0 * np.pi * (times - peak_seconds) / 86400.0)


def build_traffic_timeseries(
    dataset: TrafficDataset,
    times_seconds: np.ndarray,
    eta_sat: float,
    profile_mode: str = "problem_spec",
) -> pd.DataFrame:
    times = np.asarray(times_seconds, dtype=float)
    eta_sat = float(eta_sat)
    if not 0.0 <= eta_sat <= 1.0:
        raise ValueError("eta_sat 必须位于 [0,1]。")
    if profile_mode == "actual_profile" and dataset.hourly_profile is not None:
        hourly = dataset.hourly_profile
        source_seconds = (
            hourly["timestamp"] - hourly["timestamp"].iloc[0]
        ).dt.total_seconds().to_numpy(float)
        source_values = hourly["traffic_gbps"].to_numpy(float)
        period = max(float(source_seconds[-1]), 86400.0)
        profile = np.interp(
            np.mod(times, period), source_seconds, source_values,
            left=source_values[0], right=source_values[-1]
        )
        actual_demand = profile
    elif profile_mode == "problem_spec":
        actual_demand = dataset.average_actual_gbps * periodic_problem_spec_profile(times)
    else:
        raise ValueError(
            "traffic_profile_mode='actual_profile' 需要小时流量曲线，"
            "否则应使用 'problem_spec'。"
        )
    satellite_demand = eta_sat * actual_demand
    return pd.DataFrame(
        {
            "time_seconds": times,
            "actual_region_demand_gbps": actual_demand,
            "eta_sat": eta_sat,
            "satellite_demand_gbps": satellite_demand,
            "profile_mode": profile_mode,
        }
    )


def spatial_ground_demand(
    total_demand_gbps: float, area_weights: np.ndarray
) -> np.ndarray:
    weights = np.asarray(area_weights, dtype=float)
    if np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("Area weights must be nonnegative and have positive sum.")
    return float(total_demand_gbps) * weights / weights.sum()


def grid_cell_areas_km2(
    area_weights: np.ndarray,
    region_area_km2: float | None = None,
) -> np.ndarray:
    weights = np.asarray(area_weights, dtype=float)
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Area weights must be nonnegative and have positive sum.")
    area = (
        spherical_rectangle_area_km2()
        if region_area_km2 is None
        else float(region_area_km2)
    )
    cells = area * weights / weights.sum()
    if not math.isclose(float(cells.sum()), area, rel_tol=1e-12, abs_tol=1e-6):
        raise RuntimeError("网格面积之和与题目目标区域面积不一致。")
    return cells


def _visible_edges(visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ground_index, satellite_index = np.nonzero(np.asarray(visibility, dtype=bool))
    return ground_index.astype(int), satellite_index.astype(int)


def _ground_equality_matrix(
    ground_index: np.ndarray,
    ground_count: int,
    variable_count: int,
) -> csr_matrix:
    data = np.ones(len(ground_index), dtype=float)
    columns = np.arange(len(ground_index), dtype=int)
    return coo_matrix(
        (data, (ground_index, columns)), shape=(ground_count, variable_count)
    ).tocsr()


def fair_access_allocation(
    demands_gbps: np.ndarray,
    visibility: np.ndarray,
    *,
    capacity_gbps: float | np.ndarray = C_ACCESS_GBPS,
    assignment_cost_ms: np.ndarray | None = None,
    balance_tolerance: float = 1e-7,
) -> AccessAllocation:
    """Solve lexicographic fair admission, load balancing, and delay minimization."""

    demands = np.asarray(demands_gbps, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    if visible.ndim != 2 or visible.shape[0] != demands.size:
        raise ValueError("visibility must have shape (ground, satellite).")
    if np.any(demands < -1e-12):
        raise ValueError("Ground demands must be nonnegative.")
    ground_count, satellite_count = visible.shape
    capacities = np.asarray(capacity_gbps, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(satellite_count, float(capacities), dtype=float)
    if capacities.shape != (satellite_count,):
        raise ValueError("capacity_gbps must be a scalar or one value per satellite.")
    if np.any(capacities < -1e-12):
        raise ValueError("Satellite capacities must be nonnegative.")
    capacities = np.maximum(capacities, 0.0)
    ground_index, satellite_index = _visible_edges(visible)
    edge_count = len(ground_index)
    total_demand = float(demands.sum())
    if total_demand <= 1e-12:
        empty = pd.DataFrame(
            columns=["ground_index", "satellite_id", "assigned_gbps", "assignment_cost_ms"]
        )
        return AccessAllocation(
            1.0, 0.0, 0.0, 0.0, 0.0, empty,
            np.zeros(satellite_count), np.zeros(ground_count), "zero_demand"
        )

    # Stage 1: maximize the common service ratio r.
    variable_count = edge_count + 1
    A_eq = _ground_equality_matrix(ground_index, ground_count, variable_count).tolil()
    A_eq[:, -1] = -demands[:, None]
    A_eq = A_eq.tocsr()
    sat_rows = satellite_index
    sat_cols = np.arange(edge_count, dtype=int)
    A_ub = coo_matrix(
        (np.ones(edge_count), (sat_rows, sat_cols)),
        shape=(satellite_count, variable_count),
    ).tocsr()
    c = np.zeros(variable_count)
    c[-1] = -1.0
    stage1 = linprog(
        c,
        A_ub=A_ub,
        b_ub=capacities,
        A_eq=A_eq,
        b_eq=np.zeros(ground_count),
        bounds=[(0.0, None)] * edge_count + [(0.0, 1.0)],
        method="highs",
    )
    if not stage1.success:
        raise RuntimeError(f"公平准入 LP 求解失败：{stage1.message}")
    service_ratio = float(np.clip(stage1.x[-1], 0.0, 1.0))

    # Stage 2: keep r fixed and minimize maximum utilization u.
    variable_count2 = edge_count + 1
    A_eq2 = _ground_equality_matrix(ground_index, ground_count, variable_count2)
    b_eq2 = service_ratio * demands
    rows = np.concatenate((satellite_index, np.arange(satellite_count)))
    cols = np.concatenate(
        (np.arange(edge_count), np.full(satellite_count, edge_count))
    )
    data = np.concatenate(
        (np.ones(edge_count), -capacities)
    )
    A_ub2 = coo_matrix(
        (data, (rows, cols)), shape=(satellite_count, variable_count2)
    ).tocsr()
    c2 = np.zeros(variable_count2)
    c2[-1] = 1.0
    stage2 = linprog(
        c2,
        A_ub=A_ub2,
        b_ub=np.zeros(satellite_count),
        A_eq=A_eq2,
        b_eq=b_eq2,
        bounds=[(0.0, None)] * edge_count + [(0.0, 1.0)],
        method="highs",
    )
    if not stage2.success:
        raise RuntimeError(f"负载均衡 LP 求解失败：{stage2.message}")
    maximum_utilization = float(np.clip(stage2.x[-1], 0.0, 1.0))

    # Stage 3: minimize assignment delay without sacrificing stages 1 and 2.
    if assignment_cost_ms is None:
        edge_cost = np.zeros(edge_count, dtype=float)
    else:
        costs = np.asarray(assignment_cost_ms, dtype=float)
        if costs.shape != visible.shape:
            raise ValueError("assignment_cost_ms must match visibility shape.")
        edge_cost = costs[ground_index, satellite_index]
    A_ub3 = coo_matrix(
        (
            np.ones(edge_count),
            (satellite_index, np.arange(edge_count, dtype=int)),
        ),
        shape=(satellite_count, edge_count),
    ).tocsr()
    capacity_limit = capacities * min(
        1.0, maximum_utilization + float(balance_tolerance)
    )
    stage3 = linprog(
        edge_cost,
        A_ub=A_ub3,
        b_ub=capacity_limit,
        A_eq=_ground_equality_matrix(ground_index, ground_count, edge_count),
        b_eq=b_eq2,
        bounds=[(0.0, None)] * edge_count,
        method="highs",
    )
    flows = stage3.x if stage3.success else stage2.x[:edge_count]
    loads = np.bincount(
        satellite_index, weights=flows, minlength=satellite_count
    ).astype(float)
    ground_served = np.bincount(
        ground_index, weights=flows, minlength=ground_count
    ).astype(float)
    if np.any(loads > capacities + 1e-5):
        raise RuntimeError("优化结果违反单星接入容量。")
    utilizations = np.divide(
        loads,
        capacities,
        out=np.zeros_like(loads),
        where=capacities > 0.0,
    )
    assignments = pd.DataFrame(
        {
            "ground_index": ground_index,
            "satellite_id": satellite_index,
            "assigned_gbps": flows,
            "assignment_cost_ms": edge_cost,
        }
    )
    assignments = assignments[assignments["assigned_gbps"] > 1e-10].reset_index(drop=True)
    throughput = float(ground_served.sum())
    return AccessAllocation(
        service_ratio=service_ratio,
        throughput_gbps=throughput,
        blocked_gbps=max(0.0, total_demand - throughput),
        maximum_utilization=float(np.max(utilizations, initial=0.0)),
        utilization_standard_deviation=float(np.std(utilizations)),
        assignments=assignments,
        satellite_loads_gbps=loads,
        ground_served_gbps=ground_served,
        optimization_status="optimal" if stage3.success else "stage2_optimal_stage3_fallback",
    )


def nearest_visible_baseline(
    demands_gbps: np.ndarray,
    visibility: np.ndarray,
    slant_distances_km: np.ndarray,
    *,
    capacity_gbps: float = C_ACCESS_GBPS,
    seed: int = 0,
) -> AccessAllocation:
    """Deterministic nearest-visible allocation with capacity spillover."""

    demands = np.asarray(demands_gbps, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    slant = np.asarray(slant_distances_km, dtype=float)
    if visible.shape != slant.shape or visible.shape[0] != demands.size:
        raise ValueError("Visibility, slant distances, and demand dimensions disagree.")
    ground_count, satellite_count = visible.shape
    loads = np.zeros(satellite_count, dtype=float)
    served = np.zeros(ground_count, dtype=float)
    records: list[dict[str, float | int]] = []
    rng = np.random.default_rng(int(seed))
    order = np.arange(ground_count)
    rng.shuffle(order)
    order = order[np.argsort(-demands[order], kind="stable")]
    for ground in order:
        satellites = np.flatnonzero(visible[ground])
        satellites = satellites[np.argsort(slant[ground, satellites], kind="stable")]
        remaining = float(demands[ground])
        for satellite in satellites:
            available = max(0.0, float(capacity_gbps) - loads[satellite])
            assigned = min(remaining, available)
            if assigned > 1e-12:
                loads[satellite] += assigned
                served[ground] += assigned
                records.append(
                    {
                        "ground_index": int(ground),
                        "satellite_id": int(satellite),
                        "assigned_gbps": assigned,
                        "assignment_cost_ms": slant[ground, satellite]
                        / 299792.458
                        * 1000.0,
                    }
                )
                remaining -= assigned
            if remaining <= 1e-12:
                break
    total_demand = float(demands.sum())
    throughput = float(served.sum())
    positive = demands > 1e-12
    ratios = np.divide(served, demands, out=np.ones_like(served), where=positive)
    return AccessAllocation(
        service_ratio=float(np.min(ratios[positive])) if np.any(positive) else 1.0,
        throughput_gbps=throughput,
        blocked_gbps=max(0.0, total_demand - throughput),
        maximum_utilization=float(np.max(loads, initial=0.0) / capacity_gbps),
        utilization_standard_deviation=float(np.std(loads / capacity_gbps)),
        assignments=pd.DataFrame(records),
        satellite_loads_gbps=loads,
        ground_served_gbps=served,
        optimization_status="nearest_visible_satellite",
    )


def assignment_probabilities(
    allocation: AccessAllocation, ground_count: int, satellite_count: int
) -> np.ndarray:
    probabilities = np.zeros((ground_count, satellite_count), dtype=float)
    if allocation.assignments.empty:
        return probabilities
    frame = allocation.assignments
    probabilities[
        frame["ground_index"].to_numpy(int), frame["satellite_id"].to_numpy(int)
    ] = frame["assigned_gbps"].to_numpy(float)
    totals = probabilities.sum(axis=1, keepdims=True)
    return np.divide(probabilities, totals, out=np.zeros_like(probabilities), where=totals > 0)


def summarize_traffic_timeseries(frame: pd.DataFrame, dataset: TrafficDataset) -> dict[str, Any]:
    if frame.empty:
        return {"status": "skipped_missing_real_data"}
    summary = {
        "status": dataset.status,
        "is_synthetic_demo": dataset.is_synthetic_demo,
        "data_type": dataset.data_type,
        "source_names": dataset.source_names,
        "source_urls": dataset.source_urls,
        "actual_average_demand_gbps": dataset.average_actual_gbps,
        "satellite_mean_demand_gbps": float(frame["satellite_demand_gbps"].mean()),
        "satellite_peak_demand_gbps": float(frame["satellite_demand_gbps"].max()),
        "optimized_mean_throughput_gbps": float(frame["optimized_throughput_gbps"].mean()),
        "optimized_peak_throughput_gbps": float(frame["optimized_throughput_gbps"].max()),
        "optimized_total_blocked_gbps_samples": float(frame["optimized_blocked_gbps"].sum()),
        "optimized_blocked_ratio": float(
            frame["optimized_blocked_gbps"].sum()
            / max(frame["satellite_demand_gbps"].sum(), 1e-12)
        ),
        "minimum_service_ratio": float(frame["optimized_service_ratio"].min()),
        "maximum_satellite_utilization": float(frame["optimized_maximum_utilization"].max()),
        "baseline_mean_throughput_gbps": float(frame["baseline_throughput_gbps"].mean()),
        "baseline_blocked_ratio": float(
            frame["baseline_blocked_gbps"].sum()
            / max(frame["satellite_demand_gbps"].sum(), 1e-12)
        ),
        "baseline_maximum_satellite_utilization": float(
            frame["baseline_maximum_utilization"].max()
        ),
        "isl_capacity_modeled": False,
        "access_capacity_gbps_per_satellite": C_ACCESS_GBPS,
    }
    summary.update(dataset.metadata)
    return summary
