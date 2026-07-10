"""Problem 2: multi-plane regional constellation optimization entry point."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd
import scipy


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from starlink_modeling.problem2_coverage import (  # noqa: E402
    CoverageEvaluation,
    evaluate_constellation_coverage,
)
from starlink_modeling.problem2_ga import (  # noqa: E402
    Individual,
    apply_evaluation,
    enumerate_certification_pairs,
    is_better,
    load_problem2_config,
    local_coordinate_refinement,
    sort_population,
    run_multiseed_ga,
)
from starlink_modeling.problem2_grid import (  # noqa: E402
    GroundGrid,
    make_equal_arc_grid,
    make_grid,
    make_regular_grid,
)
from starlink_modeling.problem2_orbit import (  # noqa: E402
    ORBIT_HEIGHT_KM,
    R_EARTH_KM,
    generate_walker_constellation,
)
from starlink_modeling.problem2_reporting import (  # noqa: E402
    build_cost_comparison,
    solution_payload,
    write_method_notes,
    write_solution_bundle,
)
from starlink_modeling.problem2_visualization import (  # noqa: E402
    plot_area_coverage_time,
    plot_cost_comparison,
    plot_ga_convergence,
    plot_ground_tracks,
    plot_mn_candidates,
    plot_region_grid,
    plot_resolution_sensitivity,
    plot_solution_grid_fields,
    plot_walker_layout_3d,
    plot_worst_snapshot,
)


GROUND_RADIUS_KM = 506.0
TARGET_MAX_LATITUDE_DEG = 53.0
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "problem2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问题二：Walker-Delta 多轨道面区域覆盖遗传算法优化"
    )
    parser.add_argument(
        "--mode", choices=("quick", "standard", "full"), default="standard"
    )
    parser.add_argument(
        "--scenario", choices=("single", "double", "both"), default="both"
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-certification",
        action="store_true",
        help="仅用于调试；跳过固定 M,N 反向核验",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="仅用于调试；跳过分辨率敏感性计算",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("problem2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        output_dir / "problem2_run.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def config_path_for_mode(mode: str) -> Path:
    return PROJECT_ROOT / "configs" / f"problem2_{mode}.json"


def validate_config(config: dict[str, Any]) -> None:
    """Fail early with readable messages for malformed numerical settings."""

    required_sections = {
        "mode",
        "search_bounds",
        "region",
        "simulation",
        "ga",
        "certification",
        "local_refinement",
        "validation",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"配置文件缺少字段：{', '.join(missing)}")
    for key in ("M", "N", "inclination_deg"):
        bounds = config["search_bounds"].get(key)
        if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"search_bounds.{key} 必须是递增的二元数组。")
    simulation = config["simulation"]
    if float(simulation["simulation_days"]) <= 0.0:
        raise ValueError("simulation_days 必须为正数。")
    if float(simulation["time_step_seconds"]) <= 0.0:
        raise ValueError("time_step_seconds 必须为正数。")
    if int(config["ga"]["population_size"]) < 2:
        raise ValueError("population_size 至少为 2。")
    if int(config["ga"]["generations"]) < 1:
        raise ValueError("generations 至少为 1。")


def simulation_grid(config: dict[str, Any]) -> GroundGrid:
    return make_grid({**config["region"], **config["simulation"]})


def prefilter_metrics(simulation_days: float, upper_reach_deg: float) -> dict[str, Any]:
    duration_minutes = simulation_days * 1440.0
    shortfall = max(0.0, TARGET_MAX_LATITUDE_DEG - upper_reach_deg)
    return {
        "Q1_full_region": 0.0,
        "Q2_full_region": 0.0,
        "P1_space_time": 0.0,
        "P2_space_time": 0.0,
        "mean_multiplicity": 0.0,
        "worst_A1": 0.0,
        "worst_A2": 0.0,
        "max_gap_time_min": duration_minutes,
        "max_double_gap_time_min": duration_minutes,
        "min_availability1": 0.0,
        "min_availability2": 0.0,
        "mean_availability1": 0.0,
        "mean_availability2": 0.0,
        "std_availability1": 0.0,
        "std_mean_multiplicity": 0.0,
        "min_angular_margin_deg": -shortfall,
        "worst_grid_latitude_deg": TARGET_MAX_LATITUDE_DEG,
        "worst_grid_longitude_deg": 73.0,
        "worst_double_grid_latitude_deg": TARGET_MAX_LATITUDE_DEG,
        "worst_double_grid_longitude_deg": 73.0,
        "sample_count": 0,
        "grid_point_count": 0,
        "simulation_duration_seconds": simulation_days * 86400.0,
        "coverage_method": "geometric_prefilter",
        "single_feasible": False,
        "double_feasible": False,
        "upper_reach_deg": upper_reach_deg,
        "geometric_prefilter_failed": True,
    }


def make_candidate_evaluator(
    config: dict[str, Any],
    grid: GroundGrid,
    theta_rad: float,
    scenario: str,
) -> Callable[[Individual], Individual]:
    simulation = config["simulation"]
    duration_seconds = float(simulation["simulation_days"]) * 86400.0
    tolerance = float(simulation.get("tolerance", 1e-12))
    theta_deg = math.degrees(theta_rad)

    def evaluate(individual: Individual) -> Individual:
        upper_reach_deg = individual.inclination_deg + theta_deg
        if upper_reach_deg < TARGET_MAX_LATITUDE_DEG:
            metrics = prefilter_metrics(
                float(simulation["simulation_days"]), upper_reach_deg
            )
            evaluated = apply_evaluation(
                individual, metrics, scenario, duration_seconds, tolerance
            )
            evaluated.constraint_violation += 10.0 * (
                TARGET_MAX_LATITUDE_DEG - upper_reach_deg
            )
            return evaluated

        constellation = generate_walker_constellation(
            individual.M,
            individual.N,
            individual.inclination_deg,
            individual.phase_factor_F,
            individual.Omega0_deg,
            individual.u0_deg,
            ORBIT_HEIGHT_KM,
        )
        result = evaluate_constellation_coverage(
            constellation,
            grid,
            theta_rad,
            float(simulation["simulation_days"]),
            float(simulation["time_step_seconds"]),
            method=simulation.get("coverage_method", "auto"),
            tolerance=tolerance,
        )
        result.metrics["upper_reach_deg"] = upper_reach_deg
        result.metrics["geometric_prefilter_failed"] = False
        return apply_evaluation(
            individual, result.metrics, scenario, duration_seconds, tolerance
        )

    return evaluate


def evaluate_detailed(
    candidate: Individual,
    config: dict[str, Any],
    grid: GroundGrid,
    theta_rad: float,
    scenario: str,
) -> tuple[Individual, CoverageEvaluation]:
    constellation = generate_walker_constellation(
        candidate.M,
        candidate.N,
        candidate.inclination_deg,
        candidate.phase_factor_F,
        candidate.Omega0_deg,
        candidate.u0_deg,
    )
    simulation = config["simulation"]
    result = evaluate_constellation_coverage(
        constellation,
        grid,
        theta_rad,
        float(simulation["simulation_days"]),
        float(simulation["time_step_seconds"]),
        method=simulation.get("coverage_method", "auto"),
        return_time_series=True,
        return_grid_metrics=True,
        return_snapshots=True,
        tolerance=float(simulation.get("tolerance", 1e-12)),
    )
    result.metrics["upper_reach_deg"] = candidate.inclination_deg + math.degrees(theta_rad)
    result.metrics["geometric_prefilter_failed"] = False
    evaluated = apply_evaluation(
        candidate,
        result.metrics,
        scenario,
        float(simulation["simulation_days"]) * 86400.0,
        float(simulation.get("tolerance", 1e-12)),
    )
    return evaluated, result


def run_fixed_mn_certification(
    current_best: Individual,
    config: dict[str, Any],
    scenario: str,
    grid: GroundGrid,
    theta_rad: float,
    seed: int,
    workers: int,
    output_dir: Path,
    resume: bool,
    logger: logging.Logger,
) -> tuple[Individual, pd.DataFrame, pd.DataFrame]:
    certification = config["certification"]
    pairs = enumerate_certification_pairs(
        config["search_bounds"],
        current_best.total_satellites,
        int(certification.get("satellite_margin", 0)),
        certification.get("max_pairs"),
    )
    if pairs.empty:
        return current_best, pairs, pd.DataFrame()

    fixed_config = copy.deepcopy(config)
    fixed_config["ga"]["population_size"] = int(certification["population_size"])
    fixed_config["ga"]["generations"] = int(certification["generations"])
    fixed_config["ga"]["random_seed_count"] = int(certification["random_seed_count"])
    fixed_config["ga"]["early_stop_generations"] = 0
    evaluator = make_candidate_evaluator(fixed_config, grid, theta_rad, scenario)
    candidate_frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    best = current_best

    for pair_index, pair in pairs.iterrows():
        if not bool(pair["selected_for_search"]):
            records.append(
                {
                    **pair.to_dict(),
                    "certification_status": "not_run_mode_limit",
                    "feasible": False,
                }
            )
            continue
        fixed_MN = (int(pair["M"]), int(pair["N"]))
        logger.info(
            "固定 M,N 核验 %s：M=%d, N=%d, S=%d",
            scenario,
            fixed_MN[0],
            fixed_MN[1],
            fixed_MN[0] * fixed_MN[1],
        )
        result = run_multiseed_ga(
            fixed_config,
            scenario,
            lambda _seed: evaluator,
            seed + 1000 + int(pair_index) * 17,
            output_dir,
            workers=workers,
            resume=resume,
            fixed_MN=fixed_MN,
            seed_count=int(certification["random_seed_count"]),
        )
        candidate_frames.append(result.all_candidates)
        record = {
            **pair.to_dict(),
            **result.best.record(),
            "certification_status": "completed",
            "best_seed": result.seed,
        }
        records.append(record)
        if is_better(result.best, best, scenario):
            best = result.best

    certification_frame = pd.DataFrame(records)
    all_candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame()
    )
    return best, certification_frame, all_candidates


def run_validation_profiles(
    candidate: Individual,
    config: dict[str, Any],
    theta_rad: float,
    scenario: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    if config["mode"] != "full":
        return {}
    results: dict[str, Any] = {}
    for profile_name, profile in config["validation"].items():
        if not isinstance(profile, dict):
            continue
        if "regular_grid_step_deg" in profile:
            grid = make_regular_grid(
                float(profile["regular_grid_step_deg"]),
                tuple(config["region"]["latitude_bounds_deg"]),
                tuple(config["region"]["longitude_bounds_deg"]),
            )
            resolution = f"{profile['regular_grid_step_deg']} deg"
        else:
            grid = make_equal_arc_grid(
                float(profile["equal_arc_spacing_km"]),
                tuple(config["region"]["latitude_bounds_deg"]),
                tuple(config["region"]["longitude_bounds_deg"]),
            )
            resolution = f"{profile['equal_arc_spacing_km']} km"
        constellation = generate_walker_constellation(
            candidate.M,
            candidate.N,
            candidate.inclination_deg,
            candidate.phase_factor_F,
            candidate.Omega0_deg,
            candidate.u0_deg,
        )
        logger.info(
            "执行 full %s 验证：%s，网格 %s，步长 %.0f s，%.1f d",
            scenario,
            profile_name,
            resolution,
            profile["time_step_seconds"],
            profile["simulation_days"],
        )
        evaluation = evaluate_constellation_coverage(
            constellation,
            grid,
            theta_rad,
            float(profile["simulation_days"]),
            float(profile["time_step_seconds"]),
            method="auto",
        )
        results[profile_name] = {
            **evaluation.metrics,
            "grid_resolution": resolution,
            "time_step_seconds": profile["time_step_seconds"],
            "simulation_days": profile["simulation_days"],
        }
    return results


def validation_status(
    mode: str,
    candidate: Individual,
    scenario: str,
    validation_results: dict[str, Any],
) -> str:
    if not candidate.feasible:
        return "search_infeasible"
    if mode == "quick":
        return "quick_smoke_only"
    if mode == "standard":
        return "standard_search_passed_requires_full"
    if not validation_results:
        return "validation_failed"
    for result in validation_results.values():
        if scenario == "single" and not bool(result["single_feasible"]):
            return "validation_failed"
        if scenario == "double" and not bool(result["double_feasible"]):
            return "validation_failed"
    return "full_validated"


def _individual_from_candidate_row(row: pd.Series) -> Individual:
    chromosome_keys = {
        "M",
        "N",
        "inclination_deg",
        "phase_factor_F",
        "Omega0_deg",
        "u0_deg",
    }
    non_metric_keys = chromosome_keys | {
        "total_satellites",
        "feasible",
        "constraint_violation",
        "generation_evaluated",
        "seed",
        "search_stage",
        "scenario",
        "refinement_iteration",
    }
    candidate = Individual(
        M=int(row["M"]),
        N=int(row["N"]),
        inclination_deg=float(row["inclination_deg"]),
        phase_factor_F=int(row["phase_factor_F"]),
        Omega0_deg=float(row["Omega0_deg"]),
        u0_deg=float(row["u0_deg"]),
    )
    candidate.metrics = {
        key: value
        for key, value in row.items()
        if key not in non_metric_keys and pd.notna(value)
    }
    candidate.feasible = bool(row.get("feasible", False))
    candidate.constraint_violation = float(
        row.get("constraint_violation", math.inf)
    )
    return candidate


def select_full_validated_candidate(
    current_best: Individual,
    candidate_frames: list[pd.DataFrame],
    config: dict[str, Any],
    theta_rad: float,
    scenario: str,
    logger: logging.Logger,
) -> tuple[Individual, dict[str, Any]]:
    """Try progressively less preferred candidates when fine validation fails."""

    if config["mode"] != "full":
        return current_best, {}
    pool = [current_best]
    for frame in candidate_frames:
        if frame.empty or "scenario" not in frame.columns:
            continue
        for _, row in frame[frame["scenario"] == scenario].iterrows():
            candidate = _individual_from_candidate_row(row)
            if candidate.feasible:
                pool.append(candidate)

    unique: dict[tuple[Any, ...], Individual] = {}
    for candidate in pool:
        key = (
            candidate.M,
            candidate.N,
            round(candidate.inclination_deg, 6),
            candidate.phase_factor_F,
            round(candidate.Omega0_deg, 6),
            round(candidate.u0_deg, 6),
        )
        unique.setdefault(key, candidate)
    ranked = sort_population(list(unique.values()), scenario)
    limit = int(config["validation"].get("max_candidates_to_validate", 6))
    first_results: dict[str, Any] = {}
    for candidate_index, candidate in enumerate(ranked[:limit], start=1):
        logger.info(
            "full 稳健候选验证 %s %d/%d：M=%d N=%d S=%d",
            scenario,
            candidate_index,
            min(limit, len(ranked)),
            candidate.M,
            candidate.N,
            candidate.total_satellites,
        )
        results = run_validation_profiles(
            candidate, config, theta_rad, scenario, logger
        )
        if not first_results:
            first_results = results
        if validation_status("full", candidate, scenario, results) == "full_validated":
            return candidate, results
        logger.warning(
            "候选 S=%d 在细化验证下失效，继续检查下一稳健候选。",
            candidate.total_satellites,
        )
    return current_best, first_results


def sensitivity_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    region = config["region"]
    cases: list[dict[str, Any]] = []
    for step in (2.0, 1.0, 0.5):
        cases.append(
            {
                "sensitivity_type": "grid_step_deg",
                "sensitivity_value": step,
                "grid": make_regular_grid(
                    step,
                    tuple(region["latitude_bounds_deg"]),
                    tuple(region["longitude_bounds_deg"]),
                ),
                "simulation_days": 1.0,
                "time_step_seconds": 300.0,
            }
        )
    for time_step in (300.0, 120.0, 60.0, 30.0):
        cases.append(
            {
                "sensitivity_type": "time_step_seconds",
                "sensitivity_value": time_step,
                "grid": make_regular_grid(
                    2.0,
                    tuple(region["latitude_bounds_deg"]),
                    tuple(region["longitude_bounds_deg"]),
                ),
                "simulation_days": 1.0,
                "time_step_seconds": time_step,
            }
        )
    for days in (1.0, 3.0, 7.0):
        cases.append(
            {
                "sensitivity_type": "simulation_days",
                "sensitivity_value": days,
                "grid": make_regular_grid(
                    2.0,
                    tuple(region["latitude_bounds_deg"]),
                    tuple(region["longitude_bounds_deg"]),
                ),
                "simulation_days": days,
                "time_step_seconds": 300.0,
            }
        )
    return cases


def run_sensitivity(
    candidates: dict[str, Individual],
    config: dict[str, Any],
    theta_rad: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cases = sensitivity_cases(config)
    for scenario, candidate in candidates.items():
        constellation = generate_walker_constellation(
            candidate.M,
            candidate.N,
            candidate.inclination_deg,
            candidate.phase_factor_F,
            candidate.Omega0_deg,
            candidate.u0_deg,
        )
        for case_index, case in enumerate(cases, start=1):
            logger.info(
                "敏感性 %s %d/%d：%s=%s",
                scenario,
                case_index,
                len(cases),
                case["sensitivity_type"],
                case["sensitivity_value"],
            )
            result = evaluate_constellation_coverage(
                constellation,
                case["grid"],
                theta_rad,
                case["simulation_days"],
                case["time_step_seconds"],
                method="auto",
            )
            rows.append(
                {
                    "scenario": scenario,
                    "case_index": case_index,
                    "sensitivity_type": case["sensitivity_type"],
                    "sensitivity_value": case["sensitivity_value"],
                    "grid_type": case["grid"].grid_type,
                    "grid_resolution": case["grid"].resolution,
                    "grid_point_count": case["grid"].size,
                    "simulation_days": case["simulation_days"],
                    "time_step_seconds": case["time_step_seconds"],
                    "status": "completed",
                    **result.metrics,
                }
            )
    return pd.DataFrame(rows)


def apply_sensitivity_assessment(
    sensitivity: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Attach actual sensitivity minima to JSON/CSV and flag unstable results."""

    for scenario, payload in payloads.items():
        subset = sensitivity[
            (sensitivity.get("scenario") == scenario)
            & (sensitivity.get("status") == "completed")
        ]
        if subset.empty:
            payload["sensitivity_passed"] = None
            payload["sensitivity_min_Q1"] = None
            payload["sensitivity_min_Q2"] = None
        else:
            min_q1 = float(subset["Q1_full_region"].min())
            min_q2 = float(subset["Q2_full_region"].min())
            max_gap = float(subset["max_gap_time_min"].max())
            passed = min_q1 >= 1.0 - 1e-12 and max_gap <= 1e-12
            if scenario == "double":
                passed = passed and min_q2 >= 0.95 - 1e-12
            payload["sensitivity_passed"] = bool(passed)
            payload["sensitivity_min_Q1"] = min_q1
            payload["sensitivity_min_Q2"] = min_q2
            payload["sensitivity_max_gap_time_min"] = max_gap
            if payload["mode"] != "quick" and not passed:
                payload["validation_status"] = "validation_failed"

        (output_dir / f"best_{scenario}_constellation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metrics_path = output_dir / f"{scenario}_solution_metrics.csv"
        metrics = pd.read_csv(metrics_path)
        metrics["sensitivity_passed"] = payload["sensitivity_passed"]
        metrics["sensitivity_min_Q1"] = payload["sensitivity_min_Q1"]
        metrics["sensitivity_min_Q2"] = payload["sensitivity_min_Q2"]
        metrics["validation_status"] = payload["validation_status"]
        metrics.to_csv(metrics_path, index=False)


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    start_time: datetime,
    elapsed_seconds: float,
    payloads: dict[str, dict[str, Any]],
    completed: bool,
) -> None:
    metadata = {
        "started_at_utc": start_time.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "completed": completed,
        "mode": args.mode,
        "scenario": args.scenario,
        "seed": args.seed,
        "workers": args.workers,
        "resume": args.resume,
        "skip_certification": args.skip_certification,
        "skip_sensitivity": args.skip_sensitivity,
        "config_file": str(config_path_for_mode(args.mode).relative_to(PROJECT_ROOT)),
        "config": config,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "result_status": {
            scenario: payload["validation_status"] for scenario, payload in payloads.items()
        },
        "disclaimer": (
            "数值结果仅在记录的搜索范围、网格、时间采样和随机种子条件下成立；"
            "遗传算法结果不构成严格全局最优证明。"
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    start_clock = time.perf_counter()
    start_time = datetime.now(timezone.utc)
    logger = configure_logging(OUTPUT_DIR)
    config = load_problem2_config(config_path_for_mode(args.mode))
    validate_config(config)
    theta_rad = GROUND_RADIUS_KM / R_EARTH_KM
    grid = simulation_grid(config)
    scenarios = ["single", "double"] if args.scenario == "both" else [args.scenario]
    logger.info(
        "开始问题二：mode=%s, scenario=%s, seed=%d, workers=%d",
        args.mode,
        args.scenario,
        args.seed,
        args.workers,
    )
    logger.info(
        "主搜索网格：%s，点数=%d，时间步长=%.0f s，仿真=%.1f d",
        grid.grid_type,
        grid.size,
        config["simulation"]["time_step_seconds"],
        config["simulation"]["simulation_days"],
    )

    best_candidates: dict[str, Individual] = {}
    detailed_results: dict[str, CoverageEvaluation] = {}
    histories: dict[str, pd.DataFrame] = {}
    all_candidate_frames: list[pd.DataFrame] = []
    payloads: dict[str, dict[str, Any]] = {}
    constellations: dict[str, Any] = {}
    completed = False

    try:
        for scenario_index, scenario in enumerate(scenarios):
            evaluator = make_candidate_evaluator(config, grid, theta_rad, scenario)
            logger.info("运行 %s 场景遗传算法", scenario)
            result = run_multiseed_ga(
                config,
                scenario,
                lambda _seed: evaluator,
                args.seed + scenario_index * 100_000,
                OUTPUT_DIR,
                workers=args.workers,
                resume=args.resume,
            )
            best = result.best
            histories[scenario] = result.history
            result.history.to_csv(
                OUTPUT_DIR / f"ga_{scenario}_generation_history.csv", index=False
            )
            if not result.all_candidates.empty:
                result.all_candidates["search_stage"] = "main_ga"
                result.all_candidates["scenario"] = scenario
                all_candidate_frames.append(result.all_candidates)

            certification_frame = pd.DataFrame()
            if config["certification"]["enabled"] and not args.skip_certification:
                best, certification_frame, certification_candidates = run_fixed_mn_certification(
                    best,
                    config,
                    scenario,
                    grid,
                    theta_rad,
                    args.seed + scenario_index * 100_000,
                    args.workers,
                    OUTPUT_DIR,
                    args.resume,
                    logger,
                )
                if not certification_candidates.empty:
                    certification_candidates["search_stage"] = "fixed_MN_certification"
                    certification_candidates["scenario"] = scenario
                    all_candidate_frames.append(certification_candidates)
            certification_frame.to_csv(
                OUTPUT_DIR / f"fixed_MN_certification_{scenario}.csv", index=False
            )

            if config["local_refinement"]["enabled"]:
                refined, refinement_frame = local_coordinate_refinement(
                    best,
                    evaluator,
                    config["search_bounds"],
                    scenario,
                    config["local_refinement"],
                )
                if is_better(refined, best, scenario):
                    best = refined
                if not refinement_frame.empty:
                    refinement_frame["search_stage"] = "local_refinement"
                    refinement_frame["scenario"] = scenario
                    all_candidate_frames.append(refinement_frame)

            best, validation_results = select_full_validated_candidate(
                best,
                all_candidate_frames,
                config,
                theta_rad,
                scenario,
                logger,
            )
            best, detailed = evaluate_detailed(best, config, grid, theta_rad, scenario)
            status = validation_status(
                config["mode"], best, scenario, validation_results
            )
            constellation = generate_walker_constellation(
                best.M,
                best.N,
                best.inclination_deg,
                best.phase_factor_F,
                best.Omega0_deg,
                best.u0_deg,
            )
            payload = solution_payload(
                scenario,
                best,
                constellation,
                config,
                theta_rad,
                result.seed,
                status,
                validation_results,
            )
            metrics_frame = pd.DataFrame([{**best.record(), "validation_status": status}])
            write_solution_bundle(
                OUTPUT_DIR,
                scenario,
                payload,
                constellation,
                metrics_frame,
                detailed.time_metrics,
                detailed.grid_metrics,
            )
            best_candidates[scenario] = best
            detailed_results[scenario] = detailed
            payloads[scenario] = payload
            constellations[scenario] = constellation
            logger.info(
                "%s 当前最佳：M=%d N=%d S=%d i=%.3f Q1=%.6f Q2=%.6f status=%s",
                scenario,
                best.M,
                best.N,
                best.total_satellites,
                best.inclination_deg,
                best.metrics["Q1_full_region"],
                best.metrics["Q2_full_region"],
                status,
            )

        combined_candidates = (
            pd.concat(all_candidate_frames, ignore_index=True)
            if all_candidate_frames
            else pd.DataFrame()
        )
        for scenario in scenarios:
            scenario_candidates = combined_candidates[
                combined_candidates["scenario"] == scenario
            ] if not combined_candidates.empty else pd.DataFrame()
            scenario_candidates.to_csv(
                OUTPUT_DIR / f"ga_{scenario}_all_candidates.csv", index=False
            )

        costs = build_cost_comparison(payloads)
        costs.to_csv(OUTPUT_DIR / "cost_comparison.csv", index=False)

        if args.skip_sensitivity:
            sensitivity = pd.DataFrame(
                [
                    {
                        "scenario": scenario,
                        "status": "not_run_by_cli_request",
                    }
                    for scenario in scenarios
                ]
            )
        else:
            sensitivity = run_sensitivity(
                best_candidates, config, theta_rad, logger
            )
        sensitivity.to_csv(OUTPUT_DIR / "resolution_sensitivity.csv", index=False)
        apply_sensitivity_assessment(sensitivity, payloads, OUTPUT_DIR)

        plot_region_grid(grid, OUTPUT_DIR)
        preferred_scenario = "single" if "single" in constellations else scenarios[0]
        plot_walker_layout_3d(constellations[preferred_scenario], OUTPUT_DIR)
        for scenario in scenarios:
            plot_ground_tracks(constellations[scenario], scenario, OUTPUT_DIR)
            plot_ga_convergence(histories[scenario], scenario, OUTPUT_DIR)
            plot_solution_grid_fields(
                detailed_results[scenario].grid_metrics, scenario, OUTPUT_DIR
            )
            plot_area_coverage_time(
                detailed_results[scenario].time_metrics, scenario, OUTPUT_DIR
            )
            counts = (
                detailed_results[scenario].worst_single_counts
                if scenario == "single"
                else detailed_results[scenario].worst_double_counts
            )
            plot_worst_snapshot(grid, counts, scenario, OUTPUT_DIR)
        plot_mn_candidates(combined_candidates, OUTPUT_DIR)
        plot_cost_comparison(costs, OUTPUT_DIR)
        plot_resolution_sensitivity(sensitivity, OUTPUT_DIR)
        write_method_notes(OUTPUT_DIR, config, PROJECT_ROOT, set(scenarios))
        completed = True
    except KeyboardInterrupt:
        logger.warning("收到中断信号；遗传算法检查点已保存，可使用 --resume 恢复。")
        raise
    finally:
        elapsed = time.perf_counter() - start_clock
        write_metadata(
            OUTPUT_DIR,
            args,
            config,
            start_time,
            elapsed,
            payloads,
            completed,
        )

    print("\n问题二运行摘要")
    print(f"mode={args.mode}, grid_points={grid.size}, elapsed={elapsed / 60.0:.2f} min")
    for scenario, payload in payloads.items():
        print(
            f"{scenario}: M={payload['M']}, N={payload['N']}, "
            f"S={payload['total_satellites']}, Q1={payload['Q1_full_region']:.6f}, "
            f"Q2={payload['Q2_full_region']:.6f}, status={payload['validation_status']}"
        )
    if args.mode == "quick":
        print("注意：quick 结果仅用于代码冒烟测试，不是论文最终优化结论。")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
