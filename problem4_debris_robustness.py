"""Problem 4: debris avoidance and constellation robustness design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from starlink_modeling.problem4_capacity import (  # noqa: E402
    annual_constellation_distributions,
    evaluate_problem3_service_impact,
    simulate_capacity_timeline,
    summarize_capacity,
)
from starlink_modeling.problem4_debris import distribution_frames  # noqa: E402
from starlink_modeling.problem4_io import (  # noqa: E402
    Problem3SelectionError,
    discover_problem3_standard,
    is_formal_result,
    load_problem4_config,
    utc_now_iso,
)
from starlink_modeling.problem4_monte_carlo import (  # noqa: E402
    RiskSimulation,
    simulate_single_satellite_risk,
)
from starlink_modeling.problem4_redundancy import run_redundancy_search  # noqa: E402
from starlink_modeling.problem4_reporting import (  # noqa: E402
    write_json,
    write_method_notes,
)
from starlink_modeling.problem4_visualization import generate_problem4_figures  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "problem4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题四：碎片规避与星座鲁棒性设计")
    parser.add_argument("--mode", choices=("quick", "standard", "full"), default="standard")
    parser.add_argument("--stage", choices=("risk", "constellation", "redundancy", "all"), default="all")
    parser.add_argument("--problem3-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mc-runs", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no-traffic-impact", action="store_true")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("problem4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "problem4_run.log", mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "debris", "satellite", "velocity", "conjunction", "warning", "uncertainty",
        "avoidance", "capacity", "costs", "reliability", "redundancy", "precision",
    )
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError("问题四配置缺少：" + ", ".join(missing))
    if int(config["precision"]["risk_event_samples"]) <= 0:
        raise ValueError("risk_event_samples 必须为正数。")
    if int(config["reliability"]["mc_years"]) <= 0:
        raise ValueError("reliability.mc_years 必须为正数。")
    if not 0.0 < float(config["avoidance"]["threshold"]) < 1.0:
        raise ValueError("P_th 必须位于 (0,1)。")
    if float(config["avoidance"]["delta_v_max_mps"]) > 1.0 + 1e-12:
        raise ValueError("题目约束要求单次 Delta-v 不超过 1 m/s。")


def _clear_problem4_output() -> None:
    resolved = OUTPUT_DIR.resolve()
    expected_parent = (PROJECT_ROOT / "outputs").resolve()
    if resolved.parent != expected_parent or resolved.name != "problem4":
        raise RuntimeError("拒绝清理非 outputs/problem4 目录。")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def _load_existing_metadata() -> dict[str, Any]:
    path = OUTPUT_DIR / "problem4_run_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_metadata(
    args: argparse.Namespace,
    config: dict[str, Any],
    selected: Any,
    stage_status: dict[str, str],
    started_at: str,
    elapsed_seconds: float,
    *,
    completed: bool,
    error: str | None = None,
) -> None:
    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "elapsed_seconds": elapsed_seconds,
        "completed": bool(completed),
        "mode": args.mode,
        "stage": args.stage,
        "seed": args.seed,
        "workers": args.workers,
        "resume": args.resume,
        "force": args.force,
        "mc_runs_override": args.mc_runs,
        "threshold_override": args.threshold,
        "no_traffic_impact": args.no_traffic_impact,
        "stage_status": stage_status,
        "selected_problem3_source": selected.source_directory,
        "constellation_signature": selected.constellation_signature,
        "M": selected.M,
        "N": selected.N,
        "total_satellites": selected.total_satellites,
        "config": config,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "error": error,
        "formal_result": is_formal_result(args.mode, completed),
        "disclaimer": "Quick 仅用于代码验收，不得作为论文正式数值结论。",
    }
    write_json(OUTPUT_DIR / "problem4_run_metadata.json", payload)


def write_parameter_audit(config: dict[str, Any]) -> None:
    source = pd.read_csv(PROJECT_ROOT / "data" / "external" / "problem4_parameter_sources.csv")
    source.insert(0, "run_mode", config["mode"])
    source.to_csv(OUTPUT_DIR / "problem4_parameter_audit.csv", index=False)


def run_risk_stage(config: dict[str, Any], args: argparse.Namespace, logger: logging.Logger) -> RiskSimulation:
    logger.info("开始单星风险阶段：事件样本数=%d", config["precision"]["risk_event_samples"])
    distributions, checks = distribution_frames(config)
    pd.DataFrame(
        {
            "diameter_m": distributions["diameter_m"],
            "diameter_pdf_per_m": distributions["diameter_pdf_per_m"],
            "survival_probability": distributions["survival_probability"],
        }
    ).to_csv(OUTPUT_DIR / "debris_size_distribution.csv", index=False)
    pd.DataFrame(
        {
            "velocity_km_s": distributions["velocity_km_s"],
            "velocity_pdf_per_km_s": distributions["velocity_pdf_per_km_s"],
        }
    ).to_csv(OUTPUT_DIR / "relative_velocity_distribution.csv", index=False)
    pd.DataFrame(
        {
            "warning_hours": distributions["warning_hours"],
            "warning_pdf_per_hour": distributions["warning_pdf_per_hour"],
        }
    ).to_csv(OUTPUT_DIR / "warning_time_distribution.csv", index=False)
    simulation = simulate_single_satellite_risk(
        config, seed=args.seed, threshold_override=args.threshold
    )
    simulation.summary.update(checks)
    simulation.summary["mode"] = args.mode
    simulation.summary["formal_result"] = is_formal_result(args.mode)
    simulation.consistency["flux_collision_rate"] = simulation.consistency[
        "flux_collision_rate_per_second"
    ]
    simulation.consistency["conjunction_integrated_collision_rate"] = simulation.consistency[
        "conjunction_integrated_collision_rate_per_second"
    ]
    if args.mode in {"standard", "full"} and not simulation.consistency[
        "passes_standard_tolerance"
    ]:
        write_json(OUTPUT_DIR / "risk_model_consistency.json", simulation.consistency)
        raise RuntimeError(
            "风险模型通量一致性误差超过配置上限，停止正式运行："
            f"{simulation.consistency['relative_consistency_error']:.2%}"
        )
    write_json(OUTPUT_DIR / "single_satellite_risk_summary.json", simulation.summary)
    write_json(OUTPUT_DIR / "risk_model_consistency.json", simulation.consistency)
    simulation.threshold_comparison.to_csv(
        OUTPUT_DIR / "single_satellite_threshold_comparison.csv", index=False
    )
    simulation.sensitivity.to_csv(OUTPUT_DIR / "risk_sensitivity.csv", index=False)
    simulation.event_samples.to_csv(OUTPUT_DIR / "collision_event_samples.csv", index=False)
    simulation.maneuver_delta_v.to_csv(
        OUTPUT_DIR / "maneuver_delta_v_distribution.csv", index=False
    )
    simulation.maneuver_duration.to_csv(
        OUTPUT_DIR / "maneuver_duration_distribution.csv", index=False
    )
    logger.info(
        "风险阶段完成：单星年规避 %.6f 次，避撞前年风险 %.6e，避撞后 %.6e",
        simulation.summary["expected_annual_maneuvers"],
        simulation.summary["annual_collision_probability_pre"],
        simulation.summary["annual_collision_probability_post"],
    )
    return simulation


def _load_risk_summary() -> dict[str, Any]:
    return json.loads((OUTPUT_DIR / "single_satellite_risk_summary.json").read_text(encoding="utf-8"))


def run_constellation_stage(
    config: dict[str, Any], args: argparse.Namespace, selected: Any, logger: logging.Logger
) -> None:
    risk = _load_risk_summary()
    durations_frame = pd.read_csv(OUTPUT_DIR / "maneuver_duration_distribution.csv")
    durations = durations_frame["degraded_duration_hours"].to_numpy(float)
    rng = np.random.default_rng(args.seed + 1000)
    mc_years = int(config["reliability"]["mc_years"])
    avoidances, failures = annual_constellation_distributions(
        rng,
        selected.total_satellites,
        risk["expected_annual_maneuvers"],
        risk["annual_failure_hazard"],
        mc_years,
        config["costs"]["maneuver_cost_cny"],
    )
    avoidances.to_csv(OUTPUT_DIR / "annual_avoidance_distribution.csv", index=False)
    failures.to_csv(OUTPUT_DIR / "annual_failure_distribution.csv", index=False)
    timeline = simulate_capacity_timeline(
        rng,
        selected.total_satellites,
        risk["expected_annual_maneuvers"],
        durations,
        risk["annual_failure_hazard"],
        step_hours=1.0,
    )
    timeline.to_csv(OUTPUT_DIR / "capacity_timeseries_sample.csv", index=False)
    capacity_summary = summarize_capacity(
        selected.total_satellites,
        risk["expected_annual_maneuvers"],
        risk["mean_degraded_duration_hours"],
        timeline,
    )
    write_json(OUTPUT_DIR / "capacity_loss_summary.json", capacity_summary)
    constellation_summary = {
        "mode": config["mode"],
        "total_satellites": selected.total_satellites,
        "mc_years": mc_years,
        "mean_annual_avoidance_count": float(avoidances["annual_avoidance_count"].mean()),
        "std_annual_avoidance_count": float(avoidances["annual_avoidance_count"].std(ddof=1)),
        "q05_annual_avoidance_count": float(avoidances["annual_avoidance_count"].quantile(0.05)),
        "median_annual_avoidance_count": float(avoidances["annual_avoidance_count"].median()),
        "q95_annual_avoidance_count": float(avoidances["annual_avoidance_count"].quantile(0.95)),
        "mean_annual_avoidance_cost_cny": float(avoidances["annual_avoidance_cost_cny"].mean()),
        "std_annual_avoidance_cost_cny": float(avoidances["annual_avoidance_cost_cny"].std(ddof=1)),
        "expected_annual_failure_count": selected.total_satellites * risk["annual_failure_probability"],
        "monte_carlo_mean_annual_failure_count": float(failures["annual_failure_count"].mean()),
        "poisson_failure_hazard_mean": selected.total_satellites * risk["annual_failure_hazard"],
    }
    write_json(OUTPUT_DIR / "constellation_annual_summary.json", constellation_summary)
    thresholds = pd.read_csv(OUTPUT_DIR / "single_satellite_threshold_comparison.csv")
    thresholds["constellation_expected_annual_maneuvers"] = (
        thresholds["expected_annual_maneuvers"] * selected.total_satellites
    )
    thresholds["annual_avoidance_cost_cny"] = (
        thresholds["constellation_expected_annual_maneuvers"]
        * config["costs"]["maneuver_cost_cny"]
    )
    thresholds["analytical_capacity_loss_ratio"] = 0.5 * (
        1.0
        - np.exp(
            -thresholds["expected_annual_maneuvers"]
            * thresholds["mean_degraded_duration_hours"]
            / 8760.0
        )
    )
    thresholds.to_csv(OUTPUT_DIR / "threshold_cost_capacity_tradeoff.csv", index=False)
    if args.no_traffic_impact:
        logger.warning("按 --no-traffic-impact 跳过问题三 LP 复算。")
        pd.DataFrame(
            columns=[
                "demand_level", "time_seconds", "scenario", "state_basis", "throughput_gbps",
                "service_ratio", "blocked_gbps", "maximum_utilization",
            ]
        ).to_csv(OUTPUT_DIR / "problem3_service_impact.csv", index=False)
    else:
        service = evaluate_problem3_service_impact(
            PROJECT_ROOT,
            selected,
            risk["expected_annual_maneuvers"],
            risk["mean_degraded_duration_hours"],
            risk["annual_failure_hazard"],
            grid_step_deg=config["traffic_impact"]["grid_step_deg"],
        )
        service.to_csv(OUTPUT_DIR / "problem3_service_impact.csv", index=False)
    logger.info(
        "星座阶段完成：年均避撞 %.2f 次，年均成本 %.4f 亿元",
        constellation_summary["mean_annual_avoidance_count"],
        constellation_summary["mean_annual_avoidance_cost_cny"] / 1e8,
    )


def run_redundancy_stage(
    config: dict[str, Any], args: argparse.Namespace, selected: Any, logger: logging.Logger
) -> None:
    risk = _load_risk_summary()
    durations = pd.read_csv(OUTPUT_DIR / "maneuver_duration_distribution.csv")[
        "degraded_duration_hours"
    ].to_numpy(float)
    mc_years = int(config["reliability"]["mc_years"])
    logger.info(
        "开始冗余搜索：k<=%d, 额外轨道面<=%d, MC=%d",
        config["redundancy"]["max_k_per_plane"],
        config["redundancy"]["max_extra_planes"],
        mc_years,
    )
    (
        candidates,
        summary,
        years,
        cost_summary,
        best,
        pareto,
        extra_parameters,
    ) = run_redundancy_search(
        selected,
        risk,
        durations,
        config,
        seed=args.seed + 2000,
        mc_years=mc_years,
    )
    candidates.to_csv(OUTPUT_DIR / "redundancy_candidates.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "redundancy_monte_carlo_summary.csv", index=False)
    years.to_csv(OUTPUT_DIR / "redundancy_year_samples.csv", index=False)
    cost_summary.to_csv(OUTPUT_DIR / "redundancy_cost_summary.csv", index=False)
    pareto.to_csv(OUTPUT_DIR / "redundancy_pareto_front.csv", index=False)
    extra_parameters.to_csv(OUTPUT_DIR / "extra_plane_parameters.csv", index=False)
    write_json(OUTPUT_DIR / "best_redundancy_solution.json", best)
    launch_events = years[years["replenishment_launch_count"] > 0][
        ["candidate_id", "mc_year", "ground_spares_used", "replenishment_launch_count"]
    ].copy()
    launch_events.to_csv(OUTPUT_DIR / "ground_spare_launch_events.csv", index=False)

    timeline = pd.read_csv(OUTPUT_DIR / "capacity_timeseries_sample.csv")
    recommended = best["recommended_candidate"]
    rng = np.random.default_rng(args.seed + 3000)
    indicator = np.ones(len(timeline), dtype=int)
    outage_samples = int(round((1.0 - recommended["mean_effective_coverage_availability"]) * len(timeline)))
    if outage_samples > 0:
        indices = rng.choice(len(timeline), size=min(len(timeline), outage_samples), replace=False)
        indicator[indices] = 0
    timeline["basic_coverage_indicator"] = indicator
    timeline["recommended_candidate_id"] = int(recommended["candidate_id"])
    timeline.to_csv(OUTPUT_DIR / "example_degradation_timeline.csv", index=False)
    logger.info(
        "冗余阶段完成：推荐 candidate=%d, mean=%.6f, q05=%.6f, 五年成本=%.4f亿元",
        recommended["candidate_id"],
        recommended["mean_effective_coverage_availability"],
        recommended["q05_effective_coverage_availability"],
        recommended["five_year_expected_cost_cny"] / 1e8,
    )


def _stage_ready(stage: str) -> bool:
    required = {
        "risk": OUTPUT_DIR / "single_satellite_risk_summary.json",
        "constellation": OUTPUT_DIR / "constellation_annual_summary.json",
        "redundancy": OUTPUT_DIR / "best_redundancy_solution.json",
    }
    return required[stage].exists()


def main() -> None:
    args = parse_args()
    if args.force:
        _clear_problem4_output()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(OUTPUT_DIR)
    started_at = utc_now_iso()
    start = time.perf_counter()
    config = load_problem4_config(PROJECT_ROOT, args.mode, args.config)
    if args.mc_runs is not None:
        if args.mc_runs <= 0:
            raise ValueError("--mc-runs 必须为正数。")
        config["reliability"]["mc_years"] = int(args.mc_runs)
    if args.threshold is not None:
        config["avoidance"]["threshold"] = float(args.threshold)
    validate_config(config)
    selected = discover_problem3_standard(PROJECT_ROOT, args.problem3_dir)
    print("问题三结果候选审计：")
    for candidate in selected.candidate_audit:
        print(
            f"- {candidate['metadata_file']}: mode={candidate['mode']}, "
            f"completed={candidate['completed']}, eligible={candidate['eligible']}, "
            f"stage_status={candidate['stage_status']}"
        )
    print(f"最终选择：{selected.source_directory}")
    print(
        f"星座 M={selected.M}, N={selected.N}, S={selected.total_satellites}, "
        f"i={selected.inclination_deg:.8f}°, F={selected.phase_factor_F}"
    )
    write_json(OUTPUT_DIR / "selected_problem3_standard.json", selected.to_dict())
    write_json(OUTPUT_DIR / "problem4_config_used.json", config)
    write_parameter_audit(config)
    existing = _load_existing_metadata()
    stage_status = dict(existing.get("stage_status") or {}) if existing.get("mode") == args.mode else {}
    for name in ("risk", "constellation", "redundancy"):
        stage_status.setdefault(name, "pending")
    requested = (
        ["risk", "constellation", "redundancy"]
        if args.stage == "all"
        else [args.stage]
    )
    # Stages consume persisted predecessor outputs; missing dependencies are run.
    existing_mode_matches = existing.get("mode") == args.mode
    if any(stage in requested for stage in ("constellation", "redundancy")) and not (
        existing_mode_matches and _stage_ready("risk")
    ):
        requested.insert(0, "risk")
    if "redundancy" in requested and not (
        existing_mode_matches and _stage_ready("constellation")
    ):
        requested.insert(requested.index("redundancy"), "constellation")
    requested = list(dict.fromkeys(requested))
    try:
        for stage in requested:
            if args.resume and _stage_ready(stage) and stage_status.get(stage) == "completed":
                logger.info("--resume：跳过已完成阶段 %s", stage)
                continue
            if stage == "risk":
                run_risk_stage(config, args, logger)
            elif stage == "constellation":
                run_constellation_stage(config, args, selected, logger)
            else:
                run_redundancy_stage(config, args, selected, logger)
            stage_status[stage] = "completed"
            _write_metadata(
                args, config, selected, stage_status, started_at,
                time.perf_counter() - start, completed=all(value == "completed" for value in stage_status.values())
            )
        all_complete = all(stage_status.get(name) == "completed" for name in ("risk", "constellation", "redundancy"))
        if all_complete and not args.no_traffic_impact:
            generate_problem4_figures(OUTPUT_DIR, config, selected.total_satellites)
            write_method_notes(PROJECT_ROOT, OUTPUT_DIR, config, selected)
        _write_metadata(
            args, config, selected, stage_status, started_at,
            time.perf_counter() - start, completed=all_complete
        )
    except KeyboardInterrupt:
        logger.warning("用户中断，已保存完成阶段和元数据。")
        _write_metadata(
            args, config, selected, stage_status, started_at,
            time.perf_counter() - start, completed=False, error="KeyboardInterrupt"
        )
        raise
    except Exception as exc:
        logger.exception("问题四运行失败：%s", exc)
        _write_metadata(
            args, config, selected, stage_status, started_at,
            time.perf_counter() - start, completed=False, error=str(exc)
        )
        raise
    if all_complete:
        risk = _load_risk_summary()
        constellation = json.loads((OUTPUT_DIR / "constellation_annual_summary.json").read_text(encoding="utf-8"))
        best = json.loads((OUTPUT_DIR / "best_redundancy_solution.json").read_text(encoding="utf-8"))["recommended_candidate"]
        print("\n问题四摘要")
        print(f"单星年碰撞概率：{risk['annual_collision_probability_pre']:.6e}")
        print(f"避撞后年碰撞概率：{risk['annual_collision_probability_post']:.6e}")
        print(f"单星年规避次数：{risk['expected_annual_maneuvers']:.6f}")
        print(f"星座年规避次数：{constellation['mean_annual_avoidance_count']:.3f}")
        print(
            f"推荐方案：k={best['k_per_plane']}, r={best['extra_plane_count']}, "
            f"G={best['ground_spare_count']}, mean={best['mean_effective_coverage_availability']:.6%}, "
            f"q05={best['q05_effective_coverage_availability']:.6%}"
        )


if __name__ == "__main__":
    main()
