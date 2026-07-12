"""Redundancy deployment search, life-cycle cost, and Pareto filtering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .problem4_reliability import (
    CoverageArchive,
    ExtraPlane,
    build_coverage_archive,
    simulate_reliability_year,
)


def launch_count(satellite_count: int, launch_capacity: int = 60) -> int:
    if satellite_count < 0 or launch_capacity <= 0:
        raise ValueError("Satellite count and launch capacity are invalid.")
    return int(math.ceil(int(satellite_count) / int(launch_capacity))) if satellite_count else 0


def redundancy_costs(
    *,
    base_satellites: int,
    in_plane_spares: int,
    extra_plane_satellites: int,
    ground_spares: int,
    annual_avoidances_per_satellite: float,
    annual_failures: float,
    expected_replenishment_launches: float,
    satellite_cost_cny: float,
    launch_cost_cny: float,
    maneuver_cost_cny: float,
    launch_capacity: int,
    design_life_years: float,
) -> dict[str, float | int]:
    initial_in_orbit = int(in_plane_spares + extra_plane_satellites)
    added_satellites = initial_in_orbit + int(ground_spares)
    initial_launches = launch_count(initial_in_orbit, launch_capacity)
    initial_manufacture = added_satellites * float(satellite_cost_cny)
    initial_launch_cost = initial_launches * float(launch_cost_cny)
    active_in_orbit = base_satellites + initial_in_orbit
    annual_avoidance_cost = (
        active_in_orbit * annual_avoidances_per_satellite * float(maneuver_cost_cny)
    )
    annual_replenishment_launch_cost = expected_replenishment_launches * float(launch_cost_cny)
    annual_replacement_manufacture = min(float(ground_spares), annual_failures) * float(satellite_cost_cny)
    initial_cost = initial_manufacture + initial_launch_cost
    five_year_cost = initial_cost + float(design_life_years) * (
        annual_avoidance_cost
        + annual_replenishment_launch_cost
        + annual_replacement_manufacture
    )
    return {
        "added_satellites": added_satellites,
        "added_in_orbit_satellites": initial_in_orbit,
        "ground_inventory_satellites": int(ground_spares),
        "initial_launch_count": initial_launches,
        "initial_manufacturing_cost_cny": initial_manufacture,
        "initial_launch_cost_cny": initial_launch_cost,
        "initial_cost_cny": initial_cost,
        "annual_avoidance_cost_cny": annual_avoidance_cost,
        "annual_expected_replenishment_launch_cost_cny": annual_replenishment_launch_cost,
        "annual_expected_replacement_manufacturing_cost_cny": annual_replacement_manufacture,
        "five_year_expected_cost_cny": five_year_cost,
    }


def _largest_raan_gap_midpoint(existing_raan_deg: list[float]) -> float:
    values = np.sort(np.mod(np.asarray(existing_raan_deg, dtype=float), 360.0))
    extended = np.append(values, values[0] + 360.0)
    gaps = np.diff(extended)
    index = int(np.argmax(gaps))
    return float((extended[index] + gaps[index] / 2.0) % 360.0)


def optimize_extra_planes(
    selected: Any,
    max_extra_planes: int,
    *,
    grid_step_deg: float,
    representative_hours: float,
    time_step_minutes: float,
    phase_candidate_count: int,
) -> tuple[dict[int, CoverageArchive], pd.DataFrame]:
    """Greedily place each new plane in the current largest RAAN gap."""

    archives: dict[int, CoverageArchive] = {
        0: build_coverage_archive(
            selected,
            grid_step_deg=grid_step_deg,
            representative_hours=representative_hours,
            time_step_minutes=time_step_minutes,
        )
    }
    chosen: list[ExtraPlane] = []
    existing_raan = [float(value) for value in np.linspace(0.0, 360.0, selected.M, endpoint=False) + selected.Omega0_deg]
    rows: list[dict[str, Any]] = []
    phase_span = 360.0 / selected.N
    phases = np.linspace(0.0, phase_span, max(1, int(phase_candidate_count)), endpoint=False)
    for extra_index in range(1, int(max_extra_planes) + 1):
        raan = _largest_raan_gap_midpoint(existing_raan)
        best_archive = None
        best_plane = None
        best_score = None
        for phase in phases:
            plane = ExtraPlane(extra_index, raan, float(phase), selected.N)
            archive = build_coverage_archive(
                selected,
                grid_step_deg=grid_step_deg,
                representative_hours=representative_hours,
                time_step_minutes=time_step_minutes,
                extra_planes=chosen + [plane],
            )
            score = (
                archive.baseline_effective_availability,
                archive.minimum_coverage_multiplicity,
                -archive.critical_maneuver_fraction,
                archive.mean_coverage_multiplicity,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_archive = archive
                best_plane = plane
        assert best_archive is not None and best_plane is not None
        chosen.append(best_plane)
        existing_raan.append(raan)
        archives[extra_index] = best_archive
        rows.append(
            {
                "extra_plane_count": extra_index,
                "extra_plane_index": best_plane.extra_plane_index,
                "raan_deg": best_plane.raan_deg,
                "phase_offset_deg": best_plane.phase_offset_deg,
                "satellite_count": best_plane.satellite_count,
                "baseline_coverage_availability": best_archive.baseline_effective_availability,
                "minimum_coverage_multiplicity": best_archive.minimum_coverage_multiplicity,
                "mean_coverage_multiplicity": best_archive.mean_coverage_multiplicity,
                "critical_maneuver_fraction": best_archive.critical_maneuver_fraction,
            }
        )
    return archives, pd.DataFrame(rows)


def _scheme_type(k: int, r: int, ground: int) -> str:
    active = sum(value > 0 for value in (k, r, ground))
    if active == 0:
        return "无冗余"
    if active > 1:
        return "混合方案"
    if k:
        return "每轨在轨备用"
    if r:
        return "额外轨道面"
    return "地面备用"


def meets_coverage_target(
    mean_availability: float,
    q05_availability: float,
    target: float = 0.99,
) -> tuple[bool, bool]:
    """Classify mean and robust targets; robust requires both criteria."""

    mean_ok = float(mean_availability) >= float(target)
    robust_ok = mean_ok and float(q05_availability) >= float(target)
    return mean_ok, robust_ok


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    keep = np.ones(len(frame), dtype=bool)
    cost = frame["five_year_expected_cost_cny"].to_numpy(float)
    availability = frame["mean_effective_coverage_availability"].to_numpy(float)
    robust = frame["q05_effective_coverage_availability"].to_numpy(float)
    for index in range(len(frame)):
        dominates = (
            (cost <= cost[index])
            & (availability >= availability[index])
            & (robust >= robust[index])
            & (
                (cost < cost[index])
                | (availability > availability[index])
                | (robust > robust[index])
            )
        )
        dominates[index] = False
        if np.any(dominates):
            keep[index] = False
    return frame.loc[keep].sort_values(
        ["five_year_expected_cost_cny", "q05_effective_coverage_availability"],
        ascending=[True, False],
    ).reset_index(drop=True)


def run_redundancy_search(
    selected: Any,
    risk_summary: dict[str, Any],
    degraded_duration_samples_hours: np.ndarray,
    config: dict[str, Any],
    *,
    seed: int,
    mc_years: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    reliability = config["reliability"]
    redundancy = config["redundancy"]
    costs = config["costs"]
    archives, extra_parameters = optimize_extra_planes(
        selected,
        int(redundancy["max_extra_planes"]),
        grid_step_deg=float(reliability["coverage_grid_step_deg"]),
        representative_hours=float(reliability["representative_hours"]),
        time_step_minutes=float(reliability["coverage_time_step_minutes"]),
        phase_candidate_count=int(redundancy["extra_plane_phase_candidates"]),
    )
    values_k = range(0, int(redundancy["max_k_per_plane"]) + 1)
    values_r = range(0, int(redundancy["max_extra_planes"]) + 1)
    values_ground = [int(value) for value in redundancy["ground_spare_values"]]
    annual_maneuvers = float(risk_summary["expected_annual_maneuvers"])
    annual_failure_hazard = float(risk_summary["annual_failure_hazard"])
    rng_master = np.random.default_rng(int(seed))
    summary_rows: list[dict[str, Any]] = []
    year_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    candidate_id = 0
    for k in values_k:
        for r in values_r:
            for ground in values_ground:
                archive = archives[r]
                candidate_seed = int(rng_master.integers(0, np.iinfo(np.int32).max))
                rng = np.random.default_rng(candidate_seed)
                year_rows = [
                    simulate_reliability_year(
                        rng,
                        archive,
                        base_M=selected.M,
                        base_N=selected.N,
                        in_plane_spares_per_plane=k,
                        ground_spares=ground,
                        annual_maneuvers_per_satellite=annual_maneuvers,
                        annual_failure_hazard_per_satellite=annual_failure_hazard,
                        degraded_duration_samples_hours=degraded_duration_samples_hours,
                        adjustment_hours=float(reliability["in_plane_adjustment_days"]) * 24.0,
                        takeover_hours=float(redundancy["takeover_hours"]),
                        ground_replacement_hours=float(redundancy["ground_replacement_days"]) * 24.0,
                        standby_risk_factor=float(redundancy["standby_risk_factor"]),
                    )
                    for _ in range(int(mc_years))
                ]
                years = pd.DataFrame(year_rows)
                years.insert(0, "mc_year", np.arange(int(mc_years)))
                years.insert(0, "candidate_id", candidate_id)
                year_frames.append(years)
                mean_availability = float(years["effective_coverage_availability"].mean())
                q05_availability = float(years["effective_coverage_availability"].quantile(0.05))
                mean_ok, robust_ok = meets_coverage_target(
                    mean_availability, q05_availability
                )
                expected_launches = float(years["replenishment_launch_count"].mean())
                annual_failures = float(years["annual_failure_count"].mean())
                cost = redundancy_costs(
                    base_satellites=selected.total_satellites,
                    in_plane_spares=selected.M * k,
                    extra_plane_satellites=selected.N * r,
                    ground_spares=ground,
                    annual_avoidances_per_satellite=annual_maneuvers,
                    annual_failures=annual_failures,
                    expected_replenishment_launches=expected_launches,
                    satellite_cost_cny=float(costs["satellite_manufacturing_cost_cny"]),
                    launch_cost_cny=float(costs["launch_cost_cny"]),
                    maneuver_cost_cny=float(costs["maneuver_cost_cny"]),
                    launch_capacity=int(costs["launch_capacity_satellites"]),
                    design_life_years=float(costs["design_life_years"]),
                )
                record = {
                    "candidate_id": candidate_id,
                    "scheme_type": _scheme_type(k, r, ground),
                    "k_per_plane": k,
                    "extra_plane_count": r,
                    "ground_spare_count": ground,
                    "candidate_seed": candidate_seed,
                    "mc_years": int(mc_years),
                    "mean_effective_coverage_availability": mean_availability,
                    "q05_effective_coverage_availability": q05_availability,
                    "mean_geometric_coverage_availability": float(
                        years["geometric_coverage_availability"].mean()
                    ),
                    "mean_capacity_availability": float(years["capacity_availability"].mean()),
                    "mean_annual_avoidance_count": float(years["annual_avoidance_count"].mean()),
                    "mean_annual_failure_count": annual_failures,
                    "mean_replenishment_launch_count": expected_launches,
                    "meets_mean_99": mean_ok,
                    "meets_robust_99": robust_ok,
                    "critical_maneuver_fraction": archive.critical_maneuver_fraction,
                    "critical_failure_fraction": archive.critical_failure_fraction,
                    **cost,
                }
                summary_rows.append(record)
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "scheme_type": record["scheme_type"],
                        "k_per_plane": k,
                        "extra_plane_count": r,
                        "ground_spare_count": ground,
                        "added_satellites": cost["added_satellites"],
                    }
                )
                candidate_id += 1
    summary = pd.DataFrame(summary_rows)
    years_all = pd.concat(year_frames, ignore_index=True)
    candidates = pd.DataFrame(candidate_rows)
    feasible = summary[summary["meets_robust_99"]].copy()
    criterion = "robust_99"
    if feasible.empty:
        feasible = summary[summary["meets_mean_99"]].copy()
        criterion = "mean_99"
    if feasible.empty:
        feasible = summary.copy()
        criterion = "best_available_not_99"
    feasible = feasible.sort_values(
        [
            "five_year_expected_cost_cny",
            "q05_effective_coverage_availability",
            "added_in_orbit_satellites",
            "initial_launch_count",
            "annual_avoidance_cost_cny",
        ],
        ascending=[True, False, True, True, True],
    )
    best_row = feasible.iloc[0].to_dict()
    best = {
        "selection_criterion": criterion,
        "recommended_candidate": best_row,
        "pure_scheme_minimums": {},
    }
    for scheme in ("每轨在轨备用", "额外轨道面", "地面备用"):
        subset = summary[(summary["scheme_type"] == scheme) & summary["meets_mean_99"]]
        best["pure_scheme_minimums"][scheme] = (
            None
            if subset.empty
            else subset.sort_values("five_year_expected_cost_cny").iloc[0].to_dict()
        )
    base = summary[summary["scheme_type"] == "无冗余"].iloc[0]
    best["no_redundancy"] = base.to_dict()
    pareto = pareto_front(summary)
    cost_summary = summary[
        [
            "candidate_id", "scheme_type", "k_per_plane", "extra_plane_count",
            "ground_spare_count", "initial_cost_cny", "annual_avoidance_cost_cny",
            "five_year_expected_cost_cny", "initial_launch_count",
        ]
    ].copy()
    return candidates, summary, years_all, cost_summary, best, pareto, extra_parameters
