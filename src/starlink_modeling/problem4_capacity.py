"""Maneuver-duration, constellation-capacity, and Problem-three service impacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .problem2_grid import make_regular_grid
from .problem2_orbit import eci_to_ecef, generate_walker_constellation, propagate_constellation
from .problem3_routing import visibility_matrix
from .problem3_traffic import fair_access_allocation, spatial_ground_demand


HOURS_PER_YEAR = 365.0 * 24.0
NORMAL_CAPACITY_GBPS = 20.0
MANEUVER_CAPACITY_GBPS = 10.0
FAILED_CAPACITY_GBPS = 0.0


def burn_time_seconds(mass_kg: float, delta_v_mps: np.ndarray | float, thrust_n: float) -> np.ndarray:
    if mass_kg <= 0.0 or thrust_n <= 0.0:
        raise ValueError("Mass and thrust must be positive.")
    delta_v = np.asarray(delta_v_mps, dtype=float)
    if np.any(delta_v < 0.0):
        raise ValueError("Delta-v must be nonnegative.")
    return float(mass_kg) * delta_v / float(thrust_n)


def degraded_duration_hours(
    mass_kg: float,
    delta_v_mps: np.ndarray | float,
    thrust_n: float,
    overhead_hours: float,
) -> np.ndarray:
    if overhead_hours < 0.0:
        raise ValueError("Recovery overhead must be nonnegative.")
    return float(overhead_hours) + burn_time_seconds(mass_kg, delta_v_mps, thrust_n) / 3600.0


def satellite_capacity_gbps(state: np.ndarray | float) -> np.ndarray:
    """Map state factors 1, 0.5, and 0 to the inherited 20 Gbps limit."""

    factors = np.asarray(state, dtype=float)
    if np.any((factors < 0.0) | (factors > 1.0)):
        raise ValueError("Capacity state factors must lie in [0, 1].")
    return NORMAL_CAPACITY_GBPS * factors


def analytical_maneuver_capacity_loss_ratio(
    annual_maneuvers_per_satellite: float,
    mean_degraded_hours: float,
    capacity_factor_during_maneuver: float = 0.5,
) -> float:
    occupancy = 1.0 - math.exp(
        -float(annual_maneuvers_per_satellite) * float(mean_degraded_hours) / HOURS_PER_YEAR
    )
    return float((1.0 - capacity_factor_during_maneuver) * occupancy)


def annual_constellation_distributions(
    rng: np.random.Generator,
    total_satellites: int,
    annual_maneuvers_per_satellite: float,
    annual_failure_hazard_per_satellite: float,
    mc_years: int,
    maneuver_cost_cny: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    avoidances = rng.poisson(
        max(0.0, total_satellites * annual_maneuvers_per_satellite), size=int(mc_years)
    )
    failures = rng.poisson(
        max(0.0, total_satellites * annual_failure_hazard_per_satellite), size=int(mc_years)
    )
    return (
        pd.DataFrame(
            {
                "mc_year": np.arange(int(mc_years)),
                "annual_avoidance_count": avoidances,
                "annual_avoidance_cost_cny": avoidances * float(maneuver_cost_cny),
            }
        ),
        pd.DataFrame(
            {
                "mc_year": np.arange(int(mc_years)),
                "annual_failure_count": failures,
            }
        ),
    )


def simulate_capacity_timeline(
    rng: np.random.Generator,
    total_satellites: int,
    annual_maneuvers_per_satellite: float,
    degraded_duration_samples_hours: np.ndarray,
    annual_failure_hazard_per_satellite: float,
    *,
    step_hours: float = 1.0,
) -> pd.DataFrame:
    """Generate one example year without expanding a satellite-by-time cube."""

    sample_count = int(math.ceil(HOURS_PER_YEAR / float(step_hours)))
    time_hours = np.arange(sample_count, dtype=float) * float(step_hours)
    maneuver_delta = np.zeros(sample_count + 1, dtype=float)
    maneuver_partial_hours = np.zeros(sample_count, dtype=float)
    maneuver_count = rng.poisson(total_satellites * annual_maneuvers_per_satellite)
    durations = np.asarray(degraded_duration_samples_hours, dtype=float)
    if durations.size == 0:
        durations = np.array([1.0])
    for _ in range(int(maneuver_count)):
        start_hour = float(rng.uniform(0.0, HOURS_PER_YEAR))
        duration = float(rng.choice(durations))
        stop_hour = min(HOURS_PER_YEAR, start_hour + duration)
        start_index = min(sample_count - 1, int(start_hour // step_hours))
        last_index = min(
            sample_count - 1,
            int(np.nextafter(stop_hour, -np.inf) // step_hours),
        )
        if start_index == last_index:
            maneuver_partial_hours[start_index] += stop_hour - start_hour
            continue
        first_boundary = (start_index + 1) * step_hours
        maneuver_partial_hours[start_index] += first_boundary - start_hour
        last_boundary = last_index * step_hours
        maneuver_partial_hours[last_index] += stop_hour - last_boundary
        full_start = start_index + 1
        full_stop = last_index
        if full_start < full_stop:
            maneuver_delta[full_start] += 1.0
            maneuver_delta[full_stop] -= 1.0
    full_bin_counts = np.cumsum(maneuver_delta[:-1])
    maneuvering = np.clip(
        full_bin_counts + maneuver_partial_hours / float(step_hours),
        0.0,
        total_satellites,
    )

    failure_delta = np.zeros(sample_count + 1, dtype=float)
    failure_count = rng.poisson(total_satellites * annual_failure_hazard_per_satellite)
    if failure_count:
        starts = rng.integers(0, sample_count, size=int(failure_count))
        np.add.at(failure_delta, starts, 1.0)
    failed = np.clip(np.cumsum(failure_delta[:-1]), 0.0, total_satellites)
    nominal = total_satellites * NORMAL_CAPACITY_GBPS
    available = np.clip(
        nominal - maneuvering * (NORMAL_CAPACITY_GBPS - MANEUVER_CAPACITY_GBPS)
        - failed * NORMAL_CAPACITY_GBPS,
        0.0,
        nominal,
    )
    return pd.DataFrame(
        {
            "time_hours": time_hours,
            "maneuvering_satellite_count": maneuvering,
            "failed_satellite_count": failed,
            "available_capacity_gbps": available,
            "available_capacity_ratio": available / nominal,
        }
    )


def summarize_capacity(
    total_satellites: int,
    annual_maneuvers_per_satellite: float,
    mean_degraded_hours: float,
    timeline: pd.DataFrame,
) -> dict[str, Any]:
    nominal_gbps = total_satellites * NORMAL_CAPACITY_GBPS
    analytic_loss = analytical_maneuver_capacity_loss_ratio(
        annual_maneuvers_per_satellite, mean_degraded_hours
    )
    mc_total_loss = 1.0 - float(timeline["available_capacity_ratio"].mean())
    mc_maneuver_only = float(
        timeline["maneuvering_satellite_count"].mean()
        * (NORMAL_CAPACITY_GBPS - MANEUVER_CAPACITY_GBPS)
        / nominal_gbps
    )
    return {
        "nominal_total_access_capacity_gbps": nominal_gbps,
        "maneuver_analytical_capacity_loss_ratio": analytic_loss,
        "maneuver_mc_capacity_loss_ratio": mc_maneuver_only,
        "total_mc_capacity_loss_ratio_including_failures": mc_total_loss,
        "annual_equivalent_maneuver_capacity_loss_gbps_h": nominal_gbps
        * HOURS_PER_YEAR
        * analytic_loss,
        "annual_equivalent_maneuver_capacity_loss_tbps_h": nominal_gbps
        * HOURS_PER_YEAR
        * analytic_loss
        / 1000.0,
        "analytical_mc_relative_error": abs(mc_maneuver_only - analytic_loss)
        / max(analytic_loss, 1e-15),
    }


def _constellation_from_selected(selected: Any):
    return generate_walker_constellation(
        selected.M,
        selected.N,
        selected.inclination_deg,
        selected.phase_factor_F,
        selected.Omega0_deg,
        selected.u0_deg,
        selected.altitude_km,
    )


def evaluate_problem3_service_impact(
    project_root: Path,
    selected: Any,
    annual_maneuvers_per_satellite: float,
    mean_degraded_hours: float,
    annual_failure_hazard_per_satellite: float,
    *,
    grid_step_deg: float,
) -> pd.DataFrame:
    """Re-run the inherited three-stage LP for representative Standard demands."""

    source_dir = project_root / selected.source_directory
    traffic = pd.read_csv(source_dir / "traffic_timeseries.csv")
    mean_demand = float(traffic["satellite_demand_gbps"].mean())
    targets = {
        "mean": int((traffic["satellite_demand_gbps"] - mean_demand).abs().idxmin()),
        "peak": int(traffic["satellite_demand_gbps"].idxmax()),
    }
    load_frame = pd.read_csv(source_dir / "satellite_load_timeseries.csv")
    load_rank = (
        load_frame.groupby("satellite_id")["optimized_load_gbps"]
        .mean()
        .sort_values(ascending=False)
        .index.to_numpy(dtype=int)
    )
    constellation = _constellation_from_selected(selected)
    grid = make_regular_grid(float(grid_step_deg))
    concurrent_mean = selected.total_satellites * (
        1.0 - math.exp(-annual_maneuvers_per_satellite * mean_degraded_hours / HOURS_PER_YEAR)
    )
    maneuver_count = int(poisson.ppf(0.95, concurrent_mean)) if concurrent_mean > 0.0 else 0
    failure_mean = selected.total_satellites * annual_failure_hazard_per_satellite
    failure_count = int(poisson.ppf(0.95, failure_mean)) if failure_mean > 0.0 else 0
    # Include one conditional failure stress state even when the unconditional
    # annual 95th percentile is zero; the basis is recorded in the output.
    conditional_failure_count = max(1, failure_count)
    rows: list[dict[str, Any]] = []
    for demand_level, row_index in targets.items():
        source_row = traffic.loc[row_index]
        time_seconds = float(source_row["time_seconds"])
        eci = propagate_constellation(constellation, time_seconds)
        ecef = eci_to_ecef(eci, time_seconds)[0]
        visibility, slant = visibility_matrix(
            grid.unit_vectors, ecef, math.radians(selected.theta_deg)
        )
        demand = spatial_ground_demand(float(source_row["satellite_demand_gbps"]), grid.weights)
        scenarios: dict[str, tuple[np.ndarray, str]] = {}
        normal = np.full(selected.total_satellites, NORMAL_CAPACITY_GBPS)
        scenarios["正常基准"] = (normal.copy(), "Standard代表时刻")
        avoidance = normal.copy()
        if maneuver_count:
            avoidance[load_rank[:maneuver_count]] = MANEUVER_CAPACITY_GBPS
        scenarios["仅避撞"] = (avoidance, "并发机动数取泊松95%分位")
        combined = avoidance.copy()
        failed_ids = load_rank[maneuver_count : maneuver_count + conditional_failure_count]
        combined[failed_ids] = FAILED_CAPACITY_GBPS
        scenarios["避撞加失效"] = (combined, "至少一颗失效的条件压力情景")
        restored = avoidance.copy()
        scenarios["在轨备用接替"] = (restored, "失效位置已由在轨备用恢复")
        baseline = None
        for scenario, (capacities, state_basis) in scenarios.items():
            allocation = fair_access_allocation(
                demand,
                visibility,
                capacity_gbps=capacities,
                assignment_cost_ms=slant / 299792.458 * 1000.0,
            )
            record = {
                "demand_level": demand_level,
                "time_seconds": time_seconds,
                "scenario": scenario,
                "state_basis": state_basis,
                "maneuvering_satellite_count": int(np.count_nonzero(capacities == MANEUVER_CAPACITY_GBPS)),
                "failed_satellite_count": int(np.count_nonzero(capacities == FAILED_CAPACITY_GBPS)),
                "active_satellite_count": int(np.count_nonzero(capacities > 0.0)),
                "total_available_capacity_gbps": float(capacities.sum()),
                "throughput_gbps": allocation.throughput_gbps,
                "service_ratio": allocation.service_ratio,
                "blocked_gbps": allocation.blocked_gbps,
                "maximum_utilization": allocation.maximum_utilization,
            }
            if scenario == "正常基准":
                baseline = record.copy()
            record["throughput_change_gbps"] = (
                0.0 if baseline is None else record["throughput_gbps"] - baseline["throughput_gbps"]
            )
            record["service_ratio_change"] = (
                0.0 if baseline is None else record["service_ratio"] - baseline["service_ratio"]
            )
            record["blocked_increase_gbps"] = (
                0.0 if baseline is None else record["blocked_gbps"] - baseline["blocked_gbps"]
            )
            rows.append(record)
    return pd.DataFrame(rows)
