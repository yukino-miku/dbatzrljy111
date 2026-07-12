import math

import numpy as np

from starlink_modeling.problem3_traffic import fair_access_allocation
from starlink_modeling.problem4_capacity import (
    analytical_maneuver_capacity_loss_ratio,
    annual_constellation_distributions,
    burn_time_seconds,
    degraded_duration_hours,
    satellite_capacity_gbps,
    simulate_capacity_timeline,
    summarize_capacity,
)


def test_burn_and_degraded_duration_use_mass_delta_v_over_thrust():
    burn = float(burn_time_seconds(227.0, 0.5, 0.1))
    assert math.isclose(burn, 227.0 * 0.5 / 0.1)
    degraded = float(degraded_duration_hours(227.0, 0.5, 0.1, 1.0))
    assert degraded > burn / 3600.0


def test_capacity_states_are_20_10_and_zero_gbps():
    np.testing.assert_allclose(satellite_capacity_gbps([1.0, 0.5, 0.0]), [20.0, 10.0, 0.0])


def test_problem3_lp_accepts_per_satellite_capacity_vector():
    demand = np.array([15.0, 15.0])
    visible = np.ones((2, 3), dtype=bool)
    result = fair_access_allocation(demand, visible, capacity_gbps=np.array([20.0, 10.0, 0.0]))
    assert result.satellite_loads_gbps[0] <= 20.0 + 1e-7
    assert result.satellite_loads_gbps[1] <= 10.0 + 1e-7
    assert result.satellite_loads_gbps[2] <= 1e-7


def test_constellation_poisson_mean_is_s_times_single_satellite_rate():
    rng = np.random.default_rng(12)
    avoidance, _ = annual_constellation_distributions(rng, 1840, 2.0, 1e-5, 20_000, 20_000.0)
    assert math.isclose(avoidance["annual_avoidance_count"].mean(), 3680.0, rel_tol=0.002)
    loss = analytical_maneuver_capacity_loss_ratio(2.0, 1.0)
    assert 0.0 < loss < 1.0


def test_capacity_loss_analytical_and_event_simulation_agree():
    timeline = simulate_capacity_timeline(
        np.random.default_rng(123),
        total_satellites=100,
        annual_maneuvers_per_satellite=10.0,
        degraded_duration_samples_hours=np.array([1.0]),
        annual_failure_hazard_per_satellite=0.0,
        step_hours=1.0,
    )
    summary = summarize_capacity(100, 10.0, 1.0, timeline)
    assert summary["analytical_mc_relative_error"] < 0.15
