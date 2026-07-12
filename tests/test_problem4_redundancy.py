import math

import pandas as pd

from starlink_modeling.problem4_redundancy import (
    launch_count,
    meets_coverage_target,
    pareto_front,
    redundancy_costs,
)


def test_launch_count_uses_ceiling_and_sixty_satellite_limit():
    assert launch_count(0) == 0
    assert launch_count(60) == 1
    assert launch_count(61) == 2


def test_redundancy_cost_includes_manufacture_and_initial_launches():
    result = redundancy_costs(
        base_satellites=1840,
        in_plane_spares=40,
        extra_plane_satellites=0,
        ground_spares=0,
        annual_avoidances_per_satellite=1.0,
        annual_failures=1.0,
        expected_replenishment_launches=0.0,
        satellite_cost_cny=5_000_000.0,
        launch_cost_cny=200_000_000.0,
        maneuver_cost_cny=20_000.0,
        launch_capacity=60,
        design_life_years=5.0,
    )
    assert result["initial_launch_count"] == 1
    assert result["initial_cost_cny"] == 400_000_000.0
    assert result["five_year_expected_cost_cny"] > result["initial_cost_cny"]


def test_pareto_filter_removes_more_expensive_weaker_candidate():
    frame = pd.DataFrame(
        {
            "five_year_expected_cost_cny": [1.0, 2.0, 1.5],
            "mean_effective_coverage_availability": [0.99, 0.98, 0.995],
            "q05_effective_coverage_availability": [0.98, 0.97, 0.985],
        }
    )
    result = pareto_front(frame)
    assert 2.0 not in result["five_year_expected_cost_cny"].tolist()
    assert len(result) == 2


def test_mean_and_robust_99_percent_decisions_are_separate():
    assert meets_coverage_target(0.995, 0.995) == (True, True)
    assert meets_coverage_target(0.995, 0.985) == (True, False)
    assert meets_coverage_target(0.985, 0.999) == (False, False)
