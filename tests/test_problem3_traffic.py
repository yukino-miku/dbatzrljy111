import math
from pathlib import Path

import numpy as np
import pandas as pd

from starlink_modeling.problem3_traffic import (
    TRAFFIC_DENSITY_MBPS_PER_KM2,
    fair_access_allocation,
    grid_cell_areas_km2,
    load_region_traffic_density,
    load_traffic_data,
    nearest_visible_baseline,
    periodic_problem_spec_profile,
    region_traffic_metrics,
    spatial_ground_demand,
    spherical_rectangle_area_km2,
)


def test_target_region_uses_spherical_rectangle_area():
    area = spherical_rectangle_area_km2()
    assert math.isclose(area, 32_013_984.230224464, rel_tol=1e-12)
    assert not math.isclose(area, 9_600_000.0)


def test_density_converts_directly_to_mean_and_peak_traffic():
    metrics = region_traffic_metrics()
    expected_mbps = TRAFFIC_DENSITY_MBPS_PER_KM2 * metrics["region_area_km2"]
    assert math.isclose(metrics["mean_traffic_mbps"], expected_mbps, rel_tol=1e-12)
    assert math.isclose(metrics["mean_traffic_tbps"], 224.78266873425576, rel_tol=1e-12)
    assert math.isclose(metrics["peak_traffic_tbps"], 337.1740031013836, rel_tol=1e-12)


def test_grid_area_and_demand_sums_match_region_totals():
    metrics = region_traffic_metrics()
    weights = np.cos(np.radians(np.array([4.0, 20.0, 35.0, 53.0])))
    areas = grid_cell_areas_km2(weights)
    demands = spatial_ground_demand(metrics["mean_traffic_gbps"], weights)
    assert math.isclose(areas.sum(), metrics["region_area_km2"], rel_tol=1e-12)
    assert math.isclose(demands.sum(), metrics["mean_traffic_gbps"], rel_tol=1e-12)
    assert np.allclose(
        demands,
        areas * TRAFFIC_DENSITY_MBPS_PER_KM2 / 1000.0,
        rtol=1e-12,
    )


def test_user_density_file_is_not_divided_by_time():
    root = Path(__file__).resolve().parents[1]
    dataset = load_region_traffic_density(
        root / "data" / "external" / "problem3_region_traffic_density.csv"
    )
    metrics = region_traffic_metrics()
    assert dataset.status == "user_given_region_density"
    assert math.isclose(dataset.average_actual_gbps, metrics["mean_traffic_gbps"])
    assert periodic_problem_spec_profile(np.array([0.0])).item() == 1.5


def test_fair_lp_has_full_service_when_capacity_is_sufficient():
    demand = np.array([5.0, 5.0])
    visibility = np.array([[True, True], [True, True]])
    result = fair_access_allocation(demand, visibility)
    assert math.isclose(result.service_ratio, 1.0, abs_tol=1e-8)
    assert math.isclose(result.throughput_gbps, demand.sum(), abs_tol=1e-7)
    assert result.satellite_loads_gbps.max() <= 20.0 + 1e-7


def test_fair_lp_reduces_common_ratio_when_capacity_is_insufficient():
    demand = np.array([30.0, 30.0])
    visibility = np.array([[True, False], [True, False]])
    result = fair_access_allocation(demand, visibility)
    assert 0.0 < result.service_ratio < 1.0
    assert math.isclose(result.service_ratio, 1.0 / 3.0, rel_tol=1e-7)
    assert math.isclose(result.throughput_gbps, result.service_ratio * demand.sum(), rel_tol=1e-7)
    assert result.satellite_loads_gbps.max() <= 20.0 + 1e-7
    assigned = set(result.assignments["satellite_id"].astype(int))
    assert assigned <= {0}


def test_nearest_baseline_and_optimizer_are_reproducible():
    demand = np.array([8.0, 7.0, 6.0])
    visibility = np.array([[True, True], [True, True], [False, True]])
    slant = np.array([[550.0, 700.0], [560.0, 710.0], [9999.0, 580.0]])
    first = nearest_visible_baseline(demand, visibility, slant, seed=7)
    second = nearest_visible_baseline(demand, visibility, slant, seed=7)
    assert first.assignments.equals(second.assignments)
    optimized_a = fair_access_allocation(demand, visibility)
    optimized_b = fair_access_allocation(demand, visibility)
    assert np.allclose(optimized_a.satellite_loads_gbps, optimized_b.satellite_loads_gbps)
    assert optimized_a.satellite_loads_gbps.max() <= 20.0 + 1e-7


def test_cumulative_traffic_is_converted_to_average_gbps(tmp_path: Path):
    path = tmp_path / "traffic.csv"
    pd.DataFrame(
        [
            {
                "region_name": "测试区",
                "traffic_value": 1.0,
                "traffic_unit": "TB",
                "period_start": "2026-01-01T00:00:00Z",
                "period_end": "2026-01-02T00:00:00Z",
                "source_name": "test",
                "source_url": "https://example.invalid",
            }
        ]
    ).to_csv(path, index=False)
    dataset = load_traffic_data(path)
    expected = 1e12 * 8 / 86400 / 1e9
    assert math.isclose(dataset.average_actual_gbps, expected)
    assert dataset.data_type == "cumulative_volume"


def test_hourly_rate_curve_is_parsed_without_inventing_source(tmp_path: Path):
    path = tmp_path / "hourly.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "traffic_value": [1.0, 2.0],
            "traffic_unit": ["Tbps", "Tbps"],
            "source_name": ["audited-source", "audited-source"],
            "source_url": ["https://example.invalid", "https://example.invalid"],
        }
    ).to_csv(path, index=False)
    dataset = load_traffic_data(path)
    assert dataset.data_type == "hourly_rate_curve"
    assert math.isclose(dataset.average_actual_gbps, 1500.0)
    assert dataset.source_names == ["audited-source"]


def test_baseline_never_exceeds_access_capacity():
    demand = np.array([25.0, 25.0, 25.0])
    visibility = np.ones((3, 3), dtype=bool)
    slant = np.array([[1.0, 2.0, 3.0]] * 3)
    baseline = nearest_visible_baseline(demand, visibility, slant, seed=11)
    assert baseline.satellite_loads_gbps.max() <= 20.0 + 1e-9
    assert baseline.throughput_gbps <= 60.0 + 1e-9
