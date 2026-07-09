import math

import numpy as np

import problem1_latitude_coverage as p1


def test_coverage_theta_from_alpha_uses_spherical_geometry_formula():
    alpha = math.radians(40.46)

    theta = p1.coverage_theta_from_alpha(6371.0, 550.0, alpha)

    expected = math.asin(((6371.0 + 550.0) / 6371.0) * math.sin(alpha)) - alpha
    assert math.isclose(theta, expected, rel_tol=0.0, abs_tol=1e-12)
    assert theta > 0.0


def test_merge_intervals_sorts_and_merges_overlaps():
    merged, length = p1.merge_intervals([(2.0, 3.0), (0.0, 1.0), (0.8, 2.4), (4.0, 4.0)])

    assert merged == [(0.0, 3.0)]
    assert math.isclose(length, 3.0)


def test_periodic_max_gap_joins_false_samples_across_boundary():
    full_covered = np.array([False, True, True, False, False])

    gap_minutes = p1.max_periodic_false_gap_minutes(full_covered, period_seconds=300.0)

    assert math.isclose(gap_minutes, 3.0)


def test_periodic_max_gap_handles_all_true_and_all_false():
    assert p1.max_periodic_false_gap_minutes(np.array([True, True]), 600.0) == 0.0
    assert p1.max_periodic_false_gap_minutes(np.array([False, False]), 600.0) == 10.0


def test_ground_track_outputs_wrapped_longitude_and_bounded_latitude():
    t, lat, lon = p1.ground_track(
        inclination_deg=50.0,
        num_periods=1.0,
        num_samples=600,
        satellite_index=0,
        total_satellites=1,
    )

    assert len(t) == 600
    assert np.nanmax(np.abs(np.degrees(lat))) <= 50.0 + 1e-10
    assert np.nanmin(lon) >= -math.pi - 1e-12
    assert np.nanmax(lon) <= math.pi + 1e-12


def test_evaluate_latitude_coverage_reports_full_cover_when_one_interval_contains_band():
    metrics = p1.evaluate_latitude_coverage(
        i_deg=0.0,
        N=1,
        theta=1.0,
        target_band=(math.radians(10.0), math.radians(20.0)),
        num_time_samples=128,
        return_series=False,
    )

    assert math.isclose(metrics["min_coverage"], 1.0)
    assert math.isclose(metrics["full_coverage_time_ratio"], 1.0)
    assert math.isclose(metrics["max_gap_time_min"], 0.0)


def test_evaluate_latitude_coverage_reports_normalized_overlap_ratio():
    metrics = p1.evaluate_latitude_coverage(
        i_deg=0.0,
        N=2,
        theta=1.0,
        target_band=(math.radians(-2.0), math.radians(2.0)),
        num_time_samples=64,
        return_series=False,
    )

    assert math.isclose(metrics["mean_overlap_ratio"], 1.0)
    assert math.isclose(metrics["normalized_overlap_ratio"], 0.5)


def test_evaluate_latitude_coverage_handles_zero_sum_for_normalized_overlap():
    metrics = p1.evaluate_latitude_coverage(
        i_deg=0.0,
        N=1,
        theta=math.radians(1.0),
        target_band=(math.radians(70.0), math.radians(71.0)),
        num_time_samples=64,
        return_series=False,
    )

    assert math.isclose(metrics["normalized_overlap_ratio"], 0.0)


def test_configure_chinese_font_is_nonfatal_and_keeps_minus_sign_setting():
    selected_font = p1.configure_chinese_font()

    assert selected_font is None or isinstance(selected_font, str)
    assert p1.plt.rcParams["axes.unicode_minus"] is False
