import numpy as np
from scipy.sparse import csr_matrix

from starlink_modeling.problem4_reliability import (
    CoverageArchive,
    effective_coverage,
    interval_union_hours,
    simulate_reliability_year,
)


def test_effective_coverage_uses_state_weighted_multiplicity():
    counts = np.array([1.0, 2.0])
    assert effective_coverage(counts, np.array([0.0, 1.0]))
    assert not effective_coverage(counts, np.array([0.5, 0.0]))


def test_interval_union_handles_overlap_without_double_counting():
    assert interval_union_hours([(1.0, 3.0), (2.0, 5.0), (8.0, 9.0)]) == 5.0


def test_no_event_year_recovers_baseline_coverage():
    archive = CoverageArchive(
        grid=None,
        times_seconds=np.array([0.0]),
        incidence=[],
        counts=[],
        total_satellites=4,
        plane_index=np.array([0, 0, 1, 1]),
        baseline_effective_availability=1.0,
        critical_maneuver_fraction=0.5,
        critical_failure_fraction=0.5,
        minimum_coverage_multiplicity=1,
        mean_coverage_multiplicity=2.0,
    )
    result = simulate_reliability_year(
        np.random.default_rng(3),
        archive,
        base_M=2,
        base_N=2,
        in_plane_spares_per_plane=0,
        ground_spares=0,
        annual_maneuvers_per_satellite=0.0,
        annual_failure_hazard_per_satellite=0.0,
        degraded_duration_samples_hours=np.array([1.0]),
        adjustment_hours=168.0,
        takeover_hours=6.0,
        ground_replacement_hours=1440.0,
        standby_risk_factor=1.0,
    )
    assert result["effective_coverage_availability"] == 1.0
    assert result["geometric_coverage_availability"] == 1.0


class _FixedFailureRng:
    """Small deterministic RNG stub for failure-window boundary tests."""

    def __init__(self, failure_times, failure_satellites):
        self.failure_times = np.asarray(failure_times, dtype=float)
        self.failure_satellites = np.asarray(failure_satellites, dtype=int)
        self.poisson_calls = 0

    def poisson(self, _lam, size=None):
        self.poisson_calls += 1
        if self.poisson_calls <= 3:
            return 0
        if self.poisson_calls == 4:
            return len(self.failure_times)
        if self.poisson_calls == 5:
            return np.zeros(size, dtype=int)
        return 0

    def uniform(self, _low, _high, size=None):
        if size is not None:
            return self.failure_times.copy()
        return float(_low)

    def integers(self, _low, _high=None, size=None):
        if size is not None:
            return self.failure_satellites.copy()
        return int(self.failure_satellites[0])

    def choice(self, values):
        return np.asarray(values).ravel()[0]


def _single_critical_satellite_archive() -> CoverageArchive:
    return CoverageArchive(
        grid=None,
        times_seconds=np.array([0.0]),
        incidence=[csr_matrix([[1]], dtype=np.int8)],
        counts=[np.array([1], dtype=np.int16)],
        total_satellites=1,
        plane_index=np.array([0]),
        baseline_effective_availability=1.0,
        critical_maneuver_fraction=0.0,
        critical_failure_fraction=1.0,
        minimum_coverage_multiplicity=1,
        mean_coverage_multiplicity=1.0,
    )


def test_failure_satellite_is_absent_during_seven_day_adjustment():
    result = simulate_reliability_year(
        _FixedFailureRng([100.0], [0]),
        _single_critical_satellite_archive(),
        base_M=1,
        base_N=1,
        in_plane_spares_per_plane=0,
        ground_spares=0,
        annual_maneuvers_per_satellite=0.0,
        annual_failure_hazard_per_satellite=1.0,
        degraded_duration_samples_hours=np.array([1.0]),
        adjustment_hours=168.0,
        takeover_hours=6.0,
        ground_replacement_hours=1440.0,
        standby_risk_factor=1.0,
    )
    assert np.isclose(result["geometric_coverage_availability"], 1.0 - 168.0 / 8760.0)


def test_in_orbit_spare_is_consumed_once_and_ground_spare_waits_for_launch():
    common = dict(
        archive=_single_critical_satellite_archive(),
        base_M=1,
        base_N=1,
        annual_maneuvers_per_satellite=0.0,
        annual_failure_hazard_per_satellite=1.0,
        degraded_duration_samples_hours=np.array([1.0]),
        adjustment_hours=168.0,
        takeover_hours=6.0,
        ground_replacement_hours=1440.0,
        standby_risk_factor=1.0,
    )
    two_failures = simulate_reliability_year(
        _FixedFailureRng([100.0, 200.0], [0, 0]),
        in_plane_spares_per_plane=1,
        ground_spares=0,
        **common,
    )
    assert two_failures["remaining_in_orbit_spares"] == 0

    ground = simulate_reliability_year(
        _FixedFailureRng([100.0], [0]),
        in_plane_spares_per_plane=0,
        ground_spares=1,
        **common,
    )
    assert ground["ground_spares_used"] == 1
    assert ground["replenishment_launch_count"] == 1
    assert np.isclose(ground["capacity_availability"], 1.0 - 1440.0 / 8760.0)


def test_fixed_seed_reproduces_annual_reliability_result():
    archive = CoverageArchive(
        grid=None,
        times_seconds=np.array([0.0]),
        incidence=[],
        counts=[],
        total_satellites=10,
        plane_index=np.zeros(10, dtype=int),
        baseline_effective_availability=1.0,
        critical_maneuver_fraction=0.2,
        critical_failure_fraction=0.0,
        minimum_coverage_multiplicity=2,
        mean_coverage_multiplicity=3.0,
    )
    kwargs = dict(
        archive=archive,
        base_M=1,
        base_N=10,
        in_plane_spares_per_plane=0,
        ground_spares=0,
        annual_maneuvers_per_satellite=2.0,
        annual_failure_hazard_per_satellite=0.0,
        degraded_duration_samples_hours=np.array([1.0, 2.0]),
        adjustment_hours=168.0,
        takeover_hours=6.0,
        ground_replacement_hours=1440.0,
        standby_risk_factor=1.0,
    )
    assert simulate_reliability_year(np.random.default_rng(99), **kwargs) == simulate_reliability_year(
        np.random.default_rng(99), **kwargs
    )
