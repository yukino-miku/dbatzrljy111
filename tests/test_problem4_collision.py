import math

import numpy as np
from scipy.integrate import quad

from starlink_modeling.problem4_collision import (
    collision_probability_exact,
    collision_probability_small_circle,
    conjunction_rate_per_second,
    expected_collision_cross_section_km2,
    flux_collision_rate_per_second,
    hard_body_radius_km,
    poisson_at_least_one_probability,
)
from starlink_modeling.problem4_debris import (
    debris_size_pdf,
    sample_miss_distance_km,
    sample_relative_velocity_km_s,
    sample_truncated_power_law,
)


def test_power_law_normalizes_and_inverse_samples_stay_in_range():
    integral, _ = quad(lambda d: debris_size_pdf(d, 0.01, 1.0, 2.5), 0.01, 1.0)
    assert math.isclose(integral, 1.0, rel_tol=1e-9)
    rng = np.random.default_rng(7)
    samples = sample_truncated_power_law(rng, 100_000, 0.01, 1.0, 2.5)
    assert samples.min() >= 0.01
    assert samples.max() <= 1.0
    assert np.mean(samples < 0.02) > 0.80


def test_truncated_velocity_and_area_uniform_miss_distance():
    rng = np.random.default_rng(9)
    velocity = sample_relative_velocity_km_s(rng, 50_000, 10.0, 2.0, 5.0, 15.0)
    miss = sample_miss_distance_km(rng, 50_000, 5.0)
    assert np.all((velocity >= 5.0) & (velocity <= 15.0))
    assert math.isclose(float(np.mean((miss / 5.0) ** 2)), 0.5, abs_tol=0.01)


def test_exact_collision_probability_is_bounded_monotone_and_matches_small_circle():
    miss = np.linspace(0.0, 0.5, 200)
    hard = float(hard_body_radius_km(0.01, 15.0))
    exact = collision_probability_exact(miss, 0.1, hard)
    approx = collision_probability_small_circle(miss, 0.1, hard)
    assert np.all((exact >= 0.0) & (exact <= 1.0))
    assert np.all(np.diff(exact) <= 1e-15)
    mask = exact > 1e-12
    assert np.max(np.abs(approx[mask] - exact[mask]) / exact[mask]) < 0.01


def test_flux_and_candidate_rate_are_analytically_consistent():
    cross = expected_collision_cross_section_km2(0.01, 1.0, 2.5, 15.0)
    flux = flux_collision_rate_per_second(1e-8, cross, 10.0)
    conjunction = conjunction_rate_per_second(1e-8, 5.0, 10.0)
    reconstructed = conjunction * cross / (math.pi * 5.0**2)
    assert math.isclose(flux, reconstructed, rel_tol=1e-12)
    assert math.isclose(float(poisson_at_least_one_probability(0.2)), 1 - math.exp(-0.2))


def test_conditional_failure_probability_cannot_exceed_collision_probability():
    collision_hazard = 0.2
    conditional_failure_probability = 0.8
    collision_probability = float(poisson_at_least_one_probability(collision_hazard))
    failure_probability = float(
        poisson_at_least_one_probability(
            collision_hazard * conditional_failure_probability
        )
    )
    assert 0.0 <= failure_probability <= collision_probability <= 1.0
