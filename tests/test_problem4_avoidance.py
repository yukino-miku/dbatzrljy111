import numpy as np

from starlink_modeling.problem4_avoidance import (
    apply_avoidance_policy,
    target_miss_distance_km,
)
from starlink_modeling.problem4_collision import collision_probability_exact


def test_target_distance_reaches_nonzero_target_probability():
    target = target_miss_distance_km(0.1, 0.002, 1e-6)
    probability = float(collision_probability_exact(target, 0.1, 0.002))
    assert target > 0.0
    assert 0.0 < probability <= 1.00001e-6


def test_policy_respects_late_warning_delta_v_limit_and_residual_probability():
    miss = np.array([0.01, 0.01, 0.01])
    sigma = np.full(3, 0.1)
    hard = np.full(3, 0.002)
    pre = collision_probability_exact(miss, sigma, hard)
    result = apply_avoidance_policy(
        pre,
        miss,
        sigma,
        hard,
        np.array([2.0, 24.0, 24.0]),
        np.array([True, True, False]),
        threshold=1e-6,
        target_factor=0.1,
        minimum_warning_hours=6.0,
        delta_v_max_mps=1.0,
        rng=np.random.default_rng(1),
    )
    assert result.late_warning[0]
    assert not result.maneuver_executed[0]
    assert result.maneuver_executed[1]
    assert not result.maneuver_executed[2]
    assert np.max(result.delta_v_actual_mps) <= 1.0
    assert np.all(result.collision_probability_post <= pre)
    assert result.collision_probability_post[1] > 0.0


def test_lower_threshold_never_reduces_requested_count():
    pre = np.array([1e-3, 2e-5, 2e-7])
    common = dict(
        collision_probability_pre=pre,
        miss_distance_km=np.full(3, 0.01),
        sigma_pos_km=np.full(3, 0.1),
        hard_body_radius_km=np.full(3, 0.002),
        warning_hours=np.full(3, 24.0),
        detected=np.ones(3, dtype=bool),
        target_factor=0.1,
        minimum_warning_hours=6.0,
        delta_v_max_mps=1.0,
        rng=np.random.default_rng(2),
    )
    high = apply_avoidance_policy(threshold=1e-4, **common)
    low = apply_avoidance_policy(threshold=1e-6, **common)
    assert low.maneuver_requested.sum() >= high.maneuver_requested.sum()
