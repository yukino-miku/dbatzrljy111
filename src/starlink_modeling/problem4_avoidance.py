"""Autonomous avoidance decisions and residual-risk calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .problem4_collision import collision_probability_exact


@dataclass
class AvoidanceResult:
    detected: np.ndarray
    maneuver_requested: np.ndarray
    maneuver_executed: np.ndarray
    late_warning: np.ndarray
    delta_v_limited: np.ndarray
    target_miss_distance_km: np.ndarray
    delta_v_required_mps: np.ndarray
    delta_v_commanded_mps: np.ndarray
    delta_v_actual_mps: np.ndarray
    post_miss_distance_km: np.ndarray
    collision_probability_post: np.ndarray


def target_miss_distance_km(
    sigma_pos_km: float,
    hard_body_radius_km: float,
    target_probability: float,
) -> float:
    """Solve the minimum nonnegative miss distance attaining the target risk."""

    if not 0.0 < target_probability < 1.0:
        raise ValueError("Target collision probability must lie in (0, 1).")
    at_zero = float(
        collision_probability_exact(0.0, sigma_pos_km, hard_body_radius_km)
    )
    if at_zero <= target_probability:
        return 0.0

    def objective(distance: float) -> float:
        return float(
            collision_probability_exact(distance, sigma_pos_km, hard_body_radius_km)
        ) - target_probability

    upper = max(0.01, 4.0 * float(sigma_pos_km))
    while objective(upper) > 0.0 and upper < 1.0e4:
        upper *= 2.0
    if objective(upper) > 0.0:
        raise RuntimeError("Unable to bracket target miss distance.")
    return float(brentq(objective, 0.0, upper, xtol=1e-12, rtol=1e-11))


def apply_avoidance_policy(
    collision_probability_pre: np.ndarray,
    miss_distance_km: np.ndarray,
    sigma_pos_km: np.ndarray,
    hard_body_radius_km: np.ndarray,
    warning_hours: np.ndarray,
    detected: np.ndarray,
    *,
    threshold: float,
    target_factor: float,
    minimum_warning_hours: float,
    delta_v_max_mps: float,
    displacement_factor: float = 1.0,
    execution_error_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> AvoidanceResult:
    pre = np.asarray(collision_probability_pre, dtype=float)
    miss = np.asarray(miss_distance_km, dtype=float)
    sigma = np.asarray(sigma_pos_km, dtype=float)
    hard_body = np.asarray(hard_body_radius_km, dtype=float)
    warning = np.asarray(warning_hours, dtype=float)
    detected_array = np.asarray(detected, dtype=bool)
    arrays = (miss, sigma, hard_body, warning, detected_array)
    if any(array.shape != pre.shape for array in arrays):
        raise ValueError("All event arrays must have the same shape.")
    if threshold <= 0.0 or not 0.0 < target_factor < 1.0:
        raise ValueError("Avoidance threshold or target factor is invalid.")
    if delta_v_max_mps <= 0.0 or displacement_factor <= 0.0:
        raise ValueError("Delta-v and displacement limits must be positive.")

    requested = detected_array & (pre >= float(threshold))
    late = requested & (warning < float(minimum_warning_hours))
    executed = requested & ~late
    target_distance = miss.copy()
    required = np.zeros_like(pre)

    target_probability = float(threshold) * float(target_factor)
    for index in np.flatnonzero(executed):
        target_distance[index] = target_miss_distance_km(
            float(sigma[index]), float(hard_body[index]), target_probability
        )
        delta_b_km = max(0.0, float(target_distance[index] - miss[index]))
        required[index] = (
            1000.0
            * delta_b_km
            / (float(displacement_factor) * float(warning[index]) * 3600.0)
        )

    limited = executed & (required > float(delta_v_max_mps))
    commanded = np.where(executed, np.minimum(required, float(delta_v_max_mps)), 0.0)
    if execution_error_std > 0.0:
        if rng is None:
            raise ValueError("rng is required when execution error is enabled.")
        epsilon = rng.normal(0.0, float(execution_error_std), size=pre.size)
        epsilon = np.clip(epsilon, -0.5, 0.5)
    else:
        epsilon = np.zeros(pre.size, dtype=float)
    actual = np.where(executed, commanded * (1.0 + epsilon), 0.0)
    actual = np.clip(actual, 0.0, float(delta_v_max_mps))
    displacement_km = (
        float(displacement_factor) * actual * warning * 3600.0 / 1000.0
    )
    post_miss = miss + displacement_km
    post = pre.copy()
    if np.any(executed):
        post[executed] = collision_probability_exact(
            post_miss[executed], sigma[executed], hard_body[executed]
        )
    post = np.minimum(np.clip(post, 0.0, 1.0), np.clip(pre, 0.0, 1.0))
    return AvoidanceResult(
        detected=detected_array,
        maneuver_requested=requested,
        maneuver_executed=executed,
        late_warning=late,
        delta_v_limited=limited,
        target_miss_distance_km=target_distance,
        delta_v_required_mps=required,
        delta_v_commanded_mps=commanded,
        delta_v_actual_mps=actual,
        post_miss_distance_km=post_miss,
        collision_probability_post=post,
    )
