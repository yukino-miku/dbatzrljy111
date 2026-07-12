"""Debris-size, relative-speed, miss-distance, and warning-time distributions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import truncnorm


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def truncated_power_law_normalization(d0_m: float, dmax_m: float, beta: float) -> float:
    if not (0.0 < d0_m < dmax_m and beta > 0.0):
        raise ValueError("Require 0 < D0 < Dmax and beta > 0.")
    return 1.0 - (d0_m / dmax_m) ** beta


def debris_size_pdf(
    diameter_m: np.ndarray | float,
    d0_m: float,
    dmax_m: float,
    beta: float,
) -> np.ndarray:
    """Normalized truncated differential power-law probability density in 1/m."""

    diameter = np.asarray(diameter_m, dtype=float)
    norm = truncated_power_law_normalization(d0_m, dmax_m, beta)
    density = beta * d0_m**beta * np.power(diameter, -beta - 1.0) / norm
    return np.where((diameter >= d0_m) & (diameter <= dmax_m), density, 0.0)


def debris_survival_probability(
    diameter_m: np.ndarray | float,
    d0_m: float,
    dmax_m: float,
    beta: float,
) -> np.ndarray:
    diameter = np.asarray(diameter_m, dtype=float)
    clipped = np.clip(diameter, d0_m, dmax_m)
    numerator = (d0_m / clipped) ** beta - (d0_m / dmax_m) ** beta
    denominator = truncated_power_law_normalization(d0_m, dmax_m, beta)
    survival = numerator / denominator
    return np.where(diameter <= d0_m, 1.0, np.where(diameter >= dmax_m, 0.0, survival))


def sample_truncated_power_law(
    rng: np.random.Generator,
    size: int,
    d0_m: float,
    dmax_m: float,
    beta: float,
    uniform: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse-transform sample from the finite power-law interval."""

    u = rng.random(int(size)) if uniform is None else np.asarray(uniform, dtype=float)
    if u.shape != (int(size),):
        raise ValueError("uniform must contain exactly size values.")
    lower = d0_m ** (-beta)
    upper = dmax_m ** (-beta)
    return np.power(lower - np.clip(u, 0.0, 1.0) * (lower - upper), -1.0 / beta)


def truncated_normal_parameters(mean: float, std: float, lower: float, upper: float) -> tuple[float, float]:
    if std <= 0.0 or not lower < upper:
        raise ValueError("Truncated-normal scale and interval are invalid.")
    return (lower - mean) / std, (upper - mean) / std


def relative_velocity_mean_km_s(mean: float, std: float, lower: float, upper: float) -> float:
    a, b = truncated_normal_parameters(mean, std, lower, upper)
    return float(truncnorm.mean(a, b, loc=mean, scale=std))


def sample_relative_velocity_km_s(
    rng: np.random.Generator,
    size: int,
    mean: float,
    std: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    a, b = truncated_normal_parameters(mean, std, lower, upper)
    return np.asarray(
        truncnorm.rvs(a, b, loc=mean, scale=std, size=int(size), random_state=rng),
        dtype=float,
    )


def sample_miss_distance_km(
    rng: np.random.Generator,
    size: int,
    screen_radius_km: float,
    uniform: np.ndarray | None = None,
) -> np.ndarray:
    if screen_radius_km <= 0.0:
        raise ValueError("Screening radius must be positive.")
    u = rng.random(int(size)) if uniform is None else np.asarray(uniform, dtype=float)
    return float(screen_radius_km) * np.sqrt(np.clip(u, 0.0, 1.0))


def sample_warning_hours(
    rng: np.random.Generator,
    size: int,
    minimum: float,
    mode: float,
    maximum: float,
) -> np.ndarray:
    if not minimum <= mode <= maximum or minimum < 0.0 or minimum == maximum:
        raise ValueError("Warning-time triangular parameters are invalid.")
    return rng.triangular(minimum, mode, maximum, size=int(size))


def distribution_frames(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return deterministic grids used by tables and publication figures."""

    debris = config["debris"]
    velocity = config["velocity"]
    warning = config["warning"]
    d = np.geomspace(float(debris["D0_m"]), float(debris["Dmax_m"]), 300)
    pdf = debris_size_pdf(d, debris["D0_m"], debris["Dmax_m"], debris["beta"])
    survival = debris_survival_probability(
        d, debris["D0_m"], debris["Dmax_m"], debris["beta"]
    )
    v = np.linspace(float(velocity["lower_km_s"]), float(velocity["upper_km_s"]), 300)
    a, b = truncated_normal_parameters(
        velocity["mean_km_s"], velocity["std_km_s"], velocity["lower_km_s"], velocity["upper_km_s"]
    )
    v_pdf = truncnorm.pdf(
        v, a, b, loc=float(velocity["mean_km_s"]), scale=float(velocity["std_km_s"])
    )
    w = np.linspace(float(warning["minimum_hours"]), float(warning["maximum_hours"]), 300)
    left = (w - warning["minimum_hours"]) / (
        (warning["maximum_hours"] - warning["minimum_hours"])
        * (warning["mode_hours"] - warning["minimum_hours"])
    )
    right = (warning["maximum_hours"] - w) / (
        (warning["maximum_hours"] - warning["minimum_hours"])
        * (warning["maximum_hours"] - warning["mode_hours"])
    )
    w_pdf = 2.0 * np.where(w <= warning["mode_hours"], left, right)
    norm_error = abs(float(np.trapezoid(pdf, d)) - 1.0)
    return (
        {
            "diameter_m": d,
            "diameter_pdf_per_m": pdf,
            "survival_probability": survival,
            "velocity_km_s": v,
            "velocity_pdf_per_km_s": v_pdf,
            "warning_hours": w,
            "warning_pdf_per_hour": w_pdf,
        },
        {
            "debris_pdf_numerical_normalization_error": norm_error,
            "velocity_expected_km_s": relative_velocity_mean_km_s(
                velocity["mean_km_s"], velocity["std_km_s"], velocity["lower_km_s"], velocity["upper_km_s"]
            ),
            "warning_expected_hours": (
                float(warning["minimum_hours"])
                + float(warning["mode_hours"])
                + float(warning["maximum_hours"])
            )
            / 3.0,
        },
    )
