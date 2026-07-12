"""Long-term flux risk and per-conjunction collision probability models."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import ncx2

from .problem4_debris import debris_size_pdf


def satellite_equivalent_radius_m(effective_area_m2: float) -> float:
    if effective_area_m2 <= 0.0:
        raise ValueError("Effective projected area must be positive.")
    return math.sqrt(float(effective_area_m2) / math.pi)


def hard_body_radius_km(
    debris_diameter_m: np.ndarray | float,
    effective_area_m2: float,
) -> np.ndarray:
    debris = np.asarray(debris_diameter_m, dtype=float)
    if np.any(debris < 0.0):
        raise ValueError("Debris diameter must be nonnegative.")
    radius_m = satellite_equivalent_radius_m(effective_area_m2) + debris / 2.0
    return radius_m / 1000.0


def collision_cross_section_km2(
    debris_diameter_m: np.ndarray | float,
    effective_area_m2: float,
) -> np.ndarray:
    radius_km = hard_body_radius_km(debris_diameter_m, effective_area_m2)
    return math.pi * radius_km**2


def expected_collision_cross_section_km2(
    d0_m: float,
    dmax_m: float,
    beta: float,
    effective_area_m2: float,
) -> float:
    value, _ = quad(
        lambda diameter: float(
            debris_size_pdf(diameter, d0_m, dmax_m, beta)
            * collision_cross_section_km2(diameter, effective_area_m2)
        ),
        d0_m,
        dmax_m,
        epsabs=1e-15,
        epsrel=1e-10,
        limit=300,
    )
    return float(value)


def position_uncertainty_km(
    warning_hours: np.ndarray | float,
    sigma0_km: float,
    k_sigma_km_per_hour: float,
) -> np.ndarray:
    warning = np.asarray(warning_hours, dtype=float)
    if sigma0_km <= 0.0 or k_sigma_km_per_hour < 0.0:
        raise ValueError("Position-uncertainty parameters are invalid.")
    return np.sqrt(float(sigma0_km) ** 2 + (float(k_sigma_km_per_hour) * warning) ** 2)


def collision_probability_exact(
    miss_distance_km: np.ndarray | float,
    sigma_pos_km: np.ndarray | float,
    hard_body_radius_km_value: np.ndarray | float,
) -> np.ndarray:
    miss = np.asarray(miss_distance_km, dtype=float)
    sigma = np.asarray(sigma_pos_km, dtype=float)
    hard_body = np.asarray(hard_body_radius_km_value, dtype=float)
    if np.any(miss < 0.0) or np.any(sigma <= 0.0) or np.any(hard_body < 0.0):
        raise ValueError("Miss distance, uncertainty, or hard-body radius is invalid.")
    noncentrality = np.square(miss / sigma)
    x_h = np.square(hard_body / sigma)
    probability = ncx2.cdf(x_h, df=2, nc=noncentrality)
    return np.clip(np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def collision_probability_small_circle(
    miss_distance_km: np.ndarray | float,
    sigma_pos_km: np.ndarray | float,
    hard_body_radius_km_value: np.ndarray | float,
) -> np.ndarray:
    miss = np.asarray(miss_distance_km, dtype=float)
    sigma = np.asarray(sigma_pos_km, dtype=float)
    hard_body = np.asarray(hard_body_radius_km_value, dtype=float)
    value = hard_body**2 / (2.0 * sigma**2) * np.exp(-miss**2 / (2.0 * sigma**2))
    return np.clip(value, 0.0, 1.0)


def flux_collision_rate_per_second(
    number_density_km3: float,
    expected_cross_section_km2: float,
    expected_velocity_km_s: float,
) -> float:
    return float(number_density_km3) * float(expected_cross_section_km2) * float(expected_velocity_km_s)


def conjunction_rate_per_second(
    number_density_km3: float,
    screen_radius_km: float,
    expected_velocity_km_s: float,
) -> float:
    if screen_radius_km <= 0.0:
        raise ValueError("Screening radius must be positive.")
    return (
        float(number_density_km3)
        * math.pi
        * float(screen_radius_km) ** 2
        * float(expected_velocity_km_s)
    )


def poisson_at_least_one_probability(hazard: float | np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(hazard, dtype=float), 0.0)
    return -np.expm1(-value)
