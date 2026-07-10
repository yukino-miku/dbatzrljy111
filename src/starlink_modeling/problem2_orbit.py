"""Walker-Delta constellation construction and two-body circular propagation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


R_EARTH_KM = 6371.0
ORBIT_HEIGHT_KM = 550.0
MU_EARTH_KM3_S2 = 3.986004418e5
OMEGA_EARTH_RAD_S = 7.2921159e-5


def wrap_to_pi(angle_rad: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-pi, pi)."""

    return (np.asarray(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi


def mean_motion_rad_s(
    earth_radius_km: float = R_EARTH_KM,
    altitude_km: float = ORBIT_HEIGHT_KM,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> float:
    """Return circular-orbit mean angular speed."""

    semi_major_axis_km = earth_radius_km + altitude_km
    return math.sqrt(mu_km3_s2 / semi_major_axis_km**3)


def orbital_period_seconds(
    earth_radius_km: float = R_EARTH_KM,
    altitude_km: float = ORBIT_HEIGHT_KM,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> float:
    """Return the circular-orbit period in seconds."""

    return 2.0 * math.pi / mean_motion_rad_s(
        earth_radius_km, altitude_km, mu_km3_s2
    )


@dataclass(frozen=True)
class WalkerConstellation:
    """Expanded satellite parameters for a Walker-Delta-like constellation."""

    M: int
    N: int
    inclination_rad: float
    phase_factor_F: int
    Omega0_rad: float
    u0_rad: float
    altitude_km: float
    plane_index: np.ndarray
    satellite_index: np.ndarray
    raan_rad: np.ndarray
    initial_argument_of_latitude_rad: np.ndarray

    @property
    def total_satellites(self) -> int:
        return self.M * self.N

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per satellite for reproducible result export."""

        return pd.DataFrame(
            {
                "plane_index": self.plane_index,
                "satellite_index": self.satellite_index,
                "inclination_deg": math.degrees(self.inclination_rad),
                "raan_deg": np.degrees(self.raan_rad) % 360.0,
                "initial_argument_of_latitude_deg": (
                    np.degrees(self.initial_argument_of_latitude_rad) % 360.0
                ),
                "phase_factor": self.phase_factor_F,
                "altitude_km": self.altitude_km,
            }
        )


def generate_walker_constellation(
    M: int,
    N: int,
    inclination_deg: float,
    phase_factor_F: int,
    Omega0_deg: float = 0.0,
    u0_deg: float = 0.0,
    altitude_km: float = ORBIT_HEIGHT_KM,
) -> WalkerConstellation:
    """Expand the six Walker design variables to per-satellite parameters.

    The adjacent-plane phase offset is 2*pi*F/(M*N), while satellites in one
    plane are separated by 2*pi/N.
    """

    M = int(M)
    N = int(N)
    phase_factor_F = int(phase_factor_F)
    if M < 1 or N < 1:
        raise ValueError("M and N must both be positive integers.")
    if not 0 <= phase_factor_F < M:
        raise ValueError("phase_factor_F must satisfy 0 <= F < M.")

    plane_index = np.repeat(np.arange(M, dtype=int), N)
    satellite_index = np.tile(np.arange(N, dtype=int), M)
    Omega0_rad = math.radians(float(Omega0_deg))
    u0_rad = math.radians(float(u0_deg))

    plane_raan = Omega0_rad + 2.0 * np.pi * np.arange(M) / M
    raan_rad = plane_raan[plane_index]
    initial_u_rad = (
        u0_rad
        + 2.0 * np.pi * satellite_index / N
        + 2.0 * np.pi * phase_factor_F * plane_index / (M * N)
    )

    return WalkerConstellation(
        M=M,
        N=N,
        inclination_rad=math.radians(float(inclination_deg)),
        phase_factor_F=phase_factor_F,
        Omega0_rad=Omega0_rad,
        u0_rad=u0_rad,
        altitude_km=float(altitude_km),
        plane_index=plane_index,
        satellite_index=satellite_index,
        raan_rad=raan_rad,
        initial_argument_of_latitude_rad=initial_u_rad,
    )


def propagate_constellation(
    constellation: WalkerConstellation,
    times_seconds: np.ndarray | float,
    earth_radius_km: float = R_EARTH_KM,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> np.ndarray:
    """Propagate all satellites and return ECI coordinates.

    The result has shape ``(time, satellite, xyz)``. A scalar time still
    returns a leading time dimension of length one.
    """

    times = np.atleast_1d(np.asarray(times_seconds, dtype=float))
    a_km = earth_radius_km + constellation.altitude_km
    n_rad_s = math.sqrt(mu_km3_s2 / a_km**3)

    u = (
        constellation.initial_argument_of_latitude_rad[None, :]
        + n_rad_s * times[:, None]
    )
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_raan = np.cos(constellation.raan_rad)[None, :]
    sin_raan = np.sin(constellation.raan_rad)[None, :]
    cos_i = math.cos(constellation.inclination_rad)
    sin_i = math.sin(constellation.inclination_rad)

    x_eci = a_km * (cos_raan * cos_u - sin_raan * sin_u * cos_i)
    y_eci = a_km * (sin_raan * cos_u + cos_raan * sin_u * cos_i)
    z_eci = a_km * sin_u * sin_i
    return np.stack((x_eci, y_eci, z_eci), axis=-1)


def eci_to_ecef(
    eci_positions_km: np.ndarray,
    times_seconds: np.ndarray | float,
    omega_earth_rad_s: float = OMEGA_EARTH_RAD_S,
) -> np.ndarray:
    """Rotate ECI positions to ECEF using R3(-omega_E*t)."""

    eci = np.asarray(eci_positions_km, dtype=float)
    if eci.ndim == 2:
        eci = eci[None, :, :]
    if eci.ndim != 3 or eci.shape[-1] != 3:
        raise ValueError("ECI positions must have shape (time, satellite, 3).")

    times = np.atleast_1d(np.asarray(times_seconds, dtype=float))
    if times.size != eci.shape[0]:
        raise ValueError("Number of times must match the ECI time dimension.")

    rotation = omega_earth_rad_s * times
    cos_r = np.cos(rotation)[:, None]
    sin_r = np.sin(rotation)[:, None]
    x_eci = eci[..., 0]
    y_eci = eci[..., 1]
    x_ecef = cos_r * x_eci + sin_r * y_eci
    y_ecef = -sin_r * x_eci + cos_r * y_eci
    return np.stack((x_ecef, y_ecef, eci[..., 2]), axis=-1)


def satellite_subpoints(ecef_positions_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ECEF satellite positions to geocentric subpoint lat/lon."""

    ecef = np.asarray(ecef_positions_km, dtype=float)
    x = ecef[..., 0]
    y = ecef[..., 1]
    z = ecef[..., 2]
    latitude_rad = np.arctan2(z, np.hypot(x, y))
    longitude_rad = wrap_to_pi(np.arctan2(y, x))
    return latitude_rad, longitude_rad


def propagate_subpoint_unit_vectors(
    constellation: WalkerConstellation,
    times_seconds: np.ndarray | float,
) -> np.ndarray:
    """Return ECEF unit vectors from Earth center to satellite subpoints."""

    eci = propagate_constellation(constellation, times_seconds)
    ecef = eci_to_ecef(eci, times_seconds)
    return ecef / np.linalg.norm(ecef, axis=-1, keepdims=True)

