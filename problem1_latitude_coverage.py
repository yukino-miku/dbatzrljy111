"""Problem 1: single-orbital-plane latitude projection coverage analysis.

Run from the repository root:

    python problem1_latitude_coverage.py

On Windows, if ``python`` points to the Microsoft Store launcher, use:

    py problem1_latitude_coverage.py
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Default constants required by the problem statement.
R_EARTH_KM = 6371.0
ORBIT_HEIGHT_KM = 550.0
MU_EARTH_KM3_S2 = 3.986004418e5
OMEGA_EARTH_RAD_S = 7.2921159e-5
ALPHA_DEG = 40.46
GROUND_RADIUS_GIVEN_KM = 506.0
TARGET_LAT_MIN_DEG = 30.0
TARGET_LAT_MAX_DEG = 50.0
INCLINATION_MIN_DEG = 40.0
INCLINATION_MAX_DEG = 60.0
INCLINATION_STEP_DEG = 0.25
N_MIN_SEARCH = 1
N_MAX_SEARCH = 80
NUM_TIME_SAMPLES = 3000
COVERAGE_TOL = 1e-10

OUTPUT_DIR = Path("outputs") / "problem1"
METHOD_NOTES_PATH = Path("problem1_method_notes.md")
METHOD_NOTES_DOCS_PATH = Path("docs") / "report" / "problem1_method_notes.md"


def deg2rad(value_deg: float | np.ndarray) -> float | np.ndarray:
    """Convert degrees to radians."""
    return np.deg2rad(value_deg)


def rad2deg(value_rad: float | np.ndarray) -> float | np.ndarray:
    """Convert radians to degrees."""
    return np.rad2deg(value_rad)


def wrap_to_pi(angle_rad: np.ndarray | float) -> np.ndarray | float:
    """Wrap longitude-like angles to [-pi, pi]."""
    return (np.asarray(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def mean_motion_rad_s(
    R: float = R_EARTH_KM,
    h: float = ORBIT_HEIGHT_KM,
    mu: float = MU_EARTH_KM3_S2,
) -> float:
    """Return circular-orbit mean motion n = sqrt(mu / a^3)."""
    semi_major_axis_km = R + h
    return math.sqrt(mu / semi_major_axis_km**3)


def orbital_period_seconds(
    R: float = R_EARTH_KM,
    h: float = ORBIT_HEIGHT_KM,
    mu: float = MU_EARTH_KM3_S2,
) -> float:
    """Return circular-orbit period T = 2*pi/n."""
    return 2.0 * math.pi / mean_motion_rad_s(R=R, h=h, mu=mu)


def coverage_theta_from_alpha(R: float, h: float, alpha: float) -> float:
    """Compute coverage central angle from antenna half-cone angle.

    The spherical geometry relation used here is

        theta = asin(((R + h) / R) * sin(alpha)) - alpha

    for a nadir-pointing cone that does not exceed the horizon.
    """
    arg = ((R + h) / R) * math.sin(alpha)
    if arg > 1.0:
        raise ValueError(
            "Antenna cone reaches beyond the geometric horizon; "
            "the stated theta-alpha formula is not valid."
        )
    return math.asin(arg) - alpha


def spherical_cap_area_km2(R: float, theta: float) -> float:
    """Area of a spherical cap with central angle theta."""
    return 2.0 * math.pi * R**2 * (1.0 - math.cos(theta))


def flat_disk_area_km2(R: float, theta: float) -> float:
    """Local planar disk area approximation using radius R*theta."""
    return math.pi * (R * theta) ** 2


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> tuple[list[tuple[float, float]], float]:
    """Merge one-dimensional intervals and return merged intervals plus length.

    Intervals with hi <= lo are ignored. Inputs are in radians when used by the
    latitude coverage model, but this utility itself is unit-agnostic.
    """
    cleaned = sorted((float(lo), float(hi)) for lo, hi in intervals if hi > lo)
    if not cleaned:
        return [], 0.0

    merged: list[tuple[float, float]] = []
    cur_lo, cur_hi = cleaned[0]

    for lo, hi in cleaned[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi

    merged.append((cur_lo, cur_hi))
    total_length = sum(hi - lo for lo, hi in merged)
    return merged, total_length


def max_periodic_false_gap_minutes(full_covered: np.ndarray, period_seconds: float) -> float:
    """Return the longest periodic not-fully-covered time span in minutes.

    ``full_covered[t]`` is True when C_lat(t) >= 1 - tol. The sampled orbit is
    periodic, so a False run at the end and a False run at the beginning must be
    treated as one continuous gap.
    """
    flags = np.asarray(full_covered, dtype=bool)
    if flags.size == 0:
        return 0.0
    if np.all(flags):
        return 0.0
    if not np.any(flags):
        return period_seconds / 60.0

    true_indices = np.flatnonzero(flags)
    sample_dt_seconds = period_seconds / flags.size
    max_false_samples = 0

    for idx, current_true in enumerate(true_indices):
        next_true = true_indices[(idx + 1) % true_indices.size]
        if idx == true_indices.size - 1:
            next_true += flags.size
        false_samples = next_true - current_true - 1
        max_false_samples = max(max_false_samples, int(false_samples))

    return max_false_samples * sample_dt_seconds / 60.0


def _sin_u_matrix(num_time_samples: int, N: int) -> np.ndarray:
    """Precompute sin(u_k(t)) over one orbit for uniformly spaced satellites."""
    base_phase = np.linspace(0.0, 2.0 * math.pi, num_time_samples, endpoint=False)
    satellite_offsets = 2.0 * math.pi * np.arange(N) / N
    return np.sin(base_phase[:, None] + satellite_offsets[None, :])


def _latitudes_from_sin_u(i_rad: float, sin_u: np.ndarray) -> np.ndarray:
    """Compute sub-satellite latitudes phi = asin(sin(i) sin(u))."""
    return np.arcsin(np.sin(i_rad) * sin_u)


def _union_and_overlap_timeseries(
    latitudes_rad: np.ndarray,
    theta: float,
    target_band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute latitude union length, summed length, and overlap length.

    The target-band intersection is applied before union calculation. The union
    calculation is vectorized over time samples and iterates only over the
    satellites in a single time slice.
    """
    band_lo, band_hi = target_band

    lows = np.maximum(latitudes_rad - theta, band_lo)
    highs = np.minimum(latitudes_rad + theta, band_hi)
    valid = highs > lows

    lengths = np.where(valid, highs - lows, 0.0)
    summed_lengths = np.sum(lengths, axis=1)

    sorted_lows = np.where(valid, lows, np.inf)
    sorted_highs = np.where(valid, highs, -np.inf)
    order = np.argsort(sorted_lows, axis=1)
    sorted_lows = np.take_along_axis(sorted_lows, order, axis=1)
    sorted_highs = np.take_along_axis(sorted_highs, order, axis=1)

    num_times, num_satellites = sorted_lows.shape
    union_lengths = np.zeros(num_times, dtype=float)
    has_active_interval = np.zeros(num_times, dtype=bool)
    current_lo = np.zeros(num_times, dtype=float)
    current_hi = np.zeros(num_times, dtype=float)

    for sat_idx in range(num_satellites):
        lo = sorted_lows[:, sat_idx]
        hi = sorted_highs[:, sat_idx]
        is_valid = hi > lo
        already_active = has_active_interval.copy()

        starts_first = is_valid & ~already_active
        current_lo[starts_first] = lo[starts_first]
        current_hi[starts_first] = hi[starts_first]
        has_active_interval[starts_first] = True

        continues = is_valid & already_active
        overlaps = continues & (lo <= current_hi + 1e-14)
        current_hi[overlaps] = np.maximum(current_hi[overlaps], hi[overlaps])

        disjoint = continues & ~overlaps
        union_lengths[disjoint] += current_hi[disjoint] - current_lo[disjoint]
        current_lo[disjoint] = lo[disjoint]
        current_hi[disjoint] = hi[disjoint]

    union_lengths[has_active_interval] += (
        current_hi[has_active_interval] - current_lo[has_active_interval]
    )
    overlap_lengths = np.maximum(0.0, summed_lengths - union_lengths)
    return union_lengths, summed_lengths, overlap_lengths


def evaluate_latitude_coverage(
    i_deg: float,
    N: int,
    theta: float,
    target_band: tuple[float, float],
    num_time_samples: int = NUM_TIME_SAMPLES,
    return_series: bool = False,
    sin_u: np.ndarray | None = None,
    period_seconds: float | None = None,
    tol: float = COVERAGE_TOL,
) -> dict[str, float | int | np.ndarray]:
    """Evaluate one-orbit latitude-projection coverage metrics.

    At time t, satellite k covers [phi_k(t)-theta, phi_k(t)+theta] in latitude.
    The target-band intersection of all such intervals is merged, and its union
    length divided by the target-band width gives C_lat(t).
    """
    if N < 1:
        raise ValueError("N must be at least 1.")
    band_lo, band_hi = target_band
    if band_hi <= band_lo:
        raise ValueError("target_band must be ordered as (lower, upper).")

    if period_seconds is None:
        period_seconds = orbital_period_seconds()
    if sin_u is None:
        sin_u = _sin_u_matrix(num_time_samples, N)

    i_rad = math.radians(i_deg)
    latitudes = _latitudes_from_sin_u(i_rad, sin_u)
    union_lengths, summed_lengths, overlap_lengths = _union_and_overlap_timeseries(
        latitudes, theta, target_band
    )

    band_width = band_hi - band_lo
    coverage = np.clip(union_lengths / band_width, 0.0, 1.0)
    overlap_ratio = overlap_lengths / band_width
    full_covered = coverage >= 1.0 - tol

    metrics: dict[str, float | int | np.ndarray] = {
        "inclination_deg": float(i_deg),
        "N": int(N),
        "spacing_deg": 360.0 / N,
        "min_coverage": float(np.min(coverage)),
        "mean_coverage": float(np.mean(coverage)),
        "full_coverage_time_ratio": float(np.mean(full_covered)),
        "mean_overlap_ratio": float(np.mean(overlap_ratio)),
        "max_gap_time_min": float(
            max_periodic_false_gap_minutes(full_covered, period_seconds)
        ),
    }

    if return_series:
        time_seconds = np.linspace(0.0, period_seconds, num_time_samples, endpoint=False)
        metrics.update(
            {
                "time_seconds": time_seconds,
                "coverage_series": coverage,
                "overlap_series": overlap_ratio,
                "latitudes_rad": latitudes,
                "summed_lengths_rad": summed_lengths,
                "union_lengths_rad": union_lengths,
            }
        )

    return metrics


def ground_track(
    inclination_deg: float,
    num_periods: float = 3.0,
    num_samples: int = 3600,
    satellite_index: int = 0,
    total_satellites: int = 1,
    R: float = R_EARTH_KM,
    h: float = ORBIT_HEIGHT_KM,
    mu: float = MU_EARTH_KM3_S2,
    omega_E: float = OMEGA_EARTH_RAD_S,
    raan_rad: float = 0.0,
    u0_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, sub-satellite latitude, and longitude for a circular orbit."""
    n = mean_motion_rad_s(R=R, h=h, mu=mu)
    period = 2.0 * math.pi / n
    times = np.linspace(0.0, num_periods * period, num_samples, endpoint=False)
    i_rad = math.radians(inclination_deg)
    phase_offset = 2.0 * math.pi * satellite_index / total_satellites
    u = u0_rad + n * times + phase_offset

    lat = np.arcsin(np.sin(i_rad) * np.sin(u))
    lon_inertial = np.arctan2(np.cos(i_rad) * np.sin(u), np.cos(u))
    lon = wrap_to_pi(raan_rad + lon_inertial - omega_E * times)
    return times, lat, lon


def build_single_satellite_summary() -> tuple[pd.DataFrame, dict[str, float]]:
    """Build the single-satellite coverage geometry summary table."""
    alpha_rad = math.radians(ALPHA_DEG)
    theta_alpha = coverage_theta_from_alpha(R_EARTH_KM, ORBIT_HEIGHT_KM, alpha_rad)
    theta_given = GROUND_RADIUS_GIVEN_KM / R_EARTH_KM

    spherical_area = spherical_cap_area_km2(R_EARTH_KM, theta_given)
    flat_area = flat_disk_area_km2(R_EARTH_KM, theta_given)

    summary = {
        "alpha_deg": ALPHA_DEG,
        "theta_alpha_deg": math.degrees(theta_alpha),
        "theta_given_deg": math.degrees(theta_given),
        "ground_radius_from_alpha_km": R_EARTH_KM * theta_alpha,
        "ground_radius_given_km": GROUND_RADIUS_GIVEN_KM,
        "spherical_cap_area_km2": spherical_area,
        "flat_area_km2": flat_area,
        "area_relative_error": (flat_area - spherical_area) / spherical_area,
        "spherical_cap_area_from_alpha_km2": spherical_cap_area_km2(
            R_EARTH_KM, theta_alpha
        ),
    }
    return pd.DataFrame([summary]), summary


def scan_inclinations(
    theta: float,
    target_band: tuple[float, float],
    num_time_samples: int = NUM_TIME_SAMPLES,
    n_max: int = N_MAX_SEARCH,
    tol: float = COVERAGE_TOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan inclinations and satellite counts for latitude-continuous coverage."""
    inclinations = np.arange(
        INCLINATION_MIN_DEG,
        INCLINATION_MAX_DEG + INCLINATION_STEP_DEG / 2.0,
        INCLINATION_STEP_DEG,
    )
    period_seconds = orbital_period_seconds()
    theta_deg = math.degrees(theta)
    target_upper_deg = math.degrees(target_band[1])

    sin_u_cache = {
        N: _sin_u_matrix(num_time_samples, N) for N in range(N_MIN_SEARCH, n_max + 1)
    }
    grid_rows: list[dict[str, float | int]] = []
    min_rows: list[dict[str, float | int | bool]] = []

    for inclination_deg in inclinations:
        best_metrics: dict[str, float | int | np.ndarray] | None = None
        upper_reach_deg = inclination_deg + theta_deg
        feasible_by_upper_bound = upper_reach_deg >= target_upper_deg

        for N in range(N_MIN_SEARCH, n_max + 1):
            metrics = evaluate_latitude_coverage(
                i_deg=float(inclination_deg),
                N=N,
                theta=theta,
                target_band=target_band,
                num_time_samples=num_time_samples,
                return_series=False,
                sin_u=sin_u_cache[N],
                period_seconds=period_seconds,
                tol=tol,
            )
            grid_rows.append(
                {
                    "inclination_deg": float(inclination_deg),
                    "N": N,
                    "spacing_deg": float(metrics["spacing_deg"]),
                    "min_coverage": float(metrics["min_coverage"]),
                    "mean_coverage": float(metrics["mean_coverage"]),
                    "full_coverage_time_ratio": float(
                        metrics["full_coverage_time_ratio"]
                    ),
                    "mean_overlap_ratio": float(metrics["mean_overlap_ratio"]),
                    "max_gap_time_min": float(metrics["max_gap_time_min"]),
                }
            )

            if best_metrics is None and float(metrics["min_coverage"]) >= 1.0 - tol:
                best_metrics = metrics

        min_rows.append(
            {
                "inclination_deg": float(inclination_deg),
                "theta_deg": theta_deg,
                "upper_reach_deg": upper_reach_deg,
                "feasible_by_upper_bound": bool(feasible_by_upper_bound),
                "N_min": np.nan if best_metrics is None else int(best_metrics["N"]),
                "spacing_deg_at_N_min": (
                    np.nan if best_metrics is None else float(best_metrics["spacing_deg"])
                ),
                "mean_overlap_at_N_min": (
                    np.nan
                    if best_metrics is None
                    else float(best_metrics["mean_overlap_ratio"])
                ),
                "max_gap_time_min_at_N_min": (
                    np.nan if best_metrics is None else float(best_metrics["max_gap_time_min"])
                ),
            }
        )

    return pd.DataFrame(min_rows), pd.DataFrame(grid_rows)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.unicode_minus": False,
        }
    )


def plot_single_satellite_geometry(theta: float, alpha_rad: float, output_dir: Path) -> None:
    """Save a 2D cross-section sketch for the single-satellite geometry."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    R = R_EARTH_KM
    rs = R_EARTH_KM + ORBIT_HEIGHT_KM
    earth_angle = np.linspace(0.0, 2.0 * math.pi, 800)
    ax.plot(R * np.cos(earth_angle), R * np.sin(earth_angle), color="#377eb8", lw=2.0)

    orbit_angle = np.linspace(0.0, 2.0 * math.pi, 800)
    ax.plot(
        rs * np.cos(orbit_angle),
        rs * np.sin(orbit_angle),
        color="#bbbbbb",
        lw=1.0,
        ls="--",
        label="Circular orbit",
    )

    O = np.array([0.0, 0.0])
    P = np.array([0.0, R])
    S = np.array([0.0, rs])
    G = np.array([R * math.sin(theta), R * math.cos(theta)])

    ax.plot([O[0], P[0]], [O[1], P[1]], color="#333333", lw=1.5)
    ax.plot([O[0], G[0]], [O[1], G[1]], color="#333333", lw=1.2)
    ax.plot([S[0], P[0]], [S[1], P[1]], color="#984ea3", lw=1.8, label="Nadir")
    ax.plot([S[0], G[0]], [S[1], G[1]], color="#e41a1c", lw=1.8, label="Cone boundary")

    cap_arc = np.linspace(math.pi / 2.0 - theta, math.pi / 2.0, 120)
    ax.plot(
        R * np.cos(cap_arc),
        R * np.sin(cap_arc),
        color="#ff7f00",
        lw=4.0,
        solid_capstyle="round",
        label="Ground coverage arc",
    )

    theta_arc = np.linspace(math.pi / 2.0 - theta, math.pi / 2.0, 80)
    ax.plot(
        0.18 * R * np.cos(theta_arc),
        0.18 * R * np.sin(theta_arc),
        color="#ff7f00",
        lw=2.0,
    )
    ax.text(0.12 * R, 0.16 * R, r"$\theta$", color="#ff7f00", fontsize=14)

    # Draw a small angle arc at the satellite for alpha.
    v_nadir = P - S
    v_boundary = G - S
    angle_nadir = math.atan2(v_nadir[1], v_nadir[0])
    angle_boundary = math.atan2(v_boundary[1], v_boundary[0])
    alpha_arc = np.linspace(angle_nadir, angle_boundary, 80)
    alpha_radius = 0.22 * R
    ax.plot(
        S[0] + alpha_radius * np.cos(alpha_arc),
        S[1] + alpha_radius * np.sin(alpha_arc),
        color="#e41a1c",
        lw=2.0,
    )
    ax.text(0.08 * R, rs - 0.28 * R, r"$\alpha$", color="#e41a1c", fontsize=14)

    for label, point, offset in [
        ("O", O, (-260, -430)),
        ("P", P, (-290, 90)),
        ("G", G, (90, -120)),
        ("S", S, (-270, 120)),
    ]:
        ax.scatter(point[0], point[1], s=35, color="#222222", zorder=5)
        ax.text(point[0] + offset[0], point[1] + offset[1], label, fontsize=13)

    ax.annotate(
        "h",
        xy=(0.0, (R + rs) / 2.0),
        xytext=(-620.0, (R + rs) / 2.0),
        arrowprops={"arrowstyle": "<->", "color": "#4daf4a", "lw": 1.6},
        color="#4daf4a",
        fontsize=13,
        va="center",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.7 * R, 0.95 * R)
    ax.set_ylim(-0.12 * R, 1.18 * rs)
    ax.set_xlabel("Cross-section x (km)")
    ax.set_ylabel("Cross-section y (km)")
    ax.set_title("Single-satellite spherical coverage geometry")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_single_satellite_geometry.png", dpi=300)
    plt.close(fig)


def plot_ground_track_example(output_dir: Path) -> None:
    times, lat, lon = ground_track(
        inclination_deg=50.0,
        num_periods=3.0,
        num_samples=4200,
        satellite_index=0,
        total_satellites=1,
    )
    period = orbital_period_seconds()

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sc = ax.scatter(
        np.degrees(lon),
        np.degrees(lat),
        c=times / period,
        s=7,
        cmap="viridis",
        alpha=0.82,
        linewidths=0.0,
    )
    ax.axhspan(TARGET_LAT_MIN_DEG, TARGET_LAT_MAX_DEG, color="#fdbf6f", alpha=0.20)
    ax.axhline(TARGET_LAT_MIN_DEG, color="#e31a1c", lw=1.2, ls="--", label="30 deg N")
    ax.axhline(TARGET_LAT_MAX_DEG, color="#e31a1c", lw=1.2, ls="-.", label="50 deg N")
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-60.0, 60.0)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Ground-track example with Earth rotation, i = 50 deg")
    ax.legend(loc="lower left")
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Elapsed orbits")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_ground_track_example.png", dpi=300)
    plt.close(fig)


def plot_latitude_coverage_time_example(
    min_df: pd.DataFrame,
    theta: float,
    target_band: tuple[float, float],
    output_dir: Path,
) -> None:
    preferred = min_df.loc[np.isclose(min_df["inclination_deg"], 50.0)]
    if not preferred.empty and not pd.isna(preferred.iloc[0]["N_min"]):
        inclination = 50.0
        N = int(preferred.iloc[0]["N_min"])
    else:
        finite = min_df.dropna(subset=["N_min"])
        if finite.empty:
            inclination = 50.0
            N = N_MAX_SEARCH
        else:
            row = finite.iloc[0]
            inclination = float(row["inclination_deg"])
            N = int(row["N_min"])

    metrics = evaluate_latitude_coverage(
        i_deg=inclination,
        N=N,
        theta=theta,
        target_band=target_band,
        num_time_samples=NUM_TIME_SAMPLES,
        return_series=True,
    )
    time_min = np.asarray(metrics["time_seconds"]) / 60.0
    coverage = np.asarray(metrics["coverage_series"])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(time_min, coverage, color="#1f78b4", lw=1.6)
    ax.axhline(1.0, color="#e31a1c", lw=1.3, ls="--", label="Continuous threshold")
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Time in one orbit (min)")
    ax.set_ylabel("Latitude coverage ratio")
    ax.set_title(
        f"Latitude-projection coverage over one orbit, i = {inclination:.2f} deg, N = {N}"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_latitude_coverage_time_example.png", dpi=300)
    plt.close(fig)


def plot_min_satellites_vs_inclination(
    min_df: pd.DataFrame,
    theta_deg: float,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    infeasible_limit = TARGET_LAT_MAX_DEG - theta_deg
    if infeasible_limit > INCLINATION_MIN_DEG:
        ax.axvspan(
            INCLINATION_MIN_DEG,
            min(infeasible_limit, INCLINATION_MAX_DEG),
            color="#fb9a99",
            alpha=0.22,
            label=r"Necessary upper-reach condition fails",
        )

    finite = min_df.dropna(subset=["N_min"])
    ax.plot(
        finite["inclination_deg"],
        finite["N_min"],
        color="#1f78b4",
        marker="o",
        markersize=3.2,
        lw=1.3,
        label="Numerical N_min",
    )
    ax.set_xlabel("Orbital inclination i (deg)")
    ax.set_ylabel("Minimum satellites in one orbital plane")
    ax.set_title("Minimum N for 30 deg N to 50 deg N latitude-projection continuous coverage")
    ax.set_xlim(INCLINATION_MIN_DEG, INCLINATION_MAX_DEG)
    if not finite.empty:
        ax.set_ylim(max(0.0, finite["N_min"].min() - 2.0), finite["N_min"].max() + 3.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_min_satellites_vs_inclination.png", dpi=300)
    plt.close(fig)


def plot_spacing_overlap_curves(
    min_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    representative_inclinations = [46.0, 50.0, 55.0, 60.0]
    colors = ["#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a"]

    fig, (ax_cov, ax_ov) = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)

    for target_i, color in zip(representative_inclinations, colors):
        available = np.asarray(sorted(grid_df["inclination_deg"].unique()))
        nearest_i = float(available[np.argmin(np.abs(available - target_i))])
        sub = grid_df.loc[np.isclose(grid_df["inclination_deg"], nearest_i)].sort_values(
            "spacing_deg"
        )
        label = f"i = {nearest_i:.0f} deg"
        ax_cov.plot(
            sub["spacing_deg"],
            sub["full_coverage_time_ratio"],
            color=color,
            lw=1.5,
            label=label,
        )
        ax_ov.plot(
            sub["spacing_deg"],
            sub["mean_overlap_ratio"],
            color=color,
            lw=1.5,
            label=label,
        )

        min_row = min_df.loc[np.isclose(min_df["inclination_deg"], nearest_i)]
        if not min_row.empty and not pd.isna(min_row.iloc[0]["spacing_deg_at_N_min"]):
            threshold_spacing = float(min_row.iloc[0]["spacing_deg_at_N_min"])
            ax_cov.axvline(threshold_spacing, color=color, ls="--", alpha=0.32)
            ax_ov.axvline(threshold_spacing, color=color, ls="--", alpha=0.32)

    ax_cov.axhline(1.0, color="#e31a1c", ls="--", lw=1.1)
    ax_cov.set_ylabel("Full coverage time ratio")
    ax_cov.set_title("Spacing, full-coverage time ratio, and redundant overlap")
    ax_cov.legend(loc="lower left", ncol=2)

    ax_ov.set_xlabel("Orbital phase spacing 360/N (deg)")
    ax_ov.set_ylabel("Mean overlap ratio")
    ax_ov.legend(loc="upper right", ncol=2)
    ax_ov.invert_xaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_spacing_overlap_curves.png", dpi=300)
    plt.close(fig)


def _format_inclination_ranges(values: list[float], step: float = INCLINATION_STEP_DEG) -> str:
    if not values:
        return "无"
    ranges: list[tuple[float, float]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if abs(value - prev - step) <= 1e-9:
            prev = value
        else:
            ranges.append((start, prev))
            start = prev = value
    ranges.append((start, prev))
    chunks = [f"{lo:.2f} deg" if lo == hi else f"{lo:.2f}-{hi:.2f} deg" for lo, hi in ranges]
    return ", ".join(chunks)


def _representative_table_markdown(min_df: pd.DataFrame) -> str:
    rows = []
    for target_i in [46.0, 50.0, 55.0, 60.0]:
        sub = min_df.iloc[(min_df["inclination_deg"] - target_i).abs().argsort()[:1]]
        row = sub.iloc[0]
        n_text = "不可行" if pd.isna(row["N_min"]) else str(int(row["N_min"]))
        spacing_text = "-" if pd.isna(row["spacing_deg_at_N_min"]) else f"{row['spacing_deg_at_N_min']:.3f}"
        overlap_text = "-" if pd.isna(row["mean_overlap_at_N_min"]) else f"{row['mean_overlap_at_N_min']:.4f}"
        gap_text = "-" if pd.isna(row["max_gap_time_min_at_N_min"]) else f"{row['max_gap_time_min_at_N_min']:.4f}"
        rows.append(
            f"| {row['inclination_deg']:.2f} | {n_text} | {spacing_text} | {overlap_text} | {gap_text} |"
        )
    return "\n".join(rows)


def generate_method_notes(
    summary: dict[str, float],
    min_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Generate detailed Chinese method notes from actual computed results."""
    finite = min_df.dropna(subset=["N_min"]).copy()
    if finite.empty:
        best_n_text = "本次搜索范围内没有倾角满足连续覆盖"
        best_i_text = "无"
    else:
        best_n = int(finite["N_min"].min())
        best_values = sorted(finite.loc[finite["N_min"] == best_n, "inclination_deg"].tolist())
        best_n_text = str(best_n)
        best_i_text = _format_inclination_ranges(best_values)

    infeasible_upper = min_df.loc[~min_df["feasible_by_upper_bound"], "inclination_deg"].tolist()
    infeasible_upper_text = _format_inclination_ranges(sorted(infeasible_upper))

    no_n_found = min_df.loc[
        min_df["feasible_by_upper_bound"] & min_df["N_min"].isna(), "inclination_deg"
    ].tolist()
    no_n_found_text = _format_inclination_ranges(sorted(no_n_found))

    rep_table = _representative_table_markdown(min_df)
    period_min = orbital_period_seconds() / 60.0
    theta_deg = summary["theta_given_deg"]
    theta_alpha_deg = summary["theta_alpha_deg"]

    notes = f"""# 问题一：单轨道面覆盖特性分析方法说明

本文档由 `problem1_latitude_coverage.py` 根据实际计算结果自动生成，面向论文撰写者。它只讨论问题一中的单颗卫星覆盖几何、同一轨道面内均匀分布卫星的星下点轨迹，以及 30 deg N 到 50 deg N 纬度带的纬度投影连续覆盖能力。这里的连续覆盖不是完整二维经纬度区域覆盖；完整二维区域覆盖需要在后续问题中引入经纬度网格、多个轨道面和升交点布局。

## 问题一建模思路总览

问题一的作用是为后续多轨道面星座优化建立基础模型。建模内容可以分成四层：第一层是单颗卫星的地面覆盖几何，用覆盖地心角描述球面覆盖范围；第二层是同一圆轨道面内 N 颗卫星均匀分布时的星下点经纬度轨迹；第三层是在目标纬度带上构造一维区间并集，定义纬度投影覆盖率和重叠率；第四层是在给定倾角范围内枚举卫星数，搜索满足连续覆盖判据的最小 N。

本模型严格限定在问题一。经度模型用于说明星下点轨迹和地球自转影响，但最终连续覆盖判据只作用在纬度投影方向，不等价于对目标区域每一个经纬度网格点的持续覆盖。

## 单星覆盖几何模型

设地球半径为 R，轨道高度为 h，卫星到地心距离为 r_s=R+h，通信天线半锥角为 alpha。单颗卫星天线波束与地球球面相交后，在球面上形成以星下点为中心的球冠区域。球冠大小可以用覆盖边界点与星下点之间的地心角 theta 表征。只要知道 theta，地面大圆弧覆盖半径就是 r_c=R theta，球冠面积为

```text
A = 2 pi R^2 (1 - cos theta).
```

在天线锥没有超过地平线时，题目给定口径可写为

```text
theta_alpha = asin(((R+h)/R) sin alpha) - alpha.
```

程序同时计算由题目直接给定覆盖半径 506 km 得到的

```text
theta_given = 506 / R.
```

本次主仿真采用 `theta_given`，原因是题目明确给出 Ku 波段半锥角对应的地面覆盖半径约 506 km。由 alpha 反推的 theta 作为一致性校核列入表格。根据当前参数，`theta_alpha = {theta_alpha_deg:.6f} deg`，对应地面弧长 `{summary['ground_radius_from_alpha_km']:.3f} km`；`theta_given = {theta_deg:.6f} deg`。采用题目给定覆盖半径时，球冠面积为 `{summary['spherical_cap_area_km2']:.3f} km^2`，局部平面近似面积为 `{summary['flat_area_km2']:.3f} km^2`，二者相对差为 `{summary['area_relative_error']:.6%}`。

## 星下点经纬度轨迹模型

轨道按圆轨道处理，半长轴 a=R+h，平均角速度和周期分别为

```text
n = sqrt(mu/a^3),   T = 2 pi / n.
```

当前默认参数下轨道周期为 `{period_min:.3f} min`。同一轨道面内 N 颗卫星沿轨道均匀分布，第 k 颗卫星的纬度辐角写为

```text
u_k(t) = u0 + n t + 2 pi k / N,   k = 0,1,...,N-1.
```

星下点纬度为

```text
phi_k(t) = asin(sin i sin u_k(t)).
```

因此轨道倾角 i 决定星下点可达到的最高纬度，N 决定同轨相邻卫星的轨道相位间隔。完整星下点轨迹还必须包含经度。采用简化二体圆轨道叠加地球自转模型，升交点赤经 Omega 默认取 0，经度为

```text
lambda_k(t) = Omega + atan2(cos i sin u_k(t), cos u_k(t)) - omega_E t.
```

这里必须使用 `atan2`，不能用 `arctan(cos(i) tan(u))`，否则在跨象限时会出现经度跳变错误。地球自转项 `-omega_E t` 主要影响星下点经度，使同一轨道面的地面轨迹在经度方向逐圈偏移。但在本题采用的纬度投影覆盖模型中，纬度公式不含 omega_E，因此地球自转不直接改变纬度覆盖率。换言之，经度模型用于轨迹展示和自转影响讨论，而不参与本题问题一的纬度连续覆盖判据。

## 纬度投影覆盖率模型

问题一第三问按已经确定的口径处理：单轨道面只分析目标纬度带的基础纬度覆盖能力，不要求完整二维经纬度区域连续覆盖。于是可以把问题转化为一维区间覆盖。设目标纬度带为

```text
B = [phi_a, phi_b] = [30 deg, 50 deg].
```

在时刻 t，第 k 颗卫星的星下点纬度为 phi_k(t)，其纬度方向覆盖区间定义为

```text
I_k(t) = [phi_k(t)-theta, phi_k(t)+theta].
```

程序先把每个 I_k(t) 与目标带 B 求交集，再合并所有交集区间，得到区间并集长度 L_union(t)。目标带宽为 L_B=phi_b-phi_a，瞬时纬度覆盖率定义为

```text
C_lat(t) = L_union(t) / L_B.
```

当 C_lat(t)=1 时，说明该时刻目标纬度带在纬度投影方向被连续覆盖。一个轨道周期内所有采样时刻均满足 C_lat(t)>=1-tol，则认为该倾角和卫星数可以实现纬度投影意义上的连续覆盖。

相邻星下点纬度差不超过 2 theta 可以作为中间不断开的直观条件，但这不是完整判据。完整判断还必须确认目标纬度带的上下边界都被覆盖，并处理多颗卫星区间交叠和目标带裁剪。因此程序采用区间并集法，而不是只检查相邻纬度差。

程序还计算总覆盖长度

```text
L_sum(t) = sum length(I_k(t) intersect B),
```

以及重复覆盖长度

```text
L_overlap(t) = max(0, L_sum(t)-L_union(t)).
```

瞬时重叠率为 eta_overlap(t)=L_overlap(t)/L_B。输出指标包括最小覆盖率、平均覆盖率、全覆盖时间比例、平均重叠率，以及周期意义下最长未全覆盖空档时间。

## 最小卫星数搜索方法

程序对 i=40 deg 到 60 deg、步长 0.25 deg 的倾角序列进行扫描。对每个倾角，从 N=1 到 80 递增枚举。每一个 (i,N) 组合都在一个轨道周期内采样 `{NUM_TIME_SAMPLES}` 个时刻，计算 C_lat(t) 的最小值。若第一个 N 满足

```text
min_t C_lat(t) >= 1 - tol,
```

则记录为该倾角下的最小卫星数 N_min(i)。

必要的几何可行性判断是

```text
i + theta >= 50 deg.
```

若 i+theta<50 deg，即使同一轨道面内卫星数量无限增多，星下点最高纬度加覆盖地心角仍无法达到目标纬度带上边界 50 deg，因此该倾角在几何上不可行。当前采用 theta_given 时，因上边界条件失败而不可行的倾角范围为：{infeasible_upper_text}。但这个条件只是必要条件，最终是否连续覆盖仍以时间采样区间并集结果为准。在几何上通过上界判断但 N<=80 仍未找到连续覆盖方案的倾角范围为：{no_n_found_text}。

## 卫星间距与覆盖重叠率

同一轨道面内卫星均匀分布，轨道相位间距定义为

```text
spacing = 360 deg / N.
```

spacing 越小表示同轨卫星越密。随着 N 增加，目标纬度带内被覆盖的机会增加，连续覆盖阈值更容易达到；同时多个卫星对同一纬度段的重复覆盖也增加。程序用 L_sum-L_union 定义冗余覆盖长度，并用其相对目标带宽的均值作为 mean_overlap_ratio。

刚好达到连续覆盖的 N_min 对应最小可行方案。继续增加 N 通常会提高冗余覆盖和鲁棒性，但也会增加单轨道面卫星规模，因此论文中应把 N_min 作为规模下限，把更大 N 解释为冗余设计选择。

## 结果解释模板

程序输出的主要文件位于 `{output_dir.as_posix()}/`。单星覆盖几何结果在 `single_satellite_coverage_summary.csv`，倾角最小卫星数结果在 `inclination_min_satellites.csv`，完整扫描网格在 `coverage_metrics_grid.csv`。

当前参数下，主仿真采用的覆盖地心角为 `{theta_deg:.6f} deg`，地面覆盖半径为 `{summary['ground_radius_given_km']:.3f} km`，球冠覆盖面积为 `{summary['spherical_cap_area_km2']:.3f} km^2`。在所有扫描倾角中，最小的 N_min 为 `{best_n_text}`，对应倾角范围为：{best_i_text}。

代表性倾角的结果如下，数值均来自 `inclination_min_satellites.csv`：

| inclination_deg | N_min | spacing_deg_at_N_min | mean_overlap_at_N_min | max_gap_time_min_at_N_min |
|---:|---:|---:|---:|---:|
{rep_table}

若某些倾角显示不可行，应先检查 `feasible_by_upper_bound`。若该列为 False，原因是最高可覆盖纬度不足以达到 50 deg N。若该列为 True 但 N_min 为空，则表示虽然几何上最高纬度可达上边界，但在本程序 N<=80 的枚举范围内，一个轨道周期内仍存在纬度覆盖空档，最终判据没有通过。

## 图表使用说明

- `fig_single_satellite_geometry.png` 展示 R、h、alpha、theta 和覆盖边界之间的几何关系。
- `fig_ground_track_example.png` 展示 i=50 deg 的经纬度二维星下点轨迹，并标出 30 deg N 到 50 deg N 目标带，说明星下点轨迹不只有纬度。
- `fig_latitude_coverage_time_example.png` 展示代表性可行方案在一个轨道周期内的 C_lat(t)，y=1 参考线即连续覆盖判据。
- `fig_min_satellites_vs_inclination.png` 展示不同倾角下所需的最小单轨道面卫星数。
- `fig_spacing_overlap_curves.png` 展示代表性倾角下相位间距、全覆盖时间比例和平均重叠率之间的关系。

再次强调，本模型解决的是问题一中的单轨道面纬度投影覆盖分析，不等价于目标区域的完整二维连续覆盖。完整二维覆盖需要在问题二中引入经纬度网格、多个轨道面和升交点布局。
"""
    return notes


def write_method_notes(
    summary: dict[str, float],
    min_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    notes = generate_method_notes(summary, min_df, grid_df, output_dir)
    METHOD_NOTES_PATH.write_text(notes, encoding="utf-8")
    METHOD_NOTES_DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METHOD_NOTES_DOCS_PATH.write_text(notes, encoding="utf-8")


def write_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run all computations and save tables, figures, and method notes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_plot_style()

    single_df, summary = build_single_satellite_summary()
    theta = GROUND_RADIUS_GIVEN_KM / R_EARTH_KM
    target_band = (math.radians(TARGET_LAT_MIN_DEG), math.radians(TARGET_LAT_MAX_DEG))

    print("Scanning inclination and satellite-count grid...")
    min_df, grid_df = scan_inclinations(theta=theta, target_band=target_band)

    single_df.to_csv(OUTPUT_DIR / "single_satellite_coverage_summary.csv", index=False)
    min_df.to_csv(OUTPUT_DIR / "inclination_min_satellites.csv", index=False)
    grid_df.to_csv(OUTPUT_DIR / "coverage_metrics_grid.csv", index=False)

    plot_single_satellite_geometry(theta, math.radians(ALPHA_DEG), OUTPUT_DIR)
    plot_ground_track_example(OUTPUT_DIR)
    plot_latitude_coverage_time_example(min_df, theta, target_band, OUTPUT_DIR)
    plot_min_satellites_vs_inclination(min_df, summary["theta_given_deg"], OUTPUT_DIR)
    plot_spacing_overlap_curves(min_df, grid_df, OUTPUT_DIR)
    write_method_notes(summary, min_df, grid_df, OUTPUT_DIR)

    return single_df, min_df, grid_df, summary


def print_summary(min_df: pd.DataFrame, summary: dict[str, float]) -> None:
    """Print a concise terminal summary after successful computation."""
    finite = min_df.dropna(subset=["N_min"]).copy()
    period_min = orbital_period_seconds() / 60.0

    print("\nProblem 1 latitude-projection coverage summary")
    print("-" * 56)
    print(f"theta_given_deg: {summary['theta_given_deg']:.6f} deg")
    print(f"spherical_cap_area: {summary['spherical_cap_area_km2']:.3f} km^2")
    print(f"orbital_period: {period_min:.3f} min")

    print("\nRepresentative inclinations:")
    for target_i in [46.0, 50.0, 55.0, 60.0]:
        row = min_df.iloc[(min_df["inclination_deg"] - target_i).abs().argsort()[:1]].iloc[0]
        if pd.isna(row["N_min"]):
            print(f"  i = {row['inclination_deg']:.2f} deg: no feasible N <= {N_MAX_SEARCH}")
        else:
            print(
                "  "
                f"i = {row['inclination_deg']:.2f} deg: "
                f"N_min = {int(row['N_min'])}, "
                f"spacing = {row['spacing_deg_at_N_min']:.3f} deg, "
                f"mean_overlap = {row['mean_overlap_at_N_min']:.4f}"
            )

    if finite.empty:
        print(f"\nNo inclination reached continuous coverage for N <= {N_MAX_SEARCH}.")
    else:
        best_n = int(finite["N_min"].min())
        best_inclinations = sorted(
            finite.loc[finite["N_min"] == best_n, "inclination_deg"].tolist()
        )
        print(
            f"\nSmallest N_min over all inclinations: {best_n} "
            f"at {_format_inclination_ranges(best_inclinations)}"
        )

    print(f"\nOutputs written to: {OUTPUT_DIR.as_posix()}")
    print(f"Method notes written to: {METHOD_NOTES_PATH.as_posix()}")


def main() -> None:
    single_df, min_df, grid_df, summary = write_outputs()
    _ = (single_df, grid_df)
    print_summary(min_df, summary)


if __name__ == "__main__":
    main()
