"""Memory-conscious regional coverage evaluation for Walker constellations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .problem2_grid import GroundGrid
from .problem2_orbit import WalkerConstellation, propagate_subpoint_unit_vectors

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only in scipy-free environments
    cKDTree = None


@dataclass
class CoverageEvaluation:
    """Coverage summary plus optional time/grid detail for reporting."""

    metrics: dict[str, Any]
    time_metrics: pd.DataFrame | None = None
    grid_metrics: pd.DataFrame | None = None
    worst_single_counts: np.ndarray | None = None
    worst_double_counts: np.ndarray | None = None


def coverage_counts_direct(
    grid_unit_vectors: np.ndarray,
    satellite_unit_vectors: np.ndarray,
    theta_rad: float,
    grid_chunk_size: int = 4096,
) -> np.ndarray:
    """Count visible satellites by chunked dot products without arccos."""

    grid = np.asarray(grid_unit_vectors, dtype=float)
    satellites = np.asarray(satellite_unit_vectors, dtype=float)
    counts = np.zeros(grid.shape[0], dtype=np.int32)
    threshold = math.cos(theta_rad)
    for start in range(0, grid.shape[0], grid_chunk_size):
        stop = min(start + grid_chunk_size, grid.shape[0])
        counts[start:stop] = np.count_nonzero(
            grid[start:stop] @ satellites.T >= threshold, axis=1
        )
    return counts


def coverage_counts_kdtree(
    grid_tree: Any,
    grid_size: int,
    satellite_unit_vectors: np.ndarray,
    theta_rad: float,
) -> np.ndarray:
    """Count visible satellites using unit-sphere chord neighborhoods."""

    if cKDTree is None:
        raise RuntimeError("scipy is not available; KDTree coverage cannot be used.")
    chord_radius = 2.0 * math.sin(theta_rad / 2.0)
    neighborhoods = grid_tree.query_ball_point(
        np.asarray(satellite_unit_vectors, dtype=float), chord_radius
    )
    nonempty = [np.asarray(indices, dtype=int) for indices in neighborhoods if indices]
    if not nonempty:
        return np.zeros(grid_size, dtype=np.int32)
    return np.bincount(np.concatenate(nonempty), minlength=grid_size).astype(np.int32)


def max_false_gap_samples(flags: np.ndarray, periodic: bool = False) -> int:
    """Return the longest False run, optionally joining the two endpoints."""

    flags = np.asarray(flags, dtype=bool).ravel()
    if flags.size == 0 or np.all(flags):
        return 0
    if not np.any(flags):
        return int(flags.size)

    false_values = ~flags
    padded = np.concatenate(([False], false_values, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    longest = int(np.max(stops - starts))
    if periodic and false_values[0] and false_values[-1]:
        leading = int(np.argmax(flags))
        trailing = int(flags.size - 1 - np.flatnonzero(flags)[-1])
        longest = max(longest, leading + trailing)
    return longest


def _update_gap_state(
    covered: np.ndarray,
    current: np.ndarray,
    maximum: np.ndarray,
    leading: np.ndarray,
    seen_covered: np.ndarray,
) -> None:
    leading += ((~seen_covered) & (~covered)).astype(np.int32)
    seen_covered |= covered
    current[:] = np.where(covered, 0, current + 1)
    np.maximum(maximum, current, out=maximum)


def _nearest_margin_rad(
    grid_vectors: np.ndarray, satellite_vectors: np.ndarray, theta_rad: float
) -> float:
    if cKDTree is not None:
        distances, _ = cKDTree(satellite_vectors).query(grid_vectors, k=1)
        central_angles = 2.0 * np.arcsin(np.clip(distances / 2.0, 0.0, 1.0))
        return float(np.min(theta_rad - central_angles))
    best = np.full(grid_vectors.shape[0], -1.0)
    for start in range(0, grid_vectors.shape[0], 4096):
        stop = min(start + 4096, grid_vectors.shape[0])
        best[start:stop] = np.max(grid_vectors[start:stop] @ satellite_vectors.T, axis=1)
    central_angles = np.arccos(np.clip(best, -1.0, 1.0))
    return float(np.min(theta_rad - central_angles))


def evaluate_constellation_coverage(
    constellation: WalkerConstellation,
    grid: GroundGrid,
    theta_rad: float,
    simulation_days: float,
    time_step_seconds: float,
    *,
    method: str = "auto",
    return_time_series: bool = False,
    return_grid_metrics: bool = False,
    return_snapshots: bool = False,
    periodic_gap: bool = False,
    tolerance: float = 1e-12,
) -> CoverageEvaluation:
    """Evaluate all requested regional coverage metrics without a 3-D cube."""

    duration_seconds = float(simulation_days) * 86400.0
    if duration_seconds <= 0.0 or time_step_seconds <= 0.0:
        raise ValueError("Simulation duration and time step must be positive.")
    times = np.arange(0.0, duration_seconds, float(time_step_seconds))
    if times.size == 0:
        times = np.array([0.0])

    use_kdtree = method == "kdtree" or (method == "auto" and cKDTree is not None)
    if method not in {"auto", "kdtree", "direct"}:
        raise ValueError(f"Unknown coverage method: {method}")
    if method == "kdtree" and cKDTree is None:
        raise RuntimeError("method='kdtree' requires scipy.")
    grid_tree = cKDTree(grid.unit_vectors) if use_kdtree else None

    point_count = grid.size
    sample_count = times.size
    availability1_count = np.zeros(point_count, dtype=np.int32)
    availability2_count = np.zeros(point_count, dtype=np.int32)
    multiplicity_sum = np.zeros(point_count, dtype=np.float64)

    current_gap1 = np.zeros(point_count, dtype=np.int32)
    current_gap2 = np.zeros(point_count, dtype=np.int32)
    max_gap1 = np.zeros(point_count, dtype=np.int32)
    max_gap2 = np.zeros(point_count, dtype=np.int32)
    leading_gap1 = np.zeros(point_count, dtype=np.int32)
    leading_gap2 = np.zeros(point_count, dtype=np.int32)
    seen1 = np.zeros(point_count, dtype=bool)
    seen2 = np.zeros(point_count, dtype=bool)

    A1 = np.empty(sample_count, dtype=float)
    A2 = np.empty(sample_count, dtype=float)
    B1 = np.empty(sample_count, dtype=bool)
    B2 = np.empty(sample_count, dtype=bool)
    weighted_multiplicity = np.empty(sample_count, dtype=float)
    min_margin_rad = math.inf
    worst_single_counts = None
    worst_double_counts = None
    worst_single_value = math.inf
    worst_double_value = math.inf

    # Propagation is batched to avoid materializing time x satellite x xyz.
    propagation_batch_size = 16
    for batch_start in range(0, sample_count, propagation_batch_size):
        batch_stop = min(batch_start + propagation_batch_size, sample_count)
        batch_times = times[batch_start:batch_stop]
        satellite_batches = propagate_subpoint_unit_vectors(constellation, batch_times)
        for offset, satellite_vectors in enumerate(satellite_batches):
            time_index = batch_start + offset
            if use_kdtree:
                counts = coverage_counts_kdtree(
                    grid_tree, point_count, satellite_vectors, theta_rad
                )
            else:
                counts = coverage_counts_direct(
                    grid.unit_vectors, satellite_vectors, theta_rad
                )

            covered1 = counts >= 1
            covered2 = counts >= 2
            availability1_count += covered1
            availability2_count += covered2
            multiplicity_sum += counts
            _update_gap_state(covered1, current_gap1, max_gap1, leading_gap1, seen1)
            _update_gap_state(covered2, current_gap2, max_gap2, leading_gap2, seen2)

            A1[time_index] = float(grid.weights @ covered1)
            A2[time_index] = float(grid.weights @ covered2)
            B1[time_index] = bool(np.all(covered1))
            B2[time_index] = bool(np.all(covered2))
            weighted_multiplicity[time_index] = float(grid.weights @ counts)

            if return_snapshots and A1[time_index] < worst_single_value:
                worst_single_value = A1[time_index]
                worst_single_counts = counts.copy()
            if return_snapshots and A2[time_index] < worst_double_value:
                worst_double_value = A2[time_index]
                worst_double_counts = counts.copy()
            if return_grid_metrics:
                min_margin_rad = min(
                    min_margin_rad,
                    _nearest_margin_rad(grid.unit_vectors, satellite_vectors, theta_rad),
                )

    if periodic_gap:
        max_gap1 = np.maximum(max_gap1, np.minimum(sample_count, leading_gap1 + current_gap1))
        max_gap2 = np.maximum(max_gap2, np.minimum(sample_count, leading_gap2 + current_gap2))

    availability1 = availability1_count / sample_count
    availability2 = availability2_count / sample_count
    point_mean_multiplicity = multiplicity_sum / sample_count
    gap1_minutes = max_gap1 * float(time_step_seconds) / 60.0
    gap2_minutes = max_gap2 * float(time_step_seconds) / 60.0

    worst_grid_index = int(np.argmin(availability1))
    worst_double_grid_index = int(np.argmin(availability2))
    q1 = float(np.mean(B1))
    q2 = float(np.mean(B2))
    metrics: dict[str, Any] = {
        "Q1_full_region": q1,
        "Q2_full_region": q2,
        "P1_space_time": float(np.mean(A1)),
        "P2_space_time": float(np.mean(A2)),
        "mean_multiplicity": float(np.mean(weighted_multiplicity)),
        "worst_A1": float(np.min(A1)),
        "worst_A2": float(np.min(A2)),
        "max_gap_time_min": float(np.max(gap1_minutes)),
        "max_double_gap_time_min": float(np.max(gap2_minutes)),
        "min_availability1": float(np.min(availability1)),
        "min_availability2": float(np.min(availability2)),
        "mean_availability1": float(grid.weights @ availability1),
        "mean_availability2": float(grid.weights @ availability2),
        "std_availability1": float(np.std(availability1)),
        "std_mean_multiplicity": float(np.std(point_mean_multiplicity)),
        "min_angular_margin_deg": (
            math.degrees(min_margin_rad) if math.isfinite(min_margin_rad) else math.nan
        ),
        "worst_grid_latitude_deg": float(grid.latitude_deg[worst_grid_index]),
        "worst_grid_longitude_deg": float(grid.longitude_deg[worst_grid_index]),
        "worst_double_grid_latitude_deg": float(
            grid.latitude_deg[worst_double_grid_index]
        ),
        "worst_double_grid_longitude_deg": float(
            grid.longitude_deg[worst_double_grid_index]
        ),
        "sample_count": int(sample_count),
        "grid_point_count": int(point_count),
        "simulation_duration_seconds": duration_seconds,
        "coverage_method": "kdtree" if use_kdtree else "direct",
        "single_feasible": bool(q1 >= 1.0 - tolerance and np.max(gap1_minutes) <= tolerance),
        "double_feasible": bool(
            q1 >= 1.0 - tolerance and q2 >= 0.95 - tolerance
        ),
    }

    time_metrics = None
    if return_time_series:
        time_metrics = pd.DataFrame(
            {
                "time_seconds": times,
                "time_hours": times / 3600.0,
                "A1_area_coverage": A1,
                "A2_area_coverage": A2,
                "B1_full_region": B1.astype(int),
                "B2_full_region": B2.astype(int),
                "weighted_mean_multiplicity": weighted_multiplicity,
            }
        )

    grid_metrics = None
    if return_grid_metrics:
        grid_metrics = pd.DataFrame(
            {
                "latitude_deg": grid.latitude_deg,
                "longitude_deg": grid.longitude_deg,
                "area_weight": grid.weights,
                "availability1": availability1,
                "availability2": availability2,
                "mean_multiplicity": point_mean_multiplicity,
                "max_gap_time_min": gap1_minutes,
                "max_double_gap_time_min": gap2_minutes,
            }
        )

    return CoverageEvaluation(
        metrics=metrics,
        time_metrics=time_metrics,
        grid_metrics=grid_metrics,
        worst_single_counts=worst_single_counts,
        worst_double_counts=worst_double_counts,
    )

