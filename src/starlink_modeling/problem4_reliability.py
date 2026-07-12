"""Sparse representative coverage and event-driven annual reliability simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .problem2_grid import GroundGrid, make_regular_grid
from .problem2_orbit import (
    WalkerConstellation,
    generate_walker_constellation,
    propagate_subpoint_unit_vectors,
)


HOURS_PER_YEAR = 8760.0


@dataclass(frozen=True)
class ExtraPlane:
    extra_plane_index: int
    raan_deg: float
    phase_offset_deg: float
    satellite_count: int


@dataclass
class CoverageArchive:
    grid: GroundGrid
    times_seconds: np.ndarray
    incidence: list[csr_matrix]
    counts: list[np.ndarray]
    total_satellites: int
    plane_index: np.ndarray
    baseline_effective_availability: float
    critical_maneuver_fraction: float
    critical_failure_fraction: float
    minimum_coverage_multiplicity: int
    mean_coverage_multiplicity: float


def constellation_with_extra_planes(selected: Any, extra_planes: list[ExtraPlane]) -> WalkerConstellation:
    base = generate_walker_constellation(
        selected.M,
        selected.N,
        selected.inclination_deg,
        selected.phase_factor_F,
        selected.Omega0_deg,
        selected.u0_deg,
        selected.altitude_km,
    )
    if not extra_planes:
        return base
    plane_parts = [base.plane_index]
    satellite_parts = [base.satellite_index]
    raan_parts = [base.raan_rad]
    u_parts = [base.initial_argument_of_latitude_rad]
    q = np.arange(selected.N, dtype=int)
    for offset, plane in enumerate(extra_planes):
        plane_id = selected.M + offset
        plane_parts.append(np.full(selected.N, plane_id, dtype=int))
        satellite_parts.append(q.copy())
        raan_parts.append(np.full(selected.N, math.radians(plane.raan_deg)))
        u_parts.append(
            math.radians(selected.u0_deg + plane.phase_offset_deg)
            + 2.0 * math.pi * q / selected.N
        )
    return WalkerConstellation(
        M=selected.M + len(extra_planes),
        N=selected.N,
        inclination_rad=base.inclination_rad,
        phase_factor_F=selected.phase_factor_F,
        Omega0_rad=base.Omega0_rad,
        u0_rad=base.u0_rad,
        altitude_km=base.altitude_km,
        plane_index=np.concatenate(plane_parts),
        satellite_index=np.concatenate(satellite_parts),
        raan_rad=np.concatenate(raan_parts),
        initial_argument_of_latitude_rad=np.concatenate(u_parts),
    )


def _incidence_from_satellites(
    grid_tree: cKDTree,
    grid_size: int,
    satellite_vectors: np.ndarray,
    theta_rad: float,
) -> csr_matrix:
    chord_radius = 2.0 * math.sin(float(theta_rad) / 2.0)
    neighborhoods = grid_tree.query_ball_point(satellite_vectors, chord_radius)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for satellite_id, indices in enumerate(neighborhoods):
        if indices:
            values = np.asarray(indices, dtype=int)
            rows.append(values)
            columns.append(np.full(values.size, satellite_id, dtype=int))
    if not rows:
        return csr_matrix((grid_size, satellite_vectors.shape[0]), dtype=np.float32)
    row = np.concatenate(rows)
    column = np.concatenate(columns)
    data = np.ones(row.size, dtype=np.float32)
    return csr_matrix((data, (row, column)), shape=(grid_size, satellite_vectors.shape[0]))


def build_coverage_archive(
    selected: Any,
    *,
    grid_step_deg: float,
    representative_hours: float,
    time_step_minutes: float,
    extra_planes: list[ExtraPlane] | None = None,
) -> CoverageArchive:
    extra_planes = list(extra_planes or [])
    constellation = constellation_with_extra_planes(selected, extra_planes)
    grid = make_regular_grid(float(grid_step_deg))
    times = np.arange(
        0.0,
        float(representative_hours) * 3600.0,
        float(time_step_minutes) * 60.0,
    )
    if times.size == 0:
        times = np.array([0.0])
    tree = cKDTree(grid.unit_vectors)
    matrices: list[csr_matrix] = []
    counts_list: list[np.ndarray] = []
    critical_pairs = 0
    full_snapshots = 0
    multiplicity_sum = 0.0
    minimum = np.iinfo(np.int32).max
    for batch_start in range(0, times.size, 12):
        vectors_batch = propagate_subpoint_unit_vectors(
            constellation, times[batch_start : batch_start + 12]
        )
        for vectors in vectors_batch:
            matrix = _incidence_from_satellites(
                tree, grid.size, vectors, math.radians(selected.theta_deg)
            )
            counts = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int16)
            matrices.append(matrix)
            counts_list.append(counts)
            full_snapshots += int(np.all(counts >= 1))
            minimum = min(minimum, int(counts.min()))
            multiplicity_sum += float(grid.weights @ counts)
            single_rows = np.flatnonzero(counts == 1)
            if single_rows.size:
                critical_pairs += np.unique(matrix[single_rows].indices).size
    denominator = max(1, len(matrices) * constellation.total_satellites)
    critical_fraction = critical_pairs / denominator
    return CoverageArchive(
        grid=grid,
        times_seconds=times,
        incidence=matrices,
        counts=counts_list,
        total_satellites=constellation.total_satellites,
        plane_index=constellation.plane_index,
        baseline_effective_availability=full_snapshots / len(matrices),
        critical_maneuver_fraction=float(critical_fraction),
        critical_failure_fraction=float(critical_fraction),
        minimum_coverage_multiplicity=int(minimum),
        mean_coverage_multiplicity=multiplicity_sum / len(matrices),
    )


def effective_coverage(counts_by_grid: np.ndarray, state_losses_by_grid: np.ndarray) -> bool:
    counts = np.asarray(counts_by_grid, dtype=float)
    losses = np.asarray(state_losses_by_grid, dtype=float)
    if counts.shape != losses.shape:
        raise ValueError("Coverage counts and losses must have the same shape.")
    return bool(np.all(counts - losses >= 1.0 - 1e-12))


def interval_union_hours(intervals: list[tuple[float, float]]) -> float:
    valid = sorted((max(0.0, a), min(HOURS_PER_YEAR, b)) for a, b in intervals if b > a)
    if not valid:
        return 0.0
    total = 0.0
    start, stop = valid[0]
    for next_start, next_stop in valid[1:]:
        if next_start <= stop:
            stop = max(stop, next_stop)
        else:
            total += stop - start
            start, stop = next_start, next_stop
    return total + stop - start


def simulate_reliability_year(
    rng: np.random.Generator,
    archive: CoverageArchive,
    *,
    base_M: int,
    base_N: int,
    in_plane_spares_per_plane: int,
    ground_spares: int,
    annual_maneuvers_per_satellite: float,
    annual_failure_hazard_per_satellite: float,
    degraded_duration_samples_hours: np.ndarray,
    adjustment_hours: float,
    takeover_hours: float,
    ground_replacement_hours: float,
    standby_risk_factor: float,
) -> dict[str, float | int]:
    service_satellites = archive.total_satellites
    standby_satellites = base_M * int(in_plane_spares_per_plane)
    active_in_orbit = service_satellites + standby_satellites
    durations = np.asarray(degraded_duration_samples_hours, dtype=float)
    if durations.size == 0:
        durations = np.array([1.0])

    total_avoidances = int(rng.poisson(active_in_orbit * annual_maneuvers_per_satellite))
    service_avoidances = int(rng.poisson(service_satellites * annual_maneuvers_per_satellite))
    critical_avoidances = int(
        rng.poisson(
            service_satellites
            * annual_maneuvers_per_satellite
            * archive.critical_maneuver_fraction
        )
    )
    effective_intervals: list[tuple[float, float]] = []
    geometric_intervals: list[tuple[float, float]] = []
    for _ in range(critical_avoidances):
        start = float(rng.uniform(0.0, HOURS_PER_YEAR))
        duration = float(rng.choice(durations))
        effective_intervals.append((start, start + duration))

    failure_count = int(rng.poisson(service_satellites * annual_failure_hazard_per_satellite))
    failure_times = np.sort(rng.uniform(0.0, HOURS_PER_YEAR, size=failure_count))
    failure_satellites = rng.integers(0, service_satellites, size=failure_count)
    plane_spares = np.full(base_M, int(in_plane_spares_per_plane), dtype=int)
    standby_failures = rng.poisson(
        int(in_plane_spares_per_plane) * annual_failure_hazard_per_satellite * standby_risk_factor,
        size=base_M,
    )
    plane_spares = np.maximum(0, plane_spares - standby_failures)
    critical_failure_count = 0
    capacity_failure_hours = 0.0
    ground_used = 0
    launch_times: list[float] = []
    failure_coverage_records: list[tuple[float, float, int]] = []

    representative_step_hours = (
        float(np.median(np.diff(archive.times_seconds))) / 3600.0
        if archive.times_seconds.size > 1
        else 1.0
    )
    representative_hours = representative_step_hours * max(1, len(archive.incidence))

    def state_break_intervals(
        start_hour: float,
        stop_hour: float,
        satellite_ids: list[int],
        losses: list[float],
    ) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        cursor = max(0.0, float(start_hour))
        stop_value = min(HOURS_PER_YEAR, float(stop_hour))
        while cursor < stop_value - 1e-12:
            phase_hour = cursor % representative_hours
            phase_index = min(
                len(archive.incidence) - 1,
                int(phase_hour // representative_step_hours),
            )
            next_phase = cursor + (
                representative_step_hours - phase_hour % representative_step_hours
            )
            segment_stop = min(stop_value, next_phase)
            matrix = archive.incidence[phase_index]
            loss_by_grid = np.asarray(
                matrix[:, satellite_ids] @ np.asarray(losses, dtype=float)
            ).ravel()
            if np.any(archive.counts[phase_index] - loss_by_grid < 1.0 - 1e-12):
                intervals.append((cursor, segment_stop))
            cursor = segment_stop
        return intervals

    for event_time, satellite_id in zip(failure_times, failure_satellites):
        plane = int(archive.plane_index[int(satellite_id)])
        replacement_delay = HOURS_PER_YEAR - float(event_time)
        coverage_outage = float(adjustment_hours)
        if plane < base_M and plane_spares[plane] > 0:
            plane_spares[plane] -= 1
            coverage_outage = float(takeover_hours)
            replacement_delay = float(takeover_hours)
        elif ground_used < int(ground_spares):
            ground_used += 1
            replacement_delay = min(replacement_delay, float(ground_replacement_hours))
            launch_times.append(float(event_time))
        capacity_failure_hours += replacement_delay
        outage_stop = min(HOURS_PER_YEAR, float(event_time) + coverage_outage)
        failure_only = state_break_intervals(
            float(event_time), outage_stop, [int(satellite_id)], [1.0]
        )
        if failure_only:
            critical_failure_count += 1
            effective_intervals.extend(failure_only)
            geometric_intervals.extend(failure_only)

        # A single failure may leave exactly one full equivalent satellite. A
        # simultaneous 50%-capacity maneuver can then break c_eff even when
        # neither event is critical by itself. Only events inside the sparse
        # failure interval need exact grid verification.
        overlap_mean = (
            service_satellites
            * annual_maneuvers_per_satellite
            * max(0.0, outage_stop - float(event_time))
            / HOURS_PER_YEAR
        )
        for _ in range(int(rng.poisson(overlap_mean))):
            maneuver_start = float(rng.uniform(float(event_time), outage_stop))
            maneuver_duration = float(rng.choice(durations))
            maneuver_stop = min(outage_stop, maneuver_start + maneuver_duration)
            maneuver_satellite = int(rng.integers(0, service_satellites))
            if maneuver_satellite == int(satellite_id):
                continue
            effective_intervals.extend(
                state_break_intervals(
                    maneuver_start,
                    maneuver_stop,
                    [int(satellite_id), maneuver_satellite],
                    [1.0, 0.5],
                )
            )

        for previous_start, previous_stop, previous_satellite in failure_coverage_records:
            overlap_start = max(previous_start, float(event_time))
            overlap_stop = min(previous_stop, outage_stop)
            if overlap_stop > overlap_start:
                pair_intervals = state_break_intervals(
                    overlap_start,
                    overlap_stop,
                    [previous_satellite, int(satellite_id)],
                    [1.0, 1.0],
                )
                effective_intervals.extend(pair_intervals)
                geometric_intervals.extend(pair_intervals)
        failure_coverage_records.append(
            (float(event_time), outage_stop, int(satellite_id))
        )

    maneuver_capacity_loss_hours = 0.5 * service_avoidances * float(np.mean(durations))
    capacity_loss_ratio = min(
        1.0,
        (maneuver_capacity_loss_hours + capacity_failure_hours)
        / (service_satellites * HOURS_PER_YEAR),
    )
    effective_event_loss = interval_union_hours(effective_intervals) / HOURS_PER_YEAR
    effective_availability = archive.baseline_effective_availability * (1.0 - effective_event_loss)
    geometric_event_loss = interval_union_hours(geometric_intervals) / HOURS_PER_YEAR
    geometric_availability = archive.baseline_effective_availability * (1.0 - geometric_event_loss)
    launches = int(math.ceil(ground_used / 60.0)) if ground_used else 0
    return {
        "effective_coverage_availability": float(np.clip(effective_availability, 0.0, 1.0)),
        "geometric_coverage_availability": float(np.clip(geometric_availability, 0.0, 1.0)),
        "capacity_availability": float(np.clip(1.0 - capacity_loss_ratio, 0.0, 1.0)),
        "annual_avoidance_count": total_avoidances,
        "annual_failure_count": failure_count,
        "critical_maneuver_count": critical_avoidances,
        "critical_failure_count": critical_failure_count,
        "standby_failure_count": int(np.sum(standby_failures)),
        "ground_spares_used": ground_used,
        "replenishment_launch_count": launches,
        "remaining_in_orbit_spares": int(np.sum(plane_spares)),
    }
