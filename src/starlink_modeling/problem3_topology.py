"""Time-varying inter-satellite topology for problem three."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import heapq
import json
import math
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .problem2_orbit import (
    R_EARTH_KM,
    WalkerConstellation,
    propagate_constellation,
)


LIGHT_SPEED_KM_S = 299792.458
D_ISL_MAX_KM = 5000.0
TAU_PROCESSING_MS = 0.5


@dataclass
class TopologySnapshot:
    time_seconds: float
    positions_eci_km: np.ndarray
    edges: pd.DataFrame
    crosslink_shifts: pd.DataFrame
    degree: np.ndarray
    metrics: dict[str, object]


def satellite_id(plane_index: int, satellite_index: int, N: int) -> int:
    return int(plane_index) * int(N) + int(satellite_index)


def same_plane_neighbor_distance_km(
    constellation: WalkerConstellation,
    earth_radius_km: float = R_EARTH_KM,
) -> float:
    a_km = earth_radius_km + constellation.altitude_km
    return float(2.0 * a_km * math.sin(math.pi / constellation.N))


def segment_minimum_center_distance_km(
    positions_a_km: np.ndarray, positions_b_km: np.ndarray
) -> np.ndarray:
    """Return the minimum distance from each segment AB to Earth's center."""

    a = np.asarray(positions_a_km, dtype=float)
    b = np.asarray(positions_b_km, dtype=float)
    direction = b - a
    denominator = np.einsum("...i,...i->...", direction, direction)
    safe = np.where(denominator > 0.0, denominator, 1.0)
    fraction = -np.einsum("...i,...i->...", a, direction) / safe
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = a + fraction[..., None] * direction
    return np.linalg.norm(closest, axis=-1)


def has_line_of_sight(
    positions_a_km: np.ndarray,
    positions_b_km: np.ndarray,
    earth_radius_km: float = R_EARTH_KM,
) -> np.ndarray:
    return segment_minimum_center_distance_km(
        positions_a_km, positions_b_km
    ) > float(earth_radius_km)


def link_feasibility(
    positions_a_km: np.ndarray,
    positions_b_km: np.ndarray,
    max_distance_km: float = D_ISL_MAX_KM,
    earth_radius_km: float = R_EARTH_KM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(positions_a_km, dtype=float)
    b = np.asarray(positions_b_km, dtype=float)
    distances = np.linalg.norm(a - b, axis=-1)
    line_of_sight = has_line_of_sight(a, b, earth_radius_km)
    feasible = (distances <= float(max_distance_km)) & line_of_sight
    return feasible, distances, line_of_sight


def cyclic_shift_pairs(N: int, shift: int) -> tuple[np.ndarray, np.ndarray]:
    source = np.arange(int(N), dtype=int)
    target = (source + int(shift)) % int(N)
    return source, target


def select_cyclic_crosslink_matching(
    left_positions_km: np.ndarray,
    right_positions_km: np.ndarray,
    max_distance_km: float = D_ISL_MAX_KM,
    earth_radius_km: float = R_EARTH_KM,
) -> dict[str, object]:
    """Select a one-to-one cyclic matching by the requested lexicographic rule."""

    left = np.asarray(left_positions_km, dtype=float)
    right = np.asarray(right_positions_km, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("Adjacent-plane positions must both have shape (N, 3).")
    N = left.shape[0]
    best: dict[str, object] | None = None
    best_key: tuple[float, ...] | None = None
    for shift in range(N):
        source, target = cyclic_shift_pairs(N, shift)
        feasible, distances, line_of_sight = link_feasibility(
            left[source], right[target], max_distance_km, earth_radius_km
        )
        key = (
            -float(np.count_nonzero(feasible)),
            float(np.max(distances)),
            float(np.sum(distances)),
            float(shift),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "shift": int(shift),
                "source_indices": source,
                "target_indices": target,
                "feasible": feasible,
                "distances_km": distances,
                "line_of_sight": line_of_sight,
                "feasible_count": int(np.count_nonzero(feasible)),
                "maximum_distance_km": float(np.max(distances)),
                "total_distance_km": float(np.sum(distances)),
            }
    if best is None:
        raise RuntimeError("Cyclic-shift matching did not evaluate any shift.")
    return best


def select_hungarian_crosslink_matching(
    left_positions_km: np.ndarray,
    right_positions_km: np.ndarray,
    max_distance_km: float = D_ISL_MAX_KM,
    earth_radius_km: float = R_EARTH_KM,
) -> dict[str, object]:
    left = np.asarray(left_positions_km, dtype=float)
    right = np.asarray(right_positions_km, dtype=float)
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    los = has_line_of_sight(left[:, None, :], right[None, :, :], earth_radius_km)
    feasible_matrix = (distances <= float(max_distance_km)) & los
    penalty = max(1.0e9, float(np.max(distances)) * left.shape[0] * 100.0)
    cost = np.where(feasible_matrix, distances, penalty + distances)
    source, target = linear_sum_assignment(cost)
    selected_distances = distances[source, target]
    selected_los = los[source, target]
    selected_feasible = feasible_matrix[source, target]
    return {
        "shift": -1,
        "source_indices": source.astype(int),
        "target_indices": target.astype(int),
        "feasible": selected_feasible,
        "distances_km": selected_distances,
        "line_of_sight": selected_los,
        "feasible_count": int(np.count_nonzero(selected_feasible)),
        "maximum_distance_km": float(np.max(selected_distances)),
        "total_distance_km": float(np.sum(selected_distances)),
    }


def build_adjacency(
    satellite_count: int, edges: pd.DataFrame
) -> list[list[tuple[int, float, float]]]:
    """Build sparse adjacency entries as (neighbor, distance_km, delay_ms)."""

    adjacency: list[list[tuple[int, float, float]]] = [
        [] for _ in range(int(satellite_count))
    ]
    for row in edges.itertuples(index=False):
        source = int(row.source_satellite)
        target = int(row.target_satellite)
        distance = float(row.distance_km)
        delay_ms = distance / LIGHT_SPEED_KM_S * 1000.0 + TAU_PROCESSING_MS
        adjacency[source].append((target, distance, delay_ms))
        adjacency[target].append((source, distance, delay_ms))
    return adjacency


def _connected_components(adjacency: list[list[tuple[int, float, float]]]) -> list[list[int]]:
    seen = np.zeros(len(adjacency), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor, _, _ in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def _bfs_farthest(
    adjacency: list[list[tuple[int, float, float]]], start: int, allowed: set[int]
) -> tuple[int, int]:
    distance = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor, _, _ in adjacency[node]:
            if neighbor in allowed and neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    farthest = max(distance, key=lambda node: (distance[node], -node))
    return farthest, int(distance[farthest])


def _dijkstra_farthest(
    adjacency: list[list[tuple[int, float, float]]], start: int, allowed: set[int]
) -> tuple[int, float]:
    distances = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node] + 1e-12:
            continue
        for neighbor, _, delay_ms in adjacency[node]:
            if neighbor not in allowed:
                continue
            candidate = distance + delay_ms
            if candidate + 1e-12 < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    farthest = max(distances, key=lambda node: (distances[node], -node))
    return farthest, float(distances[farthest])


def approximate_diameters(
    adjacency: list[list[tuple[int, float, float]]], components: list[list[int]]
) -> tuple[int, float]:
    """Use a deterministic double sweep on the largest component."""

    if not components:
        return 0, 0.0
    largest = max(components, key=len)
    allowed = set(largest)
    start = min(largest)
    endpoint, _ = _bfs_farthest(adjacency, start, allowed)
    _, unweighted = _bfs_farthest(adjacency, endpoint, allowed)
    weighted_endpoint, _ = _dijkstra_farthest(adjacency, start, allowed)
    _, weighted = _dijkstra_farthest(adjacency, weighted_endpoint, allowed)
    return unweighted, weighted


def build_topology_snapshot(
    constellation: WalkerConstellation,
    time_seconds: float,
    *,
    max_distance_km: float = D_ISL_MAX_KM,
    matching_method: str = "cyclic_shift",
) -> TopologySnapshot:
    positions = propagate_constellation(constellation, float(time_seconds))[0]
    shaped = positions.reshape(constellation.M, constellation.N, 3)
    edge_records: list[dict[str, object]] = []

    for plane in range(constellation.M):
        q = np.arange(constellation.N, dtype=int)
        next_q = (q + 1) % constellation.N
        feasible, distances, los = link_feasibility(
            shaped[plane, q], shaped[plane, next_q], max_distance_km
        )
        for source_q, target_q, distance, line_ok, is_on in zip(
            q, next_q, distances, los, feasible
        ):
            if not is_on:
                continue
            source = satellite_id(plane, int(source_q), constellation.N)
            target = satellite_id(plane, int(target_q), constellation.N)
            if source > target:
                source, target = target, source
            edge_records.append(
                {
                    "source_satellite": source,
                    "target_satellite": target,
                    "link_type": "same_plane",
                    "distance_km": float(distance),
                    "line_of_sight": bool(line_ok),
                    "link_on": True,
                    "plane_pair": f"{plane}-{plane}",
                }
            )

    shift_records: list[dict[str, object]] = []
    matching_function = (
        select_cyclic_crosslink_matching
        if matching_method == "cyclic_shift"
        else select_hungarian_crosslink_matching
    )
    if matching_method not in {"cyclic_shift", "hungarian"}:
        raise ValueError(f"未知跨轨匹配方法：{matching_method}")

    for left_plane in range(constellation.M):
        right_plane = (left_plane + 1) % constellation.M
        selected = matching_function(
            shaped[left_plane], shaped[right_plane], max_distance_km
        )
        source_indices = np.asarray(selected["source_indices"], dtype=int)
        target_indices = np.asarray(selected["target_indices"], dtype=int)
        feasible = np.asarray(selected["feasible"], dtype=bool)
        distances = np.asarray(selected["distances_km"], dtype=float)
        los = np.asarray(selected["line_of_sight"], dtype=bool)
        shift_records.append(
            {
                "time_seconds": float(time_seconds),
                "left_plane": left_plane,
                "right_plane": right_plane,
                "kappa": int(selected["shift"]),
                "feasible_count": int(selected["feasible_count"]),
                "maximum_distance_km": float(selected["maximum_distance_km"]),
                "total_distance_km": float(selected["total_distance_km"]),
                "matching_method": matching_method,
            }
        )
        for source_q, target_q, distance, line_ok, is_on in zip(
            source_indices, target_indices, distances, los, feasible
        ):
            if not is_on:
                continue
            source = satellite_id(left_plane, int(source_q), constellation.N)
            target = satellite_id(right_plane, int(target_q), constellation.N)
            if source > target:
                source, target = target, source
            edge_records.append(
                {
                    "source_satellite": source,
                    "target_satellite": target,
                    "link_type": "cross_plane",
                    "distance_km": float(distance),
                    "line_of_sight": bool(line_ok),
                    "link_on": True,
                    "plane_pair": f"{left_plane}-{right_plane}",
                }
            )

    edges = pd.DataFrame(edge_records).drop_duplicates(
        subset=["source_satellite", "target_satellite"], keep="first"
    )
    if edges.empty:
        edges = pd.DataFrame(
            columns=[
                "source_satellite",
                "target_satellite",
                "link_type",
                "distance_km",
                "line_of_sight",
                "link_on",
                "plane_pair",
            ]
        )
    degree = np.zeros(constellation.total_satellites, dtype=int)
    if not edges.empty:
        np.add.at(degree, edges["source_satellite"].to_numpy(dtype=int), 1)
        np.add.at(degree, edges["target_satellite"].to_numpy(dtype=int), 1)
    if int(degree.max(initial=0)) > 4:
        raise RuntimeError("星间拓扑违反每颗卫星最多 4 条链路的约束。")

    adjacency = build_adjacency(constellation.total_satellites, edges)
    components = _connected_components(adjacency)
    largest_size = max((len(component) for component in components), default=0)
    unweighted_diameter, weighted_diameter_ms = approximate_diameters(
        adjacency, components
    )
    shift_frame = pd.DataFrame(shift_records)
    same_count = int(np.count_nonzero(edges["link_type"] == "same_plane"))
    cross_count = int(np.count_nonzero(edges["link_type"] == "cross_plane"))
    metrics: dict[str, object] = {
        "time_seconds": float(time_seconds),
        "edge_count": int(len(edges)),
        "same_plane_edge_count": same_count,
        "cross_plane_edge_count": cross_count,
        "average_degree": float(np.mean(degree)),
        "minimum_degree": int(np.min(degree)),
        "maximum_degree": int(np.max(degree)),
        "connected_component_count": int(len(components)),
        "largest_component_ratio": float(largest_size / constellation.total_satellites),
        "is_connected": bool(len(components) == 1),
        "unweighted_diameter": int(unweighted_diameter),
        "weighted_diameter_ms": float(weighted_diameter_ms),
        "diameter_method": "largest_component_double_sweep_approximation",
        "selected_crosslink_shifts": json.dumps(
            shift_frame["kappa"].astype(int).tolist(), ensure_ascii=False
        ),
    }
    return TopologySnapshot(
        time_seconds=float(time_seconds),
        positions_eci_km=positions,
        edges=edges.reset_index(drop=True),
        crosslink_shifts=shift_frame,
        degree=degree,
        metrics=metrics,
    )


def _snapshot_worker(args: tuple[WalkerConstellation, float, float, str]) -> TopologySnapshot:
    constellation, time_seconds, max_distance_km, matching_method = args
    return build_topology_snapshot(
        constellation,
        time_seconds,
        max_distance_km=max_distance_km,
        matching_method=matching_method,
    )


def build_topology_series(
    constellation: WalkerConstellation,
    times_seconds: Iterable[float],
    *,
    max_distance_km: float = D_ISL_MAX_KM,
    matching_method: str = "cyclic_shift",
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[TopologySnapshot]:
    times = [float(value) for value in times_seconds]
    arguments = [
        (constellation, value, float(max_distance_km), matching_method) for value in times
    ]
    snapshots: list[TopologySnapshot] = []
    if int(workers) > 1 and len(arguments) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            for index, snapshot in enumerate(executor.map(_snapshot_worker, arguments), 1):
                snapshots.append(snapshot)
                if progress:
                    progress(index, len(arguments))
    else:
        for index, arguments_one in enumerate(arguments, 1):
            snapshots.append(_snapshot_worker(arguments_one))
            if progress:
                progress(index, len(arguments))
    return snapshots


def _longest_false_run(mask: np.ndarray, step_seconds: float) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or np.all(mask):
        return 0.0
    if not np.any(mask):
        return float(mask.size * step_seconds)
    false = ~mask
    padded = np.concatenate(([False], false, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return float(np.max(stops - starts) * step_seconds)


def summarize_topology_series(
    constellation: WalkerConstellation,
    snapshots: list[TopologySnapshot],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if not snapshots:
        raise ValueError("Topology series must contain at least one snapshot.")
    topology_timeseries = pd.DataFrame([snapshot.metrics for snapshot in snapshots])
    shift_timeseries = pd.concat(
        [snapshot.crosslink_shifts for snapshot in snapshots], ignore_index=True
    )
    degree_timeseries = pd.concat(
        [
            pd.DataFrame(
                {
                    "time_seconds": snapshot.time_seconds,
                    "satellite_id": np.arange(constellation.total_satellites),
                    "plane_index": constellation.plane_index,
                    "satellite_index": constellation.satellite_index,
                    "degree": snapshot.degree,
                }
            )
            for snapshot in snapshots
        ],
        ignore_index=True,
    )

    active_rows: list[pd.DataFrame] = []
    masks: dict[tuple[int, int, str], np.ndarray] = {}
    distances: dict[tuple[int, int, str], list[float]] = {}
    for time_index, snapshot in enumerate(snapshots):
        frame = snapshot.edges.copy()
        frame.insert(0, "time_seconds", snapshot.time_seconds)
        active_rows.append(frame)
        for row in snapshot.edges.itertuples(index=False):
            key = (
                int(row.source_satellite),
                int(row.target_satellite),
                str(row.link_type),
            )
            masks.setdefault(key, np.zeros(len(snapshots), dtype=bool))[time_index] = True
            distances.setdefault(key, []).append(float(row.distance_km))
    link_timeseries = pd.concat(active_rows, ignore_index=True)
    if len(snapshots) > 1:
        steps = np.diff([snapshot.time_seconds for snapshot in snapshots])
        step_seconds = float(np.median(steps))
    else:
        step_seconds = 0.0
    availability_rows = []
    for key, mask in masks.items():
        values = np.asarray(distances[key], dtype=float)
        transitions = np.diff(mask.astype(np.int8))
        on_events = int(mask[0]) + int(np.count_nonzero(transitions == 1))
        off_events = int(np.count_nonzero(transitions == -1))
        availability_rows.append(
            {
                "source_satellite": key[0],
                "target_satellite": key[1],
                "link_type": key[2],
                "minimum_distance_km": float(np.min(values)),
                "maximum_distance_km": float(np.max(values)),
                "mean_distance_km": float(np.mean(values)),
                "availability_ratio": float(np.mean(mask)),
                "number_of_on_events": on_events,
                "number_of_off_events": off_events,
                "number_of_switches": int(np.count_nonzero(transitions)),
                "longest_outage_seconds": _longest_false_run(mask, step_seconds),
            }
        )
    link_availability = pd.DataFrame(availability_rows)
    worst_index = int(topology_timeseries["largest_component_ratio"].idxmin())
    snapshot_edges = snapshots[worst_index].edges.copy()
    snapshot_edges.insert(0, "time_seconds", snapshots[worst_index].time_seconds)

    analytic_same = same_plane_neighbor_distance_km(constellation)
    same_distances = link_timeseries.loc[
        link_timeseries["link_type"] == "same_plane", "distance_km"
    ].to_numpy(dtype=float)
    summary: dict[str, object] = {
        "snapshot_count": len(snapshots),
        "duration_seconds": float(
            snapshots[-1].time_seconds - snapshots[0].time_seconds
        ),
        "same_plane_distance_analytic_km": analytic_same,
        "same_plane_distance_numerical_mean_km": float(np.mean(same_distances)),
        "same_plane_distance_max_abs_error_km": float(
            np.max(np.abs(same_distances - analytic_same))
        ),
        "network_connected_time_ratio": float(topology_timeseries["is_connected"].mean()),
        "mean_edge_count": float(topology_timeseries["edge_count"].mean()),
        "mean_degree": float(topology_timeseries["average_degree"].mean()),
        "maximum_degree": int(topology_timeseries["maximum_degree"].max()),
        "mean_largest_component_ratio": float(
            topology_timeseries["largest_component_ratio"].mean()
        ),
        "minimum_largest_component_ratio": float(
            topology_timeseries["largest_component_ratio"].min()
        ),
        "topology_switch_count": int(link_availability["number_of_switches"].sum()),
        "crosslink_shift_change_count": int(
            shift_timeseries.sort_values(["left_plane", "time_seconds"])
            .groupby("left_plane")["kappa"]
            .apply(lambda series: np.count_nonzero(np.diff(series.to_numpy())))
            .sum()
        ),
        "cross_plane_distance_min_km": float(
            link_timeseries.loc[
                link_timeseries["link_type"] == "cross_plane", "distance_km"
            ].min()
        ),
        "cross_plane_distance_max_km": float(
            link_timeseries.loc[
                link_timeseries["link_type"] == "cross_plane", "distance_km"
            ].max()
        ),
        "mean_link_availability_ratio": float(link_availability["availability_ratio"].mean()),
        "minimum_link_availability_ratio": float(link_availability["availability_ratio"].min()),
        "all_distances_within_limit": bool(
            link_timeseries["distance_km"].max() <= D_ISL_MAX_KM + 1e-9
        ),
        "all_links_have_line_of_sight": bool(link_timeseries["line_of_sight"].all()),
        "degree_constraint_satisfied": bool(
            topology_timeseries["maximum_degree"].max() <= 4
        ),
        "diameter_is_approximate": True,
    }
    return (
        summary,
        topology_timeseries,
        link_timeseries,
        link_availability,
        shift_timeseries,
        degree_timeseries,
        snapshot_edges,
    )
