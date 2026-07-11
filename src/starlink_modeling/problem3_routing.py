"""Ground access and minimum-delay routing for problem three."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .problem2_grid import GroundGrid
from .problem2_orbit import R_EARTH_KM
from .problem3_topology import (
    LIGHT_SPEED_KM_S,
    TAU_PROCESSING_MS,
    TopologySnapshot,
    build_adjacency,
)


DELAY_THRESHOLD_MS = 30.0


@dataclass(frozen=True)
class VisibleSatellites:
    satellite_ids: np.ndarray
    slant_distances_km: np.ndarray


def ground_positions_ecef_km(unit_vectors: np.ndarray) -> np.ndarray:
    return R_EARTH_KM * np.asarray(unit_vectors, dtype=float)


def visible_satellites(
    ground_unit_vector: np.ndarray,
    satellite_positions_ecef_km: np.ndarray,
    theta_rad: float,
) -> VisibleSatellites:
    ground = np.asarray(ground_unit_vector, dtype=float)
    satellites = np.asarray(satellite_positions_ecef_km, dtype=float)
    satellite_units = satellites / np.linalg.norm(satellites, axis=1, keepdims=True)
    mask = satellite_units @ ground >= math.cos(float(theta_rad))
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        return VisibleSatellites(ids, np.empty(0, dtype=float))
    ground_position = R_EARTH_KM * ground
    slant = np.linalg.norm(satellites[ids] - ground_position, axis=1)
    return VisibleSatellites(ids.astype(int), slant)


def visibility_matrix(
    ground_unit_vectors: np.ndarray,
    satellite_positions_ecef_km: np.ndarray,
    theta_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ground-by-satellite visibility and true slant distances."""

    ground = np.asarray(ground_unit_vectors, dtype=float)
    satellites = np.asarray(satellite_positions_ecef_km, dtype=float)
    satellite_units = satellites / np.linalg.norm(satellites, axis=1, keepdims=True)
    visibility = ground @ satellite_units.T >= math.cos(float(theta_rad))
    dot = ground @ satellites.T
    satellite_radius_sq = np.einsum("ij,ij->i", satellites, satellites)
    slant_sq = (
        R_EARTH_KM**2 + satellite_radius_sq[None, :] - 2.0 * R_EARTH_KM * dot
    )
    slant = np.sqrt(np.maximum(slant_sq, 0.0))
    return visibility, slant


def dijkstra_shortest_paths(
    adjacency: list[list[tuple[int, float, float]]],
    sources: Iterable[tuple[int, float, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run deterministic multi-source Dijkstra.

    Each source tuple is ``(node, initial_delay_ms, ingress_satellite)``. Equal
    delays prefer fewer satellite nodes and then the lower ingress ID.
    """

    node_count = len(adjacency)
    delay = np.full(node_count, np.inf, dtype=float)
    satellite_count = np.full(node_count, np.iinfo(np.int32).max, dtype=np.int32)
    predecessor = np.full(node_count, -1, dtype=np.int32)
    ingress = np.full(node_count, -1, dtype=np.int32)
    heap: list[tuple[float, int, int, int]] = []
    for node, initial_delay, ingress_id in sources:
        node = int(node)
        key = (float(initial_delay), 1, int(ingress_id))
        current = (delay[node], int(satellite_count[node]), int(ingress[node]))
        if key < current:
            delay[node] = key[0]
            satellite_count[node] = 1
            ingress[node] = key[2]
            predecessor[node] = -1
            heapq.heappush(heap, (key[0], 1, key[2], node))

    while heap:
        current_delay, count, ingress_id, node = heapq.heappop(heap)
        if (
            current_delay > delay[node] + 1e-12
            or count != satellite_count[node]
            or ingress_id != ingress[node]
        ):
            continue
        for neighbor, _, edge_delay_ms in adjacency[node]:
            candidate_delay = current_delay + float(edge_delay_ms)
            candidate_count = count + 1
            candidate_key = (candidate_delay, candidate_count, ingress_id)
            existing_key = (
                delay[neighbor],
                int(satellite_count[neighbor]),
                int(ingress[neighbor]),
            )
            if candidate_key < existing_key:
                delay[neighbor] = candidate_delay
                satellite_count[neighbor] = candidate_count
                predecessor[neighbor] = node
                ingress[neighbor] = ingress_id
                heapq.heappush(
                    heap,
                    (candidate_delay, candidate_count, ingress_id, int(neighbor)),
                )
    return delay, predecessor, satellite_count, ingress


def _reconstruct_path(predecessor: np.ndarray, endpoint: int) -> list[int]:
    path = [int(endpoint)]
    node = int(endpoint)
    seen = {node}
    while int(predecessor[node]) >= 0:
        node = int(predecessor[node])
        if node in seen:
            raise RuntimeError("Dijkstra predecessor chain contains a cycle.")
        seen.add(node)
        path.append(node)
    path.reverse()
    return path


def _edge_distance_lookup(
    adjacency: list[list[tuple[int, float, float]]], path: list[int]
) -> float:
    total = 0.0
    for source, target in zip(path[:-1], path[1:]):
        match = next(
            (distance for neighbor, distance, _ in adjacency[source] if neighbor == target),
            None,
        )
        if match is None:
            raise RuntimeError(f"Route contains missing edge {source}-{target}.")
        total += float(match)
    return total


def great_circle_distance_km(a: np.ndarray, b: np.ndarray) -> float:
    central_angle = math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))
    return R_EARTH_KM * central_angle


def minimum_delay_ground_route(
    adjacency: list[list[tuple[int, float, float]]],
    satellite_positions_ecef_km: np.ndarray,
    source_unit_vector: np.ndarray,
    destination_unit_vector: np.ndarray,
    theta_rad: float,
    *,
    time_seconds: float = 0.0,
    source_lat: float = math.nan,
    source_lon: float = math.nan,
    destination_lat: float = math.nan,
    destination_lon: float = math.nan,
) -> dict[str, object]:
    source_visible = visible_satellites(
        source_unit_vector, satellite_positions_ecef_km, theta_rad
    )
    destination_visible = visible_satellites(
        destination_unit_vector, satellite_positions_ecef_km, theta_rad
    )
    base = {
        "source_lat": float(source_lat),
        "source_lon": float(source_lon),
        "destination_lat": float(destination_lat),
        "destination_lon": float(destination_lon),
        "time_seconds": float(time_seconds),
        "ground_distance_km": great_circle_distance_km(
            np.asarray(source_unit_vector), np.asarray(destination_unit_vector)
        ),
    }
    if source_visible.satellite_ids.size == 0 or destination_visible.satellite_ids.size == 0:
        return {
            **base,
            "ingress_satellite": -1,
            "egress_satellite": -1,
            "satellite_path": "[]",
            "ISL_hop_count": 0,
            "satellite_count": 0,
            "uplink_distance_km": math.nan,
            "ISL_total_distance_km": math.nan,
            "downlink_distance_km": math.nan,
            "propagation_delay_ms": math.nan,
            "processing_delay_ms": math.nan,
            "processing_delay_isl_only_ms": math.nan,
            "total_delay_ms": math.nan,
            "total_delay_isl_processing_ms": math.nan,
            "reachable": False,
        }

    source_delay = source_visible.slant_distances_km / LIGHT_SPEED_KM_S * 1000.0
    sources = [
        (int(satellite), float(delay + TAU_PROCESSING_MS), int(satellite))
        for satellite, delay in zip(source_visible.satellite_ids, source_delay)
    ]
    delays, predecessor, satellite_counts, ingress_ids = dijkstra_shortest_paths(
        adjacency, sources
    )
    destination_delay = (
        destination_visible.slant_distances_km / LIGHT_SPEED_KM_S * 1000.0
    )
    candidates: list[tuple[float, int, int]] = []
    for satellite, down_delay in zip(
        destination_visible.satellite_ids, destination_delay
    ):
        if math.isfinite(delays[satellite]):
            candidates.append(
                (
                    float(delays[satellite] + down_delay),
                    int(satellite_counts[satellite]),
                    int(satellite),
                )
            )
    if not candidates:
        return {
            **base,
            "ingress_satellite": -1,
            "egress_satellite": -1,
            "satellite_path": "[]",
            "ISL_hop_count": 0,
            "satellite_count": 0,
            "uplink_distance_km": math.nan,
            "ISL_total_distance_km": math.nan,
            "downlink_distance_km": math.nan,
            "propagation_delay_ms": math.nan,
            "processing_delay_ms": math.nan,
            "processing_delay_isl_only_ms": math.nan,
            "total_delay_ms": math.nan,
            "total_delay_isl_processing_ms": math.nan,
            "reachable": False,
        }

    total_delay, _, egress = min(candidates)
    path = _reconstruct_path(predecessor, egress)
    ingress_satellite = int(ingress_ids[egress])
    if path[0] != ingress_satellite:
        raise RuntimeError("Reconstructed route does not start at the selected ingress.")
    source_position = R_EARTH_KM * np.asarray(source_unit_vector)
    destination_position = R_EARTH_KM * np.asarray(destination_unit_vector)
    uplink = float(
        np.linalg.norm(satellite_positions_ecef_km[ingress_satellite] - source_position)
    )
    downlink = float(
        np.linalg.norm(satellite_positions_ecef_km[egress] - destination_position)
    )
    isl_distance = _edge_distance_lookup(adjacency, path)
    satellite_count = len(path)
    propagation_delay = (uplink + isl_distance + downlink) / LIGHT_SPEED_KM_S * 1000.0
    processing_delay = satellite_count * TAU_PROCESSING_MS
    isl_processing_delay = max(0, satellite_count - 1) * TAU_PROCESSING_MS
    return {
        **base,
        "ingress_satellite": ingress_satellite,
        "egress_satellite": int(egress),
        "satellite_path": json.dumps(path),
        "ISL_hop_count": max(0, satellite_count - 1),
        "satellite_count": satellite_count,
        "uplink_distance_km": uplink,
        "ISL_total_distance_km": isl_distance,
        "downlink_distance_km": downlink,
        "propagation_delay_ms": propagation_delay,
        "processing_delay_ms": processing_delay,
        "processing_delay_isl_only_ms": isl_processing_delay,
        "total_delay_ms": float(total_delay),
        "total_delay_isl_processing_ms": propagation_delay + isl_processing_delay,
        "reachable": True,
    }


def _nearest_grid_index(grid: GroundGrid, latitude: float, longitude: float) -> int:
    distance = (grid.latitude_deg - latitude) ** 2 + (
        (grid.longitude_deg - longitude) * math.cos(math.radians(latitude))
    ) ** 2
    return int(np.argmin(distance))


def sample_ground_pairs(
    grid: GroundGrid, pair_count: int, seed: int
) -> np.ndarray:
    """Return deterministic area-weighted pairs with mandatory boundary cases."""

    pair_count = int(pair_count)
    if pair_count < 1:
        return np.empty((0, 2), dtype=int)
    lat_min, lat_max = float(grid.latitude_deg.min()), float(grid.latitude_deg.max())
    lon_min, lon_max = float(grid.longitude_deg.min()), float(grid.longitude_deg.max())
    center_lat, center_lon = (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0
    special_points = [
        _nearest_grid_index(grid, lat_min, lon_min),
        _nearest_grid_index(grid, lat_min, lon_max),
        _nearest_grid_index(grid, lat_max, lon_min),
        _nearest_grid_index(grid, lat_max, lon_max),
        _nearest_grid_index(grid, center_lat, center_lon),
        _nearest_grid_index(grid, lat_min, center_lon),
        _nearest_grid_index(grid, lat_max, center_lon),
        _nearest_grid_index(grid, center_lat, lon_min),
        _nearest_grid_index(grid, center_lat, lon_max),
    ]
    special_pairs = [
        (special_points[0], special_points[3]),
        (special_points[1], special_points[2]),
        (special_points[0], special_points[1]),
        (special_points[2], special_points[3]),
        (special_points[4], special_points[0]),
        (special_points[4], special_points[3]),
        (special_points[5], special_points[6]),
        (special_points[7], special_points[8]),
    ]
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for source, destination in special_pairs:
        key = (min(source, destination), max(source, destination))
        if source != destination and key not in seen:
            pairs.append((source, destination))
            seen.add(key)
        if len(pairs) >= pair_count:
            return np.asarray(pairs, dtype=int)

    rng = np.random.default_rng(int(seed))
    attempts = 0
    maximum_attempts = max(1000, pair_count * 50)
    while len(pairs) < pair_count and attempts < maximum_attempts:
        source, destination = rng.choice(
            grid.size, size=2, replace=False, p=grid.weights
        )
        key = (min(int(source), int(destination)), max(int(source), int(destination)))
        if key not in seen:
            seen.add(key)
            pairs.append((int(source), int(destination)))
        attempts += 1
    if len(pairs) < pair_count:
        for source in range(grid.size):
            for destination in range(source + 1, grid.size):
                key = (source, destination)
                if key not in seen:
                    pairs.append(key)
                    seen.add(key)
                if len(pairs) >= pair_count:
                    break
            if len(pairs) >= pair_count:
                break
    return np.asarray(pairs, dtype=int)


def enumerate_ground_pairs_chunked(
    point_count: int, block_size: int = 256
) -> Iterable[np.ndarray]:
    """Yield upper-triangle ground pairs without a dense point-by-point cube."""

    for start in range(0, int(point_count), int(block_size)):
        stop = min(start + int(block_size), int(point_count))
        pairs = [
            (source, destination)
            for source in range(start, stop)
            for destination in range(source + 1, int(point_count))
        ]
        if pairs:
            yield np.asarray(pairs, dtype=int)


def evaluate_routes(
    grid: GroundGrid,
    snapshots: list[TopologySnapshot],
    satellite_positions_ecef: list[np.ndarray],
    theta_rad: float,
    pair_count: int,
    seed: int,
    pair_mode: str = "sampled",
    exact_block_size: int = 256,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    if len(snapshots) != len(satellite_positions_ecef):
        raise ValueError("Snapshot and ECEF position counts must match.")
    if not snapshots:
        raise ValueError("At least one routing snapshot is required.")
    if pair_mode == "sampled":
        sampled_pairs = sample_ground_pairs(grid, pair_count, seed)
    elif pair_mode == "exact":
        sampled_pairs = None
    else:
        raise ValueError(f"未知地面点对模式：{pair_mode}")

    records: list[dict[str, object]] = []
    access_rows: list[dict[str, object]] = []
    snapshot_count = len(snapshots)
    contexts: list[tuple[list[list[tuple[int, float, float]]], np.ndarray]] = []
    for snapshot_index, (snapshot, positions_ecef) in enumerate(
        zip(snapshots, satellite_positions_ecef)
    ):
        adjacency = build_adjacency(len(positions_ecef), snapshot.edges)
        visibility, slant = visibility_matrix(
            grid.unit_vectors, positions_ecef, theta_rad
        )
        contexts.append((adjacency, positions_ecef))
        access_rows.extend(
            {
                "time_seconds": snapshot.time_seconds,
                "ground_index": ground_index,
                "latitude_deg": float(grid.latitude_deg[ground_index]),
                "longitude_deg": float(grid.longitude_deg[ground_index]),
                "visible_satellite_count": int(np.count_nonzero(visibility[ground_index])),
                "minimum_slant_distance_km": float(
                    np.min(slant[ground_index, visibility[ground_index]])
                )
                if np.any(visibility[ground_index])
                else math.nan,
            }
            for ground_index in range(grid.size)
        )

    def evaluate_pair(pair_index: int, source_index: int, destination_index: int) -> None:
        snapshot_index = int(pair_index % snapshot_count)
        snapshot = snapshots[snapshot_index]
        adjacency, positions_ecef = contexts[snapshot_index]
        records.append(
            minimum_delay_ground_route(
                adjacency,
                positions_ecef,
                grid.unit_vectors[source_index],
                grid.unit_vectors[destination_index],
                theta_rad,
                time_seconds=snapshot.time_seconds,
                source_lat=float(grid.latitude_deg[source_index]),
                source_lon=float(grid.longitude_deg[source_index]),
                destination_lat=float(grid.latitude_deg[destination_index]),
                destination_lon=float(grid.longitude_deg[destination_index]),
            )
            | {
                "source_ground_index": int(source_index),
                "destination_ground_index": int(destination_index),
                "pair_index": int(pair_index),
            }
        )

    if sampled_pairs is not None:
        for pair_index, (source_index, destination_index) in enumerate(sampled_pairs):
            evaluate_pair(pair_index, int(source_index), int(destination_index))
    else:
        pair_offset = 0
        for chunk in enumerate_ground_pairs_chunked(grid.size, exact_block_size):
            for local_index, (source_index, destination_index) in enumerate(chunk):
                evaluate_pair(
                    pair_offset + local_index,
                    int(source_index),
                    int(destination_index),
                )
            pair_offset += len(chunk)

    routes = pd.DataFrame(records)
    reachable = routes[routes["reachable"]].copy()
    if reachable.empty:
        summary = {
            "route_request_count": int(len(routes)),
            "reachable_ratio": 0.0,
            "mean_delay_ms": math.nan,
            "median_delay_ms": math.nan,
            "p90_delay_ms": math.nan,
            "p95_delay_ms": math.nan,
            "p99_delay_ms": math.nan,
            "max_delay_ms": math.nan,
            "minimum_delay_ms": math.nan,
            "mean_hops": math.nan,
            "max_hops": math.nan,
            "ratio_below_30ms": 0.0,
        }
        worst: dict[str, object] = {"reachable": False}
        delay_distribution = pd.DataFrame(columns=["bin_left_ms", "bin_right_ms", "count"])
    else:
        delays = reachable["total_delay_ms"].to_numpy(dtype=float)
        summary = {
            "route_request_count": int(len(routes)),
            "reachable_request_count": int(len(reachable)),
            "reachable_ratio": float(len(reachable) / len(routes)),
            "mean_delay_ms": float(np.mean(delays)),
            "median_delay_ms": float(np.median(delays)),
            "p90_delay_ms": float(np.quantile(delays, 0.90)),
            "p95_delay_ms": float(np.quantile(delays, 0.95)),
            "p99_delay_ms": float(np.quantile(delays, 0.99)),
            "max_delay_ms": float(np.max(delays)),
            "minimum_delay_ms": float(np.min(delays)),
            "mean_hops": float(reachable["ISL_hop_count"].mean()),
            "max_hops": int(reachable["ISL_hop_count"].max()),
            "ratio_below_30ms": float(np.mean(delays <= DELAY_THRESHOLD_MS)),
            "mean_below_30ms": bool(np.mean(delays) <= DELAY_THRESHOLD_MS),
            "p95_below_30ms": bool(np.quantile(delays, 0.95) <= DELAY_THRESHOLD_MS),
            "maximum_below_30ms": bool(np.max(delays) <= DELAY_THRESHOLD_MS),
            "processing_definition_sensitivity_mean_ms": float(
                reachable["total_delay_ms"].mean()
                - reachable["total_delay_isl_processing_ms"].mean()
            ),
            "pair_mode": pair_mode,
        }
        worst = reachable.loc[reachable["total_delay_ms"].idxmax()].to_dict()
        bin_count = min(30, max(5, int(math.sqrt(len(delays)))))
        counts, edges = np.histogram(delays, bins=bin_count)
        delay_distribution = pd.DataFrame(
            {
                "bin_left_ms": edges[:-1],
                "bin_right_ms": edges[1:],
                "count": counts,
            }
        )

    time_metrics = (
        reachable.groupby("time_seconds", as_index=False)
        .agg(
            mean_delay_ms=("total_delay_ms", "mean"),
            p95_delay_ms=("total_delay_ms", lambda values: values.quantile(0.95)),
            max_delay_ms=("total_delay_ms", "max"),
            mean_hops=("ISL_hop_count", "mean"),
            max_hops=("ISL_hop_count", "max"),
            reachable_count=("reachable", "size"),
        )
        if not reachable.empty
        else pd.DataFrame()
    )
    access = pd.DataFrame(access_rows)
    access_summary = (
        access.groupby(
            ["ground_index", "latitude_deg", "longitude_deg"], as_index=False
        )
        .agg(
            minimum_visible_satellites=("visible_satellite_count", "min"),
            mean_visible_satellites=("visible_satellite_count", "mean"),
            maximum_visible_satellites=("visible_satellite_count", "max"),
            mean_minimum_slant_distance_km=("minimum_slant_distance_km", "mean"),
        )
    )
    return summary, time_metrics, routes, worst, delay_distribution, access_summary
