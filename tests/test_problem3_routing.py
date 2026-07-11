import json
import math

import numpy as np

from starlink_modeling.problem3_routing import (
    dijkstra_shortest_paths,
    enumerate_ground_pairs_chunked,
    minimum_delay_ground_route,
    sample_ground_pairs,
)
from starlink_modeling.problem2_grid import make_regular_grid


def _edge(adjacency, a, b, distance, delay):
    adjacency[a].append((b, distance, delay))
    adjacency[b].append((a, distance, delay))


def test_dijkstra_on_simple_sparse_graph():
    adjacency = [[] for _ in range(4)]
    _edge(adjacency, 0, 1, 1.0, 1.0)
    _edge(adjacency, 1, 3, 1.0, 1.0)
    _edge(adjacency, 0, 2, 1.0, 0.5)
    _edge(adjacency, 2, 3, 1.0, 3.0)
    distance, predecessor, _, _ = dijkstra_shortest_paths(adjacency, [(0, 0.0, 0)])
    assert math.isclose(distance[3], 2.0)
    assert predecessor[3] == 1


def test_virtual_source_selects_better_non_nearest_ingress_and_processing_count():
    radius = 6921.0
    angle = math.radians(10.0)
    satellites = np.array(
        [
            [radius, 0.0, 0.0],
            [radius * math.cos(angle), radius * math.sin(angle), 0.0],
            [0.0, radius, 0.0],
        ]
    )
    adjacency = [[] for _ in range(3)]
    _edge(adjacency, 0, 2, 5000.0, 20.0)
    _edge(adjacency, 1, 2, 1000.0, 1.0)
    source = np.array([1.0, 0.0, 0.0])
    destination = np.array([0.0, 1.0, 0.0])
    result = minimum_delay_ground_route(
        adjacency, satellites, source, destination, math.radians(15.0)
    )
    assert result["reachable"]
    assert result["ingress_satellite"] == 1
    assert json.loads(result["satellite_path"]) == [1, 2]
    assert result["satellite_count"] == 2
    assert math.isclose(result["processing_delay_ms"], 1.0)
    assert math.isclose(result["processing_delay_isl_only_ms"], 0.5)


def test_no_visible_satellite_is_unreachable():
    satellites = np.array([[0.0, 6921.0, 0.0]])
    result = minimum_delay_ground_route(
        [[]],
        satellites,
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        math.radians(1.0),
    )
    assert not result["reachable"]
    assert math.isnan(result["total_delay_ms"])


def test_sampled_pairs_are_reproducible():
    grid = make_regular_grid(10.0, (4.0, 24.0), (73.0, 103.0))
    first = sample_ground_pairs(grid, 12, 42)
    second = sample_ground_pairs(grid, 12, 42)
    assert np.array_equal(first, second)
    assert np.all(first[:, 0] != first[:, 1])


def test_exact_pair_generator_covers_upper_triangle_without_dense_matrix():
    chunks = list(enumerate_ground_pairs_chunked(7, block_size=2))
    pairs = np.concatenate(chunks)
    assert len(pairs) == 7 * 6 // 2
    assert len({tuple(pair) for pair in pairs}) == len(pairs)
    assert np.all(pairs[:, 0] < pairs[:, 1])
