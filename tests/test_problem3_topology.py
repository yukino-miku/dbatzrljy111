import math

import numpy as np

from starlink_modeling.problem2_orbit import (
    R_EARTH_KM,
    generate_walker_constellation,
    propagate_constellation,
)
from starlink_modeling.problem3_topology import (
    D_ISL_MAX_KM,
    build_topology_snapshot,
    cyclic_shift_pairs,
    has_line_of_sight,
    link_feasibility,
    same_plane_neighbor_distance_km,
    select_hungarian_crosslink_matching,
)


def test_eci_radius_is_constant():
    constellation = generate_walker_constellation(4, 8, 50.0, 1)
    positions = propagate_constellation(constellation, [0.0, 1234.0])
    radii = np.linalg.norm(positions, axis=-1)
    assert np.allclose(radii, R_EARTH_KM + 550.0, atol=1e-9)


def test_same_plane_neighbor_distance_matches_chord_formula():
    constellation = generate_walker_constellation(4, 8, 50.0, 1)
    positions = propagate_constellation(constellation, 321.0)[0].reshape(4, 8, 3)
    numerical = np.linalg.norm(positions[0] - np.roll(positions[0], -1, axis=0), axis=1)
    analytic = same_plane_neighbor_distance_km(constellation)
    assert math.isclose(analytic, 2 * (R_EARTH_KM + 550.0) * math.sin(math.pi / 8))
    assert np.allclose(numerical, analytic, atol=1e-9)


def test_cyclic_shift_is_one_to_one():
    source, target = cyclic_shift_pairs(11, 7)
    assert len(np.unique(source)) == 11
    assert len(np.unique(target)) == 11
    assert np.array_equal(target, (source + 7) % 11)


def test_snapshot_has_unique_edges_and_degree_at_most_four():
    constellation = generate_walker_constellation(6, 12, 50.0, 2)
    snapshot = build_topology_snapshot(constellation, 100.0)
    edge_pairs = snapshot.edges[["source_satellite", "target_satellite"]].to_numpy()
    assert len({tuple(pair) for pair in edge_pairs}) == len(edge_pairs)
    assert snapshot.degree.max() <= 4


def test_distance_limit_and_earth_occlusion_are_enforced():
    near_a = np.array([[R_EARTH_KM + 550.0, 0.0, 0.0]])
    near_b = np.array([[R_EARTH_KM + 550.0, 100.0, 0.0]])
    feasible, distance, _ = link_feasibility(near_a, near_b)
    assert feasible[0] and distance[0] < D_ISL_MAX_KM

    far_b = np.array([[-(R_EARTH_KM + 550.0), 0.0, 0.0]])
    feasible, distance, los = link_feasibility(near_a, far_b, max_distance_km=20000.0)
    assert distance[0] > D_ISL_MAX_KM
    assert not los[0]
    assert not feasible[0]
    assert not has_line_of_sight(near_a, far_b)[0]


def test_hungarian_matching_is_one_to_one():
    constellation = generate_walker_constellation(4, 7, 50.0, 1)
    positions = propagate_constellation(constellation, 0.0)[0].reshape(4, 7, 3)
    result = select_hungarian_crosslink_matching(positions[0], positions[1])
    assert len(np.unique(result["source_indices"])) == 7
    assert len(np.unique(result["target_indices"])) == 7


def test_snapshot_does_not_create_edges_beyond_configured_limit():
    constellation = generate_walker_constellation(4, 8, 50.0, 1)
    snapshot = build_topology_snapshot(constellation, 0.0, max_distance_km=1.0)
    assert snapshot.edges.empty
    assert snapshot.degree.max() == 0
