import math

import numpy as np

from starlink_modeling.problem2_coverage import (
    coverage_counts_direct,
    coverage_counts_kdtree,
    max_false_gap_samples,
)
from starlink_modeling.problem2_grid import (
    grid_to_unit_vectors,
    make_equal_arc_grid,
    make_regular_grid,
)


def test_regular_and_equal_arc_grids_include_all_four_corners_and_normalize_weights():
    for grid in (make_regular_grid(7.0), make_equal_arc_grid(500.0)):
        points = set(zip(grid.latitude_deg, grid.longitude_deg))
        for corner in ((4.0, 73.0), (4.0, 135.0), (53.0, 73.0), (53.0, 135.0)):
            assert corner in points
        assert np.isclose(grid.weights.sum(), 1.0)


def test_coverage_includes_subpoint_and_excludes_point_beyond_theta():
    theta = math.radians(5.0)
    satellite = grid_to_unit_vectors(np.array([20.0]), np.array([100.0]))
    grid = grid_to_unit_vectors(np.array([20.0, 20.0]), np.array([100.0, 106.0]))
    counts = coverage_counts_direct(grid, satellite, theta)
    np.testing.assert_array_equal(counts, [1, 0])


def test_kdtree_and_direct_coverage_counts_match_and_single_satellite_is_binary():
    scipy = __import__("scipy.spatial", fromlist=["cKDTree"])
    grid = make_regular_grid(5.0)
    satellite = grid_to_unit_vectors(
        np.array([10.0, 35.0]), np.array([90.0, 120.0])
    )
    theta = math.radians(4.55)
    direct = coverage_counts_direct(grid.unit_vectors, satellite, theta)
    tree = scipy.cKDTree(grid.unit_vectors)
    kd = coverage_counts_kdtree(tree, grid.size, satellite, theta)
    np.testing.assert_array_equal(kd, direct)

    one_satellite = coverage_counts_direct(grid.unit_vectors, satellite[:1], theta)
    assert np.max(one_satellite) <= 1


def test_max_false_gap_handles_full_empty_and_periodic_endpoint_join():
    assert max_false_gap_samples(np.ones(5, dtype=bool), periodic=True) == 0
    assert max_false_gap_samples(np.zeros(5, dtype=bool), periodic=True) == 5
    flags = np.array([False, False, True, False, True, False])
    assert max_false_gap_samples(flags, periodic=False) == 2
    assert max_false_gap_samples(flags, periodic=True) == 3

