import numpy as np

from starlink_modeling.problem2_orbit import (
    OMEGA_EARTH_RAD_S,
    eci_to_ecef,
    generate_walker_constellation,
    propagate_constellation,
    satellite_subpoints,
)


def test_walker_raan_spacing_is_uniform():
    constellation = generate_walker_constellation(6, 8, 53.0, 2, Omega0_deg=3.0)
    plane_raan = constellation.raan_rad.reshape(6, 8)[:, 0]
    np.testing.assert_allclose(np.diff(np.unwrap(plane_raan)), 2.0 * np.pi / 6)


def test_satellite_phase_spacing_and_walker_plane_offset_are_correct():
    constellation = generate_walker_constellation(5, 7, 52.0, 3, u0_deg=1.0)
    phase = constellation.initial_argument_of_latitude_rad.reshape(5, 7)
    np.testing.assert_allclose(np.diff(phase[0]), 2.0 * np.pi / 7)
    np.testing.assert_allclose(phase[1, 0] - phase[0, 0], 2.0 * np.pi * 3 / 35)


def test_propagated_eci_radius_equals_orbit_radius():
    constellation = generate_walker_constellation(4, 6, 50.0, 1)
    positions = propagate_constellation(constellation, np.array([0.0, 1234.0]))
    np.testing.assert_allclose(np.linalg.norm(positions, axis=-1), 6371.0 + 550.0)


def test_ecef_rotation_moves_longitude_westward_by_earth_rotation_angle():
    eci = np.array([[[7000.0, 0.0, 0.0]], [[7000.0, 0.0, 0.0]]])
    times = np.array([0.0, 1000.0])
    ecef = eci_to_ecef(eci, times)
    _, longitude = satellite_subpoints(ecef)
    np.testing.assert_allclose(longitude[1, 0], -OMEGA_EARTH_RAD_S * 1000.0)

