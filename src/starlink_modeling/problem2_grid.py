"""Ground-grid construction for the problem-two rectangular target region."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .problem2_orbit import R_EARTH_KM


TARGET_LAT_MIN_DEG = 4.0
TARGET_LAT_MAX_DEG = 53.0
TARGET_LON_MIN_DEG = 73.0
TARGET_LON_MAX_DEG = 135.0


@dataclass(frozen=True)
class GroundGrid:
    """Flattened ground points, normalized area weights, and unit vectors."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    weights: np.ndarray
    unit_vectors: np.ndarray
    grid_type: str
    resolution: float
    row_index: np.ndarray

    @property
    def size(self) -> int:
        return int(self.latitude_deg.size)


def _inclusive_axis(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    values = np.arange(start, stop + step * 0.25, step, dtype=float)
    values = values[values <= stop + 1e-12]
    if values.size == 0 or not math.isclose(values[-1], stop, abs_tol=1e-10):
        values = np.append(values, stop)
    values[0] = start
    values[-1] = stop
    return np.unique(np.round(values, 12))


def grid_to_unit_vectors(
    latitude_deg: np.ndarray, longitude_deg: np.ndarray
) -> np.ndarray:
    """Convert latitude/longitude arrays to Earth-centered unit vectors."""

    lat = np.radians(np.asarray(latitude_deg, dtype=float))
    lon = np.radians(np.asarray(longitude_deg, dtype=float))
    cos_lat = np.cos(lat)
    return np.column_stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat))
    )


def compute_area_weights(
    latitude_deg: np.ndarray, grid_type: str = "regular_latlon"
) -> np.ndarray:
    """Return normalized point weights for area-average statistics."""

    latitude_deg = np.asarray(latitude_deg, dtype=float)
    if grid_type == "regular_latlon":
        weights = np.cos(np.radians(latitude_deg))
    elif grid_type == "equal_arc":
        # Equal-arc longitude spacing grows as 1/cos(phi), so each point
        # represents approximately equal surface area.
        weights = np.ones_like(latitude_deg)
    else:
        raise ValueError(f"Unsupported grid_type: {grid_type}")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Grid weights must have a positive sum.")
    return weights / total


def make_regular_grid(
    step_deg: float,
    lat_bounds_deg: tuple[float, float] = (TARGET_LAT_MIN_DEG, TARGET_LAT_MAX_DEG),
    lon_bounds_deg: tuple[float, float] = (TARGET_LON_MIN_DEG, TARGET_LON_MAX_DEG),
) -> GroundGrid:
    """Build an inclusive equal-latitude/equal-longitude grid."""

    lat_axis = _inclusive_axis(*lat_bounds_deg, step_deg)
    lon_axis = _inclusive_axis(*lon_bounds_deg, step_deg)
    lon_mesh, lat_mesh = np.meshgrid(lon_axis, lat_axis)
    lat_flat = lat_mesh.ravel()
    lon_flat = lon_mesh.ravel()
    row_index = np.repeat(np.arange(lat_axis.size), lon_axis.size)
    return GroundGrid(
        latitude_deg=lat_flat,
        longitude_deg=lon_flat,
        weights=compute_area_weights(lat_flat, "regular_latlon"),
        unit_vectors=grid_to_unit_vectors(lat_flat, lon_flat),
        grid_type="regular_latlon",
        resolution=float(step_deg),
        row_index=row_index,
    )


def make_equal_arc_grid(
    spacing_km: float,
    lat_bounds_deg: tuple[float, float] = (TARGET_LAT_MIN_DEG, TARGET_LAT_MAX_DEG),
    lon_bounds_deg: tuple[float, float] = (TARGET_LON_MIN_DEG, TARGET_LON_MAX_DEG),
    earth_radius_km: float = R_EARTH_KM,
) -> GroundGrid:
    """Build an inclusive grid with approximately equal surface arc spacing."""

    if spacing_km <= 0:
        raise ValueError("spacing_km must be positive.")
    lat_step_deg = math.degrees(spacing_km / earth_radius_km)
    lat_axis = _inclusive_axis(*lat_bounds_deg, lat_step_deg)

    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    lon_span = lon_bounds_deg[1] - lon_bounds_deg[0]
    for row, latitude_deg in enumerate(lat_axis):
        cos_lat = max(math.cos(math.radians(latitude_deg)), 1e-8)
        lon_step_deg = math.degrees(spacing_km / (earth_radius_km * cos_lat))
        longitude_axis = _inclusive_axis(*lon_bounds_deg, lon_step_deg)
        if longitude_axis[-1] - longitude_axis[0] < lon_span - 1e-9:
            raise RuntimeError("Equal-arc longitude construction omitted a boundary.")
        lat_parts.append(np.full(longitude_axis.size, latitude_deg))
        lon_parts.append(longitude_axis)
        row_parts.append(np.full(longitude_axis.size, row, dtype=int))

    lat_flat = np.concatenate(lat_parts)
    lon_flat = np.concatenate(lon_parts)
    row_index = np.concatenate(row_parts)
    return GroundGrid(
        latitude_deg=lat_flat,
        longitude_deg=lon_flat,
        weights=compute_area_weights(lat_flat, "equal_arc"),
        unit_vectors=grid_to_unit_vectors(lat_flat, lon_flat),
        grid_type="equal_arc",
        resolution=float(spacing_km),
        row_index=row_index,
    )


def make_grid(config: dict) -> GroundGrid:
    """Construct a grid from a JSON-compatible configuration mapping."""

    grid_type = config.get("grid_type", "equal_arc")
    lat_bounds = tuple(config.get("latitude_bounds_deg", [4.0, 53.0]))
    lon_bounds = tuple(config.get("longitude_bounds_deg", [73.0, 135.0]))
    if grid_type == "equal_arc":
        return make_equal_arc_grid(
            float(config["equal_arc_spacing_km"]), lat_bounds, lon_bounds
        )
    if grid_type == "regular_latlon":
        return make_regular_grid(
            float(config["regular_grid_step_deg"]), lat_bounds, lon_bounds
        )
    raise ValueError(f"Unsupported grid_type: {grid_type}")

