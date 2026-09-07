from __future__ import annotations

import warnings

import numpy as np


def resolve_camera_model(
    intrinsics: np.ndarray | None,
    image_size: tuple[int, int] | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a usable pinhole model, falling back to a neutral 1280×720 camera."""
    fallback_size = (1280, 720)
    fallback_matrix = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if intrinsics is None:
        return fallback_matrix, fallback_size
    try:
        matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return fallback_matrix, fallback_size
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if not np.isfinite(matrix).all() or fx <= 0.0 or fy <= 0.0:
        return fallback_matrix, fallback_size
    if image_size is not None:
        width, height = int(image_size[0]), int(image_size[1])
    else:
        width = max(1, int(round(cx * 2.0)))
        height = max(1, int(round(cy * 2.0)))
    if width < 1 or height < 1:
        return fallback_matrix, fallback_size
    return matrix, (width, height)


def project_camera_points(
    points: np.ndarray,
    intrinsics: np.ndarray,
    cam_offset: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Project left-camera coordinates using z = Z + cam_offset."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    projected_z = points[:, 2] + float(cam_offset)
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = matrix[0, 0] * points[:, 0] / projected_z + matrix[0, 2]
    uv[:, 1] = matrix[1, 1] * points[:, 1] / projected_z + matrix[1, 2]
    return uv, projected_z


def _grid_representatives(
    points: np.ndarray,
    uv: np.ndarray,
    width: int,
    height: int,
    grid: int,
    tau_rel: float,
    occlusion: bool,
) -> np.ndarray:
    """Select projected grid samples and optionally remove unsupported depth jumps."""
    inside = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    inside_indices = np.flatnonzero(inside)
    if not len(inside_indices):
        return inside_indices

    projected = uv[inside_indices]
    columns = np.floor(projected[:, 0] / grid).astype(np.int64)
    rows = np.floor(projected[:, 1] / grid).astype(np.int64)
    grid_columns = max(1, (width + grid - 1) // grid)
    grid_rows = max(1, (height + grid - 1) // grid)
    keys = rows * grid_columns + columns
    depths = points[inside_indices, 2]
    cell_count = grid_rows * grid_columns
    source_positions = np.arange(len(keys), dtype=np.int64)

    if occlusion:
        # Reduce directly into the small projected grid. This avoids sorting
        # every source vertex (often more than a million) just to keep tens of
        # thousands of visible cells.
        nearest_depth = np.full(cell_count, np.inf, dtype=np.float32)
        np.minimum.at(nearest_depth, keys, depths)
        nearest_mask = depths == nearest_depth[keys]
        first_nearest = np.full(cell_count, len(keys), dtype=np.int64)
        np.minimum.at(
            first_nearest,
            keys[nearest_mask],
            source_positions[nearest_mask],
        )
        chosen_positions = first_nearest[first_nearest < len(keys)]
    else:
        # Grid sampling still controls density; disabling occlusion preserves
        # the source cloud's first sample instead of forcing the nearest one.
        first_sample = np.full(cell_count, len(keys), dtype=np.int64)
        np.minimum.at(first_sample, keys, source_positions)
        chosen_positions = first_sample[first_sample < len(keys)]

    selected = inside_indices[chosen_positions]
    selected_rows = rows[chosen_positions]
    selected_columns = columns[chosen_positions]
    selected_depths = points[selected, 2]

    if tau_rel > 0.0 and len(selected) >= 3:
        depth_map = np.full((grid_rows, grid_columns), np.nan, dtype=np.float32)
        depth_map[selected_rows, selected_columns] = selected_depths
        padded = np.pad(depth_map, 1, constant_values=np.nan)
        neighbors = np.stack(
            [
                padded[0:grid_rows, 0:grid_columns],
                padded[0:grid_rows, 1 : grid_columns + 1],
                padded[0:grid_rows, 2 : grid_columns + 2],
                padded[1 : grid_rows + 1, 0:grid_columns],
                padded[1 : grid_rows + 1, 2 : grid_columns + 2],
                padded[2 : grid_rows + 2, 0:grid_columns],
                padded[2 : grid_rows + 2, 1 : grid_columns + 1],
                padded[2 : grid_rows + 2, 2 : grid_columns + 2],
            ],
            axis=0,
        )
        neighbor_count = np.count_nonzero(np.isfinite(neighbors), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            neighbor_median = np.nanmedian(neighbors, axis=0)
        local_median = neighbor_median[selected_rows, selected_columns]
        local_count = neighbor_count[selected_rows, selected_columns]
        relative_jump = np.abs(selected_depths - local_median) / np.maximum(
            np.minimum(selected_depths, local_median),
            1e-6,
        )
        # tau_rel is exposed as filter strength: increasing it narrows the
        # accepted local depth jump, so filtering becomes more strict.
        allowed_jump = max(0.01, 0.30 * (1.0 - tau_rel))
        supported = (local_count < 2) | ~np.isfinite(relative_jump) | (
            relative_jump <= allowed_jump
        )
        selected = selected[supported]

    if occlusion and len(selected):
        # Opaque point sprites then paint from far to near while the depth
        # buffer independently guarantees that near points cover far points.
        selected = selected[np.argsort(points[selected, 2])[::-1]]
    return selected


