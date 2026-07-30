from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtGui import QImage


@dataclass(frozen=True)
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray
    original_count: int
    filtered_count: int
    max_distance: float | None


@dataclass(frozen=True)
class ImageData:
    image: QImage
    values: np.ndarray
    image_stats: ImageStats | None = None
    depth_stats: DepthStats | None = None


@dataclass(frozen=True)
class ImageStats:
    minimum: float | None
    maximum: float | None
    mean: float | None
    standard_deviation: float | None
    valid_ratio: float
    pixel_count: int
    channel_histograms: tuple[np.ndarray, ...]
    histogram_range: tuple[float, float]


@dataclass(frozen=True)
class DepthStats:
    width: int
    height: int
    minimum: float | None
    maximum: float | None
    invalid_ratio: float
    zero_ratio: float
    hole_ratio: float
    extreme_threshold: float | None
    extreme_ratio: float


def _normalize_numeric(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    valid = values[finite]
    low, high = np.percentile(valid, (1.0, 99.0)) if valid.size > 32 else (valid.min(), valid.max())
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) * (255.0 / (high - low)), 0, 255)
    scaled[~finite] = 0
    return scaled.astype(np.uint8)


def load_image_data(path: Path) -> ImageData:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in {"1", "P", "LA", "CMYK", "YCbCr", "LAB", "HSV"}:
            image = image.convert("RGB")
        array = np.asarray(image).copy()

    if array.ndim == 2:
        gray = array.astype(np.uint8, copy=False) if array.dtype == np.uint8 else _normalize_numeric(array)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        if array.shape[2] > 3:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = _normalize_numeric(array)
        rgb = np.ascontiguousarray(array)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)

    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, width * channels, QImage.Format.Format_RGB888).copy()
    return ImageData(qimage, array)


def load_image(path: Path) -> QImage:
    return load_image_data(path).image


def analyze_image(values: np.ndarray, bins: int = 128) -> ImageStats:
    """Return compact viewer statistics and per-channel histograms."""
    array = np.asarray(values)
    numeric = array.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    total = max(1, numeric.size)
    valid = numeric[finite]
    if not valid.size:
        return ImageStats(
            None,
            None,
            None,
            None,
            0.0,
            int(array.shape[0] * array.shape[1]) if array.ndim >= 2 else int(array.size),
            (np.zeros(bins, dtype=np.int64),),
            (0.0, 1.0),
        )
    low = float(valid.min())
    high = float(valid.max())
    histogram_high = high if high > low else low + 1.0
    if array.ndim == 3 and array.shape[2] >= 3:
        channels = tuple(
            np.histogram(
                numeric[..., index][np.isfinite(numeric[..., index])],
                bins=bins,
                range=(low, histogram_high),
            )[0]
            for index in range(3)
        )
    else:
        channel = numeric[..., 0] if array.ndim == 3 else numeric
        channel_valid = channel[np.isfinite(channel)]
        channels = (
            np.histogram(channel_valid, bins=bins, range=(low, histogram_high))[0],
        )
    return ImageStats(
        minimum=low,
        maximum=high,
        mean=float(valid.mean()),
        standard_deviation=float(valid.std()),
        valid_ratio=float(np.count_nonzero(finite) / total),
        pixel_count=int(array.shape[0] * array.shape[1]) if array.ndim >= 2 else int(array.size),
        channel_histograms=channels,
        histogram_range=(low, high),
    )


_COLOR_MAPS: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    "gray": (
        (0.0, (0, 0, 0)),
        (1.0, (255, 255, 255)),
    ),
    "viridis": (
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ),
    "turbo": (
        (0.0, (48, 18, 59)),
        (0.2, (50, 101, 212)),
        (0.4, (32, 192, 147)),
        (0.6, (189, 224, 52)),
        (0.8, (249, 128, 18)),
        (1.0, (122, 4, 3)),
    ),
}


def colorize_scalar(values: np.ndarray, color_map: str = "turbo") -> np.ndarray:
    """Map normalized scalar values to an RGB lookup table."""
    normalized = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    anchors = _COLOR_MAPS.get(color_map, _COLOR_MAPS["turbo"])
    positions = np.asarray([position for position, _ in anchors], dtype=np.float32)
    colors = np.asarray([color for _, color in anchors], dtype=np.float32)
    flat = normalized.reshape(-1)
    output = np.empty((flat.size, 3), dtype=np.float32)
    for channel in range(3):
        output[:, channel] = np.interp(flat, positions, colors[:, channel])
    return np.clip(output.reshape(normalized.shape + (3,)), 0, 255).astype(np.uint8)


def render_image_values(
    values: np.ndarray,
    *,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    exposure: float = 0.0,
    color_map: str = "gray",
    display_range: tuple[float, float] | None = None,
    auto_percentiles: tuple[float, float] = (1.0, 99.0),
    highlight_invalid: bool = False,
) -> QImage:
    """Render source values without modifying the loaded image data."""
    array = np.asarray(values)
    scalar = array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 1)
    if scalar:
        numeric = (
            array[..., 0] if array.ndim == 3 else array
        ).astype(np.float32, copy=False)
        finite = np.isfinite(numeric)
        normal_valid = finite & (numeric > 0) & (numeric != 65535)
        range_values = numeric[normal_valid]
        if display_range is not None:
            low, high = map(float, display_range)
        elif range_values.size:
            low, high = np.percentile(range_values, auto_percentiles)
        else:
            low, high = 0.0, 1.0
        if not np.isfinite(low):
            low = 0.0
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        normalized = np.clip((numeric - low) / (high - low), 0.0, 1.0)
        normalized[~finite] = 0.0
        normalized = (normalized - 0.5) * max(0.01, float(contrast)) + 0.5
        normalized *= 2.0 ** float(exposure)
        normalized += float(brightness)
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = np.power(normalized, 1.0 / max(0.05, float(gamma)))
        rgb = colorize_scalar(normalized, color_map)
        if highlight_invalid:
            rgb[~finite] = (255, 208, 64)
            rgb[finite & (numeric == 0)] = (0, 210, 255)
            rgb[finite & (numeric == 65535)] = (255, 70, 190)
            rgb[
                finite
                & ((numeric < low) | (numeric > high))
                & (numeric != 0)
                & (numeric != 65535)
            ] = (
                255,
                110,
                40,
            )
    else:
        rgb_source = array[..., :3].astype(np.float32, copy=False)
        maximum = 255.0
        if array.dtype.kind in {"u", "i"}:
            maximum = float(np.iinfo(array.dtype).max)
        elif np.isfinite(rgb_source).any() and float(np.nanmax(rgb_source)) <= 1.0:
            maximum = 1.0
        rgb = np.nan_to_num(rgb_source / max(maximum, 1.0), nan=0.0)
        rgb = (rgb - 0.5) * max(0.01, float(contrast)) + 0.5
        rgb *= 2.0 ** float(exposure)
        rgb += float(brightness)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = np.power(rgb, 1.0 / max(0.05, float(gamma)))
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    return QImage(
        rgb.data,
        width,
        height,
        width * channels,
        QImage.Format.Format_RGB888,
    ).copy()


def analyze_depth(values: np.ndarray) -> DepthStats:
    """Return robust quality indicators without assuming a specific depth unit."""
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[..., 0]
    numeric = array.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    zero = finite & (numeric == 0)
    holes = ~finite | (numeric <= 0)
    positive = numeric[finite & (numeric > 0)]
    total = max(1, numeric.size)
    if positive.size:
        minimum = float(positive.min())
        maximum = float(positive.max())
        extreme_threshold = float(np.percentile(positive, 99.5))
        # Saturated uint16 depth (65535) must remain visible even when the
        # percentile threshold itself equals 65535. Other constant, normal
        # depth images should not be reported as 100% extreme.
        extreme = (positive > extreme_threshold) | (positive == 65535.0)
        extreme_ratio = float(np.count_nonzero(extreme) / total)
    else:
        minimum = maximum = extreme_threshold = None
        extreme_ratio = 0.0
    height, width = numeric.shape[:2]
    return DepthStats(
        width=width,
        height=height,
        minimum=minimum,
        maximum=maximum,
        invalid_ratio=float(np.count_nonzero(~finite) / total),
        zero_ratio=float(np.count_nonzero(zero) / total),
        hole_ratio=float(np.count_nonzero(holes) / total),
        extreme_threshold=extreme_threshold,
        extreme_ratio=extreme_ratio,
    )


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


def load_point_cloud(
    path: Path,
    max_points: int = 450_000,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    max_distance: float | None = 10.0,
    intrinsics: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    cam_offset: float = 0.05,
    grid: int = 5,
    tau_rel: float = 0.15,
    occlusion: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> PointCloudData:
    from plyfile import PlyData

    def abort_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("点云加载已取消")

    abort_if_cancelled()
    ply = PlyData.read(str(path))
    abort_if_cancelled()
    if "vertex" not in {element.name for element in ply.elements}:
        raise ValueError("PLY 文件不包含 vertex 元素")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY 顶点缺少 x / y / z 坐标")

    if max_points < 1:
        raise ValueError("点云显示上限必须大于 0")
    if max_distance is not None:
        max_distance = float(max_distance)
        if not np.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("点云过滤距离必须大于 0")
    cam_offset = float(cam_offset)
    tau_rel = float(tau_rel)
    grid = int(grid)
    if not np.isfinite(cam_offset) or cam_offset < 0:
        raise ValueError("虚拟相机后移距离不能小于 0")
    if grid < 1:
        raise ValueError("点云采样网格必须大于 0")
    if not np.isfinite(tau_rel) or not 0.0 <= tau_rel <= 1.0:
        raise ValueError("飞点过滤强度必须在 0 到 1 之间")
    camera_matrix, (image_width, image_height) = resolve_camera_model(
        intrinsics,
        image_size,
    )

    rotation_array = (
        np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        if rotation is not None
        else None
    )
    translation_array = (
        np.asarray(translation, dtype=np.float32).reshape(3)
        if translation is not None
        else None
    )

    finite = np.ones(len(vertex), dtype=bool)
    for field in ("x", "y", "z"):
        finite &= np.isfinite(vertex[field])
    valid_indices = np.flatnonzero(finite)
    original_count = len(valid_indices)
    abort_if_cancelled()
    points = np.column_stack([vertex[field][valid_indices] for field in ("x", "y", "z")]).astype(
        np.float32,
        copy=False,
    )
    if rotation_array is not None:
        points = points @ rotation_array.T
    if translation_array is not None:
        points = points + translation_array.reshape(1, 3)
    transformed_finite = np.isfinite(points).all(axis=1)
    if max_distance is None:
        in_range = transformed_finite & (points[:, 2] > 0.0)
    else:
        in_range = (
            transformed_finite
            & (points[:, 2] > 0.0)
            & (points[:, 2] <= max_distance)
        )
    points = points[in_range]
    valid_indices = valid_indices[in_range]
    abort_if_cancelled()

    if len(points):
        uv, projected_z = project_camera_points(points, camera_matrix, cam_offset)
        projectable = np.isfinite(projected_z) & (projected_z > 0.0)
        points = points[projectable]
        valid_indices = valid_indices[projectable]
        uv = uv[projectable]
        selected_positions = _grid_representatives(
            points,
            uv,
            image_width,
            image_height,
            grid,
            tau_rel,
            bool(occlusion),
        )
        points = points[selected_positions]
        valid_indices = valid_indices[selected_positions]
    abort_if_cancelled()

    filtered_count = len(points)
    if filtered_count > max_points:
        sample_positions = np.linspace(0, filtered_count - 1, max_points, dtype=np.int64)
        points = points[sample_positions]
        selected_indices = valid_indices[sample_positions]
    else:
        selected_indices = valid_indices
    abort_if_cancelled()

    color_fields = next(
        (fields for fields in (("red", "green", "blue"), ("r", "g", "b")) if set(fields).issubset(names)),
        None,
    )
    if color_fields:
        colors = np.column_stack([vertex[field][selected_indices] for field in color_fields]).astype(
            np.float32,
            copy=False,
        )
        maximum = float(np.nanmax(colors)) if colors.size and np.isfinite(colors).any() else 0.0
        if maximum > 1.0:
            divisor = 255.0 if maximum <= 255.0 else 65535.0 if maximum <= 65535.0 else maximum
            colors /= divisor
        colors = np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.tile(np.array([[0.29, 0.72, 0.92]], dtype=np.float32), (len(points), 1))

    alpha = np.ones((len(colors), 1), dtype=np.float32)
    return PointCloudData(
        points,
        np.column_stack((colors, alpha)),
        original_count,
        filtered_count,
        max_distance,
    )
