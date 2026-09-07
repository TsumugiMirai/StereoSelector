from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtGui import QImage

from .datatypes import ImageData, PointCloudData
from .projection import _grid_representatives, project_camera_points, resolve_camera_model
from .stats import _normalize_numeric


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
    from .plyio import read_ply_vertices

    def abort_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("点云加载已取消")

    abort_if_cancelled()
    vertex, total_vertex_count = read_ply_vertices(
        path,
        max(1, max_points * 3),
    )
    abort_if_cancelled()
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
    original_count = (
        total_vertex_count
        if len(vertex) < total_vertex_count
        else len(valid_indices)
    )
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
