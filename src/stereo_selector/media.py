from __future__ import annotations

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


@dataclass(frozen=True)
class ImageData:
    image: QImage
    values: np.ndarray


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
        extreme_ratio = float(np.count_nonzero(positive > extreme_threshold) / total)
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


def load_point_cloud(
    path: Path,
    max_points: int = 450_000,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> PointCloudData:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    if "vertex" not in {element.name for element in ply.elements}:
        raise ValueError("PLY 文件不包含 vertex 元素")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY 顶点缺少 x / y / z 坐标")

    if max_points < 1:
        raise ValueError("点云显示上限必须大于 0")

    # Find valid rows before building Nx3/Nx4 matrices. For multi-million point
    # clouds this avoids allocating full coordinate and color copies that will
    # immediately be discarded by sampling.
    finite = np.ones(len(vertex), dtype=bool)
    for field in ("x", "y", "z"):
        finite &= np.isfinite(vertex[field])
    valid_indices = np.flatnonzero(finite)
    original_count = len(valid_indices)
    if original_count > max_points:
        sample_positions = np.linspace(0, original_count - 1, max_points, dtype=np.int64)
        selected_indices = valid_indices[sample_positions]
    else:
        selected_indices = valid_indices

    points = np.column_stack([vertex[field][selected_indices] for field in ("x", "y", "z")]).astype(
        np.float32,
        copy=False,
    )
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        points = points @ rotation.T
    if translation is not None:
        points = points + np.asarray(translation, dtype=np.float32).reshape(1, 3)

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

    # Camera coordinates: X right, Y down, Z forward -> OpenGL: X right, Y depth, Z up.
    oriented = np.column_stack((points[:, 0], points[:, 2], -points[:, 1])).astype(np.float32)
    alpha = np.ones((len(colors), 1), dtype=np.float32)
    return PointCloudData(oriented, np.column_stack((colors, alpha)), original_count)
