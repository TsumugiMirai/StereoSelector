from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


