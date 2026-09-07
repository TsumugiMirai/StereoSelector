"""Media data pipeline (loading, statistics, rendering, projection).

The package keeps the former :mod:`stereo_selector.media` public surface so
existing imports keep working while the implementation is split by concern.
"""

from __future__ import annotations

from .datatypes import DepthStats, ImageData, ImageStats, PointCloudData
from .loading import load_image, load_image_data, load_point_cloud
from .projection import project_camera_points, resolve_camera_model
from .render import colorize_scalar, downsample_pool, render_image_values
from .stats import analyze_depth, analyze_image

__all__ = [
    "DepthStats",
    "ImageData",
    "ImageStats",
    "PointCloudData",
    "analyze_depth",
    "analyze_image",
    "colorize_scalar",
    "downsample_pool",
    "load_image",
    "load_image_data",
    "load_point_cloud",
    "project_camera_points",
    "render_image_values",
    "resolve_camera_model",
]
