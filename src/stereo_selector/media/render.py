from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

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


def downsample_pool(values: np.ndarray, scale: int) -> np.ndarray:
    """Block-mean downsample for cheap preview rendering.

    ``scale`` is an integer factor; edges are padded by replication so the
    pooled array keeps the source aspect ratio.
    """
    scale = max(1, int(scale))
    if scale <= 1:
        return values
    array = np.asarray(values)
    height, width = array.shape[:2]
    pad_height = (-height) % scale
    pad_width = (-width) % scale
    padded = np.pad(
        array,
        ((0, pad_height), (0, pad_width)) + ((0, 0),) * (array.ndim - 2),
        mode="edge",
    )
    pooled_height = padded.shape[0] // scale
    pooled_width = padded.shape[1] // scale
    reshaped = padded.reshape(
        pooled_height,
        scale,
        pooled_width,
        scale,
        *padded.shape[2:],
    )
    return reshaped.mean(axis=(1, 3))


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


