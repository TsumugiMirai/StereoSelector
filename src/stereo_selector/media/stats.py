from __future__ import annotations

import numpy as np

from .datatypes import DepthStats, ImageStats


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


