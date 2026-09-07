"""Depth unit handling shared by the cursor readout, measurements and clouds.

All metric computation inside the application happens in metres. Depth
images are stored in whatever unit the sensor chose; the user selects how to
interpret them in the settings page and every readout goes through
:func:`depth_to_meters`.
"""

from __future__ import annotations

import math

import numpy as np

DEPTH_UNIT_CHOICES: tuple[tuple[str, str], ...] = (
    ("auto", "自动：整数为毫米，浮点为米"),
    ("mm", "毫米"),
    ("m", "米"),
)
DEPTH_UNIT_IDS = tuple(unit for unit, _label in DEPTH_UNIT_CHOICES)

SATURATED_UINT16 = 65535.0


def normalize_depth_unit(unit: object) -> str:
    text = str(unit or "auto").strip().casefold()
    return text if text in DEPTH_UNIT_IDS else "auto"


def resolve_depth_unit(dtype: np.dtype | type | None, unit: str) -> str:
    """Return ``"mm"`` or ``"m"`` for a depth array of ``dtype``."""
    unit = normalize_depth_unit(unit)
    if unit != "auto":
        return unit
    if dtype is None:
        return "mm"
    kind = np.dtype(dtype).kind
    return "mm" if kind in {"u", "i", "b"} else "m"


def depth_scale_to_meters(dtype: np.dtype | type | None, unit: str) -> float:
    return 0.001 if resolve_depth_unit(dtype, unit) == "mm" else 1.0


def depth_to_meters(value: float, dtype: np.dtype | type | None, unit: str) -> float:
    return float(value) * depth_scale_to_meters(dtype, unit)


def depth_is_valid(value: float | None, dtype: np.dtype | type | None) -> bool:
    """A usable depth is finite, positive and not the uint16 saturation code."""
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(number) or number <= 0:
        return False
    if dtype is not None and np.dtype(dtype).kind in {"u", "i"} and number >= SATURATED_UINT16:
        return False
    return True


def format_depth(value: float, dtype: np.dtype | type | None, unit: str) -> str:
    """Human readable depth with its native unit and the metric equivalent."""
    resolved = resolve_depth_unit(dtype, unit)
    meters = depth_to_meters(value, dtype, unit)
    if resolved == "mm":
        return f"{value:g} mm ({meters:.3f} m)"
    return f"{meters:.3f} m"


def format_length(meters: float) -> str:
    if meters < 1.0:
        return f"{meters * 1000:.1f} mm"
    return f"{meters:.3f} m"
