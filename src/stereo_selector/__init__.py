"""Stereo Selector: a stereo image, depth and point-cloud inspection workbench."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from pathlib import Path

DISTRIBUTION_NAME = "stereo-image-selector"


def _read_version() -> str:
    """Return the single project version declared in pyproject.toml.

    A source checkout reads pyproject.toml directly so the value is always
    current. Installed and frozen builds have no pyproject.toml and read the
    distribution metadata instead (PyInstaller copies it via the spec file).
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"))
    except OSError:
        match = None
    if match:
        return match.group(1)
    try:
        return _installed_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _read_version()


def release_version(version: str = __version__) -> str:
    """Return the short marketing version, e.g. ``1.1`` for ``1.1.0``."""
    parts = version.split(".")
    if len(parts) == 3 and parts[2] == "0":
        return ".".join(parts[:2])
    return version
