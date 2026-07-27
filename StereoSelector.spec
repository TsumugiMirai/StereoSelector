# -*- mode: python ; coding: utf-8 -*-

import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH)
PROJECT_TEXT = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
VERSION_MATCH = re.search(r'(?m)^version\s*=\s*"([^"]+)"', PROJECT_TEXT)
if VERSION_MATCH is None:
    raise RuntimeError("Unable to determine the application version from pyproject.toml")

PROJECT_VERSION = VERSION_MATCH.group(1)
VERSION_PARTS = PROJECT_VERSION.split(".")
RELEASE_VERSION = (
    ".".join(VERSION_PARTS[:2])
    if len(VERSION_PARTS) == 3 and VERSION_PARTS[2] == "0"
    else PROJECT_VERSION
)
APP_NAME = f"StereoSelector-v{RELEASE_VERSION}"
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
DEFAULT_ICON = PROJECT_ROOT / "src" / "stereo_selector" / "assets" / "app_icon.png"
ICON_PATH = os.environ.get("STEREO_SELECTOR_ICON", str(DEFAULT_ICON))

hiddenimports = collect_submodules("pyqtgraph.opengl")
datas = collect_data_files("pyqtgraph") + [
    (str(PROJECT_ROOT / "src" / "stereo_selector" / "assets"), "stereo_selector/assets"),
    (
        str(PROJECT_ROOT / "calibration" / "calib2000_004_b" / "calibration_param.json"),
        "calibration/calib2000_004_b",
    ),
    (
        str(PROJECT_ROOT / "calibration" / "calib3000_003" / "calibration_param.json"),
        "calibration/calib3000_003",
    ),
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe_options = {
    "name": APP_NAME,
    "debug": False,
    "bootloader_ignore_signals": False,
    "strip": False,
    "upx": IS_WINDOWS,
    "console": False,
    "disable_windowed_traceback": False,
    "argv_emulation": False,
    "target_arch": None,
    "codesign_identity": None,
    "entitlements_file": None,
    "icon": ICON_PATH,
}

if IS_WINDOWS:
    exe_options["version"] = str(PROJECT_ROOT / "version_info.txt")

if IS_MACOS:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        **exe_options,
    )
    collected = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name=APP_NAME,
    )
    app = BUNDLE(
        collected,
        name=f"{APP_NAME}.app",
        icon=ICON_PATH,
        bundle_identifier="com.toolbox.stereoselector",
        info_plist={
            "CFBundleDisplayName": "Stereo Selector",
            "CFBundleShortVersionString": PROJECT_VERSION,
            "CFBundleVersion": PROJECT_VERSION,
            "NSHighResolutionCapable": True,
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        **exe_options,
    )
