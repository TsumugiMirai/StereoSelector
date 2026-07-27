# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules("pyqtgraph.opengl")
datas = collect_data_files("pyqtgraph") + [
    ("src/stereo_selector/assets", "stereo_selector/assets"),
    ("calibration/calib2000_004_b/calibration_param.json", "calibration/calib2000_004_b"),
    ("calibration/calib3000_003/calibration_param.json", "calibration/calib3000_003"),
]

a = Analysis(
    ["main.py"],
    pathex=["src"],
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
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="StereoSelector-v1.1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="src/stereo_selector/assets/app_icon.png",
    version="version_info.txt",
)
