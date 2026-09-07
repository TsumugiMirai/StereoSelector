"""Project opening flow: background discovery, scan guard, recent list and auto calibration."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from stereo_selector import app as app_module
from stereo_selector import workers
from stereo_selector.app import MainWindow


def settle(app):
    for _ in range(3):
        app.processEvents()
        workers.wait_for_all(3000)
    app.processEvents()


def _make_project(root: Path, frames: int = 2) -> Path:
    for index in range(frames):
        for folder in ("left", "right"):
            path = root / folder / f"{index:07d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), (40, 90, 140)).save(path)
    return root


def _vendor_calibration() -> dict:
    return {
        "camera_left": {"rows": 48, "cols": 64, "camera_matrix": {"data": [50, 0, 32, 0, 50, 24, 0, 0, 1]}, "distortion_coeffs": [0] * 5},
        "camera_right": {"rows": 48, "cols": 64, "camera_matrix": {"data": [50, 0, 32, 0, 50, 24, 0, 0, 1]}, "distortion_coeffs": [0] * 5},
        "extrinsic_l_r": {"rotation_matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation_vector": [-50, 0, 0]},
        "camera_rectify": {"rows": 48, "cols": 64, "fx": 55, "fy": 55, "cx": 32, "cy": 24,
                           "rotation_left": [1, 0, 0, 0, 1, 0, 0, 0, 1], "rotation_right": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
    }


def test_project_with_calib_folder_selects_it_automatically(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(app_module, "QSettings", lambda *args: settings)
    root = _make_project(tmp_path / "capture")
    (root / "calib").mkdir()
    (root / "calib" / "calibration_param.json").write_text(json.dumps(_vendor_calibration()), encoding="utf-8")
    window = MainWindow()
    window.load_project(root)
    settle(app)
    try:
        assert window.dataset is not None
        assert window.calibration is not None
        assert window.current_calibration_id.startswith("custom:")
        assert any(option.project_local for option in window.calibration_options)
        assert window.calibration_picker.findData(window.current_calibration_id) >= 0
        assert "项目内" in window.status_text.text()
        # Project-local options are not persisted as global custom files.
        window._save_calibration_options()
        assert json.loads(str(settings.value("calibration/custom_files"))) == []
        assert root.name in [Path(p).name for p in window._recent_projects()]
    finally:
        window.close()
        settle(app)


def test_recent_projects_are_listed_on_the_empty_state(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(app_module, "QSettings", lambda *args: settings)
    first = _make_project(tmp_path / "first")
    second = _make_project(tmp_path / "second")
    window = MainWindow()
    window.load_project(first)
    settle(app)
    window.load_project(second)
    settle(app)
    try:
        recent = window._recent_projects()
        assert [Path(p).name for p in recent] == ["second", "first"]
        assert [b.text() for b in window.empty_hint.recent_buttons] == ["second", "first"]
        window.empty_hint.recent_requested.emit(str(first))
        settle(app)
        assert window.dataset.root == first.resolve()
    finally:
        window.close()
        settle(app)


def test_load_project_is_guarded_while_a_scan_runs(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    root = _make_project(tmp_path / "capture")
    window = MainWindow()
    calls: list[Path] = []
    original = MainWindow.load_project

    def spy(self, path, *args, **kwargs):
        calls.append(path)
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(MainWindow, "load_project", spy)
    window._scan_in_progress = True
    window.load_project(root)
    assert window.dataset is None
    assert "扫描" in window.status_text.text()
    window._scan_in_progress = False
    window.load_project(root)
    settle(app)
    assert window.dataset is not None
    assert len(calls) == 2
    window.close()
    settle(app)


def test_line_measurement_uses_endpoint_depths_and_millimetres(tmp_path, monkeypatch) -> None:
    import numpy as np

    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(app_module, "QSettings", lambda *args: settings)
    root = _make_project(tmp_path / "capture", frames=1)
    depth = np.full((48, 64), 1000, dtype=np.uint16)
    depth[:, 32:] = 2000
    (root / "depth_fsd").mkdir()
    Image.fromarray(depth).save(root / "depth_fsd" / "0000000.png")
    (root / "calib").mkdir()
    (root / "calib" / "calibration_param.json").write_text(json.dumps(_vendor_calibration()), encoding="utf-8")
    window = MainWindow()
    window.load_project(root)
    settle(app)
    try:
        window.checkboxes["depth_fsd"].setChecked(True)
        settle(app)
        captured = {}
        monkeypatch.setattr(
            window.inspector,
            "set_line_measurement",
            lambda pixels, physical=None, endpoint_depths=None, note="": captured.update(
                pixels=pixels, physical=physical, depths=endpoint_depths, note=note
            ),
        )
        # Horizontal line from x=16 (depth 1 m) to x=48 (depth 2 m) at y=24 (principal row).
        window._media_selection_changed("left", "line", (16 / 64, 0.5, 48 / 64, 0.5))
        assert captured["depths"] == (1.0, 2.0)
        # Rectified fx=55: X = (u - 32)/55 * Z  -> (-16/55*1, 0, 1) to (16/55*2, 0, 2)
        expected = np.hypot(-16 / 55 * 1 - 16 / 55 * 2, 1.0)
        assert captured["physical"] == pytest.approx(expected, rel=1e-6)
        assert captured["note"] == ""
        fields = window._cursor_value_fields("left", 48 / 64, 0.5)
        assert "2000 mm" in fields["depth"]
        assert fields["xyz"].startswith("XYZ ") and fields["xyz"].endswith(" m")
    finally:
        window.close()
        settle(app)
