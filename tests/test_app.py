from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtCore import QSettings, QThreadPool

from stereo_selector.app import MainWindow
from stereo_selector.mapping import MappingDialog
from stereo_selector.settings import (
    AppPreferences,
    SegmentedControl,
    SettingsDialog,
    ToggleSwitch,
    reserved_shortcut_actions,
)


def _make_project(root: Path) -> Path:
    for index in (1, 2):
        for folder, color in (("left", "navy"), ("right", "teal")):
            path = root / folder / f"frame_{index:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), color).save(path)
    return root


def test_review_focus_and_accept_flow(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    app.processEvents()

    assert window.dataset is not None
    assert len(window.dataset.samples) == 2
    assert list(window.tiles) == ["left", "right"]

    window.focus_modality("left")
    assert window.focused_modality == "left"
    assert not window.tiles["left"].isHidden()
    assert window.tiles["right"].isHidden()
    window.exit_focus()
    assert all(not tile.isHidden() for tile in window.tiles.values())

    first = window.dataset.samples[0]
    window.accept_current()
    assert first.key in window.accepted
    assert window.current_index == 1
    assert (project.parent / "capture_select" / "left" / "frame_0001.png").exists()
    assert window.accepted_count_label.text() == "1"
    window.close()


def test_existing_output_is_restored_as_accepted(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    app.processEvents()
    window.accept_current()
    window.close()

    reopened = MainWindow(project)
    app.processEvents()
    assert len(reopened.accepted) == 1
    reopened.next_unreviewed()
    assert reopened.current_index == 1
    reopened.close()


def test_failed_project_switch_keeps_current_dataset(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    empty_project = tmp_path / "empty"
    empty_project.mkdir()
    window = MainWindow(project)
    app.processEvents()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)

    window.load_project(empty_project)

    assert window.dataset is not None
    assert window.dataset.root == project.resolve()
    assert window.current_root == project.resolve()
    assert list(window.tiles) == ["left", "right"]
    window.close()


def test_integrated_titlebar_window_controls(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(_make_project(tmp_path / "capture"))
    window.show()
    app.processEvents()
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert window._native_frame_applied
    assert not hasattr(window, "_resize_handles")
    assert window._resize_hit_test(101, 101, 100, 100, 900, 700, 12) == 13
    assert window._resize_hit_test(899, 400, 100, 100, 900, 700, 12) == 11
    assert window._resize_hit_test(500, 699, 100, 100, 900, 700, 12) == 15
    assert window._resize_hit_test(500, 400, 100, 100, 900, 700, 12) is None

    window.toggle_maximized()
    app.processEvents()
    assert window.isMaximized()
    window.toggle_maximized()
    app.processEvents()
    assert not window.isMaximized()
    window.close()


def test_zero_and_single_view_layout(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(_make_project(tmp_path / "capture"))
    window.show()
    app.processEvents()

    window.checkboxes["left"].setChecked(False)
    window.checkboxes["right"].setChecked(False)
    assert window.tiles == {}
    assert not window.no_views_hint.isHidden()

    window.checkboxes["left"].setChecked(True)
    assert list(window.tiles) == ["left"]
    tile_index = window.media_grid.indexOf(window.tiles["left"])
    assert window.media_grid.getItemPosition(tile_index) == (0, 0, 1, 1)
    window.close()


def test_view_toggle_reuses_point_cloud_widget(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    ply_path = project / "ply" / "frame_0001.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    ply_path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 1\n",
        encoding="ascii",
    )
    window = MainWindow(project)
    window.show()
    app.processEvents()
    pooled = window.tile_pool["ply"]
    native_handle = int(window.winId())

    window.checkboxes["ply"].setChecked(True)
    window.checkboxes["ply"].setChecked(False)
    assert not pooled._workers
    assert not pooled._loading_delay.isActive()
    window.checkboxes["ply"].setChecked(True)
    app.processEvents()
    assert window.tiles["ply"] is pooled
    assert int(window.winId()) == native_handle
    window.close()


def test_media_loading_is_asynchronous(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(_make_project(tmp_path / "capture"))
    window.show()
    app.processEvents()
    tile = window.tiles["left"]
    first_path = window.dataset.samples[0].files["left"]
    second_path = window.dataset.samples[1].files["left"]

    QThreadPool.globalInstance().waitForDone(5000)
    app.processEvents()
    assert tile.stack.currentWidget() is tile.image_canvas

    tile.show_file(second_path)
    assert tile.stack.currentWidget() is tile.image_canvas
    assert not tile.spinner._timer.isActive()
    assert tile._loading_delay.isActive()
    QThreadPool.globalInstance().waitForDone(5000)
    app.processEvents()
    assert tile.stack.currentWidget() is tile.image_canvas

    tile.show_file(first_path)
    assert not tile._workers
    assert tile.stack.currentWidget() is tile.image_canvas
    window.close()


def test_preferences_round_trip(tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    preferences = AppPreferences(
        theme="light",
        auto_advance=False,
        point_limit=150_000,
        button_labels={"previous": "向前", "next": "向后", "accept": "保留"},
        shortcuts={"previous": "A", "next": "D", "accept": "Space", "focus": "F", "reset": "R"},
    )
    preferences.save(settings)
    loaded = AppPreferences.load(settings)
    assert loaded == preferences


def test_invalid_persisted_preferences_are_sanitized(tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "broken.ini"), QSettings.Format.IniFormat)
    settings.setValue("preferences/theme", "unknown")
    settings.setValue("preferences/point_limit", -1)
    settings.setValue("preferences/buttons/accept", "   ")
    settings.setValue("preferences/shortcuts/previous", "Ctrl+O")
    settings.setValue("preferences/shortcuts/next", "Ctrl+O")
    loaded = AppPreferences.load(settings)
    assert loaded.theme == "dark"
    assert loaded.point_limit == 50_000
    assert loaded.button_labels["accept"] == "接受并继续"
    assert loaded.shortcuts == {
        "previous": "Left",
        "next": "Right",
        "accept": "Return",
        "focus": "F",
        "reset": "R",
    }


def test_settings_use_consistent_selection_controls() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SettingsDialog(AppPreferences(theme="light", auto_advance=False))
    assert isinstance(dialog.theme_combo, SegmentedControl)
    assert dialog.theme_combo.currentData() == "light"
    assert isinstance(dialog.auto_advance_check, ToggleSwitch)
    assert not dialog.auto_advance_check.isChecked()
    assert dialog.navigation.count() == 5
    dialog.close()


def test_configurable_shortcuts_cannot_shadow_fixed_actions() -> None:
    assert reserved_shortcut_actions({"previous": "Ctrl+O", "next": "Right"}) == ["previous"]
    assert reserved_shortcut_actions({"previous": "A", "next": "D"}) == []


def test_manual_mapping_dialog_returns_selected_directories(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    left = tmp_path / "camera_a"
    right = tmp_path / "camera_b"
    left.mkdir()
    right.mkdir()
    dialog = MappingDialog(tmp_path, {"left": [left]}, force_order=True)
    dialog.path_edits["right"].setText(str(right))
    assert dialog.selected_dirs() == {"left": left.resolve(), "right": right.resolve()}
    assert dialog.force_order
    assert len(dialog.path_edits) == 5
    dialog.close()


def test_manual_mapping_preferences_are_kept_per_project(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project_a = _make_project(tmp_path / "capture_a")
    project_b = _make_project(tmp_path / "capture_b")
    window = MainWindow(project_a)
    app.processEvents()
    window.settings = QSettings(str(tmp_path / "matching.ini"), QSettings.Format.IniFormat)

    window._save_matching_for(project_a.resolve(), {"left": project_a / "left"}, True)
    window._save_matching_for(project_b.resolve(), {"right": project_b / "right"}, False)

    dirs_a, force_a = window._saved_matching_for(project_a.resolve())
    dirs_b, force_b = window._saved_matching_for(project_b.resolve())
    assert dirs_a == {"left": project_a / "left"}
    assert force_a
    assert dirs_b == {"right": project_b / "right"}
    assert not force_b
    window.close()
