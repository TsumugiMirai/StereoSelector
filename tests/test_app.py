from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtCore import QSettings, QThreadPool
from PySide6.QtTest import QTest

from stereo_selector.app import MainWindow
from stereo_selector import app as app_module
from stereo_selector.bootstrap import application_icon, asset_path
from stereo_selector.calibration import builtin_calibration_options
from stereo_selector.mapping import MappingDialog
from stereo_selector.settings import (
    AppPreferences,
    ChoiceButton,
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


def test_empty_startup_stays_idle_without_open_dialog(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    choose_calls: list[bool] = []
    monkeypatch.setattr(MainWindow, "choose_project", lambda self: choose_calls.append(True))

    window = MainWindow()
    window.show()
    QTest.qWait(260)
    app.processEvents()

    assert choose_calls == []
    assert window.dataset is None
    assert window.content_stack.currentWidget() is window.empty_hint
    assert window.status_text.text() == ""
    assert window.view_hint.isHidden()
    assert window.title_bar.context.isHidden()
    assert window.project_path_label.isHidden()
    assert window.project_summary_label.isHidden()
    assert window.change_button.isHidden()
    assert window.manual_mapping_button.isHidden()
    assert window.modes_panel.isHidden()
    assert window.quality_panel.isHidden()
    assert window.review_panel.isHidden()
    assert window.output_panel.isHidden()
    assert window.reset_button.isHidden()
    assert window.review_bar.isHidden()
    assert window.inspection_bar.isHidden()
    assert not hasattr(window, "shortcut_hint")
    window.close()


def test_application_icon_asset_is_available() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    assert asset_path("app_icon.png").is_file()
    assert not application_icon().isNull()


def test_review_focus_and_accept_flow(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    app.processEvents()

    assert window.dataset is not None
    assert len(window.dataset.samples) == 2
    assert list(window.tiles) == ["left", "right"]
    assert not window.project_path_label.isHidden()
    assert not window.project_summary_label.isHidden()
    assert window.change_button.text() == "更改项目"
    assert not window.review_bar.isHidden()
    assert "按文件名匹配" not in window.project_summary_label.text()
    assert "滚轮缩放" not in window.view_hint.text()
    assert window.previous_button.text() == window.preferences.button_labels["previous"]
    assert window.next_button.text() == window.preferences.button_labels["next"]
    assert window.accept_button.text() == window.preferences.button_labels["accept"]
    assert window._shortcut_text("previous") in window.previous_button.toolTip()
    assert window._shortcut_text("accept") in window.accept_button.toolTip()

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
    review_document = json.loads(
        (project.parent / "capture_select" / "stereo_selector_review.json").read_text(encoding="utf-8")
    )
    assert review_document["samples"][first.key]["status"] == "accepted"
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


def test_accepted_sample_changes_the_entire_review_bar_state(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    window.preferences.auto_advance = False
    app.processEvents()

    assert window.review_bar.property("reviewState") == "pending"
    assert not window.sample_status.property("accepted")
    window.accept_current()
    app.processEvents()

    assert window.review_bar.property("reviewState") == "accepted"
    assert window.sample_status.property("accepted")
    assert window.sample_status.text() == "已接受"
    assert window.accept_button.text() == "继续"

    window.next_sample()
    assert window.review_bar.property("reviewState") == "pending"
    assert window.sample_status.text() == "待审阅"
    window.close()


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
    assert loaded.button_labels["accept"] == "接受"
    assert loaded.shortcuts == {
        "previous": "Left",
        "next": "Right",
        "accept": "Return",
        "focus": "F",
        "reset": "R",
    }


def test_legacy_default_button_labels_are_simplified(tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "legacy.ini"), QSettings.Format.IniFormat)
    settings.setValue("preferences/buttons/previous", "←  上一组")
    settings.setValue("preferences/buttons/next", "跳过 / 下一组  →")
    settings.setValue("preferences/buttons/accept", "接受并继续")

    loaded = AppPreferences.load(settings)

    assert loaded.button_labels == {"previous": "上一组", "next": "下一组", "accept": "接受"}


def test_settings_use_consistent_selection_controls() -> None:
    app = QApplication.instance() or QApplication([])
    options = builtin_calibration_options()
    dialog = SettingsDialog(
        AppPreferences(theme="light", auto_advance=False),
        calibration_options=options,
        project_available=True,
    )
    assert isinstance(dialog.theme_combo, SegmentedControl)
    assert dialog.theme_combo.currentData() == "light"
    assert isinstance(dialog.auto_advance_check, ToggleSwitch)
    assert isinstance(dialog.calibration_combo, ChoiceButton)
    assert not dialog.auto_advance_check.isChecked()
    assert dialog.navigation.count() == 6
    assert dialog.calibration_combo.count() == 3
    assert dialog.calibration_combo.itemText(1).startswith("libra2000")
    assert dialog.calibration_combo.itemText(2).startswith("libra3000")
    dialog.close()


def test_settings_open_as_an_integrated_main_window_page() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()

    window.open_settings()
    app.processEvents()

    dialog = window._settings_page
    assert dialog is not None
    assert window.page_stack.currentWidget() is dialog
    assert dialog.parentWidget() is window.page_stack
    assert dialog.objectName() == "settingsPage"
    assert not dialog.isWindow()
    margins = dialog._window_layout.contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (0, 0, 0, 0)
    assert dialog.dialog_surface.property("embedded")
    assert window.title_bar.context.isHidden()
    assert window.title_bar.settings_button.isHidden()
    assert window.app_status_bar.isHidden()

    dialog.reject()
    app.processEvents()
    assert window._settings_page is None
    assert window.page_stack.currentWidget() is window.main_page
    assert not window.title_bar.settings_button.isHidden()
    window.close()


def test_choice_button_uses_theme_owned_popup() -> None:
    app = QApplication.instance() or QApplication([])
    selector = ChoiceButton()
    selector.addItem("无标定", "")
    selector.addItem("libra2000", "builtin:libra2000")

    assert selector._menu.objectName() == "choiceMenu"
    assert selector.findData("builtin:libra2000") == 1
    selector.setCurrentIndex(1)
    assert selector.currentData() == "builtin:libra2000"
    assert selector.currentText() == "libra2000"
    selector.close()


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


def test_custom_output_is_saved_per_project_and_restored(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    custom_output = tmp_path / "exports" / "manual_review"
    window = MainWindow()
    window.settings = QSettings(str(tmp_path / "output.ini"), QSettings.Format.IniFormat)
    window._save_output_for(project.resolve(), custom_output.resolve())

    window.load_project(project)

    assert window.dataset is not None
    assert window.dataset.output_root == custom_output.resolve()
    assert window.output_label.text() == str(custom_output.resolve())
    assert (custom_output / "stereo_selector_review.json").is_file()
    window.accept_current()
    assert (custom_output / "left" / "frame_0001.png").is_file()
    window.close()


def test_configure_output_updates_active_project_without_moving_old_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    selected_output = (tmp_path / "chosen" / "named_output").resolve()
    window = MainWindow(project)
    app.processEvents()
    window.settings = QSettings(str(tmp_path / "output.ini"), QSettings.Format.IniFormat)
    old_output = window.dataset.output_root

    class AcceptedOutputDialog:
        def __init__(self, *args, **kwargs) -> None:
            self.output_root = selected_output

        def exec(self) -> int:
            return 1

    monkeypatch.setattr(app_module, "OutputSettingsDialog", AcceptedOutputDialog)
    window.configure_output()

    assert window.dataset is not None
    assert window.dataset.output_root == selected_output
    assert window._saved_output_for(project.resolve()) == selected_output
    assert (selected_output / "stereo_selector_review.json").is_file()
    assert (old_output / "stereo_selector_review.json").is_file()
    assert not list(old_output.rglob("*.png"))
    window.close()


def test_annotation_button_reflects_saved_tags_and_note(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    app.processEvents()
    assert window.review_store is not None
    sample = window.dataset.samples[0]

    window.review_store.set_annotation(sample, ["模糊"], "边缘需要复查")
    window._show_current()

    assert window.annotation_button.text() == "缺陷与备注 · 2"
    assert "模糊" in window.annotation_button.toolTip()
    assert "边缘需要复查" in window.annotation_button.toolTip()
    window.close()


def test_inspection_tools_link_images_and_report_depth_quality(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    import numpy as np

    depth = project / "depth_fsd" / "frame_0001.png"
    depth.parent.mkdir()
    Image.fromarray(np.array([[0, 1000], [2000, 5000]], dtype=np.uint16)).save(depth)
    window = MainWindow(project)
    window.show()
    app.processEvents()
    QThreadPool.globalInstance().waitForDone(5000)
    app.processEvents()

    assert not window.inspection_bar.isHidden()
    assert window.overlay_button.isEnabled()
    assert window.title_bar.minimize_button.toolTip() == ""
    assert window.title_bar.maximize_button.toolTip() == ""
    window.checkboxes["depth_fsd"].setChecked(True)
    QThreadPool.globalInstance().waitForDone(5000)
    app.processEvents()
    assert "范围" in window.depth_quality_label.text()
    assert "零值" in window.depth_quality_label.text()

    window.crosshair_button.setChecked(True)
    window._media_cursor_moved("left", 0.5, 0.5)
    assert "RGB" in window.cursor_info.text()
    assert "深度" in window.cursor_info.text()
    assert window.tiles["left"].image_canvas._crosshair_vertical.isVisible()

    window.epiline_button.setChecked(True)
    window._media_cursor_moved("left", 0.5, 0.5)
    assert window.tiles["right"].image_canvas._epiline.isVisible()

    window.sync_views_button.setChecked(True)
    window._media_view_changed("left", 2.0, 0.4, 0.6)
    assert window.tiles["right"].image_canvas._zoom_factor == 2.0

    annotation_center = window.annotation_button.mapTo(window, window.annotation_button.rect().center()).y()
    accept_center = window.accept_button.mapTo(window, window.accept_button.rect().center()).y()
    assert abs(annotation_center - accept_center) <= 1
    window.close()


def test_main_page_can_switch_builtin_calibration_per_project(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    app.processEvents()

    assert window.calibration_picker.findData("builtin:libra2000") >= 0
    assert window.calibration_picker.findData("builtin:libra3000") >= 0
    window.calibration_picker.setCurrentIndex(
        window.calibration_picker.findData("builtin:libra3000")
    )
    app.processEvents()

    assert window.current_calibration_id == "builtin:libra3000"
    assert window.calibration is not None
    assert window.calibration.left is not None
    assert window.calibration.left.width == 1280
    prefix = window._matching_settings_prefix(project.resolve())
    assert window.settings.value(f"{prefix}/calibration_id") == "builtin:libra3000"
    assert not hasattr(window, "calibration_button")
    window.close()


def test_custom_calibration_is_imported_from_settings(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    calibration_path = tmp_path / "custom_camera.json"
    calibration_path.write_text(
        json.dumps(
            {
                "left": {
                    "camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
                    "distortion": [0, 0, 0, 0, 0],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        app_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(calibration_path), "标定文件"),
    )
    dialog = SettingsDialog(
        AppPreferences(),
        calibration_options=builtin_calibration_options(),
        project_available=True,
    )

    dialog._import_calibration()

    selected_id = str(dialog.calibration_combo.currentData())
    assert selected_id.startswith("custom:")
    assert any(option.id == selected_id for option in dialog.calibration_options)
    assert "左目内参" in dialog.calibration_detail.text()
    dialog.close()
    assert app is not None
