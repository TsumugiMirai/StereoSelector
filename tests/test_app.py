from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPoint, QRunnable, QSettings, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from stereo_selector import app as app_module
from stereo_selector import workers
from stereo_selector.app import MainWindow
from stereo_selector.bootstrap import application_icon, asset_path
from stereo_selector.calibration import builtin_calibration_options
from stereo_selector.mapping import MappingDialog
from stereo_selector.settings import (
    DEFAULT_SHORTCUTS,
    AppPreferences,
    ChoiceButton,
    SegmentedControl,
    SettingsDialog,
    ToggleSwitch,
    reserved_shortcut_actions,
)
from stereo_selector.theme import PALETTES, style_for
from stereo_selector.workers import ProjectScanSignals, RectifySignals


def _make_project(root: Path) -> Path:
    for index in (1, 2):
        for folder, color in (("left", "navy"), ("right", "teal")):
            path = root / folder / f"frame_{index:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), color).save(path)
    return root


def test_cancel_project_scan_returns_without_missing_result_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])

    class WaitingScanWorker(QRunnable):
        instance = None

        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            self.signals = ProjectScanSignals()
            self.cancelled = False
            WaitingScanWorker.instance = self

        def cancel(self) -> None:
            self.cancelled = True

    class HoldingPool:
        def start(self, _worker) -> None:
            return

    monkeypatch.setattr(app_module, "ProjectScanWorker", WaitingScanWorker)
    monkeypatch.setattr(app_module, "SCAN_POOL", HoldingPool())
    window = MainWindow()

    def click_cancel() -> None:
        dialogs = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, QProgressDialog)
        ]
        assert dialogs
        button = dialogs[0].findChild(QPushButton)
        assert button is not None
        button.click()

    QTimer.singleShot(20, click_cancel)
    window.load_project(tmp_path)

    assert WaitingScanWorker.instance is not None
    assert WaitingScanWorker.instance.cancelled
    assert window.dataset is None
    assert window.current_root is None
    window.close()


def test_stale_rectification_failure_cannot_cancel_new_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_project(_make_project(tmp_path / "capture"))
    workers.wait_for_all(5000)
    app.processEvents()
    assert window._apply_calibration_selection("builtin:libra3000")
    workers.wait_for_all(5000)
    app.processEvents()

    class PendingRectifyWorker:
        def __init__(self, token, *_args) -> None:
            self.token = token
            self.signals = RectifySignals()

    class HoldingPool:
        def __init__(self) -> None:
            self.workers = []

        def start(self, worker) -> None:
            self.workers.append(worker)

    pool = HoldingPool()
    monkeypatch.setattr(app_module, "RectifyWorker", PendingRectifyWorker)
    monkeypatch.setattr(app_module, "IMAGE_POOL", pool)

    window.rectify_button.blockSignals(True)
    window.rectify_button.setChecked(True)
    window.rectify_button.blockSignals(False)
    window._apply_rectified_views()
    first = pool.workers[-1]

    window.rectify_button.setChecked(False)
    window.rectify_button.blockSignals(True)
    window.rectify_button.setChecked(True)
    window.rectify_button.blockSignals(False)
    window._apply_rectified_views()
    second = pool.workers[-1]
    assert second.token > first.token

    first.signals.failed.emit(first.token, "stale failure")
    app.processEvents()

    assert window.rectify_button.isChecked()
    assert window._rectify_token == second.token
    assert window._rectify_worker is second
    window.rectify_button.setChecked(False)
    window.close()


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
    assert not window.change_button.isHidden()
    assert not window.manual_mapping_button.isEnabled()
    assert window.modes_panel.isHidden()
    assert window.quality_panel.isHidden()
    assert not hasattr(window, "review_panel")
    assert not hasattr(window, "output_panel")
    assert window.reset_button.isHidden()
    assert window.player_bar.isHidden()
    assert window.inspection_bar.isHidden()
    assert not hasattr(window, "shortcut_hint")
    window.close()


def test_application_icon_asset_is_available() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    assert asset_path("app_icon.png").is_file()
    assert not application_icon().isNull()


def test_ui_design_tokens_cover_both_themes() -> None:
    required = {
        "window",
        "surface",
        "panel",
        "border",
        "border_strong",
        "text",
        "text_secondary",
        "hover",
        "pressed",
        "focus",
        "accent",
        "success",
        "warning",
        "danger",
    }
    for theme in ("dark", "light"):
        assert required <= PALETTES[theme].keys()
        stylesheet = style_for(theme)
        assert "$" not in stylesheet
        assert '"Segoe UI Variable Text"' in stylesheet
        assert "QFrame#commandSurface" in stylesheet


def test_primary_actions_share_the_integrated_title_bar() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()

    assert window.title_bar.height() == 40
    assert not hasattr(window, "mode_picker")
    assert window.open_button.parentWidget() is window.title_bar.actions
    assert window.layout_picker.parentWidget() is window.title_bar.actions
    assert window.header_settings_button.parentWidget() is window.title_bar.actions
    assert window.topbar_button.parentWidget() is window.title_bar.actions
    assert not window.topbar_button.isEnabled()
    assert window.sidebar_button.parentWidget() is window.inspection_tools_row
    assert window.inspection_tools_row.parentWidget() is window.inspection_bar
    assert window.reset_button.parentWidget() is window.inspection_tools_row
    assert window.view_hint.parentWidget() is window.inspection_tools_row
    assert window.media_grid.contentsMargins().left() == 8
    assert window.media_grid.spacing() == 8

    window.open_settings()
    app.processEvents()
    assert window.title_bar.actions.isHidden()
    assert window._settings_page is not None
    assert window.title_bar.context.text() == "设置"
    assert window._settings_page.header.isHidden()
    window._settings_page.reject()
    app.processEvents()
    assert not window.title_bar.actions.isHidden()
    assert window.app_status_bar.isHidden()
    window.close()


def test_viewer_focus_and_navigation_flow(tmp_path: Path) -> None:
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
    assert not window.player_bar.isHidden()
    assert "按文件名匹配" not in window.project_summary_label.text()
    assert "滚轮缩放" not in window.view_hint.text()
    assert window._shortcut_text("previous") in window.playback_previous_button.toolTip()
    assert window._shortcut_text("next") in window.playback_next_button.toolTip()
    assert not hasattr(window, "product_mode")
    assert not hasattr(window, "review_controls")

    window.focus_modality("left")
    assert window.focused_modality == "left"
    assert not window.tiles["left"].isHidden()
    assert window.tiles["right"].isHidden()
    window.exit_focus()
    assert all(not tile.isHidden() for tile in window.tiles.values())

    window.next_sample()
    assert window.current_index == 1
    window.previous_sample()
    assert window.current_index == 0
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
    assert window._native_frame_applied is (sys.platform == "win32")
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
    assert not pooled.has_pending_work()
    assert not pooled.loading_delay_active()
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

    workers.wait_for_all(5000)
    app.processEvents()
    assert tile.stack.currentWidget() is tile.image_canvas

    tile.show_file(second_path)
    assert tile.stack.currentWidget() is tile.image_canvas
    assert not tile.spinner._timer.isActive()
    assert tile.loading_delay_active()
    workers.wait_for_all(5000)
    app.processEvents()
    assert tile.stack.currentWidget() is tile.image_canvas

    tile.show_file(first_path)
    assert not tile.has_pending_work()
    assert tile.stack.currentWidget() is tile.image_canvas
    window.close()


def test_preferences_round_trip(tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    preferences = AppPreferences(
        theme="light",
        point_limit=150_000,
        cloud_cam_offset=0.1,
        cloud_grid=4,
        cloud_dot_radius=2.5,
        cloud_z_max=24.5,
        cloud_tau_rel=0.2,
        cloud_occlusion=False,
        shortcuts={
            "previous": "A",
            "next": "D",
            "playback": "B",
            "focus": "F",
            "reset": "R",
        },
    )
    preferences.save(settings)
    loaded = AppPreferences.load(settings)
    assert loaded == preferences


def test_invalid_persisted_preferences_are_sanitized(tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "broken.ini"), QSettings.Format.IniFormat)
    settings.setValue("preferences/theme", "unknown")
    settings.setValue("preferences/point_limit", -1)
    settings.setValue("preferences/point_cloud/cam_offset", float("nan"))
    settings.setValue("preferences/point_cloud/grid", -1)
    settings.setValue("preferences/point_cloud/dot_radius", float("nan"))
    settings.setValue("preferences/point_cloud/ply_z_max", float("nan"))
    settings.setValue("preferences/point_cloud/tau_rel", float("nan"))
    settings.setValue("preferences/buttons/accept", "   ")
    settings.setValue("preferences/shortcuts/previous", "Ctrl+O")
    settings.setValue("preferences/shortcuts/next", "Ctrl+O")
    loaded = AppPreferences.load(settings)
    assert loaded.theme == "dark"
    assert loaded.point_limit == 50_000
    assert loaded.cloud_cam_offset == 0.05
    assert loaded.cloud_grid == 1
    assert loaded.cloud_dot_radius == 1.0
    assert loaded.cloud_z_max == 10.0
    assert loaded.cloud_tau_rel == 0.15
    assert not hasattr(loaded, "button_labels")
    assert loaded.shortcuts == DEFAULT_SHORTCUTS


def test_former_point_cloud_defaults_are_migrated_once(tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "legacy-cloud.ini"), QSettings.Format.IniFormat)
    settings.setValue("preferences/point_cloud/grid", 6)
    settings.setValue("preferences/point_cloud/dot_radius", 2.0)
    settings.setValue("preferences/point_cloud/ply_z_max", 15.0)

    migrated = AppPreferences.load(settings)

    assert migrated.cloud_grid == 5
    assert migrated.cloud_dot_radius == 1.0
    assert migrated.cloud_z_max == 10.0
    migrated.save(settings)
    settings.setValue("preferences/point_cloud/grid", 6)
    settings.setValue("preferences/point_cloud/dot_radius", 2.0)
    settings.setValue("preferences/point_cloud/ply_z_max", 15.0)

    customized = AppPreferences.load(settings)

    assert customized.cloud_grid == 6
    assert customized.cloud_dot_radius == 2.0
    assert customized.cloud_z_max == 15.0


def test_settings_use_consistent_selection_controls() -> None:
    app = QApplication.instance() or QApplication([])
    options = builtin_calibration_options()
    dialog = SettingsDialog(
        AppPreferences(theme="light"),
        calibration_options=options,
        project_available=True,
    )
    assert isinstance(dialog.theme_combo, SegmentedControl)
    assert dialog.theme_combo.currentData() == "light"
    assert isinstance(dialog.cloud_occlusion_check, ToggleSwitch)
    assert isinstance(dialog.calibration_combo, ChoiceButton)
    assert not hasattr(dialog, "auto_advance_check")
    assert dialog.cloud_cam_offset_spin.value() == 0.05
    assert dialog.cloud_grid_spin.value() == 5
    assert dialog.cloud_dot_radius_spin.value() == 1.0
    assert dialog.cloud_z_max_spin.value() == 10.0
    assert dialog.cloud_tau_rel_spin.value() == 0.15
    assert dialog.cloud_occlusion_check.isChecked()
    assert dialog.navigation.count() == 4
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
    assert window.title_bar.context.text() == "设置"
    assert not window.title_bar.context.isHidden()
    assert dialog.header.isHidden()
    assert window.title_bar.settings_button.isHidden()
    assert window.app_status_bar.isHidden()

    dialog.reject()
    app.processEvents()
    assert window._settings_page is None
    assert window.page_stack.currentWidget() is window.main_page
    assert window.title_bar.settings_button.isHidden()
    assert not window.header_settings_button.isHidden()
    window.close()


def test_choice_button_uses_theme_owned_popup() -> None:
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    host.resize(360, 240)
    layout = QVBoxLayout(host)
    selector = ChoiceButton(host)
    layout.addWidget(selector)
    selector.addItem("无标定", "")
    selector.addItem("libra2000", "builtin:libra2000")

    assert selector.findData("builtin:libra2000") == 1
    host.show()
    selector.showPopup()
    app.processEvents()
    assert selector._popup is not None
    assert selector._popup.objectName() == "choicePopup"
    assert selector._popup.surface.objectName() == "choicePopupSurface"
    assert selector._popup.parentWidget() is host
    assert not selector._popup.isWindow()
    assert selector._popup.isVisible()
    assert len(selector._popup.option_buttons) == 2
    QTest.mouseClick(selector._popup.option_buttons[1], Qt.MouseButton.LeftButton)
    assert selector.currentData() == "builtin:libra2000"
    assert selector.currentText() == "libra2000"
    assert not selector._popup.isVisible()
    selector.showPopup()
    host.resize(420, 280)
    app.processEvents()
    assert not selector._popup.isVisible()
    host.close()


def test_main_choice_controls_share_non_native_popup(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(_make_project(tmp_path / "capture"))
    window.resize(1200, 760)
    window.show()
    workers.wait_for_all(5000)
    app.processEvents()

    for selector in (
        window.layout_picker,
        window.calibration_picker,
        window.playback_speed,
    ):
        assert isinstance(selector, ChoiceButton)
        selector.showPopup()
        app.processEvents()
        assert selector._popup is not None
        assert selector._popup.parentWidget() is window
        assert not selector._popup.isWindow()
        assert window.rect().contains(selector._popup.geometry())
        selector.hidePopup()

    window.close()


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


def test_inspection_tools_link_images_and_report_depth_quality(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    import numpy as np

    depth = project / "depth_fsd" / "frame_0001.png"
    depth.parent.mkdir()
    Image.fromarray(np.array([[0, 1000], [2000, 5000]], dtype=np.uint16)).save(depth)
    pseudo = project / "depth_color" / "frame_0001.png"
    pseudo.parent.mkdir()
    Image.new("RGB", (32, 24), "orange").save(pseudo)
    window = MainWindow(project)
    window.show()
    app.processEvents()
    workers.wait_for_all(5000)
    app.processEvents()
    workers.wait_for_all(5000)
    app.processEvents()

    assert not window.inspection_bar.isHidden()
    assert window.overlay_button.isEnabled()
    assert window.title_bar.minimize_button.toolTip() == ""
    assert window.title_bar.maximize_button.toolTip() == ""
    # Depth quality loads even when raw depth is not a visible comparison tile.
    assert not window.checkboxes["depth_fsd"].isChecked()
    assert "范围" in window.depth_quality_label.text()
    assert "零值" in window.depth_quality_label.text()
    assert "（" not in window.depth_quality_label.text()
    window.checkboxes["depth_fsd"].setChecked(True)
    workers.wait_for_all(5000)
    app.processEvents()
    assert "范围" in window.depth_quality_label.text()
    assert "零值" in window.depth_quality_label.text()

    window.crosshair_button.setChecked(True)
    assert (
        window.tiles["left"].image_canvas.viewport().cursor().shape()
        == Qt.CursorShape.ArrowCursor
    )
    window._media_cursor_moved("left", 0.5, 0.5)
    assert "RGB" in window.cursor_info.text()
    assert "深度" in window.cursor_info.text()
    assert window.tiles["left"].image_canvas._crosshair_vertical.isVisible()
    assert window.cursor_info.minimumWidth() == window.cursor_info.maximumWidth()
    assert not window.cursor_info_row.isHidden()
    assert window.inspection_bar.height() > 38
    cursor_origin = window.cursor_info.mapTo(window.cursor_info_row, QPoint(0, 0))
    assert cursor_origin.x() + window.cursor_info.width() <= window.cursor_info_row.width()
    readout_labels = (
        window.cursor_info.position_label,
        window.cursor_info.stereo_label,
        window.cursor_info.rgb_label,
        window.cursor_info.depth_label,
        window.cursor_info.xyz_label,
    )
    readout_geometries = tuple(label.geometry() for label in readout_labels)
    window._media_cursor_moved("left", 0.25, 0.25)
    window._media_cursor_moved("left", 0.75, 0.75)
    QTest.qWait(20)
    app.processEvents()
    assert window._crosshair_position == (0.75, 0.75)
    assert tuple(label.geometry() for label in readout_labels) == readout_geometries

    window.checkboxes["depth_color"].setChecked(True)
    workers.wait_for_all(5000)
    app.processEvents()
    window._media_cursor_moved("depth_color", 0.5, 0.5)
    QTest.qWait(20)
    app.processEvents()
    assert "RGB" in window.cursor_info.text()
    assert "深度" in window.cursor_info.text()

    window.epiline_button.setChecked(True)
    window._media_cursor_moved("left", 0.5, 0.5)
    QTest.qWait(20)
    app.processEvents()
    assert window.tiles["right"].image_canvas._epiline.isVisible()

    window.sync_views_button.setChecked(True)
    window._media_view_changed("left", 2.0, 0.4, 0.6)
    assert window.tiles["right"].image_canvas._zoom_factor == 2.0

    window.crosshair_button.setChecked(False)
    assert window.cursor_info_row.isHidden()
    assert window.inspection_bar.height() == 38
    assert (
        window.tiles["left"].image_canvas.viewport().cursor().shape()
        == Qt.CursorShape.OpenHandCursor
    )
    window.close()


def test_cached_depth_quality_is_not_overwritten_by_loading_state(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    depth_dir = project / "depth_fsd"
    depth_dir.mkdir()
    Image.fromarray(np.array([[0, 1000], [2000, 5000]], dtype=np.uint16)).save(
        depth_dir / "frame_0001.png"
    )
    Image.fromarray(np.array([[0, 3000], [4000, 65535]], dtype=np.uint16)).save(
        depth_dir / "frame_0002.png"
    )
    window = MainWindow(project)
    window.show()
    app.processEvents()
    workers.wait_for_all(5000)
    app.processEvents()

    window.next_sample()
    workers.wait_for_all(5000)
    app.processEvents()
    workers.wait_for_all(5000)
    app.processEvents()
    assert "65535" in window.depth_quality_label.text()

    window.previous_sample()

    assert "正在计算" not in window.depth_quality_label.text()
    assert "1000" in window.depth_quality_label.text()
    window.close()


def test_timeline_playback_advances_loaded_frames_and_stops_at_end(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    window.show()
    app.processEvents()
    workers.wait_for_all(5000)
    app.processEvents()

    window.playback_speed.setCurrentIndex(window.playback_speed.findData(0.5))
    assert not window.timeline.hasTracking()
    window.timeline.setFixedWidth(300)
    QTest.mouseClick(
        window.timeline,
        Qt.MouseButton.LeftButton,
        pos=QPoint(window.timeline.width() - 4, window.timeline.height() // 2),
    )
    assert window.current_index == 1
    window.first_sample()
    assert window.current_index == 0
    window.toggle_playback()

    assert window._playback_timer.isActive()
    assert window.playback_button.playing
    window._playback_tick()
    assert window.current_index == 1

    workers.wait_for_all(5000)
    app.processEvents()
    window._playback_tick()

    assert not window._playback_timer.isActive()
    assert not window.playback_button.playing
    assert window.current_index == 1
    window.close()


def test_sidebar_and_inspection_toolbar_fold_with_reversible_animation(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    window.show()
    app.processEvents()

    assert window.sidebar_panel.isHidden()
    assert not window.sidebar.isHidden()
    window.toggle_sidebar()
    assert window._sidebar_animation is None
    assert not window.project_panel.isHidden()
    QTest.qWait(240)
    assert not window.sidebar_panel.isHidden()
    assert window.sidebar.width() >= 238
    window.toggle_sidebar()
    assert window.project_panel.isHidden()
    assert window.sidebar_panel.isHidden()
    assert not window.sidebar.isHidden()
    assert window.sidebar.width() <= 42
    icon_position = window.activity_buttons["data"].mapTo(window.sidebar, QPoint(0, 0))
    assert icon_position.x() <= 3

    window.toggle_topbar()
    QTest.qWait(220)
    assert window.inspection_bar.isHidden()
    assert window.topbar_button._direction == "down"
    window.toggle_topbar()
    QTest.qWait(220)
    assert not window.inspection_bar.isHidden()
    assert window.inspection_bar.height() > 0
    assert window.topbar_button._direction == "up"
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


def test_project_collection_opens_first_child_and_switches_from_header(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    collection = tmp_path / "20260805"
    first = _make_project(collection / "140514")
    second = _make_project(collection / "140534")
    window = MainWindow()

    window.load_project(collection)
    app.processEvents()

    assert window.project_collection_root == collection.resolve()
    assert window.project_roots == [first.resolve(), second.resolve()]
    assert window.dataset is not None
    assert window.dataset.root == first.resolve()
    assert not window.project_picker.isHidden()
    assert window.project_picker.count() == 2
    assert window.project_picker.currentText() == "140514"

    window.project_picker.setCurrentIndex(1)
    app.processEvents()

    assert window.dataset is not None
    assert window.dataset.root == second.resolve()
    assert window.project_picker.currentText() == "140534"
    assert "项目 2/2" in window.project_summary_label.text()
    window.close()


def test_compact_view_chrome_and_analysis_inspector(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    window = MainWindow(project)
    window.show()
    workers.wait_for_all(5000)
    app.processEvents()
    workers.wait_for_all(5000)
    app.processEvents()

    assert window.sidebar.width() <= 42
    window.open_inspector()
    app.processEvents()
    assert window.inspector.isVisible()
    assert "均值" in window.inspector.stats_label.text()

    window.toggle_chrome()
    assert window._chrome_hidden
    assert window.title_bar.isHidden()
    assert window.player_bar.isHidden()
    window.toggle_chrome()
    assert not window._chrome_hidden
    assert not window.title_bar.isHidden()
    window.close()


def test_activity_bar_separates_image_and_cloud_tool_families(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    project = _make_project(tmp_path / "capture")
    ply = project / "ply" / "frame_0001.ply"
    ply.parent.mkdir()
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 1\n",
        encoding="ascii",
    )
    window = MainWindow(project)
    window.show()
    workers.wait_for_all(5000)
    app.processEvents()

    assert {
        "data",
        "display",
        "adjust",
        "measure",
        "statistics",
    } <= set(window.activity_buttons)
    assert "stereo" not in window.activity_buttons
    assert window.overlay_button.parentWidget() is window.inspection_tools_row
    assert window.inspection_tools_row.parentWidget() is window.inspection_bar

    window._select_activity("statistics")
    app.processEvents()
    assert window.inspector.isVisible()
    assert window.inspector._category == "statistics"
    assert not window.inspector.analysis_section.isHidden()
    assert window.inspector.cloud_section.isHidden()

    window.checkboxes["ply"].setChecked(True)
    workers.wait_for_all(5000)
    app.processEvents()
    window._select_activity("adjust")
    window.inspector.source_picker.setCurrentIndex(window.inspector.source_picker.findData("ply"))
    app.processEvents()
    assert window.inspector._category == "adjust"
    assert not window.inspector.cloud_section.isHidden()
    assert window.inspector.source_picker.currentData() == "ply"
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
