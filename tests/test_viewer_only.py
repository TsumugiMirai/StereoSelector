"""Read-only product contract and viewer interaction regressions."""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from stereo_selector import app as app_module
from stereo_selector import workers
from stereo_selector.app import MainWindow
from stereo_selector.inspection import CloudProjectionDialog, StereoOverlayDialog
from stereo_selector.media import PointCloudData
from stereo_selector.models import DatasetScanner
from stereo_selector.settings import AppPreferences


def settle(app):
    for _ in range(3):
        app.processEvents()
        workers.wait_for_all(3000)
    app.processEvents()


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(app_module, "QSettings", lambda *args: settings)
    root = tmp_path / "capture"
    for side in ("left", "right"):
        (root / side).mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (48, 32), "navy").save(root / side / f"{index}.png")
    window = MainWindow()
    window.show()
    yield app, window, root, settings
    window.close()
    settle(app)


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_legacy_review_settings_and_output_are_untouched(viewer):
    app, window, root, settings = viewer
    output = root.with_name(root.name + "_select")
    output.mkdir()
    # Invalid legacy JSON must not be parsed, reconciled, or repaired by a viewer.
    (output / "stereo_selector_review.json").write_bytes(b"legacy-unreadable-record")
    (output / "selected.png").write_bytes(b"existing-output")
    prefix = window._matching_settings_prefix(root)
    settings.setValue(f"{prefix}/product_mode", "review")
    settings.setValue(f"{prefix}/output_root", str(output))
    before, before_output = snapshot(root), snapshot(output)
    window.load_project(root)
    settle(app)
    for key in (Qt.Key.Key_Return, Qt.Key.Key_X, Qt.Key.Key_P, Qt.Key.Key_N):
        QTest.keyClick(window, key)
    QTest.keyClick(window, Qt.Key.Key_2, Qt.KeyboardModifier.ControlModifier)
    window.next_sample()
    window.close()
    settle(app)
    assert snapshot(root) == before
    assert snapshot(output) == before_output
    assert settings.value(f"{prefix}/product_mode") == "review"
    assert not hasattr(window, "review_store")
    assert not hasattr(window, "mode_picker")
    assert not hasattr(window, "accept_current")
    assert "review" not in window.activity_buttons


def test_new_project_creates_no_output_and_no_review_commands(viewer, monkeypatch):
    app, window, root, _ = viewer
    window.load_project(root)
    settle(app)
    actions = []
    monkeypatch.setattr(app_module, "CommandPalette", lambda entries, _parent:
                        (actions.extend(entries) or SimpleNamespace(exec=lambda: None)))
    window.open_command_palette()
    assert not any(any(word in entry[0] for word in ("接受", "拒绝", "待定", "筛选", "缺陷")) for entry in actions)
    assert set(window.action_shortcuts) == {"previous", "next", "first", "last", "playback", "focus", "reset"}
    assert not root.with_name(root.name + "_select").exists()


def test_legacy_preferences_are_ignored_without_erasing_them(tmp_path):
    settings = QSettings(str(tmp_path / "legacy.ini"), QSettings.Format.IniFormat)
    settings.setValue("preferences/auto_advance", False)
    settings.setValue("preferences/buttons/accept", "保留")
    settings.setValue("preferences/shortcuts/reject", "Ctrl+J")
    preferences = AppPreferences.load(settings)
    preferences.save(settings)
    assert not hasattr(preferences, "auto_advance")
    assert "reject" not in preferences.shortcuts
    assert settings.value("preferences/buttons/accept") == "保留"
    assert settings.value("preferences/shortcuts/reject") == "Ctrl+J"


def test_manual_navigation_pauses_playback_and_close_stops_timers(viewer):
    app, window, root, _ = viewer
    window.load_project(root)
    settle(app)
    window.toggle_playback()
    window.next_sample()
    assert window.current_index == 1
    assert not window._playback_timer.isActive()
    assert not window.playback_button.playing
    window.toggle_playback()
    window._cursor_update_timer.start()
    window.close()
    assert not window._playback_timer.isActive()
    assert not window._cursor_update_timer.isActive()


def test_scrub_resumes_playback_and_stops_at_last_frame(viewer):
    app, window, root, _ = viewer
    window.load_project(root)
    settle(app)
    window.toggle_playback()
    midpoint = QPoint(window.timeline.width() // 2, window.timeline.height() // 2)
    QTest.mouseClick(window.timeline, Qt.MouseButton.LeftButton, pos=midpoint)
    assert window.current_index == 1
    assert window._playback_timer.isActive()
    endpoint = QPoint(window.timeline.width() - 1, window.timeline.height() // 2)
    QTest.mouseClick(window.timeline, Qt.MouseButton.LeftButton, pos=endpoint)
    assert window.current_index == 2
    assert not window._playback_timer.isActive()


def test_single_frame_playback_command_is_noop(viewer):
    app, window, root, _ = viewer
    window.load_project(root)
    settle(app)
    window.dataset.samples = window.dataset.samples[:1]
    window.toggle_playback()
    assert not window._playback_timer.isActive()


def test_long_status_never_moves_timeline_or_frame_counter(viewer):
    app, window, root, _ = viewer
    window.load_project(root)
    window.resize(1050, 720)
    settle(app)
    before = window.timeline.geometry()
    counter_width = window.timeline_left.width()
    window.status_text.setText("非常长的项目路径与信息" * 100)
    window._timeline_preview_changed(99999)
    app.processEvents()
    assert window.timeline.geometry() == before
    assert window.timeline_left.width() == counter_width
    assert window.status_text.toolTip() == window.status_text.text()
    assert window.player_bar.height() == 40


def test_scanner_excludes_nested_legacy_outputs(viewer):
    _app, _window, root, _ = viewer
    output = root / "old_select" / "left"
    output.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(output / "extra.png")
    dataset = DatasetScanner().scan(root)
    assert len(dataset.samples) == 3
    assert len(dataset.files["left"]) == 3


def test_cloud_projection_dialog_renders_without_review_dependency():
    app = QApplication.instance() or QApplication([])
    image = QImage(16, 16, QImage.Format.Format_RGB888)
    image.fill(Qt.GlobalColor.white)
    cloud = PointCloudData(np.array([[0., 0., 1.]]), np.array([[1., 0., 0., 1.]]), 1, 1, 10.)
    dialog = CloudProjectionDialog(image, cloud, np.array([[10., 0., 8.], [0., 10., 8.], [0., 0., 1.]]), 0.)
    assert not dialog.canvas._item.pixmap().isNull()
    assert not any(button.text() == "取消" for button in dialog.findChildren(QPushButton))
    dialog.close()


def test_overlay_opacity_preserves_zoom_and_close_stops_render_timer():
    app = QApplication.instance() or QApplication([])
    image = QImage(48, 32, QImage.Format.Format_RGB888)
    image.fill(Qt.GlobalColor.white)
    dialog = StereoOverlayDialog(image, image)
    dialog.show()
    dialog._apply_blend()
    dialog.canvas.apply_view_state(2., 0.5, 0.5)
    dialog.blend_slider.setValue(70)
    dialog._apply_blend()
    assert dialog.canvas._zoom_factor == 2.
    dialog.reject()
    assert not dialog._blend_timer.isActive()
