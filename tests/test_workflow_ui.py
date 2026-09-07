"""Regression tests for task-oriented panels and transient interaction tools."""
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from stereo_selector import media_tile, workers
from stereo_selector.app import MainWindow
from stereo_selector.inspector import InspectorPanel
from stereo_selector.media import PointCloudData, load_image_data
from stereo_selector.media_tile import MediaTile
from stereo_selector.widgets import PointCloudCanvas


def settle(app):
    # Loading, preview rendering and analysis are deliberately separate queues.
    for _ in range(4):
        app.processEvents()
        workers.wait_for_all(3000)
    app.processEvents()


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    for folder in ("left", "right"):
        (tmp_path / folder).mkdir()
        for frame in (1, 2):
            Image.new("RGB", (48, 36), (50, 80, 120)).save(tmp_path / folder / f"{frame}.png")
    w = MainWindow(tmp_path)
    w.show()
    settle(app)
    yield w
    w.close()
    settle(app)


def test_empty_pages_cannot_be_opened_by_buttons_or_commands():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.show()
    for page in ("display", "adjust", "measure", "statistics", "review", "unknown"):
        if page in w.activity_buttons:
            assert not w.activity_buttons[page].isEnabled()
        w._select_activity(page)
        assert w.sidebar_panel.isHidden()
        assert w.inspector.isHidden()
    w.open_inspector()
    assert w.inspector.isHidden()
    w._select_activity("data")
    assert w.change_button.isVisible()
    w.close()


def test_pages_follow_task_not_reference_application(window):
    panel = window.inspector
    assert set(window.activity_buttons) == {"data", "display", "adjust", "measure", "statistics"}
    window._select_activity("adjust")
    assert panel.display_section.isVisible()
    assert panel.analysis_section.isHidden()
    window._select_activity("measure")
    assert panel.view_tools_section.isVisible()
    assert panel.result_section.isVisible()
    assert panel.display_section.isHidden()
    window._select_activity("statistics")
    assert panel.analysis_section.isVisible()
    assert panel.view_tools_section.isHidden()
    assert all("ImageJ" not in b.toolTip() and "CloudCompare" not in b.toolTip()
               for b in window.activity_buttons.values())


def test_cloud_uses_the_same_task_pages():
    app = QApplication.instance() or QApplication([])
    panel = InspectorPanel()
    panel.set_sources(["left", "ply"], "ply")
    panel.set_category("measure")
    panel.show()
    assert panel.cloud_tools_section.isVisible()
    assert panel.result_section.isVisible()
    assert panel.view_tools_section.isHidden()
    panel.set_category("statistics")
    assert panel.cloud_stats_section.isVisible()
    panel.set_category("adjust")
    assert panel.cloud_section.isVisible()
    assert panel.cloud_tools_section.isHidden()
    panel.close()


def test_cloud_statistics_and_clip_bounds_refresh():
    app = QApplication.instance() or QApplication([])
    panel = InspectorPanel()
    cloud = PointCloudData(np.array([[1., 2., 3.], [4., 5., 6.]]),
                           np.ones((2, 4)), 2, 2, 10.0)
    panel.set_cloud_data(cloud, Path("first.ply"))
    assert "2 / 2" in panel.cloud_stats.text()
    assert panel.cloud_clip_spins["z"][1].value() == 6.0
    cached = panel._cloud_bounds
    panel.set_cloud_data(cloud, Path("first.ply"))
    assert panel._cloud_bounds is cached
    panel.close()


def test_depth_only_controls_hidden_for_rgb(window, tmp_path):
    panel = window.inspector
    panel.set_image_data("left", window.tiles["left"].image_data, None)
    assert panel.depth_controls.isHidden()
    path = tmp_path / "depth.png"
    Image.fromarray(np.full((10, 10), 1000, dtype=np.uint16)).save(path)
    panel.set_image_data("depth", load_image_data(path), path)
    assert not panel.depth_controls.isHidden()


@pytest.mark.parametrize("mode", ["x", "y", "z", "height"])
def test_cloud_scalar_coloring_has_valid_rgba(mode):
    cloud = PointCloudData(np.array([[1., 2., 3.], [4., 5., 6.]]),
                           np.ones((2, 4)), 2, 2, 10.0)
    colors = PointCloudCanvas._colors_for_mode(None, cloud, mode)
    assert colors.shape == (2, 4)
    assert np.isfinite(colors).all()
    assert np.all((colors >= 0) & (colors <= 1))
    assert not np.array_equal(colors[0], colors[1])


def test_command_opens_inspector_without_double_sidebars(window):
    window._select_activity("data")
    assert not window._sidebar_collapsed
    window.open_inspector()
    assert window._sidebar_collapsed
    assert window.inspector.isVisible()
    assert window.sidebar_panel.isHidden()


def test_hiding_all_views_closes_inapplicable_inspector(window):
    window._select_activity("measure")
    for checkbox in window.checkboxes.values():
        checkbox.setChecked(False)
    assert window.inspector.isHidden()
    assert not window.activity_buttons["measure"].isEnabled()
    window._select_activity("measure")
    assert window.inspector.isHidden()
    window._select_activity("display")
    assert window.modes_panel.isVisible()


def test_source_controls_restore_without_applying_previous_values(window):
    app = QApplication.instance()
    window._select_activity("adjust")
    window.tiles["left"].set_display_settings(brightness=0.5)
    settle(app)
    window._update_inspector()
    assert window.inspector.brightness.value() == 50
    panel = window.inspector
    panel.source_picker.setCurrentIndex(panel.source_picker.findData("right"))
    QTest.qWait(60)
    assert panel.brightness.value() == 0
    assert window.tiles["right"]._display_settings["brightness"] == 0


@pytest.mark.parametrize("tool", ["roi", "line"])
def test_image_measurements_return_to_pan_and_keep_result(window, tool):
    window._select_activity("measure")
    window._tool_requested(tool)
    canvas = window.tiles["left"].image_canvas
    a = canvas.mapFromScene(QPointF(8, 8))
    b = canvas.mapFromScene(QPointF(24, 24))
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(canvas.viewport(), b)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=b)
    settle(QApplication.instance())
    assert canvas._tool == "pan"
    assert window.inspector.image_tool_buttons["pan"].isChecked()
    assert canvas._selection_rect.isVisible() if tool == "roi" else canvas._selection_line.isVisible()
    assert "ROI" in window.inspector.selection_label.text() if tool == "roi" else "线段" in window.inspector.selection_label.text()


def test_tool_cancel_on_repeat_escape_source_and_frame_change(window):
    window._select_activity("measure")
    window._tool_requested("line")
    window._tool_requested("line")
    assert window.tiles["left"].image_canvas._tool == "pan"
    window._tool_requested("roi")
    window.cancel_current_tool()
    assert window.tiles["left"].image_canvas._tool == "pan"
    window._tool_requested("line")
    window.inspector.source_picker.setCurrentIndex(1)
    assert window.tiles["left"].image_canvas._tool == "pan"
    window._tool_requested("line")
    window.next_sample()
    assert all(t.image_canvas._tool == "pan" for t in window.tiles.values())


@pytest.mark.parametrize("mode,clicks", [("pick", 1), ("measure", 2)])
def test_cloud_one_shot_tools_return_to_rotate(mode, clicks):
    app = QApplication.instance() or QApplication([])
    canvas = PointCloudCanvas()
    if canvas.view is None:
        pytest.skip("OpenGL widget unavailable")
    canvas._highlight = SimpleNamespace(setData=lambda **kwargs: None)
    canvas._pick_point = lambda p: np.array([p.x(), p.y(), 2.0])
    canvas.set_interaction_mode(mode)
    finished = []
    canvas.tool_finished.connect(lambda: finished.append(True))
    for index in range(clicks):
        event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(10 + index, 10),
                            QPointF(10 + index, 10), Qt.MouseButton.LeftButton,
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        canvas.eventFilter(canvas.view, event)
    assert canvas._interaction_mode == "rotate"
    assert finished == [True]
    canvas.close()


def test_repeated_panel_switches_leave_no_animation_or_blank_sidebar(window):
    for _ in range(5):
        window._select_activity("display")
        window._select_activity("adjust")
        window._select_activity("measure")
        window._select_activity("data")
        assert window.project_panel.isVisible()
        assert window.inspector.isHidden()
    assert window._sidebar_animation is None
    assert window._topbar_animation is None
    assert all(t.image_canvas.graphicsEffect() is None for t in window.tiles.values())


def test_slow_image_load_retains_canvas(window):
    tile = window.tiles["left"]
    tile._active_worker_token = 999
    tile._workers[999] = object()
    tile._show_delayed_loading()
    assert tile.stack.currentWidget() is tile.image_canvas
    tile._workers.pop(999)
    tile._active_worker_token = None


def test_preview_updates_coalesce_and_discard_stale_results(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 24), (60, 60, 60)).save(path)
    queued = []
    monkeypatch.setattr(media_tile, "RENDER_POOL", SimpleNamespace(start=lambda w: queued.append(w)))
    tile = MediaTile("left")
    tile._analysis_enabled = False
    tile._current_path = path
    tile._display_result(path, load_image_data(path))
    original = tile.image_canvas._item.pixmap().toImage()
    for brightness in (0.1, 0.2, 0.3, 0.4):
        tile.set_display_settings(brightness=brightness)
    assert len(queued) == 1
    queued.pop(0).run()
    app.processEvents()
    assert len(queued) == 1
    assert tile.image_canvas._item.pixmap().toImage() == original
    queued.pop(0).run()
    app.processEvents()
    assert not tile.is_loading()
    assert tile.image_canvas._item.pixmap().toImage() != original
    assert tile.image_canvas.graphicsEffect() is None
    tile.dispose()
    tile.close()


def test_choice_popup_does_not_rebuild_items_or_navigate_frames(window):
    window.layout_picker.showPopup()
    popup = window.layout_picker._popup
    options = list(popup.option_buttons)
    QTest.keyClick(popup, Qt.Key.Key_Down)
    assert window.current_index == 0
    QTest.keyClick(popup, Qt.Key.Key_Return)
    assert not popup.isVisible()
    window.layout_picker.showPopup()
    assert popup.option_buttons == options
    QTest.keyClick(popup, Qt.Key.Key_Escape)
    assert not popup.isVisible()
