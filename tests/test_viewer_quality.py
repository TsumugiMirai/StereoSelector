"""Viewer-only regressions for source changes and asynchronous recovery."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from stereo_selector import media_tile, workers
from stereo_selector.inspector import HistogramWidget, InspectorPanel
from stereo_selector.media import ImageData, PointCloudData, analyze_image, load_image_data
from stereo_selector.media_tile import MediaTile


@pytest.fixture
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


class HeldPool:
    def __init__(self):
        self.queued = []

    def start(self, worker, priority=0):
        self.queued.append(worker)

    def tryTake(self, worker):
        # Simulate a running job that cannot be removed from the pool.
        return False


def test_rebuilt_sources_discard_pending_edits_and_roi(app):
    panel = InspectorPanel()
    panel.set_sources(["left", "right"], "left")
    panel.set_ready(True)
    panel.brightness.setValue(34)
    panel._clip_timer.start()
    previous_token = panel._roi_token
    panel.selection_label.setText("old result")
    panel.set_sources(["right"], "right")
    assert not panel._update_timer.isActive()
    assert not panel._clip_timer.isActive()
    assert panel._roi_token > previous_token
    assert not panel._ready
    assert "old result" not in panel.selection_label.text()
    panel._roi_ready(previous_token, analyze_image(np.ones((2, 2))))
    assert "ROI ·" not in panel.selection_label.text()
    panel.close()


def test_unchanged_source_list_keeps_current_choice(app):
    panel = InspectorPanel()
    panel.set_sources(["left", "right"], "right")
    panel.set_sources(["left", "right", "depth_fsd"])
    assert panel._source == "right"
    assert panel.source_picker.currentData() == "right"
    panel.close()


def test_reset_only_brightness_updates_preview_and_label(app):
    panel = InspectorPanel()
    panel.set_sources(["left"], "left")
    panel.set_ready(True)
    panel._update_timer.stop()
    panel.set_display_controls({"brightness": 0.6})
    changes = []
    panel.display_changed.connect(changes.append)
    panel.reset_display_controls()
    QTest.qWait(80)
    assert panel.brightness.value_label.text() == "0"
    assert len(changes) == 1
    assert changes[0]["brightness"] == 0
    panel.close()


def test_scalar_controls_remain_visible_when_task_changes(app):
    panel = InspectorPanel()
    panel.set_sources(["image"], "image")
    data = ImageData(QImage(8, 6, QImage.Format.Format_Grayscale8), np.ones((6, 8), dtype=np.uint16))
    panel.set_image_data("image", data, Path("numeric.png"))
    panel.set_category("statistics")
    panel.set_category("adjust")
    assert not panel.depth_controls.isHidden()
    assert "8 × 6" in panel.file_label.text()
    panel.close()


def test_empty_filtered_cloud_is_not_reported_as_loading(app):
    panel = InspectorPanel()
    cloud = PointCloudData(np.empty((0, 3)), np.empty((0, 4)), 10, 0, 10.0)
    panel.set_cloud_data(cloud, Path("empty.ply"))
    assert panel.cloud_stats.text() == "当前范围内无点"
    panel.close()


def test_histogram_ignores_empty_channels(app):
    widget = HistogramWidget()
    widget.set_histograms((np.array([]), np.array([])))
    assert widget._histograms == ()
    widget.resize(300, 150)
    assert not widget.grab().isNull()
    widget.close()


def test_fast_return_to_cancelled_load_starts_fresh_worker(app, monkeypatch, tmp_path):
    pool = HeldPool()
    monkeypatch.setattr(media_tile, "load_pool_for", lambda _modality: pool)
    tile = MediaTile("left")
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    tile.show_file(first)
    cancelled_token = tile._active_worker_token
    tile.show_file(second)
    assert tile._workers[cancelled_token]._cancelled
    tile.show_file(first)
    assert tile._active_worker_token != cancelled_token
    assert not tile._workers[tile._active_worker_token]._cancelled
    tile.dispose()


def test_cancelled_worker_does_not_decode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(workers, "load_image_data", lambda path: calls.append(path))
    worker = workers.MediaLoadWorker(1, "left", tmp_path / "a.png", 100)
    worker.cancel()
    worker.run()
    assert calls == []


def test_cloud_cache_identity_includes_extrinsics(app, tmp_path):
    # Avoid initializing an OpenGL viewport: cache identity only needs modality.
    tile = MediaTile("left")
    tile.modality = "ply"
    path = tmp_path / "a.ply"
    original = tile._cache_key(path)
    tile.cloud_rotation = np.eye(3)
    rotated = tile._cache_key(path)
    tile.cloud_translation = np.array([0.1, 0.0, 0.0])
    assert len({original, rotated, tile._cache_key(path)}) == 3
    tile.dispose()


def test_failed_first_preview_recovers_canvas_and_ready_signal(app, tmp_path):
    tile = MediaTile("left")
    image = QImage(12, 8, QImage.Format.Format_RGB888)
    image.fill(0)
    tile._current_path = tmp_path / "frame.png"
    tile.image_data = ImageData(image, np.zeros((8, 12, 3), np.uint8))
    tile._preview_sequence = 3
    tile._preview_pending = (3, tile.image_data.values, {}, False)
    tile.stack.setCurrentWidget(tile.loading_page)
    ready = []
    tile.data_ready.connect(ready.append)
    tile._preview_failed(3, "bad display parameter")
    assert tile.showing_canvas()
    assert not tile.is_loading()
    assert ready == ["left"]
    assert "bad display parameter" in tile.meta.toolTip()
    tile.dispose()


def test_preview_metadata_keeps_original_dimensions(app, tmp_path):
    tile = MediaTile("left")
    image = QImage(4, 3, QImage.Format.Format_RGB888)
    tile._current_path = tmp_path / "frame.png"
    tile.image_data = ImageData(image, np.zeros((60, 80, 3), np.uint8))
    tile._present_image(image, preserve_view=False)
    assert "80×60" in tile._meta_text
    tile.dispose()


def test_analysis_result_cannot_contaminate_changed_file_cache(app, monkeypatch, tmp_path):
    pool = HeldPool()
    monkeypatch.setattr(media_tile, "ANALYSIS_POOL", pool)
    path = tmp_path / "changing.png"
    Image.new("RGB", (4, 3), (20, 20, 20)).save(path)
    old_data = load_image_data(path)
    tile = MediaTile("left")
    tile._current_path = path
    old_key = tile._cache_key(path)
    tile._remember(old_key, old_data)
    tile._start_analysis(path, old_data)
    token = tile._active_analysis_token
    # The file can be replaced by a capture process while analysis runs.
    Image.new("RGB", (8, 6), (90, 90, 90)).save(path)
    os.utime(path, ns=(path.stat().st_atime_ns, old_key[1] + 1_000_000_000))
    new_data = load_image_data(path)
    new_key = tile._cache_key(path)
    tile._remember(new_key, new_data)
    tile.image_data = new_data
    tile._analysis_finished(token, path, analyze_image(old_data.values), None)
    assert tile._cache[new_key].image_stats is None
    assert tile.image_data is new_data
    tile.dispose()


def test_narrow_inspector_preserves_full_path_and_original_metadata(app):
    panel = InspectorPanel()
    panel.set_sources(["left"], "left")
    path = Path("C:/capture/projects/very-long-project-name/left/full-resolution-image-00001234.png")
    data = ImageData(QImage(4, 3, QImage.Format.Format_RGB888),
                     np.zeros((60, 80, 3), np.uint8),
                     analyze_image(np.zeros((60, 80, 3), np.uint8)))
    panel.set_image_data("left", data, path)
    panel.resize(268, 720)
    panel.show()
    app.processEvents()
    visible_path = panel.file_label.text().splitlines()[0]
    assert "…" in visible_path
    assert panel.file_label.fontMetrics().horizontalAdvance(visible_path) <= panel.file_label.contentsRect().width()
    assert "80 × 60 · uint8" in panel.file_label.text()
    assert str(path) in panel.file_label.toolTip()
    panel.copy_path_button.click()
    assert QApplication.clipboard().text() == str(path)
    panel.set_image_data("left", None, None)
    assert not panel.copy_path_button.isEnabled()
    assert panel.file_label.toolTip() == ""
    panel.close()
