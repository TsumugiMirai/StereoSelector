from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from stereo_selector import media_tile
from stereo_selector.media import ImageData, render_image_values
from stereo_selector.media_tile import MediaTile
from stereo_selector.widgets import ImageCanvas


@pytest.fixture
def render_queue(monkeypatch):
    app = QApplication.instance() or QApplication([])
    queued = []
    monkeypatch.setattr(media_tile, "RENDER_POOL", SimpleNamespace(start=queued.append))
    return app, queued


def _run_next(app, queued):
    worker = queued.pop(0)
    worker.run()
    app.processEvents()
    return worker


def _large_tile(tmp_path, app, queued, modality="depth_fsd"):
    if modality == "depth_fsd":
        values = np.broadcast_to(np.arange(1, 4201, dtype=np.uint16), (120, 4200)).copy()
    else:
        values = np.full((120, 4200, 3), (40, 100, 180), dtype=np.uint8)
    tile = MediaTile(modality)
    tile._analysis_enabled = False
    tile.resize(700, 480)
    tile.show()
    tile._current_path = tmp_path / "first.png"
    tile._display_settings["brightness"] = 0.1
    tile._display_result(tile.current_path, ImageData(render_image_values(values), values))
    assert queued[0].scale > 1
    _run_next(app, queued)
    return tile


@pytest.mark.parametrize("modality", ["depth_fsd", "left"])
def test_actual_size_renders_original_pixels_off_thread(tmp_path, render_queue, modality):
    app, queued = render_queue
    tile = _large_tile(tmp_path, app, queued, modality)
    try:
        old_image = tile.image_canvas._item.pixmap().toImage()
        assert old_image.width() < tile.image_data.values.shape[1]
        tile.set_fit_mode("actual")
        assert len(queued) == 1
        assert queued[0].scale == 1
        assert tile.is_loading()
        assert tile.image_canvas._item.pixmap().toImage() == old_image
        _run_next(app, queued)
        rendered = tile.image_canvas._item.pixmap().toImage()
        assert rendered.width() == 4200
        assert rendered.height() == 120
        assert rendered.convertToFormat(QImage.Format.Format_RGB888) == render_image_values(
            tile.image_data.values, **tile._display_settings
        )
        assert tile.image_canvas.transform().m11() == 1.0
        assert tile._presented_full_resolution
        assert not tile.is_loading()
    finally:
        tile.dispose()
        tile.close()


@pytest.mark.parametrize("mode", ["fit", "width"])
def test_new_fit_request_supersedes_pending_actual_size(tmp_path, render_queue, mode):
    app, queued = render_queue
    tile = _large_tile(tmp_path, app, queued)
    try:
        old_image = tile.image_canvas._item.pixmap().toImage()
        tile.set_fit_mode("actual")
        tile.set_fit_mode(mode)
        assert len(queued) == 1
        _run_next(app, queued)
        assert tile.image_canvas._item.pixmap().toImage() == old_image
        assert queued[0].scale > 1
        _run_next(app, queued)
        assert tile._fit_mode == mode
        assert not tile._presented_full_resolution
        assert tile.image_canvas.transform().m11() != 1.0
        if mode == "fit":
            assert not tile.image_canvas._user_zoomed
        else:
            bounds = tile.image_canvas._item.boundingRect()
            width = bounds.width() * tile.image_canvas.transform().m11()
            assert abs(width - tile.image_canvas.viewport().width()) <= 20
    finally:
        tile.dispose()
        tile.close()


def test_frame_change_discards_pending_full_resolution(tmp_path, render_queue):
    app, queued = render_queue
    tile = _large_tile(tmp_path, app, queued)
    try:
        tile.set_fit_mode("actual")
        new_path = tmp_path / "second.png"
        values = np.full((90, 3600), 1500, dtype=np.uint16)
        tile._remember(tile._cache_key(new_path), ImageData(render_image_values(values), values))
        tile.show_file(new_path)
        _run_next(app, queued)
        assert queued[0].scale > 1
        _run_next(app, queued)
        assert tile.current_path == new_path
        assert tile.image_data.values is values
        assert tile.image_canvas._item.pixmap().width() == 1800
        assert tile._fit_mode == "fit"
        assert not tile.image_canvas._user_zoomed
    finally:
        tile.dispose()
        tile.close()


def test_adjustment_during_actual_size_keeps_latest_parameters_and_view(tmp_path, render_queue):
    app, queued = render_queue
    tile = _large_tile(tmp_path, app, queued)
    try:
        tile.set_fit_mode("actual")
        tile.set_display_settings(brightness=0.3)
        _run_next(app, queued)
        assert queued[0].scale == 1
        _run_next(app, queued)
        assert tile.image_canvas.transform().m11() == 1.0
        assert tile.image_canvas._item.pixmap().toImage().convertToFormat(
            QImage.Format.Format_RGB888
        ) == render_image_values(
            tile.image_data.values, **tile._display_settings
        )
        tile.image_canvas.centerOn(2700, 60)
        center_before = tile.image_canvas.mapToScene(tile.image_canvas.viewport().rect().center())
        tile.set_display_settings(gamma=1.5)
        _run_next(app, queued)
        assert tile.image_canvas.transform().m11() == 1.0
        center_after = tile.image_canvas.mapToScene(tile.image_canvas.viewport().rect().center())
        assert abs(center_before.x() - center_after.x()) < 2
        tile.set_fit_mode("width")
        assert not queued
        assert tile.image_canvas.transform().m11() < 1.0
        tile.set_fit_mode("actual")
        assert not queued
        assert tile.image_canvas.transform().m11() == 1.0
    finally:
        tile.dispose()
        tile.close()


def test_reset_cancels_pending_actual_size(tmp_path, render_queue):
    app, queued = render_queue
    tile = _large_tile(tmp_path, app, queued)
    try:
        tile.set_fit_mode("actual")
        tile.reset_view()
        _run_next(app, queued)
        _run_next(app, queued)
        assert tile._fit_mode == "fit"
        assert not tile.image_canvas._user_zoomed
    finally:
        tile.dispose()
        tile.close()


def test_canvas_keeps_source_region_when_preview_resolution_changes(render_queue):
    app, _queued = render_queue
    canvas = ImageCanvas()
    canvas.resize(500, 300)
    canvas.show()
    image = QImage(800, 600, QImage.Format.Format_RGB888)
    image.fill(0)
    try:
        canvas.set_image(image)
        app.processEvents()
        canvas.show_actual_size()
        assert canvas._zoom_factor > 1.0
        assert canvas.transform().m11() == 1.0
        canvas.set_image(image.scaled(400, 300), preserve_view=True)
        assert canvas.transform().m11() == 2.0
        canvas.set_image(image, preserve_view=True)
        assert canvas.transform().m11() == 1.0
    finally:
        canvas.close()
