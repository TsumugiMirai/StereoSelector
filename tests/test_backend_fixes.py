from __future__ import annotations

import json
import os
import struct
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from stereo_selector import workers
from stereo_selector.calibration import load_calibration
from stereo_selector.inspection import rasterize_cloud_projection
from stereo_selector.media import downsample_pool
from stereo_selector.media.plyio import read_ply_vertices
from stereo_selector.models import DatasetScanner


def _make_project(root: Path) -> Path:
    for index in (1, 2):
        for folder in ("left", "right"):
            path = root / folder / f"frame_{index:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), "gray").save(path)
    return root


def test_rasterize_cloud_projection_blends_points() -> None:
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    uv = np.array([[1.0, 1.0]], dtype=np.float64)
    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    raster = rasterize_cloud_projection(image, uv, colors, alpha=255, block=1)

    assert raster[1, 1].tolist() == [255, 0, 0]
    assert raster[0, 0].tolist() == [255, 255, 255]

    raster_block = rasterize_cloud_projection(image, uv, colors, alpha=255, block=2)
    assert raster_block[2, 2].tolist() == [255, 0, 0]


def test_analysis_worker_failure_emits_failed_signal(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    failed: list[str] = []

    def boom(values):
        raise MemoryError("模拟统计失败")

    monkeypatch.setattr(workers, "analyze_image", boom)
    worker = workers.MediaAnalysisWorker(
        1,
        np.zeros((4, 4), dtype=np.uint16),
        include_depth=True,
    )
    worker.signals.failed.connect(lambda token, error: failed.append(error))
    worker.run()

    assert len(failed) == 1
    assert "统计失败" in failed[0]
    assert app is not None


def test_scan_reports_progress_and_supports_cancel(tmp_path: Path) -> None:
    _make_project(tmp_path / "capture")
    progress_calls: list[tuple[str, int, int]] = []
    scanner = DatasetScanner()

    dataset = scanner.scan(
        tmp_path / "capture",
        progress=lambda text, done, total: progress_calls.append((text, done, total)),
    )
    assert len(dataset.samples) == 2
    assert any("扫描" in text or "索引" in text for text, _done, _total in progress_calls)

    cancelled = False
    try:
        scanner.scan(tmp_path / "capture", cancel_check=lambda: True)
    except InterruptedError:
        cancelled = True
    assert cancelled


def test_downsample_pool_keeps_aspect_ratio_and_averages() -> None:
    values = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7)
    pooled = downsample_pool(values, 2)
    assert pooled.shape == (3, 4)
    assert pooled[0, 0] == np.mean(values[0:2, 0:2])
    assert downsample_pool(values, 1) is values


def test_configured_pools_split_by_task_type() -> None:
    workers.configure_pools()
    assert workers.IMAGE_POOL.maxThreadCount() == 2
    assert workers.CLOUD_POOL.maxThreadCount() == 2
    assert workers.ANALYSIS_POOL.maxThreadCount() == 1
    assert workers.SCAN_POOL.maxThreadCount() == 1


def test_application_entry_points_are_importable() -> None:
    from stereo_selector import app

    assert callable(app.run_application)
    assert callable(app.build_parser)
    assert callable(app.start_smoke_test)
    assert callable(app.main)


def test_calibration_singular_matrix_raises_friendly_error(tmp_path: Path) -> None:
    path = tmp_path / "singular.json"
    path.write_text(
        json.dumps(
            {
                "left": {
                    "camera_matrix": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                },
                "right": {
                    "camera_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
                "left_to_right": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation": [-0.12, 0.0, 0.0],
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        load_calibration(path)
    except ValueError as exc:
        assert "不可逆" in str(exc)
    else:
        raise AssertionError("expected ValueError for singular camera matrix")


def test_read_ply_vertices_samples_large_binary_cloud(tmp_path: Path) -> None:
    path = tmp_path / "cloud.ply"
    count = 100
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "end_header\n"
    ).encode("ascii")
    payload = b"".join(
        struct.pack("<ddd", float(index), float(index), float(index))
        for index in range(count)
    )
    path.write_bytes(header + payload)

    vertices, total = read_ply_vertices(path, sample_target=10)

    assert total == count
    assert len(vertices) == 10
    assert vertices[0]["x"] == 0.0
    assert vertices[-1]["x"] == 90.0
