from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtGui import QColor, QImage

import numpy as np

from stereo_selector.calibration import CalibrationData, CameraCalibration
from stereo_selector.inspection import StereoOverlayDialog, rectify_stereo_images
from stereo_selector.models import DatasetScanner, copy_sample
from stereo_selector.review import AnnotationDialog, MANIFEST_FILENAME, OutputSettingsDialog, ReviewStore


def _make_dataset(tmp_path: Path):
    root = tmp_path / "capture"
    for folder in ("left", "right"):
        path = root / folder / "frame_0001.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(folder.encode("ascii"))
    output = tmp_path / "exports" / "capture_review"
    return DatasetScanner().scan(root, output_root=output)


def test_review_store_persists_annotations_and_acceptance(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    sample = dataset.samples[0]
    store = ReviewStore(dataset)

    assert store.reconcile() == set()
    store.set_annotation(sample, ["模糊", "自定义"], "左上角需要复查")
    copy_sample(dataset, sample)
    store.set_status(sample, "accepted")

    manifest = dataset.output_root / MANIFEST_FILENAME
    document = json.loads(manifest.read_text(encoding="utf-8"))
    record = document["samples"][sample.key]
    assert document["project"]["source_root"] == str(dataset.root)
    assert document["project"]["output_root"] == str(dataset.output_root)
    assert record["status"] == "accepted"
    assert record["defect_tags"] == ["模糊", "自定义"]
    assert record["note"] == "左上角需要复查"
    assert ReviewStore(dataset).reconcile() == {sample.key}


def test_rejected_status_remains_authoritative_when_old_export_files_exist(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    sample = dataset.samples[0]
    store = ReviewStore(dataset)
    copy_sample(dataset, sample)
    store.set_status(sample, "rejected")

    reopened = ReviewStore(dataset)

    assert reopened.reconcile() == set()
    assert reopened.get(sample)["status"] == "rejected"


def test_output_settings_dialog_builds_custom_path(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = tmp_path / "capture"
    project.mkdir()
    output_parent = tmp_path / "exports"
    output_parent.mkdir()
    dialog = OutputSettingsDialog(project, tmp_path / "capture_select")
    dialog.parent_edit.setText(str(output_parent))
    dialog.name_edit.setText("人工筛选结果")

    dialog._validate_and_accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.output_root == (output_parent / "人工筛选结果").resolve()
    dialog.close()
    assert app is not None


def test_annotation_dialog_supports_builtin_and_custom_tags(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    dataset = _make_dataset(tmp_path)
    dialog = AnnotationDialog(
        dataset.samples[0],
        ["模糊", "纹理重复"],
        "检查深度边缘",
    )

    assert dialog.tag_checks["模糊"].isChecked()
    assert dialog.selected_tags() == ["模糊", "纹理重复"]
    assert dialog.note() == "检查深度边缘"
    dialog.close()
    assert app is not None


def test_stereo_overlay_slider_blends_loaded_images() -> None:
    app = QApplication.instance() or QApplication([])
    left = QImage(2, 2, QImage.Format.Format_RGB32)
    right = QImage(2, 2, QImage.Format.Format_RGB32)
    left.fill(QColor("red"))
    right.fill(QColor("blue"))
    dialog = StereoOverlayDialog(left, right)
    margins = dialog._window_layout.contentsMargins()
    assert min(margins.left(), margins.top(), margins.right(), margins.bottom()) >= 15
    assert dialog.dialog_surface.objectName() == "dialogSurface"

    dialog.blend_slider.setValue(0)
    left_pixel = dialog.canvas._item.pixmap().toImage().pixelColor(0, 0)
    dialog.blend_slider.setValue(100)
    right_pixel = dialog.canvas._item.pixmap().toImage().pixelColor(0, 0)

    assert left_pixel.red() > left_pixel.blue()
    assert right_pixel.blue() > right_pixel.red()
    dialog.close()
    assert app is not None


def test_stereo_difference_view_uses_absolute_rgb_difference() -> None:
    app = QApplication.instance() or QApplication([])
    left = QImage(2, 2, QImage.Format.Format_RGB32)
    right = QImage(2, 2, QImage.Format.Format_RGB32)
    left.fill(QColor(200, 20, 10))
    right.fill(QColor(50, 30, 40))

    dialog = StereoOverlayDialog(left, right, mode="difference")
    pixel = dialog.canvas._item.pixmap().toImage().pixelColor(0, 0)

    assert pixel.getRgb()[:3] == (150, 10, 30)
    assert dialog.controls.isHidden()
    dialog.close()
    assert app is not None


def test_rectified_toggle_preserves_zero_distortion_pair() -> None:
    left = QImage(8, 6, QImage.Format.Format_RGB32)
    right = QImage(8, 6, QImage.Format.Format_RGB32)
    left.fill(QColor(20, 40, 60))
    right.fill(QColor(80, 100, 120))
    camera = CameraCalibration(
        np.array([[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        8,
        6,
    )
    calibration = CalibrationData(
        Path("test.json"),
        camera,
        camera,
        None,
        None,
        None,
        None,
        None,
    )

    rectified_left, rectified_right = rectify_stereo_images(
        left,
        right,
        calibration,
    )

    assert rectified_left.size() == left.size()
    assert rectified_right.size() == right.size()
    assert rectified_left.pixelColor(4, 3).getRgb()[:3] == (20, 40, 60)
    assert rectified_right.pixelColor(4, 3).getRgb()[:3] == (80, 100, 120)
