from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from .calibration import CalibrationData, RemapTables, compute_rectify_maps, load_remap_tables
from .media import PointCloudData, project_camera_points
from .settings import DialogCloseButton, DialogHeader, ShadowDialog
from .widgets import ImageCanvas
from .workers import IMAGE_POOL, DifferenceWorker

logger = logging.getLogger(__name__)

# Vendor remap tables are 20 MB per calibration; keep the decoded arrays per
# directory so switching frames does not re-read them.
_REMAP_CACHE: dict[tuple[Path, int], RemapTables | None] = {}


def cached_remap_tables(calibration: CalibrationData) -> RemapTables | None:
    if not calibration.has_remap_tables or calibration.remap_directory is None:
        return None
    directory = calibration.remap_directory
    try:
        stamp = max(int(path.stat().st_mtime_ns) for path in directory.glob("map_*.pfm"))
    except (OSError, ValueError):
        stamp = 0
    key = (directory, stamp)
    if key not in _REMAP_CACHE:
        _REMAP_CACHE.clear()
        _REMAP_CACHE[key] = load_remap_tables(calibration)
    return _REMAP_CACHE[key]


def rasterize_cloud_projection(
    image: np.ndarray,
    uv: np.ndarray,
    colors: np.ndarray,
    *,
    alpha: int = 210,
    block: int = 2,
) -> np.ndarray:
    """Blend projected point colors into an RGB buffer with indexed writes.

    Drawing up to 80k points through per-point QPainter calls blocks the UI;
    indexed numpy writes are orders of magnitude faster and deterministic.
    """
    buffer = np.ascontiguousarray(image).copy()
    height, width = buffer.shape[:2]
    color = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0) * 255.0
    opacity = alpha / 255.0
    for dy in range(block):
        for dx in range(block):
            xx = np.clip(
                np.round(np.asarray(uv[:, 0]) + dx).astype(np.int64),
                0,
                width - 1,
            )
            yy = np.clip(
                np.round(np.asarray(uv[:, 1]) + dy).astype(np.int64),
                0,
                height - 1,
            )
            blended = (
                color * opacity
                + buffer[yy, xx].astype(np.float32) * (1.0 - opacity)
            )
            buffer[yy, xx] = np.clip(blended, 0, 255).astype(np.uint8)
    return buffer


def rectify_stereo_images(
    left: QImage,
    right: QImage,
    calibration: CalibrationData,
) -> tuple[QImage, QImage, str]:
    """Rectify a loaded stereo pair using the active calibration.

    Returns the two rectified images and a short label describing which
    mapping was used: vendor remap tables, the rectified section of the
    calibration, OpenCV ``stereoRectify`` or plain undistortion.
    """
    if calibration.left is None or calibration.right is None:
        raise ValueError("当前标定不包含双目内参")
    import cv2

    left_array = StereoOverlayDialog._rgb_array(left)
    right_array = StereoOverlayDialog._rgb_array(right)
    height, width = left_array.shape[:2]
    if right_array.shape[:2] != (height, width):
        right_array = cv2.resize(
            right_array,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    tables = cached_remap_tables(calibration)
    if tables is not None and tables.mode != calibration.rectify_mode:
        # The vendor tables describe a different optical mode (e.g. underwater).
        tables = None
    if tables is not None and tables.left_x.shape != (height, width):
        logger.info(
            "重映射表尺寸 %s 与图像 %s 不一致，改用计算映射",
            tables.left_x.shape[::-1],
            (width, height),
        )
        tables = None
    if tables is not None:
        left_result = cv2.remap(left_array, tables.left_x, tables.left_y, cv2.INTER_LINEAR)
        right_result = cv2.remap(right_array, tables.right_x, tables.right_y, cv2.INTER_LINEAR)
        method = "重映射表"
    else:
        maps = compute_rectify_maps(calibration, (width, height))
        if maps is not None:
            left_result = cv2.remap(left_array, *maps[0], cv2.INTER_LINEAR)
            right_result = cv2.remap(right_array, *maps[1], cv2.INTER_LINEAR)
            method = (
                "校正参数"
                if calibration.rectified is not None and calibration.rectified.rotation_left is not None
                else "立体校正"
            )
        else:
            left_result = cv2.undistort(
                left_array,
                calibration.left.matrix.astype(np.float64),
                calibration.left.distortion.astype(np.float64),
            )
            right_result = cv2.undistort(
                right_array,
                calibration.right.matrix.astype(np.float64),
                calibration.right.distortion.astype(np.float64),
            )
            method = "仅去畸变"

    def to_qimage(values: np.ndarray) -> QImage:
        values = np.ascontiguousarray(values)
        result_height, result_width, channels = values.shape
        return QImage(
            values.data,
            result_width,
            result_height,
            result_width * channels,
            QImage.Format.Format_RGB888,
        ).copy()

    return to_qimage(left_result), to_qimage(right_result), method


class _InspectionDialog(ShadowDialog):
    """Viewing-only dialog chrome, independent of the removed review workflow."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        header = DialogHeader()
        header.setObjectName("settingsHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 0, 5, 0)
        heading = QLabel(title)
        heading.setObjectName("settingsTitle")
        close = DialogCloseButton()
        close.clicked.connect(self.reject)
        layout.addWidget(heading)
        layout.addStretch(1)
        layout.addWidget(close)
        self.outer.addWidget(header)

    def add_footer(self, text: str, handler) -> None:
        footer = QFrame()
        footer.setObjectName("settingsFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(12, 8, 12, 8)
        close = QPushButton(text)
        close.setObjectName("primaryButton")
        close.clicked.connect(handler)
        layout.addStretch(1)
        layout.addWidget(close)
        self.outer.addWidget(footer)


class StereoOverlayDialog(_InspectionDialog):
    def __init__(
        self,
        left: QImage,
        right: QImage,
        parent: QWidget | None = None,
        *,
        mode: str = "overlay",
    ) -> None:
        self.mode = mode if mode in {"overlay", "difference"} else "overlay"
        super().__init__("左右图绝对差值" if self.mode == "difference" else "左右图叠加", parent)
        self.resize(1040, 720)
        self.setMinimumSize(720, 520)
        self.left = left
        self.right = right
        self._difference_token = 0
        self._blend_timer = QTimer(self)
        self._blend_timer.setSingleShot(True)
        self._blend_timer.setInterval(30)
        self._blend_timer.timeout.connect(self._apply_blend)

        body = QWidget()
        body.setObjectName("settingsPages")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)
        self.loading_label = QLabel("正在计算…")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setObjectName("loadingText")
        self.loading_label.hide()
        layout.addWidget(self.loading_label)

        self.controls = QWidget()
        controls = QHBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        left_label = QLabel("左图")
        left_label.setObjectName("muted")
        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        self.blend_slider.valueChanged.connect(self._update_blend)
        right_label = QLabel("右图")
        right_label.setObjectName("muted")
        self.value_label = QLabel("50%")
        self.value_label.setObjectName("sampleMeta")
        self.value_label.setFixedWidth(42)
        controls.addWidget(left_label)
        controls.addWidget(self.blend_slider, 1)
        controls.addWidget(right_label)
        controls.addWidget(self.value_label)
        layout.addWidget(self.controls)
        self.outer.addWidget(body, 1)
        self.add_footer("关闭", self.accept)
        if self.mode == "difference":
            self.controls.hide()
            self._start_difference()
        else:
            left_array = self._rgb_array(self.left)
            right_image = self.right
            if right_image.size() != self.left.size():
                right_image = right_image.scaled(
                    self.left.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self._left_arr = left_array
            self._right_arr = self._rgb_array(right_image)
            self._update_blend()

    def _update_blend(self) -> None:
        if self.mode == "difference":
            return
        self.value_label.setText(f"{self.blend_slider.value()}%")
        self._blend_timer.start()

    def _apply_blend(self) -> None:
        if self.mode == "difference":
            return
        opacity = self.blend_slider.value() / 100.0
        blended = (
            self._left_arr.astype(np.float32) * (1.0 - opacity)
            + self._right_arr.astype(np.float32) * opacity
        )
        combined = np.ascontiguousarray(np.clip(blended, 0, 255).astype(np.uint8))
        height, width, channels = combined.shape
        image = QImage(
            combined.data,
            width,
            height,
            width * channels,
            QImage.Format.Format_RGB888,
        ).copy()
        self.canvas.set_image(image, preserve_view=True)

    def _start_difference(self) -> None:
        self._difference_token += 1
        token = self._difference_token
        self.loading_label.show()
        self.loading_label.setText("正在计算差值…")
        worker = DifferenceWorker(
            token,
            self.left,
            self.right,
            StereoOverlayDialog._rgb_array,
        )
        self._difference_worker = worker

        def on_finished(result_token: int, image: object) -> None:
            if result_token != self._difference_token:
                return
            self._difference_worker = None
            self.loading_label.hide()
            self.canvas.set_image(image)

        def on_failed(result_token: int, error: str) -> None:
            if result_token != self._difference_token:
                return
            self._difference_worker = None
            self.loading_label.setText(f"差值计算失败：{error}")

        worker.signals.finished.connect(on_finished)
        worker.signals.failed.connect(on_failed)
        IMAGE_POOL.start(worker)

    def done(self, result: int) -> None:
        self._blend_timer.stop()
        self._difference_token += 1
        super().done(result)

    @staticmethod
    def _rgb_array(image: QImage) -> np.ndarray:
        converted = image.convertToFormat(QImage.Format.Format_RGB888)
        view = np.frombuffer(converted.constBits(), dtype=np.uint8)
        rows = view.reshape(converted.height(), converted.bytesPerLine())
        return rows[:, : converted.width() * 3].reshape(
            converted.height(),
            converted.width(),
            3,
        ).copy()

class CloudProjectionDialog(_InspectionDialog):
    def __init__(
        self,
        image: QImage,
        cloud: PointCloudData,
        intrinsics: np.ndarray,
        cam_offset: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("点云投影到左图", parent)
        self.resize(1040, 720)
        self.setMinimumSize(720, 520)
        body = QWidget()
        body.setObjectName("settingsPages")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)
        self.outer.addWidget(body, 1)
        self.add_footer("关闭", self.accept)

        combined = image.convertToFormat(QImage.Format.Format_RGB888)
        base_array = StereoOverlayDialog._rgb_array(combined)
        uv, depth = project_camera_points(cloud.points, intrinsics, cam_offset)
        width, height = combined.width(), combined.height()
        valid = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(depth)
            & (depth > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        indices = np.flatnonzero(valid)
        if len(indices) > 80_000:
            indices = indices[
                np.linspace(0, len(indices) - 1, 80_000, dtype=np.int64)
            ]
        raster = rasterize_cloud_projection(
            base_array,
            uv[indices],
            cloud.colors[indices, :3],
        )
        height, width, _channels = raster.shape
        projection = QImage(
            raster.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        self.canvas.set_image(projection)
