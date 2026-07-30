from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .review import _BaseDialog
from .media import PointCloudData, project_camera_points
from .calibration import CalibrationData
from .widgets import ImageCanvas


def rectify_stereo_images(
    left: QImage,
    right: QImage,
    calibration: CalibrationData,
) -> tuple[QImage, QImage]:
    """Rectify a loaded stereo pair using the active calibration."""
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
    left_matrix = calibration.left.matrix.astype(np.float64)
    right_matrix = calibration.right.matrix.astype(np.float64)
    left_distortion = calibration.left.distortion.astype(np.float64)
    right_distortion = calibration.right.distortion.astype(np.float64)
    if calibration.rotation is not None and calibration.translation is not None:
        rotation = calibration.rotation.astype(np.float64).reshape(3, 3)
        translation = calibration.translation.astype(np.float64).reshape(3)
        left_rectification, right_rectification, left_projection, right_projection, *_ = (
            cv2.stereoRectify(
                left_matrix,
                left_distortion,
                right_matrix,
                right_distortion,
                (width, height),
                rotation,
                translation,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
            )
        )
        left_map = cv2.initUndistortRectifyMap(
            left_matrix,
            left_distortion,
            left_rectification,
            left_projection,
            (width, height),
            cv2.CV_32FC1,
        )
        right_map = cv2.initUndistortRectifyMap(
            right_matrix,
            right_distortion,
            right_rectification,
            right_projection,
            (width, height),
            cv2.CV_32FC1,
        )
        left_result = cv2.remap(left_array, *left_map, cv2.INTER_LINEAR)
        right_result = cv2.remap(right_array, *right_map, cv2.INTER_LINEAR)
    else:
        left_result = cv2.undistort(
            left_array,
            left_matrix,
            left_distortion,
        )
        right_result = cv2.undistort(
            right_array,
            right_matrix,
            right_distortion,
        )

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

    return to_qimage(left_result), to_qimage(right_result)


class StereoOverlayDialog(_BaseDialog):
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

        body = QWidget()
        body.setObjectName("settingsPages")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)

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
            self._show_difference()
        else:
            self._update_blend()

    def _update_blend(self) -> None:
        opacity = self.blend_slider.value() / 100.0
        self.value_label.setText(f"{self.blend_slider.value()}%")
        target_size = self.left.size()
        right = self.right
        if right.size() != target_size:
            right = right.scaled(
                target_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        combined = QImage(target_size, QImage.Format.Format_RGB32)
        combined.fill(Qt.GlobalColor.black)
        painter = QPainter(combined)
        painter.drawImage(0, 0, self.left)
        painter.setOpacity(opacity)
        painter.drawImage(0, 0, right)
        painter.end()
        self.canvas.set_image(combined)

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

    def _show_difference(self) -> None:
        right = self.right
        if right.size() != self.left.size():
            right = right.scaled(
                self.left.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        difference = np.abs(
            self._rgb_array(self.left).astype(np.int16)
            - self._rgb_array(right).astype(np.int16)
        ).astype(np.uint8)
        height, width, channels = difference.shape
        image = QImage(
            difference.data,
            width,
            height,
            width * channels,
            QImage.Format.Format_RGB888,
        ).copy()
        self.canvas.set_image(image)


class CloudProjectionDialog(_BaseDialog):
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

        combined = image.convertToFormat(QImage.Format.Format_RGB32)
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
        painter = QPainter(combined)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        colors = np.clip(cloud.colors[indices, :3] * 255.0, 0, 255).astype(np.uint8)
        for index, color in zip(indices, colors):
            painter.setPen(QPen(QColor(int(color[0]), int(color[1]), int(color[2]), 210), 2))
            painter.drawPoint(int(round(uv[index, 0])), int(round(uv[index, 1])))
        painter.end()
        self.canvas.set_image(combined)
