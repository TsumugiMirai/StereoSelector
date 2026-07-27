from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .review import _BaseDialog
from .widgets import ImageCanvas


class StereoOverlayDialog(_BaseDialog):
    def __init__(self, left: QImage, right: QImage, parent: QWidget | None = None) -> None:
        super().__init__("左右图叠加", parent)
        self.resize(1040, 720)
        self.setMinimumSize(720, 520)
        self.left = left
        self.right = right

        body = QWidget()
        body.setObjectName("settingsPages")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(10)
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
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
        layout.addLayout(controls)
        self.outer.addWidget(body, 1)
        self.add_footer("关闭", self.accept)
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
