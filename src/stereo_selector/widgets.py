from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QMatrix4x4,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QVector3D,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRubberBand,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .media import (
    PointCloudData,
    colorize_scalar,
    project_camera_points,
    resolve_camera_model,
)
from .ui_metrics import (
    ICON_ACTIVITY_SIZE,
    ICON_PANE_SIZE,
    ICON_PLAYBACK_SIZE,
    ICON_SPINNER_SIZE,
    ICON_TOOL_SIZE,
    ICON_WINDOW_HEIGHT,
    ICON_WINDOW_WIDTH,
    TITLE_BAR_HEIGHT,
)

logger = logging.getLogger(__name__)


MODALITY_ACCENTS = {
    "left": "#4ea1ff",
    "right": "#a78bfa",
    "depth_fsd": "#f0a35e",
    "depth_color": "#e76f9b",
    "ply": "#56c596",
}

# Camera-space X is horizontal, Y is vertical (down), and Z is distance from
# the lens. Rendering maps this to OpenGL X right, Y forward, Z up. A camera at
# negative OpenGL Y therefore looks along the camera's positive Z axis.
CAMERA_ALIGNED_AZIMUTH = -90.0
CAMERA_ALIGNED_ELEVATION = 0.0


def camera_points_to_gl(points: np.ndarray) -> np.ndarray:
    """Map camera coordinates (X right, Y down, Z forward) to OpenGL axes."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return np.column_stack((points[:, 0], points[:, 2], -points[:, 1])).astype(
        np.float32,
        copy=False,
    )


class PinholeProjectionMixin:
    """Use left-camera intrinsics instead of GLViewWidget's symmetric FOV."""

    def configure_camera_projection(
        self,
        intrinsics: np.ndarray,
        image_size: tuple[int, int],
        far_distance: float,
    ) -> None:
        self._camera_intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        self._camera_image_size = (int(image_size[0]), int(image_size[1]))
        self._camera_far_distance = max(float(far_distance), 1.0)
        self.update()

    def projectionMatrix(self, region, viewport):
        matrix = getattr(self, "_camera_intrinsics", None)
        image_size = getattr(self, "_camera_image_size", None)
        if matrix is None or image_size is None:
            return super().projectionMatrix(region, viewport)
        x0, y0, width, height = (float(value) for value in viewport)
        if width <= 0.0 or height <= 0.0:
            return super().projectionMatrix(region, viewport)

        image_width, image_height = image_size
        scale = min(width / image_width, height / image_height)
        content_width = image_width * scale
        content_height = image_height * scale
        inset_x = (width - content_width) / 2.0
        inset_y = (height - content_height) / 2.0
        fx = float(matrix[0, 0]) * scale
        fy = float(matrix[1, 1]) * scale
        cx = inset_x + float(matrix[0, 2]) * scale
        cy = inset_y + float(matrix[1, 2]) * scale

        near_clip = 0.001
        far_clip = self._camera_far_distance
        full_left = -cx * near_clip / fx
        full_right = (width - cx) * near_clip / fx
        full_bottom = -(height - cy) * near_clip / fy
        full_top = cy * near_clip / fy
        span_x = full_right - full_left
        span_y = full_top - full_bottom
        region_x, region_y, region_width, region_height = (
            float(value) for value in region
        )
        left = full_left + ((region_x - x0) / width) * span_x
        right = full_left + ((region_x + region_width - x0) / width) * span_x
        bottom = full_bottom + ((region_y - y0) / height) * span_y
        top = full_bottom + ((region_y + region_height - y0) / height) * span_y
        projection = QMatrix4x4()
        projection.frustum(left, right, bottom, top, near_clip, far_clip)
        return projection


class LoadingSpinner(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(ICON_SPINNER_SIZE, ICON_SPINNER_SIZE)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance)

    def start(self) -> None:
        self._angle = 0
        self._timer.start()
        self.show()

    def stop(self) -> None:
        self._timer.stop()

    def _advance(self) -> None:
        self._angle = (self._angle + 24) % 360
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#4c9ffe"), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.drawArc(rect, (90 - self._angle) * 16, -250 * 16)


class WindowControlButton(QPushButton):
    """Paint consistent window controls without relying on font glyphs."""

    def __init__(self, control: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.control = control
        self.maximized = False
        self.setObjectName("closeWindowButton" if control == "close" else "windowControlButton")

    def set_maximized(self, maximized: bool) -> None:
        self.maximized = maximized
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().buttonText().color()
        pen = QPen(color, 1.25)
        pen.setCosmetic(True)
        painter.setPen(pen)
        center_x, center_y = self.width() / 2, self.height() / 2
        if self.control == "minimize":
            painter.drawLine(int(center_x - 5), int(center_y + 3), int(center_x + 5), int(center_y + 3))
        elif self.control == "maximize":
            if self.maximized:
                painter.drawRect(int(center_x - 4), int(center_y - 3), 8, 7)
                painter.drawRect(int(center_x - 2), int(center_y - 5), 8, 7)
            else:
                painter.drawRect(int(center_x - 5), int(center_y - 5), 10, 10)
        else:
            painter.drawLine(int(center_x - 5), int(center_y - 5), int(center_x + 5), int(center_y + 5))
            painter.drawLine(int(center_x + 5), int(center_y - 5), int(center_x - 5), int(center_y + 5))


class PaneToggleButton(QPushButton):
    """Theme-owned chevron used for collapsible editor panes."""

    def __init__(
        self,
        direction: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._direction = direction
        self.setObjectName("sideToggleButton")
        self.setFixedSize(ICON_PANE_SIZE, ICON_PANE_SIZE)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_direction(self, direction: str) -> None:
        if direction != self._direction:
            self._direction = direction
            self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.palette().buttonText().color(), 1.5)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        x, y = self.width() / 2.0, self.height() / 2.0
        if self._direction == "left":
            points = (QPointF(x + 2, y - 4), QPointF(x - 2, y), QPointF(x + 2, y + 4))
        elif self._direction == "right":
            points = (QPointF(x - 2, y - 4), QPointF(x + 2, y), QPointF(x - 2, y + 4))
        elif self._direction == "up":
            points = (QPointF(x - 4, y + 2), QPointF(x, y - 2), QPointF(x + 4, y + 2))
        else:
            points = (QPointF(x - 4, y - 2), QPointF(x, y + 2), QPointF(x + 4, y - 2))
        painter.drawLine(points[0], points[1])
        painter.drawLine(points[1], points[2])


class PlaybackControlButton(QPushButton):
    """Font-independent media control for the frame timeline."""

    def __init__(
        self,
        control: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.control = control
        self.playing = False
        self.setObjectName("playbackButton")
        self.setFixedSize(ICON_PLAYBACK_SIZE, ICON_PLAYBACK_SIZE)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_playing(self, playing: bool) -> None:
        self.playing = playing
        self.setProperty("playing", playing)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().buttonText().color()
        painter.setPen(QPen(color, 1.4))
        painter.setBrush(color)
        x, y = self.width() / 2.0, self.height() / 2.0
        if self.control == "play" and self.playing:
            painter.drawRoundedRect(int(x - 4), int(y - 5), 3, 10, 1, 1)
            painter.drawRoundedRect(int(x + 1), int(y - 5), 3, 10, 1, 1)
            return
        if self.control == "play":
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 3, y - 5),
                        QPointF(x + 5, y),
                        QPointF(x - 3, y + 5),
                    ]
                )
            )
            return
        direction = -1 if self.control == "previous" else 1
        bar_x = x - 5 if direction < 0 else x + 5
        painter.drawLine(QPointF(bar_x, y - 5), QPointF(bar_x, y + 5))
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(x - direction * 3, y - 5),
                    QPointF(x + direction * 4, y),
                    QPointF(x - direction * 3, y + 5),
                ]
            )
        )


class TimelineSlider(QSlider):
    """A precise scrubber that seeks on the pointer position, not page steps."""

    previewChanged = Signal(int)
    previewCleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("timelineSlider")
        self.setTracking(False)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _value_at(self, x: float) -> int:
        handle_margin = 7
        available = max(1, self.width() - handle_margin * 2)
        position = max(0, min(available, round(x) - handle_margin))
        return self.minimum() + round(
            position * (self.maximum() - self.minimum()) / available
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            value = self._value_at(event.position().x())
            self.setSliderDown(True)
            self.setSliderPosition(value)
            self.sliderMoved.emit(value)
            self.previewChanged.emit(value)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        value = self._value_at(event.position().x())
        self.previewChanged.emit(value)
        if self.isSliderDown():
            self.setSliderPosition(value)
            self.sliderMoved.emit(value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            value = self._value_at(event.position().x())
            self.setSliderPosition(value)
            self.setValue(value)
            self.setSliderDown(False)
            self.previewChanged.emit(value)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        if not self.isSliderDown():
            self.previewCleared.emit()
        super().leaveEvent(event)


class RoundedToolTip(QWidget):
    """Transparent tooltip window with one clipped rounded surface."""

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.ToolTip
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.BypassWindowManagerHint,
        )
        self.setObjectName("toolTipWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        self.surface = QFrame()
        self.surface.setObjectName("appToolTip")
        surface_layout = QVBoxLayout(self.surface)
        surface_layout.setContentsMargins(9, 6, 9, 6)
        self.label = QLabel()
        self.label.setObjectName("appToolTipText")
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(360)
        surface_layout.addWidget(self.label)
        shadow = QGraphicsDropShadowEffect(self.surface)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 85))
        self.surface.setGraphicsEffect(shadow)
        outer.addWidget(self.surface)

    def show_text(self, text: str, position: QPoint) -> None:
        self.label.setText(text)
        self.adjustSize()
        screen = QApplication.screenAt(position) or QApplication.primaryScreen()
        target = position + QPoint(12, 18)
        if screen is not None:
            bounds = screen.availableGeometry()
            target.setX(min(target.x(), bounds.right() - self.width()))
            target.setY(min(target.y(), bounds.bottom() - self.height()))
            target.setX(max(bounds.left(), target.x()))
            target.setY(max(bounds.top(), target.y()))
        self.move(target)
        self.show()
        self.raise_()


class ToolTipManager(QObject):
    """Replace native mixed-corner tooltips across the application."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._disposed = False
        self._application = QApplication.instance()
        self.popup: RoundedToolTip | None = RoundedToolTip()
        self.destroyed.connect(self.popup.deleteLater)
        self.popup.destroyed.connect(self._popup_destroyed)
        if parent is not None:
            parent.destroyed.connect(self.dispose)
        if self._application is not None:
            self._application.aboutToQuit.connect(self.dispose)

    @staticmethod
    def _alive(obj: QObject | None) -> bool:
        # Top-level widgets can be destroyed before their Python owner during
        # QApplication shutdown. A non-None wrapper is not proof it is usable.
        from shiboken6 import isValid

        return obj is not None and isValid(obj)

    def dispose(self) -> None:
        """Detach the global filter before releasing its top-level tooltip."""
        if self._disposed:
            return
        self._disposed = True
        application = self._application
        self._application = None
        if self._alive(application) and self._alive(self):
            application.removeEventFilter(self)
        popup = self.popup
        self.popup = None
        if self._alive(popup):
            popup.hide()
            popup.deleteLater()

    def _popup_destroyed(self, _object: QObject | None = None) -> None:
        self.popup = None
        self.dispose()

    def eventFilter(self, watched, event) -> bool:
        if self._disposed:
            return False
        popup = self.popup
        if not self._alive(popup):
            self.dispose()
            return False
        event_type = event.type()
        if event_type == QEvent.Type.DeferredDelete and (
            watched is self or watched is self.parent()
        ):
            self.dispose()
            return False
        if event_type == QEvent.Type.ToolTip and isinstance(watched, QWidget):
            text = watched.toolTip().strip()
            if text:
                popup.show_text(text, event.globalPos())
                return True
        if event_type in {
            QEvent.Type.Leave,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Wheel,
            QEvent.Type.KeyPress,
            QEvent.Type.WindowDeactivate,
        }:
            popup.hide()
        return super().eventFilter(watched, event)


class ActivityButton(QPushButton):
    """Font-independent activity-bar icon."""

    def __init__(self, kind: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("activityButton")
        self.setCheckable(True)
        self.setFixedSize(ICON_ACTIVITY_SIZE, ICON_ACTIVITY_SIZE)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().buttonText().color()
        color.setAlpha(230 if self.isChecked() else 155)
        pen = QPen(color, 1.45)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        center = QPointF(self.width() / 2, self.height() / 2)
        if self.kind == "data":
            painter.drawRoundedRect(QRectF(center.x() - 8, center.y() - 8, 16, 16), 2, 2)
            painter.drawLine(center.x() - 4, center.y() - 3, center.x() + 4, center.y() - 3)
            painter.drawLine(center.x() - 4, center.y() + 1, center.x() + 4, center.y() + 1)
            painter.drawLine(center.x() - 4, center.y() + 5, center.x() + 1, center.y() + 5)
        elif self.kind == "display":
            painter.drawRect(QRectF(center.x() - 9, center.y() - 7, 8, 7))
            painter.drawRect(QRectF(center.x() + 1, center.y() - 7, 8, 7))
            painter.drawRect(QRectF(center.x() - 9, center.y() + 2, 8, 7))
            painter.drawRect(QRectF(center.x() + 1, center.y() + 2, 8, 7))
        elif self.kind == "statistics":
            painter.drawLine(center.x() - 9, center.y() + 7, center.x() - 9, center.y() - 7)
            painter.drawLine(center.x() - 9, center.y() + 7, center.x() + 9, center.y() + 7)
            points = (
                QPointF(center.x() - 7, center.y() + 3),
                QPointF(center.x() - 2, center.y() - 2),
                QPointF(center.x() + 2, center.y() + 1),
                QPointF(center.x() + 8, center.y() - 6),
            )
            for first, second in zip(points, points[1:]):
                painter.drawLine(first, second)
        elif self.kind == "adjust":
            for y, offset in ((-6, -3), (0, 4), (6, -1)):
                painter.drawLine(center.x() - 9, center.y() + y, center.x() + 9, center.y() + y)
                painter.drawEllipse(QPointF(center.x() + offset, center.y() + y), 2, 2)
        elif self.kind == "measure":
            painter.drawRect(QRectF(center.x() - 9, center.y() - 5, 18, 10))
            for x in (-5, -1, 3, 7):
                painter.drawLine(center.x() + x, center.y() - 5, center.x() + x, center.y())
        elif self.kind == "stereo":
            painter.drawRoundedRect(
                QRectF(center.x() - 10, center.y() - 6, 8, 12), 2, 2
            )
            painter.drawRoundedRect(
                QRectF(center.x() + 2, center.y() - 6, 8, 12), 2, 2
            )
            painter.drawLine(center.x() - 2, center.y(), center.x() + 2, center.y())
        elif self.kind == "cloud":
            for offset_x, offset_y in (
                (-7, -5),
                (1, -7),
                (7, -1),
                (-4, 3),
                (4, 6),
                (-9, 7),
            ):
                painter.drawEllipse(
                    QPointF(center.x() + offset_x, center.y() + offset_y),
                    1.4,
                    1.4,
                )
        else:
            painter.drawRoundedRect(QRectF(center.x() - 8, center.y() - 8, 16, 16), 4, 4)
            painter.drawLine(center.x() - 4, center.y(), center.x() - 1, center.y() + 3)
            painter.drawLine(center.x() - 1, center.y() + 3, center.x() + 5, center.y() - 4)


class ToolIconButton(QPushButton):
    """Compact inspection tool whose icon is painted instead of font glyphs."""

    def __init__(
        self,
        kind: str,
        tooltip: str,
        *,
        checkable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("toolIconButton")
        self.setCheckable(checkable)
        self.setFixedSize(ICON_TOOL_SIZE, ICON_TOOL_SIZE)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().buttonText().color()
        color.setAlpha(230 if self.isChecked() or self.underMouse() else 155)
        pen = QPen(color, 1.35)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        x, y = self.width() / 2, self.height() / 2
        if self.kind == "crosshair":
            painter.drawEllipse(QPointF(x, y), 4, 4)
            painter.drawLine(x - 9, y, x - 5, y)
            painter.drawLine(x + 5, y, x + 9, y)
            painter.drawLine(x, y - 9, x, y - 5)
            painter.drawLine(x, y + 5, x, y + 9)
        elif self.kind == "sync":
            painter.drawArc(QRectF(x - 8, y - 7, 16, 12), 30 * 16, 145 * 16)
            painter.drawArc(QRectF(x - 8, y - 5, 16, 12), 210 * 16, 145 * 16)
            painter.drawLine(x + 7, y - 4, x + 7, y - 9)
            painter.drawLine(x - 7, y + 4, x - 7, y + 9)
        elif self.kind == "epiline":
            painter.drawRect(QRectF(x - 9, y - 7, 18, 14))
            painter.drawLine(x - 7, y + 3, x + 7, y - 3)
            painter.drawEllipse(QPointF(x - 3, y + 1.3), 1.8, 1.8)
        elif self.kind == "overlay":
            painter.drawRoundedRect(QRectF(x - 9, y - 7, 12, 14), 2, 2)
            painter.drawRoundedRect(QRectF(x - 3, y - 7, 12, 14), 2, 2)
        elif self.kind == "rectify":
            painter.drawRect(QRectF(x - 9, y - 7, 7, 14))
            painter.drawRect(QRectF(x + 2, y - 7, 7, 14))
            painter.drawLine(x - 7, y - 3, x - 4, y - 1)
            painter.drawLine(x + 4, y - 1, x + 7, y - 3)
            painter.drawLine(x - 7, y + 3, x - 4, y + 1)
            painter.drawLine(x + 4, y + 1, x + 7, y + 3)
        else:
            painter.drawRect(QRectF(x - 8, y - 7, 16, 14))


class CommandPalette(QDialog):
    """Searchable action surface opened with Ctrl+Shift+P."""

    def __init__(
        self,
        actions: list[tuple[str, str, object]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("commandPalette")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)
        self.resize(620, 390)
        self._actions = actions
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 22)
        self.surface = QFrame()
        self.surface.setObjectName("commandSurface")
        shadow = QGraphicsDropShadowEffect(self.surface)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 7)
        shadow.setColor(QColor(0, 0, 0, 110))
        self.surface.setGraphicsEffect(shadow)
        outer.addWidget(self.surface)
        layout = QVBoxLayout(self.surface)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)
        self.search = QLineEdit()
        self.search.setObjectName("commandSearch")
        self.search.setPlaceholderText("搜索命令")
        self.results = QListWidget()
        self.results.setObjectName("commandResults")
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._activate_current)
        self.results.itemActivated.connect(lambda _item: self._activate_current())
        layout.addWidget(self.search)
        layout.addWidget(self.results, 1)
        self._filter("")

    def _filter(self, query: str) -> None:
        terms = [term.lower() for term in query.split() if term]
        self.results.clear()
        for title, shortcut, _handler in self._actions:
            haystack = f"{title} {shortcut}".lower()
            if all(term in haystack for term in terms):
                item = QListWidgetItem(f"{title}    {shortcut}".rstrip())
                item.setData(Qt.ItemDataRole.UserRole, _handler)
                self.results.addItem(item)
        if self.results.count():
            self.results.setCurrentRow(0)

    def _activate_current(self) -> None:
        row = self.results.currentRow()
        if row < 0:
            return
        handler = self.results.item(row).data(Qt.ItemDataRole.UserRole)
        if handler is None:
            return
        self.accept()
        QTimer.singleShot(0, handler)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.search.clear()
        self.search.setFocus(Qt.FocusReason.PopupFocusReason)


class ImageCanvas(QGraphicsView):
    """Image viewport with editor-like zoom, pan and fit behavior."""

    cursor_moved = Signal(float, float)
    cursor_left = Signal()
    view_changed = Signal(float, float, float)
    selection_changed = Signal(str, object)
    pixel_clicked = Signal(float, float)
    fit_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self.scene().addItem(self._item)
        crosshair_pen = QPen(QColor("#54a7ff"), 0)
        crosshair_pen.setCosmetic(True)
        self._crosshair_vertical = self.scene().addLine(0, 0, 0, 0, crosshair_pen)
        self._crosshair_horizontal = self.scene().addLine(0, 0, 0, 0, crosshair_pen)
        epiline_pen = QPen(QColor("#f0a35e"), 0, Qt.PenStyle.DashLine)
        epiline_pen.setCosmetic(True)
        self._epiline = self.scene().addLine(0, 0, 0, 0, epiline_pen)
        self._crosshair_vertical.hide()
        self._crosshair_horizontal.hide()
        self._epiline.hide()
        selection_pen = QPen(QColor("#4c9ffe"), 0, Qt.PenStyle.DashLine)
        selection_pen.setCosmetic(True)
        self._selection_rect = self.scene().addRect(QRectF(), selection_pen)
        self._selection_line = self.scene().addLine(0, 0, 0, 0, selection_pen)
        self._selection_rect.hide()
        self._selection_line.hide()
        self.setBackgroundBrush(QBrush(QColor("#0d0d0d")))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.viewport().setMouseTracking(True)
        self._user_zoomed = False
        self._zoom_factor = 1.0
        self._sync_guard = False
        self._tool = "pan"
        self._crosshair_enabled = False
        self._selection_start: QPointF | None = None
        self._press_view_position: QPointF | None = None

    def set_crosshair_mode(self, enabled: bool) -> None:
        self._crosshair_enabled = enabled
        # The operating-system arrow cursor has its hot spot at the pointer
        # tip, so the tip and the scene crosshair share the inspected pixel.
        cursor = (
            Qt.CursorShape.CrossCursor
            if self._tool in {"roi", "line"}
            else Qt.CursorShape.ArrowCursor
            if enabled
            else Qt.CursorShape.OpenHandCursor
        )
        self.viewport().setCursor(QCursor(cursor))

    def set_background_color(self, color: str) -> None:
        self.setBackgroundBrush(QBrush(QColor(color)))

    def set_image(self, image, *, preserve_view: bool = False) -> None:
        previous_zoom = self._zoom_factor
        previous_scale = self.transform().m11()
        bounds = self._item.boundingRect()
        center = self.mapToScene(self.viewport().rect().center())
        center_x = center.x() / bounds.width() if bounds.width() else 0.5
        center_y = center.y() / bounds.height() if bounds.height() else 0.5
        self._item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(self._item.boundingRect())
        if preserve_view and self._user_zoomed:
            new_bounds = self._item.boundingRect()
            # Preserve the visible source region even when a preview is
            # replaced by a full-resolution render (or vice versa).
            scale = previous_scale * bounds.width() / max(1.0, new_bounds.width())
            self.resetTransform()
            self.scale(scale, scale)
            self._zoom_factor = previous_zoom
            self.centerOn(QPointF(center_x * new_bounds.width(), center_y * new_bounds.height()))
        else:
            self.reset_view()

    def reset_view(self) -> None:
        self.resetTransform()
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self._user_zoomed = False
        self._zoom_factor = 1.0

    def show_actual_size(self) -> None:
        if self._item.pixmap().isNull():
            return
        self.reset_view()
        fitted_scale = self.transform().m11()
        self.resetTransform()
        self._user_zoomed = True
        # Synchronized views interpret zoom as a multiplier of the fitted
        # image, not the raw scene transform.
        self._zoom_factor = 1.0 / max(fitted_scale, 1e-9)
        self.centerOn(self._item)
        self._emit_view_state()

    def fit_width(self) -> None:
        if self._item.pixmap().isNull():
            return
        self.reset_view()
        fitted_scale = self.transform().m11()
        self.resetTransform()
        bounds = self._item.boundingRect()
        factor = self.viewport().width() / max(1.0, bounds.width())
        self.scale(factor, factor)
        self._user_zoomed = True
        self._zoom_factor = factor / max(fitted_scale, 1e-9)
        self.centerOn(bounds.center())
        self._emit_view_state()

    def set_tool(self, tool: str) -> None:
        self._tool = tool if tool in {"pan", "roi", "line"} else "pan"
        self._selection_start = None
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if self._tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        cursor = (
            Qt.CursorShape.CrossCursor
            if self._tool in {"roi", "line"}
            else Qt.CursorShape.ArrowCursor
            if self._crosshair_enabled
            else Qt.CursorShape.OpenHandCursor
        )
        self.viewport().setCursor(QCursor(cursor))

    @property
    def tool(self) -> str:
        return self._tool

    @property
    def has_image(self) -> bool:
        return not self._item.pixmap().isNull()

    def clear_selection(self) -> None:
        self._selection_start = None
        self._selection_rect.hide()
        self._selection_line.hide()

    def mousePressEvent(self, event) -> None:
        if (
            self._tool in {"roi", "line"}
            and event.button() == Qt.MouseButton.LeftButton
            and not self._item.pixmap().isNull()
        ):
            self._selection_start = self.mapToScene(event.position().toPoint())
            if self._tool == "roi":
                self._selection_rect.setRect(QRectF(self._selection_start, self._selection_start))
                self._selection_rect.show()
                self._selection_line.hide()
            else:
                self._selection_line.setLine(
                    self._selection_start.x(),
                    self._selection_start.y(),
                    self._selection_start.x(),
                    self._selection_start.y(),
                )
                self._selection_line.show()
                self._selection_rect.hide()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_view_position = QPointF(event.position())
        super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._item.pixmap().isNull():
            return
        factor = 1.16 if event.angleDelta().y() > 0 else 1 / 1.16
        next_scale = self.transform().m11() * factor
        if 0.02 <= next_scale <= 80:
            self.scale(factor, factor)
            self._zoom_factor *= factor
            self._user_zoomed = True
            self._emit_view_state()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._selection_start is not None and self._tool in {"roi", "line"}:
            point = self.mapToScene(event.position().toPoint())
            bounds = self._item.boundingRect()
            point.setX(max(bounds.left(), min(bounds.right(), point.x())))
            point.setY(max(bounds.top(), min(bounds.bottom(), point.y())))
            if self._tool == "roi":
                self._selection_rect.setRect(
                    QRectF(self._selection_start, point).normalized()
                )
            else:
                self._selection_line.setLine(
                    self._selection_start.x(),
                    self._selection_start.y(),
                    point.x(),
                    point.y(),
                )
            event.accept()
            return
        super().mouseMoveEvent(event)
        point = self.mapToScene(event.position().toPoint())
        bounds = self._item.boundingRect()
        if bounds.contains(point) and bounds.width() and bounds.height():
            self.cursor_moved.emit(point.x() / bounds.width(), point.y() / bounds.height())

    def mouseReleaseEvent(self, event) -> None:
        if (
            self._selection_start is not None
            and self._tool in {"roi", "line"}
            and event.button() == Qt.MouseButton.LeftButton
        ):
            bounds = self._item.boundingRect()
            if self._tool == "roi":
                rect = self._selection_rect.rect().intersected(bounds)
                normalized = QRectF(
                    rect.x() / max(1.0, bounds.width()),
                    rect.y() / max(1.0, bounds.height()),
                    rect.width() / max(1.0, bounds.width()),
                    rect.height() / max(1.0, bounds.height()),
                )
                self.selection_changed.emit("roi", normalized)
            else:
                line = self._selection_line.line()
                normalized = (
                    line.x1() / max(1.0, bounds.width()),
                    line.y1() / max(1.0, bounds.height()),
                    line.x2() / max(1.0, bounds.width()),
                    line.y2() / max(1.0, bounds.height()),
                )
                self.selection_changed.emit("line", normalized)
            self._selection_start = None
            self.set_tool("pan")
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._emit_view_state()
            if (
                self._press_view_position is not None
                and (
                    abs(event.position().x() - self._press_view_position.x())
                    + abs(event.position().y() - self._press_view_position.y())
                )
                <= 4
            ):
                point = self.mapToScene(event.position().toPoint())
                bounds = self._item.boundingRect()
                if bounds.contains(point) and bounds.width() and bounds.height():
                    self.pixel_clicked.emit(
                        point.x() / bounds.width(),
                        point.y() / bounds.height(),
                    )
            self._press_view_position = None

    def leaveEvent(self, event) -> None:
        self.cursor_left.emit()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()
        self.fit_requested.emit("fit")
        self._emit_view_state()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.reset_view()

    def _emit_view_state(self) -> None:
        if self._sync_guard or self._item.pixmap().isNull():
            return
        bounds = self._item.boundingRect()
        center = self.mapToScene(self.viewport().rect().center())
        nx = center.x() / bounds.width() if bounds.width() else 0.5
        ny = center.y() / bounds.height() if bounds.height() else 0.5
        self.view_changed.emit(self._zoom_factor, nx, ny)

    def apply_view_state(self, zoom_factor: float, center_x: float, center_y: float) -> None:
        if self._item.pixmap().isNull():
            return
        self._sync_guard = True
        try:
            self.reset_view()
            zoom_factor = max(0.02, min(80.0, zoom_factor))
            if zoom_factor != 1.0:
                self.scale(zoom_factor, zoom_factor)
                self._user_zoomed = True
            self._zoom_factor = zoom_factor
            bounds = self._item.boundingRect()
            self.centerOn(QPointF(center_x * bounds.width(), center_y * bounds.height()))
        finally:
            self._sync_guard = False

    def set_crosshair(self, x: float, y: float, visible: bool) -> None:
        bounds = self._item.boundingRect()
        visible = visible and not self._item.pixmap().isNull()
        if visible:
            px = max(0.0, min(1.0, x)) * bounds.width()
            py = max(0.0, min(1.0, y)) * bounds.height()
            self._crosshair_vertical.setLine(px, 0, px, bounds.height())
            self._crosshair_horizontal.setLine(0, py, bounds.width(), py)
        self._crosshair_vertical.setVisible(visible)
        self._crosshair_horizontal.setVisible(visible)

    def set_epiline(self, line: tuple[float, float, float, float] | None) -> None:
        bounds = self._item.boundingRect()
        visible = line is not None and not self._item.pixmap().isNull()
        if line is not None:
            x1, y1, x2, y2 = line
            self._epiline.setLine(
                x1 * bounds.width(),
                y1 * bounds.height(),
                x2 * bounds.width(),
                y2 * bounds.height(),
            )
        self._epiline.setVisible(visible)


class TitleBar(QFrame):
    """Integrated title bar that delegates native move behavior to Windows."""

    minimize_requested = Signal()
    maximize_requested = Signal()
    settings_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("titleBar")
        self.setFixedHeight(TITLE_BAR_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 5, 0)
        layout.setSpacing(4)

        mark = QLabel("SS")
        mark.setObjectName("titleMark")
        mark.setFixedSize(23, 23)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Stereo Selector")
        title.setObjectName("titleBrand")
        self.context = QLabel("")
        self.context.setObjectName("titleContext")
        self.context.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.context.hide()

        self.actions = QWidget()
        self.actions.setObjectName("titleActions")
        self.action_layout = QHBoxLayout(self.actions)
        self.action_layout.setContentsMargins(0, 0, 0, 0)
        self.action_layout.setSpacing(3)

        self.settings_button = QPushButton("设置")
        self.settings_button.setObjectName("titleActionButton")
        self.settings_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.minimize_button = WindowControlButton("minimize")
        self.maximize_button = WindowControlButton("maximize")
        self.close_button = WindowControlButton("close")
        for button in (self.minimize_button, self.maximize_button, self.close_button):
            button.setFixedSize(ICON_WINDOW_WIDTH, ICON_WINDOW_HEIGHT)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.minimize_button.clicked.connect(self.minimize_requested)
        self.maximize_button.clicked.connect(self.maximize_requested)
        self.close_button.clicked.connect(self.close_requested)
        self.settings_button.clicked.connect(self.settings_requested)
        layout.addWidget(mark)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.context)
        layout.addStretch(1)
        layout.addWidget(self.actions)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)

    def set_context(self, text: str) -> None:
        self.context.setText(text)
        self.context.setVisible(bool(text))

    def set_maximized(self, maximized: bool) -> None:
        self.maximize_button.set_maximized(maximized)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if sys.platform != "win32" and event.position().y() < 7:
                # Leave the top edge to the main window's system resize.
                event.ignore()
                return
            window_handle = self.window().windowHandle()
            if window_handle is not None:
                window_handle.startSystemMove()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.maximize_requested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class PointCloudCanvas(QWidget):
    point_picked = Signal(object)
    measurement_changed = Signal(object)
    selection_changed = Signal(int)
    tool_finished = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        cam_offset: float = 0.05,
        dot_radius: float = 1.0,
        z_max: float = 10.0,
        intrinsics: np.ndarray | None = None,
        image_size: tuple[int, int] | None = None,
        background: str = "#0d0d0d",
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._scatter = None
        self._highlight = None
        self._cloud: PointCloudData | None = None
        self._color_mode = "rgb"
        self._rendered_points = np.empty((0, 3), dtype=np.float32)
        self._rendered_colors = np.empty((0, 4), dtype=np.float32)
        self._rendered_source_colors = np.empty((0, 4), dtype=np.float32)
        self._interaction_mode = "rotate"
        self._measure_points: list[np.ndarray] = []
        self._box_origin: QPoint | None = None
        self._center = np.zeros(3, dtype=np.float32)
        self._distance = 1.0
        self.cam_offset = float(cam_offset)
        self.dot_radius = float(dot_radius)
        self.z_max = float(z_max)
        self.camera_intrinsics, self.camera_image_size = resolve_camera_model(
            intrinsics,
            image_size,
        )
        self._right_pan_position: QPointF | None = None
        try:
            import pyqtgraph.opengl as gl

            self._gl = gl
            view_class = type(
                "StereoSelectorGLView",
                (PinholeProjectionMixin, gl.GLViewWidget),
                {},
            )
            self.view = view_class()
            self.view.configure_camera_projection(
                self.camera_intrinsics,
                self.camera_image_size,
                max(100.0, (self.z_max + self.cam_offset) * 2.0),
            )
            self.view.setBackgroundColor(background)
            self.view.installEventFilter(self)
            self._rubber_band = QRubberBand(
                QRubberBand.Shape.Rectangle,
                self.view,
            )
            self._axis = gl.GLAxisItem()
            self._axis.setSize(1.0, 1.0, 1.0)
            self._axis.setVisible(False)
            self.view.addItem(self._axis)
            self._grid = gl.GLGridItem()
            self._grid.setSize(10.0, 10.0)
            self._grid.setSpacing(1.0, 1.0)
            self._grid.setVisible(False)
            self.view.addItem(self._grid)
            frustum_points = self._camera_frustum_points()
            self._frustum = gl.GLLinePlotItem(
                pos=frustum_points,
                color=(0.35, 0.65, 1.0, 0.75),
                width=1.0,
                antialias=True,
                mode="lines",
            )
            self._frustum.setVisible(False)
            self.view.addItem(self._frustum)
            self._highlight = gl.GLScatterPlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=(1.0, 0.35, 0.1, 1.0),
                size=11.0,
                pxMode=True,
            )
            self.view.addItem(self._highlight)
            layout.addWidget(self.view)
        except Exception as exc:  # pragma: no cover - host OpenGL dependent
            self._gl = None
            self.view = None
            label = QLabel(f"OpenGL 视图初始化失败\n{exc}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setObjectName("errorText")
            layout.addWidget(label)

    def set_cloud_data(self, cloud: PointCloudData) -> int:
        if self.view is None or self._gl is None:
            raise RuntimeError("当前系统无法创建 OpenGL 点云视图")
        if not len(cloud.points):
            if cloud.max_distance is not None:
                raise ValueError(f"{cloud.max_distance:g} m 范围内没有有效点")
            raise ValueError("点云中没有有效顶点")
        self._cloud = cloud
        self._measure_points.clear()
        render_points = camera_points_to_gl(cloud.points)
        colors = self._colors_for_mode(cloud, self._color_mode)
        self._rendered_points = cloud.points
        self._rendered_colors = colors
        self._rendered_source_colors = cloud.colors
        if self._scatter is None:
            self._scatter = self._gl.GLScatterPlotItem(
                pos=render_points,
                color=colors,
                size=self.dot_radius * 2.0,
                pxMode=True,
                glOptions="opaque",
            )
            self.view.addItem(self._scatter)
        else:
            # Reuse the GL item and its shader/buffers. Recreating the item on
            # every frame causes a visible main-thread stall on some drivers.
            self._scatter.setData(
                pos=render_points,
                color=colors,
                size=self.dot_radius * 2.0,
                pxMode=True,
            )
        # Keep the optical axis fixed at X=0/Y=0. With this focus point and
        # camera-aligned azimuth, cameraPosition() becomes (0, -cam_offset, 0)
        # in GL coordinates, which is exactly z = Z + cam_offset.
        self._distance = max(float(np.median(cloud.points[:, 2])), 0.01)
        self._center = np.array(
            [0.0, self._distance - self.cam_offset, 0.0],
            dtype=np.float32,
        )
        self.reset_view()
        return len(cloud.points)

    @property
    def current_cloud(self) -> PointCloudData | None:
        """The point cloud currently loaded into this canvas, if any."""
        return self._cloud

    @property
    def interaction_mode(self) -> str:
        return self._interaction_mode

    @property
    def color_mode(self) -> str:
        return self._color_mode

    @property
    def guides_visible(self) -> bool:
        return self._axis.visible() if getattr(self, "_axis", None) is not None else False

    def _camera_frustum_points(self) -> np.ndarray:
        matrix = self.camera_intrinsics
        width, height = self.camera_image_size
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        distance = min(1.0, max(0.15, self.z_max * 0.08))
        corners = np.asarray(
            [
                [(0 - cx) / fx * distance, (0 - cy) / fy * distance, distance],
                [(width - cx) / fx * distance, (0 - cy) / fy * distance, distance],
                [(width - cx) / fx * distance, (height - cy) / fy * distance, distance],
                [(0 - cx) / fx * distance, (height - cy) / fy * distance, distance],
            ],
            dtype=np.float32,
        )
        origin = np.zeros((1, 3), dtype=np.float32)
        segments = []
        for corner in corners:
            segments.extend((origin[0], corner))
        for first, second in zip(corners, np.roll(corners, -1, axis=0)):
            segments.extend((first, second))
        return camera_points_to_gl(np.asarray(segments, dtype=np.float32))

    def _colors_for_mode(self, cloud: PointCloudData, mode: str) -> np.ndarray:
        if mode == "rgb":
            return cloud.colors
        column = {"x": 0, "y": 1, "z": 2, "height": 1}.get(mode, 2)
        values = cloud.points[:, column].astype(np.float32, copy=False)
        finite = np.isfinite(values)
        if not finite.any():
            normalized = np.zeros(len(values), dtype=np.float32)
        else:
            low, high = np.percentile(values[finite], (1.0, 99.0))
            if high <= low:
                high = low + 1.0
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
            if mode == "height":
                normalized = 1.0 - normalized
        rgb = colorize_scalar(normalized, "turbo").astype(np.float32) / 255.0
        return np.column_stack((rgb, np.ones(len(rgb), dtype=np.float32)))

    def set_color_mode(self, mode: str) -> None:
        self._color_mode = mode if mode in {"rgb", "x", "y", "z", "height"} else "rgb"
        if self._cloud is not None and self._scatter is not None:
            visible_cloud = PointCloudData(
                self._rendered_points,
                self._rendered_source_colors,
                self._cloud.original_count,
                len(self._rendered_points),
                self._cloud.max_distance,
            )
            colors = self._colors_for_mode(visible_cloud, self._color_mode)
            self._rendered_colors = colors
            self._scatter.setData(
                pos=camera_points_to_gl(self._rendered_points),
                color=colors,
                size=self.dot_radius * 2.0,
                pxMode=True,
            )

    def set_point_size(self, radius: float) -> None:
        self.dot_radius = max(0.5, min(12.0, float(radius)))
        if self._cloud is not None and self._scatter is not None:
            self._scatter.setData(
                pos=camera_points_to_gl(self._rendered_points),
                color=self._rendered_colors,
                size=self.dot_radius * 2.0,
                pxMode=True,
            )

    def set_clip_ranges(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
    ) -> int:
        if self._cloud is None or self._scatter is None:
            return 0
        points = self._cloud.points
        mask = (
            (points[:, 0] >= min(x_range))
            & (points[:, 0] <= max(x_range))
            & (points[:, 1] >= min(y_range))
            & (points[:, 1] <= max(y_range))
            & (points[:, 2] >= min(z_range))
            & (points[:, 2] <= max(z_range))
        )
        visible_points = points[mask]
        visible_cloud = PointCloudData(
            visible_points,
            self._cloud.colors[mask],
            self._cloud.original_count,
            len(visible_points),
            self._cloud.max_distance,
        )
        colors = self._colors_for_mode(visible_cloud, self._color_mode)
        self._rendered_points = visible_points
        self._rendered_colors = colors
        self._rendered_source_colors = self._cloud.colors[mask]
        self._scatter.setData(
            pos=camera_points_to_gl(visible_points),
            color=colors,
            size=self.dot_radius * 2.0,
            pxMode=True,
        )
        return len(visible_points)

    def set_guides_visible(self, visible: bool) -> None:
        if self.view is None:
            return
        self._axis.setVisible(visible)
        self._grid.setVisible(visible)
        self._frustum.setVisible(visible)
        self.view.update()

    def set_background(self, color: str) -> None:
        if self.view is not None:
            self.view.setBackgroundColor(color)

    def set_interaction_mode(self, mode: str) -> None:
        self._interaction_mode = (
            mode if mode in {"rotate", "pick", "measure", "box"} else "rotate"
        )
        if self._interaction_mode != "measure":
            self._measure_points.clear()
        self._box_origin = None
        if hasattr(self, "_rubber_band"):
            self._rubber_band.hide()
        if self.view is not None:
            self.view.setCursor(
                Qt.CursorShape.ArrowCursor if self._interaction_mode == "rotate"
                else Qt.CursorShape.CrossCursor
            )

    def highlight_image_point(self, x: float, y: float) -> None:
        if self._cloud is None or self._highlight is None or not len(self._cloud.points):
            return
        uv, projected_z = project_camera_points(
            self._cloud.points,
            self.camera_intrinsics,
            self.cam_offset,
        )
        width, height = self.camera_image_size
        target = np.asarray([x * width, y * height], dtype=np.float64)
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(projected_z)
        if not valid.any():
            return
        indices = np.flatnonzero(valid)
        distances = np.sum((uv[indices] - target) ** 2, axis=1)
        index = indices[int(np.argmin(distances))]
        self._highlight.setData(
            pos=camera_points_to_gl(self._cloud.points[index : index + 1]),
            color=(1.0, 0.35, 0.1, 1.0),
            size=11.0,
            pxMode=True,
        )

    def _pick_point(self, position: QPointF) -> np.ndarray | None:
        projected = self._screen_projection()
        if projected is None:
            return None
        indices, screen_x, screen_y, ndc = projected
        distance_sq = (screen_x - position.x()) ** 2 + (screen_y - position.y()) ** 2
        nearest_position = int(np.argmin(distance_sq))
        if distance_sq[nearest_position] > 14.0**2:
            return None
        # Prefer the visually front-most point among overlapping candidates.
        close = distance_sq <= max(14.0**2, distance_sq[nearest_position] + 9.0)
        close_positions = np.flatnonzero(close)
        chosen = close_positions[int(np.argmin(ndc[close_positions, 2]))]
        return self._rendered_points[indices[chosen]]

    def _screen_projection(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if self.view is None or not len(self._rendered_points):
            return None
        try:
            viewport = (0, 0, self.view.width(), self.view.height())
            matrix = self.view.projectionMatrix(viewport, viewport) * self.view.viewMatrix()
            transform = np.asarray(matrix.copyDataTo(), dtype=np.float64).reshape(4, 4)
        except (AttributeError, TypeError, ValueError):
            return None
        render = camera_points_to_gl(self._rendered_points).astype(np.float64)
        homogeneous = np.column_stack((render, np.ones(len(render), dtype=np.float64)))
        clip = homogeneous @ transform.T
        valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0)
        if not valid.any():
            return None
        indices = np.flatnonzero(valid)
        ndc = clip[indices, :3] / clip[indices, 3:4]
        visible = (
            (np.abs(ndc[:, 0]) <= 1.0)
            & (np.abs(ndc[:, 1]) <= 1.0)
            & (np.abs(ndc[:, 2]) <= 1.0)
        )
        indices = indices[visible]
        ndc = ndc[visible]
        if not len(indices):
            return None
        screen_x = (ndc[:, 0] + 1.0) * 0.5 * self.view.width()
        screen_y = (1.0 - ndc[:, 1]) * 0.5 * self.view.height()
        return indices, screen_x, screen_y, ndc

    def _select_box(self, rectangle: QRectF) -> int:
        projected = self._screen_projection()
        if projected is None:
            return 0
        indices, screen_x, screen_y, _ndc = projected
        inside = (
            (screen_x >= rectangle.left())
            & (screen_x <= rectangle.right())
            & (screen_y >= rectangle.top())
            & (screen_y <= rectangle.bottom())
        )
        selected = indices[inside]
        if not len(selected):
            return 0
        self._rendered_points = self._rendered_points[selected]
        self._rendered_source_colors = self._rendered_source_colors[selected]
        visible_cloud = PointCloudData(
            self._rendered_points,
            self._rendered_source_colors,
            self._cloud.original_count if self._cloud is not None else len(selected),
            len(selected),
            self._cloud.max_distance if self._cloud is not None else None,
        )
        self._rendered_colors = self._colors_for_mode(
            visible_cloud,
            self._color_mode,
        )
        self._scatter.setData(
            pos=camera_points_to_gl(self._rendered_points),
            color=self._rendered_colors,
            size=self.dot_radius * 2.0,
            pxMode=True,
        )
        self.selection_changed.emit(len(selected))
        return len(selected)

    def restore_full_cloud(self) -> int:
        if self._cloud is None or self._scatter is None:
            return 0
        self._rendered_points = self._cloud.points
        self._rendered_source_colors = self._cloud.colors
        self._rendered_colors = self._colors_for_mode(
            self._cloud,
            self._color_mode,
        )
        self._scatter.setData(
            pos=camera_points_to_gl(self._rendered_points),
            color=self._rendered_colors,
            size=self.dot_radius * 2.0,
            pxMode=True,
        )
        return len(self._rendered_points)

    def set_standard_view(self, name: str) -> None:
        if self.view is None:
            return
        angles = {
            "front": (CAMERA_ALIGNED_ELEVATION, CAMERA_ALIGNED_AZIMUTH),
            "back": (0.0, 90.0),
            "left": (0.0, 0.0),
            "right": (0.0, 180.0),
            "top": (89.9, -90.0),
        }
        elevation, azimuth = angles.get(name, angles["front"])
        self.view.setCameraPosition(
            pos=QVector3D(*map(float, self._center)),
            distance=self._distance,
            elevation=elevation,
            azimuth=azimuth,
        )

    def reset_view(self) -> None:
        if self.view is None:
            return
        self.view.setCameraPosition(
            pos=QVector3D(float(self._center[0]), float(self._center[1]), float(self._center[2])),
            distance=self._distance,
            elevation=CAMERA_ALIGNED_ELEVATION,
            azimuth=CAMERA_ALIGNED_AZIMUTH,
        )

    def eventFilter(self, watched, event) -> bool:
        if watched is self.view:
            event_type = event.type()
            interaction_mode = getattr(self, "_interaction_mode", "rotate")
            if (
                interaction_mode == "box"
                and event_type == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._box_origin = event.position().toPoint()
                self._rubber_band.setGeometry(
                    QRectF(
                        self._box_origin.x(),
                        self._box_origin.y(),
                        1,
                        1,
                    ).toRect()
                )
                self._rubber_band.show()
                event.accept()
                return True
            if (
                interaction_mode == "box"
                and event_type == QEvent.Type.MouseMove
                and self._box_origin is not None
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                current = event.position().toPoint()
                rectangle = QRectF(
                    self._box_origin.x(),
                    self._box_origin.y(),
                    current.x() - self._box_origin.x(),
                    current.y() - self._box_origin.y(),
                ).normalized()
                self._rubber_band.setGeometry(rectangle.toRect())
                event.accept()
                return True
            if (
                interaction_mode == "box"
                and event_type == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
                and self._box_origin is not None
            ):
                rectangle = QRectF(self._rubber_band.geometry())
                self._rubber_band.hide()
                self._box_origin = None
                self._select_box(rectangle)
                self.set_interaction_mode("rotate")
                self.tool_finished.emit()
                event.accept()
                return True
            if (
                interaction_mode in {"pick", "measure"}
                and event_type == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                event.accept()
                return True
            if (
                interaction_mode in {"pick", "measure"}
                and event_type == QEvent.Type.MouseMove
                and event.buttons() & Qt.MouseButton.LeftButton
            ):
                event.accept()
                return True
            if (
                event_type == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.RightButton
            ):
                self._right_pan_position = QPointF(event.position())
                self.view.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return True
            if (
                event_type == QEvent.Type.MouseMove
                and self._right_pan_position is not None
                and event.buttons() & Qt.MouseButton.RightButton
            ):
                current = QPointF(event.position())
                delta = current - self._right_pan_position
                self._right_pan_position = current
                self.view.pan(delta.x(), delta.y(), 0.0, relative="view")
                event.accept()
                return True
            if (
                event_type == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.RightButton
            ):
                self._right_pan_position = None
                self.view.unsetCursor()
                event.accept()
                return True
            if (
                event_type == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
                and interaction_mode in {"pick", "measure"}
            ):
                point = self._pick_point(QPointF(event.position()))
                if point is not None:
                    self._highlight.setData(
                        pos=camera_points_to_gl(point.reshape(1, 3)),
                        color=(1.0, 0.35, 0.1, 1.0),
                        size=11.0,
                        pxMode=True,
                    )
                    self.point_picked.emit(point.copy())
                    if interaction_mode == "measure":
                        self._measure_points.append(point.copy())
                        if len(self._measure_points) == 2:
                            self.measurement_changed.emit(
                                (
                                    self._measure_points[0].copy(),
                                    self._measure_points[1].copy(),
                                    float(np.linalg.norm(
                                        self._measure_points[1] - self._measure_points[0]
                                    )),
                                )
                            )
                            self._measure_points.clear()
                            self.set_interaction_mode("rotate")
                            self.tool_finished.emit()
                    else:
                        self.set_interaction_mode("rotate")
                        self.tool_finished.emit()
                event.accept()
                return True
        return super().eventFilter(watched, event)


class DropHint(QFrame):
    open_requested = Signal()
    recent_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("emptyState")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(24, 24, 24, 24)

        mark = QLabel("S")
        mark.setObjectName("emptyMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(48, 48)
        title = QLabel("打开项目")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("拖放文件夹，或从本机选择")
        hint.setObjectName("emptyHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        open_button = QPushButton("选择文件夹")
        open_button.setObjectName("primaryButton")
        open_button.clicked.connect(self.open_requested)
        layout.addWidget(mark, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(14)
        layout.addWidget(title)
        layout.addSpacing(5)
        layout.addWidget(hint)
        layout.addSpacing(16)
        layout.addWidget(open_button, 0, Qt.AlignmentFlag.AlignHCenter)
        self.recent_title = QLabel("最近项目")
        self.recent_title.setObjectName("emptyFormats")
        self.recent_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recent_title.hide()
        layout.addSpacing(20)
        layout.addWidget(self.recent_title)
        self.recent_container = QWidget()
        self.recent_layout = QVBoxLayout(self.recent_container)
        self.recent_layout.setContentsMargins(0, 4, 0, 0)
        self.recent_layout.setSpacing(2)
        self.recent_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.recent_container, 0, Qt.AlignmentFlag.AlignHCenter)
        self.recent_buttons: list[QPushButton] = []

    def set_recent(self, paths: list[str]) -> None:
        for button in self.recent_buttons:
            self.recent_layout.removeWidget(button)
            button.deleteLater()
        self.recent_buttons.clear()
        for path in paths[:5]:
            button = QPushButton(Path(path).name or path)
            button.setObjectName("ghostButton")
            button.setToolTip(path)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumWidth(220)
            button.clicked.connect(lambda _checked=False, target=path: self.recent_requested.emit(target))
            self.recent_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
            self.recent_buttons.append(button)
        self.recent_title.setVisible(bool(self.recent_buttons))
        self.recent_container.setVisible(bool(self.recent_buttons))


class NoViewsHint(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("noViewsHint")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("选择视图")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

def __getattr__(name: str):
    # Preserve the historic import without a widgets <-> media_tile cycle.
    if name == "MediaTile":
        from .media_tile import MediaTile
        return MediaTile
    raise AttributeError(name)
