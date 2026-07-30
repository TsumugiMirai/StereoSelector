from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QRunnable,
    QThreadPool,
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
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRubberBand,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .media import (
    DepthStats,
    ImageData,
    ImageStats,
    PointCloudData,
    analyze_depth,
    analyze_image,
    colorize_scalar,
    load_image_data,
    load_point_cloud,
    project_camera_points,
    render_image_values,
    resolve_camera_model,
)
from .models import modality_label


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


class MediaWorkerSignals(QObject):
    loaded = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)


class AnalysisWorkerSignals(QObject):
    loaded = Signal(int, object, object)


class MediaAnalysisWorker(QRunnable):
    def __init__(
        self,
        token: int,
        values: np.ndarray,
        *,
        include_depth: bool,
    ) -> None:
        super().__init__()
        self.token = token
        self.values = values
        self.include_depth = include_depth
        self.signals = AnalysisWorkerSignals()

    def run(self) -> None:
        image_stats = analyze_image(self.values)
        depth_stats = analyze_depth(self.values) if self.include_depth else None
        try:
            self.signals.loaded.emit(self.token, image_stats, depth_stats)
        except RuntimeError:
            pass


class MediaLoadWorker(QRunnable):
    def __init__(
        self,
        token: int,
        modality: str,
        path: Path,
        point_limit: int,
        cloud_cam_offset: float = 0.05,
        cloud_grid: int = 5,
        cloud_z_max: float = 10.0,
        cloud_tau_rel: float = 0.15,
        cloud_occlusion: bool = True,
        cloud_intrinsics: np.ndarray | None = None,
        cloud_image_size: tuple[int, int] | None = None,
        cloud_rotation: np.ndarray | None = None,
        cloud_translation: np.ndarray | None = None,
        analysis_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.token = token
        self.modality = modality
        self.path = path
        self.point_limit = point_limit
        self.cloud_cam_offset = cloud_cam_offset
        self.cloud_grid = cloud_grid
        self.cloud_z_max = cloud_z_max
        self.cloud_tau_rel = cloud_tau_rel
        self.cloud_occlusion = cloud_occlusion
        self.cloud_intrinsics = cloud_intrinsics
        self.cloud_image_size = cloud_image_size
        self.cloud_rotation = cloud_rotation
        self.cloud_translation = cloud_translation
        self.analysis_enabled = analysis_enabled
        self.signals = MediaWorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            if self.modality == "ply":
                result = load_point_cloud(
                    self.path,
                    max_points=self.point_limit,
                    rotation=self.cloud_rotation,
                    translation=self.cloud_translation,
                    max_distance=self.cloud_z_max,
                    intrinsics=self.cloud_intrinsics,
                    image_size=self.cloud_image_size,
                    cam_offset=self.cloud_cam_offset,
                    grid=self.cloud_grid,
                    tau_rel=self.cloud_tau_rel,
                    occlusion=self.cloud_occlusion,
                    cancel_check=lambda: self._cancelled,
                )
            else:
                result = load_image_data(self.path)
        except InterruptedError:
            self._publish_cancelled()
            return
        except Exception as exc:
            if self._cancelled:
                self._publish_cancelled()
            else:
                try:
                    self.signals.failed.emit(self.token, str(exc))
                except RuntimeError:
                    pass  # The window was destroyed while this task was finishing.
            return
        if self._cancelled:
            self._publish_cancelled()
            return
        try:
            self.signals.loaded.emit(self.token, result)
        except RuntimeError:
            pass  # The receiver/source may be gone during application shutdown.

    def _publish_cancelled(self) -> None:
        try:
            self.signals.cancelled.emit(self.token)
        except RuntimeError:
            pass


class LoadingSpinner(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(28, 28)
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
        self.setFixedSize(32, 32)
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
    """Font-independent media control for the review timeline."""

    def __init__(
        self,
        control: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.control = control
        self.playing = False
        self.setObjectName("playbackButton")
        self.setFixedSize(30, 30)
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


class ActivityButton(QPushButton):
    """Font-independent activity-bar icon."""

    def __init__(self, kind: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("activityButton")
        self.setCheckable(True)
        self.setFixedSize(36, 36)
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
        elif self.kind == "analysis":
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
        self.setFixedSize(28, 28)
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
                self.results.addItem(f"{title}    {shortcut}".rstrip())
        if self.results.count():
            self.results.setCurrentRow(0)

    def _activate_current(self) -> None:
        row = self.results.currentRow()
        if row < 0:
            return
        visible_text = self.results.item(row).text()
        title = visible_text.split("    ", 1)[0]
        handler = next(
            (callback for action_title, _shortcut, callback in self._actions if action_title == title),
            None,
        )
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
        self._selection_start: QPointF | None = None
        self._press_view_position: QPointF | None = None

    def set_crosshair_mode(self, enabled: bool) -> None:
        # The scene crosshair already communicates the exact location. Hiding
        # the large hand cursor prevents it from covering the inspected pixel.
        cursor = (
            Qt.CursorShape.CrossCursor
            if self._tool in {"roi", "line"}
            else Qt.CursorShape.BlankCursor
            if enabled
            else Qt.CursorShape.OpenHandCursor
        )
        self.viewport().setCursor(QCursor(cursor))

    def set_image(self, image, *, preserve_view: bool = False) -> None:
        previous_zoom = self._zoom_factor
        bounds = self._item.boundingRect()
        center = self.mapToScene(self.viewport().rect().center())
        center_x = center.x() / bounds.width() if bounds.width() else 0.5
        center_y = center.y() / bounds.height() if bounds.height() else 0.5
        self._item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(self._item.boundingRect())
        if preserve_view and self._user_zoomed:
            self.apply_view_state(previous_zoom, center_x, center_y)
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
        self.resetTransform()
        self._user_zoomed = True
        self._zoom_factor = 1.0
        self.centerOn(self._item)
        self._emit_view_state()

    def fit_width(self) -> None:
        if self._item.pixmap().isNull():
            return
        self.resetTransform()
        bounds = self._item.boundingRect()
        factor = self.viewport().width() / max(1.0, bounds.width())
        self.scale(factor, factor)
        self._user_zoomed = True
        self._zoom_factor = factor
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
            else Qt.CursorShape.OpenHandCursor
        )
        self.viewport().setCursor(QCursor(cursor))

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
        self.setFixedHeight(40)
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
            button.setFixedSize(42, 32)
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

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        cam_offset: float = 0.05,
        dot_radius: float = 1.0,
        z_max: float = 10.0,
        intrinsics: np.ndarray | None = None,
        image_size: tuple[int, int] | None = None,
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
            self.view.setBackgroundColor("#0d0d0d")
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
                event.accept()
                return True
        return super().eventFilter(watched, event)


class MediaTile(QFrame):
    focus_requested = Signal(str)
    cursor_moved = Signal(str, float, float)
    cursor_left = Signal(str)
    view_changed = Signal(str, float, float, float)
    data_ready = Signal(str)
    selection_changed = Signal(str, str, object)
    pixel_clicked = Signal(str, float, float)

    def __init__(
        self,
        modality: str,
        point_limit: int = 300_000,
        cloud_cam_offset: float = 0.05,
        cloud_grid: int = 5,
        cloud_dot_radius: float = 1.0,
        cloud_z_max: float = 10.0,
        cloud_tau_rel: float = 0.15,
        cloud_occlusion: bool = True,
        parent: QWidget | None = None,
        cloud_intrinsics: np.ndarray | None = None,
        cloud_image_size: tuple[int, int] | None = None,
        cloud_rotation: np.ndarray | None = None,
        cloud_translation: np.ndarray | None = None,
    ) -> None:
        super().__init__(parent)
        self.modality = modality
        self.point_limit = point_limit
        self.cloud_cam_offset = cloud_cam_offset
        self.cloud_grid = cloud_grid
        self.cloud_dot_radius = cloud_dot_radius
        self.cloud_z_max = cloud_z_max
        self.cloud_tau_rel = cloud_tau_rel
        self.cloud_occlusion = cloud_occlusion
        self.cloud_intrinsics = cloud_intrinsics
        self.cloud_image_size = cloud_image_size
        self.cloud_rotation = cloud_rotation
        self.cloud_translation = cloud_translation
        self._worker_sequence = 0
        self._active_worker_token: int | None = None
        self._workers: dict[int, MediaLoadWorker] = {}
        self._worker_cache_keys: dict[int, tuple[object, ...]] = {}
        self._analysis_sequence = 0
        self._analysis_workers: dict[int, MediaAnalysisWorker] = {}
        self._active_analysis_token: int | None = None
        self._cache: OrderedDict[tuple[object, ...], object] = OrderedDict()
        self._cache_costs: dict[tuple[object, ...], int] = {}
        self._cache_bytes = 0
        self._cache_limit_bytes = 192 * 1024 * 1024 if modality == "ply" else 96 * 1024 * 1024
        self._current_path: Path | None = None
        self.image_data: ImageData | None = None
        self.depth_stats: DepthStats | None = None
        self._meta_text = ""
        self._disposed = False
        self._analysis_enabled = True
        self._display_settings: dict[str, object] = {
            "brightness": 0.0,
            "contrast": 1.0,
            "gamma": 1.0,
            "exposure": 0.0,
            "color_map": "gray",
            "display_range": None,
            "highlight_invalid": False,
        }
        self._content_animation: QPropertyAnimation | None = None
        self._loading_delay = QTimer(self)
        self._loading_delay.setSingleShot(True)
        self._loading_delay.setInterval(220)
        self._loading_delay.timeout.connect(self._show_delayed_loading)
        self.setObjectName("mediaTile")
        self.setProperty("missing", False)
        self.setMinimumSize(280, 210)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QFrame()
        self.header.setObjectName("mediaHeader")
        self.header.installEventFilter(self)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 5, 6, 5)
        header_layout.setSpacing(6)
        accent = QFrame()
        accent.setFixedSize(3, 15)
        accent.setStyleSheet(f"background: {MODALITY_ACCENTS.get(modality, '#777')}; border-radius: 1px;")
        self.title = QLabel(modality_label(modality))
        self.title.setObjectName("tileTitle")
        self.meta = QLabel("")
        self.meta.setObjectName("tileMeta")
        self.meta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.focus_button = QPushButton("聚焦")
        self.focus_button.setObjectName("iconButton")
        self.focus_button.setToolTip("单独查看此视图 (F)")
        self.focus_button.clicked.connect(lambda: self.focus_requested.emit(self.modality))
        header_layout.addWidget(accent)
        header_layout.addWidget(self.title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.meta, 1)
        header_layout.addWidget(self.focus_button)
        outer.addWidget(self.header)

        self.stack = QStackedWidget()
        self.image_canvas = ImageCanvas()
        self.image_canvas.cursor_moved.connect(
            lambda x, y: self.cursor_moved.emit(self.modality, x, y)
        )
        self.image_canvas.cursor_left.connect(lambda: self.cursor_left.emit(self.modality))
        self.image_canvas.view_changed.connect(
            lambda zoom, x, y: self.view_changed.emit(self.modality, zoom, x, y)
        )
        self.image_canvas.selection_changed.connect(
            lambda tool, selection: self.selection_changed.emit(
                self.modality,
                tool,
                selection,
            )
        )
        self.image_canvas.pixel_clicked.connect(
            lambda x, y: self.pixel_clicked.emit(self.modality, x, y)
        )
        self.cloud_canvas = (
            PointCloudCanvas(
                cam_offset=self.cloud_cam_offset,
                dot_radius=self.cloud_dot_radius,
                z_max=self.cloud_z_max,
                intrinsics=self.cloud_intrinsics,
                image_size=self.cloud_image_size,
            )
            if modality == "ply"
            else None
        )
        self.message = QLabel("")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setObjectName("emptyTileText")
        self.loading_page = QWidget()
        loading_layout = QVBoxLayout(self.loading_page)
        loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.setSpacing(9)
        self.spinner = LoadingSpinner()
        self.loading_label = QLabel("正在加载…")
        self.loading_label.setObjectName("loadingText")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        loading_layout.addWidget(self.loading_label)
        self.stack.addWidget(self.image_canvas)
        if self.cloud_canvas is not None:
            self.stack.addWidget(self.cloud_canvas)
        self.stack.addWidget(self.message)
        self.stack.addWidget(self.loading_page)
        outer.addWidget(self.stack, 1)

    def _set_missing(self, missing: bool) -> None:
        if bool(self.property("missing")) == missing:
            return
        self.setProperty("missing", missing)
        self.style().unpolish(self)
        self.style().polish(self)

    def _set_meta(self, text: str, tooltip: str | None = None) -> None:
        self._meta_text = text
        self.meta.setToolTip(tooltip or text)
        available = max(24, self.meta.width() - 4)
        self.meta.setText(self.meta.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle, available))

    def show_file(self, path: Path | None) -> None:
        self._active_worker_token = None
        self._current_path = path
        self._loading_delay.stop()
        if path is None:
            self._cancel_workers_except()
            self.image_data = None
            self.depth_stats = None
            self.spinner.stop()
            self._set_missing(True)
            self._set_meta("未匹配", "")
            self.message.setText("当前样本没有匹配到此类型文件")
            self.stack.setCurrentWidget(self.message)
            return
        if self.modality != "ply":
            self.image_data = None
            self.depth_stats = None
        self._set_missing(False)
        self._set_meta(path.name, str(path))
        cache_key = self._cache_key(path)
        pool = QThreadPool.globalInstance()
        if cache_key in self._cache:
            self._cancel_workers_except(pool=pool)
            result = self._cache.pop(cache_key)
            self._cache[cache_key] = result
            self.spinner.stop()
            self._display_result(path, result)
            return

        matching_token = next(
            (
                worker_token
                for worker_token, worker_key in self._worker_cache_keys.items()
                if worker_key == cache_key
            ),
            None,
        )
        if matching_token is not None:
            self._active_worker_token = matching_token
            self._cancel_workers_except(matching_token, pool)
            worker = self._workers.get(matching_token)
            if worker is not None:
                try:
                    was_queued = pool.tryTake(worker)
                except RuntimeError:
                    was_queued = False
                if was_queued:
                    pool.start(worker, 1)
            self._begin_loading(path)
            return

        self._cancel_workers_except(pool=pool)
        token = self._new_worker_token()
        self._active_worker_token = token
        worker = self._create_worker(token, path, cache_key)
        pool.start(worker, 1)
        self._begin_loading(path)

    def prefetch_file(self, path: Path | None) -> None:
        """Prepare one adjacent file at low priority without changing the view."""
        if self._disposed or path is None:
            return
        cache_key = self._cache_key(path)
        if cache_key in self._cache or cache_key in self._worker_cache_keys.values():
            return
        token = self._new_worker_token()
        worker = self._create_worker(token, path, cache_key)
        QThreadPool.globalInstance().start(worker, -1)

    def _new_worker_token(self) -> int:
        self._worker_sequence += 1
        return self._worker_sequence

    def _create_worker(
        self,
        token: int,
        path: Path,
        cache_key: tuple[object, ...],
    ) -> MediaLoadWorker:
        self.loading_label.setText(f"正在加载 {path.name}")
        worker = MediaLoadWorker(
            token,
            self.modality,
            path,
            self.point_limit,
            cloud_cam_offset=self.cloud_cam_offset,
            cloud_grid=self.cloud_grid,
            cloud_z_max=self.cloud_z_max,
            cloud_tau_rel=self.cloud_tau_rel,
            cloud_occlusion=self.cloud_occlusion,
            cloud_intrinsics=self.cloud_intrinsics,
            cloud_image_size=self.cloud_image_size,
            cloud_rotation=self.cloud_rotation,
            cloud_translation=self.cloud_translation,
            analysis_enabled=self._analysis_enabled,
        )
        worker.signals.loaded.connect(self._load_finished)
        worker.signals.failed.connect(self._load_failed)
        worker.signals.cancelled.connect(self._load_cancelled)
        self._workers[token] = worker
        self._worker_cache_keys[token] = cache_key
        return worker

    def _begin_loading(self, path: Path) -> None:
        self.loading_label.setText(f"正在加载 {path.name}")
        if self.stack.currentWidget() is self.loading_page:
            self.spinner.start()
        else:
            self.spinner.stop()
            self._loading_delay.start()

    def _cache_key(self, path: Path) -> tuple[object, ...]:
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = 0
        intrinsics_key = (
            tuple(np.asarray(self.cloud_intrinsics, dtype=float).reshape(-1))
            if self.modality == "ply" and self.cloud_intrinsics is not None
            else ()
        )
        return (
            str(path),
            modified,
            self.point_limit if self.modality == "ply" else 0,
            self.cloud_cam_offset if self.modality == "ply" else 0.0,
            self.cloud_grid if self.modality == "ply" else 0,
            self.cloud_dot_radius if self.modality == "ply" else 0.0,
            self.cloud_z_max if self.modality == "ply" else 0.0,
            self.cloud_tau_rel if self.modality == "ply" else 0.0,
            self.cloud_occlusion if self.modality == "ply" else False,
            self.cloud_image_size if self.modality == "ply" else None,
            intrinsics_key,
        )

    def _show_delayed_loading(self) -> None:
        if (
            self._active_worker_token is None
            or self._active_worker_token not in self._workers
        ):
            return
        if (
            self.modality == "ply"
            and self.cloud_canvas is not None
            and self.cloud_canvas._scatter is not None
        ):
            # Keep the last cloud visible while the next one is generated.
            # Replacing the whole viewport with a loading page looks like a
            # freeze even though the work is correctly running in background.
            current_name = self._current_path.name if self._current_path is not None else "点云"
            self._set_meta(f"{current_name}  ·  正在生成…", str(self._current_path or ""))
            return
        self.spinner.start()
        self.stack.setCurrentWidget(self.loading_page)

    def _remember(self, key: tuple[object, ...], result: object) -> None:
        old_cost = self._cache_costs.pop(key, 0)
        self._cache_bytes -= old_cost
        self._cache[key] = result
        self._cache.move_to_end(key)
        if isinstance(result, ImageData):
            cost = int(result.values.nbytes + result.image.sizeInBytes())
        elif isinstance(result, PointCloudData):
            cost = int(result.points.nbytes + result.colors.nbytes)
        else:
            cost = 1
        self._cache_costs[key] = cost
        self._cache_bytes += cost
        while self._cache_bytes > self._cache_limit_bytes and len(self._cache) > 1:
            evicted_key, _ = self._cache.popitem(last=False)
            self._cache_bytes -= self._cache_costs.pop(evicted_key, 0)

    def _cancel_workers_except(
        self,
        keep_token: int | None = None,
        pool: QThreadPool | None = None,
    ) -> None:
        pool = pool or QThreadPool.globalInstance()
        for old_token, old_worker in list(self._workers.items()):
            if old_token == keep_token:
                continue
            old_worker.cancel()
            try:
                removed = pool.tryTake(old_worker)
            except RuntimeError:
                removed = True  # Runnable already completed; its queued result is stale.
            if removed:
                self._workers.pop(old_token, None)
                self._worker_cache_keys.pop(old_token, None)

    def dispose(self) -> None:
        """Invalidate pending presentation work before a tile is removed."""
        self._disposed = True
        self.cancel_pending()
        self._cache.clear()
        self._cache_costs.clear()
        self._cache_bytes = 0
        self._active_analysis_token = None
        self._analysis_workers.clear()

    def cancel_pending(self) -> None:
        """Stop publishing work for a view that is no longer visible."""
        self._active_worker_token = None
        self._loading_delay.stop()
        self.spinner.stop()
        self._cancel_workers_except()
        self._workers.clear()
        self._worker_cache_keys.clear()
        if self.stack.currentWidget() is self.loading_page:
            if self.cloud_canvas is not None and self.cloud_canvas._scatter is not None:
                self.stack.setCurrentWidget(self.cloud_canvas)
            elif not self.image_canvas._item.pixmap().isNull():
                self.stack.setCurrentWidget(self.image_canvas)
            else:
                self.message.setText("等待加载")
                self.stack.setCurrentWidget(self.message)

    def _load_finished(self, token: int, result: object) -> None:
        self._workers.pop(token, None)
        cache_key = self._worker_cache_keys.pop(token, None)
        if cache_key is not None and not self._disposed:
            self._remember(cache_key, result)
        if (
            self._disposed
            or token != self._active_worker_token
            or self._current_path is None
        ):
            return
        self._active_worker_token = None
        path = self._current_path
        self._loading_delay.stop()
        self.spinner.stop()
        self._display_result(path, result)

    def _display_result(self, path: Path, result: object) -> None:
        try:
            if self.modality == "ply" and self.cloud_canvas is not None:
                if not isinstance(result, PointCloudData):
                    raise TypeError("点云加载结果格式无效")
                count = self.cloud_canvas.set_cloud_data(result)
                distance = (
                    f"  ·  Z≤{result.max_distance:g} m"
                    if result.max_distance is not None
                    else ""
                )
                tooltip = (
                    f"{path}\n有效点 {result.original_count:,} · "
                    f"过滤后 {result.filtered_count:,} · 显示 {count:,}"
                )
                self._set_meta(f"{path.name}  ·  {count:,} 点{distance}", tooltip)
                self.stack.setCurrentWidget(self.cloud_canvas)
                self.data_ready.emit(self.modality)
            else:
                image_data = result
                if not isinstance(image_data, ImageData):
                    raise TypeError("图片加载结果格式无效")
                self.image_data = image_data
                self.depth_stats = image_data.depth_stats if self.modality == "depth_fsd" else None
                image = (
                    render_image_values(image_data.values, **self._display_settings)
                    if self.modality == "depth_fsd"
                    else image_data.image
                )
                self.image_canvas.set_image(image)
                self._set_meta(f"{path.name}  ·  {image.width()}×{image.height()}", str(path))
                self.stack.setCurrentWidget(self.image_canvas)
                self._animate_image_arrival()
                self.data_ready.emit(self.modality)
                if self._analysis_enabled and image_data.image_stats is None:
                    self._start_analysis(path, image_data)
        except Exception as exc:
            self._show_load_error(path, str(exc))

    def _start_analysis(self, path: Path, image_data: ImageData) -> None:
        self._analysis_sequence += 1
        token = self._analysis_sequence
        self._active_analysis_token = token
        worker = MediaAnalysisWorker(
            token,
            image_data.values,
            include_depth=self.modality == "depth_fsd",
        )
        worker.signals.loaded.connect(
            lambda result_token, image_stats, depth_stats, source=path: self._analysis_finished(
                result_token,
                source,
                image_stats,
                depth_stats,
            )
        )
        self._analysis_workers[token] = worker
        QThreadPool.globalInstance().start(worker, -1)

    def _analysis_finished(
        self,
        token: int,
        path: Path,
        image_stats: ImageStats,
        depth_stats: DepthStats | None,
    ) -> None:
        self._analysis_workers.pop(token, None)
        analyzed: ImageData | None = None
        key = self._cache_key(path)
        cached = self._cache.get(key)
        if isinstance(cached, ImageData):
            analyzed = ImageData(
                cached.image,
                cached.values,
                image_stats,
                depth_stats,
            )
            self._remember(key, analyzed)
        if (
            self._disposed
            or token != self._active_analysis_token
            or self._current_path != path
            or self.image_data is None
        ):
            return
        self._active_analysis_token = None
        self.image_data = analyzed or ImageData(
            self.image_data.image,
            self.image_data.values,
            image_stats,
            depth_stats,
        )
        self.depth_stats = depth_stats if self.modality == "depth_fsd" else None
        self.data_ready.emit(self.modality)

    def _load_failed(self, token: int, error: str) -> None:
        self._workers.pop(token, None)
        self._worker_cache_keys.pop(token, None)
        if (
            self._disposed
            or token != self._active_worker_token
            or self._current_path is None
        ):
            return
        self._active_worker_token = None
        self._loading_delay.stop()
        self.spinner.stop()
        self._show_load_error(self._current_path, error)

    def _load_cancelled(self, token: int) -> None:
        self._workers.pop(token, None)
        self._worker_cache_keys.pop(token, None)
        if token == self._active_worker_token:
            self._active_worker_token = None
            self._loading_delay.stop()
            self.spinner.stop()

    def _show_load_error(self, path: Path, error: str) -> None:
        self._set_missing(True)
        self.message.setText(f"无法预览\n{path.name}\n\n{error}")
        self.stack.setCurrentWidget(self.message)

    def _animate_image_arrival(self) -> None:
        if self._content_animation is not None:
            self._content_animation.stop()
        effect = QGraphicsOpacityEffect(self.image_canvas)
        self.image_canvas.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self.image_canvas)
        animation.setDuration(110)
        animation.setStartValue(0.9)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda: self.image_canvas.setGraphicsEffect(None))
        self._content_animation = animation
        animation.start()

    def set_focused(self, focused: bool) -> None:
        if bool(self.property("focused")) == focused:
            return
        self.setProperty("focused", focused)
        self.focus_button.setText("退出聚焦" if focused else "聚焦")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_display_settings(self, **settings: object) -> None:
        self._display_settings.update(settings)
        if self.image_data is None:
            return
        image = render_image_values(self.image_data.values, **self._display_settings)
        self.image_canvas.set_image(image, preserve_view=True)

    def set_analysis_enabled(self, enabled: bool) -> None:
        self._analysis_enabled = bool(enabled)
        if not enabled and self.modality == "depth_fsd":
            self.depth_stats = None
        if not enabled:
            self._active_analysis_token = None

    def refresh_analysis(self) -> None:
        if self._analysis_enabled and self.image_data is not None:
            if self._current_path is not None:
                self._start_analysis(self._current_path, self.image_data)

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self.header
            and event.type() == QEvent.Type.MouseButtonDblClick
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.focus_requested.emit(self.modality)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def reset_view(self) -> None:
        if self.stack.currentWidget() is self.image_canvas:
            self.image_canvas.reset_view()
        elif self.cloud_canvas is not None and self.stack.currentWidget() is self.cloud_canvas:
            self.cloud_canvas.reset_view()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._meta_text:
            self._set_meta(self._meta_text, self.meta.toolTip())


class DropHint(QFrame):
    open_requested = Signal()

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
