from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QPainter, QPen, QPixmap, QVector3D, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .media import PointCloudData, load_image, load_point_cloud
from .models import modality_label


MODALITY_ACCENTS = {
    "left": "#4ea1ff",
    "right": "#a78bfa",
    "depth_fsd": "#f0a35e",
    "depth_color": "#e76f9b",
    "ply": "#56c596",
}


class MediaWorkerSignals(QObject):
    loaded = Signal(int, object)
    failed = Signal(int, str)


class MediaLoadWorker(QRunnable):
    def __init__(self, token: int, modality: str, path: Path, point_limit: int) -> None:
        super().__init__()
        self.token = token
        self.modality = modality
        self.path = path
        self.point_limit = point_limit
        self.signals = MediaWorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            result = (
                load_point_cloud(self.path, self.point_limit)
                if self.modality == "ply"
                else load_image(self.path)
            )
        except Exception as exc:
            if not self._cancelled:
                try:
                    self.signals.failed.emit(self.token, str(exc))
                except RuntimeError:
                    pass  # The window was destroyed while this task was finishing.
            return
        if self._cancelled:
            return
        try:
            self.signals.loaded.emit(self.token, result)
        except RuntimeError:
            pass  # The receiver/source may be gone during application shutdown.


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
        color = QColor("#ffffff") if self.control == "close" and self.underMouse() else self.palette().buttonText().color()
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


class ImageCanvas(QGraphicsView):
    """Image viewport with editor-like zoom, pan and fit behavior."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self.scene().addItem(self._item)
        self.setBackgroundBrush(QBrush(QColor("#0d0d0d")))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self._user_zoomed = False

    def set_image(self, image) -> None:
        self._item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(self._item.boundingRect())
        self.reset_view()

    def reset_view(self) -> None:
        self.resetTransform()
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self._user_zoomed = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._item.pixmap().isNull():
            return
        factor = 1.16 if event.angleDelta().y() > 0 else 1 / 1.16
        next_scale = self.transform().m11() * factor
        if 0.02 <= next_scale <= 80:
            self.scale(factor, factor)
            self._user_zoomed = True
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.reset_view()


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
        layout.setContentsMargins(11, 0, 0, 0)
        layout.setSpacing(8)

        mark = QLabel("SS")
        mark.setObjectName("titleMark")
        mark.setFixedSize(23, 23)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Stereo Selector")
        title.setObjectName("titleBrand")
        self.context = QLabel("图片筛选工作区")
        self.context.setObjectName("titleContext")
        self.context.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.settings_button = QPushButton("设置")
        self.settings_button.setObjectName("titleActionButton")
        self.settings_button.setToolTip("打开设置")
        self.settings_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.minimize_button = WindowControlButton("minimize")
        self.minimize_button.setToolTip("最小化")
        self.maximize_button = WindowControlButton("maximize")
        self.maximize_button.setToolTip("最大化")
        self.close_button = WindowControlButton("close")
        self.close_button.setToolTip("关闭")
        for button in (self.minimize_button, self.maximize_button, self.close_button):
            button.setFixedSize(46, 39)
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
        layout.addWidget(self.settings_button)
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)

    def set_context(self, text: str) -> None:
        self.context.setText(text)

    def set_maximized(self, maximized: bool) -> None:
        self.maximize_button.set_maximized(maximized)
        self.maximize_button.setToolTip("还原" if maximized else "最大化")

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
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._scatter = None
        self._center = np.zeros(3, dtype=np.float32)
        self._distance = 10.0
        try:
            import pyqtgraph.opengl as gl

            self._gl = gl
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor("#0d0d0d")
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
            raise ValueError("点云中没有有效顶点")
        if self._scatter is not None:
            self.view.removeItem(self._scatter)
        self._scatter = self._gl.GLScatterPlotItem(
            pos=cloud.points,
            color=cloud.colors,
            size=2.2,
            pxMode=True,
        )
        self.view.addItem(self._scatter)
        low = np.percentile(cloud.points, 2, axis=0)
        high = np.percentile(cloud.points, 98, axis=0)
        self._center = ((low + high) / 2.0).astype(np.float32)
        self._distance = max(float(np.max(high - low)) * 1.25, 0.1)
        self.reset_view()
        return cloud.original_count

    def reset_view(self) -> None:
        if self.view is None:
            return
        self.view.setCameraPosition(
            pos=QVector3D(float(self._center[0]), float(self._center[1]), float(self._center[2])),
            distance=self._distance,
            elevation=0,
            azimuth=90,
        )


class MediaTile(QFrame):
    focus_requested = Signal(str)

    def __init__(self, modality: str, point_limit: int = 300_000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.modality = modality
        self.point_limit = point_limit
        self._load_token = 0
        self._workers: dict[int, MediaLoadWorker] = {}
        self._worker_cache_keys: dict[int, tuple[str, int, int]] = {}
        self._cache: OrderedDict[tuple[str, int, int], object] = OrderedDict()
        self._current_path: Path | None = None
        self._meta_text = ""
        self._disposed = False
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

        header = QFrame()
        header.setObjectName("mediaHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(11, 7, 8, 7)
        header_layout.setSpacing(8)
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
        outer.addWidget(header)

        self.stack = QStackedWidget()
        self.image_canvas = ImageCanvas()
        self.cloud_canvas = PointCloudCanvas() if modality == "ply" else None
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
        self._load_token += 1
        token = self._load_token
        self._current_path = path
        self._loading_delay.stop()
        if path is None:
            self.spinner.stop()
            self._set_missing(True)
            self._set_meta("未匹配", "")
            self.message.setText("当前样本没有匹配到此类型文件")
            self.stack.setCurrentWidget(self.message)
            return
        self._set_missing(False)
        self._set_meta(path.name, str(path))
        cache_key = self._cache_key(path)
        if cache_key in self._cache:
            result = self._cache.pop(cache_key)
            self._cache[cache_key] = result
            self.spinner.stop()
            self._display_result(path, result)
            return
        self.loading_label.setText(f"正在加载 {path.name}")
        pool = QThreadPool.globalInstance()
        self._cancel_queued_workers(pool)
        worker = MediaLoadWorker(token, self.modality, path, self.point_limit)
        worker.signals.loaded.connect(self._load_finished)
        worker.signals.failed.connect(self._load_failed)
        self._workers[token] = worker
        self._worker_cache_keys[token] = cache_key
        pool.start(worker)
        if self.stack.currentWidget() is self.loading_page:
            self.spinner.start()
        else:
            self.spinner.stop()
            self._loading_delay.start()

    def _cache_key(self, path: Path) -> tuple[str, int, int]:
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = 0
        return str(path), modified, self.point_limit if self.modality == "ply" else 0

    def _show_delayed_loading(self) -> None:
        if self._load_token not in self._workers:
            return
        self.spinner.start()
        self.stack.setCurrentWidget(self.loading_page)

    def _remember(self, key: tuple[str, int, int], result: object) -> None:
        self._cache[key] = result
        self._cache.move_to_end(key)
        capacity = 2 if self.modality == "ply" else 4
        while len(self._cache) > capacity:
            self._cache.popitem(last=False)

    def _cancel_queued_workers(self, pool: QThreadPool | None = None) -> None:
        pool = pool or QThreadPool.globalInstance()
        for old_token, old_worker in list(self._workers.items()):
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

    def cancel_pending(self) -> None:
        """Stop publishing work for a view that is no longer visible."""
        self._load_token += 1
        self._loading_delay.stop()
        self.spinner.stop()
        for worker in self._workers.values():
            try:
                worker.cancel()
            except RuntimeError:
                pass
        self._cancel_queued_workers()
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
        if self._disposed or token != self._load_token or self._current_path is None:
            return
        path = self._current_path
        self._loading_delay.stop()
        self.spinner.stop()
        self._display_result(path, result)

    def _display_result(self, path: Path, result: object) -> None:
        try:
            if self.modality == "ply" and self.cloud_canvas is not None:
                count = self.cloud_canvas.set_cloud_data(result)
                self._set_meta(f"{path.name}  ·  {count:,} 点", str(path))
                self.stack.setCurrentWidget(self.cloud_canvas)
            else:
                image = result
                self.image_canvas.set_image(image)
                self._set_meta(f"{path.name}  ·  {image.width()}×{image.height()}", str(path))
                self.stack.setCurrentWidget(self.image_canvas)
        except Exception as exc:
            self._show_load_error(path, str(exc))

    def _load_failed(self, token: int, error: str) -> None:
        self._workers.pop(token, None)
        self._worker_cache_keys.pop(token, None)
        if self._disposed or token != self._load_token or self._current_path is None:
            return
        self._loading_delay.stop()
        self.spinner.stop()
        self._show_load_error(self._current_path, error)

    def _show_load_error(self, path: Path, error: str) -> None:
        self._set_missing(True)
        self.message.setText(f"无法预览\n{path.name}\n\n{error}")
        self.stack.setCurrentWidget(self.message)

    def set_focused(self, focused: bool) -> None:
        if bool(self.property("focused")) == focused:
            return
        self.setProperty("focused", focused)
        self.focus_button.setText("退出聚焦" if focused else "聚焦")
        self.style().unpolish(self)
        self.style().polish(self)

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
        layout.setContentsMargins(28, 28, 28, 28)

        mark = QLabel("S")
        mark.setObjectName("emptyMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(56, 56)
        title = QLabel("打开一个双目图像项目")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("将项目文件夹拖到窗口，或从本机选择文件夹")
        hint.setObjectName("emptyHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        open_button = QPushButton("选择项目文件夹")
        open_button.setObjectName("primaryButton")
        open_button.clicked.connect(self.open_requested)
        layout.addWidget(mark, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(16)
        layout.addWidget(title)
        layout.addSpacing(5)
        layout.addWidget(hint)
        layout.addSpacing(18)
        layout.addWidget(open_button, 0, Qt.AlignmentFlag.AlignHCenter)


class NoViewsHint(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("noViewsHint")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("尚未选择对比视图")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
