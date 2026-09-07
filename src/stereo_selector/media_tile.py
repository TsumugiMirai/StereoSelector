"""Media tile widget: asynchronous loading, caching and per-modality canvas."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QEvent,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .media import (
    DepthStats,
    ImageData,
    ImageStats,
    PointCloudData,
    downsample_pool,
    render_image_values,
)
from .models import modality_label
from .ui_metrics import PREVIEW_MAX_DIMENSION, TILE_MIN_HEIGHT, TILE_MIN_WIDTH
from .widgets import MODALITY_ACCENTS, ImageCanvas, LoadingSpinner, PointCloudCanvas
from .workers import ANALYSIS_POOL, RENDER_POOL, MediaAnalysisWorker, MediaLoadWorker, PreviewWorker, load_pool_for


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
        canvas_background: str = "#0d0d0d",
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
        self.canvas_background = canvas_background
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
        self._analysis_cache_keys: dict[int, tuple[object, ...]] = {}
        self._active_analysis_token: int | None = None
        self._cache: OrderedDict[tuple[object, ...], object] = OrderedDict()
        self._cache_costs: dict[tuple[object, ...], int] = {}
        self._cache_bytes = 0
        self._cache_limit_bytes = 192 * 1024 * 1024 if modality == "ply" else 96 * 1024 * 1024
        self._current_path: Path | None = None
        self._current_key: tuple[object, ...] | None = None
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
        self._preview_sequence = 0
        self._preview_worker = None
        self._preview_pending = None
        self._fit_mode = "fit"
        self._presented_full_resolution = False
        self._displayed_key = None
        self._loading_delay = QTimer(self)
        self._loading_delay.setSingleShot(True)
        self._loading_delay.setInterval(220)
        self._loading_delay.timeout.connect(self._show_delayed_loading)
        self.setObjectName("mediaTile")
        self.setProperty("missing", False)
        self.setMinimumSize(TILE_MIN_WIDTH, TILE_MIN_HEIGHT)

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
        self.image_canvas.set_background_color(canvas_background)
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
        self.image_canvas.fit_requested.connect(self.set_fit_mode)
        self.cloud_canvas = (
            PointCloudCanvas(
                cam_offset=self.cloud_cam_offset,
                dot_radius=self.cloud_dot_radius,
                z_max=self.cloud_z_max,
                intrinsics=self.cloud_intrinsics,
                image_size=self.cloud_image_size,
                background=self.canvas_background,
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
        # One stat() per frame: the key is reused by presentation and analysis.
        cache_key = self._cache_key(path) if path is not None else None
        if (path is not None and path == self._current_path
                and self._displayed_key == cache_key
                and not self.is_loading() and self.showing_canvas()):
            return
        self._preview_sequence += 1
        self._preview_pending = None
        self._fit_mode = "fit"
        self._presented_full_resolution = False
        self._displayed_key = None
        self._active_worker_token = None
        self._current_path = path
        self._current_key = cache_key
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
        pool = load_pool_for(self.modality)
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
                and not self._workers[worker_token]._cancelled
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
        load_pool_for(self.modality).start(worker, -1)

    def is_loading(self) -> bool:
        """True while a load for the currently requested path is in flight."""
        return self._active_worker_token is not None or self._preview_pending is not None

    def has_pending_work(self) -> bool:
        """True when any worker is queued/running or a delayed spinner is armed."""
        return bool(self._workers) or self._loading_delay.isActive() or self._preview_pending is not None

    def loading_delay_active(self) -> bool:
        """True when the delayed loading indicator has not yet fired."""
        return self._loading_delay.isActive()

    @property
    def current_path(self) -> Path | None:
        """The path currently requested for display, if any."""
        return self._current_path

    def showing_canvas(self) -> bool:
        """True when the tile displays its image/point-cloud canvas rather than a placeholder."""
        expected = self.cloud_canvas if self.modality == "ply" else self.image_canvas
        return expected is not None and self.stack.currentWidget() is expected

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
            tuple(np.asarray(self.cloud_rotation, dtype=float).reshape(-1))
            if self.modality == "ply" and self.cloud_rotation is not None else (),
            tuple(np.asarray(self.cloud_translation, dtype=float).reshape(-1))
            if self.modality == "ply" and self.cloud_translation is not None else (),
        )

    def _key_for(self, path: Path) -> tuple[object, ...]:
        """Cache key of ``path``; reuses the key computed when it became current."""
        if path == self._current_path and self._current_key is not None:
            return self._current_key
        return self._cache_key(path)

    def _show_delayed_loading(self) -> None:
        if (
            self._active_worker_token is None
            or self._active_worker_token not in self._workers
        ):
            return
        if (
            (self.cloud_canvas is not None and self.cloud_canvas._scatter is not None)
            or not self.image_canvas._item.pixmap().isNull()
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
        pool = pool or load_pool_for(self.modality)
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
        self._analysis_cache_keys.clear()

    def cancel_pending(self) -> None:
        """Stop publishing work for a view that is no longer visible."""
        self._active_worker_token = None
        self._preview_sequence += 1
        self._preview_pending = None
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
                self._displayed_key = self._key_for(path)
                self.data_ready.emit(self.modality)
            else:
                image_data = result
                if not isinstance(image_data, ImageData):
                    raise TypeError("图片加载结果格式无效")
                self.image_data = image_data
                self.depth_stats = image_data.depth_stats if self.modality == "depth_fsd" else None
                settings = self._display_settings
                adjusted = (self.modality == "depth_fsd" or settings["brightness"] != 0
                            or settings["contrast"] != 1 or settings["gamma"] != 1
                            or settings["exposure"] != 0 or settings["highlight_invalid"]
                            or settings["display_range"] is not None or settings["color_map"] != "gray")
                if adjusted:
                    self._request_preview(preserve_view=False)
                else:
                    self._present_image(image_data.image, preserve_view=False)
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
        worker.signals.failed.connect(self._analysis_failed)
        self._analysis_workers[token] = worker
        self._analysis_cache_keys[token] = self._key_for(path)
        ANALYSIS_POOL.start(worker, -1)

    def _analysis_failed(self, token: int, error: str) -> None:
        self._analysis_workers.pop(token, None)
        self._analysis_cache_keys.pop(token, None)
        if token != self._active_analysis_token:
            return
        self._active_analysis_token = None
        if self.modality == "depth_fsd":
            self.depth_stats = None
        self._set_meta(
            self._meta_text,
            f"{self._meta_text}\n统计失败：{error}".strip(),
        )

    def _analysis_finished(
        self,
        token: int,
        path: Path,
        image_stats: ImageStats,
        depth_stats: DepthStats | None,
    ) -> None:
        self._analysis_workers.pop(token, None)
        analyzed: ImageData | None = None
        key = self._analysis_cache_keys.pop(token, None)
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
            or key != self._key_for(path)
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
        self.image_data = None
        self.depth_stats = None
        self._displayed_key = None
        self._set_missing(True)
        self.message.setText(f"无法预览\n{path.name}\n\n{error}")
        self.stack.setCurrentWidget(self.message)
        self.data_ready.emit(self.modality)

    def _present_image(self, image, *, preserve_view: bool, fit_mode: str | None = None) -> None:
        if self._current_path is None or self._disposed:
            return
        self.image_canvas.set_image(image, preserve_view=preserve_view)
        path = self._current_path
        height, width = self.image_data.values.shape[:2] if self.image_data is not None else (image.height(), image.width())
        self._presented_full_resolution = image.width() == width and image.height() == height
        if fit_mode == "actual":
            self.image_canvas.show_actual_size()
        elif fit_mode == "width":
            self.image_canvas.fit_width()
        elif fit_mode == "fit":
            self.image_canvas.reset_view()
        self._set_meta(f"{path.name}  ·  {width}×{height}", str(path))
        self._set_missing(False)
        self.stack.setCurrentWidget(self.image_canvas)
        self._displayed_key = self._key_for(path)
        self.data_ready.emit(self.modality)

    @property
    def display_settings(self) -> dict[str, object]:
        """Current per-tile display parameters (read-only copy)."""
        return dict(self._display_settings)

    def _request_preview(self, *, preserve_view: bool, fit_mode: str | None = None) -> None:
        if self.image_data is None or self._disposed:
            return
        if fit_mode is None and self._preview_pending is not None:
            # A display adjustment made while 1:1 is loading must keep that
            # view request while replacing only the render parameters.
            fit_mode = self._preview_pending[5]
        self._preview_sequence += 1
        scale = 1 if self._fit_mode == "actual" else self._preview_scale(self.image_data.values)
        self._preview_pending = (self._preview_sequence, self.image_data.values,
                                 dict(self._display_settings), preserve_view, scale, fit_mode)
        self._start_preview()

    def _start_preview(self) -> None:
        if self._preview_worker is not None or self._preview_pending is None or self._disposed:
            return
        token, values, settings, _preserve, scale, _fit_mode = self._preview_pending
        worker = PreviewWorker(token, values, settings, scale)
        worker.signals.loaded.connect(self._preview_finished)
        worker.signals.failed.connect(self._preview_failed)
        self._preview_worker = worker
        RENDER_POOL.start(worker)

    def _preview_finished(self, token: int, image: object) -> None:
        self._preview_worker = None
        if token == self._preview_sequence and self._preview_pending is not None and not self._disposed:
            preserve_view = self._preview_pending[3]
            fit_mode = self._preview_pending[5]
            self._preview_pending = None
            self._present_image(image, preserve_view=preserve_view, fit_mode=fit_mode)
        self._start_preview()

    def _preview_failed(self, token: int, error: str) -> None:
        self._preview_worker = None
        if token == self._preview_sequence:
            self._preview_pending = None
            if not self._disposed:
                # A failed adjustment must not leave the first frame stuck on
                # the loading page or keep its inspector disabled forever.
                if self.image_data is not None:
                    self._present_image(self.image_data.image, preserve_view=True)
                self._loading_delay.stop()
                self.spinner.stop()
                self._set_meta("显示调整失败", error)
        self._start_preview()

    def set_focused(self, focused: bool) -> None:
        if bool(self.property("focused")) == focused:
            return
        self.setProperty("focused", focused)
        self.focus_button.setText("退出聚焦" if focused else "聚焦")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_canvas_background(self, color: str) -> None:
        """Keep both image and point-cloud canvases in sync with the theme."""
        self.canvas_background = color
        self.image_canvas.set_background_color(color)
        if self.cloud_canvas is not None:
            self.cloud_canvas.set_background(color)

    def set_display_settings(self, **settings: object) -> None:
        self._display_settings.update(settings)
        if self.image_data is None:
            return
        self._request_preview(preserve_view=True)

    def set_fit_mode(self, mode: str) -> None:
        """Resolve 1:1 against source pixels, rendering large images off-thread."""
        if self.image_data is None or self._disposed:
            return
        mode = mode if mode in {"actual", "width"} else "fit"
        self._fit_mode = mode
        if mode == "actual":
            if self._presented_full_resolution and self._preview_pending is None:
                self.image_canvas.show_actual_size()
            else:
                self._request_preview(preserve_view=False, fit_mode=mode)
                self._set_meta("正在加载原始分辨率…", str(self._current_path or ""))
            return
        # Supersede any outstanding 1:1 render. Its result cannot reset a newer
        # fit/width request, but pending display adjustments are still applied.
        if self._preview_pending is not None:
            self._request_preview(preserve_view=False, fit_mode=mode)
        if mode == "width":
            self.image_canvas.fit_width()
        else:
            self.image_canvas.reset_view()

    def _render_preview(self) -> object:
        """Render depth values, downsampled for large images to keep sliders fast."""
        if self.image_data is None:
            return None
        values = self.image_data.values
        scale = self._preview_scale(values)
        if scale > 1:
            values = downsample_pool(values, scale)
        return render_image_values(values, **self._display_settings)

    @staticmethod
    def _preview_scale(values: np.ndarray) -> int:
        height, width = np.asarray(values).shape[:2]
        maximum = max(height, width)
        return max(1, (maximum + PREVIEW_MAX_DIMENSION - 1) // PREVIEW_MAX_DIMENSION)

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
            self.set_fit_mode("fit")
        elif self.cloud_canvas is not None and self.stack.currentWidget() is self.cloud_canvas:
            self.cloud_canvas.reset_view()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._meta_text:
            self._set_meta(self._meta_text, self.meta.toolTip())


