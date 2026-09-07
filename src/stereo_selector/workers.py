"""Background task workers and per-task-type thread pools.

The UI keeps the global Qt thread pool for framework-internal work; all
application loads, analyses and scans run on dedicated pools so a heavy point
cloud cannot starve image decoding or statistics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage

from .calibration import CalibrationData
from .media import (
    analyze_depth,
    analyze_image,
    load_image_data,
    load_point_cloud,
)
from .models import DatasetScanner, discover_project_roots

logger = logging.getLogger(__name__)


IMAGE_POOL = QThreadPool()
CLOUD_POOL = QThreadPool()
ANALYSIS_POOL = QThreadPool()
SCAN_POOL = QThreadPool()
RENDER_POOL = QThreadPool()


def configure_pools() -> None:
    """Size the per-task-type pools. Called once at application startup."""
    IMAGE_POOL.setMaxThreadCount(2)
    CLOUD_POOL.setMaxThreadCount(2)
    ANALYSIS_POOL.setMaxThreadCount(1)
    SCAN_POOL.setMaxThreadCount(1)
    RENDER_POOL.setMaxThreadCount(2)


def load_pool_for(modality: str) -> QThreadPool:
    """Return the pool used for media loading of the given modality."""
    return CLOUD_POOL if modality == "ply" else IMAGE_POOL


def wait_for_all(timeout_ms: int = 5000) -> bool:
    """Wait for every application pool to drain (used by tests and shutdown)."""
    settled = True
    for pool in (IMAGE_POOL, CLOUD_POOL, ANALYSIS_POOL, SCAN_POOL, RENDER_POOL):
        settled = pool.waitForDone(timeout_ms) and settled
    return settled


class MediaWorkerSignals(QObject):
    loaded = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)


class PreviewWorker(QRunnable):
    """CPU preview rendering without QWidget/QPixmap access."""

    def __init__(self, token: int, values: np.ndarray, settings: dict, scale: int) -> None:
        super().__init__()
        self.token, self.values, self.settings, self.scale = token, values, settings, scale
        self.signals = MediaWorkerSignals()

    def run(self) -> None:
        from .media import downsample_pool, render_image_values
        try:
            values = downsample_pool(self.values, self.scale) if self.scale > 1 else self.values
            result = render_image_values(values, **self.settings)
            self.signals.loaded.emit(self.token, result)
        except Exception as exc:
            self.signals.failed.emit(self.token, str(exc))


class RoiWorker(QRunnable):
    def __init__(self, token: int, values: np.ndarray) -> None:
        super().__init__()
        self.token, self.values = token, values
        self.signals = MediaWorkerSignals()

    def run(self) -> None:
        try:
            self.signals.loaded.emit(self.token, analyze_image(self.values))
        except Exception as exc:
            self.signals.failed.emit(self.token, str(exc))


class AnalysisWorkerSignals(QObject):
    loaded = Signal(int, object, object)
    failed = Signal(int, str)


class MediaAnalysisWorker(QRunnable):
    """Compute image/depth statistics without blocking the UI thread."""

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
        try:
            image_stats = analyze_image(self.values)
            depth_stats = (
                analyze_depth(self.values) if self.include_depth else None
            )
        except Exception as exc:
            try:
                self.signals.failed.emit(self.token, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self.signals.loaded.emit(self.token, image_stats, depth_stats)
        except RuntimeError:
            pass


class MediaLoadWorker(QRunnable):
    """Decode an image or generate a point cloud in the background."""

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
            if self._cancelled:
                self._publish_cancelled()
                return
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
            pass

    def _publish_cancelled(self) -> None:
        try:
            self.signals.cancelled.emit(self.token)
        except RuntimeError:
            pass


class ProjectScanSignals(QObject):
    progress = Signal(str, int, int)
    discovered = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class ProjectScanWorker(QRunnable):
    """Discover project roots and scan a project without blocking the UI thread.

    With ``discover=True`` the worker first inspects the selected folder for a
    project collection (several sibling captures). The list of roots is
    published through ``discovered`` and the first root is scanned with the
    automatic matching. The UI rescans with a saved manual mapping only when
    one exists for that root, so the common case costs a single pass.
    """

    def __init__(
        self,
        root: Path,
        manual_dirs: dict[str, Path] | None = None,
        force_order: bool = False,
        *,
        discover: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.manual_dirs = manual_dirs
        self.force_order = force_order
        self.discover = discover
        self.signals = ProjectScanSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        scanner = DatasetScanner()
        root = self.root
        try:
            if self.discover:
                self._report("正在识别项目…", 0, 0)
                roots = discover_project_roots(root)
                if self._cancelled:
                    raise InterruptedError("项目扫描已取消")
                try:
                    self.signals.discovered.emit(roots)
                except RuntimeError:
                    pass
                if len(roots) > 1:
                    root = roots[0]
            dataset = scanner.scan(
                root,
                manual_dirs=self.manual_dirs,
                force_order=self.force_order,
                progress=self._report,
                cancel_check=lambda: self._cancelled,
            )
        except InterruptedError:
            try:
                self.signals.cancelled.emit()
            except RuntimeError:
                pass
            return
        except Exception as exc:
            logger.exception("项目扫描失败：%s", root)
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass
            return
        try:
            self.signals.finished.emit(dataset)
        except RuntimeError:
            pass

    def _report(self, text: str, done: int, total: int) -> None:
        try:
            self.signals.progress.emit(text, done, total)
        except RuntimeError:
            pass


class RectifySignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)


class RectifyWorker(QRunnable):
    """Rectify a stereo pair with OpenCV without blocking the UI thread."""

    def __init__(
        self,
        token: int,
        left: QImage,
        right: QImage,
        calibration: CalibrationData,
    ) -> None:
        super().__init__()
        self.token = token
        self.left = left
        self.right = right
        self.calibration = calibration
        self.signals = RectifySignals()

    def run(self) -> None:
        from .inspection import rectify_stereo_images

        try:
            images = rectify_stereo_images(
                self.left,
                self.right,
                self.calibration,
            )
        except Exception as exc:
            try:
                self.signals.failed.emit(self.token, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self.signals.finished.emit(self.token, images)
        except RuntimeError:
            pass


class DifferenceSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)


class DifferenceWorker(QRunnable):
    """Compute the absolute difference image without blocking the UI thread."""

    def __init__(
        self,
        token: int,
        left: QImage,
        right: QImage,
        rgb_array: Callable[[QImage], np.ndarray],
    ) -> None:
        super().__init__()
        self.token = token
        self.left = left
        self.right = right
        self.rgb_array = rgb_array
        self.signals = DifferenceSignals()

    def run(self) -> None:
        try:
            left_array = self.rgb_array(self.left)
            right_array = self.rgb_array(self.right)
            if right_array.shape != left_array.shape:
                height, width = left_array.shape[:2]
                import cv2

                right_array = cv2.resize(
                    right_array,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
            difference = np.abs(
                left_array.astype(np.int16) - right_array.astype(np.int16)
            ).astype(np.uint8)
            difference = np.ascontiguousarray(difference)
            height, width, channels = difference.shape
            image = QImage(
                difference.data,
                width,
                height,
                width * channels,
                QImage.Format.Format_RGB888,
            ).copy()
        except Exception as exc:
            try:
                self.signals.failed.emit(self.token, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self.signals.finished.emit(self.token, image)
        except RuntimeError:
            pass
