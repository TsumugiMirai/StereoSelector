from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import math
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPropertyAnimation, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .calibration import (
    RECTIFY_MODE_CHOICES,
    CalibrationData,
    CalibrationOption,
    builtin_calibration_options,
    custom_calibration_option,
    discover_project_calibrations,
    epiline_for_point,
    load_calibration,
    normalize_rectify_mode,
)
from .inspection import CloudProjectionDialog, StereoOverlayDialog
from .inspector import InspectorPanel
from .mapping import MappingDialog
from .media import project_camera_points, render_image_values
from .models import (
    MODALITY_INFO,
    Dataset,
    modality_label,
    summarize_missing,
)
from .playback_controller import PlaybackController
from .settings import AppPreferences, SettingsDialog
from .settings_controller import SettingsController
from .theme import PALETTES, palette_for, style_for
from .ui_controls import ElidedLabel
from .ui_layout import build_inspector, build_sidebar, build_workspace
from .ui_metrics import (
    ACTIVITY_BAR_WIDTH,
    ICON_PANE_SIZE,
    INSPECTION_BAR_CURSOR_HEIGHT,
    INSPECTION_BAR_HEIGHT,
    MEDIA_COMPACT_MARGIN,
    MEDIA_COMPACT_SPACING,
    MEDIA_MARGIN,
    MEDIA_SPACING,
    SIDEBAR_EXPANDED_WIDTH,
    SIDEBAR_MAX_WIDTH,
    SIDEBAR_MIN_WIDTH,
    SPLITTER_INSPECTOR_WIDTH,
    SPLITTER_SIDEBAR_WIDTH,
    SPLITTER_WORKSPACE_WIDTH,
    WINDOW_DEFAULT_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from .units import depth_is_valid, depth_to_meters, format_depth, resolve_depth_unit
from .widgets import (
    CommandPalette,
    MediaTile,
    PaneToggleButton,
    TitleBar,
    ToolTipManager,
)
from .workers import IMAGE_POOL, SCAN_POOL, ProjectScanWorker, RectifyWorker, configure_pools

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow, PlaybackController, SettingsController):
    def __init__(self, initial_project: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Stereo Selector")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowSystemMenuHint
        )
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setAcceptDrops(True)

        self.settings = QSettings("ToolBox", "StereoSelector")
        self.preferences = AppPreferences.load(self.settings)
        self.dataset: Dataset | None = None
        self.current_root: Path | None = None
        self.project_collection_root: Path | None = None
        self.project_roots: list[Path] = []
        self._project_picker_guard = False
        self.manual_dirs: dict[str, Path] = {}
        self.force_order = False
        self.current_index = 0
        self.calibration: CalibrationData | None = None
        self.current_calibration_id = ""
        self.rectify_mode = "air"
        self.calibration_options = self._load_calibration_options()
        self.focused_modality: str | None = None
        self._crosshair_position: tuple[float, float] | None = None
        self._pending_cursor_update: tuple[str, float, float] | None = None
        self._cursor_update_timer = QTimer(self)
        self._cursor_update_timer.setSingleShot(True)
        self._cursor_update_timer.setInterval(16)
        self._cursor_update_timer.timeout.connect(self._drain_cursor_update)
        self.checkboxes: dict[str, QCheckBox] = {}
        self.count_labels: dict[str, QLabel] = {}
        self.tiles: dict[str, MediaTile] = {}
        self.tile_pool: dict[str, MediaTile] = {}
        self.shortcuts: list[QShortcut] = []
        self.action_shortcuts: dict[str, QShortcut] = {}
        self._native_frame_applied = False
        self._sidebar_animation: QPropertyAnimation | None = None
        self._sidebar_expanded_width = SIDEBAR_EXPANDED_WIDTH
        self._sidebar_collapsed = True
        self._active_sidebar_page = "data"
        self._topbar_animation: QPropertyAnimation | None = None
        self._topbar_expanded_height = INSPECTION_BAR_HEIGHT
        self._topbar_collapsed = False
        self._playback_fps = 2.0
        self._resume_playback_after_scrub = False
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(500)
        self._playback_timer.timeout.connect(self._playback_tick)
        self._settings_page: SettingsDialog | None = None
        self._settings_previous_context: tuple[str, bool] = ("", False)
        self._layout_columns = 0
        self._chrome_hidden = False
        self._chrome_visibility: dict[str, bool] = {}
        self._inspector_source = ""
        self._cloud_clip_ranges: dict[str, tuple[float, float]] | None = None
        self._rectified_cache: dict[tuple[str, str, str], tuple[tuple[object, object], str]] = {}
        self._rectify_sequence = 0
        self._rectify_token: int | None = None
        self._rectify_key: tuple[str, str, str] | None = None
        self._rectify_worker: RectifyWorker | None = None
        self._scan_in_progress = False
        self._resolution_warnings: set[tuple[str, int, int]] = set()

        self._build_ui()
        self.tooltip_manager = ToolTipManager(self)
        QApplication.instance().installEventFilter(self.tooltip_manager)
        self._build_shortcuts()
        self._apply_preferences()
        self._update_controls()
        self._refresh_recent_projects()
        if sys.platform != "win32":
            # Frameless windows only get the native resize frame on Windows;
            # other platforms use QWindow.startSystemResize from the edges.
            self.setMouseTracking(True)

        if initial_project is not None:
            QTimer.singleShot(0, lambda: self.load_project(initial_project))

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central)

        self.title_bar = TitleBar()
        self.title_bar.minimize_requested.connect(self.showMinimized)
        self.title_bar.maximize_requested.connect(self.toggle_maximized)
        self.title_bar.settings_requested.connect(self.open_settings)
        self.title_bar.close_requested.connect(self.close)
        self.title_bar.settings_button.hide()
        self.topbar_button = PaneToggleButton("up")
        self.topbar_button.setToolTip("收起检查工具栏")
        self.topbar_button.setEnabled(False)
        self.topbar_button.setFixedSize(ICON_PANE_SIZE, ICON_PANE_SIZE)
        self.topbar_button.clicked.connect(self.toggle_topbar)
        self.title_bar.action_layout.addWidget(self.topbar_button)
        central_layout.addWidget(self.title_bar)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("appPages")
        central_layout.addWidget(self.page_stack, 1)

        self.main_page = QWidget()
        self.main_page.setObjectName("mainPage")
        main_layout = QVBoxLayout(self.main_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.page_stack.addWidget(self.main_page)
        self.status_text = ElidedLabel("")
        self.status_text.setObjectName("playerStatus")

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.splitter, 1)

        self.sidebar = self._build_sidebar()
        workspace = self._build_workspace()
        self.inspector = self._build_inspector()
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(workspace)
        self.splitter.addWidget(self.inspector)
        self.splitter.setSizes(
            [SPLITTER_SIDEBAR_WIDTH, SPLITTER_WORKSPACE_WIDTH, SPLITTER_INSPECTOR_WIDTH]
        )
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.inspector.hide()

        self.app_status_bar = QStatusBar()
        self.app_status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.app_status_bar)
        self.app_status_bar.hide()

    def _build_sidebar(self) -> QFrame:
        return build_sidebar(self)

    def _build_workspace(self) -> QWidget:
        return build_workspace(self)

    def _build_inspector(self) -> InspectorPanel:
        return build_inspector(self)

    def _build_shortcuts(self) -> None:
        configurable = {
            "previous": self.previous_sample,
            "next": self.next_sample,
            "first": self.first_sample,
            "last": self.last_sample,
            "playback": self.toggle_playback,
            "focus": self.toggle_focus,
            "reset": self.reset_views,
        }
        for action, handler in configurable.items():
            shortcut = QShortcut(QKeySequence(self.preferences.shortcuts[action]), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            self.shortcuts.append(shortcut)
            self.action_shortcuts[action] = shortcut

        bindings = (
            ("Escape", self.cancel_current_tool),
            ("Ctrl+B", self.toggle_sidebar),
            ("Ctrl+O", self.choose_project),
            ("Ctrl+,", self.open_settings),
            ("Tab", self.toggle_chrome),
            ("F11", self.toggle_fullscreen),
            ("Ctrl+Shift+P", self.open_command_palette),
        )
        for sequence, handler in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            self.shortcuts.append(shortcut)
        for index, modality in enumerate(MODALITY_INFO, start=1):
            shortcut = QShortcut(QKeySequence(str(index)), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda name=modality: self.toggle_modality(name))
            self.shortcuts.append(shortcut)

    def choose_project(self) -> None:
        start = self.settings.value("last_project", str(Path.home()))
        selected = QFileDialog.getExistingDirectory(self, "选择双目图像项目文件夹", str(start))
        if selected:
            self.load_project(Path(selected))

    def load_project(
        self,
        root: Path,
        manual_dirs: dict[str, Path] | None = None,
        force_order: bool = False,
        *,
        collection_root: Path | None = None,
        project_roots: list[Path] | None = None,
    ) -> None:
        if self._scan_in_progress:
            # A drop or picker change arrived while the scan dialog was open.
            self.status_text.setText("正在扫描项目，请稍候")
            return
        self._stop_playback()
        previous_root = self.dataset.root if self.dataset is not None else None
        previous_collection_root = self.project_collection_root
        previous_project_roots = list(self.project_roots)
        selected_root = root.expanduser().resolve()
        if not selected_root.is_dir():
            QMessageBox.critical(self, "无法打开项目", f"项目文件夹不存在：{selected_root}")
            return
        discover = project_roots is None
        if discover:
            root = selected_root
            collection_root = None
            project_roots = [selected_root]
        else:
            root = selected_root
            project_roots = [path.expanduser().resolve() for path in project_roots]
            collection_root = (
                collection_root.expanduser().resolve()
                if collection_root is not None
                else None
            )
        self.current_root = root
        if manual_dirs is None and not discover:
            manual_dirs, force_order = self._saved_matching_for(root)
        worker = ProjectScanWorker(root, manual_dirs, force_order, discover=discover)
        dialog = QProgressDialog("正在扫描项目…", "取消", 0, 0, self)
        dialog.setWindowTitle("打开项目")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(300)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumWidth(380)
        result: dict[str, object] = {}

        def on_progress(text: str, done: int, total: int) -> None:
            dialog.setLabelText(text)
            if total > 0:
                dialog.setRange(0, total)
                dialog.setValue(done)

        def on_discovered(roots: object) -> None:
            result["roots"] = list(roots) if isinstance(roots, (list, tuple)) else []

        def on_finished(dataset: object) -> None:
            result["dataset"] = dataset
            dialog.accept()

        def on_failed(error: str) -> None:
            result["error"] = error
            dialog.accept()

        def on_cancelled() -> None:
            result["cancelled"] = True
            dialog.accept()

        def on_cancel_clicked() -> None:
            result["cancelled"] = True
            worker.cancel()

        worker.signals.progress.connect(on_progress)
        if hasattr(worker.signals, "discovered"):
            worker.signals.discovered.connect(on_discovered)
        worker.signals.finished.connect(on_finished)
        worker.signals.failed.connect(on_failed)
        worker.signals.cancelled.connect(on_cancelled)
        dialog.canceled.connect(on_cancel_clicked)
        self._scan_in_progress = True
        self.setAcceptDrops(False)
        try:
            SCAN_POOL.start(worker)
            dialog.exec()
        finally:
            self._scan_in_progress = False
            self.setAcceptDrops(True)

        if discover and result.get("roots"):
            discovered_roots = [Path(str(item)).resolve() for item in result["roots"]]
            if len(discovered_roots) > 1:
                collection_root = selected_root
                project_roots = discovered_roots
                root = discovered_roots[0]
            else:
                project_roots = discovered_roots
            self.current_root = root

        if result.get("cancelled"):
            self.current_root = previous_root
            self.project_collection_root = previous_collection_root
            self.project_roots = previous_project_roots
            self._refresh_project_picker(previous_root)
            return
        if "error" in result:
            self.current_root = previous_root
            self.project_collection_root = previous_collection_root
            self.project_roots = previous_project_roots
            self._refresh_project_picker(previous_root)
            QMessageBox.critical(self, "无法打开项目", str(result["error"]))
            return
        dataset = result.get("dataset")
        if not isinstance(dataset, Dataset):
            self.current_root = previous_root
            self.project_collection_root = previous_collection_root
            self.project_roots = previous_project_roots
            self._refresh_project_picker(previous_root)
            QMessageBox.critical(self, "无法打开项目", "扫描未返回有效结果")
            return

        if discover and manual_dirs is None:
            # The background pass used automatic matching. Only projects with
            # a saved manual mapping pay for a second, explicit scan.
            saved_dirs, saved_force = self._saved_matching_for(root)
            if saved_dirs or saved_force:
                self.load_project(
                    root,
                    saved_dirs,
                    saved_force,
                    collection_root=collection_root,
                    project_roots=project_roots,
                )
                return
            manual_dirs, force_order = {}, False

        if not dataset.available_modalities:
            answer = QMessageBox.question(
                self,
                "未识别到数据",
                "没有自动识别到支持的数据目录。\n\n是否手动指定各视图文件夹？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                QTimer.singleShot(0, self.open_mapping_settings)
            else:
                self.current_root = previous_root
                self.project_collection_root = previous_collection_root
                self.project_roots = previous_project_roots
                self._refresh_project_picker(previous_root)
            return

        self.dataset = dataset
        self.project_collection_root = collection_root
        self.project_roots = project_roots
        # Calibration files stored with the capture become selectable for this
        # project only and are preferred when nothing was chosen before.
        project_options = discover_project_calibrations(dataset.root)
        self.calibration_options = [
            option for option in self.calibration_options if not option.project_local
        ] + project_options
        self.current_calibration_id, self.calibration = self._saved_calibration_selection(
            dataset.root
        )
        auto_calibration = ""
        if self.calibration is None and project_options:
            try:
                self.calibration = load_calibration(
                    project_options[0].path, self._saved_rectify_mode(dataset.root)
                )
                self.current_calibration_id = project_options[0].id
                auto_calibration = project_options[0].label
            except (ValueError, OSError):
                logger.exception("项目内标定无法加载：%s", project_options[0].path)
        self._resolution_warnings: set[tuple[str, int, int]] = set()
        self.rectify_mode = self._saved_rectify_mode(dataset.root)
        if self.calibration is not None:
            self.calibration = self.calibration.with_rectify_mode(self.rectify_mode)
        self.manual_dirs = dict(manual_dirs or {})
        self.force_order = force_order
        self.current_index = 0
        self.focused_modality = None
        self._cloud_clip_ranges = None
        self._rectified_cache.clear()
        self.settings.setValue(
            "last_project",
            str(self.project_collection_root or dataset.root),
        )
        self._save_matching_for(dataset.root, self.manual_dirs, self.force_order)
        logger.info(
            "已打开项目 %s：%d 组样本，%s",
            dataset.root,
            len(dataset.samples),
            "、".join(dataset.available_modalities),
        )
        context = (
            f"{self.project_collection_root.name} / {dataset.root.name}"
            if self.project_collection_root is not None
            else dataset.root.name
        )
        self.setWindowTitle(f"{context} — Stereo Selector")
        self.title_bar.set_context(context)
        self._remember_recent_project(self.project_collection_root or dataset.root)
        self._refresh_project_picker(dataset.root)
        self.project_name_label.setText(dataset.root.name)
        self.project_path_label.setText(str(dataset.root))
        self.project_path_label.setToolTip(str(dataset.root))
        self.project_path_label.show()
        available_set = set(dataset.available_modalities)
        complete_count = sum(available_set.issubset(sample.files) for sample in dataset.samples)
        incomplete_count = len(dataset.samples) - complete_count
        incomplete_text = f" · {incomplete_count} 组不完整" if incomplete_count else ""
        force_order_text = " · 强制顺序" if dataset.force_order else ""
        project_text = ""
        if len(self.project_roots) > 1:
            project_index = self.project_roots.index(dataset.root) + 1
            project_text = f"项目 {project_index}/{len(self.project_roots)} · "
        self.project_summary_label.setText(
            f"{project_text}{len(dataset.samples)} 组 · "
            f"{len(dataset.available_modalities)} 种数据"
            f"{force_order_text}{incomplete_text}"
        )
        self.project_summary_label.show()
        self.change_button.setText("更改项目")
        self.change_button.show()
        self.manual_mapping_button.show()
        self.modes_panel.hide()
        self.quality_panel.hide()
        self.reset_button.show()
        self.player_bar.show()
        self.topbar_button.setEnabled(True)
        if self._topbar_collapsed:
            self.inspection_bar.hide()
        else:
            self.inspection_bar.setMaximumHeight(16_777_215)
            self.inspection_bar.show()
        self._refresh_calibration_picker(self.current_calibration_id)
        self.crosshair_button.setChecked(False)
        self.epiline_button.setChecked(False)
        self.rectify_button.setChecked(False)
        self.cursor_info.clear()

        available = dataset.available_modalities
        preferred = [name for name in ("left", "right", "depth_color", "depth_fsd", "ply") if name in available]
        initial = set(preferred[: min(2, len(preferred))])
        for name, checkbox in self.checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setEnabled(name in available)
            checkbox.setChecked(name in initial)
            checkbox.blockSignals(False)
            self.count_labels[name].setText(str(len(dataset.files.get(name, []))))

        self.timeline.blockSignals(True)
        self.timeline.setRange(0, max(0, len(dataset.samples) - 1))
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.timeline.setEnabled(bool(dataset.samples))
        self.timeline_left.setText("1" if dataset.samples else "0")
        self.timeline_right.setText(str(len(dataset.samples)))
        counter_width = max(28, self.timeline_right.fontMetrics().horizontalAdvance("0" * len(str(len(dataset.samples)))) + 8)
        self.timeline_left.setFixedWidth(counter_width)
        self.timeline_right.setFixedWidth(counter_width)
        self.inspector.set_sources(list(dataset.available_modalities), preferred[0] if preferred else "")
        self._inspector_source = preferred[0] if preferred else ""
        self.content_stack.setCurrentWidget(self.media_container)
        self._build_tile_pool()
        self._rebuild_tiles()
        self._refresh_activity_page()
        self._show_current()
        if auto_calibration:
            self.status_text.setText(f"已使用项目内标定 {auto_calibration}")
        else:
            self.status_text.setText(f"已打开 {dataset.root.name}")

    def _remember_recent_project(self, root: Path) -> None:
        recent = self._recent_projects()
        entry = str(root)
        recent = [entry, *[item for item in recent if item != entry]][:8]
        self.settings.setValue("recent_projects", json.dumps(recent, ensure_ascii=False))
        self._refresh_recent_projects()

    def _recent_projects(self) -> list[str]:
        raw = str(self.settings.value("recent_projects", "[]"))
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [str(item) for item in items if isinstance(item, str)] if isinstance(items, list) else []

    def _refresh_recent_projects(self) -> None:
        if not hasattr(self, "empty_hint"):
            return
        existing = [item for item in self._recent_projects() if Path(item).is_dir()]
        self.empty_hint.set_recent(existing)

    def _open_recent_project(self, path: str) -> None:
        target = Path(path)
        if not target.is_dir():
            self.status_text.setText("最近项目已不存在")
            self._refresh_recent_projects()
            return
        self.load_project(target)

    def _refresh_project_picker(self, selected_root: Path | None) -> None:
        if not hasattr(self, "project_picker"):
            return
        self._project_picker_guard = True
        try:
            self.project_picker.clear()
            for project_root in self.project_roots:
                self.project_picker.addItem(project_root.name, str(project_root))
            selected = str(selected_root) if selected_root is not None else ""
            index = self.project_picker.findData(selected)
            if index >= 0:
                self.project_picker.setCurrentIndex(index)
            multiple = len(self.project_roots) > 1
            self.project_picker.setVisible(multiple)
            self.project_picker.setEnabled(multiple)
            if multiple and self.project_collection_root is not None:
                self.project_picker.setToolTip(
                    f"切换项目 · 共 {len(self.project_roots)} 个\n"
                    f"{self.project_collection_root}"
                )
            else:
                self.project_picker.setToolTip("")
        finally:
            self._project_picker_guard = False

    def _project_picker_changed(self, index: int) -> None:
        if self._project_picker_guard or not 0 <= index < len(self.project_roots):
            return
        selected = Path(str(self.project_picker.itemData(index))).resolve()
        if self.dataset is not None and selected == self.dataset.root:
            return
        self.load_project(
            selected,
            collection_root=self.project_collection_root,
            project_roots=self.project_roots,
        )

    def _saved_matching_for(self, root: Path) -> tuple[dict[str, Path], bool]:
        prefix = self._matching_settings_prefix(root)
        saved_root = str(self.settings.value(f"{prefix}/root", ""))
        if saved_root != str(root):
            # Migrate the single-project format used by earlier builds.
            if str(self.settings.value("matching/root", "")) != str(root):
                return {}, False
            prefix = "matching"
        directories = {
            name: Path(str(value))
            for name in MODALITY_INFO
            if (value := self.settings.value(f"{prefix}/directories/{name}", ""))
        }
        force_order = self.settings.value(f"{prefix}/force_order", False, type=bool)
        return directories, force_order

    def _save_matching_for(self, root: Path, directories: dict[str, Path], force_order: bool) -> None:
        prefix = self._matching_settings_prefix(root)
        self.settings.setValue(f"{prefix}/root", str(root))
        self.settings.setValue(f"{prefix}/force_order", force_order)
        for name in MODALITY_INFO:
            self.settings.setValue(f"{prefix}/directories/{name}", str(directories.get(name, "")))

    def _load_calibration_options(self) -> list[CalibrationOption]:
        options = builtin_calibration_options()
        raw = str(self.settings.value("calibration/custom_files", "[]"))
        try:
            saved_paths = json.loads(raw)
        except json.JSONDecodeError:
            saved_paths = []
        if isinstance(saved_paths, list):
            for value in saved_paths:
                path = Path(str(value)).expanduser()
                if path.is_file():
                    option = custom_calibration_option(path)
                    if all(existing.id != option.id for existing in options):
                        options.append(option)
        return options

    def _save_calibration_options(self) -> None:
        paths = [
            str(option.path)
            for option in self.calibration_options
            if not option.builtin and not option.project_local and option.path.is_file()
        ]
        self.settings.setValue(
            "calibration/custom_files",
            json.dumps(paths, ensure_ascii=False),
        )

    def _calibration_option(self, option_id: str) -> CalibrationOption | None:
        return next(
            (option for option in self.calibration_options if option.id == option_id),
            None,
        )

    def _saved_calibration_selection(
        self,
        root: Path,
    ) -> tuple[str, CalibrationData | None]:
        prefix = self._matching_settings_prefix(root)
        option_id = str(self.settings.value(f"{prefix}/calibration_id", "")).strip()
        option = self._calibration_option(option_id)
        if option is None:
            # Migrate the path-only format used by the first calibration build.
            value = str(self.settings.value(f"{prefix}/calibration_file", "")).strip()
            if value:
                path = Path(value).expanduser().resolve()
                option = next(
                    (
                        item
                        for item in self.calibration_options
                        if item.path.resolve() == path
                    ),
                    None,
                )
                if option is None and path.is_file():
                    option = custom_calibration_option(path)
                    self.calibration_options.append(option)
                    self._save_calibration_options()
                option_id = option.id if option is not None else ""
        if option is None:
            return "", None
        try:
            return option.id, load_calibration(option.path, self._saved_rectify_mode(root))
        except (ValueError, OSError):
            logger.exception("已保存的标定无法加载：%s", option.path)
            return "", None

    def _save_calibration_selection(
        self,
        root: Path,
        option_id: str,
        calibration: CalibrationData | None,
    ) -> None:
        prefix = self._matching_settings_prefix(root)
        self.settings.setValue(f"{prefix}/calibration_id", option_id)
        self.settings.setValue(
            f"{prefix}/calibration_file",
            str(calibration.source) if calibration is not None else "",
        )

    def _refresh_calibration_picker(self, selected_id: str = "") -> None:
        self.calibration_picker.blockSignals(True)
        self.calibration_picker.clear()
        self.calibration_picker.addItem("无标定", "")
        for option in self.calibration_options:
            self.calibration_picker.addItem(option.label, option.id)
        index = self.calibration_picker.findData(selected_id)
        self.calibration_picker.setCurrentIndex(max(0, index))
        self.calibration_picker.blockSignals(False)
        if self.calibration is not None:
            self.calibration_picker.setToolTip(
                f"{self.calibration.details}\n{self.calibration.source}"
            )
        else:
            self.calibration_picker.setToolTip("选择标定预设")
        self._refresh_rectify_mode_picker()

    def _refresh_rectify_mode_picker(self) -> None:
        """Show the air/underwater switch only when the calibration carries both sets."""
        if not hasattr(self, "rectify_mode_picker"):
            return
        modes = self.calibration.available_rectify_modes if self.calibration is not None else []
        selectable = len(modes) > 1
        self.rectify_mode_picker.setVisible(selectable)
        self.rectify_mode_picker.setEnabled(selectable)
        if self.calibration is not None:
            self.rectify_mode_picker.blockSignals(True)
            self.rectify_mode_picker.setCurrentIndex(
                max(0, self.rectify_mode_picker.findData(self.calibration.rectify_mode))
            )
            self.rectify_mode_picker.blockSignals(False)
            self.rectify_mode_picker.setToolTip(
                "校正模式：水下相机在空气和水中使用不同的校正参数，按项目记忆"
            )

    def _saved_rectify_mode(self, root: Path) -> str:
        prefix = self._matching_settings_prefix(root)
        return normalize_rectify_mode(self.settings.value(f"{prefix}/rectify_mode", "air"))

    def _rectify_mode_changed(self) -> None:
        mode = normalize_rectify_mode(self.rectify_mode_picker.currentData())
        if mode == self.rectify_mode:
            return
        self.rectify_mode = mode
        if self.dataset is not None:
            prefix = self._matching_settings_prefix(self.dataset.root)
            self.settings.setValue(f"{prefix}/rectify_mode", mode)
        if self.calibration is None or mode not in self.calibration.available_rectify_modes:
            return
        self.calibration = self.calibration.with_rectify_mode(mode)
        self._invalidate_rectification()
        self._rectified_cache.clear()
        self._refresh_calibration_picker(self.current_calibration_id)
        if self.dataset is not None:
            # Cloud projection intrinsics come from the rectified frame.
            self.focused_modality = None
            self._build_tile_pool()
            self._rebuild_tiles()
            self._show_current()
        label = dict(RECTIFY_MODE_CHOICES).get(mode, mode)
        logger.info("校正模式切换为 %s（%s）", label, self.current_calibration_id)
        self.status_text.setText(f"校正模式：{label}")

    def _apply_calibration_selection(
        self,
        option_id: str,
        *,
        rebuild: bool = True,
    ) -> bool:
        option = self._calibration_option(option_id)
        try:
            calibration = load_calibration(option.path, self.rectify_mode) if option is not None else None
        except Exception as exc:
            logger.exception("加载标定失败：%s", option.path if option is not None else "")
            QMessageBox.critical(self, "无法加载标定", str(exc))
            self._refresh_calibration_picker(self.current_calibration_id)
            return False
        self._invalidate_rectification()
        self._rectified_cache.clear()
        self.current_calibration_id = option.id if option is not None else ""
        self.calibration = calibration
        if self.dataset is not None:
            self._save_calibration_selection(
                self.dataset.root,
                self.current_calibration_id,
                calibration,
            )
        self._refresh_calibration_picker(self.current_calibration_id)
        if rebuild and self.dataset is not None:
            self.focused_modality = None
            self._build_tile_pool()
            self._rebuild_tiles()
            self._show_current()
        label = option.label if option is not None else "无标定"
        self.status_text.setText(f"已切换标定：{label}")
        return True

    def _calibration_picker_changed(self) -> None:
        if self.dataset is None:
            return
        self._apply_calibration_selection(
            str(self.calibration_picker.currentData() or "")
        )

    @staticmethod
    def _matching_settings_prefix(root: Path) -> str:
        normalized = os.path.normcase(str(root.expanduser().resolve())).encode("utf-8")
        project_id = hashlib.sha1(normalized).hexdigest()[:16]
        return f"matching/projects/{project_id}"

    def open_mapping_settings(self) -> None:
        root = self.current_root or (self.dataset.root if self.dataset is not None else None)
        if root is None:
            self.choose_project()
            return
        configures_current_dataset = self.dataset is not None and self.dataset.root == root
        initial_dirs = (
            {name: [path] for name, path in self.manual_dirs.items()} if configures_current_dataset else {}
        )
        force_order = self.force_order if configures_current_dataset else False
        dialog = MappingDialog(root, initial_dirs, force_order, self)
        if not dialog.exec():
            if self.dataset is not None:
                self.current_root = self.dataset.root
            return
        self.load_project(root, dialog.selected_dirs(), dialog.force_order)

    def _layout_picker_changed(self, _index: int) -> None:
        try:
            self._layout_columns = int(self.layout_picker.currentData() or 0)
        except (TypeError, ValueError):
            self._layout_columns = 0
        self.focused_modality = None
        self._apply_tile_layout()

    def _select_activity(self, page: str) -> None:
        if not self._activity_available(page):
            self._refresh_activity_page()
            return
        self._reset_interaction_tools()
        if page in {"adjust", "measure", "statistics"}:
            same_open_page = (
                self.inspector.isVisible()
                and self._active_sidebar_page == page
                and self.inspector.category == page
            )
            self._active_sidebar_page = page
            if not self._sidebar_collapsed:
                self.toggle_sidebar()
            if same_open_page:
                self.close_inspector()
            else:
                self.inspector.set_category(page)
                self.open_inspector()
                self._update_inspector()
                self._refresh_activity_page()
            return
        if self.inspector.isVisible():
            self.close_inspector()
        if not self._sidebar_collapsed and page == self._active_sidebar_page:
            self.toggle_sidebar()
            return
        self._active_sidebar_page = page
        if self._sidebar_collapsed:
            self.toggle_sidebar()
        else:
            self._refresh_activity_page()

    def _refresh_activity_page(self) -> None:
        visible = not self._sidebar_collapsed
        page = self._active_sidebar_page
        self.project_panel.setVisible(visible and page == "data")
        self.modes_panel.setVisible(
            visible and page == "display" and self.dataset is not None
        )
        self.quality_panel.setVisible(
            visible
            and page == "display"
            and self.dataset is not None
            and "depth_fsd" in self.dataset.available_modalities
        )
        for name, button in self.activity_buttons.items():
            button.setChecked(
                (name == page and visible)
                or (
                    name in {"adjust", "measure", "statistics"}
                    and name == page
                    and self.inspector.isVisible()
                )
            )

    def selected_modalities(self) -> list[str]:
        return [name for name in MODALITY_INFO if self.checkboxes[name].isChecked()]

    def toggle_modality(self, name: str) -> None:
        checkbox = self.checkboxes[name]
        if checkbox.isEnabled():
            checkbox.setChecked(not checkbox.isChecked())

    def _modality_toggled(self, name: str, checked: bool) -> None:
        if self.dataset is None:
            return
        self.focused_modality = None
        self._rebuild_tiles()
        self._show_current()

    def _clear_grid(self) -> None:
        while self.media_grid.count():
            self.media_grid.takeAt(0)
        self.tiles.clear()

    def _build_tile_pool(self) -> None:
        self._clear_grid()
        for tile in self.tile_pool.values():
            tile.dispose()
            tile.hide()
            tile.deleteLater()
        self.tile_pool.clear()
        if self.dataset is None:
            return
        # Device point clouds and depth maps share the rectified left frame,
        # so the cloud projector uses the depth camera model when available.
        left_calibration = (
            self.calibration.depth_camera()
            if self.calibration is not None
            else None
        )
        cloud_image_size = (
            (left_calibration.width, left_calibration.height)
            if left_calibration is not None
            and left_calibration.width is not None
            and left_calibration.height is not None
            else None
        )
        for modality in self.dataset.available_modalities:
            tile = MediaTile(
                modality,
                point_limit=self.preferences.point_limit,
                cloud_cam_offset=self.preferences.cloud_cam_offset,
                cloud_grid=self.preferences.cloud_grid,
                cloud_dot_radius=self.preferences.cloud_dot_radius,
                cloud_z_max=self.preferences.cloud_z_max,
                cloud_tau_rel=self.preferences.cloud_tau_rel,
                cloud_occlusion=self.preferences.cloud_occlusion,
                canvas_background=PALETTES[self.preferences.theme]["media_workspace"],
                parent=self.media_container,
                cloud_intrinsics=(
                    left_calibration.matrix
                    if left_calibration is not None
                    else None
                ),
                cloud_image_size=cloud_image_size,
                cloud_rotation=(
                    self.calibration.cloud_rotation
                    if self.calibration is not None
                    else None
                ),
                cloud_translation=(
                    self.calibration.cloud_translation_m
                    if self.calibration is not None
                    else None
                ),
            )
            tile.focus_requested.connect(self.focus_modality)
            tile.cursor_moved.connect(self._media_cursor_moved)
            tile.cursor_left.connect(self._media_cursor_left)
            tile.view_changed.connect(self._media_view_changed)
            tile.data_ready.connect(self._media_data_ready)
            tile.selection_changed.connect(self._media_selection_changed)
            tile.pixel_clicked.connect(self._image_pixel_clicked)
            if tile.cloud_canvas is not None:
                tile.cloud_canvas.tool_finished.connect(self._reset_interaction_tools)
                tile.cloud_canvas.point_picked.connect(self._point_cloud_picked)
                tile.cloud_canvas.measurement_changed.connect(
                    self._point_cloud_measurement
                )
                tile.cloud_canvas.selection_changed.connect(
                    lambda count: self.status_text.setText(
                        f"已框选 {count:,} 个点"
                    )
                )
            tile.image_canvas.set_crosshair_mode(self.crosshair_button.isChecked())
            tile.hide()
            self.tile_pool[modality] = tile

    def _rebuild_tiles(self) -> None:
        self._reset_interaction_tools()
        selected_names = set(self.selected_modalities())
        for name, tile in self.tile_pool.items():
            if name not in selected_names:
                self.media_grid.removeWidget(tile)
                tile.hide()
                tile.cancel_pending()
        self.tiles = {
            modality: self.tile_pool[modality]
            for modality in MODALITY_INFO
            if modality in selected_names and modality in self.tile_pool
        }
        self._apply_tile_layout()
        self.selected_count_label.setText(f"已选 {len(self.tiles)}")
        self._update_inspector()

    def _apply_tile_layout(self) -> None:
        self.media_container.setUpdatesEnabled(False)
        try:
            self._place_tiles()
        finally:
            self.media_container.setUpdatesEnabled(True)

    def _place_tiles(self) -> None:
        while self.media_grid.count():
            self.media_grid.takeAt(0)
        for row in range(5):
            self.media_grid.setRowStretch(row, 0)
        for column in range(3):
            self.media_grid.setColumnStretch(column, 0)

        if not self.tiles:
            self.media_grid.addWidget(self.no_views_hint, 0, 0)
            self.no_views_hint.setVisible(True)
            self.media_grid.setRowStretch(0, 1)
            self.media_grid.setColumnStretch(0, 1)
            self._set_view_hint("未选择视图")
            return

        self.no_views_hint.setVisible(False)

        if self.focused_modality in self.tiles:
            for name, tile in self.tiles.items():
                focused = name == self.focused_modality
                tile.setVisible(focused)
                tile.set_focused(focused)
            self.media_grid.addWidget(self.tiles[self.focused_modality], 0, 0)
            self.media_grid.setRowStretch(0, 1)
            self.media_grid.setColumnStretch(0, 1)
            self._set_view_hint("聚焦")
            return

        self.focused_modality = None
        selected = list(self.tiles)
        columns = (
            min(max(1, self._layout_columns), len(selected))
            if self._layout_columns
            else 1
            if len(selected) == 1
            else 3
            if len(selected) in (3, 5)
            else 2
        )
        for index, modality in enumerate(selected):
            tile = self.tiles[modality]
            tile.setVisible(True)
            tile.set_focused(False)
            self.media_grid.addWidget(tile, index // columns, index % columns)
        for column in range(columns):
            self.media_grid.setColumnStretch(column, 1)
        for row in range(max(1, (len(selected) + columns - 1) // columns)):
            self.media_grid.setRowStretch(row, 1)
        self._set_view_hint("")

    def _set_view_hint(self, text: str) -> None:
        self.view_hint.setText(text)
        self.view_hint.setVisible(bool(text))

    def _toggle_crosshair(self, checked: bool) -> None:
        for tile in self.tile_pool.values():
            tile.image_canvas.set_crosshair_mode(checked)
        target_height = (
            INSPECTION_BAR_CURSOR_HEIGHT if checked else INSPECTION_BAR_HEIGHT
        )
        self.cursor_info_row.setVisible(checked)
        self._topbar_expanded_height = target_height
        if not self._topbar_collapsed and self.inspection_bar.isVisible():
            self.inspection_bar.setFixedHeight(target_height)
        if checked:
            self.cursor_info.setText("在图片上移动鼠标")
            return
        self._cursor_update_timer.stop()
        self._pending_cursor_update = None
        self._crosshair_position = None
        self.cursor_info.clear()
        for tile in self.tile_pool.values():
            tile.image_canvas.set_crosshair(0, 0, False)
            tile.image_canvas.set_epiline(None)
        if self.epiline_button.isChecked():
            self.epiline_button.setChecked(False)

    def _sync_views_toggled(self, checked: bool) -> None:
        self.status_text.setText("图片视图已同步" if checked else "")

    def _epilines_toggled(self, checked: bool) -> None:
        if checked and not self.crosshair_button.isChecked():
            self.crosshair_button.setChecked(True)
        if not checked:
            for tile in self.tile_pool.values():
                tile.image_canvas.set_epiline(None)

    def _rectification_toggled(self, checked: bool) -> None:
        if checked and (
            self.calibration is None
            or self.calibration.left is None
            or self.calibration.right is None
        ):
            self.rectify_button.blockSignals(True)
            self.rectify_button.setChecked(False)
            self.rectify_button.blockSignals(False)
            self.status_text.setText("校正切换需要左右相机内参")
            return
        if checked:
            self.status_text.setText("正在生成校正视图…")
            QTimer.singleShot(0, self._apply_rectified_views)
            return
        self._invalidate_rectification()
        for modality in ("left", "right"):
            tile = self.tile_pool.get(modality)
            if tile is not None and tile.image_data is not None:
                tile.set_display_settings()
        self.status_text.setText("已显示校正前图像")

    def _invalidate_rectification(self) -> None:
        """Make every outstanding rectification callback stale."""
        self._rectify_sequence += 1
        self._rectify_token = None
        self._rectify_key = None
        self._rectify_worker = None

    def _apply_rectified_views(self) -> None:
        if not self.rectify_button.isChecked() or self.calibration is None:
            return
        left = self.tile_pool.get("left")
        right = self.tile_pool.get("right")
        if (
            left is None
            or right is None
            or left.image_data is None
            or right.image_data is None
            or left.current_path is None
            or right.current_path is None
        ):
            return
        key = (
            str(left.current_path),
            str(right.current_path),
            self.current_calibration_id,
        )
        cached = self._rectified_cache.get(key)
        if cached is None:
            if self._rectify_token is not None:
                if self._rectify_key == key:
                    return
                self._invalidate_rectification()
            self._rectify_sequence += 1
            token = self._rectify_sequence
            self._rectify_token = token
            self._rectify_key = key
            self.status_text.setText("正在生成校正视图…")
            worker = RectifyWorker(
                token,
                left.image_data.image,
                right.image_data.image,
                self.calibration,
            )
            self._rectify_worker = worker

            def on_finished(result_token: int, images: object) -> None:
                if result_token != token or self._rectify_token != token:
                    return
                self._rectify_worker = None
                self._rectify_token = None
                self._rectify_key = None
                if not isinstance(images, tuple) or len(images) < 2:
                    return
                method = str(images[2]) if len(images) > 2 else ""
                images = (images[0], images[1])
                self._rectified_cache[key] = (images, method)
                if len(self._rectified_cache) > 6:
                    self._rectified_cache.pop(next(iter(self._rectified_cache)))
                current_key = (
                    str(left.current_path),
                    str(right.current_path),
                    self.current_calibration_id,
                )
                if current_key != key or not self.rectify_button.isChecked():
                    if self.rectify_button.isChecked():
                        QTimer.singleShot(0, self._apply_rectified_views)
                    return
                left.image_canvas.set_image(images[0], preserve_view=True)
                right.image_canvas.set_image(images[1], preserve_view=True)
                self.status_text.setText(f"已显示校正后图像 · {method}" if method else "已显示校正后图像")

            def on_failed(result_token: int, error: str) -> None:
                if result_token != token or self._rectify_token != token:
                    return
                self._rectify_worker = None
                self._rectify_token = None
                self._rectify_key = None
                self.rectify_button.blockSignals(True)
                self.rectify_button.setChecked(False)
                self.rectify_button.blockSignals(False)
                self.status_text.setText(f"无法生成校正视图：{error}")

            worker.signals.finished.connect(on_finished)
            worker.signals.failed.connect(on_failed)
            IMAGE_POOL.start(worker)
            return
        if self._rectify_token is not None and self._rectify_key != key:
            self._invalidate_rectification()
        images, method = cached
        left.image_canvas.set_image(images[0], preserve_view=True)
        right.image_canvas.set_image(images[1], preserve_view=True)
        self.status_text.setText(f"已显示校正后图像 · {method}" if method else "已显示校正后图像")

    def _media_cursor_moved(self, modality: str, x: float, y: float) -> None:
        if not self.crosshair_button.isChecked():
            return
        self._pending_cursor_update = (modality, x, y)
        if not self._cursor_update_timer.isActive():
            self._flush_cursor_update()
            self._cursor_update_timer.start()

    def _drain_cursor_update(self) -> None:
        if self._pending_cursor_update is None:
            return
        self._flush_cursor_update()
        self._cursor_update_timer.start()

    def _flush_cursor_update(self) -> None:
        pending = self._pending_cursor_update
        self._pending_cursor_update = None
        if pending is None or not self.crosshair_button.isChecked():
            return
        modality, x, y = pending
        self._crosshair_position = (x, y)
        for tile in self.tiles.values():
            if tile.image_data is not None:
                tile.image_canvas.set_crosshair(x, y, True)
                tile.image_canvas.set_epiline(None)
        self.cursor_info.set_values(**self._cursor_value_fields(modality, x, y))
        if self.epiline_button.isChecked():
            self._update_epiline(modality, x, y)

    def _media_cursor_left(self, modality: str) -> None:
        if not self.crosshair_button.isChecked():
            return
        self._cursor_update_timer.stop()
        self._pending_cursor_update = None
        self._crosshair_position = None
        for tile in self.tile_pool.values():
            tile.image_canvas.set_crosshair(0, 0, False)
            tile.image_canvas.set_epiline(None)
        self.cursor_info.setText("在图片上移动鼠标")

    def _cursor_value_fields(self, source: str, x: float, y: float) -> dict[str, str]:
        fields = {
            "position": "",
            "rgb": "",
            "stereo": "",
            "depth": "",
            "xyz": "",
        }
        left = self.tile_pool.get("left")
        right = self.tile_pool.get("right")
        source_tile = self.tile_pool.get(source)
        rgb_tile = (
            source_tile
            if source in {"left", "right", "depth_color"}
            and source_tile is not None
            and source_tile.image_data is not None
            else left
            if left is not None and left.image_data is not None
            else right
            if right is not None and right.image_data is not None
            else None
        )
        if rgb_tile is not None and rgb_tile.image_data is not None:
            image = rgb_tile.image_data.image
            px = min(image.width() - 1, max(0, int(x * image.width())))
            py = min(image.height() - 1, max(0, int(y * image.height())))
            color = image.pixelColor(px, py)
            fields["position"] = f"{modality_label(rgb_tile.modality)} {px},{py}"
            fields["rgb"] = f"RGB {color.red()},{color.green()},{color.blue()}"
        stereo_match = self._stereo_match_text(source, x, y)
        if stereo_match:
            fields["stereo"] = stereo_match

        depth_value: float | None = None
        depth_dtype = None
        depth = self.tile_pool.get("depth_fsd")
        if (
            depth is not None
            and depth.image_data is not None
        ):
            values = depth.image_data.values
            depth_dtype = values.dtype
            height, width = values.shape[:2]
            px = min(width - 1, max(0, int(x * width)))
            py = min(height - 1, max(0, int(y * height)))
            raw = values[py, px]
            if getattr(raw, "ndim", 0):
                raw = raw.flat[0]
            try:
                depth_value = float(raw)
            except (TypeError, ValueError):
                depth_value = None
            if depth_value is not None and math.isfinite(depth_value):
                if depth_is_valid(depth_value, depth_dtype):
                    fields["depth"] = f"深度 {format_depth(depth_value, depth_dtype, self.preferences.depth_unit)}"
                else:
                    fields["depth"] = f"深度 {depth_value:g} 饱和" if depth_value >= 65535 else "深度 0 空洞"
                    depth_value = None
            else:
                depth_value = None
                fields["depth"] = "深度 无效"

        camera = self.calibration.depth_camera() if self.calibration is not None else None
        if depth_value is not None and camera is not None:
            width = (
                camera.width
                if camera.width is not None
                else depth.image_data.image.width()
            )
            height = (
                camera.height
                if camera.height is not None
                else depth.image_data.image.height()
            )
            px = x * width
            py = y * height
            meters = depth_to_meters(depth_value, depth_dtype, self.preferences.depth_unit)
            point = camera.point_from_depth(px, py, meters)
            fields["xyz"] = f"XYZ {point[0]:.3f},{point[1]:.3f},{point[2]:.3f} m"
        elif depth_value is not None:
            fields["xyz"] = "XYZ 需左目内参"
        if not any(fields.values()):
            fields["position"] = "无像素数据"
        return fields

    def _cursor_value_text(self, source: str, x: float, y: float) -> str:
        """Return the complete readout for tooltips and compatibility tests."""
        fields = self._cursor_value_fields(source, x, y)
        return "   ·   ".join(value for value in fields.values() if value)

    def _stereo_match_text(self, source: str, x: float, y: float) -> str:
        if source not in {"left", "right"}:
            return ""
        left = self.tile_pool.get("left")
        right = self.tile_pool.get("right")
        if (
            left is None
            or right is None
            or left.image_data is None
            or right.image_data is None
        ):
            return ""

        source_tile = left if source == "left" else right
        target_tile = right if source == "left" else left
        source_values = np.asarray(source_tile.image_data.values)
        target_values = np.asarray(target_tile.image_data.values)
        source_height, source_width = source_values.shape[:2]
        target_height, target_width = target_values.shape[:2]
        sx = max(0, min(source_width - 1, int(x * source_width)))
        sy = max(0, min(source_height - 1, int(y * source_height)))
        ty = max(0, min(target_height - 1, int(y * target_height)))
        radius = 3
        if (
            sx < radius
            or sx >= source_width - radius
            or sy < radius
            or sy >= source_height - radius
            or ty < radius
            or ty >= target_height - radius
            or target_width < radius * 2 + 1
        ):
            return ""
        source_patch = source_values[
            sy - radius : sy + radius + 1,
            sx - radius : sx + radius + 1,
        ]
        target_strip = target_values[
            ty - radius : ty + radius + 1,
            :,
        ]
        # Convert only the 7×7 patch and seven-row search strip. Converting the
        # entire pair on every pointer event made crosshair movement stutter.
        if source_patch.ndim == 3:
            source_patch = source_patch[..., :3].astype(
                np.float32, copy=False
            ).mean(axis=2)
        else:
            source_patch = source_patch.astype(np.float32, copy=False)
        if target_strip.ndim == 3:
            target_strip = target_strip[..., :3].astype(
                np.float32, copy=False
            ).mean(axis=2)
        else:
            target_strip = target_strip.astype(np.float32, copy=False)
        try:
            candidates = np.lib.stride_tricks.sliding_window_view(
                target_strip,
                source_patch.shape,
            )[0]
        except (ValueError, IndexError):
            return ""
        expected = x * target_width
        search_radius = min(256, target_width // 2)
        start = max(0, int(expected) - search_radius - radius)
        stop = min(len(candidates), int(expected) + search_radius - radius + 1)
        if stop <= start:
            return ""
        errors = np.mean(
            np.abs(candidates[start:stop] - source_patch),
            axis=(1, 2),
        )
        best_left = start + int(np.argmin(errors))
        target_x = best_left + radius
        source_in_target_pixels = x * target_width
        disparity = (
            source_in_target_pixels - target_x
            if source == "left"
            else target_x - source_in_target_pixels
        )
        label = "R" if source == "left" else "L"
        return f"{label} {target_x},{ty} · 视差 {disparity:.2f}px"

    def _update_epiline(self, source: str, x: float, y: float) -> None:
        if source not in {"left", "right"}:
            return
        target = "right" if source == "left" else "left"
        source_tile = self.tile_pool.get(source)
        target_tile = self.tile_pool.get(target)
        if (
            source_tile is None
            or target_tile is None
            or source_tile.image_data is None
            or target_tile.image_data is None
        ):
            return
        source_image = source_tile.image_data.image
        target_image = target_tile.image_data.image
        if self.calibration is not None and self.calibration.fundamental is not None:
            line = epiline_for_point(
                self.calibration.fundamental,
                x,
                y,
                source_image.width(),
                source_image.height(),
                target_image.width(),
                target_image.height(),
                transpose=source == "right",
            )
        else:
            line = (0.0, y, 1.0, y)
        target_tile.image_canvas.set_epiline(line)

    def _media_view_changed(
        self,
        source: str,
        zoom: float,
        center_x: float,
        center_y: float,
    ) -> None:
        if not self.sync_views_button.isChecked():
            return
        for modality, tile in self.tiles.items():
            if modality != source and tile.image_data is not None:
                tile.image_canvas.apply_view_state(zoom, center_x, center_y)

    def _media_data_ready(self, modality: str) -> None:
        if modality == "depth_fsd":
            self._update_depth_quality()
        elif modality == "ply" and self._cloud_clip_ranges is not None:
            self._apply_cloud_clip()
        if modality == "left":
            self._check_calibration_resolution()
        if modality in {"left", "right"} and self.rectify_button.isChecked():
            QTimer.singleShot(0, self._apply_rectified_views)
        QTimer.singleShot(0, lambda name=modality: self._prefetch_adjacent_media(name))
        if modality == self._inspector_source:
            self._update_inspector()
        self._update_inspection_controls()

    def _check_calibration_resolution(self) -> None:
        """Warn once when the calibration was made for a different sensor size."""
        if self.calibration is None or self.calibration.left is None:
            return
        left = self.tile_pool.get("left")
        if left is None or left.image_data is None:
            return
        expected = (self.calibration.left.width, self.calibration.left.height)
        if expected[0] is None or expected[1] is None:
            return
        actual = (left.image_data.image.width(), left.image_data.image.height())
        if actual == expected:
            return
        key = (self.current_calibration_id, *actual)
        if key in self._resolution_warnings:
            return
        self._resolution_warnings.add(key)
        message = f"标定分辨率 {expected[0]}×{expected[1]} 与图像 {actual[0]}×{actual[1]} 不一致"
        logger.warning("%s（%s）", message, self.current_calibration_id)
        self.status_text.setText(message)
        self.calibration_picker.setToolTip(
            f"{self.calibration.details}\n{self.calibration.source}\n⚠ {message}"
        )

    def _prefetch_adjacent_media(self, modality: str) -> None:
        if self.dataset is None or modality not in self.tiles:
            return
        tile = self.tiles[modality]
        # Current frame has priority. Once it is visible, warm the next two
        # frames and the previous frame at low thread-pool priority.
        for index in (
            self.current_index + 1,
            self.current_index + 2,
            self.current_index - 1,
        ):
            if 0 <= index < len(self.dataset.samples):
                tile.prefetch_file(self.dataset.samples[index].files.get(modality))

    def _prefetch_next_point_cloud(self) -> None:
        # Compatibility alias retained for older tests and callers.
        self._prefetch_adjacent_media("ply")

    def _update_depth_quality(self) -> None:
        tile = self.tile_pool.get("depth_fsd")
        stats = tile.depth_stats if tile is not None else None
        if stats is None:
            self.depth_quality_label.setText(
                "正在计算…" if tile is not None and tile.image_data is not None else "等待深度图"
            )
            return
        dtype = tile.image_data.values.dtype if tile is not None and tile.image_data is not None else None
        unit = resolve_depth_unit(dtype, self.preferences.depth_unit)
        value_range = (
            f"{stats.minimum:g} – {stats.maximum:g} {unit}"
            if stats.minimum is not None and stats.maximum is not None
            else "无有效值"
        )
        self.depth_quality_label.setText(
            f"范围  {value_range}\n"
            f"无效  {stats.invalid_ratio:.2%}   零值  {stats.zero_ratio:.2%}\n"
            f"空洞  {stats.hole_ratio:.2%}   极大值  {stats.extreme_ratio:.2%}"
        )

    def _update_inspection_controls(self) -> None:
        has_left = (
            (left := self.tile_pool.get("left")) is not None
            and left.image_data is not None
            and self.checkboxes["left"].isChecked()
        )
        has_right = (
            (right := self.tile_pool.get("right")) is not None
            and right.image_data is not None
            and self.checkboxes["right"].isChecked()
        )
        self.overlay_button.setEnabled(has_left and has_right)
        self.epiline_button.setEnabled(has_left and has_right)
        self.rectify_button.setEnabled(
            has_left
            and has_right
            and self.calibration is not None
            and self.calibration.left is not None
            and self.calibration.right is not None
        )
        self.crosshair_button.setEnabled(
            any(tile.image_data is not None for tile in self.tiles.values())
        )
        self.sync_views_button.setEnabled(
            sum(tile.image_data is not None for tile in self.tiles.values()) >= 2
        )

    def open_inspector(self) -> None:
        if not self._inspection_sources():
            return
        if self.inspector.isVisible():
            return
        if self._active_sidebar_page not in {"adjust", "measure", "statistics"}:
            self._active_sidebar_page = "statistics"
            self.inspector.set_category("statistics")
        if not self._sidebar_collapsed:
            self.toggle_sidebar()
        self.inspector.show()
        width = max(290, min(340, self.inspector.sizeHint().width()))
        total = max(1, self.splitter.width())
        left = self.sidebar.width()
        self.splitter.setSizes([left, max(1, total - left - width), width])
        self._refresh_activity_page()
        self._update_inspector()

    def close_inspector(self) -> None:
        self._reset_interaction_tools()
        self.inspector.hide()
        total = max(1, self.splitter.width())
        self.splitter.setSizes([self.sidebar.width(), total - self.sidebar.width(), 0])
        self._refresh_activity_page()

    def _inspector_source_changed(self, modality: str) -> None:
        self._reset_interaction_tools()
        self._inspector_source = modality
        self._update_inspector()

    def _inspection_sources(self) -> list[str]:
        if self.dataset is None or not self.dataset.samples:
            return []
        sample = self.dataset.samples[self.current_index]
        return [name for name in self.tiles if name in sample.files]

    def _activity_available(self, page: str) -> bool:
        if page == "data":
            return True
        if page == "display":
            return self.dataset is not None and bool(self.dataset.available_modalities)
        return page in {"adjust", "measure", "statistics"} and bool(self._inspection_sources())

    def _update_inspector(self) -> None:
        if not self.inspector.isVisible() or self.dataset is None:
            return
        sources = self._inspection_sources()
        if not sources:
            self.close_inspector()
            return
        source = self._inspector_source if self._inspector_source in sources else sources[0]
        self._inspector_source = source
        self.inspector.set_sources(sources, source)
        tile = self.tile_pool.get(source)
        if tile is None:
            return
        path = tile.current_path
        if source == "ply":
            cloud = tile.cloud_canvas.current_cloud if tile.cloud_canvas is not None and not tile.is_loading() and tile.showing_canvas() else None
            self.inspector.set_ready(cloud is not None and bool(len(cloud.points)))
            self.inspector.set_cloud_data(cloud, path)
            if cloud is not None and tile.cloud_canvas is not None:
                self.inspector.set_cloud_controls(tile.cloud_canvas, self._cloud_clip_ranges)
        else:
            self.inspector.set_ready(tile.image_data is not None and not tile.is_loading())
            self.inspector.set_display_controls(tile.display_settings)
            self.inspector.set_image_data(source, tile.image_data, path)

    def _apply_display_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        tile = self.tile_pool.get(self._inspector_source)
        if tile is None or tile.image_data is None:
            return
        tile.set_display_settings(**settings)

    def _active_image_tile(self) -> MediaTile | None:
        tile = self.tiles.get(self._inspector_source)
        if tile is not None and tile.image_data is not None:
            return tile
        return next(
            (item for item in self.tiles.values() if item.image_data is not None),
            None,
        )

    def _fit_requested(self, mode: str) -> None:
        tile = self._active_image_tile()
        if tile is None:
            return
        tile.set_fit_mode(mode)

    def _tool_requested(self, tool: str) -> None:
        tile = self._active_image_tile()
        if tile is None:
            return
        previous = tile.image_canvas.tool
        self._reset_interaction_tools()
        if previous == tool:
            tool = "pan"
        tile.image_canvas.set_tool(tool)
        self.inspector.image_tool_buttons[tool].setChecked(True)
        self.status_text.setText(
            {"roi": "拖动选择矩形 ROI", "line": "拖动绘制测量线"}.get(tool, "")
        )

    def _reset_interaction_tools(self, *, clear_selection: bool = False) -> None:
        if clear_selection:
            self.inspector.invalidate_roi()
        for tile in self.tile_pool.values():
            tile.image_canvas.set_tool("pan")
            if clear_selection:
                tile.image_canvas.clear_selection()
            if tile.cloud_canvas is not None:
                tile.cloud_canvas.set_interaction_mode("rotate")
        self.inspector.reset_tools()

    def cancel_current_tool(self) -> None:
        from .ui_controls import ChoicePopup
        for popup in self.findChildren(ChoicePopup):
            if popup.isVisible():
                popup.hide()
                return
        if self._settings_page is not None:
            self._settings_page.reject()
            return
        self._reset_interaction_tools(clear_selection=True)
        self.crosshair_button.setChecked(False)
        self.status_text.clear()
        self.exit_focus()

    def _media_selection_changed(self, modality: str, tool: str, selection: object) -> None:
        self._reset_interaction_tools()
        tile = self.tile_pool.get(modality)
        if tile is None or tile.image_data is None:
            return
        values = tile.image_data.values
        height, width = values.shape[:2]
        if tool == "roi" and isinstance(selection, QRectF):
            left = max(0, min(width, int(selection.left() * width)))
            right = max(left + 1, min(width, math.ceil(selection.right() * width)))
            top = max(0, min(height, int(selection.top() * height)))
            bottom = max(top + 1, min(height, math.ceil(selection.bottom() * height)))
            self.inspector.set_roi_statistics(values[top:bottom, left:right])
            return
        if tool != "line" or not isinstance(selection, tuple) or len(selection) != 4:
            return
        x1, y1, x2, y2 = map(float, selection)
        pixel_length = math.hypot((x2 - x1) * width, (y2 - y1) * height)
        physical: float | None = None
        endpoint_depths: tuple[float, float] | None = None
        note = ""
        depth_tile = self.tile_pool.get("depth_fsd")
        camera = self.calibration.depth_camera() if self.calibration is not None else None
        if camera is not None and depth_tile is not None and depth_tile.image_data is not None:
            depth_values = depth_tile.image_data.values
            depth_height, depth_width = depth_values.shape[:2]
            unit = self.preferences.depth_unit

            def sample(nx: float, ny: float) -> float | None:
                sx = max(0, min(depth_width - 1, int(nx * depth_width)))
                sy = max(0, min(depth_height - 1, int(ny * depth_height)))
                raw = depth_values[sy, sx]
                if getattr(raw, "ndim", 0):
                    raw = raw.flat[0]
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    return None
                return value if depth_is_valid(value, depth_values.dtype) else None

            first_depth = sample(x1, y1)
            second_depth = sample(x2, y2)
            if first_depth is not None and second_depth is not None:
                frame_width = camera.width if camera.width is not None else width
                frame_height = camera.height if camera.height is not None else height
                first = camera.point_from_depth(
                    x1 * frame_width, y1 * frame_height, depth_to_meters(first_depth, depth_values.dtype, unit)
                )
                second = camera.point_from_depth(
                    x2 * frame_width, y2 * frame_height, depth_to_meters(second_depth, depth_values.dtype, unit)
                )
                physical = math.dist(first, second)
                endpoint_depths = (
                    depth_to_meters(first_depth, depth_values.dtype, unit),
                    depth_to_meters(second_depth, depth_values.dtype, unit),
                )
            else:
                note = "端点深度无效，无法换算物理长度"
        elif camera is None and depth_tile is not None and depth_tile.image_data is not None:
            note = "需要标定才能换算物理长度"
        self.inspector.set_line_measurement(pixel_length, physical, endpoint_depths, note)

    def _image_pixel_clicked(self, modality: str, x: float, y: float) -> None:
        cloud_tile = self.tile_pool.get("ply")
        if cloud_tile is None or cloud_tile.cloud_canvas is None:
            return
        cloud_tile.cloud_canvas.highlight_image_point(x, y)
        self.status_text.setText(f"{modality_label(modality)}像素已在点云中高亮")

    def _point_cloud_picked(self, point: object) -> None:
        try:
            camera_point = np.asarray(point, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return
        cloud_tile = self.tile_pool.get("ply")
        if cloud_tile is None or cloud_tile.cloud_canvas is None:
            return
        cloud = cloud_tile.cloud_canvas.current_cloud
        color = None
        if cloud is not None and len(cloud.points):
            index = int(np.argmin(np.sum((cloud.points - camera_point) ** 2, axis=1)))
            color = cloud.colors[index]
        self.inspector.set_point_information(camera_point, color)
        uv, depth = project_camera_points(
            camera_point.reshape(1, 3),
            cloud_tile.cloud_canvas.camera_intrinsics,
            cloud_tile.cloud_canvas.cam_offset,
        )
        width, height = cloud_tile.cloud_canvas.camera_image_size
        if depth[0] <= 0 or not np.isfinite(uv[0]).all():
            return
        x, y = float(uv[0, 0] / width), float(uv[0, 1] / height)
        for tile in self.tiles.values():
            if tile.image_data is not None:
                tile.image_canvas.set_crosshair(x, y, 0 <= x <= 1 and 0 <= y <= 1)
        self.cursor_info.setText(
            f"XYZ {camera_point[0]:.5g}, {camera_point[1]:.5g}, {camera_point[2]:.5g}"
        )

    def _point_cloud_measurement(self, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 3:
            return
        first, second, distance = result
        self.inspector.set_cloud_measurement(
            np.asarray(first),
            np.asarray(second),
            float(distance),
        )

    def export_current_view(self) -> None:
        tile = self.tile_pool.get(self._inspector_source)
        if tile is None:
            tile = next(iter(self.tiles.values()), None)
        if tile is None:
            return
        start = self.dataset.root if self.dataset is not None else Path.home()
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "导出当前视图",
            str(start / f"{tile.modality}_view.png"),
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)",
        )
        if not selected:
            return
        target = tile.cloud_canvas.view if tile.cloud_canvas is not None and tile.modality == "ply" else tile.image_canvas.viewport()
        if not target.grab().save(selected):
            logger.error("导出视图失败：%s", selected)
            QMessageBox.warning(self, "导出失败", f"无法写入：\n{selected}")
            return
        logger.info("已导出视图：%s", selected)
        self.status_text.setText(f"已导出 {Path(selected).name}")

    def export_adjusted_image(self) -> None:
        """Write the current display adjustments at the source resolution."""
        tile = self.tile_pool.get(self._inspector_source)
        if tile is None or tile.image_data is None:
            tile = next((item for item in self.tiles.values() if item.image_data is not None), None)
        if tile is None or tile.image_data is None:
            self.status_text.setText("没有可导出的图片")
            return
        start = self.dataset.root if self.dataset is not None else Path.home()
        stem = tile.current_path.stem if tile.current_path is not None else tile.modality
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "导出调整后的原始分辨率图像",
            str(start / f"{stem}_{tile.modality}_adjusted.png"),
            "PNG 图片 (*.png);;TIFF 图片 (*.tif *.tiff)",
        )
        if not selected:
            return
        try:
            image = render_image_values(tile.image_data.values, **tile.display_settings)
        except Exception as exc:
            logger.exception("渲染导出图像失败")
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        if not image.save(selected):
            logger.error("导出图像失败：%s", selected)
            QMessageBox.warning(self, "导出失败", f"无法写入：\n{selected}")
            return
        logger.info("已导出调整后图像：%s", selected)
        self.status_text.setText(f"已导出 {Path(selected).name}")

    def _cloud_color_changed(self, mode: str) -> None:
        tile = self.tile_pool.get("ply")
        if tile is not None and tile.cloud_canvas is not None:
            tile.cloud_canvas.set_color_mode(mode)

    def _cloud_size_changed(self, radius: float) -> None:
        tile = self.tile_pool.get("ply")
        if tile is not None and tile.cloud_canvas is not None:
            tile.cloud_canvas.set_point_size(radius)

    def _cloud_guides_changed(self, visible: bool) -> None:
        tile = self.tile_pool.get("ply")
        if tile is not None and tile.cloud_canvas is not None:
            tile.cloud_canvas.set_guides_visible(visible)

    def _cloud_view_requested(self, view: str) -> None:
        tile = self.tile_pool.get("ply")
        if tile is not None and tile.cloud_canvas is not None:
            tile.cloud_canvas.set_standard_view(view)

    def _cloud_tool_requested(self, tool: str) -> None:
        tile = self.tiles.get("ply")
        if tile is not None and tile.cloud_canvas is not None and not tile.is_loading() and tile.showing_canvas():
            previous = tile.cloud_canvas.interaction_mode
            self._reset_interaction_tools()
            if tool == "reset":
                self._cloud_clip_ranges = None
                count = tile.cloud_canvas.restore_full_cloud()
                self.inspector.reset_clip_bounds()
                self._update_inspector()
                self.status_text.setText(f"已恢复 {count:,} 个点")
                return
            if previous == tool:
                tool = "rotate"
            tile.cloud_canvas.set_interaction_mode(tool)
            self.inspector.cloud_tool_buttons[tool].setChecked(True)
            self.status_text.setText(
                {
                    "pick": "点击点云查看 XYZ",
                    "measure": "依次点击两个点测量距离",
                    "box": "拖动矩形框选局部点云",
                }.get(tool, "")
            )

    def _cloud_background_changed(self, color: str) -> None:
        tile = self.tile_pool.get("ply")
        if tile is not None and tile.cloud_canvas is not None:
            tile.cloud_canvas.set_background(color)

    def _cloud_clip_changed(self, ranges: object) -> None:
        if not isinstance(ranges, dict):
            return
        try:
            self._cloud_clip_ranges = {
                axis: tuple(map(float, ranges[axis]))
                for axis in ("x", "y", "z")
            }
        except (KeyError, TypeError, ValueError):
            return
        self._apply_cloud_clip()

    def _apply_cloud_clip(self) -> None:
        tile = self.tile_pool.get("ply")
        if (
            tile is None
            or tile.cloud_canvas is None
            or self._cloud_clip_ranges is None
        ):
            return
        count = tile.cloud_canvas.set_clip_ranges(
            self._cloud_clip_ranges["x"],
            self._cloud_clip_ranges["y"],
            self._cloud_clip_ranges["z"],
        )
        self.status_text.setText(f"点云裁剪后 {count:,} 点")

    def open_cloud_projection(self) -> None:
        left = self.tile_pool.get("left")
        cloud_tile = self.tile_pool.get("ply")
        cloud = (
            cloud_tile.cloud_canvas.current_cloud
            if cloud_tile is not None and cloud_tile.cloud_canvas is not None
            else None
        )
        if left is None or left.image_data is None or cloud is None or cloud_tile is None:
            self.status_text.setText("需要已加载的左图和点云")
            return
        canvas = cloud_tile.cloud_canvas
        CloudProjectionDialog(
            left.image_data.image,
            cloud,
            canvas.camera_intrinsics,
            canvas.cam_offset,
            self,
        ).exec()

    def open_stereo_overlay(self, *, mode: str = "overlay") -> None:
        left = self.tile_pool.get("left")
        right = self.tile_pool.get("right")
        if (
            left is None
            or right is None
            or left.image_data is None
            or right.image_data is None
        ):
            return
        StereoOverlayDialog(
            left.image_data.image,
            right.image_data.image,
            self,
            mode=mode,
        ).exec()

    def focus_modality(self, modality: str) -> None:
        self.focused_modality = None if self.focused_modality == modality else modality
        self._apply_tile_layout()

    def toggle_focus(self) -> None:
        if self.focused_modality is not None:
            self.exit_focus()
        elif self.tiles:
            self.focus_modality(next(iter(self.tiles)))

    def exit_focus(self) -> None:
        if self.focused_modality is not None:
            self.focused_modality = None
            self._apply_tile_layout()

    def _show_current(self) -> None:
        if self.dataset is None or not self.dataset.samples:
            self._update_controls()
            return
        self.current_index = max(0, min(self.current_index, len(self.dataset.samples) - 1))
        sample = self.dataset.samples[self.current_index]
        key = (self.dataset.root, sample.key)
        if getattr(self, "_interaction_frame", None) != key:
            self._interaction_frame = key
            self._reset_interaction_tools(clear_selection=True)
            self.inspector.selection_label.setText("未选择区域或测量点")
        if self.crosshair_button.isChecked():
            self._media_cursor_left("")
        depth_tile = self.tile_pool.get("depth_fsd")
        depth_path = sample.files.get("depth_fsd")
        if depth_tile is not None:
            self.depth_quality_label.setText(
                "正在计算…"
                if depth_path is not None
                else "当前样本无深度图"
            )
        for modality, tile in self.tiles.items():
            tile.show_file(sample.files.get(modality))
        # Quality analysis and cursor inspection need depth even when the raw
        # depth tile is not one of the visible comparison views.
        if (
            depth_tile is not None
            and "depth_fsd" not in self.tiles
            and not self._playback_timer.isActive()
        ):
            depth_tile.show_file(depth_path)
        if depth_tile is not None and depth_tile.depth_stats is not None:
            self._update_depth_quality()

        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_index)
        self.timeline.blockSignals(False)
        self.timeline_left.setText(str(self.current_index + 1))
        self.timeline_left.setToolTip(sample.display_name)
        self.player_bar.setToolTip(f"{sample.display_name} · {len(sample.files)} 个文件")

        missing = summarize_missing(sample, self.selected_modalities())
        if missing:
            names = "、".join(modality_label(name) for name in missing)
            self.status_text.setText(f"当前样本缺少：{names}")
        else:
            self.status_text.setText("")
        self._update_controls()
        self._update_inspector()

    def _timeline_changed(self, value: int) -> None:
        self._seek_frame(value, pause=not self.timeline.isSliderDown())

    def _update_controls(self) -> None:
        has_samples = self.dataset is not None and bool(self.dataset.samples)
        count = len(self.dataset.samples) if self.dataset else 0
        for page, button in self.activity_buttons.items():
            button.setEnabled(self._activity_available(page))
        self.manual_mapping_button.setEnabled(self.dataset is not None)
        self.reset_button.setEnabled(bool(has_samples))
        self.timeline.setEnabled(bool(has_samples))
        self.playback_previous_button.setEnabled(bool(has_samples and self.current_index > 0))
        self.playback_next_button.setEnabled(
            bool(has_samples and self.current_index < count - 1)
        )
        self.playback_button.setEnabled(bool(has_samples and count > 1))
        self.playback_speed.setEnabled(bool(has_samples and count > 1))
        self._update_inspection_controls()

    def reset_views(self) -> None:
        for tile in self.tiles.values():
            tile.reset_view()
        self.status_text.setText("视图已重置")

    def toggle_sidebar(self) -> None:
        # Avoid resizing all image/OpenGL viewports on every animation tick.
        if self._sidebar_animation is not None:
            self._sidebar_animation.stop()
            self._sidebar_animation.deleteLater()
            self._sidebar_animation = None
        self.splitter.setUpdatesEnabled(False)
        try:
            expanding = self._sidebar_collapsed
            if expanding and self._active_sidebar_page not in {"data", "display"}:
                self._active_sidebar_page = "data"
            if expanding and not self._activity_available(self._active_sidebar_page):
                self._active_sidebar_page = "data"
            if expanding and self.inspector.isVisible():
                self.close_inspector()
            if not expanding:
                self._sidebar_expanded_width = min(
                    SIDEBAR_MAX_WIDTH, max(SIDEBAR_MIN_WIDTH, self.sidebar.width())
                )
            self._sidebar_collapsed = not expanding
            self.sidebar_panel.setVisible(expanding)
            width = self._sidebar_expanded_width if expanding else ACTIVITY_BAR_WIDTH
            self.sidebar.setMinimumWidth(SIDEBAR_MIN_WIDTH if expanding else ACTIVITY_BAR_WIDTH)
            self.sidebar.setMaximumWidth(SIDEBAR_MAX_WIDTH if expanding else ACTIVITY_BAR_WIDTH)
            self.sidebar_panel.setMinimumWidth(0)
            self.sidebar_panel.setMaximumWidth(SIDEBAR_MAX_WIDTH - ACTIVITY_BAR_WIDTH)
            sizes = self.splitter.sizes()
            sizes[1] = max(1, sizes[1] + sizes[0] - width)
            sizes[0] = width
            self.splitter.setSizes(sizes)
            self._refresh_activity_page()
            self.sidebar_button.set_direction("left" if expanding else "right")
            self.sidebar_button.setToolTip(
                "收起侧栏 (Ctrl+B)" if expanding else "展开侧栏 (Ctrl+B)"
            )
        finally:
            self.splitter.setUpdatesEnabled(True)

    def toggle_topbar(self) -> None:
        if self.dataset is None:
            return
        if self._topbar_animation is not None:
            self._topbar_animation.stop()
            self._topbar_animation.deleteLater()
            self._topbar_animation = None
        self._topbar_collapsed = not self._topbar_collapsed
        self.inspection_bar.setFixedHeight(
            INSPECTION_BAR_CURSOR_HEIGHT
            if self.crosshair_button.isChecked()
            else INSPECTION_BAR_HEIGHT
        )
        self.inspection_bar.setVisible(not self._topbar_collapsed)
        self.topbar_button.set_direction("down" if self._topbar_collapsed else "up")
        self.topbar_button.setToolTip(
            ("展开" if self._topbar_collapsed else "收起") + "检查工具栏"
        )

    def toggle_chrome(self) -> None:
        self._chrome_hidden = not self._chrome_hidden
        widgets = {
            "title": self.title_bar,
            "inspection": self.inspection_bar,
            "navigation": self.sidebar,
            "inspector": self.inspector,
            "bottom": self.player_bar,
        }
        if self._chrome_hidden:
            self._chrome_visibility = {
                name: widget.isVisible() for name, widget in widgets.items()
            }
            for widget in widgets.values():
                widget.hide()
            self.media_grid.setContentsMargins(
                MEDIA_COMPACT_MARGIN,
                MEDIA_COMPACT_MARGIN,
                MEDIA_COMPACT_MARGIN,
                MEDIA_COMPACT_MARGIN,
            )
            self.media_grid.setSpacing(MEDIA_COMPACT_SPACING)
            self.status_text.setText("")
        else:
            for name, widget in widgets.items():
                widget.setVisible(self._chrome_visibility.get(name, False))
            self.media_grid.setContentsMargins(MEDIA_MARGIN, MEDIA_MARGIN, MEDIA_MARGIN, MEDIA_MARGIN)
            self.media_grid.setSpacing(MEDIA_SPACING)
            self._refresh_activity_page()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized() if getattr(self, "_was_maximized_before_fullscreen", False) else self.showNormal()
            return
        self._was_maximized_before_fullscreen = self.isMaximized()
        self.showFullScreen()

    def open_command_palette(self) -> None:
        actions: list[tuple[str, str, object]] = [
            ("打开项目", "Ctrl+O", self.choose_project),
            ("播放 / 暂停", self._shortcut_text("playback"), self.toggle_playback),
            ("第一帧", self._shortcut_text("first"), self.first_sample),
            ("最后一帧", self._shortcut_text("last"), self.last_sample),
            ("上一帧", self._shortcut_text("previous"), self.previous_sample),
            ("下一帧", self._shortcut_text("next"), self.next_sample),
            ("显示 / 隐藏界面", "Tab", self.toggle_chrome),
            ("全屏", "F11", self.toggle_fullscreen),
            ("重置视图", self._shortcut_text("reset"), self.reset_views),
            ("打开检查器", "", self.open_inspector),
            ("左右图透明叠加", "", lambda: self.open_stereo_overlay(mode="overlay")),
            ("左右图绝对差值", "", lambda: self.open_stereo_overlay(mode="difference")),
            ("将点云投影到左图", "", self.open_cloud_projection),
            (
                "切换校正前 / 校正后",
                "",
                lambda: self.rectify_button.setChecked(
                    not self.rectify_button.isChecked()
                ),
            ),
            ("导出当前视图", "", self.export_current_view),
            ("导出调整后的原始分辨率图像", "", self.export_adjusted_image),
            ("设置", "Ctrl+,", self.open_settings),
        ]
        CommandPalette(actions, self).exec()

    def _shortcut_text(self, action: str) -> str:
        text = QKeySequence(self.preferences.shortcuts[action]).toString(QKeySequence.SequenceFormat.NativeText)
        return "Enter" if text in {"Return", "回车"} else text

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_native_window_frame()

    def _apply_native_window_frame(self) -> None:
        """Restore the Windows resize/snap frame while retaining our client title bar."""
        if sys.platform != "win32" or self._native_frame_applied:
            return
        self._native_frame_applied = True
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            get_style.restype = ctypes.c_ssize_t
            set_style.restype = ctypes.c_ssize_t
            style = get_style(hwnd, -16)  # GWL_STYLE
            style |= 0x00040000 | 0x00020000 | 0x00010000 | 0x00080000
            set_style(hwnd, -16, style)
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0020)
            corner = ctypes.c_int(2)  # DWMWCP_ROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(corner), ctypes.sizeof(corner))
        except (AttributeError, OSError, TypeError):
            self._native_frame_applied = False

    @staticmethod
    def _resize_hit_test(
        x: int,
        y: int,
        left: int,
        top: int,
        right: int,
        bottom: int,
        border: int,
    ) -> int | None:
        on_left = left <= x < left + border
        on_right = right - border <= x < right
        on_top = top <= y < top + border
        on_bottom = bottom - border <= y < bottom
        if on_top and on_left:
            return 13  # HTTOPLEFT
        if on_top and on_right:
            return 14  # HTTOPRIGHT
        if on_bottom and on_left:
            return 16  # HTBOTTOMLEFT
        if on_bottom and on_right:
            return 17  # HTBOTTOMRIGHT
        if on_left:
            return 10  # HTLEFT
        if on_right:
            return 11  # HTRIGHT
        if on_top:
            return 12  # HTTOP
        if on_bottom:
            return 15  # HTBOTTOM
        return None

    def toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    RESIZE_BORDER = 7

    def _system_resize_edges(self, x: float, y: float):
        """Return the Qt edge flags under (x, y) for platforms without a native frame."""
        if sys.platform == "win32" or self.isMaximized() or self.isFullScreen():
            return None
        border = self.RESIZE_BORDER
        edges = None
        for hit, edge in (
            (x < border, Qt.Edge.LeftEdge),
            (x >= self.width() - border, Qt.Edge.RightEdge),
            (y < border, Qt.Edge.TopEdge),
            (y >= self.height() - border, Qt.Edge.BottomEdge),
        ):
            if hit:
                edges = edge if edges is None else edges | edge
        return edges

    @staticmethod
    def _cursor_for_edges(edges) -> Qt.CursorShape:
        horizontal = bool(edges & Qt.Edge.LeftEdge) or bool(edges & Qt.Edge.RightEdge)
        vertical = bool(edges & Qt.Edge.TopEdge) or bool(edges & Qt.Edge.BottomEdge)
        if horizontal and vertical:
            top_left = bool(edges & Qt.Edge.TopEdge) == bool(edges & Qt.Edge.LeftEdge)
            return Qt.CursorShape.SizeFDiagCursor if top_left else Qt.CursorShape.SizeBDiagCursor
        return Qt.CursorShape.SizeHorCursor if horizontal else Qt.CursorShape.SizeVerCursor

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._system_resize_edges(event.position().x(), event.position().y())
            handle = self.windowHandle()
            if edges is not None and handle is not None and handle.startSystemResize(edges):
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if sys.platform != "win32":
            edges = self._system_resize_edges(event.position().x(), event.position().y())
            if edges is not None:
                self.setCursor(self._cursor_for_edges(edges))
            else:
                self.unsetCursor()
        super().mouseMoveEvent(event)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self.title_bar.set_maximized(self.isMaximized())

    def nativeEvent(self, event_type, message):
        if sys.platform == "win32" and not self.isMaximized() and not self.isFullScreen():
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0084:  # WM_NCHITTEST
                hwnd = int(self.winId())
                rect = wintypes.RECT()
                user32 = ctypes.windll.user32
                if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    packed = int(msg.lParam)
                    x = ctypes.c_short(packed & 0xFFFF).value
                    y = ctypes.c_short((packed >> 16) & 0xFFFF).value
                    dpi = user32.GetDpiForWindow(hwnd) if hasattr(user32, "GetDpiForWindow") else 96
                    border = max(10, round(12 * dpi / 96))
                    hit = self._resize_hit_test(x, y, rect.left, rect.top, rect.right, rect.bottom, border)
                    if hit is not None:
                        return True, hit
        return super().nativeEvent(event_type, message)

    def closeEvent(self, event) -> None:
        self._playback_timer.stop()
        self._cursor_update_timer.stop()
        self._rectify_token = None
        for tile in self.tile_pool.values():
            tile.dispose()
        application = QApplication.instance()
        if application is not None and hasattr(self, "tooltip_manager"):
            self.tooltip_manager.dispose()
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).is_dir():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_dir():
                self.load_project(path)
                event.acceptProposedAction()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="双目图像、深度与点云查看工作台")
    parser.add_argument("project", nargs="?", type=Path, help="启动时打开的项目文件夹")
    parser.add_argument("--verbose", action="store_true", help="输出调试级别日志")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def start_smoke_test(app: QApplication, window: MainWindow, timeout_seconds: float = 30.0) -> None:
    """Exercise every available renderer in a packaged build and exit with a status code."""
    started = time.monotonic()
    configured = False
    timer = QTimer(window)
    timer.setInterval(100)

    def check() -> None:
        nonlocal configured
        if time.monotonic() - started > timeout_seconds:
            timer.stop()
            window.close()
            app.exit(3)
            return
        if window.dataset is None or not window.dataset.samples:
            return
        if not configured:
            for checkbox in window.checkboxes.values():
                checkbox.blockSignals(True)
                checkbox.setChecked(checkbox.isEnabled())
                checkbox.blockSignals(False)
            window._rebuild_tiles()
            window._show_current()
            configured = True
            return
        if any(tile.has_pending_work() for tile in window.tiles.values()):
            return
        failed = []
        for modality, tile in window.tiles.items():
            expected = tile.cloud_canvas if modality == "ply" else tile.image_canvas
            if expected is None or not tile.showing_canvas():
                failed.append(modality)
        timer.stop()
        window.close()
        app.exit(2 if failed else 0)

    timer.timeout.connect(check)
    timer.start()


def run_application(
    app: QApplication,
    argv: list[str] | None = None,
    splash: QWidget | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_test and args.project is None:
        raise SystemExit("--smoke-test requires a project folder")
    app.setStyle("Fusion")
    startup_preferences = AppPreferences.load(QSettings("ToolBox", "StereoSelector"))
    app.setStyleSheet(style_for(startup_preferences.theme))
    app.setPalette(palette_for(startup_preferences.theme))
    configure_pools()
    window = MainWindow(args.project)
    window.setWindowIcon(app.windowIcon())
    window.show()
    app.processEvents()
    if splash is not None:
        splash.close()
    if args.smoke_test:
        start_smoke_test(app, window)
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    from .bootstrap import main as bootstrap_main

    return bootstrap_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
