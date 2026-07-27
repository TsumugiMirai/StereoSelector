from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QSettings, QThreadPool, Qt, QTimer
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .models import (
    Dataset,
    DatasetScanner,
    MODALITY_INFO,
    copy_sample,
    modality_label,
    sample_is_copied,
    summarize_missing,
)
from .calibration import (
    CalibrationData,
    CalibrationOption,
    builtin_calibration_options,
    custom_calibration_option,
    epiline_for_point,
    load_calibration,
)
from .inspection import StereoOverlayDialog
from .mapping import MappingDialog
from .review import AnnotationDialog, OutputSettingsDialog, ReviewStore
from .settings import AppPreferences, ChoiceButton, SettingsDialog
from .theme import style_for
from .widgets import DropHint, MediaTile, NoViewsHint, PointCloudCanvas, TitleBar


class MainWindow(QMainWindow):
    def __init__(self, initial_project: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Stereo Selector")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowSystemMenuHint
        )
        self.resize(1520, 920)
        self.setMinimumSize(1050, 680)
        self.setAcceptDrops(True)

        self.settings = QSettings("ToolBox", "StereoSelector")
        self.preferences = AppPreferences.load(self.settings)
        self.scanner = DatasetScanner()
        self.dataset: Dataset | None = None
        self.current_root: Path | None = None
        self.manual_dirs: dict[str, Path] = {}
        self.force_order = False
        self.current_index = 0
        self.accepted: set[str] = set()
        self.review_store: ReviewStore | None = None
        self.calibration: CalibrationData | None = None
        self.current_calibration_id = ""
        self.calibration_options = self._load_calibration_options()
        self.focused_modality: str | None = None
        self._crosshair_position: tuple[float, float] | None = None
        self.checkboxes: dict[str, QCheckBox] = {}
        self.count_labels: dict[str, QLabel] = {}
        self.tiles: dict[str, MediaTile] = {}
        self.tile_pool: dict[str, MediaTile] = {}
        self.shortcuts: list[QShortcut] = []
        self.action_shortcuts: dict[str, QShortcut] = {}
        self._native_frame_applied = False
        self._sidebar_animation: QPropertyAnimation | None = None
        self._sidebar_expanded_width = 286
        self._sidebar_collapsed = False
        self._settings_page: SettingsDialog | None = None
        self._settings_previous_context: tuple[str, bool] = ("", False)

        self._build_ui()
        self._build_shortcuts()
        self._apply_preferences()
        self._update_controls()

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

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.splitter, 1)

        self.sidebar = self._build_sidebar()
        workspace = self._build_workspace()
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(workspace)
        self.splitter.setSizes([286, 1234])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        self.app_status_bar = QStatusBar()
        self.app_status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.app_status_bar)
        self.status_text = QLabel("")
        self.status_text.setContentsMargins(7, 0, 0, 0)
        self.app_status_bar.addWidget(self.status_text, 1)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(238)
        sidebar.setMaximumWidth(360)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(8, 7, 8, 8)
        side.setSpacing(6)

        project = QWidget()
        project.setObjectName("sidebarSection")
        project_layout = QVBoxLayout(project)
        project_layout.setContentsMargins(16, 18, 16, 14)
        project_layout.setSpacing(5)
        eyebrow = QLabel("PROJECT")
        eyebrow.setObjectName("eyebrow")
        self.project_name_label = QLabel("未打开项目")
        self.project_name_label.setObjectName("projectName")
        self.project_name_label.setWordWrap(True)
        self.project_path_label = QLabel("")
        self.project_path_label.setObjectName("projectPath")
        self.project_path_label.setWordWrap(True)
        self.project_path_label.hide()
        self.project_summary_label = QLabel("")
        self.project_summary_label.setObjectName("muted")
        self.project_summary_label.setWordWrap(True)
        self.project_summary_label.hide()
        self.change_button = QPushButton("打开项目")
        self.change_button.setObjectName("ghostButton")
        self.change_button.clicked.connect(self.choose_project)
        self.change_button.hide()
        project_layout.addWidget(eyebrow)
        project_layout.addWidget(self.project_name_label)
        project_layout.addWidget(self.project_path_label)
        project_layout.addSpacing(3)
        project_layout.addWidget(self.project_summary_label)
        project_actions = QHBoxLayout()
        project_actions.setSpacing(4)
        project_actions.addWidget(self.change_button)
        self.manual_mapping_button = QPushButton("视图映射")
        self.manual_mapping_button.setObjectName("ghostButton")
        self.manual_mapping_button.clicked.connect(self.open_mapping_settings)
        self.manual_mapping_button.hide()
        project_actions.addWidget(self.manual_mapping_button)
        project_actions.addStretch(1)
        project_layout.addLayout(project_actions)
        side.addWidget(project)

        self.modes_panel = QWidget()
        self.modes_panel.setObjectName("sidebarSection")
        modes_layout = QVBoxLayout(self.modes_panel)
        modes_layout.setContentsMargins(16, 9, 16, 9)
        modes_layout.setSpacing(1)
        modes_header = QHBoxLayout()
        modes_title = QLabel("对比视图")
        modes_title.setObjectName("sectionLabel")
        self.selected_count_label = QLabel("已选 0")
        self.selected_count_label.setObjectName("muted")
        modes_header.addWidget(modes_title)
        modes_header.addStretch(1)
        modes_header.addWidget(self.selected_count_label)
        modes_layout.addLayout(modes_header)
        modes_layout.addSpacing(5)
        for index, modality in enumerate(MODALITY_INFO, start=1):
            row = QHBoxLayout()
            checkbox = QCheckBox(f"{index}   {modality_label(modality)}")
            checkbox.setEnabled(False)
            checkbox.toggled.connect(lambda checked, name=modality: self._modality_toggled(name, checked))
            count = QLabel("0")
            count.setObjectName("countBadge")
            count.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.checkboxes[modality] = checkbox
            self.count_labels[modality] = count
            row.addWidget(checkbox, 1)
            row.addWidget(count)
            modes_layout.addLayout(row)
        self.modes_panel.hide()
        side.addWidget(self.modes_panel)

        self.quality_panel = QWidget()
        self.quality_panel.setObjectName("sidebarSection")
        quality_layout = QVBoxLayout(self.quality_panel)
        quality_layout.setContentsMargins(16, 10, 16, 8)
        quality_layout.setSpacing(6)
        quality_title = QLabel("深度质量")
        quality_title.setObjectName("sectionLabel")
        self.depth_quality_label = QLabel("")
        self.depth_quality_label.setObjectName("qualitySummary")
        self.depth_quality_label.setWordWrap(True)
        quality_layout.addWidget(quality_title)
        quality_layout.addWidget(self.depth_quality_label)
        self.quality_panel.hide()
        side.addWidget(self.quality_panel)

        self.review_panel = QWidget()
        self.review_panel.setObjectName("sidebarSection")
        review_wrap_layout = QVBoxLayout(self.review_panel)
        review_wrap_layout.setContentsMargins(14, 10, 14, 10)
        review_wrap_layout.setSpacing(7)
        review_title = QLabel("审核进度")
        review_title.setObjectName("sectionLabel")
        review_wrap_layout.addWidget(review_title)
        self.review_card = QFrame()
        self.review_card.setObjectName("reviewCard")
        card_layout = QVBoxLayout(self.review_card)
        card_layout.setContentsMargins(13, 11, 13, 11)
        card_layout.setSpacing(6)
        stats = QHBoxLayout()
        self.accepted_count_label = QLabel("0")
        self.accepted_count_label.setObjectName("acceptedCount")
        self.accepted_caption_label = QLabel("已接受\n剩余 0")
        self.accepted_caption_label.setObjectName("acceptedCaption")
        stats.addWidget(self.accepted_count_label)
        stats.addSpacing(4)
        stats.addWidget(self.accepted_caption_label)
        stats.addStretch(1)
        card_layout.addLayout(stats)
        self.review_progress = QProgressBar()
        self.review_progress.setTextVisible(False)
        self.review_progress.setRange(0, 1)
        card_layout.addWidget(self.review_progress)
        self.next_unreviewed_button = QPushButton("下一组未接受")
        self.next_unreviewed_button.setObjectName("ghostButton")
        self.next_unreviewed_button.clicked.connect(self.next_unreviewed)
        card_layout.addWidget(self.next_unreviewed_button)
        review_wrap_layout.addWidget(self.review_card)
        self.review_panel.hide()
        side.addWidget(self.review_panel)
        side.addStretch(1)

        self.output_panel = QWidget()
        self.output_panel.setObjectName("sidebarSection")
        output_layout = QVBoxLayout(self.output_panel)
        output_layout.setContentsMargins(14, 9, 14, 14)
        output_layout.setSpacing(6)
        output_title = QLabel("输出")
        output_title.setObjectName("sectionLabel")
        self.output_label = QLabel("")
        self.output_label.setObjectName("projectPath")
        self.output_label.setWordWrap(True)
        self.open_output_button = QPushButton("打开输出文件夹")
        self.open_output_button.setObjectName("ghostButton")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.configure_output_button = QPushButton("设置输出")
        self.configure_output_button.setObjectName("ghostButton")
        self.configure_output_button.setEnabled(False)
        self.configure_output_button.clicked.connect(self.configure_output)
        output_layout.addWidget(output_title)
        output_layout.addWidget(self.output_label)
        output_actions = QHBoxLayout()
        output_actions.setSpacing(4)
        output_actions.addWidget(self.configure_output_button)
        output_actions.addWidget(self.open_output_button)
        output_layout.addLayout(output_actions)
        self.output_panel.hide()
        side.addWidget(self.output_panel)
        return sidebar

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setObjectName("workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("workspaceHeader")
        header.setFixedHeight(58)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 9, 14, 9)
        header_layout.setSpacing(8)
        self.sidebar_button = QPushButton("‹")
        self.sidebar_button.setObjectName("sideToggleButton")
        self.sidebar_button.setFixedSize(32, 32)
        self.sidebar_button.setToolTip("收起侧栏 (Ctrl+B)")
        self.sidebar_button.clicked.connect(self.toggle_sidebar)
        breadcrumb_box = QVBoxLayout()
        breadcrumb_box.setSpacing(1)
        self.header_breadcrumb = QLabel("未打开项目")
        self.header_breadcrumb.setObjectName("breadcrumb")
        self.header_breadcrumb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.view_hint = QLabel("")
        self.view_hint.setObjectName("viewHint")
        self.view_hint.hide()
        breadcrumb_box.addWidget(self.header_breadcrumb)
        breadcrumb_box.addWidget(self.view_hint)
        self.reset_button = QPushButton("重置视图")
        self.reset_button.setObjectName("ghostButton")
        self.reset_button.clicked.connect(self.reset_views)
        self.reset_button.hide()
        self.open_button = QPushButton("打开项目…")
        self.open_button.setObjectName("primaryButton")
        self.open_button.clicked.connect(self.choose_project)
        header_layout.addWidget(self.sidebar_button)
        header_layout.addLayout(breadcrumb_box, 1)
        header_layout.addWidget(self.reset_button)
        header_layout.addWidget(self.open_button)
        workspace_layout.addWidget(header)

        self.inspection_bar = QFrame()
        self.inspection_bar.setObjectName("inspectionBar")
        inspection_layout = QHBoxLayout(self.inspection_bar)
        inspection_layout.setContentsMargins(12, 6, 14, 6)
        inspection_layout.setSpacing(5)
        self.crosshair_button = QPushButton("十字光标")
        self.crosshair_button.setObjectName("toolToggle")
        self.crosshair_button.setCheckable(True)
        self.crosshair_button.toggled.connect(self._toggle_crosshair)
        self.sync_views_button = QPushButton("同步视图")
        self.sync_views_button.setObjectName("toolToggle")
        self.sync_views_button.setCheckable(True)
        self.sync_views_button.toggled.connect(self._sync_views_toggled)
        self.epiline_button = QPushButton("极线")
        self.epiline_button.setObjectName("toolToggle")
        self.epiline_button.setCheckable(True)
        self.epiline_button.toggled.connect(self._epilines_toggled)
        self.overlay_button = QPushButton("左右叠加")
        self.overlay_button.setObjectName("toolButton")
        self.overlay_button.clicked.connect(self.open_stereo_overlay)
        calibration_caption = QLabel("标定")
        calibration_caption.setObjectName("toolMeta")
        self.calibration_picker = ChoiceButton()
        self.calibration_picker.setObjectName("calibrationPicker")
        self.calibration_picker.setMinimumWidth(132)
        self.calibration_picker.currentIndexChanged.connect(self._calibration_picker_changed)
        self.cursor_info = QLabel("")
        self.cursor_info.setObjectName("cursorInfo")
        self.cursor_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        inspection_layout.addWidget(self.crosshair_button)
        inspection_layout.addWidget(self.sync_views_button)
        inspection_layout.addWidget(self.epiline_button)
        inspection_layout.addWidget(self.overlay_button)
        inspection_layout.addSpacing(6)
        inspection_layout.addWidget(calibration_caption)
        inspection_layout.addWidget(self.calibration_picker)
        inspection_layout.addStretch(1)
        inspection_layout.addWidget(self.cursor_info)
        self.inspection_bar.hide()
        workspace_layout.addWidget(self.inspection_bar)

        self.content_stack = QStackedWidget()
        self.empty_hint = DropHint()
        self.empty_hint.open_requested.connect(self.choose_project)
        self.content_stack.addWidget(self.empty_hint)
        self.media_container = QWidget()
        self.media_container.setObjectName("mediaWorkspace")
        self.media_grid = QGridLayout(self.media_container)
        self.media_grid.setContentsMargins(10, 10, 10, 10)
        self.media_grid.setSpacing(10)
        self.no_views_hint = NoViewsHint()
        # Create the native OpenGL child before the frameless top-level window is
        # shown. The backend itself is imported in parallel while the splash is up.
        self.gl_warmup = PointCloudCanvas(self.media_container)
        self.gl_warmup.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        self.gl_warmup.setGeometry(0, 0, 2, 2)
        self.gl_warmup.show()
        self.content_stack.addWidget(self.media_container)
        workspace_layout.addWidget(self.content_stack, 1)

        self.review_bar = QFrame()
        self.review_bar.setObjectName("reviewBar")
        self.review_bar.setProperty("reviewState", "pending")
        review_layout = QVBoxLayout(self.review_bar)
        review_layout.setContentsMargins(14, 8, 14, 10)
        review_layout.setSpacing(7)
        timeline_row = QHBoxLayout()
        self.timeline_left = QLabel("0")
        self.timeline_left.setObjectName("sampleMeta")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.setEnabled(False)
        self.timeline.valueChanged.connect(self._timeline_changed)
        self.timeline_right = QLabel("0")
        self.timeline_right.setObjectName("sampleMeta")
        timeline_row.addWidget(self.timeline_left)
        timeline_row.addWidget(self.timeline, 1)
        timeline_row.addWidget(self.timeline_right)
        review_layout.addLayout(timeline_row)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.previous_button = QPushButton("上一组")
        self.previous_button.clicked.connect(self.previous_sample)
        self.next_button = QPushButton("下一组")
        self.next_button.clicked.connect(self.next_sample)
        sample_box = QVBoxLayout()
        sample_box.setSpacing(1)
        sample_top = QHBoxLayout()
        self.sample_title = QLabel("")
        self.sample_title.setObjectName("sampleTitle")
        self.sample_status = QLabel("")
        self.sample_status.setObjectName("statusPill")
        self.sample_status.setProperty("accepted", False)
        self.annotation_button = QPushButton("缺陷与备注")
        self.annotation_button.setObjectName("ghostButton")
        self.annotation_button.clicked.connect(self.open_annotation)
        sample_top.addWidget(self.sample_title)
        sample_top.addWidget(self.sample_status)
        sample_top.addStretch(1)
        self.sample_meta = QLabel("")
        self.sample_meta.setObjectName("sampleMeta")
        sample_box.addLayout(sample_top)
        sample_box.addWidget(self.sample_meta)
        self.accept_button = QPushButton("接受")
        self.accept_button.setObjectName("acceptButton")
        self.accept_button.clicked.connect(self.accept_current)
        controls.addWidget(self.previous_button)
        controls.addWidget(self.next_button)
        controls.addSpacing(10)
        controls.addLayout(sample_box, 1)
        controls.addWidget(self.annotation_button)
        controls.addWidget(self.accept_button)
        review_layout.addLayout(controls)
        self.review_bar.hide()
        workspace_layout.addWidget(self.review_bar)
        return workspace

    def _build_shortcuts(self) -> None:
        configurable = {
            "previous": self.previous_sample,
            "next": self.next_sample,
            "accept": self.accept_current,
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
            ("Home", self.first_sample),
            ("End", self.last_sample),
            ("Escape", self.exit_focus),
            ("Ctrl+B", self.toggle_sidebar),
            ("Ctrl+O", self.choose_project),
            ("N", self.next_unreviewed),
            ("Ctrl+,", self.open_settings),
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
    ) -> None:
        previous_root = self.dataset.root if self.dataset is not None else None
        root = root.expanduser().resolve()
        self.current_root = root
        if manual_dirs is None:
            manual_dirs, force_order = self._saved_matching_for(root)
        output_root = self._saved_output_for(root)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            dataset = self.scanner.scan(
                root,
                manual_dirs=manual_dirs,
                force_order=force_order,
                output_root=output_root,
            )
        except Exception as exc:
            self.current_root = previous_root
            QMessageBox.critical(self, "无法打开项目", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

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
            return

        self.dataset = dataset
        self.current_calibration_id, self.calibration = self._saved_calibration_selection(
            dataset.root
        )
        self.manual_dirs = dict(manual_dirs or {})
        self.force_order = force_order
        self.current_index = 0
        self.focused_modality = None
        try:
            self.review_store = ReviewStore(dataset)
            self.accepted = self.review_store.reconcile()
        except Exception as exc:
            self.review_store = None
            self.accepted = {
                sample.key
                for sample in dataset.samples
                if sample_is_copied(dataset, sample)
            }
            QMessageBox.warning(self, "审核文件不可用", str(exc))
        self.settings.setValue("last_project", str(dataset.root))
        self._save_matching_for(dataset.root, self.manual_dirs, self.force_order)
        self.setWindowTitle(f"{dataset.root.name} — Stereo Selector")
        self.title_bar.set_context(dataset.root.name)
        self.project_name_label.setText(dataset.root.name)
        self.project_path_label.setText(str(dataset.root))
        self.project_path_label.setToolTip(str(dataset.root))
        self.project_path_label.show()
        available_set = set(dataset.available_modalities)
        complete_count = sum(available_set.issubset(sample.files) for sample in dataset.samples)
        incomplete_count = len(dataset.samples) - complete_count
        incomplete_text = f" · {incomplete_count} 组不完整" if incomplete_count else ""
        force_order_text = " · 强制顺序" if dataset.force_order else ""
        self.project_summary_label.setText(
            f"{len(dataset.samples)} 组 · {len(dataset.available_modalities)} 种数据"
            f"{force_order_text}{incomplete_text}"
        )
        self.project_summary_label.show()
        self.output_label.setText(str(dataset.output_root))
        self.output_label.setToolTip(str(dataset.output_root))
        self.change_button.setText("更改项目")
        self.change_button.show()
        self.manual_mapping_button.show()
        self.modes_panel.show()
        self.quality_panel.setVisible("depth_fsd" in dataset.available_modalities)
        self.review_panel.show()
        self.output_panel.show()
        self.reset_button.show()
        self.review_bar.show()
        self.inspection_bar.show()
        self._refresh_calibration_picker(self.current_calibration_id)
        self.crosshair_button.setChecked(False)
        self.epiline_button.setChecked(False)
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
        self.content_stack.setCurrentWidget(self.media_container)
        self._build_tile_pool()
        self._rebuild_tiles()
        self._update_review_stats()
        self._show_current()
        self.status_text.setText(f"已打开 {dataset.root.name}")

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

    def _saved_output_for(self, root: Path) -> Path | None:
        prefix = self._matching_settings_prefix(root)
        value = str(self.settings.value(f"{prefix}/output_root", "")).strip()
        if not value:
            return None
        candidate = Path(value).expanduser().resolve()
        if candidate == root or candidate.is_relative_to(root):
            return None
        return candidate

    def _save_output_for(self, root: Path, output_root: Path) -> None:
        prefix = self._matching_settings_prefix(root)
        self.settings.setValue(f"{prefix}/output_root", str(output_root))

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
            if not option.builtin and option.path.is_file()
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
            return option.id, load_calibration(option.path)
        except (ValueError, OSError):
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
                f"{self.calibration.summary}\n{self.calibration.source}"
            )
        else:
            self.calibration_picker.setToolTip("选择标定预设")

    def _apply_calibration_selection(
        self,
        option_id: str,
        *,
        rebuild: bool = True,
    ) -> bool:
        option = self._calibration_option(option_id)
        try:
            calibration = load_calibration(option.path) if option is not None else None
        except Exception as exc:
            QMessageBox.critical(self, "无法加载标定", str(exc))
            self._refresh_calibration_picker(self.current_calibration_id)
            return False
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
        for modality in self.dataset.available_modalities:
            tile = MediaTile(
                modality,
                self.preferences.point_limit,
                self.media_container,
                cloud_rotation=(
                    self.calibration.cloud_rotation
                    if self.calibration is not None
                    else None
                ),
                cloud_translation=(
                    self.calibration.cloud_translation
                    if self.calibration is not None
                    else None
                ),
            )
            tile.focus_requested.connect(self.focus_modality)
            tile.cursor_moved.connect(self._media_cursor_moved)
            tile.cursor_left.connect(self._media_cursor_left)
            tile.view_changed.connect(self._media_view_changed)
            tile.data_ready.connect(self._media_data_ready)
            tile.hide()
            self.tile_pool[modality] = tile

    def _rebuild_tiles(self) -> None:
        self._clear_grid()
        selected_names = set(self.selected_modalities())
        for name, tile in self.tile_pool.items():
            tile.hide()
            tile.set_focused(False)
            if name not in selected_names:
                tile.cancel_pending()
        self.tiles = {
            modality: self.tile_pool[modality]
            for modality in MODALITY_INFO
            if modality in selected_names
            if modality in self.tile_pool
        }
        self._apply_tile_layout()
        self.selected_count_label.setText(f"已选 {len(self.tiles)}")

    def _apply_tile_layout(self) -> None:
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
        columns = 1 if len(selected) == 1 else (3 if len(selected) in (3, 5) else 2)
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
        if checked:
            self.cursor_info.setText("在图片上移动鼠标")
            return
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

    def _media_cursor_moved(self, modality: str, x: float, y: float) -> None:
        if not self.crosshair_button.isChecked():
            return
        self._crosshair_position = (x, y)
        for tile in self.tiles.values():
            if tile.image_data is not None:
                tile.image_canvas.set_crosshair(x, y, True)
                tile.image_canvas.set_epiline(None)
        self.cursor_info.setText(self._cursor_value_text(x, y))
        if self.epiline_button.isChecked():
            self._update_epiline(modality, x, y)

    def _media_cursor_left(self, modality: str) -> None:
        if not self.crosshair_button.isChecked():
            return
        self._crosshair_position = None
        for tile in self.tile_pool.values():
            tile.image_canvas.set_crosshair(0, 0, False)
            tile.image_canvas.set_epiline(None)
        self.cursor_info.setText("在图片上移动鼠标")

    def _cursor_value_text(self, x: float, y: float) -> str:
        parts: list[str] = []
        left = self.tile_pool.get("left")
        right = self.tile_pool.get("right")
        rgb_tile = (
            left
            if left is not None and self.checkboxes["left"].isChecked()
            else right
            if right is not None and self.checkboxes["right"].isChecked()
            else None
        )
        if rgb_tile is not None and rgb_tile.image_data is not None:
            image = rgb_tile.image_data.image
            px = min(image.width() - 1, max(0, int(x * image.width())))
            py = min(image.height() - 1, max(0, int(y * image.height())))
            color = image.pixelColor(px, py)
            parts.append(f"RGB {color.red()}, {color.green()}, {color.blue()}")

        depth_value: float | None = None
        depth = self.tile_pool.get("depth_fsd")
        if (
            depth is not None
            and depth.image_data is not None
            and self.checkboxes["depth_fsd"].isChecked()
        ):
            values = depth.image_data.values
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
                parts.append(f"深度 {depth_value:g}")
            else:
                depth_value = None
                parts.append("深度 无效")

        if (
            depth_value is not None
            and depth_value > 0
            and left is not None
            and left.image_data is not None
            and self.checkboxes["left"].isChecked()
            and self.calibration is not None
            and self.calibration.left is not None
        ):
            image = left.image_data.image
            px = x * image.width()
            py = y * image.height()
            point = self.calibration.left.point_from_depth(px, py, depth_value)
            parts.append(f"XYZ {point[0]:.3g}, {point[1]:.3g}, {point[2]:.3g}")
        elif depth_value is not None and depth_value > 0:
            parts.append("XYZ 需左目内参")
        return "   ·   ".join(parts) or "无像素数据"

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
        self._update_inspection_controls()

    def _update_depth_quality(self) -> None:
        tile = self.tile_pool.get("depth_fsd")
        stats = tile.depth_stats if tile is not None else None
        if stats is None:
            self.depth_quality_label.setText("等待深度图")
            return
        value_range = (
            f"{stats.minimum:g} – {stats.maximum:g}"
            if stats.minimum is not None and stats.maximum is not None
            else "无有效值"
        )
        extreme = (
            f"{stats.extreme_ratio:.2%}（>{stats.extreme_threshold:g}）"
            if stats.extreme_threshold is not None
            else "0.00%"
        )
        self.depth_quality_label.setText(
            f"范围  {value_range}\n"
            f"无效  {stats.invalid_ratio:.2%}   零值  {stats.zero_ratio:.2%}\n"
            f"空洞  {stats.hole_ratio:.2%}   极大值  {extreme}"
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
        self.crosshair_button.setEnabled(
            any(tile.image_data is not None for tile in self.tiles.values())
        )
        self.sync_views_button.setEnabled(
            sum(tile.image_data is not None for tile in self.tiles.values()) >= 2
        )

    def open_stereo_overlay(self) -> None:
        left = self.tile_pool.get("left")
        right = self.tile_pool.get("right")
        if (
            left is None
            or right is None
            or left.image_data is None
            or right.image_data is None
        ):
            return
        StereoOverlayDialog(left.image_data.image, right.image_data.image, self).exec()

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
        if self.crosshair_button.isChecked():
            self._media_cursor_left("")
        for modality, tile in self.tiles.items():
            tile.show_file(sample.files.get(modality))
        if "depth_fsd" in self.tile_pool:
            self.depth_quality_label.setText(
                "加载当前深度图…"
                if self.checkboxes["depth_fsd"].isChecked()
                else "选择深度图后显示分析"
            )

        self.sample_title.setText(sample.display_name)
        self.sample_meta.setText(
            f"第 {self.current_index + 1} / {len(self.dataset.samples)} 组  ·  {len(sample.files)} 个文件"
        )
        is_accepted = sample.key in self.accepted
        self.sample_status.setText("已接受" if is_accepted else "待审阅")
        review_state = "accepted" if is_accepted else "pending"
        if self.review_bar.property("reviewState") != review_state:
            self.review_bar.setProperty("reviewState", review_state)
            for widget in (self.review_bar, self.sample_title, self.accept_button):
                widget.style().unpolish(widget)
                widget.style().polish(widget)
        if bool(self.sample_status.property("accepted")) != is_accepted:
            self.sample_status.setProperty("accepted", is_accepted)
            self.sample_status.style().unpolish(self.sample_status)
            self.sample_status.style().polish(self.sample_status)
        accept_label = self.preferences.button_labels["accept"]
        self.accept_button.setText("继续" if is_accepted else accept_label)
        self._update_annotation_button(sample)
        self.header_breadcrumb.setText(f"{self.dataset.root.name}  /  {sample.display_name}")
        self.header_breadcrumb.setToolTip(str(self.dataset.root))
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_index)
        self.timeline.blockSignals(False)
        self.timeline_left.setText(str(self.current_index + 1))

        missing = summarize_missing(sample, self.selected_modalities())
        if missing:
            names = "、".join(modality_label(name) for name in missing)
            self.status_text.setText(f"当前样本缺少：{names}")
        else:
            self.status_text.setText("")
        self._update_controls()

    def _timeline_changed(self, value: int) -> None:
        if self.dataset is not None and value != self.current_index:
            self.current_index = value
            self._show_current()

    def _update_controls(self) -> None:
        has_samples = self.dataset is not None and bool(self.dataset.samples)
        count = len(self.dataset.samples) if self.dataset else 0
        self.previous_button.setEnabled(bool(has_samples and self.current_index > 0))
        self.next_button.setEnabled(bool(has_samples and self.current_index < count - 1))
        self.accept_button.setEnabled(bool(has_samples))
        self.reset_button.setEnabled(bool(has_samples))
        self.timeline.setEnabled(bool(has_samples))
        self.open_output_button.setEnabled(self.dataset is not None)
        self.configure_output_button.setEnabled(self.dataset is not None)
        self.annotation_button.setEnabled(bool(has_samples))
        self.next_unreviewed_button.setEnabled(bool(has_samples and len(self.accepted) < count))
        self._update_inspection_controls()

    def _update_review_stats(self) -> None:
        total = len(self.dataset.samples) if self.dataset else 0
        accepted = len(self.accepted)
        self.accepted_count_label.setText(str(accepted))
        self.accepted_caption_label.setText(f"已接受\n剩余 {max(0, total - accepted)}")
        self.review_progress.setRange(0, max(1, total))
        self.review_progress.setValue(accepted)
        self._update_controls()

    def previous_sample(self) -> None:
        if self.dataset and self.current_index > 0:
            self.current_index -= 1
            self._show_current()

    def next_sample(self) -> None:
        if self.dataset and self.current_index < len(self.dataset.samples) - 1:
            self.current_index += 1
            self._show_current()

    def first_sample(self) -> None:
        if self.dataset and self.dataset.samples:
            self.current_index = 0
            self._show_current()

    def last_sample(self) -> None:
        if self.dataset and self.dataset.samples:
            self.current_index = len(self.dataset.samples) - 1
            self._show_current()

    def next_unreviewed(self) -> None:
        if self.dataset is None or not self.dataset.samples:
            return
        indices = list(range(self.current_index + 1, len(self.dataset.samples))) + list(range(0, self.current_index + 1))
        for index in indices:
            if self.dataset.samples[index].key not in self.accepted:
                self.current_index = index
                self._show_current()
                return
        self.status_text.setText("全部已接受")

    def accept_current(self) -> None:
        if self.dataset is None or not self.dataset.samples:
            return
        sample = self.dataset.samples[self.current_index]
        copied_count = 0
        if sample.key not in self.accepted:
            try:
                copied_count = len(copy_sample(self.dataset, sample))
            except Exception as exc:
                QMessageBox.critical(self, "复制失败", f"无法接受当前样本：\n{exc}")
                return
            self.accepted.add(sample.key)
            if self.review_store is not None:
                try:
                    self.review_store.set_status(sample, "accepted")
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "审核记录未保存",
                        f"图片已复制，但无法更新 JSON 审核文件：\n{exc}",
                    )
        self._update_review_stats()
        if self.preferences.auto_advance and self.current_index < len(self.dataset.samples) - 1:
            self.current_index += 1
        self._show_current()
        if copied_count:
            self.status_text.setText(f"已接受 {sample.display_name}")
        else:
            self.status_text.setText("已继续")

    def reset_views(self) -> None:
        for tile in self.tiles.values():
            tile.reset_view()
        self.status_text.setText("视图已重置")

    def toggle_sidebar(self) -> None:
        expanding = self._sidebar_collapsed
        self._sidebar_collapsed = not expanding
        if self._sidebar_animation is not None:
            self._sidebar_animation.stop()
        if expanding:
            self.sidebar.show()
            self.sidebar.setMinimumWidth(0)
            self.sidebar.setMaximumWidth(0)
            start, end = 0, self._sidebar_expanded_width
        else:
            self._sidebar_expanded_width = max(238, self.sidebar.width())
            self.sidebar.setMinimumWidth(0)
            self.sidebar.setMaximumWidth(self.sidebar.width())
            start, end = self.sidebar.width(), 0

        animation = QPropertyAnimation(self.sidebar, b"maximumWidth", self)
        animation.setDuration(170)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        def finish() -> None:
            if expanding:
                self.sidebar.setMinimumWidth(238)
                self.sidebar.setMaximumWidth(360)
            else:
                self.sidebar.hide()
                self.sidebar.setMaximumWidth(360)

        animation.finished.connect(finish)
        self._sidebar_animation = animation
        animation.start()
        self.sidebar_button.setText("‹" if expanding else "›")
        self.sidebar_button.setToolTip(
            ("收起" if expanding else "展开") + "侧栏 (Ctrl+B)"
        )

    def _shortcut_text(self, action: str) -> str:
        text = QKeySequence(self.preferences.shortcuts[action]).toString(QKeySequence.SequenceFormat.NativeText)
        return "Enter" if text in {"Return", "回车"} else text

    def _apply_preferences(self) -> None:
        QApplication.instance().setStyleSheet(style_for(self.preferences.theme))
        for action, shortcut in self.action_shortcuts.items():
            shortcut.setKey(QKeySequence(self.preferences.shortcuts[action]))
        self.previous_button.setText(self.preferences.button_labels["previous"])
        self.next_button.setText(self.preferences.button_labels["next"])
        self.accept_button.setText(self.preferences.button_labels["accept"])
        self.reset_button.setText("重置视图")
        self.previous_button.setToolTip(f"上一组 ({self._shortcut_text('previous')})")
        self.next_button.setToolTip(f"下一组 ({self._shortcut_text('next')})")
        self.accept_button.setToolTip(f"接受当前组 ({self._shortcut_text('accept')})")
        self.reset_button.setToolTip(f"重置视图 ({self._shortcut_text('reset')})")

    def open_settings(self) -> None:
        if self._settings_page is not None:
            return
        dialog = SettingsDialog(
            self.preferences,
            self,
            calibration_options=self.calibration_options,
            current_calibration_id=self.current_calibration_id,
            project_available=self.dataset is not None,
        )
        dialog.set_embedded()
        self._settings_page = dialog
        self._settings_previous_context = (
            self.title_bar.context.text(),
            self.title_bar.context.isVisible(),
        )
        dialog.accepted.connect(lambda current=dialog: self._finish_settings(current, True))
        dialog.rejected.connect(lambda current=dialog: self._finish_settings(current, False))
        self.page_stack.addWidget(dialog)
        self.page_stack.setCurrentWidget(dialog)
        self.title_bar.set_context("")
        self.title_bar.settings_button.hide()
        self.app_status_bar.hide()
        for shortcut in self.shortcuts:
            shortcut.setEnabled(False)
        dialog.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def _finish_settings(self, dialog: SettingsDialog, save: bool) -> None:
        if dialog is not self._settings_page:
            return
        if save:
            self._apply_settings(dialog)
        self.page_stack.setCurrentWidget(self.main_page)
        self.page_stack.removeWidget(dialog)
        dialog.deleteLater()
        self._settings_page = None
        previous_context, was_visible = self._settings_previous_context
        self.title_bar.set_context(previous_context if was_visible else "")
        self.title_bar.settings_button.show()
        self.app_status_bar.show()
        for shortcut in self.shortcuts:
            shortcut.setEnabled(True)

    def _apply_settings(self, dialog: SettingsDialog) -> None:
        point_limit_changed = dialog.preferences.point_limit != self.preferences.point_limit
        calibration_changed = (
            self.dataset is not None
            and dialog.selected_calibration_id != self.current_calibration_id
        )
        self.preferences = dialog.preferences
        self.preferences.save(self.settings)
        self.calibration_options = list(dialog.calibration_options)
        self._save_calibration_options()
        if calibration_changed:
            self._apply_calibration_selection(
                dialog.selected_calibration_id,
                rebuild=False,
            )
        else:
            self._refresh_calibration_picker(self.current_calibration_id)
        self._apply_preferences()
        if (point_limit_changed or calibration_changed) and self.dataset is not None:
            self.focused_modality = None
            self._build_tile_pool()
            self._rebuild_tiles()
            self._show_current()
        self.status_text.setText("设置已保存")

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

    def configure_output(self) -> None:
        if self.dataset is None:
            return
        dialog = OutputSettingsDialog(self.dataset.root, self.dataset.output_root, self)
        if not dialog.exec():
            return
        selected = dialog.output_root
        if selected == self.dataset.output_root.resolve():
            return

        previous_custom_root = self.dataset.custom_output_root
        previous_store = self.review_store
        previous_accepted = set(self.accepted)
        self.dataset.custom_output_root = selected
        try:
            store = ReviewStore(self.dataset)
            accepted = store.reconcile()
        except Exception as exc:
            self.dataset.custom_output_root = previous_custom_root
            self.review_store = previous_store
            self.accepted = previous_accepted
            QMessageBox.critical(self, "无法使用输出文件夹", str(exc))
            return

        self.review_store = store
        self.accepted = accepted
        self._save_output_for(self.dataset.root, selected)
        self.output_label.setText(str(selected))
        self.output_label.setToolTip(str(selected))
        self._update_review_stats()
        self._show_current()
        self.status_text.setText("输出位置已更新；原输出文件不会移动")

    def open_annotation(self) -> None:
        if self.dataset is None or not self.dataset.samples:
            return
        sample = self.dataset.samples[self.current_index]
        if self.review_store is None:
            try:
                self.review_store = ReviewStore(self.dataset)
                self.accepted = self.review_store.reconcile()
            except Exception as exc:
                QMessageBox.critical(self, "无法保存审核记录", str(exc))
                return

        record = self.review_store.get(sample)
        dialog = AnnotationDialog(
            sample,
            list(record["defect_tags"]),
            str(record["note"]),
            self,
        )
        if not dialog.exec():
            return
        try:
            self.review_store.set_annotation(
                sample,
                dialog.selected_tags(),
                dialog.note(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "无法保存审核记录", str(exc))
            return
        self._update_annotation_button(sample)
        self.status_text.setText("缺陷与备注已保存")

    def _update_annotation_button(self, sample) -> None:
        record = self.review_store.get(sample) if self.review_store is not None else {}
        tags = [str(tag) for tag in record.get("defect_tags", [])]
        note = str(record.get("note", "")).strip()
        detail_count = len(tags) + bool(note)
        self.annotation_button.setText(
            f"缺陷与备注 · {detail_count}" if detail_count else "缺陷与备注"
        )
        details = "、".join(tags)
        if note:
            details = f"{details}\n{note}" if details else note
        self.annotation_button.setToolTip(details or "记录缺陷标签和文字备注")

    def open_output_folder(self) -> None:
        if self.dataset is None:
            return
        self.dataset.output_root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(self.dataset.output_root.as_uri())

    def closeEvent(self, event) -> None:
        for tile in self.tile_pool.values():
            tile.dispose()
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
    parser = argparse.ArgumentParser(description="双目图像与点云人工筛选器")
    parser.add_argument("project", nargs="?", type=Path, help="启动时打开的项目文件夹")
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
            for name, checkbox in window.checkboxes.items():
                checkbox.blockSignals(True)
                checkbox.setChecked(checkbox.isEnabled())
                checkbox.blockSignals(False)
            window._rebuild_tiles()
            window._show_current()
            configured = True
            return
        if any(tile._workers or tile._loading_delay.isActive() for tile in window.tiles.values()):
            return
        failed = []
        for modality, tile in window.tiles.items():
            expected = tile.cloud_canvas if modality == "ply" else tile.image_canvas
            if expected is None or tile.stack.currentWidget() is not expected:
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
    QThreadPool.globalInstance().setMaxThreadCount(3)
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
