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

import numpy as np
from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QRectF, QSettings, QThreadPool, Qt, QTimer
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
from .inspection import (
    CloudProjectionDialog,
    StereoOverlayDialog,
    rectify_stereo_images,
)
from .inspector import InspectorPanel
from .media import project_camera_points
from .mapping import MappingDialog
from .review import AnnotationDialog, MANIFEST_FILENAME, OutputSettingsDialog, ReviewStore
from .settings import AppPreferences, ChoiceButton, SettingsDialog
from .theme import style_for
from .widgets import (
    ActivityButton,
    CommandPalette,
    DropHint,
    MediaTile,
    NoViewsHint,
    PaneToggleButton,
    PlaybackControlButton,
    PointCloudCanvas,
    TitleBar,
    ToolIconButton,
)


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
        self._sidebar_expanded_width = 252
        self._sidebar_collapsed = True
        self._active_sidebar_page = "data"
        self._topbar_animation: QPropertyAnimation | None = None
        self._topbar_expanded_height = 44
        self._topbar_collapsed = False
        self._playback_fps = 2.0
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(500)
        self._playback_timer.timeout.connect(self._playback_tick)
        self._settings_page: SettingsDialog | None = None
        self._settings_previous_context: tuple[str, bool] = ("", False)
        self.product_mode = "view"
        self._layout_columns = 0
        self._chrome_hidden = False
        self._chrome_visibility: dict[str, bool] = {}
        self._inspector_source = ""
        self._cloud_clip_ranges: dict[str, tuple[float, float]] | None = None
        self._rectified_cache: dict[tuple[str, str, str], tuple[object, object]] = {}

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
        self.title_bar.settings_button.hide()
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
        self.status_text = QLabel("")
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
        self.splitter.setSizes([40, 1480, 0])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.inspector.hide()

        self.app_status_bar = QStatusBar()
        self.app_status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.app_status_bar)
        self.app_status_bar.hide()

    def _build_sidebar(self) -> QFrame:
        shell = QFrame()
        shell.setObjectName("navigationShell")
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        activity_bar = QFrame()
        activity_bar.setObjectName("activityBar")
        activity_layout = QVBoxLayout(activity_bar)
        activity_layout.setContentsMargins(2, 4, 2, 4)
        activity_layout.setSpacing(2)
        self.activity_buttons: dict[str, ActivityButton] = {}
        for name, tooltip in (
            ("data", "数据"),
            ("display", "显示"),
            ("analysis", "分析"),
            ("review", "筛选"),
        ):
            button = ActivityButton(name, tooltip)
            button.clicked.connect(
                lambda _checked=False, page=name: self._select_activity(page)
            )
            self.activity_buttons[name] = button
            activity_layout.addWidget(button)
        activity_layout.addStretch(1)
        shell_layout.addWidget(activity_bar)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(0)
        sidebar.setMaximumWidth(self._sidebar_expanded_width)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)
        self.sidebar_panel = sidebar

        project = QWidget()
        project.setObjectName("sidebarSection")
        self.project_panel = project
        project_layout = QVBoxLayout(project)
        project_layout.setContentsMargins(14, 14, 14, 12)
        project_layout.setSpacing(4)
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
        modes_layout.setContentsMargins(14, 10, 14, 10)
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
        quality_layout.setContentsMargins(14, 10, 14, 10)
        quality_layout.setSpacing(5)
        quality_title = QLabel("深度质量")
        quality_title.setObjectName("sectionLabel")
        self.depth_quality_label = QLabel("")
        self.depth_quality_label.setObjectName("qualitySummary")
        self.depth_quality_label.setWordWrap(True)
        quality_layout.addWidget(quality_title)
        quality_layout.addWidget(self.depth_quality_label)
        self.quality_panel.hide()
        side.addWidget(self.quality_panel)

        side.addStretch(1)

        self.output_panel = QWidget()
        self.output_panel.setObjectName("sidebarSection")
        output_layout = QVBoxLayout(self.output_panel)
        output_layout.setContentsMargins(14, 10, 14, 12)
        output_layout.setSpacing(5)
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
        sidebar.hide()
        shell_layout.addWidget(sidebar)
        return shell

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setObjectName("workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        self.workspace_header = QFrame()
        self.workspace_header.setObjectName("workspaceHeader")
        self.workspace_header.setFixedHeight(0)
        header_layout = QHBoxLayout(self.workspace_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        self.sidebar_button = PaneToggleButton("left")
        self.sidebar_button.setToolTip("收起侧栏 (Ctrl+B)")
        self.sidebar_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_button.setParent(self.workspace_header)
        self.sidebar_button.hide()
        self.topbar_button = PaneToggleButton("up")
        self.topbar_button.setToolTip("收起检查工具栏")
        self.topbar_button.setEnabled(False)
        self.topbar_button.clicked.connect(self.toggle_topbar)
        self.topbar_button.setParent(self.workspace_header)
        self.topbar_button.hide()
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
        self.header_breadcrumb.setParent(self.workspace_header)
        self.view_hint.setParent(self.workspace_header)
        self.header_breadcrumb.hide()
        self.reset_button = QPushButton("重置视图")
        self.reset_button.setObjectName("ghostButton")
        self.reset_button.clicked.connect(self.reset_views)
        self.reset_button.setParent(self.workspace_header)
        self.reset_button.hide()
        self.open_button = QPushButton("打开项目…")
        self.open_button.setObjectName("primaryButton")
        self.open_button.clicked.connect(self.choose_project)
        self.mode_picker = ChoiceButton()
        self.mode_picker.setObjectName("modePicker")
        self.mode_picker.setMinimumWidth(96)
        self.mode_picker.addItem("查看模式", "view")
        self.mode_picker.addItem("筛选模式", "review")
        self.mode_picker.currentIndexChanged.connect(self._mode_picker_changed)
        self.layout_picker = ChoiceButton()
        self.layout_picker.setObjectName("layoutPicker")
        self.layout_picker.setMinimumWidth(82)
        for label, columns in (("自动布局", 0), ("单列", 1), ("双列", 2), ("三列", 3)):
            self.layout_picker.addItem(label, columns)
        self.layout_picker.currentIndexChanged.connect(self._layout_picker_changed)
        self.header_settings_button = QPushButton("设置")
        self.header_settings_button.setObjectName("titleActionButton")
        self.header_settings_button.clicked.connect(self.open_settings)
        self.title_bar.action_layout.addWidget(self.mode_picker)
        self.title_bar.action_layout.addWidget(self.open_button)
        self.title_bar.action_layout.addWidget(self.layout_picker)
        self.title_bar.action_layout.addWidget(self.header_settings_button)
        self.workspace_header.hide()
        workspace_layout.addWidget(self.workspace_header)

        self.inspection_bar = QFrame()
        self.inspection_bar.setObjectName("inspectionBar")
        self.inspection_bar.setFixedHeight(38)
        inspection_layout = QHBoxLayout(self.inspection_bar)
        inspection_layout.setContentsMargins(8, 4, 10, 4)
        inspection_layout.setSpacing(3)
        self.crosshair_button = ToolIconButton("crosshair", "十字光标", checkable=True)
        self.crosshair_button.toggled.connect(self._toggle_crosshair)
        self.sync_views_button = ToolIconButton("sync", "左右图同步缩放和平移", checkable=True)
        self.sync_views_button.toggled.connect(self._sync_views_toggled)
        self.epiline_button = ToolIconButton("epiline", "极线与同名像素联动", checkable=True)
        self.epiline_button.toggled.connect(self._epilines_toggled)
        self.rectify_button = ToolIconButton(
            "rectify",
            "校正前 / 校正后",
            checkable=True,
        )
        self.rectify_button.toggled.connect(self._rectification_toggled)
        self.overlay_button = ToolIconButton("overlay", "左右图透明叠加")
        self.overlay_button.clicked.connect(
            lambda: self.open_stereo_overlay(mode="overlay")
        )
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
        inspection_layout.addWidget(self.rectify_button)
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
        self.media_grid.setContentsMargins(8, 8, 8, 8)
        self.media_grid.setSpacing(8)
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
        review_layout.setContentsMargins(8, 0, 8, 6)
        review_layout.setSpacing(4)
        self.timeline_panel = QFrame()
        self.timeline_panel.setObjectName("timelinePanel")
        timeline_row = QHBoxLayout(self.timeline_panel)
        timeline_row.setContentsMargins(2, 5, 2, 5)
        timeline_row.setSpacing(4)
        self.playback_previous_button = PlaybackControlButton("previous")
        self.playback_previous_button.setToolTip("上一帧")
        self.playback_previous_button.clicked.connect(self.previous_sample)
        self.playback_button = PlaybackControlButton("play")
        self.playback_button.setToolTip("播放")
        self.playback_button.clicked.connect(self.toggle_playback)
        self.playback_next_button = PlaybackControlButton("next")
        self.playback_next_button.setToolTip("下一帧")
        self.playback_next_button.clicked.connect(self.next_sample)
        self.timeline_left = QLabel("0")
        self.timeline_left.setObjectName("sampleMeta")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setObjectName("timelineSlider")
        self.timeline.setRange(0, 0)
        self.timeline.setEnabled(False)
        self.timeline.setTracking(False)
        self.timeline.valueChanged.connect(self._timeline_changed)
        self.timeline.sliderMoved.connect(
            lambda value: self.timeline_left.setText(str(value + 1))
        )
        self.timeline_right = QLabel("0")
        self.timeline_right.setObjectName("sampleMeta")
        self.playback_speed = ChoiceButton()
        self.playback_speed.setObjectName("playbackSpeed")
        self.playback_speed.setMinimumWidth(82)
        for label, fps in (
            ("0.5 fps", 0.5),
            ("1 fps", 1.0),
            ("2 fps", 2.0),
            ("5 fps", 5.0),
            ("10 fps", 10.0),
        ):
            self.playback_speed.addItem(label, fps)
        self.playback_speed.setCurrentIndex(self.playback_speed.findData(2.0))
        self.playback_speed.currentIndexChanged.connect(self._playback_speed_changed)
        timeline_row.addWidget(self.playback_previous_button)
        timeline_row.addWidget(self.playback_button)
        timeline_row.addWidget(self.playback_next_button)
        timeline_row.addSpacing(4)
        timeline_row.addWidget(self.timeline_left)
        timeline_row.addWidget(self.timeline, 1)
        timeline_row.addWidget(self.timeline_right)
        timeline_row.addSpacing(4)
        timeline_row.addWidget(self.playback_speed)
        self.status_text.setMinimumWidth(90)
        self.status_text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        timeline_row.addWidget(self.status_text)
        review_layout.addWidget(self.timeline_panel)

        self.review_controls = QFrame()
        self.review_controls.setObjectName("reviewControls")
        controls = QHBoxLayout(self.review_controls)
        controls.setContentsMargins(2, 2, 2, 0)
        controls.setSpacing(6)
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
        self.pending_button = QPushButton("待定")
        self.pending_button.setObjectName("secondaryButton")
        self.pending_button.clicked.connect(lambda: self.set_review_status("pending"))
        self.reject_button = QPushButton("拒绝")
        self.reject_button.setObjectName("rejectButton")
        self.reject_button.clicked.connect(lambda: self.set_review_status("rejected"))
        controls.addWidget(self.previous_button)
        controls.addWidget(self.next_button)
        controls.addSpacing(10)
        controls.addLayout(sample_box, 1)
        controls.addWidget(self.annotation_button)
        controls.addWidget(self.pending_button)
        controls.addWidget(self.reject_button)
        controls.addWidget(self.accept_button)
        review_layout.addWidget(self.review_controls)
        self.review_bar.hide()
        workspace_layout.addWidget(self.review_bar)
        return workspace

    def _build_inspector(self) -> InspectorPanel:
        inspector = InspectorPanel()
        inspector.close_requested.connect(self.close_inspector)
        inspector.source_changed.connect(self._inspector_source_changed)
        inspector.display_changed.connect(self._apply_display_settings)
        inspector.fit_requested.connect(self._fit_requested)
        inspector.tool_requested.connect(self._tool_requested)
        inspector.export_requested.connect(self.export_current_view)
        inspector.stereo_action_requested.connect(self._stereo_action)
        inspector.cloud_color_changed.connect(self._cloud_color_changed)
        inspector.cloud_size_changed.connect(self._cloud_size_changed)
        inspector.cloud_guides_changed.connect(self._cloud_guides_changed)
        inspector.cloud_view_requested.connect(self._cloud_view_requested)
        inspector.cloud_tool_requested.connect(self._cloud_tool_requested)
        inspector.cloud_background_changed.connect(self._cloud_background_changed)
        inspector.cloud_clip_changed.connect(self._cloud_clip_changed)
        return inspector

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
            ("Shift+N", lambda: self.next_with_review_status("rejected")),
            ("Ctrl+Shift+N", self.next_with_defect),
            ("Ctrl+,", self.open_settings),
            ("Tab", self.toggle_chrome),
            ("F11", self.toggle_fullscreen),
            ("Ctrl+Shift+P", self.open_command_palette),
            ("Ctrl+1", lambda: self.set_product_mode("view")),
            ("Ctrl+2", lambda: self.set_product_mode("review")),
            ("Space", self.toggle_playback),
            ("X", lambda: self.set_review_status("rejected")),
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
        self._stop_playback()
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
        self.product_mode = self._saved_product_mode(dataset.root)
        self.current_calibration_id, self.calibration = self._saved_calibration_selection(
            dataset.root
        )
        self.manual_dirs = dict(manual_dirs or {})
        self.force_order = force_order
        self.current_index = 0
        self.focused_modality = None
        self._cloud_clip_ranges = None
        self._rectified_cache.clear()
        manifest_exists = (dataset.output_root / MANIFEST_FILENAME).is_file()
        if self.product_mode == "review" or manifest_exists:
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
        else:
            self.review_store = None
            self.accepted = set()
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
        self.modes_panel.hide()
        self.quality_panel.hide()
        self.output_panel.hide()
        self.reset_button.show()
        self.review_bar.show()
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
        self.inspector.set_sources(list(dataset.available_modalities), preferred[0] if preferred else "")
        self._inspector_source = preferred[0] if preferred else ""
        self.content_stack.setCurrentWidget(self.media_container)
        self._build_tile_pool()
        self._rebuild_tiles()
        self.set_product_mode(self.product_mode, persist=False)
        self._refresh_activity_page()
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

    def _saved_product_mode(self, root: Path) -> str:
        prefix = self._matching_settings_prefix(root)
        mode = str(self.settings.value(f"{prefix}/product_mode", "view"))
        return mode if mode in {"view", "review"} else "view"

    def _save_product_mode(self, root: Path, mode: str) -> None:
        prefix = self._matching_settings_prefix(root)
        self.settings.setValue(f"{prefix}/product_mode", mode)

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

    def _mode_picker_changed(self, _index: int) -> None:
        self.set_product_mode(str(self.mode_picker.currentData() or "view"))

    def set_product_mode(self, mode: str, *, persist: bool = True) -> None:
        mode = mode if mode in {"view", "review"} else "view"
        entering_review = mode == "review" and self.product_mode != "review"
        self.product_mode = mode
        picker_index = self.mode_picker.findData(mode)
        if picker_index >= 0 and self.mode_picker.currentIndex() != picker_index:
            self.mode_picker.blockSignals(True)
            self.mode_picker.setCurrentIndex(picker_index)
            self.mode_picker.blockSignals(False)
        review_mode = mode == "review"
        self.review_controls.setVisible(review_mode and self.dataset is not None)
        self.activity_buttons["review"].setVisible(review_mode)
        if not review_mode and self._active_sidebar_page == "review":
            self._active_sidebar_page = "data"
        accept_shortcut = self.action_shortcuts.get("accept")
        if accept_shortcut is not None:
            accept_shortcut.setEnabled(review_mode)
        if entering_review and self.dataset is not None and self.review_store is None:
            try:
                self.review_store = ReviewStore(self.dataset)
                self.accepted = self.review_store.reconcile()
            except Exception as exc:
                QMessageBox.warning(self, "审核文件不可用", str(exc))
        if persist and self.dataset is not None:
            self._save_product_mode(self.dataset.root, mode)
        self._refresh_activity_page()
        if self.dataset is not None:
            self._show_current()
        self.status_text.setText("筛选模式" if review_mode else "查看模式")

    def _layout_picker_changed(self, _index: int) -> None:
        try:
            self._layout_columns = int(self.layout_picker.currentData() or 0)
        except (TypeError, ValueError):
            self._layout_columns = 0
        self.focused_modality = None
        self._apply_tile_layout()

    def _select_activity(self, page: str) -> None:
        if page == "review" and self.product_mode != "review":
            return
        if page == "analysis":
            if not self.inspector.isVisible():
                self.open_inspector()
            else:
                self.close_inspector()
            return
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
        self.output_panel.setVisible(
            visible
            and page == "review"
            and self.dataset is not None
            and self.product_mode == "review"
        )
        for name, button in self.activity_buttons.items():
            button.setChecked(
                (name == page and visible)
                or (name == "analysis" and self.inspector.isVisible())
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
        left_calibration = (
            self.calibration.left
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
            tile.selection_changed.connect(self._media_selection_changed)
            tile.pixel_clicked.connect(self._image_pixel_clicked)
            if tile.cloud_canvas is not None:
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
        for modality in ("left", "right"):
            tile = self.tile_pool.get(modality)
            if tile is not None and tile.image_data is not None:
                tile.set_display_settings()
        self.status_text.setText("已显示校正前图像")

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
            or left._current_path is None
            or right._current_path is None
        ):
            return
        key = (
            str(left._current_path),
            str(right._current_path),
            self.current_calibration_id,
        )
        images = self._rectified_cache.get(key)
        if images is None:
            try:
                images = rectify_stereo_images(
                    left.image_data.image,
                    right.image_data.image,
                    self.calibration,
                )
            except Exception as exc:
                self.rectify_button.blockSignals(True)
                self.rectify_button.setChecked(False)
                self.rectify_button.blockSignals(False)
                self.status_text.setText(f"无法生成校正视图：{exc}")
                return
            self._rectified_cache[key] = images
            if len(self._rectified_cache) > 6:
                self._rectified_cache.pop(next(iter(self._rectified_cache)))
        left.image_canvas.set_image(images[0], preserve_view=True)
        right.image_canvas.set_image(images[1], preserve_view=True)
        self.status_text.setText("已显示校正后图像")

    def _media_cursor_moved(self, modality: str, x: float, y: float) -> None:
        if not self.crosshair_button.isChecked():
            return
        self._crosshair_position = (x, y)
        for tile in self.tiles.values():
            if tile.image_data is not None:
                tile.image_canvas.set_crosshair(x, y, True)
                tile.image_canvas.set_epiline(None)
        self.cursor_info.setText(self._cursor_value_text(modality, x, y))
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

    def _cursor_value_text(self, source: str, x: float, y: float) -> str:
        parts: list[str] = []
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
            parts.append(f"{modality_label(rgb_tile.modality)} ({px}, {py})")
            parts.append(f"RGB {color.red()}, {color.green()}, {color.blue()}")
        stereo_match = self._stereo_match_text(source, x, y)
        if stereo_match:
            parts.append(stereo_match)

        depth_value: float | None = None
        depth = self.tile_pool.get("depth_fsd")
        if (
            depth is not None
            and depth.image_data is not None
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
            and self.calibration is not None
            and self.calibration.left is not None
        ):
            left_calibration = self.calibration.left
            width = (
                left.image_data.image.width()
                if left is not None and left.image_data is not None
                else left_calibration.width
                if left_calibration.width is not None
                else depth.image_data.image.width()
            )
            height = (
                left.image_data.image.height()
                if left is not None and left.image_data is not None
                else left_calibration.height
                if left_calibration.height is not None
                else depth.image_data.image.height()
            )
            px = x * width
            py = y * height
            point = self.calibration.left.point_from_depth(px, py, depth_value)
            parts.append(f"XYZ {point[0]:.3g}, {point[1]:.3g}, {point[2]:.3g}")
        elif depth_value is not None and depth_value > 0:
            parts.append("XYZ 需左目内参")
        return "   ·   ".join(parts) or "无像素数据"

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

        def gray(values: np.ndarray) -> np.ndarray:
            array = np.asarray(values)
            if array.ndim == 3:
                array = array[..., :3].astype(np.float32, copy=False).mean(axis=2)
            return array.astype(np.float32, copy=False)

        source_tile = left if source == "left" else right
        target_tile = right if source == "left" else left
        source_gray = gray(source_tile.image_data.values)
        target_gray = gray(target_tile.image_data.values)
        source_height, source_width = source_gray.shape[:2]
        target_height, target_width = target_gray.shape[:2]
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
        source_patch = source_gray[
            sy - radius : sy + radius + 1,
            sx - radius : sx + radius + 1,
        ]
        target_strip = target_gray[
            ty - radius : ty + radius + 1,
            :,
        ]
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
        return f"{label} ({target_x}, {ty}) · 视差 {disparity:.2f}px"

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
        if modality in {"left", "right"} and self.rectify_button.isChecked():
            QTimer.singleShot(0, self._apply_rectified_views)
        QTimer.singleShot(0, lambda name=modality: self._prefetch_adjacent_media(name))
        if modality == self._inspector_source:
            self._update_inspector()
        self._update_inspection_controls()

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
        value_range = (
            f"{stats.minimum:g} – {stats.maximum:g}"
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
        if self.inspector.isVisible():
            return
        self.inspector.show()
        width = max(290, min(340, self.inspector.sizeHint().width()))
        total = max(1, self.splitter.width())
        left = self.sidebar.width()
        self.splitter.setSizes([left, max(1, total - left - width), width])
        self._refresh_activity_page()
        self._update_inspector()

    def close_inspector(self) -> None:
        self.inspector.hide()
        total = max(1, self.splitter.width())
        self.splitter.setSizes([self.sidebar.width(), total - self.sidebar.width(), 0])
        self._refresh_activity_page()

    def _inspector_source_changed(self, modality: str) -> None:
        self._inspector_source = modality
        self._update_inspector()

    def _update_inspector(self) -> None:
        if not self.inspector.isVisible() or self.dataset is None:
            return
        source = self._inspector_source
        if source not in self.tile_pool:
            source = next(iter(self.tiles), next(iter(self.tile_pool), ""))
            self._inspector_source = source
        tile = self.tile_pool.get(source)
        if tile is None:
            return
        path = tile._current_path
        if source == "ply":
            cloud = tile.cloud_canvas._cloud if tile.cloud_canvas is not None else None
            self.inspector.set_cloud_data(cloud, path)
        else:
            self.inspector.set_image_data(source, tile.image_data, path)

    def _apply_display_settings(self, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        tile = self.tile_pool.get(self._inspector_source)
        if tile is None or tile.image_data is None:
            return
        tile.set_display_settings(**settings)

    def _active_image_tile(self) -> MediaTile | None:
        tile = self.tile_pool.get(self._inspector_source)
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
        if mode == "actual":
            tile.image_canvas.show_actual_size()
        elif mode == "width":
            tile.image_canvas.fit_width()
        else:
            tile.image_canvas.reset_view()

    def _tool_requested(self, tool: str) -> None:
        tile = self._active_image_tile()
        if tile is None:
            return
        for candidate in self.tile_pool.values():
            candidate.image_canvas.set_tool("pan")
        tile.image_canvas.set_tool(tool)
        self.status_text.setText(
            {"roi": "拖动选择矩形 ROI", "line": "拖动绘制测量线"}.get(tool, "")
        )

    def _media_selection_changed(self, modality: str, tool: str, selection: object) -> None:
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
        depth_tile = self.tile_pool.get("depth_fsd")
        if (
            self.calibration is not None
            and self.calibration.left is not None
            and depth_tile is not None
            and depth_tile.image_data is not None
        ):
            depth_values = depth_tile.image_data.values
            depth_height, depth_width = depth_values.shape[:2]
            mx = max(0, min(depth_width - 1, int((x1 + x2) * 0.5 * depth_width)))
            my = max(0, min(depth_height - 1, int((y1 + y2) * 0.5 * depth_height)))
            raw = depth_values[my, mx]
            if getattr(raw, "ndim", 0):
                raw = raw.flat[0]
            try:
                depth = float(raw)
            except (TypeError, ValueError):
                depth = 0.0
            if math.isfinite(depth) and 0 < depth < 65535:
                camera = self.calibration.left
                first = camera.point_from_depth(x1 * width, y1 * height, depth)
                second = camera.point_from_depth(x2 * width, y2 * height, depth)
                physical = math.dist(first, second)
                if depth > 100:
                    physical /= 1000.0
        self.inspector.set_line_measurement(pixel_length, physical)

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
        cloud = cloud_tile.cloud_canvas._cloud
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
            QMessageBox.warning(self, "导出失败", f"无法写入：\n{selected}")
            return
        self.status_text.setText(f"已导出 {Path(selected).name}")

    def _stereo_action(self, action: str) -> None:
        if action == "cloud_projection":
            self.open_cloud_projection()
        else:
            self.open_stereo_overlay(mode=action)

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
        tile = self.tile_pool.get("ply")
        if tile is not None and tile.cloud_canvas is not None:
            if tool == "reset":
                if self._cloud_clip_ranges is not None:
                    self._apply_cloud_clip()
                else:
                    count = tile.cloud_canvas.restore_full_cloud()
                    self.status_text.setText(f"已恢复 {count:,} 个点")
                return
            tile.cloud_canvas.set_interaction_mode(tool)
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
            cloud_tile.cloud_canvas._cloud
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

        self.sample_title.setText(sample.display_name)
        self.sample_meta.setText(
            f"第 {self.current_index + 1} / {len(self.dataset.samples)} 组  ·  {len(sample.files)} 个文件"
        )
        record = self.review_store.get(sample) if self.review_store is not None else {}
        review_state = (
            "accepted"
            if sample.key in self.accepted
            else str(record.get("status", "pending"))
        )
        if review_state not in {"accepted", "rejected", "pending"}:
            review_state = "pending"
        is_accepted = review_state == "accepted"
        self.sample_status.setText(
            {"accepted": "已接受", "rejected": "已拒绝", "pending": "待定"}[review_state]
        )
        visual_state = review_state if self.product_mode == "review" else "view"
        if self.review_bar.property("reviewState") != visual_state:
            self.review_bar.setProperty("reviewState", visual_state)
            for widget in (
                self.review_bar,
                self.sample_title,
                self.accept_button,
                self.reject_button,
                self.sample_status,
            ):
                widget.style().unpolish(widget)
                widget.style().polish(widget)
        self.sample_status.setProperty("reviewStatus", review_state)
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

    def _playback_speed_changed(self, index: int) -> None:
        try:
            fps = float(self.playback_speed.itemData(index))
        except (TypeError, ValueError):
            fps = 2.0
        self._playback_fps = max(0.5, min(10.0, fps))
        self._playback_timer.setInterval(max(40, round(1000 / self._playback_fps)))

    def toggle_playback(self) -> None:
        if self._playback_timer.isActive():
            self._stop_playback()
            return
        if self.dataset is None or not self.dataset.samples:
            return
        if self.current_index >= len(self.dataset.samples) - 1:
            self.current_index = 0
            self._show_current()
        for tile in self.tile_pool.values():
            tile.set_analysis_enabled(False)
        self._playback_timer.start()
        self.playback_button.set_playing(True)
        self.playback_button.setToolTip("暂停")
        self.status_text.setText(f"正在播放 · {self._playback_fps:g} fps")

    def _playback_tick(self) -> None:
        if self.dataset is None or not self.dataset.samples:
            self._stop_playback()
            return
        # Do not skip a frame while its visible media is still loading. This
        # keeps point-cloud playback ordered even at a requested high speed.
        if any(tile._active_worker_token is not None for tile in self.tiles.values()):
            return
        if self.current_index >= len(self.dataset.samples) - 1:
            self._stop_playback()
            return
        self.current_index += 1
        self._show_current()

    def _stop_playback(self) -> None:
        was_active = self._playback_timer.isActive()
        self._playback_timer.stop()
        if hasattr(self, "playback_button"):
            self.playback_button.set_playing(False)
            self.playback_button.setToolTip("播放")
        for tile in self.tile_pool.values():
            tile.set_analysis_enabled(True)
        depth_tile = self.tile_pool.get("depth_fsd")
        if was_active:
            for tile in self.tiles.values():
                if tile.image_data is not None:
                    tile.refresh_analysis()
        if was_active and depth_tile is not None and "depth_fsd" not in self.tiles:
            current = (
                self.dataset.samples[self.current_index].files.get("depth_fsd")
                if self.dataset is not None and self.dataset.samples
                else None
            )
            if depth_tile._current_path != current:
                depth_tile.show_file(current)
            else:
                depth_tile.refresh_analysis()
        if was_active and hasattr(self, "status_text"):
            self.status_text.setText("")

    def _update_controls(self) -> None:
        has_samples = self.dataset is not None and bool(self.dataset.samples)
        count = len(self.dataset.samples) if self.dataset else 0
        self.previous_button.setEnabled(bool(has_samples and self.current_index > 0))
        self.next_button.setEnabled(bool(has_samples and self.current_index < count - 1))
        review_enabled = bool(has_samples and self.product_mode == "review")
        self.accept_button.setEnabled(review_enabled)
        self.reject_button.setEnabled(review_enabled)
        self.pending_button.setEnabled(review_enabled)
        self.reset_button.setEnabled(bool(has_samples))
        self.timeline.setEnabled(bool(has_samples))
        self.playback_previous_button.setEnabled(bool(has_samples and self.current_index > 0))
        self.playback_next_button.setEnabled(
            bool(has_samples and self.current_index < count - 1)
        )
        self.playback_button.setEnabled(bool(has_samples and count > 1))
        self.playback_speed.setEnabled(bool(has_samples and count > 1))
        self.open_output_button.setEnabled(self.dataset is not None)
        self.configure_output_button.setEnabled(self.dataset is not None)
        self.annotation_button.setEnabled(review_enabled)
        self._update_inspection_controls()

    def _update_review_stats(self) -> None:
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
        if self.product_mode != "review" or self.dataset is None or not self.dataset.samples:
            return
        self.next_with_review_status("pending")

    def next_with_review_status(self, status: str) -> None:
        if self.product_mode != "review" or self.dataset is None or not self.dataset.samples:
            return
        indices = list(range(self.current_index + 1, len(self.dataset.samples))) + list(range(0, self.current_index + 1))
        for index in indices:
            sample = self.dataset.samples[index]
            sample_status = (
                "accepted"
                if sample.key in self.accepted
                else str(
                    self.review_store.get(sample).get("status", "pending")
                    if self.review_store is not None
                    else "pending"
                )
            )
            if sample_status == status:
                self.current_index = index
                self._show_current()
                return
        self.status_text.setText(f"没有更多{status}样本")

    def next_with_defect(self) -> None:
        if (
            self.product_mode != "review"
            or self.dataset is None
            or not self.dataset.samples
            or self.review_store is None
        ):
            return
        indices = list(range(self.current_index + 1, len(self.dataset.samples))) + list(
            range(0, self.current_index + 1)
        )
        for index in indices:
            record = self.review_store.get(self.dataset.samples[index])
            if record.get("defect_tags"):
                self.current_index = index
                self._show_current()
                return
        self.status_text.setText("没有带缺陷标签的样本")

    def accept_current(self) -> None:
        if self.product_mode != "review":
            return
        self.set_review_status("accepted")

    def set_review_status(self, status: str) -> None:
        if (
            self.product_mode != "review"
            or status not in {"accepted", "rejected", "pending"}
            or self.dataset is None
            or not self.dataset.samples
        ):
            return
        sample = self.dataset.samples[self.current_index]
        copied_count = 0
        if status == "accepted" and sample.key not in self.accepted:
            try:
                copied_count = len(copy_sample(self.dataset, sample))
            except Exception as exc:
                QMessageBox.critical(self, "复制失败", f"无法接受当前样本：\n{exc}")
                return
            self.accepted.add(sample.key)
        elif status != "accepted":
            self.accepted.discard(sample.key)
        if self.review_store is None:
            try:
                self.review_store = ReviewStore(self.dataset)
            except Exception as exc:
                QMessageBox.warning(self, "审核记录不可用", str(exc))
                return
        try:
            self.review_store.set_status(sample, status)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "审核记录未保存",
                f"无法更新 JSON 审核文件：\n{exc}",
            )
        self._update_review_stats()
        if (
            self.preferences.auto_advance
            and status in {"accepted", "rejected"}
            and self.current_index < len(self.dataset.samples) - 1
        ):
            self.current_index += 1
        self._show_current()
        if copied_count:
            self.status_text.setText(f"已接受 {sample.display_name}")
        else:
            self.status_text.setText(
                {"accepted": "已接受", "rejected": "已拒绝", "pending": "已设为待定"}[status]
            )

    def reset_views(self) -> None:
        for tile in self.tiles.values():
            tile.reset_view()
        self.status_text.setText("视图已重置")

    def toggle_sidebar(self) -> None:
        expanding = self._sidebar_collapsed
        self._sidebar_collapsed = not expanding
        if self._sidebar_animation is not None:
            self._sidebar_animation.stop()
        panel = self.sidebar_panel
        current_width = panel.width() if panel.isVisible() else 0
        if expanding:
            panel.show()
            panel.setMinimumWidth(0)
            panel.setMaximumWidth(max(0, current_width))
            start, end = max(0, current_width), self._sidebar_expanded_width
        else:
            if current_width >= 220:
                self._sidebar_expanded_width = min(360, current_width)
            panel.setMinimumWidth(0)
            panel.setMaximumWidth(max(0, current_width))
            start, end = max(0, current_width), 0

        animation = QPropertyAnimation(panel, b"maximumWidth", self)
        animation.setDuration(190)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def finish() -> None:
            if expanding:
                panel.setMinimumWidth(220)
                panel.setMaximumWidth(360)
            else:
                panel.hide()
                panel.setMaximumWidth(360)
            self._refresh_activity_page()

        animation.finished.connect(finish)
        self._sidebar_animation = animation
        animation.start()
        self._refresh_activity_page()

    def toggle_topbar(self) -> None:
        if self.dataset is None:
            return
        expanding = self._topbar_collapsed
        self._topbar_collapsed = not expanding
        if self._topbar_animation is not None:
            self._topbar_animation.stop()
        current_height = self.inspection_bar.height() if self.inspection_bar.isVisible() else 0
        if expanding:
            self.inspection_bar.show()
            self.inspection_bar.setMinimumHeight(0)
            self.inspection_bar.setMaximumHeight(max(0, current_height))
            start = max(0, current_height)
            end = max(self._topbar_expanded_height, self.inspection_bar.sizeHint().height())
        else:
            if current_height > 0:
                self._topbar_expanded_height = current_height
            self.inspection_bar.setMinimumHeight(0)
            self.inspection_bar.setMaximumHeight(max(0, current_height))
            start, end = max(0, current_height), 0

        animation = QPropertyAnimation(self.inspection_bar, b"maximumHeight", self)
        animation.setDuration(190)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def finish() -> None:
            if expanding:
                self.inspection_bar.setMinimumHeight(0)
                self.inspection_bar.setMaximumHeight(16_777_215)
            else:
                self.inspection_bar.hide()
                self.inspection_bar.setMaximumHeight(16_777_215)

        animation.finished.connect(finish)
        self._topbar_animation = animation
        animation.start()
        self.topbar_button.set_direction("up" if expanding else "down")
        self.topbar_button.setToolTip(
            ("收起" if expanding else "展开") + "检查工具栏"
        )

    def toggle_chrome(self) -> None:
        self._chrome_hidden = not self._chrome_hidden
        widgets = {
            "title": self.title_bar,
            "workspace_header": self.workspace_header,
            "inspection": self.inspection_bar,
            "navigation": self.sidebar,
            "inspector": self.inspector,
            "bottom": self.review_bar,
        }
        if self._chrome_hidden:
            self._chrome_visibility = {
                name: widget.isVisible() for name, widget in widgets.items()
            }
            for widget in widgets.values():
                widget.hide()
            self.media_grid.setContentsMargins(0, 0, 0, 0)
            self.media_grid.setSpacing(1)
            self.status_text.setText("")
        else:
            for name, widget in widgets.items():
                widget.setVisible(self._chrome_visibility.get(name, False))
            self.media_grid.setContentsMargins(8, 8, 8, 8)
            self.media_grid.setSpacing(8)
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
            ("切换到查看模式", "Ctrl+1", lambda: self.set_product_mode("view")),
            ("切换到筛选模式", "Ctrl+2", lambda: self.set_product_mode("review")),
            ("播放 / 暂停", "Space", self.toggle_playback),
            ("上一帧", self._shortcut_text("previous"), self.previous_sample),
            ("下一帧", self._shortcut_text("next"), self.next_sample),
            ("接受当前样本", self._shortcut_text("accept"), self.accept_current),
            ("拒绝当前样本", "X", lambda: self.set_review_status("rejected")),
            ("下一个未检查样本", "N", self.next_unreviewed),
            ("下一个拒绝样本", "Shift+N", lambda: self.next_with_review_status("rejected")),
            ("下一个带缺陷样本", "Ctrl+Shift+N", self.next_with_defect),
            ("显示 / 隐藏界面", "Tab", self.toggle_chrome),
            ("全屏", "F11", self.toggle_fullscreen),
            ("重置视图", self._shortcut_text("reset"), self.reset_views),
            ("打开检查器", "", self.open_inspector),
            ("左右图透明叠加", "", lambda: self.open_stereo_overlay(mode="overlay")),
            ("左右图绝对差值", "", lambda: self.open_stereo_overlay(mode="difference")),
            (
                "切换校正前 / 校正后",
                "",
                lambda: self.rectify_button.setChecked(
                    not self.rectify_button.isChecked()
                ),
            ),
            ("导出当前视图", "", self.export_current_view),
            ("设置", "Ctrl+,", self.open_settings),
        ]
        CommandPalette(actions, self).exec()

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
        self._stop_playback()
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
        self.title_bar.set_context("设置")
        self.title_bar.actions.hide()
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
        self.title_bar.actions.show()
        self.title_bar.settings_button.hide()
        self.app_status_bar.hide()
        for shortcut in self.shortcuts:
            shortcut.setEnabled(True)

    def _apply_settings(self, dialog: SettingsDialog) -> None:
        point_limit_changed = dialog.preferences.point_limit != self.preferences.point_limit
        cloud_render_changed = (
            dialog.preferences.cloud_cam_offset,
            dialog.preferences.cloud_grid,
            dialog.preferences.cloud_dot_radius,
            dialog.preferences.cloud_z_max,
            dialog.preferences.cloud_tau_rel,
            dialog.preferences.cloud_occlusion,
        ) != (
            self.preferences.cloud_cam_offset,
            self.preferences.cloud_grid,
            self.preferences.cloud_dot_radius,
            self.preferences.cloud_z_max,
            self.preferences.cloud_tau_rel,
            self.preferences.cloud_occlusion,
        )
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
        if (
            point_limit_changed
            or cloud_render_changed
            or calibration_changed
        ) and self.dataset is not None:
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
        if self.product_mode != "review" or self.dataset is None:
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
        if (
            self.product_mode != "review"
            or self.dataset is None
            or not self.dataset.samples
        ):
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
