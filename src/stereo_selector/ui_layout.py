"""Main-window UI construction, split out of :mod:`stereo_selector.app`.

These functions populate attributes on the window instance exactly like the
former ``MainWindow._build_*`` methods, so callers can keep using the same
attributes while ``app.py`` focuses on state and behaviour.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .calibration import RECTIFY_MODE_CHOICES
from .inspector import InspectorPanel
from .models import MODALITY_INFO, modality_label
from .ui_controls import ChoiceButton, CursorInfoWidget
from .ui_metrics import (
    ACTIVITY_BAR_WIDTH,
    CURSOR_INFO_ROW_HEIGHT,
    CURSOR_INFO_WIDTH,
    ICON_PANE_SIZE,
    INSPECTION_BAR_HEIGHT,
    MEDIA_MARGIN,
    MEDIA_SPACING,
)
from .widgets import (
    ActivityButton,
    DropHint,
    NoViewsHint,
    PaneToggleButton,
    PlaybackControlButton,
    PointCloudCanvas,
    TimelineSlider,
    ToolIconButton,
)


def build_sidebar(window) -> QFrame:
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
    window.activity_buttons: dict[str, ActivityButton] = {}
    for name, tooltip in (
        ("data", "数据"),
        ("display", "视图"),
        ("adjust", "显示调整"),
        ("measure", "测量与选择"),
        ("statistics", "统计信息"),
    ):
        button = ActivityButton(name, tooltip)
        button.clicked.connect(
            lambda _checked=False, page=name: window._select_activity(page)
        )
        window.activity_buttons[name] = button
        activity_layout.addWidget(button)
    activity_layout.addStretch(1)
    shell_layout.addWidget(activity_bar)

    sidebar = QFrame()
    sidebar.setObjectName("sidebar")
    sidebar.setMinimumWidth(0)
    sidebar.setMaximumWidth(
        max(0, window._sidebar_expanded_width - ACTIVITY_BAR_WIDTH)
    )
    side = QVBoxLayout(sidebar)
    side.setContentsMargins(0, 0, 0, 0)
    side.setSpacing(0)
    window.sidebar_panel = sidebar

    project = QWidget()
    project.setObjectName("sidebarSection")
    window.project_panel = project
    project_layout = QVBoxLayout(project)
    project_layout.setContentsMargins(14, 14, 14, 12)
    project_layout.setSpacing(4)
    eyebrow = QLabel("PROJECT")
    eyebrow.setObjectName("eyebrow")
    window.project_name_label = QLabel("未打开项目")
    window.project_name_label.setObjectName("projectName")
    window.project_name_label.setWordWrap(True)
    window.project_path_label = QLabel("")
    window.project_path_label.setObjectName("projectPath")
    window.project_path_label.setWordWrap(True)
    window.project_path_label.hide()
    window.project_summary_label = QLabel("")
    window.project_summary_label.setObjectName("muted")
    window.project_summary_label.setWordWrap(True)
    window.project_summary_label.hide()
    window.change_button = QPushButton("打开项目")
    window.change_button.setObjectName("ghostButton")
    window.change_button.clicked.connect(window.choose_project)
    project_layout.addWidget(eyebrow)
    project_layout.addWidget(window.project_name_label)
    project_layout.addWidget(window.project_path_label)
    project_layout.addSpacing(3)
    project_layout.addWidget(window.project_summary_label)
    project_actions = QHBoxLayout()
    project_actions.setSpacing(4)
    project_actions.addWidget(window.change_button)
    window.manual_mapping_button = QPushButton("视图映射")
    window.manual_mapping_button.setObjectName("ghostButton")
    window.manual_mapping_button.clicked.connect(window.open_mapping_settings)
    window.manual_mapping_button.setEnabled(False)
    project_actions.addWidget(window.manual_mapping_button)
    project_actions.addStretch(1)
    project_layout.addLayout(project_actions)
    side.addWidget(project)

    window.modes_panel = QWidget()
    window.modes_panel.setObjectName("sidebarSection")
    modes_layout = QVBoxLayout(window.modes_panel)
    modes_layout.setContentsMargins(14, 10, 14, 10)
    modes_layout.setSpacing(1)
    modes_header = QHBoxLayout()
    modes_title = QLabel("对比视图")
    modes_title.setObjectName("sectionLabel")
    window.selected_count_label = QLabel("已选 0")
    window.selected_count_label.setObjectName("muted")
    modes_header.addWidget(modes_title)
    modes_header.addStretch(1)
    modes_header.addWidget(window.selected_count_label)
    modes_layout.addLayout(modes_header)
    modes_layout.addSpacing(5)
    for index, modality in enumerate(MODALITY_INFO, start=1):
        row = QHBoxLayout()
        checkbox = QCheckBox(f"{index}   {modality_label(modality)}")
        checkbox.setEnabled(False)
        checkbox.toggled.connect(
            lambda checked, name=modality: window._modality_toggled(name, checked)
        )
        count = QLabel("0")
        count.setObjectName("countBadge")
        count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.checkboxes[modality] = checkbox
        window.count_labels[modality] = count
        row.addWidget(checkbox, 1)
        row.addWidget(count)
        modes_layout.addLayout(row)
    window.modes_panel.hide()
    side.addWidget(window.modes_panel)

    window.quality_panel = QWidget()
    window.quality_panel.setObjectName("sidebarSection")
    quality_layout = QVBoxLayout(window.quality_panel)
    quality_layout.setContentsMargins(14, 10, 14, 10)
    quality_layout.setSpacing(5)
    quality_title = QLabel("深度质量")
    quality_title.setObjectName("sectionLabel")
    window.depth_quality_label = QLabel("")
    window.depth_quality_label.setObjectName("qualitySummary")
    window.depth_quality_label.setWordWrap(True)
    quality_layout.addWidget(quality_title)
    quality_layout.addWidget(window.depth_quality_label)
    window.quality_panel.hide()
    side.addWidget(window.quality_panel)

    side.addStretch(1)

    sidebar.hide()
    shell_layout.addWidget(sidebar)
    shell.setMinimumWidth(ACTIVITY_BAR_WIDTH)
    shell.setMaximumWidth(ACTIVITY_BAR_WIDTH)
    return shell


def build_workspace(window) -> QWidget:
    workspace = QWidget()
    workspace.setObjectName("workspace")
    workspace_layout = QVBoxLayout(workspace)
    workspace_layout.setContentsMargins(0, 0, 0, 0)
    workspace_layout.setSpacing(0)

    window.open_button = QPushButton("打开项目…")
    window.open_button.setObjectName("primaryButton")
    window.open_button.clicked.connect(window.choose_project)
    window.project_picker = ChoiceButton()
    window.project_picker.setObjectName("projectPicker")
    window.project_picker.setMinimumWidth(112)
    window.project_picker.setMaximumWidth(210)
    window.project_picker.currentIndexChanged.connect(
        window._project_picker_changed
    )
    window.project_picker.hide()
    window.layout_picker = ChoiceButton()
    window.layout_picker.setObjectName("layoutPicker")
    window.layout_picker.setMinimumWidth(82)
    for label, columns in (("自动布局", 0), ("单列", 1), ("双列", 2), ("三列", 3)):
        window.layout_picker.addItem(label, columns)
    window.layout_picker.currentIndexChanged.connect(window._layout_picker_changed)
    window.header_settings_button = QPushButton("设置")
    window.header_settings_button.setObjectName("titleActionButton")
    window.header_settings_button.clicked.connect(window.open_settings)
    window.title_bar.action_layout.addWidget(window.project_picker)
    window.title_bar.action_layout.addWidget(window.open_button)
    window.title_bar.action_layout.addWidget(window.layout_picker)
    window.title_bar.action_layout.addWidget(window.header_settings_button)

    window.inspection_bar = QFrame()
    window.inspection_bar.setObjectName("inspectionBar")
    window.inspection_bar.setFixedHeight(INSPECTION_BAR_HEIGHT)
    inspection_root = QVBoxLayout(window.inspection_bar)
    inspection_root.setContentsMargins(0, 0, 0, 0)
    inspection_root.setSpacing(0)
    window.inspection_tools_row = QFrame()
    window.inspection_tools_row.setObjectName("inspectionToolsRow")
    window.inspection_tools_row.setFixedHeight(INSPECTION_BAR_HEIGHT)
    inspection_layout = QHBoxLayout(window.inspection_tools_row)
    inspection_layout.setContentsMargins(6, 4, 10, 4)
    inspection_layout.setSpacing(3)
    window.sidebar_button = PaneToggleButton("left")
    window.sidebar_button.setToolTip("收起侧栏 (Ctrl+B)")
    window.sidebar_button.setFixedSize(ICON_PANE_SIZE, ICON_PANE_SIZE)
    window.sidebar_button.clicked.connect(window.toggle_sidebar)
    window.crosshair_button = ToolIconButton("crosshair", "十字光标", checkable=True)
    window.crosshair_button.toggled.connect(window._toggle_crosshair)
    window.sync_views_button = ToolIconButton("sync", "左右图同步缩放和平移", checkable=True)
    window.sync_views_button.toggled.connect(window._sync_views_toggled)
    window.epiline_button = ToolIconButton("epiline", "极线与同名像素联动", checkable=True)
    window.epiline_button.toggled.connect(window._epilines_toggled)
    window.rectify_button = ToolIconButton(
        "rectify",
        "校正前 / 校正后",
        checkable=True,
    )
    window.rectify_button.toggled.connect(window._rectification_toggled)
    window.overlay_button = ToolIconButton("overlay", "左右图透明叠加")
    window.overlay_button.clicked.connect(
        lambda: window.open_stereo_overlay(mode="overlay")
    )
    calibration_caption = QLabel("标定")
    calibration_caption.setObjectName("toolMeta")
    window.calibration_picker = ChoiceButton()
    window.calibration_picker.setObjectName("calibrationPicker")
    window.calibration_picker.setMinimumWidth(132)
    window.calibration_picker.currentIndexChanged.connect(
        window._calibration_picker_changed
    )
    window.rectify_mode_picker = ChoiceButton()
    window.rectify_mode_picker.setObjectName("calibrationPicker")
    window.rectify_mode_picker.setMinimumWidth(72)
    window.rectify_mode_picker.setMaximumWidth(96)
    for mode, label in RECTIFY_MODE_CHOICES:
        window.rectify_mode_picker.addItem(label, mode)
    window.rectify_mode_picker.currentIndexChanged.connect(window._rectify_mode_changed)
    window.rectify_mode_picker.hide()
    window.cursor_info = CursorInfoWidget()
    window.cursor_info.setFixedWidth(CURSOR_INFO_WIDTH)
    window.view_hint = QLabel("")
    window.view_hint.setObjectName("viewHint")
    window.view_hint.hide()
    window.reset_button = QPushButton("重置视图")
    window.reset_button.setObjectName("ghostButton")
    window.reset_button.clicked.connect(window.reset_views)
    window.reset_button.hide()
    inspection_layout.addWidget(window.sidebar_button)
    inspection_layout.addWidget(window.crosshair_button)
    inspection_layout.addWidget(window.sync_views_button)
    inspection_layout.addWidget(window.epiline_button)
    inspection_layout.addWidget(window.rectify_button)
    inspection_layout.addWidget(window.overlay_button)
    inspection_layout.addWidget(window.reset_button)
    inspection_layout.addSpacing(6)
    inspection_layout.addWidget(calibration_caption)
    inspection_layout.addWidget(window.calibration_picker)
    inspection_layout.addWidget(window.rectify_mode_picker)
    inspection_layout.addStretch(1)
    inspection_layout.addWidget(window.view_hint)
    inspection_root.addWidget(window.inspection_tools_row)
    window.cursor_info_row = QFrame()
    window.cursor_info_row.setObjectName("cursorInfoRow")
    window.cursor_info_row.setFixedHeight(CURSOR_INFO_ROW_HEIGHT)
    cursor_layout = QHBoxLayout(window.cursor_info_row)
    cursor_layout.setContentsMargins(8, 0, 10, 4)
    cursor_layout.setSpacing(0)
    cursor_layout.addStretch(1)
    cursor_layout.addWidget(window.cursor_info)
    window.cursor_info_row.hide()
    inspection_root.addWidget(window.cursor_info_row)
    window.inspection_bar.hide()
    workspace_layout.addWidget(window.inspection_bar)

    window.content_stack = QStackedWidget()
    window.empty_hint = DropHint()
    window.empty_hint.open_requested.connect(window.choose_project)
    window.empty_hint.recent_requested.connect(window._open_recent_project)
    window.content_stack.addWidget(window.empty_hint)
    window.media_container = QWidget()
    window.media_container.setObjectName("mediaWorkspace")
    window.media_grid = QGridLayout(window.media_container)
    window.media_grid.setContentsMargins(MEDIA_MARGIN, MEDIA_MARGIN, MEDIA_MARGIN, MEDIA_MARGIN)
    window.media_grid.setSpacing(MEDIA_SPACING)
    window.no_views_hint = NoViewsHint()
    # Create the native OpenGL child before the frameless top-level window is
    # shown. The backend itself is imported in parallel while the splash is up.
    window.gl_warmup = PointCloudCanvas(window.media_container)
    window.gl_warmup.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.gl_warmup.setGeometry(0, 0, 2, 2)
    window.gl_warmup.show()
    window.content_stack.addWidget(window.media_container)
    workspace_layout.addWidget(window.content_stack, 1)

    window.player_bar = QFrame()
    window.player_bar.setObjectName("playerBar")
    window.player_bar.setFixedHeight(40)
    player_layout = QVBoxLayout(window.player_bar)
    player_layout.setContentsMargins(8, 0, 8, 0)
    player_layout.setSpacing(0)
    window.timeline_panel = QFrame()
    window.timeline_panel.setObjectName("timelinePanel")
    timeline_row = QHBoxLayout(window.timeline_panel)
    timeline_row.setContentsMargins(2, 5, 2, 5)
    timeline_row.setSpacing(4)
    window.playback_previous_button = PlaybackControlButton("previous")
    window.playback_previous_button.setToolTip("上一帧")
    window.playback_previous_button.clicked.connect(window.previous_sample)
    window.playback_button = PlaybackControlButton("play")
    window.playback_button.setToolTip("播放")
    window.playback_button.clicked.connect(window.toggle_playback)
    window.playback_next_button = PlaybackControlButton("next")
    window.playback_next_button.setToolTip("下一帧")
    window.playback_next_button.clicked.connect(window.next_sample)
    window.timeline_left = QLabel("0")
    window.timeline_left.setObjectName("sampleMeta")
    window.timeline_left.setFixedWidth(28)
    window.timeline_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.timeline = TimelineSlider()
    window.timeline.setRange(0, 0)
    window.timeline.setEnabled(False)
    window.timeline.setToolTip("单击或拖动跳转")
    window.timeline.valueChanged.connect(window._timeline_changed)
    window.timeline.sliderMoved.connect(
        lambda value: window.timeline_left.setText(str(value + 1))
    )
    window.timeline.previewChanged.connect(window._timeline_preview_changed)
    window.timeline.previewCleared.connect(window._timeline_preview_cleared)
    window.timeline.sliderPressed.connect(window._timeline_scrub_started)
    window.timeline.sliderReleased.connect(window._timeline_scrub_finished)
    window.timeline_right = QLabel("0")
    window.timeline_right.setObjectName("sampleMeta")
    window.timeline_right.setFixedWidth(28)
    window.timeline_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.playback_speed = ChoiceButton()
    window.playback_speed.setObjectName("playbackSpeed")
    window.playback_speed.setMinimumWidth(82)
    for label, fps in (
        ("0.5 fps", 0.5),
        ("1 fps", 1.0),
        ("2 fps", 2.0),
        ("5 fps", 5.0),
        ("10 fps", 10.0),
    ):
        window.playback_speed.addItem(label, fps)
    window.playback_speed.setCurrentIndex(window.playback_speed.findData(2.0))
    window.playback_speed.currentIndexChanged.connect(window._playback_speed_changed)
    timeline_row.addWidget(window.playback_previous_button)
    timeline_row.addWidget(window.playback_button)
    timeline_row.addWidget(window.playback_next_button)
    timeline_row.addSpacing(4)
    timeline_row.addWidget(window.timeline_left)
    timeline_row.addWidget(window.timeline, 1)
    timeline_row.addWidget(window.timeline_right)
    timeline_row.addSpacing(4)
    timeline_row.addWidget(window.playback_speed)
    window.status_text.setFixedWidth(160)
    window.status_text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    timeline_row.addWidget(window.status_text)
    player_layout.addWidget(window.timeline_panel)

    window.player_bar.hide()
    workspace_layout.addWidget(window.player_bar)
    return workspace


def build_inspector(window) -> InspectorPanel:
    inspector = InspectorPanel()
    inspector.close_requested.connect(window.close_inspector)
    inspector.source_changed.connect(window._inspector_source_changed)
    inspector.display_changed.connect(window._apply_display_settings)
    inspector.fit_requested.connect(window._fit_requested)
    inspector.tool_requested.connect(window._tool_requested)
    inspector.export_requested.connect(window.export_current_view)
    inspector.export_image_requested.connect(window.export_adjusted_image)
    inspector.cloud_color_changed.connect(window._cloud_color_changed)
    inspector.cloud_size_changed.connect(window._cloud_size_changed)
    inspector.cloud_guides_changed.connect(window._cloud_guides_changed)
    inspector.cloud_view_requested.connect(window._cloud_view_requested)
    inspector.cloud_tool_requested.connect(window._cloud_tool_requested)
    inspector.cloud_background_changed.connect(window._cloud_background_changed)
    inspector.cloud_clip_changed.connect(window._cloud_clip_changed)
    return inspector
