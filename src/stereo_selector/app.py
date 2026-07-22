from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, QThreadPool, Qt, QTimer
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
from . import __version__
from .mapping import MappingDialog
from .settings import AppPreferences, SettingsDialog
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
        self.focused_modality: str | None = None
        self.checkboxes: dict[str, QCheckBox] = {}
        self.count_labels: dict[str, QLabel] = {}
        self.tiles: dict[str, MediaTile] = {}
        self.tile_pool: dict[str, MediaTile] = {}
        self.shortcuts: list[QShortcut] = []
        self.action_shortcuts: dict[str, QShortcut] = {}
        self._native_frame_applied = False

        self._build_ui()
        self._build_shortcuts()
        self._apply_preferences()
        self._update_controls()

        if initial_project is not None:
            QTimer.singleShot(0, lambda: self.load_project(initial_project))
        else:
            QTimer.singleShot(180, self.choose_project)

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

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setChildrenCollapsible(False)
        central_layout.addWidget(self.splitter, 1)

        self.sidebar = self._build_sidebar()
        workspace = self._build_workspace()
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(workspace)
        self.splitter.setSizes([286, 1234])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        status = QStatusBar()
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)
        self.status_text = QLabel("就绪  ·  拖放项目文件夹到窗口即可打开")
        self.status_text.setContentsMargins(7, 0, 0, 0)
        status.addWidget(self.status_text, 1)
        self.shortcut_hint = QLabel("←/→ 浏览   Enter 接受   F 聚焦   Ctrl+B 侧栏")
        self.shortcut_hint.setObjectName("muted")
        status.addPermanentWidget(self.shortcut_hint)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(238)
        sidebar.setMaximumWidth(360)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)

        project = QWidget()
        project_layout = QVBoxLayout(project)
        project_layout.setContentsMargins(16, 18, 16, 14)
        project_layout.setSpacing(5)
        eyebrow = QLabel("PROJECT")
        eyebrow.setObjectName("eyebrow")
        self.project_name_label = QLabel("未打开项目")
        self.project_name_label.setObjectName("projectName")
        self.project_name_label.setWordWrap(True)
        self.project_path_label = QLabel("选择包含双目数据的文件夹")
        self.project_path_label.setObjectName("projectPath")
        self.project_path_label.setWordWrap(True)
        self.project_summary_label = QLabel("等待扫描")
        self.project_summary_label.setObjectName("muted")
        self.project_summary_label.setWordWrap(True)
        change_button = QPushButton("更改项目")
        change_button.setObjectName("ghostButton")
        change_button.clicked.connect(self.choose_project)
        project_layout.addWidget(eyebrow)
        project_layout.addWidget(self.project_name_label)
        project_layout.addWidget(self.project_path_label)
        project_layout.addSpacing(3)
        project_layout.addWidget(self.project_summary_label)
        project_actions = QHBoxLayout()
        project_actions.setSpacing(4)
        project_actions.addWidget(change_button)
        self.manual_mapping_button = QPushButton("配置视图与匹配")
        self.manual_mapping_button.setObjectName("ghostButton")
        self.manual_mapping_button.clicked.connect(self.open_mapping_settings)
        project_actions.addWidget(self.manual_mapping_button)
        project_actions.addStretch(1)
        project_layout.addLayout(project_actions)
        side.addWidget(project)

        modes = QWidget()
        modes_layout = QVBoxLayout(modes)
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
        side.addWidget(modes)

        review_wrap = QWidget()
        review_wrap_layout = QVBoxLayout(review_wrap)
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
        side.addWidget(review_wrap)
        side.addStretch(1)

        output = QWidget()
        output_layout = QVBoxLayout(output)
        output_layout.setContentsMargins(14, 9, 14, 14)
        output_layout.setSpacing(6)
        output_title = QLabel("输出")
        output_title.setObjectName("sectionLabel")
        self.output_label = QLabel("接受后按原结构复制到项目同级目录")
        self.output_label.setObjectName("projectPath")
        self.output_label.setWordWrap(True)
        self.open_output_button = QPushButton("打开输出文件夹")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output_folder)
        output_layout.addWidget(output_title)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.open_output_button)
        side.addWidget(output)
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
        self.view_hint = QLabel("选择项目后开始审阅")
        self.view_hint.setObjectName("viewHint")
        breadcrumb_box.addWidget(self.header_breadcrumb)
        breadcrumb_box.addWidget(self.view_hint)
        self.reset_button = QPushButton("重置视图  R")
        self.reset_button.setObjectName("ghostButton")
        self.reset_button.clicked.connect(self.reset_views)
        self.open_button = QPushButton("打开项目…")
        self.open_button.setObjectName("primaryButton")
        self.open_button.clicked.connect(self.choose_project)
        header_layout.addWidget(self.sidebar_button)
        header_layout.addLayout(breadcrumb_box, 1)
        header_layout.addWidget(self.reset_button)
        header_layout.addWidget(self.open_button)
        workspace_layout.addWidget(header)

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
        # A hidden OpenGL child exists before the top-level window is first shown.
        # This prevents Windows from recreating the native window when PLY is selected later.
        self.gl_warmup = PointCloudCanvas(self.media_container)
        self.gl_warmup.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        self.gl_warmup.setGeometry(0, 0, 2, 2)
        self.gl_warmup.show()
        self.content_stack.addWidget(self.media_container)
        workspace_layout.addWidget(self.content_stack, 1)

        review = QFrame()
        review.setObjectName("reviewBar")
        review_layout = QVBoxLayout(review)
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
        self.previous_button = QPushButton("←  上一组")
        self.previous_button.clicked.connect(self.previous_sample)
        self.next_button = QPushButton("跳过 / 下一组  →")
        self.next_button.clicked.connect(self.next_sample)
        sample_box = QVBoxLayout()
        sample_box.setSpacing(1)
        sample_top = QHBoxLayout()
        self.sample_title = QLabel("未加载样本")
        self.sample_title.setObjectName("sampleTitle")
        self.sample_status = QLabel("等待项目")
        self.sample_status.setObjectName("statusPill")
        self.sample_status.setProperty("accepted", False)
        sample_top.addWidget(self.sample_title)
        sample_top.addWidget(self.sample_status)
        sample_top.addStretch(1)
        self.sample_meta = QLabel("选择项目后开始审阅")
        self.sample_meta.setObjectName("sampleMeta")
        sample_box.addLayout(sample_top)
        sample_box.addWidget(self.sample_meta)
        self.accept_button = QPushButton("接受并继续  Enter")
        self.accept_button.setObjectName("acceptButton")
        self.accept_button.clicked.connect(self.accept_current)
        controls.addWidget(self.previous_button)
        controls.addWidget(self.next_button)
        controls.addSpacing(10)
        controls.addLayout(sample_box, 1)
        controls.addWidget(self.accept_button)
        review_layout.addLayout(controls)
        workspace_layout.addWidget(review)
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
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            dataset = self.scanner.scan(root, manual_dirs=manual_dirs, force_order=force_order)
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
        self.manual_dirs = dict(manual_dirs or {})
        self.force_order = force_order
        self.current_index = 0
        self.focused_modality = None
        self.accepted = {
            sample.key
            for sample in dataset.samples
            if sample_is_copied(dataset, sample)
        }
        self.settings.setValue("last_project", str(dataset.root))
        self._save_matching_for(dataset.root, self.manual_dirs, self.force_order)
        self.setWindowTitle(f"{dataset.root.name} — Stereo Selector")
        self.title_bar.set_context(dataset.root.name)
        self.project_name_label.setText(dataset.root.name)
        self.project_path_label.setText(str(dataset.root))
        self.project_path_label.setToolTip(str(dataset.root))
        match_mode = "按顺序强制匹配" if dataset.force_order else "按文件名匹配"
        available_set = set(dataset.available_modalities)
        complete_count = sum(available_set.issubset(sample.files) for sample in dataset.samples)
        incomplete_count = len(dataset.samples) - complete_count
        incomplete_text = f" · {incomplete_count} 组不完整" if incomplete_count else ""
        self.project_summary_label.setText(
            f"{len(dataset.samples)} 组样本 · {len(dataset.available_modalities)} 种数据 · {match_mode}{incomplete_text}"
        )
        self.output_label.setText(str(dataset.output_root))
        self.output_label.setToolTip(str(dataset.output_root))

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
        self.status_text.setText(
            f"已打开 {dataset.root.name}  ·  {len(dataset.samples)} 组样本"
            + (f"  ·  {incomplete_count} 组缺少部分视图" if incomplete_count else "")
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
            tile = MediaTile(modality, self.preferences.point_limit, self.media_container)
            tile.focus_requested.connect(self.focus_modality)
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
            self.view_hint.setText("未选择视图 · 使用数字键 1–5 快速切换")
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
            self.view_hint.setText("聚焦模式 · Esc 或 F 返回对比")
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
        self.view_hint.setText("滚轮缩放 · 拖动平移 · 双击适应")

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
        for modality, tile in self.tiles.items():
            tile.show_file(sample.files.get(modality))

        self.sample_title.setText(sample.display_name)
        self.sample_meta.setText(
            f"第 {self.current_index + 1} / {len(self.dataset.samples)} 组  ·  匹配 {len(sample.files)} 个文件"
        )
        is_accepted = sample.key in self.accepted
        self.sample_status.setText("已接受" if is_accepted else "待审阅")
        if bool(self.sample_status.property("accepted")) != is_accepted:
            self.sample_status.setProperty("accepted", is_accepted)
            self.sample_status.style().unpolish(self.sample_status)
            self.sample_status.style().polish(self.sample_status)
        shortcut = self._shortcut_text("accept")
        accept_label = self.preferences.button_labels["accept"]
        self.accept_button.setText(f"已接受 · 继续  {shortcut}" if is_accepted else f"{accept_label}  {shortcut}")
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
            self.status_text.setText("所有已选视图均已匹配")
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
        self.next_unreviewed_button.setEnabled(bool(has_samples and len(self.accepted) < count))

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
        self.status_text.setText("当前项目的所有样本都已接受")

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
        self._update_review_stats()
        if self.preferences.auto_advance and self.current_index < len(self.dataset.samples) - 1:
            self.current_index += 1
        self._show_current()
        if copied_count:
            self.status_text.setText(f"已接受 {sample.display_name}  ·  复制 {copied_count} 个文件")
        else:
            self.status_text.setText(f"{sample.display_name} 已接受，已继续浏览")

    def reset_views(self) -> None:
        for tile in self.tiles.values():
            tile.reset_view()
        self.status_text.setText("视图已适应窗口；点云恢复到相机拍摄正方向")

    def toggle_sidebar(self) -> None:
        visible = not self.sidebar.isVisible()
        self.sidebar.setVisible(visible)
        self.sidebar_button.setText("‹" if visible else "›")
        self.sidebar_button.setToolTip(("收起" if visible else "展开") + "侧栏 (Ctrl+B)")

    def _shortcut_text(self, action: str) -> str:
        text = QKeySequence(self.preferences.shortcuts[action]).toString(QKeySequence.SequenceFormat.NativeText)
        return "Enter" if text in {"Return", "回车"} else text

    def _apply_preferences(self) -> None:
        QApplication.instance().setStyleSheet(style_for(self.preferences.theme))
        for action, shortcut in self.action_shortcuts.items():
            shortcut.setKey(QKeySequence(self.preferences.shortcuts[action]))
        self.previous_button.setText(f"{self.preferences.button_labels['previous']}  {self._shortcut_text('previous')}")
        self.next_button.setText(f"{self.preferences.button_labels['next']}  {self._shortcut_text('next')}")
        self.accept_button.setText(f"{self.preferences.button_labels['accept']}  {self._shortcut_text('accept')}")
        self.reset_button.setText(f"重置视图  {self._shortcut_text('reset')}")
        self.shortcut_hint.setText(
            f"{self._shortcut_text('previous')}/{self._shortcut_text('next')} 浏览   "
            f"{self._shortcut_text('accept')} 接受   {self._shortcut_text('focus')} 聚焦   Ctrl+, 设置"
        )

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.preferences, self)
        if not dialog.exec():
            return
        point_limit_changed = dialog.preferences.point_limit != self.preferences.point_limit
        self.preferences = dialog.preferences
        self.preferences.save(self.settings)
        self._apply_preferences()
        if point_limit_changed and self.dataset is not None:
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_test and args.project is None:
        raise SystemExit("--smoke-test requires a project folder")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv[:1] if argv is not None else sys.argv)
    app.setApplicationName("Stereo Selector")
    app.setOrganizationName("ToolBox")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    startup_preferences = AppPreferences.load(QSettings("ToolBox", "StereoSelector"))
    app.setStyleSheet(style_for(startup_preferences.theme))
    QThreadPool.globalInstance().setMaxThreadCount(3)
    window = MainWindow(args.project)
    window.show()
    if args.smoke_test:
        start_smoke_test(app, window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
