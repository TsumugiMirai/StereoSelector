from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QLineF, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .media import ImageData, ImageStats, PointCloudData
from .models import modality_label
from .settings import DialogCloseButton
from .ui_controls import ChoiceButton
from .ui_metrics import INSPECTOR_MAX_WIDTH, INSPECTOR_MIN_WIDTH
from .units import format_length
from .workers import ANALYSIS_POOL, RoiWorker


class FileInformationLabel(QLabel):
    """Keep metadata readable while eliding only the variable-length path."""

    def __init__(self) -> None:
        super().__init__("")
        self.full_path = ""
        self.details = ""
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    def set_file(self, path: Path | None, details: str = "") -> None:
        self.full_path = str(path or "")
        self.details = details
        self.setToolTip("\n".join(value for value in (self.full_path, details) if value))
        self._refresh_text()

    def _refresh_text(self) -> None:
        path = self.fontMetrics().elidedText(
            self.full_path, Qt.TextElideMode.ElideMiddle,
            max(20, self.contentsRect().width()),
        )
        self.setText("\n".join(value for value in (path, self.details) if value))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_text()


class HistogramWidget(QWidget):
    """Small theme-aware histogram suitable for the compact inspector."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("histogram")
        self.setMinimumHeight(128)
        self._histograms: tuple[np.ndarray, ...] = ()

    def set_histograms(self, histograms: tuple[np.ndarray, ...]) -> None:
        self._histograms = tuple(np.asarray(item, dtype=np.float64) for item in histograms if np.asarray(item).size)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        bounds = self.rect().adjusted(8, 8, -8, -8)
        painter.setPen(QPen(self.palette().mid().color(), 1))
        painter.drawRoundedRect(bounds, 6, 6)
        if not self._histograms:
            painter.setPen(self.palette().mid().color())
            painter.drawText(bounds, Qt.AlignmentFlag.AlignCenter, "暂无数据")
            return
        maximum = max(float(hist.max()) for hist in self._histograms if hist.size)
        if maximum <= 0:
            return
        colors = (
            QColor(255, 92, 92, 180),
            QColor(70, 210, 125, 180),
            QColor(75, 150, 255, 180),
        )
        for channel, histogram in enumerate(self._histograms):
            if not histogram.size:
                continue
            painter.setPen(QPen(colors[channel % len(colors)] if len(self._histograms) > 1 else QColor("#4c9ffe"), 1.2))
            previous = None
            for index, value in enumerate(histogram):
                x = bounds.left() + index * bounds.width() / max(1, len(histogram) - 1)
                y = bounds.bottom() - float(value) / maximum * (bounds.height() - 2)
                if previous is not None:
                    painter.drawLine(QLineF(previous[0], previous[1], x, y))
                previous = (x, y)


class InspectorPanel(QFrame):
    close_requested = Signal()
    source_changed = Signal(str)
    display_changed = Signal(object)
    fit_requested = Signal(str)
    tool_requested = Signal(str)
    export_requested = Signal()
    export_image_requested = Signal()
    cloud_color_changed = Signal(str)
    cloud_size_changed = Signal(float)
    cloud_guides_changed = Signal(bool)
    cloud_view_requested = Signal(str)
    cloud_tool_requested = Signal(str)
    cloud_background_changed = Signal(str)
    cloud_clip_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("inspector")
        self.setMinimumWidth(INSPECTOR_MIN_WIDTH)
        self.setMaximumWidth(INSPECTOR_MAX_WIDTH)
        self._source = ""
        self._numeric_source = False
        self._category = "statistics"
        self._ready = False
        self._cloud_identity = None
        self._roi_token = 0
        self._roi_workers: dict[int, RoiWorker] = {}
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(45)
        self._update_timer.timeout.connect(self._emit_display_settings)
        self._clip_timer = QTimer(self)
        self._clip_timer.setSingleShot(True)
        self._clip_timer.setInterval(80)
        self._clip_timer.timeout.connect(self._emit_cloud_clip)
        self._clip_initialized = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QFrame()
        header.setObjectName("inspectorHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 3, 5, 3)
        self.title = QLabel("统计信息")
        self.title.setObjectName("inspectorTitle")
        close = DialogCloseButton()
        close.setToolTip("关闭检查器")
        close.clicked.connect(self.close_requested)
        header_layout.addWidget(self.title)
        header_layout.addStretch(1)
        header_layout.addWidget(close)
        outer.addWidget(header)

        scroll = QScrollArea()
        self.scroll = scroll
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("inspectorBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(0)

        self.source_picker = ChoiceButton()
        self.source_picker.setObjectName("inspectorChoice")
        self.source_picker.currentIndexChanged.connect(self._source_selected)
        self.source_section = self._section("数据源", self.source_picker)
        layout.addWidget(self.source_section)

        self.histogram = HistogramWidget()
        self.stats_label = QLabel("选择图片后显示统计")
        self.stats_label.setObjectName("inspectorStats")
        self.stats_label.setWordWrap(True)
        analysis = QFrame()
        analysis.setObjectName("inspectorSection")
        analysis_layout = QVBoxLayout(analysis)
        analysis_layout.setContentsMargins(12, 10, 12, 12)
        analysis_layout.setSpacing(7)
        analysis_layout.addWidget(self._heading("直方图与统计"))
        analysis_layout.addWidget(self.histogram)
        analysis_layout.addWidget(self.stats_label)
        self.analysis_section = analysis
        layout.addWidget(analysis)

        display = QFrame()
        display.setObjectName("inspectorSection")
        display_layout = QVBoxLayout(display)
        display_layout.setContentsMargins(12, 10, 12, 12)
        display_layout.setSpacing(6)
        display_layout.addWidget(self._heading("显示"))
        self.brightness = self._slider(-100, 100, 0, display_layout, "亮度")
        self.contrast = self._slider(10, 300, 100, display_layout, "对比度")
        self.gamma = self._slider(10, 300, 100, display_layout, "Gamma")
        self.exposure = self._slider(-40, 40, 0, display_layout, "曝光")
        self.depth_controls = QFrame()
        depth_layout = QVBoxLayout(self.depth_controls)
        depth_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.addWidget(self.depth_controls)
        self.lut = ChoiceButton()
        self.lut.setObjectName("inspectorChoice")
        for text, value in (("灰度", "gray"), ("Turbo", "turbo"), ("Viridis", "viridis")):
            self.lut.addItem(text, value)
        self.lut.currentIndexChanged.connect(lambda _index: self._schedule_display_update())
        depth_layout.addWidget(self._labeled("数值 LUT", self.lut))
        self.auto_range = QCheckBox("自动百分位拉伸")
        self.auto_range.setChecked(True)
        self.auto_range.toggled.connect(self._range_mode_changed)
        depth_layout.addWidget(self.auto_range)
        ranges = QHBoxLayout()
        self.range_min = QDoubleSpinBox()
        self.range_max = QDoubleSpinBox()
        for spin in (self.range_min, self.range_max):
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setDecimals(2)
            spin.valueChanged.connect(lambda _value: self._schedule_display_update())
        self.range_min.setValue(0)
        self.range_max.setValue(10_000)
        ranges.addWidget(self.range_min)
        ranges.addWidget(self.range_max)
        depth_layout.addLayout(ranges)
        self.highlight_invalid = QCheckBox("高亮 0 / 65535 / 无效 / 越界")
        self.highlight_invalid.toggled.connect(lambda _checked: self._schedule_display_update())
        depth_layout.addWidget(self.highlight_invalid)
        self._range_mode_changed(True)
        self.display_section = display
        layout.addWidget(display)

        view_tools = QFrame()
        view_tools.setObjectName("inspectorSection")
        view_layout = QVBoxLayout(view_tools)
        view_layout.setContentsMargins(12, 10, 12, 12)
        view_layout.setSpacing(6)
        view_layout.addWidget(self._heading("视图与测量"))
        fit_row = QHBoxLayout()
        for label, mode in (("1:1", "actual"), ("适应", "fit"), ("宽度", "width")):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=mode: self.fit_requested.emit(value))
            fit_row.addWidget(button)
        view_layout.addLayout(fit_row)
        tool_row = QHBoxLayout()
        self.image_tool_group = QButtonGroup(self)
        self.image_tool_group.setExclusive(True)
        self.image_tool_buttons: dict[str, QPushButton] = {}
        for label, tool in (("平移", "pan"), ("矩形 ROI", "roi"), ("线段", "line")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(tool == "pan")
            self.image_tool_group.addButton(button)
            self.image_tool_buttons[tool] = button
            button.clicked.connect(lambda _checked=False, value=tool: self.tool_requested.emit(value))
            tool_row.addWidget(button)
        view_layout.addLayout(tool_row)
        self.selection_label = QLabel("ROI 与测量结果显示在这里")
        self.selection_label.setObjectName("inspectorStats")
        self.selection_label.setWordWrap(True)
        self.result_section = self._section("测量结果", self.selection_label)
        export_box = QWidget()
        export_layout = QVBoxLayout(export_box)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(4)
        export = QPushButton("导出当前视图…")
        export.setToolTip("按屏幕显示内容截图")
        export.clicked.connect(self.export_requested)
        self.export_image_button = QPushButton("导出调整后图像…")
        self.export_image_button.setToolTip("按原始分辨率应用显示调整后保存")
        self.export_image_button.clicked.connect(self.export_image_requested)
        export_layout.addWidget(export)
        export_layout.addWidget(self.export_image_button)
        self.export_section = self._section("导出", export_box)
        self.view_tools_section = view_tools
        layout.addWidget(view_tools)

        cloud = QFrame()
        cloud.setObjectName("inspectorSection")
        cloud_layout = QVBoxLayout(cloud)
        cloud_layout.setContentsMargins(12, 10, 12, 12)
        cloud_layout.setSpacing(6)
        cloud_layout.addWidget(self._heading("点云"))
        self.cloud_color = ChoiceButton()
        self.cloud_color.setObjectName("inspectorChoice")
        for text, value in (
            ("RGB", "rgb"),
            ("深度 Z", "z"),
            ("X", "x"),
            ("Y", "y"),
            ("高度", "height"),
        ):
            self.cloud_color.addItem(text, value)
        self.cloud_color.currentIndexChanged.connect(
            lambda _index: self.cloud_color_changed.emit(str(self.cloud_color.currentData()))
        )
        cloud_layout.addWidget(self._labeled("着色", self.cloud_color))
        self.cloud_size = self._slider(5, 120, 10, cloud_layout, "点大小")
        self.cloud_size.valueChanged.connect(
            lambda value: self.cloud_size_changed.emit(value / 10.0)
        )
        self.cloud_guides = QCheckBox("坐标轴与网格")
        self.cloud_guides.toggled.connect(self.cloud_guides_changed)
        cloud_layout.addWidget(self.cloud_guides)
        self.cloud_section = cloud
        layout.addWidget(cloud)

        cloud_tools_section = QFrame()
        cloud_tools_section.setObjectName("inspectorSection")
        cloud_tools_layout = QVBoxLayout(cloud_tools_section)
        cloud_tools_layout.setContentsMargins(12, 10, 12, 12)
        cloud_tools_layout.addWidget(self._heading("选择与测量"))
        cloud_tools = QHBoxLayout()
        self.cloud_tool_group = QButtonGroup(self)
        self.cloud_tool_buttons: dict[str, QPushButton] = {}
        for label, tool in (
            ("旋转", "rotate"),
            ("点选", "pick"),
            ("两点测距", "measure"),
            ("框选", "box"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(tool == "rotate")
            self.cloud_tool_group.addButton(button)
            self.cloud_tool_buttons[tool] = button
            button.clicked.connect(
                lambda _checked=False, value=tool: self.cloud_tool_requested.emit(value)
            )
            cloud_tools.addWidget(button)
        cloud_tools_layout.addLayout(cloud_tools)
        reset_cloud = QPushButton("恢复全部点")
        reset_cloud.clicked.connect(
            lambda: self.cloud_tool_requested.emit("reset")
        )
        cloud_tools_layout.addWidget(reset_cloud)
        self.cloud_tools_section = cloud_tools_section
        layout.addWidget(cloud_tools_section)
        cloud_views = QHBoxLayout()
        for label, view in (("前", "front"), ("后", "back"), ("左", "left"), ("右", "right"), ("俯", "top")):
            button = QPushButton(label)
            button.setToolTip(f"{label}视图")
            button.clicked.connect(
                lambda _checked=False, value=view: self.cloud_view_requested.emit(value)
            )
            cloud_views.addWidget(button)
        cloud_layout.addLayout(cloud_views)
        cloud_clip = QFrame()
        cloud_clip.setObjectName("inspectorSection")
        clip_layout = QVBoxLayout(cloud_clip)
        clip_layout.setContentsMargins(12, 10, 12, 12)
        clip_layout.addWidget(self._heading("范围裁剪"))
        self.cloud_clip_spins: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        for axis in ("X", "Y", "Z"):
            lower = QDoubleSpinBox()
            upper = QDoubleSpinBox()
            for spin in (lower, upper):
                spin.setRange(-1_000_000, 1_000_000)
                spin.setDecimals(3)
                spin.valueChanged.connect(lambda _value: self._clip_timer.start())
            row = QHBoxLayout()
            label = QLabel(f"{axis} 裁剪")
            label.setObjectName("muted")
            row.addWidget(label)
            row.addWidget(lower)
            row.addWidget(upper)
            clip_layout.addLayout(row)
            self.cloud_clip_spins[axis.lower()] = (lower, upper)
        background = ChoiceButton()
        background.setObjectName("inspectorChoice")
        for label, value in (("深色背景", "#0d0d0d"), ("浅色背景", "#eeeeec")):
            background.addItem(label, value)
        background.currentIndexChanged.connect(
            lambda _index: self.cloud_background_changed.emit(
                str(background.currentData() or "#0d0d0d")
            )
        )
        cloud_layout.addWidget(self._labeled("背景", background))
        self.cloud_stats = QLabel("加载点云后显示范围")
        self.cloud_stats.setObjectName("inspectorStats")
        self.cloud_stats.setWordWrap(True)
        self.cloud_clip_section = cloud_clip
        self.cloud_stats_section = self._section("点数与范围", self.cloud_stats)
        layout.addWidget(cloud_clip)
        layout.addWidget(self.cloud_stats_section)
        layout.addWidget(self.result_section)
        layout.addWidget(self.export_section)

        self.file_label = FileInformationLabel()
        self.file_label.setObjectName("inspectorFile")
        self.file_label.setWordWrap(True)
        self.file_label.setMinimumWidth(0)
        self.file_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.file_section = self._section("文件", self.file_label)
        file_layout = self.file_section.layout()
        heading = file_layout.takeAt(0).widget()
        file_header = QHBoxLayout()
        file_header.addWidget(heading)
        file_header.addStretch(1)
        self.copy_path_button = QPushButton("复制路径")
        self.copy_path_button.setObjectName("ghostButton")
        self.copy_path_button.setToolTip("复制完整文件路径")
        self.copy_path_button.setEnabled(False)
        self.copy_path_button.clicked.connect(
            lambda: QApplication.clipboard().setText(self.file_label.full_path)
        )
        file_header.addWidget(self.copy_path_button)
        file_layout.insertLayout(0, file_header)
        layout.addWidget(self.file_section)
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.set_category("statistics")

    def set_category(self, category: str) -> None:
        """Group by user task, then adapt controls to the active data source."""
        category = category if category in {"adjust", "measure", "statistics"} else "statistics"
        changed = category != self._category
        self._category = category
        titles = {
            "adjust": "显示调整",
            "measure": "测量与选择",
            "statistics": "统计信息",
        }
        self.title.setText(titles[category])
        cloud = self._source == "ply"
        self.depth_controls.setVisible(self._numeric_source)
        self.export_image_button.setVisible(not cloud)
        visibility = {
            "adjust": {
                self.cloud_section if cloud else self.display_section,
                self.export_section,
            },
            "measure": {
                self.cloud_tools_section if cloud else self.view_tools_section,
                self.result_section,
                *([self.cloud_clip_section] if cloud else []),
            },
            "statistics": {
                self.cloud_stats_section if cloud else self.analysis_section,
                self.file_section,
            },
        }[category]
        visibility.add(self.source_section)
        for section in (
            self.source_section,
            self.analysis_section,
            self.display_section,
            self.view_tools_section,
            self.cloud_section,
            self.cloud_tools_section,
            self.cloud_clip_section,
            self.cloud_stats_section,
            self.result_section,
            self.export_section,
            self.file_section,
        ):
            section.setVisible(section in visibility)
        if changed:
            self.scroll.verticalScrollBar().setValue(0)
        self.set_ready(self._ready)

    def set_ready(self, ready: bool) -> None:
        self._ready = ready
        for section in (self.display_section, self.view_tools_section,
                        self.cloud_section, self.cloud_tools_section,
                        self.cloud_clip_section, self.export_section):
            section.setEnabled(ready)

    @property
    def category(self) -> str:
        return self._category

    def invalidate_roi(self) -> None:
        """Discard any ROI result still being computed."""
        self._roi_token += 1

    def reset_clip_bounds(self) -> None:
        """Re-read the clip spin boxes from the next cloud that is shown."""
        self._clip_initialized = False

    def reset_tools(self) -> None:
        self.image_tool_buttons["pan"].setChecked(True)
        self.cloud_tool_buttons["rotate"].setChecked(True)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("inspectorHeading")
        return label

    def _section(self, title: str, control: QWidget) -> QFrame:
        section = QFrame()
        section.setObjectName("inspectorSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)
        layout.addWidget(self._heading(title))
        layout.addWidget(control)
        return section

    @staticmethod
    def _labeled(text: str, control: QWidget) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(text)
        label.setObjectName("muted")
        layout.addWidget(label)
        layout.addWidget(control, 1)
        return widget

    def _slider(
        self,
        minimum: int,
        maximum: int,
        value: int,
        layout: QVBoxLayout,
        label: str,
    ) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        value_label = QLabel(str(value))
        value_label.setObjectName("inspectorValue")
        value_label.setFixedWidth(34)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider.value_label = value_label
        row = QHBoxLayout()
        name = QLabel(label)
        name.setObjectName("muted")
        row.addWidget(name)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        layout.addLayout(row)
        slider.valueChanged.connect(lambda number, target=value_label: target.setText(str(number)))
        if label != "点大小":
            slider.valueChanged.connect(lambda _number: self._schedule_display_update())
        return slider

    def set_sources(self, modalities: list[str], selected: str = "") -> None:
        selected = selected or self._source
        if [self.source_picker.itemData(i) for i in range(self.source_picker.count())] == modalities:
            index = self.source_picker.findData(selected)
            if index >= 0:
                self.source_picker.setCurrentIndex(index)
            return
        self.source_picker.blockSignals(True)
        self.source_picker.clear()
        for modality in modalities:
            self.source_picker.addItem(modality_label(modality), modality)
        index = self.source_picker.findData(selected)
        self.source_picker.setCurrentIndex(index if index >= 0 else 0)
        self.source_picker.blockSignals(False)
        self._adopt_source(str(self.source_picker.currentData() or ""))
        self.set_category(self._category)

    def _source_selected(self, _index: int) -> None:
        self._adopt_source(str(self.source_picker.currentData() or ""))
        self.set_category(self._category)
        self.source_changed.emit(self._source)

    def _adopt_source(self, source: str) -> None:
        if source == self._source:
            return
        self._update_timer.stop()
        self._clip_timer.stop()
        self._source = source
        self._numeric_source = source == "depth_fsd"
        self._roi_token += 1
        self.selection_label.setText("未选择区域或测量点")
        self.reset_tools()
        self.set_ready(False)

    def _range_mode_changed(self, automatic: bool) -> None:
        self.range_min.setEnabled(not automatic)
        self.range_max.setEnabled(not automatic)
        self._schedule_display_update()

    def _schedule_display_update(self) -> None:
        self._update_timer.start()

    def _emit_display_settings(self) -> None:
        display_range = (
            None
            if self.auto_range.isChecked()
            else (self.range_min.value(), self.range_max.value())
        )
        self.display_changed.emit(
            {
                "brightness": self.brightness.value() / 100.0,
                "contrast": self.contrast.value() / 100.0,
                "gamma": self.gamma.value() / 100.0,
                "exposure": self.exposure.value() / 10.0,
                "color_map": str(self.lut.currentData() or "gray"),
                "display_range": display_range,
                "highlight_invalid": self.highlight_invalid.isChecked(),
            }
        )

    def reset_display_controls(self) -> None:
        self._update_timer.stop()
        self.set_display_controls({})
        self._schedule_display_update()

    def set_display_controls(self, settings: dict[str, object]) -> None:
        if self._update_timer.isActive():
            return
        controls = (self.brightness, self.contrast, self.gamma, self.exposure,
                    self.auto_range, self.highlight_invalid, self.lut,
                    self.range_min, self.range_max)
        blockers = [QSignalBlocker(control) for control in controls]
        self.brightness.setValue(round(float(settings.get("brightness", 0)) * 100))
        self.contrast.setValue(round(float(settings.get("contrast", 1)) * 100))
        self.gamma.setValue(round(float(settings.get("gamma", 1)) * 100))
        self.exposure.setValue(round(float(settings.get("exposure", 0)) * 10))
        limits = settings.get("display_range")
        self.auto_range.setChecked(limits is None)
        self.range_min.setEnabled(limits is not None)
        self.range_max.setEnabled(limits is not None)
        if limits is not None:
            self.range_min.setValue(float(limits[0]))
            self.range_max.setValue(float(limits[1]))
        self.highlight_invalid.setChecked(bool(settings.get("highlight_invalid", False)))
        self.lut.setCurrentIndex(self.lut.findData(settings.get("color_map", "gray")))
        for slider in (self.brightness, self.contrast, self.gamma, self.exposure):
            slider.value_label.setText(str(slider.value()))
        del blockers

    def set_image_data(self, modality: str, data: ImageData | None, path: Path | None) -> None:
        self._numeric_source = data is not None and data.values.ndim == 2
        self.depth_controls.setVisible(self._numeric_source)
        if data is None:
            self.histogram.set_histograms(())
            self.stats_label.setText("当前数据尚未加载")
            self._set_file_information(path)
            return
        height, width = data.values.shape[:2]
        self._set_file_information(path, f"{width} × {height} · {data.values.dtype}")
        self._source = modality
        if data.image_stats is None:
            self.histogram.set_histograms(())
            self.stats_label.setText("正在计算统计…")
            return
        self._set_stats(data.image_stats)
        self.histogram.set_histograms(data.image_stats.channel_histograms)

    def _set_file_information(self, path: Path | None, details: str = "") -> None:
        self.file_label.set_file(path, details)
        self.copy_path_button.setEnabled(path is not None)

    def _set_stats(self, stats: ImageStats) -> None:
        if stats.minimum is None:
            self.stats_label.setText("没有有效像素")
            return
        self.stats_label.setText(
            f"最小值  {stats.minimum:g}    最大值  {stats.maximum:g}\n"
            f"均值  {stats.mean:g}    标准差  {stats.standard_deviation:g}\n"
            f"有效像素  {stats.valid_ratio:.2%}    像素数  {stats.pixel_count:,}"
        )

    def set_roi_statistics(self, values: np.ndarray | None) -> None:
        self._roi_token += 1
        if values is None or not values.size:
            self.selection_label.setText("ROI 为空")
            return
        for token, previous in list(self._roi_workers.items()):
            try:
                if ANALYSIS_POOL.tryTake(previous):
                    self._roi_workers.pop(token, None)
            except RuntimeError:
                self._roi_workers.pop(token, None)
        worker = RoiWorker(self._roi_token, values)
        self._roi_workers[self._roi_token] = worker
        worker.signals.loaded.connect(self._roi_ready)
        worker.signals.failed.connect(self._roi_failed)
        self.selection_label.setText("正在计算 ROI…")
        ANALYSIS_POOL.start(worker, 1)

    def _roi_failed(self, token: int, error: str) -> None:
        self._roi_workers.pop(token, None)
        if token == self._roi_token:
            self.selection_label.setText(f"ROI 统计失败：{error}")

    def _roi_ready(self, token: int, stats: ImageStats) -> None:
        self._roi_workers.pop(token, None)
        if token != self._roi_token:
            return
        if stats.minimum is None:
            self.selection_label.setText("ROI 中没有有效像素")
            return
        self.selection_label.setText(
            f"ROI · {stats.pixel_count:,} px\n"
            f"{stats.minimum:g} – {stats.maximum:g} · 均值 {stats.mean:g} · σ {stats.standard_deviation:g}"
        )

    def set_line_measurement(
        self,
        pixels: float,
        physical: float | None = None,
        endpoint_depths: tuple[float, float] | None = None,
        note: str = "",
    ) -> None:
        self._roi_token += 1
        lines = [f"线段  {pixels:.2f} px" + (f" · {format_length(physical)}" if physical is not None else "")]
        if endpoint_depths is not None:
            lines.append(f"端点深度  {endpoint_depths[0]:.3f} m → {endpoint_depths[1]:.3f} m")
        if note:
            lines.append(note)
        self.selection_label.setText("\n".join(lines))

    def set_cloud_data(self, cloud: PointCloudData | None, path: Path | None) -> None:
        if cloud is None or not len(cloud.points):
            self.cloud_stats.setText("当前点云尚未加载" if cloud is None else "当前范围内无点")
            self._set_file_information(path)
            return
        if self._cloud_identity != (path, id(cloud)):
            self._clip_initialized = False
            self._cloud_identity = (path, id(cloud))
            self._cloud_bounds = (cloud.points.min(axis=0), cloud.points.max(axis=0))
        minimum, maximum = self._cloud_bounds
        if not self._clip_initialized:
            for index, axis in enumerate(("x", "y", "z")):
                lower, upper = self.cloud_clip_spins[axis]
                lower.blockSignals(True)
                upper.blockSignals(True)
                lower.setValue(float(minimum[index]))
                upper.setValue(float(maximum[index]))
                lower.blockSignals(False)
                upper.blockSignals(False)
            self._clip_initialized = True
        self.cloud_stats.setText(
            f"显示 {len(cloud.points):,} / {cloud.original_count:,} 点\n"
            f"X {minimum[0]:.3g}…{maximum[0]:.3g}\n"
            f"Y {minimum[1]:.3g}…{maximum[1]:.3g}\n"
            f"Z {minimum[2]:.3g}…{maximum[2]:.3g}"
        )
        self._set_file_information(path)

    def _emit_cloud_clip(self) -> None:
        ranges = {
            axis: (controls[0].value(), controls[1].value())
            for axis, controls in self.cloud_clip_spins.items()
        }
        self.cloud_clip_changed.emit(ranges)

    def set_cloud_controls(self, canvas, ranges: dict | None) -> None:
        controls = [self.cloud_color, self.cloud_size, self.cloud_guides]
        controls += [spin for pair in self.cloud_clip_spins.values() for spin in pair]
        blockers = [QSignalBlocker(control) for control in controls]
        self.cloud_color.setCurrentIndex(self.cloud_color.findData(canvas.color_mode))
        self.cloud_size.setValue(round(canvas.dot_radius * 10))
        self.cloud_size.value_label.setText(str(self.cloud_size.value()))
        self.cloud_guides.setChecked(canvas.guides_visible)
        if ranges is not None and not self._clip_timer.isActive():
            for axis, (lower, upper) in self.cloud_clip_spins.items():
                lower.setValue(ranges[axis][0])
                upper.setValue(ranges[axis][1])
        del blockers

    def set_point_information(
        self,
        point: np.ndarray,
        color: np.ndarray | None = None,
    ) -> None:
        x, y, z = (float(value) for value in point[:3])
        distance = float(np.linalg.norm(point[:3]))
        color_text = ""
        if color is not None and len(color) >= 3:
            rgb = np.clip(np.asarray(color[:3]) * 255.0, 0, 255).astype(int)
            color_text = f" · RGB {rgb[0]}, {rgb[1]}, {rgb[2]}"
        self.selection_label.setText(
            f"点云点 · X {x:.5g}  Y {y:.5g}  Z {z:.5g}\n"
            f"相机距离  {distance:.5g} m{color_text}"
        )

    def set_cloud_measurement(
        self,
        first: np.ndarray,
        second: np.ndarray,
        distance: float,
    ) -> None:
        self.selection_label.setText(
            f"点云两点距离  {distance:.6g} m\n"
            f"A ({first[0]:.4g}, {first[1]:.4g}, {first[2]:.4g})\n"
            f"B ({second[0]:.4g}, {second[1]:.4g}, {second[2]:.4g})"
        )
