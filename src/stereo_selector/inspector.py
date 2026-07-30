from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QLineF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QButtonGroup,
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

from .media import ImageData, ImageStats, PointCloudData, analyze_image
from .models import modality_label
from .settings import ChoiceButton, DialogCloseButton


class HistogramWidget(QWidget):
    """Small theme-aware histogram suitable for the compact inspector."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("histogram")
        self.setMinimumHeight(128)
        self._histograms: tuple[np.ndarray, ...] = ()

    def set_histograms(self, histograms: tuple[np.ndarray, ...]) -> None:
        self._histograms = tuple(np.asarray(item, dtype=np.float64) for item in histograms)
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
            painter.setPen(QPen(colors[channel] if len(self._histograms) > 1 else QColor("#4c9ffe"), 1.2))
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
    stereo_action_requested = Signal(str)
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
        self.setMinimumWidth(268)
        self.setMaximumWidth(380)
        self._source = ""
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
        title = QLabel("检查器")
        title.setObjectName("inspectorTitle")
        close = DialogCloseButton()
        close.setToolTip("关闭检查器")
        close.clicked.connect(self.close_requested)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(close)
        outer.addWidget(header)

        scroll = QScrollArea()
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
        layout.addWidget(self._section("数据源", self.source_picker))

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
        self.lut = ChoiceButton()
        self.lut.setObjectName("inspectorChoice")
        for text, value in (("灰度", "gray"), ("Turbo", "turbo"), ("Viridis", "viridis")):
            self.lut.addItem(text, value)
        self.lut.currentIndexChanged.connect(lambda _index: self._schedule_display_update())
        display_layout.addWidget(self._labeled("深度 LUT", self.lut))
        self.auto_range = QCheckBox("自动百分位拉伸")
        self.auto_range.setChecked(True)
        self.auto_range.toggled.connect(self._range_mode_changed)
        display_layout.addWidget(self.auto_range)
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
        display_layout.addLayout(ranges)
        self.highlight_invalid = QCheckBox("高亮 0 / 65535 / 无效 / 越界")
        self.highlight_invalid.toggled.connect(lambda _checked: self._schedule_display_update())
        display_layout.addWidget(self.highlight_invalid)
        self._range_mode_changed(True)
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
        for label, tool in (("平移", "pan"), ("矩形 ROI", "roi"), ("线段", "line")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(tool == "pan")
            self.image_tool_group.addButton(button)
            button.clicked.connect(lambda _checked=False, value=tool: self.tool_requested.emit(value))
            tool_row.addWidget(button)
        view_layout.addLayout(tool_row)
        self.selection_label = QLabel("ROI 与测量结果显示在这里")
        self.selection_label.setObjectName("inspectorStats")
        self.selection_label.setWordWrap(True)
        view_layout.addWidget(self.selection_label)
        export = QPushButton("导出当前视图…")
        export.clicked.connect(self.export_requested)
        view_layout.addWidget(export)
        layout.addWidget(view_tools)

        stereo = QFrame()
        stereo.setObjectName("inspectorSection")
        stereo_layout = QVBoxLayout(stereo)
        stereo_layout.setContentsMargins(12, 10, 12, 12)
        stereo_layout.setSpacing(6)
        stereo_layout.addWidget(self._heading("双目检查"))
        stereo_row = QHBoxLayout()
        for label, action in (
            ("透明叠加", "overlay"),
            ("绝对差值", "difference"),
            ("点云投影", "cloud_projection"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=action: self.stereo_action_requested.emit(value)
            )
            stereo_row.addWidget(button)
        stereo_layout.addLayout(stereo_row)
        layout.addWidget(stereo)

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
        cloud_tools = QHBoxLayout()
        for label, tool in (
            ("旋转", "rotate"),
            ("点选", "pick"),
            ("两点测距", "measure"),
            ("框选", "box"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=tool: self.cloud_tool_requested.emit(value)
            )
            cloud_tools.addWidget(button)
        cloud_layout.addLayout(cloud_tools)
        reset_cloud = QPushButton("恢复全部点")
        reset_cloud.clicked.connect(
            lambda: self.cloud_tool_requested.emit("reset")
        )
        cloud_layout.addWidget(reset_cloud)
        cloud_views = QHBoxLayout()
        for label, view in (("前", "front"), ("后", "back"), ("左", "left"), ("右", "right"), ("俯", "top")):
            button = QPushButton(label)
            button.setToolTip(f"{label}视图")
            button.clicked.connect(
                lambda _checked=False, value=view: self.cloud_view_requested.emit(value)
            )
            cloud_views.addWidget(button)
        cloud_layout.addLayout(cloud_views)
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
            cloud_layout.addLayout(row)
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
        cloud_layout.addWidget(self.cloud_stats)
        layout.addWidget(cloud)

        self.file_label = QLabel("")
        self.file_label.setObjectName("inspectorFile")
        self.file_label.setWordWrap(True)
        self.file_label.setMinimumWidth(0)
        self.file_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(self._section("文件", self.file_label))
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

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
        row = QHBoxLayout()
        name = QLabel(label)
        name.setObjectName("muted")
        row.addWidget(name)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        layout.addLayout(row)
        slider.valueChanged.connect(lambda number, target=value_label: target.setText(str(number)))
        slider.valueChanged.connect(lambda _number: self._schedule_display_update())
        return slider

    def set_sources(self, modalities: list[str], selected: str = "") -> None:
        self._clip_initialized = False
        self.source_picker.blockSignals(True)
        self.source_picker.clear()
        for modality in modalities:
            self.source_picker.addItem(modality_label(modality), modality)
        index = self.source_picker.findData(selected)
        self.source_picker.setCurrentIndex(index if index >= 0 else 0)
        self.source_picker.blockSignals(False)
        self._source = str(self.source_picker.currentData() or "")

    def _source_selected(self, _index: int) -> None:
        self._source = str(self.source_picker.currentData() or "")
        self.source_changed.emit(self._source)

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
        for control, value in (
            (self.brightness, 0),
            (self.contrast, 100),
            (self.gamma, 100),
            (self.exposure, 0),
        ):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self.auto_range.setChecked(True)
        self.highlight_invalid.setChecked(False)
        self.lut.setCurrentIndex(0)

    def set_image_data(self, modality: str, data: ImageData | None, path: Path | None) -> None:
        if data is None:
            self.histogram.set_histograms(())
            self.stats_label.setText("当前数据尚未加载")
            self.file_label.setText(str(path or ""))
            return
        if data.image_stats is None:
            self.histogram.set_histograms(())
            self.stats_label.setText("正在计算统计…")
            self.file_label.setText(str(path or ""))
            return
        stats = data.image_stats
        self._set_stats(stats)
        self.histogram.set_histograms(stats.channel_histograms)
        self.file_label.setText(
            f"{path}\n{data.image.width()} × {data.image.height()} · {data.values.dtype}"
            if path is not None
            else f"{data.image.width()} × {data.image.height()} · {data.values.dtype}"
        )
        self._source = modality

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
        if values is None or not values.size:
            self.selection_label.setText("ROI 为空")
            return
        stats = analyze_image(values)
        if stats.minimum is None:
            self.selection_label.setText("ROI 中没有有效像素")
            return
        self.selection_label.setText(
            f"ROI · {stats.pixel_count:,} px\n"
            f"{stats.minimum:g} – {stats.maximum:g} · 均值 {stats.mean:g} · σ {stats.standard_deviation:g}"
        )

    def set_line_measurement(self, pixels: float, physical: float | None = None) -> None:
        suffix = f" · {physical:.4g} m" if physical is not None else ""
        self.selection_label.setText(f"线段  {pixels:.2f} px{suffix}")

    def set_cloud_data(self, cloud: PointCloudData | None, path: Path | None) -> None:
        if cloud is None or not len(cloud.points):
            self.cloud_stats.setText("当前点云尚未加载")
            return
        minimum = cloud.points.min(axis=0)
        maximum = cloud.points.max(axis=0)
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
        self.file_label.setText(str(path or ""))

    def _emit_cloud_clip(self) -> None:
        ranges = {
            axis: (controls[0].value(), controls[1].value())
            for axis, controls in self.cloud_clip_spins.items()
        }
        self.cloud_clip_changed.emit(ranges)

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
