from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSettings,
    Qt,
)
from PySide6.QtGui import (
    QColor,
    QKeySequence,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .calibration import CalibrationOption, custom_calibration_option, load_calibration
from .ui_controls import ChoiceButton
from .units import DEPTH_UNIT_CHOICES, normalize_depth_unit

DEFAULT_SHORTCUTS = {
    "previous": "Left",
    "next": "Right",
    "first": "Home",
    "last": "End",
    "playback": "Space",
    "focus": "F",
    "reset": "R",
}

ACTION_NAMES = {
    "previous": "上一组",
    "next": "下一组",
    "first": "第一组",
    "last": "最后一组",
    "playback": "播放 / 暂停",
    "focus": "聚焦视图",
    "reset": "重置视图",
}

RESERVED_SHORTCUTS = (
    "Escape",
    "Ctrl+B",
    "Ctrl+O",
    "Ctrl+,",
    "Tab",
    "F11",
    "Ctrl+Shift+P",
    "1",
    "2",
    "3",
    "4",
    "5",
)
_DIALOG_SHADOW_MARGINS = (20, 15, 20, 24)


def reserved_shortcut_actions(sequences: dict[str, str]) -> list[str]:
    reserved = {
        QKeySequence(value).toString(QKeySequence.SequenceFormat.PortableText)
        for value in RESERVED_SHORTCUTS
    }
    return [action for action, sequence in sequences.items() if sequence in reserved]


class ShadowDialog(QDialog):
    """Frameless modal surface with a painted shadow that does not blur its contents."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsDialog")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)
        self._embedded = False

        self._window_layout = QVBoxLayout(self)
        self._window_layout.setContentsMargins(*_DIALOG_SHADOW_MARGINS)
        self._window_layout.setSpacing(0)
        self.dialog_surface = QFrame()
        self.dialog_surface.setObjectName("dialogSurface")
        self._window_layout.addWidget(self.dialog_surface)
        self.outer = QVBoxLayout(self.dialog_surface)
        self.outer.setContentsMargins(0, 0, 0, 0)
        self.outer.setSpacing(0)

    def set_embedded(self, embedded: bool = True) -> None:
        if not embedded or self._embedded:
            return
        self._embedded = True
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setObjectName("settingsPage")
        self.setProperty("embedded", True)
        self.dialog_surface.setProperty("embedded", True)
        self._window_layout.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(0, 0)
        chrome = [self, self.dialog_surface]
        chrome.extend(
            widget
            for widget in self.findChildren(QWidget)
            if widget.objectName() in {"settingsHeader", "settingsFooter"}
        )
        for widget in chrome:
            widget.setProperty("embedded", True)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            if widget.objectName() == "settingsHeader":
                widget.hide()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._embedded or not self.dialog_surface.isVisible():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        surface_rect = QRectF(self.dialog_surface.geometry())
        for spread, alpha in ((14, 4), (12, 5), (10, 6), (8, 8), (6, 10), (4, 14), (2, 20)):
            shadow_rect = surface_rect.adjusted(-spread, -spread + 3, spread, spread + 5)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, alpha))
            painter.drawRoundedRect(shadow_rect, 14 + spread / 2, 14 + spread / 2)


@dataclass
class AppPreferences:
    theme: str = "dark"
    point_limit: int = 300_000
    cloud_cam_offset: float = 0.05
    cloud_grid: int = 5
    cloud_dot_radius: float = 1.0
    cloud_z_max: float = 10.0
    cloud_tau_rel: float = 0.15
    cloud_occlusion: bool = True
    depth_unit: str = "auto"
    shortcuts: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SHORTCUTS))

    def __post_init__(self) -> None:
        # Ignore removed actions without deleting their persisted settings.
        self.shortcuts = {
            action: self.shortcuts.get(action, default)
            for action, default in DEFAULT_SHORTCUTS.items()
        }
        self.depth_unit = normalize_depth_unit(self.depth_unit)

    @classmethod
    def load(cls, settings: QSettings) -> AppPreferences:
        shortcuts = {
            key: QKeySequence(str(settings.value(f"preferences/shortcuts/{key}", default))).toString(
                QKeySequence.SequenceFormat.PortableText
            )
            for key, default in DEFAULT_SHORTCUTS.items()
        }
        if (
            any(not value for value in shortcuts.values())
            or len(set(shortcuts.values())) != len(shortcuts)
            or reserved_shortcut_actions(shortcuts)
        ):
            shortcuts = dict(DEFAULT_SHORTCUTS)
        theme = str(settings.value("preferences/theme", "dark"))
        if theme not in {"dark", "light"}:
            theme = "dark"
        point_limit = settings.value("preferences/point_limit", 300_000, type=int)
        point_limit = max(50_000, min(1_000_000, point_limit))
        cam_offset = settings.value(
            "preferences/point_cloud/cam_offset",
            0.05,
            type=float,
        )
        defaults_version = settings.value(
            "preferences/point_cloud/defaults_version",
            0,
            type=int,
        )
        grid = settings.value("preferences/point_cloud/grid", 5, type=int)
        dot_radius = settings.value(
            "preferences/point_cloud/dot_radius",
            1.0,
            type=float,
        )
        legacy_z_max = settings.value("preferences/point_max_distance", 10.0, type=float)
        z_max = settings.value(
            "preferences/point_cloud/ply_z_max",
            legacy_z_max,
            type=float,
        )
        tau_rel = settings.value(
            "preferences/point_cloud/tau_rel",
            0.15,
            type=float,
        )
        if not math.isfinite(cam_offset):
            cam_offset = 0.05
        if not math.isfinite(dot_radius):
            dot_radius = 1.0
        if not math.isfinite(z_max):
            z_max = 10.0
        if not math.isfinite(tau_rel):
            tau_rel = 0.15
        if defaults_version < 2:
            # Migrate the former defaults while preserving values that the
            # user had already customized.
            if grid == 6:
                grid = 5
            if dot_radius == 2.0:
                dot_radius = 1.0
            if z_max == 15.0:
                z_max = 10.0
        return cls(
            theme=theme,
            point_limit=point_limit,
            cloud_cam_offset=max(0.0, min(1.0, cam_offset)),
            cloud_grid=max(1, min(32, grid)),
            cloud_dot_radius=max(0.5, min(12.0, dot_radius)),
            cloud_z_max=max(0.1, min(1_000.0, z_max)),
            cloud_tau_rel=max(0.0, min(1.0, tau_rel)),
            cloud_occlusion=settings.value(
                "preferences/point_cloud/occlusion",
                True,
                type=bool,
            ),
            depth_unit=normalize_depth_unit(settings.value("preferences/depth_unit", "auto")),
            shortcuts=shortcuts,
        )

    def save(self, settings: QSettings) -> None:
        settings.setValue("preferences/theme", self.theme)
        settings.setValue("preferences/depth_unit", self.depth_unit)
        settings.setValue("preferences/point_limit", self.point_limit)
        settings.setValue("preferences/point_cloud/cam_offset", self.cloud_cam_offset)
        settings.setValue("preferences/point_cloud/grid", self.cloud_grid)
        settings.setValue("preferences/point_cloud/dot_radius", self.cloud_dot_radius)
        settings.setValue("preferences/point_cloud/ply_z_max", self.cloud_z_max)
        settings.setValue("preferences/point_cloud/tau_rel", self.cloud_tau_rel)
        settings.setValue("preferences/point_cloud/occlusion", self.cloud_occlusion)
        settings.setValue("preferences/point_cloud/defaults_version", 2)
        for key, default in DEFAULT_SHORTCUTS.items():
            settings.setValue(f"preferences/shortcuts/{key}", self.shortcuts.get(key, default))
        settings.sync()


class ToggleSwitch(QAbstractButton):
    """Compact, theme-aware switch used for all boolean settings."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toggleSwitch")
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFixedSize(38, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._knob_progress = 1.0 if checked else 0.0
        self._knob_animation = QPropertyAnimation(self, b"knobProgress", self)
        self._knob_animation.setDuration(140)
        self._knob_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate_toggle)

    def _knob_position(self) -> float:
        return self._knob_progress

    def _set_knob_position(self, value: float) -> None:
        self._knob_progress = value
        self.update()

    knobProgress = Property(float, _knob_position, _set_knob_position)

    def _animate_toggle(self, checked: bool) -> None:
        self._knob_animation.stop()
        self._knob_animation.setStartValue(self._knob_progress)
        self._knob_animation.setEndValue(1.0 if checked else 0.0)
        self._knob_animation.start()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.isChecked():
            track = self.palette().highlight().color()
            knob = self.palette().highlightedText().color()
        else:
            track = self.palette().mid().color()
            knob = self.palette().buttonText().color()
        if not self.isEnabled():
            painter.setOpacity(0.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(0, 0, 38, 22), 11, 11)
        painter.setBrush(knob)
        x = 3 + 16 * self._knob_progress
        painter.drawEllipse(QRectF(x, 3, 16, 16))
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(self.palette().highlight().color(), 1))
            painter.drawRoundedRect(QRectF(0.5, 0.5, 37, 21), 10.5, 10.5)


class SegmentedControl(QFrame):
    def __init__(self, options: list[tuple[str, str]], current: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("segmentedControl")
        self._options = options
        self._buttons: list[QPushButton] = []
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        for index, (label, data) in enumerate(options):
            button = QPushButton(label)
            button.setObjectName("segmentButton")
            button.setCheckable(True)
            button.setProperty("segment", True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumWidth(88)
            self._group.addButton(button, index)
            self._buttons.append(button)
            layout.addWidget(button)
            if data == current:
                button.setChecked(True)
        if self._buttons and not self._group.checkedButton():
            self._buttons[0].setChecked(True)

    def currentData(self) -> str:
        index = self._group.checkedId()
        return self._options[max(index, 0)][1]

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)


class SettingRow(QFrame):
    def __init__(self, title: str, description: str, control: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingRow")
        self.setMaximumWidth(960)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 10, 2, 10)
        layout.setSpacing(16)
        copy = QVBoxLayout()
        copy.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("settingRowTitle")
        heading.setBuddy(control)
        control.setAccessibleName(title)
        copy.addWidget(heading)
        if description:
            detail = QLabel(description)
            detail.setObjectName("settingRowDescription")
            detail.setWordWrap(True)
            copy.addWidget(detail)
        layout.addLayout(copy, 1)
        layout.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)


class DialogCloseButton(QPushButton):
    """Font-independent close icon shared by every secondary surface."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsCloseButton")
        self.setFixedSize(34, 32)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.palette().buttonText().color(), 1.25)
        pen.setCosmetic(True)
        painter.setPen(pen)
        x = self.width() // 2
        y = self.height() // 2
        painter.drawLine(x - 4, y - 4, x + 4, y + 4)
        painter.drawLine(x + 4, y - 4, x - 4, y + 4)


class DialogHeader(QFrame):
    def mousePressEvent(self, event) -> None:
        if self.property("embedded"):
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()
            event.accept()
            return
        super().mousePressEvent(event)


class SettingsDialog(ShadowDialog):
    def __init__(
        self,
        preferences: AppPreferences,
        parent=None,
        *,
        calibration_options: list[CalibrationOption] | None = None,
        current_calibration_id: str = "",
        project_available: bool = False,
    ) -> None:
        super().__init__(parent)
        self.resize(820, 600)
        self.setMinimumSize(720, 540)
        self.preferences = AppPreferences(
            theme=preferences.theme,
            point_limit=preferences.point_limit,
            cloud_cam_offset=preferences.cloud_cam_offset,
            cloud_grid=preferences.cloud_grid,
            cloud_dot_radius=preferences.cloud_dot_radius,
            cloud_z_max=preferences.cloud_z_max,
            cloud_tau_rel=preferences.cloud_tau_rel,
            cloud_occlusion=preferences.cloud_occlusion,
            depth_unit=preferences.depth_unit,
            shortcuts=dict(preferences.shortcuts),
        )
        self.calibration_options = list(calibration_options or [])
        self.current_calibration_id = current_calibration_id
        self.selected_calibration_id = current_calibration_id if project_available else ""
        self.project_available = project_available

        outer = self.outer
        self.header = DialogHeader()
        self.header.setObjectName("settingsHeader")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(14, 0, 5, 0)
        title = QLabel("设置")
        title.setObjectName("settingsTitle")
        self.close_button = DialogCloseButton()
        self.close_button.clicked.connect(self.reject)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.close_button)
        outer.addWidget(self.header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.navigation = QListWidget()
        self.navigation.setObjectName("settingsNav")
        self.navigation.setFixedWidth(176)
        self.navigation.addItems(["外观", "标定", "快捷键", "点云"])
        self.pages = QStackedWidget()
        self.pages.setObjectName("settingsPages")
        self.pages.addWidget(self._appearance_page())
        self.pages.addWidget(self._calibration_page())
        self.pages.addWidget(self._shortcuts_page())
        self.pages.addWidget(self._performance_page())
        self.navigation.currentRowChanged.connect(self._switch_page)
        self.navigation.setCurrentRow(0)
        body.addWidget(self.navigation)
        body.addWidget(self.pages, 1)
        outer.addLayout(body, 1)

        self.footer = QFrame()
        self.footer.setObjectName("settingsFooter")
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(12, 8, 12, 8)
        reset = QPushButton("恢复默认")
        reset.setObjectName("ghostButton")
        reset.clicked.connect(self._reset_defaults)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        save = QPushButton("保存设置")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save_and_accept)
        footer_layout.addWidget(reset)
        footer_layout.addStretch(1)
        footer_layout.addWidget(cancel)
        footer_layout.addWidget(save)
        outer.addWidget(self.footer)

    def _page(self, title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("settingsPageTitle")
        layout.addWidget(heading)
        if description:
            subtitle = QLabel(description)
            subtitle.setObjectName("settingsDescription")
            subtitle.setWordWrap(True)
            layout.addWidget(subtitle)
        layout.addSpacing(8)
        return page, layout

    def _appearance_page(self) -> QWidget:
        page, layout = self._page("外观", "")
        self.theme_combo = SegmentedControl([("深色", "dark"), ("浅色", "light")], self.preferences.theme)
        layout.addWidget(SettingRow("界面主题", "", self.theme_combo))
        layout.addStretch(1)
        return page

    def _calibration_page(self) -> QWidget:
        page, layout = self._page("相机标定", "管理内置标定或导入自定义 JSON / YAML 文件。")
        self.calibration_combo = ChoiceButton()
        self.calibration_combo.setObjectName("settingsCombo")
        self.calibration_combo.setMinimumWidth(230)
        self.calibration_combo.addItem("不使用标定", "")
        for option in self.calibration_options:
            suffix = "内置" if option.builtin else "项目内" if option.project_local else "自定义"
            self.calibration_combo.addItem(f"{option.label}  ·  {suffix}", option.id)
        selected_index = self.calibration_combo.findData(self.current_calibration_id)
        self.calibration_combo.setCurrentIndex(max(0, selected_index))
        self.calibration_combo.currentIndexChanged.connect(self._calibration_changed)
        layout.addWidget(
            SettingRow(
                "当前项目",
                "主页面也可以快速切换已添加的标定。",
                self.calibration_combo,
            )
        )

        import_button = QPushButton("导入标定文件…")
        import_button.setObjectName("secondaryButton")
        import_button.clicked.connect(self._import_calibration)
        layout.addWidget(
            SettingRow(
                "自定义标定",
                "导入后会加入标定列表，不会修改原始文件。",
                import_button,
            )
        )
        self.calibration_detail = QLabel("")
        self.calibration_detail.setObjectName("calibrationDetail")
        self.calibration_detail.setWordWrap(True)
        self.calibration_detail.setMaximumWidth(960)
        layout.addWidget(self.calibration_detail)

        self.depth_unit_combo = ChoiceButton()
        self.depth_unit_combo.setObjectName("settingsCombo")
        self.depth_unit_combo.setMinimumWidth(230)
        for unit, label in DEPTH_UNIT_CHOICES:
            self.depth_unit_combo.addItem(label, unit)
        self.depth_unit_combo.setCurrentIndex(
            max(0, self.depth_unit_combo.findData(self.preferences.depth_unit))
        )
        layout.addWidget(
            SettingRow(
                "深度单位",
                "决定光标 XYZ、线段长度和点云外参如何换算为米。",
                self.depth_unit_combo,
            )
        )
        if not self.project_available:
            no_project = QLabel("当前未打开项目；可以先导入文件，打开项目后再选择。")
            no_project.setObjectName("settingsDescription")
            no_project.setWordWrap(True)
            no_project.setMaximumWidth(960)
            layout.addWidget(no_project)
            self.calibration_combo.setEnabled(False)
        layout.addStretch(1)
        self._calibration_changed()
        return page

    def _switch_page(self, index: int) -> None:
        # Pages contain live controls; opacity effects recreate their backing
        # surfaces and race when a user changes pages quickly.
        self.pages.setCurrentIndex(index)

    def _calibration_changed(self) -> None:
        option_id = str(self.calibration_combo.currentData() or "")
        option = next((item for item in self.calibration_options if item.id == option_id), None)
        if option is None:
            self.calibration_detail.setText("未选择标定。三维坐标和精确极线将不可用。")
            return
        try:
            calibration = load_calibration(option.path)
            source = "内置预设" if option.builtin else str(option.path)
            self.calibration_detail.setText(f"{calibration.details}\n{source}")
        except ValueError as exc:
            self.calibration_detail.setText(str(exc))

    def _import_calibration(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "导入相机标定",
            str(Path.home()),
            "标定文件 (*.json *.yaml *.yml);;所有文件 (*)",
        )
        if not selected:
            return
        path = Path(selected)
        try:
            calibration = load_calibration(path)
        except Exception as exc:
            QMessageBox.critical(self, "无法导入标定", str(exc))
            return
        option = custom_calibration_option(path)
        existing = next((item for item in self.calibration_options if item.id == option.id), None)
        if existing is None:
            self.calibration_options.append(option)
            self.calibration_combo.addItem(f"{option.label}  ·  自定义", option.id)
        self.calibration_combo.setCurrentIndex(self.calibration_combo.findData(option.id))
        self.calibration_detail.setText(f"{calibration.details}\n{option.path}")

    def _shortcuts_page(self) -> QWidget:
        content, layout = self._page("快捷键", "点击输入框后按下新的按键组合。")
        content.setObjectName("settingsScrollBody")
        self.shortcut_edits: dict[str, QKeySequenceEdit] = {}
        for action in DEFAULT_SHORTCUTS:
            edit = QKeySequenceEdit(QKeySequence(self.preferences.shortcuts[action]))
            edit.setMaximumSequenceLength(1)
            edit.setMinimumWidth(170)
            self.shortcut_edits[action] = edit
            layout.addWidget(SettingRow(ACTION_NAMES[action], "", edit))
        layout.addStretch(1)
        return self._scroll_page(content)

    @staticmethod
    def _scroll_page(content: QWidget) -> QScrollArea:
        content.setObjectName("settingsScrollBody")
        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _performance_page(self) -> QWidget:
        page, layout = self._page("点云", "渲染参数与显示性能。")
        self.point_limit_spin = QSpinBox()
        self.point_limit_spin.setRange(50_000, 1_000_000)
        self.point_limit_spin.setSingleStep(50_000)
        self.point_limit_spin.setValue(self.preferences.point_limit)
        self.point_limit_spin.setSuffix(" 点")
        self.point_limit_spin.setMinimumWidth(160)
        layout.addWidget(SettingRow("点云显示上限", "较低数值可提升加载和交互速度。", self.point_limit_spin))
        self.cloud_cam_offset_spin = QDoubleSpinBox()
        self.cloud_cam_offset_spin.setRange(0.0, 1.0)
        self.cloud_cam_offset_spin.setDecimals(2)
        self.cloud_cam_offset_spin.setSingleStep(0.01)
        self.cloud_cam_offset_spin.setValue(self.preferences.cloud_cam_offset)
        self.cloud_cam_offset_spin.setSuffix(" m")
        self.cloud_cam_offset_spin.setMinimumWidth(160)
        layout.addWidget(
            SettingRow(
                "虚拟相机后移",
                "cam_offset；增大可缩小画面并扩大初始视野。",
                self.cloud_cam_offset_spin,
            )
        )
        self.cloud_grid_spin = QSpinBox()
        self.cloud_grid_spin.setRange(1, 32)
        self.cloud_grid_spin.setValue(self.preferences.cloud_grid)
        self.cloud_grid_spin.setSuffix(" px")
        self.cloud_grid_spin.setMinimumWidth(160)
        layout.addWidget(
            SettingRow(
                "采样网格",
                "grid；数值越大点越稀疏，渲染越快。",
                self.cloud_grid_spin,
            )
        )
        self.cloud_dot_radius_spin = QDoubleSpinBox()
        self.cloud_dot_radius_spin.setRange(0.5, 12.0)
        self.cloud_dot_radius_spin.setDecimals(1)
        self.cloud_dot_radius_spin.setSingleStep(0.5)
        self.cloud_dot_radius_spin.setValue(self.preferences.cloud_dot_radius)
        self.cloud_dot_radius_spin.setSuffix(" px")
        self.cloud_dot_radius_spin.setMinimumWidth(160)
        layout.addWidget(
            SettingRow(
                "点半径",
                "dot_radius；仅改变圆点粗细，不改变几何缩放。",
                self.cloud_dot_radius_spin,
            )
        )
        self.cloud_z_max_spin = QDoubleSpinBox()
        self.cloud_z_max_spin.setRange(0.1, 1_000.0)
        self.cloud_z_max_spin.setDecimals(1)
        self.cloud_z_max_spin.setSingleStep(1.0)
        self.cloud_z_max_spin.setValue(self.preferences.cloud_z_max)
        self.cloud_z_max_spin.setSuffix(" m")
        self.cloud_z_max_spin.setMinimumWidth(160)
        layout.addWidget(
            SettingRow(
                "最远显示距离",
                "ply_z_max；只过滤可见距离，不控制缩放。",
                self.cloud_z_max_spin,
            )
        )
        self.cloud_tau_rel_spin = QDoubleSpinBox()
        self.cloud_tau_rel_spin.setRange(0.0, 1.0)
        self.cloud_tau_rel_spin.setDecimals(2)
        self.cloud_tau_rel_spin.setSingleStep(0.01)
        self.cloud_tau_rel_spin.setValue(self.preferences.cloud_tau_rel)
        self.cloud_tau_rel_spin.setMinimumWidth(160)
        layout.addWidget(
            SettingRow(
                "飞点过滤强度",
                "tau_rel；增大时过滤更严格，可能删除更多边缘点。",
                self.cloud_tau_rel_spin,
            )
        )
        self.cloud_occlusion_check = ToggleSwitch(self.preferences.cloud_occlusion)
        layout.addWidget(
            SettingRow(
                "近点遮挡远点",
                "同一投影网格只保留最近点，并按从远到近绘制。",
                self.cloud_occlusion_check,
            )
        )
        layout.addStretch(1)
        return self._scroll_page(page)

    def _reset_defaults(self) -> None:
        self.theme_combo.setCurrentIndex(0)
        self.point_limit_spin.setValue(300_000)
        self.cloud_cam_offset_spin.setValue(0.05)
        self.cloud_grid_spin.setValue(5)
        self.cloud_dot_radius_spin.setValue(1.0)
        self.cloud_z_max_spin.setValue(10.0)
        self.cloud_tau_rel_spin.setValue(0.15)
        self.cloud_occlusion_check.setChecked(True)
        self.depth_unit_combo.setCurrentIndex(0)
        if self.project_available:
            self.calibration_combo.setCurrentIndex(0)
        for key, edit in self.shortcut_edits.items():
            edit.setKeySequence(QKeySequence(DEFAULT_SHORTCUTS[key]))

    def _save_and_accept(self) -> None:
        sequences = {
            key: edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            for key, edit in self.shortcut_edits.items()
        }
        if any(not value for value in sequences.values()):
            QMessageBox.warning(self, "快捷键不能为空", "请为每个操作设置一个快捷键。")
            return
        if len(set(sequences.values())) != len(sequences):
            QMessageBox.warning(self, "快捷键重复", "同一个快捷键不能分配给多个操作。")
            return
        conflicts = reserved_shortcut_actions(sequences)
        if conflicts:
            names = "、".join(ACTION_NAMES[action] for action in conflicts)
            QMessageBox.warning(
                self,
                "快捷键已被系统操作占用",
                f"以下操作使用了保留快捷键：{names}\n\n请改用其他按键组合。",
            )
            return
        self.preferences.theme = self.theme_combo.currentData()
        self.preferences.point_limit = self.point_limit_spin.value()
        self.preferences.cloud_cam_offset = self.cloud_cam_offset_spin.value()
        self.preferences.cloud_grid = self.cloud_grid_spin.value()
        self.preferences.cloud_dot_radius = self.cloud_dot_radius_spin.value()
        self.preferences.cloud_z_max = self.cloud_z_max_spin.value()
        self.preferences.cloud_tau_rel = self.cloud_tau_rel_spin.value()
        self.preferences.cloud_occlusion = self.cloud_occlusion_check.isChecked()
        self.preferences.depth_unit = normalize_depth_unit(self.depth_unit_combo.currentData())
        self.preferences.shortcuts = sequences
        self.selected_calibration_id = (
            str(self.calibration_combo.currentData() or "")
            if self.project_available
            else ""
        )
        self.accept()
