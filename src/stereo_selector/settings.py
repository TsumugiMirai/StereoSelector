from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    Property,
    QPropertyAnimation,
    QSettings,
    Signal,
    Qt,
    QRectF,
)
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .calibration import CalibrationOption, custom_calibration_option, load_calibration

DEFAULT_LABELS = {
    "previous": "上一组",
    "next": "下一组",
    "accept": "接受",
}

LEGACY_DEFAULT_LABELS = {
    "previous": "←  上一组",
    "next": "跳过 / 下一组  →",
    "accept": "接受并继续",
}

DEFAULT_SHORTCUTS = {
    "previous": "Left",
    "next": "Right",
    "accept": "Return",
    "focus": "F",
    "reset": "R",
}

ACTION_NAMES = {
    "previous": "上一组",
    "next": "下一组",
    "accept": "接受当前组",
    "focus": "聚焦视图",
    "reset": "重置视图",
}

RESERVED_SHORTCUTS = ("Home", "End", "Escape", "Ctrl+B", "Ctrl+O", "N", "Ctrl+,", "1", "2", "3", "4", "5")
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
    auto_advance: bool = True
    point_limit: int = 300_000
    button_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABELS))
    shortcuts: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SHORTCUTS))

    @classmethod
    def load(cls, settings: QSettings) -> "AppPreferences":
        labels = {}
        for key, default in DEFAULT_LABELS.items():
            value = str(settings.value(f"preferences/buttons/{key}", default)).strip()
            labels[key] = default if not value or value == LEGACY_DEFAULT_LABELS[key] else value
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
        return cls(
            theme=theme,
            auto_advance=settings.value("preferences/auto_advance", True, type=bool),
            point_limit=point_limit,
            button_labels=labels,
            shortcuts=shortcuts,
        )

    def save(self, settings: QSettings) -> None:
        settings.setValue("preferences/theme", self.theme)
        settings.setValue("preferences/auto_advance", self.auto_advance)
        settings.setValue("preferences/point_limit", self.point_limit)
        for key, value in self.button_labels.items():
            settings.setValue(f"preferences/buttons/{key}", value)
        for key, value in self.shortcuts.items():
            settings.setValue(f"preferences/shortcuts/{key}", value)
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
            track = QColor("#4c9ffe")
            knob = QColor("#ffffff")
        else:
            track = self.palette().mid().color()
            knob = self.palette().buttonText().color()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(0, 0, 38, 22), 11, 11)
        painter.setBrush(knob)
        x = 3 + 16 * self._knob_progress
        painter.drawEllipse(QRectF(x, 3, 16, 16))


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


class ChoiceButton(QPushButton):
    """A theme-owned select control that never opens a native Windows combo popup."""

    currentIndexChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("choiceButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(120)
        self._items: list[tuple[str, object, QAction]] = []
        self._current_index = -1
        self._menu = QMenu(self)
        self._menu.setObjectName("choiceMenu")
        self._menu.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
        self.clicked.connect(self.showPopup)

    def addItem(self, text: str, data: object = None) -> None:
        index = len(self._items)
        action = QAction(text, self._menu)
        action.setCheckable(True)
        action.triggered.connect(lambda _checked=False, item_index=index: self.setCurrentIndex(item_index))
        self._menu.addAction(action)
        self._items.append((text, data, action))
        if self._current_index < 0:
            self.setCurrentIndex(0)

    def clear(self) -> None:
        self._menu.clear()
        self._items.clear()
        self._current_index = -1
        self.setText("")

    def count(self) -> int:
        return len(self._items)

    def itemText(self, index: int) -> str:
        return self._items[index][0] if 0 <= index < len(self._items) else ""

    def itemData(self, index: int) -> object:
        return self._items[index][1] if 0 <= index < len(self._items) else None

    def currentData(self) -> object:
        return self.itemData(self._current_index)

    def currentText(self) -> str:
        return self.itemText(self._current_index)

    def currentIndex(self) -> int:
        return self._current_index

    def findData(self, data: object) -> int:
        return next((index for index, (_, value, _) in enumerate(self._items) if value == data), -1)

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items) or index == self._current_index:
            return
        self._current_index = index
        text, _, _ = self._items[index]
        self.setText(text)
        for item_index, (_, _, action) in enumerate(self._items):
            action.setChecked(item_index == index)
        self.currentIndexChanged.emit(index)
        self.update()

    def showPopup(self) -> None:
        if not self.isEnabled() or not self._items:
            return
        self._menu.setMinimumWidth(self.width())
        self._menu.popup(self.mapToGlobal(QPoint(0, self.height() + 4)))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.palette().buttonText().color(), 1.25)
        pen.setCosmetic(True)
        painter.setPen(pen)
        x = self.width() - 15
        y = self.height() // 2
        painter.drawLine(x - 3, y - 2, x, y + 1)
        painter.drawLine(x, y + 1, x + 3, y - 2)


class SettingRow(QFrame):
    def __init__(self, title: str, description: str, control: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingRow")
        self.setMaximumWidth(960)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 11, 12, 11)
        layout.setSpacing(18)
        copy = QVBoxLayout()
        copy.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("settingRowTitle")
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
            auto_advance=preferences.auto_advance,
            point_limit=preferences.point_limit,
            button_labels=dict(preferences.button_labels),
            shortcuts=dict(preferences.shortcuts),
        )
        self.calibration_options = list(calibration_options or [])
        self.current_calibration_id = current_calibration_id
        self.selected_calibration_id = current_calibration_id if project_available else ""
        self.project_available = project_available
        self._page_animation: QPropertyAnimation | None = None

        outer = self.outer
        self.header = DialogHeader()
        self.header.setObjectName("settingsHeader")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(18, 0, 6, 0)
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
        self.navigation.setFixedWidth(184)
        self.navigation.addItems(["外观", "交互", "标定", "按钮文字", "快捷键", "性能"])
        self.pages = QStackedWidget()
        self.pages.setObjectName("settingsPages")
        self.pages.addWidget(self._appearance_page())
        self.pages.addWidget(self._interaction_page())
        self.pages.addWidget(self._calibration_page())
        self.pages.addWidget(self._labels_page())
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
        footer_layout.setContentsMargins(14, 10, 14, 10)
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
        layout.setContentsMargins(30, 26, 30, 26)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("settingsPageTitle")
        layout.addWidget(heading)
        if description:
            subtitle = QLabel(description)
            subtitle.setObjectName("settingsDescription")
            subtitle.setWordWrap(True)
            layout.addWidget(subtitle)
        layout.addSpacing(12)
        return page, layout

    def _appearance_page(self) -> QWidget:
        page, layout = self._page("外观", "")
        self.theme_combo = SegmentedControl([("深色", "dark"), ("浅色", "light")], self.preferences.theme)
        layout.addWidget(SettingRow("界面主题", "", self.theme_combo))
        layout.addStretch(1)
        return page

    def _interaction_page(self) -> QWidget:
        page, layout = self._page("交互", "")
        self.auto_advance_check = ToggleSwitch(self.preferences.auto_advance)
        layout.addWidget(SettingRow("接受后自动前进", "复制完成后直接显示下一组样本。", self.auto_advance_check))
        layout.addStretch(1)
        return page

    def _calibration_page(self) -> QWidget:
        page, layout = self._page("相机标定", "管理内置标定或导入自定义 JSON / YAML 文件。")
        self.calibration_combo = ChoiceButton()
        self.calibration_combo.setObjectName("settingsCombo")
        self.calibration_combo.setMinimumWidth(230)
        self.calibration_combo.addItem("不使用标定", "")
        for option in self.calibration_options:
            suffix = "内置" if option.builtin else "自定义"
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
        self.pages.setCurrentIndex(index)
        page = self.pages.currentWidget()
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", page)
        animation.setDuration(150)
        animation.setStartValue(0.35)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda: page.setGraphicsEffect(None))
        self._page_animation = animation
        animation.start()

    def _calibration_changed(self) -> None:
        option_id = str(self.calibration_combo.currentData() or "")
        option = next((item for item in self.calibration_options if item.id == option_id), None)
        if option is None:
            self.calibration_detail.setText("未选择标定。三维坐标和精确极线将不可用。")
            return
        try:
            calibration = load_calibration(option.path)
            source = "内置预设" if option.builtin else str(option.path)
            self.calibration_detail.setText(f"{calibration.summary}\n{source}")
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
        self.calibration_detail.setText(f"{calibration.summary}\n{option.path}")

    def _labels_page(self) -> QWidget:
        page, layout = self._page("按钮文字", "显示在底部审阅栏中。")
        self.label_edits: dict[str, QLineEdit] = {}
        for action in ("previous", "next", "accept"):
            edit = QLineEdit(self.preferences.button_labels[action])
            edit.setMinimumWidth(230)
            self.label_edits[action] = edit
            layout.addWidget(SettingRow(ACTION_NAMES[action], "", edit))
        layout.addStretch(1)
        return page

    def _shortcuts_page(self) -> QWidget:
        page, layout = self._page("快捷键", "点击输入框后按下新的按键组合。")
        self.shortcut_edits: dict[str, QKeySequenceEdit] = {}
        for action in DEFAULT_SHORTCUTS:
            edit = QKeySequenceEdit(QKeySequence(self.preferences.shortcuts[action]))
            edit.setMaximumSequenceLength(1)
            edit.setMinimumWidth(170)
            self.shortcut_edits[action] = edit
            layout.addWidget(SettingRow(ACTION_NAMES[action], "", edit))
        layout.addStretch(1)
        return page

    def _performance_page(self) -> QWidget:
        page, layout = self._page("性能", "")
        self.point_limit_spin = QSpinBox()
        self.point_limit_spin.setRange(50_000, 1_000_000)
        self.point_limit_spin.setSingleStep(50_000)
        self.point_limit_spin.setValue(self.preferences.point_limit)
        self.point_limit_spin.setSuffix(" 点")
        self.point_limit_spin.setMinimumWidth(160)
        layout.addWidget(SettingRow("点云显示上限", "较低数值可提升加载和交互速度。", self.point_limit_spin))
        layout.addStretch(1)
        return page

    def _reset_defaults(self) -> None:
        self.theme_combo.setCurrentIndex(0)
        self.auto_advance_check.setChecked(True)
        self.point_limit_spin.setValue(300_000)
        if self.project_available:
            self.calibration_combo.setCurrentIndex(0)
        for key, edit in self.label_edits.items():
            edit.setText(DEFAULT_LABELS[key])
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
        labels = {key: edit.text().strip() for key, edit in self.label_edits.items()}
        if any(not value for value in labels.values()):
            QMessageBox.warning(self, "按钮文字不能为空", "请填写所有按钮文字。")
            return
        self.preferences.theme = self.theme_combo.currentData()
        self.preferences.auto_advance = self.auto_advance_check.isChecked()
        self.preferences.point_limit = self.point_limit_spin.value()
        self.preferences.button_labels = labels
        self.preferences.shortcuts = sequences
        self.selected_calibration_id = (
            str(self.calibration_combo.currentData() or "")
            if self.project_available
            else ""
        )
        self.accept()
