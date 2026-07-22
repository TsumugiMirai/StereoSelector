from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QSettings, Qt, QRectF
from PySide6.QtGui import QColor, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


DEFAULT_LABELS = {
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


def reserved_shortcut_actions(sequences: dict[str, str]) -> list[str]:
    reserved = {
        QKeySequence(value).toString(QKeySequence.SequenceFormat.PortableText)
        for value in RESERVED_SHORTCUTS
    }
    return [action for action, sequence in sequences.items() if sequence in reserved]


@dataclass
class AppPreferences:
    theme: str = "dark"
    auto_advance: bool = True
    point_limit: int = 300_000
    button_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABELS))
    shortcuts: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SHORTCUTS))

    @classmethod
    def load(cls, settings: QSettings) -> "AppPreferences":
        labels = {
            key: (str(settings.value(f"preferences/buttons/{key}", default)).strip() or default)
            for key, default in DEFAULT_LABELS.items()
        }
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
        x = 19 if self.isChecked() else 3
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


class SettingRow(QFrame):
    def __init__(self, title: str, description: str, control: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settingRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 11, 12, 11)
        layout.setSpacing(18)
        copy = QVBoxLayout()
        copy.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("settingRowTitle")
        detail = QLabel(description)
        detail.setObjectName("settingRowDescription")
        detail.setWordWrap(True)
        copy.addWidget(heading)
        copy.addWidget(detail)
        layout.addLayout(copy, 1)
        layout.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)


class DialogHeader(QFrame):
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()
            event.accept()
            return
        super().mousePressEvent(event)


class SettingsDialog(QDialog):
    def __init__(self, preferences: AppPreferences, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsDialog")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.resize(820, 600)
        self.setMinimumSize(720, 540)
        self.preferences = AppPreferences(
            theme=preferences.theme,
            auto_advance=preferences.auto_advance,
            point_limit=preferences.point_limit,
            button_labels=dict(preferences.button_labels),
            shortcuts=dict(preferences.shortcuts),
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        header = DialogHeader()
        header.setObjectName("settingsHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 0, 6, 0)
        title = QLabel("设置")
        title.setObjectName("settingsTitle")
        close = QPushButton("×")
        close.setObjectName("settingsCloseButton")
        close.setFixedSize(42, 40)
        close.clicked.connect(self.reject)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(close)
        outer.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.navigation = QListWidget()
        self.navigation.setObjectName("settingsNav")
        self.navigation.setFixedWidth(184)
        self.navigation.addItems(["外观", "交互", "按钮文字", "快捷键", "性能"])
        self.pages = QStackedWidget()
        self.pages.setObjectName("settingsPages")
        self.pages.addWidget(self._appearance_page())
        self.pages.addWidget(self._interaction_page())
        self.pages.addWidget(self._labels_page())
        self.pages.addWidget(self._shortcuts_page())
        self.pages.addWidget(self._performance_page())
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        body.addWidget(self.navigation)
        body.addWidget(self.pages, 1)
        outer.addLayout(body, 1)

        footer = QFrame()
        footer.setObjectName("settingsFooter")
        footer_layout = QHBoxLayout(footer)
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
        outer.addWidget(footer)

    def _page(self, title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 26, 30, 26)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("settingsPageTitle")
        subtitle = QLabel(description)
        subtitle.setObjectName("settingsDescription")
        subtitle.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        return page, layout

    def _appearance_page(self) -> QWidget:
        page, layout = self._page("外观", "保存后统一切换窗口颜色；媒体画布保持中性深色以便比对。")
        self.theme_combo = SegmentedControl([("深色", "dark"), ("浅色", "light")], self.preferences.theme)
        layout.addWidget(SettingRow("界面主题", "统一切换窗口、侧栏和全部输入控件。", self.theme_combo))
        layout.addStretch(1)
        return page

    def _interaction_page(self) -> QWidget:
        page, layout = self._page("交互", "控制人工筛选时的浏览行为。")
        self.auto_advance_check = ToggleSwitch(self.preferences.auto_advance)
        layout.addWidget(SettingRow("接受后自动前进", "复制完成后直接显示下一组样本。", self.auto_advance_check))
        layout.addStretch(1)
        return page

    def _labels_page(self) -> QWidget:
        page, layout = self._page("按钮文字", "自定义底部三个主要审核按钮的显示文字。")
        self.label_edits: dict[str, QLineEdit] = {}
        for action in ("previous", "next", "accept"):
            edit = QLineEdit(self.preferences.button_labels[action])
            edit.setMinimumWidth(230)
            self.label_edits[action] = edit
            layout.addWidget(SettingRow(ACTION_NAMES[action], "显示在审核工具栏中。", edit))
        layout.addStretch(1)
        return page

    def _shortcuts_page(self) -> QWidget:
        page, layout = self._page("快捷键", "点击右侧输入框后按下新的按键组合；快捷键不能重复。")
        self.shortcut_edits: dict[str, QKeySequenceEdit] = {}
        for action in DEFAULT_SHORTCUTS:
            edit = QKeySequenceEdit(QKeySequence(self.preferences.shortcuts[action]))
            edit.setMaximumSequenceLength(1)
            edit.setMinimumWidth(170)
            self.shortcut_edits[action] = edit
            layout.addWidget(SettingRow(ACTION_NAMES[action], "窗口处于前台时生效。", edit))
        layout.addStretch(1)
        return page

    def _performance_page(self) -> QWidget:
        page, layout = self._page("性能", "大型点云会均匀抽样，源文件不会被修改。")
        self.point_limit_spin = QSpinBox()
        self.point_limit_spin.setRange(50_000, 1_000_000)
        self.point_limit_spin.setSingleStep(50_000)
        self.point_limit_spin.setValue(self.preferences.point_limit)
        self.point_limit_spin.setSuffix(" 点")
        self.point_limit_spin.setMinimumWidth(160)
        layout.addWidget(SettingRow("点云显示上限", "降低此值可以减少首次加载和旋转等待。", self.point_limit_spin))
        layout.addStretch(1)
        return page

    def _reset_defaults(self) -> None:
        self.theme_combo.setCurrentIndex(0)
        self.auto_advance_check.setChecked(True)
        self.point_limit_spin.setValue(300_000)
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
        self.accept()
