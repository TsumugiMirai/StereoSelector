from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .models import MODALITY_INFO, modality_label
from .settings import DialogHeader, SettingRow, ToggleSwitch


class MappingDialog(QDialog):
    """Manual modality-folder mapping and positional matching configuration."""

    def __init__(
        self,
        root: Path,
        initial_dirs: dict[str, list[Path]] | None = None,
        force_order: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.root = root.expanduser().resolve()
        self.setObjectName("settingsDialog")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.resize(850, 610)
        self.setMinimumSize(760, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        header = DialogHeader()
        header.setObjectName("settingsHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 0, 6, 0)
        title = QLabel("视图文件夹与匹配")
        title.setObjectName("settingsTitle")
        close = QPushButton("×")
        close.setObjectName("settingsCloseButton")
        close.setFixedSize(42, 40)
        close.clicked.connect(self.reject)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(close)
        outer.addWidget(header)

        body = QWidget()
        body.setObjectName("settingsPages")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(30, 24, 30, 24)
        body_layout.setSpacing(9)
        heading = QLabel("手动指定数据视图")
        heading.setObjectName("settingsPageTitle")
        description = QLabel(
            f"项目：{self.root}\n未填写的类型继续使用自动识别。所选文件夹必须位于项目目录内。"
        )
        description.setObjectName("settingsDescription")
        description.setWordWrap(True)
        body_layout.addWidget(heading)
        body_layout.addWidget(description)
        body_layout.addSpacing(8)

        self.path_edits: dict[str, QLineEdit] = {}
        initial_dirs = initial_dirs or {}
        for modality in MODALITY_INFO:
            row = QFrame()
            row.setObjectName("settingRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(14, 9, 10, 9)
            row_layout.setSpacing(10)
            label = QLabel(modality_label(modality))
            label.setObjectName("settingRowTitle")
            label.setFixedWidth(82)
            current = initial_dirs.get(modality, [])
            edit = QLineEdit(str(current[0]) if current else "")
            edit.setPlaceholderText("自动识别")
            edit.setClearButtonEnabled(True)
            browse = QPushButton("选择…")
            browse.clicked.connect(lambda checked=False, name=modality: self._choose_folder(name))
            self.path_edits[modality] = edit
            row_layout.addWidget(label)
            row_layout.addWidget(edit, 1)
            row_layout.addWidget(browse)
            body_layout.addWidget(row)

        self.force_order_switch = ToggleSwitch(force_order)
        force_row = SettingRow(
            "按顺序强制匹配",
            "忽略文件名差异；各文件夹自然排序后，第 1 个与第 1 个直接对应。",
            self.force_order_switch,
        )
        body_layout.addWidget(force_row)
        body_layout.addStretch(1)
        outer.addWidget(body, 1)

        footer = QFrame()
        footer.setObjectName("settingsFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(14, 10, 14, 10)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        apply_button = QPushButton("应用并重新扫描")
        apply_button.setObjectName("primaryButton")
        apply_button.clicked.connect(self._validate_and_accept)
        footer_layout.addStretch(1)
        footer_layout.addWidget(cancel)
        footer_layout.addWidget(apply_button)
        outer.addWidget(footer)

    @property
    def force_order(self) -> bool:
        return self.force_order_switch.isChecked()

    def selected_dirs(self) -> dict[str, Path]:
        return {
            modality: Path(edit.text().strip()).expanduser().resolve()
            for modality, edit in self.path_edits.items()
            if edit.text().strip()
        }

    def _choose_folder(self, modality: str) -> None:
        edit = self.path_edits[modality]
        start = edit.text().strip() or str(self.root)
        selected = QFileDialog.getExistingDirectory(self, f"选择{modality_label(modality)}文件夹", start)
        if selected:
            edit.setText(selected)

    def _validate_and_accept(self) -> None:
        selected = self.selected_dirs()
        for modality, directory in selected.items():
            if not directory.is_dir():
                QMessageBox.warning(self, "文件夹不存在", f"{modality_label(modality)}文件夹不存在：\n{directory}")
                return
            if not directory.is_relative_to(self.root):
                QMessageBox.warning(
                    self,
                    "文件夹不在项目内",
                    f"{modality_label(modality)}文件夹必须位于项目目录内：\n{self.root}",
                )
                return
        self.accept()
