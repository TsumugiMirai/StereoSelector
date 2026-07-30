from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .models import Dataset, Sample, sample_is_copied
from .settings import DialogCloseButton, DialogHeader, ShadowDialog


MANIFEST_FILENAME = "stereo_selector_review.json"
DEFECT_TAGS = (
    "模糊",
    "过曝",
    "欠曝",
    "左右目不匹配",
    "深度异常",
    "点云异常",
    "数据缺失",
    "其他",
)
_INVALID_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_FOLDER_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    """Atomic project review manifest stored beside the selected output."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.path = dataset.output_root / MANIFEST_FILENAME
        self.records: dict[str, dict[str, object]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取审核文件：\n{self.path}\n\n{exc}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"审核文件格式无效：\n{self.path}")
        project = document.get("project", {})
        if not isinstance(project, dict):
            raise ValueError(f"审核文件格式无效：\n{self.path}")
        source_root = str(project.get("source_root", ""))
        if source_root and Path(source_root).resolve() != self.dataset.root:
            raise ValueError(f"输出文件夹属于其他项目：\n{source_root}")
        samples = document.get("samples", {})
        if not isinstance(samples, dict):
            raise ValueError(f"审核文件格式无效：\n{self.path}")
        self.records = {
            str(key): dict(value)
            for key, value in samples.items()
            if isinstance(value, dict)
        }

    def _record_for(self, sample: Sample) -> dict[str, object]:
        record = self.records.setdefault(sample.key, {})
        record["display_name"] = sample.display_name
        record["files"] = {
            modality: path.relative_to(self.dataset.root).as_posix()
            for modality, path in sample.files.items()
        }
        record.setdefault("status", "pending")
        record.setdefault("defect_tags", [])
        record.setdefault("note", "")
        return record

    def get(self, sample: Sample) -> dict[str, object]:
        record = self.records.get(sample.key, {})
        return {
            "status": str(record.get("status", "pending")),
            "defect_tags": [
                str(tag) for tag in record.get("defect_tags", []) if str(tag).strip()
            ],
            "note": str(record.get("note", "")),
            "updated_at": str(record.get("updated_at", "")),
        }

    def set_annotation(self, sample: Sample, tags: list[str], note: str) -> None:
        record = self._record_for(sample)
        record["defect_tags"] = list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
        record["note"] = note.strip()
        record["updated_at"] = _utc_now()
        self.save()

    def set_status(self, sample: Sample, status: str) -> None:
        if status not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"不支持的审核状态：{status}")
        record = self._record_for(sample)
        record["status"] = status
        record["updated_at"] = _utc_now()
        self.save()

    def reconcile(self) -> set[str]:
        """Reconcile manifest status with complete files in the active output."""
        accepted: set[str] = set()
        changed = not self.path.exists()
        for sample in self.dataset.samples:
            copied = sample_is_copied(self.dataset, sample)
            record = self.records.get(sample.key)
            recorded_status = str(record.get("status", "pending")) if record else "pending"
            if copied and (record is None or recorded_status == "accepted"):
                accepted.add(sample.key)
                if recorded_status != "accepted":
                    record = self._record_for(sample)
                    record["status"] = "accepted"
                    record["updated_at"] = _utc_now()
                    changed = True
            elif recorded_status == "accepted":
                record = self._record_for(sample)
                record["status"] = "pending"
                record["updated_at"] = _utc_now()
                changed = True
        if changed:
            self.save()
        return accepted

    def save(self) -> None:
        document = {
            "schema_version": 1,
            "project": {
                "name": self.dataset.root.name,
                "source_root": str(self.dataset.root),
                "output_root": str(self.dataset.output_root),
                "sample_count": len(self.dataset.samples),
                "modalities": self.dataset.available_modalities,
            },
            "updated_at": _utc_now(),
            "samples": dict(sorted(self.records.items())),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.stereoselector-writing")
        try:
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


class _BaseDialog(ShadowDialog):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        header = DialogHeader()
        header.setObjectName("settingsHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 0, 5, 0)
        heading = QLabel(title)
        heading.setObjectName("settingsTitle")
        close = DialogCloseButton()
        close.clicked.connect(self.reject)
        header_layout.addWidget(heading)
        header_layout.addStretch(1)
        header_layout.addWidget(close)
        self.outer.addWidget(header)

    def add_footer(self, accept_text: str, handler) -> None:
        footer = QFrame()
        footer.setObjectName("settingsFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(12, 8, 12, 8)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        accept = QPushButton(accept_text)
        accept.setObjectName("primaryButton")
        accept.clicked.connect(handler)
        layout.addStretch(1)
        layout.addWidget(cancel)
        layout.addWidget(accept)
        self.outer.addWidget(footer)


class OutputSettingsDialog(_BaseDialog):
    def __init__(
        self,
        project_root: Path,
        current_output_root: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("输出设置", parent)
        self.project_root = project_root.resolve()
        self._selected_root: Path | None = None
        self.resize(760, 380)
        self.setMinimumSize(660, 340)

        body = QWidget()
        body.setObjectName("settingsPages")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(8)
        title = QLabel("保存位置")
        title.setObjectName("settingsPageTitle")
        layout.addWidget(title)
        layout.addSpacing(8)

        parent_row = QFrame()
        parent_row.setObjectName("settingRow")
        parent_layout = QHBoxLayout(parent_row)
        parent_layout.setContentsMargins(2, 10, 2, 10)
        parent_label = QLabel("输出位置")
        parent_label.setObjectName("settingRowTitle")
        parent_label.setFixedWidth(90)
        self.parent_edit = QLineEdit(str(current_output_root.parent))
        self.parent_edit.setClearButtonEnabled(True)
        browse = QPushButton("选择…")
        browse.clicked.connect(self._choose_parent)
        parent_layout.addWidget(parent_label)
        parent_layout.addWidget(self.parent_edit, 1)
        parent_layout.addWidget(browse)
        layout.addWidget(parent_row)

        name_row = QFrame()
        name_row.setObjectName("settingRow")
        name_layout = QHBoxLayout(name_row)
        name_layout.setContentsMargins(2, 10, 2, 10)
        name_label = QLabel("文件夹名称")
        name_label.setObjectName("settingRowTitle")
        name_label.setFixedWidth(90)
        self.name_edit = QLineEdit(current_output_root.name)
        self.name_edit.setClearButtonEnabled(True)
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_edit, 1)
        layout.addWidget(name_row)

        preview_title = QLabel("完整路径")
        preview_title.setObjectName("sectionLabel")
        self.preview = QLabel()
        self.preview.setObjectName("projectPath")
        self.preview.setWordWrap(True)
        layout.addSpacing(4)
        layout.addWidget(preview_title)
        layout.addWidget(self.preview)
        layout.addStretch(1)
        self.outer.addWidget(body, 1)
        self.add_footer("保存", self._validate_and_accept)

        self.parent_edit.textChanged.connect(self._update_preview)
        self.name_edit.textChanged.connect(self._update_preview)
        self._update_preview()

    @property
    def output_root(self) -> Path:
        if self._selected_root is None:
            raise RuntimeError("输出设置尚未保存")
        return self._selected_root

    def _choose_parent(self) -> None:
        start = self.parent_edit.text().strip() or str(self.project_root.parent)
        selected = QFileDialog.getExistingDirectory(self, "选择输出位置", start)
        if selected:
            self.parent_edit.setText(selected)

    def _candidate(self) -> Path | None:
        parent = self.parent_edit.text().strip()
        name = self.name_edit.text().strip()
        if not parent or not name:
            return None
        return Path(parent).expanduser() / name

    def _update_preview(self) -> None:
        candidate = self._candidate()
        self.preview.setText(str(candidate) if candidate is not None else "—")

    def _validate_and_accept(self) -> None:
        parent = Path(self.parent_edit.text().strip()).expanduser()
        name = self.name_edit.text().strip()
        if not parent.is_dir():
            QMessageBox.warning(self, "输出位置无效", "请选择一个存在的文件夹。")
            return
        if (
            not name
            or name in {".", ".."}
            or _INVALID_FOLDER_CHARS.search(name)
            or name.endswith((" ", "."))
            or name.upper() in _RESERVED_FOLDER_NAMES
        ):
            QMessageBox.warning(self, "文件夹名称无效", "请使用有效的文件夹名称。")
            return
        selected = (parent / name).resolve()
        if selected == self.project_root or selected.is_relative_to(self.project_root):
            QMessageBox.warning(self, "输出位置无效", "输出文件夹不能位于项目文件夹内部。")
            return
        manifest = selected / MANIFEST_FILENAME
        if manifest.is_file():
            try:
                document = json.loads(manifest.read_text(encoding="utf-8"))
                project = document.get("project", {}) if isinstance(document, dict) else {}
                source_root = str(project.get("source_root", "")) if isinstance(project, dict) else ""
            except (OSError, json.JSONDecodeError):
                QMessageBox.warning(self, "审核文件无效", f"无法读取：\n{manifest}")
                return
            if source_root and Path(source_root).resolve() != self.project_root:
                QMessageBox.warning(self, "输出文件夹已被使用", f"该文件夹属于其他项目：\n{source_root}")
                return
        self._selected_root = selected
        self.accept()


class AnnotationDialog(_BaseDialog):
    def __init__(
        self,
        sample: Sample,
        tags: list[str],
        note: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("缺陷与备注", parent)
        self.resize(700, 570)
        self.setMinimumSize(620, 520)

        body = QWidget()
        body.setObjectName("settingsPages")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(8)
        title = QLabel(sample.display_name)
        title.setObjectName("settingsPageTitle")
        layout.addWidget(title)
        layout.addSpacing(6)

        tag_title = QLabel("缺陷标签")
        tag_title.setObjectName("sectionLabel")
        layout.addWidget(tag_title)
        tag_frame = QFrame()
        tag_frame.setObjectName("settingRow")
        tag_layout = QGridLayout(tag_frame)
        tag_layout.setContentsMargins(2, 8, 2, 10)
        tag_layout.setHorizontalSpacing(16)
        tag_layout.setVerticalSpacing(6)
        selected = set(tags)
        self.tag_checks: dict[str, QCheckBox] = {}
        for index, tag in enumerate(DEFECT_TAGS):
            checkbox = QCheckBox(tag)
            checkbox.setChecked(tag in selected)
            self.tag_checks[tag] = checkbox
            tag_layout.addWidget(checkbox, index // 2, index % 2)
        layout.addWidget(tag_frame)

        custom_title = QLabel("自定义标签")
        custom_title.setObjectName("sectionLabel")
        self.custom_tags = QLineEdit(
            "，".join(tag for tag in tags if tag not in DEFECT_TAGS)
        )
        self.custom_tags.setPlaceholderText("使用逗号分隔")
        layout.addWidget(custom_title)
        layout.addWidget(self.custom_tags)

        note_title = QLabel("备注")
        note_title.setObjectName("sectionLabel")
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("记录需要复查的内容")
        self.note_edit.setPlainText(note)
        layout.addWidget(note_title)
        layout.addWidget(self.note_edit, 1)
        self.outer.addWidget(body, 1)
        self.add_footer("保存", self.accept)

    def selected_tags(self) -> list[str]:
        tags = [tag for tag, checkbox in self.tag_checks.items() if checkbox.isChecked()]
        tags.extend(
            part.strip()
            for part in re.split(r"[,，;；]+", self.custom_tags.text())
            if part.strip()
        )
        return list(dict.fromkeys(tags))

    def note(self) -> str:
        return self.note_edit.toPlainText().strip()
