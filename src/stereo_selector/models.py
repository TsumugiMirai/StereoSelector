from __future__ import annotations

import re
import shutil
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


MODALITY_INFO: dict[str, tuple[str, tuple[str, ...]]] = {
    "left": ("左目 RGB", ("left", "left_rgb", "rgb_left")),
    "right": ("右目 RGB", ("right", "right_rgb", "rgb_right")),
    "depth_fsd": ("深度图", ("depth_fsd", "depth", "depth_raw")),
    "depth_color": ("伪彩图", ("depth_color", "color_depth", "depth_vis")),
    "ply": ("点云", ("ply", "pointcloud", "point_cloud")),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
POINT_EXTENSIONS = {".ply"}

_TOKEN_RE = re.compile(
    r"(?i)(?:^|[_\-. ])(?:left|right|rgb|depth|fsd|color|colour|raw|vis|ply|pointcloud|point_cloud)(?=$|[_\-. ])"
)
_NATURAL_RE = re.compile(r"(\d+)")


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in _NATURAL_RE.split(value))


def normalized_sample_key(relative_file: Path) -> str:
    """Create a modality-independent key while retaining mirrored subdirectories."""
    parts = list(relative_file.with_suffix("").parts)
    normalized: list[str] = []
    for part in parts:
        # Preserve word boundaries before case-folding so camera names such as
        # ColorLeft / ColorRight / DepthColor normalize to the same sample id.
        cleaned = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", part)
        cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cleaned)
        cleaned = cleaned.casefold().replace(" ", "_")
        previous = None
        while cleaned != previous:
            previous = cleaned
            cleaned = _TOKEN_RE.sub("_", cleaned)
        cleaned = re.sub(r"[_\-.]+", "_", cleaned).strip("_")
        normalized.append(cleaned or part.casefold())
    return "/".join(normalized)


@dataclass(frozen=True)
class Sample:
    key: str
    files: dict[str, Path]

    @property
    def display_name(self) -> str:
        return next(iter(self.files.values())).stem if self.files else self.key


@dataclass
class Dataset:
    root: Path
    modality_dirs: dict[str, list[Path]] = field(default_factory=dict)
    files: dict[str, list[Path]] = field(default_factory=dict)
    samples: list[Sample] = field(default_factory=list)
    force_order: bool = False

    @property
    def available_modalities(self) -> list[str]:
        return [name for name in MODALITY_INFO if self.files.get(name)]

    @property
    def output_root(self) -> Path:
        return self.root.parent / f"{self.root.name}_select"


class DatasetScanner:
    def scan(
        self,
        root: Path,
        manual_dirs: dict[str, Path] | None = None,
        force_order: bool = False,
    ) -> Dataset:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"项目文件夹不存在：{root}")

        alias_to_modality = {
            alias.casefold(): modality
            for modality, (_, aliases) in MODALITY_INFO.items()
            for alias in aliases
        }
        modality_dirs: dict[str, list[Path]] = {name: [] for name in MODALITY_INFO}

        resolved_manual: dict[str, Path] = {}
        for modality, selected in (manual_dirs or {}).items():
            if modality not in MODALITY_INFO:
                continue
            selected = selected.expanduser().resolve()
            if not selected.is_dir():
                raise ValueError(f"{modality_label(modality)}文件夹不存在：{selected}")
            if not selected.is_relative_to(root):
                raise ValueError(f"{modality_label(modality)}文件夹必须位于项目目录内：{root}")
            resolved_manual[modality] = selected
            modality_dirs[modality] = [selected]

        # Discover directory names without recursively visiting every media file.
        # Once a modality directory is found, its subtree is scanned exactly once below.
        manual_paths = set(resolved_manual.values())
        for current, directory_names, _ in os.walk(root):
            current_path = Path(current)
            directory_names.sort(key=natural_key)
            descend: list[str] = []
            for name in directory_names:
                directory = current_path / name
                if manual_paths and directory.resolve() in manual_paths:
                    continue
                modality = alias_to_modality.get(name.casefold())
                if modality is None:
                    descend.append(name)
                elif modality not in resolved_manual:
                    modality_dirs[modality].append(directory)
            directory_names[:] = descend

        files: dict[str, list[Path]] = {name: [] for name in MODALITY_INFO}
        keyed: dict[str, dict[str, Path]] = {name: {} for name in MODALITY_INFO}
        for modality, directories in modality_dirs.items():
            extensions = POINT_EXTENSIONS if modality == "ply" else IMAGE_EXTENSIONS
            for directory in directories:
                for file in directory.rglob("*"):
                    if not file.is_file() or file.suffix.casefold() not in extensions:
                        continue
                    files[modality].append(file)
                    relative_inside = file.relative_to(directory)
                    # Prefix mirrored scene location when several modality folders exist.
                    scene_prefix = directory.parent.relative_to(root)
                    key_path = scene_prefix / relative_inside if scene_prefix != Path(".") else relative_inside
                    key = normalized_sample_key(key_path)
                    if key in keyed[modality]:
                        key = f"{key}::{file.relative_to(root).as_posix().casefold()}"
                    keyed[modality][key] = file
            files[modality].sort(key=lambda p: natural_key(str(p.relative_to(root))))

        if force_order:
            sample_count = max((len(paths) for paths in files.values()), default=0)
            samples = [
                Sample(
                    key=f"order/{index + 1:06d}",
                    files={name: paths[index] for name, paths in files.items() if index < len(paths)},
                )
                for index in range(sample_count)
            ]
        else:
            all_keys = sorted({key for mapping in keyed.values() for key in mapping}, key=natural_key)
            samples = [
                Sample(key=key, files={name: mapping[key] for name, mapping in keyed.items() if key in mapping})
                for key in all_keys
            ]
        return Dataset(
            root=root,
            modality_dirs={name: paths for name, paths in modality_dirs.items() if paths},
            files={name: paths for name, paths in files.items() if paths},
            samples=samples,
            force_order=force_order,
        )


def copy_sample(dataset: Dataset, sample: Sample) -> list[Path]:
    """Copy every matched asset atomically while preserving the source structure."""
    copied: list[Path] = []
    for source in sample.files.values():
        relative = source.relative_to(dataset.root)
        destination = dataset.output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.stereoselector-copying")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        copied.append(destination)
    return copied


def sample_is_copied(dataset: Dataset, sample: Sample) -> bool:
    """Return true only when every final output exists with the complete source size."""
    if not sample.files:
        return False
    try:
        return all(
            (destination := dataset.output_root / source.relative_to(dataset.root)).is_file()
            and destination.stat().st_size == source.stat().st_size
            for source in sample.files.values()
        )
    except OSError:
        return False


def modality_label(modality: str) -> str:
    return MODALITY_INFO.get(modality, (modality, ()))[0]


def summarize_missing(sample: Sample, selected: Iterable[str]) -> list[str]:
    return [name for name in selected if name not in sample.files]
