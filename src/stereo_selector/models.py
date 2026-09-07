from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

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
_SEQUENCE_PREFIX_RE = re.compile(r"^(\d+)(?:[_\-. ])")
_LONG_TIMESTAMP_RE = re.compile(r"(?:^|[_\-. ])\d{10,}(?=$|[_\-. ])")
_MODALITY_HINTS = {
    "left": ("left", "cam_l", "camera_l", "cam0", "camera0", "左目", "左相机"),
    "right": ("right", "cam_r", "camera_r", "cam1", "camera1", "右目", "右相机"),
    "depth_color": (
        "depth_color",
        "depthcolor",
        "color_depth",
        "colored_depth",
        "depth_vis",
        "depth_preview",
        "visual_depth",
        "pseudo",
        "伪彩",
        "彩色深度",
    ),
    "depth_fsd": ("depth", "distance", "range", "z16", "disparity", "深度", "视差"),
    "ply": ("ply", "pointcloud", "point_cloud", "cloud", "点云"),
}

_IGNORED_PROJECT_FOLDERS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
}


def _modality_aliases() -> dict[str, str]:
    return {
        alias.casefold(): modality
        for modality, (_, aliases) in MODALITY_INFO.items()
        for alias in aliases
    }


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


def _sequence_alignment_key(relative_file: Path) -> str | None:
    """Return a stable capture ordinal for stereo files with split timestamps."""
    prefix = _SEQUENCE_PREFIX_RE.match(relative_file.stem)
    if prefix is None or _LONG_TIMESTAMP_RE.search(relative_file.stem) is None:
        return None
    scene = "/".join(
        re.sub(r"[_\-. ]+", "_", part.casefold()).strip("_")
        for part in relative_file.parent.parts
    )
    ordinal = str(int(prefix.group(1)))
    return f"{scene}/sequence/{ordinal}" if scene else f"sequence/{ordinal}"


def infer_modality(directory: Path, media_files: list[Path]) -> str | None:
    """Infer a supported view from folder/file hints and the actual media format."""
    if not media_files:
        return None
    extensions = {path.suffix.casefold() for path in media_files}
    if extensions <= POINT_EXTENSIONS:
        return "ply"
    if not extensions & IMAGE_EXTENSIONS:
        return None
    searchable = " ".join(
        [directory.name, *(path.stem for path in media_files[:8])]
    ).casefold()
    searchable = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", searchable)
    scores = {
        modality: sum(1 for hint in hints if hint in searchable)
        for modality, hints in _MODALITY_HINTS.items()
    }
    # A color/visualized depth hint is more specific than the generic "depth" token.
    if scores["depth_color"]:
        return "depth_color"
    best = max(("left", "right", "depth_fsd"), key=scores.get)
    if scores[best]:
        return best

    # Numeric 16-bit or floating-point grayscale images are overwhelmingly likely
    # to be raw depth when folder names carry no usable camera hint.
    try:
        from PIL import Image

        with Image.open(media_files[0]) as image:
            if image.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
                return "depth_fsd"
    except OSError:
        pass
    return None


def detected_modalities(root: Path, *, max_depth: int = 3) -> set[str]:
    """Quickly identify the view types contained by a possible project root.

    Only representative files are inspected.  This keeps collection discovery
    fast even when every project contains thousands of frames.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        return set()
    aliases = _modality_aliases()
    detected: set[str] = set()
    for current, directory_names, file_names in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        directory_names[:] = [
            name
            for name in directory_names
            if name not in _IGNORED_PROJECT_FOLDERS
            and not name.casefold().endswith("_select")
        ]
        if depth >= max_depth:
            directory_names[:] = []
        media_files = [
            current_path / name
            for name in sorted(file_names, key=natural_key)
            if Path(name).suffix.casefold() in IMAGE_EXTENSIONS | POINT_EXTENSIONS
        ][:8]
        if not media_files:
            continue
        modality = aliases.get(current_path.name.casefold()) or infer_modality(
            current_path, media_files
        )
        if modality is not None:
            detected.add(modality)
            directory_names[:] = []
    return detected


def discover_project_roots(root: Path) -> list[Path]:
    """Return independently switchable projects below *root*.

    A folder whose immediate children each contain at least two recognizable
    views is treated as a project collection.  Otherwise the selected folder
    remains one project and its nested view folders are scanned recursively.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"项目文件夹不存在：{root}")
    candidates: list[Path] = []
    for child in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    ):
        if (
            child.name in _IGNORED_PROJECT_FOLDERS
            or child.name.casefold().endswith("_select")
        ):
            continue
        if len(detected_modalities(child)) >= 2:
            candidates.append(child.resolve())
    return candidates if len(candidates) >= 2 else [root]


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



class DatasetScanner:
    def scan(
        self,
        root: Path,
        manual_dirs: dict[str, Path] | None = None,
        force_order: bool = False,
        progress: Callable[[str, int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Dataset:
        def abort_if_cancelled() -> None:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("项目扫描已取消")

        root = root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"项目文件夹不存在：{root}")
        abort_if_cancelled()
        alias_to_modality = _modality_aliases()
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

        # Discover exact aliases first, then infer unconventional leaf folder names
        # from their file extensions, name hints and representative image type.
        manual_paths = set(resolved_manual.values())
        unknown_image_dirs: list[Path] = []
        discovered_files = 0
        for current, directory_names, file_names in os.walk(root):
            abort_if_cancelled()
            current_path = Path(current)
            directory_names.sort(key=natural_key)
            descend: list[str] = []
            for name in directory_names:
                if name in _IGNORED_PROJECT_FOLDERS or name.casefold().endswith("_select"):
                    continue
                directory = current_path / name
                if manual_paths and directory.resolve() in manual_paths:
                    continue
                modality = alias_to_modality.get(name.casefold())
                if modality is None:
                    descend.append(name)
                elif modality not in resolved_manual:
                    modality_dirs[modality].append(directory)
            directory_names[:] = descend
            direct_media = [
                current_path / name
                for name in sorted(file_names, key=natural_key)
                if Path(name).suffix.casefold() in IMAGE_EXTENSIONS | POINT_EXTENSIONS
            ]
            discovered_files += len(direct_media)
            if progress is not None:
                progress("正在扫描目录…", discovered_files, 0)
            if direct_media and current_path != root and current_path not in manual_paths:
                inferred = infer_modality(current_path, direct_media)
                if inferred is not None and inferred not in resolved_manual:
                    modality_dirs[inferred].append(current_path)
                    directory_names[:] = []
                elif any(path.suffix.casefold() in IMAGE_EXTENSIONS for path in direct_media):
                    unknown_image_dirs.append(current_path)

        # When names contain no hints at all, a pair of RGB folders is still a
        # useful stereo candidate. Natural ordering provides a deterministic fallback.
        missing_rgb = [name for name in ("left", "right") if not modality_dirs[name]]
        if unknown_image_dirs:
            for modality, directory in zip(missing_rgb, sorted(unknown_image_dirs, key=lambda p: natural_key(str(p)))):
                modality_dirs[modality].append(directory)

        for modality, directories in modality_dirs.items():
            modality_dirs[modality] = list(dict.fromkeys(directories))

        files: dict[str, list[Path]] = {name: [] for name in MODALITY_INFO}
        keyed: dict[str, dict[str, Path]] = {name: {} for name in MODALITY_INFO}
        sequence_groups: dict[str, list[tuple[str, str, Path]]] = {}
        indexed_files = 0
        for modality, directories in modality_dirs.items():
            extensions = POINT_EXTENSIONS if modality == "ply" else IMAGE_EXTENSIONS
            for directory in directories:
                abort_if_cancelled()
                for file in directory.rglob("*"):
                    if not file.is_file() or file.suffix.casefold() not in extensions:
                        continue
                    indexed_files += 1
                    if indexed_files % 500 == 0:
                        if progress is not None:
                            progress("正在索引文件…", indexed_files, 0)
                        abort_if_cancelled()
                    files[modality].append(file)
                    relative_inside = file.relative_to(directory)
                    # Prefix mirrored scene location when several modality folders exist.
                    scene_prefix = directory.parent.relative_to(root)
                    key_path = scene_prefix / relative_inside if scene_prefix != Path(".") else relative_inside
                    key = normalized_sample_key(key_path)
                    if key in keyed[modality]:
                        key = f"{key}::{file.relative_to(root).as_posix().casefold()}"
                    keyed[modality][key] = file
                    sequence_key = _sequence_alignment_key(key_path)
                    if sequence_key is not None:
                        sequence_groups.setdefault(sequence_key, []).append(
                            (modality, key, file)
                        )
            files[modality].sort(key=lambda p: natural_key(str(p.relative_to(root))))
        abort_if_cancelled()
        if progress is not None:
            progress("正在匹配样本…", 0, 0)

        # Some stereo recorders write an independent hardware timestamp for
        # left and right images. When a unique leading capture ordinal exists,
        # align those entries to the most widely shared primary key. Datasets
        # whose timestamps already match keep their existing stable keys.
        for entries in sequence_groups.values():
            modalities = [modality for modality, _key, _file in entries]
            if len(set(modalities)) < 2 or len(set(modalities)) != len(modalities):
                continue
            key_counts: dict[str, int] = {}
            for _modality, key, _file in entries:
                key_counts[key] = key_counts.get(key, 0) + 1
            canonical = min(
                key_counts,
                key=lambda key: (-key_counts[key], natural_key(key)),
            )
            if any(
                canonical in keyed[modality]
                and keyed[modality][canonical] != file
                for modality, _key, file in entries
            ):
                continue
            for modality, old_key, file in entries:
                if old_key == canonical:
                    continue
                if keyed[modality].get(old_key) == file:
                    del keyed[modality][old_key]
                keyed[modality][canonical] = file

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


def modality_label(modality: str) -> str:
    return MODALITY_INFO.get(modality, (modality, ()))[0]


def summarize_missing(sample: Sample, selected: Iterable[str]) -> list[str]:
    return [name for name in selected if name not in sample.files]
