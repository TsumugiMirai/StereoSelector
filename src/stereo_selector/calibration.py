"""Calibration loading, coordinate frames and vendor remap tables.

Frames and units
----------------
* ``CameraCalibration`` describes a *raw* camera: intrinsics plus distortion.
* ``RectifiedStereo`` describes the rectified stereo pair a depth sensor
  emits (``camera_rectify`` in the vendor JSON). Depth images and device
  point clouds live in this frame, so metric readouts must use it rather than
  the raw left intrinsics.
* Stereo translation is stored as written in the file together with
  ``translation_unit``; ``baseline_m`` converts it to metres. Rectification
  and the fundamental matrix are scale invariant, so the raw vector is kept.
* Depth pixel values are converted with :mod:`stereo_selector.units`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraCalibration:
    matrix: np.ndarray
    distortion: np.ndarray
    width: int | None = None
    height: int | None = None

    @property
    def is_pinhole(self) -> bool:
        return self.distortion.size == 0 or not np.any(self.distortion)

    def point_from_depth(self, x: float, y: float, depth: float) -> tuple[float, float, float]:
        """Back-project a pixel at metric ``depth`` (Z) into camera coordinates.

        Distortion is undone iteratively when coefficients are present; a
        pinhole camera (rectified frame) takes the direct path.
        """
        fx, fy = float(self.matrix[0, 0]), float(self.matrix[1, 1])
        cx, cy = float(self.matrix[0, 2]), float(self.matrix[1, 2])
        if fx == 0 or fy == 0:
            raise ValueError("相机焦距不能为零")
        distorted_x = (x - cx) / fx
        distorted_y = (y - cy) / fy
        if self.is_pinhole:
            return (distorted_x * depth, distorted_y * depth, depth)
        normalized_x, normalized_y = distorted_x, distorted_y
        coefficients = np.pad(self.distortion[:8], (0, max(0, 8 - self.distortion[:8].size)))
        k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
        for _ in range(6):
            radius2 = normalized_x * normalized_x + normalized_y * normalized_y
            radius4, radius6 = radius2 * radius2, radius2 * radius2 * radius2
            numerator = 1.0 + k1 * radius2 + k2 * radius4 + k3 * radius6
            denominator = 1.0 + k4 * radius2 + k5 * radius4 + k6 * radius6
            radial = numerator / denominator if denominator else numerator
            delta_x = 2.0 * p1 * normalized_x * normalized_y + p2 * (
                radius2 + 2.0 * normalized_x * normalized_x
            )
            delta_y = p1 * (radius2 + 2.0 * normalized_y * normalized_y) + (
                2.0 * p2 * normalized_x * normalized_y
            )
            if radial == 0:
                break
            normalized_x = (distorted_x - delta_x) / radial
            normalized_y = (distorted_y - delta_y) / radial
        return (normalized_x * depth, normalized_y * depth, depth)


@dataclass(frozen=True)
class RectifiedStereo:
    """Rectified pinhole model shared by the left and right rectified views."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    rotation_left: np.ndarray | None = None
    rotation_right: np.ndarray | None = None
    mode: str = "air"

    @property
    def mode_label(self) -> str:
        return {"air": "空气", "underwater": "水下"}.get(self.mode, self.mode)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def projection(self) -> np.ndarray:
        return np.hstack((self.matrix, np.zeros((3, 1), dtype=np.float64)))

    def camera(self) -> CameraCalibration:
        return CameraCalibration(
            matrix=self.matrix,
            distortion=np.empty(0, dtype=np.float64),
            width=self.width,
            height=self.height,
        )


REMAP_FILE_NAMES = {
    "left": ("map_left_x.pfm", "map_left_y.pfm"),
    "right": ("map_right_x.pfm", "map_right_y.pfm"),
}


@dataclass(frozen=True)
class CalibrationData:
    source: Path
    left: CameraCalibration | None
    right: CameraCalibration | None
    rotation: np.ndarray | None
    translation: np.ndarray | None
    fundamental: np.ndarray | None
    cloud_rotation: np.ndarray | None
    cloud_translation: np.ndarray | None
    rectified: RectifiedStereo | None = None
    translation_unit: str = "m"
    remap_directory: Path | None = None
    rectified_options: tuple[RectifiedStereo, ...] = ()

    @property
    def available_rectify_modes(self) -> list[str]:
        """Rectification sets the file provides, air first (e.g. ``["air", "underwater"]``)."""
        return [option.mode for option in self.rectified_options]

    @property
    def rectify_mode(self) -> str:
        return self.rectified.mode if self.rectified is not None else "air"

    def with_rectify_mode(self, mode: str) -> CalibrationData:
        """Return a copy that uses the requested rectification set when available."""
        chosen = next((option for option in self.rectified_options if option.mode == mode), None)
        if chosen is None or chosen is self.rectified:
            return self
        return replace(self, rectified=chosen)

    @property
    def baseline_m(self) -> float | None:
        if self.translation is None:
            return None
        norm = float(np.linalg.norm(self.translation))
        return norm / 1000.0 if self.translation_unit == "mm" else norm

    def depth_camera(self) -> CameraCalibration | None:
        """Camera model of the depth image: rectified when known, else raw left."""
        if self.rectified is not None:
            return self.rectified.camera()
        return self.left

    @property
    def cloud_translation_m(self) -> np.ndarray | None:
        """Point-cloud translation converted to metres.

        ``cloud_to_left`` is written in the same unit as the stereo baseline;
        millimetre files are converted so the cloud lands in the metric frame
        shared with depth readouts.
        """
        if self.cloud_translation is None:
            return None
        if self.translation_unit == "mm":
            return self.cloud_translation / 1000.0
        return self.cloud_translation

    @property
    def depth_frame_label(self) -> str:
        return "校正左目" if self.rectified is not None else "原始左目"

    @property
    def has_remap_tables(self) -> bool:
        if self.remap_directory is None:
            return False
        return all(
            (self.remap_directory / name).is_file()
            for names in REMAP_FILE_NAMES.values()
            for name in names
        )

    @property
    def summary(self) -> str:
        cameras = (
            "双目内参"
            if self.left is not None and self.right is not None
            else "左目内参"
            if self.left is not None
            else "右目内参"
        )
        extras = []
        if self.rotation is not None and self.translation is not None:
            extras.append("外参")
        if self.fundamental is not None:
            extras.append("基础矩阵")
        if self.rectified is not None:
            extras.append("校正参数")
        if self.has_remap_tables:
            extras.append("重映射表")
        if self.cloud_rotation is not None and self.cloud_translation is not None:
            extras.append("点云外参")
        return " · ".join([cameras, *extras])

    @property
    def details(self) -> str:
        """Multi-line human readable description for tooltips and settings."""
        lines = [self.summary]
        baseline = self.baseline_m
        if baseline is not None:
            lines.append(f"基线 {baseline * 1000:.2f} mm（文件单位 {self.translation_unit}）")
        if self.rectified is not None:
            lines.append(
                f"校正（{self.rectified.mode_label}）焦距 {self.rectified.fx:.1f} px"
                f" · 主点 ({self.rectified.cx:.1f}, {self.rectified.cy:.1f})"
                f" · {self.rectified.width}×{self.rectified.height}"
            )
        lines.append(f"深度坐标系：{self.depth_frame_label}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CalibrationOption:
    id: str
    label: str
    path: Path
    builtin: bool = False
    project_local: bool = False


_BUILTIN_CALIBRATIONS = (
    ("builtin:libra2000", "libra2000", "calib2000_004_b"),
    ("builtin:libra3000", "libra3000", "calib3000_003"),
)

_CALIBRATION_SUFFIXES = {".json", ".yaml", ".yml"}


def discover_project_calibrations(root: Path) -> list[CalibrationOption]:
    """Find calibration files stored with a capture (``calib/`` folders, ``*calib*.json``).

    Only the project root and its immediate ``calib*`` sub-folders are
    inspected, and each candidate must parse as a calibration.
    """
    candidates: list[Path] = []
    try:
        for entry in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if entry.is_file():
                if entry.suffix.casefold() in _CALIBRATION_SUFFIXES and "calib" in entry.name.casefold():
                    candidates.append(entry)
            elif entry.is_dir() and "calib" in entry.name.casefold():
                for child in sorted(entry.iterdir(), key=lambda item: item.name.casefold()):
                    if child.is_file() and child.suffix.casefold() in _CALIBRATION_SUFFIXES:
                        candidates.append(child)
    except OSError:
        return []
    options: list[CalibrationOption] = []
    for path in candidates:
        try:
            load_calibration(path)
        except (ValueError, OSError):
            logger.debug("跳过无法解析的项目内标定：%s", path)
            continue
        base = custom_calibration_option(path)
        folder = path.parent.name if path.parent != root else ""
        label = f"项目内 {folder}" if folder and "calib" in folder.casefold() else f"项目内 {path.stem}"
        if len(candidates) > 1 and folder:
            label = f"项目内 {folder}/{path.stem}"
        options.append(CalibrationOption(base.id, label, base.path, False, True))
    return options


def builtin_calibration_options() -> list[CalibrationOption]:
    """Return the calibrations shipped inside the package.

    The JSON files live in ``stereo_selector/assets/calibration`` so they are
    present in source checkouts, ``pip install`` builds and PyInstaller
    bundles alike. A development checkout additionally keeps the vendor remap
    tables (``*.pfm``) next to the JSON in the top-level folder, so that copy
    is preferred when it exists.
    """
    roots = [Path(__file__).resolve().parents[2] / "calibration"]
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        roots.append(Path(sys._MEIPASS) / "stereo_selector" / "assets" / "calibration")
        roots.append(Path(sys._MEIPASS) / "calibration")
    roots.append(Path(__file__).resolve().parent / "assets" / "calibration")
    options: list[CalibrationOption] = []
    for option_id, label, folder in _BUILTIN_CALIBRATIONS:
        path = next(
            (
                root / folder / "calibration_param.json"
                for root in roots
                if (root / folder / "calibration_param.json").is_file()
            ),
            None,
        )
        if path is not None:
            options.append(CalibrationOption(option_id, label, path.resolve(), True))
    return options


def custom_calibration_option(path: Path) -> CalibrationOption:
    path = path.expanduser().resolve()
    normalized = str(path).casefold().encode("utf-8")
    option_id = f"custom:{hashlib.sha1(normalized).hexdigest()[:16]}"
    return CalibrationOption(option_id, path.stem, path, False)


def _opencv_yaml_text(text: str) -> str:
    text = re.sub(r"^%YAML:\s*1\.0\s*$", "", text, flags=re.MULTILINE)
    return text.replace("!!opencv-matrix", "")


def _load_document(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.casefold() == ".json":
            document = json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ValueError("读取 YAML 标定文件需要安装 PyYAML") from exc
            document = yaml.safe_load(_opencv_yaml_text(path.read_text(encoding="utf-8-sig")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取标定文件：\n{path}\n\n{exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("标定文件根节点必须是对象")
    return document


def _lookup(mapping: dict[str, Any], *keys: str) -> Any:
    lowered = {str(key).casefold(): value for key, value in mapping.items()}
    for key in keys:
        if key.casefold() in lowered:
            return lowered[key.casefold()]
    return None


def _matrix(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = _lookup(value, "data", "values", "matrix")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if shape is not None:
        try:
            array = array.reshape(shape)
        except ValueError:
            return None
    return array


def _scalar(value: Any) -> float | None:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _camera_from(mapping: Any, matrix_fallback: Any = None, distortion_fallback: Any = None) -> CameraCalibration | None:
    camera = mapping if isinstance(mapping, dict) else {}
    matrix = _matrix(
        _lookup(camera, "camera_matrix", "intrinsic_matrix", "matrix", "K")
        if camera
        else matrix_fallback,
        (3, 3),
    )
    if matrix is None and matrix_fallback is not None:
        matrix = _matrix(matrix_fallback, (3, 3))
    if matrix is None or not np.any(matrix):
        return None
    distortion_value = (
        _lookup(
            camera,
            "distortion",
            "distortion_coefficients",
            "distortion_coeffs",
            "dist_coeffs",
            "D",
        )
        if camera
        else distortion_fallback
    )
    if distortion_value is None:
        distortion_value = distortion_fallback
    distortion = _matrix(distortion_value)
    if distortion is None:
        distortion = np.empty(0, dtype=np.float64)
    width_value = _lookup(camera, "width", "image_width", "cols") if camera else None
    height_value = _lookup(camera, "height", "image_height", "rows") if camera else None
    return CameraCalibration(
        matrix=matrix,
        distortion=distortion.reshape(-1),
        width=int(width_value) if width_value else None,
        height=int(height_value) if height_value else None,
    )


def _rectified_from(mapping: Any, left: CameraCalibration | None, mode: str = "air") -> RectifiedStereo | None:
    if not isinstance(mapping, dict):
        return None
    fx = _scalar(_lookup(mapping, "fx", "focal", "focus"))
    fy = _scalar(_lookup(mapping, "fy")) or fx
    cx = _scalar(_lookup(mapping, "cx"))
    cy = _scalar(_lookup(mapping, "cy"))
    matrix = _matrix(_lookup(mapping, "camera_matrix", "K", "P"), None)
    if (fx is None or cx is None or cy is None) and matrix is not None and matrix.size >= 9:
        flat = matrix.reshape(-1)
        fx, fy, cx, cy = float(flat[0]), float(flat[4 if matrix.size == 9 else 5]), float(flat[2]), float(
            flat[5 if matrix.size == 9 else 6]
        )
    if fx is None or fy is None or cx is None or cy is None or fx <= 0 or fy <= 0:
        return None
    width = _lookup(mapping, "cols", "width", "image_width")
    height = _lookup(mapping, "rows", "height", "image_height")
    if not width or not height:
        if left is None or left.width is None or left.height is None:
            return None
        width, height = left.width, left.height
    return RectifiedStereo(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=int(width),
        height=int(height),
        rotation_left=_matrix(_lookup(mapping, "rotation_left", "R1", "rectification_left"), (3, 3)),
        rotation_right=_matrix(_lookup(mapping, "rotation_right", "R2", "rectification_right"), (3, 3)),
        mode=mode,
    )


RECTIFY_MODE_CHOICES: tuple[tuple[str, str], ...] = (("air", "空气"), ("underwater", "水下"))


def _rectified_options(document: dict[str, Any], left: CameraCalibration | None) -> tuple[RectifiedStereo, ...]:
    """Collect every rectification set in the file, in-air first.

    Vendor files that carry ``camera_rectify_air`` (or any ``uw_*`` key) use
    ``camera_rectify`` for the underwater optics; plain files have one set
    that is treated as in-air.
    """
    air = _rectified_from(_lookup(document, "camera_rectify_air", "rectify_air", "rectified_air"), left, "air")
    underwater_hint = air is not None or any("uw" in str(key).casefold() for key in document)
    generic = _rectified_from(
        _lookup(document, "camera_rectify", "rectified", "rectification", "depth_camera"),
        left,
        "underwater" if underwater_hint else "air",
    )
    options = [option for option in (air, generic) if option is not None]
    if len(options) == 2 and options[0].mode == options[1].mode:
        options = options[:1]
    return tuple(options)


def _select_rectified(options: tuple[RectifiedStereo, ...], mode: str) -> RectifiedStereo | None:
    """Pick the requested mode, falling back to the first (in-air) set."""
    return next((option for option in options if option.mode == mode), options[0] if options else None)


def normalize_rectify_mode(mode: object) -> str:
    text = str(mode or "air").strip().casefold()
    return text if text in {choice for choice, _label in RECTIFY_MODE_CHOICES} else "air"


def _translation_unit(document: dict[str, Any], stereo_mapping: dict[str, Any], translation: np.ndarray | None) -> str:
    explicit = _lookup(stereo_mapping, "unit", "translation_unit", "distance_unit") or _lookup(
        document, "unit", "translation_unit", "distance_unit"
    )
    if isinstance(explicit, str):
        lowered = explicit.strip().casefold()
        if lowered in {"mm", "millimeter", "millimetre", "毫米"}:
            return "mm"
        if lowered in {"m", "meter", "metre", "米"}:
            return "m"
    if translation is None:
        return "m"
    # Stereo baselines are a few centimetres to a few decimetres. Anything
    # above 1.5 was clearly written in millimetres.
    return "mm" if float(np.linalg.norm(translation)) > 1.5 else "m"


def load_calibration(path: Path, rectify_mode: str = "air") -> CalibrationData:
    """Load a calibration file; ``rectify_mode`` selects air or underwater rectification when both exist."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"标定文件不存在：{path}")
    document = _load_document(path)
    left_mapping = _lookup(document, "left", "left_camera", "camera_left", "left_intrinsics")
    right_mapping = _lookup(document, "right", "right_camera", "camera_right", "right_intrinsics")
    left = _camera_from(
        left_mapping,
        _lookup(document, "M1", "K1", "left_camera_matrix"),
        _lookup(document, "D1", "left_distortion"),
    )
    right = _camera_from(
        right_mapping,
        _lookup(document, "M2", "K2", "right_camera_matrix"),
        _lookup(document, "D2", "right_distortion"),
    )

    stereo = _lookup(document, "stereo", "left_to_right", "extrinsics", "extrinsic_l_r")
    stereo_mapping = stereo if isinstance(stereo, dict) else document
    rotation = _matrix(_lookup(stereo_mapping, "rotation", "rotation_matrix", "R"), (3, 3))
    translation = _matrix(_lookup(stereo_mapping, "translation", "translation_vector", "T"))
    if translation is not None:
        translation = translation.reshape(-1)
        if translation.size != 3 or not np.any(translation):
            translation = None
    if rotation is not None and not np.any(rotation):
        rotation = None
    fundamental = _matrix(
        _lookup(stereo_mapping, "fundamental", "fundamental_matrix", "F"),
        (3, 3),
    )
    if (
        fundamental is None
        and left is not None
        and right is not None
        and rotation is not None
        and translation is not None
    ):
        tx, ty, tz = (float(value) for value in translation)
        skew = np.array(
            [[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]],
            dtype=np.float64,
        )
        try:
            left_inverse = np.linalg.inv(left.matrix)
            right_inverse = np.linalg.inv(right.matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("相机内参矩阵不可逆，无法计算基础矩阵") from exc
        fundamental = right_inverse.T @ skew @ rotation @ left_inverse
    if left is None and right is None:
        raise ValueError("未找到相机内参矩阵（camera_matrix、K、M1/M2）")
    cloud_transform = _lookup(
        document,
        "cloud_to_left",
        "pointcloud_to_left",
        "point_cloud_to_left",
    )
    cloud_mapping = cloud_transform if isinstance(cloud_transform, dict) else {}
    cloud_rotation = _matrix(
        _lookup(cloud_mapping, "rotation", "rotation_matrix", "R"),
        (3, 3),
    )
    cloud_translation = _matrix(
        _lookup(cloud_mapping, "translation", "translation_vector", "T")
    )
    if cloud_translation is not None:
        cloud_translation = cloud_translation.reshape(-1)
        if cloud_translation.size != 3:
            cloud_translation = None
    rectified_options = _rectified_options(document, left)
    rectified = _select_rectified(rectified_options, normalize_rectify_mode(rectify_mode))
    remap_directory = path.parent if any(
        (path.parent / name).is_file() for names in REMAP_FILE_NAMES.values() for name in names
    ) else None
    return CalibrationData(
        path,
        left,
        right,
        rotation,
        translation,
        fundamental,
        cloud_rotation,
        cloud_translation,
        rectified=rectified,
        translation_unit=_translation_unit(document, stereo_mapping, translation),
        remap_directory=remap_directory,
        rectified_options=rectified_options,
    )


def read_pfm(path: Path) -> np.ndarray:
    """Read a single-channel PFM image as stored (no vertical flip applied).

    The PFM specification stores rows bottom-up, but several vendors write
    them top-down. :func:`load_remap_tables` validates the orientation
    against the JSON model, so this reader stays neutral.
    """
    data = path.read_bytes()
    try:
        header, dims, scale, payload = data.split(b"\n", 3)
    except ValueError as exc:
        raise ValueError(f"PFM 文件头无效：{path}") from exc
    if header.strip() != b"Pf":
        raise ValueError(f"仅支持单通道 PFM：{path}")
    width, height = (int(value) for value in dims.split())
    dtype = "<f4" if float(scale) < 0 else ">f4"
    count = width * height
    if len(payload) < count * 4:
        raise ValueError(f"PFM 数据不完整：{path}")
    return np.frombuffer(payload, dtype=dtype, count=count).reshape(height, width).astype(np.float32)


@dataclass(frozen=True)
class RemapTables:
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    flipped: bool
    error_px: float
    mode: str = "air"  # rectification set the tables were validated against


def compute_rectify_maps(
    calibration: CalibrationData,
    size: tuple[int, int],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """Compute OpenCV remap tables from the calibration (rectified section first)."""
    import cv2

    if calibration.left is None or calibration.right is None:
        return None
    left_matrix = calibration.left.matrix.astype(np.float64)
    right_matrix = calibration.right.matrix.astype(np.float64)
    left_distortion = calibration.left.distortion.astype(np.float64)
    right_distortion = calibration.right.distortion.astype(np.float64)
    rectified = calibration.rectified
    if (
        rectified is not None
        and rectified.rotation_left is not None
        and rectified.rotation_right is not None
    ):
        left_rectification, right_rectification = rectified.rotation_left, rectified.rotation_right
        left_projection = right_projection = rectified.projection
    elif calibration.rotation is not None and calibration.translation is not None:
        left_rectification, right_rectification, left_projection, right_projection, *_ = cv2.stereoRectify(
            left_matrix,
            left_distortion,
            right_matrix,
            right_distortion,
            size,
            calibration.rotation.astype(np.float64).reshape(3, 3),
            calibration.translation.astype(np.float64).reshape(3),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
    else:
        return None
    left_map = cv2.initUndistortRectifyMap(
        left_matrix, left_distortion, left_rectification, left_projection, size, cv2.CV_32FC1
    )
    right_map = cv2.initUndistortRectifyMap(
        right_matrix, right_distortion, right_rectification, right_projection, size, cv2.CV_32FC1
    )
    return left_map, right_map


def load_remap_tables(calibration: CalibrationData, *, tolerance_px: float = 1.5) -> RemapTables | None:
    """Load the vendor PFM remap tables, validating their orientation.

    The tables are compared with maps computed from the JSON model. Whichever
    vertical orientation matches within ``tolerance_px`` wins; if neither
    does the tables are rejected and the caller falls back to the computed
    maps.
    """
    if not calibration.has_remap_tables or calibration.remap_directory is None:
        return None
    directory = calibration.remap_directory
    try:
        tables = {
            side: tuple(read_pfm(directory / name) for name in names)
            for side, names in REMAP_FILE_NAMES.items()
        }
    except (OSError, ValueError) as exc:
        logger.warning("无法读取重映射表：%s", exc)
        return None
    left_x, left_y = tables["left"]
    right_x, right_y = tables["right"]
    if not (left_x.shape == left_y.shape == right_x.shape == right_y.shape):
        logger.warning("重映射表尺寸不一致，忽略：%s", directory)
        return None
    height, width = left_x.shape
    # Vendor tables belong to exactly one rectification set (the underwater
    # one for dual-set files). Try the active set first, then the others, and
    # remember which one matched so callers use the tables only in that mode.
    candidates = [calibration] + [
        calibration.with_rectify_mode(mode)
        for mode in calibration.available_rectify_modes
        if mode != calibration.rectify_mode
    ]
    best: tuple[float, bool, str] | None = None
    for candidate in candidates:
        computed = compute_rectify_maps(candidate, (width, height))
        if computed is None:
            continue
        (cx_map, cy_map), _right = computed
        as_is = float(np.nanmean(np.abs(cx_map - left_x)) + np.nanmean(np.abs(cy_map - left_y))) / 2.0
        upside_down = float(
            np.nanmean(np.abs(cx_map - left_x[::-1])) + np.nanmean(np.abs(cy_map - left_y[::-1]))
        ) / 2.0
        error = min(as_is, upside_down)
        if best is None or error < best[0]:
            best = (error, upside_down < as_is, candidate.rectify_mode)
        if error <= tolerance_px:
            break
    flipped = False
    error = float("nan")
    mode = calibration.rectify_mode
    if best is not None:
        error, flipped, mode = best
        if error > tolerance_px:
            logger.warning(
                "重映射表与标定 JSON 不一致（平均偏差 %.2f px），改用计算映射：%s",
                error,
                directory,
            )
            return None
    if flipped:
        left_x, left_y, right_x, right_y = (table[::-1] for table in (left_x, left_y, right_x, right_y))
    logger.info("已加载重映射表 %s（模式 %s，flipped=%s，偏差 %.3f px）", directory, mode, flipped, error)
    return RemapTables(
        np.ascontiguousarray(left_x),
        np.ascontiguousarray(left_y),
        np.ascontiguousarray(right_x),
        np.ascontiguousarray(right_y),
        flipped,
        error,
        mode,
    )


def epiline_for_point(
    fundamental: np.ndarray,
    x: float,
    y: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    *,
    transpose: bool = False,
) -> tuple[float, float, float, float] | None:
    """Return normalized target-image endpoints for the corresponding epipolar line."""
    matrix = fundamental.T if transpose else fundamental
    line = matrix @ np.array([x * source_width, y * source_height, 1.0])
    a, b, c = (float(value) for value in line)
    if abs(b) > 1e-12:
        y0 = -c / b
        y1 = -(a * (target_width - 1) + c) / b
        return (0.0, y0 / target_height, 1.0, y1 / target_height)
    if abs(a) > 1e-12:
        line_x = -c / a / target_width
        return (line_x, 0.0, line_x, 1.0)
    return None
