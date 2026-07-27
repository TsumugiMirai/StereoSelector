from __future__ import annotations

import json
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    matrix: np.ndarray
    distortion: np.ndarray
    width: int | None = None
    height: int | None = None

    def point_from_depth(self, x: float, y: float, depth: float) -> tuple[float, float, float]:
        fx, fy = float(self.matrix[0, 0]), float(self.matrix[1, 1])
        cx, cy = float(self.matrix[0, 2]), float(self.matrix[1, 2])
        if fx == 0 or fy == 0:
            raise ValueError("相机焦距不能为零")
        distorted_x = (x - cx) / fx
        distorted_y = (y - cy) / fy
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
class CalibrationData:
    source: Path
    left: CameraCalibration | None
    right: CameraCalibration | None
    rotation: np.ndarray | None
    translation: np.ndarray | None
    fundamental: np.ndarray | None
    cloud_rotation: np.ndarray | None
    cloud_translation: np.ndarray | None

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
        if self.cloud_rotation is not None and self.cloud_translation is not None:
            extras.append("点云外参")
        return " · ".join([cameras, *extras])


@dataclass(frozen=True)
class CalibrationOption:
    id: str
    label: str
    path: Path
    builtin: bool = False


_BUILTIN_CALIBRATIONS = (
    ("builtin:libra2000", "libra2000", "calib2000_004_b"),
    ("builtin:libra3000", "libra3000", "calib3000_003"),
)


def builtin_calibration_options() -> list[CalibrationOption]:
    roots = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        roots.append(Path(sys._MEIPASS) / "calibration")
    roots.extend(
        (
            Path(__file__).resolve().parents[2] / "calibration",
            Path.cwd() / "calibration",
        )
    )
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
    if matrix is None:
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
        width=int(width_value) if width_value is not None else None,
        height=int(height_value) if height_value is not None else None,
    )


def load_calibration(path: Path) -> CalibrationData:
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
        if translation.size != 3:
            translation = None
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
        fundamental = (
            np.linalg.inv(right.matrix).T
            @ skew
            @ rotation
            @ np.linalg.inv(left.matrix)
        )
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
    return CalibrationData(
        path,
        left,
        right,
        rotation,
        translation,
        fundamental,
        cloud_rotation,
        cloud_translation,
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
