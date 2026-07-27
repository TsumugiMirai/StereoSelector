from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stereo_selector.calibration import (
    builtin_calibration_options,
    epiline_for_point,
    load_calibration,
)


def test_json_calibration_loads_intrinsics_distortion_and_stereo_extrinsics(tmp_path: Path) -> None:
    path = tmp_path / "stereo.json"
    path.write_text(
        json.dumps(
            {
                "left": {
                    "width": 1280,
                    "height": 720,
                    "camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
                    "distortion": [-0.1, 0.01, 0, 0, 0],
                },
                "right": {
                    "camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
                    "distortion": [0, 0, 0, 0, 0],
                },
                "left_to_right": {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation": [-0.12, 0, 0],
                },
                "cloud_to_left": {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation": [0.1, 0, 0],
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = load_calibration(path)

    assert calibration.left is not None
    assert calibration.right is not None
    assert calibration.fundamental is not None
    np.testing.assert_allclose(calibration.left.point_from_depth(640, 360, 2), [0, 0, 2])
    assert calibration.left.distortion.shape == (5,)
    assert calibration.cloud_rotation is not None
    assert calibration.cloud_translation is not None
    assert calibration.summary == "双目内参 · 外参 · 基础矩阵 · 点云外参"


def test_opencv_yaml_matrix_format_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "camera.yaml"
    path.write_text(
        "%YAML:1.0\n"
        "M1: !!opencv-matrix\n"
        "  rows: 3\n"
        "  cols: 3\n"
        "  dt: d\n"
        "  data: [800, 0, 640, 0, 800, 360, 0, 0, 1]\n"
        "D1: !!opencv-matrix\n"
        "  rows: 1\n"
        "  cols: 5\n"
        "  dt: d\n"
        "  data: [0, 0, 0, 0, 0]\n",
        encoding="utf-8",
    )

    calibration = load_calibration(path)

    assert calibration.left is not None
    assert calibration.left.matrix[0, 0] == 800


def test_epiline_endpoints_are_normalized() -> None:
    fundamental = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    line = epiline_for_point(fundamental, 0.5, 0.25, 100, 100, 100, 100)
    assert line == (0.0, 0.25, 1.0, 0.25)


def test_project_ships_two_named_builtin_calibrations() -> None:
    options = builtin_calibration_options()

    assert [option.label for option in options] == ["libra2000", "libra3000"]
    assert all(option.builtin for option in options)
    for option in options:
        calibration = load_calibration(option.path)
        assert calibration.left is not None
        assert calibration.right is not None
        assert calibration.rotation is not None
        assert calibration.translation is not None
        assert calibration.left.distortion.size == 14
