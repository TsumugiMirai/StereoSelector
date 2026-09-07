"""Depth units, rectified camera frame, vendor remap tables and per-project calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stereo_selector.calibration import (
    CameraCalibration,
    compute_rectify_maps,
    discover_project_calibrations,
    load_calibration,
    load_remap_tables,
    read_pfm,
)
from stereo_selector.units import (
    depth_is_valid,
    depth_to_meters,
    format_depth,
    format_length,
    normalize_depth_unit,
    resolve_depth_unit,
)

TEST1 = Path("C:/Users/admin/Desktop/test1")


def _vendor_document(fx: float = 450.0, baseline_mm: float = -49.9) -> dict:
    return {
        "camera_left": {
            "rows": 480,
            "cols": 640,
            "camera_matrix": {"data": [353.0, 0, 318.2, 0, 351.4, 232.0, 0, 0, 1]},
            "distortion_coeffs": [0.1, -0.05, 0.0, 0.0, 0.01, 0.2, -0.1, -0.05, 0, 0, 0, 0, 0, 0],
        },
        "camera_right": {
            "rows": 480,
            "cols": 640,
            "camera_matrix": {"data": [356.0, 0, 315.6, 0, 354.8, 235.4, 0, 0, 1]},
            "distortion_coeffs": [0.05, 0.1, 0.0, 0.0, 0.02, 0.1, 0.2, 0.05, 0, 0, 0, 0, 0, 0],
        },
        "extrinsic_l_r": {
            "rotation_matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "translation_vector": [baseline_mm, 0.7, 0.99],
        },
        "camera_rectify": {
            "rows": 480,
            "cols": 640,
            "fx": fx,
            "fy": fx,
            "cx": 323.5,
            "cy": 234.8,
            "rotation_left": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "rotation_right": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        },
        "camera_rgb": {
            "rows": 0,
            "cols": 0,
            "camera_matrix": {"data": [0] * 9},
            "distortion_coeffs": [0] * 14,
        },
        "extrinsic": {"rotation_matrix": [0] * 9, "translation_vector": [0, 0, 0]},
    }


def test_depth_unit_resolution_and_conversion() -> None:
    assert normalize_depth_unit("MM") == "mm"
    assert normalize_depth_unit("bogus") == "auto"
    assert resolve_depth_unit(np.uint16, "auto") == "mm"
    assert resolve_depth_unit(np.float32, "auto") == "m"
    assert resolve_depth_unit(np.float32, "mm") == "mm"
    assert depth_to_meters(1500, np.uint16, "auto") == pytest.approx(1.5)
    assert depth_to_meters(1.5, np.float32, "auto") == pytest.approx(1.5)
    assert depth_to_meters(1500, np.uint16, "m") == pytest.approx(1500.0)
    assert "1500 mm" in format_depth(1500, np.uint16, "auto")
    assert format_length(0.25) == "250.0 mm"
    assert format_length(2.5) == "2.500 m"


def test_saturated_and_hole_depths_are_invalid() -> None:
    assert depth_is_valid(1200, np.uint16)
    assert not depth_is_valid(0, np.uint16)
    assert not depth_is_valid(65535, np.uint16)
    assert not depth_is_valid(float("nan"), np.float32)
    assert depth_is_valid(65535.0, np.float32)  # floats have no saturation code


def test_pinhole_camera_skips_distortion_iteration() -> None:
    camera = CameraCalibration(
        matrix=np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1]]),
        distortion=np.zeros(5),
    )
    assert camera.is_pinhole
    np.testing.assert_allclose(camera.point_from_depth(720, 640, 2.0), [2.0, 2.0, 2.0])


def test_vendor_json_exposes_rectified_frame_and_millimetre_baseline(tmp_path: Path) -> None:
    path = tmp_path / "calibration_param.json"
    path.write_text(json.dumps(_vendor_document()), encoding="utf-8")

    calibration = load_calibration(path)

    assert calibration.left is not None and calibration.left.width == 640
    assert calibration.rectified is not None
    assert calibration.rectified.fx == 450.0
    assert calibration.translation_unit == "mm"
    assert calibration.baseline_m == pytest.approx(0.0499, abs=1e-3)
    depth_camera = calibration.depth_camera()
    assert depth_camera is not None and depth_camera.is_pinhole
    # Depth readouts use the rectified focal length, not the raw 353 px one.
    x, _y, z = depth_camera.point_from_depth(323.5 + 450.0, 234.8, 1.0)
    assert (x, z) == pytest.approx((1.0, 1.0))
    assert "校正参数" in calibration.summary
    assert "校正左目" in calibration.details
    assert not calibration.has_remap_tables
    maps = compute_rectify_maps(calibration, (640, 480))
    assert maps is not None and maps[0][0].shape == (480, 640)


def test_metric_calibration_keeps_metres_and_raw_left_depth_frame(tmp_path: Path) -> None:
    path = tmp_path / "stereo.json"
    path.write_text(
        json.dumps(
            {
                "left": {"camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]], "distortion": [0, 0, 0, 0, 0]},
                "right": {"camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]]},
                "left_to_right": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [-0.12, 0, 0]},
                "cloud_to_left": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [0.1, 0, 0]},
            }
        ),
        encoding="utf-8",
    )
    calibration = load_calibration(path)
    assert calibration.translation_unit == "m"
    assert calibration.baseline_m == pytest.approx(0.12)
    assert calibration.rectified is None
    assert calibration.depth_camera() is calibration.left
    np.testing.assert_allclose(calibration.cloud_translation_m, [0.1, 0, 0])


def test_millimetre_cloud_translation_is_converted(tmp_path: Path) -> None:
    document = _vendor_document()
    document["cloud_to_left"] = {"rotation_matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation_vector": [10, 0, 0]}
    path = tmp_path / "calibration_param.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    calibration = load_calibration(path)
    np.testing.assert_allclose(calibration.cloud_translation_m, [0.01, 0, 0])


def _write_pfm(path: Path, array: np.ndarray) -> None:
    height, width = array.shape
    header = f"Pf\n{width} {height}\n-1.0\n".encode("ascii")
    path.write_bytes(header + array.astype("<f4").tobytes())


def test_read_pfm_round_trip(tmp_path: Path) -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "map.pfm"
    _write_pfm(path, array)
    np.testing.assert_array_equal(read_pfm(path), array)


def test_remap_tables_are_validated_and_flipped_to_match_calibration(tmp_path: Path) -> None:
    path = tmp_path / "calibration_param.json"
    path.write_text(json.dumps(_vendor_document()), encoding="utf-8")
    calibration = load_calibration(path)
    (left_x, left_y), (right_x, right_y) = compute_rectify_maps(calibration, (640, 480))
    # Vendor writes bottom-up rows; loader must detect and undo the flip.
    for name, table in (
        ("map_left_x.pfm", left_x[::-1]),
        ("map_left_y.pfm", left_y[::-1]),
        ("map_right_x.pfm", right_x[::-1]),
        ("map_right_y.pfm", right_y[::-1]),
    ):
        _write_pfm(tmp_path / name, table)
    calibration = load_calibration(path)
    assert calibration.has_remap_tables
    assert "重映射表" in calibration.summary

    tables = load_remap_tables(calibration)

    assert tables is not None
    assert tables.flipped
    assert tables.error_px < 1e-3
    assert tables.mode == "air"  # single-set file without any uw hint is treated as in-air
    np.testing.assert_allclose(tables.left_x, left_x)
    np.testing.assert_allclose(tables.right_y, right_y)


def test_remap_tables_bind_to_the_matching_mode_in_dual_set_files(tmp_path: Path) -> None:
    document = _vendor_document()
    document["camera_rectify_air"] = {"rows": 480, "cols": 640, "fx": 318.0, "fy": 318.0, "cx": 333.2, "cy": 225.4}
    path = tmp_path / "calibration_param.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    underwater = load_calibration(path, "underwater")
    (left_x, left_y), (right_x, right_y) = compute_rectify_maps(underwater, (640, 480))
    for name, table in (
        ("map_left_x.pfm", left_x),
        ("map_left_y.pfm", left_y),
        ("map_right_x.pfm", right_x),
        ("map_right_y.pfm", right_y),
    ):
        _write_pfm(tmp_path / name, table)
    air = load_calibration(path)  # default air, tables belong to uw
    tables = load_remap_tables(air)
    assert tables is not None
    assert tables.mode == "underwater"
    assert not tables.flipped
    assert tables.error_px < 1e-3


def test_inconsistent_remap_tables_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "calibration_param.json"
    path.write_text(json.dumps(_vendor_document()), encoding="utf-8")
    for name in ("map_left_x.pfm", "map_left_y.pfm", "map_right_x.pfm", "map_right_y.pfm"):
        _write_pfm(tmp_path / name, np.full((480, 640), 5000.0, dtype=np.float32))
    calibration = load_calibration(path)
    assert load_remap_tables(calibration) is None


def test_project_local_calibrations_are_discovered(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    (root / "calib").mkdir(parents=True)
    (root / "calib" / "calibration_param.json").write_text(json.dumps(_vendor_document()), encoding="utf-8")
    (root / "calib" / "notes.json").write_text("{}", encoding="utf-8")
    (root / "left").mkdir()

    options = discover_project_calibrations(root)

    assert len(options) == 1
    assert options[0].project_local
    assert options[0].label.startswith("项目内")
    assert options[0].path == (root / "calib" / "calibration_param.json").resolve()
    assert discover_project_calibrations(tmp_path / "missing") == []


def test_air_rectification_is_preferred_over_underwater(tmp_path: Path) -> None:
    document = _vendor_document()
    document["uw_stereo_roi_offset"] = 5
    document["camera_rectify_air"] = {"rows": 480, "cols": 640, "fx": 318.0, "fy": 318.0, "cx": 333.2, "cy": 225.4}
    path = tmp_path / "calibration_param.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    calibration = load_calibration(path)

    assert calibration.rectified is not None
    assert calibration.rectified.mode == "air"
    assert calibration.rectified.fx == 318.0
    assert calibration.rectified.rotation_left is None
    assert calibration.available_rectify_modes == ["air", "underwater"]
    assert "空气" in calibration.details
    # Without air rotations the maps come from stereoRectify.
    maps = compute_rectify_maps(calibration, (640, 480))
    assert maps is not None
    # Projects captured underwater switch to the vendor uw set per project.
    underwater = calibration.with_rectify_mode("underwater")
    assert underwater.rectified.mode == "underwater"
    assert underwater.rectified.fx == 450.0
    assert underwater.rectified.rotation_left is not None
    assert "水下" in underwater.details
    assert load_calibration(path, "underwater").rectified.fx == 450.0
    assert calibration.with_rectify_mode("bogus") is calibration
    document.pop("camera_rectify_air")
    path.write_text(json.dumps(document), encoding="utf-8")
    only = load_calibration(path)
    assert only.rectified.mode == "underwater"
    assert only.available_rectify_modes == ["underwater"]


@pytest.mark.skipif(not (TEST1 / "calib" / "calibration_param.json").is_file(), reason="test1 dataset not present")
def test_real_test1_calibration_loads_rectified_frame() -> None:
    calibration = load_calibration(TEST1 / "calib" / "calibration_param.json")
    assert calibration.rectified is not None
    assert calibration.rectified.mode == "air"
    assert calibration.rectified.fx == pytest.approx(318.01, abs=0.01)
    # test1 is an underwater libra1000 capture: the uw set must be selectable.
    underwater = calibration.with_rectify_mode("underwater")
    assert underwater.rectified.fx == pytest.approx(450.96, abs=0.01)
    assert calibration.translation_unit == "mm"
    assert calibration.baseline_m == pytest.approx(0.0499, abs=1e-3)
    assert calibration.left.width == 640 and calibration.left.height == 480
