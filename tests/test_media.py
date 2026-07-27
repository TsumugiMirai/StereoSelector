from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from stereo_selector.media import analyze_depth, load_image, load_image_data, load_point_cloud
from stereo_selector.widgets import (
    CAMERA_ALIGNED_AZIMUTH,
    CAMERA_ALIGNED_ELEVATION,
    MediaLoadWorker,
    PointCloudCanvas,
)


def test_load_16bit_image(tmp_path: Path) -> None:
    path = tmp_path / "depth.png"
    Image.fromarray(np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)).save(path)
    image = load_image(path)
    assert image.width() == 2
    assert image.height() == 2
    image_data = load_image_data(path)
    assert image_data.values.dtype == np.uint16
    assert image_data.values[1, 1] == 4000


def test_depth_quality_statistics_cover_invalid_zero_holes_and_extremes() -> None:
    values = np.array([[0.0, 1.0], [np.nan, 1000.0]], dtype=np.float32)
    stats = analyze_depth(values)

    assert stats.minimum == 1.0
    assert stats.maximum == 1000.0
    assert stats.invalid_ratio == 0.25
    assert stats.zero_ratio == 0.25
    assert stats.hole_ratio == 0.5


def test_point_cloud_camera_orientation(tmp_path: Path) -> None:
    path = tmp_path / "cloud.ply"
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
        "1 2 3 255 0 0\n"
        "-1 -2 4 0 255 0\n",
        encoding="ascii",
    )
    cloud = load_point_cloud(path)
    assert cloud.original_count == 2
    # Source (x, y, z) becomes OpenGL (x, z, -y).
    np.testing.assert_allclose(cloud.points[0], [1, 3, -2])
    np.testing.assert_allclose(cloud.colors[0], [1, 0, 0, 1])


def test_point_cloud_initial_camera_looks_in_rgb_capture_direction() -> None:
    captured: dict[str, object] = {}

    class CameraView:
        def setCameraPosition(self, **kwargs) -> None:
            captured.update(kwargs)

    canvas = SimpleNamespace(
        view=CameraView(),
        _center=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        _distance=8.0,
    )
    PointCloudCanvas.reset_view(canvas)

    assert captured["azimuth"] == CAMERA_ALIGNED_AZIMUTH == -90.0
    assert captured["elevation"] == CAMERA_ALIGNED_ELEVATION == 0.0
    center = captured["pos"]
    assert (center.x(), center.y(), center.z()) == (1.0, 2.0, 3.0)


def test_point_cloud_can_be_transformed_to_left_camera_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "cloud.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n1 2 3\n",
        encoding="ascii",
    )
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    cloud = load_point_cloud(path, rotation=rotation, translation=np.array([1, 0, 0]))

    # Source [1,2,3] -> left camera [-1,1,3] -> OpenGL [-1,3,-1].
    np.testing.assert_allclose(cloud.points[0], [-1, 3, -1])


def test_point_cloud_filters_before_sampling_and_normalizes_16bit_color(tmp_path: Path) -> None:
    path = tmp_path / "large_cloud.ply"
    rows = [f"{index} {index + 1} {index + 2} 65535 32768 0" for index in range(10)]
    rows.append("nan 0 0 65535 0 0")
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 11\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property ushort red\n"
        "property ushort green\n"
        "property ushort blue\n"
        "end_header\n"
        + "\n".join(rows)
        + "\n",
        encoding="ascii",
    )

    cloud = load_point_cloud(path, max_points=3)
    assert cloud.original_count == 10
    assert cloud.points.shape == (3, 3)
    assert cloud.colors.shape == (3, 4)
    np.testing.assert_allclose(cloud.points[[0, -1], 0], [0, 9])
    np.testing.assert_allclose(cloud.colors[0], [1.0, 32768 / 65535, 0.0, 1.0], rtol=1e-5)


def test_cancelled_media_worker_does_not_publish_result(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(path)
    worker = MediaLoadWorker(1, "left", path, 100)
    published: list[int] = []
    worker.signals.loaded.connect(lambda token, result: published.append(token))
    worker.cancel()
    worker.run()
    assert published == []
