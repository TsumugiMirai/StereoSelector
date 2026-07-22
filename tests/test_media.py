from pathlib import Path

import numpy as np
from PIL import Image

from stereo_selector.media import load_image, load_point_cloud
from stereo_selector.widgets import MediaLoadWorker


def test_load_16bit_image(tmp_path: Path) -> None:
    path = tmp_path / "depth.png"
    Image.fromarray(np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)).save(path)
    image = load_image(path)
    assert image.width() == 2
    assert image.height() == 2


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
