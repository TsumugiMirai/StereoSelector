from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QVector4D
from PySide6.QtWidgets import QApplication

from stereo_selector import widgets as widgets_module
from stereo_selector.media import (
    PointCloudData,
    analyze_depth,
    analyze_image,
    colorize_scalar,
    load_image,
    load_image_data,
    load_point_cloud,
    project_camera_points,
    render_image_values,
)
from stereo_selector.widgets import (
    CAMERA_ALIGNED_AZIMUTH,
    CAMERA_ALIGNED_ELEVATION,
    MediaLoadWorker,
    MediaTile,
    PinholeProjectionMixin,
    PointCloudCanvas,
    camera_points_to_gl,
)

TEST_INTRINSICS = np.array(
    [[100.0, 0.0, 500.0], [0.0, 100.0, 500.0], [0.0, 0.0, 1.0]],
    dtype=float,
)
TEST_IMAGE_SIZE = (1000, 1000)


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


def test_depth_quality_counts_saturated_uint16_as_extreme() -> None:
    values = np.full((2, 3), 65535, dtype=np.uint16)

    stats = analyze_depth(values)

    assert stats.maximum == 65535.0
    assert stats.extreme_threshold == 65535.0
    assert stats.extreme_ratio == 1.0


def test_image_analysis_reports_statistics_and_rgb_histograms() -> None:
    values = np.array(
        [[[0, 10, 20], [30, 40, 50]], [[60, 70, 80], [90, 100, 110]]],
        dtype=np.uint8,
    )

    stats = analyze_image(values, bins=16)

    assert stats.minimum == 0
    assert stats.maximum == 110
    assert stats.pixel_count == 4
    assert stats.valid_ratio == 1.0
    assert len(stats.channel_histograms) == 3
    assert all(histogram.sum() == 4 for histogram in stats.channel_histograms)


def test_depth_render_supports_lut_range_and_invalid_highlights() -> None:
    values = np.array([[0, 1000], [2000, 65535]], dtype=np.uint16)

    image = render_image_values(
        values,
        color_map="turbo",
        display_range=(1000, 2000),
        highlight_invalid=True,
    )

    assert image.pixelColor(0, 0).getRgb()[:3] == (0, 210, 255)
    assert image.pixelColor(1, 1).getRgb()[:3] == (255, 70, 190)
    assert image.pixelColor(0, 1) != image.pixelColor(1, 0)
    assert colorize_scalar(np.array([0.0, 1.0]), "viridis").shape == (2, 3)


def test_constant_normal_depth_is_not_marked_as_extreme() -> None:
    stats = analyze_depth(np.full((2, 3), 1000, dtype=np.uint16))

    assert stats.extreme_threshold == 1000.0
    assert stats.extreme_ratio == 0.0


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
    cloud = load_point_cloud(
        path,
        intrinsics=TEST_INTRINSICS,
        image_size=TEST_IMAGE_SIZE,
        grid=1,
        tau_rel=0.0,
        occlusion=False,
    )
    assert cloud.original_count == 2
    assert cloud.filtered_count == 2
    # Loaded points remain in camera coordinates: X right, Y down, Z forward.
    red_index = np.flatnonzero(np.all(cloud.points == [1, 2, 3], axis=1))[0]
    np.testing.assert_allclose(cloud.points[red_index], [1, 2, 3])
    np.testing.assert_allclose(
        camera_points_to_gl(cloud.points)[red_index],
        [1, 3, -2],
    )
    np.testing.assert_allclose(cloud.colors[red_index], [1, 0, 0, 1])


def test_point_cloud_initial_camera_looks_in_rgb_capture_direction() -> None:
    captured: dict[str, object] = {}

    class CameraView:
        def setCameraPosition(self, **kwargs) -> None:
            captured.update(kwargs)

    canvas = SimpleNamespace(
        view=CameraView(),
        _center=np.array([0.0, 7.95, 0.0], dtype=np.float32),
        _distance=8.0,
    )
    PointCloudCanvas.reset_view(canvas)

    assert captured["azimuth"] == CAMERA_ALIGNED_AZIMUTH == -90.0
    assert captured["elevation"] == CAMERA_ALIGNED_ELEVATION == 0.0
    center = captured["pos"]
    np.testing.assert_allclose(
        (center.x(), center.y(), center.z()),
        (0.0, 7.95, 0.0),
    )
    # At azimuth -90°, camera Y = center Y - distance = -cam_offset.
    assert np.isclose(center.y() - captured["distance"], -0.05)


def test_point_cloud_can_be_transformed_to_left_camera_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "cloud.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n1 2 3\n",
        encoding="ascii",
    )
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    cloud = load_point_cloud(
        path,
        rotation=rotation,
        translation=np.array([1, 0, 0]),
        intrinsics=TEST_INTRINSICS,
        image_size=TEST_IMAGE_SIZE,
        grid=1,
        tau_rel=0.0,
        occlusion=False,
    )

    # Source [1,2,3] -> left-camera coordinates [-1,1,3].
    np.testing.assert_allclose(cloud.points[0], [-1, 1, 3])


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

    cloud = load_point_cloud(
        path,
        max_points=3,
        intrinsics=TEST_INTRINSICS,
        image_size=TEST_IMAGE_SIZE,
        grid=1,
        tau_rel=0.0,
        occlusion=False,
    )
    assert cloud.original_count == 10
    assert cloud.filtered_count == 9
    assert cloud.points.shape == (3, 3)
    assert cloud.colors.shape == (3, 4)
    np.testing.assert_allclose(cloud.points[[0, -1], 0], [0, 8])
    np.testing.assert_allclose(cloud.colors[0], [1.0, 32768 / 65535, 0.0, 1.0], rtol=1e-5)


def test_point_cloud_distance_filter_uses_transformed_camera_z(tmp_path: Path) -> None:
    path = tmp_path / "distance_filter.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 5\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
        "0 0 1\n"
        "1 2 9\n"
        "2 3 10\n"
        "3 4 10.1\n"
        "4 5 -1\n",
        encoding="ascii",
    )

    cloud = load_point_cloud(
        path,
        max_distance=10.0,
        translation=np.array([0.0, 0.0, 1.0]),
        intrinsics=TEST_INTRINSICS,
        image_size=TEST_IMAGE_SIZE,
        grid=1,
        tau_rel=0.0,
        occlusion=False,
    )

    assert cloud.original_count == 5
    assert cloud.filtered_count == 2
    assert cloud.max_distance == 10.0
    np.testing.assert_allclose(cloud.points[:, 2], [2.0, 10.0])


def test_point_cloud_projection_uses_virtual_camera_offset() -> None:
    points = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    intrinsics = np.array(
        [[200.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    uv, projected_z = project_camera_points(points, intrinsics, cam_offset=1.0)

    np.testing.assert_allclose(projected_z, [5.0])
    np.testing.assert_allclose(uv[0], [360.0, 280.0])


def test_gl_projection_matches_pinhole_formula_with_letterboxing() -> None:
    class Projection(PinholeProjectionMixin):
        def update(self) -> None:
            pass

    view = Projection()
    intrinsics = np.array(
        [[200.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    view.configure_camera_projection(intrinsics, (640, 480), 100.0)
    matrix = view.projectionMatrix((0, 0, 800, 480), (0, 0, 800, 480))
    projected = matrix.map(QVector4D(1.0, -2.0, -5.0, 1.0))
    ndc_x = projected.x() / projected.w()
    ndc_y = projected.y() / projected.w()
    screen_x = (ndc_x + 1.0) * 400.0
    screen_y = (1.0 - ndc_y) * 240.0

    # The 640×480 image is centered inside the 800×480 viewport: x gains 80 px.
    np.testing.assert_allclose((screen_x, screen_y), (440.0, 280.0), atol=1e-4)


def test_point_cloud_grid_occlusion_keeps_nearest_and_sorts_far_to_near(
    tmp_path: Path,
) -> None:
    path = tmp_path / "occlusion.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
        "0 0 2\n"
        "0 0 4\n"
        "0.5 0 5\n"
        "-0.2 0 2\n",
        encoding="ascii",
    )
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    cloud = load_point_cloud(
        path,
        intrinsics=intrinsics,
        image_size=(100, 100),
        cam_offset=0.0,
        grid=5,
        tau_rel=0.0,
        occlusion=True,
    )

    assert cloud.filtered_count == 3
    np.testing.assert_allclose(cloud.points[:, 2], [5.0, 2.0, 2.0])
    assert np.count_nonzero(np.all(cloud.points == [0.0, 0.0, 2.0], axis=1)) == 1
    assert not np.any(np.all(cloud.points == [0.0, 0.0, 4.0], axis=1))


def test_larger_tau_rel_filters_more_flying_points(tmp_path: Path) -> None:
    path = tmp_path / "flying.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
        "-0.1 0.1 2\n"
        "0.3 0.1 2\n"
        "0.1 -0.1 2\n"
        "0.2 0.2 4\n",
        encoding="ascii",
    )
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    unfiltered = load_point_cloud(
        path,
        intrinsics=intrinsics,
        image_size=(100, 100),
        cam_offset=0.0,
        grid=10,
        tau_rel=0.0,
        occlusion=False,
    )
    strict = load_point_cloud(
        path,
        intrinsics=intrinsics,
        image_size=(100, 100),
        cam_offset=0.0,
        grid=10,
        tau_rel=0.9,
        occlusion=False,
    )

    assert strict.filtered_count < unfiltered.filtered_count
    assert not np.any(np.isclose(strict.points[:, 2], 4.0))


def test_point_cloud_load_honors_cancellation_before_io(tmp_path: Path) -> None:
    missing = tmp_path / "missing.ply"

    with np.testing.assert_raises(InterruptedError):
        load_point_cloud(missing, cancel_check=lambda: True)


def test_point_cloud_right_drag_pans_in_view_plane() -> None:
    pan_calls: list[tuple[float, float, float, str]] = []

    class View:
        def setCursor(self, cursor) -> None:
            self.cursor = cursor

        def unsetCursor(self) -> None:
            self.cursor = None

        def pan(self, dx: float, dy: float, dz: float, *, relative: str) -> None:
            pan_calls.append((dx, dy, dz, relative))

    class MouseEvent:
        def __init__(
            self,
            event_type: QEvent.Type,
            position: tuple[float, float],
            *,
            button=Qt.MouseButton.NoButton,
            buttons=Qt.MouseButton.NoButton,
        ) -> None:
            self._type = event_type
            self._position = QPointF(*position)
            self._button = button
            self._buttons = buttons
            self.accepted = False

        def type(self):
            return self._type

        def position(self):
            return self._position

        def button(self):
            return self._button

        def buttons(self):
            return self._buttons

        def accept(self) -> None:
            self.accepted = True

    view = View()
    canvas = SimpleNamespace(view=view, _right_pan_position=None)
    press = MouseEvent(
        QEvent.Type.MouseButtonPress,
        (10.0, 20.0),
        button=Qt.MouseButton.RightButton,
        buttons=Qt.MouseButton.RightButton,
    )
    move = MouseEvent(
        QEvent.Type.MouseMove,
        (16.0, 15.0),
        buttons=Qt.MouseButton.RightButton,
    )
    release = MouseEvent(
        QEvent.Type.MouseButtonRelease,
        (16.0, 15.0),
        button=Qt.MouseButton.RightButton,
    )

    assert PointCloudCanvas.eventFilter(canvas, view, press)
    assert PointCloudCanvas.eventFilter(canvas, view, move)
    assert PointCloudCanvas.eventFilter(canvas, view, release)
    assert pan_calls == [(6.0, -5.0, 0.0, "view")]
    assert canvas._right_pan_position is None
    assert view.cursor is None


def test_point_cloud_xyz_clip_updates_rendered_subset() -> None:
    calls: list[dict[str, object]] = []

    class Scatter:
        def setData(self, **kwargs) -> None:
            calls.append(kwargs)

    cloud = PointCloudData(
        points=np.array(
            [[-1.0, 0.0, 1.0], [0.5, 0.5, 3.0], [2.0, 1.0, 8.0]],
            dtype=np.float32,
        ),
        colors=np.ones((3, 4), dtype=np.float32),
        original_count=3,
        filtered_count=3,
        max_distance=10.0,
    )
    canvas = SimpleNamespace(
        _cloud=cloud,
        _scatter=Scatter(),
        _color_mode="rgb",
        dot_radius=1.0,
        _colors_for_mode=lambda visible, _mode: visible.colors,
        _rendered_points=np.empty((0, 3), dtype=np.float32),
        _rendered_colors=np.empty((0, 4), dtype=np.float32),
        _rendered_source_colors=np.empty((0, 4), dtype=np.float32),
    )

    count = PointCloudCanvas.set_clip_ranges(
        canvas,
        (-2.0, 1.0),
        (-1.0, 0.75),
        (0.0, 5.0),
    )

    assert count == 2
    np.testing.assert_allclose(canvas._rendered_points[:, 2], [1.0, 3.0])
    assert calls[-1]["pos"].shape == (2, 3)


def test_point_cloud_screen_box_keeps_only_projected_points_inside() -> None:
    calls: list[dict[str, object]] = []
    emitted: list[int] = []

    class Scatter:
        def setData(self, **kwargs) -> None:
            calls.append(kwargs)

    cloud = PointCloudData(
        points=np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=np.float32),
        colors=np.ones((3, 4), dtype=np.float32),
        original_count=3,
        filtered_count=3,
        max_distance=10.0,
    )
    canvas = SimpleNamespace(
        _cloud=cloud,
        _scatter=Scatter(),
        _color_mode="rgb",
        dot_radius=1.0,
        _rendered_points=cloud.points.copy(),
        _rendered_source_colors=cloud.colors.copy(),
        _rendered_colors=cloud.colors.copy(),
        _colors_for_mode=lambda visible, _mode: visible.colors,
        _screen_projection=lambda: (
            np.array([0, 1, 2]),
            np.array([10.0, 30.0, 50.0]),
            np.array([10.0, 30.0, 50.0]),
            np.zeros((3, 3)),
        ),
        selection_changed=SimpleNamespace(emit=lambda count: emitted.append(count)),
    )

    count = PointCloudCanvas._select_box(
        canvas,
        QRectF(20.0, 20.0, 20.0, 20.0),
    )

    assert count == 1
    np.testing.assert_allclose(canvas._rendered_points[0], [1, 0, 2])
    assert emitted == [1]
    assert calls[-1]["pos"].shape == (1, 3)


def test_cancelled_media_worker_does_not_publish_result(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(path)
    worker = MediaLoadWorker(1, "left", path, 100)
    published: list[int] = []
    worker.signals.loaded.connect(lambda token, result: published.append(token))
    worker.cancel()
    worker.run()
    assert published == []


def test_prefetched_cloud_worker_is_reused_when_it_becomes_current(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "cloud.ply"
    path.write_text("", encoding="ascii")

    class FakePool:
        def __init__(self) -> None:
            self.queued: set[object] = set()
            self.starts: list[tuple[object, int]] = []

        def start(self, worker, priority: int = 0) -> None:
            self.queued.add(worker)
            self.starts.append((worker, priority))

        def tryTake(self, worker) -> bool:
            if worker not in self.queued:
                return False
            self.queued.remove(worker)
            return True

    pool = FakePool()
    monkeypatch.setattr(
        widgets_module,
        "QThreadPool",
        SimpleNamespace(globalInstance=lambda: pool),
    )
    tile = MediaTile("ply")

    tile.prefetch_file(path)
    prefetched_worker = pool.starts[0][0]
    tile.show_file(path)

    assert len({id(worker) for worker, _priority in pool.starts}) == 1
    assert pool.starts == [(prefetched_worker, -1), (prefetched_worker, 1)]
    assert len(tile._workers) == 1
    tile.dispose()
    app.processEvents()
