from pathlib import Path

from stereo_selector.models import (
    DatasetScanner,
    discover_project_roots,
    normalized_sample_key,
)


def test_normalized_sample_key_matches_modalities() -> None:
    assert normalized_sample_key(Path("scene_a/left_rgb_000123.png")) == "scene_a/000123"
    assert normalized_sample_key(Path("scene_a/right-rgb-000123.jpg")) == "scene_a/000123"
    assert normalized_sample_key(Path("scene_a/depth_color_000123.tif")) == "scene_a/000123"
    assert normalized_sample_key(Path("1_ColorLeft_1016709146_0_0_0_0.png")) == "1_1016709146_0_0_0_0"
    assert normalized_sample_key(Path("1_ColorRight_1016709146_0_0_0_0.png")) == "1_1016709146_0_0_0_0"
    assert normalized_sample_key(Path("1_DepthColor_1016709146_0_0_0_0.jpg")) == "1_1016709146_0_0_0_0"


def test_scan_is_read_only_and_preserves_source_structure(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    for folder, filename in (
        ("left", "left_0001.png"),
        ("right", "right_0001.jpg"),
        ("depth_fsd", "depth_0001.tiff"),
        ("depth_color", "depth_color_0001.png"),
        ("ply", "0001.ply"),
    ):
        path = root / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    dataset = DatasetScanner().scan(root)
    assert dataset.available_modalities == ["left", "right", "depth_fsd", "depth_color", "ply"]
    assert len(dataset.samples) == 1
    assert set(dataset.samples[0].files) == set(dataset.available_modalities)

    assert not (tmp_path / "capture_select").exists()
    assert all(path.read_bytes() == b"test" for path in dataset.samples[0].files.values())


def test_recursive_scene_folders_match(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    for scene in ("scene_01", "scene_02"):
        for side in ("left", "right"):
            path = root / scene / side / f"{side}_42.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    dataset = DatasetScanner().scan(root)
    assert len(dataset.samples) == 2
    assert all(set(sample.files) == {"left", "right"} for sample in dataset.samples)


def test_camel_case_capture_names_form_complete_groups(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    patterns = {
        "left": "{i}_ColorLeft_{stamp}_0_0_0_0.png",
        "right": "{i}_ColorRight_{stamp}_0_0_0_0.png",
        "depth_fsd": "{i}_ColorLeft_{stamp}_0_0_0_0_depth.png",
        "depth_vis": "{i}_DepthColor_{stamp}_0_0_0_0.jpg",
        "ply": "{i}_ColorLeft_{stamp}_0_0_0_0_pointcloud.ply",
    }
    for index in range(1, 4):
        for folder, pattern in patterns.items():
            path = root / folder / pattern.format(i=index, stamp=1016700000 + index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    dataset = DatasetScanner().scan(root)
    assert len(dataset.samples) == 3
    assert all(len(sample.files) == 5 for sample in dataset.samples)


def test_capture_ordinal_matches_left_and_right_with_split_timestamps(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    patterns = {
        "left": "{i}_Left_remap_{left_stamp}_0_0_0_0.png",
        "right": "{i}_Right_remap_{right_stamp}_0_0_0_0.png",
        "depth_fsd": "{i}_Left_remap_{left_stamp}_0_0_0_0_depth.png",
        "depth_vis": "{i}_Left_remap_{left_stamp}_0_0_0_0_depth_vis.png",
        "ply": "{i}_Left_remap_{left_stamp}_0_0_0_0_pointcloud.ply",
    }
    for index in range(3):
        values = {
            "i": index,
            "left_stamp": 1783241071537952 + index * 1000,
            "right_stamp": 1783241071538199 + index * 1000,
        }
        for folder, pattern in patterns.items():
            path = root / folder / pattern.format(**values)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    dataset = DatasetScanner().scan(root)

    assert len(dataset.samples) == 3
    assert all(set(sample.files) == set(dataset.available_modalities) for sample in dataset.samples)


def test_manual_directories_can_force_positional_matching(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    left_dir = root / "camera_a"
    right_dir = root / "camera_b"
    for name in ("left_2.png", "left_10.png"):
        path = left_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"left")
    for name in ("unrelated_a.png", "unrelated_b.png"):
        path = right_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"right")

    dataset = DatasetScanner().scan(
        root,
        manual_dirs={"left": left_dir, "right": right_dir},
        force_order=True,
    )

    assert dataset.force_order
    assert len(dataset.samples) == 2
    assert [sample.files["left"].name for sample in dataset.samples] == ["left_2.png", "left_10.png"]
    assert [sample.files["right"].name for sample in dataset.samples] == ["unrelated_a.png", "unrelated_b.png"]
    assert all(set(sample.files) == {"left", "right"} for sample in dataset.samples)


def test_unconventional_partial_folders_are_inferred_from_names_and_formats(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    files = (
        ("Camera-L", "shot_01.png", b"rgb"),
        ("Camera-R", "shot_01.png", b"rgb"),
        ("metric_maps", "shot_01.tiff", b"depth"),
        ("colored-depth-preview", "shot_01.jpg", b"color"),
        ("reconstruction_cloud", "shot_01.ply", b"ply"),
    )
    for folder, name, content in files:
        path = root / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if folder == "metric_maps":
            import numpy as np
            from PIL import Image

            Image.fromarray(np.array([[1000]], dtype=np.uint16)).save(path)
        else:
            path.write_bytes(content)

    dataset = DatasetScanner().scan(root)

    assert dataset.available_modalities == ["left", "right", "depth_fsd", "depth_color", "ply"]
    assert all(len(dataset.files[name]) == 1 for name in dataset.available_modalities)


def test_project_collection_discovers_switchable_children_in_natural_order(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "daily_capture"
    for project_name in ("run_10", "run_2"):
        for folder in ("camera-left-stream", "camera-right-stream"):
            path = collection / project_name / "sensors" / folder / "frame_0001.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"rgb")
    exported = collection / "run_2_select"
    for folder in ("left", "right"):
        path = exported / folder / "frame_0001.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"copy")

    projects = discover_project_roots(collection)

    assert [path.name for path in projects] == ["run_2", "run_10"]


def test_single_project_with_arbitrary_view_folder_names_stays_one_project(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    for folder in ("device-camera-left", "device-camera-right"):
        path = root / "streams" / folder / "frame_0001.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"rgb")

    assert discover_project_roots(root) == [root.resolve()]
    dataset = DatasetScanner().scan(root)
    assert dataset.available_modalities == ["left", "right"]
    assert len(dataset.samples) == 1
