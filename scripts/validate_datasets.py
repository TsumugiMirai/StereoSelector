"""Functional validation of the viewer against real datasets (offscreen).

Usage:
    python scripts/validate_datasets.py --air D:/data/capture --underwater D:/data/uw_capture

``--air`` should point at an in-air capture (optionally with depth / PLY);
``--underwater`` at a capture whose calibration carries both ``camera_rectify``
and ``camera_rectify_air``. Either may be omitted. Screenshots and the
temporary settings file are written under ``build/validate_datasets``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from stereo_selector import app as app_module
from stereo_selector import workers
from stereo_selector.app import MainWindow
from stereo_selector.theme import palette_for, style_for

OUT = ROOT / "build" / "validate_datasets"
failures: list[str] = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        failures.append(message)


def settle(app, rounds=4):
    for _ in range(rounds):
        app.processEvents()
        workers.wait_for_all(8000)
    app.processEvents()


def wait_until(app, predicate, timeout=45.0):
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def tiles_ready(window):
    return bool(window.tiles) and all(t.showing_canvas() and not t.has_pending_work() for t in window.tiles.values())


def rectify_and_wait(app, window):
    window.rectify_button.setChecked(True)
    ok = wait_until(app, lambda: "校正后" in window.status_text.text() or "无法" in window.status_text.text())
    status = window.status_text.text()
    print("  rectify status:", status)
    return ok, status


def validate_air(app, window, root: Path):
    print(f"=== air capture: {root} ===")
    window.load_project(root)
    settle(app)
    check(window.dataset is not None, "project loads")
    ds = window.dataset
    print("  modalities:", ds.available_modalities, "samples:", len(ds.samples))
    check(len(ds.samples) > 0, "samples matched")
    complete = sum(len(s.files) == len(ds.available_modalities) for s in ds.samples)
    check(complete == len(ds.samples), f"all samples complete ({complete}/{len(ds.samples)})")
    check(wait_until(app, lambda: tiles_ready(window)), "first frame rendered")
    left = window.tiles["left"].image_data.image
    print(f"  left {left.width()}x{left.height()}")
    window.grab().save(str(OUT / "air_01_loaded.png"))

    has_depth = "depth_fsd" in ds.available_modalities
    if has_depth:
        check("范围" in window.depth_quality_label.text(), "depth quality shown")
        window.checkboxes["depth_fsd"].setChecked(True)
        check(wait_until(app, lambda: tiles_ready(window)), "depth view rendered")
        fields = window._cursor_value_fields("left", 0.5, 0.5)
        print("  cursor(no calib):", fields)
        check(any(unit in fields["depth"] for unit in ("mm", " m", "空洞", "饱和", "无效")), "depth readout has explicit unit or state")

    # Pick a built-in calibration whose resolution matches the images.
    matching = next(
        (opt for opt in window.calibration_options if opt.builtin), None
    )
    from stereo_selector.calibration import load_calibration
    for opt in window.calibration_options:
        if not opt.builtin:
            continue
        cal = load_calibration(opt.path)
        if cal.left is not None and (cal.left.width, cal.left.height) == (left.width(), left.height()):
            matching = opt
            break
    if matching is None:
        print("  SKIP calibration checks: no built-in option")
        return
    window.calibration_picker.setCurrentIndex(window.calibration_picker.findData(matching.id))
    check(wait_until(app, lambda: tiles_ready(window)), f"tiles rebuilt with {matching.label}")
    cal = window.calibration
    check(cal is not None, "calibration active")
    modes = cal.available_rectify_modes
    print("  rectify modes:", modes, "remap tables:", cal.has_remap_tables)
    check(window.rectify_mode_picker.isVisible() == (len(modes) > 1), "mode picker visibility matches calibration")
    check("不一致" not in window.status_text.text(), "no resolution mismatch warning")

    if has_depth:
        fields = window._cursor_value_fields("left", 0.5, 0.5)
        print("  cursor(calib):", fields)
        check(fields["xyz"].endswith(" m") or "需" in fields["xyz"] or fields["xyz"] == "", "xyz in metres when depth valid")
        captured = {}
        original = window.inspector.set_line_measurement
        window.inspector.set_line_measurement = lambda pixels, physical=None, endpoint_depths=None, note="": captured.update(
            pixels=pixels, physical=physical, depths=endpoint_depths, note=note
        )
        window._media_selection_changed("left", "line", (0.45, 0.5, 0.55, 0.5))
        window.inspector.set_line_measurement = original
        print("  line:", captured)
        check(captured.get("physical") is not None or captured.get("note"), "line measurement reports length or reason")

    ok, status = rectify_and_wait(app, window)
    check(ok and "校正后" in status, "rectified pair shown (air)")
    window.grab().save(str(OUT / "air_02_rectified.png"))
    window.rectify_button.setChecked(False)
    settle(app, 2)
    if "underwater" in modes:
        window.rectify_mode_picker.setCurrentIndex(window.rectify_mode_picker.findData("underwater"))
        check(wait_until(app, lambda: tiles_ready(window)), "tiles rebuilt after switching to underwater")
        ok, status = rectify_and_wait(app, window)
        check(ok and "校正后" in status, "rectified pair shown (underwater)")
        if cal.has_remap_tables:
            check("重映射表" in status, "underwater mode uses vendor pfm remap tables")
        window.grab().save(str(OUT / "air_03_rectified_underwater.png"))
        window.rectify_button.setChecked(False)
        settle(app, 2)
        window.rectify_mode_picker.setCurrentIndex(window.rectify_mode_picker.findData("air"))
        check(wait_until(app, lambda: tiles_ready(window)), "back to air")

    if "ply" in ds.available_modalities:
        window.checkboxes["ply"].setChecked(True)
        tile = window.tiles.get("ply")
        if tile is not None and tile.cloud_canvas is not None and tile.cloud_canvas.view is not None:
            check(wait_until(app, lambda: tiles_ready(window), timeout=90), f"point cloud rendered ({tile.meta.text()})")
            cloud = tile.cloud_canvas.current_cloud
            if cloud is not None and len(cloud.points):
                z = cloud.points[:, 2]
                print(f"  cloud Z range {z.min():.3f}..{z.max():.3f}")
                check(float(z.max()) <= window.preferences.cloud_z_max + 1e-6, "cloud Z within distance filter")
            window.grab().save(str(OUT / "air_04_point_cloud.png"))
        else:
            print("  SKIP point cloud: OpenGL view unavailable")
        window.checkboxes["ply"].setChecked(False)
    if has_depth:
        window.checkboxes["depth_fsd"].setChecked(False)
    settle(app, 2)

    window.last_sample()
    check(wait_until(app, lambda: tiles_ready(window)), "last frame rendered")
    check(window.current_index == len(ds.samples) - 1, f"last index {window.current_index}")
    window.first_sample()
    settle(app, 2)
    if len(ds.samples) > 1:
        window.toggle_playback()
        check(window._playback_timer.isActive(), "playback started")
        time.sleep(1.5)
        settle(app, 2)
        window.toggle_playback()
        check(window.current_index >= 1, f"playback advanced to {window.current_index}")

    window.crosshair_button.setChecked(True)
    window.epiline_button.setChecked(True)
    window._media_cursor_moved("left", 0.5, 0.5)
    settle(app, 2)
    text = window.cursor_info.text()
    print("  cursor:", text)
    check("RGB" in text and "视差" in text, "cursor readout has RGB and disparity")
    check(window.tiles["right"].image_canvas._epiline.isVisible(), "epiline drawn")
    window.grab().save(str(OUT / "air_05_crosshair.png"))
    window.crosshair_button.setChecked(False)

    window._select_activity("statistics")
    settle(app, 2)
    check("均值" in window.inspector.stats_label.text(), "statistics computed")
    window.close_inspector()
    target = OUT / "air_left_adjusted.png"
    app_module.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(target), "PNG"))
    window.tiles["left"].set_display_settings(gamma=1.4)
    settle(app, 2)
    window.export_adjusted_image()
    check(target.is_file() and target.stat().st_size > 1_000, "adjusted full-resolution export written")
    with Image.open(target) as im:
        check(im.size == (left.width(), left.height()), f"export size {im.size}")
    window.tiles["left"].set_display_settings(gamma=1.0)
    window.calibration_picker.setCurrentIndex(0)
    settle(app, 2)


def validate_underwater(app, window, root: Path, other: Path | None):
    print(f"=== underwater capture: {root} ===")
    window.load_project(root)
    settle(app)
    check(window.dataset is not None, "project loads")
    check(wait_until(app, lambda: tiles_ready(window)), "pair rendered")
    cal = window.calibration
    check(cal is not None, "calibration auto-selected from project")
    if cal is None:
        return
    check(cal.available_rectify_modes == ["air", "underwater"], f"modes {cal.available_rectify_modes}")
    check(window.rectify_mode_picker.isVisible(), "rectify mode picker visible")
    check(cal.rectify_mode == "air", "defaults to air")
    window.rectify_mode_picker.setCurrentIndex(window.rectify_mode_picker.findData("underwater"))
    check(wait_until(app, lambda: tiles_ready(window)), "tiles rebuilt after mode switch")
    cal = window.calibration
    check(cal.rectify_mode == "underwater", "underwater active")
    print("  details:", cal.details.replace("\n", " | "))
    prefix = window._matching_settings_prefix(root.resolve())
    check(window.settings.value(f"{prefix}/rectify_mode") == "underwater", "mode persisted per project")
    ok, status = rectify_and_wait(app, window)
    check(ok and "校正后" in status, "rectified pair shown (underwater)")
    window.grab().save(str(OUT / "uw_01_rectified.png"))
    window.rectify_button.setChecked(False)
    settle(app, 2)
    if other is not None:
        window.load_project(other)
        settle(app)
        check(window.rectify_mode == "air", "other project keeps air")
        window.load_project(root)
        settle(app)
        check(window.rectify_mode == "underwater" and window.calibration.rectify_mode == "underwater", "underwater restored on reopen")
    window.rectify_mode_picker.setCurrentIndex(window.rectify_mode_picker.findData("air"))
    settle(app, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--air", type=Path, help="in-air capture folder")
    parser.add_argument("--underwater", type=Path, help="underwater capture folder with dual-set calibration")
    args = parser.parse_args()
    if args.air is None and args.underwater is None:
        parser.error("pass --air and/or --underwater")
    OUT.mkdir(parents=True, exist_ok=True)
    settings_path = OUT / "settings.ini"
    settings_path.unlink(missing_ok=True)
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    app_module.QSettings = lambda *a: settings
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(style_for("dark"))
    app.setPalette(palette_for("dark"))
    workers.configure_pools()
    window = MainWindow()
    window.resize(1520, 920)
    window.show()
    try:
        if args.air is not None:
            validate_air(app, window, args.air.resolve())
        if args.underwater is not None:
            validate_underwater(app, window, args.underwater.resolve(), args.air.resolve() if args.air else None)
    finally:
        window.close()
        settle(app)
    print(f"\n{len(failures)} failure(s); screenshots in {OUT}")
    for item in failures:
        print("  -", item)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
