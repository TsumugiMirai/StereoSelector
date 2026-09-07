"""Viewer UI smoke check; screenshots and settings live in a temp folder.

Usage: python scripts/check_ui_workflows.py C:/data/project
Set QT_QPA_PLATFORM=offscreen when no desktop is available.

The source tree is inventoried before and after the run to prove the viewer
is read-only. Large media files are checked by size and mtime, not hashed.
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from stereo_selector import workers
from stereo_selector.app import MainWindow
from stereo_selector.bootstrap import application_font


def tree_snapshot(root: Path) -> dict[str, tuple]:
    """Inventory without following directory links or reading large media."""
    if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
        return {".": ("link", os.readlink(root))}
    if not root.exists():
        return {".": ("absent",)}
    if not root.is_dir():
        stat = root.stat()
        return {".": ("file", stat.st_size, stat.st_mtime_ns)}
    result = {".": ("directory",)}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                if entry.is_symlink() or (
                    hasattr(entry, "is_junction") and entry.is_junction()
                ):
                    result[relative] = ("link", os.readlink(path))
                elif entry.is_dir(follow_symlinks=False):
                    result[relative] = ("directory",)
                    pending.append(path)
                else:
                    stat = entry.stat(follow_symlinks=False)
                    result[relative] = ("file", stat.st_size, stat.st_mtime_ns)
    return result


def assert_unchanged(root: Path, before: dict[str, tuple]) -> None:
    after = tree_snapshot(root)
    if before == after:
        return
    changed = sorted(
        name for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )
    raise AssertionError(f"Source/output changed: {root}: {changed[:20]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    source = args.project.resolve()
    if not source.is_dir():
        parser.error(f"Project directory does not exist: {source}")
    watched = (source,)
    snapshots = {path: tree_snapshot(path) for path in watched}
    errors = []
    original_hook = sys.excepthook

    def report_exception(*args):
        errors.append(args[1])
        original_hook(*args)

    sys.excepthook = report_exception
    folder = Path(tempfile.mkdtemp(prefix="stereo-ui-check-"))
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(folder))
    app = QApplication([])
    app.setStyle("Fusion")
    for font in ("msyh.ttc", "segoeui.ttf", "consola.ttf"):
        path = Path("C:/Windows/Fonts") / font
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    app.setFont(application_font())
    workers.configure_pools()
    window = MainWindow()
    try:
        assert set(window.activity_buttons) == {
            "data", "display", "adjust", "measure", "statistics"
        }
        assert not hasattr(window, "mode_picker"), "Obsolete mode selector remains"
        window.show()
        window.load_project(source)
        started = time.perf_counter()
        loaded = False
        while time.perf_counter() - started < 45:
            app.processEvents()
            if window.dataset and window.tiles and all(
                tile.showing_canvas() and not tile.has_pending_work()
                for tile in window.tiles.values()
            ):
                loaded = True
                break
            QTest.qWait(20)
        assert loaded, "Project/media did not finish loading within 45 seconds"
        assert window.player_bar.isVisible(), "Viewer player is not visible"
        elapsed = []
        for _ in range(10):
            for page in ("display", "adjust", "measure", "statistics", "data"):
                before = time.perf_counter()
                window._select_activity(page)
                app.processEvents()
                elapsed.append((time.perf_counter() - before) * 1000)
        print(
            f"frames={len(window.dataset.samples)}; "
            f"panel_commit_mean_ms={sum(elapsed)/len(elapsed):.2f}; "
            f"max_ms={max(elapsed):.2f}"
        )
        if not window._sidebar_collapsed:
            window.toggle_sidebar()
        for theme in ("light", "dark"):
            window.preferences.theme = theme
            window._apply_preferences()
            for width in (1050, 1520):
                window.resize(width, 800)
                for page in ("data", "display", "adjust", "measure", "statistics"):
                    window._select_activity(page)
                    if page in {"adjust", "measure", "statistics"}:
                        if not window.inspector.isVisible():
                            window._select_activity(page)
                        assert window._sidebar_collapsed, "Both side panels are open"
                        assert window.inspector.isVisible(), page
                    else:
                        if window._sidebar_collapsed:
                            window._select_activity(page)
                        assert not window.inspector.isVisible(), "Both side panels are open"
                        assert window.sidebar_panel.isVisible(), page
                    app.processEvents()
                    assert window.grab().save(str(folder / f"{theme}-{width}-{page}.png"))
                window.close_inspector()
    finally:
        window.close()
        workers.wait_for_all(10000)
        for path, snapshot in snapshots.items():
            assert_unchanged(path, snapshot)
        sys.excepthook = original_hook
    assert not errors, errors
    print(f"read_only_check=passed; entries={sum(len(items) for items in snapshots.values())}")
    print(folder)


if __name__ == "__main__":
    main()
