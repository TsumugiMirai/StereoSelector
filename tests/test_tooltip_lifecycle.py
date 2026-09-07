from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QPushButton, QWidget
from shiboken6 import isValid

from stereo_selector.widgets import ToolTipManager


def _flush_deletions() -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def _tooltip_fixture():
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    button = QPushButton("Open", window)
    button.setToolTip("Open project")
    manager = ToolTipManager(window)
    app.installEventFilter(manager)
    event = QHelpEvent(QEvent.Type.ToolTip, QPoint(2, 2), QPoint(40, 40))
    assert manager.eventFilter(button, event)
    return app, window, button, manager


def test_tooltip_dispose_is_idempotent_and_detaches_filter() -> None:
    app, window, button, manager = _tooltip_fixture()
    popup = manager.popup
    try:
        manager.dispose()
        manager.dispose()
        assert manager.popup is None
        assert manager._disposed
        _flush_deletions()
        assert not isValid(popup)
        assert not manager.eventFilter(button, QEvent(QEvent.Type.Leave))
        app.sendEvent(button, QEvent(QEvent.Type.Leave))
    finally:
        manager.dispose()
        window.deleteLater()
        _flush_deletions()


def test_deleted_popup_cannot_leave_a_live_global_event_filter() -> None:
    app, window, button, manager = _tooltip_fixture()
    popup = manager.popup
    try:
        popup.deleteLater()
        _flush_deletions()
        assert not isValid(popup)
        assert manager.popup is None
        assert manager._disposed
        for event_type in (
            QEvent.Type.Leave,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Wheel,
            QEvent.Type.KeyPress,
            QEvent.Type.WindowDeactivate,
        ):
            assert not manager.eventFilter(button, QEvent(event_type))
        app.sendEvent(button, QEvent(QEvent.Type.Leave))
    finally:
        manager.dispose()
        window.deleteLater()
        _flush_deletions()


def test_parent_delete_later_releases_unparented_tooltip() -> None:
    _app, window, _button, manager = _tooltip_fixture()
    popup = manager.popup
    window.deleteLater()
    _flush_deletions()
    _flush_deletions()
    assert not isValid(window)
    assert not isValid(manager)
    assert not isValid(popup)


def test_manager_delete_later_releases_popup_without_closing_parent() -> None:
    _app, window, _button, manager = _tooltip_fixture()
    popup = manager.popup
    try:
        manager.deleteLater()
        _flush_deletions()
        _flush_deletions()
        assert not isValid(manager)
        assert not isValid(popup)
        assert isValid(window)
    finally:
        window.deleteLater()
        _flush_deletions()


def test_tooltip_shutdown_without_explicit_window_close() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(root / "src")
    script = """
from PySide6.QtCore import QEvent, QPoint, QTimer
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QPushButton, QWidget
from stereo_selector.widgets import ToolTipManager
app = QApplication([])
window = QWidget()
button = QPushButton('Open', window)
button.setToolTip('Open project')
manager = ToolTipManager(window)
app.installEventFilter(manager)
window.show()
manager.eventFilter(button, QHelpEvent(QEvent.Type.ToolTip, QPoint(2, 2), QPoint(40, 40)))
QTimer.singleShot(10, app.quit)
app.exec()
assert manager._disposed
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RuntimeError" not in result.stderr
    assert "Traceback" not in result.stderr


def test_tooltip_interpreter_teardown_without_event_loop() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(root / "src")
    script = """
from PySide6.QtWidgets import QApplication, QWidget
from stereo_selector.widgets import ToolTipManager
app = QApplication([])
window = QWidget()
manager = ToolTipManager(window)
app.installEventFilter(manager)
window.show()
app.processEvents()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RuntimeError" not in result.stderr
    assert "Traceback" not in result.stderr
