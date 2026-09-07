from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QHelpEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from stereo_selector.app import MainWindow
from stereo_selector.settings import ChoiceButton
from stereo_selector.theme import PALETTES, palette_for, style_for
from stereo_selector.ui_controls import CursorInfoWidget
from stereo_selector.widgets import CommandPalette


def _make_project(root: Path) -> Path:
    for index in (1, 2):
        for folder, color in (("left", "navy"), ("right", "teal")):
            path = root / folder / f"frame_{index:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), color).save(path)
    return root


def test_sidebar_and_topbar_toggles_are_real_affordances(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(_make_project(tmp_path / "capture"))
    window.show()
    app.processEvents()

    assert window.sidebar_button.parentWidget() is window.inspection_tools_row
    assert window.inspection_tools_row.parentWidget() is window.inspection_bar
    assert window.reset_button.parentWidget() is window.inspection_tools_row
    assert not window.reset_button.isHidden()
    assert window.topbar_button.parentWidget() is window.title_bar.actions
    assert window.topbar_button.isEnabled()

    window.toggle_sidebar()
    QTest.qWait(240)
    assert window.sidebar_button._direction == "left"
    window.toggle_sidebar()
    QTest.qWait(240)
    assert window.sidebar_button._direction == "right"

    window.toggle_topbar()
    QTest.qWait(220)
    assert window.inspection_bar.isHidden()
    assert window.topbar_button._direction == "down"
    window.toggle_topbar()
    QTest.qWait(220)
    assert not window.inspection_bar.isHidden()
    assert window.topbar_button._direction == "up"

    window._set_view_hint("聚焦")
    assert not window.view_hint.isHidden()
    assert window.view_hint.text() == "聚焦"
    window._set_view_hint("")
    assert window.view_hint.isHidden()
    window.close()


def test_application_tooltips_use_one_rounded_transparent_surface(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(_make_project(tmp_path / "capture"))
    app.processEvents()
    event = QHelpEvent(
        QEvent.Type.ToolTip,
        QPoint(4, 4),
        QPoint(40, 40),
    )

    handled = window.tooltip_manager.eventFilter(window.sidebar_button, event)

    assert handled
    assert window.tooltip_manager.popup.surface.objectName() == "appToolTip"
    assert window.tooltip_manager.popup.testAttribute(
        Qt.WidgetAttribute.WA_TranslucentBackground
    )
    window.tooltip_manager.popup.hide()
    window.close()


def test_command_palette_activates_handler_from_item_data() -> None:
    app = QApplication.instance() or QApplication([])
    called: list[str] = []

    def first() -> None:
        called.append("first")

    def second() -> None:
        called.append("second")

    palette = CommandPalette(
        [
            ("打开项目", "Ctrl+O", first),
            ("播放 / 暂停", "Space", second),
        ],
    )
    palette._filter("播")
    assert palette.results.count() == 1
    palette.results.setCurrentRow(0)
    palette._activate_current()
    QTest.qWait(10)
    assert called == ["second"]
    palette.close()


def test_choice_button_supports_keyboard_navigation() -> None:
    app = QApplication.instance() or QApplication([])
    selector = ChoiceButton()
    selector.addItem("无标定", "")
    selector.addItem("libra2000", "builtin:libra2000")
    selector.addItem("libra3000", "builtin:libra3000")
    selector.show()
    selector.setFocus()

    QTest.keyClick(selector, Qt.Key.Key_Down)
    assert selector.currentIndex() == 1
    QTest.keyClick(selector, Qt.Key.Key_Down)
    assert selector.currentIndex() == 2
    QTest.keyClick(selector, Qt.Key.Key_Down)
    assert selector.currentIndex() == 0
    QTest.keyClick(selector, Qt.Key.Key_Up)
    assert selector.currentIndex() == 2
    QTest.keyClick(selector, Qt.Key.Key_Home)
    assert selector.currentIndex() == 0
    QTest.keyClick(selector, Qt.Key.Key_End)
    assert selector.currentIndex() == 2
    selector.close()


def test_cursor_readout_keeps_complete_values_in_stable_columns() -> None:
    app = QApplication.instance() or QApplication([])
    readout = CursorInfoWidget()
    readout.setStyleSheet(style_for("light"))
    readout.setFixedWidth(620)
    readout.set_values(
        position="左目 RGB 8191,8191",
        stereo="右目 8191,8191 · 视差 -9999.99px",
        rgb="RGB 255,255,255",
        depth="深度 65535",
        xyz="XYZ -1.23e+06,-1.23e+06,1.23e+06",
    )
    readout.show()
    app.processEvents()
    labels = (
        readout.position_label,
        readout.stereo_label,
        readout.rgb_label,
        readout.depth_label,
        readout.xyz_label,
    )
    geometries = tuple(label.geometry() for label in labels)
    assert all(
        label.fontMetrics().horizontalAdvance(label.text()) <= label.width()
        for label in labels
    )

    readout.set_values(
        position="左目 RGB 1,2",
        stereo="右目 3,2 · 视差 -2.00px",
        rgb="RGB 0,0,0",
        depth="深度 1",
        xyz="XYZ 0,0,1",
    )
    app.processEvents()
    assert tuple(label.geometry() for label in labels) == geometries
    readout.close()


def test_palette_for_matches_theme_tokens() -> None:
    app = QApplication.instance() or QApplication([])
    for theme in ("dark", "light"):
        tokens = PALETTES[theme]
        palette = palette_for(theme)
        assert palette.buttonText().color() == QColor(tokens["text"])
        assert palette.window().color() == QColor(tokens["window"])
        assert palette.highlight().color() == QColor(tokens["focus"])


def test_canvas_background_follows_theme(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(_make_project(tmp_path / "capture"))
    app.processEvents()
    tile = window.tile_pool["left"]

    initial_theme = window.preferences.theme
    assert (
        tile.image_canvas.backgroundBrush().color().name()
        == PALETTES[initial_theme]["media_workspace"]
    )

    window.preferences.theme = "light"
    window._apply_preferences()
    light_color = PALETTES["light"]["media_workspace"]
    assert tile.image_canvas.backgroundBrush().color().name() == light_color
    window.close()
