"""Reusable, application-owned PySide6 controls.

The controls in this module deliberately avoid native popup widgets.  Native
``QComboBox``/``QMenu`` windows are styled differently by each platform and,
on Windows, can expose a square backing window behind a rounded QSS surface.
Keeping the choice popup inside the main window makes its geometry, clipping,
shadow and interaction identical in every part of Stereo Selector.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ElidedLabel(QLabel):
    """Single-line status text that never pushes the timeline out of view."""

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setToolTip(text)

    def clear(self) -> None:
        self.setText("")

    def sizeHint(self) -> QSize:
        return QSize(160, self.fontMetrics().height() + 4)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self.fontMetrics().height() + 4)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        rect = self.contentsRect().adjusted(4, 0, 0, 0)
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideMiddle, rect.width())
        painter.drawText(rect, self.alignment(), text)


class CursorInfoWidget(QFrame):
    """Stable two-row readout for crosshair and point-cloud inspection.

    Each value owns a fixed semantic column, so changing digit counts cannot
    move neighbouring values.  The compact second row keeps the complete
    readout inside the inspection toolbar without stealing image height.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("cursorInfo")
        self._text = ""
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(0)
        grid.setColumnMinimumWidth(0, 180)
        grid.setColumnMinimumWidth(1, 100)
        grid.setColumnMinimumWidth(2, 324)
        grid.setColumnStretch(0, 180)
        grid.setColumnStretch(1, 100)
        grid.setColumnStretch(2, 324)

        self.position_label = self._cell("position")
        self.stereo_label = self._cell("stereo")
        self.rgb_label = self._cell("rgb")
        self.depth_label = self._cell("depth")
        self.xyz_label = self._cell("xyz")
        grid.addWidget(self.position_label, 0, 0)
        grid.addWidget(self.stereo_label, 0, 1, 1, 2)
        grid.addWidget(self.rgb_label, 1, 0)
        grid.addWidget(self.depth_label, 1, 1)
        grid.addWidget(self.xyz_label, 1, 2)

        self.message_label = self._cell("message")
        grid.addWidget(self.message_label, 0, 0, 2, 3)
        self.message_label.hide()

    def _cell(self, role: str) -> QLabel:
        label = QLabel("")
        label.setObjectName("cursorInfoCell")
        label.setProperty("cursorRole", role)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        return label

    def set_values(
        self,
        *,
        position: str = "",
        rgb: str = "",
        stereo: str = "",
        depth: str = "",
        xyz: str = "",
    ) -> None:
        values = (position, rgb, stereo, depth, xyz)
        self._text = "   ·   ".join(value for value in values if value) or "无像素数据"
        self.message_label.hide()
        for label in (
            self.position_label,
            self.stereo_label,
            self.rgb_label,
            self.depth_label,
            self.xyz_label,
        ):
            label.show()
        self.position_label.setText(position or "无像素数据")
        self.rgb_label.setText(rgb)
        self.stereo_label.setText(stereo)
        self.depth_label.setText(depth)
        self.xyz_label.setText(xyz)
        self.setToolTip(self._text)

    def setText(self, text: str) -> None:
        """QLabel-compatible message API used outside image crosshair mode."""
        self._text = text
        for label in (
            self.position_label,
            self.stereo_label,
            self.rgb_label,
            self.depth_label,
            self.xyz_label,
        ):
            label.hide()
        self.message_label.setText(text)
        self.message_label.show()
        self.setToolTip(text)

    def text(self) -> str:
        return self._text

    def clear(self) -> None:
        self.setText("")


class ChoiceOptionButton(QPushButton):
    """One option in :class:`ChoicePopup`, including a drawn check mark."""

    def __init__(self, text: str, index: int, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.index = index
        self.setObjectName("choiceOption")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedHeight(32)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.isChecked():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.palette().highlight().color(), 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        x = self.width() - 17
        y = self.height() // 2
        painter.drawLine(x - 4, y, x - 1, y + 3)
        painter.drawLine(x - 1, y + 3, x + 5, y - 4)


class ChoicePopup(QFrame):
    """Rounded in-window popup used by every select control.

    This is a regular child widget, not a native menu or combo popup.  The
    outer transparent frame reserves room for the shadow and the inner frame
    owns the rounded background.
    """

    activated = Signal(int)

    _SHADOW_MARGIN = 10
    _MAX_PANEL_HEIGHT = 292

    def __init__(self, anchor: ChoiceButton, parent: QWidget) -> None:
        super().__init__(parent)
        self.anchor = anchor
        self.setObjectName("choicePopup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.hide()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            self._SHADOW_MARGIN,
            self._SHADOW_MARGIN,
            self._SHADOW_MARGIN,
            self._SHADOW_MARGIN,
        )
        outer.setSpacing(0)

        self.surface = QFrame()
        self.surface.setObjectName("choicePopupSurface")
        shadow = QGraphicsDropShadowEffect(self.surface)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 78))
        self.surface.setGraphicsEffect(shadow)
        surface_layout = QVBoxLayout(self.surface)
        surface_layout.setContentsMargins(5, 5, 5, 5)
        surface_layout.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("choicePopupScroll")
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.body = QWidget()
        self.body.setObjectName("choicePopupBody")
        self.options_layout = QVBoxLayout(self.body)
        self.options_layout.setContentsMargins(0, 0, 0, 0)
        self.options_layout.setSpacing(2)
        self.scroll.setWidget(self.body)
        surface_layout.addWidget(self.scroll)
        outer.addWidget(self.surface)

        self.option_buttons: list[ChoiceOptionButton] = []
        self._highlight_index = -1

    def set_choices(
        self,
        choices: Sequence[tuple[str, object]],
        current_index: int,
    ) -> None:
        if [option.text() for option in self.option_buttons] == [label for label, _ in choices]:
            for index, option in enumerate(self.option_buttons):
                option.setChecked(index == current_index)
            self._set_highlight(current_index)
            return
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self.option_buttons.clear()
        for index, (label, _data) in enumerate(choices):
            option = ChoiceOptionButton(label, index, self.body)
            option.setChecked(index == current_index)
            option.clicked.connect(
                lambda _checked=False, choice_index=index: self._activate(choice_index)
            )
            self.options_layout.addWidget(option)
            self.option_buttons.append(option)
        self._set_highlight(current_index if current_index >= 0 else 0)

    def show_for(
        self,
        choices: Sequence[tuple[str, object]],
        current_index: int,
    ) -> None:
        self.set_choices(choices, current_index)
        host = self.parentWidget()
        if host is None:
            return

        text_width = max(
            (self.anchor.fontMetrics().horizontalAdvance(label) for label, _ in choices),
            default=80,
        )
        panel_width = max(self.anchor.width(), min(420, text_width + 54))
        panel_width = min(panel_width, max(120, host.width() - 16))
        option_height = len(choices) * 32 + max(0, len(choices) - 1) * 2
        # Include the surface padding and frame explicitly.  Without this
        # allowance QScrollArea can show a one-pixel scrollbar for short lists.
        content_height = max(46, option_height + 14)
        panel_height = min(self._MAX_PANEL_HEIGHT, content_height)
        total_width = panel_width + self._SHADOW_MARGIN * 2
        total_height = panel_height + self._SHADOW_MARGIN * 2

        anchor_pos = self.anchor.mapTo(host, QPoint(0, 0))
        x = anchor_pos.x() - self._SHADOW_MARGIN
        below_y = anchor_pos.y() + self.anchor.height() + 3 - self._SHADOW_MARGIN
        above_y = anchor_pos.y() - total_height + self._SHADOW_MARGIN - 3
        y = below_y if below_y + total_height <= host.height() else above_y
        x = max(0, min(x, max(0, host.width() - total_width)))
        y = max(0, min(y, max(0, host.height() - total_height)))
        self.setGeometry(x, y, total_width, total_height)
        self.show()
        self.raise_()
        self.scroll.ensureWidgetVisible(self.option_buttons[self._highlight_index])
        self.setFocus(Qt.FocusReason.PopupFocusReason)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def hideEvent(self, event) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.anchor._popup_closed()
        super().hideEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if not self.isVisible():
            return False
        event_type = event.type()
        if event_type == QEvent.Type.ShortcutOverride:
            if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Home,
                               Qt.Key.Key_End, Qt.Key.Key_Return, Qt.Key.Key_Enter,
                               Qt.Key.Key_Space, Qt.Key.Key_Escape):
                event.accept()
                return True
        if event_type == QEvent.Type.MouseButtonPress:
            if isinstance(watched, QWidget) and (
                watched is self.anchor
                or self.anchor.isAncestorOf(watched)
                or watched is self
                or self.isAncestorOf(watched)
            ):
                return False
            self.hide()
        elif event_type == QEvent.Type.Hide and watched is self.anchor:
            self.hide()
        elif event_type in {
            QEvent.Type.Hide,
            QEvent.Type.Close,
            QEvent.Type.Move,
            QEvent.Type.Resize,
            QEvent.Type.WindowDeactivate,
        } and watched is self.parentWidget():
            self.hide()
        return False

    def keyPressEvent(self, event) -> None:
        if not self.option_buttons:
            self.hide()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.anchor.setFocus(Qt.FocusReason.PopupFocusReason)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            delta = 1 if event.key() == Qt.Key.Key_Down else -1
            self._set_highlight((self._highlight_index + delta) % len(self.option_buttons))
            event.accept()
            return
        if event.key() == Qt.Key.Key_Home:
            self._set_highlight(0)
            event.accept()
            return
        if event.key() == Qt.Key.Key_End:
            self._set_highlight(len(self.option_buttons) - 1)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._activate(self._highlight_index)
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_highlight(self, index: int) -> None:
        if not self.option_buttons:
            self._highlight_index = -1
            return
        self._highlight_index = max(0, min(index, len(self.option_buttons) - 1))
        for option_index, option in enumerate(self.option_buttons):
            active = option_index == self._highlight_index
            if bool(option.property("active")) != active:
                option.setProperty("active", active)
                option.style().unpolish(option)
                option.style().polish(option)
        self.scroll.ensureWidgetVisible(self.option_buttons[self._highlight_index])

    def _activate(self, index: int) -> None:
        if not 0 <= index < len(self.option_buttons):
            return
        self.activated.emit(index)
        self.hide()
        self.anchor.setFocus(Qt.FocusReason.PopupFocusReason)


class ChoiceButton(QPushButton):
    """Uniform PySide6 select control with a non-native in-window popup."""

    currentIndexChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("choiceButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(120)
        self._items: list[tuple[str, object]] = []
        self._current_index = -1
        self._popup: ChoicePopup | None = None
        self.clicked.connect(self.showPopup)

    def addItem(self, text: str, data: object = None) -> None:
        self._items.append((text, data))
        if self._current_index < 0:
            self.setCurrentIndex(0)

    def clear(self) -> None:
        if self._popup is not None:
            self._popup.hide()
        self._items.clear()
        self._current_index = -1
        self.setText("")

    def count(self) -> int:
        return len(self._items)

    def itemText(self, index: int) -> str:
        return self._items[index][0] if 0 <= index < len(self._items) else ""

    def itemData(self, index: int) -> object:
        return self._items[index][1] if 0 <= index < len(self._items) else None

    def currentData(self) -> object:
        return self.itemData(self._current_index)

    def currentText(self) -> str:
        return self.itemText(self._current_index)

    def currentIndex(self) -> int:
        return self._current_index

    def findData(self, data: object) -> int:
        return next(
            (index for index, (_label, value) in enumerate(self._items) if value == data),
            -1,
        )

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items) or index == self._current_index:
            return
        self._current_index = index
        self.setText(self._items[index][0])
        if self._popup is not None:
            self._popup.set_choices(self._items, index)
        self.currentIndexChanged.emit(index)
        self.update()

    def showPopup(self) -> None:
        if not self.isEnabled() or not self._items:
            return
        if self._popup is not None and self._popup.isVisible():
            self._popup.hide()
            return
        host = self.window()
        if host is self:
            return
        if self._popup is None or self._popup.parentWidget() is not host:
            if self._popup is not None:
                self._popup.deleteLater()
            self._popup = ChoicePopup(self, host)
            self._popup.activated.connect(self.setCurrentIndex)
            self.destroyed.connect(self._popup.deleteLater)
        self.setProperty("popupOpen", True)
        self.style().unpolish(self)
        self.style().polish(self)
        self._popup.show_for(self._items, self._current_index)
        self.update()

    def hidePopup(self) -> None:
        if self._popup is not None:
            self._popup.hide()

    def _popup_closed(self) -> None:
        if bool(self.property("popupOpen")):
            self.setProperty("popupOpen", False)
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and self._items:
            delta = 1 if event.key() == Qt.Key.Key_Down else -1
            self.setCurrentIndex((self._current_index + delta) % len(self._items))
            event.accept()
            return
        if event.key() == Qt.Key.Key_Home and self._items:
            self.setCurrentIndex(0)
            event.accept()
            return
        if event.key() == Qt.Key.Key_End and self._items:
            self.setCurrentIndex(len(self._items) - 1)
            event.accept()
            return
        if event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Space,
        ):
            self.showPopup()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.palette().buttonText().color(), 1.25)
        pen.setCosmetic(True)
        painter.setPen(pen)
        x = self.width() - 15
        y = self.height() // 2
        if bool(self.property("popupOpen")):
            painter.drawLine(x - 3, y + 2, x, y - 1)
            painter.drawLine(x, y - 1, x + 3, y + 2)
        else:
            painter.drawLine(x - 3, y - 2, x, y + 1)
            painter.drawLine(x, y + 1, x + 3, y - 2)
