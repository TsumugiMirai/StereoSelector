from string import Template

from PySide6.QtGui import QColor, QPalette

PALETTES = {
    "dark": {
        "window": "#191a1b",
        "sidebar": "#191a1b",
        "surface": "#121314",
        "panel": "#202122",
        "elevated": "#242526",
        "border": "#2a2b2c",
        "border_strong": "#333536",
        "text": "#ededed",
        "text_secondary": "#bfbfbf",
        "muted": "#8c8c8c",
        "faint": "#626262",
        "hover": "#28292a",
        "pressed": "#303132",
        "selection": "#303132",
        "focus": "#3994bc",
        "accent": "#297aa0",
        "accent_hover": "#2b7da3",
        "success": "#72c892",
        "success_hover": "#82d7a0",
        "on_success": "#102016",
        "success_surface": "#17271d",
        "warning": "#e5ba7d",
        "danger": "#f28772",
        "on_danger": "#ffffff",
        "primary": "#eeeeee",
        "primary_hover": "#ffffff",
        "primary_text": "#191a1b",
        "input": "#191a1b",
        "disabled": "#555555",
        "nav_selected": "#242526",
        "app_border": "#333536",
        "media_workspace": "#0d0e0f",
        "media_tile": "#121314",
        "media_header": "#191a1b",
        "media_border": "#2a2b2c",
        "media_text": "#ededed",
        "media_muted": "#8c8c8c",
    },
    "light": {
        "window": "#fafafd",
        "sidebar": "#fafafd",
        "surface": "#ffffff",
        "panel": "#fafafd",
        "elevated": "#f0f0f3",
        "border": "#f0f1f2",
        "border_strong": "#d8d8d8",
        "text": "#202020",
        "text_secondary": "#3f3f3f",
        "muted": "#606060",
        "faint": "#999999",
        "hover": "#eeeef1",
        "pressed": "#e2e3e5",
        "selection": "#e6e6e8",
        "focus": "#0069cc",
        "accent": "#0069cc",
        "accent_hover": "#0063c1",
        "success": "#388a34",
        "success_hover": "#2f7c2c",
        "on_success": "#ffffff",
        "success_surface": "#e9f5eb",
        "warning": "#895503",
        "danger": "#ad0707",
        "on_danger": "#ffffff",
        "primary": "#202020",
        "primary_hover": "#343434",
        "primary_text": "#ffffff",
        "input": "#ffffff",
        "disabled": "#bbbbbb",
        "nav_selected": "#eaeaec",
        "app_border": "#d8d8d8",
        "media_workspace": "#e9eaec",
        "media_tile": "#ffffff",
        "media_header": "#fafafd",
        "media_border": "#d8d8d8",
        "media_text": "#202020",
        "media_muted": "#606060",
    },
}


STYLE = Template(r"""
QWidget {
    color: $text;
    font-family: "Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI",
                 "PingFang SC", "Noto Sans CJK SC", "Helvetica Neue", "DejaVu Sans", sans-serif;
    font-size: 12px;
}
QMainWindow { background: $window; }
QWidget#appRoot { background: $window; border: 1px solid $app_border; }
QWidget#workspace { background: $surface; }
QWidget#mediaWorkspace { background: $media_workspace; }
QToolTip {
    background: $panel; color: $text; border: 1px solid $border_strong;
    padding: 5px 8px;
}
QWidget#toolTipWindow { background: transparent; }
QFrame#appToolTip {
    background: $panel; border: 1px solid $border_strong; border-radius: 8px;
}
QLabel#appToolTipText { background: transparent; color: $text; border: 0; }

QFrame#titleBar { background: $window; border-bottom: 1px solid $border; }
QLabel#titleMark { background: $primary; color: $primary_text; border-radius: 6px; font-size: 10px; font-weight: 700; }
QLabel#titleBrand { color: $text; font-size: 12px; font-weight: 600; }
QLabel#titleContext { color: $muted; padding: 2px 10px; font-size: 11px; }
QWidget#titleActions { background: transparent; }
QPushButton#windowControlButton, QPushButton#closeWindowButton {
    background: transparent; border: 0; border-radius: 5px; padding: 0; color: $muted;
}
QPushButton#windowControlButton:hover { background: $hover; color: $text; }
QPushButton#closeWindowButton:hover { background: $danger; color: $on_danger; }
QPushButton#titleActionButton {
    min-height: 26px; background: transparent; border: 1px solid transparent;
    border-radius: 6px; color: $muted; padding: 0 8px;
}
QPushButton#titleActionButton:hover { background: $hover; color: $text; }

QFrame#navigationShell { background: $sidebar; border-right: 1px solid $border; }
QFrame#activityBar { background: $window; border-right: 1px solid $border; min-width: 39px; max-width: 39px; }
QPushButton#activityButton {
    min-height: 36px; background: transparent; border: 0; border-radius: 6px; padding: 0;
}
QPushButton#activityButton:hover { background: $hover; }
QPushButton#activityButton:checked {
    background: $nav_selected; border-left: 2px solid $focus;
}
QFrame#sidebar { background: $sidebar; border: 0; }
QWidget#sidebarSection {
    background: transparent; border: 0; border-bottom: 1px solid $border;
    border-radius: 0;
}
QLabel#eyebrow, QLabel#sectionLabel { color: $muted; font-size: 11px; font-weight: 600; }
QLabel#projectName { color: $text; font-size: 14px; font-weight: 600; }
QLabel#projectPath, QLabel#muted { color: $muted; }
QLabel#countBadge { background: $elevated; color: $muted; border-radius: 7px; padding: 1px 6px; font-size: 11px; }

QFrame#workspaceHeader { background: $surface; border-bottom: 1px solid $border; }
QFrame#inspectionBar { background: $surface; border-bottom: 1px solid $border; }
QFrame#inspectionToolsRow, QFrame#cursorInfoRow { background: transparent; border: 0; }
QLabel#breadcrumb { color: $text; font-weight: 600; }
QLabel#viewHint { color: $muted; font-size: 11px; }
QLabel#toolMeta { color: $faint; font-size: 11px; padding-left: 4px; }
QFrame#cursorInfo { background: transparent; border: 0; }
QLabel#cursorInfoCell {
    background: transparent; border: 0; color: $muted;
    font-family: "Cascadia Mono", "Consolas"; font-size: 10px;
}
QLabel#qualitySummary {
    background: $panel; color: $muted; border: 1px solid $border;
    border-radius: 6px; padding: 7px 8px; font-size: 11px;
}
QPushButton#sideToggleButton {
    background: transparent; border: 1px solid transparent; color: $muted;
    border-radius: 6px; padding: 0; font-size: 20px;
}
QPushButton#sideToggleButton:hover { background: $hover; border-color: $border; color: $text; }
QPushButton#projectPicker, QPushButton#layoutPicker,
QPushButton#calibrationPicker {
    min-height: 26px; background: transparent; color: $text; border: 1px solid transparent;
    border-radius: 6px; padding: 0 24px 0 8px; text-align: left;
}
QPushButton#projectPicker:hover, QPushButton#layoutPicker:hover, QPushButton#calibrationPicker:hover {
    background: $hover; border-color: $border;
}
QPushButton#projectPicker:focus, QPushButton#layoutPicker:focus, QPushButton#calibrationPicker:focus { border-color: $focus; }
QPushButton#toolIconButton {
    min-height: 26px; background: transparent; border: 1px solid transparent;
    border-radius: 6px; padding: 0;
}
QPushButton#toolIconButton:hover { background: $hover; border-color: $border; }
QPushButton#toolIconButton:checked { background: $nav_selected; border-color: $focus; }
QPushButton#toolIconButton:disabled { background: transparent; border-color: transparent; }

QFrame#playerBar {
    background: $surface; border-top: 1px solid $border;
}
QLabel#sampleMeta { color: $muted; font-size: 11px; }
QFrame#timelinePanel {
    background: $surface; border: 0; border-radius: 0;
}
QLabel#playerStatus { color: $muted; font-size: 11px; padding-left: 4px; }
QPushButton#playbackButton {
    background: transparent; border: 1px solid transparent; border-radius: 6px;
    color: $muted; padding: 0;
}
QPushButton#playbackButton:hover {
    background: $hover; border-color: $border; color: $text;
}
QPushButton#playbackButton[playing="true"] {
    background: $nav_selected; border-color: $focus; color: $text;
}
QPushButton#playbackButton:disabled {
    background: transparent; border-color: transparent; color: $disabled;
}
QSlider#timelineSlider::groove:horizontal {
    background: $border_strong; height: 4px; border-radius: 2px;
}
QSlider#timelineSlider::sub-page:horizontal {
    background: $accent; border-radius: 2px;
}
QSlider#timelineSlider::handle:horizontal {
    background: $text; border: 2px solid $surface; width: 12px;
    margin: -6px 0; border-radius: 8px;
}
QSlider#timelineSlider::handle:horizontal:hover {
    background: $accent; border-color: $surface;
}
QPushButton#playbackSpeed {
    min-height: 26px; background: transparent; color: $text;
    border: 1px solid transparent; border-radius: 6px; padding: 0 22px 0 7px;
    text-align: left;
}
QPushButton#playbackSpeed:hover { background: $hover; border-color: $border_strong; }
QPushButton#playbackSpeed:focus { border-color: $focus; }

QFrame#mediaTile { background: $media_tile; border: 1px solid $media_border; border-radius: 8px; }
QFrame#mediaTile[missing="true"] { border-color: $warning; }
QFrame#mediaTile[focused="true"] { border-color: $focus; }
QFrame#mediaHeader {
    background: $media_header; border-bottom: 1px solid $media_border;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}
QLabel#tileTitle { color: $media_text; font-size: 12px; font-weight: 600; }
QLabel#tileMeta, QLabel#loadingText { color: $media_muted; font-size: 11px; }
QLabel#emptyTileText { color: $warning; }
QLabel#errorText { color: $danger; }
QPushButton#iconButton { background: transparent; border: 0; color: $media_muted; border-radius: 6px; padding: 3px 7px; font-size: 11px; }
QPushButton#iconButton:hover { background: $media_border; color: $media_text; }

QLabel#emptyMark { background: $primary; color: $primary_text; border-radius: 12px; font-size: 22px; font-weight: 700; }
QLabel#emptyTitle { color: $text; font-size: 20px; font-weight: 600; }
QLabel#emptyHint { color: $muted; font-size: 12px; }
QLabel#emptyFormats { color: $faint; font-size: 11px; }

QPushButton {
    min-height: 26px; background: $panel; color: $text; border: 1px solid $border_strong;
    border-radius: 6px; padding: 0 10px;
}
QPushButton:hover { background: $hover; border-color: $border_strong; }
QPushButton:pressed { background: $pressed; }
QPushButton:disabled { color: $disabled; background: $elevated; border-color: $border; }
QPushButton:focus { border-color: $focus; }
QPushButton#primaryButton {
    background: $primary; color: $primary_text; border-color: $primary;
    font-weight: 600; padding: 0 12px;
}
QPushButton#primaryButton:hover { background: $primary_hover; border-color: $primary_hover; }
QPushButton#ghostButton { background: transparent; border-color: transparent; color: $muted; }
QPushButton#ghostButton:hover { background: $hover; color: $text; }
QPushButton#secondaryButton { background: $elevated; border-color: $border; color: $text; }
QPushButton#secondaryButton:hover { background: $hover; border-color: $border_strong; }
QPushButton#toolButton, QPushButton#toolToggle {
    min-height: 20px; background: transparent; border: 1px solid transparent;
    color: $muted; padding: 4px 9px;
}
QPushButton#toolButton:hover, QPushButton#toolToggle:hover {
    background: $hover; border-color: $border; color: $text;
}
QPushButton#toolToggle:checked {
    background: $nav_selected; border-color: $border_strong; color: $text;
}
QPushButton#toolButton:disabled, QPushButton#toolToggle:disabled {
    background: transparent; border-color: transparent; color: $disabled;
}

QCheckBox { spacing: 9px; padding: 7px 3px; color: $text; }
QCheckBox:hover { color: $text; background: $hover; border-radius: 5px; }
QCheckBox:disabled { color: $disabled; background: transparent; }
QCheckBox::indicator { width: 14px; height: 14px; }
QCheckBox::indicator:unchecked { border: 1px solid $border_strong; background: $input; border-radius: 3px; }
QCheckBox::indicator:checked { border: 1px solid $focus; background: $focus; border-radius: 3px; }

QSlider::groove:horizontal { background: $border_strong; height: 4px; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal { background: $text; border: 1px solid $surface; width: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover, QSlider::handle:horizontal:pressed { background: $focus; }
QSlider::handle:horizontal:disabled { background: $disabled; }
QSplitter::handle { background: $border; }
QStatusBar { background: $window; color: $muted; border-top: 1px solid $border; }
QStatusBar::item { border: 0; }

QFrame#inspector {
    background: $sidebar; border-left: 1px solid $border;
}
QFrame#inspectorHeader {
    background: $window; border-bottom: 1px solid $border; min-height: 37px;
}
QLabel#inspectorTitle, QLabel#inspectorHeading { color: $text; font-weight: 600; }
QScrollArea#inspectorScroll, QWidget#inspectorBody { background: $sidebar; border: 0; }
QFrame#inspectorSection {
    background: transparent; border: 0; border-bottom: 1px solid $border;
    border-radius: 0;
}
QLabel#inspectorStats, QLabel#inspectorFile {
    color: $muted; font-size: 11px;
    font-family: "Cascadia Mono", "SFMono-Regular", "Consolas", "Microsoft YaHei UI";
}
QLabel#inspectorValue {
    color: $muted; min-width: 30px; font-size: 11px;
    font-family: "Cascadia Mono", "SFMono-Regular", "Consolas";
}
QWidget#histogram { background: $input; border: 1px solid $border; border-radius: 6px; }
QPushButton#inspectorChoice {
    min-height: 26px; background: $input; color: $text; border: 1px solid $border_strong;
    border-radius: 6px; padding: 0 24px 0 8px; text-align: left;
}
QPushButton#inspectorChoice:hover { background: $hover; }
QPushButton#inspectorChoice:focus { border-color: $focus; }

QDialog#commandPalette {
    background: transparent; border: 0;
}
QFrame#commandSurface {
    background: $panel; border: 1px solid $border_strong; border-radius: 10px;
}
QLineEdit#commandSearch {
    min-height: 32px; font-size: 14px; border-color: $focus;
}
QListWidget#commandResults {
    background: $panel; border: 0; outline: 0;
}
QListWidget#commandResults::item {
    min-height: 26px; padding: 4px 8px; border-radius: 6px; color: $text_secondary;
}
QListWidget#commandResults::item:selected {
    background: $nav_selected; color: $text;
}

QDialog#settingsDialog { background: transparent; border: 0; }
QFrame#dialogSurface {
    background: $surface; border: 1px solid $border_strong; border-radius: 12px;
}
QDialog#settingsPage {
    background: $surface; border: 0; border-radius: 0;
}
QFrame#dialogSurface[embedded="true"] {
    border: 0; border-radius: 0;
}
QFrame#settingsHeader {
    background: $window; border-bottom: 1px solid $border; min-height: 39px;
    border-top-left-radius: 12px; border-top-right-radius: 12px;
}
QFrame#settingsHeader[embedded="true"] {
    border-top-left-radius: 0; border-top-right-radius: 0;
}
QLabel#settingsTitle { color: $text; font-size: 14px; font-weight: 600; }
QPushButton#settingsCloseButton {
    min-height: 28px; background: transparent; border: 0; border-radius: 6px;
    color: $muted; font-size: 16px; padding: 0;
}
QPushButton#settingsCloseButton:hover { background: $hover; color: $text; }
QListWidget#settingsNav {
    background: $sidebar; border: 0; border-right: 1px solid $border;
    padding: 10px 8px; outline: 0;
}
QListWidget#settingsNav::item {
    color: $muted; min-height: 25px; padding: 4px 10px; border-radius: 6px;
}
QListWidget#settingsNav::item:hover { background: $hover; color: $text; }
QListWidget#settingsNav::item:selected { background: $nav_selected; color: $text; }
QWidget#settingsPages, QScrollArea#settingsScroll, QWidget#settingsScrollBody {
    background: $surface; border: 0;
}
QLabel#settingsPageTitle { color: $text; font-size: 18px; font-weight: 600; }
QLabel#settingsDescription { color: $muted; }
QFrame#settingsFooter {
    background: $surface; border-top: 1px solid $border;
    border-bottom-left-radius: 14px; border-bottom-right-radius: 14px;
}
QFrame#settingsFooter[embedded="true"] {
    border-bottom-left-radius: 0; border-bottom-right-radius: 0;
}
QFrame#settingRow {
    background: transparent; border: 0; border-bottom: 1px solid $border;
    border-radius: 0;
}
QLabel#settingRowTitle { color: $text; font-weight: 600; }
QLabel#settingRowDescription { color: $muted; font-size: 11px; }

QFrame#segmentedControl { background: $elevated; border: 1px solid $border; border-radius: 7px; }
QPushButton#segmentButton {
    min-height: 24px; background: transparent; color: $muted;
    border: 0; border-radius: 5px; padding: 0 10px;
}
QPushButton#segmentButton:hover { background: $hover; color: $text; }
QPushButton#segmentButton:checked {
    background: $panel; color: $text; border: 1px solid $border_strong; font-weight: 600;
}

QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QKeySequenceEdit {
    min-height: 26px; background: $input; color: $text; border: 1px solid $border_strong;
    border-radius: 6px; padding: 0 8px; selection-background-color: $accent;
}
QPushButton#choiceButton, QPushButton#settingsCombo {
    min-height: 26px; background: $input; color: $text; border: 1px solid $border_strong;
    border-radius: 6px; padding: 0 26px 0 8px;
}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QKeySequenceEdit:focus,
QPushButton#choiceButton:focus, QPushButton#settingsCombo:focus { border-color: $focus; }
QPushButton#choiceButton:hover, QPushButton#settingsCombo:hover {
    background: $hover; border-color: $border_strong;
}
QPushButton#choiceButton[popupOpen="true"], QPushButton#settingsCombo[popupOpen="true"],
QPushButton#projectPicker[popupOpen="true"], QPushButton#layoutPicker[popupOpen="true"], QPushButton#calibrationPicker[popupOpen="true"],
QPushButton#playbackSpeed[popupOpen="true"] {
    background: $nav_selected; border-color: $focus;
}
QPushButton#choiceButton, QPushButton#settingsCombo { text-align: left; }
QFrame#choicePopup {
    background: transparent; border: 0;
}
QFrame#choicePopupSurface {
    background: $panel; border: 1px solid $border_strong; border-radius: 9px;
}
QScrollArea#choicePopupScroll, QWidget#choicePopupBody {
    background: transparent; border: 0;
}
QPushButton#choiceOption {
    min-height: 32px; background: transparent; color: $text; border: 0;
    border-radius: 6px; padding: 0 30px 0 10px; text-align: left;
}
QPushButton#choiceOption:hover, QPushButton#choiceOption[active="true"] {
    background: $hover; color: $text;
}
QPushButton#choiceOption:checked {
    background: $nav_selected; color: $text; font-weight: 600;
}
QLabel#calibrationDetail {
    background: $elevated; color: $muted; border: 1px solid $border;
    border-radius: 6px; padding: 8px 10px;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 18px; border: 0; background: transparent;
}
QScrollBar:vertical { background: transparent; width: 9px; }
QScrollBar::handle:vertical { background: $border_strong; min-height: 30px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
""")


def style_for(theme: str) -> str:
    return STYLE.substitute(PALETTES.get(theme, PALETTES["dark"]))


def palette_for(theme: str) -> QPalette:
    """Palette built from theme tokens for custom-painted widgets.

    Qt style sheets do not reliably update palette roles, so custom paint
    code (icons, chevrons, spinners) reads these roles instead of guessing.
    """
    tokens = PALETTES.get(theme, PALETTES["dark"])
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(tokens["window"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(tokens["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(tokens["input"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(tokens["panel"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(tokens["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(tokens["panel"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(tokens["text"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(tokens["panel"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(tokens["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(tokens["focus"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(tokens["text"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(tokens["muted"]))
    disabled = QPalette.ColorGroup.Disabled
    palette.setColor(disabled, QPalette.ColorRole.Text, QColor(tokens["disabled"]))
    palette.setColor(disabled, QPalette.ColorRole.ButtonText, QColor(tokens["disabled"]))
    palette.setColor(disabled, QPalette.ColorRole.WindowText, QColor(tokens["disabled"]))
    return palette
