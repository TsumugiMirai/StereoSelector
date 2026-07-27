from string import Template


PALETTES = {
    "dark": {
        "window": "#0f0f0f",
        "sidebar": "#151515",
        "surface": "#191919",
        "panel": "#1f1f1f",
        "elevated": "#252525",
        "border": "#303030",
        "border_strong": "#3d3d3d",
        "text": "#e7e7e7",
        "muted": "#909090",
        "faint": "#686868",
        "hover": "#292929",
        "pressed": "#202020",
        "primary": "#eeeeee",
        "primary_text": "#151515",
        "input": "#222222",
        "disabled": "#595959",
        "nav_selected": "#2a2a2a",
        "app_border": "#383838",
        "media_workspace": "#0c0d0f",
        "media_tile": "#121315",
        "media_header": "#1a1b1e",
        "media_border": "#303236",
        "media_text": "#ededed",
        "media_muted": "#929292",
        "accepted_bar_start": "#13251a",
        "accepted_bar_end": "#192d20",
        "accepted_border": "#357f50",
        "accepted_text": "#8bd9a8",
    },
    "light": {
        "window": "#efefec",
        "sidebar": "#f4f4f1",
        "surface": "#fafaf8",
        "panel": "#ffffff",
        "elevated": "#f5f5f2",
        "border": "#deded9",
        "border_strong": "#c9c9c3",
        "text": "#222220",
        "muted": "#737370",
        "faint": "#969690",
        "hover": "#eaeae5",
        "pressed": "#dfdfda",
        "primary": "#252523",
        "primary_text": "#ffffff",
        "input": "#ffffff",
        "disabled": "#aaa9a3",
        "nav_selected": "#e4e4df",
        "app_border": "#c8c8c2",
        "media_workspace": "#e9e9e5",
        "media_tile": "#fdfdfb",
        "media_header": "#f4f4f0",
        "media_border": "#d3d3ce",
        "media_text": "#282826",
        "media_muted": "#737370",
        "accepted_bar_start": "#e3f2e7",
        "accepted_bar_end": "#edf7ef",
        "accepted_border": "#86bf99",
        "accepted_text": "#267347",
    },
}


STYLE = Template(r"""
QWidget {
    color: $text;
    font-family: "Microsoft YaHei UI";
    font-size: 13px;
}
QMainWindow { background: $window; }
QWidget#appRoot { background: $window; border: 1px solid $app_border; }
QWidget#workspace { background: $surface; }
QWidget#mediaWorkspace { background: $media_workspace; }
QToolTip { background: $panel; color: $text; border: 1px solid $border_strong; border-radius: 7px; padding: 6px 9px; }

QFrame#titleBar { background: $window; border-bottom: 1px solid $border; }
QLabel#titleMark { background: $primary; color: $primary_text; border-radius: 6px; font-size: 10px; font-weight: 700; }
QLabel#titleBrand { color: $text; font-size: 12px; font-weight: 600; }
QLabel#titleContext { color: $muted; padding: 3px 14px; font-size: 11px; }
QPushButton#windowControlButton, QPushButton#closeWindowButton {
    background: transparent; border: 0; border-radius: 7px; padding: 0; color: $muted;
}
QPushButton#windowControlButton:hover { background: $hover; color: $text; }
QPushButton#closeWindowButton:hover { background: $hover; color: $text; }
QPushButton#titleActionButton { background: transparent; border: 0; color: $muted; padding: 5px 10px; }
QPushButton#titleActionButton:hover { background: $hover; color: $text; }

QFrame#sidebar { background: $sidebar; border-right: 1px solid $border; }
QWidget#sidebarSection {
    background: $surface; border: 1px solid $border; border-radius: 11px;
}
QLabel#eyebrow, QLabel#sectionLabel { color: $faint; font-size: 10px; font-weight: 600; }
QLabel#projectName { color: $text; font-size: 15px; font-weight: 600; }
QLabel#projectPath, QLabel#muted { color: $muted; }
QLabel#countBadge { background: $elevated; color: $muted; border-radius: 8px; padding: 1px 7px; font-size: 11px; }
QLabel#acceptedCount { color: $text; font-size: 25px; font-weight: 600; }
QLabel#acceptedCaption { color: $muted; font-size: 11px; }
QFrame#reviewCard { background: $panel; border: 1px solid $border; border-radius: 10px; }

QFrame#workspaceHeader { background: $surface; border-bottom: 1px solid $border; }
QFrame#inspectionBar { background: $surface; border-bottom: 1px solid $border; }
QLabel#breadcrumb { color: $text; font-weight: 600; }
QLabel#viewHint { color: $muted; font-size: 11px; }
QLabel#toolMeta { color: $faint; font-size: 11px; padding-left: 4px; }
QLabel#cursorInfo { color: $muted; font-family: "Cascadia Mono", "Consolas"; font-size: 11px; }
QLabel#qualitySummary {
    background: $panel; color: $muted; border: 1px solid $border;
    border-radius: 7px; padding: 8px 10px; font-size: 11px;
}
QPushButton#sideToggleButton {
    background: transparent; border: 1px solid transparent; color: $muted;
    border-radius: 6px; padding: 0; font-size: 22px;
}
QPushButton#sideToggleButton:hover { background: $hover; border-color: $border; color: $text; }

QFrame#reviewBar {
    background: $surface; border-top: 1px solid $border;
}
QFrame#reviewBar[reviewState="accepted"] {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 $accepted_bar_start, stop: 1 $accepted_bar_end
    );
    border-top: 2px solid $accepted_border;
}
QLabel#sampleTitle { color: $text; font-size: 14px; font-weight: 600; }
QLabel#sampleMeta { color: $muted; font-size: 11px; }
QLabel#statusPill {
    background: transparent; color: $muted; border: 1px solid $border_strong;
    border-radius: 9px; padding: 1px 8px; font-size: 11px;
}
QLabel#statusPill[accepted="true"] {
    background: transparent; border-color: $accepted_border; color: $accepted_text;
}
QFrame#reviewBar[reviewState="accepted"] QLabel#sampleTitle {
    color: $accepted_text;
}
QFrame#reviewBar[reviewState="accepted"] QPushButton#acceptButton {
    background: $accepted_border; border-color: $accepted_border;
}

QFrame#mediaTile { background: $media_tile; border: 1px solid $media_border; border-radius: 12px; }
QFrame#mediaTile[missing="true"] { border-color: #745037; }
QFrame#mediaTile[focused="true"] { border-color: #4c9ffe; }
QFrame#mediaHeader { background: $media_header; border-bottom: 1px solid $media_border; border-top-left-radius: 12px; border-top-right-radius: 12px; }
QLabel#tileTitle { color: $media_text; font-size: 12px; font-weight: 600; }
QLabel#tileMeta, QLabel#loadingText { color: $media_muted; font-size: 11px; }
QLabel#emptyTileText, QLabel#errorText { color: #b68b69; }
QPushButton#iconButton { background: transparent; border: 0; color: $media_muted; border-radius: 6px; padding: 3px 7px; font-size: 11px; }
QPushButton#iconButton:hover { background: $media_border; color: $media_text; }

QLabel#emptyMark { background: $primary; color: $primary_text; border-radius: 15px; font-size: 25px; font-weight: 700; }
QLabel#emptyTitle { color: $text; font-size: 21px; font-weight: 600; }
QLabel#emptyHint { color: $muted; font-size: 13px; }
QLabel#emptyFormats { color: $faint; font-size: 11px; }

QPushButton {
    min-height: 20px; background: $panel; color: $text; border: 1px solid $border_strong;
    border-radius: 8px; padding: 6px 11px;
}
QPushButton:hover { background: $hover; border-color: $border_strong; }
QPushButton:pressed { background: $pressed; }
QPushButton:disabled { color: $disabled; background: $elevated; border-color: $border; }
QPushButton#primaryButton { background: $primary; color: $primary_text; border-color: $primary; font-weight: 600; padding: 7px 14px; }
QPushButton#primaryButton:hover { background: $text; border-color: $text; }
QPushButton#acceptButton { background: #247a45; color: white; border-color: #32945a; font-weight: 600; padding: 8px 18px; }
QPushButton#acceptButton:hover { background: #2d8b51; }
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
QCheckBox::indicator:checked { border: 1px solid #4c9ffe; background: #4c9ffe; border-radius: 3px; }

QProgressBar { background: $elevated; border: 0; border-radius: 2px; max-height: 4px; }
QProgressBar::chunk { background: #4c9ffe; border-radius: 2px; }
QSlider::groove:horizontal { background: $border_strong; height: 3px; border-radius: 1px; }
QSlider::sub-page:horizontal { background: #4c9ffe; }
QSlider::handle:horizontal { background: $text; border: 2px solid $surface; width: 11px; margin: -5px 0; border-radius: 7px; }
QSplitter::handle { background: $border; }
QStatusBar { background: $window; color: $muted; border-top: 1px solid $border; }
QStatusBar::item { border: 0; }

QDialog#settingsDialog { background: transparent; border: 0; }
QFrame#dialogSurface {
    background: $surface; border: 1px solid $border_strong; border-radius: 14px;
}
QDialog#settingsPage {
    background: $surface; border: 0; border-radius: 0;
}
QFrame#dialogSurface[embedded="true"] {
    border: 0; border-radius: 0;
}
QFrame#settingsHeader {
    background: $window; border-bottom: 1px solid $border; min-height: 44px;
    border-top-left-radius: 14px; border-top-right-radius: 14px;
}
QFrame#settingsHeader[embedded="true"] {
    border-top-left-radius: 0; border-top-right-radius: 0;
}
QLabel#settingsTitle { color: $text; font-size: 15px; font-weight: 600; }
QPushButton#settingsCloseButton {
    min-height: 30px; background: transparent; border: 0; border-radius: 7px;
    color: $muted; font-size: 17px; padding: 0;
}
QPushButton#settingsCloseButton:hover { background: $hover; color: $text; }
QListWidget#settingsNav { background: $sidebar; border: 0; border-right: 1px solid $border; padding: 12px 8px; outline: 0; }
QListWidget#settingsNav::item { color: $muted; min-height: 24px; padding: 7px 11px; border-radius: 6px; }
QListWidget#settingsNav::item:hover { background: $hover; color: $text; }
QListWidget#settingsNav::item:selected { background: $nav_selected; color: $text; }
QWidget#settingsPages { background: $surface; }
QLabel#settingsPageTitle { color: $text; font-size: 20px; font-weight: 600; }
QLabel#settingsDescription { color: $muted; }
QFrame#settingsFooter {
    background: $surface; border-top: 1px solid $border;
    border-bottom-left-radius: 14px; border-bottom-right-radius: 14px;
}
QFrame#settingsFooter[embedded="true"] {
    border-bottom-left-radius: 0; border-bottom-right-radius: 0;
}
QFrame#settingRow { background: $panel; border: 1px solid $border; border-radius: 10px; }
QLabel#settingRowTitle { color: $text; font-weight: 600; }
QLabel#settingRowDescription { color: $muted; font-size: 11px; }

QFrame#segmentedControl { background: $elevated; border: 1px solid $border; border-radius: 9px; }
QPushButton#segmentButton { min-height: 24px; background: transparent; color: $muted; border: 0; border-radius: 7px; padding: 4px 12px; }
QPushButton#segmentButton:hover { background: $hover; color: $text; }
QPushButton#segmentButton:checked { background: $panel; color: $text; border: 1px solid $border_strong; font-weight: 600; }

QLineEdit, QPlainTextEdit, QSpinBox, QKeySequenceEdit {
    min-height: 22px; background: $input; color: $text; border: 1px solid $border_strong;
    border-radius: 8px; padding: 6px 9px; selection-background-color: #315f85;
}
QPushButton#choiceButton, QPushButton#settingsCombo {
    min-height: 22px; background: $input; color: $text; border: 1px solid $border_strong;
    border-radius: 8px; padding: 6px 28px 6px 9px;
}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QKeySequenceEdit:focus,
QPushButton#choiceButton:focus, QPushButton#settingsCombo:focus { border-color: #4c9ffe; }
QPushButton#calibrationPicker {
    min-height: 20px; padding: 3px 26px 3px 9px; background: $elevated;
    border: 1px solid $border; border-radius: 7px; text-align: left;
}
QPushButton#calibrationPicker:hover, QPushButton#choiceButton:hover, QPushButton#settingsCombo:hover {
    background: $hover; border-color: $border_strong;
}
QPushButton#choiceButton, QPushButton#settingsCombo { text-align: left; }
QMenu#choiceMenu {
    background: $panel; color: $text; border: 1px solid $border_strong;
    border-radius: 9px; padding: 5px;
}
QMenu#choiceMenu::item {
    min-height: 22px; padding: 6px 28px 6px 9px; border-radius: 6px;
}
QMenu#choiceMenu::item:selected {
    background: $hover; color: $text;
}
QMenu#choiceMenu::item:checked {
    background: $nav_selected; color: $text; font-weight: 600;
}
QMenu#choiceMenu::indicator { width: 0; height: 0; }
QLabel#calibrationDetail {
    background: $elevated; color: $muted; border: 1px solid $border;
    border-radius: 9px; padding: 10px 12px;
}
QSpinBox::up-button, QSpinBox::down-button { width: 18px; border: 0; background: transparent; }
QScrollBar:vertical { background: transparent; width: 9px; }
QScrollBar::handle:vertical { background: $border_strong; min-height: 30px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
""")


def style_for(theme: str) -> str:
    return STYLE.substitute(PALETTES.get(theme, PALETTES["dark"]))
