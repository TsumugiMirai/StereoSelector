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
    },
}


STYLE = Template(r"""
QWidget {
    color: $text;
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 13px;
}
QMainWindow { background: $window; }
QWidget#appRoot { background: $window; border: 1px solid $app_border; }
QWidget#workspace { background: $surface; }
QWidget#mediaWorkspace { background: #111214; }
QToolTip { background: $panel; color: $text; border: 1px solid $border_strong; border-radius: 5px; padding: 6px 8px; }

QFrame#titleBar { background: $window; border-bottom: 1px solid $border; }
QLabel#titleMark { background: $primary; color: $primary_text; border-radius: 6px; font-size: 10px; font-weight: 700; }
QLabel#titleBrand { color: $text; font-size: 12px; font-weight: 600; }
QLabel#titleContext { color: $muted; padding: 3px 14px; font-size: 11px; }
QPushButton#windowControlButton, QPushButton#closeWindowButton {
    background: transparent; border: 0; border-radius: 0; padding: 0; color: $muted;
}
QPushButton#windowControlButton:hover { background: $hover; color: $text; }
QPushButton#closeWindowButton:hover { background: #c42b1c; color: white; }
QPushButton#titleActionButton { background: transparent; border: 0; color: $muted; padding: 5px 10px; }
QPushButton#titleActionButton:hover { background: $hover; color: $text; }

QFrame#sidebar { background: $sidebar; border-right: 1px solid $border; }
QLabel#eyebrow, QLabel#sectionLabel { color: $faint; font-size: 10px; font-weight: 600; }
QLabel#projectName { color: $text; font-size: 15px; font-weight: 600; }
QLabel#projectPath, QLabel#muted { color: $muted; }
QLabel#countBadge { background: $elevated; color: $muted; border-radius: 8px; padding: 1px 7px; font-size: 11px; }
QLabel#acceptedCount { color: $text; font-size: 25px; font-weight: 600; }
QLabel#acceptedCaption { color: $muted; font-size: 11px; }
QFrame#reviewCard { background: $panel; border: 1px solid $border; border-radius: 9px; }

QFrame#workspaceHeader { background: $surface; border-bottom: 1px solid $border; }
QLabel#breadcrumb { color: $text; font-weight: 600; }
QLabel#viewHint { color: $muted; font-size: 11px; }
QPushButton#sideToggleButton {
    background: transparent; border: 1px solid transparent; color: $muted;
    border-radius: 6px; padding: 0; font-size: 22px;
}
QPushButton#sideToggleButton:hover { background: $hover; border-color: $border; color: $text; }

QFrame#reviewBar { background: $surface; border-top: 1px solid $border; }
QLabel#sampleTitle { color: $text; font-size: 14px; font-weight: 600; }
QLabel#sampleMeta { color: $muted; font-size: 11px; }
QLabel#statusPill { background: $elevated; color: $muted; border-radius: 9px; padding: 2px 9px; font-size: 11px; }
QLabel#statusPill[accepted="true"] { background: #173622; color: #78d59a; }

QFrame#mediaTile { background: #101113; border: 1px solid #303236; border-radius: 8px; }
QFrame#mediaTile[missing="true"] { border-color: #745037; }
QFrame#mediaTile[focused="true"] { border-color: #4c9ffe; }
QFrame#mediaHeader { background: #1b1c1f; border-bottom: 1px solid #303236; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QLabel#tileTitle { color: #ededed; font-size: 12px; font-weight: 600; }
QLabel#tileMeta, QLabel#loadingText { color: #929292; font-size: 11px; }
QLabel#emptyTileText, QLabel#errorText { color: #b68b69; }
QPushButton#iconButton { background: transparent; border: 0; color: #999999; padding: 3px 7px; font-size: 11px; }
QPushButton#iconButton:hover { background: #303236; color: #ffffff; }

QLabel#emptyMark { background: $primary; color: $primary_text; border-radius: 15px; font-size: 25px; font-weight: 700; }
QLabel#emptyTitle { color: $text; font-size: 21px; font-weight: 600; }
QLabel#emptyHint { color: $muted; font-size: 13px; }
QLabel#emptyFormats { color: $faint; font-size: 11px; }

QPushButton {
    min-height: 20px; background: $panel; color: $text; border: 1px solid $border_strong;
    border-radius: 6px; padding: 6px 11px;
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

QDialog#settingsDialog { background: $surface; border: 1px solid $border_strong; }
QFrame#settingsHeader { background: $window; border-bottom: 1px solid $border; min-height: 44px; }
QLabel#settingsTitle { color: $text; font-size: 15px; font-weight: 600; }
QPushButton#settingsCloseButton { background: transparent; border: 0; border-radius: 0; font-size: 18px; }
QPushButton#settingsCloseButton:hover { background: #c42b1c; color: white; }
QListWidget#settingsNav { background: $sidebar; border: 0; border-right: 1px solid $border; padding: 12px 8px; outline: 0; }
QListWidget#settingsNav::item { color: $muted; min-height: 24px; padding: 7px 11px; border-radius: 6px; }
QListWidget#settingsNav::item:hover { background: $hover; color: $text; }
QListWidget#settingsNav::item:selected { background: $nav_selected; color: $text; }
QWidget#settingsPages { background: $surface; }
QLabel#settingsPageTitle { color: $text; font-size: 20px; font-weight: 600; }
QLabel#settingsDescription { color: $muted; }
QFrame#settingsFooter { background: $surface; border-top: 1px solid $border; }
QFrame#settingRow { background: $panel; border: 1px solid $border; border-radius: 8px; }
QLabel#settingRowTitle { color: $text; font-weight: 600; }
QLabel#settingRowDescription { color: $muted; font-size: 11px; }

QFrame#segmentedControl { background: $elevated; border: 1px solid $border; border-radius: 7px; }
QPushButton#segmentButton { min-height: 24px; background: transparent; color: $muted; border: 0; border-radius: 5px; padding: 4px 12px; }
QPushButton#segmentButton:hover { background: $hover; color: $text; }
QPushButton#segmentButton:checked { background: $panel; color: $text; border: 1px solid $border_strong; font-weight: 600; }

QLineEdit, QSpinBox, QKeySequenceEdit {
    min-height: 22px; background: $input; color: $text; border: 1px solid $border_strong;
    border-radius: 6px; padding: 6px 9px; selection-background-color: #315f85;
}
QLineEdit:focus, QSpinBox:focus, QKeySequenceEdit:focus { border-color: #4c9ffe; }
QSpinBox::up-button, QSpinBox::down-button { width: 18px; border: 0; background: transparent; }
QScrollBar:vertical { background: transparent; width: 9px; }
QScrollBar::handle:vertical { background: $border_strong; min-height: 30px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
""")


def style_for(theme: str) -> str:
    return STYLE.substitute(PALETTES.get(theme, PALETTES["dark"]))
