"""Settings dialog and preference application behaviour."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from .settings import SettingsDialog
from .theme import PALETTES, palette_for, style_for


class SettingsController:

    def _set_shortcuts_enabled(self, enabled: bool) -> None:
        """Enable or disable the window shortcuts (used while the settings page is open)."""
        for shortcut in self.shortcuts:
            shortcut.setEnabled(enabled)

    def _apply_preferences(self) -> None:
        app = QApplication.instance()
        app.setStyleSheet(style_for(self.preferences.theme))
        app.setPalette(palette_for(self.preferences.theme))
        media_color = PALETTES[self.preferences.theme]["media_workspace"]
        for tile in self.tile_pool.values():
            tile.set_canvas_background(media_color)
        for action, shortcut in self.action_shortcuts.items():
            shortcut.setKey(QKeySequence(self.preferences.shortcuts[action]))
        self.reset_button.setText("重置视图")
        self.playback_previous_button.setToolTip(f"上一组 ({self._shortcut_text('previous')})")
        self.playback_next_button.setToolTip(f"下一组 ({self._shortcut_text('next')})")
        self.playback_button.setToolTip(f"播放 / 暂停 ({self._shortcut_text('playback')})")
        self.reset_button.setToolTip(f"重置视图 ({self._shortcut_text('reset')})")

    def open_settings(self) -> None:
        self._stop_playback()
        self._reset_interaction_tools()
        if self._settings_page is not None:
            return
        dialog = SettingsDialog(
            self.preferences,
            self,
            calibration_options=self.calibration_options,
            current_calibration_id=self.current_calibration_id,
            project_available=self.dataset is not None,
        )
        dialog.set_embedded()
        self._settings_page = dialog
        self._settings_previous_context = (
            self.title_bar.context.text(),
            self.title_bar.context.isVisible(),
        )
        dialog.accepted.connect(lambda current=dialog: self._finish_settings(current, True))
        dialog.rejected.connect(lambda current=dialog: self._finish_settings(current, False))
        self.page_stack.addWidget(dialog)
        self.page_stack.setCurrentWidget(dialog)
        self.title_bar.set_context("设置")
        self.title_bar.actions.hide()
        self.title_bar.settings_button.hide()
        self.app_status_bar.hide()
        self._set_shortcuts_enabled(False)
        dialog.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def _finish_settings(self, dialog: SettingsDialog, save: bool) -> None:
        if dialog is not self._settings_page:
            return
        if save:
            self._apply_settings(dialog)
        self.page_stack.setCurrentWidget(self.main_page)
        self.page_stack.removeWidget(dialog)
        dialog.deleteLater()
        self._settings_page = None
        previous_context, was_visible = self._settings_previous_context
        self.title_bar.set_context(previous_context if was_visible else "")
        self.title_bar.actions.show()
        self.title_bar.settings_button.hide()
        self.app_status_bar.hide()
        self._set_shortcuts_enabled(True)

    def _apply_settings(self, dialog: SettingsDialog) -> None:
        point_limit_changed = dialog.preferences.point_limit != self.preferences.point_limit
        cloud_render_changed = (
            dialog.preferences.cloud_cam_offset,
            dialog.preferences.cloud_grid,
            dialog.preferences.cloud_dot_radius,
            dialog.preferences.cloud_z_max,
            dialog.preferences.cloud_tau_rel,
            dialog.preferences.cloud_occlusion,
        ) != (
            self.preferences.cloud_cam_offset,
            self.preferences.cloud_grid,
            self.preferences.cloud_dot_radius,
            self.preferences.cloud_z_max,
            self.preferences.cloud_tau_rel,
            self.preferences.cloud_occlusion,
        )
        calibration_changed = (
            self.dataset is not None
            and dialog.selected_calibration_id != self.current_calibration_id
        )
        depth_unit_changed = dialog.preferences.depth_unit != self.preferences.depth_unit
        self.preferences = dialog.preferences
        self.preferences.save(self.settings)
        self.calibration_options = list(dialog.calibration_options)
        self._save_calibration_options()
        if calibration_changed:
            self._apply_calibration_selection(
                dialog.selected_calibration_id,
                rebuild=False,
            )
        else:
            self._refresh_calibration_picker(self.current_calibration_id)
        self._apply_preferences()
        if (
            point_limit_changed
            or cloud_render_changed
            or calibration_changed
        ) and self.dataset is not None:
            # Tiles are about to be destroyed; no rectification callback may
            # touch them afterwards.
            self._invalidate_rectification()
            self._rectified_cache.clear()
            self.focused_modality = None
            self._build_tile_pool()
            self._rebuild_tiles()
            self._show_current()
        elif depth_unit_changed and self.dataset is not None:
            self._update_depth_quality()
            self._update_inspector()
        self.status_text.setText("设置已保存")

