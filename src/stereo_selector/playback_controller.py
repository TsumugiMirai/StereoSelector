"""Timeline playback behaviour for the main window."""

from __future__ import annotations


class PlaybackController:

    def _seek_frame(self, index: int, *, pause: bool = True) -> None:
        if self.dataset is None or not self.dataset.samples:
            return
        if pause:
            self._resume_playback_after_scrub = False
            self._stop_playback()
        index = max(0, min(index, len(self.dataset.samples) - 1))
        if index != self.current_index:
            self.current_index = index
            self._show_current()

    def previous_sample(self) -> None:
        self._seek_frame(self.current_index - 1)

    def next_sample(self) -> None:
        self._seek_frame(self.current_index + 1)

    def first_sample(self) -> None:
        self._seek_frame(0)

    def last_sample(self) -> None:
        if self.dataset is not None:
            self._seek_frame(len(self.dataset.samples) - 1)

    def _timeline_preview_changed(self, value: int) -> None:
        self.timeline_left.setText(str(value + 1))

    def _timeline_preview_cleared(self) -> None:
        value = self.current_index + 1 if self.dataset is not None else 0
        self.timeline_left.setText(str(value))

    def _timeline_scrub_started(self) -> None:
        self._resume_playback_after_scrub = self._playback_timer.isActive()
        if self._resume_playback_after_scrub:
            self._playback_timer.stop()
            self.playback_button.set_playing(False)

    def _timeline_scrub_finished(self) -> None:
        should_resume = self._resume_playback_after_scrub
        self._resume_playback_after_scrub = False
        self._timeline_preview_cleared()
        if (
            should_resume
            and self.dataset is not None
            and self.current_index < len(self.dataset.samples) - 1
        ):
            self.toggle_playback()
        elif should_resume:
            # Preserve the normal stop path so analysis disabled during
            # playback is refreshed for the final seeked frame.
            self._playback_timer.start()
            self._stop_playback()

    def _playback_speed_changed(self, index: int) -> None:
        try:
            fps = float(self.playback_speed.itemData(index))
        except (TypeError, ValueError):
            fps = 2.0
        self._playback_fps = max(0.5, min(10.0, fps))
        self._playback_timer.setInterval(max(40, round(1000 / self._playback_fps)))

    def toggle_playback(self) -> None:
        if self._playback_timer.isActive():
            self._stop_playback()
            return
        if self.dataset is None or len(self.dataset.samples) < 2:
            return
        if self.current_index >= len(self.dataset.samples) - 1:
            self.current_index = 0
            self._show_current()
        for tile in self.tile_pool.values():
            tile.set_analysis_enabled(False)
        self._playback_timer.start()
        self.playback_button.set_playing(True)
        self.playback_button.setToolTip(
            f"暂停 ({self._shortcut_text('playback')})"
        )
        self.status_text.setText(f"正在播放 · {self._playback_fps:g} fps")

    def _playback_tick(self) -> None:
        if self.dataset is None or not self.dataset.samples:
            self._stop_playback()
            return
        # Do not skip a frame while its visible media is still loading. This
        # keeps point-cloud playback ordered even at a requested high speed.
        if any(tile.is_loading() for tile in self.tiles.values()):
            return
        if self.current_index >= len(self.dataset.samples) - 1:
            self._stop_playback()
            return
        self.current_index += 1
        self._show_current()

    def _stop_playback(self) -> None:
        was_active = self._playback_timer.isActive()
        self._playback_timer.stop()
        if hasattr(self, "playback_button"):
            self.playback_button.set_playing(False)
            self.playback_button.setToolTip(
                f"播放 / 暂停 ({self._shortcut_text('playback')})"
            )
        for tile in self.tile_pool.values():
            tile.set_analysis_enabled(True)
        depth_tile = self.tile_pool.get("depth_fsd")
        if was_active:
            for tile in self.tiles.values():
                if tile.image_data is not None:
                    tile.refresh_analysis()
        if was_active and depth_tile is not None and "depth_fsd" not in self.tiles:
            current = (
                self.dataset.samples[self.current_index].files.get("depth_fsd")
                if self.dataset is not None and self.dataset.samples
                else None
            )
            if depth_tile.current_path != current:
                depth_tile.show_file(current)
            else:
                depth_tile.refresh_analysis()
        if was_active and hasattr(self, "status_text"):
            self.status_text.setText("")

