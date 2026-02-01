"""
Audio waveform visualization widget with playback support.

This module provides a Qt widget for displaying audio waveforms using pyqtgraph,
with synchronized cursor tracking during playback.
"""

import sounddevice as sd
import numpy as np
import pyqtgraph as pg
import time, threading

from styles import COLORS


class WaveformWidget(pg.PlotWidget):
    """
    A widget for displaying audio waveforms with playback cursor.

    Features:
        - Real-time waveform visualization
        - Audio playback with synchronized cursor
        - Play/pause/stop controls
    """

    def __init__(self):
        super().__init__()

        # Theme styling - reuse colors from centralized palette
        self.setBackground(COLORS["bg_dark"])
        self.plotItem.showGrid(x=True, y=True, alpha=0.2)

        axis_pen = pg.mkPen(COLORS["text_secondary"], width=1)
        self.plotItem.getAxis('bottom').setPen(axis_pen)
        self.plotItem.getAxis('left').setPen(axis_pen)
        self.plotItem.getAxis('bottom').setTextPen(axis_pen)
        self.plotItem.getAxis('left').setTextPen(axis_pen)

        # Waveform curve with purple accent
        self.curve = self.plotItem.plot(pen=pg.mkPen(COLORS["accent"], width=1.5))

        # Playback cursor with bright purple
        self.cursor = self.plotItem.addLine(x=0, pen=pg.mkPen(COLORS["accent_hover"], width=2))

        self.audio_data = None
        self.samplerate = 16000
        self.play_thread = None
        self.playing = False

    def load_audio(self, data, samplerate):
        """
        Load and display audio waveform.

        Args:
            data: Audio samples as a numpy array.
            samplerate: Sample rate in Hz.
        """
        self.audio_data = data
        self.samplerate = samplerate
        t = np.arange(len(data)) / samplerate
        self.curve.setData(t, data)
        self.cursor.setValue(0)
        self.setLabel('bottom', 'Time (s)')
        self.setLabel('left', 'Amplitude')

    def play(self):
        """Start audio playback with synchronized cursor movement."""
        if self.audio_data is None:
            return
        if self.playing:  # Prevent multiple play threads
            return
        self.playing = True
        sd.play(self.audio_data, self.samplerate)
        total_dur = len(self.audio_data) / self.samplerate
        start = time.time()

        def update_cursor():
            while self.playing:
                elapsed = time.time() - start
                self.cursor.setValue(min(elapsed, total_dur))
                time.sleep(0.02)
            self.cursor.setValue(0)

        self.play_thread = threading.Thread(target=update_cursor, daemon=True)
        self.play_thread.start()

    def pause(self):
        """Pause audio playback."""
        self.playing = False
        sd.stop()

    def stop(self):
        """Stop audio playback and reset cursor to start."""
        self.playing = False
        sd.stop()
        self.cursor.setValue(0)