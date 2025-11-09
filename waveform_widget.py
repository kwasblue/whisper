import sounddevice as sd
import numpy as np
import pyqtgraph as pg
import time, threading


class WaveformWidget(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground("k")
        self.plotItem.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plotItem.plot(pen=pg.mkPen("c", width=1))
        self.cursor = self.plotItem.addLine(x=0, pen=pg.mkPen("y", width=2))
        self.audio_data = None
        self.samplerate = 16000
        self.play_thread = None
        self.playing = False

    def load_audio(self, data, samplerate):
        """Display the waveform."""
        self.audio_data = data
        self.samplerate = samplerate
        t = np.arange(len(data)) / samplerate
        self.curve.setData(t, data)
        self.cursor.setValue(0)
        self.setLabel('bottom', 'Time (s)')
        self.setLabel('left', 'Amplitude')

    def play(self):
        if self.audio_data is None:
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
        self.playing = False
        sd.stop()

    def stop(self):
        self.playing = False
        sd.stop()
        self.cursor.setValue(0)