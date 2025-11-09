import os, sys
from PySide6 import QtWidgets, QtCore
from whisper_recorder import WhisperRecorder
from session_manager import SessionManager
from pathlib import Path
import soundfile as sf
import numpy as np
from set_path import RECORDINGS_DIR


class MainWindow(QtWidgets.QWidget):
    """Main application window with collapsible session manager + recorder."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙️ Whisper Notebook")
        self.resize(1200, 700)

        # === Layout ===
        main_layout = QtWidgets.QVBoxLayout(self)

        # --- Toolbar ---
        toolbar = QtWidgets.QHBoxLayout()
        self.toggle_sessions_btn = QtWidgets.QPushButton("📂 Sessions")
        self.toggle_sessions_btn.setCheckable(True)
        toolbar.addWidget(self.toggle_sessions_btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # --- Splitter: session manager + recorder ---
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.session_manager = SessionManager(str(RECORDINGS_DIR))
        self.session_manager.sessionSelected.connect(self.load_session)
        self.session_manager.setVisible(False)  # hidden by default
        self.recorder = WhisperRecorder()

        self.splitter.addWidget(self.session_manager)
        self.splitter.addWidget(self.recorder)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter)

        # === Connect ===
        self.toggle_sessions_btn.toggled.connect(self.toggle_sessions)
        self.session_manager.sessionSelected.connect(self.load_session)

    # --------------------------------------------------------
    # Sidebar behavior
    # --------------------------------------------------------
    def toggle_sessions(self, checked: bool):
        """Show/hide the session manager sidebar."""
        self.session_manager.setVisible(checked)
        if checked:
            self.toggle_sessions_btn.setText("📂 Hide Sessions")
            self.splitter.setSizes([300, 900])
        else:
            self.toggle_sessions_btn.setText("📂 Sessions")
            self.splitter.setSizes([0, 1200])

    # --------------------------------------------------------
    # Session loading
    # --------------------------------------------------------
    def load_session(self, audio_path: str, txt_path: str):
        """Load selected session from the SessionManager."""
        self.recorder.text_area.clear()

        # --- Load transcript ---
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                self.recorder.text_area.setPlainText(f.read())

        # --- Load waveform if widget exists ---
        if hasattr(self.recorder, "waveform") and self.recorder.waveform:
            try:
                data, sr = sf.read(audio_path)
                if data.ndim > 1:  # handle stereo
                    data = np.mean(data, axis=1)
                self.recorder.waveform.load_audio(data, sr)
            except Exception as e:
                self.recorder.text_area.append(f"⚠️ Could not load waveform: {e}\n")

        self.recorder.text_area.append(f"\n🎧 Loaded session: {audio_path}\n")

        # --- Auto-close the drawer ---
        self.toggle_sessions_btn.setChecked(False)
        self.session_manager.setVisible(False)
        self.toggle_sessions_btn.setText("📂 Sessions")

