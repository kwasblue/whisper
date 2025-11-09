import os, sys, json
from PySide6 import QtWidgets, QtCore
from datetime import datetime
from pathlib import Path
from set_path import RECORDINGS_DIR


class SessionManager(QtWidgets.QWidget):
    """Displays a list of recorded sessions (with metadata) and emits a signal when one is selected."""
    
    sessionSelected = QtCore.Signal(str, str)  # audio_path, transcript_path

    def __init__(self, recordings_dir=str(RECORDINGS_DIR)):
        super().__init__()

        # --- Use absolute recordings directory ---
        base_dir = Path(getattr(sys, '_MEIPASS', Path.cwd()))  # Works in PyInstaller too
        if recordings_dir:
            self.recordings_dir = Path(recordings_dir).resolve()
        else:
            self.recordings_dir = (base_dir / "recordings").resolve()

        self.recordings_dir.mkdir(exist_ok=True, parents=True)
        print(f"📂 SessionManager using directory: {self.recordings_dir}")

        self.setWindowTitle("📂 Sessions")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # --- Header ---
        header_layout = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("🔄 Refresh")
        self.title_label = QtWidgets.QLabel("<b>Session Library</b>")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.refresh_btn)
        layout.addLayout(header_layout)

        # --- Table ---
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Title", "Date", "Summary", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # --- Connect ---
        self.refresh_btn.clicked.connect(self.populate_sessions)
        self.table.cellDoubleClicked.connect(self.load_selected_session)

        # --- Initialize ---
        self.populate_sessions()

    # ============================================================
    # POPULATE TABLE
    # ============================================================
    def populate_sessions(self):
        """Scan the recordings directory and list sessions using metadata if available."""
        self.table.setRowCount(0)
        if not self.recordings_dir.exists():
            print(f"⚠️ Recordings directory missing: {self.recordings_dir}")
            return

        sessions = []

        # --- Load JSON metadata if available ---
        for meta_file in self.recordings_dir.glob("*.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                base_name = meta.get("source") or meta_file.stem.replace("_clean", "").replace("_transcript", "")
                wav_path = self.recordings_dir / f"{Path(base_name).stem}.wav"
                txt_path = self.recordings_dir / f"{Path(base_name).stem}.txt"

                sessions.append({
                    "title": meta.get("title", Path(base_name).stem),
                    "summary": (meta.get("summary", "")[:90] + "...") if meta.get("summary") else "",
                    "date": meta.get("timestamp", ""),
                    "status": "✨ Cleaned" if meta.get("cleaned", False) else "📝 Raw",
                    "audio": str(wav_path),
                    "text": str(txt_path)
                })
            except Exception as e:
                print(f"⚠️ Error reading metadata {meta_file.name}: {e}")

        # --- Fallback: find any unprocessed sessions ---
        for wav_file in self.recordings_dir.glob("*.wav"):
            name = wav_file.stem
            if not any(name in s["audio"] for s in sessions):
                txt_path = self.recordings_dir / f"{name}.txt"
                sessions.append({
                    "title": name,
                    "summary": "(no metadata)",
                    "date": self._extract_date(name),
                    "status": "📝 Raw",
                    "audio": str(wav_file),
                    "text": str(txt_path)
                })

        # --- Sort by date descending ---
        sessions.sort(key=lambda s: s.get("date", ""), reverse=True)

        # --- Populate table ---
        for s in sessions:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(s["title"]))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(self._format_date(s["date"])))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(s["summary"]))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(s["status"]))
            # Store file paths as metadata
            self.table.item(row, 0).setData(QtCore.Qt.UserRole, (s["audio"], s["text"]))

    # ============================================================
    # LOAD SESSION
    # ============================================================
    def load_selected_session(self, row, _col):
        """Emit the selected session paths when a user double-clicks a row."""
        item = self.table.item(row, 0)
        if not item:
            return
        audio_path, text_path = item.data(QtCore.Qt.UserRole)
        if os.path.exists(audio_path):
            print(f"📥 Loading session: {audio_path}")
            self.sessionSelected.emit(audio_path, text_path)
        else:
            print(f"⚠️ Missing audio file: {audio_path}")

    # ============================================================
    # HELPERS
    # ============================================================
    def _extract_date(self, name):
        ts_str = name.replace("session_", "")
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            return dt.isoformat()
        except Exception:
            return ""

    def _format_date(self, iso_str):
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.strftime("%b %d, %Y  %H:%M")
        except Exception:
            return "Unknown"
