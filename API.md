# API Documentation

Technical reference for developers extending Whisper Notebook.

## Module Overview

| Module | Purpose |
|--------|---------|
| `whisper_recorder` | Core recording and transcription engine |
| `waveform_widget` | Audio visualization with pyqtgraph |
| `session_manager` | Session browsing and selection |
| `post_process_transcript2` | LLM-based transcript cleanup |
| `styles` | Centralized theming |
| `set_path` | Path configuration |

---

## whisper_recorder.py

### WhisperRecorder

Main widget for audio recording and transcription.

```python
from whisper_recorder import WhisperRecorder

recorder = WhisperRecorder()
```

#### Methods

| Method | Description |
|--------|-------------|
| `start_recording()` | Begin audio capture and live transcription |
| `stop_recording()` | Stop capture, save files, run post-processing |
| `load_and_transcribe()` | Open file dialog and transcribe selected audio |
| `cancel_transcription()` | Cancel ongoing file transcription |
| `play_audio()` | Start audio playback |
| `pause_audio()` | Pause audio playback |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `recordings_dir` | `Path` | Directory for saved sessions |
| `sample_rate` | `int` | Audio sample rate (16000) |
| `model` | `WhisperModel` | Loaded Whisper model |
| `text_area` | `QTextEdit` | Transcript display widget |
| `waveform` | `WaveformWidget` | Audio visualization widget |

#### Signals

None. Uses direct method calls and Qt meta-object invocation for thread-safe UI updates.

---

## waveform_widget.py

### WaveformWidget

PyQtGraph-based audio waveform display with playback.

```python
from waveform_widget import WaveformWidget

widget = WaveformWidget()
widget.load_audio(audio_data, sample_rate)
widget.play()
```

#### Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `load_audio(data, samplerate)` | `np.ndarray`, `int` | Display waveform |
| `play()` | - | Start playback with cursor sync |
| `pause()` | - | Pause playback |
| `stop()` | - | Stop and reset cursor |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `audio_data` | `np.ndarray` | Current audio samples |
| `samplerate` | `int` | Sample rate in Hz |
| `playing` | `bool` | Playback state |
| `cursor` | `InfiniteLine` | Playback position indicator |

---

## session_manager.py

### SessionManager

Widget for browsing and selecting recorded sessions.

```python
from session_manager import SessionManager

manager = SessionManager("/path/to/recordings")
manager.sessionSelected.connect(on_session_selected)
```

#### Signals

| Signal | Parameters | Description |
|--------|------------|-------------|
| `sessionSelected` | `str, str` | Emitted with (audio_path, transcript_path) |

#### Methods

| Method | Description |
|--------|-------------|
| `populate_sessions()` | Scan directory and refresh table |
| `load_selected_session(row, col)` | Emit signal for selected row |

---

## post_process_transcript2.py

### Functions

#### process_transcript

Main pipeline for transcript cleanup.

```python
from post_process_transcript2 import process_transcript

cleaned_path = process_transcript("/path/to/transcript.txt")
```

**Parameters:**
- `txt_path` (str | Path): Path to transcript file

**Returns:**
- `Path | None`: Path to cleaned transcript, or None if failed

**Behavior:**
1. Detects timestamps `[MM:SS]`
2. Merges timestamped lines if present
3. Cleans grammar with local Mistral LLM

---

#### summarize_transcript

Generate metadata from transcript.

```python
from post_process_transcript2 import summarize_transcript

metadata = summarize_transcript("/path/to/transcript_clean.txt")
```

**Parameters:**
- `file_path` (str | Path): Path to transcript file

**Returns:**
- `dict | None`: Metadata with keys: `title`, `summary`, `duration`, `timestamp`, `cleaned`, `source`

---

#### merge_transcript

Combine timestamped lines into paragraph.

```python
from post_process_transcript2 import merge_transcript

merged_path = merge_transcript("/path/to/transcript.txt")
```

**Returns:**
- `Path`: Path to merged file (`*_merged.txt`)

---

#### clean_with_local_mistral

Grammar correction using local LLM.

```python
from post_process_transcript2 import clean_with_local_mistral

cleaned_path = clean_with_local_mistral("/path/to/transcript.txt")
```

**Returns:**
- `Path | None`: Path to cleaned file (`*_clean.txt`)

---

## styles.py

### Constants

#### COLORS

Color palette dictionary for theming.

```python
from styles import COLORS

bg_color = COLORS["bg_dark"]      # "#1e1e2e"
accent = COLORS["accent"]          # "#9d7cd8"
```

**Available keys:**
- `bg_dark`, `bg_surface`, `bg_hover`
- `border`
- `text_primary`, `text_secondary`, `text_disabled`
- `accent`, `accent_hover`, `accent_pressed`
- `success`, `warning`, `error`, `error_hover`

#### DARK_STYLESHEET

Complete QSS stylesheet string.

```python
from styles import DARK_STYLESHEET

widget.setStyleSheet(DARK_STYLESHEET)
```

### Functions

#### apply_theme

Apply dark theme to application.

```python
from styles import apply_theme

app = QApplication(sys.argv)
apply_theme(app)
```

---

## set_path.py

### Constants

| Constant | Description |
|----------|-------------|
| `RECORDINGS_DIR` | Absolute path to recordings directory |

```python
from set_path import RECORDINGS_DIR

print(RECORDINGS_DIR)  # /path/to/whisper/recordings
```

---

## Threading Model

### Audio Processing

- **Main thread**: Qt event loop, UI updates
- **Audio callback**: Called by sounddevice for each frame
- **Processing thread**: VAD detection, transcription triggering
- **Playback thread**: Cursor position updates during playback

### Thread Safety

- `threading.Event` used for `_running` and `_cancel` flags
- `QMetaObject.invokeMethod` for thread-safe UI updates
- `queue.Queue` for audio frame buffering

---

## Extending the Application

### Adding a New Button

```python
# In whisper_recorder.py __init__
self.my_btn = QtWidgets.QPushButton("My Button")
self.my_btn.setObjectName("myBtn")  # For stylesheet targeting
btns.addWidget(self.my_btn)
self.my_btn.clicked.connect(self.my_action)
```

### Custom Button Styling

```python
# In styles.py DARK_STYLESHEET
QPushButton#myBtn {
    background-color: #custom;
}
```

### Adding Session Metadata

```python
# In post_process_transcript2.py summarize_transcript
meta = {
    "title": ...,
    "summary": ...,
    "my_field": "custom value",  # Add here
}
```
