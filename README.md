# Whisper Notebook

A desktop application for real-time audio recording and transcription using OpenAI's Whisper model, with local LLM-powered transcript cleanup.

## Features

- **Real-time Recording** - Capture audio with automatic voice activity detection
- **Live Transcription** - See transcribed text as you speak
- **Waveform Visualization** - Visual audio display with playback cursor
- **File Transcription** - Load and transcribe existing audio files (WAV, MP3, M4A, FLAC)
- **Grammar Cleanup** - Automatic post-processing with local Mistral LLM
- **Session Library** - Browse and reload previous recordings
- **Dark Theme** - Modern purple-accented dark interface

## Requirements

- Python 3.10+
- macOS, Windows, or Linux
- ~4GB RAM (for Whisper model)
- Optional: NVIDIA GPU with CUDA for faster transcription

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whisper.git
cd whisper
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Mistral model for transcript cleanup (optional):
```bash
mkdir -p models/mistral
# Download mistral-7b-instruct-v0.2.Q4_K_M.gguf from HuggingFace
# Place it in models/mistral/
```

## Usage

### Starting the Application

```bash
python3 run_app.py
```

### Recording

1. Click **Start** to begin recording
2. Speak naturally - the app detects speech and transcribes automatically
3. Click **Stop** when finished
4. Transcript is saved and cleaned automatically

### Loading Audio Files

1. Click **Load File**
2. Select an audio file (WAV, MP3, M4A, FLAC)
3. Watch the progress bar as transcription runs
4. Click **Cancel** to stop early if needed

### Playback

1. After recording or loading, click **Play** to hear the audio
2. The cursor syncs with playback position
3. Click **Pause** to stop

### Session Library

1. Click **Sessions** in the toolbar to open the sidebar
2. Double-click any session to reload it
3. Sessions show title, date, summary, and cleanup status

## Project Structure

```
whisper/
├── run_app.py              # Application entry point
├── mainwindow.py           # Main window with sidebar
├── whisper_recorder.py     # Recording and transcription
├── waveform_widget.py      # Audio visualization
├── session_manager.py      # Session browsing
├── post_process_transcript2.py  # LLM cleanup pipeline
├── styles.py               # Dark theme styling
├── set_path.py             # Path configuration
├── recordings/             # Saved sessions
└── models/                 # LLM models
    └── mistral/
```

## Configuration

### Audio Settings

Default settings in `whisper_recorder.py`:
- Sample rate: 16kHz
- VAD mode: 2 (moderate sensitivity)
- Whisper model: "small"

### Model Selection

The app auto-detects CUDA and uses:
- **GPU**: float16 precision
- **CPU**: int8 precision

To change the Whisper model size, edit `whisper_recorder.py`:
```python
self.model = WhisperModel("small", ...)  # Options: tiny, base, small, medium, large
```

## Output Files

For each session, the following files are created in `recordings/`:

| File | Description |
|------|-------------|
| `session_YYYYMMDD_HHMMSS.wav` | Raw audio |
| `session_YYYYMMDD_HHMMSS.txt` | Timestamped transcript |
| `session_*_merged.txt` | Merged transcript |
| `session_*_clean.txt` | Grammar-corrected transcript |
| `session_*_clean.json` | Session metadata |

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Start Recording | Click Start button |
| Stop Recording | Click Stop button |
| Toggle Sessions | Click Sessions button |

## Troubleshooting

### "No module named 'sounddevice'"
```bash
pip install sounddevice
```

### "Could not find CUDA"
The app will fall back to CPU mode automatically. For GPU support, install PyTorch with CUDA.

### "Missing local model"
Download the Mistral GGUF model and place it in `models/mistral/`.

### Recording not working
Check microphone permissions in your system settings.

## License

MIT License
