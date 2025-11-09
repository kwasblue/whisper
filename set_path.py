"""
paths.py — Centralized path management for Whisper Notebook
Ensures consistent, persistent file locations across source and packaged builds.
"""

import sys
from pathlib import Path

# ------------------------------------------------------------
# Determine root directory (handles both .py and PyInstaller .exe)
# ------------------------------------------------------------
if getattr(sys, "frozen", False):
    # 🧊 Running as packaged executable
    APP_ROOT = Path(sys.executable).parent
else:
    # 🧠 Running from source
    APP_ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Recordings directory (persistent)
# ------------------------------------------------------------
RECORDINGS_DIR = APP_ROOT / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

print(f"📂 [paths] Using recordings dir: {RECORDINGS_DIR}")

# ------------------------------------------------------------
# (Optional) Models directory — if you later bundle your GGUF or Whisper models
# ------------------------------------------------------------
MODELS_DIR = APP_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
