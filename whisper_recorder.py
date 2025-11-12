import os, queue, threading, time, wave, datetime, librosa, tempfile, soundfile
import webrtcvad, torch
import numpy as np
import sounddevice as sd
from pathlib import Path
from PySide6 import QtWidgets, QtCore
from faster_whisper import WhisperModel
from post_process_transcript2 import process_transcript, summarize_transcript
from waveform_widget import WaveformWidget
from set_path import RECORDINGS_DIR
import ctranslate2


# === Whisper Recorder ===
class WhisperRecorder(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙️ Whisper Recorder (VAD + Timestamped + Waveform)")
        self.resize(900, 600)
        
        # --- Centralized path setup ---
        self.recordings_dir = Path(RECORDINGS_DIR)
        self.recordings_dir.mkdir(exist_ok=True)
        print(f"📂 WhisperRecorder using directory: {self.recordings_dir}")
        # ==== GUI ====
        layout = QtWidgets.QVBoxLayout(self)

        # Splitter: waveform (top) + text (bottom)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.waveform = WaveformWidget()
        self.text_area = QtWidgets.QTextEdit(readOnly=True)
        self.splitter.addWidget(self.waveform)
        self.splitter.addWidget(self.text_area)
        self.splitter.setSizes([350, 250])
        layout.addWidget(self.splitter)

        # Buttons
        self.start_btn = QtWidgets.QPushButton("▶️ Start Recording")
        self.stop_btn = QtWidgets.QPushButton("⏹ Stop Recording")
        self.play_btn = QtWidgets.QPushButton("🔊 Play")
        self.pause_btn = QtWidgets.QPushButton("⏸ Pause")
        self.load_btn = QtWidgets.QPushButton("📂 Load Audio File")
        self.cancel_btn = QtWidgets.QPushButton("❌ Cancel Transcription")

        self.stop_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        # Button Layout
        btns = QtWidgets.QHBoxLayout()
        for b in [self.start_btn, self.stop_btn, self.play_btn, self.pause_btn, self.load_btn, self.cancel_btn]:
            btns.addWidget(b)
        layout.addLayout(btns)

        # Connect
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.load_btn.clicked.connect(self.load_and_transcribe)
        self.cancel_btn.clicked.connect(self.cancel_transcription)
        self.play_btn.clicked.connect(self.play_audio)
        self.pause_btn.clicked.connect(self.pause_audio)

        # ==== Audio setup ====
        self.sample_rate = 16000
        self.frame_ms = 30
        self.frame_len = int(self.sample_rate * self.frame_ms / 1000)
        self.vad = webrtcvad.Vad(2)
        self.audio_q = queue.Queue()
        self.running = False

        # ==== Model ====
        try:
            has_cuda = ctranslate2.get_cuda_device_count()> 0
        except Exception:
            has_cuda = False

        if has_cuda:
            device = "cuda"
            compute_type = "float16"   # good for real NVIDIA GPU
        else:
            device = "cpu"
            compute_type = "int8"      # good default for CPU

        self.model = WhisperModel("small", device=device, compute_type=compute_type)
        self.text_area.append(f"✅ Loaded Whisper 'small' on {device} ({compute_type})\n")
        #os.makedirs("recordings", exist_ok=True)
        self.text_file = None
        self.wave_file = None

    # === Audio callback ===
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        if self.running:
            self.audio_q.put(indata.copy())

    # === Start ===
    def start_recording(self):
        if self.running:
            return
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.text_area.append("🎧 Listening...\n")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audio_path = self.recordings_dir / f"session_{ts}.wav"
        self.text_path = self.recordings_dir / f"session_{ts}.txt"
        self.text_file = open(self.text_path, "w", encoding="utf-8")
        self.wave_file = wave.open(str(self.audio_path), 'wb')
        self.wave_file.setnchannels(1)
        self.wave_file.setsampwidth(2)
        self.wave_file.setframerate(self.sample_rate)

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_len,
            callback=self._audio_callback,
        )
        self.stream.start()

        threading.Thread(target=self._process_audio, daemon=True).start()

    # === Stop ===
    def stop_recording(self):
        if not self.running:
            return
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.text_area.append("\n🛑 Stopped.\n")

        if self.stream:
            self.stream.stop(); self.stream.close()
        if self.text_file:
            self.text_file.close()
        if self.wave_file:
            self.wave_file.close()

        self.text_area.append(f"💾 Audio saved to: {self.audio_path}\n")
        self.text_area.append(f"💾 Transcript saved to: {self.text_path}\n")

        # Load waveform
        data, sr = soundfile.read(self.audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        self.waveform.load_audio(data, sr)

        try:
            cleaned_path = process_transcript(self.text_path)
            self.text_area.append(f"✨ Cleaned transcript saved to: {cleaned_path}\n")
                # === New: generate metadata ===
            meta = summarize_transcript(cleaned_path)
            self.text_area.append(f"🧠 Session titled: “{meta['title']}”\n")
        except Exception as e:
            self.text_area.append(f"⚠️ Post-processing failed: {e}\n")

    # === Processing ===
    def _process_audio(self):
        ring = bytearray()
        silence_frames = 0
        speaking = False
        start_time = time.time()

        while self.running:
            try:
                indata = self.audio_q.get(timeout=1)
            except queue.Empty:
                continue

            self.wave_file.writeframes(indata.tobytes())

            frame_bytes = indata.tobytes()
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

            if is_speech:
                ring.extend(frame_bytes)
                silence_frames = 0
                speaking = True
            elif speaking:
                silence_frames += 1
                if silence_frames > 10:
                    self._transcribe_utterance(ring, start_time)
                    ring.clear()
                    silence_frames = 0
                    speaking = False
                    start_time = time.time()

        print("Processing thread ended.")

    def _transcribe_utterance(self, audio_bytes, start_time):
        if not audio_bytes:
            return
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_np, language="en")
        text = " ".join([s.text for s in segments]).strip()
        if text:
            elapsed = int(time.time() - start_time)
            stamp = time.strftime("[%M:%S]", time.gmtime(elapsed))
            line = f"{stamp} {text}"
            self.text_file.write(line + "\n")
            self.text_file.flush()
            QtCore.QMetaObject.invokeMethod(
                self.text_area, "append",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"🗣️ {line}")
            )

    # === Cancel ===
    def cancel_transcription(self):
        self.cancel_flag = True
        self.text_area.append("\n⏹ Cancel requested — will stop after current chunk.\n")

    # === Playback ===
    def play_audio(self):
        self.waveform.play()

    def pause_audio(self):
        self.waveform.pause()

    # === Load ===
    def load_and_transcribe(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.m4a *.flac)"
        )
        if not file_path:
            return

        self.text_area.append(f"\n🎧 Loading file: {file_path}\n")
        self.cancel_flag = False
        self.cancel_btn.setEnabled(True)
        self.load_btn.setEnabled(False)

        # Convert to mono/16k
        try:
            data, sr = soundfile.read(file_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != self.sample_rate:
                self.text_area.append("Resampling to 16 kHz mono...\n")
                data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
            temp_path = tempfile.mktemp(suffix=".wav")
            soundfile.write(temp_path, data, self.sample_rate)
            self.waveform.load_audio(data, self.sample_rate)
        except Exception as e:
            self.text_area.append(f"⚠️ Could not read audio: {e}\n")
            self.cancel_btn.setEnabled(False)
            self.load_btn.setEnabled(True)
            return

        # Progress bar and transcription
        progress = QtWidgets.QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        self.layout().addWidget(progress)
        self.text_area.append("🪶 Starting transcription...\n")
        self.repaint()

        start_time = time.time()
        total_dur = len(data) / self.sample_rate
        text_segments = []
        processed = 0

        try:
            segments, _ = self.model.transcribe(temp_path, language="en")
            for segment in segments:
                if self.cancel_flag:
                    self.text_area.append("\n🛑 Transcription canceled.\n")
                    progress.deleteLater()
                    self.cancel_btn.setEnabled(False)
                    self.load_btn.setEnabled(True)
                    return
                text_segments.append(segment.text)
                processed = segment.end
                percent = min(int((processed / total_dur) * 100), 100)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (total_dur - processed) / rate if rate > 0 else 0
                eta = time.strftime("%M:%S", time.gmtime(max(0, remaining)))
                progress.setValue(percent)
                progress.setFormat(f"⏳ {percent}% | ETA {eta}")
                QtCore.QCoreApplication.processEvents()
        except Exception as e:
            self.text_area.append(f"⚠️ Transcription failed: {e}\n")
            progress.deleteLater()
            self.cancel_btn.setEnabled(False)
            self.load_btn.setEnabled(True)
            return

        # Save transcript
        text = " ".join(text_segments).strip()
        base = os.path.basename(file_path)
        txt_out = self.recordings_dir / f"{Path(base).stem}_transcript.txt"
        self.recordings_dir.mkdir(exist_ok=True)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(text)
        progress.setFormat("✅ Transcription complete")
        progress.setValue(100)
        self.text_area.append(f"🗣️ Transcript saved: {txt_out.resolve()}\n")

        # Post-process
        try:
            cleaned = process_transcript(txt_out)
            self.text_area.append(f"✨ Cleaned transcript saved to: {cleaned}\n")
        except Exception as e:
            self.text_area.append(f"⚠️ Post-processing failed: {e}\n")

        progress.deleteLater()
        self.cancel_btn.setEnabled(False)
        self.load_btn.setEnabled(True)