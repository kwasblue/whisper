import re
from pathlib import Path
from llama_cpp import Llama
from datetime import datetime
import json, re, wave


# === PATHS ===
LOCAL_MISTRAL_PATH = Path(
    r"whisper\models\mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

# === HELPERS ===
def merge_transcript(txt_path):
    """Combine timestamped lines into chronological paragraph."""
    txt_path = Path(txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    for line in lines:
        m = re.match(r"\[(\d+):(\d+)\]\s*(.*)", line.strip())
        if m:
            minutes, seconds, text = m.groups()
            t = int(minutes) * 60 + int(seconds)
            entries.append((t, text))

    entries.sort(key=lambda x: x[0])
    merged_text = " ".join([t[1].strip() for t in entries])
    out_path = txt_path.with_name(txt_path.stem + "_merged.txt")
    out_path.write_text(merged_text, encoding="utf-8")
    return out_path


# === LOCAL CLEANUP ===
def clean_with_local_mistral(file_path):
    """Fix grammar/punctuation locally using a quantized Mistral GGUF model."""
    if not LOCAL_MISTRAL_PATH.exists():
        print(f"❌ Missing local model: {LOCAL_MISTRAL_PATH}")
        return None

    text = Path(file_path).read_text(encoding="utf-8")

    prompt = (
        "You are a transcript cleaner. "
        "Fix only grammar, punctuation, and spelling errors in the following transcript. "
        "Do not change phrasing or word order. Preserve meaning exactly. "
        "Return only the corrected text:\n\n"
        + text
    )

    print("🧠 Running local Mistral cleanup (GGUF)...")

    llm = Llama(
        model_path=str(LOCAL_MISTRAL_PATH),
        n_ctx=4096,
        n_threads=6,          # adjust for your CPU cores
        n_gpu_layers=-1       # auto GPU offload if available (Metal/CUDA)
    )

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2048,
    )

    output = response["choices"][0]["message"]["content"].strip()
    out_path = Path(file_path).with_name(Path(file_path).stem + "_clean.txt")
    out_path.write_text(output, encoding="utf-8")

    print(f"✅ Cleaned transcript saved to: {out_path}")
    return out_path


# === MAIN PIPELINE ===
def process_transcript(txt_path):
    """
    Detects whether the transcript includes timestamps like [00:12].
    If timestamps exist -> merge + clean.
    If no timestamps -> clean directly.
    """
    txt_path = Path(txt_path)
    raw_text = txt_path.read_text(encoding="utf-8")
    # Normalize transcript path
    if txt_path.suffix != ".txt":
        txt_path = txt_path.with_suffix(".txt")

    # Fallback: if file not found but "_transcript.txt" exists
    if not txt_path.exists():
        alt = txt_path.with_name(txt_path.stem + "_transcript.txt")
        if alt.exists():
            txt_path = alt

    has_timestamps = bool(re.search(r"\[\d{2}:\d{2}\]", raw_text))
    print(f"🔍 Checking transcript: {'timestamps found' if has_timestamps else 'no timestamps'}")

    if not raw_text.strip():
        print("⚠️ Transcript is empty — skipping cleanup.")
        return None

    target_path = merge_transcript(txt_path) if has_timestamps else txt_path
    cleaned = clean_with_local_mistral(target_path)

    if cleaned and Path(cleaned).exists():
        orig_wc = len(raw_text.split())
        new_wc = len(Path(cleaned).read_text(encoding='utf-8').split())
        print(f"✨ Cleaned transcript: {cleaned} ({orig_wc} → {new_wc} words)")
    else:
        print("⚠️ Cleanup failed or produced no output.")

    return cleaned

# sumarize the transcript generate metadata
def summarize_transcript(file_path, model_path=None):
    """
    Generate a short summary and title from a transcript using local Mistral.
    Returns metadata dict and saves it as JSON beside the transcript.
    """

    txt_path = Path(file_path)
    text = txt_path.read_text(encoding="utf-8").strip()

    if not text:
        print("⚠️ Transcript empty, skipping summary.")
        return None

    # === Prompt ===
    prompt = (
        "You are an assistant creating metadata for a recording transcript.\n"
        "1. Write a short, clear title (5–10 words max).\n"
        "2. Write a one-sentence summary of the recording.\n"
        "Respond in strict JSON format with keys: title, summary.\n\n"
        f"Transcript:\n{text[:3000]}"  # Cap input length
    )

    print("🧠 Generating session summary...")
    meta_partial = {}

    try:
        llm = Llama(
            model_path=model_path or r"C:\Users\kwasi\OneDrive\Documents\Personal Projects\whisper\models\mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=-1,
        )
        result = llm(prompt, max_tokens=128, temperature=0.4)
        raw_text = result["choices"][0]["text"].strip()
        print(f"🧠 Raw LLM output:\n{raw_text}\n")

        # === Try to parse JSON ===
        try:
            meta_partial = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try extracting JSON block if text contains extra info
            match = re.search(r'\{[\s\S]*\}', raw_text)
            if match:
                try:
                    meta_partial = json.loads(match.group(0))
                    print("✅ Extracted valid JSON block from response.")
                except Exception as e:
                    print(f"⚠️ JSON extraction failed: {e}")
                    meta_partial = {
                        "title": raw_text.split("\n")[0][:60],
                        "summary": raw_text[:200],
                    }
            else:
                print("⚠️ No JSON found in response; using fallback.")
                meta_partial = {
                    "title": raw_text.split("\n")[0][:60],
                    "summary": raw_text[:200],
                }

    except Exception as e:
        print(f"⚠️ Summary generation failed: {e}")
        meta_partial = {"title": "Untitled Session", "summary": "Summary unavailable."}

    # === Compute duration (if audio exists) ===
    audio_path = txt_path.with_suffix(".wav")
    duration = None
    if audio_path.exists():
        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = f"{frames / rate / 60:.1f} min"
        except Exception as e:
            print(f"⚠️ Could not read duration: {e}")

    # === Build final metadata ===
    meta = {
        "title": meta_partial.get("title", "Untitled Session").strip(),
        "summary": meta_partial.get("summary", "Summary unavailable.").strip(),
        "duration": duration,
        "timestamp": datetime.now().isoformat(),
        "cleaned": True,
        "source": str(txt_path.name),
    }

    # === Save metadata JSON ===
    meta_path = txt_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"🗂️ Metadata saved to: {meta_path}")

    return meta
# === ENTRYPOINT ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python post_process_transcript.py recordings/session_xxx.txt")
    else:
        process_transcript(sys.argv[1])
