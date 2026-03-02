import os, json, sys, subprocess

BASE = "/kaggle/working/tts_outputs/en"
MANIFEST = os.path.join(BASE, "manifest.jsonl")
METRICS_OUT = os.path.join(BASE, "metrics.json")

if not os.path.isfile(MANIFEST):
    raise SystemExit("❌ Manifest not found. Run Script 1 first.")

# --- deps ---
def ensure_whisper_from_git():
    try:
        import whisper
        return
    except ImportError:
        print("Installing whisper from GitHub (no pip torch downgrade)...")
        subprocess.run([
            "pip", "install", "git+https://github.com/openai/whisper.git"
        ], check=True)

ensure_whisper_from_git()
