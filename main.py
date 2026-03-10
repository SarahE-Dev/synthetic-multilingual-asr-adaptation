"""
Synthetic Multilingual ASR Adaptation — entry point.

Runs the full pipeline:
  1. (Optional) Build small EN/ES/FR dataset subsets
  2. Generate TTS audio with XTTS v2 / YourTTS
  3. Evaluate ASR with faster-whisper and report WER/CER

Usage:
    python main.py                  # TTS + ASR (default)
    python main.py --build          # Build datasets first, then TTS + ASR
    python main.py --tts-only       # Only run TTS generation
    python main.py --asr-only       # Only run ASR validation (requires existing TTS output)
"""

import argparse
import subprocess
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def run_script(path, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"\nFailed: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def build_datasets():
    for name, script in [
        ("English (LibriSpeech)", "english_dataset_downloader.py"),
        ("Spanish (MLS)", "spanish_dataset_downloader.py"),
        ("French (MLS)", "french_dataset_downloader.py"),
    ]:
        run_script(os.path.join(DATASETS_DIR, script), f"Building {name} subset")


def run_tts():
    os.environ["RUN_TTS"] = "1"
    os.environ["RUN_ASR"] = "0"
    run_script(
        os.path.join(SCRIPTS_DIR, "multi_lingual_pipeline_training.py"),
        "TTS Generation (XTTS v2 / YourTTS)",
    )


def run_tts_and_asr():
    run_script(
        os.path.join(SCRIPTS_DIR, "multi_lingual_pipeline_training.py"),
        "TTS Generation + ASR Evaluation",
    )


def run_asr():
    run_script(
        os.path.join(SCRIPTS_DIR, "asr_whisper_validation.py"),
        "ASR Validation (faster-whisper)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Multilingual ASR Adaptation Pipeline"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Download and build small EN/ES/FR dataset subsets before running the pipeline",
    )
    parser.add_argument(
        "--tts-only", action="store_true",
        help="Only run TTS generation (skip ASR evaluation)",
    )
    parser.add_argument(
        "--asr-only", action="store_true",
        help="Only run ASR validation on existing TTS output",
    )
    args = parser.parse_args()

    if args.build:
        build_datasets()

    if args.asr_only:
        run_asr()
    elif args.tts_only:
        run_tts()
    else:
        run_tts_and_asr()

    print("\nDone.")


if __name__ == "__main__":
    main()
