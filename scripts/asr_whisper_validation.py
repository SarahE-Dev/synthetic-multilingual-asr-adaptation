# === SCRIPT 2: ASR VALIDATION (faster-whisper or openai-whisper) ===

import os, json, sys

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "tts_outputs")
LANGS = ["en", "es", "fr"]

# Try faster-whisper first; fall back to openai-whisper (more reliable on macOS Apple Silicon)
ASR_BACKEND = None
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FasterWhisperModel("tiny", device="cpu", compute_type="int8")
    ASR_BACKEND = "faster-whisper"
except Exception:
    pass

if ASR_BACKEND is None:
    try:
        import whisper as openai_whisper
        ASR_BACKEND = "openai-whisper"
    except ImportError:
        print("No ASR backend found. Install one of:\n"
              "  pip install faster-whisper\n"
              "  pip install openai-whisper")
        sys.exit(1)


def wer(ref: str, hyp: str) -> float:
    r = ref.strip().lower().split()
    h = hyp.strip().lower().split()
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[n] / max(1, m)


def cer(ref: str, hyp: str) -> float:
    r = list(ref.strip().lower())
    h = list(hyp.strip().lower())
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[n] / max(1, m)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def save_manifest(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    found_any = False
    for lang in LANGS:
        manifest_path = os.path.join(BASE, lang, "manifest.jsonl")
        if os.path.isfile(manifest_path):
            found_any = True
            break

    if not found_any:
        print("No manifest files found. Run the TTS pipeline first "
              "(scripts/multi_lingual_pipeline_training.py).")
        sys.exit(1)

    print(f"=== Loading ASR model ({ASR_BACKEND}, base, CPU) ===")
    if ASR_BACKEND == "faster-whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
    else:
        import whisper as openai_whisper
        model = openai_whisper.load_model("base", device="cpu")

    def transcribe(wav_path, lang):
        if ASR_BACKEND == "faster-whisper":
            segs, _ = model.transcribe(wav_path, beam_size=1, language=lang)
            return " ".join(s.text.strip() for s in segs).strip()
        else:
            result = model.transcribe(wav_path, language=lang, fp16=False)
            return (result.get("text") or "").strip()

    metrics = {}
    for lang in LANGS:
        manifest_path = os.path.join(BASE, lang, "manifest.jsonl")
        metrics_path = os.path.join(BASE, lang, "metrics.json")

        if not os.path.isfile(manifest_path):
            print(f"\n[{lang.upper()}] No manifest found, skipping.")
            continue

        rows = load_manifest(manifest_path)
        if not rows:
            print(f"\n[{lang.upper()}] Manifest is empty, skipping.")
            continue

        wers, cers = [], []
        print(f"\n[{lang.upper()}] items={len(rows)}")
        for r in rows:
            wav, ref = r["wav"], r["ref_text"]
            if not os.path.isfile(wav):
                print(f"  skipping missing file: {wav}")
                continue
            try:
                hyp = transcribe(wav, lang)
                r["hyp_text"] = hyp
                r["WER"] = wer(ref, hyp)
                r["CER"] = cer(ref, hyp)
                wers.append(r["WER"])
                cers.append(r["CER"])
                print(f"  {os.path.basename(wav)}  WER={r['WER']:.3f}  CER={r['CER']:.3f}  | \"{hyp[:60]}\"")
            except Exception as e:
                r["hyp_text"] = None
                r["WER"] = None
                r["CER"] = None
                print(f"  ASR failed for {wav}: {e}")

        save_manifest(rows, manifest_path)

        lang_metrics = {
            "clips": len([w for w in wers if w is not None]),
            "avg_WER": (sum(wers) / len(wers)) if wers else None,
            "avg_CER": (sum(cers) / len(cers)) if cers else None,
        }
        metrics[lang] = lang_metrics

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(lang_metrics, f, indent=2, ensure_ascii=False)

    print("\n================ METRICS ================")
    for lang, v in metrics.items():
        avg_wer = None if v["avg_WER"] is None else round(v["avg_WER"], 3)
        avg_cer = None if v["avg_CER"] is None else round(v["avg_CER"], 3)
        print(f"{lang.upper()}: clips={v['clips']} | WER={avg_wer} | CER={avg_cer}")
    print(f"\nOutputs: {BASE}")


if __name__ == "__main__":
    main()
