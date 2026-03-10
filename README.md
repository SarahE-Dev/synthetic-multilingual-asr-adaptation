# Synthetic Multilingual ASR Adaptation

A research project exploring the use of synthetic speech generated through Text-to-Speech (TTS) systems for adapting and evaluating multilingual Automatic Speech Recognition (ASR) models.

## Overview

This repository investigates how synthetic multilingual speech can be leveraged to improve ASR systems across different languages, particularly for low-resource scenarios. By generating synthetic training data through TTS models, we aim to reduce the cost and time associated with collecting authentic transcribed audio while maintaining model performance.

## Key Features

- **Synthetic Speech Generation**: Generate multilingual synthetic speech using XTTS v2 / YourTTS
- **ASR Evaluation Pipeline**: Evaluate ASR model performance on synthetic speech with WER/CER metrics
- **Multilingual Support**: English, Spanish, and French with extensibility to other languages
- **Closed-loop Analysis**: Study the TTS → ASR pipeline to understand synthetic speech characteristics

## Dataset

The synthetic multilingual speech dataset used in this project is available on Hugging Face:

  --> **[nprak26/synthetic-multilingual-speech-asr](https://huggingface.co/datasets/nprak26/synthetic-multilingual-speech-asr)**

### Dataset Specifications

- **Languages**: English (en), Spanish (es), French (fr)
- **Audio Format**: 16 kHz mono WAV files
- **Size**: Initial dataset with 15 examples (expandable)
- **License**: CC BY-NC 4.0

### Dataset Structure

Each sample contains:
- `wav`: Synthetic speech audio file
- `lang`: ISO language code
- `ref_text`: Reference text used for TTS generation
- `tts_model_used`: Identifier of the TTS model
- `hyp_text`: ASR system hypothesis (decoded transcript)

## Requirements

- **Python 3.11** (the Coqui TTS package requires `<3.12`)
- macOS, Linux, or Windows

## Installation

```bash
# Clone the repository
git clone https://github.com/nishanthp/synthetic-multilingual-asr-adaptation.git
cd synthetic-multilingual-asr-adaptation

# Create a virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### macOS (Apple Silicon) note

If you don't have Python 3.11, install it with Homebrew:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the full pipeline

```bash
python main.py
```

This will:
1. Load a multilingual TTS model (XTTS v2, with YourTTS as fallback)
2. Generate synthetic speech for EN, ES, and FR using built-in sample sentences
3. Transcribe the generated audio with Whisper (faster-whisper or openai-whisper)
4. Compute WER and CER for each language and print a summary

### CLI options

```bash
python main.py                # TTS generation + ASR evaluation (default)
python main.py --build        # Download dataset subsets first, then run pipeline
python main.py --tts-only     # Only generate TTS audio
python main.py --asr-only     # Only run ASR validation on existing TTS output
```

### Build real dataset subsets

By default the pipeline uses a small set of hardcoded sample sentences. To download real audio subsets from LibriSpeech (EN) and Multilingual LibriSpeech (ES, FR):

```bash
python main.py --build
```

You can also run the dataset downloaders individually:

```bash
python datasets/english_dataset_downloader.py
python datasets/spanish_dataset_downloader.py
python datasets/french_dataset_downloader.py
```

### Output structure

After running, you'll find:

```
outputs/
  tts_outputs/
    en/
      en_0.wav ... en_4.wav
      manifest.jsonl
    es/
      es_0.wav ... es_4.wav
      manifest.jsonl
    fr/
      fr_0.wav ... fr_4.wav
      manifest.jsonl
  _xtts_refs/
    en_ref.wav, es_ref.wav, fr_ref.wav
```

Each `manifest.jsonl` contains one JSON object per line with `wav`, `lang`, `ref_text`, `tts_model_used`, and (after ASR) `hyp_text`, `WER`, and `CER`.

## Project Structure

```
├── main.py                            # Entry point — orchestrates the full pipeline
├── requirements.txt                   # Python dependencies
├── datasets/
│   ├── english_dataset_downloader.py  # LibriSpeech EN subset builder
│   ├── spanish_dataset_downloader.py  # MLS Spanish subset builder
│   └── french_dataset_downloader.py   # MLS French subset builder
├── scripts/
│   ├── multi_lingual_pipeline_training.py  # TTS generation + ASR evaluation
│   └── asr_whisper_validation.py           # Standalone ASR validation script
└── .github/
    └── workflows/
        ├── build.yml                  # CI: install, compile, test
        └── format.yml                 # CI: ruff lint & format
```

## Research Applications

This project is designed for:

1. **Low-Resource ASR Development**: Bootstrap ASR systems for languages with limited transcribed audio
2. **Data Augmentation**: Supplement existing ASR training data with diverse synthetic examples
3. **Domain Adaptation**: Adapt pre-trained ASR models to specific accents or speaking styles
4. **Robustness Testing**: Evaluate ASR model generalization to synthetic speech patterns
5. **Cost-Effective Prototyping**: Rapidly prototype multilingual ASR systems without expensive data collection

## Methodology

### TTS Generation
- Utilize multilingual TTS models (XTTS v2, YourTTS) via Coqui TTS
- Generate diverse synthetic samples across multiple languages
- Bootstrap speaker references from single-language models when no real audio is available

### ASR Evaluation
- Transcribe with Whisper (faster-whisper or openai-whisper, auto-selected)
- Compute Word Error Rate (WER) and Character Error Rate (CER)
- Per-language metrics saved to `manifest.jsonl` and `metrics.json`

## Limitations

- **Scale**: Current dataset contains 15 samples; larger scale needed for production use
- **Diversity**: Synthetic speech lacks natural speaker variability, accents, and environmental noise
- **Domain**: Limited to clean, read speech; may not generalize to spontaneous conversation
- **Languages**: Currently supports only English, Spanish, and French

## Future Work

- [ ] Expand dataset to 1M+ synthetic utterances per language
- [ ] Add support for more languages (Mandarin, Hindi, Marathi, Kannada)
- [ ] Incorporate accent and dialect variation in TTS generation
- [ ] Experiment with mixed synthetic-authentic training strategies
- [ ] Develop speaker-adaptive synthetic generation techniques

## Citation

If you use this dataset or codebase in your research, please cite:

```bibtex
@misc{synthetic-multilingual-asr-2024,
  author = {Nishanth Prakash},
  title = {Synthetic Multilingual ASR Adaptation},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/nishanthp/synthetic-multilingual-asr-adaptation}},
  note = {Dataset: \url{https://huggingface.co/datasets/nprak26/synthetic-multilingual-speech-asr}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The dataset is available under the CC BY-NC 4.0 license.
