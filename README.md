# Fintech_SP26

## Audio Alpha Pipeline

This module turns raw earnings-call MP3 files into structured audio feature
tables that can be joined with transcript, fundamentals, and market data.

The goal is not to classify whether a speaker is happy or sad, and this branch
does not predict stock returns. Earnings calls are scripted and noisy, so the
pipeline extracts measurable vocal delivery signals such as pitch instability,
energy variation, voiced speech ratio, and spectral variation. These are
combined into exploratory audio stress, confidence, instability, and clarity
proxy indices.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
brew install ffmpeg
```

Optionally install `opensmile` for standardized eGeMAPS acoustic features:

```bash
pip install ".[standard]"
```

`opensmile` is free for research and educational use, but commercial product use
requires separate licensing from audEERING. If it is unavailable, the pipeline
still produces the core transparent `librosa` features. eGeMAPS extraction is
off by default because segment-level extraction can be slow on many hours of
audio. Enable it explicitly with:

```bash
AUDIO_ALPHA_EGEMAPS=1 python -m audio_alpha.extract_features
```

## Data Layout

Google Drive is the source folder for original MP3s. Before running the
pipeline, mirror or copy the files locally into:

```text
data/raw/audio/AAPL/*.mp3
data/raw/audio/MSFT/*.mp3
data/raw/audio/NVDA/*.mp3
data/raw/audio/AMZN/*.mp3
data/raw/audio/GOOG/*.mp3
data/raw/audio/META/*.mp3
data/raw/audio/TSLA/*.mp3
```

Raw audio and generated CSV/WAV outputs are ignored by git.

## Pipeline

Run the full MVP:

```bash
python -m audio_alpha.run_pipeline
```

Or run phases one at a time:

```bash
python -m audio_alpha.build_manifest
python -m audio_alpha.preprocess_audio
python -m audio_alpha.segment_audio
python -m audio_alpha.extract_features
python -m audio_alpha.score_audio_sentiment
python -m audio_alpha.aggregate_call_features
```

Core outputs:

- `data/interim/audio_manifest.csv`
- `data/interim/audio_segment_manifest.csv`
- `data/processed/audio_call_features.csv`
- `data/processed/audio_segment_features.csv`
- `data/processed/audio_call_scores.csv`
- `data/processed/audio_segment_scores.csv`
- `data/processed/audio_call_feature_table.csv`

## Testing

Tests use synthetic audio generated in temporary directories, not real
earnings-call MP3s.

```bash
python -m pytest
```

Phase-specific tester commands:

```bash
python -m pytest tests/audio_alpha/test_build_manifest.py
python -m pytest tests/audio_alpha/test_preprocess_audio.py
python -m pytest tests/audio_alpha/test_segment_audio.py
python -m pytest tests/audio_alpha/test_extract_features.py
python -m pytest tests/audio_alpha/test_score_audio_sentiment.py
python -m pytest tests/audio_alpha/test_aggregate_call_features.py
```

After each phase, a tester should confirm row counts, required columns, stable
`call_id` keys, valid audio paths, and finite numeric feature values before the
next phase runs.
