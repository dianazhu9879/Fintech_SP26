from pathlib import Path

RAW_AUDIO_DIR = Path("data/raw/audio")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

MANIFEST_PATH = INTERIM_DIR / "audio_manifest.csv"
SEGMENT_MANIFEST_PATH = INTERIM_DIR / "audio_segment_manifest.csv"

NORMALIZED_AUDIO_DIR = INTERIM_DIR / "normalized_audio"
SEGMENT_DIR = INTERIM_DIR / "segments"

CALL_FEATURES_PATH = PROCESSED_DIR / "audio_call_features.csv"
SEGMENT_FEATURES_PATH = PROCESSED_DIR / "audio_segment_features.csv"
CALL_SCORES_PATH = PROCESSED_DIR / "audio_call_scores.csv"
SEGMENT_SCORES_PATH = PROCESSED_DIR / "audio_segment_scores.csv"
CALL_FEATURE_TABLE_PATH = PROCESSED_DIR / "audio_call_feature_table.csv"

TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA")

TARGET_SAMPLE_RATE = 16_000
WINDOW_SEC = 30
HOP_SEC = 15
MIN_SEGMENT_SEC = 5
