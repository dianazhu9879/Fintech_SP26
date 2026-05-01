from __future__ import annotations

import re
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from audio_alpha.config import (
    CALL_FEATURES_PATH,
    MANIFEST_PATH,
    SEGMENT_FEATURES_PATH,
    SEGMENT_MANIFEST_PATH,
    TARGET_SAMPLE_RATE,
)


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanstd(values)) if np.any(~np.isnan(values)) else np.nan


def _clean_feature_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return f"egemaps_{cleaned}"


def _extract_egemaps_features(audio_path: str | Path) -> dict[str, float]:
    try:
        import opensmile
    except ImportError:
        return {}

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    frame = smile.process_file(str(audio_path))
    if frame.empty:
        return {}

    row = frame.iloc[0]
    features: dict[str, float] = {}
    for name, value in row.items():
        features[_clean_feature_name(str(name))] = float(value)
    return features


def _extract_librosa_features(audio_path: str | Path) -> dict[str, float]:
    y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=4096,
    )
    rms_db = librosa.amplitude_to_db(rms, ref=np.max) if np.any(rms > 0) else rms

    features = {
        "duration_sec": float(duration),
        "energy_mean": safe_mean(rms),
        "energy_std": safe_std(rms),
        "zcr_mean": safe_mean(zcr),
        "zcr_std": safe_std(zcr),
        "spectral_centroid_mean": safe_mean(centroid),
        "spectral_centroid_std": safe_std(centroid),
        "spectral_bandwidth_mean": safe_mean(bandwidth),
        "spectral_bandwidth_std": safe_std(bandwidth),
        "pitch_mean": safe_mean(f0),
        "pitch_std": safe_std(f0),
        "voiced_ratio": float(np.mean(rms_db > -35.0)) if len(rms_db) else np.nan,
    }

    for i in range(13):
        features[f"mfcc_{i + 1}_mean"] = safe_mean(mfcc[i])
        features[f"mfcc_{i + 1}_std"] = safe_std(mfcc[i])

    return features


def should_include_egemaps() -> bool:
    return os.getenv("AUDIO_ALPHA_EGEMAPS", "").lower() in {"1", "true", "yes"}


def extract_audio_features(
    audio_path: str | Path,
    include_egemaps: bool | None = None,
) -> dict[str, float]:
    if include_egemaps is None:
        include_egemaps = should_include_egemaps()

    features = _extract_librosa_features(audio_path)
    if include_egemaps:
        features.update(_extract_egemaps_features(audio_path))
    return features


def extract_call_features(row: pd.Series) -> dict[str, object]:
    features: dict[str, object] = {
        "call_id": row["call_id"],
        "ticker": row["ticker"],
        "call_date": row.get("call_date"),
    }
    features.update(extract_audio_features(row["normalized_audio_path"]))
    return features


def extract_segment_features(row: pd.Series) -> dict[str, object]:
    features: dict[str, object] = {
        "call_id": row["call_id"],
        "ticker": row["ticker"],
        "call_date": row.get("call_date"),
        "segment_id": row["segment_id"],
        "start_sec": row["start_sec"],
        "end_sec": row["end_sec"],
    }
    features.update(extract_audio_features(row["segment_audio_path"]))
    return features


def extract_all_features(
    manifest_path: Path = MANIFEST_PATH,
    segment_manifest_path: Path = SEGMENT_MANIFEST_PATH,
    call_output_path: Path = CALL_FEATURES_PATH,
    segment_output_path: Path = SEGMENT_FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest at {manifest_path}. Run preprocess_audio first."
        )
    if not segment_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing segment manifest at {segment_manifest_path}. Run segment_audio first."
        )

    call_df = pd.read_csv(manifest_path)
    segment_df = pd.read_csv(segment_manifest_path)

    call_rows = [
        extract_call_features(row)
        for _, row in tqdm(call_df.iterrows(), total=len(call_df), desc="call features")
    ]
    segment_rows = [
        extract_segment_features(row)
        for _, row in tqdm(
            segment_df.iterrows(), total=len(segment_df), desc="segment features"
        )
    ]

    call_out = pd.DataFrame(call_rows)
    segment_out = pd.DataFrame(segment_rows)

    call_output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_output_path.parent.mkdir(parents=True, exist_ok=True)
    call_out.to_csv(call_output_path, index=False)
    segment_out.to_csv(segment_output_path, index=False)

    print(f"Wrote call features to {call_output_path}")
    print(f"Wrote segment features to {segment_output_path}")
    return call_out, segment_out


if __name__ == "__main__":
    extract_all_features()
