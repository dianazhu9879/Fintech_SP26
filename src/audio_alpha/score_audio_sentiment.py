from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from audio_alpha.config import (
    CALL_FEATURES_PATH,
    CALL_SCORES_PATH,
    SEGMENT_FEATURES_PATH,
    SEGMENT_SCORES_PATH,
)

STRESS_FEATURES = [
    "pitch_std",
    "energy_std",
    "zcr_std",
    "spectral_centroid_std",
]
CONFIDENCE_FEATURES = [
    "energy_mean",
    "voiced_ratio",
]
CLARITY_FEATURES = [
    "voiced_ratio",
    "zcr_std",
    "spectral_centroid_std",
]
SCORE_COLUMNS = [
    "audio_stress_index",
    "audio_confidence_index",
    "audio_instability_index",
    "vocal_clarity_proxy",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def zscore(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    _require_columns(df, columns)
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    medians = values.median(numeric_only=True).fillna(0.0)
    values = values.fillna(medians).fillna(0.0)

    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(values),
        columns=[f"{column}_z" for column in columns],
        index=df.index,
    )


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        for column in SCORE_COLUMNS:
            df[column] = []
        return df

    stress_z = zscore(df, STRESS_FEATURES)
    confidence_z = zscore(df, CONFIDENCE_FEATURES)

    scored = pd.concat([df.copy(), stress_z, confidence_z], axis=1)
    scored["audio_stress_index"] = (
        scored["pitch_std_z"]
        + scored["energy_std_z"]
        + scored["zcr_std_z"]
        + scored["spectral_centroid_std_z"]
    )
    scored["audio_confidence_index"] = (
        scored["energy_mean_z"] + scored["voiced_ratio_z"] - scored["audio_stress_index"]
    )
    scored["audio_instability_index"] = (
        scored["pitch_std_z"] + scored["energy_std_z"] + scored["spectral_centroid_std_z"]
    )
    scored["vocal_clarity_proxy"] = (
        scored["voiced_ratio_z"] - scored["zcr_std_z"] - scored["spectral_centroid_std_z"]
    )
    return scored


def score_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing features file at {input_path}.")

    df = pd.read_csv(input_path)
    scored = score_dataframe(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Wrote audio scores to {output_path}")
    return scored


def score_audio(
    call_input_path: Path = CALL_FEATURES_PATH,
    call_output_path: Path = CALL_SCORES_PATH,
    segment_input_path: Path = SEGMENT_FEATURES_PATH,
    segment_output_path: Path = SEGMENT_SCORES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    call_scores = score_file(call_input_path, call_output_path)
    segment_scores = score_file(segment_input_path, segment_output_path)
    return call_scores, segment_scores


if __name__ == "__main__":
    score_audio()
