from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from audio_alpha import extract_features as feature_module
from audio_alpha.extract_features import extract_all_features, extract_audio_features


def _write_tone(path: Path, sample_rate: int = 16_000, duration_sec: float = 2.0) -> None:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 220 * t)
    sf.write(path, y, sample_rate)


def test_extract_audio_features_returns_core_numeric_features(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)

    features = extract_audio_features(audio_path, include_egemaps=False)

    for column in [
        "duration_sec",
        "energy_mean",
        "energy_std",
        "zcr_mean",
        "spectral_centroid_mean",
        "pitch_mean",
        "voiced_ratio",
        "mfcc_1_mean",
    ]:
        assert column in features
        assert np.isfinite(features[column])


def test_extract_all_features_writes_call_and_segment_csvs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(feature_module, "_extract_egemaps_features", lambda _: {})

    call_audio = tmp_path / "call.wav"
    segment_audio = tmp_path / "segment.wav"
    _write_tone(call_audio)
    _write_tone(segment_audio)

    manifest_path = tmp_path / "audio_manifest.csv"
    segment_manifest_path = tmp_path / "audio_segment_manifest.csv"
    call_output_path = tmp_path / "audio_call_features.csv"
    segment_output_path = tmp_path / "audio_segment_features.csv"

    pd.DataFrame(
        [
            {
                "call_id": "AAPL_2025_05_01",
                "ticker": "AAPL",
                "call_date": "2025-05-01",
                "normalized_audio_path": str(call_audio),
            }
        ]
    ).to_csv(manifest_path, index=False)
    pd.DataFrame(
        [
            {
                "call_id": "AAPL_2025_05_01",
                "ticker": "AAPL",
                "call_date": "2025-05-01",
                "segment_id": 0,
                "start_sec": 0.0,
                "end_sec": 2.0,
                "segment_audio_path": str(segment_audio),
            }
        ]
    ).to_csv(segment_manifest_path, index=False)

    call_df, segment_df = extract_all_features(
        manifest_path=manifest_path,
        segment_manifest_path=segment_manifest_path,
        call_output_path=call_output_path,
        segment_output_path=segment_output_path,
    )

    assert call_output_path.exists()
    assert segment_output_path.exists()
    assert len(call_df) == 1
    assert len(segment_df) == 1
    assert segment_df.loc[0, "segment_id"] == 0
