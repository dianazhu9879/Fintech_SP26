import numpy as np
import pandas as pd

from audio_alpha.score_audio_sentiment import SCORE_COLUMNS, score_dataframe


def test_score_dataframe_adds_indices_and_preserves_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "call_id": "AAPL_2025_05_01",
                "ticker": "AAPL",
                "pitch_std": 1.0,
                "energy_std": 1.0,
                "zcr_std": 1.0,
                "spectral_centroid_std": 1.0,
                "energy_mean": 2.0,
                "voiced_ratio": 0.9,
            },
            {
                "call_id": "NVDA_2025_05_01",
                "ticker": "NVDA",
                "pitch_std": 3.0,
                "energy_std": 3.0,
                "zcr_std": 3.0,
                "spectral_centroid_std": 3.0,
                "energy_mean": 1.0,
                "voiced_ratio": 0.4,
            },
        ]
    )

    scored = score_dataframe(df)

    assert len(scored) == len(df)
    assert all(column in scored.columns for column in SCORE_COLUMNS)
    assert np.isfinite(scored[SCORE_COLUMNS].to_numpy()).all()
    assert scored.loc[1, "audio_stress_index"] > scored.loc[0, "audio_stress_index"]
    assert scored.loc[0, "audio_confidence_index"] > scored.loc[1, "audio_confidence_index"]


def test_score_dataframe_imputes_missing_values() -> None:
    df = pd.DataFrame(
        [
            {
                "call_id": "AAPL_2025_05_01",
                "pitch_std": np.nan,
                "energy_std": 1.0,
                "zcr_std": 1.0,
                "spectral_centroid_std": 1.0,
                "energy_mean": 2.0,
                "voiced_ratio": 0.9,
            },
            {
                "call_id": "NVDA_2025_05_01",
                "pitch_std": 3.0,
                "energy_std": 3.0,
                "zcr_std": 3.0,
                "spectral_centroid_std": 3.0,
                "energy_mean": np.nan,
                "voiced_ratio": 0.4,
            },
        ]
    )

    scored = score_dataframe(df)

    assert np.isfinite(scored[SCORE_COLUMNS].to_numpy()).all()
    assert scored["call_id"].tolist() == ["AAPL_2025_05_01", "NVDA_2025_05_01"]
