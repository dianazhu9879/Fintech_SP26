from pathlib import Path

import pandas as pd

from audio_alpha.aggregate_call_features import aggregate_call_features


def test_aggregate_call_features_outputs_one_row_per_call(tmp_path: Path) -> None:
    call_scores_path = tmp_path / "audio_call_scores.csv"
    segment_scores_path = tmp_path / "audio_segment_scores.csv"
    output_path = tmp_path / "audio_call_feature_table.csv"

    pd.DataFrame(
        [
            {
                "call_id": "AAPL_2025_05_01",
                "ticker": "AAPL",
                "call_date": "2025-05-01",
                "audio_stress_index": 1.0,
                "audio_confidence_index": 2.0,
                "audio_instability_index": 3.0,
                "vocal_clarity_proxy": 4.0,
            },
            {
                "call_id": "NVDA_2025_05_01",
                "ticker": "NVDA",
                "call_date": "2025-05-01",
                "audio_stress_index": 5.0,
                "audio_confidence_index": 6.0,
                "audio_instability_index": 7.0,
                "vocal_clarity_proxy": 8.0,
            },
        ]
    ).to_csv(call_scores_path, index=False)

    pd.DataFrame(
        [
            {
                "call_id": "AAPL_2025_05_01",
                "segment_id": 0,
                "audio_stress_index": 1.0,
                "audio_confidence_index": 4.0,
                "audio_instability_index": 2.0,
                "vocal_clarity_proxy": 7.0,
            },
            {
                "call_id": "AAPL_2025_05_01",
                "segment_id": 1,
                "audio_stress_index": 3.0,
                "audio_confidence_index": 2.0,
                "audio_instability_index": 6.0,
                "vocal_clarity_proxy": 5.0,
            },
            {
                "call_id": "NVDA_2025_05_01",
                "segment_id": 0,
                "audio_stress_index": 5.0,
                "audio_confidence_index": 6.0,
                "audio_instability_index": 7.0,
                "vocal_clarity_proxy": 8.0,
            },
        ]
    ).to_csv(segment_scores_path, index=False)

    final = aggregate_call_features(
        call_scores_path=call_scores_path,
        segment_scores_path=segment_scores_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(final) == 2
    assert final["call_id"].tolist() == ["AAPL_2025_05_01", "NVDA_2025_05_01"]
    aapl = final[final["call_id"] == "AAPL_2025_05_01"].iloc[0]
    assert aapl["segment_audio_stress_index_mean"] == 2.0
    assert aapl["segment_audio_stress_index_min"] == 1.0
    assert aapl["segment_audio_stress_index_max"] == 3.0
    assert aapl["segment_audio_stress_index_delta"] == 2.0
