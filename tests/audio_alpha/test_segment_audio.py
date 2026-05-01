from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from audio_alpha.segment_audio import segment_all_calls


def _write_tone(path: Path, sample_rate: int = 16_000, duration_sec: float = 65.0) -> None:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 220 * t)
    sf.write(path, y, sample_rate)


def test_segment_all_calls_writes_fixed_window_manifest(tmp_path: Path) -> None:
    audio_path = tmp_path / "call.wav"
    _write_tone(audio_path)

    manifest_path = tmp_path / "audio_manifest.csv"
    pd.DataFrame(
        [
            {
                "call_id": "AAPL_2025_05_01",
                "ticker": "AAPL",
                "call_date": "2025-05-01",
                "normalized_audio_path": str(audio_path),
            }
        ]
    ).to_csv(manifest_path, index=False)

    segment_dir = tmp_path / "segments"
    output_path = tmp_path / "audio_segment_manifest.csv"
    df = segment_all_calls(
        manifest_path=manifest_path,
        segment_dir=segment_dir,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(df) == 5
    assert df["segment_id"].tolist() == [0, 1, 2, 3, 4]
    assert df["start_sec"].is_monotonic_increasing
    assert (df["end_sec"] > df["start_sec"]).all()
    assert all(Path(path).exists() for path in df["segment_audio_path"])
