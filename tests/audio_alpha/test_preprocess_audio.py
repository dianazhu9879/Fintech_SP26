from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from audio_alpha.preprocess_audio import normalize_audio, preprocess_all


def _write_tone(path: Path, sample_rate: int = 8_000, duration_sec: float = 1.0) -> None:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, y, sample_rate)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_normalize_audio_writes_16khz_mono_wav(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "normalized.wav"
    _write_tone(input_path)

    normalize_audio(input_path, output_path)

    info = sf.info(output_path)
    assert output_path.exists()
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.duration > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_preprocess_all_updates_manifest(tmp_path: Path) -> None:
    raw_audio = tmp_path / "tone.wav"
    _write_tone(raw_audio)

    manifest_path = tmp_path / "audio_manifest.csv"
    pd.DataFrame(
        [
            {
                "call_id": "AAPL_2025_05_01",
                "ticker": "AAPL",
                "audio_path": str(raw_audio),
                "source_filename": raw_audio.name,
                "source_stem": raw_audio.stem,
                "call_date": "2025-05-01",
                "file_ext": ".wav",
            }
        ]
    ).to_csv(manifest_path, index=False)

    output_dir = tmp_path / "normalized"
    df = preprocess_all(manifest_path=manifest_path, output_dir=output_dir)

    normalized_path = Path(df.loc[0, "normalized_audio_path"])
    assert normalized_path.exists()
    assert df.loc[0, "sample_rate"] == 16_000
    assert df.loc[0, "channels"] == 1
    assert df.loc[0, "duration_sec"] > 0
