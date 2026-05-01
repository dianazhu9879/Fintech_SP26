from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import soundfile as sf

from audio_alpha.config import MANIFEST_PATH, NORMALIZED_AUDIO_DIR, TARGET_SAMPLE_RATE


def normalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(
            f"Could not find '{ffmpeg_bin}'. Install ffmpeg before preprocessing audio."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-vn",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def _audio_info(audio_path: str | Path) -> tuple[float, int, int]:
    info = sf.info(str(audio_path))
    return float(info.duration), int(info.samplerate), int(info.channels)


def preprocess_all(
    manifest_path: Path = MANIFEST_PATH,
    output_dir: Path = NORMALIZED_AUDIO_DIR,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest at {manifest_path}. Run build_manifest first."
        )

    df = pd.read_csv(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_paths: list[str] = []
    durations: list[float] = []
    sample_rates: list[int] = []
    channels: list[int] = []

    for _, row in df.iterrows():
        output_path = output_dir / f"{row['call_id']}.wav"
        normalize_audio(row["audio_path"], output_path)
        duration_sec, sample_rate, channel_count = _audio_info(output_path)

        normalized_paths.append(str(output_path))
        durations.append(duration_sec)
        sample_rates.append(sample_rate)
        channels.append(channel_count)

    df["normalized_audio_path"] = normalized_paths
    df["duration_sec"] = durations
    df["sample_rate"] = sample_rates
    df["channels"] = channels
    df.to_csv(manifest_path, index=False)
    print("Updated manifest with normalized audio paths.")
    return df


if __name__ == "__main__":
    preprocess_all()
