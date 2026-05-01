from __future__ import annotations

from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from audio_alpha.config import (
    HOP_SEC,
    MANIFEST_PATH,
    MIN_SEGMENT_SEC,
    SEGMENT_DIR,
    SEGMENT_MANIFEST_PATH,
    TARGET_SAMPLE_RATE,
    WINDOW_SEC,
)

SEGMENT_COLUMNS = [
    "call_id",
    "ticker",
    "call_date",
    "segment_id",
    "start_sec",
    "end_sec",
    "segment_audio_path",
]


def segment_one_call(
    row: pd.Series,
    segment_dir: Path = SEGMENT_DIR,
    window_sec: int = WINDOW_SEC,
    hop_sec: int = HOP_SEC,
    min_segment_sec: int = MIN_SEGMENT_SEC,
) -> list[dict[str, object]]:
    if "normalized_audio_path" not in row or pd.isna(row["normalized_audio_path"]):
        raise ValueError("Manifest row is missing normalized_audio_path.")

    y, sr = librosa.load(row["normalized_audio_path"], sr=TARGET_SAMPLE_RATE, mono=True)
    window = int(window_sec * sr)
    hop = int(hop_sec * sr)
    min_samples = int(min_segment_sec * sr)

    call_dir = segment_dir / str(row["call_id"])
    call_dir.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, object]] = []
    segment_id = 0

    for start in range(0, len(y), hop):
        end = min(start + window, len(y))
        segment = y[start:end]

        if len(segment) < min_samples:
            continue

        segment_path = call_dir / f"segment_{segment_id:04d}.wav"
        sf.write(segment_path, segment, sr)

        segment_rows.append(
            {
                "call_id": row["call_id"],
                "ticker": row["ticker"],
                "call_date": row.get("call_date"),
                "segment_id": segment_id,
                "start_sec": start / sr,
                "end_sec": end / sr,
                "segment_audio_path": str(segment_path),
            }
        )
        segment_id += 1

    return segment_rows


def segment_all_calls(
    manifest_path: Path = MANIFEST_PATH,
    segment_dir: Path = SEGMENT_DIR,
    output_path: Path = SEGMENT_MANIFEST_PATH,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest at {manifest_path}. Run preprocess_audio first."
        )

    df = pd.read_csv(manifest_path)
    all_rows: list[dict[str, object]] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        all_rows.extend(segment_one_call(row, segment_dir=segment_dir))

    out = pd.DataFrame(all_rows, columns=SEGMENT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote segment manifest to {output_path}")
    return out


if __name__ == "__main__":
    segment_all_calls()
