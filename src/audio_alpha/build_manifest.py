from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from audio_alpha.config import MANIFEST_PATH, RAW_AUDIO_DIR

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
MANIFEST_COLUMNS = [
    "call_id",
    "ticker",
    "audio_path",
    "source_filename",
    "source_stem",
    "call_date",
    "file_ext",
]


def parse_call_date(filename: str) -> str | None:
    match = DATE_RE.search(filename)
    return match.group(1) if match else None


def _safe_id_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value.upper() or "UNKNOWN"


def _base_call_id(ticker: str, audio_file: Path, call_date: str | None) -> str:
    if call_date:
        return f"{ticker}_{call_date.replace('-', '_')}"
    return f"{ticker}_{_safe_id_part(audio_file.stem)}"


def build_manifest(
    raw_audio_dir: Path = RAW_AUDIO_DIR,
    output_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    rows: list[dict[str, str | None]] = []
    raw_audio_dir = Path(raw_audio_dir)

    if raw_audio_dir.exists():
        ticker_dirs = sorted(path for path in raw_audio_dir.iterdir() if path.is_dir())
    else:
        ticker_dirs = []

    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name.upper()

        for audio_file in sorted(ticker_dir.glob("*.mp3")):
            call_date = parse_call_date(audio_file.name)
            rows.append(
                {
                    "base_call_id": _base_call_id(ticker, audio_file, call_date),
                    "ticker": ticker,
                    "audio_path": str(audio_file),
                    "source_filename": audio_file.name,
                    "source_stem": audio_file.stem,
                    "call_date": call_date,
                    "file_ext": audio_file.suffix.lower(),
                }
            )

    rows = sorted(
        rows,
        key=lambda row: (
            str(row["ticker"]),
            str(row["call_date"] or "9999-99-99"),
            str(row["source_filename"]),
        ),
    )

    seen: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        base_call_id = str(row.pop("base_call_id"))
        seen[base_call_id] += 1
        suffix = "" if seen[base_call_id] == 1 else f"_{seen[base_call_id]:02d}"
        row["call_id"] = f"{base_call_id}{suffix}"

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    build_manifest()
