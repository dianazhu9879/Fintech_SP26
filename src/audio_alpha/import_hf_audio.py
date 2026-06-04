from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import quote

import requests

from audio_alpha.config import RAW_AUDIO_DIR

HF_DATASET_TREE_API = "https://huggingface.co/api/datasets/{dataset_id}/tree/{revision}"
HF_DATASET_FILE_URL = "https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{path}"
HF_AUDIO_NAME_RE = re.compile(
    r"^(?P<ticker>[A-Z]+)_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<day>\d{1,2})_earnings_call_qa\.mp3$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HFAudioFile:
    source_path: str
    ticker: str
    call_date: str
    target_filename: str


def _get_dataset_tree(dataset_id: str, revision: str) -> list[dict[str, object]]:
    url = HF_DATASET_TREE_API.format(dataset_id=dataset_id, revision=revision)
    response = requests.get(url, params={"recursive": "1"}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected dataset tree response type: {type(data)}")
    return data


def _parse_hf_audio_file(path: str) -> HFAudioFile | None:
    name = Path(path).name
    match = HF_AUDIO_NAME_RE.match(name)
    if not match:
        return None

    ticker = match.group("ticker").upper()
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    call_date = f"{year:04d}-{month:02d}-{day:02d}"
    target_filename = f"{ticker}_{call_date}_earnings_call_qa.mp3"
    return HFAudioFile(
        source_path=path,
        ticker=ticker,
        call_date=call_date,
        target_filename=target_filename,
    )


def _iter_hf_audio_files(dataset_tree: list[dict[str, object]]) -> tuple[list[HFAudioFile], list[str]]:
    parsed: list[HFAudioFile] = []
    skipped: list[str] = []

    for item in dataset_tree:
        path = str(item.get("path", ""))
        if not path.lower().endswith(".mp3"):
            continue
        audio_file = _parse_hf_audio_file(path)
        if audio_file is None:
            skipped.append(path)
            continue
        parsed.append(audio_file)

    parsed = sorted(parsed, key=lambda item: (item.ticker, item.call_date, item.source_path))
    return parsed, skipped


def _download_file(url: str, target_path: Path, chunk_size: int = 1024 * 1024) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)

    tmp_path.replace(target_path)


def import_hf_audio(
    dataset_id: str = "TQTfintech/40-2025-q1-q4",
    revision: str = "main",
    raw_audio_dir: Path = RAW_AUDIO_DIR,
    overwrite: bool = False,
    max_files: int | None = None,
) -> dict[str, int]:
    dataset_tree = _get_dataset_tree(dataset_id=dataset_id, revision=revision)
    audio_files, skipped_files = _iter_hf_audio_files(dataset_tree)
    if max_files is not None:
        audio_files = audio_files[:max_files]

    downloaded = 0
    skipped_existing = 0
    failed = 0

    for index, audio in enumerate(audio_files, start=1):
        target_path = Path(raw_audio_dir) / audio.ticker / audio.target_filename
        if target_path.exists() and not overwrite:
            skipped_existing += 1
            print(
                f"[{index}/{len(audio_files)}] exists, skipping: {target_path}"
            )
            continue

        encoded_source = quote(audio.source_path, safe="/")
        source_url = HF_DATASET_FILE_URL.format(
            dataset_id=dataset_id,
            revision=revision,
            path=encoded_source,
        )
        try:
            _download_file(source_url, target_path=target_path)
            downloaded += 1
            print(f"[{index}/{len(audio_files)}] downloaded: {target_path}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{index}/{len(audio_files)}] failed: {audio.source_path} ({exc})")

    summary = {
        "found_mp3_files": len(audio_files),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "skipped_unparseable": len(skipped_files),
    }

    print("HF import summary:", summary)
    if skipped_files:
        print("Unparseable mp3 names skipped:")
        for path in skipped_files[:20]:
            print(f"- {path}")
        if len(skipped_files) > 20:
            print(f"... and {len(skipped_files) - 20} more")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import Hugging Face earnings-call mp3s into local raw/audio directories."
    )
    parser.add_argument(
        "--dataset-id",
        default="TQTfintech/40-2025-q1-q4",
        help="Hugging Face dataset id (default: TQTfintech/40-2025-q1-q4).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision/branch (default: main).",
    )
    parser.add_argument(
        "--raw-audio-dir",
        default=str(RAW_AUDIO_DIR),
        help=f"Local raw audio directory (default: {RAW_AUDIO_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for number of files to import (for smoke tests).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    import_hf_audio(
        dataset_id=args.dataset_id,
        revision=args.revision,
        raw_audio_dir=Path(args.raw_audio_dir),
        overwrite=args.overwrite,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
