from pathlib import Path

import pandas as pd

from audio_alpha.build_manifest import build_manifest, parse_call_date


def test_parse_call_date_from_filename() -> None:
    assert parse_call_date("90eed-2025-05-01-10-56-15.mp3") == "2025-05-01"
    assert parse_call_date("no-date.mp3") is None


def test_build_manifest_parses_ticker_dates_and_duplicate_ids(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "audio"
    aapl_dir = raw_dir / "AAPL"
    nvda_dir = raw_dir / "nvda"
    aapl_dir.mkdir(parents=True)
    nvda_dir.mkdir(parents=True)

    (aapl_dir / "90eed-2025-05-01-10-56-15.mp3").touch()
    (aapl_dir / "aaaaa-2025-05-01-12-00-00.mp3").touch()
    (nvda_dir / "call-without-date.mp3").touch()

    output_path = tmp_path / "data" / "interim" / "audio_manifest.csv"
    df = build_manifest(raw_audio_dir=raw_dir, output_path=output_path)

    assert output_path.exists()
    assert list(df["ticker"]) == ["AAPL", "AAPL", "NVDA"]
    assert list(df["call_id"]) == [
        "AAPL_2025_05_01",
        "AAPL_2025_05_01_02",
        "NVDA_CALL_WITHOUT_DATE",
    ]
    assert df.loc[0, "call_date"] == "2025-05-01"
    assert df.loc[2, "call_date"] is None

    saved = pd.read_csv(output_path)
    assert saved.columns.tolist() == [
        "call_id",
        "ticker",
        "audio_path",
        "source_filename",
        "source_stem",
        "call_date",
        "file_ext",
    ]
