from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from audio_alpha.config import MANIFEST_PATH, PROCESSED_DIR
from audio_alpha.extract_features import extract_call_features
from audio_alpha.score_audio_sentiment import score_dataframe


def _extract_call_features_from_dict(row: dict[str, object]) -> dict[str, object]:
    return extract_call_features(pd.Series(row))


def _merge_and_write_features(
    existing_df: pd.DataFrame,
    new_rows: list[dict[str, object]],
    output_path: Path,
) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing_df.empty:
        merged = new_df
    elif new_df.empty:
        merged = existing_df.copy()
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    if "call_id" in merged.columns:
        merged = merged.drop_duplicates(subset="call_id", keep="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def build_call_feature_table(
    manifest_path: Path = MANIFEST_PATH,
    call_features_output_path: Path = PROCESSED_DIR / "audio_call_features_comprehensive.csv",
    table_output_path: Path = PROCESSED_DIR / "audio_call_feature_table_comprehensive.csv",
    num_workers: int = max(1, (os.cpu_count() or 2) - 1),
    resume: bool = True,
    checkpoint_every: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest at {manifest_path}. Run build_manifest and preprocess_audio first."
        )

    manifest_df = pd.read_csv(manifest_path)
    existing_features_df = pd.DataFrame()
    processed_call_ids: set[str] = set()
    if resume and call_features_output_path.exists():
        existing_features_df = pd.read_csv(call_features_output_path)
        if "call_id" in existing_features_df.columns:
            processed_call_ids = set(
                existing_features_df["call_id"].dropna().astype(str).tolist()
            )

    if processed_call_ids:
        remaining_df = manifest_df[
            ~manifest_df["call_id"].astype(str).isin(processed_call_ids)
        ].copy()
    else:
        remaining_df = manifest_df.copy()

    print(
        f"Manifest rows: {len(manifest_df)} | already processed: {len(processed_call_ids)} | remaining: {len(remaining_df)}"
    )

    new_rows: list[dict[str, object]] = []
    remaining_records = remaining_df.to_dict("records")
    if remaining_records:
        if num_workers <= 1:
            for idx, row in enumerate(
                tqdm(remaining_records, total=len(remaining_records), desc="call features"),
                start=1,
            ):
                new_rows.append(_extract_call_features_from_dict(row))
                if idx % checkpoint_every == 0:
                    _merge_and_write_features(
                        existing_df=existing_features_df,
                        new_rows=new_rows,
                        output_path=call_features_output_path,
                    )
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                iterator = executor.map(
                    _extract_call_features_from_dict,
                    remaining_records,
                    chunksize=1,
                )
                for idx, feature_row in enumerate(
                    tqdm(iterator, total=len(remaining_records), desc="call features"),
                    start=1,
                ):
                    new_rows.append(feature_row)
                    if idx % checkpoint_every == 0:
                        _merge_and_write_features(
                            existing_df=existing_features_df,
                            new_rows=new_rows,
                            output_path=call_features_output_path,
                        )

    call_features_df = _merge_and_write_features(
        existing_df=existing_features_df,
        new_rows=new_rows,
        output_path=call_features_output_path,
    )
    scored_df = score_dataframe(call_features_df)

    call_features_output_path.parent.mkdir(parents=True, exist_ok=True)
    table_output_path.parent.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(table_output_path, index=False)

    print(f"Wrote call features to {call_features_output_path}")
    print(f"Wrote comprehensive call feature table to {table_output_path}")
    return call_features_df, scored_df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a single call-level audio feature table (no segment rollups)."
    )
    parser.add_argument(
        "--manifest-path",
        default=str(MANIFEST_PATH),
        help=f"Manifest CSV path (default: {MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--call-features-output",
        default=str(PROCESSED_DIR / "audio_call_features_comprehensive.csv"),
        help="Output CSV for raw call features.",
    )
    parser.add_argument(
        "--table-output",
        default=str(PROCESSED_DIR / "audio_call_feature_table_comprehensive.csv"),
        help="Output CSV for scored call-level feature table.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes for call-level feature extraction.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save intermediate call features every N processed calls.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from existing call-features output if present (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing output and recompute all calls.",
    )
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    build_call_feature_table(
        manifest_path=Path(args.manifest_path),
        call_features_output_path=Path(args.call_features_output),
        table_output_path=Path(args.table_output),
        num_workers=args.num_workers,
        resume=args.resume,
        checkpoint_every=max(1, args.checkpoint_every),
    )


if __name__ == "__main__":
    main()
