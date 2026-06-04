from __future__ import annotations

from pathlib import Path

import pandas as pd

from audio_alpha.config import (
    CALL_FEATURE_TABLE_PATH,
    CALL_SCORES_PATH,
    SEGMENT_SCORES_PATH,
)
from audio_alpha.score_audio_sentiment import SCORE_COLUMNS


def _segment_score_aggregates(segment_scores: pd.DataFrame) -> pd.DataFrame:
    if segment_scores.empty:
        return pd.DataFrame(columns=["call_id"])

    available_scores = [column for column in SCORE_COLUMNS if column in segment_scores.columns]
    if not available_scores:
        return segment_scores[["call_id"]].drop_duplicates()

    sorted_scores = segment_scores.sort_values(["call_id", "segment_id"])
    grouped = sorted_scores.groupby("call_id", dropna=False)

    pieces: list[pd.DataFrame] = []
    for stat in ("mean", "std", "min", "max"):
        agg = grouped[available_scores].agg(stat)
        agg.columns = [f"segment_{column}_{stat}" for column in agg.columns]
        pieces.append(agg)

    first = grouped[available_scores].first()
    last = grouped[available_scores].last()
    delta = last - first
    delta.columns = [f"segment_{column}_delta" for column in delta.columns]
    pieces.append(delta)

    return pd.concat(pieces, axis=1).reset_index()


def aggregate_call_features(
    call_scores_path: Path = CALL_SCORES_PATH,
    segment_scores_path: Path = SEGMENT_SCORES_PATH,
    output_path: Path = CALL_FEATURE_TABLE_PATH,
) -> pd.DataFrame:
    if not call_scores_path.exists():
        raise FileNotFoundError(f"Missing call scores at {call_scores_path}.")
    if not segment_scores_path.exists():
        raise FileNotFoundError(f"Missing segment scores at {segment_scores_path}.")

    call_scores = pd.read_csv(call_scores_path)
    segment_scores = pd.read_csv(segment_scores_path)
    segment_aggregates = _segment_score_aggregates(segment_scores)

    final = call_scores.merge(segment_aggregates, on="call_id", how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"Wrote final call feature table to {output_path}")
    return final


if __name__ == "__main__":
    aggregate_call_features()
