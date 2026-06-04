"""
normalize.py

Reads all per-clip feature JSONs produced by audio_analysis.py,
flattens them into a panel, and z-scores each acoustic feature
relative to the speaker's (ticker's) own history across quarters.

Speakers with fewer than MIN_CALLS_FOR_SPEAKER_NORM quarters of data
fall back to a population norm computed across all speakers.

Output: normalized_features.csv — one row per Q&A exchange,
with both raw and z-scored feature columns.

Usage:
    python normalize.py --features_dir ./features --output ./normalized_features.csv
"""

import os
import json
import argparse
import re
import numpy as np
import pandas as pd

# Minimum number of calls (quarters) required to use per-speaker normalization.
# Speakers below this threshold fall back to population norm.
MIN_CALLS_FOR_SPEAKER_NORM = 3

# Features to normalize — must match keys produced by audio_analysis.py
FEATURES_TO_NORMALIZE = [
    "pause_latency_sec",
    "speech_rate_wpm",
    "filler_rate_per_min",
    "intra_pause_count",
    "intra_pause_mean_sec",
    "intra_pause_max_sec",
    "intra_pause_total_sec",
    "mean_pitch_hz",
    "pitch_variance_hz",
    "mean_rms_energy",
]


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(fname: str) -> dict:
    """
    Extract ticker, year, and month from filenames like:
        AAPL_2025_10_30_earnings_call_qa.mp3
        AVGO_2025_2_6_earnings_call_qa.mp3
    Returns dict with keys: ticker, year, month, quarter
    """
    stem = fname.replace("_features.json", "").replace(".mp3", "")
    match = re.match(r"^([A-Z]+)_(\d{4})_(\d{1,2})_\d+_", stem)
    if not match:
        return {"ticker": stem, "year": None, "month": None, "quarter": None}

    ticker = match.group(1)
    year   = int(match.group(2))
    month  = int(match.group(3))

    # Approximate fiscal quarter from calendar month
    quarter = (month - 1) // 3 + 1

    return {
        "ticker":  ticker,
        "year":    year,
        "month":   month,
        "quarter": quarter,
        "period":  f"{year}_Q{quarter}",
    }


# ---------------------------------------------------------------------------
# JSON loading and flattening
# ---------------------------------------------------------------------------

def load_feature_jsons(features_dir: str) -> pd.DataFrame:
    """
    Load all *_features.json files and flatten into a DataFrame.
    One row per Q&A exchange. Skips error files (missing 'exchanges' key).
    """
    rows = []
    skipped = []

    json_files = sorted(f for f in os.listdir(features_dir) if f.endswith("_features.json"))
    print(f"Found {len(json_files)} feature JSON(s).")

    for fname in json_files:
        path = os.path.join(features_dir, fname)
        with open(path) as f:
            data = json.load(f)

        if "exchanges" not in data:
            skipped.append(fname)
            continue

        meta = parse_filename(fname)

        for ex in data["exchanges"]:
            row = {
                "file":             fname,
                "ticker":           meta["ticker"],
                "year":             meta["year"],
                "month":            meta["month"],
                "quarter":          meta["quarter"],
                "period":           meta["period"],
                "management_speaker": data.get("management_speaker"),
                "exchange_idx":     ex.get("exchange_idx"),
                "mgmt_start":       ex.get("mgmt_start"),
                "mgmt_end":         ex.get("mgmt_end"),
                "duration_sec":     ex.get("duration_sec"),
                "word_count":       ex.get("word_count"),
                "answer_text":      ex.get("answer_text", ""),
            }
            for feat in FEATURES_TO_NORMALIZE:
                row[feat] = ex.get(feat)

            rows.append(row)

    if skipped:
        print(f"  Skipped {len(skipped)} error/incomplete JSON(s): {skipped}")

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} exchanges across {df['ticker'].nunique()} ticker(s).")
    return df


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_speaker_stats(df: pd.DataFrame) -> dict:
    """
    For each ticker, compute mean and std for each feature across all their exchanges.
    Returns nested dict: {ticker: {feature: {"mean": ..., "std": ...}}}
    """
    stats = {}
    call_counts = df.groupby("ticker")["period"].nunique()

    for ticker, group in df.groupby("ticker"):
        stats[ticker] = {}
        for feat in FEATURES_TO_NORMALIZE:
            vals = group[feat].dropna()
            stats[ticker][feat] = {
                "mean":       float(vals.mean()) if len(vals) > 0 else np.nan,
                "std":        float(vals.std())  if len(vals) > 1 else np.nan,
                "n_calls":    int(call_counts.get(ticker, 0)),
                "norm_type":  "speaker" if call_counts.get(ticker, 0) >= MIN_CALLS_FOR_SPEAKER_NORM else "population",
            }
    return stats


def compute_population_stats(df: pd.DataFrame) -> dict:
    """
    Population-level stats across all speakers — used as fallback
    for speakers with too few calls.
    Returns dict: {feature: {"mean": ..., "std": ...}}
    """
    pop = {}
    for feat in FEATURES_TO_NORMALIZE:
        vals = df[feat].dropna()
        pop[feat] = {
            "mean": float(vals.mean()) if len(vals) > 0 else np.nan,
            "std":  float(vals.std())  if len(vals) > 1 else np.nan,
        }
    return pop


def z_score(value, mean, std) -> float | None:
    if value is None or np.isnan(value):
        return None
    if std is None or np.isnan(std) or std == 0:
        return None
    return round((value - mean) / std, 4)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add z-scored columns (z_{feature}) to the DataFrame.
    Uses per-speaker norm where data is sufficient, population norm otherwise.
    """
    speaker_stats = compute_speaker_stats(df)
    pop_stats     = compute_population_stats(df)

    # Print normalization summary
    print("\n  Normalization strategy per ticker:")
    call_counts = df.groupby("ticker")["period"].nunique()
    for ticker in sorted(call_counts.index):
        n = call_counts[ticker]
        strategy = "per-speaker" if n >= MIN_CALLS_FOR_SPEAKER_NORM else f"population fallback (only {n} call(s))"
        print(f"    {ticker}: {strategy}")

    z_rows = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        z_row  = {}

        for feat in FEATURES_TO_NORMALIZE:
            raw_val  = row[feat]
            s_stats  = speaker_stats[ticker][feat]

            if s_stats["norm_type"] == "speaker":
                mean = s_stats["mean"]
                std  = s_stats["std"]
            else:
                mean = pop_stats[feat]["mean"]
                std  = pop_stats[feat]["std"]

            z_row[f"z_{feat}"]       = z_score(raw_val, mean, std)
            z_row[f"norm_type_{feat}"] = s_stats["norm_type"]

        z_rows.append(z_row)

    z_df = pd.DataFrame(z_rows, index=df.index)
    return pd.concat([df, z_df], axis=1)


# ---------------------------------------------------------------------------
# Composite acoustic score
# ---------------------------------------------------------------------------

def compute_acoustic_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine z-scored features into a single acoustic sentiment proxy.

    Weighting rationale:
        Higher pitch / energy  → positive arousal → positive direction
        Higher pause latency   → hesitation → negative direction
        Higher speech rate     → confidence → positive direction
        Higher filler rate     → uncertainty → negative direction
        Higher intra pauses    → hesitation → negative direction

    All features are signed so that more positive = more confident/positive tone.
    Composite is a weighted average of available z-scores.
    """
    weights = {
        "z_mean_pitch_hz":        +0.25,
        "z_pitch_variance_hz":    +0.10,
        "z_mean_rms_energy":      +0.20,
        "z_speech_rate_wpm":      +0.20,
        "z_pause_latency_sec":    -0.10,
        "z_filler_rate_per_min":  -0.10,
        "z_intra_pause_mean_sec": -0.05,
    }

    composites = []
    for _, row in df.iterrows():
        total_weight = 0.0
        weighted_sum = 0.0
        for col, w in weights.items():
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                weighted_sum += w * val
                total_weight += abs(w)

        score = round(weighted_sum / total_weight, 4) if total_weight > 0 else None
        composites.append(score)

    df = df.copy()
    df["acoustic_composite_z"] = composites
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Normalize acoustic features from audio_analysis.py output."
    )
    parser.add_argument("--features_dir", required=True,
                        help="Directory containing *_features.json files")
    parser.add_argument("--output",       default="normalized_features.csv",
                        help="Output CSV path (default: normalized_features.csv)")
    parser.add_argument("--stats_output", default="speaker_stats.json",
                        help="Optional: save speaker normalization stats to JSON")
    args = parser.parse_args()

    print(f"\nLoading feature JSONs from: {args.features_dir}")
    df = load_feature_jsons(args.features_dir)

    if df.empty:
        print("No data loaded — check that features_dir contains valid *_features.json files.")
        return

    print("\nNormalizing features…")
    df = normalize(df)

    print("\nComputing acoustic composite score…")
    df = compute_acoustic_composite(df)

    # Drop answer_text for the stats output (keep in main CSV)
    stats_df = df.drop(columns=["answer_text"], errors="ignore")

    df.to_csv(args.output, index=False)
    print(f"\n✓ Normalized panel saved → {args.output}")
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")

    # Save speaker stats for reference / debugging
    speaker_stats = compute_speaker_stats(df)
    with open(args.stats_output, "w") as f:
        json.dump(speaker_stats, f, indent=2)
    print(f"✓ Speaker stats saved  → {args.stats_output}")

    # --- Per-call audio sentiment rollup ---
    # One row per ticker+period: mean composite (signal), std (consistency),
    # and counts. This is the primary output for downstream analysis.
    call_level = (
        df.groupby(["ticker", "year", "quarter", "period"])
        .agg(
            audio_sentiment_mean  = ("acoustic_composite_z", "mean"),
            audio_sentiment_std   = ("acoustic_composite_z", "std"),
            audio_sentiment_min   = ("acoustic_composite_z", "min"),
            audio_sentiment_max   = ("acoustic_composite_z", "max"),
            pause_latency_mean    = ("z_pause_latency_sec",   "mean"),
            speech_rate_mean      = ("z_speech_rate_wpm",     "mean"),
            filler_rate_mean      = ("z_filler_rate_per_min", "mean"),
            pitch_mean            = ("z_mean_pitch_hz",       "mean"),
            energy_mean           = ("z_mean_rms_energy",     "mean"),
            n_exchanges           = ("exchange_idx",           "count"),
        )
        .reset_index()
        .sort_values(["ticker", "year", "quarter"])
    )
    call_level = call_level.round(4)

    call_output = args.output.replace(".csv", "_per_call.csv")
    call_level.to_csv(call_output, index=False)
    print(f"✓ Per-call audio sentiment → {call_output}")

    # Quick summary
    print(f"\nAudio sentiment by ticker (mean across quarters):")
    summary = call_level.groupby("ticker")["audio_sentiment_mean"].agg(["mean", "std", "count"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()