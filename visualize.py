"""
visualize.py

Saves individual PNG charts from normalized_features_per_call.csv.
Charts are written to --output_dir (default: ./plots/).

Usage:
    python visualize.py --input ./normalized_features_per_call.csv
    python visualize.py --input ./normalized_features_per_call.csv --output_dir ./plots
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BG       = "#ffffff"
PANEL_BG = "#f8f9fa"
GRID     = "#e0e0e0"
TEXT     = "#1a1a1a"
MUTED    = "#888888"
BORDER   = "#cccccc"

POS_COLOR  = "#2e7d32"   # dark green  — positive sentiment
NEG_COLOR  = "#c62828"   # dark red    — negative sentiment
POS_LIGHT  = "#a5d6a7"   # light green — fills / bands
NEG_LIGHT  = "#ef9a9a"   # light red   — fills / bands
NEUTRAL    = "#9e9e9e"   # grey        — zero lines, neutral elements

# For multi-ticker charts, cycle through green/red shades
TICKER_PALETTE = [
    "#2e7d32", "#c62828", "#388e3c", "#d32f2f",
    "#1b5e20", "#b71c1c", "#43a047", "#e53935",
]

FEATURE_LABELS = {
    "pause_latency_mean": "Pause Latency",
    "speech_rate_mean":   "Speech Rate",
    "filler_rate_mean":   "Filler Rate",
    "pitch_mean":         "Mean Pitch",
    "energy_mean":        "RMS Energy",
}

def base_style():
    plt.rcParams.update({
        "figure.facecolor":     BG,
        "axes.facecolor":       PANEL_BG,
        "axes.edgecolor":       BORDER,
        "axes.labelcolor":      TEXT,
        "axes.titlecolor":      TEXT,
        "axes.titlesize":       12,
        "axes.titleweight":     "semibold",
        "axes.labelsize":       9,
        "xtick.color":          TEXT,
        "ytick.color":          TEXT,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "grid.color":           GRID,
        "grid.linewidth":       0.7,
        "text.color":           TEXT,
        "legend.facecolor":     BG,
        "legend.edgecolor":     BORDER,
        "legend.labelcolor":    TEXT,
        "legend.fontsize":      8,
        "font.family":          "monospace",
        "lines.linewidth":      2.0,
        "lines.markersize":     7,
    })

def fig_setup(title, figsize=(13, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    ax.set_facecolor(PANEL_BG)
    return fig, ax

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")

def sentiment_color(val, alpha=1.0):
    """Return green for positive, red for negative, grey for near-zero."""
    if val is None or np.isnan(val):
        return NEUTRAL
    return POS_COLOR if val >= 0 else NEG_COLOR

def sentiment_color_light(val):
    return POS_LIGHT if (val is not None and not np.isnan(val) and val >= 0) else NEG_LIGHT

# ---------------------------------------------------------------------------
# Chart 1 — Sentiment timeline per ticker
# ---------------------------------------------------------------------------

def chart_timeline(df, colors, out):
    fig, ax = fig_setup("Audio Sentiment Over Time — Per Ticker", figsize=(13, 5))

    for ticker in sorted(df["ticker"].unique()):
        sub   = df[df["ticker"] == ticker].sort_values("period")
        color = colors[ticker]
        ax.plot(sub["period"], sub["audio_sentiment_mean"],
                marker="o", color=color, label=ticker, zorder=3)
        ax.fill_between(
            sub["period"],
            sub["audio_sentiment_mean"] - sub["audio_sentiment_std"].fillna(0),
            sub["audio_sentiment_mean"] + sub["audio_sentiment_std"].fillna(0),
            alpha=0.12, color=color,
        )

    ax.axhline(0, color=NEUTRAL, lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Period")
    ax.set_ylabel("Composite z-score")
    ax.grid(axis="y", alpha=0.5)
    ax.legend(framealpha=0.9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, out)

# ---------------------------------------------------------------------------
# Chart 2 — Ticker ranking (avg sentiment, sorted bar)
# ---------------------------------------------------------------------------

def chart_ranking(df, colors, out):
    ranked = (df.groupby("ticker")["audio_sentiment_mean"]
                .mean()
                .sort_values(ascending=True))

    fig, ax = fig_setup("Ticker Ranking — Avg Audio Sentiment", figsize=(10, 5))

    bar_colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in ranked.values]
    bars = ax.barh(ranked.index, ranked.values,
                   color=bar_colors, alpha=0.85, height=0.55)

    for bar, val in zip(bars, ranked.values):
        if val >= 0:
            # Label just outside the right end of positive bars
            x_pos, ha = val + 0.001, "left"
        else:
            # Label just inside the bar near zero — avoids clipping into ticker label
            x_pos, ha = -0.001, "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8, color=TEXT)

    ax.axvline(0, color=NEUTRAL, lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Mean composite z-score")
    ax.grid(axis="x", alpha=0.4)
    save(fig, out)

# ---------------------------------------------------------------------------
# Chart 3 — Feature heatmap (tickers × features)
# ---------------------------------------------------------------------------

def chart_heatmap(df, out):
    feat_cols = [c for c in FEATURE_LABELS if c in df.columns]
    heat      = df.groupby("ticker")[feat_cols].mean()
    heat.columns = [FEATURE_LABELS[c] for c in feat_cols]

    data    = heat.values
    tickers = list(heat.index)
    feats   = list(heat.columns)

    fig, ax = plt.subplots(
        figsize=(len(feats) * 1.8 + 2, len(tickers) * 0.9 + 2),
        facecolor=BG
    )
    fig.suptitle("Feature Heatmap — Ticker Avg (z-score)",
                 fontsize=13, fontweight="bold", color=TEXT)
    ax.set_facecolor(PANEL_BG)

    vmax = max(abs(np.nanmax(data)), abs(np.nanmin(data)), 1.0)
    im   = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=25, ha="right", fontsize=9, color=TEXT)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=9, color=TEXT)
    ax.tick_params(colors=TEXT)

    for i in range(len(tickers)):
        for j in range(len(feats)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(val) > vmax * 0.6 else TEXT)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.ax.tick_params(colors=MUTED)

    save(fig, out)

# ---------------------------------------------------------------------------
# Chart 4 — Volatility vs sentiment scatter
# ---------------------------------------------------------------------------

def chart_volatility_scatter(df, colors, out):
    fig, ax = fig_setup("Sentiment Level vs Intra-Call Volatility", figsize=(9, 6))

    for ticker in sorted(df["ticker"].unique()):
        sub   = df[df["ticker"] == ticker]
        color = colors[ticker]
        ax.scatter(sub["audio_sentiment_mean"],
                   sub["audio_sentiment_std"].fillna(0),
                   color=color, s=90, alpha=0.85, label=ticker, zorder=3,
                   edgecolors=BORDER, linewidths=0.5)
        for _, row in sub.iterrows():
            ax.annotate(
                row["period"],
                (row["audio_sentiment_mean"],
                 row["audio_sentiment_std"] if pd.notna(row["audio_sentiment_std"]) else 0),
                textcoords="offset points", xytext=(6, 4),
                fontsize=6, color=MUTED,
            )

    ax.axvline(0, color=NEUTRAL, lw=0.9, ls="--", alpha=0.6)
    ax.set_xlabel("Audio Sentiment Mean (z-score)")
    ax.set_ylabel("Sentiment Std Dev (intra-call volatility)")
    ax.grid(alpha=0.4)
    ax.legend(framealpha=0.9)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    kw   = dict(fontsize=7, color=MUTED, alpha=0.7, ha="center")
    ax.text(xlim[1] * 0.6, ylim[1] * 0.88, "Positive\nInconsistent", **kw)
    ax.text(xlim[0] * 0.6, ylim[1] * 0.88, "Negative\nInconsistent", **kw)
    ax.text(xlim[1] * 0.6, ylim[0] + (ylim[1]-ylim[0])*0.06, "Positive\nConsistent", **kw)
    ax.text(xlim[0] * 0.6, ylim[0] + (ylim[1]-ylim[0])*0.06, "Negative\nConsistent", **kw)

    save(fig, out)

# ---------------------------------------------------------------------------
# Chart 5 — Per-feature distributions (box plots)
# ---------------------------------------------------------------------------

def chart_feature_distributions(df, out):
    feat_cols = [c for c in FEATURE_LABELS if c in df.columns]
    labels    = [FEATURE_LABELS[c] for c in feat_cols]
    data_list = [df[c].dropna().values for c in feat_cols]

    # Sign of each feature: positive = good signal, negative = stress/hesitation
    feature_sign = {
        "pause_latency_mean": -1,
        "speech_rate_mean":   +1,
        "filler_rate_mean":   -1,
        "pitch_mean":         +1,
        "energy_mean":        +1,
    }

    fig, ax = fig_setup("Feature Distributions Across All Calls (z-score)", figsize=(11, 5))

    bp = ax.boxplot(
        data_list, vert=True, patch_artist=True, labels=labels,
        medianprops=dict(color=TEXT, lw=1.8),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        flierprops=dict(marker="o", color=MUTED, markersize=3, alpha=0.5),
    )

    for patch, feat in zip(bp["boxes"], feat_cols):
        sign = feature_sign.get(feat, 1)
        patch.set_facecolor(POS_LIGHT if sign > 0 else NEG_LIGHT)
        patch.set_alpha(0.8)

    ax.axhline(0, color=NEUTRAL, lw=0.9, ls="--", alpha=0.6)
    ax.set_ylabel("z-score")
    ax.grid(axis="y", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=POS_LIGHT, label="Positive signal"),
                 Patch(facecolor=NEG_LIGHT, label="Stress / hesitation")],
        framealpha=0.9, fontsize=8,
    )

    save(fig, out)

# ---------------------------------------------------------------------------
# Chart 6 — Per-ticker feature breakdown (grouped bars, green/red per sign)
# ---------------------------------------------------------------------------

def chart_per_ticker(df, out_dir):
    """
    One chart per ticker. Each chart has one subplot per feature (grid layout),
    so feature names appear as subplot titles on the x-axis — no legend needed.
    Bars are colored green (positive) / red (negative) by value.
    """
    feat_cols  = [c for c in FEATURE_LABELS if c in df.columns]
    feat_names = [FEATURE_LABELS[c] for c in feat_cols]
    n_feats    = len(feat_cols)

    # Grid: 2 columns, enough rows to fit all features
    n_cols = 2
    n_rows = (n_feats + 1) // n_cols

    for ticker in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == ticker].sort_values("period")
        if sub.empty:
            continue

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 5.5, n_rows * 3.5),
            facecolor=BG,
        )
        fig.suptitle(f"{ticker} — Feature Breakdown by Quarter",
                     fontsize=13, fontweight="bold", color=TEXT, y=1.01)

        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, (feat, name) in enumerate(zip(feat_cols, feat_names)):
            ax  = axes_flat[i]
            ax.set_facecolor(PANEL_BG)
            ax.spines[:].set_color(BORDER)

            vals       = sub[feat].fillna(0).values
            bar_colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]

            ax.bar(sub["period"], vals, color=bar_colors, alpha=0.82, width=0.5)
            ax.axhline(0, color=NEUTRAL, lw=0.9, ls="--", alpha=0.6)
            ax.set_title(name, fontsize=10, fontweight="semibold", color=TEXT)
            ax.set_ylabel("z-score", fontsize=8)
            ax.grid(axis="y", alpha=0.35)
            ax.tick_params(colors=TEXT)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)

            # Value labels on bars
            for xi, val in enumerate(vals):
                ax.text(xi, val + (0.02 if val >= 0 else -0.02),
                        f"{val:+.2f}",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=6.5, color=TEXT)

        # Hide any unused subplots
        for j in range(n_feats, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        out = os.path.join(out_dir, f"6_per_ticker_{ticker}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  ✓ 6_per_ticker_{ticker}.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,
                        help="Path to normalized_features_per_call.csv")
    parser.add_argument("--output_dir", default="plots",
                        help="Directory to save PNG charts (default: ./plots)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} call(s) across {df['ticker'].nunique()} ticker(s).\n")

    if df.empty:
        print("No data to visualize.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    base_style()

    tickers = sorted(df["ticker"].unique())
    colors  = {t: TICKER_PALETTE[i % len(TICKER_PALETTE)] for i, t in enumerate(tickers)}

    print(f"Saving charts to ./{args.output_dir}/\n")

    chart_timeline(              df, colors, os.path.join(args.output_dir, "1_sentiment_timeline.png"))
    chart_ranking(               df, colors, os.path.join(args.output_dir, "2_ticker_ranking.png"))
    chart_heatmap(               df,         os.path.join(args.output_dir, "3_feature_heatmap.png"))
    chart_volatility_scatter(    df, colors, os.path.join(args.output_dir, "4_volatility_scatter.png"))
    chart_feature_distributions( df,         os.path.join(args.output_dir, "5_feature_distributions.png"))
    chart_per_ticker(            df,         args.output_dir)

    total = len([f for f in os.listdir(args.output_dir) if f.endswith(".png")])
    print(f"\nDone — {total} charts saved to ./{args.output_dir}/")

if __name__ == "__main__":
    main()