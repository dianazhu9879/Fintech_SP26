import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED = ROOT / "outputs" / "combined"

st.set_page_config(page_title="Event Explorer", page_icon="🔎", layout="wide")

PLOT_BG = "#111111"; PAPER_BG = "#111111"
FONT_COLOR = "#e5e7eb"; GRID_COLOR = "#2a2a2a"
HORIZONS = [1, 3, 5, 7, 10, 21]
TICKER_COLORS = {
    "NVDA": "#76b900", "AAPL": "#a2aaad", "MSFT": "#00a4ef",
    "GOOGL": "#fbbc04", "META": "#0082fb", "AMZN": "#ff9900", "TSLA": "#cc0000",
}

@st.cache_data
def load():
    return pd.read_csv(COMBINED / "panel.csv")

panel = load()
TICKERS = sorted(panel["ticker"].unique())

st.markdown("## 🔎 Event Explorer")
st.markdown(
    "<span style='color:#9ca3af;font-size:0.9rem'>"
    "Explore individual earnings events across all companies</span>",
    unsafe_allow_html=True,
)
st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
selected_tickers = col1.multiselect(
    "Companies", TICKERS, default=TICKERS,
)
horizon_pick = col2.selectbox("Return horizon", [f"{h}d" for h in HORIZONS], index=2)
h_days = int(horizon_pick.replace("d", ""))
h_col  = f"excess_return_{h_days}d"

df = panel[panel["ticker"].isin(selected_tickers)].copy()

# ── Scatter: sentiment vs return ──────────────────────────────────────────────
st.markdown(f"#### Sentiment Score vs {horizon_pick} Excess Return")
if h_col in df.columns and "avg_llm_sentiment_score" in df.columns:
    sub = df.dropna(subset=["avg_llm_sentiment_score", h_col])
    fig = go.Figure()
    for ticker in selected_tickers:
        t = sub[sub["ticker"] == ticker]
        if t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=t["avg_llm_sentiment_score"],
            y=(t[h_col] * 100).round(2),
            mode="markers+text",
            name=ticker,
            text=t["period"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(color=TICKER_COLORS.get(ticker, "#888"),
                        size=11, opacity=0.9,
                        line=dict(color="#fff", width=1.2)),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.add_vline(x=sub["avg_llm_sentiment_score"].mean(),
                  line_dash="dot", line_color="#555")
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(title="Avg LLM Sentiment Score", gridcolor=GRID_COLOR),
        yaxis=dict(title=f"{horizon_pick} Excess Return (%)", gridcolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=50, l=60, r=20),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Scatter: risk vs return ───────────────────────────────────────────────────
st.markdown(f"#### Risk Intensity vs {horizon_pick} Excess Return")
if h_col in df.columns and "avg_llm_risk_intensity_score" in df.columns:
    sub = df.dropna(subset=["avg_llm_risk_intensity_score", h_col])
    fig2 = go.Figure()
    for ticker in selected_tickers:
        t = sub[sub["ticker"] == ticker]
        if t.empty:
            continue
        fig2.add_trace(go.Scatter(
            x=t["avg_llm_risk_intensity_score"],
            y=(t[h_col] * 100).round(2),
            mode="markers",
            name=ticker,
            marker=dict(color=TICKER_COLORS.get(ticker, "#888"),
                        size=11, opacity=0.9,
                        line=dict(color="#fff", width=1.2)),
        ))
    fig2.add_hline(y=0, line_dash="dot", line_color="#555")
    fig2.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(title="Avg Risk Intensity Score", gridcolor=GRID_COLOR),
        yaxis=dict(title=f"{horizon_pick} Excess Return (%)", gridcolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=50, l=60, r=20),
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Return distribution by company ────────────────────────────────────────────
st.markdown(f"#### {horizon_pick} Excess Return by Company")
if h_col in df.columns:
    fig3 = go.Figure()
    for ticker in selected_tickers:
        t = df[df["ticker"] == ticker].dropna(subset=[h_col])
        if t.empty:
            continue
        fig3.add_trace(go.Box(
            y=(t[h_col] * 100).round(2),
            name=ticker,
            marker_color=TICKER_COLORS.get(ticker, "#888"),
            boxpoints="all", jitter=0.4, pointpos=0,
        ))
    fig3.add_hline(y=0, line_dash="dot", line_color="#555")
    fig3.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        yaxis=dict(title=f"{horizon_pick} Excess Return (%)", gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=40, l=60, r=20),
        height=380,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Raw event table ───────────────────────────────────────────────────────────
with st.expander("Raw event data"):
    cols = ["ticker", "period"] + [c for c in df.columns
            if any(k in c for k in ["avg_llm", "excess_return", "pct_negative"])]
    cols = [c for c in cols if c in df.columns][:18]
    st.dataframe(df[cols].round(3), use_container_width=True, height=400)
