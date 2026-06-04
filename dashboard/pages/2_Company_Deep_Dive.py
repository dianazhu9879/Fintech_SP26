import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PER_TICKER = ROOT / "outputs" / "per_ticker"

st.set_page_config(page_title="Company Deep Dive", page_icon="🏢", layout="wide")

PLOT_BG = "#111111"; PAPER_BG = "#111111"
FONT_COLOR = "#e5e7eb"; GRID_COLOR = "#2a2a2a"
HORIZONS = [1, 3, 5, 7, 10, 21]
TICKER_COLORS = {
    "NVDA": "#76b900", "AAPL": "#a2aaad", "MSFT": "#00a4ef",
    "GOOGL": "#fbbc04", "META": "#0082fb", "AMZN": "#ff9900", "TSLA": "#cc0000",
}

@st.cache_data
def load_ticker(ticker):
    base = PER_TICKER / ticker
    ev  = pd.read_csv(base / "event_level_summary.csv")
    top = pd.read_csv(base / "topic_scores_by_quarter.csv")
    avg = pd.read_csv(base / "average_scores_by_topic.csv")
    ret = pd.read_csv(base / "stock_return_summary.csv")
    thc = pd.read_csv(base / "topic_horizon_correlation.csv")
    return ev, top, avg, ret, thc

st.markdown("## 🏢 Company Deep Dive")
st.divider()

ticker = st.selectbox("Select company", ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"])
color  = TICKER_COLORS[ticker]
ev, top, avg, ret, thc = load_ticker(ticker)

# ── Metric cards ──────────────────────────────────────────────────────────────
n_events = len(ev)
avg_sent  = ev["avg_llm_sentiment_score"].mean()
avg_risk  = ev["avg_llm_risk_intensity_score"].mean()
avg_ret5  = ev["excess_return_5d"].mean() * 100 if "excess_return_5d" in ev.columns else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Events analyzed", n_events)
c2.metric("Avg sentiment score", f"{avg_sent:.3f}")
c3.metric("Avg risk intensity", f"{avg_risk:.3f}")
c4.metric("Avg 5d excess return", f"{avg_ret5:+.2f}%" if avg_ret5 == avg_ret5 else "—")

st.divider()

# ── Event timeline: returns + sentiment ───────────────────────────────────────
st.markdown("#### Event Timeline")
horizon_pick = st.select_slider("Return horizon", [f"{h}d" for h in HORIZONS], value="5d")
h_col = f"excess_return_{horizon_pick.replace('d', '')}d"

if h_col in ev.columns:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ev["period"], y=(ev[h_col] * 100).round(2),
        name=f"{horizon_pick} Excess Return (%)",
        marker=dict(color=[(color if v >= 0 else "#ef4444")
                           for v in ev[h_col]]),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=ev["period"], y=ev["avg_llm_sentiment_score"].round(3),
        mode="lines+markers", name="Avg Sentiment Score",
        line=dict(color="#facc15", width=2), marker=dict(size=8),
        yaxis="y2",
    ))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, title="Quarter"),
        yaxis=dict(title=f"{horizon_pick} Excess Return (%)", gridcolor=GRID_COLOR),
        yaxis2=dict(title="Sentiment Score", overlaying="y", side="right",
                    gridcolor=GRID_COLOR, showgrid=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
        margin=dict(t=50, b=40, l=60, r=60),
        height=360,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Column {h_col} not found in event data.")

# ── Risk vs sentiment scatter ─────────────────────────────────────────────────
st.markdown("#### Risk Intensity vs Excess Return")
if h_col in ev.columns and "avg_llm_risk_intensity_score" in ev.columns:
    fig2 = go.Figure(go.Scatter(
        x=ev["avg_llm_risk_intensity_score"],
        y=(ev[h_col] * 100).round(2),
        mode="markers+text",
        text=ev["period"], textposition="top center",
        textfont=dict(size=10),
        marker=dict(color=color, size=12,
                    line=dict(color="#fff", width=1.5)),
    ))
    fig2.add_hline(y=0, line_dash="dot", line_color="#555")
    fig2.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(title="Avg Risk Intensity Score", gridcolor=GRID_COLOR),
        yaxis=dict(title=f"{horizon_pick} Excess Return (%)", gridcolor=GRID_COLOR),
        margin=dict(t=30, b=50, l=60, r=20),
        height=340,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Topic sentiment by quarter heatmap ────────────────────────────────────────
st.markdown("#### Topic Sentiment by Quarter")
if "topic" in top.columns and "quarter" in top.columns:
    sent_col = [c for c in top.columns if "sentiment" in c.lower()]
    if sent_col:
        piv = top.pivot(index="topic", columns="quarter", values=sent_col[0])
        fig3 = go.Figure(go.Heatmap(
            z=piv.values.round(2).tolist(),
            x=piv.columns.tolist(),
            y=piv.index.tolist(),
            text=[[f"{v:.2f}" if v == v else "—" for v in row]
                  for row in piv.values.tolist()],
            texttemplate="%{text}",
            colorscale=[[0.0, "#b91c1c"], [0.4, "#1a1a1a"],
                        [0.6, "#1a1a1a"], [1.0, "#15803d"]],
            zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="Sentiment", tickfont=dict(color=FONT_COLOR),
                          titlefont=dict(color=FONT_COLOR)),
        ))
        fig3.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            font=dict(color=FONT_COLOR, size=11),
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, autorange="reversed"),
            margin=dict(t=20, b=40, l=220, r=20),
            height=max(300, len(piv) * 30),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── Average scores table ──────────────────────────────────────────────────────
st.divider()
st.markdown("#### Average Scores by Topic")
if not avg.empty:
    cols_show = [c for c in avg.columns
                 if any(k in c.lower() for k in ["topic", "sentiment", "risk", "uncertain", "count"])]
    st.dataframe(avg[cols_show].round(3) if cols_show else avg.round(3),
                 use_container_width=True, height=300)
