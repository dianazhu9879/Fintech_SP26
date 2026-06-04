import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED = ROOT / "outputs" / "combined"

st.set_page_config(page_title="Topic Analysis", page_icon="🔍", layout="wide")

PLOT_BG = "#111111"; PAPER_BG = "#111111"
FONT_COLOR = "#e5e7eb"; GRID_COLOR = "#2a2a2a"
HORIZONS = [1, 3, 5, 7, 10, 21]

@st.cache_data
def load():
    return pd.read_csv(COMBINED / "topic_horizon_correlations.csv")

tc = load()

st.markdown("## 🔍 Topic-Level Sentiment Signals")
st.markdown(
    "<span style='color:#9ca3af;font-size:0.9rem'>"
    "Per-topic sentiment → excess return correlation across the panel</span>",
    unsafe_allow_html=True,
)
st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([1, 1, 2])
horizon = col_a.selectbox("Return horizon", [f"{h}d" for h in HORIZONS], index=2)
h_days  = int(horizon.replace("d", ""))
metric  = col_b.selectbox("Signal", ["Sentiment", "Risk Intensity"])
min_n   = col_c.slider("Minimum events per topic (n ≥)", 2, 8, 3)
show_ind = col_c.checkbox("Include indicative-only topics", value=True)

rho_col = ("rho_sentiment_vs_excess_return" if metric == "Sentiment"
           else "rho_risk_intensity_vs_excess_return")

# ── Filter ────────────────────────────────────────────────────────────────────
df = tc[tc["horizon_days"] == h_days].copy()
df = df[df["n_events"] >= min_n]
df = df.dropna(subset=[rho_col])
if not show_ind:
    df = df[~df["indicative_only"]]
df = df.sort_values(rho_col)

# ── Top / Bottom bar chart ────────────────────────────────────────────────────
n_show = 10
bottom = df.head(n_show)
top    = df.tail(n_show).iloc[::-1]

def shorten(s, n=36):
    return s if len(s) <= n else s[:n-1] + "…"

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### 🔴 Most Negative — {metric} at {horizon}")
    if bottom.empty:
        st.info("No topics match the current filters.")
    else:
        fig = go.Figure(go.Bar(
            orientation="h",
            x=bottom[rho_col].round(3),
            y=[shorten(t) for t in bottom["topic"]],
            text=[f"n={n}" for n in bottom["n_events"]],
            textposition="outside",
            marker=dict(color=bottom[rho_col],
                        colorscale=[[0, "#b91c1c"], [1, "#ef4444"]],
                        cmin=-1, cmax=0),
        ))
        fig.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            font=dict(color=FONT_COLOR),
            xaxis=dict(title="ρ", gridcolor=GRID_COLOR, range=[-1.2, 0]),
            yaxis=dict(gridcolor=GRID_COLOR),
            margin=dict(t=20, b=40, l=220, r=60),
            height=380, showlegend=False,
        )
        fig.add_vline(x=0, line_dash="dot", line_color="#555")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown(f"#### 🟢 Most Positive — {metric} at {horizon}")
    if top.empty:
        st.info("No topics match the current filters.")
    else:
        fig2 = go.Figure(go.Bar(
            orientation="h",
            x=top[rho_col].round(3),
            y=[shorten(t) for t in top["topic"]],
            text=[f"n={n}" for n in top["n_events"]],
            textposition="outside",
            marker=dict(color=top[rho_col],
                        colorscale=[[0, "#22c55e"], [1, "#15803d"]],
                        cmin=0, cmax=1),
        ))
        fig2.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            font=dict(color=FONT_COLOR),
            xaxis=dict(title="ρ", gridcolor=GRID_COLOR, range=[0, 1.2]),
            yaxis=dict(gridcolor=GRID_COLOR),
            margin=dict(t=20, b=40, l=220, r=60),
            height=380, showlegend=False,
        )
        fig2.add_vline(x=0, line_dash="dot", line_color="#555")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Topic correlation across all horizons ─────────────────────────────────────
st.markdown("#### Correlation Across All Horizons — Search a Topic")
all_topics = sorted(tc["topic"].unique())
selected   = st.selectbox("Select topic", all_topics,
                           index=all_topics.index("AI Infrastructure")
                           if "AI Infrastructure" in all_topics else 0)

sub = tc[tc["topic"] == selected].sort_values("horizon_days")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=sub["horizon_days"], y=sub["rho_sentiment_vs_excess_return"].round(3),
    mode="lines+markers", name="Sentiment ρ",
    line=dict(color="#4ade80", width=2.5), marker=dict(size=8),
))
fig3.add_trace(go.Scatter(
    x=sub["horizon_days"], y=sub["rho_risk_intensity_vs_excess_return"].round(3),
    mode="lines+markers", name="Risk Intensity ρ",
    line=dict(color="#f87171", width=2.5), marker=dict(size=8),
))
fig3.add_hline(y=0, line_dash="dot", line_color="#555")
fig3.update_layout(
    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_COLOR),
    xaxis=dict(tickvals=HORIZONS, ticktext=[f"{h}d" for h in HORIZONS],
               gridcolor=GRID_COLOR, title="Horizon"),
    yaxis=dict(title="ρ", range=[-1.1, 1.1], gridcolor=GRID_COLOR),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=30, b=40, l=60, r=20),
    height=320,
)
n_val = int(sub["n_events"].iloc[0]) if not sub.empty else 0
ind   = bool(sub["indicative_only"].iloc[0]) if not sub.empty else True
st.plotly_chart(fig3, use_container_width=True)
st.caption(f"Topic: **{selected}** · n = {n_val} events · {'⚠️ indicative only' if ind else '✅ sufficient n'}")

st.divider()

# ── Full table ────────────────────────────────────────────────────────────────
with st.expander("Full topic table"):
    show = df[["topic", "n_events", rho_col, "indicative_only"]].copy()
    show.columns = ["Topic", "n events", "ρ", "Indicative only"]
    show["ρ"] = show["ρ"].round(3)
    st.dataframe(show, use_container_width=True, height=400)
