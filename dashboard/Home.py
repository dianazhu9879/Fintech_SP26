import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "outputs" / "combined"

st.set_page_config(
    page_title="ECT Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #111; }
.metric-card {
    background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px;
    padding: 18px 22px; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-lbl { font-size: 0.72rem; color: #9ca3af; margin-top: 5px;
               text-transform: uppercase; letter-spacing: 0.06em; }
.finding-card {
    background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px;
    padding: 18px 20px; height: 100%;
}
.finding-card h4 { font-size: 0.95rem; font-weight: 600; margin-bottom: 6px; }
.finding-card p  { font-size: 0.82rem; color: #9ca3af; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

PLOT_BG = "#111111"
PAPER_BG = "#111111"
FONT_COLOR = "#e5e7eb"
GRID_COLOR = "#2a2a2a"
HORIZONS = [1, 3, 5, 7, 10, 21]
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]
TICKER_COLORS = {
    "NVDA": "#76b900", "AAPL": "#a2aaad", "MSFT": "#00a4ef",
    "GOOGL": "#fbbc04", "META": "#0082fb", "AMZN": "#ff9900", "TSLA": "#cc0000",
}

def plot_layout(title="", height=360, **kwargs):
    return dict(
        title=dict(text=title, font=dict(size=14, color=FONT_COLOR)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, **kwargs.get("xaxis", {})),
        yaxis=dict(gridcolor=GRID_COLOR, **kwargs.get("yaxis", {})),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=40, l=60, r=20),
        height=height,
    )

@st.cache_data
def load_combined():
    return dict(
        ec=pd.read_csv(COMBINED / "event_level_correlations.csv"),
        pc=pd.read_csv(COMBINED / "per_company_correlations.csv"),
        tc=pd.read_csv(COMBINED / "topic_horizon_correlations.csv"),
        panel=pd.read_csv(COMBINED / "panel.csv"),
    )

data = load_combined()
ec, pc, tc, panel = data["ec"], data["pc"], data["tc"], data["panel"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📈 Earnings Call Sentiment & Stock Returns")
st.markdown(
    "<span style='color:#9ca3af;font-size:0.9rem'>"
    "LLM topic labeling (gpt-4o-mini) · FinBERT · 7 large-cap tech companies · 2024–2026"
    "</span>",
    unsafe_allow_html=True,
)
st.divider()

# ── Stat cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl, color in [
    (c1, "37",    "Earnings events",               "#60a5fa"),
    (c2, "7",     "Companies",                     "#60a5fa"),
    (c3, str(tc["topic"].nunique()), "Topics discovered", "#60a5fa"),
    (c4, "6",     "Return horizons (1–21d)",        "#60a5fa"),
    (c5, "−0.31", "Strongest signal ρ (risk→21d)", "#f87171"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-val" style="color:{color}">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Key findings ──────────────────────────────────────────────────────────────
st.markdown("#### Key Findings")
cols = st.columns(3)
findings = [
    ("📉", "Risk language predicts weakness",
     "Higher risk-intensity talk reliably predicts below-benchmark returns at 10–21 days (ρ = −0.31). The strongest and most consistent whole-call signal."),
    ("⏱️", "Sentiment effect is delayed",
     "Overall sentiment correlation is near zero at 1–3 days, then strengthens to ρ ≈ +0.13 at 10–21 days. Markets re-process qualitative content slowly."),
    ("🎯", "GOOGL is the most efficient",
     "Google's earnings sentiment correlates with returns at every horizon (ρ = 0.96 at 1d, 0.85 at 5d) — the most information-rich calls in this panel."),
    ("🤖", "Specific AI beats broad AI",
     "Concrete AI product sentiment (AI Infrastructure ρ = +0.85, Advertising ρ = +0.78) predicts positive returns. Vague 'AI Initiatives' predicts negative (ρ = −0.99)."),
    ("🍎", "AAPL is the outlier",
     "Apple shows a consistent inverse signal (ρ = −0.77 at 5d). Likely pre-priced before calls — earnings sentiment adds little incremental value."),
    ("⚠️", "Small sample — indicative only",
     "4–7 events per ticker is too few for statistical conclusions. All per-company and per-topic results are hypothesis-generating, not definitive."),
]
for i, (emoji, title, body) in enumerate(findings):
    with cols[i % 3]:
        st.markdown(
            f'<div class="finding-card"><div style="font-size:1.3rem;margin-bottom:8px">{emoji}</div>'
            f'<h4>{title}</h4><p>{body}</p></div><br>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Company heatmap ───────────────────────────────────────────────────────────
st.markdown("#### Sentiment → Excess Return Correlation by Company & Horizon")

pivot = (pc.pivot(index="ticker", columns="horizon_days",
                  values="rho_avg_sentiment_vs_excess_return")
           .reindex(TICKERS).reindex(columns=HORIZONS))

fig = go.Figure(go.Heatmap(
    z=pivot.values.round(2).tolist(),
    x=[f"{h}d" for h in HORIZONS],
    y=TICKERS,
    text=[[f"{v:.2f}" if v == v else "—" for v in row] for row in pivot.values.tolist()],
    texttemplate="%{text}", textfont=dict(size=13),
    colorscale=[
        [0.0, "#b91c1c"], [0.35, "#ef4444"],
        [0.47, "#1a1a1a"], [0.53, "#1a1a1a"],
        [0.65, "#22c55e"], [1.0, "#15803d"],
    ],
    zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title="ρ", tickfont=dict(color=FONT_COLOR),
                  titlefont=dict(color=FONT_COLOR)),
))
layout = plot_layout(height=320)
layout.pop("xaxis"); layout.pop("yaxis")
fig.update_layout(**layout,
    xaxis=dict(gridcolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR),
)
st.plotly_chart(fig, use_container_width=True)

# ── Signal lines ──────────────────────────────────────────────────────────────
st.markdown("#### Whole-Call Tone vs Excess Return (Pearson ρ across all 37 events)")
metric_cfg = {
    "avg_llm_sentiment_score":      ("Sentiment",         "#4ade80"),
    "avg_llm_risk_intensity_score": ("Risk Intensity",    "#f87171"),
    "avg_llm_uncertainty_score":    ("Uncertainty",       "#facc15"),
    "pct_negative_or_mixed_turns":  ("% Neg/Mixed Turns", "#c084fc"),
}
fig2 = go.Figure()
for metric, (label, color) in metric_cfg.items():
    sub = ec[ec["metric"] == metric].sort_values("horizon_days")
    fig2.add_trace(go.Scatter(
        x=sub["horizon_days"], y=sub["rho"].round(3),
        mode="lines+markers", name=label,
        line=dict(color=color, width=2.5), marker=dict(size=7),
    ))
fig2.add_hline(y=0, line_dash="dot", line_color="#555")
fig2.update_layout(**plot_layout(height=360),
    xaxis=dict(tickvals=HORIZONS, ticktext=[f"{h}d" for h in HORIZONS], gridcolor=GRID_COLOR),
    yaxis=dict(title="ρ", range=[-0.45, 0.45], gridcolor=GRID_COLOR),
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "⚠️ All correlations are Pearson (n=37 events). "
    "Per-company and per-topic results carry small-sample caveats — see individual pages."
)
