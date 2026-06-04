# Fintech_SP26 — Earnings Call Signal Research

**TQT · Spring 2026**

Exploratory research into whether public earnings-call information — audio delivery, transcript language, and LLM-derived labels — contains structured signal for post-earnings stock moves.

> This is a disciplined EDA project. The goal is hypothesis generation and candidate feature discovery, not production alpha.

---

## Research Question

Can public earnings-call data (audio + transcript) contain tradable signal for post-earnings stock returns?

We test *correlation first, not causality*, across three modalities:

| Modality | Source | Examples |
|---|---|---|
| Audio delivery | Scraped / HuggingFace audio | MFCC variability, pitch, pause rate, speech rate |
| Transcript text | HF `TQTfintech/earnings-transcripts` | Negativity rate, complexity, FinBERT sentiment |
| Stock reaction | Yahoo Finance chart API | 1d / 5d / 10d / 21d percent returns |

---

## Two Research Tracks

### Broad Multimodal Track
- **167 events**, 41 tickers, 87 merged columns
- Return targets: `return_1d_pct`, `return_5d_pct`, `return_10d_pct`
- 100% audio + stock coverage; ~49.7% transcript match rate
- Main artifacts: `notebooks/01–04`, `data/processed/earnings_analysis_table.csv`

### 7-Company Panel Track
- **37 events** across AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
- LLM / FinBERT topic-tone labels
- 6 excess-return horizons (1d → 21d)
- Main artifacts: `dashboard/`, `data/text_alpha/outputs/`

---

## Key Findings

- **No feature survives FDR correction** at 10% — all results are EDA-level hypotheses.
- **Text complexity** (avg sentence/word length, unique word ratio) shows the strongest unadjusted Spearman relationships (ρ ≈ 0.18–0.20).
- **MFCC variability** is the strongest audio candidate family (ρ ≈ 0.14–0.15), consistently appearing across return horizons.
- **Risk tone** (panel track) shows the most interesting whole-panel result: risk-intensity vs 21-day excess return ρ ≈ −0.31.
- Broad composite audio indices (stress, confidence, instability) are near zero vs returns.
- Audio–text cross-modal correlations are stronger than audio–return correlations (FDR-significant MFCC↔finance-term relationships exist).

---

## Repo Structure

```
Fintech_SP26/
├── notebooks/                         # Main EDA workflow (run in order)
│   ├── 01_data_ingest_dedupe.ipynb
│   ├── 02_text_features_eda.ipynb
│   ├── 03_stock_reaction_merge_eda.ipynb
│   └── 04_deep_feature_correlation_analysis.ipynb
│
├── src/
│   ├── audio_alpha/                   # Audio feature extraction pipeline
│   │   ├── run_pipeline.py            # Entry point
│   │   ├── import_hf_audio.py
│   │   ├── preprocess_audio.py
│   │   ├── segment_audio.py
│   │   ├── extract_features.py
│   │   ├── aggregate_call_features.py
│   │   ├── build_call_feature_table.py
│   │   ├── build_manifest.py
│   │   ├── score_audio_sentiment.py
│   │   ├── normalize.py (legacy)
│   │   └── config.py
│   └── text_alpha/
│       └── scraper.py                 # Transcript collection
│
├── dashboard/                         # Streamlit panel dashboard
│   ├── Home.py
│   └── pages/
│       ├── 1_Topic_Analysis.py
│       ├── 2_Company_Deep_Dive.py
│       └── 3_Event_Explorer.py
│
├── dashboard-static-export/           # Static HTML export of dashboard data
│
├── data/
│   ├── raw/audio/                     # Raw downloaded audio files
│   ├── interim/                       # Intermediate pipeline artifacts
│   ├── processed/                     # Final analysis-ready tables
│   ├── audio_alpha/                   # Per-call features, normalized tables, plots
│   └── text_alpha/outputs/            # LLM / FinBERT label outputs
│
├── reports/
│   └── final_results_summary.md       # Full EDA results with tables
│
├── tests/audio_alpha/                 # pytest unit tests
├── requirements.txt
└── pyproject.toml
```

---

## Setup

**Requirements:** Python ≥ 3.10

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the audio_alpha package in editable mode
pip install -e .

# Optional: opensmile features
pip install -e ".[standard]"
```

Key dependencies: `pandas`, `numpy`, `librosa`, `soundfile`, `scikit-learn`, `tqdm`, `whisperx`, `sentence-transformers`, `beautifulsoup4`, `selenium`

---

## Running the Pipeline

### Notebooks (Broad Track EDA)

Run in order from the repo root:

```bash
jupyter notebook notebooks/
```

1. `01_data_ingest_dedupe.ipynb` — load audio features, deduplicate calls, cache transcripts
2. `02_text_features_eda.ipynb` — extract dictionary-based text features from transcripts
3. `03_stock_reaction_merge_eda.ipynb` — fetch Yahoo Finance returns, merge into event table
4. `04_deep_feature_correlation_analysis.ipynb` — Pearson/Spearman correlation analysis with FDR correction

### Audio Pipeline

```bash
python -m audio_alpha.run_pipeline
```

### Streamlit Dashboard (Panel Track)

```bash
streamlit run dashboard/Home.py
```

Filters: ticker · topic · horizon. Pages: Topic Analysis, Company Deep Dive, Event Explorer.

### Tests

```bash
pytest
```

---

## Data Sources

| Source | What it provides |
|---|---|
| HuggingFace `TQTfintech/earnings-transcripts` | Earnings call transcripts (161 rows, 40 tickers) |
| HuggingFace audio dataset | Call audio for feature extraction |
| Yahoo Finance chart API | Post-call stock price data (no `yfinance` dependency) |

---

## Signal Candidates (for future modeling)

Based on EDA results, prioritized candidate features for a controlled modeling step:

1. **MFCC variability** — `mfcc_1_std` through `mfcc_13_std` (full 167-row coverage)
2. **Text complexity** — `avg_sentence_length`, `avg_word_length`, `unique_word_ratio` (83-row subset)
3. **Text negativity** — `negative_term_rate` (modest negative correlation with 1d/5d returns)
4. **Risk tone** — panel-track LLM risk-intensity label (strongest panel candidate, ρ ≈ −0.31 vs 21d)

---

## Caveats

- All correlation results are **unadjusted** and none survive FDR correction — hypothesis generation only.
- Text coverage is 49.7%; transcript matching by ticker/date is incomplete.
- Stock targets are **raw percent returns**, not market- or sector-adjusted abnormal returns.
- No predictive modeling or train/test validation has been done.
- Small sample sizes per company/topic make sliced results statistically fragile.

---

## Next Steps

- [ ] Improve transcript coverage (Q&A JSON sources, additional transcript datasets)
- [ ] Add market-adjusted returns (SPY/QQQ benchmarks for same windows)
- [ ] Add ticker fixed effects / within-ticker z-scores
- [ ] Segment-rollup audio features (tone dynamics through the call)
- [ ] Controlled modeling notebook with train/test splits, effect sizes, and confidence intervals

---

## Reports

Full results with tables: [`reports/final_results_summary.md`](reports/final_results_summary.md)

---

## Team

**Members:** Syrena, Peter, David, Prasham, Jaden

**Mentors:** Alexa, Antara, Diana, Jayani, Rudy
