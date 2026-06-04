# Earnings Call Audio, Text, and Stock Reaction EDA Results

Generated from the notebook-first EDA workflow in:

- `notebooks/01_data_ingest_dedupe.ipynb`
- `notebooks/02_text_features_eda.ipynb`
- `notebooks/03_stock_reaction_merge_eda.ipynb`
- `notebooks/04_deep_feature_correlation_analysis.ipynb`

Primary final table: `data/processed/earnings_analysis_table.csv`

## Executive Summary

We built an analysis-ready earnings-call dataset combining:

- audio delivery features from earnings-call audio,
- transcript-derived text features from Hugging Face earnings transcripts,
- post-earnings stock reactions measured as percent returns.

The cleaned final merged table contains 167 earnings-call rows and 87 columns. The stock reaction targets are percent movement columns: `return_1d_pct`, `return_5d_pct`, and `return_10d_pct`.

At this exploratory stage, the strongest observed stock-reaction signal is not from a simple linear audio index. Audio stress/confidence correlations with returns are near zero in the current sample. Text features show slightly more signal: higher `negative_term_rate` has a modest negative correlation with 1-day and 5-day returns, while `simple_sentiment_balance` is mildly more related to 5-day and 10-day returns.

These are EDA findings only, not predictive conclusions.

Expanded correlation analysis across all numeric audio/text features reinforces that caveat. No feature survived Benjamini-Hochberg false discovery correction at 10%, but the strongest unadjusted candidates came from text complexity features and MFCC variability rather than the broad audio stress/confidence indices.

## Data Pipeline Results

### Audio Dedupe

Input audio feature table: `data/processed/audio_call_feature_table_comprehensive.csv`

Output deduped table: `data/processed/audio_call_feature_table_deduped.csv`

The dedupe rule was applied only within the dataframe:

1. Build `base_call_id` by removing optional numeric suffixes like `_02`.
2. For each `base_call_id`, keep the row with the maximum `duration_sec`.
3. Save an audit trail to `data/processed/audio_call_duplicate_audit.csv`.

Results:

- Deduped audio rows: 167
- Unique tickers: 41
- Duplicate `base_call_id` values after cleaning: 0
- Duplicate groups audited: 10
- Rows removed: 10

Removed duplicate rows:

| Base call | Removed row | Removed duration sec | Kept row |
|---|---:|---:|---|
| `AAPL_2025_05_01` | `AAPL_2025_05_01_02` | 233.293 | `AAPL_2025_05_01` |
| `AAPL_2025_10_30` | `AAPL_2025_10_30_02` | 1746.949 | `AAPL_2025_10_30` |
| `AMZN_2025_05_01` | `AMZN_2025_05_01` | 1604.317 | `AMZN_2025_05_01_02` |
| `AMZN_2025_10_30` | `AMZN_2025_10_30_02` | 1479.037 | `AMZN_2025_10_30` |
| `GOOG_2025_04_24` | `GOOG_2025_04_24_02` | 1607.965 | `GOOG_2025_04_24` |
| `META_2025_04_30` | `META_2025_04_30_02` | 1730.125 | `META_2025_04_30` |
| `META_2025_10_29` | `META_2025_10_29` | 1992.949 | `META_2025_10_29_02` |
| `MSFT_2025_04_30` | `MSFT_2025_04_30` | 1409.221 | `MSFT_2025_04_30_02` |
| `MSFT_2025_10_29` | `MSFT_2025_10_29_02` | 1672.813 | `MSFT_2025_10_29` |
| `NVDA_2025_11_19` | `NVDA_2025_11_19` | 2102.773 | `NVDA_2025_11_19_02` |

## Text Feature Results

Text source: Hugging Face `TQTfintech/earnings-transcripts`

Outputs:

- Raw transcript cache: `data/processed/earnings_transcripts_hf_raw.csv`
- Text features: `data/processed/earnings_text_features.csv`

Results:

- Transcript rows: 161
- Transcript tickers: 40
- Missing parsed transcript dates: 0
- Ticker/date normalization included mapping `GOOGL` to `GOOG`.

Text features created:

- `word_count`
- `sentence_count`
- `avg_sentence_length`
- `avg_word_length`
- `unique_word_ratio`
- `positive_term_count`
- `negative_term_count`
- `finance_term_count`
- `positive_term_rate`
- `negative_term_rate`
- `finance_term_rate`
- `simple_sentiment_balance`

## Stock Reaction Results

Stock data source: Yahoo chart API via notebook code, no `yfinance` dependency required.

Output: `data/processed/earnings_event_returns.csv`

Results:

- Event rows: 167
- Rows with valid stock data: 167
- Stock match rate: 100%

Stock movement is measured as percent return, not absolute price move:

- `return_1d_pct`
- `return_5d_pct`
- `return_10d_pct`

Summary of post-call percent returns:

| Metric | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| `return_1d_pct` | -0.486 | 7.746 | -20.600 | -0.576 | 35.949 |
| `return_5d_pct` | -0.585 | 9.374 | -20.711 | -1.164 | 27.120 |
| `return_10d_pct` | -0.720 | 10.820 | -27.825 | -0.959 | 39.863 |

The overall distribution is slightly negative on average across all three horizons, with large event-driven tails.

## Final Merge Results

Output: `data/processed/earnings_analysis_table.csv`

Results:

- Final rows: 167
- Final columns: 87
- Audio coverage: 167 / 167
- Stock coverage: 167 / 167
- Text match rate: 49.7%
- Unmatched rows logged in `data/processed/earnings_analysis_unmatched_rows.csv`

The text match rate is below stock/audio coverage because the HF transcript dataset and the HF audio/audio-feature universe do not fully overlap by ticker and call date.

## Biggest Positive 1-Day Reactions

| Event | 1-day % | 5-day % | 10-day % | Text matched |
|---|---:|---:|---:|---|
| `ORCL_2025_09_09` | 35.949 | 26.972 | 29.945 | No |
| `APP_2025_02_12` | 24.019 | 18.324 | -15.731 | Yes |
| `AI_2025_05_28` | 20.765 | 14.075 | 7.081 | No |
| `SNOW_2025_08_27` | 20.265 | 10.734 | 12.047 | Yes |
| `NET_2025_02_06` | 17.764 | 24.717 | 7.377 | Yes |
| `NOW_2025_04_23` | 15.488 | 17.511 | 20.991 | No |
| `ABNB_2025_02_13` | 14.450 | 3.566 | -1.539 | Yes |
| `NET_2025_10_30` | 13.843 | 1.528 | -4.027 | Yes |
| `SNOW_2025_05_21` | 13.432 | 12.947 | 17.296 | No |
| `ORCL_2025_06_11` | 13.312 | 19.554 | 20.660 | No |

## Biggest Negative 1-Day Reactions

| Event | 1-day % | 5-day % | 10-day % | Text matched |
|---|---:|---:|---:|---|
| `VRTX_2025_08_04` | -20.600 | -20.711 | -17.446 | Yes |
| `SMCI_2025_08_05` | -18.285 | -18.914 | -24.485 | No |
| `CMG_2025_10_29` | -18.184 | -19.593 | -21.227 | Yes |
| `AMAT_2025_08_14` | -14.067 | -15.087 | -12.203 | No |
| `CMG_2025_07_23` | -13.338 | -17.090 | -19.117 | Yes |
| `ZS_2025_11_25` | -13.033 | -15.901 | -16.029 | Yes |
| `GTLB_2025_12_02` | -12.774 | -10.030 | -12.151 | No |
| `PLTR_2025_05_05` | -12.047 | -4.290 | 2.068 | No |
| `NOW_2025_01_29` | -11.444 | -10.193 | -14.098 | Yes |
| `AVGO_2025_12_11` | -11.428 | -18.823 | -13.347 | Yes |

## Ticker-Level Observations

Average 1-day post-call percent returns by ticker show large differences, but many tickers only have three or four events, so these should be treated as descriptive EDA rather than stable estimates.

Notable positive averages:

- `APP`: +12.143% average 1-day return across 4 calls
- `ORCL`: +8.832% across 4 calls
- `NET`: +8.605% across 4 calls
- `SNOW`: +7.430% across 3 calls

Notable negative averages:

- `SMCI`: -10.337% average 1-day return across 3 calls
- `GTLB`: -10.241% across 3 calls
- `VRTX`: -8.696% across 4 calls
- `CMG`: -8.120% across 4 calls
- `AMAT`: -6.563% across 4 calls
- `QCOM`: -6.001% across 4 calls

## Correlation EDA

Simple Pearson correlations with stock percent returns:

| Feature | 1-day % | 5-day % | 10-day % |
|---|---:|---:|---:|
| `audio_stress_index` | 0.004 | -0.016 | -0.023 |
| `audio_confidence_index` | -0.001 | 0.013 | -0.010 |
| `audio_instability_index` | -0.007 | -0.019 | -0.023 |
| `vocal_clarity_proxy` | -0.031 | -0.005 | -0.008 |
| `word_count` | 0.107 | 0.016 | 0.010 |
| `negative_term_rate` | -0.156 | -0.121 | -0.035 |
| `positive_term_rate` | -0.050 | 0.051 | 0.097 |
| `finance_term_rate` | 0.013 | 0.044 | 0.031 |
| `simple_sentiment_balance` | 0.008 | 0.092 | 0.106 |

Interpretation:

- The current audio composite indices do not show meaningful linear correlation with short-window returns.
- Text negativity has the clearest simple EDA relationship: higher negative-term rate is associated with lower 1-day and 5-day percent returns.
- Positive text balance is weakly more aligned with 5-day and 10-day percent returns than with 1-day returns.
- These are univariate/correlation observations only; they do not control for ticker, sector, market movement, earnings surprise, or company-specific baselines.

## Deep Correlation Analysis

Expanded notebook: `notebooks/04_deep_feature_correlation_analysis.ipynb`

Generated artifacts:

- `data/processed/deep_feature_return_correlations.csv`
- `data/processed/deep_numeric_feature_correlation_matrix.csv`
- `data/processed/deep_audio_text_correlation_pairs.csv`
- `data/processed/deep_feature_group_summary.csv`

This notebook analyzes research-motivated audio families, text features, and stock percent-return outcomes with both Pearson and Spearman correlations. Spearman is emphasized because earnings reactions are heavy-tailed and event-driven.

Audio feature families tested:

- prosody/pitch: `pitch_mean`, `pitch_std`
- energy/loudness: `energy_mean`, `energy_std`
- voice activity: `voiced_ratio`
- zero-crossing: `zcr_mean`, `zcr_std`
- spectral shape: `spectral_centroid_*`, `spectral_bandwidth_*`
- MFCC means: `mfcc_1_mean` through `mfcc_13_mean`
- MFCC variability: `mfcc_1_std` through `mfcc_13_std`
- composite scores: `audio_stress_index`, `audio_confidence_index`, `audio_instability_index`, `vocal_clarity_proxy`

Text features tested:

- length/complexity: `transcript_chars`, `word_count`, `sentence_count`, `avg_sentence_length`, `avg_word_length`, `unique_word_ratio`
- dictionary counts/rates: positive, negative, and finance term counts/rates
- `simple_sentiment_balance`

### Strongest Feature-to-Return Relationships

Top unadjusted Spearman relationships across all available features:

| Outcome | Feature group | Feature | n | Spearman rho | Spearman p | FDR q |
|---|---|---|---:|---:|---:|---:|
| `return_10d_pct` | text | `avg_sentence_length` | 83 | -0.202 | 0.067 | 0.991 |
| `return_1d_pct` | text | `unique_word_ratio` | 83 | -0.198 | 0.073 | 0.991 |
| `return_5d_pct` | text | `avg_word_length` | 83 | -0.184 | 0.095 | 0.991 |
| `return_1d_pct` | text | `word_count` | 83 | 0.176 | 0.112 | 0.991 |
| `return_1d_pct` | text | `avg_word_length` | 83 | -0.175 | 0.114 | 0.991 |
| `return_5d_pct` | MFCC variability | `mfcc_5_std` | 167 | 0.150 | 0.053 | 0.991 |
| `return_1d_pct` | MFCC variability | `mfcc_3_std` | 167 | 0.147 | 0.058 | 0.991 |
| `return_5d_pct` | MFCC variability | `mfcc_3_std` | 167 | 0.147 | 0.059 | 0.991 |
| `return_5d_pct` | MFCC variability | `mfcc_1_std` | 167 | 0.141 | 0.069 | 0.991 |
| `return_10d_pct` | MFCC variability | `mfcc_13_std` | 167 | 0.139 | 0.073 | 0.991 |

Interpretation:

- The largest observed relationships are still modest: absolute Spearman rho is roughly 0.14 to 0.20.
- Text style and complexity features ranked highest, but those use only the 83 rows with matched transcripts.
- MFCC variability features are the strongest audio-family candidates because they use all 167 event rows and appear repeatedly across horizons.
- No feature survives FDR correction, so these are hypothesis-generating candidates for later modeling, not final evidence.

### Research-Motivated Audio Results

Top audio-only unadjusted Spearman relationships:

| Outcome | Audio group | Feature | n | Spearman rho | Pearson r |
|---|---|---|---:|---:|---:|
| `return_5d_pct` | MFCC variability | `mfcc_5_std` | 167 | 0.150 | 0.145 |
| `return_1d_pct` | MFCC variability | `mfcc_3_std` | 167 | 0.147 | 0.134 |
| `return_5d_pct` | MFCC variability | `mfcc_3_std` | 167 | 0.147 | 0.088 |
| `return_5d_pct` | MFCC variability | `mfcc_1_std` | 167 | 0.141 | 0.086 |
| `return_10d_pct` | MFCC variability | `mfcc_13_std` | 167 | 0.139 | 0.096 |
| `return_10d_pct` | MFCC mean | `mfcc_11_mean` | 167 | 0.130 | 0.105 |
| `return_1d_pct` | MFCC variability | `mfcc_1_std` | 167 | 0.130 | 0.115 |
| `return_1d_pct` | MFCC variability | `mfcc_5_std` | 167 | 0.128 | 0.132 |
| `return_1d_pct` | MFCC mean | `mfcc_11_mean` | 167 | 0.118 | 0.070 |
| `return_10d_pct` | MFCC variability | `mfcc_10_std` | 167 | 0.115 | 0.072 |

Audio interpretation:

- MFCC variability is the only audio family that repeatedly appears near the top.
- Pitch and voice activity are weaker in this sample. The strongest pitch candidate was `pitch_std` vs `return_1d_pct`, with Spearman rho around -0.086.
- The broad composite indices (`audio_stress_index`, `audio_confidence_index`, `audio_instability_index`, `vocal_clarity_proxy`) remain weak.
- If audio signal exists here, it may be in lower-level timbre/voice-quality descriptors rather than the first-pass stress/confidence proxy formulas.

### Feature Family Summary

Maximum absolute Spearman correlation by family:

| Outcome | Strongest group | Max abs Spearman | Median abs Spearman | FDR hits |
|---|---|---:|---:|---:|
| `return_1d_pct` | text features | 0.198 | 0.079 | 0 |
| `return_1d_pct` | MFCC variability | 0.147 | 0.041 | 0 |
| `return_5d_pct` | text features | 0.184 | 0.027 | 0 |
| `return_5d_pct` | MFCC variability | 0.150 | 0.014 | 0 |
| `return_10d_pct` | text features | 0.202 | 0.057 | 0 |
| `return_10d_pct` | MFCC variability | 0.139 | 0.050 | 0 |

The group-level pattern is consistent: text features produce the largest unadjusted associations, MFCC variability is the strongest audio feature family, broad composite audio scores are weaker, and no group has statistically reliable FDR-adjusted evidence yet.

### Audio-Text Relationships

The deep notebook also measures audio-text correlations on matched transcript rows to see whether vocal delivery and language content capture overlapping constructs.

Strongest audio-text relationships:

| Audio group | Audio feature | Text feature | n | Spearman rho | FDR q |
|---|---|---|---:|---:|---:|
| MFCC mean | `mfcc_10_mean` | `finance_term_count` | 83 | -0.421 | 0.032 |
| MFCC mean | `mfcc_10_mean` | `simple_sentiment_balance` | 83 | -0.397 | 0.032 |
| MFCC mean | `mfcc_10_mean` | `positive_term_rate` | 83 | -0.395 | 0.032 |
| MFCC mean | `mfcc_1_mean` | `finance_term_count` | 83 | 0.387 | 0.032 |
| MFCC mean | `mfcc_10_mean` | `finance_term_rate` | 83 | -0.386 | 0.032 |
| MFCC mean | `mfcc_1_mean` | `finance_term_rate` | 83 | 0.383 | 0.032 |
| MFCC variability | `mfcc_4_std` | `finance_term_count` | 83 | -0.353 | 0.072 |
| MFCC mean | `mfcc_10_mean` | `positive_term_count` | 83 | -0.348 | 0.072 |
| MFCC variability | `mfcc_12_std` | `avg_word_length` | 83 | 0.347 | 0.072 |
| Energy | `energy_mean` | `finance_term_rate` | 83 | 0.346 | 0.072 |

Audio-text interpretation:

- Audio-text correlations are stronger than audio-return correlations.
- MFCC descriptors are associated with finance-term density and simple sentiment balance.
- `energy_mean` is positively related to finance-term rate, suggesting that more energetic calls may also contain denser finance-specific language.
- `pitch_std` has a moderate positive relationship with `negative_term_rate` (Spearman rho about 0.320, FDR q about 0.101), which is directionally consistent with stress/negative-content intuition but just outside the 10% FDR threshold.

### Modeling Implications

Based on the deeper notebook:

1. Do not rely only on `audio_stress_index`, `audio_confidence_index`, `audio_instability_index`, and `vocal_clarity_proxy`.
2. Keep lower-level MFCC variability features in the candidate set.
3. Treat text complexity and tone features as important model candidates once transcript coverage improves.
4. Use rank-based checks or robust models because percent-return outcomes have large event-driven tails.
5. Reduce feature redundancy before modeling using regularization, PCA, or family-level feature selection.

## Limitations

- Text coverage is incomplete: only 49.7% of audio/stock rows matched a transcript by ticker and date.
- The text sentiment features are lightweight dictionary-based EDA features, not a domain-trained financial NLP model.
- Audio features are call-level aggregate features; the faster table does not include segment-rollup dynamics.
- Stock reactions are raw percent returns, not market-adjusted or sector-adjusted abnormal returns.
- No predictive modeling or train/test validation has been done yet.

## Recommended Next Steps

1. Improve transcript coverage by adding Q&A transcript JSONs or additional transcript sources.
2. Add market-adjusted returns using SPY/QQQ benchmarks for the same windows.
3. Add ticker fixed effects or within-ticker z-scores so large-company baselines do not dominate.
4. Add segment-rollup audio features for tone changes through the call.
5. Move from simple correlations to a controlled modeling notebook with train/test splits.

