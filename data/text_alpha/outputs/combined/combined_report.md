# Cross-company Earnings Call Sentiment Panel

**Events:** 37  |  **Tickers:** 7

## What's in this panel

Per-event topic + tone scores from the LLM labeler, joined with
FinBERT scores and post-earnings excess returns at six trading-day
horizons (1, 3, 5, 7, 10, 21d). Excess returns are computed against
the per-ticker benchmark configured in `data/config/benchmarks.json`.

## Reading guide

- Correlations with `n` >= 8 start to mean something at this sample size,
  but no single ticker has that many events on its own — these are panel
  correlations across all tickers.
- Topic correlations with `n < 6` are flagged as
  `indicative_only` — treat them as hypothesis-generating only.
- The 21d horizon overlaps with many non-earnings news events;
  treat it as context, not as a causal earnings-call effect.

## 1) Event-level (whole-call) correlations

Whole-call average tone metrics vs excess returns across the full panel.

```
                      metric  horizon_days  n_events    rho
     avg_llm_sentiment_score             1        37 -0.010
     avg_llm_sentiment_score             3        37 -0.072
     avg_llm_sentiment_score             5        37  0.043
     avg_llm_sentiment_score             7        37  0.113
     avg_llm_sentiment_score            10        37  0.131
     avg_llm_sentiment_score            21        37  0.129
avg_llm_risk_intensity_score             1        37  0.090
avg_llm_risk_intensity_score             3        37  0.040
avg_llm_risk_intensity_score             5        37 -0.041
avg_llm_risk_intensity_score             7        37 -0.057
avg_llm_risk_intensity_score            10        37 -0.164
avg_llm_risk_intensity_score            21        37 -0.307
   avg_llm_uncertainty_score             1        37  0.162
   avg_llm_uncertainty_score             3        37  0.180
   avg_llm_uncertainty_score             5        37  0.062
   avg_llm_uncertainty_score             7        37  0.011
   avg_llm_uncertainty_score            10        37 -0.030
   avg_llm_uncertainty_score            21        37 -0.203
 pct_negative_or_mixed_turns             1        37 -0.057
 pct_negative_or_mixed_turns             3        37  0.001
 pct_negative_or_mixed_turns             5        37  0.009
 pct_negative_or_mixed_turns             7        37 -0.061
 pct_negative_or_mixed_turns            10        37 -0.040
 pct_negative_or_mixed_turns            21        37 -0.129
```

## 2) Per-topic correlations across the panel

Topics shown sorted by |ρ| at the 5-day horizon.

```
  iPad Innovations                           ρ=-1.000  n=2  (indicative)
  Sovereign AI Initiatives                   ρ=+1.000  n=2  (indicative)
  Enterprise AI Adoption                     ρ=+1.000  n=2  (indicative)
  Delivery Speed and Logistics               ρ=-1.000  n=2  (indicative)
  Regulatory and Compliance Efforts          ρ=+1.000  n=2  (indicative)
  4680 Battery Cell Production               ρ=+1.000  n=3  (indicative)
  User Growth Metrics                        ρ=-0.999  n=4  (indicative)
  Google Cloud Expansion                     ρ=+0.999  n=3  (indicative)
  AI and Machine Learning in Products        ρ=-0.986  n=5  (indicative)
  Azure Data Center Expansion                ρ=-0.961  n=3  (indicative)
  Cost Management and Efficiency Initiatives ρ=-0.855  n=3  (indicative)
  AI Infrastructure                          ρ=+0.853  n=4  (indicative)
  Advertising Performance                    ρ=+0.778  n=5  (indicative)
  Cost Management and Efficiency             ρ=+0.764  n=4  (indicative)
  Microsoft 365 Copilot Adoption             ρ=-0.754  n=4  (indicative)
  Consumer Electronics Trends                ρ=-0.739  n=3  (indicative)
  iPhone Lineup                              ρ=+0.697  n=5  (indicative)
  YouTube Growth                             ρ=+0.687  n=4  (indicative)
  Regulatory Challenges                      ρ=+0.650  n=5  (indicative)
  Hopper Architecture                        ρ=+0.649  n=3  (indicative)
  Search Innovations                         ρ=-0.637  n=4  (indicative)
  Apple Intelligence                         ρ=-0.628  n=5  (indicative)
  AI Integration and Development             ρ=+0.545  n=5  (indicative)
  Services Growth                            ρ=-0.509  n=5  (indicative)
  Mac Performance                            ρ=-0.500  n=3  (indicative)
```

## 3) Per-company correlations (n is small for each)

```
horizon_days     1      3      5      7      10     21
ticker                                                
AAPL         -0.087 -0.404 -0.774 -0.770 -0.895 -0.825
AMZN          0.548  0.271  0.426  0.707  0.284 -0.298
GOOGL         0.960  0.976  0.847  0.914  0.868  0.857
META          0.230  0.241  0.817  0.693  0.715  0.634
MSFT          0.436  0.494  0.345  0.104 -0.803  0.040
NVDA         -0.082  0.047  0.166  0.304  0.395  0.047
TSLA         -0.092  0.143  0.305  0.366  0.612  0.555
```

## Caveats

- All correlations are Pearson; non-linear effects won't show up.
- Each event is treated as independent; we are not adjusting for
  sector or time clustering.
- The LLM and FinBERT both make systematic mistakes on sarcasm,
  hedging, and forward-looking language.
- Excess returns are computed against per-ticker sector ETFs
  (SOXX for semis, QQQ otherwise) — choice of benchmark moves
  the numbers.
