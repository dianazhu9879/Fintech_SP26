# Earnings transcript sentiment + forward-return pipeline

Two files:

- `earnings_pipeline.py` — the production pipeline
- `test_pipeline.py` — verification harness (parser + return math + sentiment chunking)

## Run it

```bash
pip install transformers torch yfinance pandas

python earnings_pipeline.py \
  --input  ./earnings_transcripts \
  --output ./earnings_sentiment_returns.csv
```

Output CSV columns: `ticker, call_date, quarter, prepared_sentiment, prepared_score, qa_sentiment, qa_score, ret_1d, ret_3d, ret_5d`

## What's inside

**Parser.** Handles the two Motley Fool layouts seen in the corpus — a newer format with a `Full Conference Call Transcript` header and `Operator:` Q&A delimiter, and an older format with `Prepared Remarks:` and `Questions & Answers:` headers. It also handles TSLA's webcast variant where Q&A is opened by phrasing like *"Now we'll go to investor questions"*. The Q&A boundary is found via a prioritised list of regex cues — `Operator:`, *"open … for questions"*, *"first question comes from"*, etc. — and the earliest match wins. Date is read from line 9 when present, falling back to a date-shaped substring search in the first 30 lines. Quarter is taken from the filename (`AAPL_Q1_2026.txt` → `Q1 2026`).

**Sentiment.** FinBERT (`ProsusAI/finbert`) loaded once, run on each section. Long inputs are tokenised once and chunked into ≤510-token windows (room for `[CLS]` and `[SEP]`); chunk softmax probabilities are averaged before argmax so the label and confidence reflect the whole section, not just its first 512 tokens.

**Prices.** `yfinance` `auto_adjust=True` 5-year history per ticker, cached. `t` is anchored at the first trading day on or after the call date (so an after-hours call on a trading day uses that day's close; a Friday-night call rolls to Monday). Forward returns are `close[t+N]/close[t] − 1` for N in {1, 3, 5}. Missing forward bars come back as `None` rather than zero.

**Validation.** At the end of a run the script prints: total transcripts parsed, any with empty splits, any with missing forward returns, and the sentiment distributions across both sections.

## Verification

`test_pipeline.py` covers everything that doesn't require live network:

- All 35 transcripts in the corpus parse cleanly (ticker, date, quarter, prepared, Q&A).
- Forward-return arithmetic against a hand-crafted price fixture: `+1.50%` / `+5.00%` / `+8.00%`.
- Weekend-call anchor (Saturday call → Monday close as `t`).
- Missing forward bars return `None`.
- `score_section` chunks long input correctly across the 510-token boundary and averages probabilities.
- Empty input returns `("neutral", 0.0)`.
- Full `build_records` orchestration with mocked yfinance + mocked FinBERT yields a complete 35-row DataFrame.

```
python test_pipeline.py
# → 7 / 7 tests passed
```

## Graceful degradation

If `transformers`/`torch` aren't installed or `huggingface.co` is unreachable, sentiment columns are left empty and a warning is printed. Same for yfinance — if Yahoo is unreachable or rate-limits, return columns come back empty for the affected ticker. The CSV still writes either way.
