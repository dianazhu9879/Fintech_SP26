# Earnings Signal Dashboard Static Export

Open `index.html` to view the hardcoded dashboard snapshot.

The dashboard HTML includes the rendered data it needs to display, including the candlestick windows. The `data/` folder keeps the generated result artifacts alongside it for inspection or reuse:

- `data/signals/` - mechanical signals, comparison rows, backtest, realized move profiles, and OHLCV windows
- `data/ml/` - ML scores, predictions, forward evaluations, and trained model metadata
- `data/tables/` - final enriched source tables used for the displayed results
- `data/metadata.json` and `data/summary.json` - project/result metadata

This folder is meant for display/export only. It does not include the scripts, tests, fetched HTML cache, or full pipeline needed to regenerate the data.
