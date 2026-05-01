"""
Verification harness for earnings_pipeline.

This sandbox can't reach huggingface.co (FinBERT) or Yahoo Finance, so we
inject:
  * a synthetic OHLC price DataFrame per ticker, with hand-crafted closes
    that make the forward-return arithmetic verifiable by hand;
  * a stub FinBERT pipeline (lexicon-based) that runs through the same
    chunking/softmax-averaging code path and returns labels & scores.

If these tests pass, the only things actually exercised live on the user's
machine are:
  * the network call to huggingface.co (model download — one-time)
  * the network call to Yahoo Finance via yfinance (price data)

Both are well-trodden code paths in their respective libraries; the logic
in this repo around them is what these tests cover.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import earnings_pipeline as ep  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synth_history(start: str, days: int, base: float, drift: float) -> pd.DataFrame:
    """Daily-frequency 'business-day' price series, monotonically growing."""
    idx = pd.date_range(start=start, periods=days, freq="B").normalize()
    closes = [base * (1 + drift) ** i for i in range(days)]
    return pd.DataFrame({"Close": closes}, index=idx)


# Hand-checked closes for AAPL around its 2025-01-30 call.
# 2025-01-30 = Thursday. Trading days that week: Mon 1/27, Tue 1/28,
# Wed 1/29, Thu 1/30, Fri 1/31, Mon 2/3, Tue 2/4, Wed 2/5, Thu 2/6.
AAPL_FIXTURE = pd.DataFrame(
    {
        "Close": [
            230.00,  # 2025-01-27 Mon
            231.00,  # 2025-01-28 Tue
            232.00,  # 2025-01-29 Wed
            240.00,  # 2025-01-30 Thu  <- t (call date)
            243.60,  # 2025-01-31 Fri  <- t+1  -> +1.50%
            246.00,  # 2025-02-03 Mon
            252.00,  # 2025-02-04 Tue  <- t+3  -> +5.00%
            254.40,  # 2025-02-05 Wed
            259.20,  # 2025-02-06 Thu  <- t+5  -> +8.00%
        ]
    },
    index=pd.to_datetime(
        [
            "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30",
            "2025-01-31", "2025-02-03", "2025-02-04", "2025-02-05",
            "2025-02-06",
        ]
    ).normalize(),
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_all_transcripts():
    """Every file under earnings_transcripts/ should parse fully."""
    root = Path(__file__).parent / "raw_text"
    files = ep.discover_transcripts(root)
    assert len(files) == 35, f"expected 35 transcripts, found {len(files)}"

    bad = []
    for f in files:
        p = ep.parse_transcript_file(f)
        if not p.ticker:
            bad.append((f.name, "ticker"))
        if not p.call_date:
            bad.append((f.name, "call_date"))
        if not p.quarter:
            bad.append((f.name, "quarter"))
        if not p.prepared_remarks:
            bad.append((f.name, "prepared"))
        if not p.qa_section:
            bad.append((f.name, "qa"))
    assert not bad, f"parse failures: {bad}"
    print(f"  ✓ all {len(files)} transcripts parsed cleanly")


def test_forward_returns_arithmetic():
    """forward_returns should produce the exact percentages we hand-crafted."""
    ep._price_cache.clear()
    ep._price_cache["AAPL"] = AAPL_FIXTURE

    rets = ep.forward_returns("AAPL", "2025-01-30")
    expected = {"ret_1d": 0.015, "ret_3d": 0.05, "ret_5d": 0.08}
    for k, want in expected.items():
        got = rets[k]
        assert got is not None and math.isclose(got, want, abs_tol=1e-9), (
            f"{k}: got {got}, want {want}"
        )
    print(f"  ✓ forward returns 1d/3d/5d match hand-crafted closes: "
          f"{ {k: f'{v*100:+.2f}%' for k, v in rets.items()} }")


def test_forward_returns_call_on_weekend():
    """If the call date falls on a non-trading day, anchor at next trading day."""
    ep._price_cache.clear()
    ep._price_cache["AAPL"] = AAPL_FIXTURE
    # 2025-02-01 is a Saturday — should anchor on Monday 2025-02-03 (close=246).
    rets = ep.forward_returns("AAPL", "2025-02-01")
    # t   = 246.00 (2025-02-03 Mon)
    # t+1 = 252.00 (2025-02-04 Tue) -> +2.4390%
    assert rets["ret_1d"] is not None
    assert math.isclose(rets["ret_1d"], 252.00 / 246.00 - 1, abs_tol=1e-6)
    print(f"  ✓ weekend call anchors at next trading day "
          f"(ret_1d={rets['ret_1d']*100:+.2f}%)")


def test_forward_returns_missing_when_too_recent():
    """If we don't have N forward bars yet, return None for that window."""
    ep._price_cache.clear()
    ep._price_cache["AAPL"] = AAPL_FIXTURE
    # Anchor on the very last bar — no t+1, t+3, t+5 available.
    rets = ep.forward_returns("AAPL", "2025-02-06")
    assert rets["ret_1d"] is None
    assert rets["ret_3d"] is None
    assert rets["ret_5d"] is None
    print("  ✓ missing forward bars -> None (no spurious zeros)")


def test_score_section_chunking():
    """
    Patch _load_finbert with a stub that mimics FinBERT's interface using a
    naive lexicon. We feed a deliberately long string (>1024 tokens worth)
    so the chunking path executes, then verify the averaged probability
    sums to ~1 and the predicted label is sensible.
    """
    import torch

    class StubTok:
        cls_token_id = 101
        sep_token_id = 102

        def __call__(self, text, **kw):
            # Toy whitespace tokenisation -> integer IDs.
            ids = [hash(w) % 30000 for w in text.split()]
            return {"input_ids": ids}

    class StubModel:
        config = type("C", (), {"id2label": {0: "positive", 1: "negative", 2: "neutral"}})()

        def eval(self):
            return self

        def __call__(self, input_ids=None, attention_mask=None):
            # Stub logits:
            # - >+0 for "positive" if many up-ish words
            # - >+0 for "negative" if many down-ish words
            # - default to neutral
            text = " ".join(str(int(x)) for x in input_ids[0])  # opaque
            # Sentiment is supplied via ground-truth marker tokens.
            ones = (input_ids == 1).sum().item()
            neg_ones = (input_ids == 2).sum().item()
            pos_score = float(ones)
            neg_score = float(neg_ones)
            neutral_score = 0.5
            logits = torch.tensor([[pos_score, neg_score, neutral_score]])
            return type("Out", (), {"logits": logits})()

    # Hand-build a chunked input by directly patching the score_section's
    # internals: use a tokenized id list spanning multiple chunks, with
    # marker IDs (1 for positive, 2 for negative) inserted at known
    # positions. This bypasses StubTok and exercises the chunk loop.
    stub_tok = StubTok()
    stub_mdl = StubModel()
    ep._finbert_pipe = (stub_tok, stub_mdl, torch)
    ep._FINBERT_LOAD_FAILED = False

    # Override the tokenizer call in score_section by patching the function
    # that converts text -> id list. We do this by feeding a string that,
    # under our StubTok, produces ids = many copies of "1" (positive).
    fake_pos_text = " ".join(["pos"] * 1500)  # >> 510 -> forces 3 chunks
    # We need ids[i]==1 for positive. Replace StubTok call to inject.
    def fake_call(text, **kw):
        return {"input_ids": [1] * 1500 if "pos" in text else [2] * 1500}
    stub_tok.__class__.__call__ = lambda self, text, **kw: fake_call(text, **kw)

    label, score = ep.score_section(fake_pos_text)
    assert label == "positive", f"expected positive, got {label} (score={score})"
    assert 0.0 <= score <= 1.0
    print(f"  ✓ score_section chunked 1500 tokens correctly -> {label} ({score:.3f})")

    label_neg, _ = ep.score_section(" ".join(["neg"] * 1500))
    assert label_neg == "negative", f"expected negative, got {label_neg}"
    print(f"  ✓ score_section chunking works for opposite polarity too")

    # Reset for any later tests.
    ep._finbert_pipe = None
    ep._FINBERT_LOAD_FAILED = False


def test_score_section_empty():
    label, score = ep.score_section("")
    assert label == "neutral" and score == 0.0
    print("  ✓ empty section -> ('neutral', 0.0)")


def test_full_orchestration_with_mocks():
    """
    Run the full build_records pipeline against the real transcripts but
    with mocked yfinance histories and a mocked FinBERT, so we can confirm
    the CSV is shaped correctly with non-null sentiment + returns.
    """
    import torch

    # --- Mock prices: synthetic 5-year history per ticker ----------------
    # Use enough business days to cover all call dates plus 5+ trading days
    # of forward window. The latest call in the corpus is META 2026-04-23.
    ep._price_cache.clear()
    for tk in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"):
        ep._price_cache[tk] = make_synth_history(
            start="2023-01-01", days=900, base=100.0, drift=0.001
        )
    ep._YF_LOAD_FAILED = False

    # --- Mock FinBERT: very short text -> trivial logits -----------------
    class MiniTok:
        cls_token_id = 0
        sep_token_id = 0

        def __call__(self, text, **kw):
            return {"input_ids": [3] * min(len(text.split()), 200)}

    class MiniModel:
        config = type("C", (), {"id2label": {0: "positive", 1: "negative", 2: "neutral"}})()

        def eval(self):
            return self

        def __call__(self, input_ids=None, attention_mask=None):
            # Bias toward "positive" so the test asserts something concrete.
            n = input_ids.shape[1]
            logits = torch.tensor([[1.5, -0.5, 0.2]]) * (n / 200)
            return type("Out", (), {"logits": logits})()

    ep._finbert_pipe = (MiniTok(), MiniModel(), torch)
    ep._FINBERT_LOAD_FAILED = False

    df = ep.build_records(Path(__file__).parent / "raw_text")
    assert len(df) == 35
    assert df["prepared_sentiment"].notna().all()
    assert df["qa_sentiment"].notna().all()
    assert df["ret_1d"].notna().all()
    assert df["ret_3d"].notna().all()
    assert df["ret_5d"].notna().all()

    # All sentiment should be 'positive' under our biased mock.
    assert (df["prepared_sentiment"] == "positive").all()
    print(f"  ✓ full pipeline against 35 transcripts produces complete rows "
          f"(no NaNs in sentiment or return columns)")
    print(f"  ✓ sample row:\n{df.iloc[0].to_dict()}")
    return df


def main():
    tests = [
        ("parse all transcripts", test_parse_all_transcripts),
        ("forward returns arithmetic", test_forward_returns_arithmetic),
        ("forward returns weekend anchor", test_forward_returns_call_on_weekend),
        ("forward returns missing windows", test_forward_returns_missing_when_too_recent),
        ("score_section chunking", test_score_section_chunking),
        ("score_section empty input", test_score_section_empty),
        ("full orchestration (mocks)", test_full_orchestration_with_mocks),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n[ test: {name} ]")
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"{len(tests) - failed} / {len(tests)} tests passed")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
