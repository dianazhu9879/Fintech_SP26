"""
End-to-end earnings call sentiment + forward-return pipeline.

Reads scraped Motley Fool transcripts, splits each into prepared remarks
and analyst Q&A, scores both sections with FinBERT, fetches forward
adjusted-close returns from yfinance, and writes a clean CSV.

Usage:
    python earnings_pipeline.py \
        --input  raw_text/ \
        --output earnings_sentiment_returns.csv

Required libraries: transformers, torch, yfinance, pandas (stdlib: pathlib, re, etc.)
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. PARSER
# ---------------------------------------------------------------------------

# Months we may see in scraped headers: full names, three-letter, "Sept", etc.
_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Examples this regex must handle (taken from real files):
#   "Oct. 30, 2025 at 5:00 p.m. ET"
#   "Thursday, January 29, 2026 at 5 p.m. ET"
#   "Wednesday, July 23, 2025, at 5:30 p.m. EDT"
#   "Oct 31, 2024" (older format, on its own line)
#   "Feb 04, 2025"
_DATE_RE = re.compile(
    r"\b(?P<month>"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\.?\s+"
    r"(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)

# Q1 2026, Q4 2024, etc.  Either from filename or in-file header.
# No leading \b so it matches inside "META_Q1_2026" (underscore is a word char).
_QUARTER_RE = re.compile(r"Q([1-4])[\s_]+(\d{4})\b", re.IGNORECASE)


@dataclass
class ParsedTranscript:
    ticker: str
    call_date: str          # YYYY-MM-DD
    quarter: str            # e.g. "Q1 2026"
    prepared_remarks: str
    qa_section: str
    source_path: str


def _parse_date(text_blob: str) -> str | None:
    """Return YYYY-MM-DD for the first date-like substring, else None."""
    m = _DATE_RE.search(text_blob)
    if not m:
        return None
    month = _MONTHS[m.group("month").lower().rstrip(".")]
    day = int(m.group("day"))
    year = int(m.group("year"))
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _extract_quarter(filename: str, raw_text: str) -> str:
    """Prefer the quarter label baked into the filename (e.g. AAPL_Q1_2026.txt)."""
    m = _QUARTER_RE.search(filename) or _QUARTER_RE.search(raw_text)
    if m:
        return f"Q{m.group(1)} {m.group(2)}"
    return ""


# Fallback Q&A entry-cue patterns — used when an explicit "Operator:" header
# is absent. Each pattern matches the *moderator's* line that opens the
# analyst questioning. Patterns are tried in order; the earliest match in
# the document wins.
_QA_CUE_PATTERNS = [
    # Apple/MSFT/AMZN-style — most common, most specific:
    re.compile(r"\nOperator:\s*\n"),
    # "We will now open the call / floor / lines for questions"
    # "We will now open for questions"
    re.compile(
        r"(?i)\n[^\n]*\bopen\s+(?:the\s+(?:call|floor|lines?)\s+)?for\s+questions?\b[^\n]*"
    ),
    # "question and answer session" cues:
    re.compile(r"(?i)\n[^\n]*\bquestion\s*(?:-|and|&)\s*answer\s+session\b[^\n]*"),
    # First-question cues (any possessive — our/the/your/first):
    re.compile(r"(?i)\n[^\n]*\bfirst\s+question\s+(?:comes|will\s+be|is)\s+from\b[^\n]*"),
    # TSLA-style moderator handoffs:
    re.compile(
        r"(?i)\n[^\n]*\b(?:go|head|move)(?:ing)?\s+(?:to|over\s+to|on\s+to)\s+"
        r"(?:investor|analyst)\s+questions\b[^\n]*"
    ),
    re.compile(
        r"(?i)\n[^\n]*\blet'?s\s+(?:go|start|move)\s+(?:to\s+|with\s+)?"
        r"(?:the\s+)?(?:investor|analyst)?\s*(?:q\s*&\s*a|questions)\b[^\n]*"
    ),
]


def _find_qa_boundary(text: str, start: int = 0) -> int | None:
    """Return absolute char-index where the Q&A section begins, or None."""
    best = None
    for pat in _QA_CUE_PATTERNS:
        m = pat.search(text, start)
        if m and (best is None or m.start() < best):
            # +1 to skip the leading "\n" we anchored on
            best = m.start() + 1
    return best


def _split_prepared_and_qa(raw: str) -> tuple[str, str]:
    """
    Three observed transcript layouts:

    (A) NEWER (post-2025 Fool layout)
        ... header ...
        Full Conference Call Transcript
        <speaker>:
        <prepared remarks ...>
        Operator:                       <-- or one of several other Q&A cues
        <Q&A ...>

    (B) OLDER (pre-2025 Fool layout)
        Prepared Remarks:
        <speaker> -- <title>
        <prepared remarks ...>
        Questions & Answers:
        Operator
        <Q&A ...>

    For both, we anchor the prepared-remarks start on an explicit header,
    then search for the Q&A boundary using a prioritised list of cue
    patterns (Operator:, "first question comes from...", TSLA-style
    moderator handoffs, etc.).
    """
    # --- Format A ----------------------------------------------------------
    a_marker = "Full Conference Call Transcript"
    if a_marker in raw:
        a_idx = raw.index(a_marker)
        body_start = raw.find("\n", a_idx) + 1
        qa_pos = _find_qa_boundary(raw, start=body_start)
        if qa_pos is not None:
            return raw[body_start:qa_pos].strip(), raw[qa_pos:].strip()
        return raw[body_start:].strip(), ""

    # --- Format B ----------------------------------------------------------
    b_pr_match = re.search(r"^Prepared Remarks:\s*$", raw, flags=re.MULTILINE)
    if b_pr_match:
        body_start = b_pr_match.end() + 1  # past the newline
        # Prefer the explicit "Questions & Answers:" header.
        qa_match = re.search(
            r"^Questions\s+(?:&|and)\s+Answers:\s*$", raw, flags=re.MULTILINE
        )
        if qa_match:
            prepared = raw[body_start : qa_match.start()].strip()
            qa = raw[qa_match.end() + 1 :].strip()
            return prepared, qa

        # No explicit Q&A header — fall back to cue patterns.
        qa_pos = _find_qa_boundary(raw, start=body_start)
        if qa_pos is not None:
            return raw[body_start:qa_pos].strip(), raw[qa_pos:].strip()

        return raw[body_start:].strip(), ""

    # Couldn't recognise the layout at all.
    return "", ""


def _extract_call_date(raw: str, lines: list[str]) -> str | None:
    """
    Per the task spec, line 9 holds the date in the new format
    ("Oct. 30, 2025 at 5:00 p.m. ET"). The older Fool layout pushes it a few
    lines lower but still in the top header block. Look there first, then
    fall back to the first date-shaped substring anywhere in the first 30
    lines (defensive).
    """
    if len(lines) >= 9:
        candidate = lines[8].strip()  # 0-indexed line 9
        d = _parse_date(candidate)
        if d:
            return d
    head_blob = "\n".join(lines[:30])
    return _parse_date(head_blob)


def _extract_ticker(filename: str, raw: str) -> str:
    """Ticker is the first underscore-separated token of the filename."""
    stem = Path(filename).stem
    tok = stem.split("_", 1)[0].upper()
    if tok:
        return tok
    # Belt-and-braces: "(AAPL)" header on line 1.
    m = re.search(r"\(([A-Z]{1,6})\)", raw[:200])
    return m.group(1) if m else ""


def parse_transcript_file(path: Path) -> ParsedTranscript:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    ticker = _extract_ticker(path.name, raw)
    call_date = _extract_call_date(raw, lines) or ""
    quarter = _extract_quarter(path.name, raw)
    prepared, qa = _split_prepared_and_qa(raw)

    return ParsedTranscript(
        ticker=ticker,
        call_date=call_date,
        quarter=quarter,
        prepared_remarks=prepared,
        qa_section=qa,
        source_path=str(path),
    )


def discover_transcripts(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.txt") if p.name.lower() != "readme.txt")


# ---------------------------------------------------------------------------
# 2. SENTIMENT (FinBERT)
# ---------------------------------------------------------------------------

# FinBERT outputs three labels: positive / negative / neutral. We chunk long
# inputs at the tokeniser level (max 512 tokens), score each chunk, and
# average the softmax probabilities to get one label + score per section.

_FINBERT_NAME = "ProsusAI/finbert"
_finbert_pipe = None


_FINBERT_LOAD_FAILED = False


def _load_finbert():
    """Lazy-load FinBERT once. Returns (tokenizer, model, torch_module) or None."""
    global _finbert_pipe, _FINBERT_LOAD_FAILED
    if _finbert_pipe is not None:
        return _finbert_pipe
    if _FINBERT_LOAD_FAILED:
        return None
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        print(f"[sentiment] transformers/torch unavailable: {e}", file=sys.stderr)
        _FINBERT_LOAD_FAILED = True
        return None

    try:
        tok = AutoTokenizer.from_pretrained(_FINBERT_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(_FINBERT_NAME)
    except Exception as e:
        # Most commonly: no network access to huggingface.co, or no cached model.
        print(
            f"[sentiment] could not load FinBERT ({_FINBERT_NAME}): {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        print(
            "[sentiment] sentiment columns will be left empty. To populate them, "
            "run this script on a machine with internet access to huggingface.co, "
            "or pre-download the model and set the HF_HOME env var to its cache.",
            file=sys.stderr,
        )
        _FINBERT_LOAD_FAILED = True
        return None

    mdl.eval()
    _finbert_pipe = (tok, mdl, torch)
    return _finbert_pipe


def score_section(
    text: str,
    chunk_size: int = 510,
    stride: int = 256,
) -> tuple[str | None, float | None]:
    """
    Return (dominant_label, mean_score). Empty input -> ("neutral", 0.0).
    If FinBERT cannot be loaded, returns (None, None).

    Tokenises the full text once, then slides a chunk_size-token window
    with the given stride (overlap = chunk_size - stride) so context at
    chunk boundaries is captured by at least one well-centred window.
    Each chunk is weighted by max(0, max(p_pos, p_neg) - 1/3): its excess
    directional probability above the 3-class chance level. Operator lines
    and analyst questions score at or below chance and get zero weight;
    management-response chunks with real sentiment drive the aggregate.
    """
    text = (text or "").strip()
    if not text:
        return "neutral", 0.0

    pipe = _load_finbert()
    if pipe is None:
        return None, None
    tokenizer, model, torch = pipe

    enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]

    if not ids:
        return "neutral", 0.0

    # Overlapping windows: stride < chunk_size gives overlap = chunk_size - stride.
    chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), stride)]
    label_names = model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}

    # Resolve label indices from the model config so the code isn't brittle if
    # the id2label mapping ever changes (e.g. a fine-tuned checkpoint).
    label2idx = {v.lower(): k for k, v in label_names.items()}
    pos_idx = label2idx.get("positive", 0)
    neg_idx = label2idx.get("negative", 1)

    # Confidence-weighted accumulator.
    # Q&A sections are structurally ~75% neutral (operator lines, analyst
    # questions). Any weighting based on overall peak confidence amplifies
    # neutral further, because neutral chunks score higher confidence than
    # positive ones. Instead we weight each chunk by its *excess directional
    # probability above the 3-class chance level* (1/3):
    #   w = max(0, max(p_pos, p_neg) - 1/3)
    # Chunks at or below chance (operator lines, analyst questions) get
    # weight zero and drop out entirely. Management-response chunks with real
    # directional signal drive the aggregate proportionally to their strength.
    # If no chunk exceeds the chance level the section genuinely has no
    # directional signal and we fall back to neutral.
    _CHANCE = 1.0 / 3.0
    weighted_probs = None   # signal-weighted accumulator
    total_weight = 0.0
    probs_sum = None        # unweighted fallback accumulator
    n_chunks = 0

    with torch.no_grad():
        for chunk in chunks:
            input_ids = torch.tensor(
                [[tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]]
            )
            attn = torch.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits, dim=-1)[0]

            probs_sum = probs if probs_sum is None else probs_sum + probs
            n_chunks += 1

            directional = float(torch.max(probs[pos_idx], probs[neg_idx]).item())
            confidence = max(0.0, directional - _CHANCE)
            if confidence == 0.0:
                continue
            weighted_probs = (
                probs * confidence
                if weighted_probs is None
                else weighted_probs + probs * confidence
            )
            total_weight += confidence

    if total_weight > 0.0 and weighted_probs is not None:
        avg_probs = weighted_probs / total_weight
    elif probs_sum is not None:
        # No chunk had above-chance directional signal; the section is
        # genuinely low-sentiment — fall back to simple average.
        avg_probs = probs_sum / n_chunks
    else:
        return "neutral", 0.0

    idx = int(avg_probs.argmax().item())
    return label_names[idx].lower(), float(avg_probs[idx].item())


# ---------------------------------------------------------------------------
# 3. PRICE DATA + FORWARD RETURNS
# ---------------------------------------------------------------------------

# We pull a 30-calendar-day window starting on the call date so we have enough
# trading days for t, t+1, t+3, t+5 even across long weekends/holidays.

_price_cache: dict[str, pd.DataFrame] = {}
_YF_LOAD_FAILED = False


def _load_history(ticker: str) -> pd.DataFrame:
    """Pull a wide history once per ticker and cache it (auto-adjusted close)."""
    global _YF_LOAD_FAILED
    if ticker in _price_cache:
        return _price_cache[ticker]
    if _YF_LOAD_FAILED:
        _price_cache[ticker] = pd.DataFrame()
        return _price_cache[ticker]
    try:
        import yfinance as yf
    except ImportError as e:
        print(f"[prices] yfinance unavailable: {e}", file=sys.stderr)
        _YF_LOAD_FAILED = True
        _price_cache[ticker] = pd.DataFrame()
        return _price_cache[ticker]

    try:
        # auto_adjust=True makes "Close" already reflect splits/dividends.
        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            actions=False,
            threads=False,
        )
    except Exception as e:
        print(
            f"[prices] yfinance fetch for {ticker} failed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        _price_cache[ticker] = pd.DataFrame()
        return _price_cache[ticker]

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns a MultiIndex when a list is passed; flatten.
        df.columns = [c[0] for c in df.columns]
    if df.empty:
        print(
            f"[prices] yfinance returned empty data for {ticker} "
            "(rate limit, blocked network, or bad ticker).",
            file=sys.stderr,
        )
    else:
        df.index = pd.to_datetime(df.index).normalize()
    _price_cache[ticker] = df
    return df


def forward_returns(
    ticker: str, call_date: str, windows: Iterable[int] = (1, 3, 5)
) -> dict[str, float | None]:
    """
    Returns dict like {"ret_1d": .., "ret_3d": .., "ret_5d": ..}.

    Convention: t = first trading day on/after the call date (after-hours
    earnings calls land on the next trading day). t+N = the close N trading
    days later. ret_Nd = close_{t+N}/close_t - 1.
    """
    out = {f"ret_{n}d": None for n in windows}
    if not call_date:
        return out
    df = _load_history(ticker)
    if df.empty or "Close" not in df.columns:
        return out

    target = pd.Timestamp(call_date).normalize()

    # Trading dates >= call date.
    forward = df.index[df.index >= target]
    if len(forward) == 0:
        return out

    # If the call was after-hours, the price reaction shows up the NEXT
    # trading day — use that as t.
    t_idx = forward[0]

    # If call_date is itself a trading day we still anchor at the close that
    # day, since that's the close into which the call was reported. The
    # forward windows then count trading days beyond that.
    pos = df.index.get_loc(t_idx)

    # Defensive: get_loc can return slice/array for duplicates; we expect int.
    if not isinstance(pos, (int,)):
        try:
            pos = int(pos)
        except Exception:
            return out

    close_t = float(df["Close"].iloc[pos])
    if not (close_t and close_t == close_t):  # NaN guard
        return out

    for n in windows:
        if pos + n < len(df):
            close_tn = float(df["Close"].iloc[pos + n])
            if close_tn and close_tn == close_tn:
                out[f"ret_{n}d"] = close_tn / close_t - 1.0
    return out


# ---------------------------------------------------------------------------
# 4. ORCHESTRATION + 5. VALIDATION
# ---------------------------------------------------------------------------

def build_records(transcripts_dir: Path) -> pd.DataFrame:
    files = discover_transcripts(transcripts_dir)
    print(f"[parse] found {len(files)} transcript files under {transcripts_dir}")

    rows = []

    def _fmt_score(s: float | None) -> str:
        return f"{s:.2f}" if s is not None else "----"

    def _fmt_ret(r: float | None) -> str:
        return f"{r*100:+.2f}%" if r is not None else "n/a"

    for i, path in enumerate(files, 1):
        parsed = parse_transcript_file(path)
        prepared_label, prepared_score = score_section(parsed.prepared_remarks)
        qa_label, qa_score = score_section(parsed.qa_section)
        rets = forward_returns(parsed.ticker, parsed.call_date)
        rows.append(
            {
                "ticker": parsed.ticker,
                "call_date": parsed.call_date,
                "quarter": parsed.quarter,
                "prepared_sentiment": prepared_label,
                "prepared_score": round(prepared_score, 4) if prepared_score is not None else None,
                "qa_sentiment": qa_label,
                "qa_score": round(qa_score, 4) if qa_score is not None else None,
                "ret_1d": rets["ret_1d"],
                "ret_3d": rets["ret_3d"],
                "ret_5d": rets["ret_5d"],
                "_source": Path(parsed.source_path).name,
                "_prepared_chars": len(parsed.prepared_remarks),
                "_qa_chars": len(parsed.qa_section),
            }
        )
        print(
            f"[{i:>2}/{len(files)}] {parsed.ticker:<5} {parsed.call_date or '????-??-??'} "
            f"{parsed.quarter:<8} prep={(prepared_label or '?'):<8}({_fmt_score(prepared_score)}) "
            f"qa={(qa_label or '?'):<8}({_fmt_score(qa_score)}) "
            f"ret1d={_fmt_ret(rets['ret_1d'])}"
        )
    df = pd.DataFrame(rows)
    return df


def validate_and_summarise(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total transcripts parsed: {len(df)}")

    # Parse failures
    bad_dates = df[df["call_date"] == ""]
    if not bad_dates.empty:
        print(f"\n!! {len(bad_dates)} transcripts with no parseable call date:")
        for _, r in bad_dates.iterrows():
            print(f"   - {r['_source']}")

    bad_split = df[(df["_prepared_chars"] == 0) | (df["_qa_chars"] == 0)]
    if not bad_split.empty:
        print(f"\n!! {len(bad_split)} transcripts with empty prepared OR Q&A:")
        for _, r in bad_split.iterrows():
            print(
                f"   - {r['_source']:<28} prep_chars={r['_prepared_chars']} "
                f"qa_chars={r['_qa_chars']}"
            )

    # Missing prices
    miss = df[df[["ret_1d", "ret_3d", "ret_5d"]].isna().any(axis=1)]
    print(f"\nTranscripts with at least one missing forward return: {len(miss)}")
    for _, r in miss.iterrows():
        missing_cols = [c for c in ("ret_1d", "ret_3d", "ret_5d") if pd.isna(r[c])]
        print(f"   - {r['ticker']:<5} {r['call_date']} missing {missing_cols}")

    # Sentiment distribution
    print("\nPrepared-remarks sentiment distribution:")
    print(df["prepared_sentiment"].value_counts().to_string())
    print("\nQ&A sentiment distribution:")
    print(df["qa_sentiment"].value_counts().to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    df = build_records(args.input)

    # Final CSV — strip helper columns prefixed with "_".
    out_cols = [
        "ticker", "call_date", "quarter",
        "prepared_sentiment", "prepared_score",
        "qa_sentiment", "qa_score",
        "ret_1d", "ret_3d", "ret_5d",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["ticker", "call_date"])[out_cols].to_csv(args.output, index=False)
    print(f"\n[output] wrote {args.output}")

    validate_and_summarise(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
