"""
clip_qa.py — Earnings Call Q&A Audio Clipper (Adaptive Detection)

Usage:
    python clip_qa.py --input_dir ./earnings_calls --output_dir ./qa_clips

Requirements:
    pip install whisperx rapidfuzz sentence-transformers
    brew install ffmpeg  (macOS) or sudo apt install ffmpeg (Linux)

Transcription strategy (two-pass):
    Pass 1 — Transcribe only the back 60% of the audio using the tiny model.
             Fast (~3–4x speedup). Sufficient to find the Q&A trigger phrase.
    Pass 2 — Only runs if pass 1 fails. Transcribes the full audio with the
             configured model (default: small) as a fallback.

Detection strategy (in order of confidence):
    1. Exact phrase match   — fast, zero-cost, handles known patterns
    2. Fuzzy phrase match   — catches typos / ASR errors in known phrases
    3. Semantic search      — embedding similarity; catches novel phrasings
    4. Structural heuristic — last-resort pattern exploit (first analyst Q)

Skip logic:
    Files with an existing _qa.mp3 output are skipped automatically.
    Use --reprocess to override and reprocess all files.
"""

import os
import re
import json
import argparse
import subprocess
import tempfile
import whisperx

# ---------------------------------------------------------------------------
# ffmpeg helpers — duration probe and audio trimming for two-pass strategy
# ---------------------------------------------------------------------------

def get_audio_duration(mp3_path: str) -> float | None:
    """Return the duration of an audio file in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        mp3_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def trim_audio_to_tempfile(mp3_path: str, start_seconds: float) -> str | None:
    """
    Use ffmpeg to cut audio from start_seconds onward into a temp file.
    Returns the temp file path, or None on failure.
    The caller is responsible for deleting the temp file.
    """
    suffix = os.path.splitext(mp3_path)[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", mp3_path,
        "-ss", str(start_seconds),
        "-c", "copy",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(tmp.name)
        return None
    return tmp.name


# ---------------------------------------------------------------------------
# Known trigger phrases (still useful as anchors for fuzzy + semantic search)
# ---------------------------------------------------------------------------
QA_TRIGGER_PHRASES = [
    "we will now begin the question and answer",
    "we will now begin our question and answer",
    "we'll now begin the question and answer",
    "we'll now open the line for questions",
    "we will now open the line for questions",
    "open the floor for questions",
    "now open for questions",
    "our first question comes from",
    "first question comes from",
    "we'll take our first question",
    "we will take our first question",
    "you may begin your question",
    "let's open the call to questions",
    "we'll now move over to q&a",
    "we will now transition to q&a",
    "we're going to investor questions",
    "let's open to call the questions",
    "we're going to head over to investor questions",
    "we're gonna go to investor questions",
    "we'll now take questions",
    "now take questions from",
    "open it up for questions",
    "operator will now",
    "questions and answers session",
    "we will now begin the question and answer session",
    "we'll now turn to questions",
    "we're ready to take questions"
]

# Regex for the structural heuristic: "Our first question comes from [Name] at [Firm]"
# This pattern is remarkably stable across operators and years.
FIRST_ANALYST_RE = re.compile(
    r"(our|the)\s+first\s+question\s+(comes?\s+from|is\s+from|will\s+come\s+from)"
    r"|first\s+question.{0,30}(from|caller)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Lazy-loaded semantic model (only imported if needed)
# ---------------------------------------------------------------------------
_semantic_model = None

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            print("  → Loading semantic embedding model (first use only)...")
            _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("  ⚠ sentence-transformers not installed; skipping semantic search.")
            print("    Run: pip install sentence-transformers")
            _semantic_model = False  # sentinel: tried and failed
    return _semantic_model


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def _search_window(segments: list[dict], total_duration: float) -> list[dict]:
    """
    Earnings calls always have prepared remarks first, then Q&A.
    Restrict search to the back 60% of the call to avoid false positives
    in the intro (e.g. "we will take questions after remarks").
    """
    cutoff = total_duration * 0.12
    return [s for s in segments if s["start"] >= cutoff]


def detect_exact(segments: list[dict]) -> tuple[float | None, str]:
    """Exact substring match — same as original script."""
    for seg in segments:
        text = seg["text"].strip().lower()
        for phrase in QA_TRIGGER_PHRASES:
            if phrase in text:
                return seg["start"], f"exact match: '{phrase}'"
    return None, ""


def detect_fuzzy(segments: list[dict], threshold: int = 82) -> tuple[float | None, str]:
    """
    Fuzzy match using rapidfuzz partial_ratio.
    Catches ASR transcription errors like 'we'll now begin the question in answer'
    or 'we will now open the lines for questions'.

    threshold=82 is empirically good: catches 1–2 word errors without false positives.
    Lower it to ~75 if you have very noisy ASR; raise to 90 for precision.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        print("  ⚠ rapidfuzz not installed; skipping fuzzy match.")
        print("    Run: pip install rapidfuzz")
        return None, ""

    best_score = 0
    best_result = (None, "")

    for seg in segments:
        text = seg["text"].strip().lower()
        for phrase in QA_TRIGGER_PHRASES:
            score = fuzz.partial_ratio(phrase, text)
            if score >= threshold and score > best_score:
                best_score = score
                best_result = (seg["start"], f"fuzzy match (score={score}): '{phrase}'")

    return best_result


def detect_semantic(segments: list[dict], threshold: float = 0.55) -> tuple[float | None, str]:
    """
    Semantic embedding similarity. Encodes each segment and compares against
    a fixed 'anchor' description of a Q&A transition. Catches genuinely novel
    operator phrasings not in the trigger list.

    threshold=0.55 works well for MiniLM; cosine similarity is 0–1.
    """
    try:
        import numpy as np
    except ImportError:
        return None, ""

    model = get_semantic_model()
    if not model:
        return None, ""

    qa_anchor = (
        "We will now begin the question and answer session. "
        "Analysts may ask questions to management."
    )
    anchor_vec = model.encode(qa_anchor, normalize_embeddings=True)

    best_score = 0.0
    best_result = (None, "")

    texts = [seg["text"].strip() for seg in segments]
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    for seg, vec in zip(segments, vecs):
        score = float(np.dot(anchor_vec, vec))
        if score >= threshold and score > best_score:
            best_score = score
            best_result = (seg["start"], f"semantic match (score={score:.3f}): '{seg['text'].strip()[:80]}'")

    return best_result


def detect_structural(segments: list[dict]) -> tuple[float | None, str]:
    """
    Last-resort heuristic: look for the 'first question comes from' pattern,
    which almost universally marks the analyst Q&A start regardless of operator.
    When found, walk back to find the operator handoff sentence before it.
    """
    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if FIRST_ANALYST_RE.search(text):
            # Try to walk back a few segments to catch the operator intro sentence
            lookback = max(0, i - 3)
            for j in range(lookback, i + 1):
                candidate = segments[j]["text"].strip().lower()
                if any(kw in candidate for kw in ["question", "q&a", "open", "begin"]):
                    label = f"structural heuristic: '{segments[j]['text'].strip()[:80]}'"
                    return segments[j]["start"], label
            # Fallback: use the matched segment itself
            return seg["start"], f"structural heuristic (direct): '{text[:80]}'"
    return None, ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def find_qa_start(segments: list[dict]) -> tuple[float | None, str, str]:
    """
    Run detection methods in order of confidence. Returns:
        (start_seconds, description, method_name)
    or (None, "", "") if nothing found.
    """
    if not segments:
        return None, "", ""

    total_duration = segments[-1]["end"]
    window = _search_window(segments, total_duration)

    print(f"  → Search window: {window[0]['start']:.0f}s – {total_duration:.0f}s "
          f"({len(window)}/{len(segments)} segments)")

    # 1. Exact
    t, desc = detect_exact(window)
    if t is not None:
        return t, desc, "exact"

    # 2. Fuzzy
    t, desc = detect_fuzzy(window)
    if t is not None:
        return t, desc, "fuzzy"

    # 3. Semantic
    t, desc = detect_semantic(window)
    if t is not None:
        return t, desc, "semantic"

    # 4. Structural (run on ALL segments — first analyst Q may appear early on short calls)
    t, desc = detect_structural(segments)
    if t is not None:
        return t, desc, "structural"

    return None, "", ""


# ---------------------------------------------------------------------------
# ffmpeg clipping
# ---------------------------------------------------------------------------

def clip_audio(input_path: str, output_path: str, start_seconds: float) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_seconds),
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ ffmpeg error:\n{result.stderr}")
        return False
    return True


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, model) -> list[dict] | None:
    """Transcribe an audio file and return segments, or None on failure."""
    try:
        audio = whisperx.load_audio(audio_path)
        transcription = model.transcribe(audio, batch_size=4)
        return transcription["segments"]
    except Exception as e:
        print(f"  ✗ Transcription failed: {e}")
        return None


def process_file(
    mp3_path: str,
    output_dir: str,
    model,
    tiny_model,
    reprocess: bool = False,
) -> dict:
    filename = os.path.basename(mp3_path)
    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{stem}_qa.mp3")

    print(f"\n{'─' * 60}")
    print(f"Processing: {filename}")

    result = {
        "file": filename,
        "status": None,
        "detection_method": None,
        "detection_note": None,
        "transcription_pass": None,
        "qa_start_seconds": None,
        "output": None,
        "note": None,
    }

    # ── Skip if already done ────────────────────────────────────────────────
    if not reprocess and os.path.exists(output_path):
        print(f"  ⏭ Already processed — skipping (use --reprocess to override)")
        result["status"] = "skipped"
        result["output"] = output_path
        return result

    # ── Probe duration ───────────────────────────────────────────────────────
    duration = get_audio_duration(mp3_path)
    if duration is None:
        print("  ⚠ Could not determine duration; falling back to full transcription.")

    # ── Pass 1: Transcribe only the back 60% with tiny model ────────────────
    # The Q&A trigger phrase always appears in the back half of the call.
    # Tiny model is ~3–4x faster than small and sufficient for phrase detection.
    qa_start, detection_desc, method, segments_used = None, "", "", []
    tmp_path = None

    if duration is not None:
        trim_start = duration * 0.40
        mins_trim, secs_trim = divmod(int(trim_start), 60)
        print(f"  → Pass 1 (tiny model, back 60%): trimming from {mins_trim}m {secs_trim}s ...")
        tmp_path = trim_audio_to_tempfile(mp3_path, trim_start)

    if tmp_path is not None:
        try:
            segments_p1 = transcribe_audio(tmp_path, tiny_model)
            if segments_p1:
                # Timestamps from the trimmed file are relative — add the offset back
                for seg in segments_p1:
                    seg["start"] += trim_start
                    seg["end"]   += trim_start
                qa_start, detection_desc, method = find_qa_start(segments_p1)
                if qa_start is not None:
                    segments_used = segments_p1
                    result["transcription_pass"] = "pass1_tiny_back60"
                    print(f"  ✓ Pass 1 succeeded.")
                else:
                    print(f"  → Pass 1 found nothing. Falling back to pass 2.")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        print("  → Skipping pass 1 (trim failed or duration unknown).")

    # ── Pass 2: Full transcription with main model ───────────────────────────
    if qa_start is None:
        print(f"  → Pass 2 (full audio, {getattr(model, '_model_size', 'configured')} model)...")
        segments_p2 = transcribe_audio(mp3_path, model)
        if segments_p2 is None:
            result["status"] = "error"
            result["note"] = "Transcription failed on both passes"
            return result
        qa_start, detection_desc, method = find_qa_start(segments_p2)
        segments_used = segments_p2
        result["transcription_pass"] = "pass2_full"

    # ── Handle detection failure ─────────────────────────────────────────────
    if qa_start is None:
        result["status"] = "no_qa_found"
        result["note"] = "All detection methods failed on both passes"
        print(f"  ✗ No Q&A start detected")
        transcript_path = os.path.join(output_dir, f"{stem}_transcript.json")
        with open(transcript_path, "w") as f:
            json.dump(segments_used, f, indent=2)
        print(f"  → Transcript saved to {transcript_path} for manual review")
        return result

    mins, secs = divmod(int(qa_start), 60)
    print(f"  ✓ Q&A detected at {qa_start:.1f}s ({mins}m {secs}s)")
    print(f"    Method: {method} — {detection_desc}")
    print(f"    Pass:   {result['transcription_pass']}")

    result["detection_method"] = method
    result["detection_note"] = detection_desc

    # ── Clip ─────────────────────────────────────────────────────────────────
    print("  → Clipping with ffmpeg...")
    success = clip_audio(mp3_path, output_path, qa_start)

    if success:
        result["status"] = "success"
        result["qa_start_seconds"] = round(qa_start, 2)
        result["output"] = output_path
        print(f"  ✓ Saved: {output_path}")
    else:
        result["status"] = "error"
        result["note"] = "ffmpeg clipping failed"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clip Q&A portions from earnings call MP3s")
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_size", default="small",
                        choices=["tiny", "base", "small", "medium"],
                        help="WhisperX model for pass 2 / fallback (default: small)")
    parser.add_argument("--no_semantic", action="store_true",
                        help="Skip semantic search (faster; omit if sentence-transformers absent)")
    parser.add_argument("--reprocess", action="store_true",
                        help="Reprocess files even if a _qa.mp3 output already exists")
    args = parser.parse_args()

    if args.no_semantic:
        global detect_semantic
        detect_semantic = lambda segs, threshold=0.55: (None, "")

    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir '{args.input_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    mp3_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".mp3")
    ])

    if not mp3_files:
        print(f"No MP3 files found in {args.input_dir}")
        return

    print(f"Found {len(mp3_files)} MP3 file(s)")
    print(f"Pass 1 model: tiny (back 60% only) | Pass 2 model: {args.model_size} (full audio)")
    print(f"Output dir: {args.output_dir}")
    if not args.reprocess:
        print("Skip mode: ON  (already-processed files will be skipped; use --reprocess to override)")

    # Load tiny model for fast pass 1
    print(f"\nLoading WhisperX tiny model (pass 1)...")
    tiny_model = whisperx.load_model("tiny", "cpu", compute_type="int8")
    print("Tiny model loaded.")

    # Load main model for pass 2 fallback (skip if same as tiny)
    if args.model_size == "tiny":
        print("Pass 2 model same as pass 1 — reusing tiny model.")
        main_model = tiny_model
    else:
        print(f"Loading WhisperX {args.model_size} model (pass 2 fallback)...")
        main_model = whisperx.load_model(args.model_size, "cpu", compute_type="int8")
        print(f"{args.model_size.capitalize()} model loaded.\n")

    results = []
    for mp3_path in mp3_files:
        results.append(
            process_file(mp3_path, args.output_dir, main_model, tiny_model, args.reprocess)
        )

    # Summary
    print(f"\n{'═' * 60}")
    print("SUMMARY")
    print(f"{'═' * 60}")
    success  = [r for r in results if r["status"] == "success"]
    skipped  = [r for r in results if r["status"] == "skipped"]
    no_qa    = [r for r in results if r["status"] == "no_qa_found"]
    errors   = [r for r in results if r["status"] == "error"]

    print(f"  ✓ Success:      {len(success)}")
    print(f"  ⏭ Skipped:      {len(skipped)}")
    print(f"  ? No Q&A found: {len(no_qa)}")
    print(f"  ✗ Errors:       {len(errors)}")

    if success:
        method_counts = {}
        pass_counts   = {}
        for r in success:
            m = r.get("detection_method", "unknown")
            p = r.get("transcription_pass", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1
            pass_counts[p]   = pass_counts.get(p, 0) + 1
        print("\n  Detection method breakdown:")
        for m, c in sorted(method_counts.items()):
            print(f"    {m}: {c}")
        print("\n  Transcription pass breakdown:")
        for p, c in sorted(pass_counts.items()):
            print(f"    {p}: {c}")

    if no_qa:
        print("\nFiles where Q&A was not detected (transcripts saved):")
        for r in no_qa:
            print(f"  - {r['file']}")

    if errors:
        print("\nFiles with errors:")
        for r in errors:
            print(f"  - {r['file']}: {r['note']}")

    log_path = os.path.join(args.output_dir, "clip_qa_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull log saved to: {log_path}")


if __name__ == "__main__":
    main()