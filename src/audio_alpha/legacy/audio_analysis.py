"""
audio_analysis.py

Extracts paralinguistic features from Q&A earnings call audio clips.

Pipeline:
    1. WhisperX  → word-level transcription with forced alignment
    2. pyannote  → speaker diarization (who spoke when)
    3. Auto-ID   → management speaker = largest total speaking time
    4. Turn build → reconstruct analyst-question / exec-answer exchange pairs
    5. Features  → per-answer: pause latency, speech rate, fillers,
                   intra-answer pauses, pitch (F0), pitch variance, RMS energy
    6. Output    → JSON per clip saved to --output_dir

Usage:
    # Set token once in environment (recommended)
    export HF_TOKEN=hf_yourtoken

    # Run on a directory of Q&A clips
    python src/audio_alpha/legacy/audio_analysis.py --input_dir data/audio_alpha/qa_clips --output_dir data/audio_alpha/per_call_features

    # Override management speaker if auto-detect is wrong
    python src/audio_alpha/legacy/audio_analysis.py --input_dir data/audio_alpha/qa_clips --output_dir data/audio_alpha/per_call_features \\
        --mgmt_speaker SPEAKER_01

    # Use medium model for better transcription accuracy (slower)
    python src/audio_alpha/legacy/audio_analysis.py --input_dir data/audio_alpha/qa_clips --output_dir data/audio_alpha/per_call_features \\
        --model_size medium
"""

import os
import json
import argparse
import warnings
import numpy as np
import librosa

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTRA_PAUSE_THRESHOLD_SEC = 0.3   # gaps longer than this inside an answer count as pauses
MIN_TURN_DURATION_SEC     = 1.5   # skip management turns shorter than this
MIN_TURN_WORDS            = 4     # skip management turns with fewer words than this
SAMPLE_RATE               = 16000 # Hz — WhisperX and librosa both expect 16k

# Single-word fillers (lowercased, punctuation stripped before matching)
FILLER_WORDS_SINGLE = {
    "um", "uh", "hmm", "hm", "er", "ah", "oh",
    "actually", "basically", "literally", "honestly",
    "right", "okay", "ok", "so", "well", "yeah",
}

# Multi-word filler phrases (checked as substrings in the answer text)
FILLER_PHRASES_MULTI = [
    "you know", "i mean", "kind of", "sort of",
    "i think", "you see", "as i said",
]


# ---------------------------------------------------------------------------
# Step 1 — Transcription
# ---------------------------------------------------------------------------

def transcribe(audio_path: str, model_size: str = "small", device: str = "cpu") -> list[dict]:
    """
    Run WhisperX transcription + forced alignment.
    Returns a flat list of word dicts, each with keys:
        word, start (sec), end (sec), score
    """
    import whisperx

    print(f"    Loading WhisperX '{model_size}' model…")
    model = whisperx.load_model(model_size, device=device, compute_type="int8")
    audio = whisperx.load_audio(audio_path)

    print("    Transcribing…")
    result = model.transcribe(audio, batch_size=8)

    print("    Aligning word timestamps…")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False
    )

    # Flatten all words across all segments
    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None:
                words.append({
                    "word":  w["word"].strip(),
                    "start": float(w["start"]),
                    "end":   float(w["end"]),
                    "score": float(w.get("score", 0.0)),
                })
    return words


# ---------------------------------------------------------------------------
# Step 2 — Diarization
# ---------------------------------------------------------------------------

def diarize(audio_path: str, hf_token: str) -> list[dict]:
    """
    Run pyannote speaker diarization.
    Returns list of {start, end, speaker} dicts, sorted by start time.

    Audio is pre-loaded via torchaudio and passed as a waveform dict to
    bypass torchcodec (which fails on many Mac setups).
    """
    import torch
    import torchaudio
    from pyannote.audio import Pipeline

    print("    Loading pyannote diarization pipeline…")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # Pre-load audio with torchaudio to avoid torchcodec dependency
    print("    Loading audio for diarization…")
    waveform, sample_rate = torchaudio.load(audio_path)
    # pyannote expects mono or multi-channel float32 tensor
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

    print("    Running diarization (CPU — this takes a few minutes)…")
    diarization = pipeline(audio_dict)

    # pyannote >= 3.x wraps result in DiarizeOutput — unwrap to Annotation
    if hasattr(diarization, 'speaker_diarization'):
        diarization = diarization.speaker_diarization
    elif hasattr(diarization, 'annotation'):
        diarization = diarization.annotation

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start":   round(turn.start, 3),
            "end":     round(turn.end,   3),
            "speaker": speaker,
        })

    segments.sort(key=lambda s: s["start"])
    return segments


# ---------------------------------------------------------------------------
# Step 3 — Management speaker identification
# ---------------------------------------------------------------------------

def identify_management_speaker(diar_segments: list[dict], override: str | None = None) -> str:
    """
    Returns the speaker label with the most total speaking time.
    Pass override to skip auto-detection (e.g. when auto-detect is wrong).
    """
    if override:
        return override

    totals: dict[str, float] = {}
    for seg in diar_segments:
        spk = seg["speaker"]
        totals[spk] = totals.get(spk, 0.0) + (seg["end"] - seg["start"])

    mgmt = max(totals, key=totals.get)

    # Print the breakdown so the caller can verify or override
    print("    Speaker time breakdown:")
    for spk, secs in sorted(totals.items(), key=lambda x: -x[1]):
        flag = " ← management (auto)" if spk == mgmt else ""
        print(f"      {spk}: {secs:.1f}s{flag}")

    return mgmt


# ---------------------------------------------------------------------------
# Step 4 — Q&A turn reconstruction
# ---------------------------------------------------------------------------

def merge_consecutive_segments(diar_segments: list[dict]) -> list[dict]:
    """
    Merge back-to-back segments from the same speaker into one,
    collapsing short pauses that diarization sometimes splits.
    """
    if not diar_segments:
        return []

    merged = [diar_segments[0].copy()]
    for seg in diar_segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]
        if seg["speaker"] == last["speaker"] and gap < 0.5:
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


def build_qa_turns(diar_segments: list[dict], mgmt_speaker: str) -> list[dict]:
    """
    Reconstruct exchange pairs: (analyst/operator turn) → (management turn).

    Returns list of dicts:
        analyst_end   — when the question ended (sec)
        mgmt_start    — when the exec started answering (sec)
        mgmt_end      — when the exec finished (sec)
        exchange_idx  — sequential index
    """
    merged = merge_consecutive_segments(diar_segments)
    turns  = []
    idx    = 0

    i = 0
    while i < len(merged):
        seg = merged[i]

        # Find an analyst/operator segment
        if seg["speaker"] != mgmt_speaker:
            analyst_end = seg["end"]

            # Consume further non-management segments (multi-part question)
            while i + 1 < len(merged) and merged[i + 1]["speaker"] != mgmt_speaker:
                i += 1
                analyst_end = merged[i]["end"]

            # The next segment should be management
            if i + 1 < len(merged) and merged[i + 1]["speaker"] == mgmt_speaker:
                i += 1
                mgmt_start = merged[i]["start"]
                mgmt_end   = merged[i]["end"]

                # Consume further consecutive management segments
                while i + 1 < len(merged) and merged[i + 1]["speaker"] == mgmt_speaker:
                    i += 1
                    mgmt_end = merged[i]["end"]

                turns.append({
                    "exchange_idx": idx,
                    "analyst_end":  analyst_end,
                    "mgmt_start":   mgmt_start,
                    "mgmt_end":     mgmt_end,
                })
                idx += 1

        i += 1

    return turns


# ---------------------------------------------------------------------------
# Step 5a — Timestamp-derived features (no audio processing needed)
# ---------------------------------------------------------------------------

def get_words_in_window(words: list[dict], start: float, end: float) -> list[dict]:
    return [w for w in words if w["start"] >= start and w["end"] <= end]


def speech_rate_wpm(words: list[dict], duration_sec: float) -> float | None:
    if duration_sec <= 0 or not words:
        return None
    return round((len(words) / duration_sec) * 60, 2)


def pause_latency(analyst_end: float, mgmt_start: float) -> float:
    return round(max(0.0, mgmt_start - analyst_end), 3)


def intra_answer_pauses(words: list[dict], threshold: float = INTRA_PAUSE_THRESHOLD_SEC) -> dict:
    gaps = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > threshold:
            gaps.append(round(gap, 3))
    return {
        "intra_pause_count":    len(gaps),
        "intra_pause_mean_sec": round(float(np.mean(gaps)),  3) if gaps else 0.0,
        "intra_pause_max_sec":  round(float(np.max(gaps)),   3) if gaps else 0.0,
        "intra_pause_total_sec":round(float(np.sum(gaps)),   3) if gaps else 0.0,
    }


def filler_features(words: list[dict], duration_sec: float) -> dict:
    """
    Count single-word fillers via word list.
    Count multi-word filler phrases via reconstructed answer text.
    """
    single_count = 0
    for w in words:
        clean = w["word"].lower().strip(".,!?;:'\"")
        if clean in FILLER_WORDS_SINGLE:
            single_count += 1

    answer_text  = " ".join(w["word"] for w in words).lower()
    phrase_count = sum(answer_text.count(p) for p in FILLER_PHRASES_MULTI)

    total = single_count + phrase_count
    rate  = round((total / duration_sec) * 60, 2) if duration_sec > 0 else None

    return {
        "filler_count":       total,
        "filler_rate_per_min": rate,
    }


# ---------------------------------------------------------------------------
# Step 5b — Acoustic features (librosa)
# ---------------------------------------------------------------------------

def extract_acoustic_features(audio: np.ndarray, sr: int, start: float, end: float) -> dict | None:
    """
    Extract pitch and energy features from a time-bounded audio segment.
    Returns None if the segment is too short to be reliable.
    """
    s = int(start * sr)
    e = int(end   * sr)
    segment = audio[s:e]

    if len(segment) < sr * 0.2:   # shorter than 200ms — skip
        return None

    # --- Pitch (F0) via probabilistic YIN ---
    f0, voiced_flag, _ = librosa.pyin(
        segment,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz  (below lowest speaking voice)
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz (above highest speaking voice)
        sr=sr,
    )

    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

    mean_pitch     = round(float(np.mean(voiced_f0)),  2) if len(voiced_f0) > 0 else None
    pitch_variance = round(float(np.std(voiced_f0)),   2) if len(voiced_f0) > 0 else None

    # --- RMS Energy ---
    rms      = librosa.feature.rms(y=segment)[0]
    mean_rms = round(float(np.mean(rms)), 6)

    return {
        "mean_pitch_hz":     mean_pitch,
        "pitch_variance_hz": pitch_variance,
        "mean_rms_energy":   mean_rms,
    }


# ---------------------------------------------------------------------------
# Main per-file analysis
# ---------------------------------------------------------------------------

def analyze_clip(
    audio_path:            str,
    hf_token:              str,
    mgmt_speaker_override: str | None = None,
    model_size:            str = "small",
    cache_dir:             str | None = None,
) -> dict:

    print(f"\n{'='*60}")
    print(f"  File: {os.path.basename(audio_path)}")
    print(f"{'='*60}")

    fname_stem = os.path.basename(audio_path).replace(".mp3", "")

    # --- Transcript cache: load if exists, save after first successful run ---
    transcript_cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        transcript_cache_path = os.path.join(cache_dir, f"{fname_stem}_transcript.json")

    if transcript_cache_path and os.path.exists(transcript_cache_path):
        print("  Loading cached transcript…")
        with open(transcript_cache_path) as f:
            words = json.load(f)
    else:
        words = transcribe(audio_path, model_size=model_size)
        if transcript_cache_path:
            with open(transcript_cache_path, "w") as f:
                json.dump(words, f)
            print(f"  Transcript cached → {transcript_cache_path}")

    # Load audio once — reused for acoustic feature extraction
    print("  Loading audio…")
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Steps 2–3
    diar_segments = diarize(audio_path, hf_token)
    mgmt_speaker  = identify_management_speaker(diar_segments, mgmt_speaker_override)

    # Step 4
    qa_turns = build_qa_turns(diar_segments, mgmt_speaker)
    print(f"  Q&A exchanges found: {len(qa_turns)}")

    # Step 5 — per-exchange feature extraction
    exchanges = []
    skipped   = 0

    for turn in qa_turns:
        duration = turn["mgmt_end"] - turn["mgmt_start"]
        seg_words = get_words_in_window(words, turn["mgmt_start"], turn["mgmt_end"])

        # Skip trivially short or word-sparse turns
        if duration < MIN_TURN_DURATION_SEC or len(seg_words) < MIN_TURN_WORDS:
            skipped += 1
            continue

        answer_text = " ".join(w["word"] for w in seg_words)

        exchange = {
            "exchange_idx":      turn["exchange_idx"],
            "mgmt_start":        round(turn["mgmt_start"], 3),
            "mgmt_end":          round(turn["mgmt_end"],   3),
            "duration_sec":      round(duration, 3),
            "word_count":        len(seg_words),
            "answer_text":       answer_text,
            # Timestamp-derived
            "pause_latency_sec": pause_latency(turn["analyst_end"], turn["mgmt_start"]),
            "speech_rate_wpm":   speech_rate_wpm(seg_words, duration),
            **intra_answer_pauses(seg_words),
            **filler_features(seg_words, duration),
        }

        # Acoustic features (may return None for very short segments)
        acoustic = extract_acoustic_features(audio, sr, turn["mgmt_start"], turn["mgmt_end"])
        if acoustic:
            exchange.update(acoustic)
        else:
            exchange.update({
                "mean_pitch_hz":     None,
                "pitch_variance_hz": None,
                "mean_rms_energy":   None,
            })

        exchanges.append(exchange)

    print(f"  Exchanges extracted: {len(exchanges)}  |  Skipped (too short): {skipped}")

    return {
        "file":               os.path.basename(audio_path),
        "management_speaker": mgmt_speaker,
        "num_exchanges":      len(exchanges),
        "exchanges":          exchanges,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract paralinguistic features from Q&A earnings call clips."
    )
    parser.add_argument("--input_dir",    required=True,
                        help="Directory containing Q&A MP3 clips")
    parser.add_argument("--output_dir",   required=True,
                        help="Directory to write per-clip feature JSONs")
    parser.add_argument("--mgmt_speaker", default=None,
                        help="Override management speaker label (e.g. SPEAKER_01). "
                             "Use when auto-detection picks the wrong speaker.")
    parser.add_argument("--model_size",   default="small", choices=["small", "medium"],
                        help="WhisperX model size (default: small)")
    parser.add_argument("--hf_token",     default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token. Defaults to $HF_TOKEN env var.")
    parser.add_argument("--cache_dir",    default=None,
                        help="Directory to cache transcript JSONs. Skips re-transcription "
                             "on subsequent runs. Recommended: data/audio_alpha/transcript_cache")
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError(
            "HuggingFace token is required.\n"
            "Set it via:  export HF_TOKEN=hf_yourtoken\n"
            "Or pass:     --hf_token hf_yourtoken"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    mp3_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {args.input_dir}")
        return

    print(f"Found {len(mp3_files)} clip(s) to process.\n")

    for fname in mp3_files:
        audio_path = os.path.join(args.input_dir, fname)
        out_path   = os.path.join(args.output_dir, fname.replace(".mp3", "_features.json"))

        if os.path.exists(out_path):
            print(f"Skipping {fname}  (output already exists)")
            continue

        try:
            result = analyze_clip(
                audio_path,
                hf_token=args.hf_token,
                mgmt_speaker_override=args.mgmt_speaker,
                model_size=args.model_size,
                cache_dir=args.cache_dir,
            )
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  → Saved: {out_path}")
        except Exception as exc:
            print(f"  ERROR processing {fname}: {exc}")
            # Write a minimal error record so the file gets the 'already processed' skip
            # next run won't re-attempt a file that errored — remove it manually to retry
            with open(out_path, "w") as f:
                json.dump({"file": fname, "error": str(exc)}, f, indent=2)


if __name__ == "__main__":
    main()
