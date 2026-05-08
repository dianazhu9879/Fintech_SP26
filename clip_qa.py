# need to run this one more time

"""
clip_qa.py — Earnings Call Q&A Audio Clipper

Usage:
    python clip_qa.py --input_dir ./earnings_calls --output_dir ./qa_clips

Requirements:
    pip install whisperx
    brew install ffmpeg  (macOS) or sudo apt install ffmpeg (Linux)
"""

import os
import re
import json
import argparse
import subprocess
import whisperx

# ---------------------------------------------------------------------------
# Q&A trigger phrases — operator language that signals the Q&A start
# Add more here if you notice patterns in your specific call transcripts
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
    "Tim and I will take questions",
    "we're going to head over to investor questions",
    "we're gonna go to investor questions",
]

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def find_qa_start(segments: list[dict]) -> float | None:
    """
    Search transcript segments for a Q&A trigger phrase.
    Returns the start time in seconds, or None if not found.
    """
    # Build a list of (start_time, text) for each segment
    for segment in segments:
        text = segment["text"].strip().lower()
        for phrase in QA_TRIGGER_PHRASES:
            if phrase in text:
                print(f"  ✓ Found Q&A trigger: '{phrase}'")
                print(f"    Segment text: '{segment['text'].strip()}'")
                return segment["start"]
    return None


def clip_audio(input_path: str, output_path: str, start_seconds: float) -> bool:
    """
    Use ffmpeg to clip audio from start_seconds to end of file.
    Returns True on success.
    """
    cmd = [
        "ffmpeg",
        "-y",                          # overwrite output if exists
        "-i", input_path,              # input file
        "-ss", str(start_seconds),     # start time
        "-c", "copy",                  # lossless copy, no re-encoding
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ ffmpeg error:\n{result.stderr}")
        return False
    return True


def process_file(
    mp3_path: str,
    output_dir: str,
    model,
    device: str = "cpu",
    log: list = None
) -> dict:
    """
    Full pipeline for a single MP3:
      1. Transcribe with WhisperX
      2. Find Q&A start timestamp
      3. Clip audio with ffmpeg
    Returns a result dict for logging.
    """
    filename = os.path.basename(mp3_path)
    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{stem}_qa.mp3")

    print(f"\n{'─' * 60}")
    print(f"Processing: {filename}")

    result = {
        "file": filename,
        "status": None,
        "qa_start_seconds": None,
        "output": None,
        "note": None,
    }

    # Step 1: Transcribe
    print("  → Transcribing (this may take a few minutes on CPU)...")
    try:
        audio = whisperx.load_audio(mp3_path)
        transcription = model.transcribe(audio, batch_size=4)
        segments = transcription["segments"]
    except Exception as e:
        result["status"] = "error"
        result["note"] = f"Transcription failed: {e}"
        print(f"  ✗ {result['note']}")
        return result

    # Step 2: Find Q&A start
    print("  → Searching for Q&A trigger phrase...")
    qa_start = find_qa_start(segments)

    if qa_start is None:
        result["status"] = "no_qa_found"
        result["note"] = "No Q&A trigger phrase detected in transcript"
        print(f"  ✗ {result['note']}")
        # Save transcript for manual inspection
        transcript_path = os.path.join(output_dir, f"{stem}_transcript.json")
        with open(transcript_path, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"  → Saved transcript to {transcript_path} for manual review")
        return result

    print(f"  ✓ Q&A starts at {qa_start:.1f}s ({int(qa_start)//60}m {int(qa_start)%60}s)")

    # Step 3: Clip audio
    print(f"  → Clipping audio with ffmpeg...")
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
    parser.add_argument("--input_dir", required=True, help="Directory containing earnings call MP3s")
    parser.add_argument("--output_dir", required=True, help="Directory to save Q&A clips")
    parser.add_argument("--model_size", default="small", choices=["tiny", "base", "small", "medium"],
                        help="WhisperX model size (default: small). Use medium for better accuracy.")
    args = parser.parse_args()

    # Validate dirs
    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir '{args.input_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all MP3s
    mp3_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".mp3")
    ])

    if not mp3_files:
        print(f"No MP3 files found in {args.input_dir}")
        return

    print(f"Found {len(mp3_files)} MP3 file(s)")
    print(f"Model: {args.model_size} | Device: CPU")
    print(f"Output dir: {args.output_dir}")

    # Load WhisperX model once (reused across all files)
    print(f"\nLoading WhisperX model '{args.model_size}'...")
    device = "cpu"
    compute_type = "int8"  # most compatible for CPU
    model = whisperx.load_model(args.model_size, device, compute_type=compute_type)
    print("Model loaded.\n")

    # Process each file
    results = []
    for mp3_path in mp3_files:
        result = process_file(mp3_path, args.output_dir, model, device)
        results.append(result)

    # Summary
    print(f"\n{'═' * 60}")
    print("SUMMARY")
    print(f"{'═' * 60}")
    success = [r for r in results if r["status"] == "success"]
    no_qa   = [r for r in results if r["status"] == "no_qa_found"]
    errors  = [r for r in results if r["status"] == "error"]

    print(f"  ✓ Success:      {len(success)}")
    print(f"  ? No Q&A found: {len(no_qa)}")
    print(f"  ✗ Errors:       {len(errors)}")

    if no_qa:
        print("\nFiles where Q&A was not detected (transcripts saved for review):")
        for r in no_qa:
            print(f"  - {r['file']}")

    if errors:
        print("\nFiles with errors:")
        for r in errors:
            print(f"  - {r['file']}: {r['note']}")

    # Save full log
    log_path = os.path.join(args.output_dir, "clip_qa_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull log saved to: {log_path}")


if __name__ == "__main__":
    main()