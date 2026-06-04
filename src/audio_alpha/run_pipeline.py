from __future__ import annotations

import subprocess
import sys

STEPS = [
    "audio_alpha.build_manifest",
    "audio_alpha.preprocess_audio",
    "audio_alpha.segment_audio",
    "audio_alpha.extract_features",
    "audio_alpha.score_audio_sentiment",
    "audio_alpha.aggregate_call_features",
]


def run_pipeline() -> None:
    for module in STEPS:
        cmd = [sys.executable, "-m", module]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_pipeline()
