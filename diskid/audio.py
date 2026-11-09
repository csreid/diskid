"""ffmpeg/ffprobe helpers for pulling audio segments out of video files."""

import subprocess
from pathlib import Path


def probe_duration(video: Path) -> float:
    """Return the container duration in seconds."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def probe_title(video: Path) -> str | None:
    """Return the container's "title" format tag, if it has one."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format_tags=title",
            "-of", "csv=p=0",
            str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return out or None


def extract_audio(video: Path, start: float, duration: float, wav_out: Path) -> None:
    # -ss before -i for fast (keyframe) seeking; 16 kHz mono wav is what whisper wants
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
            str(wav_out),
        ],
        check=True,
    )
