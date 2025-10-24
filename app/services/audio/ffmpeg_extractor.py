"""Audio extraction helpers leveraging FFmpeg."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.utils.filesystem import ensure_parents


@dataclass
class AudioExtractionResult:
    audio_path: Path
    format: str
    sample_rate: int | None = None


class AudioExtractor:
    """Extract audio track from a video file using FFmpeg."""

    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        self._ffmpeg = ffmpeg_binary

    def extract(self, video_path: Path, output_dir: Path, codec: str = "mp3") -> AudioExtractionResult:
        """Extract audio using FFmpeg and return the generated file metadata."""
        if codec not in {"mp3", "wav", "flac"}:
            raise ValueError(f"Unsupported codec requested: {codec}")

        ensure_parents(output_dir / "placeholder.txt")
        output_path = (output_dir / f"{video_path.stem}.{codec}").resolve()

        command = [
            self._ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "copy" if codec != "mp3" else "libmp3lame",
            str(output_path),
        ]

        if codec == "wav":
            command = [
                self._ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                str(output_path),
            ]

        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {process.stderr.strip()}")

        return AudioExtractionResult(audio_path=output_path, format=codec)

