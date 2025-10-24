"""Whisper transcription service wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
import whisper

from app.utils.filesystem import ensure_parents


@dataclass
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: List[TranscriptSegment]
    language: str | None
    srt_path: Path | None = None


class WhisperService:
    """Encapsulates Whisper model loading and inference."""

    def __init__(
        self,
        models_dir: Path,
        model_size: str = "small",
        device_preference: str = "auto",
        compute_dtype: str = "float16",
    ) -> None:
        self._models_dir = models_dir
        self._model_size = model_size
        self._device_preference = device_preference
        self._compute_dtype = compute_dtype
        self._model = None  # Lazy loaded Whisper model instance
        self._device: str | None = None
        os.environ.setdefault("WHISPER_CACHE_DIR", str(models_dir))

    def transcribe(
        self,
        audio_path: Path,
        transcripts_dir: Path,
        generate_srt: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio and optionally create an SRT file."""
        model = self._load_model()
        device = self._resolve_device()
        fp16 = device == "cuda" and self._compute_dtype == "float16"

        result = model.transcribe(str(audio_path), fp16=fp16)
        raw_segments = result.get("segments") or []
        segments = [
            TranscriptSegment(
                index=i + 1,
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=(segment.get("text") or "").strip(),
            )
            for i, segment in enumerate(raw_segments)
        ]

        ensure_parents(transcripts_dir / "placeholder.txt")
        srt_path = None
        if generate_srt:
            srt_path = transcripts_dir / f"{audio_path.stem}.srt"
            self.write_srt(segments, srt_path)

        return TranscriptionResult(
            text=result.get("text") or "",
            segments=segments,
            language=result.get("language"),
            srt_path=srt_path,
        )

    def _load_model(self):
        if self._model is None:
            download_root = str(self._models_dir)
            device = self._resolve_device()
            self._model = whisper.load_model(self._model_size, download_root=download_root, device=device)
        return self._model

    def _resolve_device(self) -> str:
        if self._device:
            return self._device

        if self._device_preference == "cuda":
            device = "cuda"
        elif self._device_preference == "cpu":
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device
        return device

    @staticmethod
    def write_srt(segments: Iterable[TranscriptSegment], output_path: Path) -> None:
        """Write segments to an SRT file."""
        ensure_parents(output_path)

        def _format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        with output_path.open("w", encoding="utf-8") as handle:
            for segment in segments:
                handle.write(f"{segment.index}\n")
                handle.write(f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}\n")
                handle.write(f"{segment.text}\n\n")
