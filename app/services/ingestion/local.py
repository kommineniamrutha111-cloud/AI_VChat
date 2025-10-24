"""Resolver for pre-downloaded local video files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.utils.filesystem import ensure_parents
from .base import BaseVideoIngestionService, IngestionResult


class LocalVideoResolver(BaseVideoIngestionService):
    """Validate local video paths and copy them into the project workspace."""

    def __init__(self, downloads_dir: Path, copy_to_workspace: bool = True) -> None:
        self._downloads_dir = downloads_dir
        self._copy_to_workspace = copy_to_workspace

    def ingest(self, source: str) -> IngestionResult:
        """Validate the provided local path and return ingestion metadata."""
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Video path does not exist: {source_path}")

        ensure_parents(self._downloads_dir / "placeholder.txt")
        destination_path = self._resolve_destination(source_path)

        duration = self._probe_duration(destination_path)
        return IngestionResult(
            video_path=destination_path,
            thumbnail_path=None,
            title=destination_path.stem,
            duration_seconds=duration,
            source_id=str(source_path),
            extra={"original_path": str(source_path)},
        )

    def _resolve_destination(self, source_path: Path) -> Path:
        if not self._copy_to_workspace:
            return source_path

        destination = (self._downloads_dir / source_path.name).resolve()
        if destination != source_path:
            shutil.copy2(source_path, destination)
        return destination

    def _probe_duration(self, path: Path) -> float | None:
        """Use ffprobe to determine video duration when available."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        value = result.stdout.strip()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

