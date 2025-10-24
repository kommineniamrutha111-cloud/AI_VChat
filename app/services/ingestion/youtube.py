"""YouTube ingestion powered by yt-dlp."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests
from yt_dlp import YoutubeDL

from app.utils.filesystem import ensure_parents
from .base import BaseVideoIngestionService, IngestionResult

_LOGGER = logging.getLogger(__name__)


class YouTubeDownloader(BaseVideoIngestionService):
    """Download a YouTube video using yt-dlp."""

    def __init__(self, downloads_dir: Path, thumbnails_dir: Path, session: requests.Session | None = None) -> None:
        self._downloads_dir = downloads_dir
        self._thumbnails_dir = thumbnails_dir
        self._session = session or requests.Session()

    def ingest(self, source: str) -> IngestionResult:
        """Download the video and return local metadata."""
        ensure_parents(self._downloads_dir / "dummy.txt")
        ensure_parents(self._thumbnails_dir / "dummy.txt")

        ydl_opts = {
            "outtmpl": str(self._downloads_dir / "%(id)s.%(ext)s"),
            "restrictfilenames": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "writesubtitles": False,
            "writethumbnail": False,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=True)

        requested_downloads: list[dict[str, Any]] = info.get("requested_downloads") or []
        if not requested_downloads:
            raise RuntimeError("yt-dlp did not provide download metadata.")

        download_info = requested_downloads[0]
        video_path = Path(download_info["filepath"]).resolve()

        thumbnail_path = self._fetch_thumbnail(info)

        duration = self._safe_float(info.get("duration"))
        result = IngestionResult(
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            title=info.get("title") or video_path.stem,
            duration_seconds=duration,
            source_id=info.get("id"),
            extra={"yt_info": info},
        )
        return result

    def _fetch_thumbnail(self, info: dict[str, Any]) -> Path | None:
        url = info.get("thumbnail")
        video_id = info.get("id") or "thumbnail"
        if not url:
            return None

        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            _LOGGER.warning("Failed to download thumbnail for %s: %s", video_id, exc)
            return None

        extension = Path(url).suffix or ".jpg"
        destination = (self._thumbnails_dir / f"{video_id}{extension}").resolve()
        ensure_parents(destination)
        destination.write_bytes(response.content)
        return destination

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

