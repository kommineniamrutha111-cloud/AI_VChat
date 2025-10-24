"""Abstract interfaces for video ingestion services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class IngestionResult:
    """Standardised output for ingestion services."""

    video_path: Path
    thumbnail_path: Path | None = None
    title: str = ""
    duration_seconds: float | None = None
    source_id: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseVideoIngestionService(ABC):
    """Convenience base class for ingestion services."""

    @abstractmethod
    def ingest(self, source: str) -> IngestionResult:
        """Download or resolve video and return metadata."""
        raise NotImplementedError

