"""
Runtime configuration objects for the AI Video Chat project.
These helpers centralise environment-derived settings (paths, models, flags).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    """Centralised filesystem locations."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_root: Path = field(init=False)
    downloads_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)
    transcripts_dir: Path = field(init=False)
    thumbnails_dir: Path = field(init=False)
    models_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        data_root_env = os.getenv("AIVC_DATA_ROOT")
        self.data_root = Path(data_root_env) if data_root_env else self.project_root / "data"
        self.downloads_dir = self.data_root / "downloads"
        self.audio_dir = self.data_root / "audio"
        self.transcripts_dir = self.data_root / "transcripts"
        self.thumbnails_dir = self.data_root / "thumbnails"
        self.models_dir = self.project_root / "models"

    def ensure(self) -> None:
        """Create required directories if they are missing."""
        for path in (
            self.downloads_dir,
            self.audio_dir,
            self.transcripts_dir,
            self.thumbnails_dir,
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class WhisperConfig:
    """Settings for the Whisper transcription service."""

    model_size: str = os.getenv("AIVC_WHISPER_MODEL", "small")
    device_preference: str = os.getenv("AIVC_WHISPER_DEVICE", "auto")  # auto|cpu|cuda
    compute_dtype: str = os.getenv("AIVC_WHISPER_DTYPE", "float16")


@dataclass
class OllamaConfig:
    """Settings controlling local Ollama usage."""

    base_url: str = os.getenv("AIVC_OLLAMA_URL", "http://localhost:11434")
    model: str = os.getenv("AIVC_OLLAMA_MODEL", "llama3.1:8b")
    summary_prompt: str = os.getenv(
        "AIVC_SUMMARY_PROMPT",
        (
            "Summarise the following transcript into concise study notes. "
            "Highlight key takeaways, timelines, and action items for students."
        ),
    )


@dataclass
class DatabaseConfig:
    """Settings for SQLite chat persistence."""

    url: str = field(default_factory=lambda: os.getenv("AIVC_DB_URL", "sqlite:///data/aivc.db"))


@dataclass
class AppConfig:
    """Aggregate configuration accessor."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    def initialise(self) -> None:
        """Perform bootstrap steps such as ensuring directories."""
        self.paths.ensure()
        os.environ.setdefault("WHISPER_CACHE_DIR", str(self.paths.models_dir))


def load_config() -> AppConfig:
    """Factory returning the configured AppConfig instance."""
    config = AppConfig()
    config.initialise()
    return config

