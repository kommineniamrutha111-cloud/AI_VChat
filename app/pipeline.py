"""Pipeline coordinator orchestrating video processing stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import AppConfig, load_config
from app.services.audio.ffmpeg_extractor import AudioExtractor
from app.services.ingestion.base import BaseVideoIngestionService, IngestionResult
from app.services.ingestion.local import LocalVideoResolver
from app.services.ingestion.youtube import YouTubeDownloader
from app.services.llm.ollama_client import OllamaClient
from app.services.transcription.whisper_service import WhisperService
from app.storage.database import get_session, init_db
from app.storage.repositories import ChatRepository, TranscriptRepository, VideoRepository


@dataclass
class PipelineResult:
    video_id: int
    transcript_id: int
    summary: str
    audio_path: Path
    transcript_path: Path | None
    chat_session_id: int


class PipelineCoordinator:
    """High-level orchestrator for ingestion → transcription → summarisation."""

    def __init__(
        self,
        config: AppConfig | None = None,
        downloader: BaseVideoIngestionService | None = None,
        audio_extractor: AudioExtractor | None = None,
        whisper: WhisperService | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self._config = config or load_config()
        self._downloader = downloader
        self._audio_extractor = audio_extractor
        self._whisper = whisper
        self._ollama_client = ollama_client

        init_db()

    def process(self, source: str, mode: Literal["url", "file_path"] = "url") -> PipelineResult:
        """Coordinate ingestion, transcription, and summarisation."""
        ingestion_service = self._resolve_downloader(mode)
        ingestion_result = ingestion_service.ingest(source)

        audio_result = self._get_audio_extractor().extract(
            ingestion_result.video_path,
            self._config.paths.audio_dir,
            codec="mp3",
        )

        transcription = self._get_whisper_service().transcribe(
            audio_result.audio_path,
            transcripts_dir=self._config.paths.transcripts_dir,
            generate_srt=True,
        )

        summary = self._get_ollama_client().summarise(transcription.text)

        with get_session() as session:
            video_repo = VideoRepository(session)
            transcript_repo = TranscriptRepository(session)
            chat_repo = ChatRepository(session)

            video = video_repo.get_by_source(
                ingestion_result.source_id,
                video_path=str(ingestion_result.video_path),
            )
            if video is None:
                video = video_repo.create(ingestion_result)

            transcript = transcript_repo.save(
                video,
                transcription,
                summary=summary,
                audio_path=audio_result.audio_path,
            )
            chat_repo.upsert_summary_session(video, summary)
            chat_session = chat_repo.create_session(video)

            return PipelineResult(
                video_id=video.id,
                transcript_id=transcript.id,
                summary=summary,
                audio_path=audio_result.audio_path,
                transcript_path=transcription.srt_path,
                chat_session_id=chat_session.id,
            )

    def _resolve_downloader(self, mode: Literal["url", "file_path"]) -> BaseVideoIngestionService:
        if self._downloader:
            return self._downloader

        if mode == "url":
            return YouTubeDownloader(self._config.paths.downloads_dir, self._config.paths.thumbnails_dir)
        if mode == "file_path":
            return LocalVideoResolver(self._config.paths.downloads_dir)
        raise ValueError(f"Unsupported mode: {mode}")

    def _get_audio_extractor(self) -> AudioExtractor:
        if not self._audio_extractor:
            self._audio_extractor = AudioExtractor()
        return self._audio_extractor

    def _get_whisper_service(self) -> WhisperService:
        if not self._whisper:
            self._whisper = WhisperService(
                models_dir=self._config.paths.models_dir,
                model_size=self._config.whisper.model_size,
                device_preference=self._config.whisper.device_preference,
                compute_dtype=self._config.whisper.compute_dtype,
            )
        return self._whisper

    def _get_ollama_client(self) -> OllamaClient:
        if not self._ollama_client:
            self._ollama_client = OllamaClient(
                base_url=self._config.ollama.base_url,
                model=self._config.ollama.model,
                summary_prompt=self._config.ollama.summary_prompt,
            )
        return self._ollama_client

    def get_ollama_client(self) -> OllamaClient:
        """Expose the Ollama client for downstream components (e.g., chat UI)."""
        return self._get_ollama_client()
