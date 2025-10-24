"""Repository helpers for database persistence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import Session

from app.services.ingestion.base import IngestionResult
from app.services.llm.ollama_client import ChatMessage
from app.services.transcription.whisper_service import TranscriptSegment, TranscriptionResult
from app.storage import models


class VideoRepository:
    """CRUD helpers for `Video` entries."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, ingestion: IngestionResult) -> models.Video:
        video = models.Video(
            source_id=ingestion.source_id,
            title=ingestion.title or ingestion.video_path.stem,
            duration_seconds=ingestion.duration_seconds,
            video_path=str(ingestion.video_path),
            thumbnail_path=str(ingestion.thumbnail_path) if ingestion.thumbnail_path else None,
        )
        self._session.add(video)
        self._session.flush()
        return video

    def get_by_source(self, source_id: str | None, video_path: str) -> models.Video | None:
        if not source_id:
            stmt: Select = select(models.Video).where(models.Video.video_path == video_path)
        else:
            stmt = select(models.Video).where(models.Video.source_id == source_id)
        return self._session.execute(stmt).scalars().first()

    def get(self, video_id: int) -> models.Video | None:
        stmt: Select = select(models.Video).where(models.Video.id == video_id)
        return self._session.execute(stmt).scalars().first()

    def list_recent(self, limit: int = 10) -> list[models.Video]:
        stmt: Select = (
            select(models.Video)
            .order_by(models.Video.created_at.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars().all())


class TranscriptRepository:
    """Persist and query transcripts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(
        self,
        video: models.Video,
        transcription: TranscriptionResult,
        summary: str | None = None,
        audio_path: Path | None = None,
    ) -> models.Transcript:
        transcript = models.Transcript(
            video=video,
            language=transcription.language,
            text=transcription.text,
            srt_path=str(transcription.srt_path) if transcription.srt_path else None,
            summary=summary,
            audio_path=str(audio_path) if audio_path else None,
        )
        self._session.add(transcript)
        self._session.flush()

        segment_models = [
            models.TranscriptSegmentModel(
                transcript=transcript,
                index=segment.index,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
            for segment in transcription.segments
        ]
        self._session.add_all(segment_models)
        self._session.flush()
        return transcript

    def get_segments(self, video_id: int, start: float | None = None, end: float | None = None) -> List[TranscriptSegment]:
        stmt: Select = (
            select(models.TranscriptSegmentModel)
            .join(models.Transcript)
            .where(models.Transcript.video_id == video_id)
            .order_by(models.TranscriptSegmentModel.index)
        )
        if start is not None:
            stmt = stmt.where(models.TranscriptSegmentModel.end >= start)
        if end is not None:
            stmt = stmt.where(models.TranscriptSegmentModel.start <= end)

        rows = self._session.execute(stmt).scalars().all()
        return [
            TranscriptSegment(
                index=row.index,
                start=row.start,
                end=row.end,
                text=row.text,
            )
            for row in rows
        ]

    def latest_text(self, video_id: int) -> str:
        stmt = (
            select(models.Transcript.text)
            .where(models.Transcript.video_id == video_id)
            .order_by(models.Transcript.created_at.desc())
            .limit(1)
        )
        result = self._session.execute(stmt).scalar_one_or_none()
        return result or ""

    def latest_summary(self, video_id: int) -> str | None:
        stmt = (
            select(models.Transcript.summary)
            .where(models.Transcript.video_id == video_id)
            .order_by(models.Transcript.created_at.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def latest(self, video_id: int) -> models.Transcript | None:
        stmt = (
            select(models.Transcript)
            .where(models.Transcript.video_id == video_id)
            .order_by(models.Transcript.created_at.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalars().first()


class ChatRepository:
    """Persist chat messages per video."""

    def __init__(self, session: Session) -> None:
        self._session = session
    def create_session(self, video: models.Video, title: str | None = None) -> models.ChatSession:
        session = models.ChatSession(
            video=video,
            title=title or self._default_session_title(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._session.add(session)
        self._session.flush()
        return session

    def upsert_summary_session(self, video: models.Video, summary: str) -> models.ChatSession:
        stmt: Select = select(models.ChatSession).where(
            models.ChatSession.video_id == video.id,
            models.ChatSession.title == "Summary",
        )
        session = self._session.execute(stmt).scalar_one_or_none()
        if session is None:
            session = self.create_session(video, title="Summary")
        else:
            # Wipe previous summary messages
            self._session.execute(
                delete(models.ChatMessage).where(models.ChatMessage.session_id == session.id)
            )
        self._session.flush()
        self.add_assistant_message(session.id, video.id, summary, turn_index=1)
        session.updated_at = datetime.utcnow()
        self._session.flush()
        return session

    def list_sessions(self, video_id: int) -> List[models.ChatSession]:
        stmt: Select = (
            select(models.ChatSession)
            .where(models.ChatSession.video_id == video_id)
            .order_by(models.ChatSession.updated_at.desc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_session(self, session_id: int) -> models.ChatSession | None:
        stmt: Select = select(models.ChatSession).where(models.ChatSession.id == session_id)
        return self._session.execute(stmt).scalars().first()

    def list_messages(self, session_id: int) -> List[ChatMessage]:
        stmt: Select = (
            select(models.ChatMessage)
            .where(models.ChatMessage.session_id == session_id)
            .order_by(models.ChatMessage.turn_index, models.ChatMessage.id)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [ChatMessage(role=row.role, content=row.content) for row in rows]

    def append_turn(self, session_id: int, video_id: int, user_message: str, assistant_message: str) -> None:
        turn_index = self._next_turn_index(session_id)
        user_entry = models.ChatMessage(
            video_id=video_id,
            session_id=session_id,
            role="user",
            content=user_message,
            turn_index=turn_index,
        )
        assistant_entry = models.ChatMessage(
            video_id=video_id,
            session_id=session_id,
            role="assistant",
            content=assistant_message,
            turn_index=turn_index,
        )
        self._session.add_all([user_entry, assistant_entry])
        session = self.get_session(session_id)
        if session:
            if self._should_update_title(session, turn_index, user_message):
                session.title = self._derive_title(user_message)
            session.updated_at = datetime.utcnow()
        self._session.flush()

    def add_assistant_message(self, session_id: int, video_id: int, content: str, turn_index: int | None = None) -> None:
        if turn_index is None:
            turn_index = self._next_turn_index(session_id)
        assistant_entry = models.ChatMessage(
            video_id=video_id,
            session_id=session_id,
            role="assistant",
            content=content,
            turn_index=turn_index,
        )
        self._session.add(assistant_entry)
        session = self.get_session(session_id)
        if session:
            session.updated_at = datetime.utcnow()
        self._session.flush()

    def _next_turn_index(self, session_id: int) -> int:
        stmt = select(func.max(models.ChatMessage.turn_index)).where(models.ChatMessage.session_id == session_id)
        value = self._session.execute(stmt).scalar_one_or_none()
        return 1 if value is None else value + 1

    @staticmethod
    def _default_session_title() -> str:
        return f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"

    @staticmethod
    def _derive_title(prompt: str, max_length: int = 40) -> str:
        cleaned = prompt.strip().replace("\n", " ")
        return (cleaned[: max_length - 3] + "...") if len(cleaned) > max_length else cleaned

    @staticmethod
    def _should_update_title(session: models.ChatSession, turn_index: int, prompt: str) -> bool:
        if session.title == "Summary":
            return False
        if turn_index > 1:
            return False
        return bool(prompt.strip())
