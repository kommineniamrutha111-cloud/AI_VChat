"""Conversation orchestration for chat-with-video experience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from app.services.llm.ollama_client import ChatMessage, OllamaClient
from app.services.transcription.whisper_service import TranscriptSegment
from app.storage.repositories import ChatRepository, TranscriptRepository


@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str


class ConversationManager:
    """Coordinates conversation flow and persistence hooks."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        transcript_repo: TranscriptRepository,
        chat_repo: ChatRepository,
    ) -> None:
        self._ollama = ollama_client
        self._transcripts = transcript_repo
        self._chats = chat_repo

    def ask(
        self,
        video_id: int,
        session_id: int,
        prompt: str,
        time_window: Tuple[float, float] | None = None,
    ) -> ConversationTurn:
        """Retrieve relevant transcript context, call Ollama, and persist the exchange."""
        start, end = (time_window if time_window else (None, None))
        segments = self._transcripts.get_segments(video_id, start=start, end=end)
        if not segments:
            full_text = self._transcripts.latest_text(video_id)
            segments = []
        else:
            full_text = None

        context = self._compose_context(segments, fallback_text=full_text, window=time_window)

        history = self._chats.list_messages(session_id)
        truncated_history = history[-10:]  # prevent oversized context windows

        user_prompt = self._build_prompt(prompt, context, time_window)

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are an AI assistant that answers questions based on a video's transcript. "
                    "Use only the provided context and clearly acknowledge when information is unavailable."
                ),
            ),
            *truncated_history,
            ChatMessage(role="user", content=user_prompt),
        ]

        response = self._ollama.chat(messages)
        answer = response.message.content.strip()
        self._chats.append_turn(session_id, video_id, prompt, answer)

        return ConversationTurn(user_message=prompt, assistant_message=answer)

    @staticmethod
    def _compose_context(
        segments: Iterable[TranscriptSegment],
        fallback_text: str | None,
        window: Tuple[float, float] | None,
    ) -> str:
        if segments:
            formatted = []
            for segment in segments:
                formatted.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
            return "\n".join(formatted)

        if fallback_text:
            return fallback_text

        if window:
            return f"No transcript segments were found between {window[0]}s and {window[1]}s."
        return "No transcript data was available for this video."

    @staticmethod
    def _build_prompt(prompt: str, context: str, window: Tuple[float, float] | None) -> str:
        window_text = ""
        if window:
            window_text = f"\nTime window: {window[0]:.2f}s to {window[1]:.2f}s."
        return (
            "Context:\n"
            f"{context}\n\n"
            "Instructions: Answer the question using only the context above. "
            "If the answer is missing, explain what additional information would be required."
            f"{window_text}\n\n"
            f"Question: {prompt}"
        )
