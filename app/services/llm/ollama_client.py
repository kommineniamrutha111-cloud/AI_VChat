"""Ollama chat client abstraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, List

import requests


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    message: ChatMessage
    raw: dict[str, Any]


class OllamaClient:
    """Wrapper around Ollama HTTP or python-sdk client."""

    def __init__(self, base_url: str, model: str, summary_prompt: str | None = None, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._summary_prompt = summary_prompt
        self._session = requests.Session()
        self._timeout = timeout

    def chat(self, messages: Iterable[ChatMessage], stream: bool = False) -> ChatResponse:
        """Trigger a chat completion call to the Ollama HTTP API."""
        payload = {
            "model": self._model,
            "messages": [message.__dict__ for message in messages],
            "stream": stream,
        }
        url = f"{self._base_url}/api/chat"

        if stream:
            return self._stream_chat(url, payload)

        response = self._session.post(url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        message_data = data.get("message") or {}
        message = ChatMessage(role=message_data.get("role", "assistant"), content=message_data.get("content", ""))
        return ChatResponse(message=message, raw=data)

    def summarise(self, transcript: str) -> str:
        """Generate a concise summary for the provided transcript."""
        if not transcript.strip():
            return "Transcript is empty; nothing to summarise."

        prompt = self._summary_prompt or "Provide a concise summary for the provided transcript."
        messages: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content="You are an assistant that creates concise, student-friendly study notes from lecture transcripts.",
            ),
            ChatMessage(role="user", content=f"{prompt}\n\nTranscript:\n{transcript}"),
        ]
        response = self.chat(messages)
        return response.message.content.strip()

    def _stream_chat(self, url: str, payload: dict[str, Any]) -> ChatResponse:
        """Handle streaming chat responses by aggregating event chunks."""
        with self._session.post(url, json=payload, timeout=self._timeout, stream=True) as resp:
            resp.raise_for_status()
            combined_content = []
            final_message: dict[str, Any] | None = None
            for line in resp.iter_lines():
                if not line:
                    continue
                payload = json.loads(line.decode("utf-8"))
                message = payload.get("message")
                if message and message.get("content"):
                    combined_content.append(message["content"])
                    final_message = message

        if final_message is None:
            raise RuntimeError("Ollama streaming response did not return a message.")

        final_message["content"] = "".join(combined_content)
        chat_message = ChatMessage(role=final_message.get("role", "assistant"), content=final_message["content"])
        return ChatResponse(message=chat_message, raw=final_message)

