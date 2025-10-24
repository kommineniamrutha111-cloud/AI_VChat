# Project Architecture Plan

## Objectives
- Streamlit single-page app providing two ingestion modes: YouTube URL or local video path.
- Backend pipeline handles download, audio extraction, transcription, summarisation, and conversational QA.
- Modular components to swap out implementations (yt-dlp, ffmpeg, whisper, Ollama).
- Persist chat history in SQLite for each processed video.
- Run entirely inside local virtual environment; leverage GPU when available.

## High-Level Flow
1. **Frontend (Streamlit)**
   - Input controls: ingestion mode selector (`url` | `file_path`), URL/path field, process button.
   - Display: video metadata (title, thumbnail), transcript summary, chat conversation thread, follow-up prompt box.
   - State: manage session id, transcript cache, chat history retrieval/sync with SQLite.

2. **Backend Pipeline**
   - `VideoIngestionService`: orchestrates either `YouTubeDownloader` or `LocalVideoResolver`.
   - `AudioExtractor`: wraps FFmpeg command generation/execution, outputs `.wav`/`.mp3`.
   - `TranscriptionService`: wraps Whisper invocation; supports GPU/CPU fallback, SRT generation.
   - `LLMService`: wraps Ollama chat interface, with model selection & prompt templating.
   - `ConversationManager`: manages transcript chunks, temporal lookup, persistence.

3. **Persistence Layer**
   - SQLite via SQLModel/SQLAlchemy for conversations, transcripts metadata, user queries.
   - File system storage for downloaded assets under configurable `data/` root (videos, audio, transcripts, SRT).

## Module Boundaries
- `app/streamlit_app.py`: UI entrypoint.
- `app/services/ingestion/`
  - `base.py`: abstract interfaces.
  - `youtube.py`: `YouTubeDownloader`.
  - `local.py`: `LocalVideoResolver`.
- `app/services/audio/ffmpeg_extractor.py`: audio extraction.
- `app/services/transcription/whisper_service.py`: transcription logic + SRT writer.
- `app/services/llm/ollama_client.py`: chat & summary wrappers.
- `app/services/chat/conversation_manager.py`: handles Q/A, timeline queries, caching.
- `app/storage/`:
  - `database.py`: SQLite session factory.
  - `models.py`: ORM models for videos, transcripts, chat messages.
- `app/config.py`: settings (paths, model names) using `pydantic` or simple dataclass.
- `app/utils/filesystem.py`: helpers for paths, cleanup.

## Data Directories
- `data/downloads/`: raw video files from yt-dlp or local copies.
- `data/audio/`: extracted audio files.
- `data/transcripts/`: JSON/SRT transcripts.
- `data/thumbnails/`: video thumbnails.

## Key Interactions
- `Streamlit` triggers `PipelineCoordinator.process(input)` which:
  1. Resolves/Downloads video → returns metadata & file path.
  2. Extracts audio → returns audio path.
  3. Transcribes audio → returns transcript text + segments.
  4. Stores metadata/transcript in DB and disk.
  5. Generates summary via `LLMService`.
- Conversational follow-ups send user query + transcript to LLM, store responses, support range queries (use segments metadata to filter relevant text).

## GPU Considerations
- Torch installation pinned to CUDA-enabled wheel.
- Whisper service check `torch.cuda.is_available()`; load model with `device="cuda"` when true else `"cpu"`.
- Optionally allow configuration of Whisper model size per request.

## Testing & Tooling
- Provide CLI smoke test script (`scripts/smoke_test.py`) for pipeline components (mock shorter audio).
- Use `pytest` for unit tests of services (mock external dependencies).
- Add `pre-commit` config for formatting (optional).

