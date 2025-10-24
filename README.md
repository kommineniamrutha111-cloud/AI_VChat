# AI Video Chat

Local-first Streamlit application that downloads long-form videos, extracts audio, transcribes them with Whisper, and lets you chat with the resulting transcript using an Ollama model.

## Features
- Ingest via YouTube URL (yt-dlp) or local file path.
- Extract audio with FFmpeg, transcribe with Whisper (GPU-aware).
- Persist metadata, transcripts, and chat history in SQLite.
- Streamlit chat interface with optional time-window filtering.
- Modular service layer to swap out tooling (Whisper/Ollama/etc.).

## Prerequisites
- Python 3.10+ (matching your CUDA-enabled PyTorch build).
- FFmpeg available on your `PATH`.
- Ollama running locally with at least one chat model (`ollama list`).
- NVIDIA drivers + CUDA runtime compatible with PyTorch (e.g. CUDA ≥ 12.1 for RTX 2060).

## Quick Start

```pwsh
# 1. Create & activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install GPU-enabled PyTorch first (adjust cu121 to match nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/streamlit_app.py
```

### Optional Environment Overrides

| Variable | Description | Default |
| --- | --- | --- |
| `AIVC_DATA_ROOT` | Root folder for downloads/transcripts/audio | `<project>/data` |
| `AIVC_WHISPER_MODEL` | Whisper model size (`small`, `base`, `medium`, …) | `small` |
| `AIVC_WHISPER_DEVICE` | Force Whisper device (`auto`, `cpu`, `cuda`) | `auto` |
| `AIVC_OLLAMA_MODEL` | Ollama model name | `llama3.1:8b` |
| `AIVC_DB_URL` | SQLAlchemy connection string | `sqlite:///data/aivc.db` |

## Project Structure

```
app/
  config.py                  # Settings & paths
  pipeline.py                # Orchestrates ingestion → transcription → summary
  streamlit_app.py           # Frontend entrypoint
  services/                  # Modular service layer (ingestion, LLM, Whisper, etc.)
  storage/                   # SQLAlchemy models + repositories
  utils/                     # Filesystem helpers
data/                        # Downloads, audio, transcripts, thumbnails
docs/                        # Architecture notes & design docs
scripts/                     # Utility scripts and smoke tests
```
## 🏗️ Architecture

![architecture](https://raw.githubusercontent.com/kommineniamrutha111-cloud/AI_VChat/refs/heads/main/PROJ_IMAGES/Architecture.png)

## Running Notes
- Whisper models are cached under `<project>/models`. Delete that folder to force a re-download.
- First-time transcriptions can take several minutes depending on GPU and video length.
- SQLite database lives in `data/aivc.db`; delete it to reset history.
- Use the sidebar to reload previously processed videos without re-running the pipeline.


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

# Local Setup Guide

This project is designed to run entirely on your workstation using a Python virtual environment, GPU-accelerated Whisper, and a local Ollama model. Follow the steps below to get everything running smoothly.

## 1. Create & Activate the Virtual Environment

```pwsh
python -m venv venv
.\venv\Scripts\activate
```

> Keep your project code **outside** of the `venv/` directory. Only Python packages live there.

## 2. Install GPU-Enabled PyTorch

Check your GPU driver compatibility:

```pwsh
nvidia-smi
```

If the output shows `CUDA Version: 13.0` (RTX 2060 example), install the matching PyTorch build:

```pwsh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

To confirm PyTorch sees your GPU:

```pwsh
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

## 3. Install Project Dependencies

```pwsh
pip install -r requirements.txt
```

This installs Streamlit, yt-dlp, openai-whisper (which will reuse the GPU torch you already installed), SQLAlchemy, and supporting libraries.

## 4. Verify FFmpeg & Ollama

- `ffmpeg -version` should print version info. Add FFmpeg to `PATH` if it does not.
- `ollama list` should show models like `llama3.1:8b`. Start the Ollama service if needed (`ollama serve`).

## 5. Run the App

```pwsh
streamlit run app/streamlit_app.py
```

The UI will open in your browser at `http://localhost:8501`. Use the sidebar to review processed videos, and the main panel to ingest new ones or chat with transcripts.

## Optional Tweaks
- Override directories or model names with environment variables (`AIVC_DATA_ROOT`, `AIVC_WHISPER_MODEL`, etc.).
- Download Whisper models into `models/` once; the service reuses cached `.pt` files.
- Delete `data/aivc.db` and `data/` subfolders to wipe history and downloads.

