"""Streamlit frontend for AI Video Chat."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st

# Ensure project root is on sys.path when launched via `streamlit run app/streamlit_app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, load_config
from app.pipeline import PipelineCoordinator
from app.services.chat.conversation_manager import ConversationManager
from app.storage.database import get_session
from app.storage.repositories import ChatRepository, TranscriptRepository, VideoRepository

CONFIG: AppConfig = load_config()


def set_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AI Video Chat",
        page_icon="AI",
        layout="wide",
    )


def init_session_state() -> None:
    """Ensure session state keys exist."""
    defaults: Dict[str, Any] = {
        "pipeline": None,
        "video_id": None,
        "video_title": "",
        "video_duration": 0.0,
        "video_path": "",
        "thumbnail_path": "",
        "summary": "",
        "transcript_path": "",
        "audio_path": "",
        "chat_session_id": None,
        "summary_session_id": None,
        "time_window_enabled": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_pipeline() -> PipelineCoordinator:
    """Return a cached pipeline instance."""
    if st.session_state["pipeline"] is None:
        st.session_state["pipeline"] = PipelineCoordinator(config=CONFIG)
    return st.session_state["pipeline"]


def get_chat_sessions_snapshot(video_id: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return chat sessions (excluding summary) and optional summary session metadata."""
    with get_session() as session:
        chat_repo = ChatRepository(session)
        session_models = chat_repo.list_sessions(video_id)

        def _as_dict(session_obj: Any) -> Dict[str, Any]:
            return {
                "id": session_obj.id,
                "title": session_obj.title,
                "updated_at": session_obj.updated_at,
            }

        summary_model = next((item for item in session_models if item.title == "Summary"), None)
        summary = _as_dict(summary_model) if summary_model else None
        chats = [_as_dict(item) for item in session_models if item.title != "Summary"]
    return chats, summary


def create_new_chat_session(video_id: int) -> int:
    """Create a new chat session for the given video and return its id."""
    with get_session() as session:
        video_repo = VideoRepository(session)
        chat_repo = ChatRepository(session)

        video = video_repo.get(video_id)
        if video is None:
            raise ValueError(f"Video {video_id} not found.")

        new_session = chat_repo.create_session(video)
        return new_session.id


def fetch_chat_history(session_id: int) -> list[dict[str, str]]:
    """Load chat history for display."""
    with get_session() as session:
        chat_repo = ChatRepository(session)
        messages = chat_repo.list_messages(session_id)
    return [{"role": message.role, "content": message.content} for message in messages]


def render_sidebar() -> None:
    """Display sidebar with configuration info and recent videos."""
    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            f"""
            - Data Root: `{CONFIG.paths.data_root}`
            - Whisper Model: `{CONFIG.whisper.model_size}`
            - Ollama Model: `{CONFIG.ollama.model}`
            """
        )

        st.subheader("Recent Videos")
        with get_session() as session:
            video_repo = VideoRepository(session)
            recent = video_repo.list_recent(limit=10)
            recent_options = [
                (f"{video.title} (#{video.id})", video.id)
                for video in recent
            ]

        options = dict(recent_options)
        if not options:
            st.caption("No videos processed yet.")
        else:
            selection = st.selectbox("Load existing video:", ["-- Select --", *options.keys()])
            if selection != "-- Select --":
                load_video_state(options[selection])

        current_video_id = st.session_state.get("video_id")
        if current_video_id:
            st.subheader("Chats")
            sessions, summary = get_chat_sessions_snapshot(current_video_id)
            st.session_state["summary_session_id"] = summary["id"] if summary else None

            if st.button("+ New Chat", key="new_chat_btn"):
                new_session_id = create_new_chat_session(current_video_id)
                st.session_state["chat_session_id"] = new_session_id
                st.experimental_rerun()

            if sessions:
                options_map = {session["title"]: session["id"] for session in sessions}
                option_labels = list(options_map.keys())
                option_values = list(options_map.values())
                current_session_id = st.session_state.get("chat_session_id")
                default_index = option_values.index(current_session_id) if current_session_id in option_values else 0
                chosen_label = st.selectbox(
                    "Chat history:",
                    option_labels,
                    index=default_index,
                    key="chat_session_select",
                )
                st.session_state["chat_session_id"] = options_map[chosen_label]
            else:
                st.caption("No chats yet. Start a new conversation.")


def handle_summarise(mode: str, source: str) -> None:
    """Run the pipeline and refresh session state."""
    pipeline = get_pipeline()
    try:
        with st.spinner("Processing video... this may take a while for longer videos."):
            result = pipeline.process(source.strip(), mode="url" if mode == "YouTube URL" else "file_path")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to process the video: {exc}")
        return

    st.success("Video processed successfully.")
    st.session_state["chat_session_id"] = result.chat_session_id
    load_video_state(result.video_id)

    # Cache current run outputs for quick access
    st.session_state["summary"] = result.summary
    st.session_state["transcript_path"] = str(result.transcript_path) if result.transcript_path else ""
    st.session_state["audio_path"] = str(result.audio_path)


def load_video_state(video_id: int) -> None:
    """Populate session state with video metadata and available chat sessions."""
    with get_session() as session:
        video_repo = VideoRepository(session)
        transcript_repo = TranscriptRepository(session)
        chat_repo = ChatRepository(session)

        video = video_repo.get(video_id)
        if video is None:
            st.error(f"Video with id {video_id} was not found.")
            return

        transcript = transcript_repo.latest(video.id)
        sessions = chat_repo.list_sessions(video.id)

        summary_session = next((item for item in sessions if item.title == "Summary"), None)
        chat_sessions = [item for item in sessions if item.title != "Summary"]

        if not chat_sessions:
            new_session = chat_repo.create_session(video)
            chat_sessions.append(new_session)

        video_data = {
            "id": video.id,
            "title": video.title,
            "duration_seconds": float(video.duration_seconds or 0.0),
            "video_path": video.video_path,
            "thumbnail_path": video.thumbnail_path or "",
        }
        transcript_data = {
            "summary": transcript.summary if transcript and transcript.summary else "",
            "srt_path": transcript.srt_path if transcript and transcript.srt_path else "",
            "audio_path": transcript.audio_path if transcript and transcript.audio_path else "",
        }
        summary_session_id = summary_session.id if summary_session else None
        valid_session_ids = [session_item.id for session_item in chat_sessions]
        desired_session_id = st.session_state.get("chat_session_id")
        active_session_id = (
            desired_session_id if desired_session_id in valid_session_ids else chat_sessions[0].id
        )

    st.session_state["video_id"] = video_data["id"]
    st.session_state["video_title"] = video_data["title"]
    st.session_state["video_duration"] = video_data["duration_seconds"]
    st.session_state["video_path"] = video_data["video_path"]
    st.session_state["thumbnail_path"] = video_data["thumbnail_path"]
    st.session_state["summary"] = (
        transcript_data["summary"] or st.session_state.get("summary", "")
    )
    st.session_state["transcript_path"] = transcript_data["srt_path"]
    st.session_state["audio_path"] = transcript_data["audio_path"] or st.session_state.get("audio_path", "")
    st.session_state["summary_session_id"] = summary_session_id
    st.session_state["chat_session_id"] = active_session_id


def render_ingestion_form() -> None:
    """Render inputs for submitting a new video."""
    st.subheader("1. Ingest Video")

    mode = st.radio("Source Type", ["YouTube URL", "Local File Path"], horizontal=True, key="source_mode")
    placeholder = "https://www.youtube.com/watch?v=..." if mode == "YouTube URL" else r"C:\path\to\video.mp4"
    source = st.text_input("Video URL / Path", placeholder=placeholder, key="video_source")

    if st.button("Summarise Video", type="primary"):
        if not source.strip():
            st.warning("Please provide a valid URL or file path.")
            return
        handle_summarise(mode, source)


def render_video_overview() -> None:
    """Show summary and metadata for the active video."""
    if st.session_state["video_id"] is None:
        st.info("Process a video to see its summary and transcript.")
        return

    st.subheader("2. Video Summary")
    cols = st.columns([1, 3])
    with cols[0]:
        if st.session_state["thumbnail_path"] and Path(st.session_state["thumbnail_path"]).exists():
            st.image(st.session_state["thumbnail_path"], use_container_width=True)
    with cols[1]:
        st.markdown(f"**Title:** {st.session_state['video_title']}")
        duration = st.session_state["video_duration"]
        if duration:
            st.markdown(f"**Duration:** {duration / 60:.1f} minutes")
        st.markdown("**Summary:**")
        st.write(st.session_state["summary"] or "_Summary not available yet._")

        if st.session_state["transcript_path"]:
            transcript_path = Path(st.session_state["transcript_path"])
            if transcript_path.exists():
                st.download_button(
                    "Download Transcript (SRT)",
                    data=transcript_path.read_bytes(),
                    file_name=transcript_path.name,
                    mime="application/x-subrip",
                )


def render_chat_interface() -> None:
    """Display chat history and input for follow-up questions."""
    if st.session_state["video_id"] is None:
        return

    st.subheader("3. Chat With The Video")

    enable_window = st.checkbox("Filter by time window", value=st.session_state["time_window_enabled"])
    st.session_state["time_window_enabled"] = enable_window
    time_window: Tuple[float, float] | None = None
    duration = st.session_state["video_duration"]
    if enable_window and duration:
        start, end = st.slider(
            "Select time range (seconds)",
            min_value=0.0,
            max_value=float(duration),
            value=(0.0, min(120.0, float(duration))),
            step=1.0,
        )
        time_window = (start, end)
    elif enable_window:
        st.warning("Video duration unknown; time filtering disabled.")

    st.markdown("#### Conversation")
    session_id = st.session_state.get("chat_session_id")
    if session_id is None:
        st.info("Create a new chat from the sidebar to start asking questions.")
        return

    history = fetch_chat_history(session_id)
    prompt = st.chat_input("Ask a question about the video")
    if not history and prompt is None:
        st.caption("No messages yet. Ask your first question.")
    for message in history:
        with st.chat_message(message['role']):
            st.write(message['content'])
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        respond_to_prompt(prompt, time_window)


def respond_to_prompt(prompt: str, time_window: Tuple[float, float] | None) -> None:
    """Handle chat submission by invoking the conversation manager."""
    video_id = st.session_state["video_id"]
    session_id = st.session_state.get("chat_session_id")
    if session_id is None:
        st.warning("Create a new chat before asking questions.")
        return

    pipeline = get_pipeline()
    ollama_client = pipeline.get_ollama_client()

    with get_session() as session:
        transcript_repo = TranscriptRepository(session)
        chat_repo = ChatRepository(session)

        conversation_manager = ConversationManager(
            ollama_client=ollama_client,
            transcript_repo=transcript_repo,
            chat_repo=chat_repo,
        )

        with st.spinner("Thinking..."):
            turn = conversation_manager.ask(video_id, session_id, prompt, time_window=time_window)

    with st.chat_message("assistant"):
        st.write(turn.assistant_message)


def render_header() -> None:
    """Render the top-level header."""
    st.title("AI Video Chat")
    st.caption("Summarise long-form videos and converse with their transcripts using local Whisper + Ollama.")


def main() -> None:
    """Streamlit application entrypoint."""
    set_page_config()
    init_session_state()
    render_sidebar()
    render_header()
    render_ingestion_form()
    render_video_overview()
    render_chat_interface()


if __name__ == "__main__":
    main()
