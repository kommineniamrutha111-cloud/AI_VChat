# Local Setup Guide

This project is designed to run entirely on your workstation using a Python virtual environment, GPU-accelerated Whisper, and a local Ollama model. Follow the steps below to get everything running smoothly.

## 1. Create & Activate the Virtual Environment

```pwsh
python -m venv venv
.\venv\Scripts\activate
```

> 📝 Keep your project code **outside** of the `venv/` directory. Only Python packages live there.

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

