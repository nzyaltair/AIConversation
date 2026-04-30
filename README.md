# AI Conversation

A full-stack voice AI conversation platform running entirely on local inference — graduation project (April 2026, under active development).

**Pipeline**: `VAD → ASR → LLM → TTS` — microphone input to speech output, all locally.

## Features

- **Real-time Voice Chat** — WebSocket-driven bidirectional voice interaction with interruption support and state-machine-managed conversation flow
- **Text Chat** — Threaded conversations with Markdown rendering, thinking process display, and SSE streaming
- **Speech-to-Text (ASR)** — Upload or record audio, transcribe to text; supports Chinese and English
- **Text-to-Speech (TTS)** — Multiple voices available (Kokoro, Qwen3-TTS)
- **Voice Activity Detection (VAD)** — Dedicated debug page with real-time statistics and timeline visualization
- **Model Management** — Browse, download, load/unload models with SSE-based progress tracking
- **External API Proxy** — Optionally route LLM through any OpenAI-compatible API (supports `reasoning_content` streaming)
- **Dark/Light Theme** — Full theming system driven by CSS variables

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Browser (React)                         │
│  Mic → ScriptProcessor → IVWS WebSocket → AudioContext    │
└──────────────────────┬───────────────────────────────────┘
                       │ WebSocket + REST (/v1/*)
┌──────────────────────▼───────────────────────────────────┐
│               FastAPI Backend (Python)                    │
│                                                          │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐              │
│  │ VAD  │ → │ ASR  │ → │ LLM  │ → │ TTS  │              │
│  │onnx  │   │onnx+ │   │gguf/ │   │onnx/ │              │
│  │      │   │gguf  │   │onnx  │   │gguf  │              │
│  └──────┘   └──────┘   └──────┘   └──────┘              │
│                                                          │
│  Graceful degradation per stage: mock fallback when      │
│  engine is not loaded                                    │
└──────────────────────────────────────────────────────────┘
```

## Tech Stack

### Frontend

| Technology | Notes |
|------------|-------|
| React 18 + TypeScript 5.6 | UI framework |
| Vite 6 | Build tool, dev server on port 3000 |
| Tailwind CSS 3.4 | Utility-first CSS + CSS variable theming |
| Zustand 5 | Client-side state management |
| TanStack React Query 5 | Server state & caching |
| Radix UI | Accessible UI primitives |
| react-router-dom 7 | Client-side routing |
| framer-motion | Animations |

### Backend

| Technology | Notes |
|------------|-------|
| Python 3.11+ + FastAPI | Web framework |
| SQLAlchemy 2.0 + aiosqlite | Async database, SQLite with WAL mode |
| ONNX Runtime | VAD, Kokoro-TTS, Qwen3-0.6B inference |
| llama-cpp-python | GGUF LLM & TTS inference |
| ModelScope SDK | Model downloading |
| Pydantic v2 + orjson | Validation & serialization |

### Inference Engines

| Engine | Model | Runtime |
|--------|-------|---------|
| FireRedVAD | FireRedVad | onnx |
| Qwen3-ASR | Qwen3-ASR 0.6B / 1.7B | onnx+gguf |
| Qwen3.5 LLM | Qwen3.5 0.8B (Q8_0 / Q4_K_M) | gguf |
| Qwen3 LLM | Qwen3 0.6B | onnx |
| Kokoro TTS | Kokoro-82M (Chinese) | onnx |
| Qwen3-TTS | Qwen3-TTS 0.6B / 1.7B | gguf |
| External API | Any OpenAI-compatible API | external |

## Quick Start

### Prerequisites

- **Python** >= 3.11
- **Node.js** >= 18
- **Git**

> Windows users who want CUDA acceleration: install NVIDIA CUDA Toolkit 12.x and set the `CUDA_PATH` environment variable.

### 1. Clone

```bash
git clone <repo-url>
cd AIConversation
```

### 2. Start Backend

```bash
cd python-backend

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS / Linux)
source .venv/bin/activate

# Install dependencies
pip install -e .

# Start dev server (port 8000, auto-reload)
uvicorn server.main:app --reload
```

### 3. Start Frontend

```bash
cd typescript-frontend

# Install dependencies
npm install

# Start dev server (port 3000, proxies /v1 to backend)
npm run dev
```

### 4. Download Models

Open `http://localhost:3000/models` and download the models you need. At minimum, 4 models are required for the full real-time voice pipeline:

- FireRedVad-onnx (VAD)
- Qwen3-ASR-0.6B-gguf (ASR)
- Qwen3.5-0.8B.Q8_0 or Qwen3-0.6B-onnx (LLM)
- Kokoro-82M-v1.1-zh-ONNX-q4 or any Qwen3-TTS model (TTS)

Once downloaded, go to `/conversation` and click start to try real-time voice chat.

## Configuration

### Backend Environment Variables

All configuration is injected via environment variables (prefix `AI_SERVER_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_SERVER_HOST` | `0.0.0.0` | Bind address |
| `AI_SERVER_PORT` | `8000` | HTTP port |
| `AI_SERVER_DB_PATH` | Platform data dir | SQLite database path |
| `AI_SERVER_MEDIA_DIR` | Platform data dir | Media file storage |
| `AI_SERVER_MODELS_DIR` | `<project>/models/` | Model file directory |
| `AI_SERVER_CORS_ORIGINS` | `["http://localhost:3000", ...]` | Allowed CORS origins |
| `AI_SERVER_MAX_CONCURRENT_REQUESTS` | `32` | Max concurrent requests |
| `AI_SERVER_REQUEST_TIMEOUT_SECS` | `300` | Request timeout (seconds) |

### External API Mode

Visit `/conversation-API` to use an external LLM API (OpenAI-compatible) while VAD, ASR, and TTS remain local. Configure in the settings panel:

- Base URL (API endpoint)
- API Key
- Model name
- Reasoning Effort (optional, for reasoning models)

## Project Structure

```
AIConversation/
├── python-backend/
│   ├── pyproject.toml
│   ├── src/server/
│   │   ├── main.py              # FastAPI app factory
│   │   ├── config.py            # Configuration (env vars)
│   │   ├── app_state.py         # Runtime container (stores + engines)
│   │   ├── api/                 # Route modules (14 total)
│   │   │   └── voice/           # Real-time voice WebSocket pipeline
│   │   ├── stores/              # SQLite stores (8 domains)
│   │   └── services/
│   │       ├── inference/       # Engine registry + 7 engine implementations
│   │       └── seed_models.py   # Model catalog seeding at startup
│   └── tests/                   # pytest suite
├── typescript-frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── app/                 # App shell (router, providers, theme)
│       ├── pages/               # Page components (8 routes)
│       ├── components/          # UI components (voice/chat/tts/vad)
│       ├── stores/              # Zustand stores (5)
│       ├── hooks/               # React Query hooks
│       ├── api/                 # REST/SSE/WebSocket client
│       └── lib/                 # Utilities, model metadata
└── models/                      # Model files (gitignored)
```

## API Overview

All endpoints are prefixed with `/v1/`:

| Prefix | Description |
|--------|-------------|
| `/v1/admin/models` | Model catalog management, download, load/unload |
| `/v1/chat/threads` | Chat thread CRUD |
| `/v1/chat/completions` | LLM completions (POST, SSE streaming supported) |
| `/v1/audio/transcriptions` | ASR speech-to-text |
| `/v1/audio/speech` | TTS text-to-speech |
| `/v1/audio/vad` | Voice activity detection |
| `/v1/transcriptions` | Transcription history CRUD |
| `/v1/text-to-speech-generations` | TTS generation history CRUD |
| `/v1/voice/profile` | Voice profile |
| `/v1/voice/observations` | Observation memory |
| `/v1/voices` | Saved voices CRUD |
| `/v1/agent` | Agent sessions |
| `/v1/onboarding` | Onboarding status |
| `/v1/voice/realtime/ws` | Real-time voice chat WebSocket |
| `/health` | Health check |

## Notes

- **No authentication** — All API endpoints are fully open. No API keys, JWT, or auth middleware. Must be added before production deployment.
- **Model downloads** — Model files (`.gguf`, `.onnx`, `.safetensors`) are not in the Git repo; they are downloaded via ModelScope. Expect 2-5 GB disk usage on first launch.
- **OpenMP conflict** — ONNX Runtime and llama.cpp use conflicting OpenMP runtimes. The backend sets `KMP_DUPLICATE_LIB_OK=TRUE` in `main.py` to resolve this, no manual intervention needed.
- **Windows CUDA** — The backend auto-scans `CUDA_PATH` and `C:\Program Files\NVIDIA GPU Computing Toolkit` at startup to add CUDA runtime DLLs to the search path.
- **No Alembic** — Schema changes are handled via `IF NOT EXISTS` DDL directly in Store classes.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---

中文文档：[README.zh-CN.md](README.zh-CN.md)
