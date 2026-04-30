# AI Conversation

本地运行的全栈语音 AI 对话平台 — 毕业设计项目（2026 年 4 月，活跃开发中）。

**流水线**：`VAD → ASR → LLM → TTS`，从麦克风输入到语音输出，全链路本地推理。

## 功能特性

- **实时语音对话** — WebSocket 双向流式语音交互，支持打断和状态机驱动对话管理
- **文字聊天** — 线程化对话，支持 Markdown 渲染、思考过程展示和 SSE 流式响应
- **语音识别 (ASR)** — 上传或录制音频，转写为文本，支持中英文
- **语音合成 (TTS)** — 文本转语音，多种音色可选（Kokoro、Qwen3-TTS）
- **语音活动检测 (VAD)** — 独立调试页面，实时统计和时间线可视化
- **模型管理** — 浏览、下载、加载/卸载模型，SSE 实时下载进度
- **外部 API 代理** — LLM 可切换为兼容 OpenAI 的外部 API（支持 reasoning_content 流式）
- **深色/浅色主题** — CSS 变量驱动的完整主题系统

## 技术架构

```
┌──────────────────────────────────────────────────────────┐
│                    浏览器 (React)                         │
│  麦克风 → ScriptProcessor → IVWS WebSocket → AudioContext │
└──────────────────────┬───────────────────────────────────┘
                       │ WebSocket + REST (/v1/*)
┌──────────────────────▼───────────────────────────────────┐
│                 FastAPI 后端 (Python)                     │
│                                                          │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐              │
│  │ VAD  │ → │ ASR  │ → │ LLM  │ → │ TTS  │              │
│  │onnx  │   │onnx+ │   │gguf/ │   │onnx/ │              │
│  │      │   │gguf  │   │onnx  │   │gguf  │              │
│  └──────┘   └──────┘   └──────┘   └──────┘              │
│                                                          │
│  每阶段独立降级：引擎未加载时回退到模拟响应               │
└──────────────────────────────────────────────────────────┘
```

## 技术栈

### 前端

| 技术 | 说明 |
|------|------|
| React 18 + TypeScript 5.6 | UI 框架 |
| Vite 6 | 构建工具，开发服务器端口 3000 |
| Tailwind CSS 3.4 | 原子化 CSS + CSS 变量主题 |
| Zustand 5 | 客户端状态管理 |
| TanStack React Query 5 | 服务端状态与缓存 |
| Radix UI | 无障碍 UI 原语 |
| react-router-dom 7 | 客户端路由 |
| framer-motion | 动画 |

### 后端

| 技术 | 说明 |
|------|------|
| Python 3.11+ + FastAPI | Web 框架 |
| SQLAlchemy 2.0 + aiosqlite | 异步数据库，SQLite + WAL 模式 |
| ONNX Runtime | VAD、Kokoro-TTS、Qwen3-0.6B 推理 |
| llama-cpp-python | GGUF 格式 LLM、TTS 推理 |
| ModelScope SDK | 模型下载 |
| Pydantic v2 + orjson | 数据校验与序列化 |

### 推理引擎

| 引擎 | 模型 | 运行时 |
|------|------|--------|
| FireRedVAD | FireRedVad | onnx |
| Qwen3-ASR | Qwen3-ASR 0.6B / 1.7B | onnx+gguf |
| Qwen3.5 LLM | Qwen3.5 0.8B (Q8_0 / Q4_K_M) | gguf |
| Qwen3 LLM | Qwen3 0.6B | onnx |
| Kokoro TTS | Kokoro-82M (中文) | onnx |
| Qwen3-TTS | Qwen3-TTS 0.6B / 1.7B | gguf |
| 外部 API | 兼容 OpenAI 的任意 API | external |

## 快速开始

### 环境要求

- **Python** >= 3.11
- **Node.js** >= 18
- **Git**（用于拉取代码）

> Windows 用户如需 CUDA 加速，需安装 NVIDIA CUDA Toolkit 12.x 并设置 `CUDA_PATH` 环境变量。

### 1. 克隆仓库

```bash
git clone <repo-url>
cd AIConversation
```

### 2. 启动后端

```bash
cd python-backend

# 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 安装依赖
pip install -e .

# 启动开发服务器（端口 8000，支持热重载）
uvicorn server.main:app --reload
```

### 3. 启动前端

```bash
cd typescript-frontend

# 安装依赖
npm install

# 启动开发服务器（端口 3000，自动代理 /v1 到后端）
npm run dev
```

### 4. 下载模型

打开浏览器访问 `http://localhost:3000/models`，在模型管理页面下载需要的模型。至少需要下载 4 个模型才能运行完整的实时语音对话：

- FireRedVad-onnx（VAD）
- Qwen3-ASR-0.6B-gguf（ASR）
- Qwen3.5-0.8B.Q8_0 或 Qwen3-0.6B-onnx（LLM）
- Kokoro-82M-v1.1-zh-ONNX-q4 或任意 Qwen3-TTS 模型（TTS）

模型下载后，在 `/conversation` 页面点击开始即可体验实时语音对话。

## 配置

### 后端环境变量

所有配置通过环境变量注入（前缀 `AI_SERVER_`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AI_SERVER_HOST` | `0.0.0.0` | 绑定地址 |
| `AI_SERVER_PORT` | `8000` | HTTP 端口 |
| `AI_SERVER_DB_PATH` | 平台数据目录 | SQLite 数据库路径 |
| `AI_SERVER_MEDIA_DIR` | 平台数据目录 | 媒体文件存储 |
| `AI_SERVER_MODELS_DIR` | 项目根 `models/` | 模型文件目录 |
| `AI_SERVER_CORS_ORIGINS` | `["http://localhost:3000", ...]` | 允许的跨域来源 |
| `AI_SERVER_MAX_CONCURRENT_REQUESTS` | `32` | 最大并发请求数 |
| `AI_SERVER_REQUEST_TIMEOUT_SECS` | `300` | 请求超时（秒） |

### 外部 API 模式

访问 `/conversation-API` 可使用外部 LLM API（兼容 OpenAI 协议），VAD、ASR、TTS 仍由本地引擎处理。在页面设置中填入：

- Base URL（API 地址）
- API Key
- Model（模型名称）
- Reasoning Effort（可选，推理强度）

## 项目结构

```
AIConversation/
├── python-backend/
│   ├── pyproject.toml
│   ├── src/server/
│   │   ├── main.py              # FastAPI 应用工厂
│   │   ├── config.py            # 配置管理（环境变量）
│   │   ├── app_state.py         # 运行时容器（stores + 引擎）
│   │   ├── api/                 # 路由模块（14 个）
│   │   │   └── voice/           # 实时语音 WebSocket 流水线
│   │   ├── stores/              # SQLite Store（8 个领域）
│   │   └── services/
│   │       ├── inference/       # 推理引擎注册表 + 7 个引擎实现
│   │       └── seed_models.py   # 启动时种子化模型目录
│   └── tests/                   # pytest 测试
├── typescript-frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── app/                 # 应用壳（router、providers、主题）
│       ├── pages/               # 页面组件（8 个路由）
│       ├── components/          # UI 组件（voice/chat/tts/vad）
│       ├── stores/              # Zustand stores（5 个）
│       ├── hooks/               # React Query hooks
│       ├── api/                 # REST/SSE/WebSocket 客户端
│       └── lib/                 # 工具函数、模型元数据
└── models/                      # 模型文件（gitignore）
```

## API 概览

所有接口以 `/v1/` 为前缀。完整端点列表：

| 前缀 | 说明 |
|------|------|
| `/v1/admin/models` | 模型目录管理、下载、加载/卸载 |
| `/v1/chat/threads` | 聊天线程 CRUD |
| `/v1/chat/completions` | LLM 补全（POST，支持流式 SSE） |
| `/v1/audio/transcriptions` | ASR 语音转文字 |
| `/v1/audio/speech` | TTS 文字转语音 |
| `/v1/audio/vad` | VAD 语音活动检测 |
| `/v1/transcriptions` | 转写历史 CRUD |
| `/v1/text-to-speech-generations` | TTS 生成历史 CRUD |
| `/v1/voice/profile` | 语音配置文件 |
| `/v1/voice/observations` | 观察记忆 |
| `/v1/voices` | 已保存音色 CRUD |
| `/v1/agent` | Agent 会话 |
| `/v1/onboarding` | 新手引导 |
| `/v1/voice/realtime/ws` | 实时语音对话 WebSocket |
| `/health` | 健康检查 |

## 注意事项

- **无身份认证** — 所有 API 端点完全开放，不包含 API 密钥、JWT 或认证中间件。生产部署前务必添加。
- **模型下载** — 模型文件（`.gguf`、`.onnx`、`.safetensors`）不在 Git 仓库中，需通过 ModelScope 下载。首次启动至少需要 2-5 GB 磁盘空间。
- **OpenMP 冲突** — ONNX Runtime 和 llama.cpp 使用不同的 OpenMP 运行时。后端已在 `main.py` 中设置 `KMP_DUPLICATE_LIB_OK=TRUE` 解决此问题，无需手动处理。
- **Windows CUDA** — 后端启动时会自动扫描 `CUDA_PATH` 和 `C:\Program Files\NVIDIA GPU Computing Toolkit`，将 CUDA 运行时 DLL 加入搜索路径。
- **SQLite 迁移** — 不使用 Alembic，DDL 变更直接在 Store 中通过 `IF NOT EXISTS` 处理。

## 许可证

待定。

---

English version: [README.md](README.md)
