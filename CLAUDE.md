# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

AI 对话系统 — 一个本地运行的全栈语音交互平台。流水线：VAD → ASR → LLM → TTS。此为毕业设计项目，处于活跃早期开发阶段（2026 年 4 月）。

**无身份认证** — 所有 API 端点完全开放。无 API 密钥、JWT 或认证中间件。在生产部署前必须添加身份认证。

## 常用命令

### 前端 (`typescript-frontend/`)

| Action | Command |
|---|---|
| Dev server (port 3000) | `npm run dev` |
| Full build (typecheck + vite) | `npm run build` |
| Typecheck only | `npm run typecheck` |
| Lint (zero warnings enforced) | `npm run lint` |

Vite 开发服务器将 `/v1/*`（REST + WebSocket）代理到 `localhost:8000`。

### 后端 (`python-backend/`)

| Action | Command |
|---|---|
| Dev server (port 8000) | `uvicorn server.main:app` |
| With auto-reload | `uvicorn server.main:app --reload` |
| All tests | `pytest` |
| Single test | `pytest tests/test_xxx.py::test_func -v` |
| Inference engine tests | `pytest tests/test_inference_*.py -v` |
| Install (editable) | `pip install -e .` |
| Install (dev) | `pip install -e ".[dev]"` |

配置：Python `>=3.11`，[`pyproject.toml`](python-backend/pyproject.toml) 中 `pytest-asyncio` 设为 `asyncio_mode = "auto"`。测试通过 [`tests/conftest.py`](python-backend/tests/conftest.py) 提供四个异步 fixture：`tmp_config`、`app_state`、`app`、`client`（httpx `AsyncClient`）。所有异步测试使用 `@pytest.mark.anyio`。conftest.py 通过 `sys.path.insert(0, "../src")` 确保 server 包可导入；`app` fixture 手动构建 FastAPI（不含 CORS 和信号量中间件），比 `create_app()` 更精简。

## 架构

### 后端：应用工厂 + 集中式运行时

[`main.py`](python-backend/src/server/main.py) 导出 `app = create_app()`。工厂序列：

1. `build_config()` 读取 `AI_SERVER_*` 环境变量，填充平台感知默认值（`db_path`、`media_dir`、`models_dir`）
2. 构造 `AppState` — 中央运行时容器，持有所有 `Store` 实例 + 按模型变体名索引的 `dict[str, InferenceEngine]`
3. Lifespan 并行初始化所有 Store（`asyncio.gather`），然后种子化模型元数据
4. 中间件：CORS（可配置来源）、request-ID 追踪、并发信号量（可配置上限，超时返回 503）
5. 异常处理器：`ApiError` → 结构化 JSON，未处理 `Exception` → 脱敏 500

`AppState` 管理引擎生命周期：`load_engine(variant)` 从模型目录查找条目，按 `(category, runtime)` 解析引擎类，实例化后调用 `await engine.load()`。`auto_load_engine(variant)` 仅在模型已 `downloaded` 时才加载。对外部 API 引擎（`runtime == "external"`），`auto_load_engine()` 跳过本地目录检查。

### 后端：推理引擎插件系统

[`services/inference/__init__.py`](python-backend/src/server/services/inference/__init__.py) 维护 `_ENGINE_REGISTRY: dict[(category, runtime), type[InferenceEngine]]`。`@register_engine(category, runtime)` 装饰器在导入时自动注册引擎类。

基类层次（[`base.py`](python-backend/src/server/services/inference/base.py)）：

| ABC | 关键抽象方法 |
|-----|-------------|
| `VadEngine` | `detect(audio, sample_rate) → VadResult`、`process_chunk(chunk, cache_state)` |
| `AsrEngine` | `transcribe(audio, sample_rate) → AsrResult` |
| `LlmEngine` | `generate(messages, stream, max_tokens, temperature, top_p) → ChatResult` |
| `TtsEngine` | `list_voices()`、`synthesize(text, voice, speed) → AudioResult` |

已注册引擎：

| 引擎文件 | (category, runtime) | 模型 | 依赖 |
|---------|---------------------|------|------|
| `engine_fire_red_vad.py` | `(vad, onnx)` | FireRedVAD | 硬依赖 |
| `engine_kokoro_tts.py` | `(tts, onnx)` | Kokoro-82M | 硬依赖 |
| `engine_qwen3_llm_onnx.py` | `(llm, onnx)` | Qwen3 0.6B ONNX | 硬依赖 |
| `engine_qwen3_asr.py` | `(asr, onnx+gguf)` | Qwen3-ASR 0.6B/1.7B | 可选 |
| `engine_qwen35_llm.py` | `(llm, gguf)` | Qwen3.5 0.8B GGUF | 可选 |
| `engine_qwen3_tts.py` | `(tts, gguf)` | Qwen3-TTS GGUF | 可选 |
| `engine_external_api_llm.py` | `(llm, external)` | 兼容 OpenAI 的 API | 可选 |

三个硬依赖始终导入（VAD、ONNX LLM、Kokoro TTS）。四个可选引擎使用 `try/except` 导入——缺失依赖时记录警告但不阻止服务器启动。所有推理调用通过 `asyncio.to_thread()` 分派以避免阻塞事件循环。

`ExternalApiLlmEngine` 使用 `openai` SDK 委托外部 API。连接参数（`base_url`、`api_key`、`model`、`reasoning_effort`）通过 `**api_config` 按请求传递。包含 SSRF 保护（`_validate_base_url` 阻止私有/环回/链路本地 IP）。支持 `reasoning_content`（思考过程）流式传输。

**添加新引擎**：(1) 创建带 `@register_engine(category, runtime)` 装饰器的文件；(2) 在 [`seed_models.py`](python-backend/src/server/services/seed_models.py) 中添加匹配的种子模型条目。

### 后端：Store 模式

每个领域 Store 继承 `BaseStore`（[`stores/base.py`](python-backend/src/server/stores/base.py)），共享一个 SQLAlchemy 异步引擎（SQLite，WAL 模式，外键 ON，3s busy timeout）。不使用 Alembic 迁移——模式变更通过 DDL 中的 `IF NOT EXISTS` 或 `ModelStore._MIGRATIONS` 中的临时 `ALTER TABLE` 处理。

| Store | 文件 | 用途 |
|-------|------|------|
| `ModelStore` | `stores/model_store.py` | 模型目录（生命周期、变体、下载进度） |
| `ChatStore` | `stores/chat_store.py` | 聊天线程 + 消息 |
| `TranscriptionStore` | `stores/transcription_store.py` | ASR 转写历史 |
| `SpeechHistoryStore` | `stores/speech_history_store.py` | TTS 生成历史 |
| `VoiceStore` | `stores/voice_store.py` | 语音配置文件（get_or_create 模式） |
| `VoiceObservationStore` | `stores/voice_observation_store.py` | 每个配置文件的观察记忆 |
| `SavedVoiceStore` | `stores/saved_voice_store.py` | 已保存/自定义语音 |
| `OnboardingStore` | `stores/onboarding_store.py` | 单行入门标志（`CHECK(id=1)`） |

**Store 约定**：时间戳均使用 ISO 8601 UTC（`"%Y-%m-%dT%H:%M:%SZ"`）；UUID 使用 `uuid.uuid4().hex`（32 位十六进制，无连字符）。`OnboardingStore` DDL 使用 `INSERT OR IGNORE` + `CHECK(id = 1)` 实现单行自举。

### 后端：Router 工厂

13 个路由模块各自导出 `create_router(state: AppState) -> APIRouter`。[`api/router.py`](python-backend/src/server/api/router.py) 调用每个工厂并以 `/v1/...` 前缀挂载。实时语音 WebSocket 使用独立的 `register_ws(app, state)`。

| 路由模块 | 前缀 | 描述 |
|---------|------|------|
| `admin/models.py` | `/v1/admin/models` | 模型目录 CRUD，加载/卸载，下载 SSE |
| `chat/threads.py` | `/v1/chat/threads` | 聊天线程 CRUD |
| `chat/completions.py` | `/v1/chat/completions` | LLM 补全 (POST) |
| `audio/transcriptions.py` | `/v1/audio/transcriptions` | ASR 转写 (POST) |
| `audio/speech.py` | `/v1/audio/speech` | TTS 合成 (POST) |
| `audio/vad.py` | `/v1/audio/vad` | VAD 检测 (POST) |
| `transcriptions/handlers.py` | `/v1/transcriptions` | 转写历史 CRUD |
| `tts_history/handlers.py` | `/v1/text-to-speech-generations` | TTS 历史 CRUD |
| `voice/profile.py` | `/v1/voice/profile` | 语音配置文件 CRUD |
| `voice/observations.py` | `/v1/voice/observations` | 观察记忆 CRUD |
| `voices/handlers.py` | `/v1/voices` | 已保存语音 CRUD |
| `agent/handlers.py` | `/v1/agent` | Agent 会话轮次 |
| `onboarding/handlers.py` | `/v1/onboarding` | 入门状态 |

[`api/dependencies.py`](python-backend/src/server/api/dependencies.py) 提供 FastAPI 依赖函数，但路由处理器主要通过 `create_router` 闭包直接访问 `AppState`，因此很少使用。

### 后端：语音实时流水线

`/v1/voice/realtime/ws` 实现状态机流水线：

```
WebSocket → AudioRingBuffer → VAD (逐帧) → ASR → LLM (流式) → TTS (逐句) → WebSocket
```

**状态机**（[`api/voice/state_machine.py`](python-backend/src/server/api/voice/state_machine.py)）：`IDLE→LISTENING→PROCESSING→SPEAKING→LISTENING`（循环）。合法转换：`IDLE→LISTENING`、`LISTENING→{PROCESSING,IDLE}`、`PROCESSING→{SPEAKING,LISTENING,IDLE}`、`SPEAKING→{LISTENING,IDLE}`。`transition()` 非法转换返回 False；`force_transition()` 用于中断/错误恢复。使用 `asyncio.Lock` 保证原子性，`on_change(callback)` 广播状态变更。

**IVWS 协议**（[`api/voice/ivws_protocol.py`](python-backend/src/server/api/voice/ivws_protocol.py)）：24 字节固定头（Magic `"IVWS"`、Version `uint8`、Kind `uint8`、Reserved 18B）+ PCM16 小端音频。常量：`SAMPLE_RATE=16000`、`CHUNK_SIZE=160`（10ms）。24kHz→16kHz 重采样使用多相 FIR 滤波器（Kaiser 窗）防止混叠，而非简单线性插值。

**组件**：[`api/voice/ring_buffer.py`](python-backend/src/server/api/voice/ring_buffer.py)（线程安全环形缓冲区，30s 容量）、[`api/voice/vad_stream_helper.py`](python-backend/src/server/api/voice/vad_stream_helper.py)（FireRedVAD 流式包装器，可配置阈值/帧数）。

**每个引擎独立降级**：当某引擎未加载时，该阶段降级（TTS 使用模拟正弦波，ASR/LLM 使用预设文本），不影响其他阶段。

**LLM 流式桥接**：[`realtime.py`](python-backend/src/server/api/voice/realtime.py) 使用 `asyncio.Queue` + `asyncio.to_thread()` 将同步 LLM 生成器桥接到异步事件循环。**TTS 逐句合成**：完整文本按句子边界拆分，逐句合成，首句尽早发送以降低首帧延迟。

**WebSocket 会话默认值**（`handle_websocket()` 闭包初始化）：VAD→`FireRedVad-onnx`、ASR→`Qwen3-ASR-0.6B-gguf`、LLM→`Qwen3.5-0.8B.Q8_0`、TTS→`Qwen3-TTS-1.7B-VoiceDesign-gguf`；系统提示词为中文友好助手；TTS 说话人 `Vivian`、语速 `1.0`；VAD 阈值 `0.5`、最小语音 `400ms`、静音超时 `800ms`、最大话语 `20s`。

**WebSocket 错误代码**（服务器 → 客户端 JSON `{"type": "error", "code": "...", "message": "..."}`）：

| 代码 | 触发条件 |
|------|---------|
| `api_key_missing` | 外部 API 调用时 `api_key` 为空 |
| `api_error` | 外部 API 调用时 openai SDK 异常 |
| `asr_failed` | ASR 转录重试后仍失败 |
| `vad_unavailable` | `input_stream_start` 时无 VAD 引擎可用 |
| `pipeline_error` | `run_turn()` 中的未处理异常 |

**模型种子化**（[`services/seed_models.py`](python-backend/src/server/services/seed_models.py)）——每次启动时种子化 10 个模型（幂等 upsert）：

| 变体 | 类别 | 运行时 |
|------|------|--------|
| `Qwen3.5-0.8B.Q8_0` | llm | gguf |
| `Qwen3.5-0.8B.Q4_K_M` | llm | gguf |
| `FireRedVad-onnx` | vad | onnx |
| `Kokoro-82M-v1.1-zh-ONNX-q4` | tts | onnx |
| `Qwen3-TTS-0.6B-CustomVoice-gguf` | tts | gguf |
| `Qwen3-TTS-1.7B-CustomVoice-gguf` | tts | gguf |
| `Qwen3-TTS-1.7B-VoiceDesign-gguf` | tts | gguf |
| `Qwen3-ASR-0.6B-gguf` | asr | onnx+gguf |
| `Qwen3-ASR-1.7B-gguf` | asr | onnx+gguf |
| `Qwen3-0.6B-onnx` | llm | onnx |
| `external-api` | llm | external |

`external-api` 种子条目使用 `repo_id = "external/api"`（非真实 ModelScope 仓库），`seed_models()` 跳过其文件列表请求。

启动时 `seed_models()` 额外逻辑：检查模型目录是否存在，若缺失则重置 `downloaded`/`ready`/`error`→`not_downloaded`；迁移旧条目的 `runtime` 字段；清理不在 `SEED_MODELS` 中的废弃条目（仅删除 `not_downloaded`/`error` 状态，保护已下载数据）。每次 ModelScope API 调用间 `asyncio.sleep(0.5)` 避免限流。

### 前端：数据流

- **服务端状态**：TanStack React Query 5，通过 `src/api/` 模块访问。基础客户端（[`client.ts`](typescript-frontend/src/api/client.ts)）提供 `apiFetch<T>()`（REST，自动添加 `/v1` 前缀）和 `apiFetchSSE<T>()`（SSE 流，含 AbortController 支持）
- **客户端状态**：Zustand 5 stores，位于 [`stores/`](typescript-frontend/src/stores/)

| Store | 文件 | localStorage 键 | 用途 |
|-------|------|-------------------|------|
| `useThemeStore` | `theme-store.ts` | `ai-conversation.theme` | 浅色/深色/系统主题 |
| `useVoiceStore` | `voice-store.ts` | — | 会话状态、VAD 配置、对话气泡 |
| `useModelStore` | `model-store.ts` | — | 按变体键控的下载进度 |
| `useConversationSettings` | `conversation-settings-store.ts` | `ai-conversation.settings` | 全本地对话的 LLM/TTS/VAD/ASR 设置 |
| `useConversationApiSettings` | `conversation-api-settings-store.ts` | `ai-conversation.api-settings` | 外部 API 对话设置 + TTS/VAD/ASR |
| `useBackgroundStore` | `background-store.ts` | `ai-conversation.background` | 背景预设/自定义颜色/图像 |

**双轨设置架构**：`/conversation`（全本地，4 个模型本地运行）和 `/conversation-API`（混合：VAD/ASR/TTS 本地，LLM 委托外部 API）。两条路径共享相同的 `VoiceRealtimeClient` 和 Voice 组件。

Vite 全局常量 `__APP_VERSION__`（由 `npm_package_version` 注入）用于侧边栏版本号显示。`useThemeStore` 手动读写 localStorage（未使用 Zustand persist 中间件），其余持久化 store 统一使用 Zustand `persist()` + `createJSONStorage()`。

- **路由**：React Router 7（[`app/router.tsx`](typescript-frontend/src/app/router.tsx)），`@/` → `./src/`

| 路径 | 组件 | 描述 |
|------|------|------|
| `/` | Redirect | → `/conversation` |
| `/conversation` | `ConversationPage` | 实时语音对话（全本地） |
| `/conversation-API` | `ConversationApiPage` | 实时语音对话 + 外部 LLM API |
| `/chat` | `ChatPage` | 与 LLM 的文字聊天 |
| `/speech-to-text` | `SpeechToTextPage` | ASR 转写 |
| `/text-to-speech` | `TextToSpeechPage` | TTS 合成 |
| `/vad` | `VadPage` | VAD 测试 |
| `/models` | `ModelsPage` | 模型目录管理 |
| `*` | `NotFoundPage` | 404 |

- **UI**：Tailwind CSS 3（`darkMode: ['class']`，浅色主题通过 `<html class="theme-light">` 切换）+ Radix UI 原语。`@layer base, components, utilities` 结构。10 个关键帧动画（`orb-pulse`、`orb-spin`、`fade-in-up`、`fade-in`、`scale-in`、`slide-up`、`blink`、`shimmer`、`slide-in-right`、`breathe`），列表交错类 `.stagger-1`~`stagger-8`。组件工具类：`.glass-panel`、`.glass-panel-strong`、`.gradient-text`、`.gradient-border-card`、`.markdown-body`、`.chat-bubble-user`/`.chat-bubble-assistant`、`.scrollbar-thin`、`.skeleton`。

### 前端：Voice 组件与 WebSocket 客户端

[`VoiceRealtimeClient`](typescript-frontend/src/api/voice-realtime.ts) — 实时语音 WebSocket 客户端：IVWS 二进制帧编解码、`ScriptProcessorNode` 麦克风采集（16kHz 单声道，回声消除 + 降噪）、`AudioContext` 播放队列、10s 心跳 + 15s 超时重连（3 次指数退避）、RMS 音量电平。

| 组件 | 文件 | 用途 |
|------|------|------|
| `VoiceOrb` | `voice-orb.tsx` | 动画渐变球体（按状态着色 + 缩放 + 旋转） |
| `VolumeVisualizer` | `volume-visualizer.tsx` | Canvas 音量计（20 条带 lerp 平滑） |
| `StatusBar` | `status-bar.tsx` | 连接指示器、状态标签、模型徽章、延迟 |
| `ConversationStream` | `conversation-stream.tsx` | 可滚动气泡列表（自动滚动 + Markdown 渲染） |
| `SettingsPanel` | `settings-panel.tsx` | LLM/TTS/VAD/ASR 设置抽屉 |
| `ApiSettingsSheet` | `api-settings-sheet.tsx` | 外部 API 设置抽屉（专用于 `/conversation-API`） |
| `VoiceControlBar` | `voice-control-bar.tsx` | 开始/停止/静音/中断控制 |

**ConversationPage 生命周期**：挂载时并行预加载 4 个模型（`POST /v1/admin/models/{variant}/load`）→ 创建 WebSocket 连接 → 发送 `session_start` + `input_stream_start` → `useEffect` 同步设置变化 → 卸载时 `disconnect()`。

**ConversationApiPage 生命周期**：类似，但 LLM 变体固定为 `"external-api"`，`api_config` 携带 `base_url`/`api_key`/`model`/`reasoning_effort`，设置同步含 300ms 防抖。

## 关键目录

```
python-backend/src/server/
  main.py              # 应用工厂 + uvicorn 入口
  config.py            # ServeConfig（AI_SERVER_ 环境前缀）
  app_state.py         # DI 容器：store + 引擎注册表 + 生命周期
  error_handlers.py    # ApiError + FastAPI 异常处理器
  api/                 # 14 个路由模块 + voice/ 子包（WebSocket 流水线）
  stores/              # 8 个 SQLite store + base.py
  services/
    inference/         # 引擎注册表 + 7 个引擎实现 + ASR/TTS/LLM 子包
    model_downloader.py
    seed_models.py

typescript-frontend/src/
  main.tsx             # 入口
  index.css            # 全局样式 + CSS 变量 + 动画 + 组件类
  app/                 # Provider、主题、路由器
  pages/               # 8 个页面组件
  components/          # ui/（shadcn 原语）、voice/、chat/、speech/、tts/、vad/
  stores/              # 6 个 Zustand stores
  hooks/               # React Query hooks（查询键工厂）+ 工具 hooks
  api/                 # REST/SSE 客户端 + voice-realtime.ts
  lib/                 # cn() 辅助函数、model-metadata
```

## 当前实现状态

所有 7 个推理引擎均具有**真实实现**。当引擎未加载时，API 端点回退到每个引擎独立的模拟响应（非全局降级）：

- `POST /v1/chat/completions` — 真实 ONNX/GGUF 推理，否则模拟文本
- `POST /v1/audio/speech` — 真实 Kokoro/Qwen3-TTS，否则 440Hz 正弦波
- `POST /v1/audio/transcriptions` — 真实 Qwen3-ASR，否则硬编码文本
- `POST /v1/audio/vad` — 真实 FireRedVAD，否则返回错误
- `WS /v1/voice/realtime/ws` — 真实 VAD→ASR→LLM→TTS 流水线，每个引擎独立降级。支持双轨 LLM（本地引擎 或 `ExternalApiLlmEngine`）

始终可用（SQLite 支撑）：聊天线程、转写历史、语音配置文件、模型目录管理的 CRUD 端点，以及带 SSE 进度的 ModelScope 下载。Agent 会话是内存中的桩实现，未由 SQLite 支撑。

## 注意事项

- 所有繁重计算（推理、下载）必须在 `asyncio.to_thread()` 中运行
- ONNX Runtime 和 llama.cpp 带有冲突的 OpenMP 运行时。`main.py` 在任何导入前设置 `KMP_DUPLICATE_LIB_OK=TRUE` 以防止 OMP Error #15 崩溃
- 模型文件（`.gguf`、`.safetensors`、`.onnx`）和 `python-backend/models/` 被 git 忽略——运行时通过 ModelScope 下载
- 无身份认证——生产部署前必须添加
- `ExternalApiLlmEngine` 包含 SSRF 保护（阻止 `localhost`/`127.0.0.1`/`::1`/`0.0.0.0` 及私有/环回/链路本地 IP）
- [`lib/model-metadata.ts`](typescript-frontend/src/lib/model-metadata.ts) 必须与后端 `SEED_MODELS` 列表保持同步
- `services/__init__.py` 为空；真正的引擎导出在 `services/inference/__init__.py` 中
- 不使用 Alembic——模式变更直接在 Store 的 `_ddl()` 中进行
