"""
API 路由注册中心。

集中挂载所有子路由模块，定义 API 前缀命名空间：

  /v1/admin/models              — 模型管理（下载/加载/卸载/删除/状态查询）
  /v1/chat/threads              — 聊天线程 CRUD
  /v1/chat                      — OpenAI 兼容 Chat Completions 端点
  /v1/audio                     — TTS（speech）和 ASR（transcriptions）
  /v1/transcriptions            — 转录历史记录
  /v1/text-to-speech-generations — TTS 生成历史记录
  /v1/voice                     — 语音配置文件和观察记忆
  /v1/voices                    — 保存的声音
  /v1/agent                     — Agent 会话和轮次
  /v1/onboarding                — 新手引导状态
  /v1/voice/realtime/ws         — 实时语音 WebSocket（通过 register_ws 直接挂载）
"""

from __future__ import annotations

from fastapi import FastAPI

from server.app_state import AppState
from server.api.admin.models import create_router as admin_router
from server.api.chat.threads import create_router as chat_thread_router
from server.api.chat.completions import create_router as chat_completions_router
from server.api.audio.transcriptions import create_router as audio_trans_router
from server.api.audio.speech import create_router as audio_speech_router
from server.api.audio.vad import create_router as audio_vad_router
from server.api.transcriptions.handlers import create_router as create_transcription_router
from server.api.tts_history.handlers import create_router as create_tts_history_router
from server.api.voice.profile import create_router as create_voice_profile_router
from server.api.voice.observations import create_router as create_voice_obs_router
from server.api.voice.realtime import register_ws
from server.api.voices.handlers import create_router as create_saved_voice_router
from server.api.agent.handlers import create_router as create_agent_router
from server.api.onboarding.handlers import create_router as create_onboarding_router


def build_router(app: FastAPI, state: AppState) -> None:
    """将所有子路由模块挂载到应用实例上。

    每个子路由模块通过 `create_router(state)` 工厂函数创建，
    接收 AppState 以便访问共享的 Store 实例。
    WebSocket 端点使用独立的 register 函数直接挂载。
    """
    app.state.app_state = state

    # ── 模型管理 ──
    app.include_router(admin_router(state), prefix="/v1/admin/models")

    # ── 聊天 ──
    app.include_router(chat_thread_router(state), prefix="/v1/chat/threads")
    app.include_router(chat_completions_router(state), prefix="/v1/chat")

    # ── 音频处理（TTS + ASR + VAD）──
    app.include_router(audio_trans_router(state), prefix="/v1/audio")
    app.include_router(audio_speech_router(state), prefix="/v1/audio")
    app.include_router(audio_vad_router(state), prefix="/v1/audio")

    # ── 转录与 TTS 历史记录 ──
    app.include_router(create_transcription_router(state), prefix="/v1/transcriptions")
    app.include_router(create_tts_history_router(state), prefix="/v1/text-to-speech-generations")

    # ── 语音配置 ──
    app.include_router(create_voice_profile_router(state), prefix="/v1/voice")
    app.include_router(create_voice_obs_router(state), prefix="/v1/voice")

    # ── 保存的声音、Agent、引导 ──
    app.include_router(create_saved_voice_router(state), prefix="/v1/voices")
    app.include_router(create_agent_router(state), prefix="/v1/agent")
    app.include_router(create_onboarding_router(state), prefix="/v1/onboarding")

    # ── 实时语音 WebSocket ──
    register_ws(app, state)
