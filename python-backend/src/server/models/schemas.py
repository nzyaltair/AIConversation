"""
Pydantic 请求/响应模型定义集合。

定义了所有 API 端点使用的数据结构，包括模型管理、聊天、音频处理、
转录历史、语音配置、Agent 会话和引导流程的请求与响应模型。
"""

from __future__ import annotations

from pydantic import BaseModel


# ── 通用状态响应 ──
class StatusResponse(BaseModel):
    status: str


# ── 模型管理相关 ──
# 对应 /v1/admin/models 端点的请求/响应模型
class ModelInfoResponse(BaseModel):
    variant: str
    repo_id: str = ""
    category: str = ""
    status: str  # not_downloaded | downloading | downloaded | ready | error
    enabled: bool = True
    size_bytes: int | None = None
    downloaded_bytes: int = 0
    error_message: str | None = None


class DownloadProgressEvent(BaseModel):
    variant: str
    percent: float = 0.0
    current_file: str = ""
    status: str = "downloading"
    downloaded_bytes: int = 0
    total_bytes: int = 0


# ── 聊天相关 ──
# 对应 /v1/chat/threads 和 /v1/chat/completions 端点的请求/响应模型
class ThreadResponse(BaseModel):
    id: str
    title: str = ""
    model_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    last_message_preview: str | None = None


class CreateThreadRequest(BaseModel):
    title: str
    model_id: str | None = None


class UpdateThreadRequest(BaseModel):
    title: str | None = None


class MessageResponse(BaseModel):
    id: str
    thread_id: str
    role: str  # user | assistant | system
    content: str = ""
    created_at: str = ""
    model_id: str | None = None


class SendMessageRequest(BaseModel):
    content: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: bool | None = None


class BatchMessageItem(BaseModel):
    role: str
    content: str
    model_id: str = ""


class BatchSaveMessagesRequest(BaseModel):
    messages: list[BatchMessageItem]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    stream: bool = True
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    thinking: bool | None = None


# ── 音频处理相关（ASR + TTS）──
# 对应 /v1/audio/transcriptions 和 /v1/audio/speech 端点的请求/响应模型
class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None


class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str | None = None
    speed: float | None = None
    instruct: str | None = None
    response_format: str | None = None
    stream: bool = False


# ── 转录历史记录 ──
# 对应 /v1/transcriptions 端点的响应模型
class TranscriptionRecordResponse(BaseModel):
    id: str
    file_name: str = ""
    duration_secs: float | None = None
    language: str | None = None
    model_id: str | None = None
    text: str | None = None
    status: str | None = "completed"
    created_at: str | None = None


# ── TTS 生成历史记录 ──
# 对应 /v1/text-to-speech-generations 端点的请求/响应模型
class SpeechHistoryRecordResponse(BaseModel):
    id: str
    route_kind: str = ""
    model_id: str | None = None
    speaker: str | None = None
    input_text: str | None = None
    audio_path: str | None = None
    audio_duration_secs: float | None = None
    generation_time_ms: float | None = None
    created_at: str | None = None


class CreateSpeechHistoryRequest(BaseModel):
    model_id: str | None = None
    speaker: str | None = None
    input_text: str | None = None
    audio_duration_secs: float | None = None
    generation_time_ms: float | None = None
    audio_base64: str | None = None


# ── 语音配置相关 ──
# 对应 /v1/voice 端点的请求/响应模型（配置文件 + 观察记忆）
class VoiceProfileResponse(BaseModel):
    id: str
    name: str | None = None
    system_prompt: str | None = None
    observational_memory_enabled: bool = False
    default_system_prompt: str = "You are a helpful voice assistant. Be concise and direct in your responses."


class UpdateVoiceProfileRequest(BaseModel):
    name: str | None = None
    system_prompt: str | None = None
    observational_memory_enabled: bool | None = None


class VoiceObservationResponse(BaseModel):
    id: str
    profile_id: str | None = None
    category: str | None = None
    summary: str | None = None
    confidence: float | None = None
    source_text: str | None = None
    created_at: str | None = None


class AddVoiceObservationRequest(BaseModel):
    category: str | None = None
    summary: str
    confidence: float | None = None
    source_text: str | None = None


# ── 保存的声音 ──
# 对应 /v1/voices 端点的响应模型
class SavedVoiceResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    audio_path: str | None = None
    model_id: str | None = None
    created_at: str | None = None


# ── Agent 会话 ──
# 对应 /v1/agent 端点的请求/响应模型
class AgentSessionRequest(BaseModel):
    agent_id: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None
    planning_mode: str | None = None
    title: str | None = None


class AgentSessionResponse(BaseModel):
    id: str
    agent_id: str = "voice-agent"
    thread_id: str | None = None
    model_id: str | None = None
    planning_mode: str = "auto"
    created_at: str = ""
    updated_at: str = ""


class AgentTurnRequest(BaseModel):
    input: str
    model_id: str | None = None
    max_output_tokens: int = 1536


class AgentTurnResponse(BaseModel):
    session_id: str
    thread_id: str | None = None
    model_id: str | None = None
    assistant_text: str = ""


# ── 新手引导 ──
# 对应 /v1/onboarding 端点的响应模型
class OnboardingResponse(BaseModel):
    completed: bool


# ── 健康检查 ──
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
