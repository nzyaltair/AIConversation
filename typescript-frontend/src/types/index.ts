/**
 * 全局 TypeScript 类型定义。
 *
 * 按功能域组织：模型管理、聊天、转录、TTS 历史、语音配置、实时语音、
 * API 协议类型（SSE 流式响应、ChatCompletionChunk 等）。
 */

// ── 模型管理类型 ──
// 对应后端 /v1/admin/models 端点的数据模型
export type ModelStatus = 'not_downloaded' | 'downloading' | 'downloaded' | 'ready' | 'error';

export interface ModelInfo {
  variant: string;
  repo_id: string;
  category: string;
  status: ModelStatus;
  enabled: boolean;
  size_bytes: number | null;
  downloaded_bytes: number;
  error_message: string | null;
}

export interface DownloadProgress {
  variant: string;
  percent: number;
  current_file: string;
  status: string;
  downloaded_bytes: number;
  total_bytes: number;
}

export type ModelCategory = 'llm' | 'asr' | 'tts' | 'vad';

export interface DiskScanResult {
  disk_models: string[];
  orphaned: string[];
  catalog_count: number;
}

// ── 聊天类型 ──
// 对应后端 /v1/chat/threads 和 /v1/chat/completions 的数据模型
export interface Thread {
  id: string;
  title: string;
  model_id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message_preview: string | null;
}

export interface ChatMessage {
  id: string;
  thread_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  model_id: string | null;
}

// ── 转录历史类型 ──
// 对应后端 /v1/transcriptions 的数据模型
export interface TranscriptionRecord {
  id: string;
  file_name: string;
  duration_secs: number | null;
  language: string | null;
  model_id: string | null;
  text: string | null;
  status: string | null;
  created_at: string | null;
}

// ── TTS 历史类型 ──
// 对应后端 /v1/text-to-speech-generations 的数据模型
export interface SpeechHistoryRecord {
  id: string;
  route_kind: string;
  model_id: string | null;
  speaker: string | null;
  input_text: string | null;
  audio_path: string | null;
  audio_duration_secs: number | null;
  generation_time_ms: number | null;
  created_at: string | null;
}

// ── 语音配置类型 ──
// 对应后端 /v1/voice 的语音配置文件 & 观察记忆数据模型
export interface VoiceProfile {
  id: string;
  name: string | null;
  system_prompt: string | null;
  observational_memory_enabled: boolean;
  default_system_prompt: string;
}

export interface VoiceObservation {
  id: string;
  profile_id: string | null;
  category: string | null;
  summary: string | null;
  confidence: number | null;
  source_text: string | null;
  created_at: string | null;
}

export interface SavedVoice {
  id: string;
  name: string;
  description: string | null;
  audio_path: string | null;
  model_id: string | null;
  created_at: string | null;
}

// ── 实时语音 WebSocket 类型 ──
export type VoiceSessionState = 'idle' | 'connecting' | 'connected' | 'listening' | 'processing' | 'speaking';

export interface VadConfig {
  threshold: number;
  min_speech_ms: number;
  silence_duration_ms: number;
  max_utterance_ms: number;
  sample_rate: number;
}

export interface ConversationBubble {
  id: string;
  type: 'user' | 'assistant';
  text: string;
  isStreaming?: boolean;
  thinking?: string;
  timestamp: number;
}

// ── API 协议类型（SSE 流式响应）──
// ChatCompletionChunk 兼容 OpenAI API 的流式响应块格式
export interface DeviceInfo {
  device: string;
  cuda_available: boolean;
  cuda_device_count: number;
}

export interface ModelStatusInfo {
  variant: string;
  status: ModelStatus;
}

export interface ChatCompletionDelta {
  content?: string;
  role?: string;
  thinking?: string;
}

export interface ChatCompletionChoice {
  index: number;
  delta: ChatCompletionDelta;
  finish_reason: string | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
}

// ── External API configuration ──
export interface ApiConfig {
  base_url: string;
  api_key: string;
  model: string;
  reasoning_effort?: 'none' | 'low' | 'medium' | 'high';
}

// Toast
export type ToastVariant = 'default' | 'success' | 'warning' | 'destructive';

export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant: ToastVariant;
}

// SSE
export type SSECallback<T> = (data: T) => void;

// shadcn/ui animation keyframes for Tailwind config
declare global {
  interface CSSStyleDeclaration {
    '--radix-select-trigger-height'?: string;
    '--radix-accordion-content-height'?: string;
  }
}
