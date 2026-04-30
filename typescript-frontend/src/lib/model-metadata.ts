/**
 * 模型静态元数据映射表。
 *
 * 前端硬编码的模型详情（显示名称、描述、资源估算等），
 * 与后端 `SEED_MODELS`（seed_models.py）一一对应。
 */

import type { ModelCategory } from '@/types';

export interface ModelDetail {
  variant: string;
  displayName: string;
  description: string;
  category: ModelCategory;
  sizeGb: number;
  ramGb: number;
  capabilities: string[];
  isThinking?: boolean;
}

export const MODEL_DETAILS: Record<string, ModelDetail> = {
  // LLM models
  'Qwen3.5-0.8B.Q8_0': {
    variant: 'Qwen3.5-0.8B.Q8_0',
    displayName: 'Qwen3.5 0.8B (Q8_0)',
    description: '高精度 8 位量化聊天模型，最大限度保证准确性。',
    category: 'llm',
    sizeGb: 1.0,
    ramGb: 1.5,
    capabilities: ['chat', 'streaming'],
    isThinking: true,
  },
  'Qwen3.5-0.8B.Q4_K_M': {
    variant: 'Qwen3.5-0.8B.Q4_K_M',
    displayName: 'Qwen3.5 0.8B (Q4_K_M)',
    description: '均衡的 4 位量化聊天模型，兼顾效率与推理质量。',
    category: 'llm',
    sizeGb: 0.6,
    ramGb: 1.0,
    capabilities: ['chat', 'streaming'],
    isThinking: true,
  },

  // VAD model
  'FireRedVad-onnx': {
    variant: 'FireRedVad-onnx',
    displayName: 'FireRedVAD',
    description: '实时语音活动检测（VAD）模型，用于语音检测和分割。',
    category: 'vad',
    sizeGb: 0.05,
    ramGb: 0.1,
    capabilities: ['vad', 'realtime'],
  },

  // TTS models
  'Kokoro-82M-v1.1-zh-ONNX-q4': {
    variant: 'Kokoro-82M-v1.1-zh-ONNX-q4',
    displayName: 'Kokoro 82M v1.1 (Chinese Q4)',
    description: '轻量级中文 TTS 模型，采用 ONNX 运行时和 4 位量化。',
    category: 'tts',
    sizeGb: 0.3,
    ramGb: 0.5,
    capabilities: ['tts', 'streaming', 'voice-clone'],
  },
  'Qwen3-TTS-0.6B-CustomVoice-gguf': {
    variant: 'Qwen3-TTS-0.6B-CustomVoice-gguf',
    displayName: 'Qwen3 TTS 0.6B CustomVoice',
    description: 'Qwen3 TTS 模型，支持自定义语音克隆。0.6B 参数，GGUF 格式。',
    category: 'tts',
    sizeGb: 0.7,
    ramGb: 1.5,
    capabilities: ['tts', 'voice-clone', 'streaming'],
  },
  'Qwen3-TTS-1.7B-CustomVoice-gguf': {
    variant: 'Qwen3-TTS-1.7B-CustomVoice-gguf',
    displayName: 'Qwen3 TTS 1.7B CustomVoice',
    description: 'Qwen3 TTS 模型，支持自定义语音克隆。1.7B 参数，GGUF 格式。',
    category: 'tts',
    sizeGb: 1.7,
    ramGb: 3.0,
    capabilities: ['tts', 'voice-clone', 'streaming'],
  },
  'Qwen3-TTS-1.7B-VoiceDesign-gguf': {
    variant: 'Qwen3-TTS-1.7B-VoiceDesign-gguf',
    displayName: 'Qwen3 TTS 1.7B VoiceDesign',
    description: 'Qwen3 TTS 模型，支持语音设计能力。1.7B 参数，GGUF 格式。',
    category: 'tts',
    sizeGb: 1.7,
    ramGb: 3.0,
    capabilities: ['tts', 'voice-design', 'streaming'],
  },

  // ASR models
  'Qwen3-ASR-0.6B-gguf': {
    variant: 'Qwen3-ASR-0.6B-gguf',
    displayName: 'Qwen3 ASR 0.6B',
    description: '自动语音识别（ASR）模型，GGUF 格式，支持高精度多语言转录。',
    category: 'asr',
    sizeGb: 1.3,
    ramGb: 2.0,
    capabilities: ['asr', 'multilingual', 'timestamp'],
  },
  'Qwen3-ASR-1.7B-gguf': {
    variant: 'Qwen3-ASR-1.7B-gguf',
    displayName: 'Qwen3 ASR 1.7B',
    description: '大规模 ASR 模型，GGUF 格式，支持高精度多语言转录和时间戳。',
    category: 'asr',
    sizeGb: 3.0,
    ramGb: 4.0,
    capabilities: ['asr', 'multilingual', 'timestamp'],
  },

  // LLM - ONNX（默认模型，GPU 加速）
  'Qwen3-0.6B-onnx': {
    variant: 'Qwen3-0.6B-onnx',
    displayName: 'Qwen3 0.6B (ONNX)',
    description: '轻量级 ONNX 格式聊天模型，支持 GPU 加速（ONNX Runtime CUDA）。',
    category: 'llm',
    sizeGb: 0.5,
    ramGb: 1.0,
    capabilities: ['chat', 'streaming', 'gpu-accelerated'],
    isThinking: true,
  },
};

export function getModelDetail(variant: string): ModelDetail | undefined {
  return MODEL_DETAILS[variant];
}

export function getModelDisplayName(variant: string): string {
  return MODEL_DETAILS[variant]?.displayName ?? variant;
}

export function getCategoryLabel(cat: ModelCategory): string {
  switch (cat) {
    case 'llm': return 'LLM 模型';
    case 'asr': return 'ASR 模型';
    case 'tts': return 'TTS 模型';
    case 'vad': return 'VAD 模型';
  }
}
