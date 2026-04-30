/**
 * Zustand store with localStorage persist middleware for LLM/TTS/VAD settings.
 *
 * Persisted under the `ai-conversation.settings` key.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface ConversationSettings {
  // LLM
  temperature: number; // 0.0-2.0, default 0.7
  maxTokens: number; // 64-8192, default 2048
  thinkingEnabled: boolean; // default true
  systemPrompt: string; // default ""
  // TTS
  ttsSpeaker: string; // default "Vivian"
  ttsSpeed: number; // 0.5-2.0, default 1.0
  voiceDesignInstruct: string; // default ""
  // VAD
  vadThreshold: number; // 0.1-1.0, default 0.5
  vadMinSpeechMs: number; // 50-1000, default 200
  vadSilenceDurationMs: number; // 100-2000, default 500
  vadMaxUtteranceMs: number; // 1000-30000, default 10000
  // Model variants
  llmModelVariant: string; // default "Qwen3.5-0.8B.Q8_0"
  ttsModelVariant: string; // default "Qwen3-TTS-1.7B-VoiceDesign-gguf"
  asrModelVariant: string; // default "Qwen3-ASR-0.6B-gguf"
  vadModelVariant: string; // default "FireRedVad-onnx"
  // Actions
  updateSettings: (partial: Partial<ConversationSettings>) => void;
  resetSettings: () => void;
}

type PersistedSettings = Omit<ConversationSettings, 'updateSettings' | 'resetSettings'>;

const DEFAULTS: PersistedSettings = {
  temperature: 0.7,
  maxTokens: 2048,
  thinkingEnabled: false,
  systemPrompt:
    '你是一个热情又贴心的小伙伴，说话自然亲切，像朋友一样。尽量用口语化的中文回答，语气轻松活泼，不要长篇大论，像日常聊天那样回应就好。',
  ttsSpeaker: 'Vivian',
  ttsSpeed: 1.0,
  voiceDesignInstruct: '',
  vadThreshold: 0.5,
  vadMinSpeechMs: 400,
  vadSilenceDurationMs: 800,
  vadMaxUtteranceMs: 20000,
  llmModelVariant: 'Qwen3.5-0.8B.Q8_0',
  ttsModelVariant: 'Qwen3-TTS-1.7B-VoiceDesign-gguf',
  asrModelVariant: 'Qwen3-ASR-0.6B-gguf',
  vadModelVariant: 'FireRedVad-onnx',
};

export const useConversationSettings = create<ConversationSettings>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      updateSettings: (partial: Partial<ConversationSettings>) => set(partial),
      resetSettings: () => set(DEFAULTS),
    }),
    {
      name: 'ai-conversation.settings',
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
