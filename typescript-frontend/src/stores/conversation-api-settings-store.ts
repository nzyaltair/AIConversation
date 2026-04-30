/**
 * Zustand store for external-API voice conversation settings.
 *
 * Persisted under the `ai-conversation.api-settings` localStorage key.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface ConversationApiSettings {
  // API config
  baseUrl: string;
  apiKey: string;
  modelId: string;
  reasoningEffort: 'none' | 'low' | 'medium' | 'high';
  // LLM
  temperature: number;
  maxTokens: number;
  thinkingEnabled: boolean;
  systemPrompt: string;
  // TTS
  ttsSpeaker: string;
  ttsSpeed: number;
  voiceDesignInstruct: string;
  // VAD
  vadThreshold: number;
  vadMinSpeechMs: number;
  vadSilenceDurationMs: number;
  vadMaxUtteranceMs: number;
  // Model variants (VAD/ASR/TTS are local; LLM is always "external-api")
  ttsModelVariant: string;
  asrModelVariant: string;
  vadModelVariant: string;
  // Actions
  updateSettings: (partial: Partial<ConversationApiSettings>) => void;
  resetSettings: () => void;
}

type PersistedSettings = Omit<
  ConversationApiSettings,
  'updateSettings' | 'resetSettings'
>;

const DEFAULTS: PersistedSettings = {
  baseUrl: 'https://api.deepseek.com',
  apiKey: '',
  modelId: 'deepseek-v4-flash',
  reasoningEffort: 'none',
  temperature: 0.7,
  maxTokens: 2048,
  thinkingEnabled: false,
  systemPrompt: '你是一个热情又贴心的小伙伴，说话自然亲切，像朋友一样。尽量用口语化的中文回答，语气轻松活泼，不要长篇大论，像日常聊天那样回应就好。',
  ttsSpeaker: 'Vivian',
  ttsSpeed: 1.0,
  voiceDesignInstruct: '',
  vadThreshold: 0.5,
  vadMinSpeechMs: 400,
  vadSilenceDurationMs: 800,
  vadMaxUtteranceMs: 20000,
  ttsModelVariant: 'Qwen3-TTS-1.7B-VoiceDesign-gguf',
  asrModelVariant: 'Qwen3-ASR-0.6B-gguf',
  vadModelVariant: 'FireRedVad-onnx',
};

export const useConversationApiSettings = create<ConversationApiSettings>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      updateSettings: (partial: Partial<ConversationApiSettings>) => set(partial),
      resetSettings: () => set(DEFAULTS),
    }),
    {
      name: 'ai-conversation.api-settings',
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
