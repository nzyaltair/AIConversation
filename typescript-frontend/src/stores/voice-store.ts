import { create } from 'zustand';
import type { VoiceSessionState, VadConfig, ConversationBubble } from '@/types';

const DEFAULT_VAD: VadConfig = {
  threshold: 0.5,
  min_speech_ms: 200,
  silence_duration_ms: 500,
  max_utterance_ms: 10000,
  sample_rate: 16000,
};

interface VoiceStore {
  sessionState: VoiceSessionState;
  vadConfig: VadConfig;
  isTtsMuted: boolean;
  isInputDisabled: boolean;
  bubbles: ConversationBubble[];
  setSessionState: (s: VoiceSessionState) => void;
  updateVadConfig: (c: Partial<VadConfig>) => void;
  toggleTtsMute: () => void;
  setTtsMuted: (m: boolean) => void;
  toggleInputDisabled: () => void;
  setInputDisabled: (d: boolean) => void;
  addBubble: (b: ConversationBubble) => void;
  updateBubble: (id: string, text: string) => void;
  appendToBubble: (id: string, delta: string) => void;
  updateBubbleThinking: (id: string, thinking: string) => void;
  setBubbleStreaming: (id: string, isStreaming: boolean) => void;
  clearBubbles: () => void;
}

export const useVoiceStore = create<VoiceStore>((set) => ({
  sessionState: 'idle',
  vadConfig: DEFAULT_VAD,
  isTtsMuted: false,
  isInputDisabled: false,
  bubbles: [],
  setSessionState: (sessionState) => set({ sessionState }),
  updateVadConfig: (c) => set((s) => ({ vadConfig: { ...s.vadConfig, ...c } })),
  toggleTtsMute: () => set((s) => ({ isTtsMuted: !s.isTtsMuted })),
  setTtsMuted: (isTtsMuted) => set({ isTtsMuted }),
  toggleInputDisabled: () => set((s) => ({ isInputDisabled: !s.isInputDisabled })),
  setInputDisabled: (isInputDisabled) => set({ isInputDisabled }),
  addBubble: (b) => set((s) => ({ bubbles: [...s.bubbles, b] })),
  updateBubble: (id, text) =>
    set((s) => ({
      bubbles: s.bubbles.map((b) => (b.id === id ? { ...b, text } : b)),
    })),
  appendToBubble: (id, delta) =>
    set((s) => ({
      bubbles: s.bubbles.map((b) =>
        b.id === id ? { ...b, text: b.text + delta } : b,
      ),
    })),
  updateBubbleThinking: (id, thinking) =>
    set((s) => ({
      bubbles: s.bubbles.map((b) =>
        b.id === id ? { ...b, thinking } : b,
      ),
    })),
  setBubbleStreaming: (id, isStreaming) =>
    set((s) => ({
      bubbles: s.bubbles.map((b) => (b.id === id ? { ...b, isStreaming } : b)),
    })),
  clearBubbles: () => set({ bubbles: [] }),
}));
