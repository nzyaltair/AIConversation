import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type BackgroundType = 'preset' | 'color' | 'image';

export interface BackgroundPreset {
  id: string;
  label: string;
  cssValue: string;
  type: 'gradient' | 'solid';
}

interface BackgroundState {
  type: BackgroundType;
  presetId: string;
  customColor: string;
  imageUrl: string | null;
  imageFit: 'cover' | 'contain';
  opacity: number;
  enabled: boolean;

  presets: BackgroundPreset[];

  setPreset: (presetId: string) => void;
  setCustomColor: (color: string) => void;
  setImage: (dataUrl: string) => void;
  clearImage: () => void;
  setImageFit: (fit: 'cover' | 'contain') => void;
  setOpacity: (opacity: number) => void;
  toggleEnabled: () => void;
  setEnabled: (enabled: boolean) => void;
  reset: () => void;
}

const DEFAULT_PRESETS: BackgroundPreset[] = [
  {
    id: 'deep-space',
    label: '深邃星空',
    cssValue: 'linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)',
    type: 'gradient',
  },
  {
    id: 'twilight',
    label: '暮光蓝调',
    cssValue: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
    type: 'gradient',
  },
  {
    id: 'aurora',
    label: '极光森林',
    cssValue: 'linear-gradient(135deg, #0d1b2a 0%, #1b2838 40%, #1a3a2a 100%)',
    type: 'gradient',
  },
  {
    id: 'warm-ember',
    label: '温暖余烬',
    cssValue: 'linear-gradient(135deg, #2d1b00 0%, #1a0a00 50%, #1a1a2e 100%)',
    type: 'gradient',
  },
  {
    id: 'charcoal',
    label: '纯色炭灰',
    cssValue: '#1a1a2e',
    type: 'solid',
  },
];

const DEFAULT_STATE = {
  type: 'preset' as BackgroundType,
  presetId: 'deep-space',
  customColor: '#1a1a2e',
  imageUrl: null as string | null,
  imageFit: 'cover' as 'cover' | 'contain',
  opacity: 0.3,
  enabled: false,
  presets: DEFAULT_PRESETS,
};

export const useBackgroundStore = create<BackgroundState>()(
  persist(
    (set) => ({
      ...DEFAULT_STATE,

      setPreset: (presetId) => set({ type: 'preset', presetId }),
      setCustomColor: (color) => set({ type: 'color', customColor: color }),
      setImage: (dataUrl) => set({ type: 'image', imageUrl: dataUrl }),
      clearImage: () => set({ imageUrl: null, type: 'preset' }),
      setImageFit: (fit) => set({ imageFit: fit }),
      setOpacity: (opacity) => set({ opacity }),
      toggleEnabled: () => set((s) => ({ enabled: !s.enabled })),
      setEnabled: (enabled) => set({ enabled }),
      reset: () => set({ ...DEFAULT_STATE }),
    }),
    {
      name: 'ai-conversation.background',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        type: state.type,
        presetId: state.presetId,
        customColor: state.customColor,
        imageUrl: state.imageUrl,
        imageFit: state.imageFit,
        opacity: state.opacity,
        enabled: state.enabled,
      }),
    },
  ),
);
