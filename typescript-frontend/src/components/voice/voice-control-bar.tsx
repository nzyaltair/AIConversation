import { Button } from '@/components/ui/button';
import { Volume2, VolumeX, Mic, MicOff, StopCircle, Pause } from 'lucide-react';
import type { VoiceSessionState } from '@/types';

interface VoiceControlBarProps {
  sessionState: VoiceSessionState;
  isTtsMuted: boolean;
  isInputDisabled: boolean;
  isLoadingModels?: Record<string, boolean>;
  onStart: () => void;
  onStop: () => void;
  onToggleTtsMute: () => void;
  onToggleInputDisabled: () => void;
  onInterrupt: () => void;
}

export function VoiceControlBar({
  sessionState,
  isTtsMuted,
  isInputDisabled,
  isLoadingModels = {},
  onStart,
  onStop,
  onToggleTtsMute,
  onToggleInputDisabled,
  onInterrupt,
}: VoiceControlBarProps) {
  const isActive = sessionState !== 'idle';
  const anyLoading = Object.values(isLoadingModels).some(Boolean);

  return (
    <div className="flex items-center justify-between gap-4 py-5 border-t border-border/50 bg-card/40 backdrop-blur-sm">
      {/* ── Left group: TTS mute + Input disable ── */}
      <div className="flex items-center gap-2">
        <Button
          size="icon"
          variant={isTtsMuted ? 'destructive' : 'ghost'}
          onClick={onToggleTtsMute}
          disabled={!isActive}
          title="静音 TTS 音频"
          className="rounded-full h-10 w-10"
        >
          {isTtsMuted ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
        </Button>
        <Button
          size="icon"
          variant={isInputDisabled ? 'destructive' : 'ghost'}
          onClick={onToggleInputDisabled}
          disabled={!isActive}
          title="禁用麦克风输入"
          className="rounded-full h-10 w-10"
        >
          {isInputDisabled ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
        </Button>
      </div>

      {/* ── Center group: Start/Stop button ── */}
      <div>
        {!isActive ? (
          <Button size="lg" onClick={onStart} disabled={anyLoading} className="gap-2 h-12 px-10 rounded-full shadow-glow-primary">
            <Mic className="h-5 w-5" /> {anyLoading ? '模型加载中...' : '开始对话'}
          </Button>
        ) : (
          <Button size="lg" variant="destructive" onClick={onStop} className="gap-2 h-12 px-10 rounded-full">
            <StopCircle className="h-5 w-5" /> 停止
          </Button>
        )}
      </div>

      {/* ── Right group: Interrupt button ── */}
      <div>
        <Button
          size="icon"
          variant="ghost"
          onClick={onInterrupt}
          disabled={!isActive}
          title="打断 (空格键)"
          className="rounded-full h-10 w-10"
        >
          <Pause className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}
