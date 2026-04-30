import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Wifi, WifiOff, Settings2, Cpu } from 'lucide-react';
import type { VoiceSessionState } from '@/types';

interface StatusBarProps {
  sessionState: VoiceSessionState;
  isConnected: boolean;
  modelVariants?: { vad?: string; asr?: string; llm?: string; tts?: string };
  latencyMs?: number;
  gpuAvailable?: boolean;
  onToggleSettings: () => void;
}

function getStatusLabel(state: VoiceSessionState, isConnected: boolean): string {
  if (!isConnected) return '已断开';
  switch (state) {
    case 'idle':
      return '就绪';
    case 'connecting':
      return '连接中...';
    case 'connected':
      return '已连接';
    case 'listening':
      return '聆听中';
    case 'processing':
      return '处理中...';
    case 'speaking':
      return '说话中';
    default:
      return '未知';
  }
}

function getStatusDotColor(state: VoiceSessionState, isConnected: boolean): string {
  if (!isConnected) return 'bg-destructive';
  switch (state) {
    case 'connecting':
      return 'bg-warning animate-pulse';
    case 'idle':
    case 'connected':
      return 'bg-success';
    case 'listening':
    case 'processing':
    case 'speaking':
      return 'bg-success animate-pulse';
    default:
      return 'bg-muted-foreground';
  }
}

function getModelBadgeVariant(category: string): 'default' | 'secondary' | 'success' | 'warning' {
  switch (category) {
    case 'llm':
      return 'default';
    case 'tts':
      return 'success';
    case 'asr':
      return 'warning';
    case 'vad':
      return 'secondary';
    default:
      return 'secondary';
  }
}

export function StatusBar({
  sessionState,
  isConnected,
  modelVariants,
  latencyMs,
  gpuAvailable,
  onToggleSettings,
}: StatusBarProps) {
  return (
    <div className="flex items-center justify-between py-2.5 px-5 bg-card/40 backdrop-blur-sm border-b border-border/30 shrink-0">
      {/* Left: connection status */}
      <div className="flex items-center gap-2 min-w-0">
        <span className={cn('h-2.5 w-2.5 rounded-full shrink-0 ring-2 ring-background', getStatusDotColor(sessionState, isConnected))} />
        <span className="text-xs font-medium text-muted-foreground truncate">
          {getStatusLabel(sessionState, isConnected)}
        </span>
      </div>

      {/* Center: model badges (hidden on mobile) */}
      <div className="hidden md:flex items-center gap-1.5 min-w-0">
        {modelVariants?.llm && (
          <Badge variant={getModelBadgeVariant('llm')} className="text-[10px] px-1.5 py-0">
            {modelVariants.llm}
          </Badge>
        )}
        {modelVariants?.tts && (
          <Badge variant={getModelBadgeVariant('tts')} className="text-[10px] px-1.5 py-0">
            {modelVariants.tts}
          </Badge>
        )}
        {modelVariants?.asr && (
          <Badge variant={getModelBadgeVariant('asr')} className="text-[10px] px-1.5 py-0">
            {modelVariants.asr}
          </Badge>
        )}
        {modelVariants?.vad && (
          <Badge variant={getModelBadgeVariant('vad')} className="text-[10px] px-1.5 py-0">
            {modelVariants.vad}
          </Badge>
        )}
      </div>

      {/* Right: latency + gpu indicator + settings */}
      <div className="flex items-center gap-2 shrink-0">
        {gpuAvailable && (
          <span className="hidden sm:flex items-center gap-1 text-xs text-muted-foreground" title="GPU 可用">
            <Cpu className="h-3 w-3" />
          </span>
        )}
        {isConnected && latencyMs !== undefined && (
          <span className="text-xs text-muted-foreground tabular-nums bg-muted/40 rounded-md px-2 py-0.5">{latencyMs}ms</span>
        )}
        <span className={cn('flex items-center gap-1 text-xs', isConnected ? 'text-success' : 'text-destructive')}>
          {isConnected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
        </span>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSettings}
          className="h-8 w-8"
          title="设置"
        >
          <Settings2 className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}
