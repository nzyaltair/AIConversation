import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import type { VadConfig } from '@/types';

interface VadConfigPanelProps {
  config: VadConfig;
  onChange: (c: Partial<VadConfig>) => void;
}

export function VadConfigPanel({ config, onChange }: VadConfigPanelProps) {
  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between mb-1">
          <Label className="text-xs">阈值</Label>
          <span className="text-xs text-muted-foreground">{config.threshold.toFixed(2)}</span>
        </div>
        <Slider
          value={[config.threshold]}
          min={0.1}
          max={1.0}
          step={0.01}
          onValueChange={([v]) => onChange({ threshold: v })}
        />
      </div>
      <div>
        <div className="flex justify-between mb-1">
          <Label className="text-xs">最短语音（毫秒）</Label>
          <span className="text-xs text-muted-foreground">{config.min_speech_ms}ms</span>
        </div>
        <Slider
          value={[config.min_speech_ms]}
          min={50}
          max={1000}
          step={50}
          onValueChange={([v]) => onChange({ min_speech_ms: v })}
        />
      </div>
      <div>
        <div className="flex justify-between mb-1">
          <Label className="text-xs">静音时长（毫秒）</Label>
          <span className="text-xs text-muted-foreground">{config.silence_duration_ms}ms</span>
        </div>
        <Slider
          value={[config.silence_duration_ms]}
          min={100}
          max={2000}
          step={100}
          onValueChange={([v]) => onChange({ silence_duration_ms: v })}
        />
      </div>
      <div>
        <div className="flex justify-between mb-1">
          <Label className="text-xs">最长语句（毫秒）</Label>
          <span className="text-xs text-muted-foreground">{config.max_utterance_ms}ms</span>
        </div>
        <Slider
          value={[config.max_utterance_ms]}
          min={1000}
          max={30000}
          step={500}
          onValueChange={([v]) => onChange({ max_utterance_ms: v })}
        />
      </div>
    </div>
  );
}
