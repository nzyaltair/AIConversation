import type { VadResponse } from '@/api/vad';

interface VadStatsProps {
  result: VadResponse;
}

function formatDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

function StatCard({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex flex-col p-3 rounded-lg bg-muted/40 border border-border/40">
      <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</span>
      <span className={`text-lg font-semibold mt-0.5 ${highlight ? 'text-emerald-500' : ''}`}>
        {value}
      </span>
    </div>
  );
}

export function VadStats({ result }: VadStatsProps) {
  const speechTotal = result.timestamps.reduce((sum, [s, e]) => sum + (e - s), 0);
  const avgSegment = result.num_speech_segments > 0
    ? speechTotal / result.num_speech_segments
    : 0;

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">检测统计</h4>
      <div className="grid grid-cols-2 gap-2">
        <StatCard label="总时长" value={formatDuration(result.dur)} />
        <StatCard label="语音段数" value={`${result.num_speech_segments}`} highlight />
        <StatCard label="语音总时长" value={formatDuration(speechTotal)} />
        <StatCard label="语音占比" value={`${(result.speech_duration_ratio * 100).toFixed(1)}%`} />
        {result.num_speech_segments > 0 && (
          <StatCard label="平均段长" value={formatDuration(avgSegment)} />
        )}
      </div>
    </div>
  );
}
