import { cn } from '@/lib/utils';

interface VadTimelineProps {
  dur: number;
  timestamps: [number, number][];
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return m > 0 ? `${m}:${s.padStart(4, '0')}` : `${s}s`;
}

export function VadTimeline({ dur, timestamps }: VadTimelineProps) {
  if (dur <= 0) return null;

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">语音时间线</h4>
      <div className="relative h-10 bg-muted/60 rounded-lg overflow-hidden">
        {/* 网格线 */}
        {Array.from({ length: 11 }).map((_, i) => (
          <div
            key={i}
            className="absolute top-0 bottom-0 w-px bg-border/40"
            style={{ left: `${(i / 10) * 100}%` }}
          />
        ))}
        {/* 语音段 */}
        {timestamps.map(([start, end], i) => {
          const left = (start / dur) * 100;
          const width = ((end - start) / dur) * 100;
          return (
            <div
              key={i}
              title={`${formatTime(start)} → ${formatTime(end)}`}
              className={cn(
                'absolute top-1 bottom-1 rounded-sm transition-colors cursor-default',
                'bg-emerald-500/70 hover:bg-emerald-500',
              )}
              style={{ left: `${left}%`, width: `${Math.max(width, 0.3)}%` }}
            />
          );
        })}
        {/* 无声段 */}
        {timestamps.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs text-muted-foreground">未检测到语音</span>
          </div>
        )}
      </div>
      {/* 时间刻度 */}
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>0s</span>
        <span>{formatTime(dur)}</span>
      </div>
      {/* 语音段列表 */}
      {timestamps.length > 0 && (
        <div className="space-y-1 mt-2">
          {timestamps.map(([start, end], i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
              <span className="text-muted-foreground">
                语音段 {i + 1}：{formatTime(start)} → {formatTime(end)}
                <span className="text-muted-foreground/60 ml-1">
                  （{(end - start).toFixed(1)}秒）
                </span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
