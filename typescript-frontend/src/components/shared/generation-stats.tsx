interface GenerationStatsProps {
  duration?: number | null;
  generationTimeMs?: number | null;
  label?: string;
}

export function GenerationStats({ duration, generationTimeMs, label = '统计' }: GenerationStatsProps) {
  if (!duration && !generationTimeMs) return null;

  const rtf = duration && generationTimeMs
    ? ((generationTimeMs / 1000) / duration).toFixed(2)
    : null;

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span className="font-medium text-muted-foreground/70">{label}</span>
      {duration != null && <span>时长：{duration.toFixed(1)}秒</span>}
      {generationTimeMs != null && <span>生成耗时：{(generationTimeMs / 1000).toFixed(1)}秒</span>}
      {rtf && <span>RTF: {rtf}</span>}
    </div>
  );
}
