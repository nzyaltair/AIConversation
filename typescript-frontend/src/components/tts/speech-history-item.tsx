import { Play, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { SpeechHistoryRecord } from '@/types';

interface SpeechHistoryItemProps {
  record: SpeechHistoryRecord;
  onPlay: () => void;
  onDelete: (id: string) => void;
}

export function SpeechHistoryItem({ record, onPlay, onDelete }: SpeechHistoryItemProps) {
  return (
    <div className="p-3 rounded-lg border border-border bg-card/50 hover:bg-card transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-xs text-muted-foreground line-clamp-2">{record.input_text || '（无文字）'}</p>
          <div className="flex gap-2 mt-1.5 text-[10px] text-muted-foreground/60">
            {record.speaker && <span>{record.speaker}</span>}
            {record.audio_duration_secs && <span>{(record.audio_duration_secs).toFixed(0)}s</span>}
            {record.created_at && <span>{new Date(record.created_at).toLocaleDateString('zh-CN')}</span>}
          </div>
        </div>
        <div className="flex gap-0.5 shrink-0">
          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onPlay}>
            <Play className="h-3 w-3" />
          </Button>
          <Button size="icon" variant="ghost" className="h-7 w-7 text-destructive" onClick={() => onDelete(record.id)}>
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  );
}
