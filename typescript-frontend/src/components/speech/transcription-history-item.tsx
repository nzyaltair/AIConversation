import { Play, Copy, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { TranscriptionRecord } from '@/types';

interface TranscriptionHistoryItemProps {
  record: TranscriptionRecord;
  onPlay: (id: string) => void;
  onDelete: (id: string) => void;
  onCopy: (text: string) => void;
}

export function TranscriptionHistoryItem({ record, onPlay, onDelete, onCopy }: TranscriptionHistoryItemProps) {
  return (
    <div className="p-3 rounded-lg border border-border bg-card/50 hover:bg-card transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{record.file_name}</p>
          {record.text && (
            <p className="text-xs text-muted-foreground truncate mt-0.5">{record.text}</p>
          )}
          <div className="flex gap-2 mt-1.5 text-[10px] text-muted-foreground/60">
            {record.duration_secs && <span>{(record.duration_secs).toFixed(0)}s</span>}
            {record.language && <span>{record.language}</span>}
            {record.created_at && <span>{new Date(record.created_at).toLocaleDateString('zh-CN')}</span>}
          </div>
        </div>
        <div className="flex gap-0.5 shrink-0">
          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => onPlay(record.id)}>
            <Play className="h-3 w-3" />
          </Button>
          <Button
            size="icon" variant="ghost" className="h-7 w-7"
            onClick={() => record.text && onCopy(record.text)}
          >
            <Copy className="h-3 w-3" />
          </Button>
          <Button size="icon" variant="ghost" className="h-7 w-7 text-destructive" onClick={() => onDelete(record.id)}>
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  );
}
