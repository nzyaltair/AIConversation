import type { Thread } from '@/types';

interface ThreadItemProps {
  thread: Thread;
  isActive: boolean;
  onClick: () => void;
  onDelete: () => void;
}

export function ThreadItem({ thread, isActive, onClick, onDelete }: ThreadItemProps) {
  return (
    <div
      className={`group cursor-pointer px-3 py-2.5 rounded-lg border transition-all ${
        isActive ? 'bg-secondary border-primary/30' : 'border-transparent hover:bg-muted/50'
      }`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-sm font-medium truncate">{thread.title || '新对话'}</h4>
        <button
          className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-all"
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          title="删除对话"
        >
          <svg className="h-3.5 w-3.5" viewBox="0 0 15 15" fill="currentColor">
            <path d="M5.5 5.5a.5.5 0 01.5.5v4a.5.5 0 01-1 0V6a.5.5 0 01.5-.5zm4 0a.5.5 0 01.5.5v4a.5.5 0 01-1 0V6a.5.5 0 01.5-.5z" /><path fillRule="evenodd" d="M3 4.5a.5.5 0 01.5-.5h8a.5.5 0 010 1h-.5l-.5 7a1 1 0 01-1 .5h-4a1 1 0 01-1-.5l-.5-7H3.5a.5.5 0 01-.5-.5zm2.058 0l.442 6.2a.1.1 0 00.1.05h3.8a.1.1 0 00.1-.05l.442-6.2H5.058z" />
          </svg>
        </button>
      </div>
      {thread.last_message_preview && (
        <p className="text-xs text-muted-foreground truncate mt-1">{thread.last_message_preview}</p>
      )}
      <div className="flex items-center gap-2 mt-1.5">
        <span className="text-[10px] text-muted-foreground/60">{thread.message_count} 条消息</span>
        <span className="text-[10px] text-muted-foreground/60">
          {new Date(thread.updated_at).toLocaleDateString('zh-CN')}
        </span>
      </div>
    </div>
  );
}
