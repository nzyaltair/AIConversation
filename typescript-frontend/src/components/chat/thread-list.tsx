import { Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { LoadingSpinner } from '@/components/shared/loading-spinner';
import { EmptyState } from '@/components/shared/empty-state';
import { ThreadItem } from '@/components/chat/thread-item';
import type { Thread } from '@/types';

interface ThreadListProps {
  threads: Thread[];
  activeThreadId: string | null;
  isLoading: boolean;
  onThreadClick: (id: string) => void;
  onDeleteThread: (id: string) => void;
  onNewThread: () => void;
}

export function ThreadList({
  threads,
  activeThreadId,
  isLoading,
  onThreadClick,
  onDeleteThread,
  onNewThread,
}: ThreadListProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="p-3 flex items-center justify-between">
        <span className="text-sm font-medium">对话列表</span>
        <Button size="sm" variant="ghost" onClick={onNewThread}>
          <Plus className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="flex-1 px-2 pb-2">
        {isLoading ? (
          <div className="flex justify-center py-8"><LoadingSpinner /></div>
        ) : threads.length === 0 ? (
          <EmptyState title="暂无对话" description="开始一个新对话" />
        ) : (
          <div className="flex flex-col gap-1">
            {threads.map((t) => (
              <ThreadItem
                key={t.id}
                thread={t}
                isActive={t.id === activeThreadId}
                onClick={() => onThreadClick(t.id)}
                onDelete={() => onDeleteThread(t.id)}
              />
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
