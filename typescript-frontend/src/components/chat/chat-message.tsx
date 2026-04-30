import { User, Bot } from 'lucide-react';
import { MarkdownContent } from '@/components/shared/markdown-content';
import { cn } from '@/lib/utils';
import type { ChatMessage } from '@/types';

interface ChatMessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
}

export function ChatMessageBubble({ message, isStreaming }: ChatMessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('flex gap-3 animate-fade-in-up', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-1 ring-1 ring-primary/20">
          <Bot className="h-4 w-4 text-primary" />
        </div>
      )}
      <div className={cn('max-w-[75%] shadow-sm', isUser ? 'chat-bubble-user' : 'chat-bubble-assistant')}>
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <MarkdownContent content={message.content} />
        )}
        {isStreaming && <span className="animate-blink ml-0.5">|</span>}
      </div>
      {isUser && (
        <div className="h-8 w-8 rounded-full bg-secondary flex items-center justify-center shrink-0 mt-1">
          <User className="h-4 w-4 text-muted-foreground" />
        </div>
      )}
    </div>
  );
}
