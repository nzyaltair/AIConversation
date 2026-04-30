import { User, Bot } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ConversationBubbleProps {
  type: 'user' | 'assistant';
  text: string;
  isStreaming?: boolean;
  thinking?: string;
}

export function ConversationBubble({ type, text, isStreaming, thinking }: ConversationBubbleProps) {
  return (
    <div className={cn('flex gap-2.5 animate-scale-in', type === 'user' ? 'justify-end' : 'justify-start')}>
      {type === 'assistant' && (
        <div className="h-8 w-8 rounded-full bg-primary/15 ring-1 ring-primary/20 flex items-center justify-center shrink-0 mt-1">
          <Bot className="h-4 w-4 text-primary" />
        </div>
      )}
      <div
        className={cn(
          'max-w-[85%] md:max-w-[75%] px-5 py-3 rounded-2xl text-sm leading-relaxed shadow-sm',
          type === 'user'
            ? 'bg-primary/12 border border-primary/20 rounded-br-md'
            : 'bg-card border border-border/80 rounded-bl-md',
        )}
      >
        {thinking && type === 'assistant' && (
          <details className="mb-2 text-xs text-muted-foreground/70 italic border-l-2 border-muted pl-2">
            <summary className="cursor-pointer select-none text-muted-foreground/50 text-[10px] hover:text-muted-foreground/70 transition-colors">
              思考（{thinking.length} 字）
            </summary>
            <div className="mt-1 whitespace-pre-wrap">{thinking}</div>
          </details>
        )}
        {text ? <span className="whitespace-pre-wrap">{text}</span> : isStreaming && <span className="animate-blink">|</span>}
        {isStreaming && text && <span className="animate-blink">|</span>}
      </div>
      {type === 'user' && (
        <div className="h-8 w-8 rounded-full bg-secondary/80 flex items-center justify-center shrink-0 mt-1">
          <User className="h-4 w-4 text-muted-foreground" />
        </div>
      )}
    </div>
  );
}
