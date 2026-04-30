import { useEffect, useRef } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { ConversationBubble } from '@/components/voice/conversation-bubble';
import { Mic, X } from 'lucide-react';
import type { ConversationBubble as ConversationBubbleType } from '@/types';

interface ConversationStreamProps {
  bubbles: ConversationBubbleType[];
  onClear?: () => void;
}

export function ConversationStream({ bubbles, onClear }: ConversationStreamProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const prevLengthRef = useRef(bubbles.length);
  const prevLastTextRef = useRef(bubbles[bubbles.length - 1]?.text ?? '');

  // Auto-scroll when new bubbles are added or text changes
  useEffect(() => {
    const currentLastText = bubbles[bubbles.length - 1]?.text ?? '';
    const lengthChanged = bubbles.length !== prevLengthRef.current;
    const streamingTextChanged = currentLastText !== prevLastTextRef.current;

    if (lengthChanged || streamingTextChanged) {
      // Use a small delay to allow the DOM to render before scrolling
      requestAnimationFrame(() => {
        if (viewportRef.current) {
          viewportRef.current.scrollTop = viewportRef.current.scrollHeight;
        }
      });
    }

    prevLengthRef.current = bubbles.length;
    prevLastTextRef.current = currentLastText;
  }, [bubbles]);

  // Find if the last assistant bubble is streaming
  const lastAssistantIsStreaming = (() => {
    for (let i = bubbles.length - 1; i >= 0; i--) {
      if (bubbles[i].type === 'assistant') {
        return bubbles[i].isStreaming ?? false;
      }
    }
    return false;
  })();

  const isEmpty = bubbles.length === 0;

  return (
    <div className="flex-1 relative min-h-0">
      {/* Clear button */}
      {!isEmpty && onClear && (
        <div className="absolute top-2 right-3 z-10">
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            className="h-7 text-xs text-muted-foreground hover:text-foreground gap-1 rounded-full bg-card/60 backdrop-blur-sm px-3"
          >
            <X className="h-3 w-3" />
            清除
          </Button>
        </div>
      )}

      <ScrollArea className="h-full">
        <div ref={viewportRef} className="h-full px-4 py-3">
          {isEmpty ? (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-3">
              <div className="h-12 w-12 rounded-full bg-gradient-card shadow-glow-primary flex items-center justify-center">
                <Mic className="h-5 w-5 text-primary/60" />
              </div>
              <p className="text-sm text-center max-w-[220px]">
                点击麦克风按钮开始对话
              </p>
            </div>
          ) : (
            <div className="space-y-4 pb-2">
              {bubbles.map((bubble) => (
                <ConversationBubble
                  key={bubble.id}
                  type={bubble.type}
                  text={bubble.text}
                  isStreaming={bubble.isStreaming}
                  thinking={bubble.thinking}
                />
              ))}
              {/* Thinking indicator: shown when assistant bubble is streaming */}
              {lastAssistantIsStreaming && (
                <div className="flex gap-2.5 animate-fade-in">
                  <div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-1 ring-1 ring-primary/20">
                    <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                  </div>
                  <div className="flex items-center gap-1 px-3 py-2 rounded-2xl bg-card border border-border">
                    <span className="h-2 w-2 rounded-full bg-primary/40 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="h-2 w-2 rounded-full bg-primary/40 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="h-2 w-2 rounded-full bg-primary/40 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
