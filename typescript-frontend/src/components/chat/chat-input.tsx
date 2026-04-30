import { useState, useRef, type KeyboardEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSend: (content: string) => void;
  isStreaming: boolean;
  onCancelStream?: () => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, isStreaming, disabled }: ChatInputProps) {
  const [input, setInput] = useState('');
  const ref = useRef<HTMLTextAreaElement>(null);
  const isDisabled = disabled || isStreaming || !input.trim();

  const handleSend = () => {
    if (isDisabled) return;
    onSend(input.trim());
    setInput('');
    ref.current?.focus();
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-3 border-t border-border/50 bg-card/60 backdrop-blur-sm">
      <div className="flex gap-2 items-end max-w-3xl mx-auto">
        <Textarea
          ref={ref}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入消息...（Enter 发送，Shift+Enter 换行）"
          rows={2}
          className="min-h-[44px] resize-none"
          disabled={disabled}
        />
        <Button size="icon" onClick={handleSend} disabled={isDisabled} className="shrink-0 rounded-lg">
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
