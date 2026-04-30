import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface ThinkingPanelProps {
  content: string;
}

export function ThinkingPanel({ content }: ThinkingPanelProps) {
  const [isOpen, setIsOpen] = useState(true);

  if (!content) return null;

  return (
    <div className="mb-2">
      <button
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        思考
      </button>
      {isOpen && (
        <div className="mt-1 p-2 rounded-md bg-muted/40 border border-border/50 text-xs text-muted-foreground whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  );
}
