import { Copy, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface TranscriptionResultProps {
  text: string;
  onCopy: () => void;
  onDownload: () => void;
}

export function TranscriptionResult({ text, onCopy, onDownload }: TranscriptionResultProps) {
  if (!text) return null;

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium">转写结果</h3>
        <div className="flex gap-1">
          <Button size="sm" variant="ghost" onClick={onCopy}><Copy className="h-3.5 w-3.5" /></Button>
          <Button size="sm" variant="ghost" onClick={onDownload}><Download className="h-3.5 w-3.5" /></Button>
        </div>
      </div>
      <p className="text-sm whitespace-pre-wrap leading-relaxed">{text}</p>
    </div>
  );
}
