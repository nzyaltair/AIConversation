import { useState } from 'react';
import { X, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface OrphanedModelsBannerProps {
  orphaned: string[];
}

/** 孤立模型提示横幅 — 当磁盘上存在未在目录中注册的模型时显示 */
export function OrphanedModelsBanner({ orphaned }: OrphanedModelsBannerProps) {
  const [dismissed, setDismissed] = useState(false);

  if (orphaned.length === 0 || dismissed) return null;

  return (
    <div className="mb-4 flex items-start gap-3 rounded-xl border border-warning/20 bg-warning/10 px-4 py-3 text-sm">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
      <div className="flex-1 min-w-0">
        <p className="font-medium text-warning-foreground">
          发现 {orphaned.length} 个未注册的模型目录
        </p>
        <p className="text-muted-foreground mt-1">
          以下模型目录存在于磁盘上，但未在模型目录中注册：
        </p>
        <ul className="list-disc list-inside mt-1 text-muted-foreground/80">
          {orphaned.map((name) => (
            <li key={name} className="truncate">{name}</li>
          ))}
        </ul>
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 shrink-0 text-muted-foreground hover:text-foreground"
        onClick={() => setDismissed(true)}
      >
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
