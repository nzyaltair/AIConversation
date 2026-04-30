import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { ModelStatusBadge } from '@/components/shared/model-status-badge';
import { getModelDetail } from '@/lib/model-metadata';
import type { ModelInfo, DownloadProgress } from '@/types';
import { Download, Loader, Trash2, X } from 'lucide-react';

/** 模型卡片组件属性 */
interface ModelCardProps {
  model: ModelInfo;
  progress?: DownloadProgress;
  onDownload: (variant: string) => void;
  onCancelDownload: (variant: string) => void;
  onDelete: (variant: string) => void;
  isDeleting?: boolean;
}

/**
 * 模型管理卡片组件。
 *
 * 根据模型状态渲染不同的交互按钮：
 *   not_downloaded → Download 按钮
 *   downloading → Cancel 按钮 + 进度条
 *   downloaded / ready → Delete 按钮
 */
export function ModelCard({ model, progress, onDownload, onCancelDownload, onDelete, isDeleting }: ModelCardProps) {
  const detail = getModelDetail(model.variant);
  const displayName = detail?.displayName ?? model.variant;
  const description = detail?.description ?? '';
  const isDownloading = model.status === 'downloading';
  const isReady = model.status === 'ready';
  const isDownloaded = model.status === 'downloaded';
  const downloadPercent = progress?.percent ?? 0;

  return (
    <Card className="overflow-hidden group animate-fade-in-up border-l-2 border-l-primary/20">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-sm truncate">{displayName}</h4>
            <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">{description}</p>
          </div>
          <ModelStatusBadge status={model.status} />
        </div>

        {detail && (
          <div className="flex gap-3 text-xs text-muted-foreground/70 mb-3">
            <span className="inline-flex items-center gap-1">
              <span className="h-1 w-1 rounded-full bg-muted-foreground/50" />
              ~{detail.sizeGb.toFixed(1)} GB
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-1 w-1 rounded-full bg-muted-foreground/50" />
              ~{detail.ramGb.toFixed(1)} GB RAM
            </span>
          </div>
        )}

        {isDownloading && (
          <div className="mb-3">
            <Progress value={downloadPercent} className="h-1.5" />
            <div className="flex justify-between mt-1">
              <span className="text-xs text-muted-foreground">{downloadPercent.toFixed(0)}%</span>
              {progress?.current_file && (
                <span className="text-xs text-muted-foreground truncate ml-2">{progress.current_file}</span>
              )}
            </div>
          </div>
        )}

        <div className="flex gap-2">
          {/* 状态机渲染：根据 status 显示对应的操作按钮 */}
          {!isDownloading && (model.status === 'not_downloaded' || model.status === 'error') && (
            <Button size="sm" variant="outline" onClick={() => onDownload(model.variant)} className="rounded-lg">
              <Download className="h-3.5 w-3.5" /> 下载
            </Button>
          )}
          {isDownloading && (
            <Button size="sm" variant="outline" onClick={() => onCancelDownload(model.variant)} className="rounded-lg">
              <X className="h-3.5 w-3.5" /> 取消
            </Button>
          )}
          {!isDownloading && (isDownloaded || isReady || model.status === 'error') && (
            <Button
              size="sm"
              variant="outline"
              className="text-destructive hover:text-destructive border-destructive/30 hover:border-destructive rounded-lg"
              disabled={isDeleting}
              onClick={() => onDelete(model.variant)}
            >
              {isDeleting ? (
                <Loader className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
              删除
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
