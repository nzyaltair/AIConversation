import { Badge } from '@/components/ui/badge';
import type { ModelStatus } from '@/types';

// 状态 → 颜色映射，downloading 带脉冲动画（animate-pulse）
const statusConfig: Record<ModelStatus, { label: string; variant: 'success' | 'warning' | 'default' | 'secondary' | 'destructive'; pulse?: boolean }> = {
  ready: { label: '就绪', variant: 'success' },
  downloading: { label: '下载中', variant: 'warning', pulse: true },
  downloaded: { label: '已下载', variant: 'secondary' },
  not_downloaded: { label: '未下载', variant: 'secondary' },
  error: { label: '错误', variant: 'destructive' },
};

/** 模型状态徽章组件 — 根据 ModelStatus 显示对应颜色和标签 */
export function ModelStatusBadge({ status }: { status: ModelStatus }) {
  const config = statusConfig[status];
  return (
    <Badge variant={config.variant} className={config.pulse ? 'animate-pulse' : ''}>
      {config.label}
    </Badge>
  );
}
