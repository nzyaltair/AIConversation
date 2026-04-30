import { useState, useCallback } from 'react';
import { Search, SlidersHorizontal } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LoadingSpinner } from '@/components/shared/loading-spinner';
import { ErrorFallback } from '@/components/shared/error-fallback';
import { ModelCard } from '@/components/shared/model-card';
import { OrphanedModelsBanner } from '@/components/shared/orphaned-models-banner';
import { ConfirmDialog } from '@/components/shared/confirm-dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  useModels,
  useDownloadMutation,
  useDeleteMutation,
  useDiskScan,
} from '@/hooks/use-models';
import { useModelStore } from '@/stores/model-store';
import { subscribeDownloadProgress } from '@/api/models';
import { cancelDownload } from '@/api/models';
import { getModelDetail, getCategoryLabel } from '@/lib/model-metadata';
import type { ModelCategory } from '@/types';
import { useEffect, useRef } from 'react';

function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function ModelsPage() {
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [sortBy, setSortBy] = useState<string>('default');
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deletingVariant, setDeletingVariant] = useState<string | null>(null);

  const { data: models, isLoading, isError, refetch } = useModels({
    category: categoryFilter !== 'all' ? categoryFilter : undefined,
    sort_by: sortBy !== 'default' ? sortBy : undefined,
  });
  const downloadMutation = useDownloadMutation();
  const deleteMutation = useDeleteMutation();
  const { data: diskScan } = useDiskScan();
  const downloadProgress = useModelStore((s) => s.downloadProgress);
  const updateProgress = useModelStore((s) => s.updateDownloadProgress);
  const clearProgress = useModelStore((s) => s.clearDownloadProgress);
  const eventSourcesRef = useRef<Map<string, EventSource>>(new Map());

  // 组件卸载时清理所有 EventSource 连接
  useEffect(() => {
    return () => {
      eventSourcesRef.current.forEach((es) => es.close());
    };
  }, []);

  // 磁盘扫描调试日志
  useEffect(() => {
    if (diskScan) {
      console.log('Disk scan result:', diskScan);
    }
  }, [diskScan]);

  const handleDownload = useCallback(
    (variant: string) => {
      // 发起下载 POST 请求
      downloadMutation.mutate(variant);
      // 关闭旧的 SSE 连接，建立新的进度订阅
      const existing = eventSourcesRef.current.get(variant);
      if (existing) existing.close();
      const es = subscribeDownloadProgress(variant, (p) => updateProgress(p));
      eventSourcesRef.current.set(variant, es);
    },
    [downloadMutation, updateProgress],
  );

  const handleCancelDownload = useCallback(
    (variant: string) => {
      // 取消下载：调 API → 关闭 SSE → 清除 Zustand 进度状态
      cancelDownload(variant);
      const es = eventSourcesRef.current.get(variant);
      if (es) { es.close(); eventSourcesRef.current.delete(variant); }
      clearProgress(variant);
    },
    [clearProgress],
  );

  const filtered = (models ?? [])
    .filter((m) => {
      if (search) {
        const detail = getModelDetail(m.variant);
        const q = search.toLowerCase();
        if (!m.variant.toLowerCase().includes(q) && !detail?.displayName?.toLowerCase().includes(q) && !detail?.description?.toLowerCase().includes(q)) return false;
      }
      if (statusFilter !== 'all') {
        if (m.status !== statusFilter) return false;
      }
      if (categoryFilter !== 'all') {
        if (m.category !== categoryFilter) return false;
      }
      return true;
    });

  const categories: ModelCategory[] = ['llm', 'tts', 'asr', 'vad'];

  if (isError) return <ErrorFallback error={new Error('模型加载失败')} onRetry={() => refetch()} />;

  return (
    <div className="max-w-7xl mx-auto p-4 lg:p-6">
      <div className="flex items-center gap-3 mb-6">
        <h1 className="text-xl font-bold tracking-tight">模型管理</h1>
        {isLoading && <LoadingSpinner />}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2 mb-6">
        <div className="relative flex-1 min-w-[200px] max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="搜索模型..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-8"
          />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-[130px]">
            <SlidersHorizontal className="h-3.5 w-3.5" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">全部状态</SelectItem>
            <SelectItem value="ready">就绪</SelectItem>
            <SelectItem value="downloaded">已下载</SelectItem>
            <SelectItem value="not_downloaded">未下载</SelectItem>
          </SelectContent>
        </Select>
        <Tabs value={categoryFilter} onValueChange={setCategoryFilter}>
          <TabsList>
            <TabsTrigger value="all">全部</TabsTrigger>
            <TabsTrigger value="llm">LLM</TabsTrigger>
            <TabsTrigger value="tts">TTS</TabsTrigger>
            <TabsTrigger value="asr">ASR</TabsTrigger>
            <TabsTrigger value="vad">VAD</TabsTrigger>
          </TabsList>
        </Tabs>
        <Select value={sortBy} onValueChange={setSortBy}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="默认排序" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default">默认</SelectItem>
            <SelectItem value="size_asc">大小（从小到大）</SelectItem>
            <SelectItem value="size_desc">大小（从大到小）</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* 孤立模型检测 */}
      <OrphanedModelsBanner orphaned={diskScan?.orphaned ?? []} />

      {/* Model list */}
      {isLoading && !models ? (
        <div className="flex justify-center py-16"><LoadingSpinner /></div>
      ) : filtered.length === 0 ? (
        <div className="flex justify-center py-16 animate-fade-in">
          <div className="h-14 w-14 rounded-2xl bg-muted/50 flex items-center justify-center mb-4">
            <Search className="h-6 w-6 text-muted-foreground/50" />
          </div>
          <p className="text-sm text-muted-foreground">未找到符合筛选条件的模型。</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* 按 LLM → TTS → ASR → VAD 分类分组渲染模型卡片 */}
          {categories.map((cat) => {
            const catModels = filtered.filter((m) => m.category === cat);
            if (catModels.length === 0) return null;
            const readyCount = catModels.filter((m) => m.status === 'ready').length;
            const downloadedCount = catModels.filter((m) => m.status === 'downloaded').length;
            const totalSize = catModels.reduce((sum, m) => {
              if (m.size_bytes) return sum + m.size_bytes;
              return sum + ((getModelDetail(m.variant)?.sizeGb ?? 0) * 1_000_000_000);
            }, 0);
            return (
              <div key={cat}>
                <div className="flex items-center gap-2 mb-3">
                  <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                    {getCategoryLabel(cat)}
                  </h2>
                  <span className="text-xs text-muted-foreground/60">
                    ({catModels.length} 个模型 · {readyCount} 就绪 · {downloadedCount} 已下载 · {formatBytes(totalSize)} 总计)
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                  {catModels.map((m) => (
                    <ModelCard
                      key={m.variant}
                      model={m}
                      progress={downloadProgress[m.variant]}
                      onDownload={handleDownload}
                      onCancelDownload={handleCancelDownload}
                      onDelete={(v) => setDeleteTarget(v)}
                      isDeleting={deletingVariant === m.variant}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Delete confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}
        title="删除模型文件"
        description="此操作将删除已下载的模型文件并重置其状态。您可以随时重新下载。"
        confirmLabel="删除"
        variant="danger"
        onConfirm={() => {
          if (deleteTarget) {
            setDeletingVariant(deleteTarget);
            deleteMutation.mutate(deleteTarget, {
              onSettled: () => setDeletingVariant(null),
            });
            setDeleteTarget(null);
          }
        }}
      />
    </div>
  );
}
