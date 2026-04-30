/**
 * React Query 模型管理 hooks。
 *
 * 使用 TanStack React Query 封装模型数据获取和变更操作。
 * - useModels: 自动轮询（每 5s）获取模型列表，保持 UI 状态同步
 * - useDiskScan: 页面挂载时扫描磁盘检测孤立模型
 * - useDownloadMutation / useDeleteMutation: 变更操作完成后自动刷新模型列表缓存
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as modelsApi from '@/api/models';
import type { ModelInfo, DiskScanResult } from '@/types';

export const modelKeys = {
  all: ['models'] as const,
  detail: (variant: string) => ['models', variant] as const,
};

export interface UseModelsOptions {
  category?: string;
  sort_by?: string;
}

export function useModels(options?: UseModelsOptions) {
  return useQuery<ModelInfo[]>({
    queryKey: [...modelKeys.all, options ?? {}],
    queryFn: () => modelsApi.listModels(options),
    refetchInterval: 5000, // 每 5s 自动轮询，捕获下载进度和状态变更
  });
}

export function useModel(variant: string) {
  return useQuery<ModelInfo>({
    queryKey: modelKeys.detail(variant),
    queryFn: () => modelsApi.getModel(variant),
    enabled: !!variant,
  });
}

export function useDownloadMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (variant: string) => modelsApi.downloadModel(variant),
    onSuccess: () => qc.invalidateQueries({ queryKey: modelKeys.all }), // 操作成功后刷新模型列表缓存
  });
}

export function useDeleteMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (variant: string) => modelsApi.deleteModel(variant),
    onSuccess: () => qc.invalidateQueries({ queryKey: modelKeys.all }),
  });
}

export function useDiskScan() {
  return useQuery<DiskScanResult>({
    queryKey: ['models', 'disk-scan'],
    queryFn: () => modelsApi.scanDisk(),
    staleTime: Infinity,
    retry: 1,
  });
}
