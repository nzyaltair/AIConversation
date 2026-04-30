/**
 * 模型管理 API 调用层。
 *
 * 封装 /v1/admin/models 端点的所有操作：
 *   模型列表/详情、下载/取消、删除、SSE 下载进度订阅、磁盘扫描。
 */

import { apiFetch } from '@/api/client';
import type { ModelInfo, DownloadProgress, DiskScanResult } from '@/types';

export interface ListModelsParams {
  category?: string;
  sort_by?: string;
}

export function listModels(params?: ListModelsParams): Promise<ModelInfo[]> {
  const searchParams = new URLSearchParams();
  if (params?.category) searchParams.set('category', params.category);
  if (params?.sort_by) searchParams.set('sort_by', params.sort_by);
  const qs = searchParams.toString();
  return apiFetch(`/admin/models/${qs ? `?${qs}` : ''}`);
}

export function getModel(variant: string): Promise<ModelInfo> {
  return apiFetch(`/admin/models/${encodeURIComponent(variant)}`);
}

export function downloadModel(variant: string): Promise<{ status: string }> {
  return apiFetch(`/admin/models/${encodeURIComponent(variant)}/download`, { method: 'POST' });
}

export function cancelDownload(variant: string): Promise<{ status: string }> {
  return apiFetch(`/admin/models/${encodeURIComponent(variant)}/download/cancel`, { method: 'POST' });
}

export function deleteModel(variant: string): Promise<{ status: string }> {
  return apiFetch(`/admin/models/${encodeURIComponent(variant)}`, { method: 'DELETE' });
}

export function loadModel(variant: string): Promise<{ status: string; message?: string }> {
  return apiFetch(`/admin/models/${encodeURIComponent(variant)}/load`, { method: 'POST' });
}

export function unloadModel(variant: string): Promise<{ status: string; message?: string }> {
  return apiFetch(`/admin/models/${encodeURIComponent(variant)}/unload`, { method: 'POST' });
}

export function scanDisk(): Promise<DiskScanResult> {
  return apiFetch('/admin/models/scan-disk');
}

export function subscribeDownloadProgress(
  variant: string,
  onProgress: (p: DownloadProgress) => void,
): EventSource {
  /**
   * 订阅模型下载进度（SSE）。
   *
   * 通过 EventSource 连接后端 /v1/admin/models/{variant}/download/progress，
   * 每当服务端推送新进度时触发 onProgress 回调。
   * 连接出错时自动关闭 EventSource，调用方可通过返回值的 close() 手动取消订阅。
   */
  const url = `/v1/admin/models/${encodeURIComponent(variant)}/download/progress`;
  const es = new EventSource(url);
  es.addEventListener('message', (e) => {
    try {
      const data = JSON.parse(e.data);
      onProgress({ variant, ...data });
    } catch {
      // 跳过无法解析的消息
    }
  });
  es.addEventListener('error', () => {
    // SSE 连接异常时自动关闭
    es.close();
  });
  return es;
}
