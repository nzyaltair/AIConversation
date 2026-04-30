/**
 * Zustand 客户端状态管理 Store — 模型下载进度。
 *
 * 维护 variant → DownloadProgress 的映射表，
 * 由 SSE 订阅回调更新，供 UI 组件消费实时下载进度。
 */

import { create } from 'zustand';
import type { DownloadProgress } from '@/types';

interface ModelStore {
  downloadProgress: Record<string, DownloadProgress>;
  updateDownloadProgress: (p: DownloadProgress) => void;
  clearDownloadProgress: (variant: string) => void;
}

export const useModelStore = create<ModelStore>((set) => ({
  downloadProgress: {},
  updateDownloadProgress: (p) =>
    set((s) => ({ downloadProgress: { ...s.downloadProgress, [p.variant]: p } })),
  clearDownloadProgress: (variant) =>
    set((s) => {
      const next = { ...s.downloadProgress };
      delete next[variant];
      return { downloadProgress: next };
    }),
}));
