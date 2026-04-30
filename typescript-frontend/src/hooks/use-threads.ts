import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as chatApi from '@/api/chat';
import type { Thread, ChatMessage } from '@/types';

export const threadKeys = {
  all: ['threads'] as const,
  detail: (id: string) => ['threads', id] as const,
  messages: (id: string) => ['threads', id, 'messages'] as const,
};

export function useThreads() {
  return useQuery<Thread[]>({
    queryKey: threadKeys.all,
    queryFn: chatApi.listThreads,
  });
}

export function useThread(threadId: string) {
  return useQuery<Thread>({
    queryKey: threadKeys.detail(threadId),
    queryFn: () => chatApi.getThread(threadId),
    enabled: !!threadId,
  });
}

export function useThreadMessages(threadId: string) {
  return useQuery<ChatMessage[]>({
    queryKey: threadKeys.messages(threadId),
    queryFn: () => chatApi.listMessages(threadId),
    enabled: !!threadId,
  });
}

export function useCreateThread() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { title: string; model_id?: string }) => chatApi.createThread(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: threadKeys.all }),
  });
}

export function useDeleteThread() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => chatApi.deleteThread(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: threadKeys.all }),
  });
}

export function useUpdateThread() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; title: string }) =>
      chatApi.updateThread(id, { title: body.title }),
    onSuccess: () => qc.invalidateQueries({ queryKey: threadKeys.all }),
  });
}
