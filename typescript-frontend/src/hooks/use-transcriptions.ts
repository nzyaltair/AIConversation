import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as tApi from '@/api/transcriptions';
import type { TranscriptionRecord } from '@/types';

export const transcriptionKeys = {
  all: ['transcriptions'] as const,
};

export function useTranscriptions() {
  return useQuery<TranscriptionRecord[]>({
    queryKey: transcriptionKeys.all,
    queryFn: tApi.listTranscriptions,
  });
}

export function useDeleteTranscription() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => tApi.deleteTranscription(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: transcriptionKeys.all }),
  });
}
