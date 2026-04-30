import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as shApi from '@/api/speech-history';
import type { SpeechHistoryRecord } from '@/types';

export const speechHistoryKeys = {
  all: ['tts-generations'] as const,
};

export function useTTSGenerations() {
  return useQuery<SpeechHistoryRecord[]>({
    queryKey: speechHistoryKeys.all,
    queryFn: shApi.listTTSGenerations,
  });
}

export function useDeleteTTSGeneration() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => shApi.deleteTTSGeneration(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: speechHistoryKeys.all }),
  });
}
