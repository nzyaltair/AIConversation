import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as vApi from '@/api/voices';
import type { SavedVoice } from '@/types';

export const savedVoiceKeys = {
  all: ['saved-voices'] as const,
};

export function useSavedVoices() {
  return useQuery<SavedVoice[]>({
    queryKey: savedVoiceKeys.all,
    queryFn: vApi.listSavedVoices,
  });
}

export function useDeleteSavedVoice() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => vApi.deleteSavedVoice(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: savedVoiceKeys.all }),
  });
}
