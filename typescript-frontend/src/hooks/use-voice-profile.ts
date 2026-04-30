import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as vApi from '@/api/voices';
import type { VoiceProfile, VoiceObservation } from '@/types';

export const voiceProfileKeys = {
  profile: ['voice-profile'] as const,
  observations: ['voice-observations'] as const,
};

export function useVoiceProfile() {
  return useQuery<VoiceProfile>({
    queryKey: voiceProfileKeys.profile,
    queryFn: vApi.getVoiceProfile,
  });
}

export function useUpdateVoiceProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: vApi.updateVoiceProfile,
    onSuccess: () => qc.invalidateQueries({ queryKey: voiceProfileKeys.profile }),
  });
}

export function useVoiceObservations() {
  return useQuery<VoiceObservation[]>({
    queryKey: voiceProfileKeys.observations,
    queryFn: vApi.listVoiceObservations,
  });
}

export function useAddObservation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: vApi.addVoiceObservation,
    onSuccess: () => qc.invalidateQueries({ queryKey: voiceProfileKeys.observations }),
  });
}

export function useDeleteObservation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => vApi.deleteVoiceObservation(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: voiceProfileKeys.observations }),
  });
}

export function useClearObservations() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: vApi.clearVoiceObservations,
    onSuccess: () => qc.invalidateQueries({ queryKey: voiceProfileKeys.observations }),
  });
}
