import { apiFetch } from '@/api/client';
import type { VoiceProfile, VoiceObservation, SavedVoice } from '@/types';

// Voice Profile
export function getVoiceProfile(): Promise<VoiceProfile> {
  return apiFetch('/voice/profile');
}

export function updateVoiceProfile(body: {
  name?: string;
  system_prompt?: string;
  observational_memory_enabled?: boolean;
}): Promise<VoiceProfile> {
  return apiFetch('/voice/profile', { method: 'PATCH', body: JSON.stringify(body) });
}

// Observations
export function listVoiceObservations(): Promise<VoiceObservation[]> {
  return apiFetch('/voice/observations');
}

export function addVoiceObservation(body: {
  category?: string;
  summary: string;
  confidence?: number;
  source_text?: string;
}): Promise<VoiceObservation> {
  return apiFetch('/voice/observations', { method: 'POST', body: JSON.stringify(body) });
}

export function deleteVoiceObservation(id: string): Promise<{ status: string }> {
  return apiFetch(`/voice/observations/${id}`, { method: 'DELETE' });
}

export function clearVoiceObservations(): Promise<{ status: string }> {
  return apiFetch('/voice/observations', { method: 'DELETE' });
}

// Saved Voices
export function listSavedVoices(): Promise<SavedVoice[]> {
  return apiFetch('/voices');
}

export function deleteSavedVoice(id: string): Promise<{ status: string }> {
  return apiFetch(`/voices/${id}`, { method: 'DELETE' });
}
