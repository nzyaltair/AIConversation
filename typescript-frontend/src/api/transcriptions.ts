import { apiFetch } from '@/api/client';
import type { TranscriptionRecord } from '@/types';

export function listTranscriptions(): Promise<TranscriptionRecord[]> {
  return apiFetch('/transcriptions/');
}

export function getTranscription(id: string): Promise<TranscriptionRecord> {
  return apiFetch(`/transcriptions/${id}`);
}

export function createTranscription(formData: FormData): Promise<TranscriptionRecord> {
  return apiFetch('/transcriptions/', {
    method: 'POST',
    body: formData,
    headers: {},
  });
}

export function deleteTranscription(id: string): Promise<{ status: string }> {
  return apiFetch(`/transcriptions/${id}`, { method: 'DELETE' });
}
