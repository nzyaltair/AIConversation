import { apiFetch } from '@/api/client';
import type { SpeechHistoryRecord } from '@/types';

export function listTTSGenerations(): Promise<SpeechHistoryRecord[]> {
  return apiFetch('/text-to-speech-generations');
}

export function getTTSGeneration(id: string): Promise<SpeechHistoryRecord> {
  return apiFetch(`/text-to-speech-generations/${id}`);
}

export function createTTSGeneration(body: {
  model_id?: string;
  speaker?: string;
  input_text?: string;
  audio_duration_secs?: number;
  generation_time_ms?: number;
  audio_base64?: string;
}): Promise<SpeechHistoryRecord> {
  return apiFetch('/text-to-speech-generations', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function deleteTTSGeneration(id: string): Promise<{ status: string }> {
  return apiFetch(`/text-to-speech-generations/${id}`, { method: 'DELETE' });
}
