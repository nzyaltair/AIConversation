import { apiFetch } from '@/api/client';

export interface VadResponse {
  dur: number;
  timestamps: [number, number][];
  num_speech_segments: number;
  speech_duration_ratio: number;
}

export async function detectVad(file: File, model?: string): Promise<VadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (model) formData.append('model', model);
  return apiFetch('/audio/vad', {
    method: 'POST',
    body: formData,
    headers: {},
  });
}
