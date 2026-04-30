import { apiFetch, apiFetchSSE } from '@/api/client';

export interface TranscribeRequest {
  file: File;
  model?: string;
  language?: string;
  response_format?: 'json' | 'text' | 'srt' | 'vtt' | 'verbose_json';
  timestamp_granularities?: ('word' | 'segment')[];
}

export interface TranscribeResponse {
  text: string;
  language?: string;
  duration?: number;
  segments?: Array<{
    start: number;
    end: number;
    text: string;
    words?: Array<{ word: string; start: number; end: number }>;
  }>;
}

export async function transcribeAudio(req: TranscribeRequest): Promise<TranscribeResponse> {
  const formData = new FormData();
  formData.append('file', req.file);
  if (req.model) formData.append('model', req.model);
  if (req.language) formData.append('language', req.language);
  if (req.response_format) formData.append('response_format', req.response_format);
  if (req.timestamp_granularities) {
    formData.append('timestamp_granularities', JSON.stringify(req.timestamp_granularities));
  }
  return apiFetch('/audio/transcriptions', {
    method: 'POST',
    body: formData,
    headers: {}, // let browser set Content-Type for multipart
  });
}

export interface TTSRequest {
  model: string;
  input: string;
  voice?: string;
  speed?: number;
  instruct?: string;
  response_format?: string;
  stream?: boolean;
}

export async function generateSpeech(req: TTSRequest): Promise<Blob> {
  const res = await fetch('/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...req, stream: false }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({})) as Record<string, unknown>;
    const errDetail = (body as { error?: Record<string, unknown> }).error;
    throw new Error(String(errDetail?.message ?? (body as Record<string, unknown>).message ?? res.statusText));
  }
  return res.blob();
}

export interface VoicesResponse {
  voices: string[];
}

export async function fetchVoices(model?: string): Promise<string[]> {
  const params = model ? `?model=${encodeURIComponent(model)}` : '';
  const res = await apiFetch<VoicesResponse>(`/audio/voices${params}`);
  return res.voices;
}

export function generateSpeechStream(
  req: TTSRequest,
  onChunk: (base64: string) => void,
  onDone: () => void,
): AbortController {
  return apiFetchSSE<{ event: string; audio?: string; stats?: unknown }>(
    '/audio/speech',
    { ...req, stream: true },
    (data) => {
      if ((data.event === 'chunk' || data.event === 'start') && data.audio) {
        onChunk(data.audio);
      }
      if (data.event === 'done' || data.event === 'final') {
        onDone();
      }
    },
  );
}
