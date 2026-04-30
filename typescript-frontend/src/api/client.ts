const BASE = '/v1';

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export function apiUrl(path: string): string {
  return `${BASE}${path}`;
}

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      // not JSON
    }
    const msg = typeof detail === 'object' && detail !== null && 'message' in detail
      ? String((detail as Record<string, unknown>).message)
      : res.statusText;
    throw new ApiError(res.status, msg, detail);
  }

  const text = await res.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

export function apiFetchSSE<T>(
  path: string,
  body: unknown,
  onEvent: (data: T) => void,
  options?: { signal?: AbortSignal; method?: string; onError?: (error: Error) => void },
): AbortController {
  const controller = new AbortController();
  const signal = options?.signal
    ? combineSignals(controller.signal, options.signal)
    : controller.signal;

  (async () => {
    try {
      const res = await fetch(`${BASE}${path}`, {
        method: options?.method ?? 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      });

      if (!res.ok) {
        let detail: unknown;
        try { detail = await res.json(); } catch { /* */ }
        const msg = typeof detail === 'object' && detail !== null && 'message' in detail
          ? String((detail as Record<string, unknown>).message)
          : res.statusText;
        throw new ApiError(res.status, msg, detail);
      }

      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === '[DONE]' || trimmed.startsWith(':')) continue;
          if (trimmed.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(trimmed.slice(6));
              onEvent(parsed as T);
            } catch {
              // skip malformed lines
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (err instanceof ApiError) {
        options?.onError?.(err);
        return;
      }
      if (err instanceof Error) {
        options?.onError?.(err);
      }
    }
  })();

  return controller;
}

function combineSignals(s1: AbortSignal, s2: AbortSignal): AbortSignal {
  const controller = new AbortController();
  const abort = () => controller.abort();
  s1.addEventListener('abort', abort);
  s2.addEventListener('abort', abort);
  if (s1.aborted || s2.aborted) controller.abort();
  return controller.signal;
}

export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}
