import { useRef, useCallback, useEffect, useState } from 'react';

interface UseSSEOptions<T> {
  onEvent: (data: T) => void;
  onDone?: () => void;
}

export function useSSE<T>(url: string, body: unknown, options: UseSSEOptions<T>) {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const optsRef = useRef(options);
  optsRef.current = options;

  const start = useCallback(() => {
    if (controllerRef.current) controllerRef.current.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setIsConnected(true);
    setError(null);

    (async () => {
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`SSE error: ${res.status}`);

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
                optsRef.current.onEvent(parsed as T);
              } catch {
                // skip
              }
            }
          }
        }
        optsRef.current.onDone?.();
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        setIsConnected(false);
      }
    })();
  }, [url, body]);

  const abort = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsConnected(false);
  }, []);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  return { isConnected, error, start, abort };
}
