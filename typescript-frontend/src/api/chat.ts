import { apiFetch, apiFetchSSE } from '@/api/client';
import type { Thread, ChatMessage, ChatCompletionChunk } from '@/types';

// Threads
export function listThreads(): Promise<Thread[]> {
  return apiFetch('/chat/threads/');
}

export function createThread(body: { title: string; model_id?: string }): Promise<Thread> {
  return apiFetch('/chat/threads/', { method: 'POST', body: JSON.stringify(body) });
}

export function getThread(threadId: string): Promise<Thread> {
  return apiFetch(`/chat/threads/${threadId}`);
}

export function updateThread(threadId: string, body: { title?: string }): Promise<Thread> {
  return apiFetch(`/chat/threads/${threadId}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  });
}

export function deleteThread(threadId: string): Promise<{ status: string }> {
  return apiFetch(`/chat/threads/${threadId}`, { method: 'DELETE' });
}

export function listMessages(threadId: string): Promise<ChatMessage[]> {
  return apiFetch(`/chat/threads/${threadId}/messages`);
}

// Batch-save messages without triggering inference
export function saveMessages(
  threadId: string,
  messages: { role: string; content: string; model_id?: string }[],
): Promise<{ status: string; messages: ChatMessage[] }> {
  return apiFetch(`/chat/threads/${threadId}/messages/batch`, {
    method: 'POST',
    body: JSON.stringify({ messages }),
  });
}

// Chat completions (SSE)
export function chatCompletionsStream(
  body: {
    model: string;
    messages: { role: string; content: string }[];
    stream: true;
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    thinking?: boolean;
  },
  onDelta: (content: string) => void,
  onDone: (finishReason: string | null) => void,
  onThinking?: (thinking: string) => void,
  onError?: (error: Error) => void,
): AbortController {
  return apiFetchSSE<ChatCompletionChunk>(
    '/chat/completions',
    body,
    (chunk) => {
      const choice = chunk.choices?.[0];
      if (choice?.delta?.thinking && onThinking) {
        onThinking(choice.delta.thinking);
      }
      if (choice?.delta?.content) {
        onDelta(choice.delta.content);
      }
      if (choice?.finish_reason) {
        onDone(choice.finish_reason);
      }
    },
    { onError },
  );
}

// Thread message send (SSE)
export function sendMessageStream(
  threadId: string,
  body: { content: string; model?: string; temperature?: number; max_tokens?: number; thinking?: boolean },
  onDelta: (content: string) => void,
  onDone: () => void,
  signal?: AbortSignal,
): AbortController {
  return apiFetchSSE<{ event: string; text?: string; content?: string }>(
    `/chat/threads/${threadId}/messages`,
    { ...body, stream: true },
    (data) => {
      if (data.event === 'delta' || data.event === 'start') {
        if (data.text) onDelta(data.text);
        if (data.content) onDelta(data.content);
      }
      if (data.event === 'done') onDone();
    },
    { signal },
  );
}

// Send message (non-streaming)
export function sendMessage(
  threadId: string,
  body: { content: string; model?: string },
): Promise<ChatMessage> {
  return apiFetch(`/chat/threads/${threadId}/messages`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}
