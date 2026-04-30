import { useState, useRef, useCallback, useEffect } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ModelSelector } from '@/components/shared/model-selector';
import { ThreadList } from '@/components/chat/thread-list';
import { ChatMessageBubble } from '@/components/chat/chat-message';
import { ChatInput } from '@/components/chat/chat-input';
import { ThinkingPanel } from '@/components/chat/thinking-panel';
import { LoadingSpinner } from '@/components/shared/loading-spinner';
import { ConfirmDialog } from '@/components/shared/confirm-dialog';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { chatCompletionsStream, saveMessages } from '@/api/chat';
import { useThreads, useThreadMessages, useCreateThread, useDeleteThread, threadKeys } from '@/hooks/use-threads';
import { useModels } from '@/hooks/use-models';
import { useQueryClient } from '@tanstack/react-query';
import { getModelDetail } from '@/lib/model-metadata';
import { ApiError } from '@/api/client';
import type { ChatMessage } from '@/types';
import { Settings2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function ChatPage() {
  const { data: models } = useModels();
  const { data: threads, isLoading: threadsLoading } = useThreads();
  const createThread = useCreateThread();
  const deleteThread = useDeleteThread();
  const queryClient = useQueryClient();

  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('Qwen3.5-0.8B.Q4_K_M');
  const [streamingContent, setStreamingContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [thinkingContent, setThinkingContent] = useState('');
  const [showParams, setShowParams] = useState(false);
  const [temperature, setTemperature] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [thinkingEnabled, setThinkingEnabled] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [localMessages, setLocalMessages] = useState<ChatMessage[]>([]);

  const { data: serverMessages, isLoading: msgsLoading } = useThreadMessages(activeThreadId ?? '');

  const messages = serverMessages ?? localMessages;

  const currentThread = activeThreadId
    ? threads?.find((t) => t.id === activeThreadId)
    : null;

  const modelDetail = getModelDetail(selectedModel);
  const isThinkingModel = modelDetail?.isThinking;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Ensure a thread is selected (create one if none exist)
  useEffect(() => {
    if (threads && threads.length > 0 && !activeThreadId) {
      setActiveThreadId(threads[0].id);
    }
  }, [threads, activeThreadId]);

  const handleNewThread = useCallback(async () => {
    const thread = await createThread.mutateAsync({ title: '新对话', model_id: selectedModel });
    setActiveThreadId(thread.id);
    setLocalMessages([]);
  }, [createThread, selectedModel]);

  const handleDeleteThread = useCallback(
    (id: string) => {
      setDeleteTarget(id);
    },
    [],
  );

  const confirmDeleteThread = useCallback(() => {
    if (!deleteTarget) return;
    deleteThread.mutate(deleteTarget);
    if (activeThreadId === deleteTarget) {
      setActiveThreadId(threads?.find((t) => t.id !== deleteTarget)?.id ?? null);
      setLocalMessages([]);
    }
    setDeleteTarget(null);
  }, [deleteTarget, deleteThread, activeThreadId, threads]);

  const handleSend = useCallback(
    async (content: string) => {
      if (!content.trim()) return;

      setError(null);

      // Auto-create thread if none active
      let effectiveThreadId = activeThreadId;
      if (!effectiveThreadId) {
        try {
          const thread = await createThread.mutateAsync({ title: content.trim().slice(0, 50), model_id: selectedModel });
          effectiveThreadId = thread.id;
          setActiveThreadId(effectiveThreadId);
        } catch {
          // fall through — message will be sent without persistence
        }
      }

      const userMsg: ChatMessage = {
        id: `temp_${Date.now()}`,
        thread_id: effectiveThreadId ?? '',
        role: 'user',
        content: content.trim(),
        created_at: new Date().toISOString(),
        model_id: selectedModel,
      };

      setLocalMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      setStreamingContent('');
      setThinkingContent('');

      const fullMessages = [
        ...messages.map((m) => ({ role: m.role, content: m.content })),
        { role: 'user', content: content.trim() },
      ];

      const onDelta = (text: string) => {
        setStreamingContent((prev) => prev + text);
      };

      const onThinking = (thinking: string) => {
        setThinkingContent(thinking);
      };

      const onDone = () => {
        setStreamingContent((final) => {
          const assistantMsg: ChatMessage = {
            id: `temp_${Date.now() + 1}`,
            thread_id: activeThreadId ?? '',
            role: 'assistant',
            content: final,
            created_at: new Date().toISOString(),
            model_id: selectedModel,
          };
          setLocalMessages((prev) => [...prev, assistantMsg]);

          // 持久化消息到后端
          if (effectiveThreadId && final) {
            const messagesToSave = [
              { role: 'user', content: content.trim(), model_id: selectedModel },
              { role: 'assistant', content: final, model_id: selectedModel },
            ];
            const tid = effectiveThreadId;
            saveMessages(tid, messagesToSave).then(() => {
              queryClient.invalidateQueries({ queryKey: threadKeys.messages(tid) });
              queryClient.invalidateQueries({ queryKey: threadKeys.all });
            }).catch((err) => {
              console.warn('Failed to persist messages:', err);
            });
          }

          return '';
        });
        setIsStreaming(false);
      };

      const onError = (err: Error) => {
        const msg = err instanceof ApiError
          ? err.message
          : '网络错误，请检查网络连接后重试。';
        setError(msg);
        setIsStreaming(false);
        setStreamingContent('');
      };

      abortRef.current = chatCompletionsStream(
        {
          model: selectedModel,
          messages: fullMessages,
          stream: true,
          temperature,
          max_tokens: maxTokens,
          thinking: isThinkingModel ? thinkingEnabled : undefined,
        },
        onDelta,
        onDone,
        onThinking,
        onError,
      );
    },
    [activeThreadId, selectedModel, messages, isThinkingModel, temperature, maxTokens, createThread, queryClient],
  );

  return (
    <div className="flex h-[calc(100vh-3rem)] lg:h-screen">
      {/* Thread sidebar */}
      <div className="hidden md:block w-72 border-r border-border bg-card/50">
        <ThreadList
          threads={threads ?? []}
          activeThreadId={activeThreadId}
          isLoading={threadsLoading}
          onThreadClick={setActiveThreadId}
          onDeleteThread={handleDeleteThread}
          onNewThread={handleNewThread}
        />
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-border/50 bg-card/40 backdrop-blur-sm shrink-0">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <h2 className="text-sm font-medium truncate">
              {currentThread?.title ?? '聊天'}
            </h2>
            <div className="w-48 shrink-0">
              <ModelSelector
                models={models ?? []}
                category="llm"
                value={selectedModel}
                onChange={setSelectedModel}
              />
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowParams(!showParams)}
          >
            <Settings2 className="h-4 w-4" />
          </Button>
        </div>

        {/* Parameter panel */}
        {showParams && (
          <div className="px-4 py-3 border-b border-border bg-muted/20 space-y-3">
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="flex justify-between mb-1">
                  <Label className="text-xs">温度：{temperature.toFixed(2)}</Label>
                </div>
                <Slider value={[temperature]} min={0} max={2} step={0.05} onValueChange={([v]) => setTemperature(v)} />
              </div>
              <div className="flex-1">
                <div className="flex justify-between mb-1">
                  <Label className="text-xs">最大令牌数：{maxTokens}</Label>
                </div>
                <Slider value={[maxTokens]} min={64} max={8192} step={64} onValueChange={([v]) => setMaxTokens(v)} />
              </div>
            </div>
            {isThinkingModel && (
              <div className="flex items-center gap-3">
                <Label className="text-xs">思考模式</Label>
                <button
                  type="button"
                  onClick={() => setThinkingEnabled(!thinkingEnabled)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                    thinkingEnabled ? 'bg-primary' : 'bg-muted-foreground/30'
                  }`}
                >
                  <span
                    className={`inline-block h-3.5 w-3.5 rounded-full bg-background transition-transform ${
                      thinkingEnabled ? 'translate-x-4' : 'translate-x-1'
                    }`}
                  />
                </button>
                <span className="text-xs text-muted-foreground">
                  {thinkingEnabled ? '开' : '关'}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Error banner */}
        {error && (
          <div className="flex items-center justify-between px-4 py-2 bg-destructive/10 border-b border-destructive/20 text-destructive text-sm shrink-0">
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              className="hover:bg-destructive/10 rounded p-0.5"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {/* Messages */}
        <ScrollArea className="flex-1">
          <div className="max-w-3xl mx-auto p-4 space-y-4">
            {msgsLoading && activeThreadId ? (
              <div className="flex justify-center py-8"><LoadingSpinner /></div>
            ) : messages.length === 0 && !isStreaming ? (
              <p className="text-center text-muted-foreground py-16">
                发送消息开始对话
              </p>
            ) : (
              messages.map((msg) => (
                <ChatMessageBubble key={msg.id} message={msg} />
              ))
            )}
            {isStreaming && thinkingContent && (
              <ThinkingPanel content={thinkingContent} />
            )}
            {isStreaming && streamingContent && (
              <ChatMessageBubble
                message={{
                  id: 'streaming',
                  thread_id: activeThreadId ?? '',
                  role: 'assistant',
                  content: streamingContent,
                  created_at: new Date().toISOString(),
                  model_id: selectedModel,
                }}
                isStreaming
              />
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input */}
        <div className="max-w-3xl mx-auto w-full">
          <ChatInput
            onSend={handleSend}
            isStreaming={isStreaming}
            onCancelStream={() => abortRef.current?.abort()}
          />
        </div>
      </div>

      {/* Delete confirmation */}
      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}
        title="删除对话"
        description="此操作将永久删除该对话及其所有消息。"
        confirmLabel="删除"
        variant="danger"
        onConfirm={confirmDeleteThread}
      />
    </div>
  );
}
