import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { StatusBar } from '@/components/voice/status-bar';
import { VoiceOrb } from '@/components/voice/voice-orb';
import { ConversationStream } from '@/components/voice/conversation-stream';
import { ApiSettingsSheet } from '@/components/voice/api-settings-sheet';
import { VolumeVisualizer } from '@/components/voice/volume-visualizer';
import { VoiceControlBar } from '@/components/voice/voice-control-bar';
import { useVoiceStore } from '@/stores/voice-store';
import { useConversationApiSettings } from '@/stores/conversation-api-settings-store';
import { useModels } from '@/hooks/use-models';
import { loadModel, unloadModel } from '@/api/models';
import { VoiceRealtimeClient } from '@/api/voice-realtime';
import { cn } from '@/lib/utils';
import type { VoiceSessionState, ConversationBubble, ApiConfig } from '@/types';

export function ConversationApiPage() {
  // ── Stores ──
  const voiceStore = useVoiceStore();
  const settings = useConversationApiSettings();
  const { data: models } = useModels();

  // ── Local state ──
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [latencyMs, setLatencyMs] = useState<number | undefined>(undefined);
  const [isConnected, setIsConnected] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState<boolean | undefined>(undefined);
  const [micError, setMicError] = useState<string | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [loadingModels, setLoadingModels] = useState<Record<string, boolean>>({});

  // ── Refs ──
  const clientRef = useRef<VoiceRealtimeClient | null>(null);
  const currentBubbleIdRef = useRef<string | null>(null);
  const thinkingAccumRef = useRef<string>('');
  const latencyIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── API config (derived from settings, stable ref for callbacks) ──
  const apiConfig = useMemo<ApiConfig>(
    () => ({
      base_url: settings.baseUrl,
      api_key: settings.apiKey,
      model: settings.modelId,
      reasoning_effort: settings.thinkingEnabled
        ? settings.reasoningEffort
        : undefined,
    }),
    [settings.baseUrl, settings.apiKey, settings.modelId, settings.thinkingEnabled, settings.reasoningEffort],
  );

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      clientRef.current?.disconnect();
    };
  }, []);

  // ── Latency simulation ──
  useEffect(() => {
    if (isConnected) {
      const interval = setInterval(() => {
        setLatencyMs(Math.floor(Math.random() * 20) + 15);
      }, 2000);
      latencyIntervalRef.current = interval;
      return () => clearInterval(interval);
    } else {
      setLatencyMs(undefined);
      if (latencyIntervalRef.current) {
        clearInterval(latencyIntervalRef.current);
        latencyIntervalRef.current = null;
      }
    }
  }, [isConnected]);

  // ── WebSocket callbacks ──
  const callbacks = useMemo(
    () => ({
      onStateChange: (state: string) => {
        console.log('[ConvAPI] State change →', state);
        voiceStore.setSessionState(state as VoiceSessionState);
      },
      onUserTranscript: (text: string) => {
        console.log('[ConvAPI] User transcript:', text.slice(0, 100));
        const bubble: ConversationBubble = {
          id: crypto.randomUUID(),
          type: 'user' as const,
          text,
          timestamp: Date.now(),
        };
        voiceStore.addBubble(bubble);
      },
      onAssistantTextDelta: (delta: string) => {
        console.log('[ConvAPI] Assistant delta:', delta.slice(0, 80));
        const currentId = currentBubbleIdRef.current;
        if (currentId) {
          voiceStore.appendToBubble(currentId, delta);
        } else {
          const id = crypto.randomUUID();
          currentBubbleIdRef.current = id;
          const bubble: ConversationBubble = {
            id,
            type: 'assistant' as const,
            text: delta,
            isStreaming: true,
            timestamp: Date.now(),
          };
          voiceStore.addBubble(bubble);
          console.log('[ConvAPI] Created new assistant bubble:', id);
        }
      },
      onAssistantTextFinal: (text: string, thinking?: string) => {
        console.log('[ConvAPI] Assistant final:', text.slice(0, 100), 'thinking:', thinking?.slice(0, 50) || '(none)');
        const currentId = currentBubbleIdRef.current;
        if (currentId) {
          voiceStore.setBubbleStreaming(currentId, false);
          voiceStore.updateBubble(currentId, text);
          const accumulatedThinking = thinkingAccumRef.current || thinking;
          if (accumulatedThinking) {
            voiceStore.updateBubbleThinking(currentId, accumulatedThinking);
          }
          thinkingAccumRef.current = '';
          currentBubbleIdRef.current = null;
        } else {
          // Bug 4 fix: Create bubble even without prior delta (API error case)
          console.log('[ConvAPI] No current bubble — creating final bubble directly');
          const id = crypto.randomUUID();
          const bubble: ConversationBubble = {
            id,
            type: 'assistant' as const,
            text,
            isStreaming: false,
            thinking: thinking || thinkingAccumRef.current || undefined,
            timestamp: Date.now(),
          };
          voiceStore.addBubble(bubble);
          thinkingAccumRef.current = '';
          currentBubbleIdRef.current = null;
        }
      },
      onThinking: (text: string) => {
        console.log('[ConvAPI] Thinking:', text.slice(0, 80));
        thinkingAccumRef.current += text;
      },
      onTurnDone: () => {
        console.log('[ConvAPI] Turn done');
      },
      onError: (code: string, message: string) => {
        console.error(`[ConvAPI] Error from server — code: "${code}", message: "${message}"`);
        if (code === 'mic_error' || code === 'microphone_permission_denied') {
          setMicError(message);
        } else if (code === 'vad_unavailable') {
          // Bug 1 fix: Handle VAD unavailable error
          setMicError('VAD 引擎未加载。请在模型管理页面下载 FireRedVAD 模型后重试。');
          voiceStore.setSessionState('idle');
          clientRef.current?.disconnect();
          clientRef.current = null;
        } else if (code === 'api_key_missing') {
          setMicError('API 密钥未配置，请打开设置并输入您的 API 密钥。');
        } else if (code === 'api_error') {
          setMicError(message);
        } else if (
          code === 'gpu_unavailable' ||
          code === 'cuda_out_of_memory' ||
          code === 'gpu_resource_exhausted' ||
          message?.toLowerCase().includes('cuda') ||
          message?.toLowerCase().includes('out of memory')
        ) {
          setGpuAvailable(false);
          setGpuError(message);
        }
      },
      onConnectionChange: (connected: boolean) => {
        console.log('[ConvAPI] Connection change:', connected);
        setIsConnected(connected);
        if (!connected) {
          currentBubbleIdRef.current = null;
          thinkingAccumRef.current = '';
        }
      },
      onVolumeLevel: (level: number) => {
        setVolumeLevel(level);
      },
    }),
    [voiceStore],
  );

  // ── Actions ──

  const startSession = useCallback(async () => {
    if (clientRef.current) {
      console.log('[ConvAPI] startSession: already active, ignoring');
      return;
    }

    console.log('[ConvAPI] ===== startSession =====');
    console.log('[ConvAPI] Settings:', {
      baseUrl: settings.baseUrl,
      apiKey: settings.apiKey ? `${settings.apiKey.slice(0, 8)}...` : '(empty)',
      modelId: settings.modelId,
      temperature: settings.temperature,
      maxTokens: settings.maxTokens,
      thinkingEnabled: settings.thinkingEnabled,
      vadModelVariant: settings.vadModelVariant,
      asrModelVariant: settings.asrModelVariant,
      ttsModelVariant: settings.ttsModelVariant,
      vadThreshold: settings.vadThreshold,
    });
    setMicError(null);

    // ── Bug 3 fix: Validate API key before connecting ──
    if (!settings.apiKey.trim()) {
      console.error('[ConvAPI] startSession: API key is empty, aborting');
      setMicError('需要 API 密钥，请打开设置并输入您的 API 密钥。');
      voiceStore.setSessionState('idle');
      return;
    }

    // ── Load models on demand (VAD/ASR/TTS + external-api LLM engine) ──
    const variants = [
      settings.vadModelVariant,
      settings.asrModelVariant,
      settings.ttsModelVariant,
      'external-api',
    ];
    console.log('[ConvAPI] Loading models:', variants);

    const loadingMap: Record<string, boolean> = {};
    for (const v of variants) loadingMap[v] = true;
    setLoadingModels(loadingMap);

    const results = await Promise.allSettled(
      variants.map((v) => loadModel(v).catch(() => undefined)),
    );
    const succeeded = results.filter(
      (r) => r.status === 'fulfilled' && r.value !== undefined,
    ).length;
    results.forEach((r, i) => {
      console.log(`[ConvAPI] Model "${variants[i]}": ${r.status}`, r.status === 'fulfilled' ? r.value : r.reason);
    });
    console.log(`[ConvAPI] Models loaded: ${succeeded}/${variants.length} succeeded`);
    setGpuAvailable(succeeded > 0);
    setLoadingModels({});

    voiceStore.setSessionState('connecting');

    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host || 'localhost:8000';
      const wsUrl = `${protocol}//${host}/v1/voice/realtime/ws`;
      console.log('[ConvAPI] Connecting to WebSocket:', wsUrl);

      const client = new VoiceRealtimeClient(callbacks);
      clientRef.current = client;

      await client.connect(wsUrl);
      console.log('[ConvAPI] WebSocket connected');

      client.sessionStart();
      console.log('[ConvAPI] Sent session_start');

      const inputStreamPayload = {
        model_variants: {
          vad: settings.vadModelVariant,
          asr: settings.asrModelVariant,
          llm: 'external-api',
          tts: settings.ttsModelVariant,
        },
        llm_settings: {
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          thinking_enabled: settings.thinkingEnabled,
          system_prompt: settings.systemPrompt?.slice(0, 100),
        },
        tts_settings: {
          speaker: settings.ttsSpeaker,
          speed: settings.ttsSpeed,
          voice_design_instruct: settings.voiceDesignInstruct,
        },
        vad_config: {
          threshold: settings.vadThreshold,
          min_speech_ms: settings.vadMinSpeechMs,
          silence_duration_ms: settings.vadSilenceDurationMs,
          max_utterance_ms: settings.vadMaxUtteranceMs,
        },
        api_config: { ...apiConfig, api_key: apiConfig.api_key ? '***masked***' : '(empty)' },
      };
      console.log('[ConvAPI] Sending input_stream_start:', inputStreamPayload);

      client.inputStreamStart({
        model_variants: {
          vad: settings.vadModelVariant,
          asr: settings.asrModelVariant,
          llm: 'external-api',
          tts: settings.ttsModelVariant,
        },
        llm_settings: {
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          thinking_enabled: settings.thinkingEnabled,
          system_prompt: settings.systemPrompt,
        },
        tts_settings: {
          speaker: settings.ttsSpeaker,
          speed: settings.ttsSpeed,
          voice_design_instruct: settings.voiceDesignInstruct,
        },
        vad_config: {
          threshold: settings.vadThreshold,
          min_speech_ms: settings.vadMinSpeechMs,
          silence_duration_ms: settings.vadSilenceDurationMs,
          max_utterance_ms: settings.vadMaxUtteranceMs,
        },
        api_config: apiConfig,
      });

      voiceStore.setSessionState('listening');
      console.log('[ConvAPI] Session started, state → listening');
    } catch (err) {
      console.error('[ConvAPI] Failed to start voice session:', err);
      voiceStore.setSessionState('idle');
      clientRef.current = null;
    }
  }, [callbacks, settings, voiceStore, apiConfig]);

  const stopSession = useCallback(() => {
    console.log('[ConvAPI] ===== stopSession =====');
    const client = clientRef.current;
    if (client) {
      client.inputStreamStop();
      client.disconnect();
      clientRef.current = null;
    }
    currentBubbleIdRef.current = null;
    thinkingAccumRef.current = '';
    setGpuError(null);
    voiceStore.clearBubbles();
    voiceStore.setSessionState('idle');
  }, [voiceStore]);

  const interrupt = useCallback(() => {
    clientRef.current?.interrupt();
  }, []);

  const toggleTtsMute = useCallback(() => {
    const store = useVoiceStore.getState();
    const next = !store.isTtsMuted;
    store.setTtsMuted(next);
    clientRef.current?.setTtsMuted(next);
  }, []);

  const toggleInputDisabled = useCallback(() => {
    const store = useVoiceStore.getState();
    const next = !store.isInputDisabled;
    store.setInputDisabled(next);
    clientRef.current?.setInputDisabled(next);
  }, []);

  // ── Model load/unload handlers ──

  const handleLoadModel = useCallback(async (variant: string) => {
    try {
      await loadModel(variant);
    } catch (err) {
      console.warn('Failed to load model:', variant, err);
    }
  }, []);

  const handleUnloadModel = useCallback(async (variant: string) => {
    try {
      await unloadModel(variant);
    } catch (err) {
      console.warn('Failed to unload model:', variant, err);
    }
  }, []);

  const handleToggleSettings = useCallback(() => {
    setSettingsOpen((prev) => !prev);
  }, []);

  // ── Derived state ──
  const { sessionState, isTtsMuted, isInputDisabled, bubbles } = voiceStore;

  // ── Sync settings changes to active session (debounced) ──
  const syncTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (sessionState === 'idle' || !clientRef.current) return;

    if (syncTimerRef.current) clearTimeout(syncTimerRef.current);
    syncTimerRef.current = setTimeout(() => {
      if (!clientRef.current) return;
      clientRef.current.updateSettings({
        model_variants: {
          vad: settings.vadModelVariant,
          asr: settings.asrModelVariant,
          llm: 'external-api',
          tts: settings.ttsModelVariant,
        },
        llm_settings: {
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          thinking_enabled: settings.thinkingEnabled,
          system_prompt: settings.systemPrompt,
        },
        tts_settings: {
          speaker: settings.ttsSpeaker,
          speed: settings.ttsSpeed,
          voice_design_instruct: settings.voiceDesignInstruct,
        },
        vad_config: {
          threshold: settings.vadThreshold,
          min_speech_ms: settings.vadMinSpeechMs,
          silence_duration_ms: settings.vadSilenceDurationMs,
          max_utterance_ms: settings.vadMaxUtteranceMs,
        },
        api_config: apiConfig,
      });
      syncTimerRef.current = null;
    }, 300);

    return () => {
      if (syncTimerRef.current) {
        clearTimeout(syncTimerRef.current);
        syncTimerRef.current = null;
      }
    };
  }, [
    sessionState,
    settings.ttsModelVariant,
    settings.asrModelVariant,
    settings.vadModelVariant,
    settings.temperature,
    settings.maxTokens,
    settings.thinkingEnabled,
    settings.systemPrompt,
    settings.ttsSpeaker,
    settings.ttsSpeed,
    settings.voiceDesignInstruct,
    settings.vadThreshold,
    settings.vadMinSpeechMs,
    settings.vadSilenceDurationMs,
    settings.vadMaxUtteranceMs,
    apiConfig,
  ]);

  // Build model variant labels (LLM shows "External API")
  const modelVariants = useMemo(
    () => ({
      llm: '外部 API',
      tts: settings.ttsModelVariant,
      asr: settings.asrModelVariant,
      vad: settings.vadModelVariant,
    }),
    [settings],
  );

  return (
    <div className="flex flex-col h-full max-h-screen overflow-hidden">
      {/* Status bar */}
      <StatusBar
        sessionState={sessionState}
        isConnected={isConnected}
        modelVariants={modelVariants}
        latencyMs={latencyMs}
        gpuAvailable={gpuAvailable}
        onToggleSettings={handleToggleSettings}
      />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden relative">
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Voice orb */}
          <div
            className={cn(
              'flex items-center justify-center transition-all duration-500',
              bubbles.length > 0 ? 'h-28 shrink-0' : 'flex-1',
            )}
          >
            <VoiceOrb state={sessionState} />
          </div>

          {/* Conversation stream */}
          {bubbles.length > 0 && (
            <ConversationStream
              bubbles={bubbles}
              onClear={voiceStore.clearBubbles}
            />
          )}

          {/* Microphone error message */}
          {micError && (
            <div className="px-4 py-2 mx-4 mb-2 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-xs">
              {micError}
            </div>
          )}

          {/* GPU error message */}
          {gpuError && (
            <div className="px-4 py-2 mx-4 mb-2 rounded-lg bg-warning/10 border border-warning/20 text-warning text-xs">
              <strong>GPU 资源警告：</strong> {gpuError}
              <p className="mt-1">请在设置中选择更小的模型，或确保 GPU 驱动已更新。</p>
            </div>
          )}

        </div>
      </div>

      {/* Volume visualizer */}
      <div className="px-4">
        <VolumeVisualizer
          volume={volumeLevel}
          isActive={sessionState !== 'idle'}
        />
      </div>

      {/* Control bar */}
      <VoiceControlBar
        sessionState={sessionState}
        isTtsMuted={isTtsMuted}
        isInputDisabled={isInputDisabled}
        isLoadingModels={loadingModels}
        onStart={startSession}
        onStop={stopSession}
        onToggleTtsMute={toggleTtsMute}
        onToggleInputDisabled={toggleInputDisabled}
        onInterrupt={interrupt}
      />

      {/* Settings panel (overlay) */}
      <ApiSettingsSheet
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        models={models}
        onLoadModel={handleLoadModel}
        onUnloadModel={handleUnloadModel}
      />
    </div>
  );
}
