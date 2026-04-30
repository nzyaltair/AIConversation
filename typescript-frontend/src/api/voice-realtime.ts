/**
 * WebSocket client for real-time voice conversation with IVWS protocol.
 *
 * Handles bidirectional audio streaming: captures microphone PCM16 audio,
 * sends it via IVWS binary frames, and plays back assistant audio responses.
 * Includes heartbeat, exponential-backoff reconnect, and audio playback queue.
 */

import type { ApiConfig } from '@/types';

// ── IVWS Protocol Constants ──

const IVWS_HEADER_SIZE = 24;
const IVWS_MAGIC = new TextEncoder().encode('IVWS');
const IVWS_VERSION = 1;

const KIND_USER_AUDIO = 1;
const KIND_ASSISTANT_AUDIO = 2;

// ── Utility: float32 ↔ PCM16 conversion ──

function float32ToPcm16(float32: Float32Array): ArrayBuffer {
  const buf = new ArrayBuffer(float32.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function pcm16ToFloat32(pcm16: ArrayBuffer): Float32Array {
  const int16 = new Int16Array(pcm16);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768;
  }
  return float32;
}

// ── Callbacks Interface ──

export interface VoiceRealtimeCallbacks {
  onStateChange?: (state: string, previous?: string) => void;
  onUserTranscript?: (text: string) => void;
  onAssistantTextDelta?: (delta: string) => void;
  onAssistantTextFinal?: (text: string, thinking?: string) => void;
  onThinking?: (text: string) => void;
  onAssistantAudio?: (pcm16: ArrayBuffer) => void;
  onTurnDone?: () => void;
  onError?: (code: string, message: string) => void;
  onConnectionChange?: (connected: boolean) => void;
  onVolumeLevel?: (level: number) => void;
}

// ── WebSocket ReadyState enum (for readability) ──

const enum WsState {
  CONNECTING = 0,
  OPEN = 1,
  CLOSING = 2,
  CLOSED = 3,
}

// ── Client Class ──

export class VoiceRealtimeClient {
  // WebSocket
  private ws: WebSocket | null = null;
  private wsUrl: string = '';

  // Audio capture
  private audioContext: AudioContext | null = null;
  private scriptNode: ScriptProcessorNode | null = null;
  private mediaStream: MediaStream | null = null;
  private audioInitialized = false;

  // Mute / input-disable flags (reactive, toggled at runtime)
  private isTtsMuted = false;
  private isInputDisabled = false;

  // Audio playback
  private playbackQueue: ArrayBuffer[] = [];
  private scheduledSources = new Set<AudioBufferSourceNode>();
  private playbackContext: AudioContext | null = null;
  private playbackGeneration = 0;
  private rejectPlayback = false;
  private nextStartTime = 0;

  // Connection lifecycle
  private intentionalDisconnect = false;
  private isConnecting = false;
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts = 3;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // Heartbeat
  private pingIntervalId: ReturnType<typeof setInterval> | null = null;
  private lastPongTime: number = 0;

  // Callbacks
  private callbacks: VoiceRealtimeCallbacks;

  constructor(callbacks?: VoiceRealtimeCallbacks) {
    this.callbacks = callbacks ?? {};
  }

  // ── Public accessors ──

  get connected(): boolean {
    return this.ws !== null && this.ws.readyState === WsState.OPEN;
  }

  setTtsMuted(muted: boolean): void {
    this.isTtsMuted = muted;
    if (muted) this.stopPlayback();
  }

  setInputDisabled(disabled: boolean): void {
    this.isInputDisabled = disabled;
  }

  // ── Connection ──

  connect(url: string): Promise<void> {
    if (this.isConnecting) {
      return Promise.reject(new Error('Already attempting to connect'));
    }
    if (this.ws && this.ws.readyState === WsState.OPEN) {
      return Promise.resolve();
    }

    console.log('[WS] connect() called, url:', url);
    this.isConnecting = true;
    this.intentionalDisconnect = false;
    this.wsUrl = url;

    return new Promise<void>((resolve, reject) => {
      try {
        console.log('[WS] Creating WebSocket...');
        const ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
          console.log('[WS] WebSocket connected:', url);
          this.ws = ws;
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.setupMicrophone().catch((err) => {
            console.warn('[WS] Microphone setup failed in onopen:', err);
            const msg = err instanceof Error ? err.message : String(err);
            this.callbacks.onError?.('mic_error', msg || '麦克风访问失败');
          });
          this.callbacks.onConnectionChange?.(true);
          resolve();
        };

        ws.onmessage = (event: MessageEvent) => {
          if (typeof event.data === 'string') {
            console.log('[WS] ← text:', event.data.slice(0, 200));
          } else if (event.data instanceof ArrayBuffer) {
            console.log('[WS] ← binary:', event.data.byteLength, 'bytes');
          }
          this.handleMessage(event);
        };

        ws.onclose = (event) => {
          console.log('[WS] closed:', event.code, event.reason, 'intentional:', this.intentionalDisconnect);
          this.ws = null;
          this.isConnecting = false;
          this.stopHeartbeat();
          this.cleanupAudio();
          this.callbacks.onConnectionChange?.(false);
          if (!this.intentionalDisconnect) {
            this.attemptReconnect();
          }
        };

        ws.onerror = () => {
          console.error('[WS] error event');
          // onclose will fire after onerror, so reconnect logic lives there.
          // Reject only if we are still waiting for the initial connection.
          if (this.isConnecting) {
            this.isConnecting = false;
            reject(new Error(`WebSocket connection failed: ${url}`));
          }
        };
      } catch (err) {
        this.isConnecting = false;
        reject(err);
      }
    });
  }

  disconnect(): void {
    console.log('[WS] disconnect() called');
    this.intentionalDisconnect = true;
    this.stopHeartbeat();
    this.stopPlayback();
    this.cleanupAudio();

    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.onclose = null; // prevent reconnect
      this.ws.onerror = null;
      this.ws.onmessage = null;
      if (this.ws.readyState === WsState.OPEN || this.ws.readyState === WsState.CONNECTING) {
        this.ws.close();
      }
      this.ws = null;
    }

    this.reconnectAttempts = 0;
    this.isConnecting = false;
    this.callbacks.onConnectionChange?.(false);
  }

  // ── Control messages (JSON) ──

  sessionStart(): void {
    console.log('[WS] → session_start');
    this.sendJson({ type: 'session_start' });
  }

  inputStreamStart(options?: {
    model_variants?: {
      vad?: string;
      asr?: string;
      llm?: string;
      tts?: string;
    };
    llm_settings?: {
      temperature?: number;
      max_tokens?: number;
      thinking_enabled?: boolean;
      system_prompt?: string;
    };
    tts_settings?: {
      speaker?: string;
      speed?: number;
      voice_design_instruct?: string;
    };
    vad_config?: {
      threshold?: number;
      min_speech_ms?: number;
      silence_duration_ms?: number;
      max_utterance_ms?: number;
    };
    api_config?: ApiConfig;
  }): void {
    this.sendJson({
      type: 'input_stream_start',
      ...(options?.model_variants ? { model_variants: options.model_variants } : {}),
      ...(options?.llm_settings ? { llm_settings: options.llm_settings } : {}),
      ...(options?.tts_settings ? { tts_settings: options.tts_settings } : {}),
      ...(options?.vad_config ? { vad_config: options.vad_config } : {}),
      ...(options?.api_config ? { api_config: options.api_config } : {}),
    });
  }

  updateSettings(options?: {
    model_variants?: {
      vad?: string;
      asr?: string;
      llm?: string;
      tts?: string;
    };
    llm_settings?: {
      temperature?: number;
      max_tokens?: number;
      thinking_enabled?: boolean;
      system_prompt?: string;
    };
    tts_settings?: {
      speaker?: string;
      speed?: number;
      voice_design_instruct?: string;
    };
    vad_config?: {
      threshold?: number;
      min_speech_ms?: number;
      silence_duration_ms?: number;
      max_utterance_ms?: number;
    };
    api_config?: ApiConfig;
  }): void {
    this.sendJson({
      type: 'update_settings',
      ...(options?.model_variants ? { model_variants: options.model_variants } : {}),
      ...(options?.llm_settings ? { llm_settings: options.llm_settings } : {}),
      ...(options?.tts_settings ? { tts_settings: options.tts_settings } : {}),
      ...(options?.vad_config ? { vad_config: options.vad_config } : {}),
      ...(options?.api_config ? { api_config: options.api_config } : {}),
    });
  }

  inputStreamStop(): void {
    console.log('[WS] → input_stream_stop');
    this.sendJson({ type: 'input_stream_stop' });
  }

  interrupt(): void {
    console.log('[WS] → interrupt');
    this.stopPlayback();
    this.sendJson({ type: 'interrupt' });
  }

  stopPlayback(): void {
    this.playbackGeneration++;
    this.rejectPlayback = true;
    for (const source of this.scheduledSources) {
      try { source.stop(); } catch { /* already stopped */ }
    }
    this.scheduledSources.clear();
    this.playbackQueue = [];
    this.nextStartTime = 0;
  }

  // ── Audio capture setup ──

  private async setupMicrophone(): Promise<void> {
    if (this.audioInitialized) {
      console.log('[WS] setupMicrophone: already initialized');
      return;
    }

    console.log('[WS] setupMicrophone: requesting mic access...');
    try {
      // Create AudioContext — 16000 Hz is a hint; browser may use the default rate.
      const audioContext = new AudioContext({ sampleRate: 16000 });
      this.audioContext = audioContext;

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: { ideal: 16000 },
          channelCount: { ideal: 1 },
          echoCancellation: { ideal: true },
          noiseSuppression: { ideal: true },
        },
      });
      this.mediaStream = stream;

      const source = audioContext.createMediaStreamSource(stream);

      // ScriptProcessorNode for audio processing (deprecated but universally supported).
      // TODO: Migrate to AudioWorklet when browser support is ubiquitous.
      const scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
      this.scriptNode = scriptNode;

      scriptNode.onaudioprocess = (event: AudioProcessingEvent) => {
        const inputBuffer = event.inputBuffer;
        const channelData = inputBuffer.getChannelData(0);
        if (channelData.length === 0) return;

        // Convert float32 to PCM16 and send
        const pcm16 = float32ToPcm16(channelData);
        this.sendBinaryAudio(pcm16);

        // Calculate volume level for visualization
        const level = this.calculateVolumeLevel(channelData);
        this.callbacks.onVolumeLevel?.(level);
      };

      source.connect(scriptNode);
      scriptNode.connect(audioContext.destination);

      this.audioInitialized = true;
      console.log('[WS] setupMicrophone: success, audio capture active');
    } catch (err) {
      console.warn('[WS] VoiceRealtimeClient: microphone setup failed', err);
      const msg = err instanceof Error ? err.message : String(err);
      if ((err as DOMException)?.name === 'NotAllowedError') {
        this.callbacks.onError?.('microphone_permission_denied', '麦克风访问被拒绝，请在浏览器设置中允许麦克风访问。');
      } else {
        this.callbacks.onError?.('mic_error', msg || '麦克风设置失败');
      }
      this.audioInitialized = false;
    }
  }

  // ── IVWS frame building and parsing ──

  private buildIvwsFrame(kind: number, pcm16: ArrayBuffer): ArrayBuffer {
    const frame = new ArrayBuffer(IVWS_HEADER_SIZE + pcm16.byteLength);
    const header = new Uint8Array(frame);

    // Magic: "IVWS"
    header[0] = IVWS_MAGIC[0];
    header[1] = IVWS_MAGIC[1];
    header[2] = IVWS_MAGIC[2];
    header[3] = IVWS_MAGIC[3];

    // Version
    header[4] = IVWS_VERSION;

    // Kind
    header[5] = kind;

    // Reserved bytes 6-23 remain zero (default Uint8Array value)

    // Payload at offset 24
    const payload = new Uint8Array(pcm16);
    header.set(payload, IVWS_HEADER_SIZE);

    return frame;
  }

  private parseBinary(data: ArrayBuffer): { kind: number; payload: ArrayBuffer } {
    if (data.byteLength < IVWS_HEADER_SIZE) {
      throw new Error(`IVWS frame too small: ${data.byteLength} < ${IVWS_HEADER_SIZE}`);
    }

    const header = new Uint8Array(data, 0, 4);
    if (
      header[0] !== IVWS_MAGIC[0] ||
      header[1] !== IVWS_MAGIC[1] ||
      header[2] !== IVWS_MAGIC[2] ||
      header[3] !== IVWS_MAGIC[3]
    ) {
      throw new Error('Invalid IVWS magic bytes');
    }

    const view = new DataView(data);
    const kind = view.getUint8(5);

    // Return the IVWS header and payload as-is for callers that need the raw header
    const payload = data.slice(IVWS_HEADER_SIZE);
    return { kind, payload };
  }

  // ── Audio playback ──

  private playAudioBuffer(pcm16: ArrayBuffer): void {
    if (this.rejectPlayback) {
      return;
    }
    this.playbackQueue.push(pcm16);
    this.processPlaybackQueue();
  }

  private processPlaybackQueue(): void {
    if (this.playbackQueue.length === 0) {
      return;
    }

    const generation = this.playbackGeneration;

    try {
      if (!this.playbackContext) {
        this.playbackContext = new AudioContext();
      }
      if (this.playbackContext.state === 'suspended') {
        this.playbackContext.resume().catch((err) => {
          console.warn('VoiceRealtimeClient: resume() failed', err);
        });
      }

      // Schedule all queued chunks with contiguous start times (gapless)
      while (this.playbackQueue.length > 0) {
        const pcm16 = this.playbackQueue.shift()!;
        const float32 = pcm16ToFloat32(pcm16);
        const audioBuffer = this.playbackContext.createBuffer(1, float32.length, 24000);
        audioBuffer.getChannelData(0).set(float32);

        const source = this.playbackContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.playbackContext.destination);

        // Compute contiguous start time; reset if we fell behind
        const now = this.playbackContext.currentTime;
        if (this.nextStartTime < now) {
          this.nextStartTime = now + 0.01;
        }
        const startTime = this.nextStartTime;
        const duration = float32.length / 24000;
        this.nextStartTime = startTime + duration;

        // Cleanup when this source finishes
        source.onended = () => {
          this.scheduledSources.delete(source);
          if (generation !== this.playbackGeneration) return;
          if (this.scheduledSources.size === 0 && this.playbackQueue.length === 0) {
            this.nextStartTime = 0;
          }
        };

        this.scheduledSources.add(source);
        source.start(startTime);
      }
    } catch (err) {
      console.warn('VoiceRealtimeClient: playback error', err);
      if (generation === this.playbackGeneration) {
        this.nextStartTime = 0;
        setTimeout(() => this.processPlaybackQueue(), 50);
      }
    }
  }

  // ── Send helpers ──

  private sendBinaryAudio(pcm16: ArrayBuffer): void {
    if (this.isInputDisabled) return;
    if (!this.ws || this.ws.readyState !== WsState.OPEN) return;
    try {
      const frame = this.buildIvwsFrame(KIND_USER_AUDIO, pcm16);
      this.ws.send(frame);
    } catch (err) {
      console.warn('VoiceRealtimeClient: send error', err);
    }
  }

  private sendJson(data: Record<string, unknown>): void {
    if (!this.ws || this.ws.readyState !== WsState.OPEN) {
      console.warn('[WS] sendJson skipped (ws state:', this.ws?.readyState, ')');
      return;
    }
    try {
      const json = JSON.stringify(data);
      // Log safely — redact api_key from any nested api_config
      const safeForLog = JSON.stringify(data, (key, value) =>
        key === 'api_key' ? '***REDACTED***' : value,
      );
      console.log('[WS] →', safeForLog.slice(0, 300));
      this.ws.send(json);
    } catch (err) {
      console.warn('[WS] send JSON error', err);
    }
  }

  // ── Heartbeat ──

  private startHeartbeat(): void {
    this.lastPongTime = Date.now();
    this.stopHeartbeat();

    this.pingIntervalId = setInterval(() => {
      // Check if we've waited too long for a pong
      if (Date.now() - this.lastPongTime > 15000) {
        console.warn('VoiceRealtimeClient: pong timeout, reconnecting');
        this.intentionalDisconnect = false;
        this.disconnectWsOnly();
        this.attemptReconnect();
        return;
      }
      this.sendJson({ type: 'ping' });
    }, 10000);
  }

  private stopHeartbeat(): void {
    if (this.pingIntervalId !== null) {
      clearInterval(this.pingIntervalId);
      this.pingIntervalId = null;
    }
  }

  // ── Reconnect ──

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.callbacks.onError?.(
        'reconnect_failed',
        `重连失败，已尝试 ${this.maxReconnectAttempts} 次`,
      );
      return;
    }

    if (this.intentionalDisconnect) return;

    this.reconnectAttempts++;
    const delay = Math.pow(2, this.reconnectAttempts - 1) * 1000; // 1s, 2s, 4s

    this.reconnectTimer = setTimeout(() => {
      if (this.intentionalDisconnect) return;
      this.connect(this.wsUrl).catch(() => {
        // onclose handler will trigger another reconnect attempt
      });
    }, delay);
  }

  // ── Message handling ──

  private handleMessage(event: MessageEvent): void {
    if (typeof event.data === 'string') {
      this.handleTextMessage(event.data);
    } else if (event.data instanceof ArrayBuffer) {
      this.handleBinaryMessage(event.data);
    } else if (event.data instanceof Blob) {
      // If binaryType is not 'arraybuffer', convert blob to ArrayBuffer
      event.data.arrayBuffer().then(
        (buf) => this.handleBinaryMessage(buf),
        () => { /* ignore conversion errors */ },
      );
    }
  }

  private handleTextMessage(text: string): void {
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(text) as Record<string, unknown>;
    } catch {
      // Not JSON — ignore
      return;
    }

    const type = data.type as string | undefined;
    if (!type) return;

    switch (type) {
      case 'state_change': {
        const state = data.state as string;
        const previous = data.previous as string | undefined;
        this.callbacks.onStateChange?.(state, previous);
        break;
      }
      case 'user_transcript_final': {
        const text_ = data.text as string;
        if (text_ !== undefined) {
          this.callbacks.onUserTranscript?.(text_);
        }
        break;
      }
      case 'assistant_text_delta': {
        const delta = data.text as string;
        if (delta !== undefined) {
          this.callbacks.onAssistantTextDelta?.(delta);
        }
        break;
      }
      case 'assistant_text_final': {
        const finalText = data.text as string;
        const thinking = data.thinking as string | undefined;
        if (finalText !== undefined) {
          this.callbacks.onAssistantTextFinal?.(finalText, thinking);
        }
        break;
      }
      case 'thinking_delta': {
        const thinkText = data.text as string;
        if (thinkText !== undefined) {
          this.callbacks.onThinking?.(thinkText);
        }
        break;
      }
      case 'turn_done': {
        this.rejectPlayback = false;
        this.callbacks.onTurnDone?.();
        break;
      }
      case 'error': {
        const code = data.code as string;
        const message = data.message as string;
        this.callbacks.onError?.(code ?? 'unknown', message ?? 'Unknown error');
        break;
      }
      case 'pong': {
        this.lastPongTime = Date.now();
        break;
      }
      default:
        // Unknown message type — silently ignore
        break;
    }
  }

  private handleBinaryMessage(data: ArrayBuffer): void {
    try {
      const { kind, payload } = this.parseBinary(data);
      if (kind === KIND_ASSISTANT_AUDIO) {
        if (!this.isTtsMuted) {
          this.callbacks.onAssistantAudio?.(payload);
          this.playAudioBuffer(payload);
        }
      }
      // Other kinds are ignored for now (future extension)
    } catch (err) {
      console.warn('VoiceRealtimeClient: binary parse error', err);
    }
  }

  // ── Audio cleanup ──

  private cleanupAudio(): void {
    this.audioInitialized = false;

    // Disconnect script node
    if (this.scriptNode) {
      try {
        this.scriptNode.disconnect();
      } catch {
        // Already disconnected
      }
      this.scriptNode = null;
    }

    // Stop media tracks
    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop();
      }
      this.mediaStream = null;
    }

    // Close audio context
    if (this.audioContext) {
      this.audioContext.close().catch(() => { /* ignore */ });
      this.audioContext = null;
    }

    if (this.playbackContext) {
      this.playbackContext.close().catch(() => { /* ignore */ });
      this.playbackContext = null;
    }
  }

  // ── Volume calculation (RMS-based) ──

  private calculateVolumeLevel(float32: Float32Array): number {
    let sumSquares = 0;
    for (let i = 0; i < float32.length; i++) {
      sumSquares += float32[i] * float32[i];
    }
    const rms = Math.sqrt(sumSquares / float32.length);
    // Scale RMS to a human-friendly 0.0-1.0 range
    return Math.min(1, rms * 3);
  }

  // ── Internal: disconnect WebSocket only (for reconnect after heartbeat timeout) ──

  private disconnectWsOnly(): void {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      if (this.ws.readyState === WsState.OPEN || this.ws.readyState === WsState.CONNECTING) {
        this.ws.close();
      }
      this.ws = null;
    }
  }
}
