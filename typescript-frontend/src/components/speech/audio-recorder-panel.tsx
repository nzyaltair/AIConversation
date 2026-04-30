import { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { AudioPlayer } from '@/components/shared/audio-player';
import { useAudioRecorder } from '@/hooks/use-audio-recorder';
import { Mic, Square, Trash2 } from 'lucide-react';

interface AudioRecorderPanelProps {
  onRecordingChange: (blob: Blob | null, url: string | null) => void;
}

function formatTimer(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function AudioRecorderPanel({ onRecordingChange }: AudioRecorderPanelProps) {
  const { isRecording, audioBlob, audioUrl, error, startRecording, stopRecording, clear } =
    useAudioRecorder();

  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Timer: start ticking while recording, freeze on stop
  useEffect(() => {
    if (isRecording) {
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isRecording]);

  // Notify parent when blob/url changes
  useEffect(() => {
    // Auto-clear recordings shorter than 1 second
    if (audioBlob && elapsed < 1) {
      clear();
      return;
    }
    onRecordingChange(audioBlob, audioUrl);
  }, [audioBlob, audioUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clear();
      onRecordingChange(null, null);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleClear = useCallback(() => {
    clear();
    setElapsed(0);
    onRecordingChange(null, null);
  }, [clear, onRecordingChange]);

  // ── State: Recording complete, show preview ──
  if (audioBlob && !isRecording) {
    const sizeKB = (audioBlob.size / 1024).toFixed(0);
    return (
      <div className="border rounded-xl p-5 space-y-4 bg-muted/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-green-500" />
            <span className="text-sm font-medium">录音完成</span>
          </div>
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground" onClick={handleClear}>
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>

        <AudioPlayer audioUrl={audioUrl} audioBlob={audioBlob} />

        <div className="flex gap-4 text-xs text-muted-foreground">
          <span>时长：{formatTimer(elapsed)}</span>
          <span>大小：{sizeKB} KB</span>
          <span>格式：webm</span>
        </div>
      </div>
    );
  }

  // ── Error state ──
  if (error) {
    return (
      <div className="border rounded-xl p-5 space-y-3 bg-muted/20">
        <p className="text-sm text-destructive">{error}</p>
        <Button variant="outline" size="sm" onClick={() => { clear(); onRecordingChange(null, null); }}>
          重试
        </Button>
      </div>
    );
  }

  // ── Idle or Recording state ──
  return (
    <div className="border rounded-xl p-5 space-y-4 bg-muted/20">
      {isRecording ? (
        <>
          {/* Recording in progress */}
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500" />
            </span>
            <span className="text-sm font-medium">录音中...</span>
          </div>

          <p className="text-2xl font-mono tabular-nums text-center">{formatTimer(elapsed)}</p>

          <Button
            variant="destructive"
            className="w-full"
            onClick={stopRecording}
          >
            <Square className="h-4 w-4 mr-2" /> 停止录音
          </Button>
        </>
      ) : (
        <>
          {/* Idle: ready to record */}
          <div className="flex flex-col items-center gap-3 py-4">
            <Button
              variant="destructive"
              size="lg"
              className="h-16 w-16 rounded-full"
              onClick={startRecording}
            >
              <Mic className="h-6 w-6" />
            </Button>
            <p className="text-sm font-medium">点击开始录音</p>
            <p className="text-xs text-muted-foreground">浏览器将请求麦克风权限</p>
          </div>
        </>
      )}
    </div>
  );
}
