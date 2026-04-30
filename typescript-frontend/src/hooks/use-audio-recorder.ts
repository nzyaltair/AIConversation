import { useState, useRef, useCallback } from 'react';

interface UseAudioRecorderReturn {
  isRecording: boolean;
  audioBlob: Blob | null;
  audioUrl: string | null;
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  clear: () => void;
}

export function useAudioRecorder(): UseAudioRecorderReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      chunks.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: { ideal: 16000 },
          channelCount: { ideal: 1 },
          echoCancellation: false,
          noiseSuppression: false,
        },
      });
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      });
      mediaRecorder.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioUrl(url);
        setIsRecording(false);
      };

      recorder.onerror = () => {
        setError('录音失败');
        setIsRecording(false);
      };

      recorder.start();
      setIsRecording(true);
    } catch (err) {
      const msg = err instanceof DOMException && err.name === 'NotAllowedError'
        ? '麦克风访问被拒绝'
        : err instanceof Error
          ? err.message
          : '启动录音失败';
      setError(msg);
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorder.current?.state === 'recording') {
      mediaRecorder.current.stop();
    }
  }, []);

  const clear = useCallback(() => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioBlob(null);
    setAudioUrl(null);
    setError(null);
  }, [audioUrl]);

  return { isRecording, audioBlob, audioUrl, error, startRecording, stopRecording, clear };
}
