import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { ModelSelector } from '@/components/shared/model-selector';
import { AudioUploadZone } from '@/components/speech/audio-upload-zone';
import { AudioRecorderPanel } from '@/components/speech/audio-recorder-panel';
import { LoadingSpinner } from '@/components/shared/loading-spinner';
import { EmptyState } from '@/components/shared/empty-state';
import { VadTimeline } from '@/components/vad/vad-timeline';
import { VadStats } from '@/components/vad/vad-stats';
import { useModels } from '@/hooks/use-models';
import { detectVad, type VadResponse } from '@/api/vad';
import { Activity, Mic, Upload } from 'lucide-react';

export function VadPage() {
  const { data: models } = useModels();
  const [selectedModel, setSelectedModel] = useState('FireRedVad-onnx');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
  const [inputMode, setInputMode] = useState<'upload' | 'record'>('upload');
  const [result, setResult] = useState<VadResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDetect = useCallback(async () => {
    const fileToProcess = recordingBlob
      ? new File([recordingBlob], 'recording.webm', { type: 'audio/webm' })
      : selectedFile;

    if (!fileToProcess) return;
    setIsProcessing(true);
    setError(null);
    setResult(null);
    try {
      const res = await detectVad(fileToProcess, selectedModel);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'VAD 检测失败');
    } finally {
      setIsProcessing(false);
    }
  }, [selectedFile, recordingBlob, selectedModel]);

  return (
    <div className="max-w-7xl mx-auto p-4 lg:p-6">
      <h1 className="text-xl font-bold mb-4 tracking-tight">语音活动检测</h1>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* 左侧：输入 */}
        <div className="flex-1 space-y-4">
          <div className="w-56">
            <ModelSelector
              models={models ?? []}
              category="vad"
              value={selectedModel}
              onChange={setSelectedModel}
            />
          </div>

          <Tabs value={inputMode} onValueChange={(v) => {
            const mode = v as 'upload' | 'record';
            setInputMode(mode);
            if (mode === 'upload') setRecordingBlob(null);
            else setSelectedFile(null);
          }}>
            <TabsList className="w-full">
              <TabsTrigger value="upload" className="flex-1">
                <Upload className="h-3.5 w-3.5 mr-1.5" /> 上传
              </TabsTrigger>
              <TabsTrigger value="record" className="flex-1">
                <Mic className="h-3.5 w-3.5 mr-1.5" /> 录音
              </TabsTrigger>
            </TabsList>
            <TabsContent value="upload" className="mt-3">
              <AudioUploadZone onFile={setSelectedFile} selectedFile={selectedFile} />
            </TabsContent>
            <TabsContent value="record" className="mt-3">
              <AudioRecorderPanel onRecordingChange={(blob) => setRecordingBlob(blob)} />
            </TabsContent>
          </Tabs>

          <Button
            onClick={handleDetect}
            disabled={(!selectedFile && !recordingBlob) || isProcessing}
            className="w-full"
          >
            {isProcessing ? <LoadingSpinner className="h-4 w-4" /> : <Activity className="h-4 w-4" />}
            {isProcessing ? '检测中...' : '运行 VAD 检测'}
          </Button>

          {error && (
            <p className="text-sm text-destructive p-3 rounded-md bg-destructive/10">{error}</p>
          )}
        </div>

        {/* 右侧：结果 */}
        <div className="flex-1 space-y-4">
          {isProcessing ? (
            <div className="flex items-center justify-center py-16">
              <LoadingSpinner className="h-8 w-8" />
            </div>
          ) : result ? (
            <>
              <VadStats result={result} />
              <VadTimeline dur={result.dur} timestamps={result.timestamps} />
            </>
          ) : (
            <EmptyState
              icon={Activity}
              title="运行 VAD 检测"
              description="上传或录制音频，然后点击检测按钮分析语音片段。"
            />
          )}
        </div>
      </div>
    </div>
  );
}
