import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { ModelSelector } from '@/components/shared/model-selector';
import { AudioUploadZone } from '@/components/speech/audio-upload-zone';
import { AudioRecorderPanel } from '@/components/speech/audio-recorder-panel';
import { TranscriptionResult } from '@/components/speech/transcription-result';
import { TranscriptionHistoryItem } from '@/components/speech/transcription-history-item';
import { AudioPlayer } from '@/components/shared/audio-player';
import { LoadingSpinner } from '@/components/shared/loading-spinner';
import { EmptyState } from '@/components/shared/empty-state';
import { ConfirmDialog } from '@/components/shared/confirm-dialog';
import { useModels } from '@/hooks/use-models';
import { useTranscriptions, useDeleteTranscription } from '@/hooks/use-transcriptions';
import { transcribeAudio } from '@/api/audio';
import { createTranscription } from '@/api/transcriptions';
import { Play, FileAudio, Mic, Upload } from 'lucide-react';

export function SpeechToTextPage() {
  const { data: models } = useModels();
  const { data: transcriptions, isLoading: histLoading } = useTranscriptions();
  const deleteMutation = useDeleteTranscription();

  const [selectedModel, setSelectedModel] = useState('Qwen3-ASR-0.6B-gguf');
  const [selectedLanguage, setSelectedLanguage] = useState('auto');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
  const [inputMode, setInputMode] = useState<'upload' | 'record'>('upload');
  const [resultText, setResultText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const handleTranscribe = useCallback(async () => {
    const fileToTranscribe = recordingBlob
      ? new File([recordingBlob], 'recording.webm', { type: 'audio/webm' })
      : selectedFile;

    if (!fileToTranscribe) return;
    setIsProcessing(true);
    setError(null);
    try {
      const result = await transcribeAudio({
        file: fileToTranscribe,
        model: selectedModel,
        language: selectedLanguage === 'auto' ? undefined : selectedLanguage,
        response_format: 'verbose_json',
        timestamp_granularities: ['word'],
      });
      setResultText(result.text);
      // Save to history
      const formData = new FormData();
      formData.append('file', fileToTranscribe);
      formData.append('text', result.text);
      await createTranscription(formData).catch(() => { /* 历史记录保存失败不影响主功能 */ });
    } catch (err) {
      setError(err instanceof Error ? err.message : '转写失败');
    } finally {
      setIsProcessing(false);
    }
  }, [selectedFile, recordingBlob, selectedModel, selectedLanguage]);

  const handleCopyResult = () => {
    navigator.clipboard.writeText(resultText);
  };

  const handleDownloadResult = () => {
    const blob = new Blob([resultText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transcription.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePlay = (id: string) => {
    setPlayingId(id);
  };

  return (
    <div className="max-w-7xl mx-auto p-4 lg:p-6">
      <h1 className="text-xl font-bold mb-4 tracking-tight">语音转文字</h1>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left: Audio input + result */}
        <div className="flex-1 space-y-4">
          {/* Model & language */}
          <div className="flex flex-wrap gap-3">
            <div className="w-56">
              <ModelSelector
                models={models ?? []}
                category="asr"
                value={selectedModel}
                onChange={setSelectedModel}
              />
            </div>
            <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">自动检测</SelectItem>
                <SelectItem value="en">英语</SelectItem>
                <SelectItem value="zh">中文</SelectItem>
                <SelectItem value="ja">日语</SelectItem>
                <SelectItem value="ko">韩语</SelectItem>
                <SelectItem value="fr">法语</SelectItem>
                <SelectItem value="de">德语</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Input method tabs */}
          <Tabs value={inputMode} onValueChange={(v) => {
            const mode = v as 'upload' | 'record';
            setInputMode(mode);
            if (mode === 'upload') {
              setRecordingBlob(null);
            } else {
              setSelectedFile(null);
            }
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
              <AudioRecorderPanel onRecordingChange={(blob, _url) => {
                setRecordingBlob(blob);
              }} />
            </TabsContent>
          </Tabs>

          {/* Transcribe button */}
          <Button
            onClick={handleTranscribe}
            disabled={(!selectedFile && !recordingBlob) || isProcessing}
            className="w-full"
          >
            {isProcessing ? <LoadingSpinner className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isProcessing ? '转写中...' : '开始转写'}
          </Button>

          {/* Error */}
          {error && (
            <p className="text-sm text-destructive p-3 rounded-md bg-destructive/10">{error}</p>
          )}

          {/* Result */}
          <TranscriptionResult
            text={resultText}
            onCopy={handleCopyResult}
            onDownload={handleDownloadResult}
          />
        </div>

        {/* Right: History */}
        <div className="w-full lg:w-80 shrink-0">
          <h3 className="text-sm font-medium mb-3">历史记录</h3>
          <ScrollArea className="h-[calc(100vh-14rem)]">
            {histLoading ? (
              <div className="flex justify-center py-8"><LoadingSpinner /></div>
            ) : !transcriptions?.length ? (
              <EmptyState icon={FileAudio} title="暂无转写记录" />
            ) : (
              <div className="space-y-2 pr-2">
                {transcriptions.map((r) => (
                  <TranscriptionHistoryItem
                    key={r.id}
                    record={r}
                    onPlay={handlePlay}
                    onDelete={(id) => setDeleteTarget(id)}
                    onCopy={(text) => navigator.clipboard.writeText(text)}
                  />
                ))}
              </div>
            )}
          </ScrollArea>
        </div>
      </div>

      {/* Player for playing transcription audio */}
      {playingId && (
        <div className="mt-4">
          <AudioPlayer audioUrl={`/v1/transcriptions/${playingId}/audio`} />
        </div>
      )}

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}
        title="删除转写"
        description="此操作将永久删除此转写记录及其音频。"
        confirmLabel="删除"
        variant="danger"
        onConfirm={() => {
          if (deleteTarget) {
            deleteMutation.mutate(deleteTarget);
            setDeleteTarget(null);
          }
        }}
      />
    </div>
  );
}
