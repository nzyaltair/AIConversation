import { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { ModelSelector } from '@/components/shared/model-selector';
import { VoiceSelector } from '@/components/tts/voice-selector';
import { SpeechHistoryItem } from '@/components/tts/speech-history-item';
import { AudioPlayer } from '@/components/shared/audio-player';
import { GenerationStats } from '@/components/shared/generation-stats';
import { LoadingSpinner } from '@/components/shared/loading-spinner';
import { EmptyState } from '@/components/shared/empty-state';
import { ConfirmDialog } from '@/components/shared/confirm-dialog';
import { useModels } from '@/hooks/use-models';
import { useTTSGenerations, useDeleteTTSGeneration } from '@/hooks/use-speech-history';
import { generateSpeech } from '@/api/audio';
import { createTTSGeneration } from '@/api/speech-history';
import { getModelDetail } from '@/lib/model-metadata';
import { Volume2, Play } from 'lucide-react';

export function TextToSpeechPage() {
  const { data: models } = useModels();
  const { data: history, isLoading: histLoading } = useTTSGenerations();
  const deleteMutation = useDeleteTTSGeneration();

  const [selectedModel, setSelectedModel] = useState('Qwen3-TTS-1.7B-VoiceDesign-gguf');
  const [selectedVoice, setSelectedVoice] = useState('af_bella');
  const [instruct, setInstruct] = useState('');
  const [speed, setSpeed] = useState(1.0);
  const [inputText, setInputText] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{ duration?: number; genTimeMs?: number }>({});
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  // When model changes, reset voice to a sensible default
  useEffect(() => {
    const detail = getModelDetail(selectedModel);
    const caps = detail?.capabilities ?? [];

    if (caps.includes('voice-design')) {
      // VoiceDesign: no built-in voices, instruct is required
      setSelectedVoice('');
      // Keep existing instruct if user already typed something
    } else if (selectedModel.includes('Qwen') && caps.includes('voice-clone')) {
      // Qwen CustomVoice: default to Vivian
      setSelectedVoice('Vivian');
      setInstruct('');
    } else {
      // Kokoro or fallback
      setSelectedVoice('af_bella');
      setInstruct('');
    }
  }, [selectedModel]);

  const handleGenerate = useCallback(async () => {
    if (!inputText.trim()) return;
    setIsGenerating(true);
    setError(null);
    const startTime = Date.now();

    try {
      const blob = await generateSpeech({
        model: selectedModel,
        input: inputText.trim(),
        voice: selectedVoice || undefined,
        instruct: instruct.trim() || undefined,
        speed,
        response_format: 'wav',
        stream: false,
      });

      const url = URL.createObjectURL(blob);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(url);
      setAudioBlob(blob);
      setStats({ duration: undefined, genTimeMs: Date.now() - startTime });

      await createTTSGeneration({
        model_id: selectedModel,
        speaker: selectedVoice || instruct || undefined,
        input_text: inputText.trim(),
        generation_time_ms: Date.now() - startTime,
      }).catch(() => { /* 历史记录保存失败不影响主功能 */ });
    } catch (err) {
      setError(err instanceof Error ? err.message : '语音生成失败');
    } finally {
      setIsGenerating(false);
    }
  }, [inputText, selectedModel, selectedVoice, instruct, speed, audioUrl]);

  const handlePlay = useCallback(() => {
    // Playback handled by AudioPlayer
  }, []);

  return (
    <div className="max-w-7xl mx-auto p-4 lg:p-6">
      <h1 className="text-xl font-bold mb-4 tracking-tight">文字转语音</h1>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left: Voice panel */}
        <div className="w-full lg:w-56 shrink-0 space-y-4">
          <ModelSelector
            models={models ?? []}
            category="tts"
            value={selectedModel}
            onChange={setSelectedModel}
          />
          <VoiceSelector
            selectedVoice={selectedVoice}
            onSelectVoice={setSelectedVoice}
            model={selectedModel}
            instruct={instruct}
            onInstructChange={setInstruct}
          />
          <div>
            <div className="flex justify-between mb-1">
              <Label className="text-xs">语速：{speed.toFixed(1)}x</Label>
            </div>
            <Slider value={[speed]} min={0.5} max={2.0} step={0.1} onValueChange={([v]) => setSpeed(v)} />
          </div>
        </div>

        {/* Center: Text input + player */}
        <div className="flex-1 space-y-4">
          <Textarea
            placeholder="输入要转换为语音的文字..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={6}
            className="min-h-[150px] resize-y"
          />

          <Button
            onClick={handleGenerate}
            disabled={!inputText.trim() || isGenerating}
            className="w-full"
          >
            {isGenerating ? <LoadingSpinner className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isGenerating ? '生成中...' : '生成语音'}
          </Button>

          {error && (
            <div className="text-sm text-destructive p-3 rounded-md bg-destructive/10 space-y-2">
              <p>{error}</p>
              {(error.includes('not loaded') || error.includes('not found') || error.includes('未加载') || error.includes('下载')) && (
                <p className="text-xs text-muted-foreground">
                  <a href="/models" className="text-primary underline font-medium">前往模型管理页面</a> 下载 TTS 模型。
                </p>
              )}
            </div>
          )}

          {stats.genTimeMs != null && (
            <GenerationStats
              duration={stats.duration}
              generationTimeMs={stats.genTimeMs}
              label="上次生成"
            />
          )}

          <AudioPlayer audioUrl={audioUrl} audioBlob={audioBlob} />
        </div>

        {/* Right: History */}
        <div className="w-full lg:w-72 shrink-0">
          <h3 className="text-sm font-medium mb-3">历史记录</h3>
          <ScrollArea className="h-[calc(100vh-14rem)]">
            {histLoading ? (
              <div className="flex justify-center py-8"><LoadingSpinner /></div>
            ) : !history?.length ? (
              <EmptyState icon={Volume2} title="暂无生成记录" />
            ) : (
              <div className="space-y-2 pr-2">
                {history.map((r) => (
                  <SpeechHistoryItem
                    key={r.id}
                    record={r}
                    onPlay={handlePlay}
                    onDelete={(id) => setDeleteTarget(id)}
                  />
                ))}
              </div>
            )}
          </ScrollArea>
        </div>
      </div>

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}
        title="删除生成记录"
        description="此操作将永久删除此语音生成记录。"
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
