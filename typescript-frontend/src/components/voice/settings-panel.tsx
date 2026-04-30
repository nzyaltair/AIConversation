import { useState, useEffect, useCallback } from 'react';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { VadConfigPanel } from '@/components/voice/vad-config-panel';
import { PresetChipGroup } from '@/components/voice/preset-chip-group';
import { useConversationSettings } from '@/stores/conversation-settings-store';
import { getModelDetail } from '@/lib/model-metadata';
import { SYSTEM_PROMPT_PRESETS, VOICE_DESIGN_PRESETS } from '@/lib/presets';
import { RotateCcw, Loader2 } from 'lucide-react';
import type { ModelInfo, VadConfig } from '@/types';

export interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  models?: ModelInfo[];
  onLoadModel?: (variant: string) => Promise<void>;
  onUnloadModel?: (variant: string) => Promise<void>;
}

const TTS_SPEAKERS = [
  'Vivian',
  'Serena',
  'Uncle_Fu',
  'Ryan',
  'Aiden',
  'Ono_Anna',
  'Sohee',
  'Eric',
  'Dylan',
];

export function SettingsPanel({
  open,
  onClose,
  models,
  onLoadModel,
  onUnloadModel,
}: SettingsPanelProps) {
  const settings = useConversationSettings();
  const [isDesktop, setIsDesktop] = useState(true);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 768px)');
    setIsDesktop(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  const handleModelChange = useCallback(
    async (category: 'llm' | 'tts' | 'asr' | 'vad', newVariant: string) => {
      const currentVariant =
        category === 'llm'
          ? settings.llmModelVariant
          : category === 'tts'
            ? settings.ttsModelVariant
            : category === 'asr'
              ? settings.asrModelVariant
              : settings.vadModelVariant;

      if (currentVariant === newVariant) return;

      // Update the setting immediately
      const update: Record<string, string> = {};
      const key = `${category}ModelVariant` as const;
      update[key] = newVariant;
      settings.updateSettings(update as any);

      // Handle model loading/unloading if callbacks provided
      if (onUnloadModel && currentVariant) {
        setLoadingModel(currentVariant);
        try {
          await onUnloadModel(currentVariant);
        } catch {
          // Ignore unload errors
        }
      }

      if (onLoadModel) {
        setLoadingModel(newVariant);
        try {
          await onLoadModel(newVariant);
        } catch {
          // Ignore load errors
        }
      }

      setLoadingModel(null);
    },
    [settings, onLoadModel, onUnloadModel],
  );

  // Filter models by category
  const llmModels = models?.filter((m) => {
    const detail = getModelDetail(m.variant);
    return detail?.category === 'llm';
  }) ?? [];

  const ttsModels = models?.filter((m) => {
    const detail = getModelDetail(m.variant);
    return detail?.category === 'tts';
  }) ?? [];

  const vadModels = models?.filter((m) => {
    const detail = getModelDetail(m.variant);
    return detail?.category === 'vad';
  }) ?? [];

  const asrModels = models?.filter((m) => {
    const detail = getModelDetail(m.variant);
    return detail?.category === 'asr';
  }) ?? [];

  // Map settings to VadConfig for the VadConfigPanel
  const vadConfig: VadConfig = {
    threshold: settings.vadThreshold,
    min_speech_ms: settings.vadMinSpeechMs,
    silence_duration_ms: settings.vadSilenceDurationMs,
    max_utterance_ms: settings.vadMaxUtteranceMs,
    sample_rate: 16000,
  };

  const handleVadChange = useCallback(
    (c: Partial<VadConfig>) => {
      const updates: Record<string, number> = {};
      if (c.threshold !== undefined) updates.vadThreshold = c.threshold;
      if (c.min_speech_ms !== undefined) updates.vadMinSpeechMs = c.min_speech_ms;
      if (c.silence_duration_ms !== undefined) updates.vadSilenceDurationMs = c.silence_duration_ms;
      if (c.max_utterance_ms !== undefined) updates.vadMaxUtteranceMs = c.max_utterance_ms;
      settings.updateSettings(updates as any);
    },
    [settings],
  );

  const sideContent = (
    <div className="flex-1 overflow-y-auto px-1">
      <Accordion type="multiple" defaultValue={['llm', 'tts', 'vad', 'asr']}>
        {/* ── LLM Settings ── */}
        <AccordionItem value="llm">
          <AccordionTrigger className="text-sm font-medium hover:bg-accent/30 rounded-lg px-2">LLM 设置</AccordionTrigger>
          <AccordionContent className="space-y-4">
            {/* Temperature */}
            <div>
              <div className="flex justify-between mb-1">
                <Label className="text-xs">温度</Label>
                <span className="text-xs text-muted-foreground tabular-nums">{settings.temperature.toFixed(1)}</span>
              </div>
              <Slider
                value={[settings.temperature]}
                min={0}
                max={2.0}
                step={0.1}
                onValueChange={([v]) => settings.updateSettings({ temperature: v } as any)}
              />
            </div>

            {/* Max Tokens */}
            <div>
              <div className="flex justify-between mb-1">
                <Label className="text-xs">最大令牌数</Label>
                <span className="text-xs text-muted-foreground tabular-nums">{settings.maxTokens}</span>
              </div>
              <Slider
                value={[settings.maxTokens]}
                min={64}
                max={8192}
                step={64}
                onValueChange={([v]) => settings.updateSettings({ maxTokens: v } as any)}
              />
            </div>

            {/* Thinking Mode */}
            <div className="flex items-center justify-between">
              <Label className="text-xs">思考模式</Label>
              <Switch
                checked={settings.thinkingEnabled}
                onCheckedChange={(checked) => settings.updateSettings({ thinkingEnabled: checked } as any)}
              />
            </div>

            {/* System Prompt */}
            <div>
              <Label className="text-xs mb-1 block">系统提示词</Label>
              <PresetChipGroup
                presets={SYSTEM_PROMPT_PRESETS}
                onSelect={(p) => settings.updateSettings({ systemPrompt: p } as any)}
              />
              <Textarea
                value={settings.systemPrompt}
                onChange={(e) => settings.updateSettings({ systemPrompt: e.target.value } as any)}
                placeholder="可选的 LLM 系统提示词..."
                rows={3}
                className="font-mono text-xs resize-none mt-1.5"
              />
            </div>

            {/* LLM Model Select */}
            <div>
              <Label className="text-xs mb-1 block">模型</Label>
              <Select
                value={settings.llmModelVariant}
                onValueChange={(v) => handleModelChange('llm', v)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {llmModels.length > 0 ? (
                    llmModels.map((m) => (
                      <SelectItem key={m.variant} value={m.variant}>
                        <span className="flex items-center gap-2">
                          {getModelDetail(m.variant)?.displayName ?? m.variant}
                          {loadingModel === m.variant && (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          )}
                        </span>
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value={settings.llmModelVariant} disabled>
                      {getModelDetail(settings.llmModelVariant)?.displayName ?? settings.llmModelVariant}
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
          </AccordionContent>
        </AccordionItem>

        {/* ── TTS Settings ── */}
        <AccordionItem value="tts">
          <AccordionTrigger className="text-sm font-medium hover:bg-accent/30 rounded-lg px-2">TTS 设置</AccordionTrigger>
          <AccordionContent className="space-y-4">
            {/* Speaker */}
            <div>
              <Label className="text-xs mb-1 block">说话人</Label>
              <Select
                value={settings.ttsSpeaker}
                onValueChange={(v) => settings.updateSettings({ ttsSpeaker: v } as any)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TTS_SPEAKERS.map((speaker) => (
                    <SelectItem key={speaker} value={speaker}>
                      {speaker}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Speed */}
            <div>
              <div className="flex justify-between mb-1">
                <Label className="text-xs">语速</Label>
                <span className="text-xs text-muted-foreground tabular-nums">{settings.ttsSpeed.toFixed(1)}x</span>
              </div>
              <Slider
                value={[settings.ttsSpeed]}
                min={0.5}
                max={2.0}
                step={0.1}
                onValueChange={([v]) => settings.updateSettings({ ttsSpeed: v } as any)}
              />
            </div>

            {/* Voice Design Instruct */}
            <div>
              <Label className="text-xs mb-1 block">语音设计指令</Label>
              <PresetChipGroup
                presets={VOICE_DESIGN_PRESETS}
                onSelect={(p) => settings.updateSettings({ voiceDesignInstruct: p } as any)}
              />
              <Textarea
                value={settings.voiceDesignInstruct}
                onChange={(e) => settings.updateSettings({ voiceDesignInstruct: e.target.value } as any)}
                placeholder="描述您想要创建的语音..."
                rows={2}
                className="font-mono text-xs resize-none mt-1.5"
              />
            </div>

            {/* TTS Model Select */}
            <div>
              <Label className="text-xs mb-1 block">模型</Label>
              <Select
                value={settings.ttsModelVariant}
                onValueChange={(v) => handleModelChange('tts', v)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ttsModels.length > 0 ? (
                    ttsModels.map((m) => (
                      <SelectItem key={m.variant} value={m.variant}>
                        <span className="flex items-center gap-2">
                          {getModelDetail(m.variant)?.displayName ?? m.variant}
                          {loadingModel === m.variant && (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          )}
                        </span>
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value={settings.ttsModelVariant} disabled>
                      {getModelDetail(settings.ttsModelVariant)?.displayName ?? settings.ttsModelVariant}
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
          </AccordionContent>
        </AccordionItem>

        {/* ── VAD Settings ── */}
        <AccordionItem value="vad">
          <AccordionTrigger className="text-sm font-medium hover:bg-accent/30 rounded-lg px-2">VAD 设置</AccordionTrigger>
          <AccordionContent className="space-y-4">
            {/* VAD Model Select */}
            <div>
              <Label className="text-xs mb-1 block">模型</Label>
              <Select
                value={settings.vadModelVariant}
                onValueChange={(v) => handleModelChange('vad', v)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {vadModels.length > 0 ? (
                    vadModels.map((m) => (
                      <SelectItem key={m.variant} value={m.variant}>
                        <span className="flex items-center gap-2">
                          {getModelDetail(m.variant)?.displayName ?? m.variant}
                          {loadingModel === m.variant && (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          )}
                        </span>
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value={settings.vadModelVariant} disabled>
                      {getModelDetail(settings.vadModelVariant)?.displayName ?? settings.vadModelVariant}
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            <VadConfigPanel config={vadConfig} onChange={handleVadChange} />
          </AccordionContent>
        </AccordionItem>

        {/* ── ASR Settings ── */}
        <AccordionItem value="asr">
          <AccordionTrigger className="text-sm font-medium hover:bg-accent/30 rounded-lg px-2">ASR 设置</AccordionTrigger>
          <AccordionContent className="space-y-4">
            {/* ASR Model Select */}
            <div>
              <Label className="text-xs mb-1 block">模型</Label>
              <Select
                value={settings.asrModelVariant}
                onValueChange={(v) => handleModelChange('asr', v)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {asrModels.length > 0 ? (
                    asrModels.map((m) => (
                      <SelectItem key={m.variant} value={m.variant}>
                        <span className="flex items-center gap-2">
                          {getModelDetail(m.variant)?.displayName ?? m.variant}
                          {loadingModel === m.variant && (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          )}
                        </span>
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value={settings.asrModelVariant} disabled>
                      {getModelDetail(settings.asrModelVariant)?.displayName ?? settings.asrModelVariant}
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {/* Reset button */}
      <div className="mt-4 pt-3 border-t border-border">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => settings.resetSettings()}
          className="w-full text-xs text-muted-foreground gap-1.5"
        >
          <RotateCcw className="h-3 w-3" />
          重置为默认
        </Button>
      </div>
    </div>
  );

  return (
    <Sheet open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      {isDesktop ? (
        <SheetContent side="right" className="w-[400px] sm:max-w-[400px] flex flex-col p-0">
          <SheetHeader className="px-5 pt-5 pb-3 shrink-0">
            <SheetTitle>设置</SheetTitle>
          </SheetHeader>
          <div className="px-5 pb-5 flex-1 flex flex-col overflow-hidden">
            {sideContent}
          </div>
        </SheetContent>
      ) : (
        <SheetContent side="bottom" className="flex flex-col p-0 max-h-[85vh]">
          <SheetHeader className="px-5 pt-5 pb-3 shrink-0">
            <SheetTitle>设置</SheetTitle>
          </SheetHeader>
          <div className="px-5 pb-5 flex-1 flex flex-col overflow-hidden">
            {sideContent}
          </div>
        </SheetContent>
      )}
    </Sheet>
  );
}
