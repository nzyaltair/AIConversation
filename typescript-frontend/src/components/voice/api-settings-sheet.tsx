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
import { Input } from '@/components/ui/input';
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
import { useConversationApiSettings } from '@/stores/conversation-api-settings-store';
import { getModelDetail } from '@/lib/model-metadata';
import { SYSTEM_PROMPT_PRESETS, VOICE_DESIGN_PRESETS } from '@/lib/presets';
import { RotateCcw, Loader2, Eye, EyeOff } from 'lucide-react';
import type { ModelInfo } from '@/types';

export interface ApiSettingsSheetProps {
  open: boolean;
  onClose: () => void;
  models?: ModelInfo[];
  onLoadModel?: (variant: string) => Promise<void>;
  onUnloadModel?: (variant: string) => Promise<void>;
}

const TTS_SPEAKERS = [
  'Vivian', 'Serena', 'Uncle_Fu', 'Ryan', 'Aiden',
  'Ono_Anna', 'Sohee', 'Eric', 'Dylan',
];

const REASONING_EFFORTS = [
  { value: 'none', label: '无' },
  { value: 'low', label: '低' },
  { value: 'medium', label: '中' },
  { value: 'high', label: '高' },
];

export function ApiSettingsSheet({
  open,
  onClose,
  models,
  onLoadModel,
  onUnloadModel,
}: ApiSettingsSheetProps) {
  const settings = useConversationApiSettings();
  const [isDesktop, setIsDesktop] = useState(true);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 768px)');
    setIsDesktop(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  const handleModelChange = useCallback(
    async (category: 'tts' | 'asr' | 'vad', newVariant: string) => {
      const currentVariant =
        category === 'tts'
          ? settings.ttsModelVariant
          : category === 'asr'
            ? settings.asrModelVariant
            : settings.vadModelVariant;

      if (currentVariant === newVariant) return;

      // Unload old model
      if (onUnloadModel) {
        try {
          await onUnloadModel(currentVariant);
        } catch {
          // ignore
        }
      }

      // Update variant in settings
      if (category === 'tts') settings.updateSettings({ ttsModelVariant: newVariant });
      else if (category === 'asr') settings.updateSettings({ asrModelVariant: newVariant });
      else settings.updateSettings({ vadModelVariant: newVariant });

      // Load new model
      if (onLoadModel) {
        setLoadingModel(newVariant);
        try {
          await onLoadModel(newVariant);
        } finally {
          setLoadingModel(null);
        }
      }
    },
    [settings, onLoadModel, onUnloadModel],
  );

  const modelSelect = (
    category: 'tts' | 'asr' | 'vad',
    currentVariant: string,
  ) => {
    const variantModels = (models ?? []).filter((m) => m.category === category);
    return (
      <Select
        value={currentVariant}
        onValueChange={(v) => handleModelChange(category, v)}
      >
        <SelectTrigger className="h-9 text-xs rounded-lg">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {variantModels.map((m) => (
            <SelectItem key={m.variant} value={m.variant} className="text-xs">
              {getModelDetail(m.variant)?.displayName ?? m.variant}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  };

  const side = isDesktop ? 'right' : 'bottom';

  return (
    <Sheet open={open} onOpenChange={onClose}>
      <SheetContent side={side} className="w-full max-w-md sm:max-w-md overflow-y-auto">
        <SheetHeader className="mb-4">
          <SheetTitle className="text-lg">API 对话设置</SheetTitle>
        </SheetHeader>

        <Accordion type="multiple" defaultValue={['api', 'inference']} className="space-y-1">
          {/* ── API Configuration ── */}
          <AccordionItem value="api" className="border-none">
            <AccordionTrigger className="py-2 text-sm font-medium">
              API 配置
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pb-3">
              {/* Base URL */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">基础地址</Label>
                <Input
                  className="h-9 text-xs rounded-lg"
                  placeholder="https://api.deepseek.com"
                  value={settings.baseUrl}
                  onChange={(e) =>
                    settings.updateSettings({ baseUrl: e.target.value })
                  }
                />
              </div>

              {/* Model ID */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">模型 ID</Label>
                <Input
                  className="h-9 text-xs rounded-lg"
                  placeholder="deepseek-v4-pro"
                  value={settings.modelId}
                  onChange={(e) =>
                    settings.updateSettings({ modelId: e.target.value })
                  }
                />
              </div>

              {/* API Key */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">API 密钥</Label>
                <div className="relative">
                  <Input
                    className="h-9 text-xs rounded-lg pr-9"
                    type={showApiKey ? 'text' : 'password'}
                    placeholder="sk-..."
                    value={settings.apiKey}
                    onChange={(e) =>
                      settings.updateSettings({ apiKey: e.target.value })
                    }
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute right-0 top-0 h-9 w-9 rounded-lg"
                    onClick={() => setShowApiKey((v) => !v)}
                  >
                    {showApiKey ? (
                      <EyeOff className="h-3.5 w-3.5" />
                    ) : (
                      <Eye className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* ── Inference Settings ── */}
          <AccordionItem value="inference" className="border-none">
            <AccordionTrigger className="py-2 text-sm font-medium">
              推理设置
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pb-3">
              {/* Temperature */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">温度</Label>
                  <span className="text-xs font-mono tabular-nums text-muted-foreground">
                    {settings.temperature.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[settings.temperature]}
                  onValueChange={([v]) => settings.updateSettings({ temperature: v })}
                  min={0}
                  max={2}
                  step={0.05}
                  className="rounded-lg"
                />
              </div>

              {/* Max tokens */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">最大令牌数</Label>
                  <span className="text-xs font-mono tabular-nums text-muted-foreground">
                    {settings.maxTokens}
                  </span>
                </div>
                <Slider
                  value={[settings.maxTokens]}
                  onValueChange={([v]) => settings.updateSettings({ maxTokens: v })}
                  min={64}
                  max={8192}
                  step={64}
                  className="rounded-lg"
                />
              </div>

              {/* Thinking mode */}
              <div className="flex items-center justify-between">
                <Label className="text-xs text-muted-foreground">思考模式</Label>
                <Switch
                  checked={settings.thinkingEnabled}
                  onCheckedChange={(v) =>
                    settings.updateSettings({ thinkingEnabled: v })
                  }
                />
              </div>

              {/* Reasoning effort */}
              {settings.thinkingEnabled && (
                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">
                    推理强度
                  </Label>
                  <Select
                    value={settings.reasoningEffort}
                    onValueChange={(v) =>
                      settings.updateSettings({
                        reasoningEffort: v as 'none' | 'low' | 'medium' | 'high',
                      })
                    }
                  >
                    <SelectTrigger className="h-9 text-xs rounded-lg">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {REASONING_EFFORTS.map((r) => (
                        <SelectItem key={r.value} value={r.value} className="text-xs">
                          {r.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              {/* System prompt */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">系统提示词</Label>
                <PresetChipGroup
                  presets={SYSTEM_PROMPT_PRESETS}
                  onSelect={(p) => settings.updateSettings({ systemPrompt: p })}
                />
                <Textarea
                  className="text-xs rounded-lg min-h-[60px]"
                  placeholder="你是一个有帮助的助手。"
                  value={settings.systemPrompt}
                  onChange={(e) =>
                    settings.updateSettings({ systemPrompt: e.target.value })
                  }
                  rows={2}
                />
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* ── TTS Settings ── */}
          <AccordionItem value="tts" className="border-none">
            <AccordionTrigger className="py-2 text-sm font-medium">
              TTS 设置
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pb-3">
              {/* Speaker */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">说话人</Label>
                <Select
                  value={settings.ttsSpeaker}
                  onValueChange={(v) => settings.updateSettings({ ttsSpeaker: v })}
                >
                  <SelectTrigger className="h-9 text-xs rounded-lg">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TTS_SPEAKERS.map((s) => (
                      <SelectItem key={s} value={s} className="text-xs">
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Speed */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">语速</Label>
                  <span className="text-xs font-mono tabular-nums text-muted-foreground">
                    {settings.ttsSpeed.toFixed(1)}x
                  </span>
                </div>
                <Slider
                  value={[settings.ttsSpeed]}
                  onValueChange={([v]) => settings.updateSettings({ ttsSpeed: v })}
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  className="rounded-lg"
                />
              </div>

              {/* Voice design instruct */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  语音设计指令
                </Label>
                <PresetChipGroup
                  presets={VOICE_DESIGN_PRESETS}
                  onSelect={(p) => settings.updateSettings({ voiceDesignInstruct: p })}
                />
                <Textarea
                  className="text-xs rounded-lg min-h-[50px]"
                  placeholder="描述期望的语音风格..."
                  value={settings.voiceDesignInstruct}
                  onChange={(e) =>
                    settings.updateSettings({ voiceDesignInstruct: e.target.value })
                  }
                  rows={2}
                />
              </div>

              {/* TTS model */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  TTS 模型
                  {loadingModel === settings.ttsModelVariant && (
                    <Loader2 className="inline ml-1.5 h-3 w-3 animate-spin" />
                  )}
                </Label>
                {modelSelect('tts', settings.ttsModelVariant)}
              </div>
            </AccordionContent>
          </AccordionItem>

          {/* ── VAD Settings ── */}
          <AccordionItem value="vad" className="border-none">
            <AccordionTrigger className="py-2 text-sm font-medium">
              VAD 设置
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pb-3">
              {/* VAD model */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  VAD 模型
                  {loadingModel === settings.vadModelVariant && (
                    <Loader2 className="inline ml-1.5 h-3 w-3 animate-spin" />
                  )}
                </Label>
                {modelSelect('vad', settings.vadModelVariant)}
              </div>

              <VadConfigPanel
                config={{
                  threshold: settings.vadThreshold,
                  min_speech_ms: settings.vadMinSpeechMs,
                  silence_duration_ms: settings.vadSilenceDurationMs,
                  max_utterance_ms: settings.vadMaxUtteranceMs,
                  sample_rate: 16000,
                }}
                onChange={(partial) => {
                  if (partial.threshold !== undefined)
                    settings.updateSettings({ vadThreshold: partial.threshold });
                  if (partial.min_speech_ms !== undefined)
                    settings.updateSettings({ vadMinSpeechMs: partial.min_speech_ms });
                  if (partial.silence_duration_ms !== undefined)
                    settings.updateSettings({ vadSilenceDurationMs: partial.silence_duration_ms });
                  if (partial.max_utterance_ms !== undefined)
                    settings.updateSettings({ vadMaxUtteranceMs: partial.max_utterance_ms });
                }}
              />
            </AccordionContent>
          </AccordionItem>

          {/* ── ASR Settings ── */}
          <AccordionItem value="asr" className="border-none">
            <AccordionTrigger className="py-2 text-sm font-medium">
              ASR 设置
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pb-3">
              {/* ASR model */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  ASR 模型
                  {loadingModel === settings.asrModelVariant && (
                    <Loader2 className="inline ml-1.5 h-3 w-3 animate-spin" />
                  )}
                </Label>
                {modelSelect('asr', settings.asrModelVariant)}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Reset */}
        <div className="mt-4">
          <Button
            variant="outline"
            size="sm"
            className="w-full gap-2 text-xs rounded-lg"
            onClick={() => settings.resetSettings()}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            重置为默认
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}
