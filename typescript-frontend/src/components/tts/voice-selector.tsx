import { useState, useEffect, useMemo } from 'react';
import { cn } from '@/lib/utils';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { fetchVoices } from '@/api/audio';
import { getModelDetail } from '@/lib/model-metadata';
import { Mic, Info } from 'lucide-react';

interface VoiceSelectorProps {
  selectedVoice: string;
  onSelectVoice: (voice: string) => void;
  model?: string;
  instruct: string;
  onInstructChange: (instruct: string) => void;
}

export function VoiceSelector({
  selectedVoice,
  onSelectVoice,
  model,
  instruct,
  onInstructChange,
}: VoiceSelectorProps) {
  const [voices, setVoices] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  const capabilities = useMemo(() => {
    if (!model) return [];
    return getModelDetail(model)?.capabilities ?? [];
  }, [model]);

  const isVoiceDesign = capabilities.includes('voice-design');
  const hasBuiltinVoices = !isVoiceDesign;
  const defaultTab = isVoiceDesign ? 'design' : 'builtin';

  useEffect(() => {
    setLoading(true);
    fetchVoices(model)
      .then(setVoices)
      .catch(() => setVoices([]))
      .finally(() => setLoading(false));
  }, [model]);

  useEffect(() => {
    // If switching to a VoiceDesign model and current voice is empty,
    // keep the voice selector consistent
    if (isVoiceDesign && voices.length === 0) {
      // VoiceDesign has no built-in voices
    }
  }, [isVoiceDesign, voices]);

  return (
    <div>
      <h3 className="text-sm font-medium mb-2">语音</h3>
      <Tabs defaultValue={defaultTab}>
        <TabsList className="w-full">
          <TabsTrigger value="builtin" className="flex-1 text-xs">内置</TabsTrigger>
          <TabsTrigger value="clone" className="flex-1 text-xs">克隆</TabsTrigger>
          <TabsTrigger value="design" className="flex-1 text-xs">设计</TabsTrigger>
        </TabsList>

        {/* Built-in voices */}
        <TabsContent value="builtin">
          {loading ? (
            <div className="grid grid-cols-2 gap-1 max-h-40">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-8 rounded-md bg-muted/50 animate-pulse" />
              ))}
            </div>
          ) : !hasBuiltinVoices ? (
            <div className="flex flex-col items-center gap-1.5 p-4 text-center">
              <Info className="h-4 w-4 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">
                此模型没有内置语音，请使用"设计"选项卡创建语音。
              </p>
            </div>
          ) : voices.length === 0 ? (
            <p className="text-xs text-muted-foreground p-4 text-center">
              暂无可用语音，请先加载 TTS 模型。
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-1 max-h-40 overflow-y-auto">
              {voices.map((v) => (
                <button
                  key={v}
                  className={cn(
                    'flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs text-left transition-colors',
                    selectedVoice === v
                      ? 'bg-primary/10 text-primary border border-primary/30'
                      : 'hover:bg-muted border border-transparent',
                  )}
                  onClick={() => onSelectVoice(v)}
                >
                  <Mic className="h-3 w-3 shrink-0" />
                  <span className="truncate">{v}</span>
                </button>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Clone (placeholder) */}
        <TabsContent value="clone">
          <p className="text-xs text-muted-foreground p-4 text-center">
            上传参考音频以克隆语音。
          </p>
        </TabsContent>

        {/* Design / Instruct */}
        <TabsContent value="design">
          <div className="space-y-2 p-1">
            <Textarea
              placeholder={
                isVoiceDesign
                  ? '描述您想要设计的语音...\n\n例如："一位温暖、温柔的女性声音，带着淡淡的微笑，语速舒缓。"'
                  : '可选：描述语音应如何发音...\n\n例如："用非常悲伤和含泪的声音说话。"'
              }
              value={instruct}
              onChange={(e) => onInstructChange(e.target.value)}
              rows={4}
              className="resize-y text-xs min-h-[80px]"
            />
            {isVoiceDesign && (
              <p className="text-xs text-muted-foreground">
                此模型需要语音设计指令。
              </p>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
