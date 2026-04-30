import { useRef, useState, useCallback } from 'react';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { useBackgroundStore } from '@/stores/background-store';
import { Check, ImagePlus, Trash2, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BackgroundSettingsProps {
  open: boolean;
  onClose: () => void;
}

function compressImage(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const maxW = 1920;
      const maxH = 1080;
      let w = img.width;
      let h = img.height;
      if (w > maxW || h > maxH) {
        const ratio = Math.min(maxW / w, maxH / h);
        w = Math.round(w * ratio);
        h = Math.round(h * ratio);
      }
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) { reject(new Error('Canvas error')); return; }
      ctx.drawImage(img, 0, 0, w, h);
      resolve(canvas.toDataURL('image/jpeg', 0.85));
    };
    img.onerror = () => reject(new Error('Image load failed'));
    img.src = URL.createObjectURL(file);
  });
}

export function BackgroundSettings({ open, onClose }: BackgroundSettingsProps) {
  const bg = useBackgroundStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [imageWarning, setImageWarning] = useState<string | null>(null);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageWarning(null);

    if (file.size > 5 * 1024 * 1024) {
      setImageWarning('图片文件较大，建议使用 5MB 以内的图片以获得最佳体验。');
    }

    try {
      const dataUrl = await compressImage(file);
      bg.setImage(dataUrl);
    } catch {
      setImageWarning('图片加载失败，请重试。');
    }
  }, [bg]);

  return (
    <Sheet open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <SheetContent side="right" className="w-[380px] sm:max-w-[380px] overflow-y-auto p-0">
        <SheetHeader className="px-5 pt-5 pb-3 shrink-0">
          <SheetTitle>背景设置</SheetTitle>
        </SheetHeader>

        <div className="px-5 pb-5 space-y-5">
          {/* Enable/Disable */}
          <div className="flex items-center justify-between py-2 border-b border-border/50">
            <Label className="text-sm">启用自定义背景</Label>
            <Switch checked={bg.enabled} onCheckedChange={bg.setEnabled} />
          </div>

          {bg.enabled && (
            <>
              {/* Opacity */}
              <div>
                <div className="flex justify-between mb-2">
                  <Label className="text-xs text-muted-foreground">背景透明度</Label>
                  <span className="text-xs text-muted-foreground tabular-nums">{Math.round(bg.opacity * 100)}%</span>
                </div>
                <Slider
                  value={[bg.opacity]}
                  min={0.05}
                  max={1}
                  step={0.05}
                  onValueChange={([v]) => bg.setOpacity(v)}
                />
              </div>

              {/* Presets */}
              <div>
                <Label className="text-xs text-muted-foreground mb-2 block">预设背景</Label>
                <div className="grid grid-cols-5 gap-2">
                  {bg.presets.map((preset) => (
                    <button
                      key={preset.id}
                      className={cn(
                        'flex flex-col items-center gap-1 p-1.5 rounded-lg border transition-all',
                        bg.type === 'preset' && bg.presetId === preset.id
                          ? 'border-primary ring-1 ring-primary/30'
                          : 'border-border hover:border-primary/30',
                      )}
                      onClick={() => bg.setPreset(preset.id)}
                    >
                      <div
                        className="h-8 w-full rounded-md border border-border/50"
                        style={{ background: preset.cssValue }}
                      />
                      {bg.type === 'preset' && bg.presetId === preset.id && (
                        <Check className="h-3 w-3 text-primary" />
                      )}
                      <span className="text-[9px] text-muted-foreground leading-tight text-center">
                        {preset.label}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Custom Color */}
              <div>
                <Label className="text-xs text-muted-foreground mb-2 block">自定义纯色</Label>
                <div className="flex items-center gap-3">
                  <input
                    type="color"
                    value={bg.customColor}
                    onChange={(e) => bg.setCustomColor(e.target.value)}
                    className="h-9 w-12 rounded-md border border-border cursor-pointer bg-transparent"
                  />
                  <span className="text-xs font-mono text-muted-foreground">{bg.customColor}</span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => bg.setCustomColor(bg.customColor)}
                    className="text-xs h-8"
                  >
                    应用纯色
                  </Button>
                </div>
              </div>

              {/* Image Upload */}
              <div>
                <Label className="text-xs text-muted-foreground mb-2 block">上传背景图片</Label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileChange}
                />

                {bg.type === 'image' && bg.imageUrl ? (
                  <div className="space-y-2">
                    <div className="relative rounded-lg overflow-hidden border border-border">
                      <img
                        src={bg.imageUrl}
                        alt="背景预览"
                        className="w-full h-24 object-cover"
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        className="absolute top-1 right-1 h-6 w-6 bg-background/60 backdrop-blur-sm rounded-full hover:bg-destructive/20"
                        onClick={bg.clearImage}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant={bg.imageFit === 'cover' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => bg.setImageFit('cover')}
                        className="text-xs h-7 flex-1"
                      >
                        填充 (Cover)
                      </Button>
                      <Button
                        variant={bg.imageFit === 'contain' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => bg.setImageFit('contain')}
                        className="text-xs h-7 flex-1"
                      >
                        适应 (Contain)
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Button
                    variant="outline"
                    className="w-full gap-2 h-16 border-dashed"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <ImagePlus className="h-5 w-5 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">点击上传图片</span>
                  </Button>
                )}

                {imageWarning && (
                  <p className="text-xs text-warning mt-2">{imageWarning}</p>
                )}
              </div>
            </>
          )}

          {/* Reset */}
          {bg.enabled && (
            <div className="pt-3 border-t border-border">
              <Button
                variant="ghost"
                size="sm"
                onClick={bg.reset}
                className="w-full text-xs text-muted-foreground gap-1.5"
              >
                <Trash2 className="h-3 w-3" />
                重置为默认
              </Button>
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
