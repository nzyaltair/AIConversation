import { useBackgroundStore } from '@/stores/background-store';

export function BackgroundLayer() {
  const bg = useBackgroundStore();
  if (!bg.enabled) return null;

  const style: React.CSSProperties = {
    position: 'fixed',
    inset: 0,
    pointerEvents: 'none',
    zIndex: 0,
    opacity: bg.opacity,
    transition: 'opacity 0.5s, background 0.5s',
  };

  if (bg.type === 'preset') {
    const preset = bg.presets.find((p) => p.id === bg.presetId);
    if (preset) style.background = preset.cssValue;
  } else if (bg.type === 'color') {
    style.backgroundColor = bg.customColor;
  } else if (bg.type === 'image' && bg.imageUrl) {
    style.backgroundImage = `url(${bg.imageUrl})`;
    style.backgroundSize = bg.imageFit;
    style.backgroundPosition = 'center';
    style.backgroundRepeat = 'no-repeat';
  }

  return <div style={style} aria-hidden="true" />;
}
