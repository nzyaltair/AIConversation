import { useRef, useEffect, useCallback } from 'react';

interface VolumeVisualizerProps {
  volume: number;
  isActive: boolean;
  barCount?: number;
  height?: number;
}

function lerp(current: number, target: number, factor: number): number {
  return current + (target - current) * factor;
}

// Seeded pseudo-random based on index and time phase
function barHeightModifier(index: number, timePhase: number): number {
  // Simple deterministic pattern using index and time
  const seed = (index * 7 + index * index * 3 + timePhase * 5) % 1;
  return 0.3 + 0.7 * Math.abs(seed);
}

export function VolumeVisualizer({
  volume,
  isActive,
  barCount = 20,
  height = 48,
}: VolumeVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const currentLevelRef = useRef(0);
  const animFrameRef = useRef<number | null>(null);
  const timeRef = useRef(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;

    // Set canvas size matching display size
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Smooth interpolation toward target volume
    currentLevelRef.current = lerp(currentLevelRef.current, volume, 0.3);
    timeRef.current += 0.02;

    const barWidth = (width - (barCount - 1) * 2) / barCount;
    const gap = 2;

    ctx.clearRect(0, 0, width, height);

    if (!isActive || currentLevelRef.current < 0.01) {
      // Draw a subtle pulsing line when inactive
      const pulseAlpha = 0.15 + 0.05 * Math.sin(timeRef.current * 2);
      ctx.strokeStyle = `hsl(var(--muted-foreground) / ${pulseAlpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    const timePhase = timeRef.current;

    for (let i = 0; i < barCount; i++) {
      const modifier = barHeightModifier(i, timePhase);
      const barHeight = Math.max(1, currentLevelRef.current * height * modifier);
      const x = i * (barWidth + gap);
      const y = height - barHeight;

      // Monochromatic blue palette — elegant, professional
      const t = currentLevelRef.current * modifier;
      const hue = 212 + t * 10;
      const saturation = 60 + t * 30;
      const lightness = 50 + t * 15;

      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, [1, 1, 0, 0]);
      ctx.fill();
    }
  }, [volume, isActive, barCount, height]);

  useEffect(() => {
    const animate = () => {
      draw();
      animFrameRef.current = requestAnimationFrame(animate);
    };

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    };
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ height: `${height}px` }}
    />
  );
}
