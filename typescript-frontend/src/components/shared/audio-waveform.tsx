import { useRef, useEffect } from 'react';

interface AudioWaveformProps {
  analyserNode?: AnalyserNode | null;
  isActive?: boolean;
  className?: string;
}

export function AudioWaveform({ analyserNode, isActive = false, className = '' }: AudioWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (!isActive || !analyserNode || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;
      analyserNode.getByteTimeDomainData(dataArray);

      ctx.fillStyle = 'hsl(var(--background))';
      ctx.fillRect(0, 0, W, H);
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'hsl(var(--primary))';
      ctx.beginPath();

      const sliceWidth = W / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * H / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();
      rafRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [isActive, analyserNode]);

  return (
    <canvas
      ref={canvasRef}
      className={`w-full h-16 rounded-md ${className}`}
      width={300}
      height={64}
    />
  );
}
