import { useAudioPlayer } from '@/hooks/use-audio-player';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, Pause, Square, Download } from 'lucide-react';
import { useEffect } from 'react';

interface AudioPlayerProps {
  audioUrl: string | null;
  audioBlob?: Blob | null;
}

export function AudioPlayer({ audioUrl, audioBlob }: AudioPlayerProps) {
  const { isPlaying, currentTime, duration, play, pause, stop, seek, setAudioSource } = useAudioPlayer();

  useEffect(() => {
    if (audioUrl) setAudioSource(audioUrl);
  }, [audioUrl, setAudioSource]);

  if (!audioUrl) return null;

  const formatTime = (s: number) => {
    if (!isFinite(s) || s < 0) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const handleDownload = () => {
    const url = audioBlob
      ? URL.createObjectURL(audioBlob)
      : audioUrl;
    const a = document.createElement('a');
    a.href = url;
    a.download = 'audio.wav';
    a.click();
    if (audioBlob) URL.revokeObjectURL(url);
  };

  return (
    <div className="flex items-center gap-3 p-3 rounded-full bg-card/80 backdrop-blur-md border border-border/50">
      <Button size="icon" variant="ghost" className="h-8 w-8 rounded-full hover:bg-accent/60" onClick={isPlaying ? pause : play}>
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </Button>
      <Button size="icon" variant="ghost" className="h-8 w-8 rounded-full hover:bg-accent/60" onClick={stop}>
        <Square className="h-3 w-3" />
      </Button>
      <div className="flex-1">
        <Slider
          value={[currentTime]}
          min={0}
          max={duration || 1}
          step={0.1}
          onValueChange={([v]) => seek(v)}
          className="h-4"
        />
      </div>
      <span className="text-xs text-muted-foreground tabular-nums min-w-[70px] text-right">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
      <Button size="icon" variant="ghost" className="h-8 w-8 rounded-full hover:bg-accent/60" onClick={handleDownload}>
        <Download className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
