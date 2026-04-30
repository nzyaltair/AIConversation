import { cn } from '@/lib/utils';
import type { VoiceSessionState } from '@/types';

interface VoiceOrbProps {
  state: VoiceSessionState;
}

function getOrbColors(state: VoiceSessionState): { ring: string; glow: string; pulse: boolean; spin: boolean } {
  switch (state) {
    case 'listening':
      return { ring: 'border-success/60', glow: 'shadow-glow-success', pulse: true, spin: false };
    case 'processing':
      return { ring: 'border-primary/60', glow: 'shadow-glow-primary', pulse: false, spin: true };
    case 'speaking':
      return { ring: 'border-warning/60', glow: 'shadow-glow-warning', pulse: true, spin: false };
    case 'connecting':
      return { ring: 'border-warning/40', glow: '', pulse: true, spin: false };
    default:
      return { ring: 'border-border/40', glow: '', pulse: false, spin: false };
  }
}

export function VoiceOrb({ state }: VoiceOrbProps) {
  const colors = getOrbColors(state);
  const isActive = state !== 'idle' && state !== 'connected';

  return (
    <div className="relative flex items-center justify-center">
      {/* Outer glow ring */}
      {isActive && (
        <div
          className={cn(
            'absolute w-28 h-28 rounded-full border-2 opacity-60',
            colors.ring,
            colors.pulse && 'orb-pulse',
            colors.spin && 'orb-spin',
          )}
        />
      )}

      {/* Inner orb */}
      <div
        className={cn(
          'w-20 h-20 rounded-full flex items-center justify-center transition-all duration-700 ease-out',
          'bg-gradient-primary',
          colors.glow && colors.glow,
          isActive ? 'opacity-100 scale-100' : 'opacity-40 scale-90',
          state === 'listening' && 'orb-pulse',
          state === 'speaking' && 'animate-pulse',
        )}
      >
        {/* Center dot */}
        <div
          className={cn(
            'w-4 h-4 rounded-full bg-white',
            isActive ? 'opacity-90' : 'opacity-40',
          )}
        />
      </div>
    </div>
  );
}
