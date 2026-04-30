import { motion } from 'framer-motion';
import type { VoiceSessionState } from '@/types';

interface VoiceOrbProps {
  state: VoiceSessionState;
  size?: number;
}

const stateConfig: Record<VoiceSessionState, { gradient: string; ring: string; shadow: string }> = {
  idle: {
    gradient: 'from-slate-700 to-slate-800',
    ring: 'rgba(100,116,139,0.3)',
    shadow: 'rgba(100,116,139,0.15)',
  },
  connecting: {
    gradient: 'from-yellow-600 to-orange-700',
    ring: 'rgba(251,191,36,0.4)',
    shadow: 'rgba(251,191,36,0.2)',
  },
  connected: {
    gradient: 'from-blue-600 to-cyan-700',
    ring: 'rgba(59,130,246,0.4)',
    shadow: 'rgba(59,130,246,0.2)',
  },
  listening: {
    gradient: 'from-blue-500 to-indigo-600',
    ring: 'rgba(59,130,246,0.5)',
    shadow: 'rgba(59,130,246,0.25)',
  },
  processing: {
    gradient: 'from-amber-500 to-orange-600',
    ring: 'rgba(245,158,11,0.5)',
    shadow: 'rgba(245,158,11,0.25)',
  },
  speaking: {
    gradient: 'from-emerald-500 to-teal-600',
    ring: 'rgba(16,185,129,0.5)',
    shadow: 'rgba(16,185,129,0.25)',
  },
};

export function VoiceOrb({ state, size = 120 }: VoiceOrbProps) {
  const config = stateConfig[state];
  const isActive = state === 'listening' || state === 'speaking';

  return (
    <div className="relative flex items-center justify-center">
      {/* Outer ring */}
      {isActive && (
        <motion.div
          className="absolute rounded-full"
          style={{
            width: size + 20,
            height: size + 20,
            border: `2px solid ${config.ring}`,
          }}
          animate={{ scale: [1, 1.15, 1], opacity: [0.6, 0.2, 0.6] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}
      {/* Orb */}
      <motion.div
        className={`rounded-full bg-gradient-to-br ${config.gradient}`}
        style={{
          width: size,
          height: size,
          boxShadow: `0 0 ${size / 2}px ${config.shadow}`,
        }}
        animate={
          isActive
            ? { scale: [1, 1.06, 1] }
            : state === 'processing'
              ? { rotate: 360 }
              : {}
        }
        transition={
          isActive
            ? { duration: 1.5, repeat: Infinity, ease: 'easeInOut' }
            : state === 'processing'
              ? { duration: 3, repeat: Infinity, ease: 'linear' }
              : {}
        }
      >
        {/* Inner highlight */}
        <div
          className="absolute rounded-full bg-white/10"
          style={{
            width: size * 0.4,
            height: size * 0.4,
            top: size * 0.15,
            left: size * 0.2,
          }}
        />
      </motion.div>
    </div>
  );
}
