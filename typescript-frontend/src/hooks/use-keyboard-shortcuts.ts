import { useEffect } from 'react';

export interface Shortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  handler: () => void;
  preventDefault?: boolean;
}

export function useKeyboardShortcuts(shortcuts: Shortcut[]) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      for (const s of shortcuts) {
        const ctrlMatch = !!s.ctrl === (e.ctrlKey || e.metaKey);
        if (
          ctrlMatch &&
          !!s.shift === e.shiftKey &&
          !!s.alt === e.altKey &&
          e.key.toLowerCase() === s.key.toLowerCase()
        ) {
          if (s.preventDefault !== false) e.preventDefault();
          s.handler();
          return;
        }
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [shortcuts]);
}
