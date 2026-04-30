import { createContext, useContext, useEffect, type ReactNode } from 'react';
import { useThemeStore } from '@/stores/theme-store';

type ThemeContextValue = {
  theme: 'light' | 'dark';
  preference: 'system' | 'light' | 'dark';
  setTheme: (t: 'system' | 'light' | 'dark') => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const preference = useThemeStore((s) => s.preference);
  const resolved = useThemeStore((s) => s.resolved);
  const setPreference = useThemeStore((s) => s.setPreference);

  useEffect(() => {
    document.documentElement.classList.toggle('theme-light', resolved === 'light');
  }, [resolved]);

  return (
    <ThemeContext.Provider value={{ theme: resolved, preference, setTheme: setPreference }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}
