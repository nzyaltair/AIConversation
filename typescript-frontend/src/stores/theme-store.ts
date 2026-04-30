import { create } from 'zustand';

type ThemePreference = 'system' | 'light' | 'dark';

interface ThemeStore {
  preference: ThemePreference;
  resolved: 'light' | 'dark';
  setPreference: (p: ThemePreference) => void;
  setResolved: (t: 'light' | 'dark') => void;
}

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function resolveTheme(pref: ThemePreference): 'light' | 'dark' {
  return pref === 'system' ? getSystemTheme() : pref;
}

function loadPreference(): ThemePreference {
  try {
    const stored = localStorage.getItem('ai-conversation.theme');
    if (stored === 'light' || stored === 'dark' || stored === 'system') return stored;
  } catch {
    // localStorage unavailable
  }
  return 'system';
}

function applyThemeClass(theme: 'light' | 'dark') {
  const root = document.documentElement;
  if (theme === 'light') {
    root.classList.add('theme-light');
  } else {
    root.classList.remove('theme-light');
  }
}

const initialPref = loadPreference();
const initialResolved = resolveTheme(initialPref);
applyThemeClass(initialResolved);

export const useThemeStore = create<ThemeStore>((set) => ({
  preference: initialPref,
  resolved: initialResolved,
  setPreference: (pref) => {
    const resolved = resolveTheme(pref);
    applyThemeClass(resolved);
    try {
      localStorage.setItem('ai-conversation.theme', pref);
    } catch {
      // ignore
    }
    set({ preference: pref, resolved });
  },
  setResolved: (resolved) => {
    applyThemeClass(resolved);
    set({ resolved });
  },
}));

if (typeof window !== 'undefined') {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const store = useThemeStore.getState();
    if (store.preference === 'system') {
      store.setResolved(getSystemTheme());
    }
  });
}
