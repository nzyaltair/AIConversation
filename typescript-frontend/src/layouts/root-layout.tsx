import { type ReactNode, useState } from 'react';
import { Sidebar } from '@/layouts/sidebar';
import { MobileHeader } from '@/layouts/mobile-header';
import { BackgroundLayer } from '@/components/background/background-layer';
import { BackgroundSettings } from '@/components/background/background-settings';

export function RootLayout({ children }: { children: ReactNode }) {
  const [bgSettingsOpen, setBgSettingsOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/3 via-background to-background">
      <BackgroundLayer />
      <Sidebar onOpenBackgroundSettings={() => setBgSettingsOpen(true)} />
      <MobileHeader />
      <main className="lg:pl-[--sidebar-width] pt-12 lg:pt-0">
        {children}
      </main>
      <BackgroundSettings open={bgSettingsOpen} onClose={() => setBgSettingsOpen(false)} />
    </div>
  );
}
