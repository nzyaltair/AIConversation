import { Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Sidebar } from '@/layouts/sidebar';

export function MobileHeader() {
  return (
    <header className="lg:hidden fixed top-0 left-0 right-0 z-20 flex items-center justify-between h-14 px-4 border-b border-border/60 bg-card/80 backdrop-blur-xl shadow-sm">
      <div className="flex items-center gap-2.5">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="h-9 w-9 rounded-lg">
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="p-0 w-64">
            <Sidebar />
          </SheetContent>
        </Sheet>
        <div className="h-7 w-7 rounded-lg bg-gradient-primary flex items-center justify-center">
          <span className="text-[10px] font-bold text-primary-foreground">AI</span>
        </div>
        <span className="font-semibold text-sm">AI 对话</span>
      </div>
    </header>
  );
}
