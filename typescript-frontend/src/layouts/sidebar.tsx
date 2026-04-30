import { NavLink } from 'react-router-dom';
import {
  MessageSquare,
  Mic,
  Globe,
  FileAudio,
  Volume2,
  Activity,
  Box,
  ChevronLeft,
  Sun,
  Moon,
  Monitor,
  Palette,
  type LucideIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';
import { useTheme } from '@/app/theme-provider';
import { cn } from '@/lib/utils';
import { useState, useEffect } from 'react';

const COLLAPSED_KEY = 'ai-conversation.sidebar.collapsed';

interface NavItem {
  to: string;
  icon: LucideIcon;
  label: string;
}

const primaryNav: NavItem[] = [
  { to: '/conversation', icon: Mic, label: '语音对话' },
  { to: '/conversation-API', icon: Globe, label: 'API 语音' },
];

const secondaryNav: NavItem[] = [
  { to: '/chat', icon: MessageSquare, label: '聊天' },
  { to: '/speech-to-text', icon: FileAudio, label: '语音转文字' },
  { to: '/text-to-speech', icon: Volume2, label: '文字转语音' },
  { to: '/vad', icon: Activity, label: 'VAD 测试' },
];

const bottomNav: NavItem[] = [
  { to: '/models', icon: Box, label: '模型管理' },
];

export function Sidebar({ onOpenBackgroundSettings }: { onOpenBackgroundSettings?: () => void }) {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(COLLAPSED_KEY) === 'true';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSED_KEY, String(collapsed));
    } catch {
      // ignore
    }
  }, [collapsed]);

  const { theme, preference, setTheme } = useTheme();

  const themeIcon =
    preference === 'system' ? <Monitor className="h-4 w-4" /> :
    theme === 'dark' ? <Moon className="h-4 w-4" /> :
    <Sun className="h-4 w-4" />;

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 z-30 flex h-full flex-col border-r border-border/60 bg-card/90 backdrop-blur-xl transition-all duration-300',
        collapsed ? 'w-[4.5rem]' : 'w-64',
      )}
    >
      {/* Header */}
      <div className={cn('flex items-center h-14 px-3', collapsed ? 'justify-center' : 'gap-2.5')}>
        {!collapsed && (
          <>
            <div className="h-9 w-9 rounded-lg bg-gradient-primary flex items-center justify-center shadow-glow-primary">
              <span className="text-xs font-bold text-primary-foreground">AI</span>
            </div>
            <span className="font-semibold text-sm tracking-tight">AI 对话</span>
          </>
        )}
        {collapsed && (
          <div className="h-8 w-8 rounded-lg bg-gradient-primary flex items-center justify-center shadow-glow-primary">
            <span className="text-xs font-bold text-primary-foreground">AI</span>
          </div>
        )}
      </div>

      <Separator />

      {/* Nav */}
      <nav className="flex-1 flex flex-col gap-0.5 px-3 py-3">
        <NavSection items={primaryNav} collapsed={collapsed} />
        <Separator className="my-2" />
        <NavSection items={secondaryNav} collapsed={collapsed} />
        <div className="flex-1" />
        <Separator className="my-2" />
        <NavSection items={bottomNav} collapsed={collapsed} />
      </nav>

      {/* Footer */}
      <div className="px-3 pb-3 pt-1 border-t border-border/60 space-y-2">
        {onOpenBackgroundSettings && (
          <Tooltip delayDuration={300}>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size={collapsed ? 'icon' : 'sm'}
                className="gap-2 rounded-lg w-full"
                onClick={onOpenBackgroundSettings}
                title="背景设置"
              >
                <Palette className="h-4 w-4" />
                {!collapsed && <span className="text-xs">背景</span>}
              </Button>
            </TooltipTrigger>
            {collapsed && (
              <TooltipContent side="right" className="ml-1">背景设置</TooltipContent>
            )}
          </Tooltip>
        )}
        <div className={cn('flex items-center', collapsed ? 'flex-col gap-2' : 'justify-between')}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="gap-2 rounded-lg">
                {themeIcon}
                {!collapsed && <span className="text-xs capitalize">{preference}</span>}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" side="top" className="rounded-lg">
              <DropdownMenuItem onClick={() => setTheme('dark')}>
                <Moon className="h-4 w-4" /> 深色
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme('light')}>
                <Sun className="h-4 w-4" /> 浅色
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme('system')}>
                <Monitor className="h-4 w-4" /> 跟随系统
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 rounded-lg"
            onClick={() => setCollapsed(!collapsed)}
          >
            <ChevronLeft className={cn('h-4 w-4 transition-transform duration-300', collapsed && 'rotate-180')} />
          </Button>
        </div>

        {!collapsed && (
          <p className="text-[10px] text-muted-foreground/50 mt-1.5 text-center tracking-wide">
            v{__APP_VERSION__}
          </p>
        )}
      </div>
    </aside>
  );
}

function NavSection({ items, collapsed }: { items: NavItem[]; collapsed: boolean }) {
  return (
    <>
      {items.map((item, idx) => {
        const link = (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                'sidebar-link',
                collapsed && 'justify-center px-2',
                isActive && 'active',
                `stagger-${idx + 1}`,
              )
            }
          >
            <item.icon className={cn('h-5 w-5 shrink-0')} />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        );
        if (!collapsed) return link;
        return (
          <Tooltip key={item.to} delayDuration={300}>
            <TooltipTrigger asChild>{link}</TooltipTrigger>
            <TooltipContent side="right" className="ml-1">
              {item.label}
            </TooltipContent>
          </Tooltip>
        );
      })}
    </>
  );
}
