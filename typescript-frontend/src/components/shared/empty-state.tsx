import type { LucideIcon } from 'lucide-react';
import { Inbox } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: { label: string; onClick: () => void };
}

export function EmptyState({ icon: Icon = Inbox, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center animate-fade-in-up">
      <div className="h-14 w-14 rounded-2xl bg-muted/50 flex items-center justify-center mb-4">
        <Icon className="h-6 w-6 text-muted-foreground/50" />
      </div>
      <p className="text-base font-medium text-muted-foreground">{title}</p>
      {description && (
        <p className="text-xs text-muted-foreground/60 mt-1.5 max-w-xs leading-relaxed">{description}</p>
      )}
      {action && (
        <Button size="sm" variant="outline" className="mt-4 rounded-lg" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </div>
  );
}
