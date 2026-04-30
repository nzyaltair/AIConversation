import { Component, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorFallbackProps {
  error?: Error | null;
  onRetry?: () => void;
}

export class ErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode; fallback?: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? <ErrorFallback error={this.state.error} />;
    }
    return this.props.children;
  }
}

export function ErrorFallback({ error, onRetry }: ErrorFallbackProps) {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center animate-fade-in-up border border-destructive/10 rounded-xl">
      <div className="h-14 w-14 rounded-2xl bg-destructive/10 flex items-center justify-center mb-4">
        <AlertTriangle className="h-6 w-6 text-destructive/70" />
      </div>
      <p className="text-base font-medium">出错了</p>
      <p className="text-xs text-muted-foreground mt-1.5 max-w-sm leading-relaxed">
        {error?.message ?? '发生了意外错误。'}
      </p>
      {onRetry && (
        <Button size="sm" variant="outline" className="mt-4 rounded-lg" onClick={onRetry}>
          <RefreshCw className="h-3.5 w-3.5" /> 重试
        </Button>
      )}
    </div>
  );
}
