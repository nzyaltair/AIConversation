import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';

export function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[calc(100vh-3rem)] lg:min-h-screen p-8 text-center animate-fade-in">
      <h1 className="text-9xl font-bold gradient-text mb-4">404</h1>
      <p className="text-lg text-muted-foreground mb-6">页面未找到</p>
      <Button asChild variant="outline" className="rounded-lg">
        <Link to="/">
          <ArrowLeft className="h-4 w-4" /> 返回首页
        </Link>
      </Button>
    </div>
  );
}
