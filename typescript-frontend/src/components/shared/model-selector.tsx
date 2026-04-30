import { useState } from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ModelStatusBadge } from '@/components/shared/model-status-badge';
import { getModelDetail, getCategoryLabel } from '@/lib/model-metadata';
import type { ModelInfo, ModelCategory } from '@/types';

interface ModelSelectorProps {
  models: ModelInfo[];
  category: ModelCategory;
  value: string;
  onChange: (variant: string) => void;
  placeholder?: string;
}

/**
 * 模型选择器下拉组件。
 *
 * 按 category 过滤模型列表，每个选项显示模型名称和状态徽章。
 * 当没有匹配的模型时，提示用户前往 Models 页面下载。
 */
export function ModelSelector({ models, category, value, onChange, placeholder = '选择模型' }: ModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const filtered = models.filter((m) => {
    const detail = getModelDetail(m.variant);
    return detail?.category === category;
  });

  if (filtered.length === 0) {
    return (
      <p className="text-xs text-muted-foreground py-1">
        暂无可用{getCategoryLabel(category)}。请前往{' '}
        <a href="/models" className="text-primary underline">模型管理</a> 下载。
      </p>
    );
  }

  const selectedModel = filtered.find((m) => m.variant === value);

  return (
    <Select open={open} onOpenChange={setOpen} value={value} onValueChange={onChange}>
      <SelectTrigger className="w-full">
        <SelectValue asChild>
          {value && selectedModel ? (
            <div className="flex items-center gap-2 text-left">
              <span className="text-sm truncate">{getModelDetail(value)?.displayName ?? value}</span>
              <ModelStatusBadge status={selectedModel.status} />
            </div>
          ) : (
            placeholder
          )}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {filtered.map((m) => (
          <SelectItem key={m.variant} value={m.variant}>
            <div className="flex items-center gap-2">
              <span className="truncate">{getModelDetail(m.variant)?.displayName ?? m.variant}</span>
              <ModelStatusBadge status={m.status} />
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
