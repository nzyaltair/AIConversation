import { badgeVariants } from '@/components/ui/badge';

export interface Preset {
  id: string;
  label: string;
  description: string;
  prompt: string;
}

export interface PresetChipGroupProps {
  presets: Preset[];
  onSelect: (prompt: string) => void;
  className?: string;
}

export function PresetChipGroup({
  presets,
  onSelect,
  className,
}: PresetChipGroupProps) {
  return (
    <div className={`flex flex-wrap gap-1.5 ${className ?? ''}`}>
      {presets.map((preset) => (
        <button
          key={preset.id}
          type="button"
          title={preset.description}
          onClick={() => onSelect(preset.prompt)}
          className={badgeVariants({
            variant: 'outline',
            className:
              'cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors',
          })}
        >
          {preset.label}
        </button>
      ))}
    </div>
  );
}
