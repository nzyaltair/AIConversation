import { useRef, useState, type DragEvent, type ChangeEvent } from 'react';
import { Upload, FileAudio } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AudioUploadZoneProps {
  onFile: (file: File) => void;
  selectedFile: File | null;
}

export function AudioUploadZone({ onFile, selectedFile }: AudioUploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = () => setIsDragOver(false);
  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      className={cn(
        'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all',
        isDragOver ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/30 hover:bg-muted/20',
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={handleFileChange}
      />
      {selectedFile ? (
        <div className="flex flex-col items-center gap-2">
          <FileAudio className="h-8 w-8 text-primary" />
          <p className="text-sm font-medium">{selectedFile.name}</p>
          <p className="text-xs text-muted-foreground">
            {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
          </p>
          <p className="text-xs text-muted-foreground">点击或拖放以替换</p>
        </div>
      ) : (
        <>
          <Upload className="h-10 w-10 text-muted-foreground/40 mx-auto mb-3" />
          <p className="text-sm font-medium">拖放音频文件到此处，或点击浏览</p>
          <p className="text-xs text-muted-foreground mt-1">支持 MP3、WAV、FLAC、M4A、OGG 格式</p>
        </>
      )}
    </div>
  );
}
