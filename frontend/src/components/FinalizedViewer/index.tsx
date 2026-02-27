import { useState, useRef, useCallback, useEffect } from 'react';
import { useAppStore } from '@/stores/appStore';
import { usePostProcessStatus } from '@/hooks/usePostProcess';
import { api } from '@/api/client';
import type { PostProcessStage } from '@/types';

interface FinalizedViewerProps {
  projectName: string;
}

const STAGES: { id: PostProcessStage; label: string }[] = [
  { id: 'assembled', label: 'Assembled' },
  { id: 'composed', label: 'Composed' },
  { id: 'labeled', label: 'Labeled' },
  { id: 'bordered', label: 'Bordered' },
  { id: 'outpainted', label: 'Outpainted' },
];

export default function FinalizedViewer({ projectName }: FinalizedViewerProps) {
  const { finalizedStage, setFinalizedStage } = useAppStore();
  const { data: status } = usePostProcessStatus(projectName);

  // Auto-select the latest available stage
  useEffect(() => {
    if (status && !finalizedStage) {
      const latest = status.latest_stage;
      if (latest) setFinalizedStage(latest);
    }
  }, [status, finalizedStage, setFinalizedStage]);

  const currentStage = finalizedStage || status?.latest_stage || 'assembled';
  const imageUrl = api.getPostProcessImageUrl(projectName, currentStage);

  // Zoom/pan state
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setScale((s) => Math.max(0.1, Math.min(10, s * delta)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  }, [position]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  }, [dragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setDragging(false);
  }, []);

  const resetView = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  // Available stages
  const availableStages = STAGES.filter(
    (s) => status && status[s.id]
  );

  if (availableStages.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-100">
        <div className="text-center text-slate-500 space-y-2">
          <div className="text-4xl">✨</div>
          <div className="text-sm">No finalized outputs yet.</div>
          <div className="text-xs">Generate tiles and run the Finalize pipeline to view results here.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col">
      {/* Stage selector */}
      <div className="bg-white border-b border-slate-200 px-4 py-2 flex items-center gap-2">
        <span className="text-sm text-slate-500">Stage:</span>
        {availableStages.map((s) => (
          <button
            key={s.id}
            onClick={() => setFinalizedStage(s.id)}
            className={`px-3 py-1 text-sm rounded ${
              currentStage === s.id
                ? 'bg-blue-100 text-blue-700'
                : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            {s.label}
          </button>
        ))}
        <div className="flex-1" />
        <button
          onClick={resetView}
          className="px-2 py-1 text-xs text-slate-500 hover:text-slate-700 border border-slate-200 rounded"
        >
          Reset View
        </button>
        <span className="text-xs text-slate-400">{Math.round(scale * 100)}%</span>
      </div>

      {/* Image viewer */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden bg-slate-800 cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: 'center center',
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <img
            src={imageUrl}
            alt={`${currentStage} output`}
            className="max-w-none"
            style={{ imageRendering: scale > 2 ? 'pixelated' : 'auto' }}
            draggable={false}
          />
        </div>
      </div>
    </div>
  );
}
