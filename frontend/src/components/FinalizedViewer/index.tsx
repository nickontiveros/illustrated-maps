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
  { id: 'perspective', label: 'Perspective' },
  { id: 'bordered', label: 'Bordered' },
  { id: 'outpainted', label: 'Outpainted' },
];

export default function FinalizedViewer({ projectName }: FinalizedViewerProps) {
  const { finalizedStage, setFinalizedStage } = useAppStore();
  const { data: status } = usePostProcessStatus(projectName);

  // 3D preview state from store
  const perspectivePreview = useAppStore((s) => s.perspectivePreview);
  const setPerspectivePreview = useAppStore((s) => s.setPerspectivePreview);
  const previewTilt = useAppStore((s) => s.previewTilt);
  const previewRotation = useAppStore((s) => s.previewRotation);
  const setPreviewTilt = useAppStore((s) => s.setPreviewTilt);
  const setPreviewRotation = useAppStore((s) => s.setPreviewRotation);
  const resetPerspectivePreview = useAppStore((s) => s.resetPerspectivePreview);

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
  const [rotating, setRotating] = useState(false);
  const [rotateStart, setRotateStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setScale((s) => Math.max(0.1, Math.min(10, s * delta)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    // Shift+click or right-click = rotate (when 3D preview is on)
    if (perspectivePreview && (e.shiftKey || e.button === 2)) {
      e.preventDefault();
      setRotating(true);
      setRotateStart({ x: e.clientX, y: e.clientY });
      return;
    }
    // Normal drag = pan
    setDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  }, [position, perspectivePreview]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (rotating) {
      const dx = e.clientX - rotateStart.x;
      const dy = e.clientY - rotateStart.y;
      setPreviewRotation(Math.max(-180, Math.min(180, previewRotation + dx * 0.5)));
      setPreviewTilt(Math.max(0, Math.min(70, previewTilt + dy * 0.3)));
      setRotateStart({ x: e.clientX, y: e.clientY });
      return;
    }
    if (!dragging) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  }, [dragging, dragStart, rotating, rotateStart, previewRotation, previewTilt, setPreviewRotation, setPreviewTilt]);

  const handleMouseUp = useCallback(() => {
    setDragging(false);
    setRotating(false);
  }, []);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    if (perspectivePreview) {
      e.preventDefault();
    }
  }, [perspectivePreview]);

  const resetView = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
    if (perspectivePreview) {
      resetPerspectivePreview();
    }
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

  // Build the CSS transform
  const perspectiveTransform = perspectivePreview
    ? `perspective(1200px) rotateX(${previewTilt}deg) rotateZ(${previewRotation}deg)`
    : '';

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
          onClick={() => setPerspectivePreview(!perspectivePreview)}
          className={`px-2 py-1 text-xs border rounded ${
            perspectivePreview
              ? 'bg-indigo-100 text-indigo-700 border-indigo-300'
              : 'text-slate-500 border-slate-200 hover:text-slate-700'
          }`}
        >
          3D Preview
        </button>
        <button
          onClick={resetView}
          className="px-2 py-1 text-xs text-slate-500 hover:text-slate-700 border border-slate-200 rounded"
        >
          Reset View
        </button>
        <span className="text-xs text-slate-400">{Math.round(scale * 100)}%</span>
      </div>

      {/* 3D controls bar */}
      {perspectivePreview && (
        <div className="bg-slate-50 border-b border-slate-200 px-4 py-2 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 w-8">Tilt</span>
            <input
              type="range"
              min={0}
              max={70}
              step={1}
              value={previewTilt}
              onChange={(e) => setPreviewTilt(Number(e.target.value))}
              className="w-24 h-1.5 accent-indigo-600"
            />
            <span className="text-xs text-slate-400 w-8">{previewTilt}&deg;</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 w-12">Rotate</span>
            <input
              type="range"
              min={-180}
              max={180}
              step={1}
              value={previewRotation}
              onChange={(e) => setPreviewRotation(Number(e.target.value))}
              className="w-24 h-1.5 accent-indigo-600"
            />
            <span className="text-xs text-slate-400 w-8">{previewRotation}&deg;</span>
          </div>
          <span className="text-xs text-slate-400">Shift+drag to rotate</span>
        </div>
      )}

      {/* Image viewer */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden bg-slate-800 cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onContextMenu={handleContextMenu}
      >
        <div
          style={{
            transform: `translate(${position.x}px, ${position.y}px) ${perspectiveTransform} scale(${scale})`,
            transformOrigin: 'center center',
            transformStyle: perspectivePreview ? 'preserve-3d' : undefined,
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
