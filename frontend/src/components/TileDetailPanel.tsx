import { useState, useRef, useCallback, useEffect } from 'react';
import { useAppStore } from '@/stores/appStore';
import { useRegenerateTile, useSetTileOffset } from '@/hooks/useTiles';
import { api } from '@/api/client';
import type { TileSpec } from '@/types';

interface TileDetailPanelProps {
  projectName: string;
}

type DetailViewMode = 'side-by-side' | 'overlay';

export default function TileDetailPanel({ projectName }: TileDetailPanelProps) {
  const { selectedTile } = useAppStore();

  if (!selectedTile) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400">
        <div className="text-center">
          <div className="text-lg mb-2">No tile selected</div>
          <div className="text-sm">Select a tile from the grid in the sidebar</div>
        </div>
      </div>
    );
  }

  return <TileDetailContent tile={selectedTile} projectName={projectName} />;
}

function TileDetailContent({ tile, projectName }: { tile: TileSpec; projectName: string }) {
  const regenerateTile = useRegenerateTile(projectName);
  const [viewMode, setViewMode] = useState<DetailViewMode>('side-by-side');

  const handleRegenerate = async () => {
    await regenerateTile.mutateAsync({ col: tile.col, row: tile.row, force: true });
  };

  const canOverlay = tile.has_reference && tile.has_generated;

  return (
    <div className="p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">
            Tile ({tile.col}, {tile.row})
          </h2>
          <div className="flex items-center gap-3 mt-1 text-sm text-slate-500">
            <span>{tile.position_desc}</span>
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                tile.status === 'completed'
                  ? 'bg-green-100 text-green-700'
                  : tile.status === 'generating'
                  ? 'bg-blue-100 text-blue-700'
                  : tile.status === 'failed'
                  ? 'bg-red-100 text-red-700'
                  : 'bg-slate-100 text-slate-600'
              }`}
            >
              {tile.status}
            </span>
            {(tile.offset_dx !== 0 || tile.offset_dy !== 0) && (
              <span className="text-xs text-yellow-600">
                offset: ({tile.offset_dx}, {tile.offset_dy})
              </span>
            )}
          </div>
        </div>
        <button
          onClick={handleRegenerate}
          disabled={regenerateTile.isPending}
          className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {regenerateTile.isPending ? 'Regenerating...' : 'Regenerate'}
        </button>
      </div>

      {/* View mode toggle */}
      {canOverlay && (
        <div className="flex rounded-lg bg-slate-100 p-0.5 mb-6 w-fit">
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              viewMode === 'side-by-side'
                ? 'bg-white text-slate-800 shadow-sm'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            Side by Side
          </button>
          <button
            onClick={() => setViewMode('overlay')}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              viewMode === 'overlay'
                ? 'bg-white text-slate-800 shadow-sm'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            Alignment Overlay
          </button>
        </div>
      )}

      {viewMode === 'overlay' && canOverlay ? (
        <LargeAlignmentOverlay tile={tile} projectName={projectName} />
      ) : (
        <div className="grid grid-cols-2 gap-6">
          {tile.has_reference && (
            <div>
              <div className="text-sm font-medium text-slate-500 mb-2">Reference</div>
              <img
                src={api.getTileReferenceUrl(projectName, tile.col, tile.row)}
                alt="Reference"
                className="w-full rounded-lg border border-slate-200"
              />
            </div>
          )}
          {tile.has_generated && (
            <div>
              <div className="text-sm font-medium text-slate-500 mb-2">Generated</div>
              <img
                src={api.getTileGeneratedUrl(projectName, tile.col, tile.row)}
                alt="Generated"
                className="w-full rounded-lg border border-slate-200"
              />
            </div>
          )}
        </div>
      )}

      {tile.error && (
        <div className="mt-4 p-3 bg-red-50 rounded-lg text-sm text-red-600">
          {tile.error}
        </div>
      )}
    </div>
  );
}

function LargeAlignmentOverlay({ tile, projectName }: { tile: TileSpec; projectName: string }) {
  const setTileOffset = useSetTileOffset(projectName);
  const [dx, setDx] = useState(tile.offset_dx);
  const [dy, setDy] = useState(tile.offset_dy);
  const [opacity, setOpacity] = useState(0.5);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setDx(tile.offset_dx);
    setDy(tile.offset_dy);
  }, [tile.col, tile.row, tile.offset_dx, tile.offset_dy]);

  // Scale: display uses full-res images in a large viewport, offsets are in full-res pixels (2048)
  // The overlay image fills its container via CSS; we need to scale offsets to the rendered size
  const tileSize = 2048;
  const [containerWidth, setContainerWidth] = useState(600);
  const imgRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!imgRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(imgRef.current);
    return () => observer.disconnect();
  }, []);

  const scale = containerWidth / tileSize;

  const nudge = useCallback((ddx: number, ddy: number) => {
    setDx(prev => Math.max(-50, Math.min(50, prev + ddx)));
    setDy(prev => Math.max(-50, Math.min(50, prev + ddy)));
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Always capture arrow keys when this panel is visible
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
      // Don't capture if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      e.preventDefault();
      let step = 1;
      if (e.shiftKey) step = 10;
      else if (e.ctrlKey || e.metaKey) step = 5;

      switch (e.key) {
        case 'ArrowLeft':  nudge(-step, 0); break;
        case 'ArrowRight': nudge(step, 0); break;
        case 'ArrowUp':    nudge(0, -step); break;
        case 'ArrowDown':  nudge(0, step); break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [nudge]);

  const handleSave = async () => {
    await setTileOffset.mutateAsync({ col: tile.col, row: tile.row, dx, dy });
  };

  const handleReset = () => {
    setDx(0);
    setDy(0);
  };

  const hasChanges = dx !== tile.offset_dx || dy !== tile.offset_dy;
  const displayDx = dx * scale;
  const displayDy = dy * scale;

  return (
    <div ref={containerRef} className="space-y-4">
      {/* Large overlay viewport */}
      <div
        ref={imgRef}
        className="relative overflow-hidden rounded-lg border border-slate-200 bg-slate-100"
        style={{ width: '100%', maxWidth: '800px', aspectRatio: '1' }}
      >
        <img
          src={api.getTileReferenceUrl(projectName, tile.col, tile.row)}
          alt="Reference"
          className="absolute inset-0 w-full h-full object-cover"
          draggable={false}
        />
        <img
          src={api.getTileGeneratedUrl(projectName, tile.col, tile.row)}
          alt="Generated"
          className="absolute inset-0 w-full h-full object-cover"
          style={{
            opacity,
            transform: `translate(${displayDx}px, ${displayDy}px)`,
          }}
          draggable={false}
        />
      </div>

      {/* Controls bar */}
      <div className="flex items-center gap-6 flex-wrap" style={{ maxWidth: '800px' }}>
        {/* Opacity slider */}
        <div className="flex items-center gap-3 flex-1 min-w-[200px]">
          <span className="text-xs text-slate-500 whitespace-nowrap">Reference</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
          />
          <span className="text-xs text-slate-500 whitespace-nowrap">Generated</span>
        </div>

        {/* Arrow controls */}
        <div className="flex items-center gap-1">
          <div className="grid grid-cols-3 gap-0.5">
            <div />
            <button onClick={() => nudge(0, -1)} className="px-2 py-1 bg-slate-100 rounded text-xs hover:bg-slate-200">&#9650;</button>
            <div />
            <button onClick={() => nudge(-1, 0)} className="px-2 py-1 bg-slate-100 rounded text-xs hover:bg-slate-200">&#9664;</button>
            <div className="px-2 py-1 bg-slate-50 rounded text-xs text-center text-slate-500 font-mono">
              {dx},{dy}
            </div>
            <button onClick={() => nudge(1, 0)} className="px-2 py-1 bg-slate-100 rounded text-xs hover:bg-slate-200">&#9654;</button>
            <div />
            <button onClick={() => nudge(0, 1)} className="px-2 py-1 bg-slate-100 rounded text-xs hover:bg-slate-200">&#9660;</button>
            <div />
          </div>
        </div>

        {/* Save / Reset */}
        <div className="flex gap-2">
          <button
            onClick={handleSave}
            disabled={setTileOffset.isPending || !hasChanges}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {setTileOffset.isPending ? 'Saving...' : 'Save Offset'}
          </button>
          <button
            onClick={handleReset}
            disabled={dx === 0 && dy === 0}
            className="px-4 py-2 text-sm bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 disabled:opacity-50"
          >
            Reset
          </button>
        </div>
      </div>

      <p className="text-xs text-slate-400">
        Arrow keys: 1px | Ctrl+Arrow: 5px | Shift+Arrow: 10px
      </p>
    </div>
  );
}
