import { useRef, useState, useCallback, useEffect } from 'react';
import { useTileGrid, useRegenerateTile, useStyleReference, useUploadStyleReference, useDeleteStyleReference, useSetTileOffset } from '@/hooks/useTiles';
import { useAppStore } from '@/stores/appStore';
import { api } from '@/api/client';
import type { TileSpec, TileStatus } from '@/types';

interface TileGridProps {
  projectName: string;
}

function TileGrid({ projectName }: TileGridProps) {
  const { data: tileGrid, isLoading } = useTileGrid(projectName);
  const { selectedTile, setSelectedTile } = useAppStore();

  if (isLoading) {
    return (
      <div className="p-4 flex justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!tileGrid) {
    return <div className="p-4 text-slate-500">Failed to load tile grid</div>;
  }

  // Count tile statuses
  const statusCounts = tileGrid.tiles.reduce(
    (acc, tile) => {
      acc[tile.status] = (acc[tile.status] || 0) + 1;
      return acc;
    },
    {} as Record<TileStatus, number>
  );

  return (
    <div className="flex flex-col h-full">
      {/* Stats */}
      <div className="p-4 border-b border-slate-200">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-slate-50 rounded p-2">
            <div className="text-slate-500">Total</div>
            <div className="font-semibold">{tileGrid.tiles.length}</div>
          </div>
          <div className="bg-green-50 rounded p-2">
            <div className="text-green-600">Completed</div>
            <div className="font-semibold text-green-700">{statusCounts.completed || 0}</div>
          </div>
          <div className="bg-blue-50 rounded p-2">
            <div className="text-blue-600">Generating</div>
            <div className="font-semibold text-blue-700">{statusCounts.generating || 0}</div>
          </div>
          <div className="bg-red-50 rounded p-2">
            <div className="text-red-600">Failed</div>
            <div className="font-semibold text-red-700">{statusCounts.failed || 0}</div>
          </div>
        </div>
      </div>

      {/* Style Reference */}
      <StyleReferencePanel projectName={projectName} />

      {/* Grid view */}
      <div className="p-4 border-b border-slate-200">
        <div
          className="grid gap-1"
          style={{
            gridTemplateColumns: `repeat(${tileGrid.cols}, 1fr)`,
          }}
        >
          {/* Sort tiles by row, then col for proper grid layout */}
          {[...tileGrid.tiles]
            .sort((a, b) => a.row * 1000 + a.col - (b.row * 1000 + b.col))
            .map((tile) => {
              const isSelected = selectedTile?.col === tile.col && selectedTile?.row === tile.row;

              const hasOffset = tile.offset_dx !== 0 || tile.offset_dy !== 0;

              return (
                <button
                  key={`${tile.col}-${tile.row}`}
                  onClick={() => setSelectedTile(isSelected ? null : tile)}
                  className={`aspect-square rounded-sm transition-all relative ${
                    isSelected
                      ? 'ring-2 ring-purple-500 ring-offset-1'
                      : 'hover:ring-1 hover:ring-slate-300'
                  } ${
                    tile.status === 'completed'
                      ? 'bg-green-500'
                      : tile.status === 'generating'
                      ? 'bg-blue-500 animate-pulse'
                      : tile.status === 'failed'
                      ? 'bg-red-500'
                      : 'bg-slate-200'
                  } ${hasOffset ? 'border-2 border-yellow-400' : ''}`}
                  title={`Tile (${tile.col}, ${tile.row}) - ${tile.status}${hasOffset ? ` [offset: ${tile.offset_dx},${tile.offset_dy}]` : ''}`}
                >
                  {hasOffset && (
                    <span className="absolute top-0 right-0 w-1.5 h-1.5 bg-yellow-400 rounded-full" />
                  )}
                </button>
              );
            })}
        </div>
      </div>

      {/* Selected tile details */}
      {selectedTile && (
        <TileDetails tile={selectedTile} projectName={projectName} />
      )}
    </div>
  );
}

function StyleReferencePanel({ projectName }: { projectName: string }) {
  const { data: hasRef, isLoading } = useStyleReference(projectName);
  const uploadRef = useUploadStyleReference(projectName);
  const deleteRef = useDeleteStyleReference(projectName);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadRef.mutateAsync(file);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="p-4 border-b border-slate-200">
      <div className="text-sm text-slate-500 mb-2">Style Reference</div>
      {isLoading ? (
        <div className="text-xs text-slate-400">Loading...</div>
      ) : hasRef ? (
        <div className="space-y-2">
          <img
            src={api.getStyleReferenceUrl(projectName)}
            alt="Style reference"
            className="w-full rounded-lg border border-slate-200"
          />
          <div className="flex gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadRef.isPending}
              className="flex-1 px-3 py-1.5 text-xs bg-slate-100 text-slate-700 rounded hover:bg-slate-200 disabled:opacity-50"
            >
              Replace
            </button>
            <button
              onClick={() => deleteRef.mutate()}
              disabled={deleteRef.isPending}
              className="flex-1 px-3 py-1.5 text-xs bg-red-50 text-red-600 rounded hover:bg-red-100 disabled:opacity-50"
            >
              Remove
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-slate-400">
            No style reference set. The central tile will be generated first and auto-used.
          </p>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadRef.isPending}
            className="w-full px-3 py-1.5 text-xs bg-slate-100 text-slate-700 rounded hover:bg-slate-200 disabled:opacity-50"
          >
            {uploadRef.isPending ? 'Uploading...' : 'Upload Style Reference'}
          </button>
        </div>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleUpload}
        className="hidden"
      />
    </div>
  );
}

interface TileDetailsProps {
  tile: TileSpec;
  projectName: string;
}

type ViewMode = 'side-by-side' | 'overlay';

function TileDetails({ tile, projectName }: TileDetailsProps) {
  const regenerateTile = useRegenerateTile(projectName);
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');

  const handleRegenerate = async () => {
    await regenerateTile.mutateAsync({ col: tile.col, row: tile.row, force: true });
  };

  const canOverlay = tile.has_reference && tile.has_generated;

  return (
    <div className="flex-1 p-4 overflow-auto">
      <h3 className="font-semibold text-slate-800 mb-3">
        Tile ({tile.col}, {tile.row})
      </h3>

      <div className="space-y-4">
        <div>
          <div className="text-sm text-slate-500 mb-1">Position</div>
          <div className="text-sm">{tile.position_desc}</div>
        </div>

        <div>
          <div className="text-sm text-slate-500 mb-1">Status</div>
          <span
            className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
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
        </div>

        {tile.generation_time && (
          <div>
            <div className="text-sm text-slate-500 mb-1">Generation Time</div>
            <div className="text-sm">{tile.generation_time.toFixed(1)}s</div>
          </div>
        )}

        {tile.error && (
          <div>
            <div className="text-sm text-slate-500 mb-1">Error</div>
            <div className="text-sm text-red-600">{tile.error}</div>
          </div>
        )}

        {/* View mode toggle */}
        {canOverlay && (
          <div className="flex rounded-lg bg-slate-100 p-0.5">
            <button
              onClick={() => setViewMode('side-by-side')}
              className={`flex-1 px-3 py-1.5 text-xs rounded-md transition-colors ${
                viewMode === 'side-by-side'
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              Side by Side
            </button>
            <button
              onClick={() => setViewMode('overlay')}
              className={`flex-1 px-3 py-1.5 text-xs rounded-md transition-colors ${
                viewMode === 'overlay'
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              Overlay
            </button>
          </div>
        )}

        {viewMode === 'overlay' && canOverlay ? (
          <TileAlignmentOverlay tile={tile} projectName={projectName} />
        ) : (
          <>
            {/* Reference image */}
            {tile.has_reference && (
              <div>
                <div className="text-sm text-slate-500 mb-2">Reference</div>
                <img
                  src={api.getTileReferenceUrl(projectName, tile.col, tile.row, 512)}
                  alt="Reference"
                  className="w-full rounded-lg border border-slate-200"
                />
              </div>
            )}

            {/* Generated image */}
            {tile.has_generated && (
              <div>
                <div className="text-sm text-slate-500 mb-2">Generated</div>
                <img
                  src={api.getTileGeneratedUrl(projectName, tile.col, tile.row, 512)}
                  alt="Generated"
                  className="w-full rounded-lg border border-slate-200"
                />
              </div>
            )}
          </>
        )}

        {/* Offset display */}
        {(tile.offset_dx !== 0 || tile.offset_dy !== 0) && viewMode !== 'overlay' && (
          <div>
            <div className="text-sm text-slate-500 mb-1">Offset</div>
            <div className="text-sm">dx: {tile.offset_dx}, dy: {tile.offset_dy}</div>
          </div>
        )}

        {/* Actions */}
        <div className="pt-2">
          <button
            onClick={handleRegenerate}
            disabled={regenerateTile.isPending}
            className="w-full px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {regenerateTile.isPending ? 'Regenerating...' : 'Regenerate Tile'}
          </button>
        </div>
      </div>
    </div>
  );
}

const DISPLAY_SIZE = 512;

function TileAlignmentOverlay({ tile, projectName }: TileDetailsProps) {
  const setTileOffset = useSetTileOffset(projectName);
  const [dx, setDx] = useState(tile.offset_dx);
  const [dy, setDy] = useState(tile.offset_dy);
  const [opacity, setOpacity] = useState(0.5);
  const containerRef = useRef<HTMLDivElement>(null);

  // Reset local state when tile changes
  useEffect(() => {
    setDx(tile.offset_dx);
    setDy(tile.offset_dy);
  }, [tile.col, tile.row, tile.offset_dx, tile.offset_dy]);

  // Calculate scale factor: display at DISPLAY_SIZE, offsets stored in full-res pixels
  // tileSize is the actual tile resolution (e.g. 2048)
  const tileSize = 2048; // default tile size
  const scale = DISPLAY_SIZE / tileSize;

  const nudge = useCallback((ddx: number, ddy: number) => {
    setDx(prev => Math.max(-50, Math.min(50, prev + ddx)));
    setDy(prev => Math.max(-50, Math.min(50, prev + ddy)));
  }, []);

  // Arrow key handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!containerRef.current?.contains(document.activeElement) &&
          document.activeElement !== containerRef.current) return;

      let step = 1;
      if (e.shiftKey) step = 10;
      else if (e.ctrlKey || e.metaKey) step = 5;

      switch (e.key) {
        case 'ArrowLeft':  e.preventDefault(); nudge(-step, 0); break;
        case 'ArrowRight': e.preventDefault(); nudge(step, 0); break;
        case 'ArrowUp':    e.preventDefault(); nudge(0, -step); break;
        case 'ArrowDown':  e.preventDefault(); nudge(0, step); break;
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
    <div ref={containerRef} tabIndex={0} className="space-y-3 outline-none">
      {/* Overlay viewport */}
      <div
        className="relative overflow-hidden rounded-lg border border-slate-200"
        style={{ width: '100%', aspectRatio: '1' }}
      >
        {/* Reference image (background) */}
        <img
          src={api.getTileReferenceUrl(projectName, tile.col, tile.row, DISPLAY_SIZE)}
          alt="Reference"
          className="absolute inset-0 w-full h-full object-cover"
          draggable={false}
        />
        {/* Generated image (foreground, shifted by offset) */}
        <img
          src={api.getTileGeneratedUrl(projectName, tile.col, tile.row, DISPLAY_SIZE)}
          alt="Generated"
          className="absolute inset-0 w-full h-full object-cover"
          style={{
            opacity,
            transform: `translate(${displayDx}px, ${displayDy}px)`,
          }}
          draggable={false}
        />
      </div>

      {/* Opacity slider */}
      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>Reference</span>
          <span>Generated</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
          className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Arrow controls */}
      <div className="flex items-center justify-center gap-1">
        <div className="grid grid-cols-3 gap-1">
          <div />
          <button onClick={() => nudge(0, -1)} className="px-2 py-1 bg-slate-100 rounded text-xs hover:bg-slate-200">
            &#9650;
          </button>
          <div />
          <button onClick={() => nudge(-1, 0)} className="px-2 py-1 bg-slate-100 rounded text-xs hover:bg-slate-200">
            &#9664;
          </button>
          <div className="px-2 py-1 bg-slate-50 rounded text-xs text-center text-slate-400">
            {dx},{dy}
          </div>
          <button onClick={() => nudge(1, 0)} className="px-2 py-1 bg-slate-100 rounded text-xs hover:bg-slate-200">
            &#9654;
          </button>
          <div />
          <button onClick={() => nudge(0, 1)} className="px-2 py-1 bg-slate-100 rounded text-xs hover:bg-slate-200">
            &#9660;
          </button>
          <div />
        </div>
      </div>

      <p className="text-xs text-slate-400 text-center">
        Arrow keys: 1px, Ctrl: 5px, Shift: 10px
      </p>

      {/* Save / Reset */}
      <div className="flex gap-2">
        <button
          onClick={handleSave}
          disabled={setTileOffset.isPending || !hasChanges}
          className="flex-1 px-3 py-1.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {setTileOffset.isPending ? 'Saving...' : 'Save Offset'}
        </button>
        <button
          onClick={handleReset}
          disabled={dx === 0 && dy === 0}
          className="flex-1 px-3 py-1.5 text-xs bg-slate-100 text-slate-700 rounded hover:bg-slate-200 disabled:opacity-50"
        >
          Reset
        </button>
      </div>
    </div>
  );
}

export default TileGrid;
export { TileAlignmentOverlay, TileDetails };
