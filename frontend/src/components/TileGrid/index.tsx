import { useTileGrid, useRegenerateTile } from '@/hooks/useTiles';
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

              return (
                <button
                  key={`${tile.col}-${tile.row}`}
                  onClick={() => setSelectedTile(isSelected ? null : tile)}
                  className={`aspect-square rounded-sm transition-all ${
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
                  }`}
                  title={`Tile (${tile.col}, ${tile.row}) - ${tile.status}`}
                />
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

interface TileDetailsProps {
  tile: TileSpec;
  projectName: string;
}

function TileDetails({ tile, projectName }: TileDetailsProps) {
  const regenerateTile = useRegenerateTile(projectName);

  const handleRegenerate = async () => {
    await regenerateTile.mutateAsync({ col: tile.col, row: tile.row, force: true });
  };

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

export default TileGrid;
