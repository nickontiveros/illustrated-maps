import { useSeams, useRepairSeam, useRepairAllSeams } from '@/hooks/useSeams';
import { useAppStore } from '@/stores/appStore';
import { api } from '@/api/client';
import type { SeamInfo } from '@/types';

interface SeamRepairProps {
  projectName: string;
}

function SeamRepair({ projectName }: SeamRepairProps) {
  const { data: seamList, isLoading } = useSeams(projectName);
  const { selectedSeam, setSelectedSeam } = useAppStore();
  const repairAllSeams = useRepairAllSeams(projectName);

  if (isLoading) {
    return (
      <div className="p-4 flex justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!seamList) {
    return <div className="p-4 text-slate-500">Failed to load seams</div>;
  }

  const unrepairedSeams = seamList.seams.filter((s) => !s.is_repaired);

  return (
    <div className="flex flex-col h-full">
      {/* Stats */}
      <div className="p-4 border-b border-slate-200">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-slate-50 rounded p-2">
            <div className="text-slate-500">Total Seams</div>
            <div className="font-semibold">{seamList.total_seams}</div>
          </div>
          <div className="bg-green-50 rounded p-2">
            <div className="text-green-600">Repaired</div>
            <div className="font-semibold text-green-700">{seamList.repaired_seams}</div>
          </div>
        </div>

        {unrepairedSeams.length > 0 && (
          <button
            onClick={() => repairAllSeams.mutate()}
            disabled={repairAllSeams.isPending}
            className="mt-3 w-full px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {repairAllSeams.isPending ? 'Repairing...' : `Repair All (${unrepairedSeams.length})`}
          </button>
        )}
      </div>

      {/* Seam list */}
      <div className="flex-1 overflow-auto">
        <div className="divide-y divide-slate-100">
          {seamList.seams.map((seam) => {
            const isSelected = selectedSeam?.id === seam.id;

            return (
              <button
                key={seam.id}
                onClick={() => setSelectedSeam(isSelected ? null : seam)}
                className={`w-full p-3 text-left hover:bg-slate-50 transition-colors ${
                  isSelected ? 'bg-blue-50' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-slate-800">
                    {seam.description}
                  </div>
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                      seam.is_repaired
                        ? 'bg-green-100 text-green-700'
                        : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {seam.is_repaired ? 'Repaired' : 'Pending'}
                  </span>
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  {seam.orientation} • {seam.width}×{seam.height}px
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected seam details */}
      {selectedSeam && (
        <SeamDetails seam={selectedSeam} projectName={projectName} />
      )}
    </div>
  );
}

interface SeamDetailsProps {
  seam: SeamInfo;
  projectName: string;
}

function SeamDetails({ seam, projectName }: SeamDetailsProps) {
  const repairSeam = useRepairSeam(projectName);

  const handleRepair = async () => {
    await repairSeam.mutateAsync(seam.id);
  };

  return (
    <div className="border-t border-slate-200 p-4 space-y-4">
      <h3 className="font-semibold text-slate-800">
        {seam.description}
      </h3>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <div className="text-slate-500 mb-1">Orientation</div>
          <div>{seam.orientation}</div>
        </div>
        <div>
          <div className="text-slate-500 mb-1">Size</div>
          <div>{seam.width}×{seam.height}px</div>
        </div>
        <div>
          <div className="text-slate-500 mb-1">Position</div>
          <div>({seam.x}, {seam.y})</div>
        </div>
        <div>
          <div className="text-slate-500 mb-1">Status</div>
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
              seam.is_repaired
                ? 'bg-green-100 text-green-700'
                : 'bg-slate-100 text-slate-600'
            }`}
          >
            {seam.is_repaired ? 'Repaired' : 'Pending'}
          </span>
        </div>
      </div>

      {/* Before/After comparison */}
      <div className="space-y-2">
        <div className="text-sm text-slate-500">Preview</div>
        <img
          src={api.getSeamPreviewUrl(projectName, seam.id)}
          alt="Seam preview"
          className="w-full rounded-lg border border-slate-200"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = 'none';
          }}
        />
      </div>

      {seam.is_repaired && (
        <div className="space-y-2">
          <div className="text-sm text-slate-500">Repaired</div>
          <img
            src={api.getSeamRepairedUrl(projectName, seam.id)}
            alt="Repaired seam"
            className="w-full rounded-lg border border-slate-200"
          />
        </div>
      )}

      {!seam.is_repaired && (
        <button
          onClick={handleRepair}
          disabled={repairSeam.isPending}
          className="w-full px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {repairSeam.isPending ? 'Repairing...' : 'Repair Seam'}
        </button>
      )}
    </div>
  );
}

export default SeamRepair;
