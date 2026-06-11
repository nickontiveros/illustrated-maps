import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { v2api, type V2Poi, type V2ProjectConfig } from '@/api/v2';

const DEFAULT_REGION = { north: 40.735, south: 40.695, east: -73.985, west: -74.025 };

function emptyPoi(): V2Poi {
  return { name: '', lat: DEFAULT_REGION.north, lon: DEFAULT_REGION.west, tier: 2 };
}

function ProjectListV2() {
  const queryClient = useQueryClient();
  const { data: projects, isLoading, error } = useQuery({
    queryKey: ['v2-projects'],
    queryFn: v2api.listProjects,
  });

  const [showCreate, setShowCreate] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [region, setRegion] = useState(DEFAULT_REGION);
  const [pois, setPois] = useState<V2Poi[]>([emptyPoi()]);

  const createProject = useMutation({
    mutationFn: (config: V2ProjectConfig) => v2api.createProject(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['v2-projects'] });
      setShowCreate(false);
      setName('');
      setPois([emptyPoi()]);
      setCreateError(null);
    },
    onError: (e: Error) => setCreateError(e.message),
  });

  const deleteProject = useMutation({
    mutationFn: (id: string) => v2api.deleteProject(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['v2-projects'] }),
  });

  const submit = () => {
    const validPois = pois.filter((p) => p.name.trim());
    createProject.mutate({ name: name.trim(), region, pois: validPois });
  };

  const setPoi = (i: number, patch: Partial<V2Poi>) =>
    setPois((prev) => prev.map((p, j) => (j === i ? { ...p, ...patch } : p)));

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">V2 Maps</h1>
          <p className="text-sm text-slate-500">
            Asset-composition pipeline: plan &rarr; assets &rarr; compose
          </p>
        </div>
        <div className="flex gap-3">
          <Link to="/v1" className="px-3 py-2 text-sm text-slate-600 hover:text-slate-900">
            V1 projects (legacy)
          </Link>
          <button
            onClick={() => setShowCreate(true)}
            className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700"
          >
            New V2 Map
          </button>
        </div>
      </div>

      {isLoading && <p className="text-slate-500">Loading…</p>}
      {error && <p className="text-red-600">Failed to load: {(error as Error).message}</p>}

      <div className="grid gap-4 sm:grid-cols-2">
        {projects?.map((p) => (
          <div key={p.id} className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
            <div className="flex items-start justify-between">
              <Link to={`/v2/${p.id}`} className="font-semibold text-slate-800 hover:text-indigo-600">
                {p.name}
              </Link>
              <button
                onClick={() => {
                  if (confirm(`Delete "${p.name}"?`)) deleteProject.mutate(p.id);
                }}
                className="text-xs text-slate-400 hover:text-red-600"
              >
                Delete
              </button>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              {p.poi_count} POIs · {p.region.north.toFixed(3)},{p.region.west.toFixed(3)} →{' '}
              {p.region.south.toFixed(3)},{p.region.east.toFixed(3)}
            </p>
            <div className="flex gap-2 mt-3">
              <StageChip
                label={p.plan_stale ? 'Plan (stale)' : 'Plan'}
                done={p.has_plan && !p.plan_stale}
                state={p.plan_stale ? 'stale' : p.status?.plan?.state}
              />
              <StageChip
                label="Assets"
                done={p.status?.assets?.state === 'done'}
                state={p.status?.assets?.state}
              />
              <StageChip label="Poster" done={p.has_poster} state={p.status?.compose?.state} />
            </div>
          </div>
        ))}
        {projects?.length === 0 && (
          <p className="text-slate-500 col-span-2">No V2 maps yet. Create one to get started.</p>
        )}
      </div>

      {showCreate && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto p-6">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">New V2 Map</h2>

            <label className="block text-sm font-medium text-slate-600 mb-1">Map title</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Lower Manhattan"
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm mb-4"
            />

            <label className="block text-sm font-medium text-slate-600 mb-1">Region (degrees)</label>
            <div className="grid grid-cols-4 gap-2 mb-4">
              {(['north', 'south', 'east', 'west'] as const).map((edge) => (
                <div key={edge}>
                  <span className="text-xs text-slate-400 capitalize">{edge}</span>
                  <input
                    type="number"
                    step="0.001"
                    value={region[edge]}
                    onChange={(e) => setRegion({ ...region, [edge]: parseFloat(e.target.value) })}
                    className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
                  />
                </div>
              ))}
            </div>

            <div className="flex items-center justify-between mb-1">
              <label className="text-sm font-medium text-slate-600">
                Points of interest (illustrated, labeled, placed)
              </label>
              <button
                onClick={() => setPois((prev) => [...prev, emptyPoi()])}
                className="text-xs text-indigo-600 hover:text-indigo-800"
              >
                + Add POI
              </button>
            </div>
            <div className="space-y-2 mb-4">
              {pois.map((poi, i) => (
                <div key={i} className="grid grid-cols-[1fr_90px_90px_70px_24px] gap-2 items-center">
                  <input
                    value={poi.name}
                    onChange={(e) => setPoi(i, { name: e.target.value })}
                    placeholder="Name"
                    className="border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
                  />
                  <input
                    type="number"
                    step="0.0001"
                    value={poi.lat}
                    onChange={(e) => setPoi(i, { lat: parseFloat(e.target.value) })}
                    title="Latitude"
                    className="border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
                  />
                  <input
                    type="number"
                    step="0.0001"
                    value={poi.lon}
                    onChange={(e) => setPoi(i, { lon: parseFloat(e.target.value) })}
                    title="Longitude"
                    className="border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
                  />
                  <select
                    value={poi.tier}
                    onChange={(e) => setPoi(i, { tier: parseInt(e.target.value) })}
                    title="Importance tier"
                    className="border border-slate-300 rounded-lg px-1 py-1.5 text-sm"
                  >
                    <option value={1}>Hero</option>
                    <option value={2}>Major</option>
                    <option value={3}>Minor</option>
                  </select>
                  <button
                    onClick={() => setPois((prev) => prev.filter((_, j) => j !== i))}
                    className="text-slate-400 hover:text-red-600 text-sm"
                    title="Remove"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>

            {createError && <p className="text-sm text-red-600 mb-3">{createError}</p>}

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowCreate(false)}
                className="px-4 py-2 text-sm text-slate-600 hover:text-slate-900"
              >
                Cancel
              </button>
              <button
                onClick={submit}
                disabled={!name.trim() || createProject.isPending}
                className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StageChip({ label, done, state }: { label: string; done: boolean; state?: string }) {
  const color =
    state === 'running' || state === 'stale'
      ? 'bg-amber-100 text-amber-700'
      : state === 'error'
        ? 'bg-red-100 text-red-700'
        : done
          ? 'bg-emerald-100 text-emerald-700'
          : 'bg-slate-100 text-slate-500';
  return <span className={`text-xs px-2 py-0.5 rounded-full ${color}`}>{label}</span>;
}

export default ProjectListV2;
