import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { v2api, type V2JobState } from '@/api/v2';

/** V2 workflow: Plan (free) -> Assets (AI, cached) -> Compose (deterministic). */
function ProjectViewV2() {
  const { id = '' } = useParams();
  const queryClient = useQueryClient();
  const [useStub, setUseStub] = useState(false);
  const [harmonize, setHarmonize] = useState(false);
  const [scale, setScale] = useState(0.25);
  const [cacheBust, setCacheBust] = useState(0);

  const { data: project } = useQuery({
    queryKey: ['v2-project', id],
    queryFn: () => v2api.getProject(id),
    enabled: !!id,
  });

  const { data: status } = useQuery({
    queryKey: ['v2-status', id],
    queryFn: () => v2api.getStatus(id),
    enabled: !!id,
    refetchInterval: (query) => {
      const s = query.state.data;
      const running = s && Object.values(s).some((j) => j.state === 'running');
      return running ? 1500 : false;
    },
  });

  const anyRunning = status && Object.values(status).some((j) => j.state === 'running');

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['v2-status', id] });
    queryClient.invalidateQueries({ queryKey: ['v2-project', id] });
    queryClient.invalidateQueries({ queryKey: ['v2-assets', id] });
    setCacheBust(Date.now());
  };

  const startPlan = useMutation({ mutationFn: () => v2api.startPlan(id), onSettled: refresh });
  const startAssets = useMutation({
    mutationFn: (only_ids?: string[]) => v2api.startAssets(id, { stub: useStub, only_ids, force: !!only_ids }),
    onSettled: refresh,
  });
  const startCompose = useMutation({
    mutationFn: () => v2api.startCompose(id, { scale, harmonize }),
    onSettled: refresh,
  });

  const { data: assets } = useQuery({
    queryKey: ['v2-assets', id],
    queryFn: () => v2api.listAssets(id),
    enabled: !!id && !!project?.has_plan,
  });

  if (!project) return <div className="p-8 text-slate-500">Loading…</div>;

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link to="/v2" className="text-sm text-slate-500 hover:text-slate-800">
            &larr; V2 maps
          </Link>
          <h1 className="text-2xl font-bold text-slate-800">{project.name}</h1>
          <p className="text-sm text-slate-500">
            {project.poi_count} POIs · {project.config.output?.width_px}×
            {project.config.output?.height_px}px @ {project.config.output?.dpi} DPI
          </p>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3 mb-8">
        {/* Stage 1: Plan */}
        <StageCard
          title="1 · Plan"
          subtitle="Geometry, perspective, placement — free"
          job={status?.plan}
          action={
            <button
              onClick={() => startPlan.mutate()}
              disabled={anyRunning}
              className="px-3 py-1.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50"
            >
              {project.has_plan ? 'Re-plan' : 'Build plan'}
            </button>
          }
        />

        {/* Stage 2: Assets */}
        <StageCard
          title="2 · Assets"
          subtitle="AI illustrations, cached per asset"
          job={status?.assets}
          action={
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1 text-xs text-slate-500">
                <input type="checkbox" checked={useStub} onChange={(e) => setUseStub(e.target.checked)} />
                stub (free)
              </label>
              <button
                onClick={() => startAssets.mutate(undefined)}
                disabled={anyRunning || !project.has_plan}
                className="px-3 py-1.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                Generate
              </button>
            </div>
          }
        />

        {/* Stage 3: Compose */}
        <StageCard
          title="3 · Compose"
          subtitle="Deterministic render at print DPI"
          job={status?.compose}
          action={
            <div className="flex items-center gap-3">
              <select
                value={scale}
                onChange={(e) => setScale(parseFloat(e.target.value))}
                className="border border-slate-300 rounded-lg px-2 py-1 text-xs"
              >
                <option value={0.1}>10%</option>
                <option value={0.25}>25%</option>
                <option value={0.5}>50%</option>
                <option value={1}>100% (print)</option>
              </select>
              <label className="flex items-center gap-1 text-xs text-slate-500">
                <input
                  type="checkbox"
                  checked={harmonize}
                  onChange={(e) => setHarmonize(e.target.checked)}
                />
                harmonize
              </label>
              <button
                onClick={() => startCompose.mutate()}
                disabled={anyRunning || !project.has_plan}
                className="px-3 py-1.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                Render
              </button>
            </div>
          }
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Plan preview */}
        <section className="bg-white rounded-xl border border-slate-200 p-4">
          <h2 className="font-semibold text-slate-700 mb-3">Plan preview (free)</h2>
          {project.has_plan ? (
            <img
              src={`${v2api.previewUrl(id)}?t=${cacheBust}`}
              alt="Plan preview"
              className="w-full rounded-lg border border-slate-100"
            />
          ) : (
            <p className="text-sm text-slate-400">Run the plan stage to see the layout.</p>
          )}
        </section>

        {/* Poster */}
        <section className="bg-white rounded-xl border border-slate-200 p-4">
          <h2 className="font-semibold text-slate-700 mb-3">Poster</h2>
          {project.has_poster ? (
            <a href={v2api.posterUrl(id)} target="_blank" rel="noreferrer">
              <img
                src={`${v2api.posterUrl(id)}?t=${cacheBust}`}
                alt="Poster"
                className="w-full rounded-lg border border-slate-100"
              />
            </a>
          ) : (
            <p className="text-sm text-slate-400">Compose to render the poster.</p>
          )}
        </section>
      </div>

      {/* Asset gallery */}
      {assets && assets.length > 0 && (
        <section className="mt-8">
          <h2 className="font-semibold text-slate-700 mb-3">
            Assets ({assets.filter((a) => a.cached).length}/{assets.length} generated)
          </h2>
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-4 lg:grid-cols-6">
            {assets.map((asset) => (
              <div
                key={asset.id}
                className="bg-white rounded-lg border border-slate-200 p-2 flex flex-col"
              >
                <div className="aspect-square bg-slate-50 rounded flex items-center justify-center overflow-hidden">
                  {asset.cached ? (
                    <img
                      src={`${asset.url}?t=${cacheBust}`}
                      alt={asset.subject}
                      className="object-contain w-full h-full"
                    />
                  ) : (
                    <span className="text-xs text-slate-300">pending</span>
                  )}
                </div>
                <p className="text-xs text-slate-600 mt-1 truncate" title={asset.subject}>
                  {asset.subject}
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-400">{asset.kind}</span>
                  <button
                    onClick={() => startAssets.mutate([asset.id])}
                    disabled={anyRunning}
                    className="text-[10px] text-indigo-600 hover:text-indigo-800 disabled:opacity-40"
                    title="Regenerate just this asset"
                  >
                    regen
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function StageCard({
  title,
  subtitle,
  job,
  action,
}: {
  title: string;
  subtitle: string;
  job?: V2JobState;
  action: React.ReactNode;
}) {
  const state = job?.state ?? 'idle';
  const badge =
    state === 'running'
      ? 'bg-amber-100 text-amber-700'
      : state === 'done'
        ? 'bg-emerald-100 text-emerald-700'
        : state === 'error'
          ? 'bg-red-100 text-red-700'
          : 'bg-slate-100 text-slate-500';
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-1">
        <h3 className="font-semibold text-slate-800">{title}</h3>
        <span className={`text-xs px-2 py-0.5 rounded-full ${badge}`}>{state}</span>
      </div>
      <p className="text-xs text-slate-500 mb-3">{subtitle}</p>
      {state === 'running' && job && job.total > 0 && (
        <div className="mb-3">
          <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all"
              style={{ width: `${(job.current / job.total) * 100}%` }}
            />
          </div>
          <p className="text-[10px] text-slate-400 mt-1 truncate">
            {job.current}/{job.total} {job.detail}
          </p>
        </div>
      )}
      {state === 'error' && job?.detail && (
        <p className="text-xs text-red-600 mb-3 break-words">{job.detail}</p>
      )}
      {action}
    </div>
  );
}

export default ProjectViewV2;
