import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  v2api,
  type V2JobState,
  type V2Poi,
  type V2ProjectDetail,
  type V2RepaintDryRun,
} from '@/api/v2';

/** V2 workflow: Plan (free) -> Assets (AI, cached) -> Compose (deterministic)
 * -> optional Repaint (tiled AI texture pass). */
function ProjectViewV2() {
  const { id = '' } = useParams();
  const queryClient = useQueryClient();
  const [useStub, setUseStub] = useState(false);
  const [harmonize, setHarmonize] = useState(false);
  const [scale, setScale] = useState(0.25);
  // Seed from mount time so returning from the editor re-fetches preview.svg /
  // poster.png (which the apply/compose steps may have just rewritten).
  const [cacheBust, setCacheBust] = useState(() => Date.now());

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
  const [titleDraft, setTitleDraft] = useState<string | null>(null);
  const retitle = useMutation({
    mutationFn: (title: string) => v2api.retitle(id, title),
    onSettled: () => {
      setTitleDraft(null);
      refresh();
    },
  });
  const startAssets = useMutation({
    mutationFn: (opts?: { only_ids?: string[]; prompt_overrides?: Record<string, string> }) =>
      v2api.startAssets(id, {
        stub: useStub,
        only_ids: opts?.only_ids,
        force: !!opts?.only_ids,
        prompt_overrides: opts?.prompt_overrides,
      }),
    onSettled: refresh,
  });
  const [promptDrafts, setPromptDrafts] = useState<Record<string, string>>({});
  const [promptOpen, setPromptOpen] = useState<string | null>(null);
  const startCompose = useMutation({
    mutationFn: () => v2api.startCompose(id, { scale, harmonize }),
    onSettled: refresh,
  });
  const startLayered = useMutation({
    mutationFn: () => v2api.startLayered(id, { scale }),
    onSettled: refresh,
  });

  const [dryRun, setDryRun] = useState<V2RepaintDryRun | null>(null);
  const planRepaint = useMutation({
    mutationFn: () => v2api.startRepaint(id, { dry_run: true }),
    onSuccess: (data) => setDryRun(data as V2RepaintDryRun),
  });
  const startRepaint = useMutation({
    mutationFn: () => v2api.startRepaint(id, { scale, stub: useStub }),
    onSettled: refresh,
  });

  const { data: assets } = useQuery({
    queryKey: ['v2-assets', id],
    queryFn: () => v2api.listAssets(id),
    enabled: !!id && !!project?.has_plan,
  });

  const flagAsset = useMutation({
    mutationFn: ({ assetId, flagged }: { assetId: string; flagged: boolean }) =>
      v2api.flagAsset(id, assetId, flagged),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['v2-assets', id] }),
  });
  const flaggedIds = (assets ?? []).filter((a) => a.flagged).map((a) => a.id);

  if (!project) return <div className="p-8 text-slate-500">Loading…</div>;

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link to="/v2" className="text-sm text-slate-500 hover:text-slate-800">
            &larr; V2 maps
          </Link>
          {titleDraft === null ? (
            <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
              {project.config.title || project.name}
              <button
                title="Edit the poster title (no re-plan needed)"
                onClick={() => setTitleDraft(project.config.title || project.name)}
                className="text-slate-400 hover:text-slate-700 text-base"
              >
                ✎
              </button>
            </h1>
          ) : (
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (titleDraft.trim()) retitle.mutate(titleDraft.trim());
              }}
            >
              <input
                autoFocus
                value={titleDraft}
                onChange={(e) => setTitleDraft(e.target.value)}
                className="text-2xl font-bold text-slate-800 border-b-2 border-indigo-400 bg-transparent focus:outline-none"
              />
              <button
                type="submit"
                disabled={!titleDraft.trim() || retitle.isPending}
                className="px-2 py-1 bg-indigo-600 text-white text-sm rounded-lg disabled:opacity-50"
              >
                Save
              </button>
              <button
                type="button"
                onClick={() => setTitleDraft(null)}
                className="px-2 py-1 text-sm text-slate-500"
              >
                Cancel
              </button>
            </form>
          )}
          {retitle.data?.plan_patched && (
            <p className="text-xs text-amber-600">
              Title updated in the plan — re-run Compose to see it on the poster.
            </p>
          )}
          <p className="text-sm text-slate-500">
            {project.poi_count} POIs · {project.config.output?.width_px}×
            {project.config.output?.height_px}px @ {project.config.output?.dpi} DPI
          </p>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-4 mb-8">
        {/* Stage 1: Plan */}
        <StageCard
          title="1 · Plan"
          subtitle="Geometry, perspective, placement — free"
          note={
            project.plan_stale
              ? 'Project changed since this plan was built — re-plan to apply.'
              : undefined
          }
          job={status?.plan}
          action={
            <div className="flex items-center gap-2">
              {project.has_plan && (
                <Link
                  to={`/v2/${id}/edit`}
                  className="px-3 py-1.5 bg-slate-100 text-slate-700 text-sm rounded-lg hover:bg-slate-200"
                >
                  Edit layout
                </Link>
              )}
              {project.has_plan && (
                <Link
                  to={`/v2/${id}/edit-gl`}
                  className="px-3 py-1.5 bg-violet-100 text-violet-700 text-sm rounded-lg hover:bg-violet-200"
                  title="WebGL editor: live 2.5D poster with WYSIWYG label placement"
                >
                  WebGL editor
                </Link>
              )}
              <button
                onClick={() => startPlan.mutate()}
                disabled={anyRunning}
                className="px-3 py-1.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                {project.has_plan ? 'Re-plan' : 'Build plan'}
              </button>
            </div>
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
          subtitle={
            project.poster_stale
              ? '⚠ Poster is out of date with the plan — re-compose to apply layout edits'
              : 'Deterministic render at print DPI'
          }
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

        {/* Stage 4: Repaint (optional) */}
        <StageCard
          title="4 · Repaint"
          subtitle="Hand-painted texture pass (1 AI call) — optional"
          note={
            dryRun
              ? `${dryRun.calls_planned} calls planned (~$${dryRun.estimated_cost_usd})`
              : undefined
          }
          job={status?.repaint}
          action={
            <div className="flex items-center gap-3">
              <button
                onClick={() => planRepaint.mutate()}
                disabled={anyRunning || !project.has_plan}
                className="px-2 py-1.5 text-xs text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50 disabled:opacity-50"
                title="Plan only: shows call count and cost, spends nothing"
              >
                Dry run
              </button>
              <button
                onClick={() => startRepaint.mutate()}
                disabled={anyRunning || !project.has_plan}
                className="px-3 py-1.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                Repaint
              </button>
            </div>
          }
        />
      </div>

      <PoiEditor project={project} disabled={!!anyRunning} onSaved={refresh} />

      <RepaintQuadrants projectId={id} disabled={!!anyRunning} repaintJob={status?.repaint} />

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

          {/* Layered export: a Photoshop-editable PSD of the same scene. */}
          <div className="mt-4 border-t border-slate-100 pt-3">
            <div className="flex items-center gap-3 flex-wrap">
              <button
                onClick={() => startLayered.mutate()}
                disabled={anyRunning || !project.has_plan}
                className="px-3 py-1.5 bg-slate-700 text-white text-sm rounded-lg hover:bg-slate-800 disabled:opacity-50"
                title="Render a layered PSD: each sprite and label on its own editable layer"
              >
                Export layered PSD ({Math.round(scale * 100)}%)
              </button>
              {status?.layered?.state === 'running' && (
                <span className="text-xs text-amber-600">exporting…</span>
              )}
              {status?.layered?.state === 'error' && (
                <span className="text-xs text-red-600 break-words">
                  {status.layered.detail || 'export failed'}
                </span>
              )}
              {project.has_layered && status?.layered?.state !== 'running' && (
                <a
                  href={`${v2api.posterPsdUrl(id)}?t=${cacheBust}`}
                  className="text-sm text-indigo-600 hover:text-indigo-800 underline"
                >
                  Download .psd{project.layered_stale ? ' (out of date)' : ''}
                </a>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Sprites and text labels each become their own layer (the haze too); ground
              textures, roads and buildings are flattened into a base. Renders at the
              Compose scale above.
            </p>
          </div>
        </section>
      </div>

      {/* Asset gallery */}
      {assets && assets.length > 0 && (
        <section className="mt-8">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-slate-700">
              Assets ({assets.filter((a) => a.cached).length}/{assets.length} generated)
            </h2>
            {flaggedIds.length > 0 && (
              <button
                onClick={() => startAssets.mutate({ only_ids: flaggedIds })}
                disabled={anyRunning}
                className="px-3 py-1.5 bg-amber-600 text-white text-xs rounded-lg hover:bg-amber-700 disabled:opacity-50"
              >
                Regenerate flagged ({flaggedIds.length})
              </button>
            )}
          </div>
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-4 lg:grid-cols-6">
            {assets.map((asset) => (
              <div
                key={asset.id}
                className={`bg-white rounded-lg border p-2 flex flex-col ${
                  asset.flagged
                    ? 'border-amber-400'
                    : asset.palette_outlier
                      ? 'border-orange-300'
                      : 'border-slate-200'
                }`}
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
                  {asset.palette_outlier && (
                    <span
                      className="ml-1 text-orange-500"
                      title={`Palette outlier (ΔE ${asset.palette_score ?? '?'}) — may not match the map style`}
                    >
                      ⚠
                    </span>
                  )}
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-400">{asset.kind}</span>
                  <div className="flex items-center gap-2">
                    {asset.cached && (
                      <button
                        onClick={() =>
                          flagAsset.mutate({ assetId: asset.id, flagged: !asset.flagged })
                        }
                        disabled={anyRunning}
                        className={`text-[10px] disabled:opacity-40 ${
                          asset.flagged
                            ? 'text-amber-600 hover:text-amber-800'
                            : 'text-slate-400 hover:text-amber-600'
                        }`}
                        title={asset.flagged ? 'Clear flag' : 'Flag for regeneration'}
                      >
                        {asset.flagged ? '⚑ flagged' : '⚐ flag'}
                      </button>
                    )}
                    <button
                      onClick={() => setPromptOpen(promptOpen === asset.id ? null : asset.id)}
                      className={`text-[10px] hover:text-slate-700 ${
                        asset.prompt_overridden ? 'text-indigo-600' : 'text-slate-400'
                      }`}
                      title="Edit this asset's prompt"
                    >
                      {asset.prompt_overridden ? '✎ prompt*' : '✎ prompt'}
                    </button>
                    <button
                      onClick={() => startAssets.mutate({ only_ids: [asset.id] })}
                      disabled={anyRunning}
                      className="text-[10px] text-indigo-600 hover:text-indigo-800 disabled:opacity-40"
                      title="Regenerate just this asset"
                    >
                      regen
                    </button>
                  </div>
                </div>
                {promptOpen === asset.id && (
                  <div className="mt-2 border-t pt-2">
                    <textarea
                      value={promptDrafts[asset.id] ?? asset.prompt_hints}
                      onChange={(e) =>
                        setPromptDrafts((d) => ({ ...d, [asset.id]: e.target.value }))
                      }
                      rows={3}
                      className="w-full rounded border border-slate-200 p-1 text-[10px]"
                      placeholder="Describe this sprite/texture…"
                    />
                    <button
                      onClick={() =>
                        startAssets.mutate({
                          only_ids: [asset.id],
                          prompt_overrides: {
                            [asset.id]: promptDrafts[asset.id] ?? asset.prompt_hints,
                          },
                        })
                      }
                      disabled={anyRunning}
                      className="mt-1 w-full rounded bg-indigo-600 px-2 py-1 text-[10px] text-white disabled:opacity-40"
                    >
                      Regenerate with this prompt
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

/** Review the repaint quadrant grid: click a painted cell to flag it for
 * redo on the next repaint run (it repaints with context from its painted
 * neighbors), click again to restore. */
function RepaintQuadrants({
  projectId,
  disabled,
  repaintJob,
}: {
  projectId: string;
  disabled: boolean;
  repaintJob?: V2JobState;
}) {
  const queryClient = useQueryClient();
  const { data } = useQuery({
    queryKey: ['v2-repaint-quadrants', projectId],
    queryFn: () => v2api.repaintQuadrants(projectId),
    enabled: !!projectId && repaintJob?.state !== 'running',
  });

  const flag = useMutation({
    mutationFn: ({ x, y, flagged }: { x: number; y: number; flagged: boolean }) =>
      v2api.flagRepaintQuadrant(projectId, x, y, flagged),
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: ['v2-repaint-quadrants', projectId] }),
  });

  if (!data?.grid || data.quadrants.length === 0) return null;
  const { cols, rows } = data.grid;
  const byCell = new Map(data.quadrants.map((q) => [`${q.x},${q.y}`, q.status]));
  const flaggedCount = data.quadrants.filter((q) => q.status === 'flagged').length;

  const color: Record<string, string> = {
    generated: 'bg-emerald-300 hover:bg-amber-300 cursor-pointer',
    flagged: 'bg-amber-500 hover:bg-emerald-300 cursor-pointer',
    skipped: 'bg-slate-200',
    pending: 'bg-slate-100 border border-slate-200',
  };

  return (
    <section className="bg-white rounded-xl border border-slate-200 p-4 mb-8">
      <div className="flex items-center justify-between mb-1">
        <h2 className="font-semibold text-slate-700">Repaint quadrants</h2>
        <span className="text-xs text-slate-500">
          {data.calls_made ?? 0} calls made
          {flaggedCount > 0 && ` · ${flaggedCount} flagged — run Repaint to redo`}
        </span>
      </div>
      <p className="text-xs text-slate-500 mb-3">
        Tiled-mode runs only. Click a painted cell to flag it for redo; flagged cells repaint
        on the next tiled run.
      </p>
      <div
        className="grid gap-px w-fit"
        style={{ gridTemplateColumns: `repeat(${cols}, 18px)` }}
      >
        {Array.from({ length: rows }, (_, y) =>
          Array.from({ length: cols }, (_, x) => {
            const status = byCell.get(`${x},${y}`) ?? 'pending';
            const clickable =
              !disabled && (status === 'generated' || status === 'flagged');
            return (
              <div
                key={`${x},${y}`}
                onClick={() =>
                  clickable && flag.mutate({ x, y, flagged: status === 'generated' })
                }
                className={`h-[18px] rounded-sm ${color[status]} ${
                  disabled ? 'pointer-events-none' : ''
                }`}
                title={`(${x},${y}) ${status}`}
              />
            );
          })
        )}
      </div>
    </section>
  );
}

/** Edit the project's POIs after creation; saving marks the plan stale. */
function PoiEditor({
  project,
  disabled,
  onSaved,
}: {
  project: V2ProjectDetail;
  disabled: boolean;
  onSaved: () => void;
}) {
  // null = not editing; the saved config is the source of truth until then.
  const [draft, setDraft] = useState<V2Poi[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [geocoding, setGeocoding] = useState<number | null>(null);
  const pois = draft ?? project.config.pois;
  const region = project.config.region;

  const save = useMutation({
    mutationFn: (next: V2Poi[]) =>
      v2api.updateProject(project.id, { ...project.config, pois: next }),
    onSuccess: () => {
      setDraft(null);
      setError(null);
      onSaved();
    },
    onError: (e: Error) => setError(e.message),
  });

  const edit = (i: number, patch: Partial<V2Poi>) =>
    setDraft(pois.map((p, j) => (j === i ? { ...p, ...patch } : p)));
  const lookup = async (i: number, name: string) => {
    if (!name.trim()) return;
    setGeocoding(i);
    setError(null);
    try {
      const r = await v2api.geocode(name);
      const round4 = (n: number) => Math.round(n * 1e4) / 1e4;
      edit(i, { lat: round4(r.lat), lon: round4(r.lon), feature_type: r.feature_type });
    } catch (e) {
      setError(`Couldn't locate "${name}": ${(e as Error).message}`);
    } finally {
      setGeocoding(null);
    }
  };
  const add = () =>
    setDraft([
      ...pois,
      { name: '', lat: (region.north + region.south) / 2, lon: (region.east + region.west) / 2, tier: 2 },
    ]);
  const remove = (i: number) => setDraft(pois.filter((_, j) => j !== i));

  const valid = pois.every(
    (p) =>
      p.name.trim() &&
      Number.isFinite(p.lat) &&
      Number.isFinite(p.lon) &&
      p.lat <= region.north &&
      p.lat >= region.south &&
      p.lon <= region.east &&
      p.lon >= region.west
  );

  return (
    <section className="bg-white rounded-xl border border-slate-200 p-4 mb-8">
      <div className="flex items-center justify-between mb-1">
        <h2 className="font-semibold text-slate-700">Points of interest</h2>
        <button onClick={add} disabled={disabled} className="text-xs text-indigo-600 hover:text-indigo-800 disabled:opacity-40">
          + Add POI
        </button>
      </div>
      <p className="text-xs text-slate-500 mb-3">
        Each POI becomes an illustrated, labeled landmark. After saving, re-run Plan (and
        generate any new assets) to apply.
      </p>
      <div className="space-y-2 mb-3">
        {pois.map((poi, i) => (
          <div key={i} className="grid grid-cols-[1fr_110px_110px_80px_24px] gap-2 items-center">
            <div className="flex gap-1">
              <input
                value={poi.name}
                onChange={(e) => edit(i, { name: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') lookup(i, poi.name);
                }}
                placeholder="Name"
                disabled={disabled}
                className="flex-1 border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
              />
              <button
                type="button"
                onClick={() => lookup(i, poi.name)}
                disabled={disabled || geocoding === i || !poi.name.trim()}
                title="Look up coordinates by name"
                className="px-2 rounded-lg border border-slate-300 text-sm hover:bg-slate-50 disabled:opacity-40"
              >
                {geocoding === i ? '…' : '🔍'}
              </button>
            </div>
            <input
              type="number"
              step="0.0001"
              value={poi.lat}
              onChange={(e) => edit(i, { lat: parseFloat(e.target.value) })}
              title={`Latitude (${region.south} – ${region.north})`}
              disabled={disabled}
              className="border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
            />
            <input
              type="number"
              step="0.0001"
              value={poi.lon}
              onChange={(e) => edit(i, { lon: parseFloat(e.target.value) })}
              title={`Longitude (${region.west} – ${region.east})`}
              disabled={disabled}
              className="border border-slate-300 rounded-lg px-2 py-1.5 text-sm"
            />
            <select
              value={poi.tier}
              onChange={(e) => edit(i, { tier: parseInt(e.target.value) })}
              title="Importance tier"
              disabled={disabled}
              className="border border-slate-300 rounded-lg px-1 py-1.5 text-sm"
            >
              <option value={1}>Hero</option>
              <option value={2}>Major</option>
              <option value={3}>Minor</option>
            </select>
            <button
              onClick={() => remove(i)}
              disabled={disabled}
              className="text-slate-400 hover:text-red-600 text-sm disabled:opacity-40"
              title="Remove"
            >
              ×
            </button>
          </div>
        ))}
        {pois.length === 0 && <p className="text-sm text-slate-400">No POIs — add at least one.</p>}
      </div>
      {!valid && draft && (
        <p className="text-xs text-amber-600 mb-2">
          Every POI needs a name and coordinates inside the region.
        </p>
      )}
      {error && <p className="text-xs text-red-600 mb-2">{error}</p>}
      {draft && (
        <div className="flex gap-3">
          <button
            onClick={() => save.mutate(pois.filter((p) => p.name.trim()))}
            disabled={disabled || !valid || save.isPending}
            className="px-3 py-1.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50"
          >
            {save.isPending ? 'Saving…' : 'Save POIs'}
          </button>
          <button
            onClick={() => {
              setDraft(null);
              setError(null);
            }}
            className="px-3 py-1.5 text-sm text-slate-600 hover:text-slate-900"
          >
            Discard changes
          </button>
        </div>
      )}
    </section>
  );
}

function StageCard({
  title,
  subtitle,
  note,
  job,
  action,
}: {
  title: string;
  subtitle: string;
  note?: string;
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
      {note && <p className="text-xs text-amber-600 mb-3">{note}</p>}
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
