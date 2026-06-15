import { useEffect, useRef } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { v2api } from '@/api/v2';
import { useEditor, type EditorMode } from './editorStore';

/** Layout editor: interact with the source geography in normalized frame
 * space (left) and watch the warped result update live (right). Edits build
 * a CompositionSpec that the pipeline applies on the next plan run. */

const VBW = 1000;

const GROUND_FILL: Record<string, string> = {
  water: '#bcd6e6',
  park: '#cfe3c0',
  forest: '#bcd3ad',
  urban: '#e8ddc7',
  sand: '#efe6cf',
  farmland: '#e3e6c8',
  land: '#f0e9d8',
};

const MODES: { id: EditorMode; label: string; hint: string }[] = [
  { id: 'select', label: 'Select', hint: 'Click a road, river, POI or place to show/hide it.' },
  { id: 'warp', label: 'Warp', hint: 'Drag on the map to draw a magnify region; click one to edit.' },
  { id: 'poi', label: 'POIs', hint: 'Drag a POI to nudge it; click to select and resize.' },
];

function MapEditorV2() {
  const { id = '' } = useParams();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<null | {
    kind: 'region-new' | 'region-move' | 'poi-move';
    start: [number, number];
    target?: string;
    origin?: [number, number];
  }>(null);

  const project = useQuery({ queryKey: ['v2-project', id], queryFn: () => v2api.getProject(id), enabled: !!id });
  const composition = useQuery({ queryKey: ['v2-composition', id], queryFn: () => v2api.getComposition(id), enabled: !!id });
  const source = useQuery({ queryKey: ['v2-source', id], queryFn: () => v2api.getSourceGeojson(id), enabled: !!id });

  const ed = useEditor();

  useEffect(() => {
    if (composition.data && source.data && id) ed.init(id, composition.data, source.data);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [composition.data, source.data, id]);

  const out = project.data?.config.output;
  const aspect = out ? out.height_px / out.width_px : 1.414;
  const VBH = VBW * aspect;

  // viewport <-> normalized helpers
  const toNorm = (clientX: number, clientY: number): [number, number] => {
    const rect = svgRef.current!.getBoundingClientRect();
    return [(clientX - rect.left) / rect.width, (clientY - rect.top) / rect.height];
  };
  const px = (u: number, v: number): [number, number] => [u * VBW, v * VBH];

  if (source.isError || composition.isError) {
    return (
      <Shell id={id}>
        <div className="p-8 text-sm text-red-600">
          No source/plan yet. Run the <b>Plan</b> stage first, then open the editor.
        </div>
      </Shell>
    );
  }
  if (!ed.spec || !ed.source) {
    return (
      <Shell id={id}>
        <div className="p-8 text-sm text-gray-500">Loading layout…</div>
      </Shell>
    );
  }

  const spec = ed.spec;
  const src = ed.source;

  // --- pointer handling on the canvas ---
  const onPointerDown = (e: React.PointerEvent) => {
    if (ed.mode !== 'warp') return;
    const n = toNorm(e.clientX, e.clientY);
    drag.current = { kind: 'region-new', start: n };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const n = toNorm(e.clientX, e.clientY);
    if (d.kind === 'region-move' && d.target && d.origin) {
      const r = spec.warp.regions.find((x) => x.id === d.target);
      if (r) {
        const [w, h] = [r.bounds[2] - r.bounds[0], r.bounds[3] - r.bounds[1]];
        const cx = d.origin[0] + (n[0] - d.start[0]);
        const cy = d.origin[1] + (n[1] - d.start[1]);
        ed.updateRegion(d.target, { bounds: [cx, cy, cx + w, cy + h] });
      }
    } else if (d.kind === 'poi-move' && d.target) {
      ed.movePoi(d.target, [
        round(n[0] - d.start[0] + (d.origin?.[0] ?? 0)),
        round(n[1] - d.start[1] + (d.origin?.[1] ?? 0)),
      ]);
    }
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.kind === 'region-new') {
      const n = toNorm(e.clientX, e.clientY);
      const b: [number, number, number, number] = [
        Math.min(d.start[0], n[0]),
        Math.min(d.start[1], n[1]),
        Math.max(d.start[0], n[0]),
        Math.max(d.start[1], n[1]),
      ];
      if (b[2] - b[0] > 0.02 && b[3] - b[1] > 0.02) ed.addRegion(b);
    }
  };

  const selectedRegion = spec.warp.regions.find((r) => r.id === ed.selectedRegionId) ?? null;
  const selectedPoi = ed.selectedPoiId
    ? src.pois.find((p) => p.id === ed.selectedPoiId) ?? null
    : null;

  return (
    <Shell id={id}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 border-b bg-white px-4 py-2">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => ed.setMode(m.id)}
            className={`rounded px-3 py-1 text-sm ${
              ed.mode === m.id ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-700'
            }`}
          >
            {m.label}
          </button>
        ))}
        <span className="ml-2 text-xs text-gray-500">{MODES.find((m) => m.id === ed.mode)?.hint}</span>
        <div className="ml-auto flex items-center gap-3">
          {ed.previewing && <span className="text-xs text-gray-400">previewing…</span>}
          {ed.error && <span className="text-xs text-red-500">{ed.error}</span>}
          <span className={`text-xs ${ed.dirty ? 'text-amber-600' : 'text-gray-400'}`}>
            {ed.dirty ? 'unsaved' : 'saved'}
          </span>
          <button
            onClick={() => ed.save()}
            disabled={!ed.dirty || ed.saving}
            className="rounded bg-emerald-600 px-3 py-1 text-sm text-white disabled:opacity-40"
          >
            {ed.saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Interactive source canvas */}
        <div className="flex-1 overflow-auto bg-stone-100 p-4">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${VBW} ${VBH}`}
            className="mx-auto block max-h-full w-full max-w-3xl touch-none bg-[#f7f1e1] shadow"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          >
            {/* ground */}
            {src.ground.map((g) => (
              <polygon
                key={g.id}
                points={g.exterior.map(([u, v]) => px(u, v).join(',')).join(' ')}
                fill={GROUND_FILL[g.cls] ?? GROUND_FILL.land}
                opacity={ed.isVisible('roads', g.id) ? 0.6 : 0.6}
                stroke="none"
              />
            ))}
            {/* roads + rivers */}
            {src.roads.map((r) => {
              const isRiver = r.cls === 'river' || r.cls === 'stream';
              const layer = isRiver ? 'rivers' : 'roads';
              const visible = ed.isVisible(layer, r.id);
              return (
                <polyline
                  key={r.id}
                  points={r.points.map(([u, v]) => px(u, v).join(',')).join(' ')}
                  fill="none"
                  stroke={isRiver ? '#5b8fb0' : '#8a7f6b'}
                  strokeWidth={isRiver ? 3 : 2}
                  strokeOpacity={visible ? 0.9 : 0.18}
                  strokeDasharray={visible ? undefined : '4 4'}
                  className={ed.mode === 'select' ? 'cursor-pointer' : ''}
                  onClick={(e) => {
                    if (ed.mode !== 'select') return;
                    e.stopPropagation();
                    ed.toggleFeature(layer, r.id);
                  }}
                />
              );
            })}
            {/* places */}
            {src.places.map((p) => {
              const visible = ed.isVisible('places', p.id);
              const [x, y] = px(p.point[0], p.point[1]);
              return (
                <text
                  key={p.id}
                  x={x}
                  y={y}
                  fontSize={12}
                  textAnchor="middle"
                  fill={visible ? '#6b5b3e' : '#bbb'}
                  className={ed.mode === 'select' ? 'cursor-pointer' : ''}
                  onClick={(e) => {
                    if (ed.mode !== 'select') return;
                    e.stopPropagation();
                    ed.toggleFeature('places', p.id);
                  }}
                >
                  {p.name}
                </text>
              );
            })}
            {/* POIs */}
            {src.pois.map((p) => {
              const visible = ed.isVisible('pois', p.id);
              const ov = spec.pois[p.id];
              const off = ov?.offset_uv ?? [0, 0];
              const [x, y] = px(p.point[0] + off[0], p.point[1] + off[1]);
              const selected = ed.selectedPoiId === p.id;
              const size = ov?.size ?? 1;
              return (
                <g key={p.id}>
                  <circle
                    cx={x}
                    cy={y}
                    r={5 + 5 * size}
                    fill={visible ? (selected ? '#dc2626' : '#b45309') : '#ccc'}
                    fillOpacity={0.85}
                    stroke={selected ? '#dc2626' : '#7c4a02'}
                    strokeWidth={selected ? 2 : 1}
                    className={ed.mode === 'select' || ed.mode === 'poi' ? 'cursor-pointer' : ''}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (ed.mode === 'select') ed.toggleFeature('pois', p.id);
                      else if (ed.mode === 'poi') ed.selectPoi(p.id);
                    }}
                    onPointerDown={(e) => {
                      if (ed.mode !== 'poi') return;
                      e.stopPropagation();
                      ed.selectPoi(p.id);
                      drag.current = { kind: 'poi-move', target: p.id, start: toNorm(e.clientX, e.clientY), origin: off };
                      (e.target as Element).setPointerCapture?.(e.pointerId);
                    }}
                  />
                  <text x={x + 8} y={y + 4} fontSize={11} fill="#4b3f29">
                    {p.name}
                  </text>
                </g>
              );
            })}
            {/* warp regions */}
            {spec.warp.regions.map((r) => {
              const [x0, y0] = px(r.bounds[0], r.bounds[1]);
              const [x1, y1] = px(r.bounds[2], r.bounds[3]);
              const selected = ed.selectedRegionId === r.id;
              return (
                <rect
                  key={r.id}
                  x={Math.min(x0, x1)}
                  y={Math.min(y0, y1)}
                  width={Math.abs(x1 - x0)}
                  height={Math.abs(y1 - y0)}
                  fill="#6366f1"
                  fillOpacity={selected ? 0.18 : 0.1}
                  stroke="#6366f1"
                  strokeWidth={selected ? 2 : 1}
                  strokeDasharray="6 3"
                  className={ed.mode === 'warp' ? 'cursor-move' : ''}
                  onClick={(e) => {
                    if (ed.mode !== 'warp') return;
                    e.stopPropagation();
                    ed.selectRegion(r.id);
                  }}
                  onPointerDown={(e) => {
                    if (ed.mode !== 'warp') return;
                    e.stopPropagation();
                    ed.selectRegion(r.id);
                    drag.current = {
                      kind: 'region-move',
                      target: r.id,
                      start: toNorm(e.clientX, e.clientY),
                      origin: [Math.min(r.bounds[0], r.bounds[2]), Math.min(r.bounds[1], r.bounds[3])],
                    };
                    (e.target as Element).setPointerCapture?.(e.pointerId);
                  }}
                />
              );
            })}
          </svg>
        </div>

        {/* Right column: inspector + live preview */}
        <div className="flex w-80 flex-col border-l bg-white">
          <Inspector selectedRegion={selectedRegion} selectedPoi={selectedPoi} />
          <div className="flex-1 overflow-auto border-t p-2">
            <div className="mb-1 text-xs font-semibold text-gray-500">Live preview</div>
            {ed.preview ? (
              <div
                className="w-full [&>svg]:h-auto [&>svg]:w-full"
                dangerouslySetInnerHTML={{ __html: ed.preview.svg }}
              />
            ) : (
              <div className="text-xs text-gray-400">rendering…</div>
            )}
            {ed.preview?.warnings?.map((w, i) => (
              <div key={i} className="mt-1 text-[11px] text-amber-600">
                {w}
              </div>
            ))}
          </div>
        </div>
      </div>
    </Shell>
  );
}

function Inspector({
  selectedRegion,
  selectedPoi,
}: {
  selectedRegion: import('@/api/v2').WarpRegion | null;
  selectedPoi: import('@/api/v2').SourcePoiFeature | null;
}) {
  const ed = useEditor();
  if (ed.mode === 'warp' && selectedRegion) {
    const r = selectedRegion;
    return (
      <div className="space-y-2 p-3 text-sm">
        <div className="font-semibold">Warp region</div>
        <label className="block text-xs text-gray-600">
          Magnify ×{r.magnify.toFixed(1)}
          <input
            type="range"
            min={0.5}
            max={3}
            step={0.1}
            value={r.magnify}
            onChange={(e) => ed.updateRegion(r.id, { magnify: Number(e.target.value) })}
            className="w-full"
          />
        </label>
        <button onClick={() => ed.removeRegion(r.id)} className="rounded bg-red-50 px-2 py-1 text-xs text-red-600">
          Delete region
        </button>
      </div>
    );
  }
  if (ed.mode === 'poi' && selectedPoi) {
    const p = selectedPoi;
    const ov = ed.spec?.pois[p.id];
    return (
      <div className="space-y-2 p-3 text-sm">
        <div className="font-semibold">{p.name}</div>
        <div className="text-xs text-gray-500">size ×{(ov?.size ?? 1).toFixed(2)}</div>
        <div className="flex gap-2">
          <button onClick={() => ed.resizePoi(p.id, 1 / 1.2)} className="rounded bg-gray-100 px-2 py-1 text-xs">
            − smaller
          </button>
          <button onClick={() => ed.resizePoi(p.id, 1.2)} className="rounded bg-gray-100 px-2 py-1 text-xs">
            + bigger
          </button>
        </div>
        <button onClick={() => ed.cyclePoiLeader(p.id)} className="rounded bg-gray-100 px-2 py-1 text-xs">
          Leader: {ov?.leader ?? 'auto'}
        </button>
        {ov?.offset_uv && (
          <button
            onClick={() => ed.movePoi(p.id, [0, 0])}
            className="block rounded bg-gray-100 px-2 py-1 text-xs"
          >
            Reset position
          </button>
        )}
      </div>
    );
  }
  return <div className="p-3 text-xs text-gray-400">Nothing selected.</div>;
}

function Shell({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <div className="flex h-screen flex-col">
      <div className="flex items-center gap-3 border-b bg-white px-4 py-2">
        <Link to={`/v2/${id}`} className="text-sm text-indigo-600">
          ← Back to project
        </Link>
        <span className="text-sm font-semibold">Layout editor</span>
      </div>
      {children}
    </div>
  );
}

const round = (n: number) => Math.round(n * 1000) / 1000;

export default MapEditorV2;
