import { useEffect, useRef, useState } from 'react';
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
  { id: 'roads', label: 'Roads', hint: 'Click a road to route it: warped / straight / hidden, or reshape its path.' },
  { id: 'labels', label: 'Labels', hint: 'Drag a label to place it by hand; double-click to reset.' },
];

function MapEditorV2() {
  const { id = '' } = useParams();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const gRef = useRef<SVGGElement | null>(null);
  // Pan/zoom lives in a ref + imperative transform on the content group, so
  // panning never re-renders the (thousands of) feature elements.
  const view = useRef({ scale: 1, tx: 0, ty: 0 });
  const drag = useRef<null | {
    kind:
      | 'region-new'
      | 'region-move'
      | 'region-resize'
      | 'poi-move'
      | 'reshape-vertex'
      | 'label-move'
      | 'pan';
    start: [number, number];
    target?: string;
    origin?: [number, number];
    index?: number;
    corner?: import('./editorStore').RegionCorner;
  }>(null);

  // Dots/labels counter-scale by 1/zoom so they stay a constant screen size
  // (zooming in then de-clutters by spreading positions, not enlarging marks).
  const [markerScale, setMarkerScale] = useState(1);
  const [previewExpanded, setPreviewExpanded] = useState(false);

  const applyTransform = () => {
    const { scale, tx, ty } = view.current;
    gRef.current?.setAttribute('transform', `translate(${tx} ${ty}) scale(${scale})`);
  };
  const resetView = () => {
    view.current = { scale: 1, tx: 0, ty: 0 };
    applyTransform();
    setMarkerScale(1);
  };

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

  // viewport <-> normalized helpers (account for pan/zoom transform)
  const toVB = (clientX: number, clientY: number): [number, number] => {
    const rect = svgRef.current!.getBoundingClientRect();
    return [((clientX - rect.left) / rect.width) * VBW, ((clientY - rect.top) / rect.height) * VBH];
  };
  const toNorm = (clientX: number, clientY: number): [number, number] => {
    const [vbx, vby] = toVB(clientX, clientY);
    const { scale, tx, ty } = view.current;
    return [(vbx - tx) / scale / VBW, (vby - ty) / scale / VBH];
  };
  const px = (u: number, v: number): [number, number] => [u * VBW, v * VBH];

  // Native wheel listener (passive:false so we can preventDefault) -- attaches
  // once the canvas mounts (ready) and zooms toward the cursor.
  const ready = !!ed.spec && !!ed.source;
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const vbx = ((e.clientX - rect.left) / rect.width) * VBW;
      const vby = ((e.clientY - rect.top) / rect.height) * VBH;
      const v = view.current;
      const ns = Math.max(1, Math.min(24, v.scale * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      const real = ns / v.scale;
      v.tx = vbx - (vbx - v.tx) * real;
      v.ty = vby - (vby - v.ty) * real;
      v.scale = ns;
      applyTransform();
      setMarkerScale(1 / ns);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, VBH]);

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
    if (ed.mode === 'warp') {
      drag.current = { kind: 'region-new', start: toNorm(e.clientX, e.clientY) };
      (e.target as Element).setPointerCapture?.(e.pointerId);
      return;
    }
    // Background drag in non-warp modes pans the canvas.
    drag.current = {
      kind: 'pan',
      start: toVB(e.clientX, e.clientY),
      origin: [view.current.tx, view.current.ty],
    };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === 'pan' && d.origin) {
      const vb = toVB(e.clientX, e.clientY);
      view.current.tx = d.origin[0] + (vb[0] - d.start[0]);
      view.current.ty = d.origin[1] + (vb[1] - d.start[1]);
      applyTransform();
      return;
    }
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
    } else if (d.kind === 'reshape-vertex' && d.target && d.index != null) {
      ed.updateReshapeVertex(d.target, d.index, [round(n[0]), round(n[1])]);
    } else if (d.kind === 'region-resize' && d.target && d.corner) {
      ed.resizeRegionCorner(d.target, d.corner, [round(n[0]), round(n[1])]);
    } else if (d.kind === 'label-move' && d.target) {
      ed.moveLabel(d.target, [round(n[0]), round(n[1])]);
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
  const selectedRoad = ed.selectedRoadId
    ? src.roads.find((r) => r.id === ed.selectedRoadId) ?? null
    : null;
  // Editable labels (POIs + cities/water), keyed by source-feature id. Road
  // labels are omitted for now (only a handful of roads get one).
  const labelMarkers =
    ed.mode === 'labels'
      ? [
          ...src.pois.map((p) => ({ key: p.id, text: p.name, nat: p.point })),
          ...src.places.map((p) => ({ key: p.id, text: p.name, nat: p.point })),
        ]
      : [];
  // Which roads actually render (from the live preview): the editor matches it.
  const renderedRoads = new Set(ed.preview?.road_ids ?? []);
  const hasPreview = !!ed.preview;

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
        {src.counts && src.counts.roads_shown < src.counts.roads_total && (
          <span className="text-[11px] text-gray-400">
            (showing {src.counts.roads_shown} editable of {src.counts.roads_total} roads; washes/minor
            roads hidden)
          </span>
        )}
        <div className="ml-auto flex items-center gap-3">
          <span className="text-[11px] text-gray-400">scroll = zoom · drag = pan</span>
          <button onClick={resetView} className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-700">
            Reset view
          </button>
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
            <g
              ref={gRef}
              transform={`translate(${view.current.tx} ${view.current.ty}) scale(${view.current.scale})`}
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
              const ov = spec.roads[r.id];
              const treatment = ov?.treatment ?? 'warped';
              const selected = ed.mode === 'roads' && ed.selectedRoadId === r.id;
              // Draw the user's reshaped path if present, else the source path.
              const geom = ov?.reshape ?? r.points;
              const pts = geom.map(([u, v]) => px(u, v).join(',')).join(' ');
              const hittable = ed.mode === 'select' || ed.mode === 'roads';
              // Match the preview: once we have one, a road is solid iff it is
              // actually rendered there; otherwise it is a faint "available" ghost.
              const dim = hasPreview ? !renderedRoads.has(r.id) : !visible || treatment === 'hidden';
              return (
                <g key={r.id}>
                  <polyline
                    points={pts}
                    fill="none"
                    stroke={selected ? '#dc2626' : isRiver ? '#5b8fb0' : '#8a7f6b'}
                    strokeWidth={selected ? 4 : isRiver ? 3 : 2}
                    strokeOpacity={dim ? 0.15 : 0.9}
                    strokeDasharray={dim ? '4 4' : ov?.reshape ? '8 4' : undefined}
                    pointerEvents="none"
                  />
                  {hittable && (
                    <polyline
                      points={pts}
                      fill="none"
                      stroke="transparent"
                      strokeWidth={14}
                      pointerEvents="stroke"
                      className="cursor-pointer"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (ed.mode === 'select') ed.toggleFeature(layer, r.id);
                        else ed.selectRoad(r.id);
                      }}
                    />
                  )}
                  {/* reshape vertex handles */}
                  {selected &&
                    ov?.reshape?.map((uv, i) => {
                      const [hx, hy] = px(uv[0], uv[1]);
                      return (
                        <circle
                          key={i}
                          cx={hx}
                          cy={hy}
                          r={7 * markerScale}
                          fill="#fff"
                          stroke="#dc2626"
                          strokeWidth={2 * markerScale}
                          className="cursor-grab"
                          onPointerDown={(e) => {
                            e.stopPropagation();
                            drag.current = {
                              kind: 'reshape-vertex',
                              target: r.id,
                              index: i,
                              start: toNorm(e.clientX, e.clientY),
                            };
                            (e.target as Element).setPointerCapture?.(e.pointerId);
                          }}
                        />
                      );
                    })}
                </g>
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
                  fontSize={12 * markerScale}
                  textAnchor="middle"
                  fill={visible ? '#6b5b3e' : '#bbb'}
                  pointerEvents={ed.mode === 'select' ? 'auto' : 'none'}
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
                    r={(5 + 5 * size) * markerScale}
                    fill={visible ? (selected ? '#dc2626' : '#b45309') : '#ccc'}
                    fillOpacity={0.85}
                    stroke={selected ? '#dc2626' : '#7c4a02'}
                    strokeWidth={(selected ? 2 : 1) * markerScale}
                    pointerEvents={ed.mode === 'select' || ed.mode === 'poi' ? 'auto' : 'none'}
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
                  <text
                    x={x + 8 * markerScale}
                    y={y + 4 * markerScale}
                    fontSize={11 * markerScale}
                    fill="#4b3f29"
                    pointerEvents="none"
                  >
                    {p.name}
                  </text>
                </g>
              );
            })}
            {/* warp regions */}
            {spec.warp.regions.map((r) => {
              const [x0, y0] = px(r.bounds[0], r.bounds[1]);
              const [x1, y1] = px(r.bounds[2], r.bounds[3]);
              const selected = ed.mode === 'warp' && ed.selectedRegionId === r.id;
              const corners: [import('./editorStore').RegionCorner, number, number][] = [
                ['nw', r.bounds[0], r.bounds[1]],
                ['ne', r.bounds[2], r.bounds[1]],
                ['sw', r.bounds[0], r.bounds[3]],
                ['se', r.bounds[2], r.bounds[3]],
              ];
              return (
                <g key={r.id}>
                  <rect
                    x={Math.min(x0, x1)}
                    y={Math.min(y0, y1)}
                    width={Math.abs(x1 - x0)}
                    height={Math.abs(y1 - y0)}
                    fill="#6366f1"
                    fillOpacity={selected ? 0.18 : 0.1}
                    stroke="#6366f1"
                    strokeWidth={selected ? 2 : 1}
                    strokeDasharray="6 3"
                    // Only interactive in warp mode -- otherwise the box would
                    // swallow clicks meant for the POIs / roads underneath it.
                    pointerEvents={ed.mode === 'warp' ? 'auto' : 'none'}
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
                  {selected &&
                    corners.map(([corner, cu, cv]) => {
                      const [hx, hy] = px(cu, cv);
                      return (
                        <rect
                          key={corner}
                          x={hx - 6 * markerScale}
                          y={hy - 6 * markerScale}
                          width={12 * markerScale}
                          height={12 * markerScale}
                          fill="#fff"
                          stroke="#6366f1"
                          strokeWidth={2 * markerScale}
                          style={{ cursor: 'nwse-resize' }}
                          onPointerDown={(e) => {
                            e.stopPropagation();
                            drag.current = {
                              kind: 'region-resize',
                              target: r.id,
                              corner,
                              start: toNorm(e.clientX, e.clientY),
                            };
                            (e.target as Element).setPointerCapture?.(e.pointerId);
                          }}
                        />
                      );
                    })}
                </g>
              );
            })}
            {/* editable labels */}
            {labelMarkers.map((m) => {
              const ov = spec.labels.overrides[m.key];
              const [lx, ly] = px((ov ?? m.nat)[0], (ov ?? m.nat)[1]);
              return (
                <text
                  key={`lbl-${m.key}`}
                  x={lx}
                  y={ly}
                  fontSize={13 * markerScale}
                  textAnchor="middle"
                  fill={ov ? '#dc2626' : '#1f2937'}
                  stroke="#fff"
                  strokeWidth={3 * markerScale}
                  paintOrder="stroke"
                  style={{ cursor: 'grab', userSelect: 'none' }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    drag.current = { kind: 'label-move', target: m.key, start: toNorm(e.clientX, e.clientY) };
                    (e.target as Element).setPointerCapture?.(e.pointerId);
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    ed.resetLabel(m.key);
                  }}
                >
                  {m.text}
                </text>
              );
            })}
            </g>
          </svg>
        </div>

        {/* Right column: inspector + live preview */}
        <div className="flex w-96 flex-col border-l bg-white">
          <Inspector
            selectedRegion={selectedRegion}
            selectedPoi={selectedPoi}
            selectedRoad={selectedRoad}
          />
          <div className="flex-1 overflow-auto border-t p-2">
            <div className="mb-1 flex items-center gap-2">
              <span className="text-xs font-semibold text-gray-500">Live preview</span>
              <span
                className={`inline-block h-2.5 w-2.5 rounded-full ${
                  ed.previewing
                    ? 'animate-pulse bg-amber-400'
                    : ed.preview
                      ? 'bg-emerald-500'
                      : 'bg-gray-300'
                }`}
                title={ed.previewing ? 'updating…' : ed.preview ? 'up to date' : 'not rendered yet'}
              />
              <span className="text-[11px] text-gray-400">
                {ed.previewing ? 'updating…' : ed.preview ? 'up to date' : ''}
              </span>
              {ed.preview && (
                <button
                  onClick={() => setPreviewExpanded(true)}
                  className="ml-auto rounded bg-gray-100 px-2 py-0.5 text-[11px] text-gray-700"
                >
                  ⤢ Expand
                </button>
              )}
            </div>
            {ed.preview ? (
              <div
                className="w-full cursor-zoom-in [&>svg]:h-auto [&>svg]:w-full"
                onClick={() => setPreviewExpanded(true)}
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

      {/* Expanded preview overlay */}
      {previewExpanded && ed.preview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
          onClick={() => setPreviewExpanded(false)}
        >
          <div
            className="relative max-h-full overflow-auto rounded bg-white p-2 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setPreviewExpanded(false)}
              className="absolute right-3 top-3 z-10 rounded bg-gray-800/80 px-2 py-1 text-xs text-white"
            >
              Close ✕
            </button>
            <div
              className="[&>svg]:h-auto [&>svg]:max-h-[85vh] [&>svg]:w-auto"
              dangerouslySetInnerHTML={{ __html: ed.preview.svg }}
            />
          </div>
        </div>
      )}
    </Shell>
  );
}

function Inspector({
  selectedRegion,
  selectedPoi,
  selectedRoad,
}: {
  selectedRegion: import('@/api/v2').WarpRegion | null;
  selectedPoi: import('@/api/v2').SourcePoiFeature | null;
  selectedRoad: import('@/api/v2').SourceRoadFeature | null;
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
            max={4}
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
  if (ed.mode === 'roads') {
    if (!selectedRoad) return <div className="p-3 text-xs text-gray-400">Click a road to route it.</div>;
    const r = selectedRoad;
    const ov = ed.spec?.roads[r.id];
    const treatment = ov?.treatment ?? 'warped';
    const label = r.ref || r.name || r.cls;
    const treatments: import('./editorStore').RoadTreatment[] = ['warped', 'straight', 'hidden'];
    return (
      <div className="space-y-3 p-3 text-sm">
        <div className="font-semibold">{label}</div>
        <div>
          <div className="mb-1 text-xs text-gray-500">Routing</div>
          <div className="flex gap-1">
            {treatments.map((t) => (
              <button
                key={t}
                onClick={() => ed.setRoadTreatment(r.id, t)}
                className={`rounded px-2 py-1 text-xs ${
                  treatment === t ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-700'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <p className="mt-1 text-[11px] text-gray-400">
            warped follows the cartogram (can bend); straight ignores the warp; hidden drops it.
          </p>
        </div>
        <div>
          <div className="mb-1 text-xs text-gray-500">Reshape (draw its path by hand)</div>
          {ov?.reshape ? (
            <button onClick={() => ed.clearReshape(r.id)} className="rounded bg-gray-100 px-2 py-1 text-xs">
              Reset path · drag the red handles
            </button>
          ) : (
            <button
              onClick={() => ed.beginReshape(r.id, decimateForReshape(r.points))}
              className="rounded bg-amber-100 px-2 py-1 text-xs text-amber-800"
            >
              Reshape path
            </button>
          )}
          <p className="mt-1 text-[11px] text-gray-400">
            e.g. bow I-10 away from the airport to open a gap.
          </p>
        </div>
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
  if (ed.mode === 'labels') {
    const n = Object.keys(ed.spec?.labels.overrides ?? {}).length;
    return (
      <div className="space-y-2 p-3 text-sm">
        <div className="font-semibold">Labels</div>
        <p className="text-xs text-gray-500">
          Drag a POI or city label to place it by hand. Double-click a moved (red) label to reset
          it. Watch the preview for the warped result.
        </p>
        <div className="text-xs text-gray-400">{n} hand-placed</div>
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

/** A coarse, evenly-spaced handle set to start a reshape from (endpoints kept). */
function decimateForReshape(points: [number, number][], n = 7): [number, number][] {
  if (points.length <= n) return points.map((p) => [p[0], p[1]] as [number, number]);
  const step = (points.length - 1) / (n - 1);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const p = points[Math.round(i * step)];
    out.push([p[0], p[1]]);
  }
  return out;
}

export default MapEditorV2;
