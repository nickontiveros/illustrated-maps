import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { v2api } from '@/api/v2';
import { useEditor } from '../editor/editorStore';
import { Inspector, MODES, round } from '../editor/editorShared';
import { createGLMap, type GLMap } from './glRenderer';
import { buildCDF } from './warp';
import { DEFAULT_CAMERA, makeTransform, project, type CameraParams } from './project';

/** WebGL editor: the source map on a GPU canvas that renders the *projected*
 * 2.5D poster geometry (warp THEN oblique camera) and deforms live as you edit.
 * Same five tools, store and endpoints as the SVG editor — but because the
 * canvas shows the real poster shape, dragging a label/POI lands it exactly
 * where it prints (the SVG editor edits the flat pre-warp source instead).
 *
 * Every overlay mark is positioned by the client transform replica (project.ts
 * + warp.ts, verified against the backend); every drag is inverse-mapped back
 * into the space the backend stores it in:
 *   - label override  -> pre-warp uv      (warp⁻¹∘camera⁻¹ = transform.inv)
 *   - POI offset_uv   -> post-warp delta  (camera⁻¹ = transform.unwarpProject)
 *   - reshape vertex  -> flat uv (no warp; straight roads bypass it)
 *   - warp region     -> pre-warp bounds  (transform.inv) */

const VBW = 1000;
type Corner = 'nw' | 'ne' | 'sw' | 'se';

function MapEditorGL() {
  const { id = '' } = useParams();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayG = useRef<SVGGElement | null>(null);
  const gl = useRef<GLMap | null>(null);
  const view = useRef({ scale: 1, tx: 0, ty: 0 });
  const drag = useRef<
    | null
    | { kind: 'pan'; start: [number, number]; origin: [number, number] }
    | { kind: 'region-new'; start: [number, number]; startVB: [number, number] }
    | { kind: 'region-move'; id: string; start: [number, number]; origin: [number, number] }
    | { kind: 'region-resize'; id: string; corner: Corner }
    | { kind: 'poi-move'; id: string; base: [number, number] }
    | { kind: 'reshape-vertex'; id: string; index: number }
    | { kind: 'label-move'; key: string }
  >(null);
  const [rubber, setRubber] = useState<null | [number, number, number, number]>(null);
  const [markerScale, setMarkerScale] = useState(1);
  const [previewExpanded, setPreviewExpanded] = useState(false);

  const project_ = useQuery({ queryKey: ['v2-project', id], queryFn: () => v2api.getProject(id), enabled: !!id });
  const composition = useQuery({ queryKey: ['v2-composition', id], queryFn: () => v2api.getComposition(id), enabled: !!id });
  const source = useQuery({ queryKey: ['v2-source', id], queryFn: () => v2api.getSourceGeojson(id), enabled: !!id });
  const ed = useEditor();

  useEffect(() => {
    if (composition.data && source.data && id) ed.init(id, composition.data, source.data);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [composition.data, source.data, id]);

  const out = project_.data?.config.output;
  const aspect = out ? out.height_px / out.width_px : 1.414;
  const VBH = VBW * aspect;

  const camera: CameraParams = useMemo(
    () => project_.data?.config.camera ?? DEFAULT_CAMERA,
    [project_.data?.config.camera]
  );

  const regions = ed.spec?.warp.regions ?? [];
  // CDFs recomputed synchronously from the regions, so the overlay marks and
  // the GPU morph always share one warp.
  const fx = useMemo(() => buildCDF(regions, 0), [regions]);
  const fy = useMemo(() => buildCDF(regions, 1), [regions]);
  const tf = useMemo(() => makeTransform(fx, fy, camera), [fx, fy, camera]);

  // --- coordinate helpers (overlay marks live in viewBox units; the <g> applies
  // the pan/zoom transform, identical to the GL camera) ---
  // pre-warp uv -> viewBox px (warp + camera): where a feature actually prints.
  const fwd = (u: number, v: number): [number, number] => {
    const [X, Y] = tf.fwd(u, v);
    return [X * VBW, Y * VBH];
  };
  // flat uv -> viewBox px (camera only, no warp): straight roads / reshape paths.
  const fwdFlat = (u: number, v: number): [number, number] => {
    const [X, Y] = project(u, v, camera);
    return [X * VBW, Y * VBH];
  };
  const toVB = (e: { clientX: number; clientY: number }): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [((e.clientX - rect.left) / rect.width) * VBW, ((e.clientY - rect.top) / rect.height) * VBH];
  };
  // client px -> poster-normalized (X,Y), removing the pan/zoom transform.
  const toPoster = (clientX: number, clientY: number): [number, number] => {
    const [vbx, vby] = toVB({ clientX, clientY });
    const { scale, tx, ty } = view.current;
    return [(vbx - tx) / scale / VBW, (vby - ty) / scale / VBH];
  };

  const applyView = () => {
    overlayG.current?.setAttribute(
      'transform',
      `translate(${view.current.tx} ${view.current.ty}) scale(${view.current.scale})`
    );
    gl.current?.setView(view.current);
    gl.current?.draw();
  };
  const resetView = () => {
    view.current = { scale: 1, tx: 0, ty: 0 };
    applyView();
    setMarkerScale(1);
  };

  // --- create the GL map once canvas + source are ready ---
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !source.data) return;
    const map = createGLMap(canvas, VBW, VBH, camera);
    map.buildGeometry(source.data);
    gl.current = map;
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = container.clientWidth;
      map.setSize(Math.round(w * dpr), Math.round(w * aspect * dpr));
      map.setView(view.current);
      map.draw();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    return () => {
      ro.disconnect();
      map.destroy();
      gl.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source.data, VBH]);

  // push warp + camera to the GPU whenever they change
  useEffect(() => {
    gl.current?.setCDF(fx, fy);
    gl.current?.setCamera(camera);
    gl.current?.draw();
  }, [fx, fy, camera]);

  // wheel zoom (native, passive:false) toward the cursor
  const ready = !!ed.spec && !!source.data;
  useEffect(() => {
    const el = canvasRef.current;
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
      applyView();
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

  // --- canvas pointer handling: warp mode draws a region, else pan ---
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    if (ed.mode === 'warp') {
      drag.current = { kind: 'region-new', start: toPoster(e.clientX, e.clientY), startVB: toVB(e) };
    } else {
      drag.current = { kind: 'pan', start: toVB(e), origin: [view.current.tx, view.current.ty] };
    }
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === 'pan') {
      const vb = toVB(e);
      view.current.tx = d.origin[0] + (vb[0] - d.start[0]);
      view.current.ty = d.origin[1] + (vb[1] - d.start[1]);
      applyView();
      return;
    }
    if (d.kind === 'region-new') {
      const vb = toVB(e);
      setRubber([d.startVB[0], d.startVB[1], vb[0], vb[1]]);
      return;
    }
    const p = toPoster(e.clientX, e.clientY);
    if (d.kind === 'region-move') {
      const [u, v] = tf.inv(p[0], p[1]);
      const r = regions.find((x) => x.id === d.id);
      if (r) {
        const w = r.bounds[2] - r.bounds[0];
        const h = r.bounds[3] - r.bounds[1];
        const cx = d.origin[0] + (u - d.start[0]);
        const cy = d.origin[1] + (v - d.start[1]);
        ed.updateRegion(d.id, { bounds: [cx, cy, cx + w, cy + h] });
      }
    } else if (d.kind === 'region-resize') {
      const [u, v] = tf.inv(p[0], p[1]);
      ed.resizeRegionCorner(d.id, d.corner, [round(u), round(v)]);
    } else if (d.kind === 'poi-move') {
      // POI offset is a post-warp delta (builder.py): camera⁻¹ to the warped
      // frame, then subtract the POI's warped base anchor.
      const [wu, wv] = tf.unwarpProject(p[0], p[1]);
      ed.movePoi(d.id, [round(wu - d.base[0]), round(wv - d.base[1])]);
    } else if (d.kind === 'reshape-vertex') {
      // Straight roads bypass the warp: the reshape vertex is a flat uv.
      const [u, v] = tf.unwarpProject(p[0], p[1]);
      ed.updateReshapeVertex(d.id, d.index, [round(u), round(v)]);
    } else if (d.kind === 'label-move') {
      // Label overrides are pre-warp: full inverse (camera⁻¹ then warp⁻¹).
      const [u, v] = tf.inv(p[0], p[1]);
      ed.moveLabel(d.key, [round(u), round(v)]);
    }
  };
  // Clicks on overlay marks stopPropagation, so a click that reaches the canvas
  // is on empty space: deselect. (Marks select via their own handlers.)
  const clearSelection = () => {
    ed.selectRegion(null);
    ed.selectPoi(null);
    ed.selectRoad(null);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    if (d?.kind === 'region-new') {
      const end = toPoster(e.clientX, e.clientY);
      const a = tf.inv(d.start[0], d.start[1]);
      const b = tf.inv(end[0], end[1]);
      const bounds: [number, number, number, number] = [
        Math.min(a[0], b[0]),
        Math.min(a[1], b[1]),
        Math.max(a[0], b[0]),
        Math.max(a[1], b[1]),
      ];
      if (bounds[2] - bounds[0] > 0.02 && bounds[3] - bounds[1] > 0.02) ed.addRegion(bounds);
      else clearSelection(); // a click (not a drag) on empty space
    } else if (d?.kind === 'pan') {
      const vb = toVB(e);
      if (Math.hypot(vb[0] - d.start[0], vb[1] - d.start[1]) < 5) clearSelection();
    }
    setRubber(null);
  };

  const selectedRegion = spec.warp.regions.find((r) => r.id === ed.selectedRegionId) ?? null;
  const selectedPoi = ed.selectedPoiId ? src.pois.find((p) => p.id === ed.selectedPoiId) ?? null : null;
  const selectedRoad = ed.selectedRoadId ? src.roads.find((r) => r.id === ed.selectedRoadId) ?? null : null;
  const labelMarkers =
    ed.mode === 'labels'
      ? [
          ...src.pois.map((p) => ({ key: p.id, text: p.name, nat: p.point })),
          ...src.places.map((p) => ({ key: p.id, text: p.name, nat: p.point })),
        ]
      : [];
  const showRoadHits = ed.mode === 'select' || ed.mode === 'roads';

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
            (showing {src.counts.roads_shown} editable of {src.counts.roads_total} roads)
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
        {/* GPU canvas + interactive overlay */}
        <div className="flex-1 overflow-auto bg-stone-200 p-4">
          <div
            ref={containerRef}
            className="relative mx-auto w-full max-w-3xl touch-none shadow"
            style={{ aspectRatio: `${VBW} / ${VBH}` }}
            // Handlers live on the container (ancestor of both the canvas and the
            // overlay), so a mark that captures the pointer still delivers its
            // pointerup here — otherwise mark drags would never release.
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          >
            <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
            <svg
              viewBox={`0 0 ${VBW} ${VBH}`}
              className="absolute inset-0 h-full w-full"
              style={{ pointerEvents: 'none' }}
            >
              <g ref={overlayG} transform={`translate(${view.current.tx} ${view.current.ty}) scale(${view.current.scale})`}>
                {/* road hit-targets + selection (base roads are drawn on the GL canvas) */}
                {showRoadHits &&
                  src.roads.map((r) => {
                    const isRiver = r.cls === 'river' || r.cls === 'stream';
                    const layer = isRiver ? 'rivers' : 'roads';
                    const ov = spec.roads[r.id];
                    const selected = ed.mode === 'roads' && ed.selectedRoadId === r.id;
                    // warped roads follow warp+camera; straight/reshape bypass warp.
                    const straight = ov?.treatment === 'straight';
                    const geom = ov?.reshape ?? r.points;
                    const pts = geom.map(([u, v]) => (straight ? fwdFlat(u, v) : fwd(u, v)).join(',')).join(' ');
                    return (
                      <g key={r.id}>
                        {selected && (
                          <polyline points={pts} fill="none" stroke="#dc2626" strokeWidth={4 * markerScale} pointerEvents="none" />
                        )}
                        <polyline
                          points={pts}
                          fill="none"
                          stroke="transparent"
                          strokeWidth={14 * markerScale}
                          pointerEvents="stroke"
                          className="cursor-pointer"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (ed.mode === 'select') ed.toggleFeature(layer, r.id);
                            else ed.selectRoad(r.id);
                          }}
                        />
                        {/* reshape vertex handles */}
                        {selected &&
                          ov?.reshape?.map((uv, i) => {
                            const [hx, hy] = fwdFlat(uv[0], uv[1]);
                            return (
                              <circle
                                key={i}
                                cx={hx}
                                cy={hy}
                                r={7 * markerScale}
                                fill="#fff"
                                stroke="#dc2626"
                                strokeWidth={2 * markerScale}
                                style={{ pointerEvents: 'auto', cursor: 'grab' }}
                                onPointerDown={(e) => {
                                  e.stopPropagation();
                                  drag.current = { kind: 'reshape-vertex', id: r.id, index: i };
                                  (e.target as Element).setPointerCapture?.(e.pointerId);
                                }}
                              />
                            );
                          })}
                      </g>
                    );
                  })}

                {/* places (cities) */}
                {(ed.mode === 'select' ? src.places : []).map((p) => {
                  const visible = ed.isVisible('places', p.id);
                  const [x, y] = fwd(p.point[0], p.point[1]);
                  return (
                    <text
                      key={p.id}
                      x={x}
                      y={y}
                      fontSize={12 * markerScale}
                      textAnchor="middle"
                      fill={visible ? '#6b5b3e' : '#bbb'}
                      stroke="#f7f1e1"
                      strokeWidth={2 * markerScale}
                      paintOrder="stroke"
                      style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        ed.toggleFeature('places', p.id);
                      }}
                    >
                      {p.name}
                    </text>
                  );
                })}

                {/* POIs */}
                {(ed.mode === 'select' || ed.mode === 'poi' ? src.pois : []).map((p) => {
                  const visible = ed.isVisible('pois', p.id);
                  const ov = spec.pois[p.id];
                  const off = ov?.offset_uv ?? [0, 0];
                  const wb = tf.warp(p.point[0], p.point[1]);
                  const [X, Y] = project(wb[0] + off[0], wb[1] + off[1], camera);
                  const [x, y] = [X * VBW, Y * VBH];
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
                        style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (ed.mode === 'select') ed.toggleFeature('pois', p.id);
                          else ed.selectPoi(p.id);
                        }}
                        onPointerDown={(e) => {
                          if (ed.mode !== 'poi') return;
                          e.stopPropagation();
                          ed.selectPoi(p.id);
                          drag.current = { kind: 'poi-move', id: p.id, base: wb };
                          (e.target as Element).setPointerCapture?.(e.pointerId);
                        }}
                      />
                      <text x={x + 8 * markerScale} y={y + 4 * markerScale} fontSize={11 * markerScale} fill="#4b3f29" pointerEvents="none">
                        {p.name}
                      </text>
                    </g>
                  );
                })}

                {/* warp regions (projected as trapezoids; corners are the handles) */}
                {(ed.mode === 'warp' ? spec.warp.regions : []).map((r) => {
                  const selected = ed.selectedRegionId === r.id;
                  const corners: [Corner, number, number][] = [
                    ['nw', r.bounds[0], r.bounds[1]],
                    ['ne', r.bounds[2], r.bounds[1]],
                    ['se', r.bounds[2], r.bounds[3]],
                    ['sw', r.bounds[0], r.bounds[3]],
                  ];
                  const poly = corners.map(([, u, v]) => fwd(u, v).join(',')).join(' ');
                  return (
                    <g key={r.id}>
                      <polygon
                        points={poly}
                        fill="#6366f1"
                        fillOpacity={selected ? 0.18 : 0.1}
                        stroke="#6366f1"
                        strokeWidth={(selected ? 2 : 1) * markerScale}
                        strokeDasharray="6 3"
                        style={{ pointerEvents: 'auto', cursor: 'move' }}
                        onPointerDown={(e) => {
                          e.stopPropagation();
                          ed.selectRegion(r.id);
                          const p = toPoster(e.clientX, e.clientY);
                          drag.current = {
                            kind: 'region-move',
                            id: r.id,
                            start: tf.inv(p[0], p[1]),
                            origin: [Math.min(r.bounds[0], r.bounds[2]), Math.min(r.bounds[1], r.bounds[3])],
                          };
                          (e.target as Element).setPointerCapture?.(e.pointerId);
                        }}
                      />
                      {selected &&
                        corners.map(([corner, cu, cv]) => {
                          const [hx, hy] = fwd(cu, cv);
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
                              style={{ pointerEvents: 'auto', cursor: 'nwse-resize' }}
                              onPointerDown={(e) => {
                                e.stopPropagation();
                                ed.selectRegion(r.id);
                                drag.current = { kind: 'region-resize', id: r.id, corner };
                                (e.target as Element).setPointerCapture?.(e.pointerId);
                              }}
                            />
                          );
                        })}
                    </g>
                  );
                })}

                {/* rubber-band for a new region (drawn in screen space) */}
                {rubber && (
                  <rect
                    x={Math.min(rubber[0], rubber[2])}
                    y={Math.min(rubber[1], rubber[3])}
                    width={Math.abs(rubber[2] - rubber[0])}
                    height={Math.abs(rubber[3] - rubber[1])}
                    fill="#6366f1"
                    fillOpacity={0.1}
                    stroke="#6366f1"
                    strokeDasharray="6 3"
                    pointerEvents="none"
                  />
                )}

                {/* editable labels (WYSIWYG: dragged on the projected poster) */}
                {labelMarkers.map((m) => {
                  const ov = spec.labels.overrides[m.key];
                  const [lx, ly] = fwd((ov ?? m.nat)[0], (ov ?? m.nat)[1]);
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
                      style={{ pointerEvents: 'auto', cursor: 'grab', userSelect: 'none' }}
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        drag.current = { kind: 'label-move', key: m.key };
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
        </div>

        {/* Right column: inspector + backend live preview */}
        <div className="flex w-96 flex-col border-l bg-white">
          <Inspector selectedRegion={selectedRegion} selectedPoi={selectedPoi} selectedRoad={selectedRoad} />
          <div className="flex-1 overflow-auto border-t p-2">
            <div className="mb-1 flex items-center gap-2">
              <span className="text-xs font-semibold text-gray-500">Live preview</span>
              <span
                className={`inline-block h-2.5 w-2.5 rounded-full ${
                  ed.previewing ? 'animate-pulse bg-amber-400' : ed.preview ? 'bg-emerald-500' : 'bg-gray-300'
                }`}
                title={ed.previewing ? 'updating…' : ed.preview ? 'up to date' : 'not rendered yet'}
              />
              <span className="text-[11px] text-gray-400">
                {ed.previewing ? 'updating…' : ed.preview ? 'authoritative render' : ''}
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={() => setPreviewExpanded(false)}>
          <div className="relative max-h-full overflow-auto rounded bg-white p-2 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setPreviewExpanded(false)}
              className="absolute right-3 top-3 z-10 rounded bg-gray-800/80 px-2 py-1 text-xs text-white"
            >
              Close ✕
            </button>
            <div className="[&>svg]:h-auto [&>svg]:max-h-[85vh] [&>svg]:w-auto" dangerouslySetInnerHTML={{ __html: ed.preview.svg }} />
          </div>
        </div>
      )}
    </Shell>
  );
}

function Shell({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <div className="flex h-screen flex-col">
      <div className="flex items-center gap-3 border-b bg-white px-4 py-2">
        <Link to={`/v2/${id}`} className="text-sm text-indigo-600">
          ← Back to project
        </Link>
        <span className="text-sm font-semibold">Layout editor · WebGL</span>
        <span className="text-[11px] text-gray-400">live 2.5D poster · drag labels where they print</span>
        <Link to={`/v2/${id}/edit`} className="ml-auto text-xs text-gray-500 hover:text-gray-700">
          ↩ SVG editor
        </Link>
      </div>
      {children}
    </div>
  );
}

export default MapEditorGL;
