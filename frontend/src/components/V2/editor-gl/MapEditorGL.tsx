import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { v2api } from '@/api/v2';
import { useEditor } from '../editor/editorStore';
import { createGLMap, type GLMap } from './glRenderer';
import { buildCDF, warpValue, inverseWarpValue } from './warp';

/** WebGL editor spike: the source map on a GPU canvas that DEFORMS LIVE as you
 * drag a warp region (vs the SVG editor's static side-preview). Shares the
 * CompositionSpec, editor store, and endpoints with the SVG editor. */

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
    | { kind: 'region-new'; start: [number, number] }
    | { kind: 'region-move'; id: string; start: [number, number]; origin: [number, number] }
    | { kind: 'region-resize'; id: string; corner: Corner }
  >(null);
  const [rubber, setRubber] = useState<null | [number, number, number, number]>(null);
  const panHeld = useRef(false);

  const project = useQuery({ queryKey: ['v2-project', id], queryFn: () => v2api.getProject(id), enabled: !!id });
  const composition = useQuery({ queryKey: ['v2-composition', id], queryFn: () => v2api.getComposition(id), enabled: !!id });
  const source = useQuery({ queryKey: ['v2-source', id], queryFn: () => v2api.getSourceGeojson(id), enabled: !!id });
  const ed = useEditor();

  useEffect(() => {
    if (composition.data && source.data && id) ed.init(id, composition.data, source.data);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [composition.data, source.data, id]);

  const aspect = project.data?.config.output
    ? project.data.config.output.height_px / project.data.config.output.width_px
    : 1.414;
  const VBH = VBW * aspect;

  const regions = ed.spec?.warp.regions ?? [];
  // CDFs recomputed synchronously from the regions, so the overlay handles
  // (warped positions) and the GPU morph always use the same warp.
  const fx = useMemo(() => buildCDF(regions, 0), [regions]);
  const fy = useMemo(() => buildCDF(regions, 1), [regions]);

  const applyView = () => {
    overlayG.current?.setAttribute(
      'transform',
      `translate(${view.current.tx} ${view.current.ty}) scale(${view.current.scale})`
    );
    gl.current?.setView(view.current);
    gl.current?.draw();
  };

  // --- create the GL map once canvas + source are ready ---
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !source.data) return;
    const map = createGLMap(canvas, VBW, VBH);
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

  // push CDFs to the GPU whenever the warp changes
  useEffect(() => {
    gl.current?.setCDF(fx, fy);
    gl.current?.draw();
  }, [fx, fy]);

  // hold Space to pan (background drag otherwise draws a region)
  useEffect(() => {
    const set = (down: boolean) => (e: KeyboardEvent) => {
      if (e.code === 'Space') panHeld.current = down;
    };
    const d = set(true);
    const u = set(false);
    window.addEventListener('keydown', d);
    window.addEventListener('keyup', u);
    return () => {
      window.removeEventListener('keydown', d);
      window.removeEventListener('keyup', u);
    };
  }, []);

  // wheel zoom (native, passive:false)
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
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, VBH]);

  // viewport(px) -> warped-normalized (u', v'); inverse-warp for pre-warp bound
  const toWarpedNorm = (clientX: number, clientY: number): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const vbx = ((clientX - rect.left) / rect.width) * VBW;
    const vby = ((clientY - rect.top) / rect.height) * VBH;
    const { scale, tx, ty } = view.current;
    return [(vbx - tx) / scale / VBW, (vby - ty) / scale / VBH];
  };
  const toPreWarp = (clientX: number, clientY: number): [number, number] => {
    const [wu, wv] = toWarpedNorm(clientX, clientY);
    return [inverseWarpValue(fx, wu), inverseWarpValue(fy, wv)];
  };

  const toVB = (e: { clientX: number; clientY: number }): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [((e.clientX - rect.left) / rect.width) * VBW, ((e.clientY - rect.top) / rect.height) * VBH];
  };
  // Background-drag draws a new region (the core action); hold Space to pan.
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    if (panHeld.current) {
      drag.current = { kind: 'pan', start: toVB(e), origin: [view.current.tx, view.current.ty] };
    } else {
      drag.current = { kind: 'region-new', start: toPreWarp(e.clientX, e.clientY) };
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
    } else if (d.kind === 'region-new') {
      const n = toPreWarp(e.clientX, e.clientY);
      setRubber([d.start[0], d.start[1], n[0], n[1]]);
    } else if (d.kind === 'region-move') {
      const n = toPreWarp(e.clientX, e.clientY);
      const r = regions.find((x) => x.id === d.id);
      if (r) {
        const w = r.bounds[2] - r.bounds[0];
        const h = r.bounds[3] - r.bounds[1];
        const cx = d.origin[0] + (n[0] - d.start[0]);
        const cy = d.origin[1] + (n[1] - d.start[1]);
        ed.updateRegion(d.id, { bounds: [cx, cy, cx + w, cy + h] });
      }
    } else if (d.kind === 'region-resize') {
      const n = toPreWarp(e.clientX, e.clientY);
      ed.resizeRegionCorner(d.id, d.corner, [round(n[0]), round(n[1])]);
    }
  };
  const onPointerUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d?.kind === 'region-new' && rubber) {
      const b: [number, number, number, number] = [
        Math.min(rubber[0], rubber[2]),
        Math.min(rubber[1], rubber[3]),
        Math.max(rubber[0], rubber[2]),
        Math.max(rubber[1], rubber[3]),
      ];
      if (b[2] - b[0] > 0.02 && b[3] - b[1] > 0.02) ed.addRegion(b);
    }
    setRubber(null);
  };

  // warped screen position (viewBox units) of a pre-warp (u,v)
  const wpx = (u: number, v: number): [number, number] => [warpValue(fx, u) * VBW, warpValue(fy, v) * VBH];

  const warpMode = true; // spike: warp is the only tool
  const selected = ed.selectedRegionId;

  return (
    <div className="flex h-screen flex-col">
      <div className="flex items-center gap-3 border-b bg-white px-4 py-2">
        <Link to={`/v2/${id}`} className="text-sm text-indigo-600">
          ← Back to project
        </Link>
        <span className="text-sm font-semibold">Layout editor · WebGL (beta)</span>
        <span className="text-xs text-gray-400">
          Drag to draw a magnify region — the whole map deforms live. scroll = zoom · hold Space + drag = pan
        </span>
        <div className="ml-auto flex items-center gap-3">
          <Link to={`/v2/${id}/edit`} className="text-xs text-gray-500 hover:text-gray-700">
            ↩ SVG editor
          </Link>
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

      {!ready ? (
        <div className="p-8 text-sm text-gray-500">
          {source.isError ? 'No source yet — run Plan first.' : 'Loading layout…'}
        </div>
      ) : (
        <div className="flex-1 overflow-auto bg-stone-200 p-4">
          <div
            ref={containerRef}
            className="relative mx-auto w-full max-w-3xl shadow"
            style={{ aspectRatio: `${VBW} / ${VBH}` }}
          >
            <canvas
              ref={canvasRef}
              className="absolute inset-0 h-full w-full touch-none"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
            />
            <svg
              viewBox={`0 0 ${VBW} ${VBH}`}
              className="pointer-events-none absolute inset-0 h-full w-full"
            >
              <g ref={overlayG}>
                {regions.map((r) => {
                  const [x0, y0] = wpx(r.bounds[0], r.bounds[1]);
                  const [x1, y1] = wpx(r.bounds[2], r.bounds[3]);
                  const sel = selected === r.id;
                  const corners: [Corner, number, number][] = [
                    ['nw', r.bounds[0], r.bounds[1]],
                    ['ne', r.bounds[2], r.bounds[1]],
                    ['sw', r.bounds[0], r.bounds[3]],
                    ['se', r.bounds[2], r.bounds[3]],
                  ];
                  return (
                    <g key={r.id} style={{ pointerEvents: warpMode ? 'auto' : 'none' }}>
                      <rect
                        x={Math.min(x0, x1)}
                        y={Math.min(y0, y1)}
                        width={Math.abs(x1 - x0)}
                        height={Math.abs(y1 - y0)}
                        fill="#6366f1"
                        fillOpacity={sel ? 0.16 : 0.08}
                        stroke="#6366f1"
                        strokeWidth={sel ? 2 : 1}
                        strokeDasharray="6 3"
                        className="cursor-move"
                        onPointerDown={(e) => {
                          e.stopPropagation();
                          ed.selectRegion(r.id);
                          drag.current = {
                            kind: 'region-move',
                            id: r.id,
                            start: toPreWarp(e.clientX, e.clientY),
                            origin: [Math.min(r.bounds[0], r.bounds[2]), Math.min(r.bounds[1], r.bounds[3])],
                          };
                          (e.target as Element).setPointerCapture?.(e.pointerId);
                        }}
                      />
                      {sel &&
                        corners.map(([corner, cu, cv]) => {
                          const [hx, hy] = wpx(cu, cv);
                          return (
                            <rect
                              key={corner}
                              x={hx - 7}
                              y={hy - 7}
                              width={14}
                              height={14}
                              fill="#fff"
                              stroke="#6366f1"
                              strokeWidth={2}
                              style={{ cursor: 'nwse-resize' }}
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
                {rubber && (
                  <rect
                    x={Math.min(wpx(rubber[0], 0)[0], wpx(rubber[2], 0)[0])}
                    y={Math.min(wpx(0, rubber[1])[1], wpx(0, rubber[3])[1])}
                    width={Math.abs(wpx(rubber[2], 0)[0] - wpx(rubber[0], 0)[0])}
                    height={Math.abs(wpx(0, rubber[3])[1] - wpx(0, rubber[1])[1])}
                    fill="#6366f1"
                    fillOpacity={0.1}
                    stroke="#6366f1"
                    strokeDasharray="6 3"
                  />
                )}
              </g>
            </svg>
          </div>

          {selected && (
            <div className="mx-auto mt-3 flex max-w-3xl items-center gap-3 rounded bg-white p-2 text-sm shadow">
              <span className="font-semibold">Region</span>
              <label className="flex flex-1 items-center gap-2 text-xs text-gray-600">
                Magnify ×{(regions.find((r) => r.id === selected)?.magnify ?? 1).toFixed(1)}
                <input
                  type="range"
                  min={0.5}
                  max={4}
                  step={0.1}
                  value={regions.find((r) => r.id === selected)?.magnify ?? 1}
                  onChange={(e) => ed.updateRegion(selected, { magnify: Number(e.target.value) })}
                  className="flex-1"
                />
              </label>
              <button
                onClick={() => ed.removeRegion(selected)}
                className="rounded bg-red-50 px-2 py-1 text-xs text-red-600"
              >
                Delete
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const round = (n: number) => Math.round(n * 1000) / 1000;

export default MapEditorGL;
