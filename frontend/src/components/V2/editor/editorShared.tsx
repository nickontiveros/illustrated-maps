/** Shared between the SVG editor (MapEditorV2) and the WebGL editor
 * (MapEditorGL): the mode list, the right-column Inspector (driven entirely by
 * the shared editorStore, so it is identical for both), and small helpers. */

import { useEditor, type EditorMode } from './editorStore';

export const MODES: { id: EditorMode; label: string; hint: string }[] = [
  { id: 'select', label: 'Select', hint: 'Click a road, river, POI or place to show/hide it.' },
  { id: 'warp', label: 'Warp', hint: 'Drag on the map to draw a magnify region; click one to edit.' },
  { id: 'poi', label: 'POIs', hint: 'Drag a POI to nudge it; click to select and resize.' },
  { id: 'roads', label: 'Roads', hint: 'Click a road to route it: warped / straight / hidden, or reshape its path.' },
  { id: 'labels', label: 'Labels', hint: 'Drag a label to place it by hand; double-click to reset.' },
];

export const round = (n: number) => Math.round(n * 1000) / 1000;

/** A coarse, evenly-spaced handle set to start a reshape from (endpoints kept). */
export function decimateForReshape(points: [number, number][], n = 7): [number, number][] {
  if (points.length <= n) return points.map((p) => [p[0], p[1]] as [number, number]);
  const step = (points.length - 1) / (n - 1);
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const p = points[Math.round(i * step)];
    out.push([p[0], p[1]]);
  }
  return out;
}

export function Inspector({
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
          <p className="mt-1 text-[11px] text-gray-400">e.g. bow I-10 away from the airport to open a gap.</p>
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
          <button onClick={() => ed.movePoi(p.id, [0, 0])} className="block rounded bg-gray-100 px-2 py-1 text-xs">
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
          Drag a POI or city label to place it by hand. Double-click a moved (red) label to reset it. Watch the
          preview for the warped result.
        </p>
        <div className="text-xs text-gray-400">{n} hand-placed</div>
      </div>
    );
  }
  return <div className="p-3 text-xs text-gray-400">Nothing selected.</div>;
}
