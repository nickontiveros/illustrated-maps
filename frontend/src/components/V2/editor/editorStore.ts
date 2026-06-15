/** Working state for the layout editor: the in-flight CompositionSpec, the
 * source feed, the current tool, selection, and a debounced live preview.
 *
 * The spec is edited locally; every change schedules a `preview-plan` call
 * (no persistence, no AI) so the warped result updates within a beat. The
 * edited spec is persisted only on an explicit Save (PUT /composition), after
 * which a re-plan applies it to the real pipeline. */

import { create } from 'zustand';
import {
  v2api,
  type CompositionSpec,
  type PreviewPlanResult,
  type SourceGeojson,
  type WarpRegion,
} from '@/api/v2';

export type EditorMode = 'select' | 'warp' | 'poi' | 'roads' | 'labels';
export type SelectableLayer = 'roads' | 'rivers' | 'pois' | 'places';
export type RoadTreatment = 'warped' | 'straight' | 'hidden';
export type RegionCorner = 'nw' | 'ne' | 'sw' | 'se';

interface EditorState {
  projectId: string | null;
  spec: CompositionSpec | null;
  source: SourceGeojson | null;
  mode: EditorMode;
  selectedRegionId: string | null;
  selectedPoiId: string | null;
  selectedRoadId: string | null;
  preview: PreviewPlanResult | null;
  previewing: boolean;
  dirty: boolean;
  saving: boolean;
  error: string | null;

  init: (projectId: string, spec: CompositionSpec, source: SourceGeojson) => void;
  reset: () => void;
  setMode: (mode: EditorMode) => void;

  // generic spec edit: mutate a draft, mark dirty, schedule a preview
  edit: (mutate: (draft: CompositionSpec) => void) => void;

  toggleFeature: (layer: SelectableLayer, id: string) => void;
  isVisible: (layer: SelectableLayer, id: string) => boolean;

  addRegion: (bounds: [number, number, number, number]) => void;
  updateRegion: (id: string, patch: Partial<WarpRegion>) => void;
  resizeRegionCorner: (id: string, corner: RegionCorner, uv: [number, number]) => void;
  removeRegion: (id: string) => void;
  selectRegion: (id: string | null) => void;

  selectPoi: (id: string | null) => void;
  movePoi: (id: string, offset: [number, number]) => void;
  resizePoi: (id: string, factor: number) => void;
  cyclePoiLeader: (id: string) => void;

  moveLabel: (key: string, uv: [number, number]) => void;
  resetLabel: (key: string) => void;

  selectRoad: (id: string | null) => void;
  roadTreatment: (id: string) => RoadTreatment;
  setRoadTreatment: (id: string, treatment: RoadTreatment) => void;
  beginReshape: (id: string, points: [number, number][]) => void;
  updateReshapeVertex: (id: string, index: number, uv: [number, number]) => void;
  clearReshape: (id: string) => void;

  save: () => Promise<void>;
}

const clone = (spec: CompositionSpec): CompositionSpec =>
  JSON.parse(JSON.stringify(spec)) as CompositionSpec;

let previewTimer: ReturnType<typeof setTimeout> | null = null;
let regionSeq = 0;

export const useEditor = create<EditorState>((set, get) => {
  const schedulePreview = () => {
    if (previewTimer) clearTimeout(previewTimer);
    previewTimer = setTimeout(() => {
      const { projectId, spec } = get();
      if (!projectId || !spec) return;
      set({ previewing: true });
      v2api
        .previewPlan(projectId, spec)
        .then((preview) => set({ preview, previewing: false, error: null }))
        .catch((e) => set({ previewing: false, error: String(e?.message ?? e) }));
    }, 400);
  };

  const applyEdit = (mutate: (draft: CompositionSpec) => void) => {
    const current = get().spec;
    if (!current) return;
    const draft = clone(current);
    mutate(draft);
    set({ spec: draft, dirty: true });
    schedulePreview();
  };

  return {
    projectId: null,
    spec: null,
    source: null,
    mode: 'select',
    selectedRegionId: null,
    selectedPoiId: null,
    selectedRoadId: null,
    preview: null,
    previewing: false,
    dirty: false,
    saving: false,
    error: null,

    init: (projectId, spec, source) => {
      regionSeq = spec.warp.regions.length;
      set({
        projectId,
        spec,
        source,
        mode: 'select',
        selectedRegionId: null,
        selectedPoiId: null,
        selectedRoadId: null,
        preview: null,
        dirty: false,
        error: null,
      });
      schedulePreview(); // show the current layout immediately
    },

    reset: () =>
      set({ projectId: null, spec: null, source: null, preview: null, dirty: false }),

    setMode: (mode) =>
      set({ mode, selectedRegionId: null, selectedPoiId: null, selectedRoadId: null }),

    edit: (mutate) => applyEdit(mutate),

    isVisible: (layer, id) => {
      const spec = get().spec;
      if (!spec) return true;
      const sel = spec.features[layer];
      if (sel.exclude.includes(id)) return false;
      if (sel.include.includes(id)) return true;
      if (sel.default === 'none') return false;
      return true; // "auto"/"all": visible unless the heuristic dropped it
    },

    toggleFeature: (layer, id) =>
      applyEdit((draft) => {
        const sel = draft.features[layer];
        const visible = !sel.exclude.includes(id);
        sel.exclude = sel.exclude.filter((x) => x !== id);
        sel.include = sel.include.filter((x) => x !== id);
        if (visible) sel.exclude.push(id);
        else sel.include.push(id);
      }),

    addRegion: (bounds) => {
      const id = `r${++regionSeq}`;
      applyEdit((draft) => {
        draft.warp.mode = 'manual';
        draft.warp.regions.push({ id, bounds, magnify: 1.8 });
      });
      set({ selectedRegionId: id });
    },

    updateRegion: (id, patch) =>
      applyEdit((draft) => {
        const r = draft.warp.regions.find((x) => x.id === id);
        if (r) Object.assign(r, patch);
      }),

    resizeRegionCorner: (id, corner, uv) =>
      applyEdit((draft) => {
        const r = draft.warp.regions.find((x) => x.id === id);
        if (!r) return;
        let [u0, v0, u1, v1] = r.bounds;
        if (corner === 'nw') [u0, v0] = uv;
        else if (corner === 'ne') [u1, v0] = uv;
        else if (corner === 'sw') [u0, v1] = uv;
        else [u1, v1] = uv;
        r.bounds = [Math.min(u0, u1), Math.min(v0, v1), Math.max(u0, u1), Math.max(v0, v1)];
      }),

    removeRegion: (id) => {
      applyEdit((draft) => {
        draft.warp.regions = draft.warp.regions.filter((x) => x.id !== id);
        if (draft.warp.regions.length === 0) draft.warp.mode = 'auto';
      });
      set({ selectedRegionId: null });
    },

    selectRegion: (id) => set({ selectedRegionId: id }),

    selectPoi: (id) => set({ selectedPoiId: id }),

    movePoi: (id, offset) =>
      applyEdit((draft) => {
        const ov = (draft.pois[id] ??= { leader: 'auto', label_side: 'auto' });
        ov.offset_uv = offset;
      }),

    resizePoi: (id, factor) =>
      applyEdit((draft) => {
        const ov = (draft.pois[id] ??= { leader: 'auto', label_side: 'auto' });
        const next = Math.max(0.3, Math.min(4, (ov.size ?? 1) * factor));
        ov.size = Math.round(next * 100) / 100;
      }),

    cyclePoiLeader: (id) =>
      applyEdit((draft) => {
        const ov = (draft.pois[id] ??= { leader: 'auto', label_side: 'auto' });
        ov.leader = ov.leader === 'auto' ? 'force' : ov.leader === 'force' ? 'suppress' : 'auto';
      }),

    moveLabel: (key, uv) =>
      applyEdit((draft) => {
        draft.labels.overrides[key] = uv;
      }),

    resetLabel: (key) =>
      applyEdit((draft) => {
        delete draft.labels.overrides[key];
      }),

    selectRoad: (id) => set({ selectedRoadId: id }),

    roadTreatment: (id) => get().spec?.roads[id]?.treatment ?? 'warped',

    setRoadTreatment: (id, treatment) =>
      applyEdit((draft) => {
        const ov = (draft.roads[id] ??= { treatment: 'warped', reshape: null });
        ov.treatment = treatment;
        if (treatment !== 'straight') ov.reshape = null; // reshape only applies when straight
      }),

    beginReshape: (id, points) =>
      applyEdit((draft) => {
        const ov = (draft.roads[id] ??= { treatment: 'straight', reshape: null });
        ov.treatment = 'straight';
        ov.reshape = points.map((p) => [...p] as [number, number]);
      }),

    updateReshapeVertex: (id, index, uv) =>
      applyEdit((draft) => {
        const ov = draft.roads[id];
        if (ov?.reshape && ov.reshape[index]) ov.reshape[index] = uv;
      }),

    clearReshape: (id) =>
      applyEdit((draft) => {
        const ov = draft.roads[id];
        if (ov) ov.reshape = null;
      }),

    save: async () => {
      const { projectId, spec } = get();
      if (!projectId || !spec) return;
      set({ saving: true });
      try {
        const saved = clone(spec);
        saved.seeded_from = 'manual';
        await v2api.putComposition(projectId, saved);
        set({ spec: saved, dirty: false, saving: false, error: null });
      } catch (e) {
        set({ saving: false, error: String((e as Error)?.message ?? e) });
      }
    },
  };
});
