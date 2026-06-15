# V2 Map-Editing Tool — Full Design (Phases A–D)

## 1. Problem & guiding principle

Every layout problem we hit at state scale — "give Mesa/Gilbert more horizontal
room," "trail US 60 off to the side," "only the Salt River should appear,"
"make the airport bigger" — is an **editorial, per-poster, subjective** choice.
No global automatic transform yields *straight roads + spread POIs + perfect
alignment* simultaneously; that is a design decision, not a solvable equation.

**Principle:** make those choices directly manipulable and persistent. Today's
heuristics (plateau clustering, river top-6, mainline-only roads,
`SIZE_BY_FEATURE`, auto-leaders) are demoted from *deciders* to *defaults that
seed an editable document*. The user edits a draft; they never start from blank.

The document is a **`CompositionSpec`** (`composition.json`, beside
`project.yaml`). The planner's job becomes "apply the spec, falling back to the
heuristic for any section left on `auto`." This keeps a fresh project working
exactly as it does today (the spec is absent ⇒ all-auto), while every decision
becomes overridable.

---

## 2. The `CompositionSpec`

### 2.1 Coordinate space (the key decision)

The spec is authored in **normalized frame space** `(u, v) ∈ [0,1]²`
(`GeoFrame.to_normalized`, `ingest.py:103`), **not** poster-pixel space.

Rationale: poster pixels are *post-warp, post-camera* — they move every time the
warp changes, so a position stored there would drift. Normalized frame space is
warp-independent and rotation-baked, so warp regions and POI nudges authored
there stay put as the user iterates. Feature selection and road routing key off
**stable feature ids** (§2.3), which are coordinate-free.

The editor canvas therefore works *natively in normalized space*: it renders the
source layers at their `(u,v)` positions and overlays the warped plan as a
result layer. Every user gesture is already normalized — no inverse-warp math in
the client.

### 2.2 Schema

```jsonc
// composition.json
{
  "version": "1.0",
  "seeded_from": "heuristics",        // "heuristics" until the user edits → "manual"

  "warp": {
    "mode": "auto",                    // "auto" = current plateau fit | "manual" | "off"
    "regions": [                       // used only when mode == "manual"
      { "id": "phx", "label": "Phoenix metro",
        "bounds": [0.30, 0.38, 0.62, 0.60],   // [u0,v0,u1,v1] normalized
        "magnify": 2.4 }                       // target relative magnification
    ]
  },

  "features": {                        // visibility per layer + per-id overrides
    "roads":  { "default": "auto", "include": [], "exclude": [] },
    "rivers": { "default": "auto", "include": [], "exclude": [] },
    "pois":   { "default": "auto", "exclude": [] },
    "places": { "default": "auto", "exclude": [] }    // city / district labels
  },

  "roads": {                           // per-road routing, keyed by road id
    "<road_id>": {
      "treatment": "warped",           // "warped" | "straight" | "hidden"
      "reshape": null                  // optional [[u,v],…] manual centerline
    }
  },

  "pois": {                            // per-POI overrides, keyed by poi id
    "<poi_id>": {
      "size": 1.4,                     // multiplier on tier width
      "tier": 1,                       // override tier
      "offset_uv": [0.0, 0.0],         // normalized nudge from true point
      "leader": "auto",                // "auto" | "force" | "suppress"
      "label_side": "auto"             // "auto" | "left" | "right" | "above" | "below"
    }
  },

  "labels": {
    "title": { "anchor_uv": null }     // optional cartouche placement override
  }
}
```

Every section is optional and defaults to `auto`/empty ⇒ **absent spec
reproduces today's plan exactly** (locked by a golden test, §6).

### 2.3 Prerequisite — stable feature ids

Roads/rivers/places currently lack stable identifiers (`SourceRoad` has only
`name`/`ref`; `ingest.py:20`). Feature selection and road routing need them.

- Carry the OSM id through `from_osm_data` (`ingest.py:241`) and the PBF backend
  when available; otherwise synthesize a deterministic id =
  `f"{cls}:{slug(name or ref)}:{hash(rounded coords)}"`.
- Preserve the id through `_merge_roads` (`generalize.py:69`) — when segments
  merge, keep the majority-vote id alongside the existing name/ref vote.
- Add `id: str` to `RoadPath`, `GroundPolygon`, and surface `SourcePlace.id`, so
  `plan.json` and the GeoJSON feed both reference the same ids the spec uses.

This is the one schema change that touches several files; it lands in Phase A and
everything else builds on it.

---

## 3. Backend — "apply the spec"

### 3.1 Planner integration

- New module `mapgen/v2/compose_spec.py`: the `CompositionSpec` pydantic model
  with `load(path)`, `save(path)`, `default()`, and `seed_from(plan, source)`
  (captures the current heuristic decisions as an explicit draft).
- `build_plan(project, source, spec=None)` (`pipeline.py:182`) threads the spec
  into `PlanBuilder(..., spec=spec)`. The plan stage reads `composition.json` if
  present.
- `PlanBuilder.build()` (`builder.py:165`) consults the spec at four existing
  seams, each falling back to today's code when the section is `auto`/absent:
  - **Warp** (`_fit_warp`, `builder.py:179`): `mode=="manual"` builds plateau
    bands directly from `warp.regions` (bounds → `ImportanceWarp(bands=…)`,
    skipping the auto clustering); `"off"` → `IdentityWarp`; `"auto"` → unchanged.
  - **Features** (roads/ground/poi/place loops): filter source items by
    `include`/`exclude` before building — a spec-aware extension of
    `generalize_source` (`generalize.py:152`).
  - **Roads** (`builder.py:212`): per-road `treatment` overrides the global
    `road_treatment`; a `reshape` polyline is projected straight (unwarped).
  - **POIs** (`builder.py:266`): apply `size`/`tier`/`offset_uv`/`leader`/
    `label_side` in `sized_slot` + `assign_leader_lines`.

### 3.2 API endpoints (extend `mapgen/api/routers/v2.py`)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/{id}/composition` | Current spec (seeded default if none on disk) |
| `PUT`  | `/{id}/composition` | Persist edited spec → `composition.json` |
| `GET`  | `/{id}/source.geojson` | Source layers in **normalized frame space** + ids, kind, name/ref, current visibility — the editor's render feed |
| `POST` | `/{id}/preview-plan` | Run layout on an **in-flight** spec (body), return `plan.json` + `preview.svg` **without persisting** — sub-second, layout only, no AI |

`preview-plan` reuses `build_plan` with the posted spec and the *cached* source
(no OSM re-fetch), so live edits round-trip in well under a second.

---

## 4. Frontend — the editor

**Stack:** SVG in a pan/zoom viewport — the preview is already SVG
(`plan_to_svg`), geometry counts are modest (hundreds of paths after
generalization), and SVG gives free hit-testing and drag handles with **no new
heavy dependency**. State: a Zustand slice for the working spec + dirty
tracking; `source.geojson` and `preview-plan` fetched via the existing `v2api`
client pattern (`frontend/src/api/v2.ts`).

**New components** under `frontend/src/components/V2/editor/`:
- `MapEditorV2.tsx` — the viewport: source layers (normalized) + warped plan
  overlay + pan/zoom.
- `EditorToolbar.tsx` — mode switch (Select / Warp / POI / Road), save, reset.
- `WarpLayer.tsx`, `FeatureLayer.tsx`, `PoiLayer.tsx`, `RoadLayer.tsx` — one
  interaction surface per phase.
- `useComposition.ts` — Zustand store: working spec, debounced PUT, debounced
  `preview-plan`, applied-plan result.

**Live-preview loop:** gesture → mutate working spec → debounced `preview-plan`
→ swap the overlay → (separately) debounced `PUT /composition` to persist.

Entry point: a "Edit layout" action on `ProjectViewV2.tsx` opens the editor for
a planned project.

---

## 5. Phases (each ships standalone value)

### Phase A — Spec foundation (backend only) — ✅ DONE (commit pending)
- Stable feature ids (§2.3); `compose_spec.py` model + load/save/default;
  `build_plan`/`PlanBuilder` consume the spec at the four seams;
  `GET/PUT /composition`; the plan stage reads `composition.json`.
- **No UI** — spec editable as JSON. Makes every heuristic overridable and
  de-risks all later phases.
- Tests (in `tests/v2/test_compose_spec.py` + `test_api.py`): spec round-trip;
  **golden test — `auto` spec reproduces today's plan byte-for-byte**; warp
  `manual`/`off`; POI size/tier/offset/leader overrides; deterministic+unique
  ids; `exclude` hides road/poi (rivers a separate layer); road
  hidden/straight/reshape; API round-trip applied to the plan. 230 v2 tests pass.
- **Deferred to Phase C:** the *richer* auto-seed that converts the auto
  plateau fit into explicit, editable `warp.regions`. For now the all-auto
  default returned by `GET /composition` is the starting draft (everything
  visible, warp auto) — the editor toggles from there.

### Phase B — Feature-selection UI
- `GET /source.geojson` + `POST /preview-plan`; `MapEditorV2` renders source
  layers; click-to-toggle include/exclude; live preview; save.
- Delivers the highest-value, simplest workflow: **pick what appears** (drop a
  minor wash, hide a redundant US route, remove a town label).
- Tests: geojson endpoint shape/normalization; `preview-plan` determinism &
  no-persist; (component) toggle mutates spec and re-previews.

### Phase C — Warp + POI direct manipulation
- Draw/drag/resize warp-region rectangles (→ `warp.regions`, `mode="manual"`);
  drag POIs (→ `offset_uv`), resize (→ `size`), toggle leader, set label side.
- This is where "give Mesa more room" and "make the airport bigger" become
  direct gestures instead of constants.
- Tests: a region magnifies its interior in the resulting plan; POI
  offset/size/tier/leader overrides honored; warp stays fold-safe (separable).

### Phase D — Road routing
- Select a road → treatment (warped / straight / hidden); optional vertex
  reshape to "trail US 60 off to the side" (→ `roads[id].reshape`, projected
  straight).
- Retires the global `road_treatment` flag to a mere seed; full manual road
  composition.
- Tests: per-road treatment overrides the global default; a reshaped road
  follows its spec polyline; hidden roads drop from the plan.

---

## 6. Cross-cutting

- **Back-compat / safety:** absent spec ⇒ identical plan (golden test gates
  every phase). `composition.json` lives in `projects/` (gitignored), like
  `project.yaml` — per-poster working data, not source.
- **Determinism:** spec-driven layout stays deterministic; existing
  determinism/compose tests must keep passing. No RNG in spec application.
- **Versioning:** `composition.json.version`; loader migrates older specs.
- **Performance:** the editor renders the *generalized/candidate* layers, never
  the raw 112k-feature wash set; `preview-plan` skips OSM fetch and AI.
- **Asset reuse:** layout edits never touch assets (sprites are scene-free and
  cached) — the editor is free to iterate; only `compose`/`repaint` cost money.

---

## 7. Open design choices (recommendations, not blockers)

1. **Editor coordinate space** — recommend **normalized frame space** authored
   client-side (§2.1); the alternative (poster-pixel with inverse-warp) is more
   code for no benefit.
2. **Render tech** — recommend **SVG** over a canvas lib (Konva/Pixi) until a
   profiled need appears; reuses the existing preview path, zero new deps.
3. **Warp-region model** — recommend **axis-aligned rectangles with a single
   `magnify` scalar** (maps cleanly onto the existing separable plateau bands);
   freeform/rotated regions are a later enhancement.
