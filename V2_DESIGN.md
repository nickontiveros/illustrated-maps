# V2 Design: Asset-Composition Architecture for Illustrated Map Generation

> A ground-up redesign proposal. V1 (the current codebase) treats the poster as one
> giant AI-generated image and fights the consequences. V2 treats the poster as a
> **composition of small AI-generated assets on a deterministic vector substrate**.

## Table of Contents

1. [Why V1 Failed — Root Cause Analysis](#1-why-v1-failed--root-cause-analysis)
2. [The V2 Design Principle](#2-the-v2-design-principle)
3. [Architecture Overview](#3-architecture-overview)
4. [Stage Details](#4-stage-details)
5. [How V2 Eliminates Each V1 Failure Mode](#5-how-v2-eliminates-each-v1-failure-mode)
6. [Alternatives Considered](#6-alternatives-considered)
7. [What to Reuse / Discard from V1](#7-what-to-reuse--discard-from-v1)
8. [Resolution & Cost Budget](#8-resolution--cost-budget)
9. [Risks and Mitigations](#9-risks-and-mitigations)
10. [Phased Implementation Plan](#10-phased-implementation-plan)

---

## 1. Why V1 Failed — Root Cause Analysis

V1 explored four generation modes (flat tiled, hierarchical, upscale, sectional). All
share one assumption: **the image model produces the map surface itself**. Every
failure traces back to that assumption colliding with two hard constraints of current
image models:

### Constraint A: Output resolution ceiling

Gemini image models emit roughly 1–2 megapixel images (~2048 px max dimension). An A1
poster at 300 DPI is 7016 × 9933 ≈ 70 megapixels — a **~35× gap**. One-shot generation
therefore produces a map where an entire city block is a few dozen pixels; no amount
of Real-ESRGAN upscaling adds the missing *illustration content* (individual buildings,
trees, boats, window lines). Upscalers sharpen edges; they do not invent detail.

### Constraint B: No cross-call global coherence

Each Gemini call is independent. When the canvas is split into tiles, the model
re-interprets the overlap geography differently each call: a road that one tile renders
as a cream 12-px ribbon, the neighbor renders as tan and 18 px wide, at a slightly
different position because "follow the reference exactly" is only approximately honored.
V1's countermeasures — style-reference tiles, histogram matching, 256-px overlaps,
seam-repair calls, hierarchical overview-guided refinement — all *reduce* drift but
cannot eliminate it, because the model has no mechanism to be pixel-consistent across
calls. The codebase documents this arms race: water-tile heuristics, "should rarely
need seam repair" hopes, a ~20% seam-repair budget per map.

### Compounding error: perspective applied after assembly

V1 assembles tiles edge-to-edge in flat (orthographic) space, then applies one global
perspective warp (`PerspectiveService`, post-assembly). A vertical seam at x = 1792 in
flat space maps to *different* x positions at the top versus the bottom of the warped
image — tile boundaries that were straight lines become non-collinear curves, so even a
perfectly repaired flat seam reopens as a visible warp discontinuity. Resampling during
the warp also softens exactly the detail the tiles were generated to provide.

### Meta-failure: patching symptoms

Five generation services, a seam-repair service, a blending service, a color-consistency
service, and outpainting all exist to mitigate consequences of the core architecture.
Each patch adds cost, latency, and new edge cases (e.g., rotation × perspective ordering,
water-tile detection heuristics). The architecture is the bug.

---

## 2. The V2 Design Principle

> **Code owns everything global. AI paints only things that are local and small.**

Invert the division of labor:

| Concern | V1 owner | V2 owner |
|---|---|---|
| Geography (roads, water, parks, coastline) | Gemini (via reference image) | **Vector engine** (OSM geometry) |
| Bird's-eye perspective | Raster warp after assembly | **Vector-space camera transform** |
| Exaggeration & distortion (roads 10× wide, fisheye, POI enlargement) | Prompt hints | **Vector engine** (explicit, parameterized) |
| Poster resolution (70 MP) | Tiling + upscaling | **Compositor renders natively at 300 DPI** |
| Style / palette | Prompt + histogram matching | **Style bible** (one set of AI assets) + procedural finish |
| Landmark illustrations | Gemini per landmark (kept) | Gemini per landmark (**kept — this part worked**) |
| Map fabric detail (generic buildings, trees, boats, cars) | Gemini tiles | **Sprite library + 2.5D building extrusion** |
| Text / labels | None (or AI garbling risk) | **Font rendering on curves** (never AI text) |
| Global mood / painterly unity | Per-tile prompts | **Low-frequency AI harmonization pass** |

This is also how human illustrators actually produce these maps: a pencil plan of
distorted geography, ink linework, flat washes, then individually drawn buildings and
lettering, composited in layers. V2 mechanizes that workflow and uses the image model
precisely where it is excellent — drawing *one charming building*, *one tileable water
texture*, *one compass rose* — and never where it is weak (global geometric coherence
at high resolution).

**The poster is a composition, not an image.**

---

## 3. Architecture Overview

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 1. INGEST            region bbox + POI list (+ photos)               │
 │    OSM roads/water/parks/buildings · DEM terrain · POI photo lookup  │
 └──────────────────────────────┬───────────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 2. MAP PLAN (deterministic, vector — the heart of V2)                │
 │    · simplify + smooth + exaggerate road network (per-class widths)  │
 │    · selective distortion (importance warp / fisheye / POI spacing)  │
 │    · oblique bird's-eye camera applied to ALL geometry               │
 │    · POI placement, scale tiers, collision resolution                │
 │    · label plan (curved street names, districts, POI callouts)      │
 │    OUTPUT: plan.json — a complete scene graph in poster pixel space  │
 │    + instant low-cost vector preview (no AI spend yet)               │
 └──────────────────────────────┬───────────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 3. ASSET STUDIO (AI, parallel, cached, individually retryable)       │
 │    · style bible: 1 hero swatch defining palette + brushwork         │
 │    · seamless ground textures: water, park, urban, sand, forest…     │
 │    · sprite library: generic houses, trees, boats, cars, people      │
 │    · POI illustrations: photo → oblique illustrated sprite, matted   │
 │    · ornaments: cartouche, compass rose, border corners              │
 └──────────────────────────────┬───────────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 4. COMPOSITOR (deterministic, full 300-DPI resolution)               │
 │    ground textures → painterly polygon edges → roads w/ wobble &     │
 │    taper → 2.5D extruded fabric buildings → scattered sprites →      │
 │    POI sprites + shadows → atmospheric depth gradient → labels →     │
 │    border/title/legend → paper grain                                 │
 └──────────────────────────────┬───────────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 5. HARMONIZE (AI, low-frequency only — optional)                     │
 │    downscale composite → Gemini painterly pass → blend back ONLY     │
 │    the low-frequency color/light; native detail untouched            │
 └──────────────────────────────┬───────────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 6. EXPORT  PNG @300 DPI · layered PSD · deep-zoom DZI               │
 └──────────────────────────────────────────────────────────────────────┘
```

The user-facing workflow mirrors the stages: **Plan → review (free) → Generate assets
→ review/regenerate individual assets (cheap) → Compose → done.** All expensive AI
work happens on small, independent, cacheable units.

---

## 4. Stage Details

### 4.1 Ingest

Largely reuses V1 services (`osm_service`, `terrain_service`, `satellite_service`,
`wikipedia_image_service`, `landmark_discovery_service`). Inputs:

- Region bbox (+ orientation: which cardinal direction is "up").
- POI list: name, lat/lon, optional photo, optional scale tier. Auto-discovery from
  OSM/Wikipedia remains available to seed the list.
- Output spec: poster dimensions and DPI.

### 4.2 Map Plan — the new core

A pure-geometry engine that converts geographic vectors into a **scene graph in final
poster pixel space**. Everything here is deterministic, instant to iterate on, and
previewable for free.

**4.2.1 Stylized geometry.** Per road class: Douglas-Peucker simplification, Chaikin/
spline smoothing, and width exaggeration (motorway 30 px, arterial 20 px, local 10 px
at poster scale — configurable presets; tourist-map roads are 5–50× true width per the
research in `RESEARCH.md` §Factor 4). Waterways widened similarly; minor roads in dense
areas pruned by importance so the map breathes. Coastlines and park polygons smoothed
into confident, hand-drawable shapes.

**4.2.2 Selective distortion.** Because geometry is vector, distortion is a coordinate
transform, applied once, exactly:
- *Importance warp*: expand space around POI clusters, compress empty stretches
  (V1's piecewise-linear `distortion_service` generalizes to a smooth thin-plate-spline
  or mesh warp driven by feature density).
- *Peripheral fisheye* (Yoshida-style) as an optional mode.
- *Minimum-spacing solver*: guarantee every POI has enough room for its illustration
  and label before any pixel is rendered.

**4.2.3 The camera.** The bird's-eye oblique view is a single projective transform
applied to all plan coordinates: y-axis foreshortening, top-edge convergence, horizon
margin. Since it's applied to *vectors before rendering*, there is no raster warp, no
resampling blur, and no possibility of seam misalignment — a road is a smooth curve in
poster space by construction. Depth (distance from viewer) becomes an attribute on
every element, driving sprite scale and atmospheric treatment later.

**4.2.4 Placement & labels.** POI anchor points project through distortion + camera;
collision is resolved in poster space (reuse `collision_service`). The label plan
assigns curved baselines along roads, district labels, POI callouts with leader lines
(reuse/extend `typography_service`), with a font hierarchy of hand-lettering typefaces.

**4.2.5 Output: `plan.json`** — the single contract between planning, asset generation,
and composition. Contains every polyline/polygon in poster pixels, every sprite slot
(position, footprint, depth, tier), every label (text, baseline curve, font, size), and
the asset manifest the Asset Studio must fill. The UI renders it as an SVG-style
preview so the user can adjust exaggeration, distortion, and POI placement **before
spending a single AI dollar**.

### 4.3 Asset Studio — all the AI, none of the seams

Every asset is small (≤2048 px), independent, cached by content hash, and individually
regenerable. Consistency comes from one shared **style bible**.

**4.3.1 Style bible.** Generate one hero swatch — a small imaginary map corner in the
target style (e.g., "vintage tourist, warm muted palette") — iterate until the user
approves it. Extract its palette (reuse `palette_service`). The swatch image is then
attached as a style reference to *every subsequent asset call*, and the palette drives
the compositor's procedural colors. One source of truth for style.

**4.3.2 Ground textures.** Per land-cover class (water, park, urban block fill, sand,
forest, farmland): one seamlessly tileable texture at 1024–2048 px, requested in the
bible's palette ("seamless tileable watercolor wash of calm sea water, soft blue-green,
subtle wave strokes"). Tileability is verified programmatically (offset-wrap test) and
enforced with cross-fade edge correction. Because polygons are *filled* with these
textures at compose time, ground detail is unlimited-resolution and perfectly uniform
in style.

**4.3.3 Sprite library.** Sprite *sheets* of generic fabric elements — houses, shops,
mid-rises, trees (3–4 species), boats, cars, buses, people — requested as "6 variations,
aerial oblique view from the south at ~40°, sun from the northwest, flat magenta
background". Solid-key backgrounds make matting trivial and reliable. Cut into
individual sprites with alpha. ~10–20 calls total yields a library of 60–120 sprites
that the compositor scatters by the thousands.

**4.3.4 POI illustrations (the marquee feature).** For each POI: photo (user-supplied
or Wikipedia) → Gemini → illustrated building in the bible style, same camera
("aerial oblique, ~40°, facades visible, sun from northwest"), same flat-key background
→ matte → sprite at 1024–2048 px. On the poster a hero landmark occupies ~400–1200 px,
so a 2048-px sprite is *more* than print density — this is where V2 banks its detail
win. Each POI is reviewable and regenerable in isolation for ~$0.13. V1's landmark
pipeline already proved this works; V2 standardizes the camera, lighting, and keying so
sprites composite cleanly.

**4.3.5 Ornaments.** Cartouche frame, compass rose, border corners, legend icons —
each a single small AI call in bible style, on flat key. All *text* in the cartouche
and legend is font-rendered by the compositor, never AI-drawn.

### 4.4 Compositor — deterministic 300-DPI rendering

Renders `plan.json` + assets to the full poster, bottom layer up. Internally it may
render in chunks for memory, but chunked rendering of deterministic graphics is exact —
**there are no visible seams because adjacent chunks compute identical pixels at their
boundary**.

Layer stack (each kept separate for PSD export):

1. **Ground**: land-cover polygons filled with tiled textures; polygon boundaries get
   painterly edge treatment (darkened watercolor edge line, 1–3 px wobble displacement)
   so fills read as hand-washed shapes, not GIS polygons.
2. **Water effects**: shore darkening gradient, procedural wave/ripple strokes along
   coasts, illustrated flow strokes on rivers.
3. **Roads**: stroked with width per class, organic wobble (low-amplitude noise
   displacement — reuse the idea in `road_style_service`), tapered/rounded
   intersections, casing color per class.
4. **Urban fabric**: OSM building footprints extruded to 2.5D blocks under the plan
   camera (roof + two visible facades, palette-shaded, wobbled outlines) for dense
   authentic massing; sprite-library buildings/trees scattered along blocks and in
   parks where footprints are sparse. Sprites scale with depth attribute.
5. **POI sprites**: alpha-composited at plan positions with soft drop shadows and
   ground-contact anchoring; scale = tier × depth.
6. **Atmospheric perspective**: vertical gradient — desaturate, lighten, blue-shift,
   reduce contrast toward the horizon; optional haze band (reuse `atmosphere_service`).
7. **Labels**: hand-lettering fonts on curved baselines, halo/offset for legibility;
   street names along roads, district names, POI callouts.
8. **Frame**: decorative border with AI ornaments, font-rendered title, legend,
   credits (reuse/extend `border_service`).
9. **Finish**: global paper-grain overlay and subtle ink-bleed filter — shared noise
   across the whole poster is itself a powerful unifier of style.

### 4.5 Harmonize — AI for mood, at a frequency where it cannot break anything

Optional final pass that buys painterly global light without reintroducing seams:

1. Downscale the composite to ~2048 px.
2. One Gemini img2img call: "repaint as a unified hand-painted poster, preserve all
   shapes exactly" — at this scale the model is good at global color/light.
3. Upscale the result and **blend only its low-frequency component** (Gaussian-split,
   e.g. blur radius ≈ 40 px at poster scale) over the native-resolution composite.

Color and lighting mood come from the AI; every crisp detail (linework, sprites, text)
comes from the native render. Misalignment is invisible because only frequencies far
below the misalignment scale are taken from the AI image. Strength is a user dial
(0 = off).

### 4.6 Export

PNG at print resolution, layered PSD, and DZI deep-zoom for the web viewer (reuse
`dzi_service`).

**Layered PSD** (`mapgen v2 layered`, `pipeline.compose_layered`). The layer stack above
maps to PSD layers, but with a deliberate split for hand-editing: the hard-to-edit
surface — ground/water textures, the road network, 2.5D buildings and the atmospheric
haze (layers 1–4, 6) — is pre-flattened into a single **Base** layer, while everything a
user tweaks by hand is peeled into its own named, positioned layer: one per scatter sprite
*kind*, one per POI sprite, the POI leader lines, one per text label, and the frame. So you
can open the file in Photoshop/GIMP/Affinity and move the Empire State Building sprite,
retype a street name, or hide all the trees, without disturbing the painted base. Paper
grain is omitted (a global finish best re-applied to a flattened copy), so the canonical
print output stays `poster.png`. `Compositor.render_layers()` produces the stack;
`compose/psd_writer.py` writes the `.psd`.

> Implementation note: the V1 API path used `pytoshop` (`services/psd_service.py`), but its
> legacy `setup.py` no longer builds against modern setuptools. The V2 export therefore
> ships a small dependency-free PSD writer (8-bit RGB, RLE/raw channels, Unicode layer
> names) that round-trips through `psd-tools`.

---

## 5. How V2 Eliminates Each V1 Failure Mode

| V1 failure | Mechanism in V1 | V2 answer |
|---|---|---|
| Insufficient poster detail | Gemini's ~2 MP ceiling vs 70 MP poster | Compositor renders natively at 300 DPI; vector linework and fonts are resolution-independent; sprites generated at ≥ their printed size |
| Tile seams | Independent calls disagree about overlap pixels | There are no AI tiles. The only chunking is deterministic rendering, which is exact |
| Perspective warping at seams | Raster warp after edge-to-edge assembly bends seam lines | Camera is a vector transform in the plan; rendering happens directly in poster space; nothing is warped after rendering |
| Style drift across map | Per-call reinterpretation; histogram patching | One style bible referenced by every asset call + shared procedural palette + global finish layers; drift surface is per-sprite and individually fixable |
| Garbled/absent text | AI text unreliable; V1 had no labels | All text font-rendered on curves; AI never draws text |
| Expensive, coarse iteration | Bad tile ⇒ regenerate + seam-repair cascade | Bad asset ⇒ regenerate that one sprite ($0.13); plan changes are free |
| Geography hallucination | "Follow the reference exactly" is approximate | Geography never passes through the model; it is drawn from OSM vectors |
| POI placement imprecision | POIs composited onto AI tiles that may disagree with GPS→pixel math | Plan owns placement; the ground truth and the render share one coordinate system |

The requirement statement — *region + POIs in, each POI illustrated/labeled/placed,
poster-density output, exaggerated stylized geography* — maps directly onto plan
(placement, exaggeration), asset studio (illustration), compositor (labels, density).

---

## 6. Alternatives Considered

**A. Keep tiling, apply perspective per-tile before generation.** Fixes the warp-seam
interaction but not Constraint B: tiles still disagree in style and overlap content,
seam repair remains, and per-tile projective reference warping confuses the model
further. Rejected — patches one symptom of the wrong architecture.

**B. Seamless latent-space tiling (MultiDiffusion / SyncDiffusion / Mixture-of-
Diffusers) + ControlNet on the vector linework, using open-weights models (SDXL/Flux).**
Technically the strongest "whole-surface AI" option: shared-latent tiling genuinely
eliminates seams, and ControlNet would honor the vector geography far better than
prompt-pleading. Rejected as the *core* for V2 because it requires self-hosted GPU
inference (cost/ops), Gemini's API exposes no latent access, and style control is
weaker than the asset approach. **Kept as an upgrade path**: the V2 ground layer
(stage 4.4 layers 1–3) could later be re-rendered by a ControlNet-conditioned tiled
diffusion for an even more painterly base — sprites, labels, and frame composite on
top unchanged. The plan/compose architecture is agnostic to how the ground texture is
produced.

**C. One-shot + smarter upscaling (e.g., diffusion-based detail-adding upscalers,
"creative upscale").** Detail-hallucinating upscalers operate tiled internally and
reintroduce the coherence problem; they also invent geography. Acceptable for casual
output, not for a labeled, POI-accurate poster. Rejected.

**D. Pure procedural/NPR rendering, no AI.** Stylized-cartography renderers can do
wobble lines and washes, but the charm of these posters lives in the illustrated
landmarks and organic textures — exactly what AI provides cheaply. V2 is this option
*plus* AI assets where they matter.

---

## 7. What to Reuse / Discard from V1

**Reuse (mostly as-is):** `osm_service`, `terrain_service`, `satellite_service`,
`geo_utils`, `wikipedia_image_service`, `landmark_discovery_service`,
`palette_service`, `collision_service`, `psd_service`, `dzi_service`, FastAPI + React
shell, WebSocket progress, project.yaml persistence, caching/retry patterns.

**Reuse with refactoring:** `gemini_service` (becomes the Asset Studio client: style-
bible reference plumbing, flat-key matting, tileability verification),
`typography_service` (curved baselines, hierarchy), `border_service` (AI ornament
slots), `road_style_service` (wobble/taper into the compositor), `atmosphere_service`
(depth gradient), `distortion_service` (generalize to smooth mesh warp inside the
plan engine), `composition_service` (sprite placement math).

**Discard:** `generation_service`, `hierarchical_generation_service`,
`upscale_generation_service`, `sectional_generation_service`, `seam_repair_service`,
`blending_service`, `outpainting_service`, `color_consistency_service`,
`perspective_service` as a raster warp (reborn as the vector camera), and the
seam-repair UI. These exist to patch the architecture V2 removes.

**New components:** plan engine (geometry stylization, distortion, camera, placement,
label planning, `plan.json`), asset studio orchestration (manifest, matting, tileable
verification, review states), compositor (layered 300-DPI renderer), harmonizer
(frequency-split blend), plan-preview UI.

---

## 8. Resolution & Cost Budget

Poster: A1 @ 300 DPI = 7016 × 9933 (≈70 MP). Hero POI on poster ≈ 400–1200 px →
2048-px sprite = ~2–5× oversampled. Fabric sprites ≈ 100–300 px on poster → sheet cuts
at 300–500 px suffice. Ground textures tile, so density is unbounded.

| Asset class | Calls | Est. cost @ $0.13 |
|---|---|---|
| Style bible (with 2–3 iterations) | 3 | $0.39 |
| Ground textures (~8 classes) | 8 | $1.04 |
| Sprite sheets | 12 | $1.56 |
| POI illustrations (20 POIs) | 20 | $2.60 |
| Ornaments (cartouche, compass, corners, legend) | 5 | $0.65 |
| Harmonization pass | 1 | $0.13 |
| Retry/regeneration headroom (~25%) | ~12 | $1.56 |
| **Total (typical 20-POI map)** | **~61** | **≈ $8** |

Comparable to V1's ~$5–6 *before* V1's seam-repair and full-map regeneration cycles —
and V2 retries are always single small assets, never cascades. Compute: compositor is
CPU/Pillow-or-Skia work; a 70 MP layered render fits comfortably in a few GB with
chunked rendering.

---

## 9. Risks and Mitigations

1. **Ground plane looks "rendered", not hand-painted.** Highest aesthetic risk.
   Mitigations, in order: AI textures (not flat fills) + painterly polygon-edge
   treatment + line wobble + paper grain + harmonization pass; if still short, the
   ControlNet tiled-diffusion ground upgrade (§6B). *De-risked first — Phase 0.*
2. **Sprite style coherence.** Mitigated by the style bible reference in every call,
   fixed camera/light wording, palette post-check (ΔE against bible palette, auto-flag
   outliers), and cheap per-sprite regeneration. Residual variety reads as hand-drawn
   charm rather than seam error because sprites are discrete objects.
3. **Matting quality.** Flat-key backgrounds make this mostly trivial; add despill +
   alpha feathering; fall back to an off-the-shelf matting model for stubborn cases.
4. **2.5D extrusion looks too CAD-like.** Wobble outlines, palette shading, mixing
   extrusions with sprite buildings, and harmonization all soften it; density can be
   dialed down in favor of sprites per-district.
5. **Plan engine scope creep** (label placement and distortion are deep fields).
   Ship v2.0 with greedy label placement + fixed presets; the plan.json contract lets
   each subsystem improve independently later.
6. **Model behavior drift** (prompt phrasing, keying compliance). All asset prompts
   centralized with golden-image regression tests (hash-stable stub in CI, visual
   checklist for live runs) — V1's e2e stub-service pattern carries over well.

---

## 10. Phased Implementation Plan

**Phase 0 — Look validation spike (1 region, mostly manual).** Hand-assemble one
poster section (~2000 px square) for a real neighborhood using the exact V2 recipe:
OSM vectors → stylized/oblique linework → AI textures + a few sprites + 2 POI
illustrations → procedural finish → harmonization. *Gate: does it look like a tourist
pamphlet map? This validates the aesthetic before any framework code.*

**Phase 1 — Plan engine.** Geometry stylization, distortion, vector camera, POI
placement, label plan, `plan.json`, SVG preview endpoint + minimal UI. *Gate: free,
instant, attractive linework preview for any region; POIs never collide.*

**Phase 2 — Asset studio.** Style bible flow, texture generation with tileability
verification, sprite sheets + matting, POI illustration pipeline, manifest/cache,
review-and-regenerate UI. *Gate: full asset set for a 20-POI region, every asset
individually approvable.*

**Phase 3 — Compositor.** Layered 300-DPI renderer (chunked), all nine layers,
PNG/PSD/DZI export. *Gate: full A1 poster from plan + assets; zoom to 100% anywhere
and find crisp, intentional detail.*

**Phase 4 — Harmonizer + polish.** Frequency-split mood pass, atmospheric tuning,
border/cartouche integration, finish filters. *Gate: side-by-side with reference
pamphlet maps passes a squint test.*

**Phase 5 — Product workflow.** Rebuild the web UI around Plan → Assets → Compose;
cost preview per stage; delete the V1 generation/seam-repair surfaces and dead
services. *Gate: a new user goes from region + POI list to printable poster without
touching the CLI.*

Phases 1–3 are the critical path; 0 is cheap insurance and should happen first.
