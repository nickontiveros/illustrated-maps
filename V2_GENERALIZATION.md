# V2 Generalization Audit: the Arizona Case Study

Run artifacts: `projects/az_v2/` (Phoenix + Tucson, 16 POIs, southwest style,
region 268 x 410 km — roughly 4,000x the area of the Manhattan example).
Produced 2026-06-11 with the stock pipeline, no code changes. Everything below
is an observed failure or measurement from that run, followed by
recommendations. Short version: **the V2 pipeline is sound for a single dense
city at ~10 km scale and degrades at every stage when the region is a sparse
multi-city state.**

## What happened, stage by stage

### Plan / ingest

1. **One-size OSM fetch.** `pipeline.fetch_source` calls
   `OSMService.fetch_region_data(bbox)` with the default `detail_level="full"`.
   V1 already has `simplified` / `regional` / `country` tiers keyed to region
   area; V2 never selects them. Result: osmnx warned the bbox is **44x the max
   Overpass query area**, the roads fetch took ~15 min / 860 MB cache /
   667,291 features, and then **Overpass rate-limited every subsequent layer**
   — buildings timed out; water, parks, terrain, railways, washes all got
   connection-refused.
2. **Silent data loss.** Each failed layer prints a `Warning:` and returns
   nothing, and the plan stage happily wrote a "successful" plan with **0
   ground polygons** — no water, no parks, hence also **0 scatter sprites**.
   The same thing happened in the earlier `projects/phoenix` run, so it is the
   expected behavior at this region size, not a transient. A tourist map of
   Arizona with no Salt River reservoirs, no parks and no terrain shipped as
   `state: done`.
3. **Road pruning doesn't scale.** `prune_roads` caps only LOCAL/PATH (at 400);
   everything `secondary` and up is always kept. The Arizona plan retains
   **125,183 roads (101,833 "secondary")**. Consequences: `plan.json` is 43 MB,
   `preview.svg` is 40 MB (the "free, instant" preview is unloadable in a
   browser), and metro Phoenix renders as a solid circuit-board mat of
   exaggerated road ribbons (width fractions are tuned for a city: a secondary
   road gets 0.42% of poster width, which at this scale is a ~1.7 km-wide
   ribbon in ground units).
4. **No place names.** `from_osm_data` never populates `source.places`, so
   district/water labels don't exist at all. A two-city map has no "PHOENIX"
   or "TUCSON" anywhere. (V1's OSMData also fetches `railways` and `washes` —
   desert arroyos! — which V2 ingest silently drops.)
5. **Street label selection is degenerate at scale.** Greedy priority
   `0.3 + 0.02 * len(points)` favors long rural ways: of all Arizona roads the
   two street labels placed were **"Indian Route 34" and "Road 600"**.
   Meanwhile three POI labels (Arizona State University, Salt River,
   Children's Museum) were dropped to collisions inside the scrambled cluster.

### Layout (the user-visible composition)

6. **POI clustering breaks placement** *(questions 1 and 5)*. Twelve of the
   sixteen POIs occupy 0.116 x 0.028 of the normalized map (~31 x 11 km of
   metro Phoenix). The ImportanceWarp (strength 0.5, sigma 0.18) expands that
   to 0.205 x 0.047 — about 2x, but the twelve sprites need ~7x. The pairwise
   overlap solver then shoves sprites up to **2,615 px** (Phoenix Zoo) from
   their anchors — over a third of the poster width — and destroys relative
   geography: Heard Museum (x=299) and Phoenix Art Museum (x=3552) are two
   blocks apart in reality. Sprites end up anchored on empty desert with no
   relationship to the road network under them.
7. **Aspect and orientation** *(questions 3 and 4)*. `geo_to_normalized` maps
   bbox degrees straight onto the canvas: no cos(latitude) correction (E-W is
   stretched ~1.08x here; worse at higher latitudes) and no fit logic — any
   bbox is warped to A1 regardless of its shape. There is also no rotation
   support anywhere in V2 (the V1 Arizona config used
   `orientation_degrees: 340`). Measured: the POI footprint is h/w = 1.61
   north-up, vs the A1 canvas's 1.416; rotated -20° to -25° (NW up) it becomes
   1.28–1.33, a far better fit, with the Phoenix→Tucson axis running down the
   poster diagonal.

### Assets

8. **Non-building landmarks** *(question 2)*. V2's `PoiConfig` is
   `{name, lat, lon, tier, photo}` — V1's `feature_type`
   (monument/park/river/mountain/campus/airport), `path_coordinates` (river
   polylines!) and `horizon_feature` were all dropped. Every POI goes through
   the single POI_SPRITE prompt, which explicitly coerces: *"If the subject is
   a street, district, or area rather than a single building, draw only its
   single most iconic building instead."* Observed results: **Salt River
   rendered as an old-west "GENERAL STORE" building** (with signage);
   Petrified Forest National Park as a generic adobe pueblo; Papago Park as a
   ghostly half-matted butte. Ironically the types and compositor already
   support `RoadClass.RIVER/STREAM` waterways with proper ribbon rendering —
   ingest just never emits them.
9. **Scene vocabulary is hard-coded coastal-city** *(question 6)*.
   `ScatterKind` = {TREE, HOUSE, BOAT}; the style-bible prompt hard-codes
   *"Show a little coastline"* — our desert style bible contains **an ocean
   with sailboats**, and it is the style reference attached to every other
   asset call. No desert ground class (SAND's hint is beach-flavored "dry
   sandy ground"), no cactus/mesa/rock scatter, nothing to fill empty land
   (scatter only spawns inside park/forest/water/urban polygons, never on the
   base land plate — Manhattan had no empty land, Arizona is mostly empty
   land).
10. **Asset robustness.** A 0-byte reference photo aborted the entire asset
    batch mid-run (no pre-flight validation, no skip-and-continue), after
    spending two paid calls. `AssetSpec.content_hash` covers the photo *path*,
    not its content. White-roofed subjects (State Farm Stadium) come out
    half-transparent from magenta keying and invisible against parchment.
11. **What did generalize:** the style override. `style.description` +
    palette carried a convincing southwest look into the style bible, land
    texture, and all 16 sprites (palette ΔE scores 13–24, no outliers);
    Biosphere 2's glass pyramids, Camelback's red rocks and Wupatki's ruins
    are charming and on-style. The repaint stage also generalizes by
    construction: its grid is canvas-sized, so it stays 45 calls / ~$5.85
    per poster for any region, and its prompt is style-parameterized.

## Recommendations, mapped to the six questions

1. **Sparse in-between areas** — three layers of fix:
   - Region-aware ingest: choose detail level from bbox area (the tiers exist
     in V1); per-layer retry/backoff; **fail the plan stage loudly** (or mark
     `plan.json` with provenance warnings surfaced in the UI) when a layer
     returns nothing.
   - Ink budget: prune roads by class+length against a per-poster ink budget
     instead of "keep all secondary+". At state scale that means
     motorway/primary only, with secondary kept only near POI clusters.
   - Fill the desert with *art*, not data: terrain texture regions (mesa/butte
     silhouettes, saguaro and wash scatter on bare land), and a `SCATTER_FOR_LAND`
     concept so empty ground gets style-appropriate set dressing.
2. **Rivers/forests/mountains as landmarks** — restore `feature_type` on
   `PoiConfig` and branch representation: `river` → RIVER waterway polyline
   (from `path_coordinates` or an OSM way lookup) + curved water label, never
   a sprite; `park/forest` → ground polygon + texture + scatter + label;
   `mountain` → wide horizon-style silhouette sprite; `area/campus` → either a
   boundary tint or a multi-building vignette sprite. The plan types already
   support most of this.
3. **Region shape vs A1** — never stretch: apply cos(lat) correction, then fit
   the (rotated) region into the canvas with letterboxing absorbed by a wider
   decorative frame / title band / legend, the way vintage posters actually
   handle it. The camera's `horizon_margin` is already a primitive for
   reserving non-map canvas.
4. **Orientation** — add `rotation_deg` to the project (applied in flat space
   before the camera), with an "auto" mode that PCA-fits the POI cloud to the
   canvas aspect. For this map: **rotate ~20–25° (NW up)**; measured POI
   extent goes from h/w 1.61 to 1.28–1.33 against a 1.416 canvas, and the
   Phoenix→Tucson corridor becomes the poster's diagonal spine.
5. **Artistic spacing of far-out landmarks** — yes, but by warping space, not
   by lying about anchors: raise warp strength / use per-cluster sigma so the
   warp (which moves roads and POIs together, preserving local truth) absorbs
   the spacing, and iterate strength until the overlap solver's residual
   nudges fall under a threshold (e.g. 1% of poster width). The current
   solver's silent 2,600 px shoves should be an error, not a default. For
   hyper-dense clusters (11 POIs in central Phoenix), the genre solution is an
   **inset panel** ("Downtown Phoenix" magnifier bubble) — V1's
   `sectional_layout` was exactly this and V2 dropped it.
6. **Style-appropriate textures/sprites** — make the scene vocabulary part of
   the style preset: each preset carries its scatter kinds (southwest:
   saguaro, ocotillo, mesa, rock outcrop, adobe house; no boats), its ground
   classes (desert), its style-bible scene description (no hard-coded
   coastline), and ornament motifs. Boats are already data-driven (only
   generated when WATER polygons exist) — the bug is that the *style bible*
   prompt forces a coastline even when the region has none.

## Repaint stage on sparse terrain (observed after the full $5.85 run)

The 45-call repaint completed and the painterly texture it adds is genuinely
nice (the corridor now has watercolor wash brushwork), but two NYC-density
assumptions surfaced:

- **Per-window tone drift is glaring on flat desert.** The recipe is seam-free
  for *content* edges, and on dense city imagery small color drift between
  windows hides in the detail. On windows that are 90% bare terracotta, each
  call settles on a slightly different wash tone and the 512 px quadrant grid
  reads as a faint patchwork across the whole poster. `color_norm.unify_water`
  exists precisely because water showed this on NYC — the desert needs the
  same treatment for the LAND class (or a global low-frequency tone
  normalization against the guide).
- **Blank windows invite invention.** Despite the "NEVER invent content"
  clause, several near-empty windows came back with invented terrain —
  mesa/canyon vignettes along the poster edges and a dark lake-like blob in
  the northeast that does not exist in the guide. With nothing to anchor on,
  the model fills the vacuum with style-bible scenery. Mostly-blank windows
  should either be skipped (the quadrant store already has a SKIPPED status
  for blank cells — its blankness heuristic evidently doesn't trigger on
  textured land fill) or pinned harder via a higher guide-fidelity prompt.

## Cross-cutting fixes worth doing regardless

- Validate POI reference photos before spending API calls; skip+warn on
  per-asset failures instead of aborting the batch.
- Hash photo content (not path) into the asset cache key.
- Cap plan.json/preview.svg size (the preview should subsample to a few
  thousand features — it's a layout review tool, not an archive).
- Populate `source.places` (city/town names) — a map with no city labels
  fails at any scale, it just happened not to matter for single-city posters.

## Status: fixes implemented (2026-06-11)

Everything above is now addressed in code:

- **Ingest**: area-based detail tiers (`pipeline.DETAIL_TIERS`), per-layer
  provenance + warnings in `plan.json` (surfaced by CLI/UI), automatic
  cool-down retry for Overpass-refused layers, waterway-line rivers
  (`OSMService.extract_rivers`), washes/railways/terrain ingestion, city
  names into `source.places`.
- **Layout**: `ingest.GeoFrame` — cos(lat)-true metric mapping,
  `rotation_deg` (incl. `"auto"` PCA fit), aspect fit by *extending* the
  covered region (never stretch/letterbox); adaptive importance warp with
  cluster-sized sigma that escalates strength then shrinks sprites until
  overlap-solver residuals are honest; road pruning by ink budget
  (`stylize.ROAD_INK_BUDGET`); preview SVG feature caps.
- **POIs**: `feature_type` on `PoiConfig` (river → RIVER ribbon + water
  label, never a sprite; mountain/park/campus/etc. get per-type prompts).
- **Assets**: key-family matting gate (palette-shifted keys can't eat white
  subjects), key-shift flagging, `assets --reprocess` (re-matte from raw,
  no API spend), aspect-preserving sprite fit, photo validation +
  continue-on-error, photo-content cache keys.
- **Style**: preset registry (`v2/styles.py`) carrying scene vocabulary —
  `southwest_desert` ships cactus/shrub/rock land scatter, no boats,
  desert texture hints, and a non-coastline style-bible scene.
- **Labels/title**: bigger POI labels on paper plates, never dropped;
  street labels picked by class; banner-capped cartouche; `title` is
  editable post-plan via `mapgen v2 retitle` / `PATCH .../title` / the UI
  pencil (patches plan.json in place, recompose only).
- **Repaint**: blur-based featureless-window skip, invention guard for
  flat windows, post-stitch low-frequency tone match
  (`color_norm.match_low_frequency`) killing quadrant patchwork.
- **Memory** (WSL OOM hardening): per-polygon bbox-local ground rendering,
  striped grain/tone/water passes, low-res repaint gain field, API-side
  parsed-plan cache; heavy stages run under `systemd-run -p MemoryMax`.

Coverage: `tests/v2/test_generalization.py`.
