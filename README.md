# Illustrated Map Generator

Generate poster-size illustrated maps in the style of hand-illustrated tourist maps. Pick a region and a handful of landmarks; get a print-ready A1 poster with bird's-eye perspective, AI-illustrated landmarks, hand-lettered labels, and a painterly finish.

## How it works

The generator treats the poster as a **composition of small AI-generated assets on a deterministic vector substrate** (see [V2_DESIGN.md](V2_DESIGN.md) for the full design):

> **Code owns everything global. AI paints only things that are local and small.**

| Concern | Owner |
|---|---|
| Geography (roads, water, parks, buildings) | Vector engine (live OSM data) |
| Bird's-eye perspective & artistic distortion | Vector-space camera transform |
| Poster resolution (70 MP at print DPI) | Compositor renders natively |
| Landmark illustrations, textures, ornaments | Gemini — one small image at a time |
| Text & labels | Font rendering on curves (never AI text) |

This runs as a three-stage pipeline, each stage independently re-runnable:

1. **Plan** (free) — fetch OSM geometry, stylize and distort it, place POIs and labels, emit `plan.json` + an SVG preview
2. **Assets** (AI, cached) — generate the style bible, ground textures, sprite sheets, POI landmarks, and ornaments; each asset is cached by content hash, so edits only regenerate what changed
3. **Compose** (deterministic) — render all layers at print resolution: paper, textures, roads, 2.5D buildings, sprites, landmarks, labels, frame

A typical map costs **~15 Gemini image calls** (one per asset), regardless of poster size. A `--stub` mode runs the whole pipeline offline with procedural placeholder art — useful for layout review and CI.

## Installation

### Prerequisites

- Python 3.10+
- A Google Gemini API key (for real asset generation; stub mode needs none)

### CLI only

```bash
git clone https://github.com/nickontiveros/illustrated-maps.git
cd illustrated-maps
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

export GOOGLE_API_KEY="your-google-api-key"
```

### Full stack (CLI + web UI)

```bash
# Python environment as above, then:
pip install -e ".[api]"

# Frontend (requires Node.js 18+)
cd frontend && npm install && cd ..
```

## Quick start

### CLI

A project is a directory with one `project.yaml`:

```yaml
name: "Lower Manhattan"
region:
  north: 40.7350
  south: 40.6950
  east: -73.9850
  west: -74.0250
output:
  width_px: 7016    # A1 @ 300 DPI
  height_px: 9933
  dpi: 300
camera:
  convergence: 0.78      # top-edge narrowing for bird's-eye view
  vertical_scale: 0.55   # vertical compression toward the horizon
  horizon_margin: 0.06
style: vintage_tourist
distortion_strength: 0.5
pois:
  - { name: "Statue of Liberty", lat: 40.6892, lon: -74.0445, tier: 1 }
  - { name: "Brooklyn Bridge",   lat: 40.7061, lon: -73.9969, tier: 1 }
  - { name: "Wall Street",       lat: 40.7066, lon: -74.0090, tier: 2 }
```

Then run the stages (see `projects/v2_example/` for a complete example):

```bash
mapgen v2 plan projects/my_map        # plan.json + preview.svg (free, no AI)
mapgen v2 assets projects/my_map      # AI asset generation (cached)
mapgen v2 compose projects/my_map     # render the poster

# Or everything at once:
mapgen v2 generate projects/my_map

# Iterate cheaply:
mapgen v2 assets projects/my_map --stub          # offline placeholder art, no API key
mapgen v2 assets projects/my_map --only poi_wall_street --force   # redo one asset
mapgen v2 compose projects/my_map --scale 0.25   # quick low-res preview render
mapgen v2 compose projects/my_map --harmonize    # painterly mood pass (1 extra AI call)
```

### Web UI

```bash
# Terminal 1: API backend
uvicorn mapgen.api.main:app --reload --port 8000

# Terminal 2: frontend
cd frontend && npm run dev
```

Open http://localhost:5173:

- **Create a map** — title, region bounds, and POIs (name, coordinates, importance tier)
- **Manage POIs after creation** — add, edit, or remove points of interest at any time; the UI flags the plan as stale until you re-plan, and only new/changed assets are regenerated
- **Run the three stages** with live progress: Plan (free) → Assets (AI or stub) → Compose (pick render scale, optional harmonize pass)
- **Review everything** — SVG plan preview, per-asset gallery with one-click regeneration, and the final poster
- The Gemini key can be supplied via the backend `GOOGLE_API_KEY` env var or per-request from the browser (stored in localStorage, sent as `X-Google-API-Key`)

POI tiers control prominence on the map: **Hero** (tier 1) landmarks render large, **Major** (tier 2) medium, **Minor** (tier 3) small.

## V2 CLI reference

| Command | Description |
|---------|-------------|
| `mapgen v2 plan PROJECT` | Build `plan.json` + `preview.svg` from live OSM data (free) |
| `mapgen v2 assets PROJECT` | Generate all assets in the plan manifest (cached by content hash) |
| `mapgen v2 compose PROJECT` | Render the poster from plan + assets |
| `mapgen v2 generate PROJECT` | Full pipeline: plan → assets → compose |

Useful flags: `--stub` (offline procedural assets, no cost), `--force` (ignore cache), `--only ID` (regenerate specific assets), `--scale 0.25` (preview-quality render), `--harmonize` (low-frequency AI mood pass).

## Cost

| Operation | Gemini calls |
|-----------|--------------|
| Plan stage | 0 (free) |
| Assets, typical map | ~14 (1 style bible + 3 textures + 2 sprite sheets + 6 POIs + 2 ornaments) |
| Re-render / re-plan after edits | 0 (assets are cached) |
| Adding one POI | 1 |
| Harmonize pass | 1 |

Raw generations are kept under `assets/raw/`, so post-processing improvements re-apply without new API calls.

## Project layout

```
projects/my_map/
├── project.yaml      # the product: region + POIs + output size
├── plan.json         # built by the plan stage
├── preview.svg       # free layout preview
├── assets/           # generated assets (cached) + raw/ originals
├── cache/            # OSM data cache
└── poster.png        # final render
```

## Development

```bash
# V2 test suite (offline; uses synthetic geometry + stub assets)
python -m pytest tests/v2 -v

# Full test suite
python -m pytest tests/ -v

# Frontend type check
cd frontend && npx tsc --noEmit
```

## V1 (legacy tile pipeline)

The original pipeline generated the map surface itself with AI, tile by tile, and fought the consequences — seam repair, histogram matching, overlap blending, per-tile style drift. It remains available (`mapgen init`, `mapgen generate-tiles`, `mapgen outpaint`, …, and the web UI under `/v1`) but is superseded by V2. [V2_DESIGN.md](V2_DESIGN.md) documents the failure analysis and the architectural reasoning in depth.

## License

MIT License — see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.
