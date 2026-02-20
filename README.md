# Illustrated Map Generator

Generate poster-size illustrated maps in the style of hand illustrated tourist maps. Transform any geographic region into a beautiful hand-painted style illustration with AI-powered generation.

## Features

- **AI-Powered Illustration**: Uses Google Gemini to transform satellite imagery and map data into illustrated style
- **High Resolution Output**: Generates A1 poster size (7016 x 9933 px @ 300 DPI)
- **Landmark Integration**: Add custom landmarks with photos that get illustrated to match the map style
- **Landmark Discovery**: Auto-discover notable landmarks from OpenStreetMap data
- **Perspective Transform**: Creates an aerial/isometric view with horizon and atmospheric perspective
- **Typography & Labels**: Automatic road, water, park, and district labels with hand-drawn styling
- **Color Palettes**: Preset palettes (vintage tourist, modern pop, ink wash) with enforcement and consistency across tiles
- **Road Styling**: Exaggerated, hand-drawn road rendering with wobble effects and presets
- **Decorative Borders**: Four border styles (vintage scroll, art deco, modern minimal, ornate victorian) with title, compass, and legend
- **Atmospheric Perspective**: Depth-based haze, contrast reduction, and fog layers
- **Terrain Exaggeration**: Adjustable terrain height emphasis for dramatic relief
- **Layered PSD Export**: Export as layered Photoshop file for post-editing
- **Tiled Generation**: Handles large maps by generating overlapping tiles and blending seamlessly
- **Flexible Orientation**: Set any cardinal direction (north, south, east, west) as "up" on the map
- **Web UI**: Full-stack React frontend for interactive project management and generation
- **Cache Management**: Clear cached tiles to force regeneration after prompt or setting changes

## Installation

### Prerequisites

- Python 3.10+
- Google Cloud account with Gemini API access

### Option 1: CLI Only (Recommended for most users)

```bash
# Clone the repository
git clone https://github.com/nickontiveros/illustrated-maps.git
cd illustrated-maps

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install CLI package
pip install -e .

# Set up API keys
export GOOGLE_API_KEY="your-google-api-key"
export MAPBOX_ACCESS_TOKEN="your-mapbox-token"  # For satellite imagery
```

### Option 2: Full Stack (CLI + Web UI)

Includes a visual web interface for interactive map generation.

```bash
# Clone and set up Python environment (same as above)
git clone https://github.com/nickontiveros/illustrated-maps.git
cd illustrated-maps
python -m venv .venv
source .venv/bin/activate

# Install CLI + API dependencies
pip install -e ".[api]"

# Install frontend (requires Node.js 18+)
cd frontend
npm install
cd ..

# Set up API keys
export GOOGLE_API_KEY="your-google-api-key"
export MAPBOX_ACCESS_TOKEN="your-mapbox-token"
```

### Optional Dependencies

```bash
# Real-ESRGAN for high-quality outpainting
pip install -e ".[upscaling]"

# PSD export support
pip install -e ".[psd]"

# All optional features
pip install -e ".[all]"
```

## Quick Start

### Using the CLI

```bash
# 1. Initialize a new project (optionally set orientation)
mapgen init --name "My City Map" \
  --north 40.758 --south 40.700 --east -73.970 --west -74.020 \
  --orientation north \
  -o projects/my_city

# 2. Generate the illustrated map tiles
mapgen generate-tiles projects/my_city/ -o output/illustrated.png

# 3. Apply perspective and outpaint edges
mapgen outpaint projects/my_city/ -i output/illustrated.png

# 4. Export as layered PSD
mapgen export-psd projects/my_city/ --base-map output/illustrated_outpainted.png -o map.psd
```

### Using the Web UI

If you installed the full stack (Option 2), start the development servers:

```bash
# Start both backend and frontend
./run_dev.sh

# Or start them separately:
# Terminal 1: API backend
uvicorn mapgen.api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

Then open http://localhost:5173 in your browser.

**Web UI Features:**
- **Project Management** - Create and manage map projects visually
- **Interactive Map** - Geographic view with tile grid overlay
- **Real-time Progress** - Watch tiles generate with WebSocket updates
- **Tile Inspector** - Click tiles to view reference/generated images
- **Seam Repair** - Visual interface for fixing tile boundaries
- **Landmark Placement** - Drag-and-drop landmark positioning
- **Landmark Discovery** - Discover and batch-add notable places from OSM
- **Project Settings** - Edit typography, road style, border, atmosphere, palette, and narrative settings
- **Deep Zoom Viewer** - Pan and zoom on large assembled images

## Complete Workflow

### Step 1: Create a Project

Initialize a new project with geographic bounds:

```bash
mapgen init \
  --name "NYC Illustrated Map" \
  --north 40.7580 \
  --south 40.7000 \
  --east -73.9700 \
  --west -74.0200 \
  --orientation north \
  -o projects/nyc
```

The `--orientation` option sets which cardinal direction appears at the top of the map (default: north). Use `east`, `south`, or `west` for different orientations.

This creates:
```
projects/nyc/
├── project.yaml      # Configuration file
├── landmarks/        # Place landmark photos here
├── logos/           # Place logo PNGs here
└── output/          # Generated files go here
```

### Step 2: Preview and Configure

View project info and estimated costs:

```bash
mapgen info projects/nyc/
```

Preview the OSM data extraction:

```bash
mapgen preview-osm projects/nyc/
```

Preview the composite reference (satellite + OSM):

```bash
mapgen preview-composite projects/nyc/ --size 2048
```

### Step 3: Test a Single Tile

Before generating the full map, test with one tile:

```bash
mapgen test-tile projects/nyc/ --col 0 --row 0
```

Review the output to ensure the style is acceptable.

### Step 4: Generate All Tiles

Generate the full illustrated map:

```bash
mapgen generate-tiles projects/nyc/ -o output/illustrated.png
```

Options:
- `--skip-existing`: Resume interrupted generation
- `--dry-run`: Preview what will be generated
- `--no-perspective`: Generate flat (non-perspective) map

### Step 5: Add Landmarks (Optional)

Add landmarks to make key buildings stand out:

```bash
# Add a landmark
mapgen add-landmark projects/nyc/ \
  --name "Empire State Building" \
  --lat 40.7484 --lon -73.9857 \
  --photo landmarks/empire_state.jpg \
  --logo logos/empire_state.png \
  --scale 2.5

# List all landmarks
mapgen list-landmarks projects/nyc/

# Remove a landmark
mapgen remove-landmark projects/nyc/ --name "Empire State Building"
```

### Step 6: Illustrate Landmarks

Transform landmark photos to match the map style:

```bash
# Illustrate all landmarks
mapgen illustrate-landmarks projects/nyc/

# Illustrate a specific landmark
mapgen illustrate-landmarks projects/nyc/ --name "Empire State Building"
```

### Step 7: Compose Landmarks onto Map

Place illustrated landmarks on the base map:

```bash
mapgen compose projects/nyc/ \
  --base-map output/illustrated.png \
  -o output/with_landmarks.png
```

Options:
- `--no-logos`: Exclude logo labels
- `--no-shadows`: Disable drop shadows
- `--no-perspective`: For flat maps

### Step 8: Outpaint Edges

Fill the empty regions from perspective transform:

```bash
mapgen outpaint projects/nyc/ \
  -i output/with_landmarks.png \
  -o output/final.png
```

### Step 9: Export to PSD

Export as a layered Photoshop file:

```bash
# Export as layered PSD
mapgen export-psd projects/nyc/ \
  --base-map output/final.png \
  -o output/map.psd

# Or export as separate layer PNGs
mapgen export-psd projects/nyc/ \
  --base-map output/final.png \
  --format layers \
  -o output/layers/
```

## Configuration Reference

### project.yaml

```yaml
name: "NYC Illustrated Map"

region:
  north: 40.7580    # Northern latitude boundary
  south: 40.7000    # Southern latitude boundary
  east: -73.9700    # Eastern longitude boundary
  west: -74.0200    # Western longitude boundary

output:
  width: 7016       # Output width in pixels
  height: 9933      # Output height in pixels
  dpi: 300          # Resolution for print

style:
  perspective_angle: 35.264   # Isometric angle
  orientation: north          # Which direction is "up" (north/south/east/west)
  palette_preset: vintage_tourist  # Color palette (vintage_tourist, modern_pop, ink_wash)
  palette_enforcement_strength: 0.5  # How strongly to enforce palette (0.0-1.0)
  color_consistency_strength: 0.5    # Cross-tile color consistency (0.0-1.0)
  terrain_exaggeration: 1.0          # Terrain height emphasis (1.0-5.0)
  typography:
    enabled: true
    road_labels: true
    water_labels: true
    park_labels: true
    district_labels: true
    font_scale: 1.0           # Label size multiplier (0.5-3.0)
    max_labels: 50
  road_style:
    enabled: true
    preset: vintage_tourist    # Road rendering preset
    wobble_amount: 1.5         # Hand-drawn wobble (0.0-5.0)
  atmosphere:
    enabled: false
    haze_strength: 0.3         # Distance haze (0.0-1.0)
    contrast_reduction: 0.15   # Depth contrast reduction (0.0-0.5)
  prompt: |
    Transform this map into a hand illustrated tourist map style.
    Use a hand-painted illustration aesthetic with warm, muted colors...

border:
  enabled: false
  style: vintage_scroll        # vintage_scroll, art_deco, modern_minimal, ornate_victorian
  margin: 200                  # Border width in pixels (50-500)
  show_compass: true
  show_legend: true

narrative:
  auto_discover: false         # Auto-discover landmarks from OSM
  max_landmarks: 50
  min_importance_score: 0.3

tiles:
  size: 2048        # Tile size in pixels
  overlap: 256      # Overlap between tiles for blending

landmarks:
  - name: "Empire State Building"
    latitude: 40.7484
    longitude: -73.9857
    photo: "landmarks/empire_state.jpg"
    logo: "logos/empire_state.png"
    scale: 2.5      # Exaggeration factor (1.0 = actual size)
    z_index: 10     # Layer order (higher = on top)
```

### Common Output Sizes

| Format | Pixels | DPI | Physical Size |
|--------|--------|-----|---------------|
| A1 Portrait | 7016 x 9933 | 300 | 23.4" x 33.1" |
| A2 Portrait | 4961 x 7016 | 300 | 16.5" x 23.4" |
| A3 Portrait | 3508 x 4961 | 300 | 11.7" x 16.5" |
| 24x36 Poster | 7200 x 10800 | 300 | 24" x 36" |

## CLI Commands Reference

### Project Management

| Command | Description |
|---------|-------------|
| `mapgen init` | Create a new project |
| `mapgen info PROJECT` | Show project information |
| `mapgen set-orientation PROJECT DIRECTION` | Set map orientation (north/south/east/west) |
| `mapgen clear-cache PROJECT` | Clear cached tiles to force regeneration |

### Preview & Testing

| Command | Description |
|---------|-------------|
| `mapgen preview-osm PROJECT` | Preview OSM data extraction |
| `mapgen preview-composite PROJECT` | Preview satellite + OSM composite |
| `mapgen test-tile PROJECT` | Generate a single test tile |

### Map Generation

| Command | Description |
|---------|-------------|
| `mapgen generate-tiles PROJECT` | Generate all illustrated tiles |
| `mapgen assemble PROJECT` | Assemble tiles into final image |
| `mapgen regenerate-tile PROJECT` | Regenerate specific tiles |
| `mapgen outpaint PROJECT` | Fill empty edges after perspective |

### Seam Repair

| Command | Description |
|---------|-------------|
| `mapgen list-seams PROJECT` | List all tile seam locations |
| `mapgen repair-seam PROJECT` | Repair specific seams |

### Landmarks

| Command | Description |
|---------|-------------|
| `mapgen add-landmark PROJECT` | Add a landmark |
| `mapgen list-landmarks PROJECT` | List all landmarks |
| `mapgen remove-landmark PROJECT` | Remove a landmark |
| `mapgen discover-landmarks PROJECT` | Auto-discover landmarks from OSM data |
| `mapgen illustrate-landmarks PROJECT` | Illustrate landmark photos |
| `mapgen compose PROJECT` | Composite landmarks onto map |

### Post-Processing

| Command | Description |
|---------|-------------|
| `mapgen add-labels PROJECT` | Add typography labels to a map image |
| `mapgen add-border PROJECT` | Add a decorative border to a map image |

### Color Palettes

| Command | Description |
|---------|-------------|
| `mapgen palette list-presets` | List available color palette presets |
| `mapgen palette extract IMAGE` | Extract a color palette from an image |

### Export

| Command | Description |
|---------|-------------|
| `mapgen export-psd PROJECT` | Export as layered PSD |

### Full Pipeline

| Command | Description |
|---------|-------------|
| `mapgen generate PROJECT` | Run complete pipeline with all enhancements |
| `mapgen generate-full PROJECT` | Run complete pipeline (tiles + landmarks + compose) |
| `mapgen generate-sectional PROJECT` | Generate large maps in sections |

## Customization

### Map Orientation

Set which cardinal direction appears at the top of your map:

```bash
# During project creation
mapgen init --name "Tokyo Bay" --orientation east ...

# For existing projects
mapgen set-orientation projects/tokyo/ east

# Clear cache after changing orientation
mapgen clear-cache projects/tokyo/
```

| Orientation | Effect |
|-------------|--------|
| `north` | North is up (default, traditional orientation) |
| `east` | East is up (map rotated 90° counter-clockwise) |
| `south` | South is up (map rotated 180°) |
| `west` | West is up (map rotated 90° clockwise) |

### Style Prompt

The illustration style is controlled by the `style.prompt` field in `project.yaml`. Customize it to achieve different looks:

**Hand Illustrated Tourist Map (Default)**
```yaml
prompt: |
  Transform this map into a hand illustrated tourist map.
  Use a hand-painted illustration aesthetic with warm, muted colors.
  Follow the geography exactly - do not add or remove features.
```

**Watercolor Style**
```yaml
prompt: |
  Transform this map into a soft watercolor illustration. Use gentle
  washes of color, visible brush strokes, and a dreamy atmospheric quality.
  Keep the map readable while creating an artistic, hand-painted feel.
```

**Vintage Style**
```yaml
prompt: |
  Transform this map into a vintage illustrated map from the 1950s.
  Use muted, sepia-toned colors with hand-drawn buildings and landmarks.
  Include subtle paper texture and aged appearance.
```

### Perspective Parameters

Control the aerial perspective effect:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--convergence` | 0.7 | How much the top edge narrows (0.0-1.0) |
| `--vertical-scale` | 0.4 | Vertical compression at top (0.0-1.0) |
| `--horizon-margin` | 0.15 | Space for horizon/sky (0.0-0.5) |

Example with custom perspective:
```bash
mapgen generate-tiles projects/nyc/ \
  --convergence 0.6 \
  --vertical-scale 0.35 \
  --horizon-margin 0.2
```

### Landmark Scaling

Landmarks can be exaggerated to stand out on the map:

| Scale | Effect |
|-------|--------|
| 1.0 | Actual proportional size |
| 1.5 | 50% larger than actual |
| 2.0 | Double size (recommended for major landmarks) |
| 3.0+ | Very prominent, use sparingly |

## Troubleshooting

### API Key Issues

```
ValueError: Google API key required
```

Set the environment variable:
```bash
export GOOGLE_API_KEY="your-key-here"
```

### Out of Memory

For very large maps, reduce tile size in `project.yaml`:
```yaml
tiles:
  size: 1536  # Smaller tiles use less memory
  overlap: 192
```

### Inconsistent Tile Styles

If tiles don't match each other:
1. Use `mapgen regenerate-tile` to regenerate problem tiles
2. Use `mapgen repair-seam` to fix discontinuities at seams
3. Consider a more specific style prompt
4. Clear cache and regenerate: `mapgen clear-cache PROJECT && mapgen generate-tiles PROJECT`

### Aspect Ratio Distortion

If `mapgen info` shows an aspect ratio warning, your map region doesn't match the output dimensions:

```
Warning: Aspect ratio mismatch!
  Geographic aspect ratio: 1.150
  Output aspect ratio: 0.707
  Distortion: 0.61x (horizontal compression)
```

Fix by updating `output.width` and `output.height` in `project.yaml` to match the recommended dimensions shown.

### Large PSD Files

PSD files can be very large (500MB+) due to uncompressed format. Alternatives:
- Use `--format layers` to export separate PNGs
- Compress the PSD after export using Photoshop

## Cost Estimation

Gemini API costs approximately $0.13 per image generation:

| Operation | Typical Cost |
|-----------|-------------|
| Single tile generation | $0.13 |
| Full map (24 tiles) | ~$3.12 |
| Landmark illustration (each) | $0.13 |
| Seam repair (each) | $0.13 |
| Outpainting | $0.13 |
| **Total (typical map)** | **~$5-6** |

Use `--dry-run` to preview costs before generating:
```bash
mapgen generate-tiles projects/nyc/ --dry-run
```

## Examples

### NYC Manhattan

```bash
mapgen init --name "Manhattan" \
  --north 40.82 --south 40.70 --east -73.93 --west -74.02 \
  -o projects/manhattan

mapgen generate-tiles projects/manhattan/
```

### San Francisco

```bash
mapgen init --name "San Francisco" \
  --north 37.81 --south 37.70 --east -122.35 --west -122.52 \
  -o projects/sf

mapgen generate-tiles projects/sf/
```

### Tokyo Bay (East-Up Orientation)

```bash
# Create with east-up orientation for a different perspective
mapgen init --name "Tokyo Bay" \
  --north 35.7 --south 35.4 --east 140.1 --west 139.6 \
  --orientation east \
  -o projects/tokyo

mapgen generate-tiles projects/tokyo/
```

### Custom Small Area

```bash
mapgen init --name "My Neighborhood" \
  --north 40.75 --south 40.74 --east -73.98 --west -73.99 \
  -o projects/neighborhood

# Use smaller output for faster generation
# Edit project.yaml: width: 2048, height: 2048

mapgen generate-tiles projects/neighborhood/
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run full test suite
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run only integration tests
python -m pytest tests/integration/ -v
```

The test suite includes 481 tests covering models, services, CLI commands, and API endpoints.

### Type Checking (Frontend)

```bash
cd frontend && npx tsc --noEmit
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.
