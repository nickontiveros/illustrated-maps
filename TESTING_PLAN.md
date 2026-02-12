# Testing Plan: Illustrated Map Generator

This document defines a layered testing strategy for the codebase, organized by
test type (unit, integration, E2E) and prioritized by impact and feasibility.

---

## Current State

- **Framework:** pytest + pytest-cov (already in `[dev]` dependencies)
- **Existing tests:** 1 file (`tests/test_osm_service.py`) with 4 tests covering only `OSMService`
- **CI/CD:** None configured
- **Linting/Formatting:** Black + Ruff configured in `pyproject.toml`
- **Frontend:** No tests, no test runner configured

---

## Testing Principles

1. **Test at the right level.** Pure functions get unit tests. Service orchestration
   gets integration tests with mocked dependencies. Real API calls are E2E only.
2. **Prioritize by risk.** Image-processing math (perspective, blending, geo
   transforms) silently produces wrong results when broken -- these get tests first.
3. **Keep tests fast.** The default `pytest` run should complete in seconds, not
   minutes. Tests that hit the network or cost money are opt-in via markers.
4. **Use dependency injection.** Services already accept optional collaborators in
   their constructors -- use this to inject mocks and test doubles.

---

## Proposed Test Structure

```
tests/
  conftest.py                    # Shared fixtures (BoundingBox, Project, test images)
  unit/
    test_bounding_box.py         # BoundingBox model
    test_project_model.py        # Project, OutputSettings, TileSettings, etc.
    test_landmark_model.py       # Landmark model
    test_geo_utils.py            # All geo utility functions
    test_image_utils.py          # Pure image operations
    test_perspective_service.py  # Perspective transforms (pure math)
    test_blending_service.py     # Tile blending (pure image ops)
    test_generation_calc.py      # GenerationService pure methods (tile specs, cost)
    test_schemas.py              # API Pydantic schemas
  integration/
    test_render_service.py       # Render pipeline with real matplotlib, mock OSM data
    test_generation_service.py   # Generation pipeline with mocked Gemini/satellite/OSM
    test_gemini_service.py       # Gemini service with mocked API client
    test_osm_service.py          # OSM service (existing, hits network -- keep as-is)
    test_project_io.py           # Project YAML load/save with temp files
    test_api_projects.py         # FastAPI TestClient for project endpoints
    test_api_tiles.py            # FastAPI TestClient for tile endpoints
  e2e/
    test_full_pipeline.py        # End-to-end generation (requires API keys, slow)
frontend/
  src/**/*.test.tsx              # Component tests (Vitest + React Testing Library)
```

### pytest markers (add to `pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "network: tests that require network access (OSM, satellite)",
    "gemini: tests that call the real Gemini API (costs money)",
    "slow: tests that take more than 10 seconds",
]
```

Default runs exclude expensive tests:

```bash
pytest                             # unit + integration (no network)
pytest -m network                  # include network tests
pytest -m gemini                   # include Gemini API tests ($$)
pytest --cov=mapgen --cov-report=term-missing  # with coverage
```

---

## Layer 1: Unit Tests (Priority: High)

These tests are fast, deterministic, and require no mocking or external services.

### 1.1 `test_bounding_box.py` -- BoundingBox Model

| Test | What It Validates |
|------|-------------------|
| `test_valid_construction` | Accepts valid lat/lon ranges |
| `test_invalid_coordinates_rejected` | Pydantic rejects out-of-range values |
| `test_center_property` | `center` returns midpoint of bbox |
| `test_width_and_height_degrees` | `width_degrees`, `height_degrees` are correct |
| `test_geographic_aspect_ratio` | Accounts for latitude cosine correction |
| `test_calculate_area_km2` | Known bounding box returns expected area |
| `test_get_recommended_detail_level` | Area thresholds map to correct `DetailLevel` |
| `test_expanded_for_rotation` | Expansion geometry is correct for a known angle |
| `test_to_tuple_and_osmnx_format` | Serialization round-trips correctly |

### 1.2 `test_project_model.py` -- Project & Supporting Models

| Test | What It Validates |
|------|-------------------|
| `test_default_output_settings` | Defaults (7016x9933, 300 DPI) are applied |
| `test_default_tile_settings` | Defaults (2048 tile size, 256 overlap) are applied |
| `test_calculate_adjusted_dimensions` | Pixel budget respects max_dimension constraint |
| `test_cardinal_direction_rotation` | Each direction maps to correct degrees (0, 90, 180, 270) |
| `test_detail_level_ordering` | `DetailLevel` enum values are ordered correctly |

### 1.3 `test_landmark_model.py` -- Landmark Model

| Test | What It Validates |
|------|-------------------|
| `test_coordinates_property` | Returns `(lat, lon)` tuple |
| `test_resolve_photo_path` | Resolves relative paths against project directory |
| `test_resolve_logo_path` | Resolves logo paths correctly |
| `test_feature_type_values` | All `FeatureType` enum members are valid |

### 1.4 `test_geo_utils.py` -- Geographic Utilities

All functions in `mapgen/utils/geo_utils.py` are pure. Test with known inputs/outputs.

| Test | What It Validates |
|------|-------------------|
| `test_bbox_to_polygon` | Returns Shapely polygon with correct vertices |
| `test_calculate_aspect_ratio` | Known bbox returns expected ratio |
| `test_meters_per_pixel` | Known bbox + image size returns expected scale |
| `test_gps_to_pixel_basic` | GPS center maps to image center |
| `test_gps_to_pixel_corners` | GPS corners map to image corners |
| `test_pixel_to_gps_roundtrip` | `pixel_to_gps(gps_to_pixel(lat, lon)) == (lat, lon)` |
| `test_haversine_distance` | Known city pairs return expected km |
| `test_calculate_map_scale` | Returns correct meters-per-pixel scale |

### 1.5 `test_image_utils.py` -- Image Processing Utilities

Use synthetic PIL images (solid colors, gradients) as inputs.

| Test | What It Validates |
|------|-------------------|
| `test_resize_image` | Output dimensions match requested size |
| `test_create_alpha_mask` | Mask is binary with correct threshold behavior |
| `test_apply_drop_shadow` | Output is larger than input by shadow offset |
| `test_blend_images_full_opacity` | Overlay fully replaces base at position |
| `test_blend_images_half_opacity` | Pixel values are averaged at 50% opacity |
| `test_crop_to_content` | Trims transparent borders, respects padding |

### 1.6 `test_perspective_service.py` -- Perspective Transforms

All methods are pure math/NumPy. Use known inputs and verify geometric properties.

| Test | What It Validates |
|------|-------------------|
| `test_default_parameters` | Default angle, convergence, scales are set |
| `test_transform_coordinates_center` | Center point stays near center after transform |
| `test_transform_coordinates_invertible` | Forward + inverse transform round-trips |
| `test_get_output_size` | Output height is scaled by vertical_scale |
| `test_get_rotated_output_size` | 90-degree rotation swaps width/height |
| `test_isometric_matrix_shape` | Matrix is 4x4 |
| `test_isometric_matrix_determinant` | Matrix is invertible (non-zero determinant) |
| `test_transform_image_produces_output` | 100x100 test image produces valid output |
| `test_transform_image_dimensions` | Output size matches `get_output_size()` |
| `test_calculate_y_offset` | Known elevation returns expected offset |

### 1.7 `test_blending_service.py` -- Tile Blending

Use small synthetic tiles (e.g., 64x64 solid colors) for fast tests.

| Test | What It Validates |
|------|-------------------|
| `test_create_gradient_mask_horizontal` | Gradient runs left-to-right, values in [0,1] |
| `test_create_gradient_mask_vertical` | Gradient runs top-to-bottom |
| `test_multiband_blend_identity` | Blending identical images returns same image |
| `test_multiband_blend_half_mask` | 50% mask produces averaged pixels |
| `test_blend_tiles_single_tile` | Single tile returns the tile image |
| `test_blend_tiles_two_tiles_overlap` | Overlap region is smoothly blended (no hard seam) |
| `test_gaussian_pyramid_levels` | Pyramid has expected number of levels |
| `test_laplacian_pyramid_reconstruction` | Reconstruct from Laplacian matches original |

### 1.8 `test_generation_calc.py` -- GenerationService Pure Methods

| Test | What It Validates |
|------|-------------------|
| `test_calculate_tile_specs_count` | Grid dimensions match expected rows x cols |
| `test_calculate_tile_specs_overlap` | Adjacent tiles overlap by configured amount |
| `test_calculate_tile_specs_bbox_coverage` | Tile bboxes cover full project region |
| `test_estimate_cost` | Returns dict with expected keys and reasonable values |
| `test_is_water_tile` | Mostly-blue image classified as water |
| `test_is_not_water_tile` | Mostly-green image not classified as water |
| `test_position_description` | Corner/edge/center tiles described correctly |

### 1.9 `test_schemas.py` -- API Schemas

| Test | What It Validates |
|------|-------------------|
| `test_project_create_validation` | Rejects missing required fields |
| `test_project_create_valid` | Accepts valid payloads |
| `test_tile_spec_schema` | Serialization/deserialization round-trips |
| `test_generation_start_request` | Accepts valid generation configs |
| `test_project_detail_from_project` | `from_project()` maps all fields correctly |

---

## Layer 2: Integration Tests (Priority: Medium)

These tests verify that components work together correctly. They mock external
services (Gemini API, satellite imagery, OSM network) but use real internal logic.

### 2.1 `test_render_service.py`

**Setup:** Create mock `OSMData` objects with known GeoDataFrames (a few roads,
a park polygon, a water body). No network calls.

| Test | What It Validates |
|------|-------------------|
| `test_render_base_map_produces_image` | Returns a PIL Image of correct size |
| `test_render_base_map_without_perspective` | Skipping perspective still renders |
| `test_render_tile_dimensions` | Output matches requested tile size |
| `test_render_composite_reference` | Combines satellite + OSM into single image |
| `test_render_with_elevation_data` | Elevation shading modifies output pixels |

### 2.2 `test_generation_service.py`

**Setup:** Mock `GeminiService`, `SatelliteService`, `OSMService`. Inject via
constructor. Return synthetic images from mocks.

| Test | What It Validates |
|------|-------------------|
| `test_generate_tile_calls_gemini` | GeminiService.generate_base_tile is called |
| `test_generate_tile_retries_on_failure` | Retries up to max_retries times |
| `test_generate_tile_caches_result` | Second call loads from cache, no API call |
| `test_generate_all_tiles_progress` | Progress callback receives updates |
| `test_assemble_tiles_correct_size` | Assembled image matches expected dimensions |
| `test_assemble_tiles_tile_placement` | Each tile placed at correct position |
| `test_generate_single_test_tile` | Generates tile at specified grid position |

### 2.3 `test_gemini_service.py`

**Setup:** Mock `google.genai.Client` to return canned responses.

| Test | What It Validates |
|------|-------------------|
| `test_generate_base_tile_sends_prompt` | Prompt includes style and position info |
| `test_generate_base_tile_extracts_image` | Image extracted from mock response |
| `test_stylize_landmark_sends_photo` | Photo is included in API request |
| `test_inpaint_seam_orientation` | Prompt varies by orientation parameter |
| `test_handles_api_error_gracefully` | Returns None or raises clear error on failure |
| `test_lazy_client_initialization` | Client not created until first API call |

### 2.4 `test_project_io.py`

**Setup:** Use `tmp_path` fixture for temp directories.

| Test | What It Validates |
|------|-------------------|
| `test_project_yaml_roundtrip` | `to_yaml` then `from_yaml` preserves all fields |
| `test_ensure_directories_creates_dirs` | Output, landmarks, logos dirs are created |
| `test_from_yaml_missing_file` | Raises clear error for missing file |
| `test_from_yaml_invalid_yaml` | Raises clear error for malformed YAML |

### 2.5 `test_api_projects.py` and `test_api_tiles.py`

**Setup:** Use FastAPI `TestClient`. Create temp project directories with fixture
YAML files.

| Test | What It Validates |
|------|-------------------|
| `test_list_projects` | GET `/api/projects` returns project list |
| `test_get_project_detail` | GET `/api/projects/{name}` returns full config |
| `test_create_project` | POST `/api/projects` creates YAML + directories |
| `test_update_project` | PUT `/api/projects/{name}` modifies config |
| `test_get_tile_grid` | GET `/api/projects/{name}/tiles` returns grid info |
| `test_404_for_missing_project` | Returns 404 for nonexistent project name |

---

## Layer 3: End-to-End Tests (Priority: Low, Run Manually)

These tests hit real external services. They are slow, cost money (Gemini API),
and require network access. Run them manually or in a dedicated CI job.

Mark with `@pytest.mark.gemini` and/or `@pytest.mark.network`.

### 3.1 `test_full_pipeline.py`

| Test | What It Validates |
|------|-------------------|
| `test_single_tile_generation` | Generates one tile for a small bbox via real Gemini |
| `test_osm_data_fetch_and_render` | Fetches real OSM data and renders a reference map |
| `test_satellite_fetch` | Fetches real satellite imagery for a known bbox |
| `test_seam_repair` | Repairs a seam between two real tiles via Gemini |

---

## Layer 4: Frontend Tests (Priority: Medium)

### 4.1 Setup

Add to `frontend/package.json`:

```json
{
  "devDependencies": {
    "vitest": "^1.0.0",
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "jsdom": "^24.0.0"
  },
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

Add `vite.config.ts` test configuration:

```ts
export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test-setup.ts',
  },
})
```

### 4.2 Component Tests

| Test File | What It Validates |
|-----------|-------------------|
| `ProjectList.test.tsx` | Renders project list, handles loading/error states |
| `ProjectView.test.tsx` | Displays project config, handles missing project |
| `TileGrid.test.tsx` | Renders correct grid dimensions, click handlers work |
| `GlobalProgressBar.test.tsx` | Progress bar reflects generation state |
| `MapCanvas.test.tsx` | Leaflet map initializes with correct bounds |

### 4.3 Hook Tests

| Test File | What It Validates |
|-----------|-------------------|
| `useProjects.test.ts` | Fetches and caches project data via React Query |
| `useTiles.test.ts` | Returns tile grid, handles refetch |
| `useWebSocket.test.ts` | Connects to WebSocket, processes messages |

### 4.4 API Client Tests

| Test File | What It Validates |
|-----------|-------------------|
| `client.test.ts` | Constructs correct URLs, handles error responses |

---

## Shared Test Fixtures (`tests/conftest.py`)

```python
import pytest
from PIL import Image
from mapgen.models.project import BoundingBox, Project, OutputSettings, TileSettings

@pytest.fixture
def sample_bbox():
    """Small bounding box around Central Park, NYC."""
    return BoundingBox(north=40.775, south=40.768, east=-73.968, west=-73.978)

@pytest.fixture
def sample_project(tmp_path, sample_bbox):
    """Minimal project config for testing."""
    return Project(
        name="test-project",
        region=sample_bbox,
        output=OutputSettings(width=1024, height=1024, dpi=72),
        tiles=TileSettings(size=512, overlap=64),
    )

@pytest.fixture
def solid_red_image():
    """64x64 solid red RGBA image."""
    return Image.new("RGBA", (64, 64), (255, 0, 0, 255))

@pytest.fixture
def solid_blue_image():
    """64x64 solid blue RGBA image."""
    return Image.new("RGBA", (64, 64), (0, 0, 255, 255))

@pytest.fixture
def transparent_image():
    """64x64 fully transparent image."""
    return Image.new("RGBA", (64, 64), (0, 0, 0, 0))

@pytest.fixture
def gradient_image():
    """64x64 horizontal gradient from black to white."""
    import numpy as np
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    arr[:, :, 1] = arr[:, :, 0]
    arr[:, :, 2] = arr[:, :, 0]
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")
```

---

## CI/CD Integration (GitHub Actions)

Recommended workflow (`.github/workflows/test.yml`):

```yaml
name: Tests
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: ruff check mapgen/
      - run: black --check mapgen/ tests/

  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ --cov=mapgen --cov-report=xml
      - uses: codecov/codecov-action@v4

  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev,api]"
      - run: pytest tests/integration/ -m "not network and not gemini"

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: cd frontend && npm ci && npm test
```

---

## Suggested pytest Configuration

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "network: tests that require network access",
    "gemini: tests that call the real Gemini API",
    "slow: tests that take more than 10 seconds",
]
addopts = "-m 'not network and not gemini and not slow'"
```

This ensures `pytest` runs only fast, offline tests by default.

---

## Coverage Goals

| Layer | Target | Rationale |
|-------|--------|-----------|
| Models (`models/`) | 90%+ | Pure data, easy to test exhaustively |
| Utilities (`utils/`) | 90%+ | Pure functions, critical math |
| PerspectiveService | 85%+ | Core visual correctness |
| BlendingService | 85%+ | Core visual correctness |
| GenerationService (pure methods) | 80%+ | Tile layout math |
| API schemas | 80%+ | Request validation |
| API endpoints | 70%+ | Integration coverage |
| RenderService | 60%+ | Complex matplotlib, harder to assert on |
| GeminiService | 50%+ | Mostly mocked, limited testable surface |
| Frontend components | 70%+ | User-facing correctness |

---

## Implementation Priority

The recommended order for writing tests, balancing impact with effort:

1. **`test_geo_utils.py`** -- Pure functions, high impact, fast to write
2. **`test_bounding_box.py`** -- Core data model used everywhere
3. **`test_perspective_service.py`** -- Silent regressions in transforms are costly
4. **`test_blending_service.py`** -- Visual correctness of tile assembly
5. **`test_image_utils.py`** -- Shared utilities, broad impact
6. **`test_generation_calc.py`** -- Tile layout correctness
7. **`test_project_model.py` + `test_landmark_model.py`** -- Data integrity
8. **`test_schemas.py`** -- API contract validation
9. **`conftest.py` + pytest config** -- Shared infrastructure
10. **Integration tests** -- After unit tests provide a safety net
11. **Frontend tests** -- After backend is well-covered
12. **CI/CD pipeline** -- After test suite is stable
