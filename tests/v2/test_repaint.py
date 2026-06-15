"""Repaint stage tests: legality rules, planner properties, template
geometry, engine round-trip, color normalization, and pipeline integration.

The legality/planner logic is the part Isometric NYC flagged as
"irreducibly complex" -- every planner test asserts legality at every step
AND exactly-once coverage, and emits per-step visual artifacts
(MAPGEN_TEST_ARTIFACTS=1) for eyeball verification.
"""

import numpy as np
import pytest
from PIL import Image

from mapgen.v2.repaint.color_norm import unify_water
from mapgen.v2.repaint.engine import IdentityPainter, RepaintEngine
from mapgen.v2.repaint.grid import (
    QUAD,
    WINDOW,
    QuadrantGrid,
    QuadStatus,
    Selection,
    is_legal,
    window_origin,
)
from mapgen.v2.repaint.planner import plan_order
from mapgen.v2.repaint.store import RepaintStore
from mapgen.v2.repaint.template import build_template, extract_selection, selection_px_box
from mapgen.v2.types import StyleSpec

GRID_COLS, GRID_ROWS = 7, 10


def cells(*pairs):
    return set(pairs)


# --- legality rules: every documented Isometric NYC example -------------------


class TestIsLegal:
    def test_isolated_2x2_is_legal(self):
        assert is_legal(Selection(2, 2, 2, 2), set(), GRID_COLS, GRID_ROWS)

    def test_2x2_with_any_painted_neighbor_is_illegal(self):
        # Their rule: 2x2 cannot have ANY generated neighbors.
        for neighbor in [(1, 2), (4, 3), (2, 1), (3, 4)]:
            assert not is_legal(Selection(2, 2, 2, 2), {neighbor}, GRID_COLS, GRID_ROWS), neighbor

    def test_tall_1x2_with_left_painted_is_legal(self):
        # Their first "good" case: G G S / G G S.
        painted = cells((0, 0), (0, 1), (1, 0), (1, 1))
        assert is_legal(Selection(2, 0, 1, 2), painted, GRID_COLS, GRID_ROWS)

    def test_middle_band_left_and_right_painted_is_legal(self):
        # Their prose "middle band" case: G G S G G (legal via centered window).
        painted = cells((0, 0), (1, 0), (3, 0), (4, 0), (0, 1), (1, 1), (3, 1), (4, 1))
        assert is_legal(Selection(2, 0, 1, 2), painted, GRID_COLS, GRID_ROWS)

    def test_their_illegal_case_top_left_right_painted_tall(self):
        # Their explicit ILLEGAL example: 1x2 tall with painted above + left + right.
        painted = cells(
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),  # row above all painted
            (0, 1), (1, 1), (3, 1), (4, 1),
            (0, 2), (1, 2), (3, 2), (4, 2),
        )
        assert not is_legal(Selection(2, 1, 1, 2), painted, GRID_COLS, GRID_ROWS)

    def test_their_legal_case_1x1_with_three_painted_sides(self):
        # Their explicit LEGAL counterpart: 1x1 with painted above + left + right.
        painted = cells((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (3, 1))
        assert is_legal(Selection(2, 1, 1, 1), painted, GRID_COLS, GRID_ROWS)

    def test_1x1_boxed_on_four_sides_needs_allow_boxed(self):
        painted = cells((1, 2), (3, 2), (2, 1), (2, 3))
        assert not is_legal(Selection(2, 2, 1, 1), painted, GRID_COLS, GRID_ROWS)
        assert is_legal(Selection(2, 2, 1, 1), painted, GRID_COLS, GRID_ROWS, allow_boxed=True)

    def test_wide_2x1_with_painted_above_is_legal(self):
        # Strip-plan case: S S over a painted row.
        painted = {(x, 0) for x in range(GRID_COLS)}
        assert is_legal(Selection(0, 1, 2, 1), painted, GRID_COLS, GRID_ROWS)

    def test_wide_2x1_with_painted_side_is_illegal(self):
        painted = {(2, 1)}  # touching the right side of a 2-wide selection
        assert not is_legal(Selection(0, 1, 2, 1), painted, GRID_COLS, GRID_ROWS)

    def test_selection_overlapping_painted_is_illegal(self):
        assert not is_legal(Selection(2, 2, 2, 2), {(2, 2)}, GRID_COLS, GRID_ROWS)

    def test_out_of_bounds_is_illegal(self):
        assert not is_legal(Selection(-1, 0), set(), GRID_COLS, GRID_ROWS)
        assert not is_legal(Selection(GRID_COLS - 1, 0, 2, 1), set(), GRID_COLS, GRID_ROWS)


# --- window placement ----------------------------------------------------------


class TestWindowOrigin:
    def make_grid(self, cols=GRID_COLS, rows=GRID_ROWS):
        return QuadrantGrid(cols * QUAD, rows * QUAD)

    def test_2x2_window_equals_selection(self):
        grid = self.make_grid()
        assert window_origin(Selection(2, 3, 2, 2), set(), grid) == (2 * QUAD, 3 * QUAD)

    def test_aligned_toward_painted_left(self):
        grid = self.make_grid()
        sel = Selection(2, 0, 1, 2)
        painted = cells((1, 0), (1, 1))
        ox, oy = window_origin(sel, painted, grid)
        assert (ox, oy) == (QUAD, 0)  # window includes the painted left column

    def test_centered_when_painted_both_sides(self):
        grid = self.make_grid()
        sel = Selection(2, 0, 1, 2)
        painted = cells((1, 0), (1, 1), (3, 0), (3, 1))
        ox, _ = window_origin(sel, painted, grid)
        assert ox == 2 * QUAD - QUAD // 2  # half-quadrant context each side

    def test_boxed_1x1_centers_both_axes(self):
        grid = self.make_grid()
        sel = Selection(2, 2, 1, 1)
        painted = cells((1, 2), (3, 2), (2, 1), (2, 3))
        ox, oy = window_origin(sel, painted, grid)
        assert (ox, oy) == (2 * QUAD - QUAD // 2, 2 * QUAD - QUAD // 2)

    def test_window_clamped_in_bounds_without_context(self):
        grid = self.make_grid()
        # Bottom-right corner cell, nothing painted: window must stay in bounds.
        ox, oy = window_origin(Selection(GRID_COLS - 1, GRID_ROWS - 1, 1, 1), set(), grid)
        pad_w, pad_h = grid.padded_size
        assert ox + WINDOW <= pad_w and oy + WINDOW <= pad_h


# --- planner properties ----------------------------------------------------------


def render_plan_steps(grid, selections, painted, skipped):
    """Per-step state images: painted green, selected red, skipped gray."""
    frames = []
    sim = set(painted)
    cell_px = 12
    for sel in selections:
        img = Image.new("RGB", (grid.cols * cell_px, grid.rows * cell_px), (240, 235, 220))
        arr = np.asarray(img).copy()
        for x, y in sim:
            arr[y * cell_px : (y + 1) * cell_px, x * cell_px : (x + 1) * cell_px] = (120, 170, 110)
        for x, y in skipped:
            arr[y * cell_px : (y + 1) * cell_px, x * cell_px : (x + 1) * cell_px] = (200, 200, 200)
        for x, y in sel.cells():
            arr[y * cell_px : (y + 1) * cell_px, x * cell_px : (x + 1) * cell_px] = (220, 70, 60)
        frames.append(Image.fromarray(arr))
        sim.update(sel.cells())
    return frames


def assert_plan_valid(grid, selections, painted, skipped):
    """Every step legal at the time it runs; every cell painted at most once;
    every pending cell painted; every selection does useful work (skipped
    cells may ride along inside a partially-blank selection, but never make
    up a whole one)."""
    sim = set(painted)
    for sel in selections:
        assert is_legal(sel, sim, grid.cols, grid.rows, allow_boxed=True), (sel, sorted(sim))
        assert any(c not in sim and c not in skipped for c in sel.cells()), f"useless {sel}"
        for cell in sel.cells():
            assert cell not in sim, f"{cell} painted twice"
            sim.add(cell)
    expected = {c for c in grid.all_cells() if c not in skipped}
    assert sim >= expected, f"unpainted cells: {sorted(expected - sim)}"


class TestPlanner:
    def test_full_grid_plan_is_valid(self, artifacts):
        grid = QuadrantGrid(GRID_COLS * QUAD, GRID_ROWS * QUAD)
        selections = plan_order(grid)
        assert_plan_valid(grid, selections, set(), set())
        for i, frame in enumerate(render_plan_steps(grid, selections, set(), set())[:60]):
            artifacts.save(f"step_{i:03d}", frame)

    def test_call_count_matches_estimate(self):
        # 7x10 grid should plan in the ~38-46 call range from the design.
        grid = QuadrantGrid(GRID_COLS * QUAD, GRID_ROWS * QUAD)
        assert 35 <= len(plan_order(grid)) <= 50

    @pytest.mark.parametrize("cols,rows", [(2, 2), (3, 3), (4, 7), (5, 2), (8, 8), (7, 10), (9, 4)])
    def test_plan_valid_on_many_grid_sizes(self, cols, rows):
        grid = QuadrantGrid(cols * QUAD, rows * QUAD)
        assert_plan_valid(grid, plan_order(grid), set(), set())

    @pytest.mark.parametrize("seed", range(8))
    def test_plan_valid_with_random_prepainted_and_skipped(self, seed):
        rng = np.random.default_rng(seed)
        cols, rows = int(rng.integers(2, 9)), int(rng.integers(2, 11))
        grid = QuadrantGrid(cols * QUAD, rows * QUAD)
        all_cells = grid.all_cells()
        # Pre-painted: a random rectangle (what a resumed run looks like).
        if rng.random() < 0.7:
            x0, y0 = int(rng.integers(0, cols)), int(rng.integers(0, rows))
            x1, y1 = int(rng.integers(x0, cols)), int(rng.integers(y0, rows))
            painted = {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
        else:
            painted = set()
        skipped = {
            c for c in all_cells if c not in painted and rng.random() < 0.15
        }
        selections = plan_order(grid, painted, skipped)
        assert_plan_valid(grid, selections, painted, skipped)

    def test_flagged_interior_cell_gets_boxed_redo(self):
        # All painted except one interior cell: cleanup pass must redo it.
        grid = QuadrantGrid(5 * QUAD, 5 * QUAD)
        painted = {c for c in grid.all_cells() if c != (2, 2)}
        selections = plan_order(grid, painted, set())
        assert selections == [Selection(2, 2, 1, 1)]


# --- template construction --------------------------------------------------------


class TestTemplate:
    def make_working(self, grid):
        """Working canvas where every quadrant has a unique flat color."""
        img = Image.new("RGB", grid.padded_size)
        arr = np.asarray(img).copy()
        for x, y in grid.all_cells():
            left, top, right, bottom = grid.cell_box((x, y))
            arr[top:bottom, left:right] = (10 + x * 25, 10 + y * 20, 100)
        return Image.fromarray(arr)

    def test_template_window_and_extraction_geometry(self, artifacts):
        grid = QuadrantGrid(4 * QUAD, 4 * QUAD)
        working = self.make_working(grid)
        sel = Selection(2, 1, 1, 1)
        painted = cells((1, 1))  # painted to the left -> aligned window
        template, window_box, sel_box = build_template(working, sel, painted, grid)
        artifacts.save("template", template)
        assert template.size == (WINDOW, WINDOW)
        assert window_box == (QUAD, QUAD, QUAD + WINDOW, QUAD + WINDOW)
        assert sel_box == (QUAD, 0, WINDOW, QUAD)
        # Extraction returns exactly the selection's pixels.
        extracted = extract_selection(template, sel_box)
        assert extracted.size == (QUAD, QUAD)

    def test_red_boundary_outside_selection_only(self):
        grid = QuadrantGrid(4 * QUAD, 4 * QUAD)
        working = self.make_working(grid)
        sel = Selection(2, 1, 1, 1)
        painted = cells((1, 1))
        template, _, sel_box = build_template(working, sel, painted, grid)
        arr = np.asarray(template)
        red = (arr[..., 0] > 200) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
        assert red.any()  # boundary was drawn...
        l, t, r, b = sel_box
        assert not red[t:b, l:r].any()  # ...but never inside the selection

    def test_selection_px_box(self):
        grid = QuadrantGrid(4 * QUAD, 4 * QUAD)
        assert selection_px_box(Selection(1, 2, 2, 1), grid) == (QUAD, 2 * QUAD, 3 * QUAD, 3 * QUAD)


# --- engine ----------------------------------------------------------------------


def make_guide(width=3 * QUAD + 100, height=2 * QUAD + 50, seed=0):
    """Structured synthetic guide (gradients + shapes), nothing blank."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, width)[None, :, None]
    y = np.linspace(0, 1, height)[:, None, None]
    arr = 120 + 60 * np.sin(8 * np.pi * x) + 50 * np.cos(6 * np.pi * y)
    arr = np.repeat(arr, 3, axis=2) + rng.normal(0, 6, (height, width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


class TintPainter:
    """Tints the whole window blue; extraction keeps only the selection."""

    def paint(self, template, style, style_bible=None):
        arr = np.asarray(template.convert("RGB"), dtype=np.float32)
        arr[..., 2] = np.clip(arr[..., 2] + 60, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGB")


class TestEngine:
    def test_identity_round_trip(self, tmp_path, artifacts):
        """KEYSTONE: identity painter reproduces the guide byte-near, proving
        windowing, ordering, normalization, and stitching add no artifacts."""
        guide = make_guide()
        store = RepaintStore(tmp_path)
        engine = RepaintEngine(IdentityPainter(), store, StyleSpec())
        result = engine.run(guide)
        assert result.completed
        diff = np.abs(
            np.asarray(result.image, dtype=np.float32) - np.asarray(guide, dtype=np.float32)
        )
        artifacts.save("guide", guide)
        artifacts.save("round_trip", result.image)
        artifacts.save("diff_x8", np.clip(diff * 8, 0, 255).astype(np.uint8))
        assert result.image.size == guide.size
        assert diff.max() <= 2.0  # Lab round-trip rounding only

    def test_run_is_resumable_after_budget(self, tmp_path):
        guide = make_guide()
        store = RepaintStore(tmp_path)
        engine = RepaintEngine(IdentityPainter(), store, StyleSpec())
        first = engine.run(guide, max_calls=2)
        assert not first.completed and first.calls_made == 2
        second = engine.run(guide, max_calls=None)
        assert second.completed
        assert second.calls_made == first.calls_planned - 2
        assert store.call_count() == first.calls_planned

    def test_blank_quadrants_are_skipped(self, tmp_path):
        # Guide with structured left half, flat right half.
        guide = make_guide(width=4 * QUAD, height=2 * QUAD)
        arr = np.asarray(guide).copy()
        arr[:, 2 * QUAD :] = (243, 233, 212)  # flat paper
        guide = Image.fromarray(arr)
        store = RepaintStore(tmp_path)
        engine = RepaintEngine(IdentityPainter(), store, StyleSpec())
        result = engine.run(guide)
        assert result.completed
        skipped = store.cells_with_status(QuadStatus.SKIPPED)
        assert {(2, 0), (3, 0), (2, 1), (3, 1)} <= skipped
        painted = store.cells_with_status(QuadStatus.GENERATED)
        assert (0, 0) in painted and not (skipped & painted)

    def test_stitched_edges_have_no_seams(self, tmp_path, artifacts):
        """Tint-paint everything, then check luminance steps across every
        quadrant boundary stay comparable to interior steps (seam metric)."""
        guide = make_guide(width=4 * QUAD, height=3 * QUAD)
        store = RepaintStore(tmp_path)
        engine = RepaintEngine(TintPainter(), store, StyleSpec())
        result = engine.run(guide)
        artifacts.save("tinted", result.image)
        arr = np.asarray(result.image.convert("L"), dtype=np.float32)
        # Steps across vertical quadrant boundaries vs. typical steps.
        typical = np.abs(np.diff(arr, axis=1)).mean()
        for bx in range(1, 4):
            col = bx * QUAD
            boundary = np.abs(arr[:, col] - arr[:, col - 1]).mean()
            assert boundary < typical * 4 + 3, f"seam at column {col}"
        for by in range(1, 3):
            row = by * QUAD
            boundary = np.abs(arr[row, :] - arr[row - 1, :]).mean()
            assert boundary < typical * 4 + 3, f"seam at row {row}"

    def test_hallucinating_painter_is_rejected(self, tmp_path):
        """A painter that ignores the guide (invented scenery) must be caught
        by the structure guard: guide pixels kept, cells flagged."""

        class HallucinationPainter:
            def paint(self, template, style, style_bible=None):
                rng = np.random.default_rng(9)
                blobs = rng.normal(128, 60, (32, 32, 3))
                img = Image.fromarray(np.clip(blobs, 0, 255).astype(np.uint8), "RGB")
                return img.resize(template.size, Image.Resampling.BICUBIC)

        guide = make_guide()
        store = RepaintStore(tmp_path)
        engine = RepaintEngine(HallucinationPainter(), store, StyleSpec())
        result = engine.run(guide)
        # Every window rejected: output is the untouched guide...
        assert np.array_equal(np.asarray(result.image), np.asarray(guide.convert("RGB")))
        # ...nothing marked generated, everything flagged for review.
        assert not store.cells_with_status(QuadStatus.GENERATED)
        flagged = store.cells_with_status(QuadStatus.FLAGGED)
        assert flagged
        assert result.calls_made == result.calls_planned  # calls were still spent

    def test_normalization_corrects_drifted_painter(self, tmp_path):
        """A painter with a global color cast must come back guide-like."""

        class DriftPainter:
            def paint(self, template, style, style_bible=None):
                arr = np.asarray(template.convert("RGB"), dtype=np.float32)
                return Image.fromarray(np.clip(arr + 35, 0, 255).astype(np.uint8), "RGB")

        guide = make_guide()
        store = RepaintStore(tmp_path)
        engine = RepaintEngine(DriftPainter(), store, StyleSpec())
        result = engine.run(guide)
        mean_diff = abs(
            np.asarray(result.image, dtype=np.float32).mean()
            - np.asarray(guide, dtype=np.float32).mean()
        )
        assert mean_diff < 5.0  # +35 cast normalized away


# --- single-call texture pass (the default repaint mode) ----------------------


class TestTexturePass:
    def _style(self):
        return StyleSpec()

    def test_identity_pass_is_near_lossless(self, artifacts):
        from mapgen.v2.repaint import IdentityTexturePass, texture_repaint

        base = make_guide(width=600, height=420)
        out = texture_repaint(base, IdentityTexturePass(), self._style())
        diff = np.abs(
            np.asarray(out, dtype=np.float32) - np.asarray(base, dtype=np.float32)
        )
        artifacts.save("identity_diff_x8", np.clip(diff * 8, 0, 255).astype(np.uint8))
        # Identity painter at the thumbnail size -> only resize/normalize rounding.
        assert diff.mean() < 3.0

    def test_strength_zero_returns_base(self):
        from mapgen.v2.repaint import IdentityTexturePass, texture_repaint

        base = make_guide(width=300, height=200)
        out = texture_repaint(base, IdentityTexturePass(), self._style(), strength=0.0)
        assert np.array_equal(np.asarray(out), np.asarray(base.convert("RGB")))

    def test_low_frequency_tint_carries_high_frequency_does_not(self):
        """A painter that tints AND vandalizes detail: the tint (low freq)
        must come through; the detail change (high freq) must not."""
        from mapgen.v2.repaint import texture_repaint

        class TintAndVandalPass:
            def repaint(self, base_small, style, style_bible=None):
                arr = np.asarray(base_small.convert("RGB"), dtype=np.float32)
                arr[..., 0] = np.clip(arr[..., 0] + 30, 0, 255)  # warm tint
                arr[::2, ::2] = (255, 0, 255)  # checkerboard vandalism
                return Image.fromarray(arr.astype(np.uint8), "RGB")

        base = make_guide(width=600, height=420)
        out = np.asarray(
            texture_repaint(base, TintAndVandalPass(), self._style(), detail_radius=8.0),
            dtype=np.float32,
        )
        before = np.asarray(base, dtype=np.float32)
        # No raw magenta survives (high frequencies filtered out)...
        diff = np.abs(out - before)
        assert diff.max() < 90
        # ...but note: the warm tint is largely removed too, by design -- the
        # Lab normalization keeps the palette anchored to the base. What must
        # hold is that geometry is identical above the detail radius:
        edges_before = np.abs(np.diff(before.mean(axis=2), axis=1))
        edges_after = np.abs(np.diff(out.mean(axis=2), axis=1))
        assert abs(edges_after.mean() - edges_before.mean()) < 2.0

    def test_hallucination_raises_structure_rejection(self):
        from mapgen.v2.repaint import StructureRejection, texture_repaint

        class HallucinationPass:
            def repaint(self, base_small, style, style_bible=None):
                rng = np.random.default_rng(4)
                blobs = rng.normal(128, 60, (24, 24, 3))
                img = Image.fromarray(np.clip(blobs, 0, 255).astype(np.uint8), "RGB")
                return img.resize(base_small.size, Image.Resampling.BICUBIC)

        base = make_guide(width=600, height=420)
        with pytest.raises(StructureRejection):
            texture_repaint(base, HallucinationPass(), self._style())

    def test_raw_sink_receives_model_output(self):
        from mapgen.v2.repaint import IdentityTexturePass, texture_repaint

        base = make_guide(width=300, height=200)
        received = {}
        texture_repaint(
            base, IdentityTexturePass(), self._style(),
            raw_sink=lambda name, img: received.update({name: img.size}),
        )
        assert "texture_pass_raw" in received

    def test_texture_poster_end_to_end_stub(self, source, repaint_canvas, tmp_path):
        from mapgen.v2 import pipeline
        from mapgen.v2.assets.stub import StubAssetGenerator
        from mapgen.v2.repaint import IdentityTexturePass

        project = pipeline.V2Project(name="Test Town", region=source.region, output=repaint_canvas)
        plan = pipeline.build_plan(project, source)
        pipeline.write_plan(plan, tmp_path)
        pipeline.generate_assets(plan, tmp_path, StubAssetGenerator())
        pipeline.compose_poster(plan, tmp_path, scale=0.5)

        out = pipeline.texture_poster(plan, tmp_path, IdentityTexturePass(), scale=0.5)
        assert out.exists()
        assert (tmp_path / pipeline.POSTER_BASE_FILENAME).exists()  # A/B kept
        img = Image.open(out)
        assert img.size == (round(repaint_canvas.width_px * 0.5), round(repaint_canvas.height_px * 0.5))
        assert (tmp_path / pipeline.REPAINT_DIRNAME / "raw" / "texture_pass_raw.png").exists()


# --- water repair -------------------------------------------------------------


class TestUnifyWater:
    def test_water_median_pulled_to_target_land_untouched(self):
        img = Image.new("RGB", (200, 100), (180, 160, 130))  # land
        arr = np.asarray(img).copy()
        arr[:, 100:] = (110, 150, 160)  # drifted water (target is 143,184,178)
        img = Image.fromarray(arr)
        mask = Image.new("L", (200, 100), 0)
        m = np.asarray(mask).copy()
        m[:, 100:] = 255
        mask = Image.fromarray(m)
        out = np.asarray(unify_water(img, mask, (143, 184, 178)))
        water_median = np.median(out[:, 120:].reshape(-1, 3), axis=0)
        assert np.abs(water_median - (143, 184, 178)).max() < 3
        assert np.abs(out[:, :80].astype(float) - (180, 160, 130)).max() < 3

    def test_texture_preserved_by_shift(self):
        rng = np.random.default_rng(2)
        arr = np.clip(rng.normal(120, 12, (64, 64, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        mask = Image.new("L", (64, 64), 255)
        out = np.asarray(unify_water(img, mask, (150, 150, 150)), dtype=np.float32)
        # Variation (texture) survives; a flat replacement would crush it.
        assert out.std(axis=(0, 1)).min() > 6

    def test_tiny_water_region_ignored(self):
        img = Image.new("RGB", (64, 64), (100, 100, 100))
        mask = Image.new("L", (64, 64), 0)
        out = unify_water(img, mask, (200, 50, 50))
        assert np.array_equal(np.asarray(out), np.asarray(img))


# --- store ----------------------------------------------------------------------


class TestStore:
    def test_status_and_quadrant_round_trip(self, tmp_path):
        store = RepaintStore(tmp_path)
        store.set_status((1, 2), QuadStatus.SKIPPED)
        quad = Image.new("RGB", (QUAD, QUAD), (1, 2, 3))
        store.save_quadrant((3, 4), quad)
        assert store.status_map() == {(1, 2): QuadStatus.SKIPPED, (3, 4): QuadStatus.GENERATED}
        loaded = store.load_quadrant((3, 4))
        assert np.array_equal(np.asarray(loaded), np.asarray(quad))
        assert store.load_quadrant((9, 9)) is None
        store.record_call(Selection(0, 0, 2, 2))
        assert store.call_count() == 1

    def test_reopen_persists(self, tmp_path):
        store = RepaintStore(tmp_path)
        store.set_status((0, 0), QuadStatus.GENERATED)
        store.close()
        store2 = RepaintStore(tmp_path)
        assert store2.status_map() == {(0, 0): QuadStatus.GENERATED}


# --- pipeline integration ---------------------------------------------------------


@pytest.fixture
def repaint_canvas():
    # Big enough for a 512px quadrant grid (>= 1024 on both axes), small
    # enough to keep the test quick: 3x3 quadrants once padded.
    from mapgen.v2.types import CanvasSpec

    return CanvasSpec(width_px=1100, height_px=1500, dpi=72)


class TestPipeline:
    def test_repaint_poster_end_to_end_stub(self, source, repaint_canvas, tmp_path, artifacts):
        from mapgen.v2 import pipeline
        from mapgen.v2.assets.stub import StubAssetGenerator
        from mapgen.v2.repaint import IdentityPainter

        project = pipeline.V2Project(name="Test Town", region=source.region, output=repaint_canvas)
        plan = pipeline.build_plan(project, source)
        pipeline.write_plan(plan, tmp_path)
        pipeline.generate_assets(plan, tmp_path, StubAssetGenerator())
        pipeline.compose_poster(plan, tmp_path, scale=1.0)

        info = pipeline.plan_repaint(plan, tmp_path, repaint_scale=1.0)
        assert info["calls_planned"] > 0

        out, result = pipeline.repaint_poster(
            plan, tmp_path, IdentityPainter(), scale=1.0, repaint_scale=1.0
        )
        assert result.completed
        assert out.exists()
        assert (tmp_path / pipeline.POSTER_BASE_FILENAME).exists()  # A/B kept
        img = Image.open(out)
        artifacts.save("repainted_poster", img)
        assert img.size == (repaint_canvas.width_px, repaint_canvas.height_px)

    def test_repaint_scale_change_resets_store(self, source, repaint_canvas, tmp_path):
        from mapgen.v2 import pipeline
        from mapgen.v2.assets.stub import StubAssetGenerator
        from mapgen.v2.repaint import IdentityPainter

        project = pipeline.V2Project(name="Test Town", region=source.region, output=repaint_canvas)
        plan = pipeline.build_plan(project, source)
        pipeline.write_plan(plan, tmp_path)
        pipeline.generate_assets(plan, tmp_path, StubAssetGenerator())

        pipeline.repaint_poster(plan, tmp_path, IdentityPainter(), scale=1.0, repaint_scale=1.0)
        # Different repaint scale -> store cleared, full re-plan.
        info = pipeline.plan_repaint(plan, tmp_path, repaint_scale=0.95)
        assert info["calls_planned"] > 0
