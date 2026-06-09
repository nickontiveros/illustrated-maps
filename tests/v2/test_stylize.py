from mapgen.v2.plan.stylize import (
    chaikin_smooth,
    polyline_length,
    prune_roads,
    road_width_px,
    simplify_polyline,
    stylize_polyline,
)
from mapgen.v2.types import RoadClass


def test_simplify_removes_collinear_noise():
    pts = [(0.0, 0.0), (1.0, 0.01), (2.0, -0.01), (3.0, 0.0), (10.0, 0.0)]
    out = simplify_polyline(pts, tolerance=0.5)
    assert out == [(0.0, 0.0), (10.0, 0.0)]


def test_simplify_keeps_real_corners():
    pts = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    out = simplify_polyline(pts, tolerance=0.5)
    assert (5.0, 0.0) in out


def test_chaikin_rounds_corners():
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    out = chaikin_smooth(pts, iterations=2)
    assert len(out) > len(pts)
    assert (10.0, 0.0) not in out[1:-1]  # the sharp corner is cut
    assert out[0] == (0.0, 0.0) and out[-1] == (10.0, 10.0)  # endpoints fixed


def test_road_width_hierarchy():
    w = 7016
    assert road_width_px(RoadClass.MOTORWAY, w) > road_width_px(RoadClass.PRIMARY, w)
    assert road_width_px(RoadClass.PRIMARY, w) > road_width_px(RoadClass.LOCAL, w)
    assert road_width_px(RoadClass.RIVER, w) > road_width_px(RoadClass.MOTORWAY, w)
    assert road_width_px(RoadClass.PATH, 100) >= 2.0  # floor


def test_prune_drops_short_minor_roads_keeps_majors():
    short_local = (RoadClass.LOCAL, [(0.0, 0.0), (5.0, 0.0)])
    long_local = (RoadClass.LOCAL, [(0.0, 0.0), (500.0, 0.0)])
    motorway = (RoadClass.MOTORWAY, [(0.0, 0.0), (3.0, 0.0)])
    kept = prune_roads([short_local, long_local, motorway], canvas_width_px=1000)
    assert motorway in kept
    assert long_local in kept
    assert short_local not in kept


def test_stylize_polyline_pipeline():
    pts = [(0.0, 0.0), (1.0, 0.05), (2.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    out = stylize_polyline(pts, simplify_tolerance=0.3)
    assert len(out) >= 2
    assert polyline_length(out) > 0
