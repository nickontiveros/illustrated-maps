import pytest

from mapgen.v2.plan.distortion import IdentityWarp, ImportanceWarp


def test_identity_when_no_strength():
    warp = ImportanceWarp(centers=[(0.5, 0.5)], strength=0.0)
    for p in [(0.0, 0.0), (0.3, 0.7), (1.0, 1.0)]:
        warped = warp.warp_point(p)
        assert warped[0] == pytest.approx(p[0], abs=1e-6)
        assert warped[1] == pytest.approx(p[1], abs=1e-6)


def test_endpoints_are_fixed():
    warp = ImportanceWarp(centers=[(0.3, 0.6)], strength=1.0)
    assert warp.warp_point((0.0, 0.0)) == pytest.approx((0.0, 0.0))
    assert warp.warp_point((1.0, 1.0)) == pytest.approx((1.0, 1.0))


def test_space_expands_around_importance_center():
    """A small interval near the POI maps to a larger interval than far away."""
    warp = ImportanceWarp(centers=[(0.5, 0.5)], strength=1.0, radius=0.1)
    near = warp.warp_point((0.55, 0.5))[0] - warp.warp_point((0.45, 0.5))[0]
    far = warp.warp_point((0.1, 0.5))[0] - warp.warp_point((0.0, 0.5))[0]
    assert near > 0.1  # expanded beyond its original 0.1 width
    assert far < 0.1  # compressed to compensate


def test_warp_is_monotonic():
    warp = ImportanceWarp(centers=[(0.2, 0.8), (0.7, 0.3)], strength=1.5)
    xs = [warp.warp_point((i / 50, 0.5))[0] for i in range(51)]
    ys = [warp.warp_point((0.5, i / 50))[1] for i in range(51)]
    assert all(a < b for a, b in zip(xs, xs[1:]))
    assert all(a < b for a, b in zip(ys, ys[1:]))


def test_warp_points_matches_warp_point():
    warp = ImportanceWarp(centers=[(0.4, 0.4)], strength=0.8)
    pts = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]
    batch = warp.warp_points(pts)
    single = [warp.warp_point(p) for p in pts]
    for b, s in zip(batch, single):
        assert b == pytest.approx(s)


def test_identity_warp():
    warp = IdentityWarp()
    assert warp.warp_point((0.3, 0.4)) == (0.3, 0.4)
    assert warp.warp_points([(0.1, 0.2)]) == [(0.1, 0.2)]


def test_default_weights_radii_match_legacy_formula():
    """Omitting weights/radii reproduces the original single-radius warp."""
    centers = [(0.3, 0.6), (0.7, 0.4)]
    legacy = ImportanceWarp(centers=centers, strength=1.0, radius=0.15)
    explicit = ImportanceWarp(
        centers=centers, strength=1.0, radius=0.15, weights=[1.0, 1.0], radii=[0.15, 0.15]
    )
    for p in [(0.1, 0.2), (0.5, 0.5), (0.85, 0.7)]:
        assert explicit.warp_point(p) == pytest.approx(legacy.warp_point(p))


def test_heavier_weight_expands_more():
    """A higher-demand center claims more space than an equal-radius peer."""
    light = ImportanceWarp(centers=[(0.3, 0.5), (0.7, 0.5)], strength=1.0, radius=0.08,
                           weights=[1.0, 1.0], radii=[0.08, 0.08])
    heavy = ImportanceWarp(centers=[(0.3, 0.5), (0.7, 0.5)], strength=1.0, radius=0.08,
                           weights=[3.0, 1.0], radii=[0.08, 0.08])

    def span(w, c):
        return w.warp_point((c + 0.04, 0.5))[0] - w.warp_point((c - 0.04, 0.5))[0]

    assert span(heavy, 0.3) > span(light, 0.3)  # heavy center gets more room


def test_narrow_radius_keeps_expansion_local():
    """A tight per-center sigma concentrates magnification near the center."""
    wide = ImportanceWarp(centers=[(0.5, 0.5)], strength=2.0, radii=[0.18])
    narrow = ImportanceWarp(centers=[(0.5, 0.5)], strength=2.0, radii=[0.03])
    # Far from the center, the narrow bump perturbs the mapping far less.
    assert abs(narrow.warp_point((0.15, 0.5))[0] - 0.15) < abs(
        wide.warp_point((0.15, 0.5))[0] - 0.15
    )
