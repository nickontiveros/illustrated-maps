"""New land-use ground classes are wired end to end: enum -> texture hint ->
asset manifest, and the OSM terrain classifier recognises them."""

from mapgen.services.osm_service import OSMService
from mapgen.v2.assets.manifest import TEXTURE_HINTS, build_manifest
from mapgen.v2.types import GroundClass


def test_every_ground_class_has_a_texture_hint():
    # manifest indexes TEXTURE_HINTS[cls] directly -- a missing member KeyErrors
    # at plan time, so this guards every future enum addition.
    assert set(GroundClass) <= set(TEXTURE_HINTS)


def test_new_classes_present():
    for cls in (GroundClass.GOLF, GroundClass.CEMETERY, GroundClass.BEACH, GroundClass.WETLAND):
        assert cls in GroundClass
        assert TEXTURE_HINTS[cls]


def test_build_manifest_emits_new_textures():
    classes = {GroundClass.GOLF, GroundClass.WETLAND, GroundClass.LAND}
    manifest = build_manifest(classes, set(), [])
    ids = {a.id for a in manifest}
    assert "texture_golf" in ids
    assert "texture_wetland" in ids


def test_classify_terrain_recognises_new_tags():
    svc = OSMService.__new__(OSMService)  # no network/cache setup needed
    assert svc._classify_terrain({"landuse": "cemetery"}) == "cemetery"
    assert svc._classify_terrain({"amenity": "grave_yard"}) == "cemetery"
    assert svc._classify_terrain({"natural": "beach"}) == "beach"
    assert svc._classify_terrain({"natural": "wetland"}) == "wetland"
    # Plain water still classifies as water, not wetland.
    assert svc._classify_terrain({"natural": "water"}) == "water"
