"""V2 asset-composition architecture.

See V2_DESIGN.md. Code owns everything global (geometry, perspective,
layout, resolution); AI generates only small local assets (POI sprites,
textures, sprite libraries, ornaments) composited at native print DPI.
"""

__all__ = ["types", "plan", "assets", "compose", "pipeline"]
