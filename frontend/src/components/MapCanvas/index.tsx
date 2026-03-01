import { useCallback, useRef, useState, useEffect } from 'react';
import Map, {
  Marker,
  Popup,
  NavigationControl,
  type MapRef,
} from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { ProjectDetail, TileSpec, LandmarkDetail, BoundingBox, OrientedRegion } from '@/types';
import { useTileGrid } from '@/hooks/useTiles';
import { useLandmarks } from '@/hooks/useLandmarks';
import { useAppStore } from '@/stores/appStore';
import { api } from '@/api/client';

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';

interface MapCanvasProps {
  project: ProjectDetail;
}

// Rotate a geographic point around a center, accounting for latitude scaling
function rotateGeoPoint(
  lng: number, lat: number,
  centerLng: number, centerLat: number,
  angleDeg: number
): [number, number] {
  const rad = (angleDeg * Math.PI) / 180;
  const cosLat = Math.cos((centerLat * Math.PI) / 180);
  // Convert to local Cartesian with latitude scaling
  const dx = (lng - centerLng) * cosLat;
  const dy = lat - centerLat;
  // Rotate
  const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
  const ry = dx * Math.sin(rad) + dy * Math.cos(rad);
  // Convert back
  return [centerLng + rx / cosLat, centerLat + ry];
}

// SVG overlay that draws bounding box and tile grid using map.project()
function MapOverlay({
  map,
  project,
  tileGrid,
  selectedTile,
  showTiles,
}: {
  map: maplibregl.Map;
  project: ProjectDetail;
  tileGrid: { tiles: TileSpec[]; cols: number; rows: number } | null;
  selectedTile: TileSpec | null;
  showTiles: boolean;
}) {
  const [, setTick] = useState(0);
  const rerender = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    map.on('move', rerender);
    map.on('zoom', rerender);
    map.on('resize', rerender);
    return () => {
      map.off('move', rerender);
      map.off('zoom', rerender);
      map.off('resize', rerender);
    };
  }, [map, rerender]);

  const orientationDeg = project.style.orientation_degrees ?? 0;
  const or = project.oriented_region;

  // Convert km to degrees at a given latitude
  const kmToLonDeg = (km: number, lat: number) => km / (111.32 * Math.cos((lat * Math.PI) / 180));
  const kmToLatDeg = (km: number) => km / 111.32;

  // Get rotated corners of an OrientedRegion in [lng, lat]
  const getOrientedCorners = (region: OrientedRegion): [number, number][] => {
    const halfW = kmToLonDeg(region.width_km / 2, region.center_lat);
    const halfH = kmToLatDeg(region.height_km / 2);
    const unrotated: [number, number][] = [
      [region.center_lon - halfW, region.center_lat + halfH],
      [region.center_lon + halfW, region.center_lat + halfH],
      [region.center_lon + halfW, region.center_lat - halfH],
      [region.center_lon - halfW, region.center_lat - halfH],
    ];
    if (region.rotation_deg === 0) return unrotated;
    return unrotated.map(([lng, lat]) =>
      rotateGeoPoint(lng, lat, region.center_lon, region.center_lat, region.rotation_deg)
    );
  };

  // Project a bbox to an SVG path, optionally rotated by orientation
  const projectRect = (bbox: BoundingBox, rotate = false) => {
    const corners: [number, number][] = [
      [bbox.west, bbox.north], // NW
      [bbox.east, bbox.north], // NE
      [bbox.east, bbox.south], // SE
      [bbox.west, bbox.south], // SW
    ];

    let projected;
    if (rotate && orientationDeg !== 0) {
      const centerLng = (bbox.west + bbox.east) / 2;
      const centerLat = (bbox.north + bbox.south) / 2;
      projected = corners.map(([lng, lat]) => {
        const [rLng, rLat] = rotateGeoPoint(lng, lat, centerLng, centerLat, orientationDeg);
        return map.project([rLng, rLat]);
      });
    } else {
      projected = corners.map(([lng, lat]) => map.project([lng, lat]));
    }

    return `M${projected[0].x},${projected[0].y} L${projected[1].x},${projected[1].y} L${projected[2].x},${projected[2].y} L${projected[3].x},${projected[3].y} Z`;
  };

  // Project oriented region corners to SVG path
  const projectOrientedRegion = (region: OrientedRegion) => {
    const corners = getOrientedCorners(region);
    const projected = corners.map(([lng, lat]) => map.project([lng, lat]));
    return `M${projected[0].x},${projected[0].y} L${projected[1].x},${projected[1].y} L${projected[2].x},${projected[2].y} L${projected[3].x},${projected[3].y} Z`;
  };

  const canvas = map.getCanvas();
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  return (
    <svg
      style={{ position: 'absolute', top: 0, left: 0, width: w, height: h, pointerEvents: 'none' }}
    >
      {/* Project bounds — oriented region or rotated bbox */}
      <path
        d={or ? projectOrientedRegion(or) : projectRect(project.region, true)}
        fill="rgba(37, 99, 235, 0.1)"
        stroke="#2563eb"
        strokeWidth={2}
      />

      {/* Tile grid */}
      {showTiles && tileGrid && (() => {
        // For oriented regions, subdivide the oriented rectangle into a visual grid
        // so tiles align with the blue bounding box instead of the expanded generation bbox.
        if (or) {
          const { cols, rows } = tileGrid;
          const halfW = kmToLonDeg(or.width_km / 2, or.center_lat);
          const halfH = kmToLatDeg(or.height_km / 2);
          // Build a lookup from (col,row) -> tile for status/selection
          const tileLookup: Record<string, TileSpec> = {};
          for (const t of tileGrid.tiles) tileLookup[`${t.col},${t.row}`] = t;

          const cells = [];
          for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
              // Cell bounds in unrotated local coords (fraction of region)
              const left = -halfW + (col / cols) * 2 * halfW;
              const right = -halfW + ((col + 1) / cols) * 2 * halfW;
              const top = halfH - (row / rows) * 2 * halfH;
              const bottom = halfH - ((row + 1) / rows) * 2 * halfH;

              // 4 corners in geo coords, then rotate
              const cellCorners: [number, number][] = [
                [or.center_lon + left, or.center_lat + top],
                [or.center_lon + right, or.center_lat + top],
                [or.center_lon + right, or.center_lat + bottom],
                [or.center_lon + left, or.center_lat + bottom],
              ];
              const rotated = or.rotation_deg === 0
                ? cellCorners
                : cellCorners.map(([lng, lat]) =>
                    rotateGeoPoint(lng, lat, or.center_lon, or.center_lat, or.rotation_deg)
                  );

              const projected = rotated.map(([lng, lat]) => map.project([lng, lat]));
              const d = `M${projected[0].x},${projected[0].y} L${projected[1].x},${projected[1].y} L${projected[2].x},${projected[2].y} L${projected[3].x},${projected[3].y} Z`;

              const tile = tileLookup[`${col},${row}`];
              const isSelected = selectedTile?.col === col && selectedTile?.row === row;
              let color = '#64748b';
              if (tile?.status === 'completed') color = '#16a34a';
              else if (tile?.status === 'generating') color = '#2563eb';
              else if (tile?.status === 'failed') color = '#dc2626';

              cells.push(
                <path
                  key={`${col}-${row}`}
                  d={d}
                  fill={isSelected ? 'rgba(139, 92, 246, 0.3)' : `${color}19`}
                  stroke={isSelected ? '#8b5cf6' : color}
                  strokeWidth={isSelected ? 3 : 1}
                />
              );
            }
          }
          return cells;
        }

        // Legacy: draw tile bboxes directly
        return tileGrid.tiles.map((tile) => {
          const isSelected = selectedTile?.col === tile.col && selectedTile?.row === tile.row;
          let color = '#64748b';
          if (tile.status === 'completed') color = '#16a34a';
          else if (tile.status === 'generating') color = '#2563eb';
          else if (tile.status === 'failed') color = '#dc2626';

          return (
            <path
              key={`${tile.col}-${tile.row}`}
              d={projectRect(tile.bbox)}
              fill={isSelected ? 'rgba(139, 92, 246, 0.3)' : `${color}19`}
              stroke={isSelected ? '#8b5cf6' : color}
              strokeWidth={isSelected ? 3 : 1}
            />
          );
        });
      })()}
    </svg>
  );
}

function MapCanvas({ project }: MapCanvasProps) {
  const mapRef = useRef<MapRef>(null);
  const [mapInstance, setMapInstance] = useState<maplibregl.Map | null>(null);
  const { data: tileGrid } = useTileGrid(project.name);
  const { data: landmarks } = useLandmarks(project.name);
  const { selectedTile, setSelectedTile, selectedLandmark, setSelectedLandmark, sidebarTab } = useAppStore();
  const [popupTile, setPopupTile] = useState<TileSpec | null>(null);
  const [popupLandmark, setPopupLandmark] = useState<LandmarkDetail | null>(null);

  const bearing = project.style.orientation_degrees ?? 0;

  const center: [number, number] = [
    (project.region.east + project.region.west) / 2,
    (project.region.north + project.region.south) / 2,
  ];

  // Fit to project bounds on load
  const onLoad = useCallback((evt: { target: maplibregl.Map }) => {
    const map = evt.target;
    setMapInstance(map);
    map.fitBounds(
      [[project.region.west, project.region.south], [project.region.east, project.region.north]],
      { padding: 50, bearing, duration: 0 }
    );
  }, [project.region, bearing]);

  // Animate bearing changes
  useEffect(() => {
    if (mapInstance) {
      mapInstance.easeTo({ bearing, duration: 500 });
    }
  }, [bearing, mapInstance]);

  // Handle clicks on tile grid area
  const handleClick = useCallback(
    (evt: maplibregl.MapLayerMouseEvent) => {
      if (sidebarTab !== 'tiles' || !tileGrid) return;
      const map = mapRef.current;
      if (!map) return;

      // Find which tile bbox contains the clicked point
      const { lng, lat } = evt.lngLat;
      const tile = tileGrid.tiles.find(
        (t) =>
          lng >= t.bbox.west &&
          lng <= t.bbox.east &&
          lat >= t.bbox.south &&
          lat <= t.bbox.north
      );
      if (!tile) return;

      const isSelected = selectedTile?.col === tile.col && selectedTile?.row === tile.row;
      setSelectedTile(isSelected ? null : tile);
      setPopupTile(isSelected ? null : tile);
    },
    [sidebarTab, tileGrid, selectedTile, setSelectedTile]
  );

  return (
    <div className="h-full w-full relative">
      <Map
        key={project.name}
        ref={mapRef}
        initialViewState={{
          longitude: center[0],
          latitude: center[1],
          zoom: 10,
          bearing,
        }}
        style={{ width: '100%', height: '100%' }}
        mapStyle={MAP_STYLE}
        onLoad={onLoad}
        onClick={handleClick}
      >
        <NavigationControl position="top-right" showCompass visualizePitch={false} />

        {/* Tile popup */}
        {popupTile && sidebarTab === 'tiles' && (
          <Popup
            longitude={(popupTile.bbox.east + popupTile.bbox.west) / 2}
            latitude={(popupTile.bbox.north + popupTile.bbox.south) / 2}
            onClose={() => setPopupTile(null)}
            closeOnClick={false}
            maxWidth="250px"
          >
            <div className="text-sm">
              <div className="font-semibold">Tile ({popupTile.col}, {popupTile.row})</div>
              <div className="text-slate-500">{popupTile.position_desc}</div>
              <div className="mt-1">
                Status:{' '}
                <span
                  className={`font-medium ${
                    popupTile.status === 'completed'
                      ? 'text-green-600'
                      : popupTile.status === 'failed'
                        ? 'text-red-600'
                        : 'text-slate-600'
                  }`}
                >
                  {popupTile.status}
                </span>
              </div>
              {popupTile.has_generated && (
                <img
                  src={api.getTileThumbnailUrl(project.name, popupTile.col, popupTile.row)}
                  alt="Tile preview"
                  className="mt-2 w-32 h-32 object-cover rounded"
                />
              )}
            </div>
          </Popup>
        )}

        {/* Landmark markers */}
        {sidebarTab === 'landmarks' &&
          landmarks?.map((landmark) => {
            const isSelected = selectedLandmark?.name === landmark.name;
            return (
              <Marker
                key={landmark.name}
                longitude={landmark.longitude}
                latitude={landmark.latitude}
                anchor="bottom"
                onClick={(e) => {
                  e.originalEvent.stopPropagation();
                  setSelectedLandmark(isSelected ? null : landmark);
                  setPopupLandmark(isSelected ? null : landmark);
                }}
              >
                <div
                  className={`w-6 h-6 rounded-full border-2 flex items-center justify-center text-xs cursor-pointer ${
                    isSelected
                      ? 'bg-blue-600 border-blue-800 text-white'
                      : 'bg-white border-blue-500 text-blue-700'
                  }`}
                >
                  {landmark.has_illustration ? '★' : '●'}
                </div>
              </Marker>
            );
          })}

        {/* Landmark popup */}
        {popupLandmark && sidebarTab === 'landmarks' && (
          <Popup
            longitude={popupLandmark.longitude}
            latitude={popupLandmark.latitude}
            onClose={() => setPopupLandmark(null)}
            closeOnClick={false}
            offset={[0, -10] as [number, number]}
            maxWidth="250px"
          >
            <div className="text-sm">
              <div className="font-semibold">{popupLandmark.name}</div>
              <div className="text-slate-500">
                Scale: {popupLandmark.scale}x | Z-index: {popupLandmark.z_index}
              </div>
              {popupLandmark.has_photo && (
                <img
                  src={api.getLandmarkPhotoUrl(project.name, popupLandmark.name)}
                  alt={popupLandmark.name}
                  className="mt-2 w-32 h-24 object-cover rounded"
                />
              )}
            </div>
          </Popup>
        )}
      </Map>

      {/* SVG overlay for bounding box and tile grid */}
      {mapInstance && (
        <MapOverlay
          map={mapInstance}
          project={project}
          tileGrid={tileGrid ?? null}
          selectedTile={selectedTile}
          showTiles={sidebarTab === 'tiles'}
        />
      )}
    </div>
  );
}

export default MapCanvas;
