import { useCallback, useRef, useState, useEffect } from 'react';
import Map, {
  Marker,
  Popup,
  NavigationControl,
  type MapRef,
} from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { ProjectDetail, TileSpec, LandmarkDetail, BoundingBox } from '@/types';
import { useTileGrid } from '@/hooks/useTiles';
import { useLandmarks } from '@/hooks/useLandmarks';
import { useAppStore } from '@/stores/appStore';
import { api } from '@/api/client';

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';

interface MapCanvasProps {
  project: ProjectDetail;
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
  tileGrid: { tiles: TileSpec[] } | null;
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

  const projectRect = (bbox: BoundingBox) => {
    const nw = map.project([bbox.west, bbox.north]);
    const ne = map.project([bbox.east, bbox.north]);
    const se = map.project([bbox.east, bbox.south]);
    const sw = map.project([bbox.west, bbox.south]);
    return `M${nw.x},${nw.y} L${ne.x},${ne.y} L${se.x},${se.y} L${sw.x},${sw.y} Z`;
  };

  const canvas = map.getCanvas();
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  return (
    <svg
      style={{ position: 'absolute', top: 0, left: 0, width: w, height: h, pointerEvents: 'none' }}
    >
      {/* Project bounds */}
      <path
        d={projectRect(project.region)}
        fill="rgba(37, 99, 235, 0.1)"
        stroke="#2563eb"
        strokeWidth={2}
      />

      {/* Tile grid */}
      {showTiles &&
        tileGrid?.tiles.map((tile) => {
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
        })}
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
