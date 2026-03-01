import { useCallback, useRef, useState, useEffect } from 'react';
import Map, { NavigationControl, type MapRef } from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { OrientedRegion } from '@/types';

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';

// Default dimensions for initial click placement
const DEFAULT_WIDTH_KM = 10;
const DEFAULT_HEIGHT_KM = 15;

interface RegionDrawerProps {
  value: OrientedRegion | null;
  onChange: (region: OrientedRegion | null) => void;
  rotation: number;
}

// Rotate a geographic point around a center, accounting for latitude scaling
function rotateGeoPoint(
  lng: number, lat: number,
  centerLng: number, centerLat: number,
  angleDeg: number
): [number, number] {
  const rad = (angleDeg * Math.PI) / 180;
  const cosLat = Math.cos((centerLat * Math.PI) / 180);
  const dx = (lng - centerLng) * cosLat;
  const dy = lat - centerLat;
  const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
  const ry = dx * Math.sin(rad) + dy * Math.cos(rad);
  return [centerLng + rx / cosLat, centerLat + ry];
}

// Convert km to degrees at a given latitude
function kmToLonDeg(km: number, lat: number): number {
  return km / (111.32 * Math.cos((lat * Math.PI) / 180));
}
function kmToLatDeg(km: number): number {
  return km / 111.32;
}

// Get the 4 corners of an oriented region in [lng, lat] format
function getCorners(region: OrientedRegion): [number, number][] {
  const halfW = kmToLonDeg(region.width_km / 2, region.center_lat);
  const halfH = kmToLatDeg(region.height_km / 2);

  // Unrotated corners (NW, NE, SE, SW)
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
}

// Midpoint between two [lng, lat] points
function midpoint(a: [number, number], b: [number, number]): [number, number] {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

type DragType = 'n' | 's' | 'e' | 'w' | null;

// SVG overlay for the region rectangle with handles
function RegionOverlay({
  map,
  region,
  onDragStart,
  onDrag,
  onDragEnd,
}: {
  map: maplibregl.Map;
  region: OrientedRegion;
  onDragStart: (type: DragType, startLngLat: [number, number]) => void;
  onDrag: (lngLat: [number, number]) => void;
  onDragEnd: () => void;
}) {
  const [, setTick] = useState(0);
  const rerender = useCallback(() => setTick((t) => t + 1), []);
  const dragging = useRef(false);

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

  const corners = getCorners(region);
  const projected = corners.map(([lng, lat]) => map.project([lng, lat]));

  const path = `M${projected[0].x},${projected[0].y} L${projected[1].x},${projected[1].y} L${projected[2].x},${projected[2].y} L${projected[3].x},${projected[3].y} Z`;

  // Edge midpoints for resize handles: N (top), E (right), S (bottom), W (left)
  const edgeMids = {
    n: midpoint(corners[0], corners[1]),
    e: midpoint(corners[1], corners[2]),
    s: midpoint(corners[2], corners[3]),
    w: midpoint(corners[3], corners[0]),
  };

  const centerPx = map.project([region.center_lon, region.center_lat]);

  const canvas = map.getCanvas();
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  const handleSize = 8;

  const handleMouseDown = (type: DragType) => (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    dragging.current = true;
    const rect = canvas.getBoundingClientRect();
    const point = map.unproject([e.clientX - rect.left, e.clientY - rect.top]);
    onDragStart(type, [point.lng, point.lat]);
  };

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const rect = canvas.getBoundingClientRect();
      const point = map.unproject([e.clientX - rect.left, e.clientY - rect.top]);
      onDrag([point.lng, point.lat]);
    };
    const onMouseUp = () => {
      if (dragging.current) {
        dragging.current = false;
        onDragEnd();
      }
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [map, canvas, onDrag, onDragEnd]);

  const renderHandle = (key: DragType, lngLat: [number, number], cursor: string) => {
    const px = map.project(lngLat);
    return (
      <rect
        key={key}
        x={px.x - handleSize / 2}
        y={px.y - handleSize / 2}
        width={handleSize}
        height={handleSize}
        fill="white"
        stroke="#2563eb"
        strokeWidth={2}
        style={{ cursor, pointerEvents: 'all' }}
        onMouseDown={handleMouseDown(key)}
      />
    );
  };

  return (
    <svg
      style={{ position: 'absolute', top: 0, left: 0, width: w, height: h, pointerEvents: 'none' }}
    >
      {/* Region fill + border (not interactive — map pans through it) */}
      <path
        d={path}
        fill="rgba(37, 99, 235, 0.1)"
        stroke="#2563eb"
        strokeWidth={2}
      />

      {/* Edge resize handles */}
      {renderHandle('n', edgeMids.n, 'ns-resize')}
      {renderHandle('s', edgeMids.s, 'ns-resize')}
      {renderHandle('e', edgeMids.e, 'ew-resize')}
      {renderHandle('w', edgeMids.w, 'ew-resize')}

      {/* Center dot (visual only) */}
      <circle
        cx={centerPx.x}
        cy={centerPx.y}
        r={4}
        fill="#2563eb"
        stroke="white"
        strokeWidth={2}
      />
    </svg>
  );
}

export default function RegionDrawer({ value, onChange, rotation }: RegionDrawerProps) {
  const mapRef = useRef<MapRef>(null);
  const [mapInstance, setMapInstance] = useState<maplibregl.Map | null>(null);
  const dragRef = useRef<{
    type: DragType;
    startLngLat: [number, number];
    startRegion: OrientedRegion;
  } | null>(null);

  // Sync rotation prop into the region
  useEffect(() => {
    if (value && value.rotation_deg !== rotation) {
      onChange({ ...value, rotation_deg: rotation });
    }
  }, [rotation]); // eslint-disable-line react-hooks/exhaustive-deps

  const onLoad = useCallback((evt: { target: maplibregl.Map }) => {
    setMapInstance(evt.target);
  }, []);

  // Double-click to place or reposition region center
  const handleMapDblClick = useCallback((evt: maplibregl.MapLayerMouseEvent) => {
    evt.preventDefault(); // prevent default zoom-in on double-click
    const { lng, lat } = evt.lngLat;
    onChange({
      center_lat: lat,
      center_lon: lng,
      width_km: value?.width_km ?? DEFAULT_WIDTH_KM,
      height_km: value?.height_km ?? DEFAULT_HEIGHT_KM,
      rotation_deg: rotation,
    });
  }, [value, onChange, rotation]);

  // Fit map to region when it changes
  useEffect(() => {
    if (!mapInstance || !value) return;
    const corners = getCorners(value);
    const lngs = corners.map(c => c[0]);
    const lats = corners.map(c => c[1]);
    const bounds = new maplibregl.LngLatBounds(
      [Math.min(...lngs), Math.min(...lats)],
      [Math.max(...lngs), Math.max(...lats)]
    );
    mapInstance.fitBounds(bounds, { padding: 80, duration: 0 });
  }, [mapInstance, value?.center_lat, value?.center_lon]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDragStart = useCallback((type: DragType, startLngLat: [number, number]) => {
    if (!value) return;
    dragRef.current = { type, startLngLat, startRegion: { ...value } };
  }, [value]);

  const handleDrag = useCallback((lngLat: [number, number]) => {
    const drag = dragRef.current;
    if (!drag || !drag.type) return;

    const sr = drag.startRegion;
    const [lng, lat] = lngLat;

    // For edge handles, compute distance from center along the rotated axis
    const cosLat = Math.cos((sr.center_lat * Math.PI) / 180);
    const dlng = (lng - sr.center_lon) * cosLat * 111.32; // km east
    const dlat = (lat - sr.center_lat) * 111.32; // km north

    // Rotate delta into region-local coordinates
    const rad = (-sr.rotation_deg * Math.PI) / 180;
    const localX = dlng * Math.cos(rad) - dlat * Math.sin(rad); // east in region frame
    const localY = dlng * Math.sin(rad) + dlat * Math.cos(rad); // north in region frame

    let newW = sr.width_km;
    let newH = sr.height_km;

    // Each edge handle extends from center, so new half-extent = |local distance|
    if (drag.type === 'n' || drag.type === 's') {
      newH = Math.max(1, Math.abs(localY) * 2);
    } else if (drag.type === 'e' || drag.type === 'w') {
      newW = Math.max(1, Math.abs(localX) * 2);
    }

    onChange({ ...sr, width_km: newW, height_km: newH });
  }, [onChange]);

  const handleDragEnd = useCallback(() => {
    dragRef.current = null;
  }, []);

  const initialViewState = value
    ? { longitude: value.center_lon, latitude: value.center_lat, zoom: 10 }
    : { longitude: -98.5, latitude: 39.8, zoom: 3 }; // Default: center of US

  return (
    <div className="relative w-full h-64 rounded-lg overflow-hidden border border-slate-300">
      <Map
        ref={mapRef}
        initialViewState={initialViewState}
        style={{ width: '100%', height: '100%' }}
        mapStyle={MAP_STYLE}
        onLoad={onLoad}
        onDblClick={handleMapDblClick}
        doubleClickZoom={false}
      >
        <NavigationControl position="top-right" showCompass={false} />
      </Map>

      {mapInstance && value && (
        <RegionOverlay
          map={mapInstance}
          region={value}
          onDragStart={handleDragStart}
          onDrag={handleDrag}
          onDragEnd={handleDragEnd}
        />
      )}

      {/* Reset button */}
      {value && (
        <button
          onClick={() => onChange(null)}
          className="absolute top-2 left-2 bg-white/90 text-slate-600 text-xs px-2 py-1 rounded shadow hover:bg-white"
        >
          Reset
        </button>
      )}

      {/* Instruction overlay */}
      {!value && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="bg-black/50 text-white px-4 py-2 rounded-lg text-sm">
            Double-click to place region center
          </div>
        </div>
      )}
    </div>
  );
}
