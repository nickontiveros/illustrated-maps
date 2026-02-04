import { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Rectangle, Marker, Popup, useMap } from 'react-leaflet';
import { LatLngBounds, LatLng } from 'leaflet';
import type { ProjectDetail, TileSpec, LandmarkDetail } from '@/types';
import { useTileGrid } from '@/hooks/useTiles';
import { useLandmarks } from '@/hooks/useLandmarks';
import { useAppStore } from '@/stores/appStore';
import { api } from '@/api/client';
import 'leaflet/dist/leaflet.css';

interface MapCanvasProps {
  project: ProjectDetail;
}

function MapCanvas({ project }: MapCanvasProps) {
  const { data: tileGrid } = useTileGrid(project.name);
  const { data: landmarks } = useLandmarks(project.name);
  const { selectedTile, setSelectedTile, selectedLandmark, setSelectedLandmark, sidebarTab } = useAppStore();

  const bounds = new LatLngBounds(
    [project.region.south, project.region.west],
    [project.region.north, project.region.east]
  );

  const center: [number, number] = [
    (project.region.north + project.region.south) / 2,
    (project.region.east + project.region.west) / 2,
  ];

  return (
    <MapContainer
      center={center}
      zoom={14}
      className="h-full w-full"
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* Project bounds */}
      <Rectangle
        bounds={bounds}
        pathOptions={{
          color: '#2563eb',
          weight: 2,
          fillOpacity: 0.1,
        }}
      />

      {/* Tile grid overlay */}
      {sidebarTab === 'tiles' && tileGrid && (
        <TileGridOverlay
          tiles={tileGrid.tiles}
          projectName={project.name}
          selectedTile={selectedTile}
          onSelectTile={setSelectedTile}
        />
      )}

      {/* Landmarks */}
      {sidebarTab === 'landmarks' && landmarks && (
        <LandmarkMarkers
          landmarks={landmarks}
          projectName={project.name}
          selectedLandmark={selectedLandmark}
          onSelectLandmark={setSelectedLandmark}
        />
      )}

      <FitBoundsOnMount bounds={bounds} />
    </MapContainer>
  );
}

function FitBoundsOnMount({ bounds }: { bounds: LatLngBounds }) {
  const map = useMap();

  useEffect(() => {
    map.fitBounds(bounds, { padding: [20, 20] });
  }, [map, bounds]);

  return null;
}

interface TileGridOverlayProps {
  tiles: TileSpec[];
  projectName: string;
  selectedTile: TileSpec | null;
  onSelectTile: (tile: TileSpec | null) => void;
}

function TileGridOverlay({ tiles, projectName, selectedTile, onSelectTile }: TileGridOverlayProps) {
  return (
    <>
      {tiles.map((tile) => {
        const isSelected = selectedTile?.col === tile.col && selectedTile?.row === tile.row;
        const bounds = new LatLngBounds(
          [tile.bbox.south, tile.bbox.west],
          [tile.bbox.north, tile.bbox.east]
        );

        let color = '#64748b'; // slate
        if (tile.status === 'completed') color = '#16a34a'; // green
        else if (tile.status === 'generating') color = '#2563eb'; // blue
        else if (tile.status === 'failed') color = '#dc2626'; // red

        return (
          <Rectangle
            key={`${tile.col}-${tile.row}`}
            bounds={bounds}
            pathOptions={{
              color: isSelected ? '#8b5cf6' : color,
              weight: isSelected ? 3 : 1,
              fillOpacity: isSelected ? 0.3 : 0.1,
              fillColor: color,
            }}
            eventHandlers={{
              click: () => onSelectTile(isSelected ? null : tile),
            }}
          >
            <Popup>
              <div className="text-sm">
                <div className="font-semibold">Tile ({tile.col}, {tile.row})</div>
                <div className="text-slate-500">{tile.position_desc}</div>
                <div className="mt-1">
                  Status: <span className={`font-medium ${
                    tile.status === 'completed' ? 'text-green-600' :
                    tile.status === 'failed' ? 'text-red-600' :
                    'text-slate-600'
                  }`}>{tile.status}</span>
                </div>
                {tile.has_generated && (
                  <img
                    src={api.getTileThumbnailUrl(projectName, tile.col, tile.row)}
                    alt="Tile preview"
                    className="mt-2 w-32 h-32 object-cover rounded"
                  />
                )}
              </div>
            </Popup>
          </Rectangle>
        );
      })}
    </>
  );
}

interface LandmarkMarkersProps {
  landmarks: LandmarkDetail[];
  projectName: string;
  selectedLandmark: LandmarkDetail | null;
  onSelectLandmark: (landmark: LandmarkDetail | null) => void;
}

function LandmarkMarkers({ landmarks, projectName, selectedLandmark, onSelectLandmark }: LandmarkMarkersProps) {
  return (
    <>
      {landmarks.map((landmark) => {
        const isSelected = selectedLandmark?.name === landmark.name;

        return (
          <Marker
            key={landmark.name}
            position={[landmark.latitude, landmark.longitude]}
            eventHandlers={{
              click: () => onSelectLandmark(isSelected ? null : landmark),
            }}
          >
            <Popup>
              <div className="text-sm">
                <div className="font-semibold">{landmark.name}</div>
                <div className="text-slate-500">
                  Scale: {landmark.scale}x • Z-index: {landmark.z_index}
                </div>
                {landmark.has_photo && (
                  <img
                    src={api.getLandmarkPhotoUrl(projectName, landmark.name)}
                    alt={landmark.name}
                    className="mt-2 w-32 h-24 object-cover rounded"
                  />
                )}
              </div>
            </Popup>
          </Marker>
        );
      })}
    </>
  );
}

export default MapCanvas;
