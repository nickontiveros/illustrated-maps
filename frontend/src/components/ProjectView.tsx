import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useProject } from '@/hooks/useProjects';
import { useAppStore } from '@/stores/appStore';
import MapCanvas from './MapCanvas';
import TileGrid from './TileGrid';
import TileDetailPanel from './TileDetailPanel';
import SeamRepair from './SeamRepair';
import Landmarks from './Landmarks';
import Generation from './Generation';
import DeepZoomViewer from './DeepZoomViewer';

function ProjectView() {
  const { name } = useParams<{ name: string }>();
  const { data: project, isLoading, error } = useProject(name);
  const { sidebarTab, setSidebarTab, mapViewMode, setMapViewMode, setCurrentProject } = useAppStore();

  // Track which project is currently being viewed (for legacy store accessors)
  React.useEffect(() => {
    setCurrentProject(name ?? null);
    return () => setCurrentProject(null);
  }, [name, setCurrentProject]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="bg-red-50 text-red-600 p-4 rounded-lg">
          {error ? `Failed to load project: ${error.message}` : 'Project not found'}
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'tiles' as const, label: 'Tiles', icon: '🔲' },
    { id: 'seams' as const, label: 'Seams', icon: '🔗' },
    { id: 'landmarks' as const, label: 'Landmarks', icon: '📍' },
    { id: 'settings' as const, label: 'Settings', icon: '⚙️' },
  ];

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-slate-400 hover:text-slate-600">
            ← Back
          </Link>
          <h1 className="text-xl font-semibold text-slate-800">{project.name}</h1>
          <div className="text-sm text-slate-500">
            {project.grid_cols}×{project.grid_rows} tiles • {project.area_km2.toFixed(1)} km²
          </div>
        </div>
        <Generation projectName={name!} />
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 bg-white border-r border-slate-200 flex flex-col">
          {/* Tabs */}
          <div className="flex border-b border-slate-200">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSidebarTab(tab.id)}
                className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                  sidebarTab === tab.id
                    ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                    : 'text-slate-600 hover:text-slate-800 hover:bg-slate-50'
                }`}
              >
                <span className="mr-1">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto">
            {sidebarTab === 'tiles' && <TileGrid projectName={name!} />}
            {sidebarTab === 'seams' && <SeamRepair projectName={name!} />}
            {sidebarTab === 'landmarks' && <Landmarks projectName={name!} />}
            {sidebarTab === 'settings' && <ProjectSettings project={project} />}
          </div>
        </div>

        {/* Map */}
        <div className="flex-1 flex flex-col">
          {/* View toggle */}
          <div className="bg-white border-b border-slate-200 px-4 py-2 flex items-center gap-2">
            <span className="text-sm text-slate-500">View:</span>
            <button
              onClick={() => setMapViewMode('geographic')}
              className={`px-3 py-1 text-sm rounded ${
                mapViewMode === 'geographic'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              Geographic
            </button>
            <button
              onClick={() => setMapViewMode('tiles')}
              className={`px-3 py-1 text-sm rounded ${
                mapViewMode === 'tiles'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              Assembled Image
            </button>
            <button
              onClick={() => setMapViewMode('tile-detail')}
              className={`px-3 py-1 text-sm rounded ${
                mapViewMode === 'tile-detail'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              Tile Detail
            </button>
          </div>

          {/* Map, Deep Zoom viewer, or Tile Detail */}
          <div className="flex-1 overflow-auto">
            {mapViewMode === 'geographic' ? (
              <MapCanvas project={project} />
            ) : mapViewMode === 'tiles' ? (
              <DeepZoomViewer projectName={name!} className="w-full h-full" />
            ) : (
              <TileDetailPanel projectName={name!} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ProjectSettings({ project }: { project: import('@/types').ProjectDetail }) {
  return (
    <div className="p-4 space-y-6">
      <section>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Region</h3>
        <div className="bg-slate-50 rounded-lg p-3 text-sm">
          <div className="grid grid-cols-2 gap-2">
            <div>North: {project.region.north.toFixed(4)}°</div>
            <div>South: {project.region.south.toFixed(4)}°</div>
            <div>East: {project.region.east.toFixed(4)}°</div>
            <div>West: {project.region.west.toFixed(4)}°</div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Output</h3>
        <div className="bg-slate-50 rounded-lg p-3 text-sm space-y-1">
          <div>Size: {project.output.width}×{project.output.height}px</div>
          <div>DPI: {project.output.dpi}</div>
        </div>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Style</h3>
        <div className="bg-slate-50 rounded-lg p-3 text-sm space-y-1">
          <div>Orientation: {project.style.orientation}</div>
          <div>Perspective: {project.style.perspective_angle}°</div>
        </div>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Tiles</h3>
        <div className="bg-slate-50 rounded-lg p-3 text-sm space-y-1">
          <div>Size: {project.tiles.size}px</div>
          <div>Overlap: {project.tiles.overlap}px</div>
          <div>Grid: {project.grid_cols}×{project.grid_rows}</div>
          <div>Total: {project.tile_count} tiles</div>
        </div>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Detail Level</h3>
        <div className="bg-slate-50 rounded-lg p-3 text-sm">
          {project.detail_level}
        </div>
      </section>
    </div>
  );
}

export default ProjectView;
