import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useProjects, useCreateProject, useDeleteProject } from '@/hooks/useProjects';
import { useAppStore } from '@/stores/appStore';

function ProjectList() {
  const { data: projects, isLoading, error } = useProjects();
  const createProject = useCreateProject();
  const deleteProject = useDeleteProject();
  const activeGenerations = useAppStore((s) => s.activeGenerations);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [newProject, setNewProject] = useState({
    name: '',
    north: 40.758,
    south: 40.7,
    east: -73.97,
    west: -74.02,
  });

  const handleCreate = async () => {
    if (!newProject.name.trim()) return;
    setCreateError(null);

    try {
      await createProject.mutateAsync({
        name: newProject.name,
        region: {
          north: newProject.north,
          south: newProject.south,
          east: newProject.east,
          west: newProject.west,
        },
      });
      setShowCreateModal(false);
      setNewProject({ name: '', north: 40.758, south: 40.7, east: -73.97, west: -74.02 });
    } catch (e: any) {
      const message = e?.data?.detail || e?.message || 'Failed to create project';
      setCreateError(message);
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete project "${name}"? This cannot be undone.`)) return;
    await deleteProject.mutateAsync({ name });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="bg-red-50 text-red-600 p-4 rounded-lg">
          Failed to load projects: {error.message}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">MapGen</h1>
          <p className="text-slate-600 mt-1">Illustrated Map Generator</p>
        </div>
        <button
          onClick={() => { setCreateError(null); setShowCreateModal(true); }}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          New Project
        </button>
      </div>

      {projects?.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm p-12 text-center">
          <div className="text-slate-400 text-6xl mb-4">🗺️</div>
          <h2 className="text-xl font-semibold text-slate-700 mb-2">No projects yet</h2>
          <p className="text-slate-500 mb-6">Create your first illustrated map project to get started.</p>
          <button
            onClick={() => { setCreateError(null); setShowCreateModal(true); }}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Create Project
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects?.map((project) => (
            <div key={project.name} className="bg-white rounded-lg shadow-sm overflow-hidden">
              <Link to={`/project/${encodeURIComponent(project.name)}`} className="block p-6 hover:bg-slate-50">
                <h2 className="text-lg font-semibold text-slate-800 mb-2">{project.name}</h2>
                <div className="text-sm text-slate-500 space-y-1">
                  <div>Area: {project.area_km2.toFixed(1)} km²</div>
                  <div>Tiles: {project.tile_count}</div>
                  <div>Landmarks: {project.landmark_count}</div>
                </div>
                <div className="mt-4 flex items-center gap-2">
                  {activeGenerations[project.name] ? (
                    <>
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700">
                        <span className="inline-block w-1.5 h-1.5 bg-blue-600 rounded-full animate-pulse mr-1" />
                        Generating
                      </span>
                      {activeGenerations[project.name].progress && (
                        <span className="text-xs text-slate-500">
                          {activeGenerations[project.name].progress!.completed_tiles}/
                          {activeGenerations[project.name].progress!.total_tiles} tiles
                        </span>
                      )}
                    </>
                  ) : project.has_generated_tiles ? (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700">
                      Generated
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600">
                      Not generated
                    </span>
                  )}
                </div>
              </Link>
              <div className="px-6 py-3 bg-slate-50 border-t border-slate-100 flex justify-end">
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    handleDelete(project.name);
                  }}
                  className="text-sm text-red-600 hover:text-red-700"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Project Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
            <h2 className="text-xl font-semibold mb-4">Create New Project</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Project Name</label>
                <input
                  type="text"
                  value={newProject.name}
                  onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="My Map Project"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">North</label>
                  <input
                    type="number"
                    step="0.001"
                    value={newProject.north}
                    onChange={(e) => setNewProject({ ...newProject, north: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">South</label>
                  <input
                    type="number"
                    step="0.001"
                    value={newProject.south}
                    onChange={(e) => setNewProject({ ...newProject, south: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">East</label>
                  <input
                    type="number"
                    step="0.001"
                    value={newProject.east}
                    onChange={(e) => setNewProject({ ...newProject, east: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">West</label>
                  <input
                    type="number"
                    step="0.001"
                    value={newProject.west}
                    onChange={(e) => setNewProject({ ...newProject, west: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
            </div>

            {createError && (
              <div className="mt-4 p-3 bg-red-50 text-red-700 text-sm rounded-lg">
                {createError}
              </div>
            )}

            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 text-slate-600 hover:text-slate-800"
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={createProject.isPending || !newProject.name.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {createProject.isPending ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProjectList;
