import { useState } from 'react';
import {
  useLandmarks,
  useCreateLandmark,
  useUpdateLandmark,
  useDeleteLandmark,
  useIllustrateLandmark,
  useIllustrateAllLandmarks,
} from '@/hooks/useLandmarks';
import { useAppStore } from '@/stores/appStore';
import { api } from '@/api/client';
import type { LandmarkDetail } from '@/types';

interface LandmarksProps {
  projectName: string;
}

function Landmarks({ projectName }: LandmarksProps) {
  const { data: landmarks, isLoading } = useLandmarks(projectName);
  const { selectedLandmark, setSelectedLandmark } = useAppStore();
  const illustrateAll = useIllustrateAllLandmarks(projectName);
  const [showAddModal, setShowAddModal] = useState(false);

  if (isLoading) {
    return (
      <div className="p-4 flex justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!landmarks) {
    return <div className="p-4 text-slate-500">Failed to load landmarks</div>;
  }

  const needsIllustration = landmarks.filter((l) => l.has_photo && !l.has_illustration);

  return (
    <div className="flex flex-col h-full">
      {/* Stats & Actions */}
      <div className="p-4 border-b border-slate-200 space-y-3">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-slate-50 rounded p-2">
            <div className="text-slate-500">Total</div>
            <div className="font-semibold">{landmarks.length}</div>
          </div>
          <div className="bg-green-50 rounded p-2">
            <div className="text-green-600">Illustrated</div>
            <div className="font-semibold text-green-700">
              {landmarks.filter((l) => l.has_illustration).length}
            </div>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setShowAddModal(true)}
            className="flex-1 px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700"
          >
            Add Landmark
          </button>
          {needsIllustration.length > 0 && (
            <button
              onClick={() => illustrateAll.mutate()}
              disabled={illustrateAll.isPending}
              className="flex-1 px-3 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700 disabled:opacity-50"
            >
              {illustrateAll.isPending ? 'Working...' : `Illustrate (${needsIllustration.length})`}
            </button>
          )}
        </div>
      </div>

      {/* Landmark list */}
      <div className="flex-1 overflow-auto">
        {landmarks.length === 0 ? (
          <div className="p-8 text-center text-slate-500">
            <div className="text-4xl mb-2">📍</div>
            <div>No landmarks yet</div>
            <div className="text-sm mt-1">Click "Add Landmark" to get started</div>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {landmarks.map((landmark) => {
              const isSelected = selectedLandmark?.name === landmark.name;

              return (
                <button
                  key={landmark.name}
                  onClick={() => setSelectedLandmark(isSelected ? null : landmark)}
                  className={`w-full p-3 text-left hover:bg-slate-50 transition-colors ${
                    isSelected ? 'bg-blue-50' : ''
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {landmark.has_photo && (
                      <img
                        src={api.getLandmarkPhotoUrl(projectName, landmark.name)}
                        alt={landmark.name}
                        className="w-10 h-10 rounded object-cover"
                      />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-slate-800 truncate">
                        {landmark.name}
                      </div>
                      <div className="text-xs text-slate-500">
                        Scale: {landmark.scale}x • Z: {landmark.z_index}
                      </div>
                    </div>
                    <div className="flex flex-col gap-1">
                      {landmark.has_photo && (
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs bg-blue-100 text-blue-700">
                          Photo
                        </span>
                      )}
                      {landmark.has_illustration && (
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs bg-green-100 text-green-700">
                          Illustrated
                        </span>
                      )}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Selected landmark details */}
      {selectedLandmark && (
        <LandmarkDetails landmark={selectedLandmark} projectName={projectName} />
      )}

      {/* Add landmark modal */}
      {showAddModal && (
        <AddLandmarkModal
          projectName={projectName}
          onClose={() => setShowAddModal(false)}
        />
      )}
    </div>
  );
}

interface LandmarkDetailsProps {
  landmark: LandmarkDetail;
  projectName: string;
}

function LandmarkDetails({ landmark, projectName }: LandmarkDetailsProps) {
  const updateLandmark = useUpdateLandmark(projectName);
  const deleteLandmark = useDeleteLandmark(projectName);
  const illustrateLandmark = useIllustrateLandmark(projectName);
  const { setSelectedLandmark } = useAppStore();

  const [scale, setScale] = useState(landmark.scale);
  const [zIndex, setZIndex] = useState(landmark.z_index);

  const handleSave = async () => {
    await updateLandmark.mutateAsync({
      landmarkName: landmark.name,
      data: { scale, z_index: zIndex },
    });
  };

  const handleDelete = async () => {
    if (!confirm(`Delete landmark "${landmark.name}"?`)) return;
    await deleteLandmark.mutateAsync(landmark.name);
    setSelectedLandmark(null);
  };

  const handleIllustrate = async () => {
    await illustrateLandmark.mutateAsync(landmark.name);
  };

  return (
    <div className="border-t border-slate-200 p-4 space-y-4 max-h-96 overflow-auto">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-slate-800">{landmark.name}</h3>
        <button
          onClick={handleDelete}
          className="text-sm text-red-600 hover:text-red-700"
        >
          Delete
        </button>
      </div>

      <div className="text-sm text-slate-500">
        {landmark.latitude.toFixed(6)}°, {landmark.longitude.toFixed(6)}°
      </div>

      {/* Scale slider */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-500">Scale</span>
          <span className="font-medium">{scale.toFixed(1)}x</span>
        </div>
        <input
          type="range"
          min="0.5"
          max="5"
          step="0.1"
          value={scale}
          onChange={(e) => setScale(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Z-index */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-500">Z-Index</span>
          <span className="font-medium">{zIndex}</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          value={zIndex}
          onChange={(e) => setZIndex(parseInt(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Save button */}
      {(scale !== landmark.scale || zIndex !== landmark.z_index) && (
        <button
          onClick={handleSave}
          disabled={updateLandmark.isPending}
          className="w-full px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700 disabled:opacity-50"
        >
          {updateLandmark.isPending ? 'Saving...' : 'Save Changes'}
        </button>
      )}

      {/* Photo preview */}
      {landmark.has_photo && (
        <div>
          <div className="text-sm text-slate-500 mb-2">Photo</div>
          <img
            src={api.getLandmarkPhotoUrl(projectName, landmark.name)}
            alt={landmark.name}
            className="w-full rounded-lg border border-slate-200"
          />
        </div>
      )}

      {/* Illustration preview */}
      {landmark.has_illustration && (
        <div>
          <div className="text-sm text-slate-500 mb-2">Illustration</div>
          <img
            src={api.getLandmarkIllustrationUrl(projectName, landmark.name)}
            alt={`${landmark.name} illustrated`}
            className="w-full rounded-lg border border-slate-200"
          />
        </div>
      )}

      {/* Illustrate button */}
      {landmark.has_photo && !landmark.has_illustration && (
        <button
          onClick={handleIllustrate}
          disabled={illustrateLandmark.isPending}
          className="w-full px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {illustrateLandmark.isPending ? 'Illustrating...' : 'Generate Illustration'}
        </button>
      )}
    </div>
  );
}

interface AddLandmarkModalProps {
  projectName: string;
  onClose: () => void;
}

function AddLandmarkModal({ projectName, onClose }: AddLandmarkModalProps) {
  const createLandmark = useCreateLandmark(projectName);
  const [formData, setFormData] = useState({
    name: '',
    latitude: 0,
    longitude: 0,
    scale: 1.5,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name.trim()) return;

    try {
      await createLandmark.mutateAsync(formData);
      onClose();
    } catch (error) {
      console.error('Failed to create landmark:', error);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
        <h2 className="text-xl font-semibold mb-4">Add Landmark</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Empire State Building"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Latitude</label>
              <input
                type="number"
                step="0.000001"
                value={formData.latitude}
                onChange={(e) => setFormData({ ...formData, latitude: parseFloat(e.target.value) })}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Longitude</label>
              <input
                type="number"
                step="0.000001"
                value={formData.longitude}
                onChange={(e) => setFormData({ ...formData, longitude: parseFloat(e.target.value) })}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Scale</label>
            <input
              type="number"
              step="0.1"
              min="0.5"
              max="5"
              value={formData.scale}
              onChange={(e) => setFormData({ ...formData, scale: parseFloat(e.target.value) })}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-slate-600 hover:text-slate-800"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={createLandmark.isPending || !formData.name.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {createLandmark.isPending ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Landmarks;
