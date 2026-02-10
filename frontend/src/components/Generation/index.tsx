import { useState, useEffect } from 'react';
import {
  useGenerationStatus,
  useStartGeneration,
  useCancelGeneration,
} from '@/hooks/useTiles';
import { useAppStore } from '@/stores/appStore';
import type { GenerationProgress } from '@/types';

interface GenerationProps {
  projectName: string;
}

function Generation({ projectName }: GenerationProps) {
  const { data: statusData } = useGenerationStatus(projectName);
  const startGeneration = useStartGeneration(projectName);
  const cancelGeneration = useCancelGeneration(projectName);
  const { activeGenerations, setActiveGeneration } = useAppStore();

  const [showModal, setShowModal] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const activeGen = activeGenerations[projectName];
  const currentProgress: GenerationProgress | null =
    activeGen?.progress || statusData || null;

  const isRunning = currentProgress?.status === 'running';
  const isFailed = currentProgress?.status === 'failed';
  const progressPercent = currentProgress
    ? Math.round((currentProgress.completed_tiles / currentProgress.total_tiles) * 100)
    : 0;

  const handleStart = async () => {
    setStartError(null);
    try {
      const result = await startGeneration.mutateAsync({ skip_existing: true });
      setActiveGeneration(projectName, {
        taskId: result.task_id,
        projectName,
        progress: null,
      });
    } catch (error: any) {
      const message = error?.data?.detail || error?.message || 'Failed to start generation';
      setStartError(message);
    }
  };

  const handleCancel = async () => {
    try {
      await cancelGeneration.mutateAsync();
      setActiveGeneration(projectName, null);
    } catch (error) {
      console.error('Failed to cancel generation:', error);
    }
  };

  // ESC key to close modal
  useEffect(() => {
    if (!showModal) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowModal(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [showModal]);

  return (
    <>
      <div className="flex items-center gap-3">
        {isRunning ? (
          <>
            <div className="flex items-center gap-2">
              <div className="w-32 h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-600 transition-all duration-300"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <span className="text-sm text-slate-600">
                {currentProgress?.completed_tiles}/{currentProgress?.total_tiles}
              </span>
            </div>
            <button
              onClick={() => setShowModal(true)}
              className="text-sm text-blue-600 hover:text-blue-700"
            >
              Details
            </button>
            <button
              onClick={handleCancel}
              disabled={cancelGeneration.isPending}
              className="px-3 py-1.5 text-sm bg-red-100 text-red-600 rounded hover:bg-red-200 disabled:opacity-50"
            >
              Cancel
            </button>
          </>
        ) : (
          <div className="flex items-center gap-3">
            {isFailed && currentProgress?.error && (
              <span className="text-sm text-red-600 max-w-xs truncate" title={currentProgress.error}>
                Failed: {currentProgress.error}
              </span>
            )}
            <button
              onClick={handleStart}
              disabled={startGeneration.isPending}
              className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {startGeneration.isPending
                ? 'Starting...'
                : isFailed
                ? 'Retry'
                : 'Generate'}
            </button>
          </div>
        )}
      </div>

      {startError && (
        <div className="fixed top-4 right-4 bg-red-50 text-red-700 text-sm p-4 rounded-lg shadow-xl border border-red-200 max-w-md" style={{ zIndex: 10000 }}>
          <div className="flex items-start gap-2">
            <div className="flex-1">{startError}</div>
            <button
              onClick={() => setStartError(null)}
              className="text-red-400 hover:text-red-600 font-bold flex-shrink-0"
            >
              x
            </button>
          </div>
        </div>
      )}

      {/* Progress Modal */}
      {showModal && (
        <GenerationModal
          progress={currentProgress}
          onClose={() => setShowModal(false)}
          onCancel={handleCancel}
          isCancelling={cancelGeneration.isPending}
        />
      )}
    </>
  );
}

interface GenerationModalProps {
  progress: GenerationProgress | null;
  onClose: () => void;
  onCancel: () => void;
  isCancelling: boolean;
}

function GenerationModal({ progress, onClose, onCancel, isCancelling }: GenerationModalProps) {
  if (!progress) {
    return null;
  }

  const progressPercent = Math.round((progress.completed_tiles / progress.total_tiles) * 100);
  const isRunning = progress.status === 'running';

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={(e) => {
        // Close on overlay click (not modal content click)
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold">Generation Progress</h2>
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                progress.status === 'running'
                  ? 'bg-blue-100 text-blue-700'
                  : progress.status === 'completed'
                  ? 'bg-green-100 text-green-700'
                  : progress.status === 'failed'
                  ? 'bg-red-100 text-red-700'
                  : 'bg-slate-100 text-slate-600'
              }`}
            >
              {progress.status}
            </span>
          </div>
        </div>

        {/* Progress bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-600">
              {progress.completed_tiles} of {progress.total_tiles} tiles
            </span>
            <span className="font-medium">{progressPercent}%</span>
          </div>
          <div className="h-4 bg-slate-200 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                progress.status === 'completed'
                  ? 'bg-green-500'
                  : progress.status === 'failed'
                  ? 'bg-red-500'
                  : 'bg-blue-600'
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-slate-50 rounded-lg p-3">
            <div className="text-sm text-slate-500">Elapsed</div>
            <div className="font-semibold">{formatTime(progress.elapsed_seconds)}</div>
          </div>
          <div className="bg-slate-50 rounded-lg p-3">
            <div className="text-sm text-slate-500">Remaining</div>
            <div className="font-semibold">
              {progress.estimated_remaining_seconds
                ? formatTime(progress.estimated_remaining_seconds)
                : '—'}
            </div>
          </div>
          <div className="bg-green-50 rounded-lg p-3">
            <div className="text-sm text-green-600">Completed</div>
            <div className="font-semibold text-green-700">{progress.completed_tiles}</div>
          </div>
          <div className="bg-red-50 rounded-lg p-3">
            <div className="text-sm text-red-600">Failed</div>
            <div className="font-semibold text-red-700">{progress.failed_tiles}</div>
          </div>
        </div>

        {/* Current tile */}
        {progress.current_tile && isRunning && (
          <div className="bg-blue-50 rounded-lg p-3 mb-6">
            <div className="text-sm text-blue-600 mb-1">Currently generating</div>
            <div className="font-medium text-blue-800">
              Tile ({progress.current_tile[0]}, {progress.current_tile[1]})
            </div>
          </div>
        )}

        {/* Error */}
        {progress.error && (
          <div className="bg-red-50 rounded-lg p-3 mb-6">
            <div className="text-sm text-red-600 mb-1">Error</div>
            <div className="text-red-800 text-sm">{progress.error}</div>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3">
          {isRunning && (
            <button
              onClick={onCancel}
              disabled={isCancelling}
              className="px-4 py-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 disabled:opacity-50"
            >
              {isCancelling ? 'Cancelling...' : 'Cancel'}
            </button>
          )}
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default Generation;
