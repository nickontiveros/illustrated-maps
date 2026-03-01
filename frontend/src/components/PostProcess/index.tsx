import { useState } from 'react';
import {
  usePostProcessStatus,
  useComposeLandmarks,
  useAddLabels,
  useApplyPerspective,
  useAddBorder,
  useStartOutpaint,
  useStartPipeline,
} from '@/hooks/usePostProcess';
import { useAppStore } from '@/stores/appStore';
import PipelineStep from './PipelineStep';

interface PostProcessProps {
  projectName: string;
}

const STEP_LABELS: Record<string, string> = {
  compose: 'Compose Landmarks',
  labels: 'Add Labels',
  perspective: 'Perspective',
  border: 'Add Border',
  outpaint: 'Outpaint Edges',
};

export default function PostProcess({ projectName }: PostProcessProps) {
  const { data: status, isLoading } = usePostProcessStatus(projectName);
  const setMapViewMode = useAppStore((s) => s.setMapViewMode);
  const setFinalizedStage = useAppStore((s) => s.setFinalizedStage);

  const compose = useComposeLandmarks(projectName);
  const labels = useAddLabels(projectName);
  const perspective = useApplyPerspective(projectName);
  const border = useAddBorder(projectName);
  const outpaint = useStartOutpaint(projectName);
  const pipeline = useStartPipeline(projectName);

  const [pipelineSteps, setPipelineSteps] = useState<Set<string>>(
    new Set(['compose', 'labels', 'perspective', 'border'])
  );
  const [includeShields, setIncludeShields] = useState(true);
  const [showPerspectiveParams, setShowPerspectiveParams] = useState(false);
  const [perspectiveParams, setPerspectiveParams] = useState({
    angle: 35,
    convergence: 0.7,
    vertical_scale: 0.4,
    horizon_margin: 0.15,
  });

  if (isLoading) {
    return (
      <div className="p-4 flex justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
      </div>
    );
  }

  if (!status) {
    return (
      <div className="p-4 text-sm text-slate-500">
        Failed to load post-processing status.
      </div>
    );
  }

  const toggleStep = (step: string) => {
    setPipelineSteps((prev) => {
      const next = new Set(prev);
      if (next.has(step)) next.delete(step);
      else next.add(step);
      return next;
    });
  };

  const handleRunAll = () => {
    const orderedSteps = ['compose', 'labels', 'perspective', 'border', 'outpaint'].filter((s) =>
      pipelineSteps.has(s)
    );
    if (orderedSteps.length > 0) {
      pipeline.mutate({ steps: orderedSteps, includeShields });
    }
  };

  const handleViewStage = (stage: string) => {
    setFinalizedStage(stage);
    setMapViewMode('finalized');
  };

  const handleExportPSD = async () => {
    try {
      const response = await import('@/api/client').then((m) =>
        m.api.exportPSD(projectName)
      );
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${projectName}.psd`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch {
      // Error handled silently
    }
  };

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">Post-Processing Pipeline</h3>
        {status.latest_stage && (
          <button
            onClick={() => handleViewStage(status.latest_stage!)}
            className="text-xs text-blue-600 hover:text-blue-700"
          >
            View Latest
          </button>
        )}
      </div>

      {!status.assembled && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-sm text-amber-800">
          No assembled image found. Generate and assemble tiles first.
        </div>
      )}

      {/* Pipeline Steps */}
      <div className="space-y-2">
        <PipelineStep
          label="Assembled"
          stage="assembled"
          projectName={projectName}
          done={status.assembled}
          running={false}
          available={false}
          onRun={() => {}}
        />

        <div className="flex items-center justify-center">
          <span className="text-slate-300 text-xs">&darr;</span>
        </div>

        <PipelineStep
          label="Compose Landmarks"
          stage="composed"
          projectName={projectName}
          done={status.composed}
          running={compose.isPending}
          available={status.assembled}
          onRun={() => compose.mutate()}
          error={compose.error?.message}
        />

        <div className="flex items-center justify-center">
          <span className="text-slate-300 text-xs">&darr;</span>
        </div>

        <PipelineStep
          label="Add Labels"
          stage="labeled"
          projectName={projectName}
          done={status.labeled}
          running={labels.isPending}
          available={status.assembled}
          onRun={() => labels.mutate(includeShields)}
          error={labels.error?.message}
        >
          <label className="flex items-center gap-2 text-xs text-slate-500 mt-1">
            <input
              type="checkbox"
              checked={includeShields}
              onChange={(e) => setIncludeShields(e.target.checked)}
              className="rounded"
            />
            Highway Shields
          </label>
        </PipelineStep>

        <div className="flex items-center justify-center">
          <span className="text-slate-300 text-xs">&darr;</span>
        </div>

        <PipelineStep
          label="Perspective (Bird's Eye)"
          stage="perspective"
          projectName={projectName}
          done={status.perspective}
          running={perspective.isPending}
          available={status.assembled}
          onRun={() => perspective.mutate(perspectiveParams)}
          error={perspective.error?.message}
        >
          <div className="mt-1">
            <button
              onClick={() => setShowPerspectiveParams(!showPerspectiveParams)}
              className="text-xs text-blue-600 hover:text-blue-700"
            >
              {showPerspectiveParams ? 'Hide' : 'Show'} Parameters
            </button>
            {showPerspectiveParams && (
              <div className="mt-2 space-y-2">
                <div>
                  <label className="flex items-center justify-between text-xs text-slate-500">
                    <span>Angle: {perspectiveParams.angle}&deg;</span>
                    <span className="text-slate-400">10-80</span>
                  </label>
                  <input
                    type="range"
                    min={10}
                    max={80}
                    step={1}
                    value={perspectiveParams.angle}
                    onChange={(e) => setPerspectiveParams((p) => ({ ...p, angle: Number(e.target.value) }))}
                    className="w-full h-1.5 accent-blue-600"
                  />
                </div>
                <div>
                  <label className="flex items-center justify-between text-xs text-slate-500">
                    <span>Convergence: {perspectiveParams.convergence.toFixed(2)}</span>
                    <span className="text-slate-400">0.3-1.0</span>
                  </label>
                  <input
                    type="range"
                    min={0.3}
                    max={1.0}
                    step={0.05}
                    value={perspectiveParams.convergence}
                    onChange={(e) => setPerspectiveParams((p) => ({ ...p, convergence: Number(e.target.value) }))}
                    className="w-full h-1.5 accent-blue-600"
                  />
                </div>
                <div>
                  <label className="flex items-center justify-between text-xs text-slate-500">
                    <span>Vertical Scale: {perspectiveParams.vertical_scale.toFixed(2)}</span>
                    <span className="text-slate-400">0.1-1.0</span>
                  </label>
                  <input
                    type="range"
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    value={perspectiveParams.vertical_scale}
                    onChange={(e) => setPerspectiveParams((p) => ({ ...p, vertical_scale: Number(e.target.value) }))}
                    className="w-full h-1.5 accent-blue-600"
                  />
                </div>
                <div>
                  <label className="flex items-center justify-between text-xs text-slate-500">
                    <span>Horizon Margin: {perspectiveParams.horizon_margin.toFixed(2)}</span>
                    <span className="text-slate-400">0.0-0.5</span>
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={0.5}
                    step={0.05}
                    value={perspectiveParams.horizon_margin}
                    onChange={(e) => setPerspectiveParams((p) => ({ ...p, horizon_margin: Number(e.target.value) }))}
                    className="w-full h-1.5 accent-blue-600"
                  />
                </div>
                <button
                  onClick={() => setPerspectiveParams({ angle: 35, convergence: 0.7, vertical_scale: 0.4, horizon_margin: 0.15 })}
                  className="text-xs text-slate-500 hover:text-slate-700"
                >
                  Reset to defaults
                </button>
              </div>
            )}
          </div>
        </PipelineStep>

        <div className="flex items-center justify-center">
          <span className="text-slate-300 text-xs">&darr;</span>
        </div>

        <PipelineStep
          label="Add Border"
          stage="bordered"
          projectName={projectName}
          done={status.bordered}
          running={border.isPending}
          available={status.assembled}
          onRun={() => border.mutate()}
          error={border.error?.message}
        />

        <div className="flex items-center justify-center">
          <span className="text-slate-300 text-xs">&darr;</span>
        </div>

        <PipelineStep
          label="Outpaint Edges"
          stage="outpainted"
          projectName={projectName}
          done={status.outpainted}
          running={outpaint.isPending}
          available={status.assembled}
          onRun={() => outpaint.mutate()}
          error={outpaint.error?.message}
        />
      </div>

      {/* Run All Section */}
      {status.assembled && (
        <div className="border border-slate-200 rounded-lg p-4 space-y-3">
          <h4 className="text-sm font-medium text-slate-700">Run Pipeline</h4>
          <div className="space-y-1.5">
            {['compose', 'labels', 'perspective', 'border', 'outpaint'].map((step) => (
              <label key={step} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={pipelineSteps.has(step)}
                  onChange={() => toggleStep(step)}
                />
                <span>{STEP_LABELS[step]}</span>
              </label>
            ))}
          </div>
          <button
            onClick={handleRunAll}
            disabled={pipeline.isPending || pipelineSteps.size === 0}
            className="w-full px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {pipeline.isPending ? 'Running Pipeline...' : `Run ${pipelineSteps.size} Step${pipelineSteps.size !== 1 ? 's' : ''}`}
          </button>
          {pipeline.error && (
            <div className="text-xs text-red-600">{(pipeline.error as Error).message}</div>
          )}
        </div>
      )}

      {/* Export Section */}
      {status.assembled && (
        <div className="border border-slate-200 rounded-lg p-4 space-y-3">
          <h4 className="text-sm font-medium text-slate-700">Export</h4>
          <div className="flex gap-2">
            <button
              onClick={handleExportPSD}
              className="flex-1 px-3 py-2 text-sm border border-slate-300 rounded-lg hover:bg-slate-50"
            >
              Export PSD
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
