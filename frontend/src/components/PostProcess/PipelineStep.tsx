import { api } from '@/api/client';

interface PipelineStepProps {
  label: string;
  stage: string;
  projectName: string;
  done: boolean;
  running: boolean;
  available: boolean;
  onRun: () => void;
  error?: string | null;
  children?: React.ReactNode;
}

export default function PipelineStep({
  label,
  stage,
  projectName,
  done,
  running,
  available,
  onRun,
  error,
  children,
}: PipelineStepProps) {
  const thumbnailUrl = done ? api.getPostProcessImageUrl(projectName, stage, 256) : null;

  return (
    <div className={`border rounded-lg overflow-hidden ${done ? 'border-green-200 bg-green-50/30' : 'border-slate-200'}`}>
      <div className="px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {/* Status indicator */}
          {done ? (
            <span className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center text-white text-xs">✓</span>
          ) : running ? (
            <span className="w-5 h-5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
          ) : (
            <span className="w-5 h-5 rounded-full border-2 border-slate-300" />
          )}
          <span className="text-sm font-medium text-slate-700">{label}</span>
        </div>

        <button
          onClick={onRun}
          disabled={!available || running}
          className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${
            !available || running
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {running ? 'Running...' : done ? 'Re-run' : 'Run'}
        </button>
      </div>

      {children && (
        <div className="px-4 pb-2">
          {children}
        </div>
      )}

      {error && (
        <div className="px-4 pb-3">
          <div className="text-xs text-red-600 bg-red-50 rounded px-2 py-1">{error}</div>
        </div>
      )}

      {thumbnailUrl && (
        <div className="px-4 pb-3">
          <img
            src={thumbnailUrl}
            alt={`${label} result`}
            className="w-full rounded border border-slate-200 cursor-pointer hover:opacity-90"
            onClick={() => {
              const fullUrl = api.getPostProcessImageUrl(projectName, stage);
              window.open(fullUrl, '_blank');
            }}
          />
        </div>
      )}
    </div>
  );
}
