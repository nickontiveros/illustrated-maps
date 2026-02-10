import { Link } from 'react-router-dom';
import { useAppStore } from '@/stores/appStore';

function GlobalProgressBar() {
  const activeGenerations = useAppStore((s) => s.activeGenerations);
  const entries = Object.values(activeGenerations);

  if (entries.length === 0) return null;

  return (
    <div className="bg-blue-600 text-white px-4 py-2 flex items-center gap-4 text-sm">
      <span className="font-medium flex items-center gap-2">
        <span className="inline-block w-2 h-2 bg-white rounded-full animate-pulse" />
        Generating
      </span>
      <div className="flex-1 flex items-center gap-4 overflow-x-auto">
        {entries.map((gen) => {
          const progress = gen.progress;
          const percent = progress
            ? Math.round((progress.completed_tiles / progress.total_tiles) * 100)
            : 0;
          const label = progress
            ? `${progress.completed_tiles}/${progress.total_tiles} tiles`
            : 'Starting...';

          return (
            <Link
              key={gen.projectName}
              to={`/project/${encodeURIComponent(gen.projectName)}`}
              className="flex items-center gap-2 hover:bg-blue-500 rounded px-2 py-1 transition-colors min-w-0"
            >
              <span className="truncate font-medium">{gen.projectName}</span>
              <div className="w-24 h-1.5 bg-blue-400 rounded-full overflow-hidden flex-shrink-0">
                <div
                  className="h-full bg-white transition-all duration-300"
                  style={{ width: `${percent}%` }}
                />
              </div>
              <span className="text-blue-100 whitespace-nowrap">{label}</span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

export default GlobalProgressBar;
