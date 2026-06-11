import { Routes, Route } from 'react-router-dom';
import ProjectList from './components/ProjectList';
import ProjectView from './components/ProjectView';
import ProjectListV2 from './components/V2/ProjectListV2';
import ProjectViewV2 from './components/V2/ProjectViewV2';
import GlobalProgressBar from './components/GlobalProgressBar';
import { useGlobalGeneration } from './hooks/useGlobalGeneration';

function App() {
  // Mount global generation tracking at the app root
  useGlobalGeneration();

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <GlobalProgressBar />
      <div className="flex-1">
        <Routes>
          {/* V2 asset-composition pipeline is the default experience */}
          <Route path="/" element={<ProjectListV2 />} />
          <Route path="/v2" element={<ProjectListV2 />} />
          <Route path="/v2/:id" element={<ProjectViewV2 />} />
          {/* V1 tile-generation pipeline (legacy) */}
          <Route path="/v1" element={<ProjectList />} />
          <Route path="/project/:name" element={<ProjectView />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;
