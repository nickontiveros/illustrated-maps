import { Routes, Route } from 'react-router-dom';
import ProjectList from './components/ProjectList';
import ProjectView from './components/ProjectView';
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
          <Route path="/" element={<ProjectList />} />
          <Route path="/project/:name" element={<ProjectView />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;
