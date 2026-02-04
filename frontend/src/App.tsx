import { Routes, Route } from 'react-router-dom';
import ProjectList from './components/ProjectList';
import ProjectView from './components/ProjectView';

function App() {
  return (
    <div className="min-h-screen bg-slate-50">
      <Routes>
        <Route path="/" element={<ProjectList />} />
        <Route path="/project/:name" element={<ProjectView />} />
      </Routes>
    </div>
  );
}

export default App;
