import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import LandingPage from './pages/LandingPage';
import ImagePage from './pages/ImagePage';
import VideoPage from './pages/VideoPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-white text-black">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/image" element={<ImagePage />} />
            <Route path="/video" element={<VideoPage />} />
          </Routes>
        </main>
        <footer className="border-t border-gray-200 py-6 text-center text-sm text-gray-500">
          GastroLens-AI &copy; 2026 &middot; ResNet50 Transfer Learning
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;
