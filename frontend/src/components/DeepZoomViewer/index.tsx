import { useEffect, useRef, useState } from 'react';
import OpenSeadragon from 'openseadragon';

interface DeepZoomViewerProps {
  projectName: string;
  className?: string;
}

interface DZIInfo {
  is_generated: boolean;
  width: number;
  height: number;
  tile_size: number;
  overlap: number;
  format: string;
  max_level: number;
  num_levels: number;
}

function DeepZoomViewer({ projectName, className = '' }: DeepZoomViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const [dziInfo, setDziInfo] = useState<DZIInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAssembling, setIsAssembling] = useState(false);
  const [needsAssembly, setNeedsAssembly] = useState(false);

  // Fetch DZI info
  const fetchInfo = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setNeedsAssembly(false);

      const response = await fetch(`/api/projects/${encodeURIComponent(projectName)}/dzi/info`);

      if (!response.ok) {
        if (response.status === 404) {
          setNeedsAssembly(true);
          setError(null);
        } else {
          throw new Error(`Failed to fetch DZI info: ${response.statusText}`);
        }
        return;
      }

      const info = await response.json();
      setDziInfo(info);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load image info');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchInfo();
  }, [projectName]);

  const handleAssemble = async () => {
    try {
      setIsAssembling(true);
      setError(null);

      const response = await fetch(
        `/api/projects/${encodeURIComponent(projectName)}/assemble`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.detail || `Assembly failed: ${response.statusText}`);
      }

      // Refresh info after assembly
      await fetchInfo();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to assemble tiles');
    } finally {
      setIsAssembling(false);
    }
  };

  // Initialize OpenSeadragon viewer
  useEffect(() => {
    if (!containerRef.current || !dziInfo?.is_generated) return;

    // Destroy existing viewer
    if (viewerRef.current) {
      viewerRef.current.destroy();
    }

    // Create new viewer
    viewerRef.current = OpenSeadragon({
      element: containerRef.current,
      tileSources: `/api/projects/${encodeURIComponent(projectName)}/dzi/assembled.dzi`,
      prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      navigatorSizeRatio: 0.15,
      showRotationControl: true,
      showFullPageControl: true,
      showHomeControl: true,
      showZoomControl: true,
      animationTime: 0.5,
      blendTime: 0.3,
      constrainDuringPan: true,
      maxZoomPixelRatio: 2,
      minZoomImageRatio: 0.8,
      visibilityRatio: 0.5,
      springStiffness: 10,
    });

    return () => {
      if (viewerRef.current) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, [projectName, dziInfo?.is_generated]);

  const handleGenerateDZI = async () => {
    try {
      setIsGenerating(true);
      setError(null);

      const response = await fetch(
        `/api/projects/${encodeURIComponent(projectName)}/dzi/generate`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error(`Failed to generate DZI: ${response.statusText}`);
      }

      // Refresh info
      const infoResponse = await fetch(
        `/api/projects/${encodeURIComponent(projectName)}/dzi/info`
      );
      if (infoResponse.ok) {
        const info = await infoResponse.json();
        setDziInfo(info);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate tiles');
    } finally {
      setIsGenerating(false);
    }
  };

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center bg-slate-100 ${className}`}>
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (needsAssembly) {
    return (
      <div className={`flex flex-col items-center justify-center bg-slate-100 ${className}`}>
        <div className="text-center">
          <div className="text-slate-500 mb-4">
            No assembled image yet. Assemble generated tiles to view the full map.
          </div>
          <button
            onClick={handleAssemble}
            disabled={isAssembling}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {isAssembling ? 'Assembling...' : 'Assemble Tiles'}
          </button>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`flex flex-col items-center justify-center bg-slate-100 ${className}`}>
        <div className="text-slate-500 mb-4">{error}</div>
      </div>
    );
  }

  if (!dziInfo?.is_generated) {
    return (
      <div className={`flex flex-col items-center justify-center bg-slate-100 ${className}`}>
        <div className="text-center">
          <div className="text-slate-500 mb-4">
            Deep Zoom tiles not generated yet.
          </div>
          {dziInfo && (
            <div className="text-sm text-slate-400 mb-4">
              Image size: {dziInfo.width} × {dziInfo.height}px
            </div>
          )}
          <button
            onClick={handleGenerateDZI}
            disabled={isGenerating}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {isGenerating ? 'Generating...' : 'Generate Deep Zoom Tiles'}
          </button>
        </div>
      </div>
    );
  }

  const handleReassemble = async () => {
    try {
      setIsAssembling(true);
      setError(null);

      // Re-assemble tiles
      const response = await fetch(
        `/api/projects/${encodeURIComponent(projectName)}/assemble`,
        { method: 'POST' }
      );
      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.detail || `Assembly failed: ${response.statusText}`);
      }

      // Regenerate DZI from the new assembled image
      const dziResponse = await fetch(
        `/api/projects/${encodeURIComponent(projectName)}/dzi/generate?force=true`,
        { method: 'POST' }
      );
      if (!dziResponse.ok) {
        throw new Error(`Failed to regenerate DZI: ${dziResponse.statusText}`);
      }

      // Refresh the viewer
      await fetchInfo();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to re-assemble');
    } finally {
      setIsAssembling(false);
    }
  };

  return (
    <div className={`relative ${className}`}>
      <div ref={containerRef} className="w-full h-full" />

      {/* Info overlay */}
      <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm rounded-lg px-3 py-2 text-sm shadow">
        <div className="font-medium">Assembled Map</div>
        <div className="text-slate-500">
          {dziInfo.width} × {dziInfo.height}px
        </div>
        <button
          onClick={handleReassemble}
          disabled={isAssembling}
          className="mt-2 px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {isAssembling ? 'Re-assembling...' : 'Re-assemble'}
        </button>
      </div>
    </div>
  );
}

export default DeepZoomViewer;
