import { useState, useEffect } from 'react';
import { useAPIKeys } from '@/hooks/useAPIKeys';
import { api } from '@/api/client';

interface APIKeySettingsProps {
  open: boolean;
  onClose: () => void;
}

export default function APIKeySettings({ open, onClose }: APIKeySettingsProps) {
  const { googleKey, mapboxToken, setGoogleKey, setMapboxToken } = useAPIKeys();

  const [localGoogle, setLocalGoogle] = useState(googleKey);
  const [localMapbox, setLocalMapbox] = useState(mapboxToken);
  const [showGoogle, setShowGoogle] = useState(false);
  const [showMapbox, setShowMapbox] = useState(false);
  const [googleSource, setGoogleSource] = useState<string>('unknown');
  const [mapboxSource, setMapboxSource] = useState<string>('unknown');

  useEffect(() => {
    if (open) {
      setLocalGoogle(googleKey);
      setLocalMapbox(mapboxToken);
      // Fetch current config to check sources
      api.getConfig().then((config) => {
        setGoogleSource(config.google_api_key_source);
        setMapboxSource(config.mapbox_token_source);
      }).catch(() => {});
    }
  }, [open, googleKey, mapboxToken]);

  if (!open) return null;

  const handleSave = () => {
    setGoogleKey(localGoogle);
    setMapboxToken(localMapbox);
    // Refresh sources
    setTimeout(() => {
      api.getConfig().then((config) => {
        setGoogleSource(config.google_api_key_source);
        setMapboxSource(config.mapbox_token_source);
      }).catch(() => {});
    }, 100);
  };

  const handleClear = () => {
    setLocalGoogle('');
    setLocalMapbox('');
    setGoogleKey('');
    setMapboxToken('');
  };

  const sourceLabel = (source: string) => {
    switch (source) {
      case 'client': return <span className="text-green-600 text-xs font-medium">Configured (browser)</span>;
      case 'server': return <span className="text-blue-600 text-xs font-medium">Configured (server)</span>;
      default: return <span className="text-red-500 text-xs font-medium">Missing</span>;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4" onClick={(e) => e.stopPropagation()}>
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-800">API Key Settings</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-xl leading-none">&times;</button>
        </div>

        <div className="p-6 space-y-5">
          <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-sm text-amber-800">
            Keys are stored in your browser only and sent as headers with each request. They are never saved on the server.
          </div>

          {/* Google API Key */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-700">Google API Key (Gemini)</label>
              {sourceLabel(googleSource)}
            </div>
            <div className="relative">
              <input
                type={showGoogle ? 'text' : 'password'}
                value={localGoogle}
                onChange={(e) => setLocalGoogle(e.target.value)}
                placeholder="AIza..."
                className="w-full px-3 py-2 pr-16 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <button
                onClick={() => setShowGoogle(!showGoogle)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-slate-500 hover:text-slate-700 px-2 py-1"
              >
                {showGoogle ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>

          {/* Mapbox Access Token */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-700">Mapbox Access Token</label>
              {sourceLabel(mapboxSource)}
            </div>
            <div className="relative">
              <input
                type={showMapbox ? 'text' : 'password'}
                value={localMapbox}
                onChange={(e) => setLocalMapbox(e.target.value)}
                placeholder="pk.ey..."
                className="w-full px-3 py-2 pr-16 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <button
                onClick={() => setShowMapbox(!showMapbox)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-slate-500 hover:text-slate-700 px-2 py-1"
              >
                {showMapbox ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>
        </div>

        <div className="px-6 py-4 border-t border-slate-200 flex items-center justify-between">
          <button
            onClick={handleClear}
            className="text-sm text-red-600 hover:text-red-700"
          >
            Clear All
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
