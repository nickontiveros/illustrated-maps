import { useState, useCallback } from 'react';

const GOOGLE_KEY_STORAGE = 'mapgen_google_api_key';
const MAPBOX_TOKEN_STORAGE = 'mapgen_mapbox_token';

export function useAPIKeys() {
  const [googleKey, setGoogleKeyState] = useState(
    () => localStorage.getItem(GOOGLE_KEY_STORAGE) || ''
  );
  const [mapboxToken, setMapboxTokenState] = useState(
    () => localStorage.getItem(MAPBOX_TOKEN_STORAGE) || ''
  );

  const setGoogleKey = useCallback((key: string) => {
    if (key) {
      localStorage.setItem(GOOGLE_KEY_STORAGE, key);
    } else {
      localStorage.removeItem(GOOGLE_KEY_STORAGE);
    }
    setGoogleKeyState(key);
  }, []);

  const setMapboxToken = useCallback((token: string) => {
    if (token) {
      localStorage.setItem(MAPBOX_TOKEN_STORAGE, token);
    } else {
      localStorage.removeItem(MAPBOX_TOKEN_STORAGE);
    }
    setMapboxTokenState(token);
  }, []);

  const clearAll = useCallback(() => {
    localStorage.removeItem(GOOGLE_KEY_STORAGE);
    localStorage.removeItem(MAPBOX_TOKEN_STORAGE);
    setGoogleKeyState('');
    setMapboxTokenState('');
  }, []);

  return {
    googleKey,
    mapboxToken,
    setGoogleKey,
    setMapboxToken,
    clearAll,
    hasGoogleKey: !!googleKey,
    hasMapboxToken: !!mapboxToken,
  };
}

/** Get auth headers from localStorage (for use outside React components). */
export function getAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  const googleKey = localStorage.getItem(GOOGLE_KEY_STORAGE);
  const mapboxToken = localStorage.getItem(MAPBOX_TOKEN_STORAGE);
  if (googleKey) headers['X-Google-API-Key'] = googleKey;
  if (mapboxToken) headers['X-Mapbox-Access-Token'] = mapboxToken;
  return headers;
}
