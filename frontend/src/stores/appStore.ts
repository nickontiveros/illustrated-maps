import { create } from 'zustand';
import type { TileSpec, SeamInfo, LandmarkDetail } from '@/types';

interface AppState {
  // Selected project
  currentProject: string | null;
  setCurrentProject: (name: string | null) => void;

  // Selected tile
  selectedTile: TileSpec | null;
  setSelectedTile: (tile: TileSpec | null) => void;

  // Selected seam
  selectedSeam: SeamInfo | null;
  setSelectedSeam: (seam: SeamInfo | null) => void;

  // Selected landmark
  selectedLandmark: LandmarkDetail | null;
  setSelectedLandmark: (landmark: LandmarkDetail | null) => void;

  // UI state
  sidebarTab: 'tiles' | 'seams' | 'landmarks' | 'settings';
  setSidebarTab: (tab: 'tiles' | 'seams' | 'landmarks' | 'settings') => void;

  // Generation task
  activeTaskId: string | null;
  setActiveTaskId: (taskId: string | null) => void;

  // Map view mode
  mapViewMode: 'geographic' | 'tiles';
  setMapViewMode: (mode: 'geographic' | 'tiles') => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentProject: null,
  setCurrentProject: (name) => set({ currentProject: name }),

  selectedTile: null,
  setSelectedTile: (tile) => set({ selectedTile: tile }),

  selectedSeam: null,
  setSelectedSeam: (seam) => set({ selectedSeam: seam }),

  selectedLandmark: null,
  setSelectedLandmark: (landmark) => set({ selectedLandmark: landmark }),

  sidebarTab: 'tiles',
  setSidebarTab: (tab) => set({ sidebarTab: tab }),

  activeTaskId: null,
  setActiveTaskId: (taskId) => set({ activeTaskId: taskId }),

  mapViewMode: 'geographic',
  setMapViewMode: (mode) => set({ mapViewMode: mode }),
}));
