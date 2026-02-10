import { create } from 'zustand';
import type { TileSpec, SeamInfo, LandmarkDetail, GenerationProgress } from '@/types';

export interface ActiveGeneration {
  taskId: string;
  projectName: string;
  progress: GenerationProgress | null;
}

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

  // Active generations (keyed by project name)
  activeGenerations: Record<string, ActiveGeneration>;
  setActiveGeneration: (projectName: string, gen: ActiveGeneration | null) => void;
  updateGenerationProgress: (projectName: string, progress: GenerationProgress) => void;

  // Map view mode
  mapViewMode: 'geographic' | 'tiles' | 'tile-detail';
  setMapViewMode: (mode: 'geographic' | 'tiles' | 'tile-detail') => void;
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

  activeGenerations: {},
  setActiveGeneration: (projectName, gen) =>
    set((state) => {
      const next = { ...state.activeGenerations };
      if (gen) {
        next[projectName] = gen;
      } else {
        delete next[projectName];
      }
      return { activeGenerations: next };
    }),
  updateGenerationProgress: (projectName, progress) =>
    set((state) => {
      const existing = state.activeGenerations[projectName];
      if (!existing) return state;
      return {
        activeGenerations: {
          ...state.activeGenerations,
          [projectName]: { ...existing, progress },
        },
      };
    }),

  mapViewMode: 'geographic',
  setMapViewMode: (mode) => set({ mapViewMode: mode }),
}));
