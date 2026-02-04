// Project types
export interface BoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

export interface OutputSettings {
  width: number;
  height: number;
  dpi: number;
}

export interface StyleSettings {
  perspective_angle: number;
  orientation: 'north' | 'south' | 'east' | 'west';
  prompt: string;
  color_palette: string[] | null;
}

export interface TileSettings {
  size: number;
  overlap: number;
}

export interface ProjectSummary {
  name: string;
  region: BoundingBox;
  area_km2: number;
  tile_count: number;
  landmark_count: number;
  has_generated_tiles: boolean;
  last_modified: string | null;
}

export interface ProjectDetail {
  name: string;
  region: BoundingBox;
  output: OutputSettings;
  style: StyleSettings;
  tiles: TileSettings;
  landmarks: Landmark[];
  area_km2: number;
  detail_level: string;
  grid_cols: number;
  grid_rows: number;
  tile_count: number;
  estimated_cost: number | null;
}

// Tile types
export type TileStatus = 'pending' | 'generating' | 'completed' | 'failed';

export interface TileSpec {
  col: number;
  row: number;
  x_offset: number;
  y_offset: number;
  bbox: BoundingBox;
  position_desc: string;
  status: TileStatus;
  has_reference: boolean;
  has_generated: boolean;
  generation_time: number | null;
  error: string | null;
}

export interface TileGrid {
  project_name: string;
  cols: number;
  rows: number;
  tile_size: number;
  overlap: number;
  effective_size: number;
  tiles: TileSpec[];
}

// Seam types
export interface SeamInfo {
  id: string;
  orientation: 'horizontal' | 'vertical';
  tile_a: [number, number];
  tile_b: [number, number];
  x: number;
  y: number;
  width: number;
  height: number;
  description: string;
  is_repaired: boolean;
}

export interface SeamList {
  project_name: string;
  total_seams: number;
  repaired_seams: number;
  seams: SeamInfo[];
}

// Landmark types
export interface Landmark {
  name: string;
  latitude: number;
  longitude: number;
  photo: string | null;
  logo: string | null;
  scale: number;
  z_index: number;
  rotation: number;
  illustrated_path: string | null;
  pixel_position: [number, number] | null;
}

export interface LandmarkDetail extends Landmark {
  has_photo: boolean;
  has_illustration: boolean;
}

// Generation types
export type GenerationStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface GenerationProgress {
  status: GenerationStatus;
  total_tiles: number;
  completed_tiles: number;
  failed_tiles: number;
  current_tile: [number, number] | null;
  elapsed_seconds: number;
  estimated_remaining_seconds: number | null;
  error: string | null;
}

export interface GenerationStartResponse {
  task_id: string;
  status: GenerationStatus;
  total_tiles: number;
  websocket_url: string;
}

// WebSocket message types
export interface WSProgressMessage {
  type: 'progress';
  data: GenerationProgress;
}

export interface WSTileCompleteMessage {
  type: 'tile_complete';
  data: {
    col: number;
    row: number;
  };
}

export interface WSErrorMessage {
  type: 'error';
  data: {
    error: string;
  };
}

export interface WSDoneMessage {
  type: 'done';
  data: {
    status: GenerationStatus;
    error: string | null;
  };
}

export type WSMessage = WSProgressMessage | WSTileCompleteMessage | WSErrorMessage | WSDoneMessage;

// API response types
export interface SuccessResponse {
  success: boolean;
  message: string;
}

export interface ErrorResponse {
  success: boolean;
  error: string;
  detail?: string;
}

export interface CostEstimate {
  project_name: string;
  breakdown: {
    tiles: {
      count: number;
      unit_cost: number;
      total: number;
    };
    landmarks: {
      count: number;
      unit_cost: number;
      total: number;
    };
    seams: {
      count: number;
      unit_cost: number;
      total: number;
    };
  };
  total_estimated_cost_usd: number;
}
