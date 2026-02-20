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

export interface TypographySettings {
  enabled: boolean;
  road_labels: boolean;
  district_labels: boolean;
  water_labels: boolean;
  park_labels: boolean;
  title_text: string | null;
  subtitle_text: string | null;
  font_scale: number;
  halo_width: number;
  max_labels: number;
  min_road_length_px: number;
}

export interface RoadStyleSettings {
  enabled: boolean;
  motorway_exaggeration: number;
  primary_exaggeration: number;
  secondary_exaggeration: number;
  residential_exaggeration: number;
  motorway_color: string | null;
  primary_color: string | null;
  secondary_color: string | null;
  residential_color: string | null;
  outline_color: string | null;
  wobble_amount: number;
  wobble_frequency: number;
  overlay_on_output: boolean;
  overlay_opacity: number;
  reference_opacity: number;
  preset: string | null;
}

export interface AtmosphereSettings {
  enabled: boolean;
  haze_color: string;
  haze_strength: number;
  contrast_reduction: number;
  saturation_reduction: number;
  gradient_curve: number;
}

export interface BorderSettings {
  enabled: boolean;
  style: 'vintage_scroll' | 'art_deco' | 'modern_minimal' | 'ornate_victorian';
  margin: number;
  show_compass: boolean;
  show_legend: boolean;
  show_scale_bar: boolean;
  border_color: string | null;
  background_color: string | null;
  ornament_opacity: number;
}

export interface NarrativeSettings {
  auto_discover: boolean;
  max_landmarks: number;
  show_activities: boolean;
  max_activity_markers: number;
  min_importance_score: number;
}

export interface StyleSettings {
  perspective_angle: number;
  orientation: 'north' | 'south' | 'east' | 'west';
  orientation_degrees: number | null;
  prompt: string;
  color_palette: string[] | null;
  palette_preset: string | null;
  palette_enforcement_strength: number;
  color_consistency_strength: number;
  typography: TypographySettings | null;
  road_style: RoadStyleSettings | null;
  atmosphere: AtmosphereSettings | null;
  terrain_exaggeration: number;
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
  title: string | null;
  subtitle: string | null;
  border: BorderSettings | null;
  narrative: NarrativeSettings | null;
}

export interface LandmarkDiscoverRequest {
  min_importance_score?: number;
  max_landmarks?: number;
}

export interface LandmarkDiscoverResponse {
  discovered: number;
  landmarks: LandmarkDetail[];
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
  offset_dx: number;
  offset_dy: number;
}

export interface TileOffset {
  col: number;
  row: number;
  dx: number;
  dy: number;
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

// Active task types
export interface ActiveTaskInfo {
  task_id: string;
  project_name: string;
  task_type: string;
  status: GenerationStatus;
  created_at: string | null;
  progress?: {
    total_tiles: number;
    completed_tiles: number;
    failed_tiles: number;
    elapsed_seconds: number;
    estimated_remaining_seconds: number | null;
  };
}

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
