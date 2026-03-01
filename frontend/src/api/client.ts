import { getAuthHeaders } from '@/hooks/useAPIKeys';

const API_BASE = '/api';

class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public data?: unknown
  ) {
    super(`API Error: ${status} ${statusText}`);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let data;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    throw new ApiError(response.status, response.statusText, data);
  }
  return response.json();
}

/** Build headers with API keys and optional extra headers. */
function buildHeaders(extra?: Record<string, string>): Record<string, string> {
  return { ...getAuthHeaders(), ...extra };
}

/** Build headers for JSON requests. */
function jsonHeaders(): Record<string, string> {
  return buildHeaders({ 'Content-Type': 'application/json' });
}

export const api = {
  // Projects
  async listProjects() {
    const response = await fetch(`${API_BASE}/projects`, { headers: buildHeaders() });
    return handleResponse<import('@/types').ProjectSummary[]>(response);
  },

  async getProject(name: string) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}`, { headers: buildHeaders() });
    return handleResponse<import('@/types').ProjectDetail>(response);
  },

  async createProject(data: {
    name: string;
    region?: import('@/types').BoundingBox;
    oriented_region?: import('@/types').OrientedRegion;
    output?: import('@/types').OutputSettings;
    style?: import('@/types').StyleSettings;
    tiles?: import('@/types').TileSettings;
  }) {
    const response = await fetch(`${API_BASE}/projects`, {
      method: 'POST',
      headers: jsonHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse<import('@/types').ProjectDetail>(response);
  },

  async updateProject(name: string, data: {
    output?: import('@/types').OutputSettings;
    style?: import('@/types').StyleSettings;
    tiles?: import('@/types').TileSettings;
    oriented_region?: import('@/types').OrientedRegion;
    title?: string;
    subtitle?: string;
    border?: import('@/types').BorderSettings;
    narrative?: import('@/types').NarrativeSettings;
  }) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}`, {
      method: 'PUT',
      headers: jsonHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse<import('@/types').ProjectDetail>(response);
  },

  async deleteProject(name: string, deleteCache = true) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(name)}?delete_cache=${deleteCache}`,
      { method: 'DELETE', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async getProjectCostEstimate(name: string) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}/cost-estimate`, { headers: buildHeaders() });
    return handleResponse<import('@/types').CostEstimate>(response);
  },

  // Tiles
  async getTileGrid(projectName: string) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles`, { headers: buildHeaders() });
    return handleResponse<import('@/types').TileGrid>(response);
  },

  async getTileInfo(projectName: string, col: number, row: number) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/${col}/${row}`,
      { headers: buildHeaders() }
    );
    return handleResponse<import('@/types').TileSpec>(response);
  },

  getTileReferenceUrl(projectName: string, col: number, row: number, size?: number) {
    const params = size ? `?size=${size}` : '';
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/${col}/${row}/reference${params}`;
  },

  getTileGeneratedUrl(projectName: string, col: number, row: number, size?: number) {
    const params = size ? `?size=${size}` : '';
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/${col}/${row}/generated${params}`;
  },

  getTileThumbnailUrl(projectName: string, col: number, row: number) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/${col}/${row}/thumbnail`;
  },

  async regenerateTile(projectName: string, col: number, row: number, force = false) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/${col}/${row}/regenerate`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ force }),
      }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  // Generation
  async startGeneration(projectName: string, options?: {
    skip_existing?: boolean;
    tile_filter?: [number, number][];
  }) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/generate`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify(options ?? {}),
      }
    );
    return handleResponse<import('@/types').GenerationStartResponse>(response);
  },

  async getGenerationStatus(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/generate/status`,
      { headers: buildHeaders() }
    );
    return handleResponse<import('@/types').GenerationProgress>(response);
  },

  async cancelGeneration(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/generate/cancel`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  // Seams
  async listSeams(projectName: string) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/seams`, { headers: buildHeaders() });
    return handleResponse<import('@/types').SeamList>(response);
  },

  async getSeam(projectName: string, seamId: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/${encodeURIComponent(seamId)}`,
      { headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SeamInfo>(response);
  },

  getSeamPreviewUrl(projectName: string, seamId: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/${encodeURIComponent(seamId)}/preview`;
  },

  getSeamRepairedUrl(projectName: string, seamId: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/${encodeURIComponent(seamId)}/repaired`;
  },

  async repairSeam(projectName: string, seamId: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/${encodeURIComponent(seamId)}/repair`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async repairSeamsBatch(projectName: string, seamIds: string[]) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/repair-batch`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ seam_ids: seamIds }),
      }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async repairAllSeams(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/repair-all`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  // Landmarks
  async listLandmarks(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks`,
      { headers: buildHeaders() }
    );
    return handleResponse<import('@/types').LandmarkDetail[]>(response);
  },

  async createLandmark(projectName: string, data: {
    name: string;
    latitude: number;
    longitude: number;
    photo?: string;
    logo?: string;
    scale?: number;
    z_index?: number;
    rotation?: number;
    wikipedia_url?: string;
    wikidata_id?: string;
  }) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify(data),
      }
    );
    return handleResponse<import('@/types').LandmarkDetail>(response);
  },

  async updateLandmark(projectName: string, landmarkName: string, data: {
    latitude?: number;
    longitude?: number;
    scale?: number;
    z_index?: number;
    rotation?: number;
    wikipedia_url?: string;
    wikidata_id?: string;
  }) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}`,
      {
        method: 'PUT',
        headers: jsonHeaders(),
        body: JSON.stringify(data),
      }
    );
    return handleResponse<import('@/types').LandmarkDetail>(response);
  },

  async deleteLandmark(projectName: string, landmarkName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}`,
      { method: 'DELETE', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  getLandmarkPhotoUrl(projectName: string, landmarkName: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}/photo`;
  },

  getLandmarkIllustrationUrl(projectName: string, landmarkName: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}/illustration`;
  },

  async illustrateLandmark(projectName: string, landmarkName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}/illustrate`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async uploadLandmarkPhoto(projectName: string, landmarkName: string, file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}/upload-photo`,
      {
        method: 'POST',
        headers: buildHeaders(),
        body: formData,
      }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async discoverLandmarks(projectName: string, data?: import('@/types').LandmarkDiscoverRequest) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/discover`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify(data ?? {}),
      }
    );
    return handleResponse<import('@/types').LandmarkDiscoverResponse>(response);
  },

  async illustrateAllLandmarks(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/illustrate-all`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async fetchWikipediaPhoto(projectName: string, landmarkName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}/fetch-photo`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  // Tile Offsets
  async getTileOffset(projectName: string, col: number, row: number) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/${col}/${row}/offset`,
      { headers: buildHeaders() }
    );
    return handleResponse<import('@/types').TileOffset>(response);
  },

  async setTileOffset(projectName: string, col: number, row: number, dx: number, dy: number) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/${col}/${row}/offset`,
      {
        method: 'PUT',
        headers: jsonHeaders(),
        body: JSON.stringify({ dx, dy }),
      }
    );
    return handleResponse<import('@/types').TileOffset>(response);
  },

  async getAllTileOffsets(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/offsets`,
      { headers: buildHeaders() }
    );
    return handleResponse<{ project_name: string; offsets: import('@/types').TileOffset[] }>(response);
  },

  // Style Reference
  async uploadStyleReference(projectName: string, file: File) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/style-reference`,
      { method: 'POST', headers: buildHeaders(), body: formData }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  getStyleReferenceUrl(projectName: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/style-reference`;
  },

  async deleteStyleReference(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/style-reference`,
      { method: 'DELETE', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async hasStyleReference(projectName: string): Promise<boolean> {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/style-reference`,
      { method: 'HEAD', headers: buildHeaders() }
    );
    return response.ok;
  },

  // Assembly
  async assembleTiles(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/assemble`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  // Active Tasks
  async getActiveTasks() {
    const response = await fetch(`${API_BASE}/tasks/active`, { headers: buildHeaders() });
    return handleResponse<import('@/types').ActiveTaskInfo[]>(response);
  },

  // Config
  async getConfig() {
    const response = await fetch(`${API_BASE}/config`, { headers: buildHeaders() });
    return handleResponse<import('@/types').APIConfig>(response);
  },

  // DZI (Deep Zoom Images)
  async getDZIInfo(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/dzi/info`,
      { headers: buildHeaders() }
    );
    return handleResponse<{
      project_name: string;
      is_generated: boolean;
      width: number;
      height: number;
      tile_size: number;
      overlap: number;
      format: string;
      max_level: number;
      num_levels: number;
    }>(response);
  },

  async generateDZI(projectName: string, force = false) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/dzi/generate?force=${force}`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  getDZIDescriptorUrl(projectName: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/dzi/assembled.dzi`;
  },

  getDZITileUrl(projectName: string, level: number, col: number, row: number, format = 'jpg') {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/dzi/assembled_files/${level}/${col}_${row}.${format}`;
  },

  // Post-processing
  async getPostProcessStatus(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/status`,
      { headers: buildHeaders() }
    );
    return handleResponse<import('@/types').PostProcessStatus>(response);
  },

  async composeLandmarks(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/compose`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async addLabels(projectName: string, includeShields: boolean = true) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/labels`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ include_shields: includeShields }),
      }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async addBorder(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/border`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async startOutpaint(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/outpaint`,
      { method: 'POST', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').GenerationStartResponse>(response);
  },

  async startPipeline(projectName: string, steps: string[], includeShields: boolean = true) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/pipeline`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ steps, include_shields: includeShields }),
      }
    );
    return handleResponse<import('@/types').GenerationStartResponse>(response);
  },

  async applyPerspective(projectName: string, params?: {
    angle?: number;
    convergence?: number;
    vertical_scale?: number;
    horizon_margin?: number;
  }) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/perspective`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify(params ?? {}),
      }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  getPostProcessImageUrl(projectName: string, stage: string, size?: number) {
    const params = size ? `?size=${size}` : '';
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/${stage}/image${params}`;
  },

  async exportPSD(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/postprocess/export-psd`,
      { method: 'POST', headers: buildHeaders() }
    );
    return response;
  },

  // Previews
  getPreviewOSMUrl(projectName: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/preview/osm`;
  },

  getPreviewCompositeUrl(projectName: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/preview/composite`;
  },

  // Cache
  async clearCache(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/cache`,
      { method: 'DELETE', headers: buildHeaders() }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },
};

export { ApiError };
