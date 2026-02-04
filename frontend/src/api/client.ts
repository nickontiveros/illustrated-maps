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

export const api = {
  // Projects
  async listProjects() {
    const response = await fetch(`${API_BASE}/projects`);
    return handleResponse<import('@/types').ProjectSummary[]>(response);
  },

  async getProject(name: string) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}`);
    return handleResponse<import('@/types').ProjectDetail>(response);
  },

  async createProject(data: {
    name: string;
    region: import('@/types').BoundingBox;
    output?: import('@/types').OutputSettings;
    style?: import('@/types').StyleSettings;
    tiles?: import('@/types').TileSettings;
  }) {
    const response = await fetch(`${API_BASE}/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return handleResponse<import('@/types').ProjectDetail>(response);
  },

  async updateProject(name: string, data: {
    output?: import('@/types').OutputSettings;
    style?: import('@/types').StyleSettings;
    tiles?: import('@/types').TileSettings;
  }) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return handleResponse<import('@/types').ProjectDetail>(response);
  },

  async deleteProject(name: string, deleteCache = true) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(name)}?delete_cache=${deleteCache}`,
      { method: 'DELETE' }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async getProjectCostEstimate(name: string) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}/cost-estimate`);
    return handleResponse<import('@/types').CostEstimate>(response);
  },

  // Tiles
  async getTileGrid(projectName: string) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles`);
    return handleResponse<import('@/types').TileGrid>(response);
  },

  async getTileInfo(projectName: string, col: number, row: number) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/tiles/${col}/${row}`
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
        headers: { 'Content-Type': 'application/json' },
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(options ?? {}),
      }
    );
    return handleResponse<import('@/types').GenerationStartResponse>(response);
  },

  async getGenerationStatus(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/generate/status`
    );
    return handleResponse<import('@/types').GenerationProgress>(response);
  },

  async cancelGeneration(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/generate/cancel`,
      { method: 'POST' }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  // Seams
  async listSeams(projectName: string) {
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/seams`);
    return handleResponse<import('@/types').SeamList>(response);
  },

  async getSeam(projectName: string, seamId: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/${encodeURIComponent(seamId)}`
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
      { method: 'POST' }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async repairSeamsBatch(projectName: string, seamIds: string[]) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/repair-batch`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seam_ids: seamIds }),
      }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async repairAllSeams(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/seams/repair-all`,
      { method: 'POST' }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  // Landmarks
  async listLandmarks(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks`
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
  }) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
  }) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }
    );
    return handleResponse<import('@/types').LandmarkDetail>(response);
  },

  async deleteLandmark(projectName: string, landmarkName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/${encodeURIComponent(landmarkName)}`,
      { method: 'DELETE' }
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
      { method: 'POST' }
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
        body: formData,
      }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  async illustrateAllLandmarks(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/landmarks/illustrate-all`,
      { method: 'POST' }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  // DZI (Deep Zoom Images)
  async getDZIInfo(projectName: string) {
    const response = await fetch(
      `${API_BASE}/projects/${encodeURIComponent(projectName)}/dzi/info`
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
      { method: 'POST' }
    );
    return handleResponse<import('@/types').SuccessResponse>(response);
  },

  getDZIDescriptorUrl(projectName: string) {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/dzi/assembled.dzi`;
  },

  getDZITileUrl(projectName: string, level: number, col: number, row: number, format = 'jpg') {
    return `${API_BASE}/projects/${encodeURIComponent(projectName)}/dzi/assembled_files/${level}/${col}_${row}.${format}`;
  },
};

export { ApiError };
