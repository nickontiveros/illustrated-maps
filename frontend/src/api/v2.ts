/** Typed client for the V2 asset-composition API (/api/v2). */

import { getAuthHeaders } from '@/hooks/useAPIKeys';

const API_BASE = '/api/v2';

export interface V2Region {
  north: number;
  south: number;
  east: number;
  west: number;
}

export interface V2Poi {
  name: string;
  lat: number;
  lon: number;
  tier: number;
  photo?: string | null;
  feature_type?: string;
  path?: [number, number][] | null;
}

export interface V2ProjectConfig {
  name: string;
  title?: string | null;
  region: V2Region;
  output?: { width_px: number; height_px: number; dpi: number };
  camera?: { convergence: number; vertical_scale: number; horizon_margin: number };
  distortion_strength?: number;
  rotation_deg?: number | 'auto';
  pois: V2Poi[];
}

export interface V2JobState {
  stage: string;
  state: 'idle' | 'running' | 'done' | 'error';
  detail: string;
  current: number;
  total: number;
}

export type V2Status = Record<'plan' | 'assets' | 'compose' | 'repaint', V2JobState>;

export interface V2RepaintQuadrant {
  x: number;
  y: number;
  status: 'pending' | 'generated' | 'skipped' | 'flagged';
}

export interface V2RepaintState {
  grid: { cols: number; rows: number; repaint_scale: number } | null;
  quadrants: V2RepaintQuadrant[];
  calls_made?: number;
}

export interface V2RepaintDryRun {
  stage: string;
  state: string;
  mode: 'single' | 'tiled';
  calls_planned: number;
  estimated_cost_usd: number;
  // tiled mode only:
  repaint_scale?: number;
  grid?: { cols: number; rows: number };
}

export interface V2ProjectSummary {
  id: string;
  name: string;
  region: V2Region;
  poi_count: number;
  has_plan: boolean;
  plan_stale: boolean;
  has_poster: boolean;
  status: V2Status;
}

export interface V2ProjectDetail extends V2ProjectSummary {
  config: V2ProjectConfig;
}

export interface V2Asset {
  id: string;
  kind: string;
  subject: string;
  cached: boolean;
  url: string;
  palette_score: number | null;
  palette_outlier: boolean;
  flagged: boolean;
}

class V2ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'V2ApiError';
  }
}

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      if (data?.detail) detail = data.detail;
    } catch {
      /* keep statusText */
    }
    throw new V2ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

function headers(json = false): Record<string, string> {
  return { ...getAuthHeaders(), ...(json ? { 'Content-Type': 'application/json' } : {}) };
}

export const v2api = {
  listProjects: () =>
    fetch(API_BASE + '/projects', { headers: headers() }).then((r) => handle<V2ProjectSummary[]>(r)),

  getProject: (id: string) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}`, { headers: headers() }).then((r) =>
      handle<V2ProjectDetail>(r)
    ),

  createProject: (config: V2ProjectConfig) =>
    fetch(API_BASE + '/projects', {
      method: 'POST',
      headers: headers(true),
      body: JSON.stringify(config),
    }).then((r) => handle<V2ProjectSummary>(r)),

  updateProject: (id: string, config: V2ProjectConfig) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}`, {
      method: 'PUT',
      headers: headers(true),
      body: JSON.stringify(config),
    }).then((r) => handle<V2ProjectSummary>(r)),

  retitle: (id: string, title: string) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/title`, {
      method: 'PATCH',
      headers: headers(true),
      body: JSON.stringify({ title }),
    }).then((r) => handle<{ title: string; plan_patched: boolean }>(r)),

  deleteProject: (id: string) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}`, {
      method: 'DELETE',
      headers: headers(),
    }).then((r) => handle<void>(r)),

  getStatus: (id: string) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/status`, { headers: headers() }).then(
      (r) => handle<V2Status>(r)
    ),

  startPlan: (id: string) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/plan`, {
      method: 'POST',
      headers: headers(true),
    }).then((r) => handle<{ stage: string }>(r)),

  listAssets: (id: string) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/assets`, { headers: headers() }).then(
      (r) => handle<V2Asset[]>(r)
    ),

  startAssets: (id: string, opts: { stub?: boolean; force?: boolean; only_ids?: string[] }) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/assets`, {
      method: 'POST',
      headers: headers(true),
      body: JSON.stringify(opts),
    }).then((r) => handle<{ stage: string }>(r)),

  flagAsset: (id: string, assetId: string, flagged: boolean) =>
    fetch(
      `${API_BASE}/projects/${encodeURIComponent(id)}/assets/${encodeURIComponent(assetId)}/flag`,
      { method: 'POST', headers: headers(true), body: JSON.stringify({ flagged }) }
    ).then((r) => handle<{ id: string; flagged: boolean }>(r)),

  startCompose: (id: string, opts: { scale?: number; harmonize?: boolean }) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/compose`, {
      method: 'POST',
      headers: headers(true),
      body: JSON.stringify(opts),
    }).then((r) => handle<{ stage: string }>(r)),

  startRepaint: (
    id: string,
    opts: { scale?: number; repaint_scale?: number; max_calls?: number | null; dry_run?: boolean; stub?: boolean }
  ) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/repaint`, {
      method: 'POST',
      headers: headers(true),
      body: JSON.stringify(opts),
    }).then((r) => handle<V2RepaintDryRun | { stage: string; state: string }>(r)),

  repaintQuadrants: (id: string) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/repaint/quadrants`, {
      headers: headers(),
    }).then((r) => handle<V2RepaintState>(r)),

  flagRepaintQuadrant: (id: string, x: number, y: number, flagged: boolean) =>
    fetch(`${API_BASE}/projects/${encodeURIComponent(id)}/repaint/quadrants/${x}/${y}/flag`, {
      method: 'POST',
      headers: headers(true),
      body: JSON.stringify({ flagged }),
    }).then((r) => handle<{ x: number; y: number; status: string }>(r)),

  previewUrl: (id: string) => `${API_BASE}/projects/${encodeURIComponent(id)}/preview.svg`,
  posterUrl: (id: string) => `${API_BASE}/projects/${encodeURIComponent(id)}/poster`,
  assetUrl: (id: string, assetId: string) =>
    `${API_BASE}/projects/${encodeURIComponent(id)}/assets/${encodeURIComponent(assetId)}`,
};
