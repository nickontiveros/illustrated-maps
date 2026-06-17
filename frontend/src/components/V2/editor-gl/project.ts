/** Client-side replica of the oblique 2.5D camera (mapgen/v2/plan/camera.py
 * ObliqueCamera) so the WebGL editor can show the *projected* poster geometry,
 * not just the flat-warped map. MUST match the Python math exactly, or dragged
 * labels/POIs land in the wrong place. Verified numerically against the backend.
 *
 * Spaces (all axis-normalized to [0,1]):
 *   pre-warp uv  --warp-->  flat-warped (wu,wv)  --project-->  poster (X,Y)
 * The warp half lives in ./warp.ts; this file is the camera half + combinators.
 */

import { warpValue, inverseWarpValue } from './warp';

export interface CameraParams {
  convergence: number; // far-edge width relative to near edge (1 = none)
  vertical_scale: number; // far-edge vertical compression (1 = none)
  horizon_margin: number; // fraction of height reserved above the far edge
}

// Mirrors mapgen/v2/types.py CameraSpec field defaults.
export const DEFAULT_CAMERA: CameraParams = {
  convergence: 0.78,
  vertical_scale: 0.55,
  horizon_margin: 0.06,
};

/** flat-warped (wu,wv) -> poster-normalized (X,Y). Mirrors project_point:
 * X narrows toward the far edge (convergence); Y is the integral of the linear
 * vertical-scale profile, normalized so the near edge lands at the bottom. */
export function project(wu: number, wv: number, cam: CameraParams): [number, number] {
  const { convergence: conv, vertical_scale: vs, horizon_margin: hm } = cam;
  const t = Math.min(1, Math.max(0, wv));
  const ws = conv + (1 - conv) * t;
  const X = 0.5 + (wu - 0.5) * ws;
  const mean = (vs + 1) / 2;
  const integral = vs * t + ((1 - vs) * t * t) / 2;
  const Y = hm + (1 - hm) * (integral / mean);
  return [X, Y];
}

/** Inverse of project: poster-normalized (X,Y) -> flat-warped (wu,wv).
 * Vertical depends only on wv, so recover t (=wv) from Y first (the quadratic
 * solved in camera.py t_at_poster_y), then wu from X and the width scale. */
export function unproject(X: number, Y: number, cam: CameraParams): [number, number] {
  const { convergence: conv, vertical_scale: vs, horizon_margin: hm } = cam;
  const mean = (vs + 1) / 2;
  const c = Math.max(0, (Y - hm) / (1 - hm)) * mean;
  const a = (1 - vs) / 2;
  let t: number;
  if (a < 1e-9) t = c / Math.max(1e-9, vs);
  else t = (-vs + Math.sqrt(vs * vs + 4 * a * c)) / (2 * a);
  t = Math.min(1, Math.max(0, t));
  const ws = conv + (1 - conv) * t;
  const wu = 0.5 + (X - 0.5) / (ws || 1e-9);
  return [wu, t];
}

export interface Transform {
  /** pre-warp uv -> poster-normalized (X,Y): where it actually prints. */
  fwd: (u: number, v: number) => [number, number];
  /** poster-normalized (X,Y) -> pre-warp uv: for a dragged label drop. */
  inv: (X: number, Y: number) => [number, number];
  /** pre-warp uv -> flat-warped (wu,wv): the post-warp frame POI offsets live in. */
  warp: (u: number, v: number) => [number, number];
  /** poster-normalized (X,Y) -> flat-warped (wu,wv): for a dragged POI drop. */
  unwarpProject: (X: number, Y: number) => [number, number];
}

/** Bundle the current warp CDFs + camera into the four maps the editor needs. */
export function makeTransform(fx: Float32Array, fy: Float32Array, cam: CameraParams): Transform {
  return {
    fwd: (u, v) => project(warpValue(fx, u), warpValue(fy, v), cam),
    inv: (X, Y) => {
      const [wu, wv] = unproject(X, Y, cam);
      return [inverseWarpValue(fx, wu), inverseWarpValue(fy, wv)];
    },
    warp: (u, v) => [warpValue(fx, u), warpValue(fy, v)],
    unwarpProject: (X, Y) => unproject(X, Y, cam),
  };
}
