/** Client-side replica of the backend warp (mapgen/v2/plan/distortion.py
 * ImportanceWarp, manual/plateau path) so the GPU can morph the map in real
 * time. MUST match the Python constants exactly, or the live morph drifts from
 * the authoritative render. Verified numerically against the backend.
 *
 * The warp is separable + monotonic: u' = fx(u), v' = fy(v), each fx/fy a
 * normalized cumulative density (CDF) over a 1024-sample grid. */

export const SAMPLES = 1024;

export interface WarpRegionLike {
  bounds: [number, number, number, number]; // [u0, v0, u1, v1] normalized
  magnify: number;
}

function smoothstep(t: number): number {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
}

// builder.py _magnify_weight: plateau weight that stretches a band by `magnify`.
function magnifyWeight(width: number, magnify: number): number {
  if (magnify === 1 || width <= 0) return 0;
  const denom = 1 - magnify * width;
  if (denom <= 1e-3) return 1000; // region too wide -> saturate
  return Math.max(-0.9, (magnify - 1) / denom);
}

/** Per-axis CDF (Float32Array of SAMPLES) from the spec's warp regions.
 * axis 0 = u (bounds[0],bounds[2]); axis 1 = v (bounds[1],bounds[3]). */
export function buildCDF(regions: WarpRegionLike[], axis: 0 | 1): Float32Array {
  const n = SAMPLES;
  // density (distortion.py _plateau_density): base 1 + plateau bumps.
  const density = new Float32Array(n).fill(1);
  for (const r of regions) {
    const lo = Math.min(r.bounds[axis], r.bounds[axis + 2]);
    const hi = Math.max(r.bounds[axis], r.bounds[axis + 2]);
    const width = hi - lo;
    if (width <= 0) continue;
    const w = magnifyWeight(width, r.magnify);
    const soft = Math.max(0.025, width * 0.3);
    for (let i = 0; i < n; i++) {
      const g = i / (n - 1);
      const up = smoothstep((g - (lo - soft)) / soft);
      const down = 1 - smoothstep((g - hi) / soft);
      const plateau = Math.min(1, Math.max(0, Math.min(up, down)));
      density[i] += w * plateau; // strength = 1.0
    }
  }
  // cdf (distortion.py _cdf): trapezoid cumsum, prepend 0, normalize by last.
  const cdf = new Float32Array(n);
  for (let i = 1; i < n; i++) cdf[i] = cdf[i - 1] + (density[i] + density[i - 1]) / 2;
  const last = cdf[n - 1] || 1;
  for (let i = 0; i < n; i++) cdf[i] /= last;
  return cdf;
}

export function identityCDF(): Float32Array {
  const cdf = new Float32Array(SAMPLES);
  for (let i = 0; i < SAMPLES; i++) cdf[i] = i / (SAMPLES - 1);
  return cdf;
}

/** warp(u) = interp(u, grid, cdf). For positioning handles in warped space. */
export function warpValue(cdf: Float32Array, u: number): number {
  const n = cdf.length;
  const x = Math.min(1, Math.max(0, u)) * (n - 1);
  const i = Math.floor(x);
  if (i >= n - 1) return cdf[n - 1];
  const f = x - i;
  return cdf[i] * (1 - f) + cdf[i + 1] * f;
}

/** inverseWarp(u') = interp(u', cdf, grid). For mapping a dragged handle in
 * warped screen space back to its pre-warp bound. cdf is monotonic. */
export function inverseWarpValue(cdf: Float32Array, up: number): number {
  const n = cdf.length;
  const t = Math.min(cdf[n - 1], Math.max(cdf[0], up));
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (cdf[m] < t) lo = m;
    else hi = m;
  }
  const span = cdf[hi] - cdf[lo] || 1e-9;
  const f = (t - cdf[lo]) / span;
  return (lo + f) / (n - 1);
}
