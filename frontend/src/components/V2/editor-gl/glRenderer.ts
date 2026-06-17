/** regl renderer for the WebGL editor spike: draws the source geography as
 * lines and warps every vertex on the GPU by sampling the per-axis CDF
 * lookup textures (identity CDF => unwarped). Pan/zoom matches the SVG
 * editor's camera (viewBox units + view transform) so an HTML overlay aligns. */

import createREGL, { type Regl, type Texture2D, type DrawCommand } from 'regl';
import type { SourceGeojson } from '@/api/v2';
import { SAMPLES, identityCDF } from './warp';
import { DEFAULT_CAMERA, type CameraParams } from './project';

export interface ViewState {
  scale: number;
  tx: number;
  ty: number;
}

// Projection (uProj = [convergence, vertical_scale, horizon_margin]) mirrors
// ./project.ts project() and mapgen/v2/plan/camera.py exactly, so the GPU draws
// the real 2.5D poster geometry (warp THEN camera), not just the flat-warped map.
const VERT = `
precision highp float;
attribute vec2 uv;      // normalized source position (0..1)
attribute float kind;   // 0 road, 1 river, 2 ground
uniform vec2 uViewBox;  // (VBW, VBH)
uniform vec2 uCanvas;   // drawing-buffer px
uniform vec2 uTranslate;// view.tx, view.ty (viewBox units)
uniform float uScale;   // view.scale
uniform vec3 uProj;     // convergence, vertical_scale, horizon_margin
uniform sampler2D uCdfU, uCdfV;
varying float vKind;
void main() {
  // separable warp: sample the CDF lookup (identity CDF => passthrough)
  float wu = texture2D(uCdfU, vec2(clamp(uv.x, 0.0, 1.0), 0.5)).r;
  float wv = texture2D(uCdfV, vec2(clamp(uv.y, 0.0, 1.0), 0.5)).r;
  // oblique camera: flat-warped (wu,wv) -> poster-normalized (X,Y)
  float conv = uProj.x, vs = uProj.y, hm = uProj.z;
  float t = clamp(wv, 0.0, 1.0);
  float ws = conv + (1.0 - conv) * t;
  float X = 0.5 + (wu - 0.5) * ws;
  float mean = (vs + 1.0) / 2.0;
  float integral = vs * t + (1.0 - vs) * t * t / 2.0;
  float Y = hm + (1.0 - hm) * (integral / mean);
  vec2 world = vec2(X * uViewBox.x, Y * uViewBox.y);
  vec2 screenVB = world * uScale + uTranslate;          // matches SVG <g> transform
  float sFit = min(uCanvas.x / uViewBox.x, uCanvas.y / uViewBox.y);
  vec2 px = screenVB * sFit + 0.5 * (uCanvas - uViewBox * sFit);
  vec2 clip = px / uCanvas * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);         // flip Y
  vKind = kind;
}`;

const FRAG = `
precision highp float;
varying float vKind;
void main() {
  vec3 c = vKind > 1.5 ? vec3(0.70, 0.64, 0.53)   // ground outline
         : vKind > 0.5 ? vec3(0.36, 0.56, 0.69)   // river
         : vec3(0.52, 0.48, 0.40);                // road
  gl_FragColor = vec4(c, vKind > 1.5 ? 0.5 : 0.92);
}`;

export interface GLMap {
  buildGeometry(src: SourceGeojson): void;
  setView(v: ViewState): void;
  setSize(cw: number, ch: number): void;
  setCDF(fx: Float32Array, fy: Float32Array): void;
  setCamera(cam: CameraParams): void;
  draw(): void;
  destroy(): void;
}

export function createGLMap(
  canvas: HTMLCanvasElement,
  VBW: number,
  VBH: number,
  cam: CameraParams = DEFAULT_CAMERA
): GLMap {
  const regl: Regl = createREGL({
    canvas,
    attributes: { antialias: true, premultipliedAlpha: false },
    extensions: ['OES_texture_float'],
    optionalExtensions: ['OES_texture_float_linear'],
  });
  const filter = regl.hasExtension('OES_texture_float_linear') ? 'linear' : 'nearest';

  const mkCdf = (data: Float32Array): Texture2D =>
    regl.texture({ width: SAMPLES, height: 1, data, format: 'luminance', type: 'float', mag: filter, min: filter, wrapS: 'clamp', wrapT: 'clamp' });

  let cdfU = mkCdf(identityCDF());
  let cdfV = mkCdf(identityCDF());
  const view: ViewState = { scale: 1, tx: 0, ty: 0 };
  let camera: CameraParams = cam;
  let canvasSize: [number, number] = [canvas.width, canvas.height];

  // --- geometry: roads + ground outlines as line segments ---
  const uvArr: number[] = [];
  const kindArr: number[] = [];
  // Subdivide so no segment exceeds ~0.02 in normalized space: the camera
  // projection bends straight lines, so long segments would look angular.
  const MAX_SEG = 0.02;
  const pushSeg = (a: [number, number], b: [number, number], k: number) => {
    const steps = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / MAX_SEG));
    for (let s = 0; s < steps; s++) {
      const f0 = s / steps;
      const f1 = (s + 1) / steps;
      uvArr.push(a[0] + (b[0] - a[0]) * f0, a[1] + (b[1] - a[1]) * f0);
      uvArr.push(a[0] + (b[0] - a[0]) * f1, a[1] + (b[1] - a[1]) * f1);
      kindArr.push(k, k);
    }
  };
  // placeholder; filled by setSource below before first draw
  const drawLines: DrawCommand = regl({
    vert: VERT,
    frag: FRAG,
    attributes: { uv: regl.prop<{ uv: number[] }, 'uv'>('uv'), kind: regl.prop<{ kind: number[] }, 'kind'>('kind') },
    uniforms: {
      uViewBox: [VBW, VBH],
      uCanvas: () => canvasSize,
      uScale: () => view.scale,
      uTranslate: () => [view.tx, view.ty],
      uProj: () => [camera.convergence, camera.vertical_scale, camera.horizon_margin],
      uCdfU: () => cdfU,
      uCdfV: () => cdfV,
    },
    count: regl.prop<{ count: number }, 'count'>('count'),
    primitive: 'lines',
    lineWidth: 1,
    blend: {
      enable: true,
      func: { srcRGB: 'src alpha', srcAlpha: 1, dstRGB: 'one minus src alpha', dstAlpha: 1 },
    },
    depth: { enable: false },
  });

  let uvBuffer = regl.buffer([0, 0]);
  let kindBuffer = regl.buffer([0]);
  let count = 0;

  function buildGeometry(src: SourceGeojson) {
    uvArr.length = 0;
    kindArr.length = 0;
    for (const r of src.roads) {
      const k = r.cls === 'river' || r.cls === 'stream' ? 1 : 0;
      for (let i = 1; i < r.points.length; i++) pushSeg(r.points[i - 1], r.points[i], k);
    }
    for (const g of src.ground) {
      const pts = g.exterior;
      for (let i = 1; i < pts.length; i++) pushSeg(pts[i - 1], pts[i], 2);
      if (pts.length > 2) pushSeg(pts[pts.length - 1], pts[0], 2); // close
    }
    uvBuffer = regl.buffer(new Float32Array(uvArr));
    kindBuffer = regl.buffer(new Float32Array(kindArr));
    count = kindArr.length;
  }

  return {
    buildGeometry,
    setView(v) {
      view.scale = v.scale;
      view.tx = v.tx;
      view.ty = v.ty;
    },
    setSize(cw, ch) {
      canvas.width = cw;
      canvas.height = ch;
      canvasSize = [cw, ch];
    },
    setCDF(fx, fy) {
      cdfU.destroy();
      cdfV.destroy();
      cdfU = mkCdf(fx);
      cdfV = mkCdf(fy);
    },
    setCamera(c) {
      camera = c;
    },
    draw() {
      regl.clear({ color: [0.969, 0.945, 0.882, 1], depth: 1 });
      if (count > 0) drawLines({ uv: uvBuffer, kind: kindBuffer, count });
    },
    destroy() {
      cdfU.destroy();
      cdfV.destroy();
      regl.destroy();
    },
  };
}
