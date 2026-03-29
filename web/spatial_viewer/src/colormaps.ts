export type ColormapId = "viridis" | "magma" | "diverging";

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [183, 222, 39],
  [253, 231, 37],
];

const MAGMA: [number, number, number][] = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 13, 99],
  [156, 37, 99],
  [205, 71, 93],
  [238, 116, 84],
  [251, 173, 81],
  [252, 229, 91],
  [252, 253, 191],
];

function lerpStops(
  t: number,
  stops: [number, number, number][],
): [number, number, number] {
  const n = stops.length - 1;
  const x = t * n;
  const i = Math.min(Math.floor(x), n - 1);
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

function divergingRgb(t: number): [number, number, number] {
  const lo: [number, number, number] = [49, 54, 149];
  const mid: [number, number, number] = [247, 247, 247];
  const hi: [number, number, number] = [165, 0, 38];
  if (t <= 0.5) {
    const u = t * 2;
    return [
      Math.round(lo[0] + (mid[0] - lo[0]) * u),
      Math.round(lo[1] + (mid[1] - lo[1]) * u),
      Math.round(lo[2] + (mid[2] - lo[2]) * u),
    ];
  }
  const u = (t - 0.5) * 2;
  return [
    Math.round(mid[0] + (hi[0] - mid[0]) * u),
    Math.round(mid[1] + (hi[1] - mid[1]) * u),
    Math.round(mid[2] + (hi[2] - mid[2]) * u),
  ];
}

export function applyColors(
  values: Float32Array | null,
  colors: Uint8Array,
  n: number,
  cmap: ColormapId,
): { lo: number; hi: number } {
  if (!values || values.length !== n) {
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      colors[o] = 55;
      colors[o + 1] = 60;
      colors[o + 2] = 68;
      colors[o + 3] = 220;
    }
    return { lo: 0, hi: 1 };
  }
  let lo: number;
  let hi: number;
  if (cmap === "diverging") {
    const s = symmetricDivergingRange(values);
    lo = s.lo;
    hi = s.hi;
  } else {
    const r = robustRange(values);
    lo = r.lo;
    hi = r.hi;
  }
  for (let i = 0; i < n; i++) {
    const rgb = mapValueToRgb(values[i]!, lo, hi, cmap);
    const o = i * 4;
    colors[o] = rgb[0];
    colors[o + 1] = rgb[1];
    colors[o + 2] = rgb[2];
    colors[o + 3] = 240;
  }
  return { lo, hi };
}

export function colormapLegendGradientCss(map: ColormapId): string {
  const n = 28;
  const parts: string[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const rgb =
      map === "diverging"
        ? mapValueToRgb(-1 + 2 * t, -1, 1, "diverging")
        : mapValueToRgb(t, 0, 1, map);
    parts.push(`rgb(${rgb[0]},${rgb[1]},${rgb[2]}) ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${parts.join(", ")})`;
}

export function mapValueToRgb(
  v: number,
  lo: number,
  hi: number,
  map: ColormapId,
): [number, number, number] {
  if (!Number.isFinite(v) || !Number.isFinite(lo) || !Number.isFinite(hi)) {
    return [128, 128, 128];
  }
  if (map === "diverging") {
    const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-12);
    let t = (v + m) / (2 * m);
    t = Math.max(0, Math.min(1, t));
    return divergingRgb(t);
  }
  if (hi <= lo) {
    return lerpStops(0.5, VIRIDIS);
  }
  let t = (v - lo) / (hi - lo);
  t = Math.max(0, Math.min(1, t));
  return map === "magma" ? lerpStops(t, MAGMA) : lerpStops(t, VIRIDIS);
}

export function robustRange(
  values: Float32Array,
  lowQ = 0.02,
  highQ = 0.98,
): { lo: number; hi: number } {
  const n = values.length;
  if (n === 0) return { lo: 0, hi: 1 };
  const tmp = new Float32Array(values);
  tmp.sort();
  const ilo = Math.min(Math.floor(lowQ * (n - 1)), n - 1);
  const ihi = Math.min(Math.floor(highQ * (n - 1)), n - 1);
  let lo = tmp[ilo];
  let hi = tmp[ihi];
  if (hi <= lo) {
    hi = lo + 1e-6;
  }
  return { lo, hi };
}

export function symmetricDivergingRange(values: Float32Array): {
  lo: number;
  hi: number;
} {
  const n = values.length;
  if (n === 0) return { lo: -1, hi: 1 };
  const abs = new Float32Array(n);
  for (let i = 0; i < n; i++) abs[i] = Math.abs(values[i]);
  abs.sort();
  const i98 = Math.min(Math.floor(0.98 * (n - 1)), n - 1);
  const m = Math.max(abs[i98], 1e-12);
  return { lo: -m, hi: m };
}

const ALL_ZERO_EPS = 1e-12;

function setNoSignalColor(colors: Uint8Array, i: number) {
  const o = i * 4;
  colors[o] = 55;
  colors[o + 1] = 60;
  colors[o + 2] = 68;
  colors[o + 3] = 220;
}

/**
 * Betadata coloring: min–max (or symmetric diverging) is computed **per cluster** so one
 * cluster cannot dominate the colormap. Clusters whose coefficients are all ~0 are drawn
 * in neutral gray.
 */
export function applyBetadataColorsPerCluster(
  values: Float32Array,
  clusters: Uint32Array,
  colors: Uint8Array,
  n: number,
  cmap: ColormapId,
): void {
  if (values.length !== n || clusters.length !== n) {
    for (let i = 0; i < n; i++) setNoSignalColor(colors, i);
    return;
  }

  type Agg = { minV: number; maxV: number; maxAbs: number };
  const byCluster = new Map<number, Agg>();

  for (let i = 0; i < n; i++) {
    const c = clusters[i]!;
    const v = values[i]!;
    const a = Math.abs(v);
    let g = byCluster.get(c);
    if (!g) {
      byCluster.set(c, { minV: v, maxV: v, maxAbs: a });
    } else {
      g.minV = Math.min(g.minV, v);
      g.maxV = Math.max(g.maxV, v);
      g.maxAbs = Math.max(g.maxAbs, a);
    }
  }

  for (let i = 0; i < n; i++) {
    const c = clusters[i]!;
    const g = byCluster.get(c);
    if (!g || g.maxAbs < ALL_ZERO_EPS) {
      setNoSignalColor(colors, i);
      continue;
    }
    let lo: number;
    let hi: number;
    if (cmap === "diverging") {
      const m = Math.max(g.maxAbs, 1e-12);
      lo = -m;
      hi = m;
    } else {
      lo = g.minV;
      hi = g.maxV;
      if (hi <= lo) {
        hi = lo + 1e-6;
      }
    }
    const rgb = mapValueToRgb(values[i]!, lo, hi, cmap);
    const o = i * 4;
    colors[o] = rgb[0];
    colors[o + 1] = rgb[1];
    colors[o + 2] = rgb[2];
    colors[o + 3] = 240;
  }
}
