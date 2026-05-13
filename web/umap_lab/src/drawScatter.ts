import type { ClusterPalette } from "@/clusterPalette"

export function hslForCategory(code: number, nCategories: number): string {
  if (nCategories <= 0) return "oklch(0.55 0.12 250)"
  const golden = 0.618_033_988_749_895
  const hue = ((code * golden) % 1) * 360
  return `oklch(0.62 0.13 ${hue})`
}

export function rgbForCategory(code: number, nCategories: number): [number, number, number] {
  if (nCategories <= 0) return [110, 140, 200]
  const golden = 0.618_033_988_749_895
  const hue = ((code * golden) % 1)
  const s = 0.65
  const l = 0.55
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((hue * 6) % 2) - 1))
  const m = l - c / 2
  let r1: number, g1: number, b1: number
  const h6 = hue * 6
  if (h6 < 1) { r1 = c; g1 = x; b1 = 0 }
  else if (h6 < 2) { r1 = x; g1 = c; b1 = 0 }
  else if (h6 < 3) { r1 = 0; g1 = c; b1 = x }
  else if (h6 < 4) { r1 = 0; g1 = x; b1 = c }
  else if (h6 < 5) { r1 = x; g1 = 0; b1 = c }
  else { r1 = c; g1 = 0; b1 = x }
  return [
    Math.round((r1 + m) * 255),
    Math.round((g1 + m) * 255),
    Math.round((b1 + m) * 255),
  ]
}

export function axisBounds(values: ArrayLike<number>): { min: number; max: number } {
  const n = values.length
  if (n === 0) return { min: 0, max: 1 }
  let min = values[0]!
  let max = values[0]!
  for (let i = 1; i < n; i++) {
    const v = values[i]!
    if (v < min) min = v
    if (v > max) max = v
  }
  if (min === max) {
    const pad = Math.abs(min) * 0.05 + 0.05
    return { min: min - pad, max: max + pad }
  }
  const pad = (max - min) * 0.04
  return { min: min - pad, max: max + pad }
}

export type ClusterCentroid = {
  code: number
  label: string
  cx: number
  cy: number
  count: number
}

export interface ScatterState {
  w: number
  h: number
  xBounds: { min: number; max: number }
  yBounds: { min: number; max: number }
  dpr: number
  centroids: ClusterCentroid[]
}

export type ScatterContinuousColor = {
  values: ArrayLike<number>
  vmin: number
  vmax: number
}

export type ScatterJitter = {
  enabled: boolean
  amp01: number
  tSec: number
}

export type ClusterLabel = {
  code: number
  label: string
  fillCss: string
}

export type DrawScatterOpts = {
  pointRadiusMul?: number
  jitter?: ScatterJitter | null
  continuous?: ScatterContinuousColor | null
  clusterLabels?: ClusterLabel[] | null
}

function fillCssForCategory(
  code: number,
  nCategories: number,
  palette: ClusterPalette | null | undefined,
): string {
  if (palette?.fillCss.length) {
    const i = Math.min(Math.max(0, code), palette.fillCss.length - 1)
    return palette.fillCss[i]!
  }
  return hslForCategory(code, nCategories)
}

function rgbForCategoryOrPalette(
  code: number,
  nCategories: number,
  palette: ClusterPalette | null | undefined,
): [number, number, number] {
  if (palette?.rgb.length) {
    const i = Math.min(Math.max(0, code), palette.rgb.length - 1)
    return palette.rgb[i]!
  }
  return rgbForCategory(code, nCategories)
}

function normExpr(v: number, vmin: number, vmax: number): number {
  if (!(vmax > vmin)) return 0.5
  return Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)))
}

function viridisRgb(t: number): [number, number, number] {
  const u = Math.max(0, Math.min(1, t))
  const s = u * u * (3 - 2 * u)
  const a: [number, number, number] = [68, 1, 84]
  const b: [number, number, number] = [253, 231, 37]
  return [
    Math.round(a[0] + (b[0] - a[0]) * s),
    Math.round(a[1] + (b[1] - a[1]) * s),
    Math.round(a[2] + (b[2] - a[2]) * s),
  ]
}

function makeMappers(
  w: number,
  h: number,
  xb: { min: number; max: number },
  yb: { min: number; max: number },
) {
  const mapX = (v: number) => ((v - xb.min) / (xb.max - xb.min)) * (w - 12) + 6
  const mapY = (v: number) => h - 6 - ((v - yb.min) / (yb.max - yb.min)) * (h - 12)
  return { mapX, mapY }
}

function jitterOffset(
  i: number,
  sx: number,
  sy: number,
  jitter: ScatterJitter | null | undefined,
  spanPx: number,
): [number, number] {
  if (!jitter?.enabled || jitter.amp01 <= 0) return [sx, sy]
  const amp = spanPx * 0.003 * Math.max(0, Math.min(1, jitter.amp01))
  const g = i * 0.813492075
  return [
    sx + amp * Math.sin(jitter.tSec * 1.12 + g),
    sy + amp * Math.cos(jitter.tSec * 0.97 + g * 1.71),
  ]
}

export function drawScatter2d(
  canvas: HTMLCanvasElement,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  colorCodes: ArrayLike<number>,
  nCategories: number,
  highlightCluster?: number | null,
  palette?: ClusterPalette | null,
  opts?: DrawScatterOpts | null,
): ScatterState | null {
  const n = x.length
  const ctx = canvas.getContext("2d")
  if (!ctx || n === 0) return null

  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (w < 2 || h < 2) return null

  canvas.width = Math.floor(w * dpr)
  canvas.height = Math.floor(h * dpr)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const xb = axisBounds(x)
  const yb = axisBounds(y)

  ctx.clearRect(0, 0, w, h)
  ctx.fillStyle = "oklch(0.99 0 0)"
  ctx.fillRect(0, 0, w, h)

  const { mapX, mapY } = makeMappers(w, h, xb, yb)
  const spanPx = Math.min(w, h)
  const cont = opts?.continuous ?? null
  const jitter = opts?.jitter ?? null
  const mul = opts?.pointRadiusMul ?? 1
  const baseR = (n > 50_000 ? 1.1 : n > 10_000 ? 1.35 : 2) * mul
  const hasHighlight = highlightCluster != null

  const screenXY = (i: number) => {
    const sx0 = mapX(x[i]!)
    const sy0 = mapY(y[i]!)
    return jitterOffset(i, sx0, sy0, jitter, spanPx)
  }

  if (cont && cont.values.length >= n) {
    const { vmin, vmax, values } = cont
    if (hasHighlight) {
      for (let i = 0; i < n; i++) {
        if ((colorCodes[i] ?? 0) === highlightCluster) continue
        const t = normExpr(values[i] ?? 0, vmin, vmax)
        const [cr, cg, cb] = viridisRgb(t)
        const [sx, sy] = screenXY(i)
        ctx.fillStyle = `rgba(${cr},${cg},${cb},0.1)`
        ctx.beginPath()
        ctx.arc(sx, sy, baseR, 0, Math.PI * 2)
        ctx.fill()
      }
      for (let i = 0; i < n; i++) {
        if ((colorCodes[i] ?? 0) !== highlightCluster) continue
        const t = normExpr(values[i] ?? 0, vmin, vmax)
        const [cr, cg, cb] = viridisRgb(t)
        const [sx, sy] = screenXY(i)
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`
        ctx.beginPath()
        ctx.arc(sx, sy, baseR * 1.35, 0, Math.PI * 2)
        ctx.fill()
      }
    } else {
      for (let i = 0; i < n; i++) {
        const t = normExpr(values[i] ?? 0, vmin, vmax)
        const [cr, cg, cb] = viridisRgb(t)
        const [sx, sy] = screenXY(i)
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`
        ctx.beginPath()
        ctx.arc(sx, sy, baseR, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  } else if (hasHighlight) {
    for (let i = 0; i < n; i++) {
      if ((colorCodes[i] ?? 0) === highlightCluster) continue
      const [cr, cg, cb] = rgbForCategoryOrPalette(colorCodes[i] ?? 0, nCategories, palette)
      const [sx, sy] = screenXY(i)
      ctx.fillStyle = `rgba(${cr},${cg},${cb},0.08)`
      ctx.beginPath()
      ctx.arc(sx, sy, baseR, 0, Math.PI * 2)
      ctx.fill()
    }
    for (let i = 0; i < n; i++) {
      if ((colorCodes[i] ?? 0) !== highlightCluster) continue
      const [sx, sy] = screenXY(i)
      ctx.fillStyle = fillCssForCategory(colorCodes[i] ?? 0, nCategories, palette)
      ctx.beginPath()
      ctx.arc(sx, sy, baseR * 1.4, 0, Math.PI * 2)
      ctx.fill()
    }
  } else {
    for (let i = 0; i < n; i++) {
      const [sx, sy] = screenXY(i)
      ctx.fillStyle = fillCssForCategory(colorCodes[i] ?? 0, nCategories, palette)
      ctx.beginPath()
      ctx.arc(sx, sy, baseR, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  const labels = opts?.clusterLabels
  const centroids: ClusterCentroid[] = []
  if (labels && labels.length > 0 && !cont) {
    const sumX = new Float64Array(nCategories)
    const sumY = new Float64Array(nCategories)
    const cnt = new Uint32Array(nCategories)
    for (let i = 0; i < n; i++) {
      const c = colorCodes[i] ?? 0
      if (c < nCategories) {
        const [sx, sy] = screenXY(i)
        sumX[c] += sx
        sumY[c] += sy
        cnt[c]++
      }
    }
    ctx.textAlign = "center"
    ctx.textBaseline = "middle"
    const fontSize = Math.max(11, Math.min(15, spanPx * 0.024))
    ctx.font = `700 ${fontSize}px ui-sans-serif, system-ui, sans-serif`
    for (const cl of labels) {
      if (cl.code >= nCategories || cnt[cl.code] === 0) continue
      const ccx = sumX[cl.code]! / cnt[cl.code]!
      const ccy = sumY[cl.code]! / cnt[cl.code]!
      centroids.push({ code: cl.code, label: cl.label, cx: ccx, cy: ccy, count: cnt[cl.code]! })
      ctx.strokeStyle = "rgba(255,255,255,0.92)"
      ctx.lineWidth = 3.5
      ctx.lineJoin = "round"
      ctx.strokeText(cl.label, ccx, ccy)
      ctx.fillStyle = "oklch(0.18 0 0)"
      ctx.fillText(cl.label, ccx, ccy)
    }
  }

  return { w, h, xBounds: xb, yBounds: yb, dpr, centroids }
}

export function findNearestPoint(
  state: ScatterState,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  canvasX: number,
  canvasY: number,
  maxRadiusPx: number,
  opts?: DrawScatterOpts | null,
): number | null {
  const { w, h, xBounds: xb, yBounds: yb } = state
  const { mapX, mapY } = makeMappers(w, h, xb, yb)
  const spanPx = Math.min(w, h)
  const jitter = opts?.jitter ?? null
  const mul = opts?.pointRadiusMul ?? 1
  const n = x.length
  const baseR = (n > 50_000 ? 1.1 : n > 10_000 ? 1.35 : 2) * mul
  const hitR = baseR * 1.6 + maxRadiusPx * 0.25

  let bestIdx = -1
  let bestDist = hitR * hitR
  for (let i = 0; i < n; i++) {
    const sx0 = mapX(x[i]!)
    const sy0 = mapY(y[i]!)
    const [sx, sy] = jitterOffset(i, sx0, sy0, jitter, spanPx)
    const dx = sx - canvasX
    const dy = sy - canvasY
    const d2 = dx * dx + dy * dy
    if (d2 < bestDist) {
      bestDist = d2
      bestIdx = i
    }
  }
  return bestIdx >= 0 ? bestIdx : null
}
