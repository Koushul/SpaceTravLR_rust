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

export interface ScatterState {
  w: number
  h: number
  xBounds: { min: number; max: number }
  yBounds: { min: number; max: number }
  dpr: number
}

export function drawScatter2d(
  canvas: HTMLCanvasElement,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  colorCodes: ArrayLike<number>,
  nCategories: number,
  highlightCluster?: number | null,
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

  const mapX = (v: number) => ((v - xb.min) / (xb.max - xb.min)) * (w - 12) + 6
  const mapY = (v: number) => h - 6 - ((v - yb.min) / (yb.max - yb.min)) * (h - 12)

  const r = n > 50_000 ? 1.1 : n > 10_000 ? 1.35 : 2
  const hasHighlight = highlightCluster != null

  if (hasHighlight) {
    for (let i = 0; i < n; i++) {
      if ((colorCodes[i] ?? 0) === highlightCluster) continue
      const [cr, cg, cb] = rgbForCategory(colorCodes[i] ?? 0, nCategories)
      ctx.fillStyle = `rgba(${cr},${cg},${cb},0.08)`
      ctx.beginPath()
      ctx.arc(mapX(x[i]!), mapY(y[i]!), r, 0, Math.PI * 2)
      ctx.fill()
    }
    for (let i = 0; i < n; i++) {
      if ((colorCodes[i] ?? 0) !== highlightCluster) continue
      ctx.fillStyle = hslForCategory(colorCodes[i] ?? 0, nCategories)
      ctx.beginPath()
      ctx.arc(mapX(x[i]!), mapY(y[i]!), r * 1.4, 0, Math.PI * 2)
      ctx.fill()
    }
  } else {
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = hslForCategory(colorCodes[i] ?? 0, nCategories)
      ctx.beginPath()
      ctx.arc(mapX(x[i]!), mapY(y[i]!), r, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  return { w, h, xBounds: xb, yBounds: yb, dpr }
}

export function findNearestPoint(
  state: ScatterState,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  canvasX: number,
  canvasY: number,
  maxRadiusPx: number,
): number | null {
  const { w, h, xBounds: xb, yBounds: yb } = state
  const mapX = (v: number) => ((v - xb.min) / (xb.max - xb.min)) * (w - 12) + 6
  const mapY = (v: number) => h - 6 - ((v - yb.min) / (yb.max - yb.min)) * (h - 12)

  let bestIdx = -1
  let bestDist = maxRadiusPx * maxRadiusPx
  const n = x.length
  for (let i = 0; i < n; i++) {
    const dx = mapX(x[i]!) - canvasX
    const dy = mapY(y[i]!) - canvasY
    const d2 = dx * dx + dy * dy
    if (d2 < bestDist) {
      bestDist = d2
      bestIdx = i
    }
  }
  return bestIdx >= 0 ? bestIdx : null
}
