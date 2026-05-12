export function hslForCategory(code: number, nCategories: number): string {
  if (nCategories <= 0) return "oklch(0.55 0.12 250)"
  const golden = 0.618_033_988_749_895
  const hue = ((code * golden) % 1) * 360
  return `oklch(0.62 0.13 ${hue})`
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

export function drawScatter2d(
  canvas: HTMLCanvasElement,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  colorCodes: ArrayLike<number>,
  nCategories: number,
): void {
  const n = x.length
  const ctx = canvas.getContext("2d")
  if (!ctx || n === 0) return

  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (w < 2 || h < 2) return

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
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = hslForCategory(colorCodes[i] ?? 0, nCategories)
    ctx.beginPath()
    ctx.arc(mapX(x[i]!), mapY(y[i]!), r, 0, Math.PI * 2)
    ctx.fill()
  }
}
