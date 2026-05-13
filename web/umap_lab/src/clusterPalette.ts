import createIWantHue from "iwanthue-api"

export type ClusterPalette = {
  fillCss: string[]
  rgb: [number, number, number][]
}

function toRgbTuple(c: { rgb: number[] }): [number, number, number] {
  const [r, g, b] = c.rgb
  return [Math.round(r), Math.round(g), Math.round(b)]
}

export function buildClusterPalette(nCategories: number): ClusterPalette {
  if (nCategories <= 0) {
    return { fillCss: [], rgb: [] }
  }
  if (nCategories === 1) {
    return { fillCss: ["rgb(88,126,186)"], rgb: [[88, 126, 186]] }
  }

  const api = createIWantHue()
  const checkColor = (color: { hcl: () => number[] }) => {
    const hcl = color.hcl()
    return (
      Number.isFinite(hcl[0]) &&
      hcl[0] >= 0 &&
      hcl[0] <= 360 &&
      hcl[1] >= 0.2 &&
      hcl[1] <= 3 &&
      hcl[2] >= 0.22 &&
      hcl[2] <= 0.92
    )
  }

  const forceVector = nCategories > 0 && nCategories <= 36
  const quality = forceVector
    ? Math.min(95, 48 + Math.round(nCategories * 1.1))
    : nCategories > 80
      ? 26
      : nCategories > 40
        ? 40
        : 58
  const generated = api.generate(nCategories, checkColor, forceVector, quality)
  const sorted = api.diffSort(generated.slice())
  const rgb = sorted.map(toRgbTuple)
  const fillCss = rgb.map(([r, g, b]) => `rgb(${r},${g},${b})`)
  return { fillCss, rgb }
}
