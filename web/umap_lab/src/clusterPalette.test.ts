import { describe, expect, it } from "vitest"
import { buildClusterPalette } from "./clusterPalette"

describe("buildClusterPalette", () => {
  it("returns one color for a single cluster", () => {
    const p = buildClusterPalette(1)
    expect(p.fillCss).toHaveLength(1)
    expect(p.rgb).toHaveLength(1)
  })

  it("returns distinct iWantHue colors for several clusters", () => {
    const p = buildClusterPalette(6)
    expect(p.rgb).toHaveLength(6)
    const keys = new Set(p.rgb.map(([r, g, b]) => `${r},${g},${b}`))
    expect(keys.size).toBe(6)
  })
})
