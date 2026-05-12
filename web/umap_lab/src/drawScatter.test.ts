import { describe, expect, it } from "vitest"
import { axisBounds } from "./drawScatter"

describe("axisBounds", () => {
  it("pads equal values", () => {
    const b = axisBounds([3, 3, 3])
    expect(b.max).toBeGreaterThan(b.min)
  })

  it("includes extrema", () => {
    const b = axisBounds([0, 10])
    expect(b.min).toBeLessThanOrEqual(0)
    expect(b.max).toBeGreaterThanOrEqual(10)
  })
})
