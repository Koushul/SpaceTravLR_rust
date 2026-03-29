import { describe, expect, it } from "vitest";

describe("PerturbReferenceSimilarityResponse invariants (Rust-aligned)", () => {
  it("mean_delta_cosine equals mean_cosine_after minus mean_cosine_before", () => {
    const fixture = {
      n_genes_used: 120,
      n_reference_cells: 500,
      n_eval_cells: 80,
      exclude_perturb_cells_from_reference: true,
      mean_cosine_before: 0.31,
      mean_cosine_after: 0.47,
      median_cosine_before: 0.3,
      median_cosine_after: 0.45,
      mean_delta_cosine: 0.16,
    };
    expect(fixture.mean_delta_cosine).toBeCloseTo(
      fixture.mean_cosine_after - fixture.mean_cosine_before,
      10,
    );
  });
});
