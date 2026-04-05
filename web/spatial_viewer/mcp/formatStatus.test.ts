import { describe, expect, it } from "vitest";

import { formatStatus, type MetaSnapshot } from "./mcpHttp.js";

function baseMeta(over: Partial<MetaSnapshot> = {}): MetaSnapshot {
  return {
    n_obs: 100,
    n_vars: 500,
    dataset_ready: true,
    perturb_ready: false,
    perturb_loading: false,
    ...over,
  };
}

describe("formatStatus", () => {
  it("shows not loaded when dataset_ready is false", () => {
    const m = baseMeta({ dataset_ready: false, perturb_ready: false });
    expect(formatStatus(m)).toContain("Dataset: not loaded yet");
    expect(formatStatus(m)).toContain("Perturbation: not configured");
  });

  it("shows ready dataset line and optional fields", () => {
    const m = baseMeta({
      adata_path: "/tmp/d.h5ad",
      cluster_annot: "leiden",
      cell_type_categories: ["A", "B"],
      network_loaded: true,
      network_species: "human",
      betadata_dir: "/b",
      betadata_row_id: "row1",
    });
    const s = formatStatus(m);
    expect(s).toContain("Dataset: ready — 100 cells, 500 genes (/tmp/d.h5ad)");
    expect(s).toContain("clusters: leiden");
    expect(s).toContain("cell types: A, B");
    expect(s).toContain("GRN: human");
    expect(s).toContain("betadata: /b");
    expect(s).toContain("betadata row id: row1");
    expect(s).toContain("Perturbation: not configured");
  });

  it("shows perturb loading with permille percent", () => {
    const m = baseMeta({
      perturb_loading: true,
      perturb_progress_permille: 125,
      perturb_progress_label: "loading wl",
    });
    expect(formatStatus(m)).toContain("Perturbation: loading 12.5% — loading wl");
  });

  it("falls back to perturb_progress_percent when no permille", () => {
    const m = baseMeta({
      perturb_loading: true,
      perturb_progress_percent: 33,
    });
    expect(formatStatus(m)).toContain("Perturbation: loading 33%");
  });

  it("shows perturb error before ready branch", () => {
    const m = baseMeta({ perturb_error: "missing parquet", perturb_ready: false });
    expect(formatStatus(m)).toContain("Perturbation: ERROR — missing parquet");
  });

  it("shows perturb ready", () => {
    const m = baseMeta({ perturb_ready: true });
    expect(formatStatus(m)).toContain("Perturbation: ready");
  });

  it("shows foyer cache and spatial ligand lines when spatial_model is present", () => {
    const m = baseMeta({
      perturb_ready: true,
      spatial_model: {
        grn_foyer_cache: "active",
        spatial_ligand_mode: "grid_approx",
        ligand_grid_factor: 2.5,
      },
    });
    const s = formatStatus(m);
    expect(s).toContain("Perturbation: ready");
    expect(s).toContain("GRN foyer cache: on");
    expect(s).toContain("grid approximation (grid factor 2.5)");
  });
});
