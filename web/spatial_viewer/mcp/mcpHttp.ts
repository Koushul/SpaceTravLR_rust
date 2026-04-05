import {
  formatGrnFoyerCacheLine,
  formatSpatialLigandLine,
} from "../src/perturbRuntimeLabels.js";

export interface SpatialModelMetaSnapshot {
  grn_foyer_cache?: string;
  spatial_ligand_mode?: string;
  ligand_grid_factor?: number | null;
}

export interface MetaSnapshot {
  n_obs: number;
  n_vars: number;
  dataset_ready: boolean;
  perturb_ready: boolean;
  perturb_loading: boolean;
  perturb_progress_percent?: number;
  perturb_progress_permille?: number;
  perturb_progress_label?: string;
  perturb_error?: string;
  cell_type_categories?: string[];
  cell_type_column?: string;
  cluster_annot?: string;
  adata_path?: string;
  betadata_dir?: string;
  network_loaded?: boolean;
  network_species?: string;
  betadata_row_id?: string;
  spatial_model?: SpatialModelMetaSnapshot | null;
  [k: string]: unknown;
}

export function connectOrigin(apiBase: string): string {
  try {
    return new URL(apiBase).origin;
  } catch {
    return "http://127.0.0.1:8080";
  }
}

export function makeConnectDomainList(
  defaultApiBase: string,
  connectOriginsCsv: string | undefined,
): string[] {
  const base = connectOrigin(defaultApiBase);
  const extra = connectOriginsCsv
    ?.split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (!extra?.length) return [base];
  return [...new Set([base, ...extra])];
}

export function normalizeApiBase(input: string | undefined, fallback: string): string {
  return (input?.trim() || fallback).replace(/\/$/, "");
}

export function parseDelimitedGenes(s: string): string[] {
  return String(s)
    .split(/[,;\s]+/)
    .map((x) => x.trim())
    .filter(Boolean);
}

export type PerturbScopeJson =
  | { type: "all" }
  | { type: "cell_type_name"; name: string }
  | { type: "cluster"; cluster_id: number };

export function buildPerturbScopeBody(
  scope: string | undefined,
  cell_type_label: string | undefined,
  cluster_id: number | undefined,
): PerturbScopeJson {
  const s = scope ?? "all";
  if (s === "cell_type" && cell_type_label != null && String(cell_type_label).trim() !== "") {
    return { type: "cell_type_name", name: String(cell_type_label).trim() };
  }
  if (s === "cluster" && cluster_id != null) {
    return { type: "cluster", cluster_id };
  }
  return { type: "all" };
}

export function buildReferenceCentroidScopeBody(
  reference_scope: string | undefined,
  reference_cell_type_label: string | undefined,
  reference_cluster_id: number | undefined,
): PerturbScopeJson {
  const refS = reference_scope ?? "cell_type";
  if (
    refS === "cell_type" &&
    reference_cell_type_label != null &&
    String(reference_cell_type_label).trim() !== ""
  ) {
    return { type: "cell_type_name", name: String(reference_cell_type_label).trim() };
  }
  if (refS === "cluster" && typeof reference_cluster_id === "number") {
    return { type: "cluster", cluster_id: reference_cluster_id };
  }
  return { type: "all" };
}

export function formatStatus(m: MetaSnapshot): string {
  const lines: string[] = [];
  if (!m.dataset_ready) {
    lines.push("Dataset: not loaded yet");
  } else {
    lines.push(
      `Dataset: ready — ${m.n_obs} cells, ${m.n_vars} genes` +
        (m.adata_path ? ` (${m.adata_path})` : ""),
    );
    if (m.cluster_annot) lines.push(`  clusters: ${m.cluster_annot}`);
    if (m.cell_type_categories?.length)
      lines.push(`  cell types: ${m.cell_type_categories.join(", ")}`);
    if (m.network_loaded) lines.push(`  GRN: ${m.network_species ?? "loaded"}`);
    if (m.betadata_dir) lines.push(`  betadata: ${m.betadata_dir}`);
    if (m.betadata_row_id) lines.push(`  betadata row id: ${m.betadata_row_id}`);
  }

  if (m.perturb_loading) {
    const pm = m.perturb_progress_permille;
    const pctStr =
      pm != null && Number.isFinite(pm)
        ? `${(pm / 10).toFixed(1)}%`
        : `${m.perturb_progress_percent ?? 0}%`;
    const label = m.perturb_progress_label ?? "";
    lines.push(`Perturbation: loading ${pctStr} — ${label}`);
  } else if (m.perturb_error) {
    lines.push(`Perturbation: ERROR — ${m.perturb_error}`);
  } else if (m.perturb_ready) {
    lines.push("Perturbation: ready");
    const sm = m.spatial_model;
    if (sm?.grn_foyer_cache) {
      lines.push(`  ${formatGrnFoyerCacheLine(sm.grn_foyer_cache)}`);
    }
    if (sm?.spatial_ligand_mode) {
      lines.push(
        `  ${formatSpatialLigandLine(sm.spatial_ligand_mode, sm.ligand_grid_factor)}`,
      );
    }
  } else {
    lines.push("Perturbation: not configured (no --run-toml)");
  }
  return lines.join("\n");
}

export async function fetchMetaWith(
  fetchImpl: typeof fetch,
  api: string,
): Promise<MetaSnapshot> {
  const res = await fetchImpl(`${api}/api/meta`);
  if (!res.ok) throw new Error(`/api/meta ${res.status}: ${await res.text()}`);
  return (await res.json()) as MetaSnapshot;
}
