export function formatGrnFoyerCacheLine(cache: string | undefined): string {
  switch (cache) {
    case "active":
      return "GRN foyer cache: on";
    case "skipped_small":
      return "GRN foyer cache: off (below size threshold)";
    case "skipped_large":
      return "GRN foyer cache: off (over configured cap)";
    default:
      return cache ? `GRN foyer cache: ${cache}` : "GRN foyer cache: unknown";
  }
}

export function formatSpatialLigandLine(
  mode: string | undefined,
  ligandGridFactor: number | null | undefined,
): string {
  if (mode === "grid_approx") {
    const g =
      ligandGridFactor != null && Number.isFinite(ligandGridFactor)
        ? ` (grid factor ${ligandGridFactor})`
        : "";
    return `Spatial ligand model during perturb: grid approximation${g}`;
  }
  if (mode === "exact_pairwise") {
    return "Spatial ligand model during perturb: exact pairwise";
  }
  return mode
    ? `Spatial ligand model during perturb: ${mode}`
    : "Spatial ligand model during perturb: unknown";
}

export function grnFoyerCacheStatusFragment(cache: string | undefined): string {
  switch (cache) {
    case "active":
      return "foyer GRN cache: on";
    case "skipped_small":
      return "foyer GRN cache: off (small)";
    case "skipped_large":
      return "foyer GRN cache: off (cap)";
    default:
      return cache ? `foyer GRN: ${cache}` : "foyer GRN: ?";
  }
}

export function spatialLigandStatusFragment(
  mode: string | undefined,
  ligandGridFactor: number | null | undefined,
): string {
  if (mode === "grid_approx") {
    const g =
      ligandGridFactor != null && Number.isFinite(ligandGridFactor)
        ? ` ×${ligandGridFactor}`
        : "";
    return `perturb ligands: grid${g}`;
  }
  if (mode === "exact_pairwise") {
    return "perturb ligands: pairwise";
  }
  return mode ? `perturb ligands: ${mode}` : "perturb ligands: ?";
}

export function perturbReadyTooltip(sm: {
  grn_foyer_cache?: string;
  spatial_ligand_mode?: string;
  ligand_grid_factor?: number | null;
} | null | undefined): string {
  if (!sm) return "Perturbation runtime ready";
  return [
    "Perturbation runtime ready",
    formatGrnFoyerCacheLine(sm.grn_foyer_cache),
    formatSpatialLigandLine(sm.spatial_ligand_mode, sm.ligand_grid_factor),
  ].join(" — ");
}
