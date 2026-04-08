export interface Meta {
  n_obs: number;
  n_vars: number;
  spatial_obsm_key: string;
  layer: string;
  cluster_annot: string;
  bounds: { min_x: number; max_x: number; min_y: number; max_y: number };
  umap_obsm_key?: string | null;
  umap_bounds?: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
  } | null;
  cell_type_column?: string | null;
  cell_type_categories?: string[];
  network_loaded?: boolean;
  network_species?: string | null;
  betadata_row_id?: string | null;
  perturb_ready?: boolean;
  perturb_loading?: boolean;
  perturb_error?: string | null;
  perturb_progress_percent?: number | null;
  perturb_progress_permille?: number | null;
  perturb_progress_label?: string | null;
  perturb_betadata_phase?: "reading" | "expanding" | null;
  perturb_betadata_done?: number | null;
  perturb_betadata_total?: number | null;
  adata_path: string;
  betadata_dir: string;
  network_dir?: string | null;
  run_toml?: string | null;
  perturb_overlay?: string | null;
  dataset_ready?: boolean;
  spatial_model?: {
    weighted_ligand_scale_factor: number;
    spatial_radius: number;
    contact_distance: number;
    spatial_dim: number;
    received_ligand_n_channels: number;
    received_ligand_columns_sample: string[];
    tfl_ligand_n_channels: number;
    grn_foyer_cache?: string;
    spatial_ligand_mode?: string;
    ligand_grid_factor?: number | null;
  } | null;
}

export interface SessionConfigureResponse {
  ok: boolean;
  message: string;
  meta: Meta;
}

export interface SessionConfigurePayload {
  adata_path: string;
  layer: string;
  cluster_annot: string;
  network_dir: string;
  run_toml: string;
  perturb_overlay: string;
}

export interface CollectedInteractionRow {
  interaction: string;
  gene: string;
  beta: number;
  interaction_type: string;
}

export interface CollectInteractionsApiResponse {
  interactions: CollectedInteractionRow[];
  n_reported: number;
  n_total: number;
  capped: boolean;
}

export interface PairLrRow {
  target_gene: string;
  interaction: string;
  beta_cell_a: number;
  beta_cell_b: number;
  score: number;
}

export interface PairLrApiResponse {
  cell_a: number;
  cell_b: number;
  betadata_row_id?: string | null;
  rows: PairLrRow[];
  n_genes_scanned: number;
}

export interface UmapFieldResponse {
  nx: number;
  ny: number;
  grid_x: number[];
  grid_y: number[];
  u: number[];
  v: number[];
  cell_u?: number[];
  cell_v?: number[];
  svg_export_path?: string | null;
}

export interface UmapSignatureFieldResponse extends UmapFieldResponse {
  signature_per_cell: number[];
}

export interface CellContextResponse {
  focus_gene: string;
  cell_index: number;
  modulators: {
    regulators: string[];
    ligands: string[];
    receptors: string[];
    tfl_ligands: string[];
    tfl_regulators: string[];
    lr_pairs: string[];
    tfl_pairs: string[];
  };
  neighbors: {
    index: number;
    distance_sq: number;
    distance?: number;
    cell_type?: string | null;
    max_support_score?: number | null;
    lr_edges: {
      ligand: string;
      receptor: string;
      lig_expr_sender: number;
      rec_expr_neighbor: number;
      support_score: number;
      linked_tf?: string;
      linked_tf_expr?: number;
    }[];
  }[];
  sender_regulator_exprs: { gene: string; expr: number }[];
  sender_ligand_exprs: { gene: string; expr: number }[];
  neighbor_query?: string | null;
  radius_used?: number | null;
  neighbors_in_query?: number | null;
}

export interface InteractionLineDatum {
  sourcePosition: [number, number, number];
  targetPosition: [number, number, number];
  color?: [number, number, number, number];
}

export interface QuiverSegDatum extends InteractionLineDatum {
  width?: number;
}
