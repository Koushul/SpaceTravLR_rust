use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anndata::AnnData;
use anndata::AnnDataOp;
use anndata_hdf5::H5;
use anyhow::{Context, anyhow};
use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, get_service, post};
use clap::Parser;
use ndarray::{Array2, Axis};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use space_trav_lr_rust::adata_query::{
    cell_expression_map, cell_type_encoding, clusters_as_u32_le_bytes,
    expression_matrix_genes_subset, expression_profiles_for_cells, f32_vec_to_le_bytes,
    gene_expression_f32, genes_with_prefix, obs_names, open_adata, spatial_obsm_key_used,
    spatial_xy, spatial_xy_f32_interleaved, try_umap_xy, u16_vec_to_le_bytes, var_names,
};
use space_trav_lr_rust::betadata::{
    BetadataCollectAggregate, BetadataUiProgress, CollectedInteraction, GeneMatrix, PairLrBetaRow,
    TopBetaCoefficient, betadata_collect_interactions_parallel,
    betadata_feather_modulator_beta_means_for_cells, betadata_feather_per_cell_column,
    betadata_feather_plottable_columns, betadata_feather_row_id_column,
    betadata_feather_top_coefficients_for_selection, betadata_pair_lr_parallel,
};
use space_trav_lr_rust::betadata_view::{betadata_feather_path, list_betadata_target_genes};
use space_trav_lr_rust::config::{SpaceshipConfig, expand_user_path, normalize_ui_path};
use space_trav_lr_rust::foyer_perturb_cache::{
    self, FoyerCacheLimits, FoyerPerturbCaches, PerturbCacheKey, UmapGridBlob,
};
use space_trav_lr_rust::ligand::{calculate_weighted_ligands, calculate_weighted_ligands_grid};
use space_trav_lr_rust::network::{GeneNetwork, infer_species};
use space_trav_lr_rust::perturb::{
    PerturbConfig, PerturbResult, PerturbTarget, PerturbTimings, compute_splash_all_progress,
    perturb_with_targets,
};
use space_trav_lr_rust::perturb_mode::PerturbRuntime;
use space_trav_lr_rust::transition_umap::{
    SignatureUmapParams, TransitionGrid, TransitionUmapParams, compute_signature_umap_grid,
    compute_umap_transition_grid, signature_sum_per_cell,
};
use tokio::sync::RwLock;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

const DEFAULT_GENE_SEARCH_LIMIT: usize = 500;
const FULL_GENE_LIST_THRESHOLD: usize = 50_000;
const MAX_TOP_BETA_INDICES: usize = 250_000;
const MAX_UMAP_TRANSITION_CELLS: usize = 40_000;
const MAX_SIGNATURE_GENES: usize = 128;
const MAX_LR_NEIGHBORS_RADIUS: usize = 500;
const MAX_RECEIVED_LIGAND_GENES: usize = 96;

fn foyer_cache_limits_from_cli(cli: &Cli) -> FoyerCacheLimits {
    const MIB: usize = 1024 * 1024;
    let mut l = FoyerCacheLimits::from_env();
    if let Some(v) = cli.foyer_grn_memory_mb {
        l.grn_memory = (v as usize).saturating_mul(MIB);
    }
    if let Some(v) = cli.foyer_grn_disk_mb {
        l.grn_disk = (v as usize).saturating_mul(MIB);
    }
    if let Some(v) = cli.foyer_grid_memory_mb {
        l.grid_memory = (v as usize).saturating_mul(MIB);
    }
    if let Some(v) = cli.foyer_grid_disk_mb {
        l.grid_disk = (v as usize).saturating_mul(MIB);
    }
    l
}

fn deserialize_usize_flex<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    struct U;
    impl<'de> Visitor<'de> for U {
        type Value = usize;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a non-negative whole number (JSON integer or float like 42.0)")
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<usize, E> {
            usize::try_from(v).map_err(E::custom)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<usize, E> {
            if v < 0 {
                return Err(E::custom("negative index"));
            }
            Ok(v as usize)
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<usize, E> {
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
                return Err(E::custom(
                    "index must be a finite non-negative whole number (integers or e.g. 12.0)",
                ));
            }
            Ok(v as usize)
        }
    }
    deserializer.deserialize_any(U)
}

#[derive(Parser, Debug)]
#[command(name = "spatial_viewer")]
struct Cli {
    /// AnnData path; omit to start the UI only and load via **Dataset paths**.
    #[arg(long)]
    h5ad: Option<PathBuf>,
    #[arg(long, default_value = "imputed_count")]
    layer: String,
    #[arg(long, default_value = "cell_type")]
    cluster_annot: String,
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Must contain `index.html` from `npm run build` in `web/spatial_viewer` (not `build:mcp` alone).
    #[arg(long, default_value = "web/spatial_viewer/dist")]
    static_dir: PathBuf,
    /// Directory containing `{mouse|human}_network.parquet` (optional; otherwise same search as training).
    #[arg(long)]
    network_dir: Option<PathBuf>,
    /// `spacetravlr_run_repro.toml`: enables perturbation + betadata from `[execution].output_dir` (or the TOML’s directory if that field is empty). The loaded AnnData path is taken from `data.adata_path` in the TOML (overrides `--h5ad` when both are set). You may pass only `--run-toml` with no `--h5ad`.
    #[arg(long)]
    run_toml: Option<PathBuf>,
    /// Allow permissive CORS on `/api/*` (needed for MCP App iframe → local API). Also set `SPATIAL_VIEWER_ALLOW_CORS=1`.
    #[arg(long)]
    allow_cors: bool,
    /// Directory for foyer hybrid disk cache (GRN perturbation + UMAP grid). Default: system temp `spacetravlr_foyer_perturb/`.
    #[arg(long)]
    perturb_cache_dir: Option<PathBuf>,
    /// Override foyer GRN in-memory tier size (MiB). Default: env `SPACETRAVLR_FOYER_GRN_MEMORY_MB` or 256.
    #[arg(long)]
    foyer_grn_memory_mb: Option<u64>,
    /// Override foyer GRN on-disk capacity (MiB). Default: env `SPACETRAVLR_FOYER_GRN_DISK_MB` or 512.
    #[arg(long)]
    foyer_grn_disk_mb: Option<u64>,
    /// Override foyer UMAP grid in-memory tier (MiB). Default: env `SPACETRAVLR_FOYER_GRID_MEMORY_MB` or 64.
    #[arg(long)]
    foyer_grid_memory_mb: Option<u64>,
    /// Override foyer UMAP grid on-disk capacity (MiB). Default: env `SPACETRAVLR_FOYER_GRID_DISK_MB` or 128.
    #[arg(long)]
    foyer_grid_disk_mb: Option<u64>,
}

#[derive(Clone)]
struct AppDataset {
    adata: Arc<RwLock<AnnData<H5>>>,
    adata_path: PathBuf,
    layer: String,
    cluster_annot: String,
    betadata_dir: Option<PathBuf>,
    network_dir: Option<PathBuf>,
    run_toml: Option<PathBuf>,
    obs_names: Arc<Vec<String>>,
    clusters: Arc<Vec<usize>>,
    /// Feather `Cluster` join strings (category names, not codes).
    betadata_cluster_keys: Arc<Vec<String>>,
    spatial_key: String,
    spatial_f32: Arc<Vec<f32>>,
    umap_key: Option<String>,
    umap_f32: Option<Arc<Vec<f32>>>,
    umap_bounds: Option<MetaBounds>,
    clusters_bin: Arc<Vec<u8>>,
    n_vars: usize,
    meta_bounds: MetaBounds,
    cell_type_column: Option<String>,
    cell_type_categories: Arc<Vec<String>>,
    cell_type_codes_bin: Option<Arc<Vec<u8>>>,
    var_names: Arc<Vec<String>>,
    grn: Option<Arc<GeneNetwork>>,
    betadata_row_id: Option<String>,
    perturb_runtime: Option<Arc<PerturbRuntime>>,
    perturb_load_error: Option<String>,
}

fn spatial_model_meta_from_runtime(
    rt: &PerturbRuntime,
    n_obs: usize,
    n_vars: usize,
) -> SpatialModelMeta {
    let sc = &rt.cfg.spatial;
    let sample: Vec<String> = rt
        .rw_ligands_init
        .col_names
        .iter()
        .take(80)
        .cloned()
        .collect();
    let est = perturb_matrix_payload_est_bytes(n_obs, n_vars);
    let grn_foyer_cache = if should_use_foyer_perturb_cache(est) {
        "active".to_string()
    } else if est > MAX_GRN_PERTURB_CACHE_BYTES {
        "skipped_large".to_string()
    } else {
        "skipped_small".to_string()
    };
    let (spatial_ligand_mode, ligand_grid_factor) = match rt.perturb_cfg.ligand_grid_factor {
        Some(gf) if gf.is_finite() && gf > 0.0 => ("grid_approx".to_string(), Some(gf)),
        _ => ("exact_pairwise".to_string(), None),
    };
    SpatialModelMeta {
        weighted_ligand_scale_factor: sc.weighted_ligand_scale_factor,
        spatial_radius: sc.radius,
        contact_distance: sc.contact_distance,
        spatial_dim: sc.spatial_dim,
        received_ligand_n_channels: rt.rw_ligands_init.col_names.len(),
        received_ligand_columns_sample: sample,
        tfl_ligand_n_channels: rt.rw_tfligands_init.col_names.len(),
        grn_foyer_cache,
        spatial_ligand_mode,
        ligand_grid_factor,
    }
}

fn aggregate_received_ligand_rows(mat: &Array2<f64>, mode: &str) -> anyhow::Result<Vec<f32>> {
    let n = mat.nrows();
    let k = mat.ncols();
    anyhow::ensure!(k > 0, "no ligand columns");
    let mut out = Vec::with_capacity(n);
    match mode.to_ascii_lowercase().as_str() {
        "sum" => {
            for row in mat.axis_iter(Axis(0)) {
                out.push(row.sum() as f32);
            }
        }
        "max" => {
            for row in mat.axis_iter(Axis(0)) {
                let v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                out.push(v as f32);
            }
        }
        "mean" => {
            let inv = 1.0 / k as f64;
            for row in mat.axis_iter(Axis(0)) {
                out.push((row.sum() * inv) as f32);
            }
        }
        _ => anyhow::bail!("aggregate must be sum, max, or mean"),
    }
    Ok(out)
}

fn compute_received_ligand_from_adata(
    adata: &AnnData<H5>,
    layer: &str,
    spatial_f32: &[f32],
    var_names: &[String],
    genes: &[String],
    radius: f64,
    scale: f64,
    use_grid: bool,
    grid_factor: f64,
    aggregate: &str,
) -> anyhow::Result<Vec<f32>> {
    let n = spatial_f32.len() / 2;
    let mut xy = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        xy[[i, 0]] = spatial_f32[i * 2] as f64;
        xy[[i, 1]] = spatial_f32[i * 2 + 1] as f64;
    }
    let (lig_mat, _) = expression_matrix_genes_subset(adata, layer, genes, var_names)?;
    anyhow::ensure!(
        lig_mat.nrows() == n,
        "expression rows {} != spatial n {}",
        lig_mat.nrows(),
        n
    );
    let received = if use_grid {
        calculate_weighted_ligands_grid(&xy, &lig_mat, radius, scale, grid_factor)
    } else {
        calculate_weighted_ligands(&xy, &lig_mat, radius, scale)
    };
    aggregate_received_ligand_rows(&received, aggregate)
}

fn runtime_received_ligand_column(
    rt: &PerturbRuntime,
    gene: &str,
    matrix: &str,
) -> anyhow::Result<Vec<f32>> {
    let gm = if matrix.eq_ignore_ascii_case("tfl") {
        &rt.rw_tfligands_init
    } else {
        &rt.rw_ligands_init
    };
    let view = gm.col(gene).ok_or_else(|| {
        anyhow::anyhow!(
            "column {:?} not found in {} received-ligand matrix ({} channels)",
            gene,
            if matrix.eq_ignore_ascii_case("tfl") {
                "TFL"
            } else {
                "LR"
            },
            gm.col_names.len()
        )
    })?;
    Ok(view.iter().copied().collect())
}

fn cell_type_label_for_obs(ds: &AppDataset, obs_i: usize) -> Option<String> {
    let bin = ds.cell_type_codes_bin.as_ref()?;
    let o = obs_i.checked_mul(2)?;
    if o + 2 > bin.len() {
        return None;
    }
    let code = u16::from_le_bytes([bin[o], bin[o + 1]]);
    if code == u16::MAX {
        return Some("(unknown)".to_string());
    }
    Some(
        ds.cell_type_categories
            .get(code as usize)
            .cloned()
            .unwrap_or_else(|| format!("(code {})", code)),
    )
}

#[derive(Clone)]
struct AppState {
    dataset: Option<AppDataset>,
    default_layer: String,
    default_cluster_annot: String,
    default_network_dir: Option<PathBuf>,
    default_run_toml: Option<PathBuf>,
    perturb_bg_gen: Arc<AtomicU64>,
    perturb_bg_in_flight: Arc<AtomicBool>,
    /// Another `spawn_perturb_background_load` ran while a load was active; run one more pass after it finishes.
    perturb_bg_pending: Arc<AtomicBool>,
    perturb_load_progress_permille: Arc<AtomicU32>,
    perturb_job_progress_permille: Arc<AtomicU32>,
    perturb_job_active: Arc<AtomicBool>,
    /// Set true from POST /api/cancel; checked between perturb propagation iterations.
    perturb_job_cancel: Arc<AtomicBool>,
    /// After cancel, hide “perturbation loading” until a new load starts (blocking work may still finish).
    perturb_suppress_bg_loading_ui: Arc<AtomicBool>,
    perturb_progress_message: Arc<Mutex<String>>,
    perturb_betadata_ui: Arc<BetadataUiProgress>,
    /// Splash network job: UI polls `GET /api/perturb/splash_progress` while POST splash_network runs.
    splash_job_progress_permille: Arc<AtomicU32>,
    splash_job_active: Arc<AtomicBool>,
    foyer_caches: Arc<FoyerPerturbCaches>,
    dataset_cache_epoch: Arc<AtomicU64>,
}

#[derive(Clone, Serialize)]
struct MetaBounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

/// Training spatial + received-ligand channel metadata (present when `perturb_ready`).
#[derive(Clone, Serialize)]
struct SpatialModelMeta {
    weighted_ligand_scale_factor: f64,
    spatial_radius: f64,
    contact_distance: f64,
    spatial_dim: usize,
    received_ligand_n_channels: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    received_ligand_columns_sample: Vec<String>,
    tfl_ligand_n_channels: usize,
    /// `active` | `skipped_small` | `skipped_large` — whether hybrid GRN δ cache is used for this matrix size.
    grn_foyer_cache: String,
    /// `grid_approx` | `exact_pairwise` — spatial received-ligand recomputation during GRN perturbation.
    spatial_ligand_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ligand_grid_factor: Option<f64>,
}

#[derive(Serialize)]
struct MetaJson {
    n_obs: usize,
    n_vars: usize,
    spatial_obsm_key: String,
    layer: String,
    cluster_annot: String,
    bounds: MetaBounds,
    #[serde(skip_serializing_if = "Option::is_none")]
    umap_obsm_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    umap_bounds: Option<MetaBounds>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cell_type_column: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cell_type_categories: Vec<String>,
    network_loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    network_species: Option<String>,
    /// `Cluster` = seed-only (β per cluster); `CellID` = spatial CNN (per-cell β). Probed from first `*_betadata.feather`.
    #[serde(skip_serializing_if = "Option::is_none")]
    betadata_row_id: Option<String>,
    perturb_ready: bool,
    #[serde(default)]
    perturb_loading: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perturb_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perturb_progress_percent: Option<u8>,
    /// Raw 0–1000 permille when a load or perturb job is active (finer than `perturb_progress_percent`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perturb_progress_permille: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perturb_progress_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perturb_betadata_phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perturb_betadata_done: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perturb_betadata_total: Option<u32>,
    adata_path: String,
    /// Training run directory when `--run-toml` was passed (same dir as `*_betadata.feather`); empty otherwise.
    betadata_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    network_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_toml: Option<String>,
    #[serde(default)]
    dataset_ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    spatial_model: Option<SpatialModelMeta>,
}

#[derive(serde::Deserialize)]
struct GenesQuery {
    prefix: Option<String>,
    limit: Option<usize>,
}

#[derive(serde::Deserialize)]
struct GeneExprQuery {
    gene: String,
}

#[derive(serde::Deserialize)]
struct BetadataColumnsQuery {
    gene: String,
}

#[derive(serde::Deserialize)]
struct BetadataValuesQuery {
    gene: String,
    column: String,
}

#[derive(Deserialize)]
struct TopBetasBody {
    gene: String,
    #[serde(default)]
    indices: Vec<usize>,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

#[derive(Deserialize)]
struct CellContextBody {
    #[serde(deserialize_with = "deserialize_usize_flex")]
    cell_index: usize,
    focus_gene: String,
    #[serde(
        default = "default_neighbor_k",
        deserialize_with = "deserialize_usize_flex"
    )]
    neighbor_k: usize,
    #[serde(default)]
    tf_ligand_cutoff: f64,
    #[serde(default = "default_expr_threshold")]
    expr_threshold: f64,
    /// `"knn"` (default) or `"radius"` (use `radius` in coordinate units).
    #[serde(default)]
    neighbor_mode: Option<String>,
    /// Euclidean radius in **same units as spatial coordinates** (e.g. pixels / µm).
    #[serde(default)]
    radius: Option<f64>,
}

#[derive(Deserialize)]
struct ReceivedLigandBody {
    /// `adata` (compute from expression + spatial) or `runtime` / `model` (slice training `GeneMatrix`).
    #[serde(default = "default_rl_source")]
    source: String,
    #[serde(default)]
    genes: Vec<String>,
    /// `lr` (default) or `tfl` when source is runtime/model.
    #[serde(default = "default_rl_matrix")]
    matrix: String,
    #[serde(default)]
    radius: Option<f64>,
    #[serde(default)]
    scale_factor: Option<f64>,
    #[serde(default = "default_rl_use_grid")]
    use_grid: bool,
    #[serde(default)]
    grid_factor: Option<f64>,
    #[serde(default = "default_rl_aggregate")]
    aggregate: String,
}

fn default_rl_source() -> String {
    "adata".to_string()
}

fn default_rl_matrix() -> String {
    "lr".to_string()
}

fn default_rl_use_grid() -> bool {
    true
}

fn default_rl_aggregate() -> String {
    "sum".to_string()
}

fn default_neighbor_k() -> usize {
    24
}

fn default_expr_threshold() -> f64 {
    1e-6
}

fn probe_betadata_row_id(dir: &std::path::Path) -> Option<String> {
    let dir_s = dir.to_string_lossy().to_string();
    let genes = list_betadata_target_genes(&dir_s).ok()?;
    let g = genes.first()?;
    let p = betadata_feather_path(&dir_s, g);
    if !p.is_file() {
        return None;
    }
    betadata_feather_row_id_column(p.to_string_lossy().as_ref())
        .ok()
        .flatten()
}

#[derive(Serialize)]
struct GeneExprEntry {
    gene: String,
    expr: f64,
}

#[derive(Serialize)]
struct LrEdgeJson {
    ligand: String,
    receptor: String,
    lig_expr_sender: f64,
    rec_expr_neighbor: f64,
    /// √(L·R) for UI coloring when both > 0.
    support_score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    linked_tf: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    linked_tf_expr: Option<f64>,
}

#[derive(Serialize)]
struct NeighborContextJson {
    index: usize,
    distance_sq: f64,
    /// Euclidean distance (√distance_sq) in spatial coordinate units.
    distance: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cell_type: Option<String>,
    lr_edges: Vec<LrEdgeJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_support_score: Option<f64>,
}

#[derive(Serialize)]
struct CellContextResponse {
    focus_gene: String,
    cell_index: usize,
    modulators: space_trav_lr_rust::network::Modulators,
    neighbors: Vec<NeighborContextJson>,
    sender_regulator_exprs: Vec<GeneExprEntry>,
    sender_ligand_exprs: Vec<GeneExprEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    neighbor_query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    radius_used: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    neighbors_in_query: Option<usize>,
}

fn spatial_k_nearest(
    spatial_f32: &[f32],
    n: usize,
    cell_idx: usize,
    k: usize,
) -> Vec<(usize, f64)> {
    if n == 0 || cell_idx >= n {
        return vec![];
    }
    let x0 = spatial_f32[cell_idx * 2] as f64;
    let y0 = spatial_f32[cell_idx * 2 + 1] as f64;
    let mut dists: Vec<(usize, f64)> = Vec::with_capacity(n.saturating_sub(1));
    for j in 0..n {
        if j == cell_idx {
            continue;
        }
        let x = spatial_f32[j * 2] as f64 - x0;
        let y = spatial_f32[j * 2 + 1] as f64 - y0;
        dists.push((j, x * x + y * y));
    }
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k.min(dists.len()));
    dists
}

fn spatial_within_radius(
    spatial_f32: &[f32],
    n: usize,
    cell_idx: usize,
    radius: f64,
    max_neighbors: usize,
) -> Vec<(usize, f64)> {
    if n == 0 || cell_idx >= n || !radius.is_finite() || radius <= 0.0 {
        return vec![];
    }
    let r2 = radius * radius;
    let x0 = spatial_f32[cell_idx * 2] as f64;
    let y0 = spatial_f32[cell_idx * 2 + 1] as f64;
    let mut dists: Vec<(usize, f64)> = Vec::new();
    for j in 0..n {
        if j == cell_idx {
            continue;
        }
        let x = spatial_f32[j * 2] as f64 - x0;
        let y = spatial_f32[j * 2 + 1] as f64 - y0;
        let d2 = x * x + y * y;
        if d2 <= r2 {
            dists.push((j, d2));
        }
    }
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(max_neighbors.min(dists.len()));
    dists
}

fn parse_lr_pair(s: &str) -> Option<(String, String)> {
    let (a, b) = s.split_once('$')?;
    let a = a.trim();
    let b = b.trim();
    if a.is_empty() || b.is_empty() {
        return None;
    }
    Some((a.to_string(), b.to_string()))
}

fn parse_tfl_pair(s: &str) -> Option<(String, String)> {
    let (lig, tf) = s.split_once('#')?;
    let lig = lig.trim();
    let tf = tf.trim();
    if lig.is_empty() || tf.is_empty() {
        return None;
    }
    Some((lig.to_string(), tf.to_string()))
}

fn default_top_k() -> usize {
    25
}

fn binary_response(bytes: Vec<u8>) -> Response {
    (
        [(header::CONTENT_TYPE, "application/octet-stream")],
        Bytes::from(bytes),
    )
        .into_response()
}

fn perturb_export_feather_filename(gene: &str) -> String {
    let mut s: String = gene
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        s = "perturb".into();
    }
    s.truncate(64);
    format!("{s}_perturb_simulated_expr.feather")
}

fn feather_download_response(bytes: Vec<u8>, filename: &str) -> Response {
    let safe: String = filename
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let cd = format!("attachment; filename=\"{safe}\"");
    let Ok(disposition) = HeaderValue::from_str(&cd) else {
        return binary_response(bytes);
    };
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            ),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        Bytes::from(bytes),
    )
        .into_response()
}

type SharedState = Arc<RwLock<AppState>>;

#[derive(Clone)]
struct ViewerLoadInputs {
    h5ad: PathBuf,
    layer: String,
    cluster_annot: String,
    network_dir: Option<PathBuf>,
    run_toml: Option<PathBuf>,
}

fn attach_perturb_runtime_to_dataset(
    ds: &mut AppDataset,
    pr: PerturbRuntime,
    run_toml: &Path,
) -> anyhow::Result<()> {
    let cfg_path = expand_user_path(pr.cfg.resolve_adata_path().as_str());
    let tomlp = Path::new(cfg_path.as_str());
    let vcan = ds
        .adata_path
        .canonicalize()
        .unwrap_or_else(|_| ds.adata_path.clone());
    let tcan = tomlp.canonicalize().unwrap_or_else(|_| tomlp.to_path_buf());
    if vcan != tcan {
        anyhow::bail!(
            "h5ad {} must match data.adata_path in {} ({}) when using run_toml",
            vcan.display(),
            run_toml.display(),
            tcan.display()
        );
    }
    if pr.obs_names.len() != ds.obs_names.len() {
        anyhow::bail!(
            "n_obs mismatch: TOML adata has {} cells, viewer h5ad has {}",
            pr.obs_names.len(),
            ds.obs_names.len()
        );
    }
    ds.perturb_runtime = Some(Arc::new(pr));
    tracing::info!(
        "perturbation runtime ready (loaded from {})",
        run_toml.display()
    );
    Ok(())
}

struct ClearPerturbInFlight(Arc<AtomicBool>);
impl Drop for ClearPerturbInFlight {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

struct PerturbJobGuard(Arc<AtomicBool>);
impl Drop for PerturbJobGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn permille_to_percent(p: u32) -> u8 {
    ((p.min(1000) as u64 * 100 + 500) / 1000).min(100) as u8
}

fn schedule_perturb_progress_permille_clear(p: Arc<AtomicU32>) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(700)).await;
        p.store(0, Ordering::Relaxed);
    });
}

/// Default floor for [`min_foyer_grn_payload_bytes`]: below this estimated δ payload (~`8 × n_cells × n_genes`), GRN and UMAP grid skip hybrid cache.
const DEFAULT_MIN_FOYER_GRN_PAYLOAD_BYTES: usize = 1024 * 1024;

/// Skip foyer above this estimated δ payload (too large for practical cache entries / policy).
const MAX_GRN_PERTURB_CACHE_BYTES: usize = 512 * 1024 * 1024;

#[inline]
fn perturb_matrix_payload_est_bytes(n_cells: usize, n_genes: usize) -> usize {
    n_cells.saturating_mul(n_genes).saturating_mul(8)
}

/// Minimum serialized δ payload (bytes) before using foyer for GRN / UMAP grid.  
/// Env: `SPACETRAVLR_FOYER_GRN_MIN_PAYLOAD_MB` (mebibytes, default 1). Set `0` to always use foyer up to [`MAX_GRN_PERTURB_CACHE_BYTES`].
fn min_foyer_grn_payload_bytes() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        const MIB: usize = 1024 * 1024;
        match std::env::var("SPACETRAVLR_FOYER_GRN_MIN_PAYLOAD_MB") {
            Ok(s) => s
                .parse::<usize>()
                .ok()
                .map(|mb| mb.saturating_mul(MIB))
                .unwrap_or(DEFAULT_MIN_FOYER_GRN_PAYLOAD_BYTES),
            Err(_) => DEFAULT_MIN_FOYER_GRN_PAYLOAD_BYTES,
        }
    })
}

#[inline]
fn should_use_foyer_perturb_cache(est_payload_bytes: usize) -> bool {
    est_payload_bytes >= min_foyer_grn_payload_bytes()
        && est_payload_bytes <= MAX_GRN_PERTURB_CACHE_BYTES
}

fn transition_grid_to_blob(grid: &TransitionGrid) -> UmapGridBlob {
    let nx = grid.grid_x.len();
    let ny = grid.grid_y.len();
    UmapGridBlob {
        nx,
        ny,
        grid_x: grid.grid_x.clone(),
        grid_y: grid.grid_y.clone(),
        u: grid.vectors.iter().map(|w| w[0]).collect(),
        v: grid.vectors.iter().map(|w| w[1]).collect(),
        cell_u: grid.cell_vectors.iter().map(|w| w[0] as f32).collect(),
        cell_v: grid.cell_vectors.iter().map(|w| w[1] as f32).collect(),
    }
}

fn transition_grid_from_blob(b: UmapGridBlob) -> anyhow::Result<TransitionGrid> {
    let n = b.nx * b.ny;
    anyhow::ensure!(
        b.u.len() == n && b.v.len() == n,
        "quiver u/v length mismatch"
    );
    anyhow::ensure!(
        b.cell_u.len() == b.cell_v.len(),
        "cell quiver length mismatch"
    );
    let vectors: Vec<[f64; 2]> = b.u.iter().zip(&b.v).map(|(&u, &v)| [u, v]).collect();
    let cell_vectors: Vec<[f64; 2]> = b
        .cell_u
        .iter()
        .zip(&b.cell_v)
        .map(|(&u, &v)| [f64::from(u), f64::from(v)])
        .collect();
    Ok(TransitionGrid {
        grid_x: b.grid_x,
        grid_y: b.grid_y,
        vectors,
        cell_vectors,
    })
}

async fn direct_grn_perturb_result(
    rt: Arc<PerturbRuntime>,
    targets: Vec<PerturbTarget>,
    cfg: PerturbConfig,
    job_p: Arc<AtomicU32>,
    job_msg: Arc<Mutex<String>>,
    cancel: Arc<AtomicBool>,
) -> Result<PerturbResult, ()> {
    tokio::task::spawn_blocking(move || {
        let mut no_timings: Option<PerturbTimings> = None;
        perturb_with_targets(
            &rt.bb,
            &rt.gene_mtx,
            &rt.gene_names,
            &rt.xy,
            &rt.rw_ligands_init,
            &rt.rw_tfligands_init,
            &targets,
            &cfg,
            &rt.lr_radii,
            Some(&job_p),
            Some(&job_msg),
            Some(&*cancel),
            Some(&rt.baseline_splash_cache),
            &mut no_timings,
        )
    })
    .await
    .unwrap_or(Err(()))
}

async fn cached_grn_perturb_result(
    foyer: &FoyerPerturbCaches,
    cache_key: PerturbCacheKey,
    n_cells: usize,
    n_genes: usize,
    rt: Arc<PerturbRuntime>,
    targets: Vec<PerturbTarget>,
    cfg: PerturbConfig,
    job_p: Arc<AtomicU32>,
    job_msg: Arc<Mutex<String>>,
    cancel: Arc<AtomicBool>,
) -> Result<PerturbResult, ()> {
    let est = perturb_matrix_payload_est_bytes(n_cells, n_genes);
    if !should_use_foyer_perturb_cache(est) {
        return direct_grn_perturb_result(rt, targets, cfg, job_p, job_msg, cancel).await;
    }

    let grn = foyer.grn.clone();
    let rt_c = Arc::clone(&rt);
    let targets_c = targets.clone();
    let cfg_c = cfg.clone();
    let job_p_c = Arc::clone(&job_p);
    let job_msg_c = Arc::clone(&job_msg);
    let cancel_c = Arc::clone(&cancel);
    let entry = grn
        .get_or_fetch(&cache_key, move || {
            let rt_c = Arc::clone(&rt_c);
            let targets_c = targets_c.clone();
            let cfg_c = cfg_c.clone();
            let job_p_c = Arc::clone(&job_p_c);
            let job_msg_c = Arc::clone(&job_msg_c);
            let cancel_c = Arc::clone(&cancel_c);
            async move {
                let pr = tokio::task::spawn_blocking(move || {
                    let mut no_timings: Option<PerturbTimings> = None;
                    perturb_with_targets(
                        &rt_c.bb,
                        &rt_c.gene_mtx,
                        &rt_c.gene_names,
                        &rt_c.xy,
                        &rt_c.rw_ligands_init,
                        &rt_c.rw_tfligands_init,
                        &targets_c,
                        &cfg_c,
                        &rt_c.lr_radii,
                        Some(&job_p_c),
                        Some(&job_msg_c),
                        Some(&*cancel_c),
                        Some(&rt_c.baseline_splash_cache),
                        &mut no_timings,
                    )
                    .map_err(|_| anyhow!("perturbation cancelled"))
                })
                .await
                .map_err(|e| anyhow!("{e}"))?
                .map_err(|e| e)?;
                let enc =
                    foyer_perturb_cache::encode_perturb_result(&pr).map_err(|e| anyhow!("{e}"))?;
                Ok::<Vec<u8>, anyhow::Error>(enc)
            }
        })
        .await
        .map_err(|_| ())?;
    foyer_perturb_cache::decode_perturb_cache_entry(
        entry.value(),
        &rt.gene_mtx,
        &rt.gene_names,
        &targets,
    )
    .map_err(|_| ())
}

fn spawn_perturb_background_load(state: SharedState) {
    tokio::spawn(async move {
        loop {
            let (run_toml, committed_gen) = {
                let g = state.read().await;
                let Some(ds) = g.dataset.as_ref() else {
                    return;
                };
                if ds.perturb_runtime.is_some() {
                    return;
                }
                if ds.perturb_load_error.is_some() {
                    return;
                }
                let Some(p) = ds.run_toml.clone() else {
                    return;
                };
                (p, g.perturb_bg_gen.load(Ordering::SeqCst))
            };

            let in_flight_flag = Arc::clone(&state.read().await.perturb_bg_in_flight);
            if in_flight_flag
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                state
                    .read()
                    .await
                    .perturb_bg_pending
                    .store(true, Ordering::SeqCst);
                return;
            }

            {
                let _in_flight_guard = ClearPerturbInFlight(Arc::clone(&in_flight_flag));
                {
                    let g = state.read().await;
                    g.perturb_suppress_bg_loading_ui
                        .store(false, Ordering::SeqCst);
                }
                tracing::info!(
                    "loading perturbation runtime in background from {} (large betabase runs can take several minutes)",
                    run_toml.display()
                );
                let (load_perm, prog_msg, betadata_ui) = {
                    let g = state.read().await;
                    g.perturb_betadata_ui.reset();
                    (
                        Arc::clone(&g.perturb_load_progress_permille),
                        Arc::clone(&g.perturb_progress_message),
                        Arc::clone(&g.perturb_betadata_ui),
                    )
                };
                load_perm.store(0, Ordering::Relaxed);
                if let Ok(mut m) = prog_msg.lock() {
                    *m = "Starting perturbation runtime load…".into();
                }
                let run_toml_for_blocking = run_toml.clone();
                let load_perm_block = Arc::clone(&load_perm);
                let betadata_ui_block = Arc::clone(&betadata_ui);
                let pr_result = tokio::task::spawn_blocking(move || {
                    PerturbRuntime::from_run_toml_with_progress(
                        run_toml_for_blocking.as_path(),
                        Some(load_perm_block),
                        Some(prog_msg),
                        Some(betadata_ui_block),
                    )
                })
                .await;
                let mut w = state.write().await;
                if w.perturb_bg_gen.load(Ordering::SeqCst) != committed_gen {
                    load_perm.store(0, Ordering::Relaxed);
                } else if let Some(ds) = w.dataset.as_mut() {
                    if ds.run_toml.as_ref() != Some(&run_toml) {
                        load_perm.store(0, Ordering::Relaxed);
                    } else if ds.perturb_runtime.is_some() {
                        load_perm.store(0, Ordering::Relaxed);
                    } else {
                        match pr_result {
                            Ok(Ok(pr)) => {
                                if let Err(e) =
                                    attach_perturb_runtime_to_dataset(ds, pr, run_toml.as_path())
                                {
                                    tracing::error!("perturbation runtime rejected: {:#}", e);
                                    load_perm.store(0, Ordering::Relaxed);
                                    ds.perturb_load_error = Some(format!("{:#}", e));
                                } else {
                                    schedule_perturb_progress_permille_clear(Arc::clone(
                                        &load_perm,
                                    ));
                                }
                            }
                            Ok(Err(e)) => {
                                tracing::error!("perturbation runtime load failed: {:#}", e);
                                load_perm.store(0, Ordering::Relaxed);
                                ds.perturb_load_error = Some(format!("{:#}", e));
                            }
                            Err(e) => {
                                load_perm.store(0, Ordering::Relaxed);
                                ds.perturb_load_error =
                                    Some(format!("perturb load task join: {}", e));
                            }
                        }
                    }
                } else {
                    load_perm.store(0, Ordering::Relaxed);
                }
            }

            let pending = state
                .read()
                .await
                .perturb_bg_pending
                .swap(false, Ordering::SeqCst);
            if pending {
                let mut w = state.write().await;
                if let Some(ds) = w.dataset.as_mut() {
                    ds.perturb_load_error = None;
                }
                continue;
            }
            break;
        }
    });
}

fn require_dataset(st: &AppState) -> Result<&AppDataset, (StatusCode, String)> {
    st.dataset.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "No dataset loaded; use Dataset paths → Load dataset.".into(),
    ))
}

fn perturb_runtime_or_status(
    ds: &AppDataset,
) -> Result<&Arc<PerturbRuntime>, (StatusCode, String)> {
    if let Some(rt) = ds.perturb_runtime.as_ref() {
        return Ok(rt);
    }
    if let Some(e) = &ds.perturb_load_error {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Perturbation load failed: {}", e),
        ));
    }
    if ds.run_toml.is_some() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Perturbation runtime is still loading or was cancelled; use GET /api/meta (perturb_loading) or POST /api/cancel then reload dataset."
                .into(),
        ));
    }
    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        "Perturbation requires --run-toml (spacetravlr_run_repro.toml); data.adata_path must match --h5ad."
            .into(),
    ))
}

fn meta_json(st: &AppState) -> MetaJson {
    let Some(ds) = &st.dataset else {
        return MetaJson {
            n_obs: 0,
            n_vars: 0,
            spatial_obsm_key: String::new(),
            layer: st.default_layer.clone(),
            cluster_annot: st.default_cluster_annot.clone(),
            bounds: MetaBounds {
                min_x: 0.0,
                max_x: 1.0,
                min_y: 0.0,
                max_y: 1.0,
            },
            umap_obsm_key: None,
            umap_bounds: None,
            cell_type_column: None,
            cell_type_categories: vec![],
            network_loaded: false,
            network_species: None,
            betadata_row_id: None,
            perturb_ready: false,
            perturb_loading: false,
            perturb_error: None,
            perturb_progress_percent: None,
            perturb_progress_permille: None,
            perturb_progress_label: None,
            perturb_betadata_phase: None,
            perturb_betadata_done: None,
            perturb_betadata_total: None,
            adata_path: String::new(),
            betadata_dir: String::new(),
            network_dir: st
                .default_network_dir
                .as_ref()
                .map(|p| p.display().to_string()),
            run_toml: st
                .default_run_toml
                .as_ref()
                .map(|p| p.display().to_string()),
            dataset_ready: false,
            spatial_model: None,
        };
    };
    MetaJson {
        n_obs: ds.obs_names.len(),
        n_vars: ds.n_vars,
        spatial_obsm_key: ds.spatial_key.clone(),
        layer: ds.layer.clone(),
        cluster_annot: ds.cluster_annot.clone(),
        bounds: ds.meta_bounds.clone(),
        umap_obsm_key: ds.umap_key.clone(),
        umap_bounds: ds.umap_bounds.clone(),
        cell_type_column: ds.cell_type_column.clone(),
        cell_type_categories: ds.cell_type_categories.as_ref().clone(),
        network_loaded: ds.grn.is_some(),
        network_species: ds.grn.as_ref().map(|g| g.species.clone()),
        betadata_row_id: ds.betadata_row_id.clone(),
        perturb_ready: ds.perturb_runtime.is_some(),
        perturb_loading: {
            let suppress = st.perturb_suppress_bg_loading_ui.load(Ordering::Relaxed);
            (!suppress && st.perturb_bg_in_flight.load(Ordering::Relaxed))
                || st.perturb_job_active.load(Ordering::Relaxed)
        },
        perturb_error: ds.perturb_load_error.clone(),
        perturb_progress_percent: {
            let job_on = st.perturb_job_active.load(Ordering::Relaxed);
            let suppress = st.perturb_suppress_bg_loading_ui.load(Ordering::Relaxed);
            let load_on = !suppress && st.perturb_bg_in_flight.load(Ordering::Relaxed);
            let job_perm = st.perturb_job_progress_permille.load(Ordering::Relaxed);
            let load_perm = st.perturb_load_progress_permille.load(Ordering::Relaxed);
            let p = if job_on {
                Some(job_perm)
            } else if load_on {
                Some(load_perm)
            } else if job_perm > 0 {
                Some(job_perm)
            } else if !suppress && load_perm > 0 {
                Some(load_perm)
            } else {
                None
            };
            p.map(permille_to_percent)
        },
        perturb_progress_permille: {
            let job_on = st.perturb_job_active.load(Ordering::Relaxed);
            let suppress = st.perturb_suppress_bg_loading_ui.load(Ordering::Relaxed);
            let load_on = !suppress && st.perturb_bg_in_flight.load(Ordering::Relaxed);
            let job_perm = st.perturb_job_progress_permille.load(Ordering::Relaxed);
            let load_perm = st.perturb_load_progress_permille.load(Ordering::Relaxed);
            let p = if job_on {
                Some(job_perm)
            } else if load_on {
                Some(load_perm)
            } else if job_perm > 0 {
                Some(job_perm)
            } else if !suppress && load_perm > 0 {
                Some(load_perm)
            } else {
                None
            };
            p.filter(|&v| v > 0 && v <= 1000)
        },
        perturb_progress_label: {
            let suppress = st.perturb_suppress_bg_loading_ui.load(Ordering::Relaxed);
            let load_on = !suppress && st.perturb_bg_in_flight.load(Ordering::Relaxed);
            let job_on = st.perturb_job_active.load(Ordering::Relaxed);
            let job_perm = st.perturb_job_progress_permille.load(Ordering::Relaxed);
            let load_perm = st.perturb_load_progress_permille.load(Ordering::Relaxed);
            let show = job_on || load_on || job_perm > 0 || (!suppress && load_perm > 0);
            if !show {
                None
            } else {
                st.perturb_progress_message
                    .lock()
                    .ok()
                    .map(|g| g.clone())
                    .filter(|s| !s.is_empty())
            }
        },
        perturb_betadata_phase: {
            let bdp = st.perturb_betadata_ui.phase.load(Ordering::Relaxed);
            if bdp == 0 {
                None
            } else if bdp == 1 {
                Some("reading".to_string())
            } else {
                Some("expanding".to_string())
            }
        },
        perturb_betadata_done: {
            let bdp = st.perturb_betadata_ui.phase.load(Ordering::Relaxed);
            if bdp == 0 {
                None
            } else {
                Some(st.perturb_betadata_ui.done.load(Ordering::Relaxed))
            }
        },
        perturb_betadata_total: {
            let bdp = st.perturb_betadata_ui.phase.load(Ordering::Relaxed);
            if bdp == 0 {
                None
            } else {
                Some(st.perturb_betadata_ui.total.load(Ordering::Relaxed))
            }
        },
        adata_path: ds.adata_path.display().to_string(),
        betadata_dir: ds
            .betadata_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        network_dir: ds.network_dir.as_ref().map(|p| p.display().to_string()),
        run_toml: ds.run_toml.as_ref().map(|p| p.display().to_string()),
        dataset_ready: true,
        spatial_model: ds
            .perturb_runtime
            .as_ref()
            .map(|rt| spatial_model_meta_from_runtime(rt, ds.obs_names.len(), ds.n_vars)),
    }
}

fn resolve_load_inputs(
    mut inputs: ViewerLoadInputs,
) -> anyhow::Result<(ViewerLoadInputs, Option<SpaceshipConfig>)> {
    let run_spaceship_cfg = if let Some(ref rtp) = inputs.run_toml {
        let cfg = SpaceshipConfig::from_file(rtp)
            .map_err(|e| anyhow::anyhow!("failed to read run TOML {}: {e}", rtp.display()))?;
        let ap = normalize_ui_path(cfg.resolve_adata_path().as_str());
        if ap.is_empty() {
            anyhow::bail!("run TOML {} has empty data.adata_path", rtp.display());
        }
        inputs.h5ad = PathBuf::from(ap);
        Some(cfg)
    } else {
        if inputs.h5ad.as_os_str().is_empty() {
            anyhow::bail!("adata path is required (or pass run_toml with data.adata_path)");
        }
        None
    };
    Ok((inputs, run_spaceship_cfg))
}

fn paths_eq_resolved(a: &Path, b: &Path) -> bool {
    let ca = a.canonicalize().unwrap_or_else(|_| a.to_path_buf());
    let cb = b.canonicalize().unwrap_or_else(|_| b.to_path_buf());
    ca == cb
}

fn dataset_matches_resolved_inputs(ds: &AppDataset, inputs: &ViewerLoadInputs) -> bool {
    paths_eq_resolved(&ds.adata_path, &inputs.h5ad)
        && ds.layer == inputs.layer
        && ds.cluster_annot == inputs.cluster_annot
        && ds.network_dir == inputs.network_dir
        && ds.run_toml == inputs.run_toml
}

fn load_app_state(inputs: ViewerLoadInputs) -> anyhow::Result<AppDataset> {
    let (inputs, run_spaceship_cfg) = resolve_load_inputs(inputs)?;

    let adata = open_adata(inputs.h5ad.to_string_lossy().as_ref()).with_context(|| {
        let mut msg = format!("failed to open AnnData at `{}`", inputs.h5ad.display());
        if run_spaceship_cfg.is_some() {
            msg.push_str(
                " (when Run TOML is set, this path comes from the TOML's data.adata_path, not the AnnData field in Dataset paths — clear Run TOML to load only what you paste there)",
            );
        }
        msg
    })?;
    let spatial_key = spatial_obsm_key_used(&adata)?;
    let xy = spatial_xy(&adata)?;
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..xy.nrows() {
        let x = xy[[i, 0]];
        let y = xy[[i, 1]];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let spatial_f32 = Arc::new(spatial_xy_f32_interleaved(&xy));
    let onames = Arc::new(obs_names(&adata));
    let (umap_key, umap_f32, umap_bounds) = match try_umap_xy(&adata, onames.len())? {
        Some((key, u_xy)) => {
            let mut umin_x = f64::INFINITY;
            let mut umax_x = f64::NEG_INFINITY;
            let mut umin_y = f64::INFINITY;
            let mut umax_y = f64::NEG_INFINITY;
            for i in 0..u_xy.nrows() {
                let ux = u_xy[[i, 0]];
                let uy = u_xy[[i, 1]];
                umin_x = umin_x.min(ux);
                umax_x = umax_x.max(ux);
                umin_y = umin_y.min(uy);
                umax_y = umax_y.max(uy);
            }
            let uf = Arc::new(spatial_xy_f32_interleaved(&u_xy));
            (
                Some(key),
                Some(uf),
                Some(MetaBounds {
                    min_x: umin_x,
                    max_x: umax_x,
                    min_y: umin_y,
                    max_y: umax_y,
                }),
            )
        }
        None => (None, None, None),
    };
    if let Some(ref k) = umap_key {
        tracing::info!("UMAP layout available (obsm['{}'])", k);
    }
    let obs_df = adata.read_obs()?;
    let clusters = Arc::new(
        space_trav_lr_rust::adata_query::clusters_from_obs_dataframe(
            &obs_df,
            &inputs.cluster_annot,
        )?,
    );
    let betadata_key_col = space_trav_lr_rust::betadata::resolve_betadata_cluster_key_column(
        &obs_df,
        &inputs.cluster_annot,
    );
    if betadata_key_col != inputs.cluster_annot {
        tracing::info!(
            "betadata feather join uses obs column {:?} (cluster_annot is {:?})",
            betadata_key_col,
            inputs.cluster_annot
        );
    }
    let betadata_cluster_keys = Arc::new(
        space_trav_lr_rust::betadata::betadata_cluster_keys_from_obs_dataframe(
            &obs_df,
            betadata_key_col.as_str(),
        )?,
    );
    anyhow::ensure!(
        onames.len() == clusters.len(),
        "cluster column length mismatch"
    );
    anyhow::ensure!(
        onames.len() == betadata_cluster_keys.len(),
        "betadata cluster keys length mismatch"
    );
    let clusters_bin = Arc::new(clusters_as_u32_le_bytes(clusters.as_ref()));
    let n_vars = adata.n_vars();
    let cell_type_enc = cell_type_encoding(&adata)?;
    let (cell_type_column, cell_type_categories, cell_type_codes_bin) = match cell_type_enc {
        Some((name, cats, codes)) => (
            Some(name),
            Arc::new(cats),
            Some(Arc::new(u16_vec_to_le_bytes(&codes))),
        ),
        None => (None, Arc::new(vec![]), None),
    };
    let vn = Arc::new(var_names(&adata));
    let net_dir = inputs
        .network_dir
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());
    let species = infer_species(vn.as_ref());
    let grn: Option<Arc<GeneNetwork>> =
        match GeneNetwork::new(species, vn.as_ref(), net_dir.as_deref()) {
            Ok(g) => {
                tracing::info!("loaded GRN species={} path={}", g.species, g.network_path);
                Some(Arc::new(g))
            }
            Err(e) => {
                tracing::warn!("GRN not available: {}", e);
                None
            }
        };
    let (perturb_runtime, betadata_dir) =
        if let (Some(rtp), Some(cfg)) = (&inputs.run_toml, &run_spaceship_cfg) {
            let bd = cfg.resolve_training_output_dir(rtp.as_path());
            tracing::info!(
                "perturbation API will load in background from {} (betadata dir {})",
                rtp.display(),
                bd.display()
            );
            (None, Some(bd))
        } else {
            (None, None)
        };
    let betadata_row_id = betadata_dir.as_ref().and_then(|d| probe_betadata_row_id(d));
    let adata = Arc::new(RwLock::new(adata));

    Ok(AppDataset {
        adata,
        adata_path: inputs.h5ad,
        layer: inputs.layer,
        cluster_annot: inputs.cluster_annot,
        betadata_dir,
        network_dir: inputs.network_dir,
        run_toml: inputs.run_toml,
        obs_names: onames,
        clusters,
        betadata_cluster_keys,
        spatial_key,
        spatial_f32,
        umap_key,
        umap_f32,
        umap_bounds,
        clusters_bin,
        n_vars,
        meta_bounds: MetaBounds {
            min_x,
            max_x,
            min_y,
            max_y,
        },
        cell_type_column,
        cell_type_categories,
        cell_type_codes_bin,
        var_names: vn,
        grn,
        betadata_row_id,
        perturb_runtime,
        perturb_load_error: None,
    })
}

#[derive(Deserialize)]
struct SessionConfigureBody {
    adata_path: String,
    #[serde(default)]
    layer: String,
    #[serde(default)]
    cluster_annot: String,
    #[serde(default)]
    network_dir: String,
    #[serde(default)]
    run_toml: String,
}

#[derive(Serialize)]
struct SessionConfigureResponse {
    ok: bool,
    message: String,
    meta: MetaJson,
}

fn session_path_field(s: &str) -> Option<PathBuf> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(PathBuf::from(normalize_ui_path(t)))
    }
}

fn non_empty_trimmed(s: &str, default: &str) -> String {
    let t = s.trim();
    if t.is_empty() {
        default.to_string()
    } else {
        t.to_string()
    }
}

async fn api_session_configure(
    State(state): State<SharedState>,
    Json(body): Json<SessionConfigureBody>,
) -> Result<Json<SessionConfigureResponse>, (StatusCode, String)> {
    let run_toml = session_path_field(&body.run_toml);
    let h5ad = PathBuf::from(normalize_ui_path(&body.adata_path));
    if h5ad.as_os_str().is_empty() && run_toml.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "adata_path is required unless run_toml is set (then data.adata_path from the TOML is used)"
                .into(),
        ));
    }
    let layer = non_empty_trimmed(&body.layer, "imputed_count");
    let cluster_annot = non_empty_trimmed(&body.cluster_annot, "cell_type");
    let network_dir = session_path_field(&body.network_dir);
    let inputs = ViewerLoadInputs {
        h5ad,
        layer: layer.clone(),
        cluster_annot: cluster_annot.clone(),
        network_dir: network_dir.clone(),
        run_toml: run_toml.clone(),
    };
    let resolved = resolve_load_inputs(inputs.clone())
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("{:#}", e)))?;

    {
        let r = state.read().await;
        if let Some(ds) = r.dataset.as_ref() {
            if dataset_matches_resolved_inputs(ds, &resolved.0) {
                let ready = ds.perturb_runtime.is_some();
                let load_busy = r.perturb_bg_in_flight.load(Ordering::SeqCst)
                    || r.perturb_bg_pending.load(Ordering::SeqCst);
                if ready || load_busy {
                    let meta = meta_json(&*r);
                    return Ok(Json(SessionConfigureResponse {
                        ok: true,
                        message: if ready {
                            "dataset unchanged (already loaded)".into()
                        } else {
                            "dataset unchanged (perturbation runtime still loading)".into()
                        },
                        meta,
                    }));
                }
                if ds.run_toml.is_some() && ds.perturb_load_error.is_some() {
                    drop(r);
                    let mut w = state.write().await;
                    let Some(ds_mut) = w.dataset.as_mut() else {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "dataset missing during perturb retry".into(),
                        ));
                    };
                    if !dataset_matches_resolved_inputs(ds_mut, &resolved.0) {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "dataset mismatch during perturb retry".into(),
                        ));
                    }
                    ds_mut.perturb_load_error = None;
                    w.perturb_bg_gen.fetch_add(1, Ordering::SeqCst);
                    w.perturb_suppress_bg_loading_ui
                        .store(false, Ordering::SeqCst);
                    let meta = meta_json(&*w);
                    drop(w);
                    spawn_perturb_background_load(state.clone());
                    return Ok(Json(SessionConfigureResponse {
                        ok: true,
                        message: "retrying perturbation runtime load (same dataset)".into(),
                        meta,
                    }));
                }
            }
        }
    }

    let new_dataset = tokio::task::spawn_blocking(move || load_app_state(inputs))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let mut w = state.write().await;
    w.dataset_cache_epoch.fetch_add(1, Ordering::SeqCst);
    w.perturb_bg_gen.fetch_add(1, Ordering::SeqCst);
    w.perturb_bg_pending.store(false, Ordering::SeqCst);
    w.perturb_suppress_bg_loading_ui
        .store(false, Ordering::SeqCst);
    w.dataset = Some(new_dataset);
    w.default_layer = layer;
    w.default_cluster_annot = cluster_annot;
    w.default_network_dir = network_dir;
    w.default_run_toml = run_toml;
    let meta = meta_json(&*w);
    drop(w);
    spawn_perturb_background_load(state.clone());
    Ok(Json(SessionConfigureResponse {
        ok: true,
        message: "dataset loaded".into(),
        meta,
    }))
}

async fn api_meta(State(state): State<SharedState>) -> impl IntoResponse {
    let st = state.read().await;
    axum::Json(meta_json(&st))
}

async fn api_cancel(State(state): State<SharedState>) -> impl IntoResponse {
    let w = state.write().await;
    w.perturb_bg_gen.fetch_add(1, Ordering::SeqCst);
    w.perturb_suppress_bg_loading_ui
        .store(true, Ordering::SeqCst);
    w.perturb_job_cancel.store(true, Ordering::SeqCst);
    w.perturb_job_active.store(false, Ordering::SeqCst);
    w.perturb_job_progress_permille.store(0, Ordering::Relaxed);
    w.perturb_load_progress_permille.store(0, Ordering::Relaxed);
    w.splash_job_active.store(false, Ordering::SeqCst);
    w.splash_job_progress_permille.store(0, Ordering::Relaxed);
    if let Ok(mut m) = w.perturb_progress_message.lock() {
        *m = "Cancelled.".into();
    }
    w.perturb_betadata_ui.reset();
    Json(serde_json::json!({ "ok": true, "message": "cancel requested" }))
}

async fn api_spatial(State(state): State<SharedState>) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    Ok(binary_response(f32_vec_to_le_bytes(
        ds.spatial_f32.as_ref(),
    )))
}

async fn api_umap(State(state): State<SharedState>) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let Some(ref u) = ds.umap_f32 else {
        return Err((
            StatusCode::NOT_FOUND,
            "no obsm['X_umap'] or obsm['umap'] in this dataset".into(),
        ));
    };
    Ok(binary_response(f32_vec_to_le_bytes(u.as_ref())))
}

async fn api_clusters(State(state): State<SharedState>) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    Ok(binary_response(ds.clusters_bin.as_ref().clone()))
}

async fn api_cell_type_codes(
    State(state): State<SharedState>,
) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let Some(ref bin) = ds.cell_type_codes_bin else {
        return Err((StatusCode::NOT_FOUND, "no cell_type column".into()));
    };
    Ok(binary_response(bin.as_ref().clone()))
}

async fn api_genes(
    State(state): State<SharedState>,
    Query(q): Query<GenesQuery>,
) -> Result<axum::Json<Vec<String>>, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let prefix = q.prefix.unwrap_or_default();
    let limit = q.limit.unwrap_or(DEFAULT_GENE_SEARCH_LIMIT).min(10_000);
    let adata = ds.adata.read().await;
    let list = genes_with_prefix(&adata, &prefix, limit);
    Ok(axum::Json(list))
}

async fn api_genes_full(
    State(state): State<SharedState>,
) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let adata = ds.adata.read().await;
    if adata.n_vars() > FULL_GENE_LIST_THRESHOLD {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "n_vars {} exceeds {}; use /api/genes?prefix=...",
                adata.n_vars(),
                FULL_GENE_LIST_THRESHOLD
            ),
        ));
    }
    let names = var_names(&adata);
    Ok(axum::Json(names).into_response())
}

async fn api_gene_expression(
    State(state): State<SharedState>,
    Query(q): Query<GeneExprQuery>,
) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let path = ds.adata_path.clone();
    let layer = ds.layer.clone();
    let gene = q.gene;
    drop(st);
    let vec = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
        let adata = open_adata(path.to_string_lossy().as_ref()).map_err(|e| e.to_string())?;
        gene_expression_f32(&adata, &layer, &gene).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::NOT_FOUND, e))?;
    Ok(binary_response(f32_vec_to_le_bytes(&vec)))
}

async fn api_received_ligand(
    State(state): State<SharedState>,
    Json(body): Json<ReceivedLigandBody>,
) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let n = ds.obs_names.len();
    let path = ds.adata_path.clone();
    let layer = ds.layer.clone();
    let spatial = ds.spatial_f32.as_ref().clone();
    let vn = ds.var_names.as_ref().clone();
    let rt_opt = ds.perturb_runtime.clone();
    drop(st);

    let mut genes: Vec<String> = body
        .genes
        .iter()
        .filter_map(|g| {
            let t = g.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect();
    genes.truncate(MAX_RECEIVED_LIGAND_GENES);

    let source = body.source.to_ascii_lowercase();
    let aggregate = body.aggregate.clone();
    let matrix_key = body.matrix.clone();

    let vec: Vec<f32> = if source == "runtime" || source == "model" {
        let rt = rt_opt.ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "Perturbation runtime not loaded; use --run-toml and wait until perturb_ready."
                    .to_string(),
            )
        })?;
        let col = genes.first().cloned().ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "runtime/model source requires genes[0] = column name in the received-ligand matrix"
                    .to_string(),
            )
        })?;
        tokio::task::spawn_blocking(move || runtime_received_ligand_column(&*rt, &col, &matrix_key))
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    } else {
        if genes.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "adata source requires a non-empty genes list (ligand symbols in var)".into(),
            ));
        }
        let radius = body.radius.unwrap_or(120.0);
        if !radius.is_finite() || radius <= 0.0 {
            return Err((
                StatusCode::BAD_REQUEST,
                "radius must be finite and > 0 (spatial coordinate units)".into(),
            ));
        }
        let scale = body.scale_factor.unwrap_or(1.0);
        if !scale.is_finite() {
            return Err((
                StatusCode::BAD_REQUEST,
                "scale_factor must be finite".into(),
            ));
        }
        let grid_factor = body.grid_factor.unwrap_or(0.5);
        if !grid_factor.is_finite() || grid_factor <= 0.0 {
            return Err((
                StatusCode::BAD_REQUEST,
                "grid_factor must be finite and > 0".into(),
            ));
        }
        let use_grid = body.use_grid;
        tokio::task::spawn_blocking(move || {
            let adata = open_adata(path.to_string_lossy().as_ref())
                .map_err(|e| format!("open adata: {}", e))?;
            compute_received_ligand_from_adata(
                &adata,
                &layer,
                &spatial,
                vn.as_ref(),
                &genes,
                radius,
                scale,
                use_grid,
                grid_factor,
                &aggregate,
            )
            .map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?
    };

    if vec.len() != n {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "received_ligand length {} != n_obs {} (check adata vs spatial alignment)",
                vec.len(),
                n
            ),
        ));
    }
    Ok(binary_response(f32_vec_to_le_bytes(&vec)))
}

async fn api_betadata_genes(
    State(state): State<SharedState>,
) -> Result<axum::Json<Vec<String>>, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let Some(ref bd) = ds.betadata_dir else {
        return Ok(axum::Json(Vec::<String>::new()));
    };
    let dir = bd.to_string_lossy().to_string();
    match list_betadata_target_genes(&dir) {
        Ok(v) => Ok(axum::Json(v)),
        Err(e) => Err((StatusCode::BAD_REQUEST, e.to_string())),
    }
}

async fn api_betadata_columns(
    State(state): State<SharedState>,
    Query(q): Query<BetadataColumnsQuery>,
) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let Some(ref bd) = ds.betadata_dir else {
        return Err((
            StatusCode::BAD_REQUEST,
            "No betadata directory configured".into(),
        ));
    };
    let path = betadata_feather_path(bd.to_string_lossy().as_ref(), &q.gene);
    if !path.is_file() {
        return Err((StatusCode::NOT_FOUND, format!("missing {:?}", path)));
    }
    let path_s = path.to_string_lossy().into_owned();
    let cols = tokio::task::spawn_blocking(move || betadata_feather_plottable_columns(&path_s))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(axum::Json(cols).into_response())
}

async fn api_betadata_values(
    State(state): State<SharedState>,
    Query(q): Query<BetadataValuesQuery>,
) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let bd = ds.betadata_dir.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "No betadata directory configured".into(),
        )
    })?;
    let path = betadata_feather_path(bd.to_string_lossy().as_ref(), &q.gene);
    if !path.is_file() {
        return Err((StatusCode::NOT_FOUND, format!("missing {:?}", path)));
    }
    let obs = ds.obs_names.clone();
    let cluster_keys = Arc::clone(&ds.betadata_cluster_keys);
    let path_s = path.to_string_lossy().to_string();
    let column = q.column;
    let vec = tokio::task::spawn_blocking(move || {
        betadata_feather_per_cell_column(&path_s, &column, &obs, cluster_keys.as_ref())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(binary_response(f32_vec_to_le_bytes(&vec)))
}

async fn api_betadata_top(
    State(state): State<SharedState>,
    Json(body): Json<TopBetasBody>,
) -> Result<Json<Vec<TopBetaCoefficient>>, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    if body.indices.is_empty() {
        return Ok(Json(vec![]));
    }
    if body.indices.len() > MAX_TOP_BETA_INDICES {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "indices len {} exceeds max {}",
                body.indices.len(),
                MAX_TOP_BETA_INDICES
            ),
        ));
    }
    let bd = ds.betadata_dir.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "No betadata directory configured".into(),
        )
    })?;
    let path = betadata_feather_path(bd.to_string_lossy().as_ref(), &body.gene);
    if !path.is_file() {
        return Err((StatusCode::NOT_FOUND, format!("missing {:?}", path)));
    }
    let top_k = if body.top_k == 0 {
        25
    } else {
        body.top_k.min(100)
    };
    let obs = ds.obs_names.clone();
    let cluster_keys = Arc::clone(&ds.betadata_cluster_keys);
    let path_s = path.to_string_lossy().to_string();
    let indices = body.indices;
    let out = tokio::task::spawn_blocking(move || {
        betadata_feather_top_coefficients_for_selection(
            &path_s,
            &obs,
            cluster_keys.as_ref(),
            &indices,
            top_k,
        )
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(out))
}

const MAX_COLLECT_INTERACTIONS_OUT: usize = 80_000;

fn default_collect_aggregate_str() -> String {
    "mean".into()
}

fn default_collect_filter_mode() -> String {
    "cell_type".into()
}

fn default_collect_max_genes() -> usize {
    2048
}

#[derive(Deserialize)]
struct CollectInteractionsBody {
    #[serde(default = "default_collect_aggregate_str")]
    aggregate: String,
    #[serde(default = "default_collect_filter_mode")]
    filter: String,
    #[serde(default)]
    cell_type: Option<String>,
    #[serde(default)]
    cluster_id: Option<usize>,
    #[serde(default = "default_collect_max_genes")]
    max_genes: usize,
    #[serde(default)]
    gene_subset: Option<Vec<String>>,
}

#[derive(Serialize)]
struct CollectInteractionsResponse {
    interactions: Vec<CollectedInteraction>,
    n_reported: usize,
    n_total: usize,
    capped: bool,
}

async fn api_betadata_collect_interactions(
    State(state): State<SharedState>,
    Json(body): Json<CollectInteractionsBody>,
) -> Result<Json<CollectInteractionsResponse>, (StatusCode, String)> {
    let mode = BetadataCollectAggregate::parse(body.aggregate.trim()).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "aggregate must be mean|min|max|sum|positive|negative".into(),
        )
    })?;
    let max_genes = body.max_genes.clamp(1, 4096);

    let (
        bd_dir,
        obs_names,
        betadata_ck,
        clusters_usize,
        cell_type_categories,
        cell_type_bin,
        n_obs,
    ) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let Some(ref bd) = ds.betadata_dir else {
            return Err((
                StatusCode::BAD_REQUEST,
                "No betadata directory configured".into(),
            ));
        };
        (
            bd.to_string_lossy().into_owned(),
            Arc::clone(&ds.obs_names),
            Arc::clone(&ds.betadata_cluster_keys),
            Arc::clone(&ds.clusters),
            Arc::clone(&ds.cell_type_categories),
            ds.cell_type_codes_bin.clone(),
            ds.obs_names.len(),
        )
    };

    let filter = body.filter.to_ascii_lowercase();
    let mask: Vec<bool> = match filter.as_str() {
        "cell_type" | "celltype" => {
            let label = body
                .cell_type
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        "cell_type label required when filter=cell_type".into(),
                    )
                })?;
            let want_codes: HashSet<u16> = cell_type_categories
                .iter()
                .enumerate()
                .filter(|(_, c)| c.as_str() == label)
                .map(|(i, _)| i as u16)
                .collect();
            if want_codes.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("cell_type {:?} not in annotation categories", label),
                ));
            }
            let Some(ref bin) = cell_type_bin else {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "dataset has no cell_type column for this filter".into(),
                ));
            };
            let codes: Vec<u16> = bin
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            if codes.len() != n_obs {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "cell_type codes length mismatch".into(),
                ));
            }
            (0..n_obs)
                .map(|i| codes[i] != u16::MAX && want_codes.contains(&codes[i]))
                .collect()
        }
        "cluster" => {
            let cid = body.cluster_id.ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "cluster_id required when filter=cluster".into(),
                )
            })?;
            (0..n_obs).map(|i| clusters_usize[i] == cid).collect()
        }
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "filter must be cell_type or cluster".into(),
            ));
        }
    };

    if !mask.iter().any(|&m| m) {
        return Err((StatusCode::BAD_REQUEST, "no cells match the filter".into()));
    }

    let mut genes = if let Some(gs) = body.gene_subset {
        gs
    } else {
        list_betadata_target_genes(&bd_dir).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    };
    genes.retain(|g| !g.trim().is_empty());
    genes.truncate(max_genes);

    if genes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "no target genes to scan (empty gene list)".into(),
        ));
    }

    let obs = (*obs_names).clone();
    let ck = (*betadata_ck).clone();
    let dir = bd_dir.clone();
    let mask_clone = mask.clone();
    let mut interactions = tokio::task::spawn_blocking(move || {
        betadata_collect_interactions_parallel(&dir, &genes, &obs, &ck, &mask_clone, mode)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let n_total = interactions.len();
    let capped = interactions.len() > MAX_COLLECT_INTERACTIONS_OUT;
    if capped {
        interactions.truncate(MAX_COLLECT_INTERACTIONS_OUT);
    }
    let n_reported = interactions.len();

    Ok(Json(CollectInteractionsResponse {
        interactions,
        n_reported,
        n_total,
        capped,
    }))
}

fn default_pair_lr_top_n() -> usize {
    25
}

#[derive(Deserialize)]
struct PairLrBody {
    #[serde(deserialize_with = "deserialize_usize_flex")]
    cell_a: usize,
    #[serde(deserialize_with = "deserialize_usize_flex")]
    cell_b: usize,
    #[serde(
        default = "default_pair_lr_top_n",
        deserialize_with = "deserialize_usize_flex"
    )]
    top_n: usize,
    #[serde(
        default = "default_collect_max_genes",
        deserialize_with = "deserialize_usize_flex"
    )]
    max_genes: usize,
    #[serde(default)]
    gene_subset: Option<Vec<String>>,
}

#[derive(Serialize)]
struct PairLrResponse {
    cell_a: usize,
    cell_b: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    betadata_row_id: Option<String>,
    rows: Vec<PairLrBetaRow>,
    n_genes_scanned: usize,
}

async fn api_betadata_pair_lr(
    State(state): State<SharedState>,
    Json(body): Json<PairLrBody>,
) -> Result<Json<PairLrResponse>, (StatusCode, String)> {
    if body.cell_a == body.cell_b {
        return Err((
            StatusCode::BAD_REQUEST,
            "cell_a and cell_b must differ".into(),
        ));
    }
    let top_n = body.top_n.clamp(1, 200);
    let max_genes = body.max_genes.clamp(1, 4096);

    let (bd_dir, obs_names, betadata_ck, row_id) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let Some(ref bd) = ds.betadata_dir else {
            return Err((
                StatusCode::BAD_REQUEST,
                "No betadata directory configured".into(),
            ));
        };
        if body.cell_a >= ds.obs_names.len() || body.cell_b >= ds.obs_names.len() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("cell index out of range (n_obs = {})", ds.obs_names.len()),
            ));
        };
        (
            bd.to_string_lossy().into_owned(),
            Arc::clone(&ds.obs_names),
            Arc::clone(&ds.betadata_cluster_keys),
            ds.betadata_row_id.clone(),
        )
    };

    let mut genes = if let Some(gs) = body.gene_subset {
        gs
    } else {
        list_betadata_target_genes(&bd_dir).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    };
    genes.retain(|g| !g.trim().is_empty());
    genes.truncate(max_genes);

    if genes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "no target genes to scan (empty gene list)".into(),
        ));
    }

    let obs = (*obs_names).clone();
    let ck = (*betadata_ck).clone();
    let dir = bd_dir.clone();
    let ca = body.cell_a;
    let cb = body.cell_b;
    let n_genes = genes.len();
    let mut rows = tokio::task::spawn_blocking(move || {
        betadata_pair_lr_parallel(&dir, &genes, &obs, &ck, ca, cb)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    rows.truncate(top_n);

    Ok(Json(PairLrResponse {
        cell_a: body.cell_a,
        cell_b: body.cell_b,
        betadata_row_id: row_id,
        rows,
        n_genes_scanned: n_genes,
    }))
}

async fn api_network_cell_context(
    State(state): State<SharedState>,
    Json(body): Json<CellContextBody>,
) -> Result<Json<CellContextResponse>, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let Some(ref grn) = ds.grn else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "GRN not loaded; use --network-dir or place mouse_network.parquet / human_network.parquet under data/ (or set SPACETRAVLR_DATA_DIR)."
                .into(),
        ));
    };

    let n = ds.obs_names.len();
    if body.cell_index >= n {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("cell_index {} >= n_obs {}", body.cell_index, n),
        ));
    }

    let focus_gene = body.focus_gene.trim();
    if focus_gene.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "focus_gene is empty".into()));
    }

    let var_set: HashSet<&str> = ds.var_names.iter().map(|s| s.as_str()).collect();
    if !var_set.contains(focus_gene) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("focus_gene {:?} not in var_names", focus_gene),
        ));
    }

    let neighbor_k = body.neighbor_k.clamp(1, 200);
    let thr = body.expr_threshold;
    let spatial = ds.spatial_f32.as_ref();
    let use_radius = body
        .neighbor_mode
        .as_deref()
        .map(|m| m.eq_ignore_ascii_case("radius"))
        .unwrap_or(false)
        && body
            .radius
            .map(|r| r > 0.0 && r.is_finite())
            .unwrap_or(false);
    let (neighbors_spatial, neighbor_query, radius_used) = if use_radius {
        let r = body.radius.unwrap_or(0.0);
        (
            spatial_within_radius(spatial, n, body.cell_index, r, MAX_LR_NEIGHBORS_RADIUS),
            Some("radius".to_string()),
            Some(r),
        )
    } else {
        (
            spatial_k_nearest(spatial, n, body.cell_index, neighbor_k),
            Some("knn".to_string()),
            None,
        )
    };

    let expr_map = {
        let adata = ds.adata.read().await;
        cell_expression_map(&adata, &ds.layer, body.cell_index)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("expression read: {}", e)))?
    };

    let modulators = grn
        .get_modulators(
            focus_gene,
            body.tf_ligand_cutoff,
            Some(120),
            Some(&expr_map),
        )
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("modulators: {}", e)))?;

    let mut genes_union: HashSet<String> = HashSet::new();
    for g in &modulators.regulators {
        genes_union.insert(g.clone());
    }
    for g in &modulators.ligands {
        genes_union.insert(g.clone());
    }
    for g in &modulators.receptors {
        genes_union.insert(g.clone());
    }
    for g in &modulators.tfl_ligands {
        genes_union.insert(g.clone());
    }
    for g in &modulators.tfl_regulators {
        genes_union.insert(g.clone());
    }
    genes_union.insert(focus_gene.to_string());

    let genes_vec: Vec<String> = genes_union.into_iter().collect();
    let mut all_cells = vec![body.cell_index];
    for (j, _) in &neighbors_spatial {
        all_cells.push(*j);
    }

    let profiles = {
        let adata = ds.adata.read().await;
        expression_profiles_for_cells(
            &adata,
            &ds.layer,
            &all_cells,
            &genes_vec,
            ds.var_names.as_ref(),
        )
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("profiles: {}", e)))?
    };

    let cell_idx = body.cell_index;
    let sender_expr = |g: &str| -> f64 {
        profiles
            .get(&cell_idx)
            .and_then(|m| m.get(g))
            .copied()
            .unwrap_or(0.0)
    };
    let neighbor_expr = |nj: usize, g: &str| -> f64 {
        profiles
            .get(&nj)
            .and_then(|m| m.get(g))
            .copied()
            .unwrap_or(0.0)
    };

    let mut lig_to_tfs: HashMap<String, Vec<String>> = HashMap::new();
    for tp in &modulators.tfl_pairs {
        if let Some((lig, tf)) = parse_tfl_pair(tp) {
            lig_to_tfs.entry(lig).or_default().push(tf);
        }
    }

    const MAX_EDGES: usize = 48;
    let mut neighbors_out = Vec::with_capacity(neighbors_spatial.len());
    for (nj, d2) in neighbors_spatial {
        let mut lr_edges = Vec::new();
        for pair in &modulators.lr_pairs {
            let Some((lig, rec)) = parse_lr_pair(pair) else {
                continue;
            };
            let ls = sender_expr(&lig);
            let rn = neighbor_expr(nj, &rec);
            if ls <= thr || rn <= thr {
                continue;
            }
            let support_score = (ls * rn).max(0.0).sqrt();
            let linked_tf_opt = lig_to_tfs.get(&lig).and_then(|tfs| {
                tfs.iter()
                    .filter_map(|tf| {
                        let v = sender_expr(tf);
                        (v > thr).then_some((tf.clone(), v))
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            let (linked_tf, linked_tf_expr) = match linked_tf_opt {
                Some((a, b)) => (Some(a), Some(b)),
                None => (None, None),
            };
            lr_edges.push(LrEdgeJson {
                ligand: lig,
                receptor: rec,
                lig_expr_sender: ls,
                rec_expr_neighbor: rn,
                support_score,
                linked_tf,
                linked_tf_expr,
            });
            if lr_edges.len() >= MAX_EDGES {
                break;
            }
        }
        let max_support_score = lr_edges
            .iter()
            .map(|e| e.support_score)
            .fold(0.0_f64, f64::max);
        let max_support_score = (max_support_score > 0.0).then_some(max_support_score);
        let distance = d2.sqrt();
        let cell_type = cell_type_label_for_obs(ds, nj);
        neighbors_out.push(NeighborContextJson {
            index: nj,
            distance_sq: d2,
            distance,
            cell_type,
            lr_edges,
            max_support_score,
        });
    }

    let sender_regulator_exprs: Vec<GeneExprEntry> = modulators
        .regulators
        .iter()
        .filter_map(|g| {
            let e = sender_expr(g);
            (e > thr).then_some(GeneExprEntry {
                gene: g.clone(),
                expr: e,
            })
        })
        .collect();

    let sender_ligand_exprs: Vec<GeneExprEntry> = modulators
        .ligands
        .iter()
        .filter_map(|g| {
            let e = sender_expr(g);
            (e > thr).then_some(GeneExprEntry {
                gene: g.clone(),
                expr: e,
            })
        })
        .collect();

    let neighbors_in_query = neighbors_out.len();
    Ok(Json(CellContextResponse {
        focus_gene: focus_gene.to_string(),
        cell_index: cell_idx,
        modulators,
        neighbors: neighbors_out,
        sender_regulator_exprs,
        sender_ligand_exprs,
        neighbor_query,
        radius_used,
        neighbors_in_query: Some(neighbors_in_query),
    }))
}

#[derive(Deserialize)]
struct PerturbPreviewBody {
    gene: String,
    #[serde(default)]
    desired_expr: f64,
    #[serde(default)]
    scope: PerturbScopeBody,
    /// Overrides run TOML `n_propagation` for this request (clamped 1–32).
    #[serde(default)]
    n_propagation: Option<usize>,
}

#[derive(Clone, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PerturbScopeBody {
    #[default]
    All,
    Indices {
        indices: Vec<usize>,
    },
    CellType {
        category: u16,
    },
    /// All cells whose annotation string equals `name` (unions every cluster/category with that label).
    CellTypeName {
        name: String,
    },
    Cluster {
        cluster_id: usize,
    },
}

fn perturb_cfg_for_request(base: &PerturbConfig, n_propagation: Option<usize>) -> PerturbConfig {
    let mut c = base.clone();
    if let Some(n) = n_propagation {
        c.n_propagation = n.clamp(1, 32);
    }
    c
}

fn cell_indices_for_perturb_scope(
    st: &AppDataset,
    scope: &PerturbScopeBody,
    n_obs: usize,
) -> Result<Option<Vec<usize>>, (StatusCode, String)> {
    match scope {
        PerturbScopeBody::All => Ok(None),
        PerturbScopeBody::Indices { indices } => {
            if indices.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "indices must be non-empty for scope indices".into(),
                ));
            }
            let mut v = Vec::new();
            for &i in indices {
                if i >= n_obs {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("cell index {} out of range (n={})", i, n_obs),
                    ));
                }
                v.push(i);
            }
            v.sort_unstable();
            v.dedup();
            Ok(Some(v))
        }
        PerturbScopeBody::CellType { category } => {
            let bin = st.cell_type_codes_bin.as_ref().ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "no cell_type annotation in this dataset (obs column missing)".to_string(),
                )
            })?;
            let codes: Vec<u16> = bin
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            if codes.len() != n_obs {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "cell_type codes length mismatch".into(),
                ));
            }
            let idx: Vec<usize> = codes
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| (c != u16::MAX && c == *category).then_some(i))
                .collect();
            if idx.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "no cells for that cell type category".into(),
                ));
            }
            Ok(Some(idx))
        }
        PerturbScopeBody::CellTypeName { name } => {
            let name = name.trim();
            if name.is_empty() {
                return Err((StatusCode::BAD_REQUEST, "cell_type_name is empty".into()));
            }
            let cats = st.cell_type_categories.as_ref();
            let bin = st.cell_type_codes_bin.as_ref().ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "no cell_type annotation in this dataset (obs column missing)".to_string(),
                )
            })?;
            let codes: Vec<u16> = bin
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            if codes.len() != n_obs {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "cell_type codes length mismatch".into(),
                ));
            }
            let want_codes: HashSet<u16> = cats
                .iter()
                .enumerate()
                .filter(|(_, s)| s.as_str() == name)
                .map(|(i, _)| i as u16)
                .collect();
            if want_codes.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("no annotation category named {:?}", name),
                ));
            }
            let idx: Vec<usize> = codes
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| (c != u16::MAX && want_codes.contains(&c)).then_some(i))
                .collect();
            if idx.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("no cells with annotation {:?}", name),
                ));
            }
            Ok(Some(idx))
        }
        PerturbScopeBody::Cluster { cluster_id } => {
            let idx: Vec<usize> = st
                .clusters
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (*c == *cluster_id).then_some(i))
                .collect();
            if idx.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("no cells with cluster id {}", cluster_id),
                ));
            }
            Ok(Some(idx))
        }
    }
}

fn build_perturb_targets(
    st: &AppDataset,
    body: &PerturbPreviewBody,
    n_obs: usize,
) -> Result<Vec<PerturbTarget>, (StatusCode, String)> {
    let gene = body.gene.trim();
    if gene.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "gene is empty".into()));
    }
    let cell_indices = cell_indices_for_perturb_scope(st, &body.scope, n_obs)?;
    Ok(vec![PerturbTarget {
        gene: gene.to_string(),
        desired_expr: body.desired_expr,
        cell_indices,
    }])
}

fn mask_from_cell_selection(n_obs: usize, cell_indices: Option<Vec<usize>>) -> Vec<bool> {
    match cell_indices {
        None => vec![true; n_obs],
        Some(idx) => {
            let mut m = vec![false; n_obs];
            for i in idx {
                if i < n_obs {
                    m[i] = true;
                }
            }
            m
        }
    }
}

fn count_true_mask(m: &[bool]) -> usize {
    m.iter().filter(|&&x| x).count()
}

fn mean_splash_col(splash: &GeneMatrix, col: usize, mask: &[bool]) -> f32 {
    let colv = splash.data.column(col);
    let mut s = 0.0f64;
    let mut n = 0usize;
    for (i, &on) in mask.iter().enumerate() {
        if on && i < colv.len() {
            s += f64::from(colv[i]);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { (s / n as f64) as f32 }
}

fn plain_modulator_name(s: &str) -> &str {
    s.strip_prefix("beta_").unwrap_or(s)
}

fn splash_directed_bfs_path(
    adj: &HashMap<String, Vec<(String, f32)>>,
    start: &str,
    goal: &str,
    max_steps: usize,
) -> Option<Vec<String>> {
    if start == goal {
        return Some(vec![start.to_string()]);
    }
    let mut q = VecDeque::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut parent: HashMap<String, String> = HashMap::new();
    q.push_back((start.to_string(), 0usize));
    visited.insert(start.to_string());
    while let Some((u, d)) = q.pop_front() {
        if u == goal {
            let mut out = vec![goal.to_string()];
            let mut cur = goal.to_string();
            while cur != start {
                let p = parent.get(&cur)?.clone();
                out.push(p.clone());
                cur = p;
            }
            out.reverse();
            return Some(out);
        }
        if d >= max_steps {
            continue;
        }
        for (v, _) in adj.get(&u).map(|x| x.as_slice()).unwrap_or(&[]) {
            if visited.contains(v) {
                continue;
            }
            visited.insert(v.clone());
            parent.insert(v.clone(), u.clone());
            q.push_back((v.clone(), d + 1));
        }
    }
    None
}

fn undirected_adj_from_edges(edges: &[(String, String, f32)]) -> HashMap<String, Vec<String>> {
    let mut g: HashMap<String, Vec<String>> = HashMap::new();
    for (a, b, _) in edges {
        g.entry(a.clone()).or_default().push(b.clone());
        g.entry(b.clone()).or_default().push(a.clone());
    }
    g
}

fn nodes_within_undirected_hops(
    sources: &HashSet<String>,
    und: &HashMap<String, Vec<String>>,
    max_hops: u32,
) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    let mut q = VecDeque::new();
    for s in sources {
        if out.insert(s.clone()) {
            q.push_back((s.clone(), 0u32));
        }
    }
    while let Some((u, h)) = q.pop_front() {
        if h < max_hops {
            for v in und.get(&u).map(|x| x.as_slice()).unwrap_or(&[]) {
                if out.insert(v.clone()) {
                    q.push_back((v.clone(), h + 1));
                }
            }
        }
    }
    out
}

fn node_incident_strength(edges: &[(String, String, f32)]) -> HashMap<String, f32> {
    let mut m: HashMap<String, f32> = HashMap::new();
    for (a, b, w) in edges {
        let t = w.abs();
        *m.entry(a.clone()).or_insert(0.0) += t;
        *m.entry(b.clone()).or_insert(0.0) += t;
    }
    m
}

fn trim_nodes_to_budget(
    candidates: HashSet<String>,
    must_keep: HashSet<String>,
    max_n: usize,
    strength: &HashMap<String, f32>,
) -> HashSet<String> {
    if candidates.len() <= max_n {
        return candidates;
    }
    let mut kept: HashSet<String> = must_keep.intersection(&candidates).cloned().collect();
    let mut rest: Vec<String> = candidates.difference(&kept).cloned().collect();
    rest.sort_by(|a, b| {
        let wa = strength.get(a).copied().unwrap_or(0.0);
        let wb = strength.get(b).copied().unwrap_or(0.0);
        wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
    });
    for n in rest {
        if kept.len() >= max_n {
            break;
        }
        kept.insert(n);
    }
    kept
}

#[derive(Serialize)]
struct SplashNetworkNode {
    id: String,
    on_path: bool,
    role: String,
}

#[derive(Serialize)]
struct SplashNetworkLink {
    source: String,
    target: String,
    weight: f32,
    abs_weight: f32,
    /// Mean β for this modulator in the **target** gene's betadata row, over the same cells as splash scope.
    #[serde(skip_serializing_if = "Option::is_none")]
    beta_mean: Option<f32>,
}

#[derive(Serialize)]
struct SplashNetworkResponse {
    gene_a: String,
    gene_b: String,
    n_cells_used: usize,
    path: Option<Vec<String>>,
    path_found: bool,
    nodes: Vec<SplashNetworkNode>,
    links: Vec<SplashNetworkLink>,
    message: Option<String>,
    surround_hops: u32,
    max_nodes: usize,
}

fn compute_splash_network(
    rt: &PerturbRuntime,
    gene_a: &str,
    gene_b: &str,
    mask: &[bool],
    surround_hops: u32,
    max_nodes: usize,
    progress: Option<Arc<AtomicU32>>,
) -> Result<SplashNetworkResponse, String> {
    let n_used = count_true_mask(mask);
    if n_used == 0 {
        return Err("no cells in scope".into());
    }
    if !rt.bb.data.contains_key(gene_b) {
        return Err(format!(
            "gene_b {:?} is not a trained target (no betabase model); pick a gene with *_betadata.feather",
            gene_b
        ));
    }
    if let Some(p) = progress.as_ref() {
        p.store(5, Ordering::Relaxed);
    }
    let gex_f32 = rt.gene_mtx.mapv(|v| v as f32);
    let gex_df = GeneMatrix::new(gex_f32, rt.gene_names.clone());
    let splashed = compute_splash_all_progress(
        &rt.bb,
        &rt.rw_ligands_init,
        &rt.rw_tfligands_init,
        &gex_df,
        rt.perturb_cfg.beta_scale_factor as f32,
        rt.perturb_cfg.beta_cap.map(|c| c as f32),
        progress.as_deref(),
        None,
    )
    .expect("compute_splash_all_progress without cancel must return Some");
    if let Some(p) = progress.as_ref() {
        p.store(760, Ordering::Relaxed);
    }
    let n_targets = splashed.len().max(1);
    let tgt_step = (n_targets / 22).max(1);
    let mut max_abs = 0.0f32;
    let mut raw_edges: Vec<(String, String, f32)> = Vec::new();
    for (ti, (target, mat)) in splashed.iter().enumerate() {
        for (j, mod_col) in mat.col_names.iter().enumerate() {
            let w = mean_splash_col(mat, j, mask);
            max_abs = max_abs.max(w.abs());
            let src = plain_modulator_name(mod_col).to_string();
            raw_edges.push((src, target.clone(), w));
        }
        if let Some(p) = progress.as_ref() {
            if ti % tgt_step == 0 || ti + 1 == n_targets {
                let v = 760u32 + ((ti as u32 + 1) * 120 / n_targets as u32);
                p.store(v.min(880), Ordering::Relaxed);
            }
        }
    }
    let eps = (max_abs * 1e-5f32).max(1e-12f32);
    let edges: Vec<(String, String, f32)> = raw_edges
        .into_iter()
        .filter(|(_, _, w)| w.abs() > eps)
        .collect();
    let strength = node_incident_strength(&edges);
    let mut adj: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    for (a, b, w) in &edges {
        adj.entry(a.clone()).or_default().push((b.clone(), *w));
    }
    let path = splash_directed_bfs_path(&adj, gene_a, gene_b, 16);
    let path_found = path.is_some();
    let mut path_set: HashSet<String> = HashSet::new();
    if let Some(p) = &path {
        for n in p {
            path_set.insert(n.clone());
        }
    } else {
        path_set.insert(gene_a.to_string());
        path_set.insert(gene_b.to_string());
    }
    let und = undirected_adj_from_edges(&edges);
    let expanded = nodes_within_undirected_hops(&path_set, &und, surround_hops);
    let must_keep: HashSet<String> = path_set.intersection(&expanded).cloned().collect();
    let final_nodes = trim_nodes_to_budget(expanded, must_keep, max_nodes, &strength);
    let mut links: Vec<SplashNetworkLink> = Vec::new();
    for (a, b, w) in &edges {
        if final_nodes.contains(a) && final_nodes.contains(b) {
            links.push(SplashNetworkLink {
                source: a.clone(),
                target: b.clone(),
                weight: *w,
                abs_weight: w.abs(),
                beta_mean: None,
            });
        }
    }
    links.sort_by(|x, y| {
        y.abs_weight
            .partial_cmp(&x.abs_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(p) = progress.as_ref() {
        p.store(910, Ordering::Relaxed);
    }
    let path_ids: HashSet<String> = path
        .as_ref()
        .map(|p| p.iter().cloned().collect())
        .unwrap_or_default();
    let mut nodes_out: Vec<SplashNetworkNode> = final_nodes
        .iter()
        .map(|id| {
            let on_path = path_ids.contains(id);
            let role = if id == gene_a {
                "source"
            } else if id == gene_b {
                "sink"
            } else if on_path {
                "path"
            } else {
                "context"
            };
            SplashNetworkNode {
                id: id.clone(),
                on_path,
                role: role.into(),
            }
        })
        .collect();
    nodes_out.sort_by(|a, b| a.id.cmp(&b.id));
    let message = if !path_found {
        Some(format!(
            "No directed splash path from {} to {} within 16 hops (showing {}-hop neighborhood around {{A,B}} in the undirected derivative graph).",
            gene_a, gene_b, surround_hops
        ))
    } else {
        None
    };
    Ok(SplashNetworkResponse {
        gene_a: gene_a.to_string(),
        gene_b: gene_b.to_string(),
        n_cells_used: n_used,
        path,
        path_found,
        nodes: nodes_out,
        links,
        message,
        surround_hops,
        max_nodes,
    })
}

fn enrich_splash_response_with_betadata(
    ds: &AppDataset,
    mask: &[bool],
    resp: &mut SplashNetworkResponse,
) {
    let Some(ref bd) = ds.betadata_dir else {
        return;
    };
    let obs = ds.obs_names.as_ref();
    let cluster_keys = ds.betadata_cluster_keys.as_ref();
    let cell_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (*m).then_some(i))
        .collect();
    if cell_indices.is_empty() {
        return;
    }
    let dir = bd.to_string_lossy().to_string();

    let mut by_target: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, link) in resp.links.iter().enumerate() {
        by_target.entry(link.target.clone()).or_default().push(i);
    }

    for (target, idxs) in by_target {
        let path = betadata_feather_path(&dir, &target);
        if !path.is_file() {
            continue;
        }
        let path_s = path.to_string_lossy().into_owned();
        let mods: Vec<String> = idxs
            .iter()
            .map(|&li| resp.links[li].source.clone())
            .collect();
        let means = match betadata_feather_modulator_beta_means_for_cells(
            &path_s,
            &mods,
            obs.as_ref(),
            cluster_keys.as_ref(),
            &cell_indices,
        ) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("splash betadata enrich target {}: {}", target, e);
                continue;
            }
        };
        for (j, &link_idx) in idxs.iter().enumerate() {
            if let Some(m) = means.get(j).copied().flatten() {
                resp.links[link_idx].beta_mean = Some(m as f32);
            }
        }
    }
}

#[derive(Deserialize)]
struct SplashNetworkBody {
    gene_a: String,
    gene_b: String,
    #[serde(default)]
    scope: PerturbScopeBody,
    #[serde(default = "default_splash_surround_hops")]
    surround_hops: u32,
    #[serde(default = "default_splash_max_nodes")]
    max_nodes: usize,
}

fn default_splash_surround_hops() -> u32 {
    1
}

fn default_splash_max_nodes() -> usize {
    24
}

#[derive(Serialize)]
struct SplashProgressJson {
    active: bool,
    permille: u32,
}

async fn api_splash_progress(State(state): State<SharedState>) -> Json<SplashProgressJson> {
    let st = state.read().await;
    Json(SplashProgressJson {
        active: st.splash_job_active.load(Ordering::Relaxed),
        permille: st.splash_job_progress_permille.load(Ordering::Relaxed),
    })
}

async fn api_perturb_splash_network(
    State(state): State<SharedState>,
    Json(body): Json<SplashNetworkBody>,
) -> Result<Json<SplashNetworkResponse>, (StatusCode, String)> {
    let gene_a = body.gene_a.trim().to_string();
    let gene_b = body.gene_b.trim().to_string();
    if gene_a.is_empty() || gene_b.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "gene_a and gene_b required".into()));
    }
    if gene_a == gene_b {
        return Err((
            StatusCode::BAD_REQUEST,
            "gene_a and gene_b must differ".into(),
        ));
    }
    let surround_hops = body.surround_hops.min(4);
    let max_nodes = body.max_nodes.clamp(6, 64);
    let (rt, mask, splash_prog, splash_active) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let rt = Arc::clone(perturb_runtime_or_status(ds)?);
        let n_obs = ds.obs_names.len();
        let cell_sel = cell_indices_for_perturb_scope(ds, &body.scope, n_obs)?;
        let mask = mask_from_cell_selection(n_obs, cell_sel);
        let n_used = count_true_mask(&mask);
        if n_used == 0 {
            return Err((StatusCode::BAD_REQUEST, "no cells in scope".into()));
        }
        let vn: HashSet<&str> = rt.gene_names.iter().map(|s| s.as_str()).collect();
        if !vn.contains(gene_a.as_str()) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("gene_a {:?} not in model var_names", gene_a),
            ));
        }
        if !vn.contains(gene_b.as_str()) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("gene_b {:?} not in model var_names", gene_b),
            ));
        }
        (
            rt,
            mask,
            Arc::clone(&st.splash_job_progress_permille),
            Arc::clone(&st.splash_job_active),
        )
    };
    splash_active.store(true, Ordering::SeqCst);
    splash_prog.store(0, Ordering::Relaxed);
    let mask_for_betadata = mask.clone();
    let ga = gene_a.clone();
    let gb = gene_b.clone();
    let prog_for_block = Arc::clone(&splash_prog);
    let block_result = tokio::task::spawn_blocking(move || {
        compute_splash_network(
            &*rt,
            &ga,
            &gb,
            &mask,
            surround_hops,
            max_nodes,
            Some(prog_for_block),
        )
    })
    .await;
    let mut res = match block_result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            splash_active.store(false, Ordering::SeqCst);
            splash_prog.store(0, Ordering::Relaxed);
            return Err((StatusCode::BAD_REQUEST, e));
        }
        Err(e) => {
            splash_active.store(false, Ordering::SeqCst);
            splash_prog.store(0, Ordering::Relaxed);
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };
    splash_prog.store(940, Ordering::Relaxed);
    {
        let st = state.read().await;
        if let Ok(ds) = require_dataset(&st) {
            enrich_splash_response_with_betadata(ds, &mask_for_betadata, &mut res);
        }
    }
    splash_prog.store(1000, Ordering::Relaxed);
    splash_active.store(false, Ordering::SeqCst);
    Ok(Json(res))
}

async fn api_perturb_preview(
    State(state): State<SharedState>,
    Json(body): Json<PerturbPreviewBody>,
) -> Result<Response, (StatusCode, String)> {
    let n_propagation = body.n_propagation;
    let (
        n_obs,
        targets,
        gj,
        rt,
        job_p,
        job_active,
        job_msg,
        cancel,
        n_vars,
        adata_path,
        foyer,
        epoch,
    ) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let rt = perturb_runtime_or_status(ds)?;
        let n_obs = ds.obs_names.len();
        let targets = build_perturb_targets(ds, &body, n_obs)?;
        let gene = targets[0].gene.clone();
        if !rt.gene_names.iter().any(|g| g == &gene) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("gene {:?} not in model var_names", gene),
            ));
        }
        let gj = rt
            .gene_names
            .iter()
            .position(|g| g == &gene)
            .ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal: gene index missing".into(),
                )
            })?;
        let n_vars = ds.n_vars;
        let adata_path = ds.adata_path.to_string_lossy().into_owned();
        let epoch = st.dataset_cache_epoch.load(Ordering::SeqCst);
        let foyer = Arc::clone(&st.foyer_caches);
        (
            n_obs,
            targets,
            gj,
            Arc::clone(rt),
            Arc::clone(&st.perturb_job_progress_permille),
            Arc::clone(&st.perturb_job_active),
            Arc::clone(&st.perturb_progress_message),
            Arc::clone(&st.perturb_job_cancel),
            n_vars,
            adata_path,
            foyer,
            epoch,
        )
    };
    cancel.store(false, Ordering::SeqCst);
    let cfg = perturb_cfg_for_request(&rt.perturb_cfg, n_propagation);
    job_p.store(0, Ordering::Relaxed);
    job_active.store(true, Ordering::Relaxed);
    if let Ok(mut m) = job_msg.lock() {
        *m = "GRN perturbation…".into();
    }
    let job_active_move = job_active.clone();
    let _guard = PerturbJobGuard(job_active_move);
    let cache_key = foyer_perturb_cache::grn_perturb_cache_key(
        epoch,
        false,
        &adata_path,
        n_obs,
        n_vars,
        &targets,
        &cfg,
    );
    let pr = cached_grn_perturb_result(
        foyer.as_ref(),
        cache_key,
        n_obs,
        rt.gene_names.len(),
        Arc::clone(&rt),
        targets,
        cfg,
        Arc::clone(&job_p),
        Arc::clone(&job_msg),
        Arc::clone(&cancel),
    )
    .await
    .map_err(|_| {
        job_p.store(0, Ordering::Relaxed);
        (StatusCode::REQUEST_TIMEOUT, "Perturbation cancelled".into())
    })?;
    job_p.store(1000, Ordering::Relaxed);
    let vec: Vec<f32> = pr.delta.column(gj).iter().map(|x| *x as f32).collect();
    if vec.len() != n_obs {
        job_p.store(0, Ordering::Relaxed);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "perturbation output length mismatch".into(),
        ));
    }
    schedule_perturb_progress_permille_clear(Arc::clone(&job_p));
    Ok(binary_response(f32_vec_to_le_bytes(&vec)))
}

async fn api_perturb_export_feather(
    State(state): State<SharedState>,
    Json(body): Json<PerturbPreviewBody>,
) -> Result<Response, (StatusCode, String)> {
    let n_propagation = body.n_propagation;
    let fname_gene = body.gene.trim().to_string();
    let (targets, rt, job_p, job_active, job_msg, cancel, n_obs, n_vars, adata_path, foyer, epoch) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let rt = perturb_runtime_or_status(ds)?;
        let n_obs = ds.obs_names.len();
        let targets = build_perturb_targets(ds, &body, n_obs)?;
        let gene = targets[0].gene.clone();
        if !rt.gene_names.iter().any(|g| g == &gene) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("gene {:?} not in model var_names", gene),
            ));
        }
        let n_vars = ds.n_vars;
        let adata_path = ds.adata_path.to_string_lossy().into_owned();
        let epoch = st.dataset_cache_epoch.load(Ordering::SeqCst);
        let foyer = Arc::clone(&st.foyer_caches);
        (
            targets,
            Arc::clone(rt),
            Arc::clone(&st.perturb_job_progress_permille),
            Arc::clone(&st.perturb_job_active),
            Arc::clone(&st.perturb_progress_message),
            Arc::clone(&st.perturb_job_cancel),
            n_obs,
            n_vars,
            adata_path,
            foyer,
            epoch,
        )
    };
    cancel.store(false, Ordering::SeqCst);
    let cfg = perturb_cfg_for_request(&rt.perturb_cfg, n_propagation);
    job_p.store(0, Ordering::Relaxed);
    job_active.store(true, Ordering::Relaxed);
    if let Ok(mut m) = job_msg.lock() {
        *m = "GRN perturbation · export simulated expression…".into();
    }
    let job_active_move = job_active.clone();
    let _guard = PerturbJobGuard(job_active_move);
    let cache_key = foyer_perturb_cache::grn_perturb_cache_key(
        epoch,
        false,
        &adata_path,
        n_obs,
        n_vars,
        &targets,
        &cfg,
    );
    let pr = cached_grn_perturb_result(
        foyer.as_ref(),
        cache_key,
        n_obs,
        rt.gene_names.len(),
        Arc::clone(&rt),
        targets,
        cfg,
        Arc::clone(&job_p),
        Arc::clone(&job_msg),
        Arc::clone(&cancel),
    )
    .await
    .map_err(|_| {
        job_p.store(0, Ordering::Relaxed);
        (StatusCode::REQUEST_TIMEOUT, "Perturbation cancelled".into())
    })?;
    job_p.store(1000, Ordering::Relaxed);
    let obs_names = rt.obs_names.clone();
    let gene_names = rt.gene_names.clone();
    let simulated = pr.simulated;
    let bytes_result = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ()> {
        let mut buf = Vec::new();
        space_trav_lr_rust::betadata::write_betadata_feather_to_writer(
            &mut buf,
            "CellID",
            &obs_names,
            &gene_names,
            &simulated,
        )
        .map_err(|_| ())?;
        Ok(buf)
    })
    .await
    .map_err(|e| {
        job_p.store(0, Ordering::Relaxed);
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;
    let bytes = bytes_result.map_err(|_| {
        job_p.store(0, Ordering::Relaxed);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "feather export failed".into(),
        )
    })?;
    if bytes.is_empty() {
        job_p.store(0, Ordering::Relaxed);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "feather export produced no bytes".into(),
        ));
    }
    schedule_perturb_progress_permille_clear(Arc::clone(&job_p));
    let filename = perturb_export_feather_filename(&fname_gene);
    Ok(feather_download_response(bytes, &filename))
}

fn default_transition_neighbors() -> usize {
    150
}

fn default_transition_temperature() -> f64 {
    0.05
}

fn default_transition_grid_scale() -> f64 {
    1.0
}

fn default_transition_vector_scale() -> f64 {
    0.85
}

fn highlight_cell_type_keep_mask(
    n_obs: usize,
    highlight_names: &[String],
    categories: &[String],
    codes_bin: &[u8],
) -> Result<Vec<bool>, (StatusCode, String)> {
    let codes: Vec<u16> = codes_bin
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    if codes.len() != n_obs {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "cell_type codes length mismatch".into(),
        ));
    }
    let mut allowed = HashSet::new();
    for name in highlight_names {
        if let Some(i) = categories.iter().position(|c| c == name) {
            allowed.insert(i as u16);
        }
    }
    if allowed.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "highlight_cell_types did not match any category in cell_type_column".into(),
        ));
    }
    Ok(codes
        .into_iter()
        .map(|c| c != u16::MAX && allowed.contains(&c))
        .collect())
}

fn default_full_graph_max_cells() -> usize {
    4096
}

fn tmp_umap_signature_svg_path(label: &str) -> PathBuf {
    let safe: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(48)
        .collect();
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let name = format!("spacetravlr_umap_signature_{safe}_{ms}.svg");
    #[cfg(unix)]
    {
        PathBuf::from("/tmp").join(name)
    }
    #[cfg(not(unix))]
    {
        std::env::temp_dir().join(name)
    }
}

fn tmp_umap_quiver_svg_path(gene: &str) -> PathBuf {
    let safe: String = gene
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let name = format!("spacetravlr_umap_quiver_{safe}_{ms}.svg");
    #[cfg(unix)]
    {
        PathBuf::from("/tmp").join(name)
    }
    #[cfg(not(unix))]
    {
        std::env::temp_dir().join(name)
    }
}

fn svg_escape_text(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            '"' => "&quot;".to_string(),
            _ => c.to_string(),
        })
        .collect()
}

/// Matches default **Quiver display** sliders in `web/spatial_viewer/src/main.ts` (vis 185%, head 28%, stride 1).
fn write_umap_quiver_svg(path: &Path, grid: &TransitionGrid, title: &str) -> std::io::Result<()> {
    use std::io::Write;
    const VIS_SCALE: f64 = 1.85;
    const HEAD_FRAC: f64 = 0.28_f64;
    const STRIDE: usize = 1;
    const STROKE_SHAFT_FRAC: f64 = 0.0020;
    const STROKE_HEAD_FRAC: f64 = 0.0026;

    let nx = grid.grid_x.len();
    let ny = grid.grid_y.len();
    if nx == 0 || ny == 0 || grid.vectors.len() != nx * ny {
        return Ok(());
    }

    #[derive(Clone, Copy)]
    struct Seg {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        head: bool,
    }
    let mut segs: Vec<Seg> = Vec::new();

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut bump = |x: f64, y: f64| {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    };

    for ix in (0..nx).step_by(STRIDE) {
        let gx = grid.grid_x[ix];
        for iy in (0..ny).step_by(STRIDE) {
            let gy = grid.grid_y[iy];
            let k = ix * ny + iy;
            let u = grid.vectors[k][0] * VIS_SCALE;
            let v = grid.vectors[k][1] * VIS_SCALE;
            let len = (u * u + v * v).sqrt();
            if len < 1e-12 {
                continue;
            }
            let dx = u / len;
            let dy = v / len;
            let hl = (len * HEAD_FRAC).min(len * 0.98);
            let tx = gx + u;
            let ty = gy + v;
            let bx = tx - hl * dx;
            let by = ty - hl * dy;
            let px = -dy;
            let py = dx;
            let hw = hl * 0.48;
            let lx = bx + hw * px;
            let ly = by + hw * py;
            let rx = bx - hw * px;
            let ry = by - hw * py;

            bump(gx, gy);
            bump(bx, by);
            bump(tx, ty);
            bump(lx, ly);
            bump(rx, ry);

            segs.push(Seg {
                x1: gx,
                y1: gy,
                x2: bx,
                y2: by,
                head: false,
            });
            segs.push(Seg {
                x1: lx,
                y1: ly,
                x2: tx,
                y2: ty,
                head: true,
            });
            segs.push(Seg {
                x1: rx,
                y1: ry,
                x2: tx,
                y2: ty,
                head: true,
            });
        }
    }

    if segs.is_empty() {
        return Ok(());
    }

    let span = (max_x - min_x).max(max_y - min_y).max(1e-9);
    let pad = 0.06 * span;
    min_x -= pad;
    max_x += pad;
    min_y -= pad;
    max_y += pad;
    let w = (max_x - min_x).max(1e-9);
    let h = (max_y - min_y).max(1e-9);
    let stroke_shaft = STROKE_SHAFT_FRAC * w.max(h);
    let stroke_head = STROKE_HEAD_FRAC * w.max(h);

    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(
        f,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x} {min_y} {w} {h}" width="960" height="960">"#,
    )?;
    writeln!(f, r#"<title>{}</title>"#, svg_escape_text(title))?;
    writeln!(
        f,
        r#"<rect x="{min_x}" y="{min_y}" width="{w}" height="{h}" fill='#12161c'/>"#
    )?;
    writeln!(
        f,
        r#"<g fill='none' stroke-linecap='round' opacity='0.92'>"#
    )?;
    for s in &segs {
        let sw = if s.head { stroke_head } else { stroke_shaft };
        writeln!(
            f,
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke='#eb6234' stroke-width="{sw}"/>"#,
            s.x1, s.y1, s.x2, s.y2,
        )?;
    }
    writeln!(f, "</g></svg>")?;
    f.flush()?;
    Ok(())
}

#[derive(Deserialize)]
struct UmapTransitionBody {
    #[serde(flatten)]
    perturb: PerturbPreviewBody,
    #[serde(default = "default_transition_neighbors")]
    n_neighbors: usize,
    #[serde(default = "default_transition_temperature")]
    temperature: f64,
    #[serde(default = "default_true")]
    remove_null: bool,
    /// `normalize` in SpaceTravLR `Cartography.plot_umap_quiver` (default `False` in Python).
    #[serde(default)]
    unit_directions: bool,
    /// When `true` with `highlight_cell_types`, cells outside those types get δ=0 before the field (Python `limit_clusters` + `highlight_clusters`).
    #[serde(default)]
    limit_clusters: bool,
    #[serde(default)]
    highlight_cell_types: Vec<String>,
    #[serde(default = "default_transition_grid_scale")]
    grid_scale: f64,
    #[serde(default = "default_transition_vector_scale")]
    vector_scale: f64,
    #[serde(default = "default_one_f64")]
    delta_rescale: f64,
    #[serde(default)]
    magnitude_threshold: f64,
    #[serde(default)]
    use_full_graph: bool,
    #[serde(default = "default_full_graph_max_cells")]
    full_graph_max_cells: usize,
    #[serde(default)]
    include_cell_vectors: bool,
    /// If true, skip GRN perturbation: δ = target_expr − expr on the chosen gene (and scope) only.
    #[serde(default)]
    quick_ko_sanity: bool,
    /// When true, write a quiver-only SVG under `/tmp` (Unix) or the system temp dir and return the path in `svg_export_path`.
    #[serde(default)]
    export_svg: bool,
}

/// Single-gene “virtual KO/OE”: delta[i,g] = target − expr[i,g] for affected cells; other genes 0.
fn delta_single_gene_to_target(
    gene_mtx: &Array2<f64>,
    gene_col: usize,
    cell_indices: Option<&[usize]>,
    target_expr: f64,
) -> Array2<f64> {
    let (n, g) = gene_mtx.dim();
    let mut delta = Array2::<f64>::zeros((n, g));
    if gene_col >= g {
        return delta;
    }
    match cell_indices {
        None => {
            for i in 0..n {
                delta[[i, gene_col]] = target_expr - gene_mtx[[i, gene_col]];
            }
        }
        Some(idxs) => {
            for &i in idxs {
                if i < n {
                    delta[[i, gene_col]] = target_expr - gene_mtx[[i, gene_col]];
                }
            }
        }
    }
    delta
}

fn default_true() -> bool {
    true
}

fn default_one_f64() -> f64 {
    1.0
}

#[derive(Serialize)]
struct UmapFieldResponse {
    nx: usize,
    ny: usize,
    grid_x: Vec<f64>,
    grid_y: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cell_u: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cell_v: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    svg_export_path: Option<String>,
}

async fn api_perturb_umap_field(
    State(state): State<SharedState>,
    Json(body): Json<UmapTransitionBody>,
) -> Result<Json<UmapFieldResponse>, (StatusCode, String)> {
    let export_svg = body.export_svg;
    let gene_for_svg = body.perturb.gene.trim().to_string();
    let n_propagation = body.perturb.n_propagation;
    let (
        umap_pts,
        highlight_keep,
        params,
        include_cv,
        quick,
        gj,
        targets,
        rt,
        job_p,
        job_active,
        job_msg,
        cancel,
        n_obs,
        n_vars,
        adata_path,
        foyer,
        epoch,
    ) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let rt = perturb_runtime_or_status(ds)?;
        let Some(umap_f32) = ds.umap_f32.as_ref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "This dataset has no 2D UMAP in obsm (X_umap / umap).".into(),
            ));
        };
        let n_obs = ds.obs_names.len();
        if n_obs > MAX_UMAP_TRANSITION_CELLS {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "n_obs {} exceeds transition-field limit {} (subset cells or use a smaller dataset)",
                    n_obs, MAX_UMAP_TRANSITION_CELLS
                ),
            ));
        }
        if umap_f32.len() != n_obs * 2 {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "UMAP coordinate length mismatch".into(),
            ));
        }
        let targets = build_perturb_targets(ds, &body.perturb, n_obs)?;
        let gene = targets[0].gene.clone();
        if !rt.gene_names.iter().any(|g| g == &gene) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("gene {:?} not in model var_names", gene),
            ));
        }
        let gj = rt
            .gene_names
            .iter()
            .position(|g| g == &gene)
            .ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal: gene index missing".into(),
                )
            })?;
        let umap_pts: Vec<[f64; 2]> = umap_f32
            .chunks_exact(2)
            .map(|c| [c[0] as f64, c[1] as f64])
            .collect();
        let highlight_keep = if body.limit_clusters {
            if body.highlight_cell_types.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "limit_clusters requires non-empty highlight_cell_types (SpaceTravLR plot_umap_quiver)"
                        .into(),
                ));
            }
            let bin = ds.cell_type_codes_bin.as_ref().ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "limit_clusters requires cell_type_column in this dataset (see /api/meta)"
                        .into(),
                )
            })?;
            Some(highlight_cell_type_keep_mask(
                n_obs,
                &body.highlight_cell_types,
                ds.cell_type_categories.as_ref(),
                bin.as_ref(),
            )?)
        } else {
            None
        };

        let params = TransitionUmapParams {
            n_neighbors: body.n_neighbors.clamp(3, 500),
            temperature: body.temperature.max(1e-6),
            remove_null: body.remove_null,
            unit_directions: body.unit_directions,
            grid_scale: body.grid_scale.max(1e-6),
            vector_scale: body.vector_scale.max(1e-9),
            delta_rescale: body.delta_rescale,
            magnitude_threshold: body.magnitude_threshold.max(0.0),
            use_full_graph: body.use_full_graph,
            full_graph_max_cells: body.full_graph_max_cells.max(64).min(8192),
        };
        let include_cv = body.include_cell_vectors;
        let quick = body.quick_ko_sanity;
        let n_vars = ds.n_vars;
        let adata_path = ds.adata_path.to_string_lossy().into_owned();
        let epoch = st.dataset_cache_epoch.load(Ordering::SeqCst);
        let foyer = Arc::clone(&st.foyer_caches);
        (
            umap_pts,
            highlight_keep,
            params,
            include_cv,
            quick,
            gj,
            targets,
            Arc::clone(rt),
            Arc::clone(&st.perturb_job_progress_permille),
            Arc::clone(&st.perturb_job_active),
            Arc::clone(&st.perturb_progress_message),
            Arc::clone(&st.perturb_job_cancel),
            n_obs,
            n_vars,
            adata_path,
            foyer,
            epoch,
        )
    };

    cancel.store(false, Ordering::SeqCst);
    let cfg = perturb_cfg_for_request(&rt.perturb_cfg, n_propagation);
    job_p.store(0, Ordering::Relaxed);
    job_active.store(true, Ordering::Relaxed);
    if let Ok(mut m) = job_msg.lock() {
        *m = if quick {
            "UMAP field (quick δ)…".into()
        } else {
            "UMAP transition field…".into()
        };
    }
    let job_active_move = job_active.clone();
    let _guard = PerturbJobGuard(job_active_move);

    let grn_key = foyer_perturb_cache::grn_perturb_cache_key(
        epoch,
        quick,
        &adata_path,
        n_obs,
        n_vars,
        &targets,
        &cfg,
    );
    let grid_key = foyer_perturb_cache::umap_grid_cache_key(
        epoch,
        grn_key.fingerprint,
        body.limit_clusters,
        &body.highlight_cell_types,
        &params,
        include_cv,
    );

    let mut delta = if quick {
        job_p.store(20, Ordering::Relaxed);
        if let Ok(mut m) = job_msg.lock() {
            *m = "Local expression delta…".into();
        }
        job_p.store(120, Ordering::Relaxed);
        let rt_q = Arc::clone(&rt);
        let targets_q = targets.clone();
        let d = tokio::task::spawn_blocking(move || {
            delta_single_gene_to_target(
                &rt_q.gene_mtx,
                gj,
                targets_q[0].cell_indices.as_deref(),
                targets_q[0].desired_expr,
            )
        })
        .await
        .map_err(|e| {
            job_p.store(0, Ordering::Relaxed);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;
        job_p.store(450, Ordering::Relaxed);
        d
    } else {
        if let Ok(mut m) = job_msg.lock() {
            *m = "GRN perturbation…".into();
        }
        let pr = cached_grn_perturb_result(
            foyer.as_ref(),
            grn_key,
            n_obs,
            rt.gene_names.len(),
            Arc::clone(&rt),
            targets.clone(),
            cfg.clone(),
            Arc::clone(&job_p),
            Arc::clone(&job_msg),
            Arc::clone(&cancel),
        )
        .await
        .map_err(|_| {
            job_p.store(0, Ordering::Relaxed);
            (StatusCode::REQUEST_TIMEOUT, "Perturbation cancelled".into())
        })?;
        job_p.store(940, Ordering::Relaxed);
        pr.delta
    };

    if let Some(ref keep) = highlight_keep {
        let nrows = delta.nrows();
        for i in 0..nrows {
            if i < keep.len() && !keep[i] {
                delta.row_mut(i).fill(0.0);
            }
        }
    }
    if let Ok(mut m) = job_msg.lock() {
        *m = "UMAP projection & grid…".into();
    }
    job_p.store(965, Ordering::Relaxed);

    let grid_payload_est = perturb_matrix_payload_est_bytes(n_obs, rt.gene_mtx.ncols());
    let grid = if should_use_foyer_perturb_cache(grid_payload_est) {
        let grid_cache = foyer.grid.clone();
        let rt_g = Arc::clone(&rt);
        let umap_pts_g = umap_pts.clone();
        let params_g = params.clone();
        let delta_g = delta.clone();
        let gentry = grid_cache
            .get_or_fetch(&grid_key, move || {
                let rt_g = Arc::clone(&rt_g);
                let umap_pts_g = umap_pts_g.clone();
                let params_g = params_g.clone();
                let delta_g = delta_g.clone();
                async move {
                    let g = tokio::task::spawn_blocking(move || {
                        compute_umap_transition_grid(
                            &rt_g.gene_mtx,
                            &delta_g,
                            &umap_pts_g,
                            &params_g,
                        )
                    })
                    .await
                    .map_err(|e| anyhow!("{e}"))?;
                    let enc =
                        foyer_perturb_cache::encode_umap_grid_blob(&transition_grid_to_blob(&g))
                            .map_err(|e| anyhow!("{e}"))?;
                    Ok::<Vec<u8>, anyhow::Error>(enc)
                }
            })
            .await
            .map_err(|e| {
                job_p.store(0, Ordering::Relaxed);
                (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))
            })?;

        transition_grid_from_blob(
            foyer_perturb_cache::decode_umap_grid_blob(gentry.value()).map_err(|e| {
                job_p.store(0, Ordering::Relaxed);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("grid cache decode: {e}"),
                )
            })?,
        )
        .map_err(|e| {
            job_p.store(0, Ordering::Relaxed);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("grid restore: {e}"),
            )
        })?
    } else {
        let rt_g = Arc::clone(&rt);
        let umap_pts_g = umap_pts.clone();
        let params_g = params.clone();
        let delta_g = delta.clone();
        tokio::task::spawn_blocking(move || {
            compute_umap_transition_grid(&rt_g.gene_mtx, &delta_g, &umap_pts_g, &params_g)
        })
        .await
        .map_err(|e| {
            job_p.store(0, Ordering::Relaxed);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?
    };

    job_p.store(1000, Ordering::Relaxed);
    schedule_perturb_progress_permille_clear(Arc::clone(&job_p));
    let nx = grid.grid_x.len();
    let ny = grid.grid_y.len();
    let u: Vec<f64> = grid.vectors.iter().map(|w| w[0]).collect();
    let v: Vec<f64> = grid.vectors.iter().map(|w| w[1]).collect();
    let (cell_u, cell_v) = if include_cv {
        let cu: Vec<f32> = grid.cell_vectors.iter().map(|w| w[0] as f32).collect();
        let cv: Vec<f32> = grid.cell_vectors.iter().map(|w| w[1] as f32).collect();
        (Some(cu), Some(cv))
    } else {
        (None, None)
    };
    let svg_export_path = if export_svg {
        let path = tmp_umap_quiver_svg_path(gene_for_svg.as_str());
        let title = format!("UMAP transition quiver · {}", gene_for_svg);
        match write_umap_quiver_svg(&path, &grid, &title) {
            Ok(()) => Some(path.display().to_string()),
            Err(e) => {
                tracing::warn!("failed to write UMAP quiver SVG: {:#}", e);
                None
            }
        }
    } else {
        None
    };
    Ok(Json(UmapFieldResponse {
        nx,
        ny,
        grid_x: grid.grid_x,
        grid_y: grid.grid_y,
        u,
        v,
        cell_u,
        cell_v,
        svg_export_path,
    }))
}

fn default_sig_n_knn() -> usize {
    30
}

fn default_sig_gradient_gain() -> f64 {
    2.0
}

#[derive(Deserialize)]
struct UmapSignatureBody {
    genes: Vec<String>,
    #[serde(default = "default_sig_n_knn")]
    n_knn: usize,
    #[serde(default = "default_transition_grid_scale")]
    grid_scale: f64,
    #[serde(default = "default_transition_vector_scale")]
    vector_scale: f64,
    #[serde(default)]
    magnitude_threshold: f64,
    #[serde(default = "default_sig_gradient_gain")]
    gradient_gain: f64,
    /// When true with `mask_perturb`, zero signature arrows where the perturbation quiver is zero (`VirtualTissue.signature2gradient`).
    #[serde(default)]
    mask_with_perturb_quiver: bool,
    /// If true with masking, use single-gene δ only (fast). If false, full GRN perturbation (slow).
    #[serde(default = "default_true")]
    mask_quick_ko: bool,
    #[serde(default)]
    mask_perturb: Option<PerturbPreviewBody>,
    #[serde(default)]
    export_svg: bool,
}

#[derive(Serialize)]
struct UmapSignatureFieldResponse {
    nx: usize,
    ny: usize,
    grid_x: Vec<f64>,
    grid_y: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    signature_per_cell: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    svg_export_path: Option<String>,
}

async fn api_umap_signature_field(
    State(state): State<SharedState>,
    Json(body): Json<UmapSignatureBody>,
) -> Result<Json<UmapSignatureFieldResponse>, (StatusCode, String)> {
    let export_svg = body.export_svg;
    let mut genes: Vec<String> = body
        .genes
        .iter()
        .filter_map(|g| {
            let t = g.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect();
    genes.truncate(MAX_SIGNATURE_GENES);
    if genes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "genes must list at least one non-empty symbol".into(),
        ));
    }
    if body.mask_with_perturb_quiver && body.mask_perturb.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "mask_with_perturb_quiver requires mask_perturb (gene, scope, etc.)".into(),
        ));
    }

    let (path, layer, vn, umap_pts, n_obs, sig_params, mask_pack, svg_label, foyer, epoch, n_vars) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let Some(umap_f32) = ds.umap_f32.as_ref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "This dataset has no 2D UMAP in obsm (X_umap / umap).".into(),
            ));
        };
        let n_obs = ds.obs_names.len();
        if n_obs > MAX_UMAP_TRANSITION_CELLS {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "n_obs {} exceeds UMAP field limit {}",
                    n_obs, MAX_UMAP_TRANSITION_CELLS
                ),
            ));
        }
        if umap_f32.len() != n_obs * 2 {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "UMAP coordinate length mismatch".into(),
            ));
        }
        let umap_pts: Vec<[f64; 2]> = umap_f32
            .chunks_exact(2)
            .map(|c| [c[0] as f64, c[1] as f64])
            .collect();
        let sig_params = SignatureUmapParams {
            n_knn: body.n_knn.clamp(3, 200),
            grid_scale: body.grid_scale.max(1e-6),
            vector_scale: body.vector_scale.max(1e-9),
            magnitude_threshold: body.magnitude_threshold.max(0.0),
            gradient_gain: body.gradient_gain,
        };
        let mask_pack = if body.mask_with_perturb_quiver {
            let rt = perturb_runtime_or_status(ds)?;
            let pb = body.mask_perturb.as_ref().unwrap();
            let targets = build_perturb_targets(ds, pb, n_obs)?;
            let gene = targets[0].gene.clone();
            if !rt.gene_names.iter().any(|g| g == &gene) {
                return Err((
                    StatusCode::NOT_FOUND,
                    format!("mask gene {:?} not in model var_names", gene),
                ));
            }
            let gj = rt
                .gene_names
                .iter()
                .position(|g| g == &gene)
                .ok_or_else(|| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "internal: mask gene index missing".into(),
                    )
                })?;
            let cfg = perturb_cfg_for_request(&rt.perturb_cfg, pb.n_propagation);
            Some((Arc::clone(rt), targets, gj, cfg, body.mask_quick_ko))
        } else {
            None
        };
        let svg_label = if genes.len() <= 3 {
            genes.join("+")
        } else {
            format!("{}+{}genes", genes[0], genes.len())
        };
        (
            ds.adata_path.clone(),
            ds.layer.clone(),
            ds.var_names.as_ref().clone(),
            umap_pts,
            n_obs,
            sig_params,
            mask_pack,
            svg_label,
            Arc::clone(&st.foyer_caches),
            st.dataset_cache_epoch.load(Ordering::SeqCst),
            ds.var_names.as_ref().len(),
        )
    };

    let pre_delta: Option<Array2<f64>> =
        if let Some((rt, targets, _gj, cfg, quick)) = mask_pack.as_ref() {
            if *quick {
                None
            } else {
                let adata_s = path.to_string_lossy();
                let cache_key = foyer_perturb_cache::grn_perturb_cache_key(
                    epoch,
                    false,
                    adata_s.as_ref(),
                    n_obs,
                    n_vars,
                    targets,
                    cfg,
                );
                let job_p = Arc::new(AtomicU32::new(0));
                let job_msg = Arc::new(Mutex::new(String::new()));
                let cancel = Arc::new(AtomicBool::new(false));
                let pr = cached_grn_perturb_result(
                    foyer.as_ref(),
                    cache_key,
                    n_obs,
                    rt.gene_names.len(),
                    Arc::clone(rt),
                    targets.clone(),
                    cfg.clone(),
                    job_p,
                    job_msg,
                    cancel,
                )
                .await
                .map_err(|_| (StatusCode::REQUEST_TIMEOUT, "Perturbation cancelled".into()))?;
                Some(pr.delta)
            }
        } else {
            None
        };

    let (grid, sig) =
        tokio::task::spawn_blocking(move || -> Result<(TransitionGrid, Vec<f64>), String> {
            let adata = open_adata(path.to_string_lossy().as_ref()).map_err(|e| e.to_string())?;
            let (mat, _) = expression_matrix_genes_subset(&adata, &layer, &genes, vn.as_ref())
                .map_err(|e| e.to_string())?;
            if mat.nrows() != n_obs {
                return Err(format!(
                    "expression rows {} != n_obs {}",
                    mat.nrows(),
                    n_obs
                ));
            }
            let sig = signature_sum_per_cell(&mat);
            let base_storage = if let Some((rt, targets, gj, cfg, quick)) = mask_pack {
                let delta = if let Some(d) = pre_delta {
                    d
                } else if quick {
                    delta_single_gene_to_target(
                        &rt.gene_mtx,
                        gj,
                        targets[0].cell_indices.as_deref(),
                        targets[0].desired_expr,
                    )
                } else {
                    let mut no_timings: Option<PerturbTimings> = None;
                    perturb_with_targets(
                        &rt.bb,
                        &rt.gene_mtx,
                        &rt.gene_names,
                        &rt.xy,
                        &rt.rw_ligands_init,
                        &rt.rw_tfligands_init,
                        &targets,
                        &cfg,
                        &rt.lr_radii,
                        None,
                        None,
                        None,
                        Some(&rt.baseline_splash_cache),
                        &mut no_timings,
                    )
                    .map_err(|_| "GRN perturbation failed (mask field)".to_string())?
                    .delta
                };
                let tparams = TransitionUmapParams::default();
                let tg = compute_umap_transition_grid(&rt.gene_mtx, &delta, &umap_pts, &tparams);
                Some(tg.vectors)
            } else {
                None
            };
            let base_ref = base_storage.as_deref();
            let grid = compute_signature_umap_grid(&umap_pts, &sig, base_ref, &sig_params);
            Ok((grid, sig))
        })
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let nx = grid.grid_x.len();
    let ny = grid.grid_y.len();
    let u: Vec<f64> = grid.vectors.iter().map(|w| w[0]).collect();
    let v: Vec<f64> = grid.vectors.iter().map(|w| w[1]).collect();
    let signature_per_cell: Vec<f32> = sig.iter().map(|&x| x as f32).collect();
    let svg_export_path = if export_svg {
        let path = tmp_umap_signature_svg_path(&svg_label);
        let title = format!("UMAP gene signature · {}", svg_label);
        match write_umap_quiver_svg(&path, &grid, &title) {
            Ok(()) => Some(path.display().to_string()),
            Err(e) => {
                tracing::warn!("failed to write UMAP signature SVG: {:#}", e);
                None
            }
        }
    } else {
        None
    };
    Ok(Json(UmapSignatureFieldResponse {
        nx,
        ny,
        grid_x: grid.grid_x,
        grid_y: grid.grid_y,
        u,
        v,
        signature_per_cell,
        svg_export_path,
    }))
}

// ── Perturbation summary statistics ──

#[derive(Serialize)]
struct PerturbSummaryResponse {
    gene: String,
    n_obs: usize,
    mean_delta: f64,
    max_abs_delta: f64,
    n_positive: usize,
    n_negative: usize,
    n_zero: usize,
    top_affected_genes: Vec<PerturbGeneEffect>,
}

#[derive(Serialize)]
struct PerturbGeneEffect {
    gene: String,
    mean_delta: f64,
    max_abs_delta: f64,
}

async fn api_perturb_summary(
    State(state): State<SharedState>,
    Json(body): Json<PerturbPreviewBody>,
) -> Result<Json<PerturbSummaryResponse>, (StatusCode, String)> {
    let n_propagation = body.n_propagation;
    let (
        n_obs,
        targets,
        gene,
        rt,
        gene_names,
        job_p,
        job_active,
        job_msg,
        cancel,
        n_vars,
        adata_path,
        foyer,
        epoch,
    ) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let rt = perturb_runtime_or_status(ds)?;
        let n_obs = ds.obs_names.len();
        let targets = build_perturb_targets(ds, &body, n_obs)?;
        let gene = targets[0].gene.clone();
        if !rt.gene_names.iter().any(|g| g == &gene) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("gene {:?} not in model var_names", gene),
            ));
        }
        let n_vars = ds.n_vars;
        let adata_path = ds.adata_path.to_string_lossy().into_owned();
        let epoch = st.dataset_cache_epoch.load(Ordering::SeqCst);
        let foyer = Arc::clone(&st.foyer_caches);
        (
            n_obs,
            targets,
            gene.clone(),
            Arc::clone(rt),
            rt.gene_names.clone(),
            Arc::clone(&st.perturb_job_progress_permille),
            Arc::clone(&st.perturb_job_active),
            Arc::clone(&st.perturb_progress_message),
            Arc::clone(&st.perturb_job_cancel),
            n_vars,
            adata_path,
            foyer,
            epoch,
        )
    };
    cancel.store(false, Ordering::SeqCst);
    let cfg = perturb_cfg_for_request(&rt.perturb_cfg, n_propagation);
    job_p.store(0, Ordering::Relaxed);
    job_active.store(true, Ordering::Relaxed);
    if let Ok(mut m) = job_msg.lock() {
        *m = "GRN perturbation (summary)…".into();
    }
    let job_active_move = job_active.clone();
    let _guard = PerturbJobGuard(job_active_move);
    let cache_key = foyer_perturb_cache::grn_perturb_cache_key(
        epoch,
        false,
        &adata_path,
        n_obs,
        n_vars,
        &targets,
        &cfg,
    );
    let result = cached_grn_perturb_result(
        foyer.as_ref(),
        cache_key,
        n_obs,
        rt.gene_names.len(),
        Arc::clone(&rt),
        targets,
        cfg,
        Arc::clone(&job_p),
        Arc::clone(&job_msg),
        Arc::clone(&cancel),
    )
    .await
    .map_err(|_| {
        job_p.store(0, Ordering::Relaxed);
        (StatusCode::REQUEST_TIMEOUT, "Perturbation cancelled".into())
    })?;
    job_p.store(1000, Ordering::Relaxed);

    let n_genes = gene_names.len();
    let mut gene_effects: Vec<PerturbGeneEffect> = (0..n_genes)
        .map(|j| {
            let col = result.delta.column(j);
            let mean_d: f64 = col.iter().sum::<f64>() / n_obs as f64;
            let max_abs: f64 = col.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            PerturbGeneEffect {
                gene: gene_names[j].clone(),
                mean_delta: mean_d,
                max_abs_delta: max_abs,
            }
        })
        .filter(|e| e.max_abs_delta > 1e-12)
        .collect();
    gene_effects.sort_by(|a, b| {
        b.max_abs_delta
            .partial_cmp(&a.max_abs_delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    gene_effects.truncate(50);

    let perturbed_gene_idx = gene_names.iter().position(|g| g == &gene).unwrap_or(0);
    let col = result.delta.column(perturbed_gene_idx);
    let mean_delta: f64 = col.iter().sum::<f64>() / n_obs as f64;
    let max_abs_delta: f64 = col.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let n_positive = col.iter().filter(|&&v| v > 1e-12).count();
    let n_negative = col.iter().filter(|&&v| v < -1e-12).count();
    let n_zero = n_obs - n_positive - n_negative;

    schedule_perturb_progress_permille_clear(Arc::clone(&job_p));
    Ok(Json(PerturbSummaryResponse {
        gene: gene.clone(),
        n_obs,
        mean_delta,
        max_abs_delta,
        n_positive,
        n_negative,
        n_zero,
        top_affected_genes: gene_effects,
    }))
}

#[derive(Deserialize)]
struct PerturbReferenceSimilarityBody {
    #[serde(flatten)]
    perturb: PerturbPreviewBody,
    /// Mean expression over cells in this scope defines the reference profile (same JSON shape as `perturb.scope`).
    reference: PerturbScopeBody,
    /// Model `var_names` to use; empty = all genes in the perturb runtime (same space as GRN δ).
    #[serde(default)]
    genes: Vec<String>,
    /// When true, cells matching `perturb.scope` are omitted from the reference centroid (recommended when reference and perturb types match).
    #[serde(default = "default_true")]
    exclude_perturb_cells_from_reference: bool,
}

#[derive(Serialize)]
struct PerturbReferenceSimilarityResponse {
    n_genes_used: usize,
    n_reference_cells: usize,
    n_eval_cells: usize,
    exclude_perturb_cells_from_reference: bool,
    mean_cosine_before: f64,
    mean_cosine_after: f64,
    median_cosine_before: f64,
    median_cosine_after: f64,
    mean_delta_cosine: f64,
}

fn resolve_gene_column_indices(
    gene_names: &[String],
    genes: &[String],
) -> Result<Vec<usize>, (StatusCode, String)> {
    if genes.is_empty() {
        return Ok((0..gene_names.len()).collect());
    }
    let mut cols = Vec::new();
    for g in genes {
        let t = g.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(p) = gene_names.iter().position(|x| x == t) {
            cols.push(p);
        }
    }
    cols.sort_unstable();
    cols.dedup();
    if cols.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "genes: no requested symbols found in model var_names".into(),
        ));
    }
    Ok(cols)
}

fn median_sorted_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut w: Vec<f64> = v.to_vec();
    w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = w.len() / 2;
    if w.len() % 2 == 1 {
        w[m]
    } else {
        0.5 * (w[m - 1] + w[m])
    }
}

async fn api_perturb_reference_similarity(
    State(state): State<SharedState>,
    Json(body): Json<PerturbReferenceSimilarityBody>,
) -> Result<Json<PerturbReferenceSimilarityResponse>, (StatusCode, String)> {
    let n_propagation = body.perturb.n_propagation;
    let exclude = body.exclude_perturb_cells_from_reference;
    let genes_filter = body.genes.clone();

    let (
        targets,
        rt,
        cols,
        eval_rows,
        ref_rows,
        job_p,
        job_active,
        job_msg,
        cancel,
        n_obs,
        n_vars,
        adata_path,
        foyer,
        epoch,
    ) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let rt = perturb_runtime_or_status(ds)?;
        let n_obs = ds.obs_names.len();
        let targets = build_perturb_targets(ds, &body.perturb, n_obs)?;
        let gene = targets[0].gene.clone();
        if !rt.gene_names.iter().any(|g| g == &gene) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("gene {:?} not in model var_names", gene),
            ));
        }
        let cols = resolve_gene_column_indices(&rt.gene_names, &genes_filter)?;
        let eval_opt = cell_indices_for_perturb_scope(ds, &body.perturb.scope, n_obs)?;
        let eval_rows: Vec<usize> = match eval_opt {
            None => (0..n_obs).collect(),
            Some(v) => v,
        };
        let ref_opt = cell_indices_for_perturb_scope(ds, &body.reference, n_obs)?;
        let mut ref_rows: Vec<usize> = match ref_opt {
            None => (0..n_obs).collect(),
            Some(v) => v,
        };
        if exclude {
            let perturb_set: HashSet<usize> = match &targets[0].cell_indices {
                None => (0..n_obs).collect(),
                Some(v) => v.iter().copied().collect(),
            };
            ref_rows.retain(|i| !perturb_set.contains(i));
        }
        if ref_rows.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "reference cell set is empty after excluding perturb cells; set exclude_perturb_cells_from_reference=false or use a broader reference scope"
                    .into(),
            ));
        }
        let n_vars = ds.n_vars;
        let adata_path = ds.adata_path.to_string_lossy().into_owned();
        let epoch = st.dataset_cache_epoch.load(Ordering::SeqCst);
        let foyer = Arc::clone(&st.foyer_caches);
        (
            targets,
            Arc::clone(rt),
            cols,
            eval_rows,
            ref_rows,
            Arc::clone(&st.perturb_job_progress_permille),
            Arc::clone(&st.perturb_job_active),
            Arc::clone(&st.perturb_progress_message),
            Arc::clone(&st.perturb_job_cancel),
            n_obs,
            n_vars,
            adata_path,
            foyer,
            epoch,
        )
    };

    cancel.store(false, Ordering::SeqCst);
    let cfg = perturb_cfg_for_request(&rt.perturb_cfg, n_propagation);
    job_p.store(0, Ordering::Relaxed);
    job_active.store(true, Ordering::Relaxed);
    if let Ok(mut m) = job_msg.lock() {
        *m = "GRN perturbation (reference similarity)…".into();
    }
    let job_active_move = job_active.clone();
    let _guard = PerturbJobGuard(job_active_move);
    let cache_key = foyer_perturb_cache::grn_perturb_cache_key(
        epoch,
        false,
        &adata_path,
        n_obs,
        n_vars,
        &targets,
        &cfg,
    );
    let pr = cached_grn_perturb_result(
        foyer.as_ref(),
        cache_key,
        n_obs,
        rt.gene_names.len(),
        Arc::clone(&rt),
        targets,
        cfg,
        Arc::clone(&job_p),
        Arc::clone(&job_msg),
        Arc::clone(&cancel),
    )
    .await
    .map_err(|_| {
        job_p.store(0, Ordering::Relaxed);
        (StatusCode::REQUEST_TIMEOUT, "Perturbation cancelled".into())
    })?;
    job_p.store(1000, Ordering::Relaxed);

    let delta_arr = pr.delta;
    let pack = tokio::task::spawn_blocking(move || -> Result<(Vec<f64>, Vec<f64>, f64, f64, usize, usize, usize), String> {
        let gene_mtx = &rt.gene_mtx;
        let delta = &delta_arr;
        let g = cols.len();
        let mut centroid = vec![0.0_f64; g];
        for k in 0..g {
            let c = cols[k];
            let mut s = 0.0_f64;
            for &r in &ref_rows {
                s += gene_mtx[[r, c]];
            }
            centroid[k] = s / ref_rows.len() as f64;
        }
        let mut norm_ref = 0.0_f64;
        for &x in &centroid {
            norm_ref += x * x;
        }
        norm_ref = norm_ref.sqrt();
        if norm_ref < 1e-14 {
            return Err(
                "reference centroid has near-zero norm in the chosen gene subspace; add genes or disable exclusion"
                    .into(),
            );
        }

        let mut before_v: Vec<f64> = Vec::with_capacity(eval_rows.len());
        let mut after_v: Vec<f64> = Vec::with_capacity(eval_rows.len());
        for &i in &eval_rows {
            let mut dot_b = 0.0_f64;
            let mut na_b = 0.0_f64;
            let mut dot_a = 0.0_f64;
            let mut na_a = 0.0_f64;
            for k in 0..g {
                let c = cols[k];
                let b = gene_mtx[[i, c]];
                let a = b + delta[[i, c]];
                let ck = centroid[k];
                dot_b += b * ck;
                na_b += b * b;
                dot_a += a * ck;
                na_a += a * a;
            }
            let nb = na_b.sqrt();
            let na = na_a.sqrt();
            let cb = if nb < 1e-14 {
                0.0_f64
            } else {
                dot_b / (nb * norm_ref)
            };
            let ca = if na < 1e-14 {
                0.0_f64
            } else {
                dot_a / (na * norm_ref)
            };
            before_v.push(cb);
            after_v.push(ca);
        }

        let n_ev = before_v.len();
        if n_ev == 0 {
            return Err("no cells to evaluate (empty perturb scope)".into());
        }
        let mean_b: f64 = before_v.iter().sum::<f64>() / n_ev as f64;
        let mean_a: f64 = after_v.iter().sum::<f64>() / n_ev as f64;
        Ok((
            before_v,
            after_v,
            mean_b,
            mean_a,
            ref_rows.len(),
            eval_rows.len(),
            g,
        ))
    })
    .await
    .map_err(|e| {
        job_p.store(0, Ordering::Relaxed);
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?
    .map_err(|msg| {
        job_p.store(0, Ordering::Relaxed);
        let code = if msg.contains("perturbation failed") {
            StatusCode::REQUEST_TIMEOUT
        } else {
            StatusCode::BAD_REQUEST
        };
        (code, msg)
    })?;

    let (before_v, after_v, mean_b, mean_a, n_ref, n_eval, n_genes) = pack;

    let med_b = median_sorted_f64(&before_v);
    let med_a = median_sorted_f64(&after_v);
    schedule_perturb_progress_permille_clear(Arc::clone(&job_p));
    Ok(Json(PerturbReferenceSimilarityResponse {
        n_genes_used: n_genes,
        n_reference_cells: n_ref,
        n_eval_cells: n_eval,
        exclude_perturb_cells_from_reference: exclude,
        mean_cosine_before: mean_b,
        mean_cosine_after: mean_a,
        median_cosine_before: med_b,
        median_cosine_after: med_a,
        mean_delta_cosine: mean_a - mean_b,
    }))
}

#[derive(Deserialize)]
struct NeighborSanityBody {
    gene: String,
    #[serde(default)]
    desired_expr: f64,
    cell_index: usize,
    #[serde(default)]
    n_propagation: Option<usize>,
    #[serde(default)]
    neighbor_radius: Option<f64>,
    #[serde(default)]
    require_cluster_id: Option<usize>,
}

#[derive(Serialize)]
struct NeighborSanityResponse {
    gene: String,
    desired_expr: f64,
    cell_index: usize,
    cluster_id_at_cell: usize,
    neighbor_radius_used: f64,
    n_propagation: usize,
    n_neighbors_within_radius: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    neighbor_sample: Vec<usize>,
    ligand_gene_in_lr_model: bool,
    sender_abs_delta_perturbed_gene: f64,
    neighbors_mean_abs_delta_perturbed_gene: f64,
    remote_mean_abs_delta_perturbed_gene: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    neighbors_vs_remote_ratio_perturbed_gene: Option<f64>,
    neighbors_mean_l1_delta: f64,
    remote_mean_l1_delta: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    neighbors_vs_remote_ratio_l1: Option<f64>,
    far_gt_2r_mean_l1_delta: f64,
    n_remote: usize,
    n_far: usize,
    interpretation: String,
}

fn euclidean_xy(xy: &Array2<f64>, i: usize, j: usize) -> f64 {
    let dx = xy[[i, 0]] - xy[[j, 0]];
    let dy = xy[[i, 1]] - xy[[j, 1]];
    (dx * dx + dy * dy).sqrt()
}

fn max_ligand_radius(lr_radii: &HashMap<String, f64>) -> f64 {
    lr_radii.values().copied().fold(0.0_f64, f64::max).max(1e-6)
}

fn row_l1_norm(delta: &Array2<f64>, row: usize) -> f64 {
    delta.row(row).iter().map(|v| v.abs()).sum()
}

async fn api_perturb_neighbor_sanity(
    State(state): State<SharedState>,
    Json(body): Json<NeighborSanityBody>,
) -> Result<Json<NeighborSanityResponse>, (StatusCode, String)> {
    let gene = body.gene.trim().to_string();
    if gene.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "gene is empty".into()));
    }
    let (
        n_obs,
        cluster_at,
        rt,
        job_p,
        job_active,
        job_msg,
        cancel,
        n_vars,
        adata_path,
        foyer,
        epoch,
    ) = {
        let st = state.read().await;
        let ds = require_dataset(&st)?;
        let rt = perturb_runtime_or_status(ds)?;
        let n_obs = ds.obs_names.len();
        if body.cell_index >= n_obs {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "cell_index {} out of range (n_obs={})",
                    body.cell_index, n_obs
                ),
            ));
        }
        if let Some(want) = body.require_cluster_id {
            let got = ds.clusters[body.cell_index];
            if got != want {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "cell_index {} has cluster_id {} but require_cluster_id was {}",
                        body.cell_index, got, want
                    ),
                ));
            }
        }
        let preview_body = PerturbPreviewBody {
            gene: gene.clone(),
            desired_expr: body.desired_expr,
            scope: PerturbScopeBody::Indices {
                indices: vec![body.cell_index],
            },
            n_propagation: body.n_propagation,
        };
        build_perturb_targets(ds, &preview_body, n_obs)?;
        if !rt.gene_names.iter().any(|g| g == &gene) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("gene {:?} not in model var_names", gene),
            ));
        }
        let cluster_at = ds.clusters[body.cell_index];
        let n_vars = ds.n_vars;
        let adata_path = ds.adata_path.to_string_lossy().into_owned();
        let epoch = st.dataset_cache_epoch.load(Ordering::SeqCst);
        let foyer = Arc::clone(&st.foyer_caches);
        (
            n_obs,
            cluster_at,
            Arc::clone(rt),
            Arc::clone(&st.perturb_job_progress_permille),
            Arc::clone(&st.perturb_job_active),
            Arc::clone(&st.perturb_progress_message),
            Arc::clone(&st.perturb_job_cancel),
            n_vars,
            adata_path,
            foyer,
            epoch,
        )
    };

    let radius = body
        .neighbor_radius
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or_else(|| max_ligand_radius(&rt.lr_radii));

    let desired_expr = body.desired_expr;
    let targets = vec![PerturbTarget {
        gene: gene.clone(),
        desired_expr,
        cell_indices: Some(vec![body.cell_index]),
    }];

    cancel.store(false, Ordering::SeqCst);
    let cfg = perturb_cfg_for_request(&rt.perturb_cfg, body.n_propagation);
    let n_propagation_used = cfg.n_propagation;
    job_p.store(0, Ordering::Relaxed);
    job_active.store(true, Ordering::Relaxed);
    if let Ok(mut m) = job_msg.lock() {
        *m = "GRN perturbation (neighbor sanity)…".into();
    }
    let cell_index = body.cell_index;
    let job_active_move = job_active.clone();
    let _guard = PerturbJobGuard(job_active_move);
    let cache_key = foyer_perturb_cache::grn_perturb_cache_key(
        epoch,
        false,
        &adata_path,
        n_obs,
        n_vars,
        &targets,
        &cfg,
    );
    let result = cached_grn_perturb_result(
        foyer.as_ref(),
        cache_key,
        n_obs,
        rt.gene_names.len(),
        Arc::clone(&rt),
        targets,
        cfg,
        Arc::clone(&job_p),
        Arc::clone(&job_msg),
        Arc::clone(&cancel),
    )
    .await
    .map_err(|_| {
        job_p.store(0, Ordering::Relaxed);
        (StatusCode::REQUEST_TIMEOUT, "Perturbation cancelled".into())
    })?;
    job_p.store(1000, Ordering::Relaxed);

    schedule_perturb_progress_permille_clear(Arc::clone(&job_p));

    let gj = rt
        .gene_names
        .iter()
        .position(|g| g == gene.as_str())
        .unwrap_or(0);

    let xy = &rt.xy;
    let delta = &result.delta;
    let mut neighbors: Vec<(usize, f64)> = Vec::new();
    let mut remote: Vec<usize> = Vec::new();
    let mut far: Vec<usize> = Vec::new();
    let r2 = radius * 2.0;

    for j in 0..n_obs {
        if j == cell_index {
            continue;
        }
        let d = euclidean_xy(xy, cell_index, j);
        if d <= radius {
            neighbors.push((j, d));
        } else {
            remote.push(j);
        }
        if d > r2 {
            far.push(j);
        }
    }

    neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let neighbor_indices: Vec<usize> = neighbors.iter().map(|x| x.0).collect();
    let neighbor_sample: Vec<usize> = neighbor_indices.iter().copied().take(100).collect();

    let sender_abs = delta[[cell_index, gj]].abs();

    let mean_abs_pg = |rows: &[usize]| -> f64 {
        if rows.is_empty() {
            return 0.0;
        }
        let s: f64 = rows.iter().map(|&r| delta[[r, gj]].abs()).sum();
        s / rows.len() as f64
    };

    let mean_l1 = |rows: &[usize]| -> f64 {
        if rows.is_empty() {
            return 0.0;
        }
        let s: f64 = rows.iter().map(|&r| row_l1_norm(delta, r)).sum();
        s / rows.len() as f64
    };

    let nbr_m_pg = mean_abs_pg(&neighbor_indices);
    let rem_m_pg = mean_abs_pg(&remote);
    let nbr_l1 = mean_l1(&neighbor_indices);
    let rem_l1 = mean_l1(&remote);
    let far_l1 = mean_l1(&far);

    let ratio =
        |a: f64, b: f64| -> Option<f64> { if b.abs() > 1e-14 { Some(a / b) } else { None } };

    let lig_lr = rt.bb.ligands_set.contains(gene.as_str());

    let interpretation = if neighbor_indices.is_empty() {
        "No spatial neighbors within the radius (other cells are farther than neighbor_radius_used). \
         Increase neighbor_radius or choose a denser region."
            .to_string()
    } else if remote.is_empty() {
        "All other cells are within one radius of the sender; remote comparison is not defined."
            .to_string()
    } else {
        let r2r = ratio(nbr_l1, rem_l1);
        let mut parts = vec![format!(
            "Compared {} neighbors (dist ≤ {:.4}) vs {} remote cells (dist > {:.4}) on spatial coordinates.",
            neighbor_indices.len(),
            radius,
            remote.len(),
            radius
        )];
        if let Some(rv) = r2r {
            if lig_lr && rv > 1.15 {
                parts.push(format!(
                    "Mean L1 |Δ| is {:.2}× higher in neighbors than in remote cells — consistent with spatial ligand coupling (gene is in the LR ligand set).",
                    rv
                ));
            } else if lig_lr && rv < 0.9 {
                parts.push(
                    "Mean L1 |Δ| is not higher in neighbors than remote despite this gene being modeled as a ligand; propagation may be dominated by intracellular GRN or the chosen cell may have weak outgoing signal in this state."
                        .to_string(),
                );
            } else if !lig_lr {
                parts.push(
                    "This gene is not listed as an LR ligand in the loaded betabase; neighbor enrichment is not required for a well-specified spatial GRN (effects may spread through other ligands / TFL edges)."
                        .to_string(),
                );
            } else {
                parts.push(format!(
                    "Neighbor vs remote L1 |Δ| ratio ≈ {:.2} (modest); review top affected genes or increase propagation depth if needed.",
                    rv
                ));
            }
        } else {
            parts.push("Remote mean |Δ| is near zero; global effect of this single-cell perturbation may be very small.".to_string());
        }
        parts.join(" ")
    };

    Ok(Json(NeighborSanityResponse {
        gene: gene.clone(),
        desired_expr: body.desired_expr,
        cell_index,
        cluster_id_at_cell: cluster_at,
        neighbor_radius_used: radius,
        n_propagation: n_propagation_used,
        n_neighbors_within_radius: neighbor_indices.len(),
        neighbor_sample,
        ligand_gene_in_lr_model: lig_lr,
        sender_abs_delta_perturbed_gene: sender_abs,
        neighbors_mean_abs_delta_perturbed_gene: nbr_m_pg,
        remote_mean_abs_delta_perturbed_gene: rem_m_pg,
        neighbors_vs_remote_ratio_perturbed_gene: ratio(nbr_m_pg, rem_m_pg),
        neighbors_mean_l1_delta: nbr_l1,
        remote_mean_l1_delta: rem_l1,
        neighbors_vs_remote_ratio_l1: ratio(nbr_l1, rem_l1),
        far_gt_2r_mean_l1_delta: far_l1,
        n_remote: remote.len(),
        n_far: far.len(),
        interpretation,
    }))
}

#[derive(Deserialize)]
struct ClusterMeanExprBody {
    genes: Vec<String>,
}

#[derive(Serialize)]
struct ClusterMeanExprResponse {
    cluster_ids: Vec<usize>,
    genes: HashMap<String, Vec<f64>>,
    n_cells_per_cluster: Vec<usize>,
}

async fn api_cluster_mean_expression(
    State(state): State<SharedState>,
    Json(body): Json<ClusterMeanExprBody>,
) -> Result<Json<ClusterMeanExprResponse>, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let path = ds.adata_path.clone();
    let layer = ds.layer.clone();
    let clusters = Arc::clone(&ds.clusters);
    let n_obs = ds.obs_names.len();
    drop(st);

    let genes = body.genes;
    if genes.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "genes list is empty".into()));
    }
    if genes.len() > 200 {
        return Err((StatusCode::BAD_REQUEST, "max 200 genes at a time".into()));
    }

    let mut unique_clusters: Vec<usize> = clusters
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_clusters.sort_unstable();
    let cluster_to_idx: HashMap<usize, usize> = unique_clusters
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let n_clusters = unique_clusters.len();
    let mut n_cells_per_cluster = vec![0usize; n_clusters];
    for &c in clusters.iter() {
        if let Some(&idx) = cluster_to_idx.get(&c) {
            n_cells_per_cluster[idx] += 1;
        }
    }

    let genes_clone = genes.clone();
    let clusters_clone = clusters.clone();
    let result =
        tokio::task::spawn_blocking(move || -> Result<HashMap<String, Vec<f64>>, String> {
            let adata = open_adata(path.to_string_lossy().as_ref()).map_err(|e| e.to_string())?;
            let mut out = HashMap::new();
            for gene in &genes_clone {
                match gene_expression_f32(&adata, &layer, gene) {
                    Ok(expr) => {
                        let mut sums = vec![0.0f64; n_clusters];
                        let mut counts = vec![0usize; n_clusters];
                        for i in 0..n_obs.min(expr.len()) {
                            let c = clusters_clone[i];
                            if let Some(&idx) = cluster_to_idx.get(&c) {
                                sums[idx] += expr[i] as f64;
                                counts[idx] += 1;
                            }
                        }
                        let means: Vec<f64> = (0..n_clusters)
                            .map(|j| {
                                if counts[j] > 0 {
                                    sums[j] / counts[j] as f64
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        out.insert(gene.clone(), means);
                    }
                    Err(_) => {
                        out.insert(gene.clone(), vec![0.0; n_clusters]);
                    }
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    Ok(Json(ClusterMeanExprResponse {
        cluster_ids: unique_clusters,
        genes: result,
        n_cells_per_cluster,
    }))
}

#[derive(Deserialize)]
struct LabelClustersBody {
    labels: HashMap<String, String>,
}

async fn api_label_clusters(
    State(state): State<SharedState>,
    Json(body): Json<LabelClustersBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut w = state.write().await;
    let ds = w
        .dataset
        .as_mut()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "No dataset loaded".into()))?;

    let mut unique_clusters: Vec<usize> = ds
        .clusters
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_clusters.sort_unstable();

    let mut categories: Vec<String> = Vec::with_capacity(unique_clusters.len());
    for &cid in &unique_clusters {
        let label = body
            .labels
            .get(&cid.to_string())
            .cloned()
            .unwrap_or_else(|| format!("Cluster {}", cid));
        categories.push(label);
    }

    let cluster_to_cat: HashMap<usize, u16> = unique_clusters
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as u16))
        .collect();

    let n_obs = ds.obs_names.len();
    let mut codes = vec![0u16; n_obs];
    for i in 0..n_obs {
        codes[i] = *cluster_to_cat.get(&ds.clusters[i]).unwrap_or(&u16::MAX);
    }
    let codes_bin: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();

    ds.cell_type_column = Some("annotated_type".into());
    ds.cell_type_categories = Arc::new(categories.clone());
    ds.cell_type_codes_bin = Some(Arc::new(codes_bin));

    Ok(Json(serde_json::json!({
        "ok": true,
        "categories": categories,
        "n_clusters": unique_clusters.len(),
    })))
}

fn resolve_static_dir(path: &Path) -> anyhow::Result<PathBuf> {
    let cwd = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("static-dir: cannot read current working directory: {e}"))?;
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    joined.canonicalize().map_err(|e| {
        anyhow::anyhow!(
            "static-dir `{}` does not exist or is unreachable (cwd `{}`): {e}",
            joined.display(),
            cwd.display()
        )
    })
}

/// All HDF5 / Polars / GRN work runs here on a plain OS thread (no Tokio runtime). Starting Tokio
/// only for `axum::serve` avoids nested-runtime panics when Polars or other code calls into async
/// runtimes during `LazyFrame::collect()` etc.
fn build_app(
    cli: Cli,
    foyer: FoyerPerturbCaches,
) -> anyhow::Result<(SocketAddr, Router, SharedState)> {
    let allow_cors = cli.allow_cors
        || std::env::var("SPATIAL_VIEWER_ALLOW_CORS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

    let h5ad_empty = cli.h5ad.as_ref().map_or(true, |p| p.as_os_str().is_empty());
    let run_toml_set = cli
        .run_toml
        .as_ref()
        .map_or(false, |p| !p.as_os_str().is_empty());
    let dataset = if !h5ad_empty || run_toml_set {
        Some(load_app_state(ViewerLoadInputs {
            h5ad: cli.h5ad.clone().unwrap_or_default(),
            layer: cli.layer.clone(),
            cluster_annot: cli.cluster_annot.clone(),
            network_dir: cli.network_dir.clone(),
            run_toml: cli.run_toml.clone(),
        })?)
    } else {
        None
    };
    let dataset_cache_epoch = Arc::new(AtomicU64::new(if dataset.is_some() { 1 } else { 0 }));
    let state = Arc::new(RwLock::new(AppState {
        dataset,
        default_layer: cli.layer.clone(),
        default_cluster_annot: cli.cluster_annot.clone(),
        default_network_dir: cli.network_dir.clone(),
        default_run_toml: cli.run_toml.clone(),
        perturb_bg_gen: Arc::new(AtomicU64::new(0)),
        perturb_bg_in_flight: Arc::new(AtomicBool::new(false)),
        perturb_bg_pending: Arc::new(AtomicBool::new(false)),
        perturb_load_progress_permille: Arc::new(AtomicU32::new(0)),
        perturb_job_progress_permille: Arc::new(AtomicU32::new(0)),
        perturb_job_active: Arc::new(AtomicBool::new(false)),
        perturb_job_cancel: Arc::new(AtomicBool::new(false)),
        perturb_suppress_bg_loading_ui: Arc::new(AtomicBool::new(false)),
        perturb_progress_message: Arc::new(Mutex::new(String::new())),
        perturb_betadata_ui: Arc::new(BetadataUiProgress::new()),
        splash_job_progress_permille: Arc::new(AtomicU32::new(0)),
        splash_job_active: Arc::new(AtomicBool::new(false)),
        foyer_caches: Arc::new(foyer),
        dataset_cache_epoch,
    }));

    let api = Router::new()
        .route("/meta", get(api_meta))
        .route("/cancel", post(api_cancel))
        .route("/session/configure", post(api_session_configure))
        .route("/spatial", get(api_spatial))
        .route("/umap", get(api_umap))
        .route("/clusters", get(api_clusters))
        .route("/cell_type/codes", get(api_cell_type_codes))
        .route("/genes", get(api_genes))
        .route("/genes/full", get(api_genes_full))
        .route("/gene/expression", get(api_gene_expression))
        .route("/spatial/received_ligand", post(api_received_ligand))
        .route("/betadata/genes", get(api_betadata_genes))
        .route("/betadata/columns", get(api_betadata_columns))
        .route("/betadata/values", get(api_betadata_values))
        .route("/betadata/top", post(api_betadata_top))
        .route(
            "/betadata/collect_interactions",
            post(api_betadata_collect_interactions),
        )
        .route("/betadata/pair_lr", post(api_betadata_pair_lr))
        .route("/network/cell-context", post(api_network_cell_context))
        .route("/perturb/preview", post(api_perturb_preview))
        .route("/perturb/export_feather", post(api_perturb_export_feather))
        .route("/perturb/splash_network", post(api_perturb_splash_network))
        .route("/perturb/splash_progress", get(api_splash_progress))
        .route("/perturb/umap-field", post(api_perturb_umap_field))
        .route("/umap/signature_field", post(api_umap_signature_field))
        .route("/perturb/summary", post(api_perturb_summary))
        .route(
            "/perturb/reference_similarity",
            post(api_perturb_reference_similarity),
        )
        .route(
            "/perturb/neighbor_sanity",
            post(api_perturb_neighbor_sanity),
        )
        .route(
            "/cluster/mean_expression",
            post(api_cluster_mean_expression),
        )
        .route("/meta/label_clusters", post(api_label_clusters))
        .with_state(state.clone())
        .layer(CompressionLayer::new());

    let api = if allow_cors {
        tracing::warn!(
            "CORS enabled on /api (MCP / cross-origin); do not expose this server untrusted"
        );
        api.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(Any),
        )
    } else {
        api
    };

    let static_dir = resolve_static_dir(cli.static_dir.as_path())?;
    let index = static_dir.join("index.html");
    if !index.is_file() {
        anyhow::bail!(
            "--static-dir must contain index.html after resolving path (missing: {})",
            index.display()
        );
    }

    let static_files = ServeDir::new(&static_dir).fallback(ServeFile::new(index.clone()));
    let app = Router::new()
        .nest("/api", api)
        .route_service("/", get_service(ServeFile::new(index.clone())))
        .fallback_service(static_files)
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", cli.bind, cli.port).parse()?;
    Ok((addr, app, state))
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "spatial_viewer=info,tower_http=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let foyer = rt
        .block_on(foyer_perturb_cache::open_foyer_perturb_caches_with_limits(
            cli.perturb_cache_dir.as_deref(),
            foyer_cache_limits_from_cli(&cli),
        ))
        .map_err(|e| anyhow!("foyer hybrid cache: {e}"))?;
    let (addr, app, state) = build_app(cli, foyer)?;

    rt.block_on(async move {
        tracing::info!("listening on http://{}", addr);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        spawn_perturb_background_load(state.clone());
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        let fc = state.read().await.foyer_caches.clone();
        let _ = foyer_perturb_cache::close_foyer_caches(fc.as_ref()).await;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}
