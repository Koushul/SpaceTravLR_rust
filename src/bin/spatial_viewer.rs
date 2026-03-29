use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anndata::AnnData;
use anndata::AnnDataOp;
use anndata_hdf5::H5;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use clap::Parser;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use space_trav_lr_rust::adata_query::{
    cell_expression_map, cell_type_encoding, clusters_as_u32_le_bytes,
    expression_profiles_for_cells, f32_vec_to_le_bytes, gene_expression_f32, genes_with_prefix,
    obs_names, open_adata, spatial_obsm_key_used, spatial_xy,
    spatial_xy_f32_interleaved, try_umap_xy, u16_vec_to_le_bytes, var_names,
};
use space_trav_lr_rust::network::{infer_species, GeneNetwork};
use space_trav_lr_rust::betadata::{
    betadata_feather_per_cell_column, betadata_feather_plottable_columns,
    betadata_feather_row_id_column, betadata_feather_top_coefficients_for_selection,
    TopBetaCoefficient,
};
use space_trav_lr_rust::betadata_view::{betadata_feather_path, list_betadata_target_genes};
use space_trav_lr_rust::config::expand_user_path;
use space_trav_lr_rust::perturb::{perturb_with_targets, PerturbTarget};
use space_trav_lr_rust::perturb_mode::PerturbRuntime;
use space_trav_lr_rust::transition_umap::{compute_umap_transition_grid, TransitionUmapParams};
use tokio::sync::RwLock;
use tower_http::compression::CompressionLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

const DEFAULT_GENE_SEARCH_LIMIT: usize = 500;
const FULL_GENE_LIST_THRESHOLD: usize = 50_000;
const MAX_TOP_BETA_INDICES: usize = 250_000;
const MAX_UMAP_TRANSITION_CELLS: usize = 40_000;
const MAX_LR_NEIGHBORS_RADIUS: usize = 500;

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
    #[arg(long, default_value = "web/spatial_viewer/dist")]
    static_dir: PathBuf,
    /// Directory containing `{mouse|human}_network.parquet` (optional; otherwise same search as training).
    #[arg(long)]
    network_dir: Option<PathBuf>,
    /// `spacetravlr_run_repro.toml` from a training run: enables perturbation and betadata from the TOML’s directory (`data.adata_path` must match `--h5ad`).
    #[arg(long)]
    run_toml: Option<PathBuf>,
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

#[derive(Clone)]
struct AppState {
    dataset: Option<AppDataset>,
    default_layer: String,
    default_cluster_annot: String,
    default_network_dir: Option<PathBuf>,
    default_run_toml: Option<PathBuf>,
    perturb_bg_gen: Arc<AtomicU64>,
    perturb_bg_in_flight: Arc<AtomicBool>,
    perturb_load_progress_permille: Arc<AtomicU32>,
    perturb_job_progress_permille: Arc<AtomicU32>,
    perturb_job_active: Arc<AtomicBool>,
    perturb_progress_message: Arc<Mutex<String>>,
}

#[derive(Clone, Serialize)]
struct MetaBounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perturb_progress_label: Option<String>,
    adata_path: String,
    /// Training run directory when `--run-toml` was passed (same dir as `*_betadata.feather`); empty otherwise.
    betadata_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    network_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_toml: Option<String>,
    #[serde(default)]
    dataset_ready: bool,
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
    cell_index: usize,
    focus_gene: String,
    #[serde(default = "default_neighbor_k")]
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

fn spatial_k_nearest(spatial_f32: &[f32], n: usize, cell_idx: usize, k: usize) -> Vec<(usize, f64)> {
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

type SharedState = Arc<RwLock<AppState>>;

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
    let tcan = tomlp
        .canonicalize()
        .unwrap_or_else(|_| tomlp.to_path_buf());
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

fn spawn_perturb_background_load(state: SharedState) {
    tokio::spawn(async move {
        let run_toml = {
            let g = state.read().await;
            let Some(ds) = g.dataset.as_ref() else {
                return;
            };
            if ds.perturb_runtime.is_some() || ds.perturb_load_error.is_some() {
                return;
            }
            let Some(p) = ds.run_toml.clone() else {
                return;
            };
            p
        };
        let committed_gen = state.read().await.perturb_bg_gen.load(Ordering::SeqCst);
        let in_flight = Arc::clone(&state.read().await.perturb_bg_in_flight);
        in_flight.store(true, Ordering::SeqCst);
        let _in_flight_guard = ClearPerturbInFlight(in_flight);
        tracing::info!(
            "loading perturbation runtime in background from {} (large betabase runs can take several minutes)",
            run_toml.display()
        );
        let (load_perm, prog_msg) = {
            let g = state.read().await;
            (
                Arc::clone(&g.perturb_load_progress_permille),
                Arc::clone(&g.perturb_progress_message),
            )
        };
        load_perm.store(0, Ordering::Relaxed);
        if let Ok(mut m) = prog_msg.lock() {
            *m = "Starting perturbation runtime load…".into();
        }
        let run_toml_for_blocking = run_toml.clone();
        let pr_result = tokio::task::spawn_blocking(move || {
            PerturbRuntime::from_run_toml_with_progress(
                run_toml_for_blocking.as_path(),
                Some(load_perm),
                Some(prog_msg),
            )
        })
        .await;
        let mut w = state.write().await;
        if w.perturb_bg_gen.load(Ordering::SeqCst) != committed_gen {
            return;
        }
        let Some(ds) = w.dataset.as_mut() else {
            return;
        };
        if ds.run_toml.as_ref() != Some(&run_toml) {
            return;
        }
        if ds.perturb_runtime.is_some() {
            return;
        }
        match pr_result {
            Ok(Ok(pr)) => {
                if let Err(e) = attach_perturb_runtime_to_dataset(ds, pr, run_toml.as_path()) {
                    tracing::error!("perturbation runtime rejected: {:#}", e);
                    ds.perturb_load_error = Some(format!("{:#}", e));
                }
            }
            Ok(Err(e)) => {
                tracing::error!("perturbation runtime load failed: {:#}", e);
                ds.perturb_load_error = Some(format!("{:#}", e));
            }
            Err(e) => {
                ds.perturb_load_error = Some(format!("perturb load task join: {}", e));
            }
        }
    });
}

fn require_dataset(st: &AppState) -> Result<&AppDataset, (StatusCode, String)> {
    st.dataset.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "No dataset loaded; use Dataset paths → Load dataset.".into(),
    ))
}

fn perturb_runtime_or_status(ds: &AppDataset) -> Result<&Arc<PerturbRuntime>, (StatusCode, String)> {
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
            "Perturbation runtime is still loading (large betabase). Poll GET /api/meta until perturb_ready is true."
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
            perturb_progress_label: None,
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
        perturb_loading: st.perturb_bg_in_flight.load(Ordering::Relaxed)
            || (ds.run_toml.is_some()
                && ds.perturb_runtime.is_none()
                && ds.perturb_load_error.is_none()),
        perturb_error: ds.perturb_load_error.clone(),
        perturb_progress_percent: {
            let job_on = st.perturb_job_active.load(Ordering::Relaxed);
            let load_on = st.perturb_bg_in_flight.load(Ordering::Relaxed)
                || (ds.run_toml.is_some()
                    && ds.perturb_runtime.is_none()
                    && ds.perturb_load_error.is_none());
            let p = if job_on {
                Some(st.perturb_job_progress_permille.load(Ordering::Relaxed))
            } else if load_on {
                Some(st.perturb_load_progress_permille.load(Ordering::Relaxed))
            } else {
                None
            };
            p.map(permille_to_percent)
        },
        perturb_progress_label: {
            let show = st.perturb_job_active.load(Ordering::Relaxed)
                || st.perturb_bg_in_flight.load(Ordering::Relaxed)
                || (ds.run_toml.is_some()
                    && ds.perturb_runtime.is_none()
                    && ds.perturb_load_error.is_none());
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
        adata_path: ds.adata_path.display().to_string(),
        betadata_dir: ds
            .betadata_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        network_dir: ds.network_dir.as_ref().map(|p| p.display().to_string()),
        run_toml: ds.run_toml.as_ref().map(|p| p.display().to_string()),
        dataset_ready: true,
    }
}

fn load_app_state(inputs: ViewerLoadInputs) -> anyhow::Result<AppDataset> {
    let adata = open_adata(inputs.h5ad.to_string_lossy().as_ref())?;
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
    let clusters = Arc::new(space_trav_lr_rust::adata_query::clusters_from_obs_column(
        &adata,
        &inputs.cluster_annot,
    )?);
    anyhow::ensure!(
        onames.len() == clusters.len(),
        "cluster column length mismatch"
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
                tracing::info!(
                    "loaded GRN species={} path={}",
                    g.species,
                    g.network_path
                );
                Some(Arc::new(g))
            }
            Err(e) => {
                tracing::warn!("GRN not available: {}", e);
                None
            }
        };
    let (perturb_runtime, betadata_dir) = if let Some(ref rtp) = inputs.run_toml {
        let parent = rtp
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| anyhow::anyhow!("run TOML path has no parent directory"))?;
        tracing::info!(
            "perturbation API will load in background from {} (betadata dir {})",
            rtp.display(),
            parent.display()
        );
        (None, Some(parent))
    } else {
        (None, None)
    };
    let betadata_row_id = betadata_dir
        .as_ref()
        .and_then(|d| probe_betadata_row_id(d));
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
        Some(PathBuf::from(expand_user_path(t)))
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
    let h5ad = PathBuf::from(expand_user_path(body.adata_path.trim()));
    if h5ad.as_os_str().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "adata_path is required".into(),
        ));
    }
    let layer = non_empty_trimmed(&body.layer, "imputed_count");
    let cluster_annot = non_empty_trimmed(&body.cluster_annot, "cell_type");
    let network_dir = session_path_field(&body.network_dir);
    let run_toml = session_path_field(&body.run_toml);
    let inputs = ViewerLoadInputs {
        h5ad,
        layer: layer.clone(),
        cluster_annot: cluster_annot.clone(),
        network_dir: network_dir.clone(),
        run_toml: run_toml.clone(),
    };
    let new_dataset = tokio::task::spawn_blocking(move || load_app_state(inputs))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let mut w = state.write().await;
    w.perturb_bg_gen.fetch_add(1, Ordering::SeqCst);
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

async fn api_spatial(
    State(state): State<SharedState>,
) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    Ok(binary_response(f32_vec_to_le_bytes(ds.spatial_f32.as_ref())))
}

async fn api_umap(
    State(state): State<SharedState>,
) -> Result<Response, (StatusCode, String)> {
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

async fn api_clusters(
    State(state): State<SharedState>,
) -> Result<Response, (StatusCode, String)> {
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
    let layer = ds.layer.clone();
    let gene = q.gene;
    let adata = ds.adata.read().await;
    let vec = gene_expression_f32(&adata, &layer, &gene).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            format!("{}", e),
        )
    })?;
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
    let path = betadata_feather_path(
        bd.to_string_lossy().as_ref(),
        &q.gene,
    );
    if !path.is_file() {
        return Err((StatusCode::NOT_FOUND, format!("missing {:?}", path)));
    }
    match betadata_feather_plottable_columns(path.to_string_lossy().as_ref()) {
        Ok(cols) => Ok(axum::Json(cols).into_response()),
        Err(e) => Err((StatusCode::BAD_REQUEST, e.to_string())),
    }
}

async fn api_betadata_values(
    State(state): State<SharedState>,
    Query(q): Query<BetadataValuesQuery>,
) -> Result<Response, (StatusCode, String)> {
    let st = state.read().await;
    let ds = require_dataset(&st)?;
    let bd = ds.betadata_dir.as_ref().ok_or_else(|| {
        (StatusCode::BAD_REQUEST, "No betadata directory configured".into())
    })?;
    let path = betadata_feather_path(
        bd.to_string_lossy().as_ref(),
        &q.gene,
    );
    if !path.is_file() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("missing {:?}", path),
        ));
    }
    let obs = ds.obs_names.clone();
    let clusters = ds.clusters.clone();
    let path_s = path.to_string_lossy().to_string();
    let column = q.column;
    let vec = tokio::task::spawn_blocking(move || {
        betadata_feather_per_cell_column(&path_s, &column, &obs, &clusters)
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
        (StatusCode::BAD_REQUEST, "No betadata directory configured".into())
    })?;
    let path = betadata_feather_path(
        bd.to_string_lossy().as_ref(),
        &body.gene,
    );
    if !path.is_file() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("missing {:?}", path),
        ));
    }
    let top_k = if body.top_k == 0 {
        25
    } else {
        body.top_k.min(100)
    };
    let obs = ds.obs_names.clone();
    let clusters = ds.clusters.clone();
    let path_s = path.to_string_lossy().to_string();
    let indices = body.indices;
    let out = tokio::task::spawn_blocking(move || {
        betadata_feather_top_coefficients_for_selection(
            &path_s,
            &obs,
            &clusters,
            &indices,
            top_k,
        )
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(out))
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
        && body.radius.map(|r| r > 0.0 && r.is_finite()).unwrap_or(false);
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
        cell_expression_map(&adata, &ds.layer, body.cell_index).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("expression read: {}", e),
            )
        })?
    };

    let modulators = grn
        .get_modulators(
            focus_gene,
            body.tf_ligand_cutoff,
            Some(800),
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
        neighbors_out.push(NeighborContextJson {
            index: nj,
            distance_sq: d2,
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
}

#[derive(Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PerturbScopeBody {
    #[default]
    All,
    Indices { indices: Vec<usize> },
    CellType { category: u16 },
    Cluster { cluster_id: usize },
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
    let cell_indices = match &body.scope {
        PerturbScopeBody::All => None,
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
            Some(v)
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
            Some(idx)
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
            Some(idx)
        }
    };
    Ok(vec![PerturbTarget {
        gene: gene.to_string(),
        desired_expr: body.desired_expr,
        cell_indices,
    }])
}

async fn api_perturb_preview(
    State(state): State<SharedState>,
    Json(body): Json<PerturbPreviewBody>,
) -> Result<Response, (StatusCode, String)> {
    let (n_obs, targets, gj, rt, job_p, job_active, job_msg) = {
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
        (
            n_obs,
            targets,
            gj,
            Arc::clone(rt),
            Arc::clone(&st.perturb_job_progress_permille),
            Arc::clone(&st.perturb_job_active),
            Arc::clone(&st.perturb_progress_message),
        )
    };
    job_p.store(0, Ordering::Relaxed);
    job_active.store(true, Ordering::Relaxed);
    if let Ok(mut m) = job_msg.lock() {
        *m = "GRN perturbation…".into();
    }
    let job_active_move = job_active.clone();
    let vec: Vec<f32> = tokio::task::spawn_blocking(move || {
        let _guard = PerturbJobGuard(job_active_move);
        let result = perturb_with_targets(
            &rt.bb,
            &rt.gene_mtx,
            &rt.gene_names,
            &rt.xy,
            &rt.rw_ligands_init,
            &rt.rw_tfligands_init,
            &targets,
            &rt.perturb_cfg,
            &rt.lr_radii,
            Some(&job_p),
        );
        job_p.store(1000, Ordering::Relaxed);
        result
            .delta
            .column(gj)
            .iter()
            .map(|x| *x as f32)
            .collect()
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    if vec.len() != n_obs {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "perturbation output length mismatch".into(),
        ));
    }
    Ok(binary_response(f32_vec_to_le_bytes(&vec)))
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
}

async fn api_perturb_umap_field(
    State(state): State<SharedState>,
    Json(body): Json<UmapTransitionBody>,
) -> Result<Json<UmapFieldResponse>, (StatusCode, String)> {
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
        )
    };

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
    let grid = tokio::task::spawn_blocking(move || {
        let _guard = PerturbJobGuard(job_active_move);
        job_p.store(20, Ordering::Relaxed);
        let mut delta = if quick {
            if let Ok(mut m) = job_msg.lock() {
                *m = "Local expression delta…".into();
            }
            job_p.store(120, Ordering::Relaxed);
            let d = delta_single_gene_to_target(
                &rt.gene_mtx,
                gj,
                targets[0].cell_indices.as_deref(),
                targets[0].desired_expr,
            );
            job_p.store(450, Ordering::Relaxed);
            d
        } else {
            if let Ok(mut m) = job_msg.lock() {
                *m = "GRN perturbation…".into();
            }
            let result = perturb_with_targets(
                &rt.bb,
                &rt.gene_mtx,
                &rt.gene_names,
                &rt.xy,
                &rt.rw_ligands_init,
                &rt.rw_tfligands_init,
                &targets,
                &rt.perturb_cfg,
                &rt.lr_radii,
                Some(&job_p),
            );
            job_p.store(880, Ordering::Relaxed);
            result.delta
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
        job_p.store(900, Ordering::Relaxed);
        let g = compute_umap_transition_grid(&rt.gene_mtx, &delta, &umap_pts, &params);
        job_p.store(1000, Ordering::Relaxed);
        g
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
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
    Ok(Json(UmapFieldResponse {
        nx,
        ny,
        grid_x: grid.grid_x,
        grid_y: grid.grid_y,
        u,
        v,
        cell_u,
        cell_v,
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
    let (n_obs, targets, gene, rt, gene_names, job_p, job_active, job_msg) = {
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
        (
            n_obs,
            targets,
            gene.clone(),
            Arc::clone(rt),
            rt.gene_names.clone(),
            Arc::clone(&st.perturb_job_progress_permille),
            Arc::clone(&st.perturb_job_active),
            Arc::clone(&st.perturb_progress_message),
        )
    };
    job_p.store(0, Ordering::Relaxed);
    job_active.store(true, Ordering::Relaxed);
    if let Ok(mut m) = job_msg.lock() {
        *m = "GRN perturbation (summary)…".into();
    }
    let job_active_move = job_active.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _guard = PerturbJobGuard(job_active_move);
        let r = perturb_with_targets(
            &rt.bb,
            &rt.gene_mtx,
            &rt.gene_names,
            &rt.xy,
            &rt.rw_ligands_init,
            &rt.rw_tfligands_init,
            &targets,
            &rt.perturb_cfg,
            &rt.lr_radii,
            Some(&job_p),
        );
        job_p.store(1000, Ordering::Relaxed);
        r
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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

    let perturbed_gene_idx = gene_names
        .iter()
        .position(|g| g == &gene)
        .unwrap_or(0);
    let col = result.delta.column(perturbed_gene_idx);
    let mean_delta: f64 = col.iter().sum::<f64>() / n_obs as f64;
    let max_abs_delta: f64 = col.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let n_positive = col.iter().filter(|&&v| v > 1e-12).count();
    let n_negative = col.iter().filter(|&&v| v < -1e-12).count();
    let n_zero = n_obs - n_positive - n_negative;

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

/// All HDF5 / Polars / GRN work runs here on a plain OS thread (no Tokio runtime). Starting Tokio
/// only for `axum::serve` avoids nested-runtime panics when Polars or other code calls into async
/// runtimes during `LazyFrame::collect()` etc.
fn build_app(cli: Cli) -> anyhow::Result<(SocketAddr, Router, SharedState)> {
    let dataset = match cli.h5ad.as_ref() {
        Some(p) if !p.as_os_str().is_empty() => Some(load_app_state(ViewerLoadInputs {
            h5ad: p.clone(),
            layer: cli.layer.clone(),
            cluster_annot: cli.cluster_annot.clone(),
            network_dir: cli.network_dir.clone(),
            run_toml: cli.run_toml.clone(),
        })?),
        _ => None,
    };
    let state = Arc::new(RwLock::new(AppState {
        dataset,
        default_layer: cli.layer.clone(),
        default_cluster_annot: cli.cluster_annot.clone(),
        default_network_dir: cli.network_dir.clone(),
        default_run_toml: cli.run_toml.clone(),
        perturb_bg_gen: Arc::new(AtomicU64::new(0)),
        perturb_bg_in_flight: Arc::new(AtomicBool::new(false)),
        perturb_load_progress_permille: Arc::new(AtomicU32::new(0)),
        perturb_job_progress_permille: Arc::new(AtomicU32::new(0)),
        perturb_job_active: Arc::new(AtomicBool::new(false)),
        perturb_progress_message: Arc::new(Mutex::new(String::new())),
    }));

    let api = Router::new()
        .route("/meta", get(api_meta))
        .route("/session/configure", post(api_session_configure))
        .route("/spatial", get(api_spatial))
        .route("/umap", get(api_umap))
        .route("/clusters", get(api_clusters))
        .route("/cell_type/codes", get(api_cell_type_codes))
        .route("/genes", get(api_genes))
        .route("/genes/full", get(api_genes_full))
        .route("/gene/expression", get(api_gene_expression))
        .route("/betadata/genes", get(api_betadata_genes))
        .route("/betadata/columns", get(api_betadata_columns))
        .route("/betadata/values", get(api_betadata_values))
        .route("/betadata/top", post(api_betadata_top))
        .route("/network/cell-context", post(api_network_cell_context))
        .route("/perturb/preview", post(api_perturb_preview))
        .route("/perturb/umap-field", post(api_perturb_umap_field))
        .route("/perturb/summary", post(api_perturb_summary))
        .with_state(state.clone())
        .layer(CompressionLayer::new());

    let index = cli.static_dir.join("index.html");
    let app = Router::new()
        .nest("/api", api)
        .fallback_service(
            ServeDir::new(&cli.static_dir).not_found_service(ServeFile::new(index)),
        )
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
    let (addr, app, state) = build_app(cli)?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        tracing::info!("listening on http://{}", addr);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        spawn_perturb_background_load(state.clone());
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}
