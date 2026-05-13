use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::{Method, StatusCode};
use axum::routing::{get, post};
use axum::serve;
use clap::Parser;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use spacetravlr::config::expand_user_path;
use spacetravlr::{
    FuzzyGraph, leiden_labels_from_graph, leiden_labels_subcluster_into, umap_lab_gene_expression_from_h5ad,
    umap_lab_load_pca_session, umap_lab_read_obs_column,
    umap_lab_run_embedding, RustPreprocessParams, UmapLabKnnCache, UmapLabLoaded,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

#[derive(Parser, Debug)]
#[command(name = "umap_lab")]
struct Cli {
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    #[arg(long, default_value_t = 8765)]
    port: u16,
    #[arg(long, default_value = "web/umap_lab/dist")]
    static_dir: PathBuf,
    /// Serve only `/api` (no static UI). Use with `npm run dev` in `web/umap_lab` for Vite HMR.
    #[arg(long, default_value_t = false)]
    api_only: bool,
    #[arg(long, default_value_t = true)]
    allow_cors: bool,
}

struct AppState {
    session: Mutex<Option<LoadedSession>>,
}

struct LoadedSession {
    path: PathBuf,
    pca: Array2<f64>,
    umap_param_base: RustPreprocessParams,
    color_column: Option<String>,
    color_categories: Vec<String>,
    color_codes: Vec<u32>,
    fuzzy_graph: Option<FuzzyGraph>,
    umap_knn_cache: Option<UmapLabKnnCache>,
    leiden_cache: Option<CachedLeiden>,
    /// Last **full** `POST /api/leiden` partition (per-cell labels). Subcluster updates `leiden_cache` only.
    leiden_baseline_labels: Option<Vec<String>>,
    obs_names: Vec<String>,
    obs_columns: Vec<String>,
}

#[derive(Clone)]
struct CachedLeiden {
    categories: Vec<String>,
    codes: Vec<u32>,
}

fn pack_labels(labels: &[String]) -> (Vec<String>, Vec<u32>) {
    if labels.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut cats: Vec<String> = Vec::new();
    let mut idx_by: HashMap<String, u32> = HashMap::new();
    let mut codes = Vec::with_capacity(labels.len());
    for l in labels {
        let c = *idx_by.entry(l.clone()).or_insert_with(|| {
            let i = cats.len() as u32;
            cats.push(l.clone());
            i
        });
        codes.push(c);
    }
    (cats, codes)
}

#[derive(Deserialize)]
struct LoadRequest {
    path: String,
    #[serde(default)]
    n_top_hvg: Option<usize>,
    #[serde(default)]
    n_pca_components: Option<usize>,
}

#[derive(Serialize)]
struct LoadResponse {
    path: String,
    n_cells: usize,
    n_pca_available: usize,
    color_column: Option<String>,
    color_categories: Vec<String>,
    color_codes: Vec<u32>,
    ef_construction: usize,
    obs_columns: Vec<String>,
}

#[derive(Deserialize, Default)]
struct UmapRequest {
    n_neighbors: Option<usize>,
    min_dist: Option<f32>,
    n_epochs: Option<usize>,
    ef_construction: Option<usize>,
    n_pca_components: Option<usize>,
    spread: Option<f32>,
    umap_learning_rate: Option<f32>,
}

#[derive(Serialize)]
struct UmapResponse {
    x: Vec<f32>,
    y: Vec<f32>,
    timings_sec: Vec<(String, f64)>,
}

#[derive(Serialize)]
struct StatusResponse {
    loaded: bool,
    path: Option<String>,
    n_cells: Option<usize>,
    n_pca_available: Option<usize>,
    color_column: Option<String>,
    color_categories: Option<Vec<String>>,
    color_codes: Option<Vec<u32>>,
}

fn merge_preprocess_params(req: &LoadRequest) -> RustPreprocessParams {
    let mut p = RustPreprocessParams::default();
    if let Some(n) = req.n_top_hvg {
        p.n_top_hvg = n;
    }
    if let Some(n) = req.n_pca_components {
        p.n_pca_components = n;
    }
    p
}

fn merge_umap_params(base: &RustPreprocessParams, req: &UmapRequest) -> RustPreprocessParams {
    const SPREAD_MIN: f32 = 0.1;
    const SPREAD_MAX: f32 = 1.0;

    let mut p = base.clone();
    if let Some(n) = req.n_neighbors {
        p.n_neighbors = n.max(2);
    }
    if let Some(v) = req.min_dist {
        p.min_dist = v.max(1e-6);
    }
    if let Some(n) = req.n_epochs {
        p.n_epochs = Some(n.max(2));
    }
    if let Some(n) = req.ef_construction {
        p.ef_construction = n.max(4);
    }
    if let Some(n) = req.n_pca_components {
        p.n_pca_components = n.max(2);
    }
    if let Some(v) = req.spread {
        p.spread = v.clamp(SPREAD_MIN, SPREAD_MAX);
    }
    if let Some(v) = req.umap_learning_rate {
        p.umap_learning_rate = v.max(1e-6);
    }
    p.spread = p.spread.clamp(SPREAD_MIN, SPREAD_MAX);
    let (md, sp) = spacetravlr::rust_preprocess::clamp_umap_min_dist_spread(p.min_dist, p.spread);
    p.min_dist = md;
    p.spread = sp;
    p
}

async fn api_status(State(st): State<Arc<AppState>>) -> Json<StatusResponse> {
    let g = st.session.lock().unwrap();
    let Some(s) = g.as_ref() else {
        return Json(StatusResponse {
            loaded: false,
            path: None,
            n_cells: None,
            n_pca_available: None,
            color_column: None,
            color_categories: None,
            color_codes: None,
        });
    };
    Json(StatusResponse {
        loaded: true,
        path: Some(s.path.to_string_lossy().to_string()),
        n_cells: Some(s.pca.nrows()),
        n_pca_available: Some(s.pca.ncols()),
        color_column: s.color_column.clone(),
        color_categories: Some(s.color_categories.clone()),
        color_codes: Some(s.color_codes.clone()),
    })
}

async fn api_load(
    State(st): State<Arc<AppState>>,
    Json(body): Json<LoadRequest>,
) -> Result<Json<LoadResponse>, (StatusCode, String)> {
    let expanded = expand_user_path(body.path.trim());
    let path = PathBuf::from(expanded);
    if !path.is_file() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("not a file: {}", path.display()),
        ));
    }
    let prep = merge_preprocess_params(&body);
    let loaded: UmapLabLoaded = tokio::task::spawn_blocking({
        let path = path.clone();
        let prep = prep.clone();
        move || umap_lab_load_pca_session(&path, &prep).map_err(|e| format!("{:#}", e))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let n_cells = loaded.pca.nrows();
    let n_pca_available = loaded.pca.ncols();
    let color_column = loaded.color_column.clone();
    let (color_categories, color_codes) = pack_labels(&loaded.color_labels);
    let ef_construction = prep.ef_construction;
    let obs_columns = loaded.obs_columns.clone();

    let sess = LoadedSession {
        path: path.clone(),
        pca: loaded.pca,
        umap_param_base: prep,
        color_column: color_column.clone(),
        color_categories: color_categories.clone(),
        color_codes: color_codes.clone(),
        fuzzy_graph: None,
        umap_knn_cache: None,
        leiden_cache: None,
        leiden_baseline_labels: None,
        obs_names: loaded.obs_names,
        obs_columns: loaded.obs_columns,
    };
    *st.session.lock().unwrap() = Some(sess);

    Ok(Json(LoadResponse {
        path: path.to_string_lossy().to_string(),
        n_cells,
        n_pca_available,
        color_column,
        color_categories,
        color_codes,
        ef_construction,
        obs_columns,
    }))
}

async fn api_umap(
    State(st): State<Arc<AppState>>,
    Json(body): Json<UmapRequest>,
) -> Result<Json<UmapResponse>, (StatusCode, String)> {
    let (pca, mut params, knn_cache_in) = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "load a dataset first (POST /api/load)".into(),
            ));
        };
        (
            s.pca.clone(),
            merge_umap_params(&s.umap_param_base, &body),
            s.umap_knn_cache.clone(),
        )
    };
    params.n_pca_components = params
        .n_pca_components
        .max(2)
        .min(pca.ncols());

    let (emb, graph, timings, new_knn_cache) = tokio::task::spawn_blocking(move || {
        umap_lab_run_embedding(&pca, &params, knn_cache_in.as_ref()).map_err(|e| format!("{:#}", e))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let n = emb.nrows();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        x.push(emb[(i, 0)]);
        y.push(emb[(i, 1)]);
    }

    {
        let mut g = st.session.lock().unwrap();
        if let Some(s) = g.as_mut() {
            s.fuzzy_graph = Some(graph);
            s.umap_knn_cache = Some(new_knn_cache);
            s.leiden_cache = None;
            s.leiden_baseline_labels = None;
        }
    }

    Ok(Json(UmapResponse {
        x,
        y,
        timings_sec: timings,
    }))
}

#[derive(Deserialize)]
struct LeidenRequest {
    #[serde(default = "default_leiden_resolution")]
    resolution: f64,
}

fn default_leiden_resolution() -> f64 {
    1.0
}

#[derive(Serialize)]
struct LeidenResponse {
    labels: Vec<String>,
    categories: Vec<String>,
    codes: Vec<u32>,
    n_clusters: usize,
    elapsed_sec: f64,
}

async fn api_leiden(
    State(st): State<Arc<AppState>>,
    Json(body): Json<LeidenRequest>,
) -> Result<Json<LeidenResponse>, (StatusCode, String)> {
    let graph = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "load a dataset first (POST /api/load)".into(),
            ));
        };
        let Some(ref fg) = s.fuzzy_graph else {
            return Err((
                StatusCode::BAD_REQUEST,
                "run UMAP first to build the fuzzy graph".into(),
            ));
        };
        fg.clone()
    };

    let resolution = body.resolution.max(0.01);

    let (labels, elapsed) = tokio::task::spawn_blocking(move || {
        let t = std::time::Instant::now();
        let labels = leiden_labels_from_graph(&graph, resolution, 100);
        (labels, t.elapsed().as_secs_f64())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let n_clusters = {
        let mut seen = std::collections::HashSet::new();
        for l in &labels {
            seen.insert(l.as_str());
        }
        seen.len()
    };

    let (categories, codes) = pack_labels(&labels);

    {
        let mut g = st.session.lock().unwrap();
        if let Some(s) = g.as_mut() {
            s.leiden_baseline_labels = Some(labels.clone());
            s.leiden_cache = Some(CachedLeiden {
                categories: categories.clone(),
                codes: codes.clone(),
            });
        }
    }

    Ok(Json(LeidenResponse {
        labels,
        categories,
        codes,
        n_clusters,
        elapsed_sec: elapsed,
    }))
}

#[derive(Deserialize)]
struct LeidenSubclusterRequest {
    parent_code: u32,
    #[serde(default = "default_leiden_resolution")]
    resolution: f64,
}

async fn api_leiden_subcluster(
    State(st): State<Arc<AppState>>,
    Json(body): Json<LeidenSubclusterRequest>,
) -> Result<Json<LeidenResponse>, (StatusCode, String)> {
    let (graph, cache) = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "load a dataset first (POST /api/load)".into(),
            ));
        };
        let Some(ref fg) = s.fuzzy_graph else {
            return Err((
                StatusCode::BAD_REQUEST,
                "run UMAP first to build the fuzzy graph".into(),
            ));
        };
        let Some(ref c) = s.leiden_cache else {
            return Err((
                StatusCode::BAD_REQUEST,
                "run full Leiden first (POST /api/leiden), then pick a cluster".into(),
            ));
        };
        (fg.clone(), c.clone())
    };

    let resolution = body.resolution.max(0.01);
    let parent_code = body.parent_code;

    let (labels, elapsed) = tokio::task::spawn_blocking(move || {
        let t = std::time::Instant::now();
        let out = leiden_labels_subcluster_into(
            &graph,
            &cache.codes,
            &cache.categories,
            parent_code,
            resolution,
            100,
        )
        .map_err(|e| e.to_string())?;
        Ok::<_, String>((out, t.elapsed().as_secs_f64()))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let n_clusters = {
        let mut seen = std::collections::HashSet::new();
        for l in &labels {
            seen.insert(l.as_str());
        }
        seen.len()
    };
    let (categories, codes) = pack_labels(&labels);

    {
        let mut g = st.session.lock().unwrap();
        if let Some(s) = g.as_mut() {
            s.leiden_cache = Some(CachedLeiden {
                categories: categories.clone(),
                codes: codes.clone(),
            });
        }
    }

    Ok(Json(LeidenResponse {
        labels,
        categories,
        codes,
        n_clusters,
        elapsed_sec: elapsed,
    }))
}

async fn api_leiden_reset(
    State(st): State<Arc<AppState>>,
) -> Result<Json<LeidenResponse>, (StatusCode, String)> {
    let (labels, categories, codes, n_clusters) = {
        let mut g = st.session.lock().unwrap();
        let Some(s) = g.as_mut() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "load a dataset first (POST /api/load)".into(),
            ));
        };
        let Some(ref baseline) = s.leiden_baseline_labels else {
            return Err((
                StatusCode::BAD_REQUEST,
                "run full Leiden (POST /api/leiden) before reset".into(),
            ));
        };
        let labels = baseline.clone();
        let (categories, codes) = pack_labels(&labels);
        let mut seen = std::collections::HashSet::new();
        for l in &labels {
            seen.insert(l.as_str());
        }
        let n_clusters = seen.len();
        s.leiden_cache = Some(CachedLeiden {
            categories: categories.clone(),
            codes: codes.clone(),
        });
        (labels, categories, codes, n_clusters)
    };

    Ok(Json(LeidenResponse {
        labels,
        categories,
        codes,
        n_clusters,
        elapsed_sec: 0.0,
    }))
}

#[derive(Deserialize)]
struct GeneRequest {
    gene: String,
}

#[derive(Serialize)]
struct GeneResponse {
    gene: String,
    values: Vec<f32>,
    vmin: f32,
    vmax: f32,
}

async fn api_gene(
    State(st): State<Arc<AppState>>,
    Json(body): Json<GeneRequest>,
) -> Result<Json<GeneResponse>, (StatusCode, String)> {
    let (path, n_expected) = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "load a dataset first (POST /api/load)".into(),
            ));
        };
        (s.path.clone(), s.pca.nrows())
    };
    let gene = body.gene.trim().to_string();
    if gene.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "gene name is empty".into()));
    }
    let (resolved, values, vmin, vmax) = tokio::task::spawn_blocking(move || {
        umap_lab_gene_expression_from_h5ad(&path, &gene).map_err(|e| format!("{:#}", e))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    if values.len() != n_expected {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "gene vector length {} does not match loaded session ({} cells)",
                values.len(),
                n_expected
            ),
        ));
    }

    Ok(Json(GeneResponse {
        gene: resolved,
        values,
        vmin,
        vmax,
    }))
}

#[derive(Deserialize)]
struct ColorByRequest {
    column: String,
}

#[derive(Serialize)]
struct ColorByResponse {
    column: String,
    categories: Vec<String>,
    codes: Vec<u32>,
}

async fn api_color_by(
    State(st): State<Arc<AppState>>,
    Json(body): Json<ColorByRequest>,
) -> Result<Json<ColorByResponse>, (StatusCode, String)> {
    let (path, obs_columns) = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((StatusCode::BAD_REQUEST, "load a dataset first".into()));
        };
        (s.path.clone(), s.obs_columns.clone())
    };
    let column = body.column.trim().to_string();
    if column.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "column name is empty".into()));
    }
    if !obs_columns.contains(&column) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("column {column:?} not in obs; available: {obs_columns:?}"),
        ));
    }
    let col = column.clone();
    let labels = tokio::task::spawn_blocking(move || {
        umap_lab_read_obs_column(&path, &col).map_err(|e| format!("{:#}", e))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let (categories, codes) = pack_labels(&labels);
    {
        let mut g = st.session.lock().unwrap();
        if let Some(s) = g.as_mut() {
            s.color_column = Some(column.clone());
            s.color_categories = categories.clone();
            s.color_codes = codes.clone();
        }
    }
    Ok(Json(ColorByResponse {
        column,
        categories,
        codes,
    }))
}

#[derive(Deserialize)]
struct MaltRequest {
    reference_path: String,
    #[serde(default)]
    groupby: Option<String>,
    #[serde(default)]
    outdir: Option<String>,
    #[serde(default)]
    no_leiden_map: Option<bool>,
}

#[derive(Serialize)]
struct MaltResponse {
    outdir: String,
    csv_path: String,
    csv_columns: Vec<String>,
    elapsed_sec: f64,
}

async fn api_malt(
    State(st): State<Arc<AppState>>,
    Json(body): Json<MaltRequest>,
) -> Result<Json<MaltResponse>, (StatusCode, String)> {
    let query_path = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((StatusCode::BAD_REQUEST, "load a dataset first".into()));
        };
        s.path.clone()
    };

    let ref_expanded = expand_user_path(body.reference_path.trim());
    let ref_path = PathBuf::from(&ref_expanded);
    if !ref_path.is_file() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("reference not a file: {}", ref_path.display()),
        ));
    }

    let outdir = body
        .outdir
        .as_deref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| expand_user_path(s))
        .unwrap_or_else(|| "/tmp/malt_results".to_string());

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/malt_label_transfer.py");
    if !script.is_file() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("MALT script not found at {}", script.display()),
        ));
    }

    let groupby = body.groupby.clone();
    let no_leiden = body.no_leiden_map.unwrap_or(false);
    let outdir2 = outdir.clone();

    let (status, elapsed) = tokio::task::spawn_blocking(move || {
        let mut cmd = std::process::Command::new("python");
        cmd.env("PYTHONUNBUFFERED", "1")
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .arg(&script)
            .arg("--reference")
            .arg(&ref_path)
            .arg("--query")
            .arg(&query_path)
            .arg("--outdir")
            .arg(&outdir2);

        if let Some(ref gb) = groupby {
            let gb = gb.trim();
            if !gb.is_empty() {
                cmd.arg("--groupby").arg(gb);
            }
        }
        if no_leiden {
            cmd.arg("--no-leiden-map");
        }

        tracing::info!("Running MALT: {:?}", cmd);

        let t0 = std::time::Instant::now();
        let status = cmd
            .status()
            .map_err(|e| format!("failed to spawn MALT python: {e}"))?;

        Ok::<_, String>((status, t0.elapsed().as_secs_f64()))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    if !status.success() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "MALT exited with code {:?} (stdout/stderr were streamed to this process)",
                status.code(),
            ),
        ));
    }

    let csv_path = PathBuf::from(&outdir).join("malt_labels.csv");
    if !csv_path.is_file() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "MALT completed but malt_labels.csv not found at {}",
                csv_path.display()
            ),
        ));
    }

    let csv_columns = {
        let rdr = std::io::BufRead::lines(std::io::BufReader::new(
            std::fs::File::open(&csv_path).map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("open CSV: {e}"))
            })?,
        ));
        let first_line = rdr
            .into_iter()
            .next()
            .transpose()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("read CSV: {e}")))?
            .unwrap_or_default();
        first_line
            .split(',')
            .skip(1)
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>()
    };

    tracing::info!(
        "MALT finished in {:.1}s → {} ({} columns)",
        elapsed,
        csv_path.display(),
        csv_columns.len(),
    );

    Ok(Json(MaltResponse {
        outdir,
        csv_path: csv_path.to_string_lossy().to_string(),
        csv_columns,
        elapsed_sec: elapsed,
    }))
}

#[derive(Deserialize)]
struct LoadCsvRequest {
    csv_path: String,
    column: String,
}

#[derive(Serialize)]
struct LoadCsvResponse {
    column: String,
    categories: Vec<String>,
    codes: Vec<u32>,
    n_matched: usize,
    n_missing: usize,
}

async fn api_load_csv(
    State(st): State<Arc<AppState>>,
    Json(body): Json<LoadCsvRequest>,
) -> Result<Json<LoadCsvResponse>, (StatusCode, String)> {
    let obs_names = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((StatusCode::BAD_REQUEST, "load a dataset first".into()));
        };
        s.obs_names.clone()
    };

    let csv_path = PathBuf::from(expand_user_path(body.csv_path.trim()));
    if !csv_path.is_file() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("CSV file not found: {}", csv_path.display()),
        ));
    }
    let column = body.column.trim().to_string();

    let (labels, n_matched, n_missing) = tokio::task::spawn_blocking(move || {
        let content = std::fs::read_to_string(&csv_path)
            .map_err(|e| format!("read CSV: {e}"))?;
        let mut lines = content.lines();
        let header_line = lines.next().ok_or("CSV is empty")?;
        let headers: Vec<&str> = header_line.split(',').map(|s| s.trim()).collect();
        let col_idx = headers
            .iter()
            .position(|&h| h == column)
            .ok_or_else(|| {
                format!(
                    "column {column:?} not in CSV headers: {headers:?}"
                )
            })?;
        let idx_col = headers.iter().position(|&h| h == "obs_name").unwrap_or(0);

        let mut csv_map: HashMap<String, String> = HashMap::new();
        for line in lines {
            let fields: Vec<&str> = line.split(',').collect();
            let key = fields.get(idx_col).unwrap_or(&"").trim().to_string();
            let val = fields.get(col_idx).unwrap_or(&"").trim().to_string();
            if !key.is_empty() {
                csv_map.insert(key, val);
            }
        }

        let mut labels = Vec::with_capacity(obs_names.len());
        let mut matched = 0usize;
        let mut missing = 0usize;
        for name in &obs_names {
            if let Some(val) = csv_map.get(name) {
                labels.push(val.clone());
                matched += 1;
            } else {
                labels.push("unmapped".to_string());
                missing += 1;
            }
        }
        Ok::<_, String>((labels, matched, missing))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let (categories, codes) = pack_labels(&labels);
    let col_name = body.column.trim().to_string();
    {
        let mut g = st.session.lock().unwrap();
        if let Some(s) = g.as_mut() {
            s.color_column = Some(col_name.clone());
            s.color_categories = categories.clone();
            s.color_codes = codes.clone();
        }
    }

    Ok(Json(LoadCsvResponse {
        column: col_name,
        categories,
        codes,
        n_matched,
        n_missing,
    }))
}

#[derive(Serialize)]
struct MaltOptimizedResponse {
    column: String,
    categories: Vec<String>,
    codes: Vec<u32>,
    n_subsample: usize,
    n_total: usize,
    min_cluster_count: usize,
    elapsed_sec: f64,
}

async fn api_malt_optimized(
    State(st): State<Arc<AppState>>,
    Json(body): Json<MaltRequest>,
) -> Result<Json<MaltOptimizedResponse>, (StatusCode, String)> {
    let (query_path, obs_names, leiden) = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((StatusCode::BAD_REQUEST, "load a dataset first".into()));
        };
        let Some(ref lc) = s.leiden_cache else {
            return Err((
                StatusCode::BAD_REQUEST,
                "run Leiden first — the optimized path subsamples per cluster".into(),
            ));
        };
        (
            s.path.clone(),
            s.obs_names.clone(),
            lc.clone(),
        )
    };

    let ref_expanded = expand_user_path(body.reference_path.trim());
    let ref_path = PathBuf::from(&ref_expanded);
    if !ref_path.is_file() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("reference not a file: {}", ref_path.display()),
        ));
    }

    let groupby = body.groupby.clone();
    let no_leiden = body.no_leiden_map.unwrap_or(true);

    let n_total = obs_names.len();
    let n_cats = leiden.categories.len();

    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); n_cats];
    for (i, &code) in leiden.codes.iter().enumerate() {
        cluster_indices[code as usize].push(i);
    }
    let min_count = cluster_indices
        .iter()
        .filter(|v| !v.is_empty())
        .map(|v| v.len())
        .min()
        .unwrap_or(0);

    if min_count == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "Leiden produced an empty cluster — cannot subsample".into(),
        ));
    }

    let mut subsample_indices: Vec<usize> = Vec::with_capacity(min_count * n_cats);
    {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        for idxs in &mut cluster_indices {
            idxs.shuffle(&mut rng);
            subsample_indices.extend_from_slice(&idxs[..min_count]);
        }
    }
    subsample_indices.sort_unstable();

    let subset_names: Vec<String> = subsample_indices.iter().map(|&i| obs_names[i].clone()).collect();
    let n_subsample = subset_names.len();
    tracing::info!(
        "MALT optimized: subsampling {n_subsample}/{n_total} cells (min cluster = {min_count}, {n_cats} clusters)"
    );

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/malt_subsample_helper.py");
    if !script.is_file() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("subsample helper not found at {}", script.display()),
        ));
    }

    let (subset_csv_content, elapsed) = tokio::task::spawn_blocking(move || {
        let tmpdir = tempfile::tempdir().map_err(|e| format!("tmpdir: {e}"))?;
        let names_json = tmpdir.path().join("subset_names.json");
        let outdir = tmpdir.path().join("malt_out");
        std::fs::create_dir_all(&outdir).map_err(|e| format!("mkdir: {e}"))?;

        std::fs::write(
            &names_json,
            serde_json::to_string(&subset_names).map_err(|e| format!("json: {e}"))?,
        )
        .map_err(|e| format!("write names json: {e}"))?;

        let mut cmd = std::process::Command::new("python");
        cmd.env("PYTHONUNBUFFERED", "1")
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .arg(&script)
            .arg("--query")
            .arg(&query_path)
            .arg("--reference")
            .arg(&ref_path)
            .arg("--subset-names-json")
            .arg(&names_json)
            .arg("--outdir")
            .arg(&outdir);

        if let Some(ref gb) = groupby {
            let gb = gb.trim();
            if !gb.is_empty() {
                cmd.arg("--groupby").arg(gb);
            }
        }
        if no_leiden {
            cmd.arg("--no-leiden-map");
        }

        tracing::info!("Running MALT subsample: {:?}", cmd);
        let t0 = std::time::Instant::now();
        let status = cmd
            .status()
            .map_err(|e| format!("spawn: {e}"))?;
        let elapsed = t0.elapsed().as_secs_f64();

        if !status.success() {
            return Err(format!(
                "MALT subsample helper exited {:?} (stdout/stderr were streamed to this process)",
                status.code(),
            ));
        }

        let csv_path = outdir.join("malt_labels.csv");
        if !csv_path.is_file() {
            return Err("MALT produced no malt_labels.csv".into());
        }
        let csv_content = std::fs::read_to_string(&csv_path)
            .map_err(|e| format!("read CSV: {e}"))?;

        // tmpdir drops here, cleaning up everything
        Ok::<_, String>((csv_content, elapsed))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // Parse subset CSV into a map: obs_name → label
    let (subset_label_map, label_col_name) = {
        let mut lines = subset_csv_content.lines();
        let header = lines.next().unwrap_or("");
        let headers: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
        let malt_col = headers
            .iter()
            .position(|h| h.starts_with("malt_label"))
            .unwrap_or(1);
        let idx_col = headers.iter().position(|&h| h == "obs_name").unwrap_or(0);
        let col_name = headers.get(malt_col).unwrap_or(&"malt_label").to_string();

        let mut map: HashMap<String, String> = HashMap::new();
        for line in lines {
            let fields: Vec<&str> = line.split(',').collect();
            let key = fields.get(idx_col).unwrap_or(&"").trim().to_string();
            let val = fields.get(malt_col).unwrap_or(&"").trim().to_string();
            if !key.is_empty() {
                map.insert(key, val);
            }
        }
        (map, col_name)
    };

    // Project labels to all cells via Leiden cluster majority vote
    let mut cluster_label_votes: Vec<HashMap<&str, usize>> = vec![HashMap::new(); n_cats];
    for &i in &subsample_indices {
        let cluster = leiden.codes[i] as usize;
        let label = subset_label_map
            .get(&obs_names[i])
            .map(|s| s.as_str())
            .unwrap_or("unmapped");
        *cluster_label_votes[cluster].entry(label).or_insert(0) += 1;
    }
    let cluster_majority: Vec<String> = cluster_label_votes
        .iter()
        .map(|votes| {
            votes
                .iter()
                .max_by_key(|(_, count)| **count)
                .map(|(label, _)| label.to_string())
                .unwrap_or_else(|| "unmapped".to_string())
        })
        .collect();

    let labels: Vec<String> = leiden
        .codes
        .iter()
        .map(|&code| cluster_majority[code as usize].clone())
        .collect();

    let (categories, codes) = pack_labels(&labels);
    {
        let mut g = st.session.lock().unwrap();
        if let Some(s) = g.as_mut() {
            s.color_column = Some(label_col_name.clone());
            s.color_categories = categories.clone();
            s.color_codes = codes.clone();
        }
    }

    tracing::info!(
        "MALT optimized done in {:.1}s: {n_subsample} subset → {n_total} projected ({} categories)",
        elapsed,
        categories.len(),
    );

    Ok(Json(MaltOptimizedResponse {
        column: label_col_name,
        categories,
        codes,
        n_subsample,
        n_total,
        min_cluster_count: min_count,
        elapsed_sec: elapsed,
    }))
}

#[derive(Deserialize)]
struct ExportCsvRequest {
    #[serde(default)]
    annotations: HashMap<String, String>,
}

async fn api_export_csv(
    State(st): State<Arc<AppState>>,
    Json(body): Json<ExportCsvRequest>,
) -> Result<(StatusCode, [(axum::http::header::HeaderName, String); 2], String), (StatusCode, String)> {
    let g = st.session.lock().unwrap();
    let Some(s) = g.as_ref() else {
        return Err((StatusCode::BAD_REQUEST, "load a dataset first".into()));
    };

    let n = s.obs_names.len();
    let mut csv = String::with_capacity(n * 80);

    let has_leiden = s.leiden_cache.is_some();
    let has_color = !s.color_categories.is_empty() && s.color_codes.len() == n;
    let color_col_name = s.color_column.as_deref().unwrap_or("color_label");
    let has_annotations = !body.annotations.is_empty();

    csv.push_str("obs_name");
    if has_leiden {
        csv.push_str(",leiden");
    }
    if has_color {
        csv.push(',');
        csv.push_str(color_col_name);
    }
    if has_annotations {
        csv.push_str(",annotation");
    }
    csv.push('\n');

    let leiden_ref = s.leiden_cache.as_ref();
    for i in 0..n {
        csv.push_str(&s.obs_names[i]);
        if let Some(lc) = leiden_ref {
            csv.push(',');
            csv.push_str(&lc.categories[lc.codes[i] as usize]);
        }
        if has_color {
            csv.push(',');
            csv.push_str(&s.color_categories[s.color_codes[i] as usize]);
        }
        if has_annotations {
            csv.push(',');
            let raw_label = if let Some(lc) = leiden_ref {
                &lc.categories[lc.codes[i] as usize]
            } else if has_color {
                &s.color_categories[s.color_codes[i] as usize]
            } else {
                ""
            };
            if let Some(ann) = body.annotations.get(raw_label) {
                csv.push_str(ann);
            }
        }
        csv.push('\n');
    }

    Ok((
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/csv; charset=utf-8".to_string()),
            (
                axum::http::header::CONTENT_DISPOSITION,
                "attachment; filename=\"umap_lab_export.csv\"".to_string(),
            ),
        ],
        csv,
    ))
}

fn resolve_static_dir(cli: &Path) -> anyhow::Result<PathBuf> {
    fn has_index(dir: &Path) -> bool {
        dir.join("index.html").is_file()
    }

    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    if cli.is_absolute() {
        if has_index(cli) {
            return Ok(cli.to_path_buf());
        }
        anyhow::bail!(
            "missing {}; run `npm ci && npm run build` in web/umap_lab",
            cli.join("index.html").display()
        );
    }

    let cwd = std::env::current_dir().context("cwd")?;
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(cwd.join(cli));
    if let Ok(suffix) = cli.strip_prefix("web/umap_lab/") {
        candidates.push(cwd.join(suffix));
    }
    candidates.push(cwd.join("dist"));
    candidates.push(crate_root.join(cli));

    for c in candidates {
        if has_index(&c) {
            return Ok(c);
        }
    }

    anyhow::bail!(
        "could not find index.html for --static-dir {:?} (cwd {}). Tried cwd-relative paths and {}. Run `npm ci && npm run build` in web/umap_lab.",
        cli,
        cwd.display(),
        crate_root.join(cli).display()
    )
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    spacetravlr::ensure_hdf5_no_file_locking();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "umap_lab=info,tower_http=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let state = Arc::new(AppState {
        session: Mutex::new(None),
    });

    let mut api = Router::new()
        .route("/status", get(api_status))
        .route("/load", post(api_load))
        .route("/umap", post(api_umap))
        .route("/leiden", post(api_leiden))
        .route("/leiden/subcluster", post(api_leiden_subcluster))
        .route("/leiden/reset", post(api_leiden_reset))
        .route("/gene", post(api_gene))
        .route("/color_by", post(api_color_by))
        .route("/malt", post(api_malt))
        .route("/malt_optimized", post(api_malt_optimized))
        .route("/load_csv", post(api_load_csv))
        .route("/export_csv", post(api_export_csv))
        .with_state(state.clone());

    if cli.allow_cors {
        api = api.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(Any),
        );
    }

    let app = if cli.api_only {
        tracing::info!(
            "API only (no bundled UI). For hot reload: `cd web/umap_lab && npm run dev`, then open the Vite URL (proxies /api to this server)."
        );
        Router::new()
            .nest("/api", api)
            .layer(TraceLayer::new_for_http())
    } else {
        let static_dir = resolve_static_dir(cli.static_dir.as_path())?;
        tracing::info!(
            "Serving UI from {}. For hot reload without rebuilds: `cd web/umap_lab && npm run dev` (Vite proxies /api to this port).",
            static_dir.display()
        );
        let index = static_dir.join("index.html");
        let static_files = ServeDir::new(&static_dir).fallback(ServeFile::new(index));
        Router::new()
            .nest("/api", api)
            .fallback_service(static_files)
            .layer(TraceLayer::new_for_http())
    };

    let addr: SocketAddr = format!("{}:{}", cli.bind, cli.port)
        .parse()
        .context("bind address")?;
    tracing::info!("UMAP lab API → http://{}/api/…", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    serve(listener, app).await?;
    Ok(())
}
