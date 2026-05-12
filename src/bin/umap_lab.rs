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
    umap_lab_load_pca_session, umap_lab_run_embedding, RustPreprocessParams, UmapLabLoaded,
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
        move || umap_lab_load_pca_session(&path, &prep).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let n_cells = loaded.pca.nrows();
    let n_pca_available = loaded.pca.ncols();
    let color_column = loaded.color_column.clone();
    let (color_categories, color_codes) = pack_labels(&loaded.color_labels);

    let sess = LoadedSession {
        path: path.clone(),
        pca: loaded.pca,
        umap_param_base: prep,
        color_column: color_column.clone(),
        color_categories: color_categories.clone(),
        color_codes: color_codes.clone(),
    };
    *st.session.lock().unwrap() = Some(sess);

    Ok(Json(LoadResponse {
        path: path.to_string_lossy().to_string(),
        n_cells,
        n_pca_available,
        color_column,
        color_categories,
        color_codes,
    }))
}

async fn api_umap(
    State(st): State<Arc<AppState>>,
    Json(body): Json<UmapRequest>,
) -> Result<Json<UmapResponse>, (StatusCode, String)> {
    let (pca, mut params) = {
        let g = st.session.lock().unwrap();
        let Some(s) = g.as_ref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "load a dataset first (POST /api/load)".into(),
            ));
        };
        (s.pca.clone(), merge_umap_params(&s.umap_param_base, &body))
    };
    params.n_pca_components = params
        .n_pca_components
        .max(2)
        .min(pca.ncols());

    let (emb, timings) = tokio::task::spawn_blocking(move || {
        umap_lab_run_embedding(&pca, &params).map_err(|e| e.to_string())
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

    Ok(Json(UmapResponse {
        x,
        y,
        timings_sec: timings,
    }))
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
        .with_state(state.clone());

    if cli.allow_cors {
        api = api.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(Any),
        );
    }

    let static_dir = resolve_static_dir(cli.static_dir.as_path())?;
    let index = static_dir.join("index.html");
    let static_files = ServeDir::new(&static_dir).fallback(ServeFile::new(index.clone()));
    let app = Router::new()
        .nest("/api", api)
        .fallback_service(static_files)
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", cli.bind, cli.port)
        .parse()
        .context("bind address")?;
    tracing::info!("UMAP lab → http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    serve(listener, app).await?;
    Ok(())
}
