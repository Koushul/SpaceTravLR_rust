use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use clap::Parser;
use serde::{Deserialize, Serialize};
use spacetravlr::cell_comm_network::{
    CellCommNetworkParams, communication_edges_from_receiver, communication_edges_from_sender,
};
use spacetravlr::perturb_mode::PerturbRuntime;
use tokio::sync::RwLock;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

const INDEX_HTML: &str = include_str!("cell_comm_viewer/index.html");

#[derive(Parser, Debug)]
#[command(name = "cell-comm-viewer")]
struct Cli {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 3847)]
    port: u16,
    #[arg(
        long = "run-toml",
        help = "Optional path to spacetravlr_run_repro.toml; if set, dataset loads at startup."
    )]
    run_toml: Option<PathBuf>,
    #[arg(long, default_value_t = 24)]
    top_k: usize,
    #[arg(long, default_value_t = 0.0)]
    threshold: f64,
    #[arg(long)]
    no_contact_cutoff: bool,
    #[arg(long, default_value_t = 50_000usize)]
    max_cells_json: usize,
}

#[derive(Clone)]
struct AppState {
    rt: Arc<RwLock<Option<Arc<PerturbRuntime>>>>,
    params: CellCommNetworkParams,
    max_cells_json: usize,
    default_context_top: usize,
}

fn effective_params(st: &AppState, rt: &PerturbRuntime) -> CellCommNetworkParams {
    let mut p = st.params.clone();
    p.beta_scale_factor = rt.perturb_cfg.beta_scale_factor as f32;
    p
}

#[derive(Serialize)]
struct MetaResponse {
    ready: bool,
    n_cells: usize,
    n_genes: usize,
    n_targets: usize,
    run_dir: Option<String>,
    message: Option<String>,
    context_top: usize,
}

#[derive(Serialize)]
struct CellDto {
    i: usize,
    id: String,
    x: f64,
    y: f64,
}

#[derive(Serialize)]
struct CellsResponse {
    cells: Vec<CellDto>,
    truncated: bool,
    n_total: usize,
}

#[derive(Serialize)]
struct NeighborDto {
    i: usize,
    id: String,
    w: f64,
}

#[derive(Serialize)]
struct ContextResponse {
    i: usize,
    id: String,
    x: f64,
    y: f64,
    incoming: Vec<NeighborDto>,
    outgoing: Vec<NeighborDto>,
}

#[derive(Deserialize)]
struct LoadBody {
    run_toml: String,
}

#[derive(Deserialize)]
struct GenesQuery {
    #[serde(default)]
    q: String,
    #[serde(default = "default_gene_limit")]
    limit: usize,
}

fn default_gene_limit() -> usize {
    80
}

async fn get_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn get_meta(State(st): State<AppState>) -> Json<MetaResponse> {
    let g = st.rt.read().await;
    let Some(rt) = g.as_ref() else {
        return Json(MetaResponse {
            ready: false,
            n_cells: 0,
            n_genes: 0,
            n_targets: 0,
            run_dir: None,
            message: Some("Load a run via the panel or POST /api/load.".into()),
            context_top: st.default_context_top,
        });
    };
    Json(MetaResponse {
        ready: true,
        n_cells: rt.obs_names.len(),
        n_genes: rt.gene_names.len(),
        n_targets: rt.bb.data.len(),
        run_dir: rt.run_dir.to_str().map(String::from),
        message: None,
        context_top: st.default_context_top,
    })
}

async fn get_cells(State(st): State<AppState>) -> impl IntoResponse {
    let g = st.rt.read().await;
    let Some(rt) = g.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error":"not loaded"}))).into_response();
    };
    let n = rt.obs_names.len();
    let cap = st.max_cells_json.min(n);
    let truncated = cap < n;
    let cells: Vec<CellDto> = (0..cap)
        .map(|i| CellDto {
            i,
            id: rt.obs_names[i].clone(),
            x: rt.xy[[i, 0]],
            y: if rt.xy.ncols() > 1 {
                rt.xy[[i, 1]]
            } else {
                0.0
            },
        })
        .collect();
    (StatusCode::OK, Json(CellsResponse { cells, truncated, n_total: n })).into_response()
}

#[derive(Deserialize)]
struct ContextQuery {
    top: Option<usize>,
}

async fn get_context(
    State(st): State<AppState>,
    Path(i): Path<usize>,
    Query(q): Query<ContextQuery>,
) -> impl IntoResponse {
    let top = q
        .top
        .unwrap_or(st.default_context_top)
        .max(1)
        .min(256);
    let g = st.rt.read().await;
    let Some(rt) = g.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error":"not loaded"}))).into_response();
    };
    if i >= rt.obs_names.len() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"cell index out of range"}))).into_response();
    }
    let eff = effective_params(&st, rt);
    let inc = communication_edges_from_receiver(rt.as_ref(), &eff, i, top);
    let out = communication_edges_from_sender(rt.as_ref(), &eff, i, top);
    let incoming: Vec<NeighborDto> = inc
        .into_iter()
        .map(|(j, w)| NeighborDto {
            i: j,
            id: rt.obs_names[j].clone(),
            w,
        })
        .collect();
    let outgoing: Vec<NeighborDto> = out
        .into_iter()
        .map(|(j, w)| NeighborDto {
            i: j,
            id: rt.obs_names[j].clone(),
            w,
        })
        .collect();
    let body = ContextResponse {
        i,
        id: rt.obs_names[i].clone(),
        x: rt.xy[[i, 0]],
        y: if rt.xy.ncols() > 1 {
            rt.xy[[i, 1]]
        } else {
            0.0
        },
        incoming,
        outgoing,
    };
    (StatusCode::OK, Json(body)).into_response()
}

async fn get_genes(State(st): State<AppState>, Query(q): Query<GenesQuery>) -> Json<Vec<String>> {
    let needle = q.q.trim().to_ascii_lowercase();
    let g = st.rt.read().await;
    let Some(rt) = g.as_ref() else {
        return Json(vec![]);
    };
    let lim = q.limit.min(500);
    let mut out: Vec<String> = rt
        .gene_names
        .iter()
        .filter(|g| {
            if needle.is_empty() {
                return true;
            }
            g.to_ascii_lowercase().contains(&needle)
        })
        .take(lim)
        .cloned()
        .collect();
    out.sort();
    Json(out)
}

async fn get_gene_expr(
    State(st): State<AppState>,
    Path(gene): Path<String>,
) -> impl IntoResponse {
    let g = st.rt.read().await;
    let Some(rt) = g.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error":"not loaded"}))).into_response();
    };
    let Some(gi) = rt.gene_names.iter().position(|x| x == &gene) else {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"unknown gene"}))).into_response();
    };
    let n = rt.obs_names.len().min(st.max_cells_json);
    let expr: Vec<f64> = (0..n).map(|i| rt.gene_mtx[[i, gi]]).collect();
    (StatusCode::OK, Json(serde_json::json!({"gene": gene, "expr": expr}))).into_response()
}

async fn post_load(
    State(st): State<AppState>,
    Json(body): Json<LoadBody>,
) -> impl IntoResponse {
    let path = PathBuf::from(body.run_toml.trim());
    let res = tokio::task::spawn_blocking(move || PerturbRuntime::from_run_toml(&path)).await;
    match res {
        Ok(Ok(rt)) => {
            st.rt.write().await.replace(Arc::new(rt));
            (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"ok": false, "error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"ok": false, "error": e.to_string()})),
        )
            .into_response(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=warn".into()),
        )
        .init();

    let cli = Cli::parse();
    let params = CellCommNetworkParams {
        beta_scale_factor: 1.0,
        min_expression: 1e-9,
        edge_threshold_abs: cli.threshold,
        include_self_loops: false,
        ignore_contact_distance: cli.no_contact_cutoff,
    };

    let rt_init: Option<Arc<PerturbRuntime>> = if let Some(p) = &cli.run_toml {
        tracing::info!(path = %p.display(), "loading run at startup…");
        Some(Arc::new(PerturbRuntime::from_run_toml(p)?))
    } else {
        None
    };

    let state = AppState {
        rt: Arc::new(RwLock::new(rt_init)),
        params,
        max_cells_json: cli.max_cells_json.max(1000),
        default_context_top: cli.top_k.max(1).min(256),
    };

    let app = Router::new()
        .route("/", get(get_index))
        .route("/api/meta", get(get_meta))
        .route("/api/cells", get(get_cells))
        .route("/api/cell/{i}/context", get(get_context))
        .route("/api/genes", get(get_genes))
        .route("/api/gene/{gene}/expr", get(get_gene_expr))
        .route("/api/load", post(post_load))
        .layer(CompressionLayer::new())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    tracing::info!("cell-comm-viewer → http://{addr}/");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
