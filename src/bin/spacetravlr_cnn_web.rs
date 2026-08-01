//! SpaceTravLR CNN Web: prepare Lasso+spatial packs natively, train CNN in browser WASM.
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::routing::{get, post};
use axum::serve;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn_autodiff::Autodiff;
use clap::Parser;
use serde::{Deserialize, Serialize};
use spacetravlr::config::{
    CnnTrainingMode, ModelExportConfig, SpaceshipConfig, expand_user_path,
    resolve_spaceship_config_toml_path,
};
use spacetravlr::spatial_estimator::SpatialCellularProgramsEstimator;
use spacetravlr_cnn_pack::cnn_gene_train_pack_from_npz;
use spacetravlr_cnn_wasm::{encode_pack, train_gene_pack};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

type TrainBackend = Autodiff<NdArray<f32, i32>>;

#[derive(Parser, Debug)]
#[command(name = "spacetravlr-cnn-web")]
struct Cli {
    #[arg(long, default_value = "0.0.0.0")]
    bind: String,
    #[arg(long, default_value_t = 8787)]
    port: u16,
    #[arg(long, default_value = "web/cnn_train/dist")]
    static_dir: PathBuf,
    #[arg(long)]
    h5ad: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = "/tmp/spacetravlr_cnn_web")]
    work_dir: PathBuf,
    #[arg(long, default_value = "AICDA,CD74")]
    default_genes: String,
    #[arg(long, default_value_t = 8)]
    spatial_dim: usize,
    #[arg(long, default_value_t = 64)]
    max_ligands: usize,
    #[arg(long, default_value_t = 8)]
    wasm_epochs: u32,
    #[arg(long, default_value_t = true)]
    allow_cors: bool,
}

struct AppState {
    h5ad: PathBuf,
    config_path: PathBuf,
    work_dir: PathBuf,
    default_genes: Vec<String>,
    spatial_dim: usize,
    max_ligands: usize,
    wasm_epochs: u32,
    job: Mutex<Option<JobStatus>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobStatus {
    phase: String,
    message: String,
    genes: Vec<String>,
    results: Vec<GeneSummary>,
    error: Option<String>,
    elapsed_ms: u64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeneSummary {
    gene: String,
    n_clusters: usize,
    wall_ms: u32,
    mean_final_mse: Option<f32>,
    clusters: Vec<ClusterSummary>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ClusterSummary {
    cluster_id: u32,
    n_cells: u32,
    mse_epochs: Vec<f32>,
    diverged: bool,
    wall_ms: u32,
}

#[derive(Deserialize)]
struct TrainRequest {
    genes: Option<String>,
    epochs: Option<u32>,
    /// When true, run CNN with native Burn NdArray (same pack). Default: prepare packs only
    /// and let the browser WASM module train.
    native_cnn: Option<bool>,
}

#[derive(Deserialize)]
struct PackQuery {
    gene: String,
}

#[derive(Serialize)]
struct InfoResponse {
    h5ad: String,
    default_genes: Vec<String>,
    spatial_dim: usize,
    max_ligands: usize,
    wasm_epochs: u32,
    backend: &'static str,
}

fn parse_genes(s: &str) -> Vec<String> {
    s.split(',')
        .map(|g| g.trim().to_string())
        .filter(|g| !g.is_empty())
        .collect()
}

fn resolve_static_dir(cli: &Path) -> anyhow::Result<PathBuf> {
    let candidates = [
        cli.to_path_buf(),
        PathBuf::from("web/cnn_train/dist"),
        PathBuf::from("web/cnn_train"),
    ];
    for c in candidates {
        if c.join("index.html").is_file() {
            return Ok(c);
        }
    }
    anyhow::bail!(
        "could not find index.html for --static-dir {:?} — run npm build in web/cnn_train",
        cli
    )
}

async fn api_info(State(st): State<Arc<AppState>>) -> Json<InfoResponse> {
    Json(InfoResponse {
        h5ad: st.h5ad.display().to_string(),
        default_genes: st.default_genes.clone(),
        spatial_dim: st.spatial_dim,
        max_ligands: st.max_ligands,
        wasm_epochs: st.wasm_epochs,
        backend: "Burn NdArray (prep) + WASM CNN",
    })
}

async fn api_status(State(st): State<Arc<AppState>>) -> Json<Option<JobStatus>> {
    Json(st.job.lock().unwrap_or_else(|e| e.into_inner()).clone())
}

fn set_job(st: &AppState, status: JobStatus) {
    *st.job.lock().unwrap_or_else(|e| e.into_inner()) = Some(status);
}

fn pack_npz_path(st: &AppState, gene: &str) -> PathBuf {
    st.work_dir.join("packs").join(format!("{gene}_cnn_train_data.npz"))
}

fn prepare_packs_for_genes(
    st: &AppState,
    genes: &[String],
    epochs: u32,
) -> anyhow::Result<Vec<(String, PathBuf)>> {
    let packs_dir = st.work_dir.join("packs");
    std::fs::create_dir_all(&packs_dir)?;

    // Stable dir per gene set for caching of the heavy Lasso dump
    let gene_key = genes.join("_");
    let run_dir = st.work_dir.join(format!("prep_{gene_key}_d{}", st.spatial_dim));
    std::fs::create_dir_all(&run_dir)?;

    let mut cfg = SpaceshipConfig::from_file(&st.config_path)
        .with_context(|| format!("load config {}", st.config_path.display()))?;
    cfg.data.adata_path = st.h5ad.display().to_string();
    cfg.spatial.spatial_dim = st.spatial_dim;
    cfg.grn.max_ligands = Some(st.max_ligands);
    cfg.training.mode = Some(CnnTrainingMode::Full);
    cfg.training.epochs = 0;
    cfg.training.genes = Some(genes.to_vec());
    cfg.execution.output_dir = run_dir.display().to_string();
    cfg.execution.n_parallel = 1;
    cfg.model_export = ModelExportConfig {
        save_cnn_weights: false,
        write_cnn_train_data_npz: true,
        compressed_npz: false,
        output_subdir: "CNN_weights".into(),
    };
    cfg.cnn.cnn_max_cells_per_epoch = Some(512);
    cfg.cnn.cnn_minibatch_size = 64;
    cfg.cnn.cnn_early_stop_patience = 0;

    let mut out = Vec::new();
    let mut missing = Vec::new();
    for g in genes {
        let cached = pack_npz_path(st, g);
        if cached.is_file() {
            out.push((g.clone(), cached));
        } else {
            missing.push(g.clone());
        }
    }
    if missing.is_empty() {
        return Ok(out);
    }

    set_job(
        st,
        JobStatus {
            phase: "prepare".into(),
            message: format!(
                "Lasso + spatial pack dump for {} (spatial_dim={})",
                missing.join(","),
                st.spatial_dim
            ),
            genes: genes.to_vec(),
            results: vec![],
            error: None,
            elapsed_ms: 0,
        },
    );

    let device = NdArrayDevice::Cpu;
    let network_dir = cfg
        .grn
        .network_data_dir
        .clone()
        .or_else(|| std::env::var("SPACETRAVLR_DATA_DIR").ok())
        .or_else(|| {
            let d = PathBuf::from("data");
            if d.join("human_network.parquet").is_file() {
                Some(d.display().to_string())
            } else {
                None
            }
        });

    SpatialCellularProgramsEstimator::<TrainBackend, anndata_hdf5::H5>::fit_all_genes(
        &st.h5ad.display().to_string(),
        None,
        cfg.spatial.radius,
        cfg.spatial.spatial_dim,
        cfg.spatial.contact_distance,
        cfg.grn.tf_ligand_cutoff,
        cfg.grn.max_ligands,
        cfg.grn.use_tf_modulators,
        cfg.grn.use_lr_modulators,
        cfg.grn.use_tfl_modulators,
        &cfg.data.layer,
        &cfg.data.cluster_annot,
        &cfg.cnn,
        0,
        cfg.training.learning_rate,
        cfg.training.score_threshold,
        cfg.lasso.l1_reg,
        cfg.lasso.group_reg,
        cfg.lasso.n_iter,
        cfg.lasso.tol,
        CnnTrainingMode::Full,
        Some(missing.clone()),
        None,
        1,
        &run_dir.display().to_string(),
        &cfg.model_export,
        None,
        network_dir.as_deref(),
        cfg.grn.tf_priors_feather.as_deref(),
        false,
        &cfg,
        Some(st.config_path.clone()),
        false,
        false,
        None,
        &device,
    )?;

    out.clear();
    for g in genes {
        let src = run_dir
            .join("CNN_weights")
            .join(format!("{g}_cnn_train_data.npz"));
        let dest = pack_npz_path(st, g);
        if src.is_file() {
            std::fs::copy(&src, &dest)
                .with_context(|| format!("copy {} -> {}", src.display(), dest.display()))?;
            let _ = epochs;
            out.push((g.clone(), dest));
        } else if dest.is_file() {
            out.push((g.clone(), dest));
        } else {
            eprintln!(
                "spacetravlr-cnn-web: no CNN train pack for {g} (Lasso-only / orphan / score gate)"
            );
        }
    }
    if out.is_empty() {
        anyhow::bail!(
            "no CNN train packs written under {}/CNN_weights for {:?}",
            run_dir.display(),
            genes
        );
    }
    Ok(out)
}

async fn api_prepare(
    State(st): State<Arc<AppState>>,
    Json(req): Json<TrainRequest>,
) -> Result<Json<JobStatus>, (StatusCode, String)> {
    let genes = req
        .genes
        .as_deref()
        .map(parse_genes)
        .filter(|g| !g.is_empty())
        .unwrap_or_else(|| st.default_genes.clone());
    let epochs = req.epochs.unwrap_or(st.wasm_epochs);
    let native = req.native_cnn.unwrap_or(false);
    let st2 = Arc::clone(&st);
    let t0 = Instant::now();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<JobStatus> {
        let packs = prepare_packs_for_genes(&st2, &genes, epochs)?;
        let mut summaries = Vec::new();
        if native {
            set_job(
                &st2,
                JobStatus {
                    phase: "native_cnn".into(),
                    message: "Training CNN natively (NdArray) on prepared packs".into(),
                    genes: genes.clone(),
                    results: vec![],
                    error: None,
                    elapsed_ms: t0.elapsed().as_millis() as u64,
                },
            );
            for (gene, npz) in &packs {
                let mut pack = cnn_gene_train_pack_from_npz(npz, gene, Some(epochs))?;
                pack.hyperparams.epochs = epochs;
                let trained = train_gene_pack(&pack);
                let mean_final = trained
                    .clusters
                    .iter()
                    .filter_map(|c| c.mse_epochs.last().copied())
                    .filter(|v| v.is_finite())
                    .collect::<Vec<_>>();
                let mean_final_mse = if mean_final.is_empty() {
                    None
                } else {
                    Some(mean_final.iter().sum::<f32>() / mean_final.len() as f32)
                };
                summaries.push(GeneSummary {
                    gene: gene.clone(),
                    n_clusters: trained.clusters.len(),
                    wall_ms: trained.wall_ms,
                    mean_final_mse,
                    clusters: trained
                        .clusters
                        .into_iter()
                        .map(|c| ClusterSummary {
                            cluster_id: c.cluster_id,
                            n_cells: c.n_cells,
                            mse_epochs: c.mse_epochs,
                            diverged: c.diverged,
                            wall_ms: c.wall_ms,
                        })
                        .collect(),
                });
            }
        } else {
            for (gene, npz) in &packs {
                let pack = cnn_gene_train_pack_from_npz(npz, gene, Some(epochs))?;
                summaries.push(GeneSummary {
                    gene: gene.clone(),
                    n_clusters: pack.clusters.len(),
                    wall_ms: 0,
                    mean_final_mse: None,
                    clusters: pack
                        .clusters
                        .iter()
                        .map(|c| ClusterSummary {
                            cluster_id: c.cluster_id,
                            n_cells: c.n_cells,
                            mse_epochs: vec![],
                            diverged: false,
                            wall_ms: 0,
                        })
                        .collect(),
                });
            }
        }
        Ok(JobStatus {
            phase: if native {
                "done_native".into()
            } else {
                "packs_ready".into()
            },
            message: if native {
                "Native CNN training finished".into()
            } else {
                "Packs ready — run WASM train in the browser".into()
            },
            genes,
            results: summaries,
            error: None,
            elapsed_ms: t0.elapsed().as_millis() as u64,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    match result {
        Ok(status) => {
            set_job(&st, status.clone());
            Ok(Json(status))
        }
        Err(e) => {
            let status = JobStatus {
                phase: "error".into(),
                message: "failed".into(),
                genes: vec![],
                results: vec![],
                error: Some(e.to_string()),
                elapsed_ms: t0.elapsed().as_millis() as u64,
            };
            set_job(&st, status.clone());
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn api_pack(
    State(st): State<Arc<AppState>>,
    Query(q): Query<PackQuery>,
) -> Result<(HeaderMap, Bytes), (StatusCode, String)> {
    let epochs = st.wasm_epochs;
    let gene = q.gene.trim().to_string();
    if gene.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "gene required".into()));
    }
    let st2 = Arc::clone(&st);
    let bytes = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<u8>> {
        let gene_name = gene.clone();
        let packs = prepare_packs_for_genes(&st2, &[gene_name.clone()], epochs)?;
        let (_, npz) = packs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no pack"))?;
        let mut pack = cnn_gene_train_pack_from_npz(&npz, &gene_name, Some(epochs))?;
        pack.hyperparams.epochs = epochs;
        encode_pack(&pack).map_err(|e| anyhow::anyhow!("{e}"))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename=\"{}.stcnnbin\"",
            q.gene.trim()
        ))
        .unwrap_or(HeaderValue::from_static(
            "attachment; filename=\"pack.stcnnbin\"",
        )),
    );
    Ok((headers, Bytes::from(bytes)))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    spacetravlr::ensure_process_env();
    unsafe {
        if std::env::var_os("SPACETRAVLR_FORCE_CPU").is_none() {
            std::env::set_var("SPACETRAVLR_FORCE_CPU", "1");
        }
        if std::env::var_os("SPACETRAVLR_QUIET_COMPUTE").is_none() {
            std::env::set_var("SPACETRAVLR_QUIET_COMPUTE", "1");
        }
    }

    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "spacetravlr_cnn_web=info,tower_http=info".into()),
        )
        .init();

    let h5ad = expand_user_path(&cli.h5ad.display().to_string());
    let h5ad = PathBuf::from(h5ad);
    anyhow::ensure!(h5ad.is_file(), "h5ad not found: {}", h5ad.display());

    let config_path = match cli.config {
        Some(p) => PathBuf::from(expand_user_path(&p.display().to_string())),
        None => resolve_spaceship_config_toml_path()
            .context("spaceship_config.toml not found; pass --config")?,
    };
    std::fs::create_dir_all(&cli.work_dir)?;

    let state = Arc::new(AppState {
        h5ad,
        config_path,
        work_dir: cli.work_dir,
        default_genes: parse_genes(&cli.default_genes),
        spatial_dim: cli.spatial_dim.max(8),
        max_ligands: cli.max_ligands,
        wasm_epochs: cli.wasm_epochs,
        job: Mutex::new(None),
    });

    let mut api = Router::new()
        .route("/api/info", get(api_info))
        .route("/api/status", get(api_status))
        .route("/api/prepare", post(api_prepare))
        .route("/api/pack", get(api_pack))
        .with_state(state);

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
    let static_files = ServeDir::new(&static_dir).fallback(ServeFile::new(index));
    let app = Router::new()
        .merge(api)
        .fallback_service(static_files)
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", cli.bind, cli.port).parse()?;
    eprintln!(
        "spacetravlr-cnn-web listening on http://{} (WASM CNN UI)",
        addr
    );
    let listener = tokio::net::TcpListener::bind(addr).await?;
    serve(listener, app).await?;
    // keep type happy if serve returns
    let _ = Duration::from_secs(0);
    Ok(())
}
