//! `spacetravlr-niche` — train a CNN over per-cell splash gene×gene images and
//! emit microniche labels.
//!
//! Two modes:
//!   * `from-run`: load a finished training run via `spacetravlr_run_repro.toml`,
//!     run splash, train, write `niche_labels.feather` + `.csv`.
//!   * `synthetic`: build an in-memory synthetic spatial run with known niches
//!     (no h5ad needed), train, write the labels alongside ground truth so you
//!     can inspect the result with any plotting tool.

use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use burn_autodiff::Autodiff;
use clap::{Parser, Subcommand};
use spacetravlr::niche::image::StandardizeMode;
use spacetravlr::niche::{
    NicheLabels, NicheRuntime, NicheRuntimeBuilder, NicheTrainConfig, adjusted_rand_index,
    make_synthetic_run, normalized_mutual_info, spatial_purity_knn, write_niche_labels_csv,
    write_niche_labels_feather,
};

#[derive(Parser, Debug)]
#[command(
    name = "spacetravlr-niche",
    about = "CNN microniche detection from per-cell splash gene×gene images.",
    long_about = "Loads (or simulates) a SpaceTravLR run, computes per-cell splash \
                  Jacobians, trains a CNN encoder with functional + spatial-coherence \
                  + reconstruction heads, then clusters the embedding with k-means and \
                  writes per-cell niche labels."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Load a finished training run via spacetravlr_run_repro.toml and detect niches.
    FromRun(FromRunArgs),
    /// Build a synthetic spatial run with known microniches and detect them.
    Synthetic(SyntheticArgs),
}

#[derive(Parser, Debug)]
struct CommonArgs {
    /// Output directory for niche_labels.{feather,csv}.
    #[arg(short = 'o', long)]
    out_dir: PathBuf,
    /// Number of microniches (k) to recover.
    #[arg(long, default_value_t = 5)]
    n_clusters: usize,
    #[arg(long, default_value_t = 60)]
    epochs: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 16)]
    n_programs: usize,
    #[arg(long, default_value_t = 8)]
    spatial_k: usize,
    #[arg(long, default_value_t = 1.0)]
    lambda_recon: f32,
    #[arg(long, default_value_t = 2.0)]
    lambda_func: f32,
    #[arg(long, default_value_t = 1.0)]
    lambda_spatial: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Print epoch-level training summaries.
    #[arg(long)]
    verbose: bool,
}

#[derive(Parser, Debug)]
struct FromRunArgs {
    #[arg(long)]
    run_toml: PathBuf,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Parser, Debug)]
struct SyntheticArgs {
    #[arg(long, default_value_t = 80)]
    cells_per_niche: usize,
    #[arg(long, default_value_t = 5)]
    n_niches: usize,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Clone, Debug)]
enum ComputeChoice {
    Wgpu(WgpuDevice),
    NdArray(NdArrayDevice),
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn select_compute_backend() -> ComputeChoice {
    if env_truthy("SPACETRAVLR_FORCE_CPU") || env_truthy("SPACETRAVLR_DISABLE_WGPU") {
        return ComputeChoice::NdArray(NdArrayDevice::Cpu);
    }
    let probe = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map(|a| a.get_info())
    });
    match probe {
        Some(_) => ComputeChoice::Wgpu(WgpuDevice::default()),
        None => ComputeChoice::NdArray(NdArrayDevice::Cpu),
    }
}

fn make_train_cfg(c: &CommonArgs) -> NicheTrainConfig {
    NicheTrainConfig {
        epochs: c.epochs,
        batch_size: c.batch_size,
        learning_rate: c.learning_rate,
        embedding_dim: c.embedding_dim,
        n_programs: c.n_programs,
        spatial_k: c.spatial_k,
        lambda_recon: c.lambda_recon,
        lambda_func: c.lambda_func,
        lambda_spatial: c.lambda_spatial,
        recon_down: 4,
        conv_channels: (32, 64, 64),
        mlp_hidden: 128,
        projection_dim: 16,
        seed: c.seed,
        verbose: c.verbose,
    }
}

fn run_from_run(args: FromRunArgs) -> Result<()> {
    let common = args.common;
    let run_toml = args.run_toml.clone();
    println!("Loading run TOML: {}", run_toml.display());
    let builder = NicheRuntimeBuilder::from_run_toml(&run_toml, StandardizeMode::PerEntry)
        .with_context(|| format!("loading run from {}", run_toml.display()))?;
    println!(
        "Loaded splash image stack: n_cells={}, n_targets={}, n_modulators={}",
        builder.stack.n_cells, builder.stack.n_targets, builder.stack.n_modulators
    );
    fit_and_write(builder, common, /*synth_gt=*/ None)
}

fn run_synthetic(args: SyntheticArgs) -> Result<()> {
    let common = args.common;
    println!(
        "Generating synthetic run: cells_per_niche={}, n_niches={}",
        args.cells_per_niche, args.n_niches
    );
    let synth = make_synthetic_run(args.cells_per_niche, args.n_niches, common.seed);
    let gt = synth.niche_gt.clone();
    let coords: Vec<[f64; 2]> = (0..synth.n_cells)
        .map(|i| [synth.xy[[i, 0]], synth.xy[[i, 1]]])
        .collect();
    let builder = NicheRuntimeBuilder::from_synthetic(synth, StandardizeMode::PerEntry);
    fit_and_write(builder, common, Some((gt, coords)))
}

fn fit_and_write(
    builder: NicheRuntimeBuilder,
    common: CommonArgs,
    synth_gt: Option<(Vec<usize>, Vec<[f64; 2]>)>,
) -> Result<()> {
    std::fs::create_dir_all(&common.out_dir)
        .with_context(|| format!("create_dir_all {}", common.out_dir.display()))?;
    let train_cfg = make_train_cfg(&common);

    let backend = select_compute_backend();
    println!("Using backend: {}", match &backend {
        ComputeChoice::Wgpu(_) => "WebGPU",
        ComputeChoice::NdArray(_) => "CPU (NdArray)",
    });

    match backend {
        ComputeChoice::Wgpu(dev) => {
            type B = Autodiff<Wgpu>;
            let outputs = NicheRuntime::fit::<B>(&dev, builder, &train_cfg, common.n_clusters);
            write_outputs(&common.out_dir, &outputs, synth_gt)?;
        }
        ComputeChoice::NdArray(dev) => {
            type B = Autodiff<NdArray<f32, i32>>;
            let outputs = NicheRuntime::fit::<B>(&dev, builder, &train_cfg, common.n_clusters);
            write_outputs(&common.out_dir, &outputs, synth_gt)?;
        }
    };

    Ok(())
}

fn write_outputs<B: burn::tensor::backend::AutodiffBackend>(
    out_dir: &std::path::Path,
    outputs: &spacetravlr::niche::NicheRuntimeOutputs<B>,
    synth_gt: Option<(Vec<usize>, Vec<[f64; 2]>)>,
) -> Result<()> {
    let feather = out_dir.join("niche_labels.feather");
    write_niche_labels_feather(
        &feather,
        NicheLabels {
            obs_names: &outputs.obs_names,
            labels: &outputs.labels,
            embeddings: &outputs.embeddings,
        },
    )?;
    println!("wrote {}", feather.display());
    let csv = out_dir.join("niche_labels.csv");
    write_niche_labels_csv(
        &csv,
        NicheLabels {
            obs_names: &outputs.obs_names,
            labels: &outputs.labels,
            embeddings: &outputs.embeddings,
        },
    )?;
    println!("wrote {}", csv.display());

    if let Some((gt, coords)) = synth_gt {
        let ari = adjusted_rand_index(&gt, &outputs.labels);
        let nmi = normalized_mutual_info(&gt, &outputs.labels);
        let purity = spatial_purity_knn(&coords, &outputs.labels, 10);
        let purity_gt = spatial_purity_knn(&coords, &gt, 10);
        println!(
            "metrics vs ground-truth niches: ARI={:.3} NMI={:.3} spatial_purity_k10={:.3} (GT purity = {:.3})",
            ari, nmi, purity, purity_gt
        );
        let metrics = out_dir.join("niche_metrics.json");
        std::fs::write(
            &metrics,
            serde_json::to_string_pretty(&serde_json::json!({
                "ari": ari,
                "nmi": nmi,
                "spatial_purity_k10": purity,
                "spatial_purity_k10_ground_truth": purity_gt,
                "n_clusters_pred": outputs.n_clusters,
                "n_clusters_gt": gt.iter().copied().collect::<std::collections::HashSet<_>>().len(),
                "epoch_losses": outputs.train.epoch_losses.iter().map(|s| {
                    serde_json::json!({
                        "epoch": s.epoch,
                        "total": s.total,
                        "recon": s.recon,
                        "functional": s.functional,
                        "spatial": s.spatial,
                    })
                }).collect::<Vec<_>>(),
            }))?,
        )?;
        println!("wrote {}", metrics.display());
    }
    Ok(())
}

fn main() -> Result<()> {
    spacetravlr::ensure_hdf5_no_file_locking();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::FromRun(args) => run_from_run(args),
        Cmd::Synthetic(args) => run_synthetic(args),
    }
}
