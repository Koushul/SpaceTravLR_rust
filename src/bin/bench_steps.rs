//! Per-step microbenchmarks for SpaceTravLR (Rust side).
//!
//! Each invocation runs **one** named step on a synthetic dataset of fixed size
//! and prints a single JSON line to stdout describing the wall time and the
//! current process's peak resident set size. The companion Python script
//! (`benchmarks/rust_vs_python/bench_steps.py`) implements the same steps
//! against the reference Python `SpaceTravLR` package; the driver script
//! (`benchmarks/rust_vs_python/run_bench.py`) launches this binary and the
//! Python script as separate subprocesses, then aggregates the results.
//!
//! Steps are intentionally narrow and deterministic so Rust ↔ Python timings
//! compare like-for-like:
//!   - `received_ligands`  : Gaussian-weighted received-ligand aggregation.
//!   - `spatial_features`  : Per-cluster neighbor counts within a radius.
//!   - `xyc2spatial`       : Distance-map "spatial maps" tensor.
//!   - `group_lasso`       : Per-cluster grouped FISTA lasso fit.
//!   - `train_one_gene`    : Tiny end-to-end CNN+Lasso fit on a synthetic gene.
//!
//! Synthetic data is generated from `--seed` so Rust and Python see the same
//! geometry / cluster layout / signal (modulo deterministic RNG differences
//! across language stacks; absolute results are not compared, only timings).

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use ndarray::{Array1, Array2, Array4};
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Normal, Uniform};
use serde_json::json;
use std::time::Instant;

use spacetravlr::estimator::{create_spatial_features, xyc2spatial_fast};
use spacetravlr::lasso::{ClusteredGroupLasso, GroupLassoParams};
use spacetravlr::ligand::calculate_weighted_ligands;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Step {
    ReceivedLigands,
    SpatialFeatures,
    Xyc2spatial,
    GroupLasso,
    TrainOneGene,
}

impl Step {
    fn as_str(self) -> &'static str {
        match self {
            Step::ReceivedLigands => "received_ligands",
            Step::SpatialFeatures => "spatial_features",
            Step::Xyc2spatial => "xyc2spatial",
            Step::GroupLasso => "group_lasso",
            Step::TrainOneGene => "train_one_gene",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Per-step Rust microbenchmark for spacetravlr")]
struct Cli {
    /// Which step to benchmark.
    #[arg(long, value_enum)]
    step: Step,

    /// Number of synthetic cells.
    #[arg(long)]
    n_cells: usize,

    /// Random seed for synthetic data generation.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of ligands (received_ligands / train_one_gene).
    #[arg(long, default_value_t = 32)]
    n_ligands: usize,

    /// Number of synthetic gene/modulator features (group_lasso / train_one_gene).
    #[arg(long, default_value_t = 64)]
    n_features: usize,

    /// Number of clusters.
    #[arg(long, default_value_t = 8)]
    n_clusters: usize,

    /// Spatial extent (xy uniform in [-extent/2, extent/2]^2).
    #[arg(long, default_value_t = 5_000.0)]
    extent: f64,

    /// Gaussian radius for received ligands / spatial features.
    #[arg(long, default_value_t = 300.0)]
    radius: f64,

    /// CNN spatial map grid edge for xyc2spatial / train_one_gene.
    #[arg(long, default_value_t = 24)]
    spatial_dim: usize,

    /// CNN epochs for `train_one_gene`.
    #[arg(long, default_value_t = 4)]
    epochs: usize,

    /// FISTA max iterations for the Lasso step.
    #[arg(long, default_value_t = 200)]
    n_iter: usize,

    /// Number of repeats; the median wall time is reported.
    #[arg(long, default_value_t = 1)]
    repeats: usize,
}

fn make_xy(rng: &mut ChaCha8Rng, n: usize, extent: f64) -> Array2<f64> {
    let half = extent * 0.5;
    let dist = Uniform::new_inclusive(-half, half);
    let mut xy = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        xy[[i, 0]] = dist.sample(rng);
        xy[[i, 1]] = dist.sample(rng);
    }
    xy
}

fn make_clusters(rng: &mut ChaCha8Rng, n: usize, k: usize) -> Array1<usize> {
    let dist = Uniform::new(0usize, k.max(1));
    let mut c = Array1::<usize>::zeros(n);
    for i in 0..n {
        c[i] = dist.sample(rng);
    }
    c
}

fn make_dense(rng: &mut ChaCha8Rng, n: usize, p: usize) -> Array2<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut m = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let v: f64 = normal.sample(rng);
            m[[i, j]] = v.abs();
        }
    }
    m
}

fn peak_rss_mb() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest
                .split_whitespace()
                .next()?
                .parse()
                .ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

fn run_received_ligands(args: &Cli) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let xy = make_xy(&mut rng, args.n_cells, args.extent);
    let lig = make_dense(&mut rng, args.n_cells, args.n_ligands);
    let t0 = Instant::now();
    let _ = calculate_weighted_ligands(&xy, &lig, args.radius, 1.0);
    t0.elapsed().as_secs_f64()
}

fn run_spatial_features(args: &Cli) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let xy = make_xy(&mut rng, args.n_cells, args.extent);
    let clusters = make_clusters(&mut rng, args.n_cells, args.n_clusters);
    let t0 = Instant::now();
    let _ = create_spatial_features(&xy, &clusters, args.n_clusters, args.radius);
    t0.elapsed().as_secs_f64()
}

fn run_xyc2spatial(args: &Cli) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let xy = make_xy(&mut rng, args.n_cells, args.extent);
    let clusters = make_clusters(&mut rng, args.n_cells, args.n_clusters);
    let t0 = Instant::now();
    let _: Array4<f32> = xyc2spatial_fast(
        &xy,
        &clusters,
        args.n_clusters,
        args.spatial_dim,
        args.spatial_dim,
        false,
    );
    t0.elapsed().as_secs_f64()
}

fn run_group_lasso(args: &Cli) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let x = make_dense(&mut rng, args.n_cells, args.n_features);
    let normal = Normal::new(0.0, 0.5).unwrap();
    let mut beta = Array1::<f64>::zeros(args.n_features);
    for j in 0..args.n_features {
        beta[j] = if j < args.n_features / 4 {
            normal.sample(&mut rng)
        } else {
            0.0
        };
    }
    let mut y = Array2::<f64>::zeros((args.n_cells, 1));
    for i in 0..args.n_cells {
        let mut s = 0.0;
        for j in 0..args.n_features {
            s += x[[i, j]] * beta[j];
        }
        y[[i, 0]] = s + 0.01 * normal.sample(&mut rng);
    }
    let mut clusters = Array1::<i64>::zeros(args.n_cells);
    let cd = Uniform::new(0i64, args.n_clusters as i64);
    for i in 0..args.n_cells {
        clusters[i] = cd.sample(&mut rng);
    }
    let groups: Vec<i64> = (0..args.n_features as i64).collect();
    let params = GroupLassoParams {
        groups,
        group_reg: 1e-4,
        l1_reg: 1e-4,
        n_iter: args.n_iter,
        tol: 1e-4,
        gram_override: Some(true),
        ..Default::default()
    };
    let mut model = ClusteredGroupLasso::new(params);
    let t0 = Instant::now();
    let _ = model.fit(&x, &y, &clusters);
    t0.elapsed().as_secs_f64()
}

fn run_train_one_gene(args: &Cli) -> f64 {
    use spacetravlr::config::{CnnConfig, CnnOutputActivation};
    use spacetravlr::estimator::train_cluster_cnn_epochs;
    use spacetravlr::model::{CellularNicheNetwork, CellularNicheNetworkConfig};
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    use burn_autodiff::Autodiff;

    type B = Autodiff<NdArray<f32, i32>>;
    let device = Default::default();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    let xy = make_xy(&mut rng, args.n_cells, args.extent);
    let clusters_usize = make_clusters(&mut rng, args.n_cells, args.n_clusters);

    let received = calculate_weighted_ligands(&xy, &make_dense(&mut rng, args.n_cells, args.n_ligands), args.radius, 1.0);
    let modulators = args.n_features.min(32);
    let x_full = make_dense(&mut rng, args.n_cells, modulators);
    let _ = received;

    let mut betas_true = Array1::<f64>::zeros(modulators);
    let normal = Normal::new(0.0, 0.3).unwrap();
    for j in 0..modulators / 2 {
        betas_true[j] = normal.sample(&mut rng);
    }
    let mut y = Array1::<f64>::zeros(args.n_cells);
    for i in 0..args.n_cells {
        let mut s = 0.0;
        for j in 0..modulators {
            s += x_full[[i, j]] * betas_true[j];
        }
        y[i] = s + 0.05 * normal.sample(&mut rng);
    }

    let lasso_anchor = Array2::<f64>::from_shape_fn((modulators, 1), |(j, _)| betas_true[j] * 0.5);
    let intercept = 0.0_f32;

    let h = args.spatial_dim;
    let w = args.spatial_dim;
    let sm = xyc2spatial_fast(&xy, &clusters_usize, args.n_clusters, h, w, false);
    let _ = sm;

    let cluster_id = 0usize;
    let mut idx = Vec::new();
    for (i, &c) in clusters_usize.iter().enumerate() {
        if c == cluster_id {
            idx.push(i);
        }
    }
    if idx.is_empty() {
        return 0.0;
    }
    let n_c = idx.len();
    let sm_c = Array4::<f32>::from_elem((n_c, 1, h, w), 0.05f32);
    let mut x_c = Array2::<f64>::zeros((n_c, modulators));
    let mut y_c = Array1::<f64>::zeros(n_c);
    let mut sf_c = Array2::<f64>::zeros((n_c, args.n_clusters));
    for (k, &i) in idx.iter().enumerate() {
        for j in 0..modulators {
            x_c[[k, j]] = x_full[[i, j]];
        }
        y_c[k] = y[i];
        sf_c[[k, cluster_id]] = 1.0;
    }

    let anchors_vec: Vec<f32> = std::iter::once(intercept)
        .chain((0..modulators).map(|j| lasso_anchor[[j, 0]] as f32))
        .collect();
    let anchors_tensor = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(anchors_vec.clone(), [anchors_vec.len()]),
        &device,
    );
    let cfg = CellularNicheNetworkConfig {
        n_modulators: modulators,
        n_clusters: args.n_clusters,
        vision_in_channels: 1,
    };
    let model: CellularNicheNetwork<B> = cfg.init::<B>(&device, anchors_tensor, CnnOutputActivation::Sigmoid);

    let mut cnn = CnnConfig::default();
    cnn.lasso_pred_align_weight = 0.05;
    cnn.cnn_minibatch_size = 64;
    cnn.cnn_early_stop_patience = 0;

    let t0 = Instant::now();
    let _ = train_cluster_cnn_epochs::<B>(
        model,
        &device,
        &sm_c,
        &x_c,
        &sf_c,
        &y_c,
        n_c,
        cluster_id,
        None,
        0.0_f32,
        &cnn,
        4e-4,
        args.epochs,
        None,
        args.seed,
    );
    t0.elapsed().as_secs_f64()
}

fn run_step(args: &Cli) -> Vec<f64> {
    (0..args.repeats.max(1))
        .map(|_| match args.step {
            Step::ReceivedLigands => run_received_ligands(args),
            Step::SpatialFeatures => run_spatial_features(args),
            Step::Xyc2spatial => run_xyc2spatial(args),
            Step::GroupLasso => run_group_lasso(args),
            Step::TrainOneGene => run_train_one_gene(args),
        })
        .collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { xs[n / 2] } else { 0.5 * (xs[n / 2 - 1] + xs[n / 2]) }
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let times = run_step(&args);
    let wall = median(times.clone());
    let peak = peak_rss_mb();
    let out = json!({
        "impl": "rust",
        "step": args.step.as_str(),
        "n_cells": args.n_cells,
        "n_features": args.n_features,
        "n_ligands": args.n_ligands,
        "n_clusters": args.n_clusters,
        "spatial_dim": args.spatial_dim,
        "epochs": args.epochs,
        "n_iter": args.n_iter,
        "seed": args.seed,
        "repeats": args.repeats,
        "wall_s": wall,
        "wall_s_runs": times,
        "peak_rss_mb": peak,
    });
    println!("{}", serde_json::to_string(&out).context("serialize result")?);
    Ok(())
}
