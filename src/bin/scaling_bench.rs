//! CLI bench for SpaceTravLR's CNN training that scales with `--n` cells.
//!
//! Builds synthetic 32x32 spatial maps, modulator features `X`, cluster one-hots `SF`,
//! and a target `Y` linear in `X` plus a small spatial signal, then runs
//! `train_cluster_cnn_epochs` for the requested number of epochs.
//!
//! Optional `--dropout p` zeros each cell of the spatial maps with probability `p` to
//! simulate sparsity / dropout in the input.
//!
//! Output: a single JSON line on stdout with timing and final MSE.
//!
//! Designed to be paired with `benchmarks/bench_cnn_python.py` for a head-to-head
//! Rust-vs-Python scaling plot.

use std::time::Instant;

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::prelude::*;
use burn_autodiff::Autodiff;
use ndarray::{Array1, Array2, Array4};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use spacetravlr::config::{CnnConfig, CnnOutputActivation};
use spacetravlr::estimator::{TrainClusterCnnEpochsInput, train_cluster_cnn_epochs};
use spacetravlr::model::CellularNicheNetworkConfig;

#[derive(Serialize)]
struct BenchResult {
    impl_: &'static str,
    backend: &'static str,
    n_cells: usize,
    spatial_dim: usize,
    n_modulators: usize,
    n_clusters: usize,
    epochs: usize,
    minibatch_size: usize,
    dropout: f32,
    learning_rate: f64,
    total_seconds: f64,
    epoch_seconds: Vec<f64>,
    final_mse: f32,
    mse_history: Vec<f32>,
    diverged: bool,
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    if let Some(pos) = args.iter().position(|a| a == flag) {
        if pos + 1 < args.len() {
            return args[pos + 1].parse().expect("invalid flag value");
        }
    }
    default
}

fn build_inputs(
    n: usize,
    spatial_dim: usize,
    n_modulators: usize,
    n_clusters: usize,
    dropout: f32,
    seed: u64,
) -> (Array4<f32>, Array2<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let h = spatial_dim;
    let w = spatial_dim;
    let mut sm = Array4::<f32>::zeros((n, 1, h, w));
    for i in 0..n {
        for r in 0..h {
            for c in 0..w {
                let v: f32 = rng.gen_range(0.0..1.0);
                let keep: f32 = rng.gen_range(0.0..1.0);
                sm[[i, 0, r, c]] = if keep < dropout { 0.0 } else { v };
            }
        }
    }
    let mut x = Array2::<f64>::zeros((n, n_modulators));
    for i in 0..n {
        for j in 0..n_modulators {
            x[[i, j]] = rng.gen_range(0.0..1.0);
        }
    }
    let mut sf = Array2::<f64>::zeros((n, n_clusters));
    for i in 0..n {
        let c = (i % n_clusters) as usize;
        sf[[i, c]] = 1.0;
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.1;
        for j in 0..n_modulators.min(6) {
            acc += x[[i, j]] * (0.05 + 0.02 * (j as f64));
        }
        let mut sp = 0.0f64;
        for r in 0..h {
            for c in 0..w {
                sp += sm[[i, 0, r, c]] as f64;
            }
        }
        acc += 0.02 * (sp / (h * w) as f64);
        y[i] = acc + rng.gen_range(0.0..0.05);
    }
    (sm, x, sf, y)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = parse_arg(&args, "--n", 700usize);
    let spatial_dim: usize = parse_arg(&args, "--spatial-dim", 32usize);
    let n_mod: usize = parse_arg(&args, "--n-modulators", 16usize);
    let n_clusters: usize = parse_arg(&args, "--n-clusters", 12usize);
    let epochs: usize = parse_arg(&args, "--epochs", 4usize);
    let mbs: usize = parse_arg(&args, "--minibatch", 128usize);
    let lr: f64 = parse_arg(&args, "--lr", 1e-3f64);
    let dropout: f32 = parse_arg(&args, "--dropout", 0.0f32);
    let seed: u64 = parse_arg(&args, "--seed", 42u64);

    eprintln!(
        "[rust-bench] n={n} spatial_dim={spatial_dim} n_mod={n_mod} n_clusters={n_clusters} \
         epochs={epochs} minibatch={mbs} lr={lr} dropout={dropout}"
    );

    let (sm, x, sf, y) = build_inputs(n, spatial_dim, n_mod, n_clusters, dropout, seed);

    type B = Autodiff<NdArray<f32, i32>>;
    let device = Default::default();
    let backend = "NdArray-CPU";

    let mut anchors_vec: Vec<f32> = Vec::with_capacity(n_mod + 1);
    anchors_vec.push(0.5);
    for _ in 0..n_mod {
        anchors_vec.push(0.1);
    }
    let anchors_tensor = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(anchors_vec.clone(), [anchors_vec.len()]),
        &device,
    );
    let cfg = CellularNicheNetworkConfig {
        n_modulators: n_mod,
        n_clusters,
        vision_in_channels: 1,
    };
    let model = cfg.init::<B>(&device, anchors_tensor, CnnOutputActivation::Sigmoid);

    let cnn = CnnConfig {
        lasso_pred_align_weight: 0.0,
        cnn_minibatch_size: mbs,
        cnn_early_stop_patience: 0,
        cnn_max_batches_per_epoch: None,
        cnn_max_cells_per_epoch: None,
        ..Default::default()
    };

    let mut epoch_seconds = Vec::with_capacity(epochs);
    let mut mse_history: Vec<f32> = Vec::with_capacity(epochs);
    let t0 = Instant::now();
    let (_m, mse_ep, diverged) = train_cluster_cnn_epochs(TrainClusterCnnEpochsInput {
        model,
        device: &device,
        sm_c: &sm,
        x_c: &x,
        sf_c: &sf,
        y_c: &y,
        cluster_n: n,
        cluster_id: 0usize,
        y_lasso_cpu: None,
        beta_prior_w: 0.01f32,
        cnn: &cnn,
        learning_rate: lr,
        epochs,
        cnn_epoch_slot: None,
        shuffle_seed: seed,
    });
    let total_seconds = t0.elapsed().as_secs_f64();

    let per = total_seconds / (epochs.max(1) as f64);
    for _ in 0..epochs {
        epoch_seconds.push(per);
    }
    mse_history.extend(mse_ep.iter().copied());
    let final_mse = mse_history.last().copied().unwrap_or(f32::NAN);

    let out = BenchResult {
        impl_: "rust-burn",
        backend,
        n_cells: n,
        spatial_dim,
        n_modulators: n_mod,
        n_clusters,
        epochs,
        minibatch_size: mbs,
        dropout,
        learning_rate: lr,
        total_seconds,
        epoch_seconds,
        final_mse,
        mse_history,
        diverged,
    };
    let s = serde_json::to_string(&out).context("serialize")?;
    println!("{s}");
    Ok(())
}
