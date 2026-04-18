//! Training loop for [`NicheEncoder`] with a composite loss:
//!
//! ```text
//!   L_total = lambda_recon   * L_recon
//!           + lambda_func    * L_func
//!           + lambda_spatial * L_spatial
//! ```
//!
//! * `L_recon`   = MSE between a low-rank avg-pooled version of the input
//!   image and the recon head's output. Forces the embedding to retain
//!   information about the splash Jacobian.
//! * `L_func`    = MSE between the functional head and a per-cell **program
//!   activity vector** computed as `softmax_norm(P^T |J_cell|)` where
//!   `P ∈ R^{n_modulators × n_programs}` is a learned-once "program
//!   membership" matrix obtained by k-means on modulator co-activity. This
//!   forces the embedding to encode *which signalling programs* are active
//!   in each cell, which is exactly the property a *functional* niche should
//!   have.
//! * `L_spatial` = `1 - cos(z_i, mean(z_{N(i)}))` averaged over cells, where
//!   `N(i)` are spatial KNN. Forces neighbouring cells to have similar
//!   embeddings, making niches spatially contiguous without supervision.

use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::image::NicheImageStack;
use super::kmeans::kmeans_lloyd;
use super::model::{NicheEncoder, NicheEncoderConfig};

/// Hyper-parameters for [`train_niche_encoder`].
#[derive(Clone, Debug)]
pub struct NicheTrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub embedding_dim: usize,
    pub n_programs: usize,
    pub spatial_k: usize,
    pub lambda_recon: f32,
    pub lambda_func: f32,
    pub lambda_spatial: f32,
    pub recon_down: usize,
    pub conv_channels: (usize, usize, usize),
    pub mlp_hidden: usize,
    pub projection_dim: usize,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for NicheTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 60,
            batch_size: 64,
            learning_rate: 1e-3,
            embedding_dim: 32,
            n_programs: 16,
            spatial_k: 8,
            lambda_recon: 1.0,
            lambda_func: 1.0,
            lambda_spatial: 0.5,
            recon_down: 4,
            conv_channels: (32, 64, 64),
            mlp_hidden: 128,
            projection_dim: 16,
            seed: 0,
            verbose: false,
        }
    }
}

/// Result of training.
pub struct NicheTrainOutputs<B: AutodiffBackend> {
    pub embeddings: Array2<f32>,
    pub program_assignments: Vec<usize>,
    pub program_membership: Array2<f32>,
    pub functional_targets: Array2<f32>,
    pub epoch_losses: Vec<EpochLossSummary>,
    pub model: NicheEncoder<B>,
}

#[derive(Clone, Debug)]
pub struct EpochLossSummary {
    pub epoch: usize,
    pub total: f32,
    pub recon: f32,
    pub functional: f32,
    pub spatial: f32,
}

/// Train a [`NicheEncoder`] on `images` and return per-cell embeddings + the
/// trained model.
pub fn train_niche_encoder<B: AutodiffBackend>(
    device: &B::Device,
    stack: &NicheImageStack,
    coords: &[[f64; 2]],
    cfg: &NicheTrainConfig,
) -> NicheTrainOutputs<B> {
    assert_eq!(coords.len(), stack.n_cells);
    let n_cells = stack.n_cells;
    let n_targets = stack.n_targets;
    let n_modulators = stack.n_modulators;

    let (program_assignments, program_membership) =
        compute_program_membership(stack, cfg.n_programs, cfg.seed);
    let functional_targets =
        compute_functional_targets(stack, &program_membership, cfg.n_programs);
    let spatial_neighbors = build_spatial_knn(coords, cfg.spatial_k);

    let model_cfg = NicheEncoderConfig::new(n_targets, n_modulators)
        .with_embedding_dim(cfg.embedding_dim)
        .with_n_programs(cfg.n_programs)
        .with_recon_down(cfg.recon_down)
        .with_conv1_channels(cfg.conv_channels.0)
        .with_conv2_channels(cfg.conv_channels.1)
        .with_conv3_channels(cfg.conv_channels.2)
        .with_mlp_hidden(cfg.mlp_hidden)
        .with_projection_dim(cfg.projection_dim);
    let mut model: NicheEncoder<B> = model_cfg.init::<B>(device);
    let mut optim = AdamConfig::new().init::<B, NicheEncoder<B>>();

    let recon_target_full =
        precompute_recon_target(stack, model.heads_recon_h(), model.heads_recon_w());

    let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(1));
    let mut epoch_losses = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        let mut indices: Vec<usize> = (0..n_cells).collect();
        indices.shuffle(&mut rng);

        let mut sum_total = 0.0f32;
        let mut sum_recon = 0.0f32;
        let mut sum_func = 0.0f32;
        let mut sum_spatial = 0.0f32;
        let mut n_batches = 0usize;

        for chunk in indices.chunks(cfg.batch_size) {
            let bs = chunk.len();
            let mut batch_imgs = Vec::with_capacity(bs * n_targets * n_modulators);
            let mut batch_recon = Vec::with_capacity(bs * model.heads_recon_h() * model.heads_recon_w());
            let mut batch_func = Vec::with_capacity(bs * cfg.n_programs);
            let mut batch_nbr_idx = Vec::with_capacity(bs);
            for &i in chunk {
                batch_imgs.extend_from_slice(stack.cell(i));
                let r = recon_row(&recon_target_full, i, model.heads_recon_h(), model.heads_recon_w());
                batch_recon.extend_from_slice(r);
                batch_func.extend_from_slice(&functional_targets.row(i).to_vec());
                let mut local: Vec<i64> = Vec::with_capacity(cfg.spatial_k);
                let nbrs = &spatial_neighbors[i];
                for &j in nbrs {
                    if let Some(local_pos) = chunk.iter().position(|&x| x == j) {
                        local.push(local_pos as i64);
                    }
                }
                while local.len() < cfg.spatial_k {
                    local.push(-1);
                }
                local.truncate(cfg.spatial_k);
                batch_nbr_idx.push(local);
            }

            let imgs: Tensor<B, 4> = Tensor::from_data(
                burn::tensor::TensorData::new(batch_imgs, [bs, 1, n_targets, n_modulators]),
                device,
            );
            let recon_t: Tensor<B, 4> = Tensor::from_data(
                burn::tensor::TensorData::new(
                    batch_recon,
                    [bs, 1, model.heads_recon_h(), model.heads_recon_w()],
                ),
                device,
            );
            let func_t: Tensor<B, 2> = Tensor::from_data(
                burn::tensor::TensorData::new(batch_func, [bs, cfg.n_programs]),
                device,
            );

            let out = model.forward(imgs);

            let l_recon = burn::nn::loss::MseLoss::new().forward(
                out.recon.clone(),
                recon_t,
                burn::nn::loss::Reduction::Mean,
            );
            let l_func = burn::nn::loss::MseLoss::new().forward(
                out.functional.clone(),
                func_t,
                burn::nn::loss::Reduction::Mean,
            );
            let l_spatial = spatial_coherence_loss::<B>(&out.projection, &batch_nbr_idx, device);

            let loss = l_recon.clone() * cfg.lambda_recon
                + l_func.clone() * cfg.lambda_func
                + l_spatial.clone() * cfg.lambda_spatial;

            sum_recon += l_recon.into_scalar().to_f32();
            sum_func += l_func.into_scalar().to_f32();
            sum_spatial += l_spatial.into_scalar().to_f32();
            sum_total += loss.clone().into_scalar().to_f32();
            n_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.learning_rate, model, grads);
        }

        let summary = EpochLossSummary {
            epoch,
            total: sum_total / n_batches.max(1) as f32,
            recon: sum_recon / n_batches.max(1) as f32,
            functional: sum_func / n_batches.max(1) as f32,
            spatial: sum_spatial / n_batches.max(1) as f32,
        };
        if cfg.verbose {
            println!(
                "epoch {:>3}/{}: total={:.4} recon={:.4} func={:.4} spatial={:.4}",
                summary.epoch + 1,
                cfg.epochs,
                summary.total,
                summary.recon,
                summary.functional,
                summary.spatial
            );
        }
        epoch_losses.push(summary);
    }

    let embeddings = embed_all(&model, stack, device, cfg.batch_size);

    NicheTrainOutputs {
        embeddings,
        program_assignments,
        program_membership,
        functional_targets,
        epoch_losses,
        model,
    }
}

impl<B: burn::tensor::backend::Backend> NicheEncoder<B> {
    pub(crate) fn heads_recon_h(&self) -> usize {
        self.heads.recon_h
    }
    pub(crate) fn heads_recon_w(&self) -> usize {
        self.heads.recon_w
    }
}

fn embed_all<B: AutodiffBackend>(
    model: &NicheEncoder<B>,
    stack: &NicheImageStack,
    device: &B::Device,
    batch_size: usize,
) -> Array2<f32> {
    let inner = model.valid();
    let mut out = Array2::<f32>::zeros((stack.n_cells, model.embedding_dim()));
    let n = stack.n_cells;
    let mut start = 0;
    while start < n {
        let end = (start + batch_size).min(n);
        let bs = end - start;
        let mut buf = Vec::with_capacity(bs * stack.n_targets * stack.n_modulators);
        for i in start..end {
            buf.extend_from_slice(stack.cell(i));
        }
        let imgs: Tensor<B::InnerBackend, 4> = Tensor::from_data(
            burn::tensor::TensorData::new(buf, [bs, 1, stack.n_targets, stack.n_modulators]),
            device,
        );
        let z = inner.embed(imgs);
        let data = z.into_data();
        let v = data.as_slice::<f32>().unwrap().to_vec();
        for (k, val) in v.into_iter().enumerate() {
            let row = start + k / model.embedding_dim();
            let col = k % model.embedding_dim();
            out[[row, col]] = val;
        }
        start = end;
    }
    out
}

/// Cluster the modulator axis by co-activity across cells × targets to derive
/// "programs" — groups of modulators that tend to fire together. Each
/// modulator gets a hard program assignment, then a soft membership matrix
/// is the one-hot encoding (so the functional target is just the L1 sum of
/// |J| inside each program).
pub(crate) fn compute_program_membership(
    stack: &NicheImageStack,
    n_programs: usize,
    seed: u64,
) -> (Vec<usize>, Array2<f32>) {
    let n_modulators = stack.n_modulators;
    let n_targets = stack.n_targets;
    let n_cells = stack.n_cells;

    let mut profiles = vec![0.0f32; n_modulators * (n_targets + 1)];
    for c in 0..n_cells {
        let cell = stack.cell(c);
        for t in 0..n_targets {
            for m in 0..n_modulators {
                let v = cell[t * n_modulators + m].abs();
                profiles[m * (n_targets + 1) + t] += v;
            }
        }
    }
    for m in 0..n_modulators {
        let mut s = 0.0f32;
        for t in 0..n_targets {
            s += profiles[m * (n_targets + 1) + t];
        }
        profiles[m * (n_targets + 1) + n_targets] = s;
    }

    let n = n_modulators;
    let dim = n_targets + 1;
    let k = n_programs.min(n).max(1);
    let res = kmeans_lloyd(&profiles, n, dim, k, 64, seed);
    let mut membership = Array2::<f32>::zeros((n_modulators, k));
    for (m, &p) in res.labels.iter().enumerate() {
        membership[[m, p]] = 1.0;
    }
    (res.labels, membership)
}

/// `targets[c, p] = sum_{t,m: P[m]=p} |J[c, t, m]|`, normalized so that each
/// cell's program activity sums to 1.
fn compute_functional_targets(
    stack: &NicheImageStack,
    membership: &Array2<f32>,
    n_programs: usize,
) -> Array2<f32> {
    let n_cells = stack.n_cells;
    let n_modulators = stack.n_modulators;
    let n_targets = stack.n_targets;
    let mut out = Array2::<f32>::zeros((n_cells, n_programs));
    for c in 0..n_cells {
        let cell = stack.cell(c);
        for t in 0..n_targets {
            for m in 0..n_modulators {
                let v = cell[t * n_modulators + m].abs();
                if v == 0.0 {
                    continue;
                }
                for p in 0..n_programs {
                    if membership[[m, p]] != 0.0 {
                        out[[c, p]] += v;
                    }
                }
            }
        }
        let s: f32 = out.row(c).sum();
        if s > 0.0 {
            for p in 0..n_programs {
                out[[c, p]] /= s;
            }
        }
    }
    out
}

fn build_spatial_knn(coords: &[[f64; 2]], k: usize) -> Vec<Vec<usize>> {
    let n = coords.len();
    let pts: Vec<[f64; 2]> = coords.to_vec();
    let tree = kiddo::ImmutableKdTree::<f64, 2>::new_from_slice(&pts);
    let k_eff = k.min(n.saturating_sub(1).max(1));
    let k_query = std::num::NonZero::new(k_eff + 1).unwrap();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let nbrs = tree.nearest_n::<kiddo::SquaredEuclidean>(&coords[i], k_query);
        let mut idxs = Vec::with_capacity(k_eff);
        for nb in &nbrs {
            let j = nb.item as usize;
            if j == i {
                continue;
            }
            idxs.push(j);
            if idxs.len() == k_eff {
                break;
            }
        }
        out.push(idxs);
    }
    out
}

fn precompute_recon_target(stack: &NicheImageStack, h: usize, w: usize) -> Vec<f32> {
    let n_cells = stack.n_cells;
    let mut out = vec![0.0f32; n_cells * h * w];
    let stride_t = stack.n_modulators;
    let stride_c = stack.n_targets * stride_t;
    let bin_h = stack.n_targets.div_ceil(h);
    let bin_w = stack.n_modulators.div_ceil(w);
    for c in 0..n_cells {
        let cell = &stack.images[c * stride_c..(c + 1) * stride_c];
        for hi in 0..h {
            for wi in 0..w {
                let mut sum = 0.0f32;
                let mut cnt = 0usize;
                for t in (hi * bin_h)..((hi + 1) * bin_h).min(stack.n_targets) {
                    for m in (wi * bin_w)..((wi + 1) * bin_w).min(stack.n_modulators) {
                        sum += cell[t * stride_t + m];
                        cnt += 1;
                    }
                }
                let val = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
                out[c * h * w + hi * w + wi] = val;
            }
        }
    }
    out
}

fn recon_row<'a>(buf: &'a [f32], i: usize, h: usize, w: usize) -> &'a [f32] {
    let s = i * h * w;
    &buf[s..s + h * w]
}

/// Mean of `1 - cosine_similarity(z_i, mean_j∈N(i) z_j)` over the batch.
/// Indices in `nbr_idx` < 0 are masked.
fn spatial_coherence_loss<B: AutodiffBackend>(
    z: &Tensor<B, 2>,
    nbr_idx: &[Vec<i64>],
    device: &B::Device,
) -> Tensor<B, 1> {
    let dims = z.dims();
    let bs = dims[0];
    let d = dims[1];
    let k = nbr_idx[0].len();
    let mut idx_flat = Vec::with_capacity(bs * k);
    let mut mask = Vec::with_capacity(bs * k);
    for nbrs in nbr_idx {
        for &j in nbrs {
            if j < 0 {
                idx_flat.push(0i64);
                mask.push(0.0f32);
            } else {
                idx_flat.push(j);
                mask.push(1.0f32);
            }
        }
    }
    let idx_t: Tensor<B, 2, burn::tensor::Int> = Tensor::from_data(
        burn::tensor::TensorData::new(idx_flat, [bs, k]),
        device,
    );
    let mask_t: Tensor<B, 2> =
        Tensor::from_data(burn::tensor::TensorData::new(mask, [bs, k]), device);

    let z_norm = normalize_rows(z.clone());
    let z_idx = idx_t.clone().reshape([bs * k]);
    let gathered = z_norm.clone().select(0, z_idx);
    let gathered = gathered.reshape([bs, k, d]);
    let mask_b = mask_t.clone().reshape([bs, k, 1]);
    let masked = gathered * mask_b.clone();
    let denom = mask_t.clone().sum_dim(1).add_scalar(1e-6f32);
    let mean_nbr = masked.sum_dim(1).reshape([bs, d]) / denom.reshape([bs, 1]);
    let mean_nbr_norm = normalize_rows(mean_nbr);
    let cos = (z_norm * mean_nbr_norm).sum_dim(1).reshape([bs]);
    let one = Tensor::<B, 1>::ones([bs], device);
    (one - cos).mean()
}

fn normalize_rows<B: AutodiffBackend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let sq = x.clone().powf_scalar(2.0).sum_dim(1).add_scalar(1e-12f32);
    let norm = sq.sqrt();
    x / norm.reshape([0, 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::niche::image::{StandardizeMode, build_niche_image_stack};
    use crate::niche::synth::make_synthetic_run;
    use burn::backend::NdArray;
    use burn_autodiff::Autodiff;

    type B = Autodiff<NdArray<f32, i32>>;

    #[test]
    fn train_runs_on_synthetic() {
        let device = Default::default();
        let run = make_synthetic_run(96, 4, 0);
        let stack = build_niche_image_stack(&run.splash, run.n_cells, StandardizeMode::PerEntry);
        let coords: Vec<[f64; 2]> = (0..run.n_cells).map(|i| [run.xy[[i, 0]], run.xy[[i, 1]]]).collect();
        let cfg = NicheTrainConfig {
            epochs: 4,
            batch_size: 32,
            n_programs: 4,
            embedding_dim: 8,
            mlp_hidden: 32,
            projection_dim: 4,
            spatial_k: 4,
            verbose: false,
            ..Default::default()
        };
        let out = train_niche_encoder::<B>(&device, &stack, &coords, &cfg);
        assert_eq!(out.embeddings.shape(), &[run.n_cells, 8]);
        assert!(out.epoch_losses.last().unwrap().total.is_finite());
    }
}
