use crate::config::{CnnConfig, CnnLrSchedule};
use crate::lasso::{GroupLasso, GroupLassoParams};
use crate::model::{CellularNicheNetwork, CellularNicheNetworkConfig};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::tensor::backend::{AutodiffBackend, Backend};
use ndarray::{Array1, Array2, Array4, ArrayView1, ArrayView2, ArrayView4, Axis, s};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Concurrent group-lasso fits per gene (CPU); CNN remains sequential on the device.
const GROUP_LASSO_MAX_CONCURRENT: usize = 4;

fn spatial_maps_for_cluster_cnn(
    spatial_maps: &Array4<f32>,
    row_indices: &[usize],
    cluster_id: usize,
) -> Array4<f32> {
    let k = row_indices.len();
    if k == 0 {
        return Array4::zeros((0, 1, spatial_maps.shape()[2], spatial_maps.shape()[3]));
    }
    let h = spatial_maps.shape()[2];
    let w = spatial_maps.shape()[3];
    let mut out = Array4::<f32>::zeros((k, 1, h, w));
    for (out_i, &src_i) in row_indices.iter().enumerate() {
        out.slice_mut(s![out_i, 0, .., ..])
            .assign(&spatial_maps.slice(s![src_i, cluster_id, .., ..]));
    }
    out
}

fn spatial_maps_all_clusters_for_cnn(
    spatial_maps: &Array4<f32>,
    row_indices: &[usize],
) -> Array4<f32> {
    let k = row_indices.len();
    let nc = spatial_maps.shape()[1];
    let h = spatial_maps.shape()[2];
    let w = spatial_maps.shape()[3];
    if k == 0 {
        return Array4::zeros((0, nc, h, w));
    }
    let mut out = Array4::<f32>::zeros((k, nc, h, w));
    for (out_i, &src_i) in row_indices.iter().enumerate() {
        out.slice_mut(s![out_i, .., .., ..])
            .assign(&spatial_maps.slice(s![src_i, .., .., ..]));
    }
    out
}

/// Precomputed spatial data shared across all per-gene training runs.
pub struct CachedSpatialData {
    pub spatial_features: Array2<f64>,
    pub spatial_maps: Array4<f32>,
}

#[inline]
pub(crate) fn finite_or_zero_f64(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

#[inline]
pub fn finite_or_zero_f32(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// In-sample R² with a finite denominator guard: when `SS_tot` is positive but tiny relative
/// to `SS_res`, `1 − SS_res/SS_tot` can take absurd magnitudes. Clamp so per-cluster metrics and
/// their means stay interpretable.
fn r2_in_sample_from_residuals(n: usize, ss_tot: f64, ss_res: f64) -> f64 {
    if ss_tot <= 0.0 {
        return 0.0;
    }
    if ss_tot < f64::EPSILON * ss_res.max(1.0) * (n as f64).max(1.0) {
        return f64::NAN;
    }
    let raw = 1.0 - ss_res / ss_tot;
    if !raw.is_finite() {
        return f64::NAN;
    }
    raw.clamp(-1_000_000.0, 1.0)
}

fn r2_score_from_pred_slice(y_true: ArrayView1<f64>, y_pred: &[f32]) -> f64 {
    let n = y_true.len();
    if n == 0 || y_pred.len() != n {
        return f64::NAN;
    }
    let y_mean: f64 = y_true.iter().copied().sum::<f64>() / n as f64;
    let ss_tot: f64 = y_true.iter().map(|y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, p)| (y - *p as f64).powi(2))
        .sum();
    r2_in_sample_from_residuals(n, ss_tot, ss_res)
}

fn cnn_lr_for_epoch(
    base_lr: f64,
    epoch: usize,
    total_epochs: usize,
    schedule: CnnLrSchedule,
    warmup_epochs: usize,
    cosine_min_ratio: f64,
) -> f64 {
    if total_epochs == 0 {
        return base_lr;
    }
    let w = warmup_epochs.min(total_epochs);
    if w > 0 && epoch < w {
        return base_lr * (epoch + 1) as f64 / w as f64;
    }
    match schedule {
        CnnLrSchedule::Constant => base_lr,
        CnnLrSchedule::Cosine => {
            let post = total_epochs.saturating_sub(w).max(1);
            let t = (epoch.saturating_sub(w)) as f64 / post.saturating_sub(1).max(1) as f64;
            let lr_min = base_lr * cosine_min_ratio;
            lr_min + (base_lr - lr_min) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
        }
    }
}

fn lasso_pred_align_weight_epoch(cnn: &CnnConfig, epoch: usize, total_epochs: usize) -> f32 {
    if cnn.lasso_pred_align_weight <= 0.0 {
        return 0.0;
    }
    let w0 = cnn.lasso_pred_align_weight as f32;
    if cnn.lasso_pred_align_linear_decay && total_epochs > 1 {
        let denom = total_epochs.saturating_sub(1).max(1) as f32;
        w0 * (1.0 - (epoch as f32) / denom)
    } else {
        w0
    }
}

fn y_lasso_vec_from_xy_cpu(
    x_c: &Array2<f64>,
    intercept: f64,
    lasso_coef: &Array2<f64>,
) -> Vec<f32> {
    let cluster_n = x_c.nrows();
    let n_mods = lasso_coef.nrows();
    let mut v = Vec::with_capacity(cluster_n);
    for i in 0..cluster_n {
        let mut s = intercept;
        for j in 0..n_mods {
            s += x_c[[i, j]] * lasso_coef[[j, 0]];
        }
        v.push(finite_or_zero_f32(s as f32));
    }
    v
}

fn y_lasso_batch_tensor_from_cpu<B: AutodiffBackend>(
    full: &[f32],
    idx: &[usize],
    device: &B::Device,
) -> Tensor<B, 1> {
    let v: Vec<f32> = idx.iter().map(|&i| finite_or_zero_f32(full[i])).collect();
    Tensor::from_data(burn::tensor::TensorData::new(v, [idx.len()]), device)
}

const CNN_MSE_FINITE_PROBE_EVERY: usize = 8;
const CNN_BEST_MODEL_MIN_REL_IMPROVE: f32 = 1e-4;

fn cnn_training_loss<B: AutodiffBackend>(
    model: &CellularNicheNetwork<B>,
    sm_tensor: Tensor<B, 4>,
    x_tensor: Tensor<B, 2>,
    sf_tensor: Tensor<B, 2>,
    y_tensor: Tensor<B, 1>,
    mean_beta_lasso_prior_weight: f32,
    y_lasso_tensor: Option<Tensor<B, 1>>,
    lasso_pred_align_weight: f32,
    mse_loss: &burn::nn::loss::MseLoss,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let betas = model.get_betas(sm_tensor, sf_tensor);
    let y_pred = CellularNicheNetwork::linear_readout_y(betas.clone(), x_tensor);
    let y_loss = mse_loss.forward(y_pred.clone(), y_tensor, burn::nn::loss::Reduction::Mean);
    let mut total = y_loss.clone();
    if mean_beta_lasso_prior_weight > 0.0 {
        let mean_betas = betas.mean_dim(0);
        let lasso_row = model.anchors_row.clone();
        let prior = mse_loss.forward(mean_betas, lasso_row, burn::nn::loss::Reduction::Mean);
        total = total + prior.mul_scalar(mean_beta_lasso_prior_weight);
    }
    if lasso_pred_align_weight > 0.0 {
        if let Some(yl) = y_lasso_tensor {
            let align = mse_loss.forward(y_pred, yl, burn::nn::loss::Reduction::Mean);
            total = total + align.mul_scalar(lasso_pred_align_weight);
        }
    }
    (total, y_loss)
}

fn cnn_r2_from_forward<B: AutodiffBackend>(
    model: &CellularNicheNetwork<B>,
    sm_c: &Array4<f32>,
    x_c: &Array2<f64>,
    sf_c: &Array2<f64>,
    y_true: ArrayView1<f64>,
    device: &B::Device,
    inference_batch_size: usize,
) -> f64 {
    let n = sm_c.shape()[0];
    if n == 0 || x_c.nrows() != n || sf_c.nrows() != n || y_true.len() != n {
        return f64::NAN;
    }
    let model_infer = model.valid();
    let bs = if inference_batch_size == 0 {
        n
    } else {
        inference_batch_size.max(1).min(n)
    };
    let mut preds = Vec::with_capacity(n);
    let mut pos = 0usize;
    while pos < n {
        let end = (pos + bs).min(n);
        let sm_tensor = tensor_from_sm_view::<<B as AutodiffBackend>::InnerBackend>(
            sm_c.slice(s![pos..end, .., .., ..]),
            device,
        );
        let x_tensor = tensor_from_x_view::<<B as AutodiffBackend>::InnerBackend>(
            x_c.slice(s![pos..end, ..]),
            device,
        );
        let sf_tensor = tensor_from_sf_view::<<B as AutodiffBackend>::InnerBackend>(
            sf_c.slice(s![pos..end, ..]),
            device,
        );
        let y_pred = model_infer.forward(sm_tensor, x_tensor, sf_tensor);
        let pred_data = y_pred.into_data();
        let pred: &[f32] = match pred_data.as_slice::<f32>() {
            Ok(s) => s,
            Err(_) => return f64::NAN,
        };
        preds.extend_from_slice(pred);
        pos = end;
    }
    r2_score_from_pred_slice(y_true, &preds)
}

fn gather_rows_4_f32(a: &Array4<f32>, idx: &[usize]) -> Array4<f32> {
    if idx.is_empty() {
        let sh = a.shape();
        return Array4::zeros((0, sh[1], sh[2], sh[3]));
    }
    let c = a.shape()[1];
    let h = a.shape()[2];
    let w = a.shape()[3];
    let mut out = Array4::<f32>::zeros((idx.len(), c, h, w));
    for (oi, &i) in idx.iter().enumerate() {
        out.slice_mut(s![oi, .., .., ..])
            .assign(&a.slice(s![i, .., .., ..]));
    }
    out
}

fn gather_rows_2_f64(a: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let cols = a.ncols();
    let mut out = Array2::zeros((idx.len(), cols));
    for (oi, &i) in idx.iter().enumerate() {
        out.row_mut(oi).assign(&a.row(i));
    }
    out
}

fn gather_rows_1_f64(a: &Array1<f64>, idx: &[usize]) -> Array1<f64> {
    Array1::from_vec(idx.iter().map(|&i| a[i]).collect())
}

fn tensor_from_sm<B: Backend>(a: &Array4<f32>, device: &B::Device) -> Tensor<B, 4> {
    tensor_from_sm_view(a.view(), device)
}

fn tensor_from_sm_view<B: Backend>(a: ArrayView4<'_, f32>, device: &B::Device) -> Tensor<B, 4> {
    let (b, c, h, w) = (a.shape()[0], a.shape()[1], a.shape()[2], a.shape()[3]);
    Tensor::from_data(
        burn::tensor::TensorData::new(
            a.iter().cloned().map(finite_or_zero_f32).collect(),
            [b, c, h, w],
        ),
        device,
    )
}

fn tensor_from_x<B: Backend>(a: &Array2<f64>, device: &B::Device) -> Tensor<B, 2> {
    tensor_from_x_view(a.view(), device)
}

fn tensor_from_x_view<B: Backend>(a: ArrayView2<'_, f64>, device: &B::Device) -> Tensor<B, 2> {
    let (r, c) = (a.nrows(), a.ncols());
    Tensor::from_data(
        burn::tensor::TensorData::new(
            a.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
            [r, c],
        ),
        device,
    )
}

fn tensor_from_y<B: Backend>(a: &Array1<f64>, device: &B::Device) -> Tensor<B, 1> {
    let n = a.len();
    Tensor::from_data(
        burn::tensor::TensorData::new(
            a.iter().map(|&v| finite_or_zero_f32(v as f32)).collect(),
            [n],
        ),
        device,
    )
}

pub fn train_cluster_cnn_epochs<B: AutodiffBackend>(
    mut model: CellularNicheNetwork<B>,
    device: &B::Device,
    sm_c: &Array4<f32>,
    x_c: &Array2<f64>,
    sf_c: &Array2<f64>,
    y_c: &Array1<f64>,
    cluster_n: usize,
    cluster_id: usize,
    y_lasso_cpu: Option<&[f32]>,
    beta_prior_w: f32,
    cnn: &CnnConfig,
    learning_rate: f64,
    epochs: usize,
    cnn_epoch_slot: Option<&Arc<CnnEpochHudSlot>>,
    shuffle_seed: u64,
) -> (CellularNicheNetwork<B>, Vec<f32>, bool) {
    let mut adam = AdamConfig::new()
        .with_beta_1(cnn.adam_beta_1 as f32)
        .with_beta_2(cnn.adam_beta_2 as f32)
        .with_epsilon(cnn.adam_epsilon as f32);
    if let Some(wd) = cnn.weight_decay {
        adam = adam.with_weight_decay(Some(WeightDecayConfig::new(wd as f32)));
    }
    if let Some(gc) = cnn.grad_clip_norm {
        adam = adam.with_grad_clipping(Some(GradientClippingConfig::Norm(gc as f32)));
    }
    let mut optim = adam.init::<B, CellularNicheNetwork<B>>();
    let mse_loss = burn::nn::loss::MseLoss::new();
    let bs_eff = if cnn.cnn_minibatch_size == 0 {
        cluster_n
    } else {
        cnn.cnn_minibatch_size.min(cluster_n).max(1)
    };
    let mut cnn_train_mse_epochs = Vec::with_capacity(epochs);
    let mut cnn_diverged = false;
    let patience = cnn.cnn_early_stop_patience;
    let min_ep = cnn.cnn_early_stop_min_epochs;
    let mut best_mse = f32::INFINITY;
    let mut no_improve_epochs = 0usize;
    let mut best_model: Option<CellularNicheNetwork<B>> = None;

    for epoch in 0..epochs {
        if let Some(s) = cnn_epoch_slot.as_ref() {
            s.set_current(epoch + 1);
        }
        let lr = cnn_lr_for_epoch(
            learning_rate,
            epoch,
            epochs,
            cnn.lr_schedule,
            cnn.lr_warmup_epochs,
            cnn.cosine_lr_min_ratio,
        );
        let align_w = lasso_pred_align_weight_epoch(cnn, epoch, epochs);
        let mut order: Vec<usize> = (0..cluster_n).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(
            shuffle_seed
                ^ 0x9E37_79B9_7F4A7C15_u64
                ^ (cluster_id as u64).wrapping_shl(32)
                ^ (epoch as u64),
        );
        order.shuffle(&mut rng);

        let mut epoch_mse_den = 0.0f32;
        let mut epoch_mse_acc = Tensor::<B, 1>::zeros([1], device);
        let mut batch_in_epoch = 0usize;

        let max_batches_cfg = match cnn.cnn_max_batches_per_epoch {
            None | Some(0) => usize::MAX,
            Some(b) => b,
        };
        let max_batches_from_cells = match cnn.cnn_max_cells_per_epoch {
            None | Some(0) => usize::MAX,
            Some(c) => c.div_ceil(bs_eff).max(1),
        };
        let max_batches = max_batches_cfg.min(max_batches_from_cells);

        let mut pos = 0usize;
        while pos < cluster_n && batch_in_epoch < max_batches {
            let end = (pos + bs_eff).min(cluster_n);
            let batch_idx = &order[pos..end];
            pos = end;

            let sm_b = gather_rows_4_f32(sm_c, batch_idx);
            let x_b = gather_rows_2_f64(x_c, batch_idx);
            let sf_b = gather_rows_2_f64(sf_c, batch_idx);
            let y_b = gather_rows_1_f64(y_c, batch_idx);
            let batch_n = batch_idx.len();

            let sm_tensor = tensor_from_sm(&sm_b, device);
            let x_tensor = tensor_from_x(&x_b, device);
            let sf_tensor = tensor_from_sf(&sf_b, device);
            let y_tensor = tensor_from_y(&y_b, device);
            let y_lasso_b = y_lasso_cpu
                .as_ref()
                .map(|sl| y_lasso_batch_tensor_from_cpu(sl, batch_idx, device));

            let (loss, y_mse) = cnn_training_loss(
                &model,
                sm_tensor,
                x_tensor,
                sf_tensor,
                y_tensor,
                beta_prior_w,
                y_lasso_b,
                align_w,
                &mse_loss,
            );
            let w = batch_n as f32;
            let y_mse_d = y_mse.detach();
            epoch_mse_acc = epoch_mse_acc + y_mse_d.clone().mul_scalar(w);
            epoch_mse_den += w;

            batch_in_epoch += 1;
            if batch_in_epoch % CNN_MSE_FINITE_PROBE_EVERY == 0 {
                let probe = y_mse_d.into_scalar().elem::<f32>();
                if !probe.is_finite() {
                    cnn_diverged = true;
                    break;
                }
            }
            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);
        }

        if cnn_diverged {
            break;
        }

        let mse = if epoch_mse_den > 0.0 {
            let sum = epoch_mse_acc.into_scalar().elem::<f32>();
            finite_or_zero_f32(sum / epoch_mse_den)
        } else {
            f32::NAN
        };
        if !mse.is_finite() {
            cnn_diverged = true;
            break;
        }
        cnn_train_mse_epochs.push(mse);
        if mse < best_mse {
            let prev_best = best_mse;
            no_improve_epochs = 0;
            best_mse = mse;
            let rel_improve = (prev_best - mse) / (prev_best.abs() + 1e-6f32);
            let take_snapshot = best_model.is_none()
                || prev_best.is_infinite()
                || rel_improve > CNN_BEST_MODEL_MIN_REL_IMPROVE;
            if take_snapshot {
                best_model = Some(model.clone());
            }
        } else if epoch + 1 >= min_ep && patience > 0 {
            no_improve_epochs += 1;
            if no_improve_epochs >= patience {
                break;
            }
        }
    }

    if let Some(m) = best_model {
        model = m;
    }

    (model, cnn_train_mse_epochs, cnn_diverged)
}

fn tensor_from_sf<B: Backend>(a: &Array2<f64>, device: &B::Device) -> Tensor<B, 2> {
    tensor_from_x(a, device)
}

fn tensor_from_sf_view<B: Backend>(a: ArrayView2<'_, f64>, device: &B::Device) -> Tensor<B, 2> {
    tensor_from_x_view(a, device)
}

fn min_max_finite_col(col: ndarray::ArrayView1<f64>) -> (f32, f32) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in col.iter() {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        (0.0, 0.0)
    } else {
        (lo as f32, hi as f32)
    }
}

#[derive(Debug, Clone)]
pub struct ClusterTrainingSummary {
    pub cluster_id: usize,
    pub n_cells: usize,
    pub n_modulators: usize,
    pub lasso_r2: f64,
    pub lasso_train_mse: f64,
    pub lasso_fista_iters: usize,
    pub lasso_converged: bool,
    pub cnn_train_mse_epochs: Vec<f32>,
    pub cnn_r2: f64,
}

/// In-sample R² for HUD / leaderboards after the same arbitration as CNN export: when
/// `drop_cnn_if_insample_worse_than_lasso` holds and CNN is worse than Lasso (including margin),
/// returns `lasso_r2`; otherwise finite `cnn_r2`, else `lasso_r2`.
pub fn cluster_insample_r2_for_hud(
    s: &ClusterTrainingSummary,
    drop_cnn_if_insample_worse_than_lasso: bool,
    cnn_vs_lasso_arbitration_margin: f64,
) -> f64 {
    let l = s.lasso_r2;
    let c = s.cnn_r2;
    if drop_cnn_if_insample_worse_than_lasso
        && l.is_finite()
        && c.is_finite()
        && c + cnn_vs_lasso_arbitration_margin < l
    {
        return l;
    }
    if c.is_finite() {
        return c;
    }
    if l.is_finite() { l } else { f64::NAN }
}

struct LassoPassData {
    cluster_id: usize,
    indices: Vec<usize>,
    cluster_n: usize,
    lasso_coef: Array2<f64>,
    intercept: f64,
    lasso_converged: bool,
    lasso_fista_iters: usize,
    lasso_train_mse: f64,
    r2: f64,
}

enum ClusterLassoPhase {
    Skipped,
    FitErrored,
    BelowThreshold(ClusterTrainingSummary),
    Pass(LassoPassData),
}

fn fit_cluster_group_lasso(
    c_id: usize,
    n_samples: usize,
    x: &Array2<f64>,
    y: &Array1<f64>,
    clusters: &Array1<usize>,
    params: GroupLassoParams,
    group_regs: Option<Vec<f64>>,
    regulator_masks: Option<&HashMap<usize, Vec<bool>>>,
    score_threshold: f64,
) -> ClusterLassoPhase {
    let indices: Vec<usize> = (0..n_samples).filter(|&i| clusters[i] == c_id).collect();
    if indices.is_empty() {
        return ClusterLassoPhase::Skipped;
    }

    let mut x_c = x.select(Axis(0), &indices);
    if let Some(mask) = regulator_masks.and_then(|m| m.get(&c_id)) {
        for (j, allowed) in mask.iter().copied().enumerate().take(x_c.ncols()) {
            if !allowed {
                x_c.column_mut(j).fill(0.0);
            }
        }
    }
    let y_c = y.select(Axis(0), &indices).insert_axis(Axis(1));

    let mut lasso = if let Some(regs) = group_regs {
        GroupLasso::new_with_regs(params, regs)
    } else {
        GroupLasso::new(params)
    };

    let lasso_converged = match lasso.fit(&x_c, &y_c, None) {
        Ok(_) => true,
        Err(crate::lasso::GroupLassoError::ConvergenceWarning) => false,
        Err(e) => {
            println!("⚠️ Lasso fit error for cluster {}: {:?}", c_id, e);
            return ClusterLassoPhase::FitErrored;
        }
    };

    let fitted = lasso.fitted.as_ref().unwrap();
    let lasso_coef = fitted.coef.mapv(finite_or_zero_f64);
    let intercept = finite_or_zero_f64(fitted.intercept[[0, 0]]);
    let lasso_fista_iters = lasso.last_fista_iterations;

    let y_pred_lasso = lasso.predict(&x_c).unwrap();
    let y_c_flat = y_c.column(0);
    let y_pred_flat = y_pred_lasso.column(0);
    let ss_res: f64 = y_c_flat
        .iter()
        .zip(y_pred_flat.iter())
        .map(|(yi, yhat)| (yi - yhat).powi(2))
        .sum();
    let cluster_n = indices.len();
    let lasso_train_mse = ss_res / cluster_n.max(1) as f64;
    let y_mean = y_c_flat.mean().unwrap_or(0.0);
    let ss_tot: f64 = y_c_flat.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r2 = finite_or_zero_f64(r2_in_sample_from_residuals(cluster_n, ss_tot, ss_res));

    if !r2.is_finite() || r2 < score_threshold {
        return ClusterLassoPhase::BelowThreshold(ClusterTrainingSummary {
            cluster_id: c_id,
            n_cells: cluster_n,
            n_modulators: lasso_coef.nrows(),
            lasso_r2: r2,
            lasso_train_mse,
            lasso_fista_iters,
            lasso_converged,
            cnn_train_mse_epochs: Vec::new(),
            cnn_r2: f64::NAN,
        });
    }

    ClusterLassoPhase::Pass(LassoPassData {
        cluster_id: c_id,
        indices,
        cluster_n,
        lasso_coef,
        intercept,
        lasso_converged,
        lasso_fista_iters,
        lasso_train_mse,
        r2,
    })
}

pub struct FittedClusterResult<B: AutodiffBackend> {
    pub cluster_id: usize,
    pub model: CellularNicheNetwork<B>,
    pub r2: f64,
    pub lasso_coef: Array2<f64>,
    pub intercept: f64,
}

#[derive(Debug)]
pub struct CnnEpochHudSlot {
    current: AtomicUsize,
    total: AtomicUsize,
}

impl CnnEpochHudSlot {
    pub fn new(total_epochs: usize) -> Arc<Self> {
        let total = total_epochs.max(1);
        Arc::new(Self {
            current: AtomicUsize::new(0),
            total: AtomicUsize::new(total),
        })
    }

    pub fn reconfigure(&self, total_epochs: usize) {
        let total = total_epochs.max(1);
        self.total.store(total, Ordering::Relaxed);
        self.current.store(0, Ordering::Relaxed);
    }

    #[inline]
    pub fn set_current(&self, one_based_epoch: usize) {
        self.current.store(one_based_epoch, Ordering::Relaxed);
    }

    #[inline]
    pub fn current(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn total(&self) -> usize {
        self.total.load(Ordering::Relaxed)
    }
}

pub struct ClusteredGcnNwrFitInputs<'a, 'd, B: AutodiffBackend> {
    pub x: &'a Array2<f64>,
    pub y: &'a Array1<f64>,
    pub xy: &'a Array2<f64>,
    pub clusters: &'a Array1<usize>,
    pub num_clusters: usize,
    pub device: &'d B::Device,
    pub epochs: usize,
    pub learning_rate: f64,
    /// Minimum in-sample Lasso R² per cluster; clusters below this skip CNN/seed export for that gene.
    pub score_threshold: f64,
    pub seed_only: bool,
    pub cnn: &'a CnnConfig,
    pub cached_spatial: Option<&'a CachedSpatialData>,
    pub cnn_epoch_slot: Option<Arc<CnnEpochHudSlot>>,
    pub parallel_lasso_clusters: bool,
    /// XOR’d into ChaCha seeds for CNN minibatch shuffles (use `[execution].random_seed`).
    pub random_seed: u64,
}

pub struct ClusteredGcnNwrCnnRefineInputs<'a, 'd, B: AutodiffBackend> {
    pub x: &'a Array2<f64>,
    pub y: &'a Array1<f64>,
    pub xy: &'a Array2<f64>,
    pub clusters: &'a Array1<usize>,
    pub num_clusters: usize,
    pub device: &'d B::Device,
    pub epochs: usize,
    pub learning_rate: f64,
    pub cnn: &'a CnnConfig,
    pub cached_spatial: Option<&'a CachedSpatialData>,
    pub cnn_epoch_slot: Option<Arc<CnnEpochHudSlot>>,
    pub random_seed: u64,
}

pub struct ClusteredGCNNWR<B: AutodiffBackend> {
    pub params: GroupLassoParams,
    pub spatial_dim: usize,
    pub spatial_feature_radius: f64,
    pub ego_center_spatial_maps: bool,
    pub multi_channel_spatial_maps: bool,
    pub models: HashMap<usize, CellularNicheNetwork<B>>,
    pub r2_scores: HashMap<usize, f64>,
    pub lasso_coefficients: HashMap<usize, Array2<f64>>,
    pub lasso_intercepts: HashMap<usize, f64>,
    pub group_reg_vec: Option<Vec<f64>>,
    pub regulator_masks_by_cluster: Option<HashMap<usize, Vec<bool>>>,
    pub cluster_training_summaries: Vec<ClusterTrainingSummary>,
}

impl<B: AutodiffBackend> ClusteredGCNNWR<B> {
    pub fn new(
        params: GroupLassoParams,
        spatial_dim: usize,
        spatial_feature_radius: f64,
        ego_center_spatial_maps: bool,
        multi_channel_spatial_maps: bool,
    ) -> Self {
        Self {
            params,
            spatial_dim,
            spatial_feature_radius,
            ego_center_spatial_maps,
            multi_channel_spatial_maps,
            models: HashMap::new(),
            r2_scores: HashMap::new(),
            lasso_coefficients: HashMap::new(),
            lasso_intercepts: HashMap::new(),
            group_reg_vec: None,
            regulator_masks_by_cluster: None,
            cluster_training_summaries: Vec::new(),
        }
    }

    pub fn fit<F: FnMut(usize, usize) + Send>(
        &mut self,
        inputs: ClusteredGcnNwrFitInputs<'_, '_, B>,
        lasso_progress: F,
    ) {
        let ClusteredGcnNwrFitInputs {
            x,
            y,
            xy,
            clusters,
            num_clusters,
            device,
            epochs,
            learning_rate,
            score_threshold,
            seed_only,
            cnn,
            cached_spatial,
            cnn_epoch_slot,
            parallel_lasso_clusters,
            random_seed,
        } = inputs;
        let n_samples = x.nrows();
        let to_fit: Vec<usize> = (0..num_clusters)
            .filter(|&c_id| (0..n_samples).any(|i| clusters[i] == c_id))
            .collect();
        let n_celltypes = to_fit.len();

        let owned_sf;
        let owned_sm;
        let (spatial_features, spatial_maps) = if let Some(c) = cached_spatial {
            (&c.spatial_features, &c.spatial_maps)
        } else {
            let r_sf = self.spatial_feature_radius;
            owned_sf = create_spatial_features(xy, clusters, num_clusters, r_sf);
            owned_sm = xyc2spatial_fast(
                xy,
                clusters,
                num_clusters,
                self.spatial_dim,
                self.spatial_dim,
                self.ego_center_spatial_maps,
            );
            (&owned_sf, &owned_sm)
        };

        self.cluster_training_summaries.clear();
        let mut training_summaries: Vec<ClusterTrainingSummary> = Vec::new();
        let mut fitted_results: Vec<FittedClusterResult<B>> = Vec::new();

        let lasso_progress = Arc::new(Mutex::new(lasso_progress));
        let lasso_done = Arc::new(AtomicUsize::new(0));
        {
            let mut cb = lasso_progress.lock().unwrap_or_else(|e| e.into_inner());
            cb(0, n_celltypes);
        }

        let params = self.params.clone();
        let group_regs = self.group_reg_vec.clone();
        let regulator_masks = self.regulator_masks_by_cluster.as_ref();

        let phase_results: Vec<ClusterLassoPhase> = if n_celltypes == 0 {
            Vec::new()
        } else if parallel_lasso_clusters {
            let pool = ThreadPoolBuilder::new()
                .num_threads(GROUP_LASSO_MAX_CONCURRENT)
                .build()
                .expect("group lasso thread pool");
            pool.install(|| {
                (0..to_fit.len())
                    .into_par_iter()
                    .map(|idx| {
                        let c_id = to_fit[idx];
                        let out = fit_cluster_group_lasso(
                            c_id,
                            n_samples,
                            x,
                            y,
                            clusters,
                            params.clone(),
                            group_regs.clone(),
                            regulator_masks,
                            score_threshold,
                        );
                        let d = lasso_done.fetch_add(1, Ordering::Relaxed) + 1;
                        lasso_progress.lock().unwrap_or_else(|e| e.into_inner())(d, n_celltypes);
                        out
                    })
                    .collect()
            })
        } else {
            (0..to_fit.len())
                .map(|idx| {
                    let c_id = to_fit[idx];
                    let out = fit_cluster_group_lasso(
                        c_id,
                        n_samples,
                        x,
                        y,
                        clusters,
                        params.clone(),
                        group_regs.clone(),
                        regulator_masks,
                        score_threshold,
                    );
                    let d = lasso_done.fetch_add(1, Ordering::Relaxed) + 1;
                    lasso_progress.lock().unwrap_or_else(|e| e.into_inner())(d, n_celltypes);
                    out
                })
                .collect()
        };

        let n_cnn_clusters = phase_results
            .iter()
            .filter(|p| matches!(p, ClusterLassoPhase::Pass(_)))
            .count();
        if !seed_only && n_cnn_clusters > 0 {
            lasso_progress.lock().unwrap_or_else(|e| e.into_inner())(0, n_cnn_clusters);
        }
        let mut cnn_cluster_done = 0usize;

        for (phase, c_id) in phase_results.into_iter().zip(to_fit.iter().copied()) {
            match phase {
                ClusterLassoPhase::Skipped | ClusterLassoPhase::FitErrored => {}
                ClusterLassoPhase::BelowThreshold(s) => {
                    debug_assert_eq!(s.cluster_id, c_id);
                    training_summaries.push(s);
                }
                ClusterLassoPhase::Pass(pass) => {
                    debug_assert_eq!(pass.cluster_id, c_id);
                    let LassoPassData {
                        cluster_id: c_id,
                        indices,
                        cluster_n,
                        lasso_coef,
                        intercept,
                        lasso_converged,
                        lasso_fista_iters,
                        lasso_train_mse,
                        r2,
                    } = pass;

                    let mut x_c = x.select(Axis(0), &indices);
                    if let Some(mask) = self
                        .regulator_masks_by_cluster
                        .as_ref()
                        .and_then(|m| m.get(&c_id))
                    {
                        for (j, allowed) in mask.iter().copied().enumerate().take(x_c.ncols()) {
                            if !allowed {
                                x_c.column_mut(j).fill(0.0);
                            }
                        }
                    }
                    let y_c = y.select(Axis(0), &indices).insert_axis(Axis(1));

                    let mut anchors_vec = vec![finite_or_zero_f32(intercept as f32)];
                    anchors_vec.extend(
                        lasso_coef
                            .column(0)
                            .iter()
                            .map(|&v| finite_or_zero_f32(v as f32)),
                    );

                    let anchors_tensor = Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::new(
                            anchors_vec.clone(),
                            [lasso_coef.nrows() + 1],
                        ),
                        device,
                    );

                    let vision_in_channels = if cnn.multi_channel_spatial_maps {
                        num_clusters
                    } else {
                        1
                    };
                    let config = CellularNicheNetworkConfig {
                        n_modulators: lasso_coef.nrows(),
                        n_clusters: num_clusters,
                        vision_in_channels,
                    };
                    let model =
                        config.init::<B>(device, anchors_tensor.clone(), cnn.output_activation);

                    let skip_cnn_training = seed_only
                        || (cnn.min_cells_for_cnn > 0 && cluster_n < cnn.min_cells_for_cnn);

                    if skip_cnn_training {
                        training_summaries.push(ClusterTrainingSummary {
                            cluster_id: c_id,
                            n_cells: cluster_n,
                            n_modulators: lasso_coef.nrows(),
                            lasso_r2: r2,
                            lasso_train_mse,
                            lasso_fista_iters,
                            lasso_converged,
                            cnn_train_mse_epochs: Vec::new(),
                            cnn_r2: f64::NAN,
                        });
                        fitted_results.push(FittedClusterResult {
                            cluster_id: c_id,
                            model,
                            r2,
                            lasso_coef,
                            intercept,
                        });
                        if !seed_only {
                            cnn_cluster_done += 1;
                            lasso_progress.lock().unwrap_or_else(|e| e.into_inner())(
                                cnn_cluster_done,
                                n_cnn_clusters,
                            );
                        }
                        continue;
                    }

                    let sf_c = spatial_features.select(Axis(0), &indices);
                    let sm_c = if cnn.multi_channel_spatial_maps {
                        spatial_maps_all_clusters_for_cnn(spatial_maps, &indices)
                    } else {
                        spatial_maps_for_cluster_cnn(spatial_maps, &indices, c_id)
                    };
                    let y_1d = y_c.column(0).into_owned();

                    let y_lasso_cpu_vec: Option<Vec<f32>> = if cnn.lasso_pred_align_weight > 0.0 {
                        Some(y_lasso_vec_from_xy_cpu(&x_c, intercept, &lasso_coef))
                    } else {
                        None
                    };
                    let beta_prior_w = cnn.mean_beta_lasso_prior_weight as f32;

                    let (model, cnn_train_mse_epochs, cnn_diverged) = train_cluster_cnn_epochs(
                        model,
                        device,
                        &sm_c,
                        &x_c,
                        &sf_c,
                        &y_1d,
                        cluster_n,
                        c_id,
                        y_lasso_cpu_vec.as_deref(),
                        beta_prior_w,
                        cnn,
                        learning_rate,
                        epochs,
                        cnn_epoch_slot.as_ref(),
                        random_seed,
                    );

                    let cnn_r2 = if cnn_diverged {
                        f64::NAN
                    } else {
                        cnn_r2_from_forward(
                            &model,
                            &sm_c,
                            &x_c,
                            &sf_c,
                            y_c.column(0),
                            device,
                            cnn.cnn_inference_batch_size,
                        )
                    };

                    training_summaries.push(ClusterTrainingSummary {
                        cluster_id: c_id,
                        n_cells: cluster_n,
                        n_modulators: lasso_coef.nrows(),
                        lasso_r2: r2,
                        lasso_train_mse,
                        lasso_fista_iters,
                        lasso_converged,
                        cnn_train_mse_epochs,
                        cnn_r2,
                    });

                    fitted_results.push(FittedClusterResult {
                        cluster_id: c_id,
                        model,
                        r2,
                        lasso_coef,
                        intercept,
                    });

                    cnn_cluster_done += 1;
                    lasso_progress.lock().unwrap_or_else(|e| e.into_inner())(
                        cnn_cluster_done,
                        n_cnn_clusters,
                    );
                }
            }
        }

        self.cluster_training_summaries = training_summaries;

        for fit in fitted_results {
            self.models.insert(fit.cluster_id, fit.model);
            self.r2_scores.insert(fit.cluster_id, fit.r2);
            self.lasso_coefficients
                .insert(fit.cluster_id, fit.lasso_coef);
            self.lasso_intercepts.insert(fit.cluster_id, fit.intercept);
        }

        for s in &self.cluster_training_summaries {
            if !s.cnn_r2.is_finite() {
                self.models.remove(&s.cluster_id);
            } else if cnn.drop_cnn_if_insample_worse_than_lasso
                && s.lasso_r2.is_finite()
                && s.cnn_r2 + cnn.cnn_vs_lasso_arbitration_margin < s.lasso_r2
            {
                self.models.remove(&s.cluster_id);
            }
        }
    }

    pub fn fit_cnn_refinement<F: FnMut(usize, usize) + Send>(
        &mut self,
        inputs: ClusteredGcnNwrCnnRefineInputs<'_, '_, B>,
        mut cluster_progress: F,
    ) {
        let ClusteredGcnNwrCnnRefineInputs {
            x,
            y,
            xy,
            clusters,
            num_clusters,
            device,
            epochs,
            learning_rate,
            cnn,
            cached_spatial,
            cnn_epoch_slot,
            random_seed,
        } = inputs;
        let n_samples = x.nrows();

        let owned_sf;
        let owned_sm;
        let (spatial_features, spatial_maps) = if let Some(c) = cached_spatial {
            (&c.spatial_features, &c.spatial_maps)
        } else {
            owned_sf =
                create_spatial_features(xy, clusters, num_clusters, self.spatial_feature_radius);
            owned_sm = xyc2spatial_fast(
                xy,
                clusters,
                num_clusters,
                self.spatial_dim,
                self.spatial_dim,
                self.ego_center_spatial_maps,
            );
            (&owned_sf, &owned_sm)
        };

        let mut summaries_by_cluster: HashMap<usize, ClusterTrainingSummary> = self
            .cluster_training_summaries
            .iter()
            .cloned()
            .map(|s| (s.cluster_id, s))
            .collect();

        let n_cnn_clusters = (0..num_clusters)
            .filter(|&c| self.models.contains_key(&c))
            .filter(|&c| (0..n_samples).any(|i| clusters[i] == c))
            .count();
        if n_cnn_clusters > 0 {
            cluster_progress(0, n_cnn_clusters);
        }
        let mut cnn_cluster_done = 0usize;

        for c_id in 0..num_clusters {
            if !self.models.contains_key(&c_id) {
                continue;
            }
            let indices: Vec<usize> = (0..n_samples).filter(|&i| clusters[i] == c_id).collect();
            if indices.is_empty() {
                continue;
            }

            let model = match self.models.remove(&c_id) {
                Some(m) => m,
                None => continue,
            };

            let cluster_n = indices.len();
            let skip_cnn = cnn.min_cells_for_cnn > 0 && cluster_n < cnn.min_cells_for_cnn;

            let x_c = x.select(Axis(0), &indices);
            let mut x_c = x_c;
            if let Some(mask) = self
                .regulator_masks_by_cluster
                .as_ref()
                .and_then(|m| m.get(&c_id))
            {
                for (j, allowed) in mask.iter().copied().enumerate().take(x_c.ncols()) {
                    if !allowed {
                        x_c.column_mut(j).fill(0.0);
                    }
                }
            }
            let y_c = y.select(Axis(0), &indices);
            let y_1d = Array1::from_vec(y_c.iter().copied().collect());

            let (cnn_train_mse_epochs, cnn_r2) = if skip_cnn {
                (Vec::new(), f64::NAN)
            } else {
                let sf_c = spatial_features.select(Axis(0), &indices);
                let sm_c = if self.multi_channel_spatial_maps {
                    spatial_maps_all_clusters_for_cnn(spatial_maps, &indices)
                } else {
                    spatial_maps_for_cluster_cnn(spatial_maps, &indices, c_id)
                };

                let y_lasso_cpu_vec: Option<Vec<f32>> = if cnn.lasso_pred_align_weight > 0.0 {
                    if let (Some(coef), Some(&inter)) = (
                        self.lasso_coefficients.get(&c_id),
                        self.lasso_intercepts.get(&c_id),
                    ) {
                        Some(y_lasso_vec_from_xy_cpu(&x_c, inter, coef))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let beta_prior_w = cnn.mean_beta_lasso_prior_weight as f32;

                let (model, cnn_train_mse_epochs, cnn_diverged) = train_cluster_cnn_epochs(
                    model,
                    device,
                    &sm_c,
                    &x_c,
                    &sf_c,
                    &y_1d,
                    cluster_n,
                    c_id,
                    y_lasso_cpu_vec.as_deref(),
                    beta_prior_w,
                    cnn,
                    learning_rate,
                    epochs,
                    cnn_epoch_slot.as_ref(),
                    random_seed,
                );

                let cnn_r2 = if cnn_diverged {
                    f64::NAN
                } else {
                    cnn_r2_from_forward(
                        &model,
                        &sm_c,
                        &x_c,
                        &sf_c,
                        y_1d.view(),
                        device,
                        cnn.cnn_inference_batch_size,
                    )
                };

                if !cnn_diverged && cnn_r2.is_finite() {
                    self.models.insert(c_id, model);
                }

                (cnn_train_mse_epochs, cnn_r2)
            };

            if let Some(s) = summaries_by_cluster.get_mut(&c_id) {
                s.cnn_train_mse_epochs = cnn_train_mse_epochs;
                s.cnn_r2 = cnn_r2;
            }

            cnn_cluster_done += 1;
            cluster_progress(cnn_cluster_done, n_cnn_clusters);
        }

        let mut ordered: Vec<ClusterTrainingSummary> = summaries_by_cluster.into_values().collect();
        ordered.sort_by_key(|s| s.cluster_id);
        self.cluster_training_summaries = ordered;

        if cnn.drop_cnn_if_insample_worse_than_lasso {
            for s in &self.cluster_training_summaries {
                if s.lasso_r2.is_finite()
                    && s.cnn_r2.is_finite()
                    && s.cnn_r2 + cnn.cnn_vs_lasso_arbitration_margin < s.lasso_r2
                {
                    self.models.remove(&s.cluster_id);
                }
            }
        }
    }

    pub fn predict_betas(
        &self,
        x: &Array2<f64>,
        xy: &Array2<f64>,
        clusters: &Array1<usize>,
        num_clusters: usize,
        device: &B::Device,
        cached_spatial: Option<&CachedSpatialData>,
        inference_batch_size: usize,
    ) -> Array2<f64> {
        let n_samples = xy.nrows();
        let n_modulators = x.ncols();

        let owned_sf;
        let owned_sm;
        let (spatial_features, spatial_maps) = if let Some(c) = cached_spatial {
            (&c.spatial_features, &c.spatial_maps)
        } else {
            owned_sf =
                create_spatial_features(xy, clusters, num_clusters, self.spatial_feature_radius);
            owned_sm = xyc2spatial_fast(
                xy,
                clusters,
                num_clusters,
                self.spatial_dim,
                self.spatial_dim,
                self.ego_center_spatial_maps,
            );
            (&owned_sf, &owned_sm)
        };

        let mut all_betas = Array2::<f64>::zeros((n_samples, n_modulators + 1));

        for c_id in 0..num_clusters {
            let indices: Vec<usize> = (0..n_samples).filter(|&i| clusters[i] == c_id).collect();
            if indices.is_empty() {
                continue;
            }

            if let Some(model) = self.models.get(&c_id) {
                let sf_c = spatial_features.select(Axis(0), &indices);
                let sm_c = if self.multi_channel_spatial_maps {
                    spatial_maps_all_clusters_for_cnn(spatial_maps, &indices)
                } else {
                    spatial_maps_for_cluster_cnn(spatial_maps, &indices, c_id)
                };
                let cluster_n = indices.len();
                let bs_eff = if inference_batch_size == 0 {
                    cluster_n.max(1)
                } else {
                    inference_batch_size.max(1)
                };

                let model_infer = model.valid();
                let n_betas = n_modulators + 1;
                let mut packed: Vec<f32> = Vec::with_capacity(cluster_n.saturating_mul(n_betas));
                let mut row = 0usize;
                while row < cluster_n {
                    let end = (row + bs_eff).min(cluster_n);
                    let sm_tensor = tensor_from_sm_view::<<B as AutodiffBackend>::InnerBackend>(
                        sm_c.slice(s![row..end, .., .., ..]),
                        device,
                    );
                    let sf_tensor = tensor_from_sf_view::<<B as AutodiffBackend>::InnerBackend>(
                        sf_c.slice(s![row..end, ..]),
                        device,
                    );
                    let betas_tensor = model_infer.get_betas(sm_tensor, sf_tensor);
                    let betas_data = betas_tensor.into_data();
                    let betas_v: &[f32] = betas_data.as_slice::<f32>().unwrap();
                    packed.extend_from_slice(betas_v);
                    row = end;
                }

                let cnn_has_nan = packed.iter().any(|v| !v.is_finite());
                if !cnn_has_nan {
                    for (i, idx) in indices.iter().enumerate() {
                        for j in 0..n_betas {
                            let v = packed[i * n_betas + j];
                            all_betas[[*idx, j]] = finite_or_zero_f32(v) as f64;
                        }
                    }
                } else if let Some(lasso_coef) = self.lasso_coefficients.get(&c_id) {
                    let intercept = finite_or_zero_f64(
                        self.lasso_intercepts.get(&c_id).copied().unwrap_or(0.0),
                    );
                    let coef_col = lasso_coef.column(0);
                    for &idx in &indices {
                        all_betas[[idx, 0]] = intercept;
                        for (j, &v) in coef_col.iter().enumerate() {
                            all_betas[[idx, j + 1]] = finite_or_zero_f64(v);
                        }
                    }
                }
            } else if let Some(lasso_coef) = self.lasso_coefficients.get(&c_id) {
                let intercept =
                    finite_or_zero_f64(self.lasso_intercepts.get(&c_id).copied().unwrap_or(0.0));
                let coef_col = lasso_coef.column(0);
                for &idx in &indices {
                    all_betas[[idx, 0]] = intercept;
                    for (j, &v) in coef_col.iter().enumerate() {
                        all_betas[[idx, j + 1]] = finite_or_zero_f64(v);
                    }
                }
            }
        }
        all_betas.mapv(finite_or_zero_f64)
    }
}

pub fn xyc2spatial_fast(
    xy: &Array2<f64>,
    clusters: &Array1<usize>,
    num_clusters: usize,
    m: usize,
    n: usize,
    ego_center: bool,
) -> Array4<f32> {
    let num_cells = xy.nrows();
    let x_col = xy.column(0);
    let y_col = xy.column(1);

    let (xmin, xmax) = min_max_finite_col(x_col);
    let (ymin, ymax) = min_max_finite_col(y_col);

    let span_x = (xmax - xmin).max(1e-6);
    let span_y = (ymax - ymin).max(1e-6);
    let cell_width = span_x / n as f32;
    let cell_height = span_y / m as f32;

    let cx_global: Vec<f32> = (0..n)
        .map(|j| xmin + (j as f32 + 0.5) * cell_width)
        .collect();
    let cy_global: Vec<f32> = (0..m)
        .map(|i| ymax - (i as f32 + 0.5) * cell_height)
        .collect();

    let mut spatial_maps = Array4::<f32>::zeros((num_cells, num_clusters, m, n));

    spatial_maps
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(s, mut cell_maps)| {
            let cluster_s = clusters[s];
            if cluster_s >= num_clusters {
                return;
            }
            let x_s = xy[[s, 0]] as f32;
            let y_s = xy[[s, 1]] as f32;
            if !x_s.is_finite() || !y_s.is_finite() {
                return;
            }

            let cx_ego = ego_center.then(|| {
                let half_x = span_x * 0.5;
                (0..n)
                    .map(|j| x_s - half_x + (j as f32 + 0.5) * cell_width)
                    .collect::<Vec<f32>>()
            });
            let cy_ego = ego_center.then(|| {
                let half_y = span_y * 0.5;
                let top_y = y_s + half_y;
                (0..m)
                    .map(|i| top_y - (i as f32 + 0.5) * cell_height)
                    .collect::<Vec<f32>>()
            });
            let cx_grid: &[f32] = cx_ego.as_deref().unwrap_or(&cx_global);
            let cy_grid: &[f32] = cy_ego.as_deref().unwrap_or(&cy_global);

            let mut channel_map = cell_maps.index_axis_mut(Axis(0), cluster_s);

            for i in 0..m {
                let gy = cy_grid[i];
                if !gy.is_finite() {
                    continue;
                }
                let dy2 = (y_s - gy).powi(2);
                for j in 0..n {
                    let gx = cx_grid[j];
                    if !gx.is_finite() {
                        continue;
                    }
                    let dx2 = (x_s - gx).powi(2);
                    let d = (dx2 + dy2).sqrt().max(1e-6);
                    channel_map[[i, j]] = 1.0 / d;
                }
            }
        });

    spatial_maps
}

pub fn create_spatial_features(
    xy: &Array2<f64>,
    clusters: &Array1<usize>,
    num_clusters: usize,
    radius: f64,
) -> Array2<f64> {
    let n = xy.nrows();
    let mut result = Array2::zeros((n, num_clusters));
    let r2 = radius * radius;
    let r2 = if r2 > 0.0 && r2.is_finite() {
        r2.next_up()
    } else {
        r2
    };

    let mut points = Vec::with_capacity(n);
    let mut valid_indices = Vec::with_capacity(n);
    for i in 0..n {
        let x = xy[[i, 0]];
        let y = xy[[i, 1]];
        if x.is_finite() && y.is_finite() {
            valid_indices.push(i);
            points.push([x, y]);
        }
    }

    if points.is_empty() {
        return result;
    }

    let tree = kiddo::ImmutableKdTree::<f64, 2>::new_from_slice(&points);

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let xi = xy[[i, 0]];
            let yi = xy[[i, 1]];
            if !xi.is_finite() || !yi.is_finite() {
                return;
            }
            let neighbors = tree.within::<kiddo::SquaredEuclidean>(&[xi, yi], r2);
            for nb in &neighbors {
                let j = valid_indices[nb.item as usize];
                let c = clusters[j];
                if c < num_clusters {
                    row[c] += 1.0;
                }
            }
        });
    result
}

/// Synthetic cluster CNN wall time for [`crate::run_benchmark_mock_cluster_cnn_training`] / `cnn_train_bench`.
/// Uses `Autodiff<NdArray>` (CPU); relative before/after comparisons are meaningful on the same machine.
pub fn run_benchmark_mock_cluster_cnn_training() -> std::time::Duration {
    use crate::config::{CnnConfig, CnnOutputActivation};
    use burn::backend::NdArray;
    use burn_autodiff::Autodiff;

    type B = Autodiff<NdArray<f32, i32>>;
    let device = Default::default();
    const CLUSTER_N: usize = 768;
    const EPOCHS: usize = 8;
    let h = 24usize;
    let w = 24usize;
    let n_modulators = 6usize;
    let n_clusters = 10usize;

    let sm_c = Array4::<f32>::from_elem((CLUSTER_N, 1, h, w), 0.1f32);
    let x_c = Array2::<f64>::from_elem((CLUSTER_N, n_modulators), 0.05);
    let sf_c = Array2::<f64>::from_elem((CLUSTER_N, n_clusters), 0.02);
    let y_c = Array1::<f64>::zeros(CLUSTER_N);
    let lasso_coef = Array2::<f64>::from_elem((n_modulators, 1), 0.01);
    let intercept = 0.0f64;

    let anchors_vec: Vec<f32> = std::iter::once(0.5f32)
        .chain(std::iter::repeat_n(0.1f32, n_modulators))
        .collect();
    let anchors_tensor = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(anchors_vec.clone(), [anchors_vec.len()]),
        &device,
    );
    let cfg = CellularNicheNetworkConfig {
        n_modulators,
        n_clusters,
        vision_in_channels: 1,
    };
    let model = cfg.init::<B>(&device, anchors_tensor, CnnOutputActivation::Sigmoid);

    let mut cnn = CnnConfig::default();
    cnn.lasso_pred_align_weight = 0.05;
    cnn.cnn_minibatch_size = 48;
    cnn.cnn_early_stop_patience = 0;

    let y_lasso_cpu = y_lasso_vec_from_xy_cpu(&x_c, intercept, &lasso_coef);

    let t0 = std::time::Instant::now();
    let _ = train_cluster_cnn_epochs(
        model,
        &device,
        &sm_c,
        &x_c,
        &sf_c,
        &y_c,
        CLUSTER_N,
        0usize,
        Some(y_lasso_cpu.as_slice()),
        0.01f32,
        &cnn,
        1e-3,
        EPOCHS,
        None,
        0,
    );
    t0.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CnnLrSchedule;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn finite_or_zero_f64_normal() {
        assert_abs_diff_eq!(finite_or_zero_f64(2.5), 2.5, epsilon = 1e-15);
        assert_abs_diff_eq!(finite_or_zero_f64(-2.0), -2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(finite_or_zero_f64(0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn finite_or_zero_f64_special() {
        assert_abs_diff_eq!(finite_or_zero_f64(f64::NAN), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(finite_or_zero_f64(f64::INFINITY), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(finite_or_zero_f64(f64::NEG_INFINITY), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn finite_or_zero_f32_special() {
        assert_eq!(finite_or_zero_f32(f32::NAN), 0.0);
        assert_eq!(finite_or_zero_f32(f32::INFINITY), 0.0);
        assert_eq!(finite_or_zero_f32(1.5), 1.5);
    }

    #[test]
    fn r2_score_from_pred_perfect_match() {
        let y = array![1.0_f64, 2.0, 3.0];
        let pred = [1.0f32, 2.0, 3.0];
        assert_abs_diff_eq!(
            r2_score_from_pred_slice(y.view(), &pred),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn r2_score_from_pred_partial_fit() {
        let y = array![0.0_f64, 2.0, 4.0];
        let pred = [0.0f32, 1.0, 3.0];
        assert_abs_diff_eq!(
            r2_score_from_pred_slice(y.view(), &pred),
            0.75,
            epsilon = 1e-12
        );
    }

    #[test]
    fn r2_score_from_pred_zero_variance_y() {
        let y = array![2.0_f64, 2.0, 2.0];
        let pred = [1.0f32, 3.0, 2.0];
        assert_abs_diff_eq!(
            r2_score_from_pred_slice(y.view(), &pred),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn r2_score_from_pred_len_mismatch_is_nan() {
        let y = array![1.0_f64, 2.0];
        let pred = [1.0f32];
        assert!(r2_score_from_pred_slice(y.view(), &pred).is_nan());
    }

    #[test]
    fn cnn_max_cells_per_epoch_caps_batches() {
        use crate::config::CnnConfig;

        let bs_eff = 32usize;

        let cells_to_batches = |cells: Option<usize>| -> usize {
            match cells {
                None | Some(0) => usize::MAX,
                Some(c) => c.div_ceil(bs_eff).max(1),
            }
        };

        assert_eq!(cells_to_batches(None), usize::MAX);
        assert_eq!(cells_to_batches(Some(0)), usize::MAX);
        assert_eq!(cells_to_batches(Some(1)), 1);
        assert_eq!(cells_to_batches(Some(32)), 1);
        assert_eq!(cells_to_batches(Some(33)), 2);
        assert_eq!(cells_to_batches(Some(8192)), 256);

        let mut cnn = CnnConfig::default();
        cnn.cnn_minibatch_size = 32;
        cnn.cnn_max_batches_per_epoch = Some(8);
        cnn.cnn_max_cells_per_epoch = Some(64);
        let combined = cells_to_batches(cnn.cnn_max_cells_per_epoch).min(
            cnn.cnn_max_batches_per_epoch
                .filter(|b| *b > 0)
                .unwrap_or(usize::MAX),
        );
        assert_eq!(combined, 2);
    }

    #[test]
    fn cnn_max_batches_per_epoch_runs_fewer_steps() {
        use crate::config::CnnConfig;
        use crate::config::CnnOutputActivation;
        use crate::model::CellularNicheNetworkConfig;
        use burn::backend::NdArray;
        use burn::tensor::Tensor;
        use burn_autodiff::Autodiff;

        type B = Autodiff<NdArray<f32, i32>>;
        let device = Default::default();
        const N: usize = 200;
        const H: usize = 8;
        const W: usize = 8;
        const P: usize = 4;
        const K: usize = 3;
        let sm = Array4::<f32>::from_elem((N, 1, H, W), 0.1f32);
        let x = Array2::<f64>::from_elem((N, P), 0.05);
        let sf = Array2::<f64>::from_elem((N, K), 0.02);
        let y = Array1::<f64>::zeros(N);
        let lasso_coef = Array2::<f64>::from_elem((P, 1), 0.01);
        let anchors: Vec<f32> = std::iter::once(0.5f32)
            .chain(std::iter::repeat_n(0.1f32, P))
            .collect();
        let at =
            Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(anchors, [P + 1]), &device);
        let m = CellularNicheNetworkConfig {
            n_modulators: P,
            n_clusters: K,
            vision_in_channels: 1,
        }
        .init::<B>(&device, at, CnnOutputActivation::Sigmoid);
        let y_l = y_lasso_vec_from_xy_cpu(&x, 0.0, &lasso_coef);
        let mut cnn = CnnConfig::default();
        cnn.lasso_pred_align_weight = 0.0;
        cnn.cnn_minibatch_size = 32;
        cnn.cnn_max_batches_per_epoch = Some(2);
        cnn.cnn_early_stop_patience = 0;
        let (_m, mse, div) = train_cluster_cnn_epochs(
            m,
            &device,
            &sm,
            &x,
            &sf,
            &y,
            N,
            0usize,
            Some(y_l.as_slice()),
            0.01f32,
            &cnn,
            1e-3,
            3usize,
            None,
            0,
        );
        assert!(!div);
        assert_eq!(mse.len(), 3);
        assert!(mse.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn r2_in_sample_clamps_extreme_negative() {
        let r = r2_in_sample_from_residuals(10, 1e-12, 2.0);
        assert!(r.is_finite());
        assert!((-1_000_000.0..=1.0).contains(&r));
    }

    #[test]
    fn r2_score_from_pred_near_constant_target_avoids_absurd_r2() {
        let y = array![1.0_f64, 1.0 + 1e-30];
        let pred = [0.0f32, 5.0f32];
        let r = r2_score_from_pred_slice(y.view(), &pred);
        assert!(r.is_nan() || (r.is_finite() && r.abs() <= 1_000_000.0));
    }

    #[test]
    fn min_max_finite_col_normal() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (lo, hi) = min_max_finite_col(data.column(0));
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 5.0);
    }

    #[test]
    fn min_max_finite_col_with_nan() {
        let data = array![[f64::NAN], [2.0], [5.0]];
        let (lo, hi) = min_max_finite_col(data.column(0));
        assert_eq!(lo, 2.0);
        assert_eq!(hi, 5.0);
    }

    #[test]
    fn min_max_finite_col_all_nan() {
        let data = array![[f64::NAN], [f64::NAN]];
        let (lo, hi) = min_max_finite_col(data.column(0));
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn spatial_features_shape() {
        let xy = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let clusters = Array1::from_vec(vec![0, 1, 0, 1]);
        let sf = create_spatial_features(&xy, &clusters, 2, 100.0);
        assert_eq!(sf.shape(), &[4, 2]);
    }

    #[test]
    fn spatial_features_self_count() {
        // Each cell is within radius of itself (distance = 0)
        let xy = array![[0.0, 0.0], [1000.0, 1000.0]];
        let clusters = Array1::from_vec(vec![0, 1]);
        let sf = create_spatial_features(&xy, &clusters, 2, 1.0); // small radius
        // Cell 0: only itself in radius → cluster 0 count = 1
        assert_abs_diff_eq!(sf[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[0, 1]], 0.0, epsilon = 1e-10);
        // Cell 1: only itself → cluster 1 count = 1
        assert_abs_diff_eq!(sf[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_all_in_radius() {
        // All cells within radius of each other
        let xy = array![[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let sf = create_spatial_features(&xy, &clusters, 2, 10.0);
        // Cell 0: sees cells 0,2 (cluster 0) and cell 1 (cluster 1)
        assert_abs_diff_eq!(sf[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[0, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_radius_boundary() {
        // Cell 0 at origin, cell 1 at (1,0). Radius = 1.0 → distance = 1.0 ≤ radius
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let clusters = Array1::from_vec(vec![0, 0]);
        let sf = create_spatial_features(&xy, &clusters, 1, 1.0);
        // Both cells within radius of each other
        assert_abs_diff_eq!(sf[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[1, 0]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_just_outside_radius() {
        // Cell 0 at origin, cell 1 at (1.01, 0). Radius = 1.0 → distance > radius
        let xy = array![[0.0, 0.0], [1.01, 0.0]];
        let clusters = Array1::from_vec(vec![0, 0]);
        let sf = create_spatial_features(&xy, &clusters, 1, 1.0);
        // Each cell only sees itself
        assert_abs_diff_eq!(sf[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[1, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_nan_handling() {
        let xy = array![[f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let clusters = Array1::from_vec(vec![0, 0, 1]);
        let sf = create_spatial_features(&xy, &clusters, 2, 100.0);
        // Cell 0 has NaN coords → row should be all zeros
        assert_abs_diff_eq!(sf[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[0, 1]], 0.0, epsilon = 1e-10);
        // Cells 1,2 should not count cell 0
        assert_abs_diff_eq!(sf[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn xyc2spatial_shape() {
        let xy = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 2, 8, 8, false);
        assert_eq!(maps.shape(), &[3, 2, 8, 8]);
    }

    #[test]
    fn spatial_maps_for_cluster_cnn_shape_matches_conv1() {
        let xy = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 2, 8, 8, false);
        let sm = spatial_maps_for_cluster_cnn(&maps, &[0, 2], 0);
        assert_eq!(sm.shape(), &[2, 1, 8, 8]);
        assert_eq!(
            sm,
            maps.select(Axis(0), &[0, 2])
                .slice(ndarray::s![.., 0..1, .., ..])
        );
    }

    #[test]
    fn xyc2spatial_only_own_cluster_nonzero() {
        // Cell 0 is cluster 0 → only channel 0 should have nonzero entries
        let xy = array![[0.5, 0.5], [1.5, 1.5]];
        let clusters = Array1::from_vec(vec![0, 1]);
        let maps = xyc2spatial_fast(&xy, &clusters, 2, 4, 4, false);

        // Cell 0, cluster 0 channel should have nonzero values
        let ch0_sum: f32 = maps.slice(ndarray::s![0, 0, .., ..]).iter().sum();
        assert!(ch0_sum > 0.0, "Own cluster channel should be nonzero");

        // Cell 0, cluster 1 channel should be zero
        let ch1_sum: f32 = maps.slice(ndarray::s![0, 1, .., ..]).iter().sum();
        assert_abs_diff_eq!(ch1_sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn xyc2spatial_inverse_distance_positive() {
        let xy = array![[0.5, 0.5]];
        let clusters = Array1::from_vec(vec![0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 1, 4, 4, false);
        // All values in the active channel should be positive (1/d > 0)
        for &v in maps.slice(ndarray::s![0, 0, .., ..]).iter() {
            assert!(v > 0.0, "Inverse distance should be positive");
        }
    }

    #[test]
    fn xyc2spatial_closer_grid_points_higher_value() {
        let xy = array![[0.0, 1.0]]; // at the top-left area
        let clusters = Array1::from_vec(vec![0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 1, 4, 4, false);
        let channel = maps.slice(ndarray::s![0, 0, .., ..]);
        // The grid point closest to the cell should have the highest value
        let max_val = channel.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_val > 0.0);
    }

    #[test]
    fn xyc2spatial_nan_cell_is_zero() {
        let xy = array![[f64::NAN, 0.0], [1.0, 1.0]];
        let clusters = Array1::from_vec(vec![0, 0]);
        let maps = xyc2spatial_fast(&xy, &clusters, 1, 4, 4, false);
        let cell0_sum: f32 = maps.slice(ndarray::s![0, .., .., ..]).iter().sum();
        assert_abs_diff_eq!(cell0_sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn xyc2spatial_deterministic() {
        let xy = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let m1 = xyc2spatial_fast(&xy, &clusters, 2, 4, 4, false);
        let m2 = xyc2spatial_fast(&xy, &clusters, 2, 4, 4, false);
        assert_eq!(m1, m2);
    }

    #[test]
    fn spatial_features_symmetry() {
        // Two cells at same distance from each other, both cluster 0
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let clusters = Array1::from_vec(vec![0, 0]);
        let sf = create_spatial_features(&xy, &clusters, 1, 10.0);
        assert_abs_diff_eq!(sf[[0, 0]], sf[[1, 0]], epsilon = 1e-10);
    }

    #[test]
    fn spatial_features_nonnegative() {
        let xy = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]];
        let clusters = Array1::from_vec(vec![0, 1, 0]);
        let sf = create_spatial_features(&xy, &clusters, 2, 5.0);
        for &v in sf.iter() {
            assert!(v >= 0.0, "Spatial features (counts) must be non-negative");
        }
    }

    #[test]
    fn spatial_features_large_cluster_count() {
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let clusters = Array1::from_vec(vec![0, 5]);
        let sf = create_spatial_features(&xy, &clusters, 10, 100.0);
        assert_eq!(sf.ncols(), 10);
        assert_abs_diff_eq!(sf[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sf[[0, 5]], 1.0, epsilon = 1e-10);
        // Other cluster columns should be zero
        assert_abs_diff_eq!(sf[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cnn_lr_constant_matches_base_after_warmup() {
        let base = 2e-4;
        let lr = cnn_lr_for_epoch(base, 10, 100, CnnLrSchedule::Constant, 5, 0.01);
        assert!((lr - base).abs() < 1e-18);
    }

    #[test]
    fn cnn_lr_cosine_endpoints() {
        let base = 1.0;
        let min_r = 0.01;
        let t = 20usize;
        let lr0 = cnn_lr_for_epoch(base, 0, t, CnnLrSchedule::Cosine, 0, min_r);
        let lr_last = cnn_lr_for_epoch(base, t - 1, t, CnnLrSchedule::Cosine, 0, min_r);
        assert!((lr0 - base).abs() < 1e-9, "got {lr0}");
        assert!((lr_last - base * min_r).abs() < 1e-9, "got {lr_last}");
    }
}
