use crate::model::{CellularNicheNetwork, CellularNicheNetworkConfig};
use crate::pack::{
    ClusterTrainResult, CnnClusterPack, CnnGeneTrainPack, CnnTrainHyperparams, GeneTrainResult,
};
use burn::backend::NdArray;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{ElementConversion, Tensor, backend::Backend};
use burn_autodiff::Autodiff;
use ndarray::{Array1, Array2, Array4};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

#[cfg(target_arch = "wasm32")]
use js_sys;

type TrainBackend = Autodiff<NdArray<f32, i32>>;

#[cfg(target_arch = "wasm32")]
fn now_ms() -> u64 {
    js_sys::Date::now() as u64
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn elapsed_ms(t0: u64) -> u32 {
    now_ms().saturating_sub(t0).min(u64::from(u32::MAX)) as u32
}

fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

fn cnn_lr_for_epoch(hp: &CnnTrainHyperparams, epoch: usize) -> f64 {
    let total = hp.epochs as usize;
    if total == 0 {
        return hp.learning_rate;
    }
    let warmup = (hp.lr_warmup_epochs as usize).min(total);
    if epoch < warmup {
        let t = (epoch + 1) as f64 / warmup as f64;
        return hp.learning_rate * t;
    }
    if !hp.lr_schedule_cosine {
        return hp.learning_rate;
    }
    let post = total.saturating_sub(warmup).max(1);
    let i = epoch.saturating_sub(warmup);
    let t = i as f64 / (post as f64 - 1.0).max(1.0);
    let min_lr = hp.learning_rate * hp.cosine_lr_min_ratio;
    min_lr + 0.5 * (hp.learning_rate - min_lr) * (1.0 + (std::f64::consts::PI * t).cos())
}

fn gather_rows_4(a: &Array4<f32>, idx: &[usize]) -> Array4<f32> {
    let c = a.shape()[1];
    let h = a.shape()[2];
    let w = a.shape()[3];
    let mut out = Array4::<f32>::zeros((idx.len(), c, h, w));
    for (oi, &i) in idx.iter().enumerate() {
        out.slice_mut(ndarray::s![oi, .., .., ..])
            .assign(&a.slice(ndarray::s![i, .., .., ..]));
    }
    out
}

fn gather_rows_2(a: &Array2<f32>, idx: &[usize]) -> Array2<f32> {
    let cols = a.ncols();
    let mut out = Array2::zeros((idx.len(), cols));
    for (oi, &i) in idx.iter().enumerate() {
        out.row_mut(oi).assign(&a.row(i));
    }
    out
}

fn gather_rows_1(a: &Array1<f32>, idx: &[usize]) -> Array1<f32> {
    Array1::from_iter(idx.iter().map(|&i| a[i]))
}

fn tensor4<B: Backend>(a: &Array4<f32>, device: &B::Device) -> Tensor<B, 4> {
    let shape = a.shape();
    let data: Vec<f32> = a.iter().copied().map(finite_or_zero).collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(data, [shape[0], shape[1], shape[2], shape[3]]),
        device,
    )
}

fn tensor2<B: Backend>(a: &Array2<f32>, device: &B::Device) -> Tensor<B, 2> {
    let data: Vec<f32> = a.iter().copied().map(finite_or_zero).collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(data, [a.nrows(), a.ncols()]),
        device,
    )
}

fn tensor1<B: Backend>(a: &Array1<f32>, device: &B::Device) -> Tensor<B, 1> {
    let data: Vec<f32> = a.iter().copied().map(finite_or_zero).collect();
    Tensor::from_data(burn::tensor::TensorData::new(data, [a.len()]), device)
}

fn train_one_cluster(pack: &CnnClusterPack, hp: &CnnTrainHyperparams) -> ClusterTrainResult {
    let t0 = now_ms();
    let device = Default::default();
    let n = pack.n_cells as usize;
    let n_mod = pack.n_modulators as usize;
    let n_clust = pack.n_clusters as usize;
    let h = pack.spatial_h as usize;
    let w = pack.spatial_w as usize;
    let ch = pack.vision_in_channels as usize;
    let epochs = hp.epochs as usize;

    let sm = Array4::from_shape_vec((n, ch, h, w), pack.spatial_maps.clone())
        .expect("spatial_maps shape");
    let x = Array2::from_shape_vec((n, n_mod), pack.x.clone()).expect("x shape");
    let sf = Array2::from_shape_vec((n, n_clust), pack.spatial_features.clone()).expect("sf shape");
    let y = Array1::from_vec(pack.y.clone());

    let anchors = Tensor::<TrainBackend, 1>::from_data(
        burn::tensor::TensorData::new(pack.anchors.clone(), [pack.anchors.len()]),
        &device,
    );
    let cfg = CellularNicheNetworkConfig {
        n_modulators: n_mod,
        n_clusters: n_clust,
        vision_in_channels: ch.max(1),
    };
    let mut model = cfg.init::<TrainBackend>(&device, anchors, hp.output_activation);

    let mut adam = AdamConfig::new()
        .with_beta_1(hp.adam_beta_1)
        .with_beta_2(hp.adam_beta_2)
        .with_epsilon(hp.adam_epsilon);
    if let Some(wd) = hp.weight_decay {
        adam = adam.with_weight_decay(Some(WeightDecayConfig::new(wd)));
    }
    if let Some(gc) = hp.grad_clip_norm {
        adam = adam.with_grad_clipping(Some(GradientClippingConfig::Norm(gc)));
    }
    let mut optim = adam.init::<TrainBackend, CellularNicheNetwork<TrainBackend>>();
    let mse_loss = burn::nn::loss::MseLoss::new();

    let bs_eff = if hp.cnn_minibatch_size == 0 {
        n
    } else {
        (hp.cnn_minibatch_size as usize).min(n).max(1)
    };

    let mut mse_epochs = Vec::with_capacity(epochs);
    let mut diverged = false;
    let patience = hp.cnn_early_stop_patience as usize;
    let min_ep = hp.cnn_early_stop_min_epochs as usize;
    let mut best_mse = f32::INFINITY;
    let mut no_improve = 0usize;
    let mut best_model: Option<CellularNicheNetwork<TrainBackend>> = None;

    for epoch in 0..epochs {
        let lr = cnn_lr_for_epoch(hp, epoch);
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(
            hp.shuffle_seed
                ^ 0x9E37_79B9_7F4A_7C15_u64
                ^ (pack.cluster_id as u64).wrapping_shl(32)
                ^ (epoch as u64),
        );
        order.shuffle(&mut rng);

        let max_batches_from_cells = match hp.cnn_max_cells_per_epoch {
            None | Some(0) => usize::MAX,
            Some(c) => (c as usize).div_ceil(bs_eff).max(1),
        };

        let mut epoch_mse_den = 0.0f32;
        let mut epoch_mse_acc = Tensor::<TrainBackend, 1>::zeros([1], &device);
        let mut batch_in_epoch = 0usize;
        let mut pos = 0usize;
        while pos < n && batch_in_epoch < max_batches_from_cells {
            let end = (pos + bs_eff).min(n);
            let batch_idx = &order[pos..end];
            pos = end;
            let batch_n = batch_idx.len();

            let sm_b = gather_rows_4(&sm, batch_idx);
            let x_b = gather_rows_2(&x, batch_idx);
            let sf_b = gather_rows_2(&sf, batch_idx);
            let y_b = gather_rows_1(&y, batch_idx);

            let sm_t = tensor4::<TrainBackend>(&sm_b, &device);
            let x_t = tensor2::<TrainBackend>(&x_b, &device);
            let sf_t = tensor2::<TrainBackend>(&sf_b, &device);
            let y_t = tensor1::<TrainBackend>(&y_b, &device);

            let betas = model.get_betas(sm_t, sf_t);
            let y_pred = CellularNicheNetwork::linear_readout_y(betas.clone(), x_t);
            let y_loss =
                mse_loss.forward(y_pred.clone(), y_t, burn::nn::loss::Reduction::Mean);
            let mut total = y_loss.clone();
            if hp.mean_beta_lasso_prior_weight > 0.0 {
                let mean_betas = betas.mean_dim(0);
                let prior = mse_loss.forward(
                    mean_betas,
                    model.anchors_row.clone(),
                    burn::nn::loss::Reduction::Mean,
                );
                total = total + prior.mul_scalar(hp.mean_beta_lasso_prior_weight);
            }
            if hp.lasso_pred_align_weight > 0.0 {
                if let Some(ref yl) = pack.y_lasso {
                    let yl_b: Vec<f32> = batch_idx.iter().map(|&i| yl[i]).collect();
                    let yl_t = Tensor::<TrainBackend, 1>::from_data(
                        burn::tensor::TensorData::new(yl_b, [batch_n]),
                        &device,
                    );
                    let align =
                        mse_loss.forward(y_pred, yl_t, burn::nn::loss::Reduction::Mean);
                    total = total + align.mul_scalar(hp.lasso_pred_align_weight);
                }
            }

            let w = batch_n as f32;
            let y_mse_d = y_loss.detach();
            epoch_mse_acc = epoch_mse_acc + y_mse_d.mul_scalar(w);
            epoch_mse_den += w;
            batch_in_epoch += 1;

            let grads = total.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);
        }

        let mse = if epoch_mse_den > 0.0 {
            let sum = epoch_mse_acc.into_scalar().elem::<f32>();
            finite_or_zero(sum / epoch_mse_den)
        } else {
            f32::NAN
        };
        if !mse.is_finite() {
            diverged = true;
            break;
        }
        mse_epochs.push(mse);
        if mse < best_mse {
            best_mse = mse;
            no_improve = 0;
            best_model = Some(model.clone());
        } else if epoch + 1 >= min_ep && patience > 0 {
            no_improve += 1;
            if no_improve >= patience {
                break;
            }
        }
    }

    if let Some(m) = best_model {
        let _ = m;
    }

    ClusterTrainResult {
        cluster_id: pack.cluster_id,
        n_cells: pack.n_cells,
        lasso_r2: pack.lasso_r2,
        mse_epochs,
        diverged,
        wall_ms: elapsed_ms(t0),
    }
}

pub fn train_gene_pack(pack: &CnnGeneTrainPack) -> GeneTrainResult {
    let t0 = now_ms();
    let mut clusters = Vec::with_capacity(pack.clusters.len());
    for c in &pack.clusters {
        clusters.push(train_one_cluster(c, &pack.hyperparams));
    }
    GeneTrainResult {
        gene: pack.gene.clone(),
        clusters,
        wall_ms: elapsed_ms(t0),
    }
}

pub fn train_gene_pack_bytes(bytes: &[u8]) -> Result<GeneTrainResult, String> {
    let pack = crate::pack::decode_pack(bytes).map_err(|e| e.to_string())?;
    Ok(train_gene_pack(&pack))
}
