//! Convert SpaceTravLR `{gene}_cnn_train_data.npz` dumps into WASM train packs.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, bail};
use ndarray::{ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::{NpzReader, ReadNpzError};
use spacetravlr_cnn_wasm::{CnnClusterPack, CnnGeneTrainPack, CnnTrainHyperparams};

fn npz_err(name: &str) -> impl Fn(ReadNpzError) -> anyhow::Error + '_ {
    move |e| anyhow::anyhow!("npz key `{name}`: {e}")
}

fn read_meta_u32(npz: &mut NpzReader<File>, name: &str) -> anyhow::Result<u32> {
    if let Ok(a) = npz.by_name::<OwnedRepr<u32>, IxDyn>(name) {
        return Ok(a.iter().copied().next().unwrap_or(0));
    }
    if let Ok(a) = npz.by_name::<OwnedRepr<i32>, IxDyn>(name) {
        return Ok(a.iter().map(|&v| v as u32).next().unwrap_or(0));
    }
    let a: ArrayD<f64> = npz.by_name(name).map_err(npz_err(name))?;
    Ok(a.iter().map(|&v| v as u32).next().unwrap_or(0))
}

fn read_meta_f64(npz: &mut NpzReader<File>, name: &str) -> anyhow::Result<f64> {
    let a: ArrayD<f64> = npz.by_name(name).map_err(npz_err(name))?;
    Ok(a.iter().copied().next().unwrap_or(0.0))
}

fn read_i32_vec(npz: &mut NpzReader<File>, name: &str) -> anyhow::Result<Vec<u32>> {
    if let Ok(a) = npz.by_name::<OwnedRepr<i32>, IxDyn>(name) {
        return Ok(a.iter().map(|&v| v as u32).collect());
    }
    if let Ok(a) = npz.by_name::<OwnedRepr<u32>, IxDyn>(name) {
        return Ok(a.iter().copied().collect());
    }
    let a: ArrayD<f64> = npz.by_name(name).map_err(npz_err(name))?;
    Ok(a.iter().map(|&v| v as u32).collect())
}

fn read_f32_dyn(npz: &mut NpzReader<File>, name: &str) -> anyhow::Result<ArrayD<f32>> {
    if let Ok(a) = npz.by_name::<OwnedRepr<f32>, IxDyn>(name) {
        return Ok(a);
    }
    let a: ArrayD<f64> = npz.by_name(name).map_err(npz_err(name))?;
    Ok(a.mapv(|v| v as f32))
}

pub fn cnn_gene_train_pack_from_npz(
    npz_path: &Path,
    gene: &str,
    epochs_override: Option<u32>,
) -> anyhow::Result<CnnGeneTrainPack> {
    let f = File::open(npz_path).with_context(|| format!("open {}", npz_path.display()))?;
    let mut npz =
        NpzReader::new(f).with_context(|| format!("npz reader {}", npz_path.display()))?;

    let cluster_ids = read_i32_vec(&mut npz, "cluster_ids")?;
    let spatial_dim = read_meta_u32(&mut npz, "meta_spatial_dim").unwrap_or(8);
    let n_clusters = read_meta_u32(&mut npz, "meta_n_clusters").unwrap_or(1);
    let n_modulators = read_meta_u32(&mut npz, "meta_n_modulators").unwrap_or(0);
    let output_activation = read_meta_u32(&mut npz, "meta_cnn_output_activation").unwrap_or(3) as u8;
    let epochs = epochs_override.unwrap_or_else(|| read_meta_u32(&mut npz, "meta_epochs").unwrap_or(8));
    let minibatch = read_meta_u32(&mut npz, "meta_cnn_minibatch_size").unwrap_or(64);
    let learning_rate = read_meta_f64(&mut npz, "meta_learning_rate").unwrap_or(1e-3);
    let adam_beta_1 = read_meta_f64(&mut npz, "meta_adam_beta_1").unwrap_or(0.9) as f32;
    let adam_beta_2 = read_meta_f64(&mut npz, "meta_adam_beta_2").unwrap_or(0.999) as f32;
    let adam_epsilon = read_meta_f64(&mut npz, "meta_adam_epsilon").unwrap_or(1e-5) as f32;
    let mean_beta_prior =
        read_meta_f64(&mut npz, "meta_mean_beta_lasso_prior_weight").unwrap_or(0.005) as f32;
    let weight_decay = read_meta_f64(&mut npz, "meta_weight_decay")
        .ok()
        .filter(|v| v.is_finite())
        .map(|v| v as f32);
    let grad_clip = read_meta_f64(&mut npz, "meta_grad_clip_norm")
        .ok()
        .filter(|v| v.is_finite())
        .map(|v| v as f32);

    let names: BTreeMap<String, ()> = npz
        .names()
        .map_err(|e| anyhow::anyhow!("npz names: {e}"))?
        .into_iter()
        .map(|s| (s, ()))
        .collect();

    let mut clusters = Vec::new();
    for &c_id in &cluster_ids {
        let prefix = format!("c{c_id:04}_");
        let x_key = format!("{prefix}x_scaled");
        if !names.contains_key(&x_key) {
            continue;
        }
        let x = read_f32_dyn(&mut npz, &x_key)?;
        let y = read_f32_dyn(&mut npz, &format!("{prefix}y"))?;
        let sf = read_f32_dyn(&mut npz, &format!("{prefix}spatial_features"))?;
        let sm = read_f32_dyn(&mut npz, &format!("{prefix}spatial_maps_used"))?;
        let anchors = read_f32_dyn(&mut npz, &format!("{prefix}anchors_init"))?;

        let n_cells = x.shape()[0] as u32;
        let n_mod = if x.ndim() > 1 {
            x.shape()[1] as u32
        } else {
            n_modulators
        };
        let sm_shape = sm.shape();
        let (ch, h, w) = match sm_shape {
            [_, c, hh, ww] => (*c as u32, *hh as u32, *ww as u32),
            [_, hh, ww] => (1, *hh as u32, *ww as u32),
            _ => (1, spatial_dim, spatial_dim),
        };

        clusters.push(CnnClusterPack {
            cluster_id: c_id,
            n_cells,
            n_modulators: n_mod,
            n_clusters,
            spatial_h: h,
            spatial_w: w,
            vision_in_channels: ch,
            spatial_maps: sm.iter().copied().collect(),
            x: x.iter().copied().collect(),
            spatial_features: sf.iter().copied().collect(),
            y: y.iter().copied().collect(),
            anchors: anchors.iter().copied().collect(),
            y_lasso: None,
            lasso_r2: f32::NAN,
        });
    }

    if clusters.is_empty() {
        bail!(
            "no cluster arrays in {} (cluster_ids={:?})",
            npz_path.display(),
            cluster_ids
        );
    }

    Ok(CnnGeneTrainPack {
        gene: gene.to_string(),
        hyperparams: CnnTrainHyperparams {
            learning_rate,
            epochs,
            adam_beta_1,
            adam_beta_2,
            adam_epsilon,
            weight_decay,
            grad_clip_norm: grad_clip,
            mean_beta_lasso_prior_weight: mean_beta_prior,
            lasso_pred_align_weight: 0.0,
            cnn_minibatch_size: minibatch.min(128).max(16),
            cnn_max_cells_per_epoch: Some(512),
            cnn_early_stop_patience: 0,
            cnn_early_stop_min_epochs: 0,
            lr_schedule_cosine: true,
            cosine_lr_min_ratio: 0.01,
            lr_warmup_epochs: 0,
            output_activation,
            shuffle_seed: 42,
        },
        clusters,
    })
}
