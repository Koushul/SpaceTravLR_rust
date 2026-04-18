//! High-level orchestrator: from `(adata, betadata, repro.toml)` (or the
//! synthetic run) → niche labels.

use std::path::Path;

use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use ndarray::Array2;

use super::image::{NicheImageStack, StandardizeMode, build_niche_image_stack};
use super::kmeans::kmeans_lloyd;
use super::model::NicheEncoder;
use super::synth::SyntheticNicheRun;
use super::train::{NicheTrainConfig, NicheTrainOutputs, train_niche_encoder};
use crate::perturb_mode::PerturbRuntime;

/// All inputs needed by [`NicheRuntime::fit`].
pub struct NicheRuntimeBuilder {
    pub stack: NicheImageStack,
    pub xy: Array2<f64>,
    pub obs_names: Vec<String>,
    pub niche_gt: Option<Vec<usize>>,
}

impl NicheRuntimeBuilder {
    /// Build a runtime from a [`SyntheticNicheRun`].
    pub fn from_synthetic(run: SyntheticNicheRun, standardize: StandardizeMode) -> Self {
        let stack = build_niche_image_stack(&run.splash, run.n_cells, standardize);
        let obs_names: Vec<String> = (0..run.n_cells).map(|i| format!("cell_{:05}", i)).collect();
        Self {
            stack,
            xy: run.xy,
            obs_names,
            niche_gt: Some(run.niche_gt),
        }
    }

    /// Build a runtime by running splash on a [`PerturbRuntime`].
    pub fn from_perturb_runtime(rt: &PerturbRuntime, standardize: StandardizeMode) -> Self {
        let splash = crate::perturb::compute_splash_all(
            &rt.bb,
            &rt.rw_ligands_init,
            &rt.rw_tfligands_init,
            &gex_gm_from_rt(rt),
            rt.perturb_cfg.beta_scale_factor as f32,
            rt.perturb_cfg.beta_cap.map(|v| v as f32),
        );
        let n_cells = rt.gene_mtx.nrows();
        let stack = build_niche_image_stack(&splash, n_cells, standardize);
        let xy = rt.xy.clone();
        let obs_names = rt.obs_names.clone();
        Self {
            stack,
            xy,
            obs_names,
            niche_gt: None,
        }
    }

    /// Convenience: load a `spacetravlr_run_repro.toml` then call
    /// [`from_perturb_runtime`].
    pub fn from_run_toml(run_toml: &Path, standardize: StandardizeMode) -> Result<Self> {
        let rt = PerturbRuntime::from_run_toml(run_toml)?;
        Ok(Self::from_perturb_runtime(&rt, standardize))
    }
}

fn gex_gm_from_rt(rt: &PerturbRuntime) -> crate::betadata::GeneMatrix {
    let n_cells = rt.gene_mtx.nrows();
    let n_genes = rt.gene_mtx.ncols();
    let mut out = Array2::<f32>::zeros((n_cells, n_genes));
    for c in 0..n_cells {
        for g in 0..n_genes {
            let v = rt.gene_mtx[[c, g]];
            out[[c, g]] = if v > 0.0 { v as f32 } else { 0.0 };
        }
    }
    crate::betadata::GeneMatrix::new(out, rt.gene_names.clone())
}

/// One trained niche detector + its outputs.
pub struct NicheRuntimeOutputs<B: AutodiffBackend> {
    pub embeddings: Array2<f32>,
    pub labels: Vec<usize>,
    pub n_clusters: usize,
    pub stack: NicheImageStack,
    pub xy: Array2<f64>,
    pub obs_names: Vec<String>,
    pub niche_gt: Option<Vec<usize>>,
    pub train: NicheTrainOutputs<B>,
}

/// Wraps `train_niche_encoder` + a final k-means on embeddings.
pub struct NicheRuntime;

impl NicheRuntime {
    pub fn fit<B: AutodiffBackend>(
        device: &B::Device,
        builder: NicheRuntimeBuilder,
        train_cfg: &NicheTrainConfig,
        n_clusters: usize,
    ) -> NicheRuntimeOutputs<B> {
        let NicheRuntimeBuilder {
            stack,
            xy,
            obs_names,
            niche_gt,
        } = builder;

        let coords: Vec<[f64; 2]> = (0..stack.n_cells).map(|i| [xy[[i, 0]], xy[[i, 1]]]).collect();
        let train_out = train_niche_encoder::<B>(device, &stack, &coords, train_cfg);
        let n = stack.n_cells;
        let dim = train_out.embeddings.shape()[1];
        let flat: Vec<f32> = train_out.embeddings.iter().copied().collect();
        let km = kmeans_lloyd(&flat, n, dim, n_clusters, 200, train_cfg.seed);
        NicheRuntimeOutputs {
            embeddings: train_out.embeddings.clone(),
            labels: km.labels,
            n_clusters,
            stack,
            xy,
            obs_names,
            niche_gt,
            train: train_out,
        }
    }
}

impl<B: AutodiffBackend> NicheRuntimeOutputs<B> {
    /// Helper: fetch the embedding row for a cell as a slice.
    pub fn embedding_row(&self, i: usize) -> Vec<f32> {
        self.embeddings.row(i).to_vec()
    }
    /// Encoder that consumed the splash image stack — useful for downstream
    /// inference on held-out cells.
    pub fn model(&self) -> &NicheEncoder<B> {
        &self.train.model
    }
}
