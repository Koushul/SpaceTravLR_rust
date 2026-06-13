use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const SPACESHIP_MERGE_SECTIONS: &[&str] = &[
    "data",
    "preprocess",
    "spatial",
    "grn",
    "cnn",
    "lasso",
    "training",
    "execution",
    "perturbation",
    "model_export",
];

fn merge_toml_table_maps(
    base: &mut toml::map::Map<String, toml::Value>,
    overlay: &toml::map::Map<String, toml::Value>,
) {
    for (k, v) in overlay {
        match (base.get_mut(k), v) {
            (Some(toml::Value::Table(base_sub)), toml::Value::Table(ov_sub)) => {
                merge_toml_table_maps(base_sub, ov_sub);
            }
            _ => {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

fn merge_toml_table_underlay_maps(
    base: &mut toml::map::Map<String, toml::Value>,
    fill: &toml::map::Map<String, toml::Value>,
) {
    for (k, v) in fill {
        match base.get_mut(k) {
            None => {
                base.insert(k.clone(), v.clone());
            }
            Some(base_val) => {
                if let (Some(bt), Some(ft)) = (base_val.as_table_mut(), v.as_table()) {
                    merge_toml_table_underlay_maps(bt, ft);
                }
            }
        }
    }
}

/// `GrnConfig` deserializes `max_lr` / `max_lr_pairs` as aliases of `max_ligands`. After underlay
/// merge, both `max_ligands` and `max_lr` can appear (repro vs repo spaceship); serde rejects
/// duplicate fields. Prefer the value already in the repro (`max_ligands`).
fn dedupe_grn_max_ligand_toml_keys_after_underlay(grn: &mut toml::map::Map<String, toml::Value>) {
    if grn.contains_key("max_ligands") && grn.contains_key("max_lr") {
        grn.remove("max_lr");
    }
    if grn.contains_key("max_ligands") && grn.contains_key("max_lr_pairs") {
        grn.remove("max_lr_pairs");
    }
}

/// After `--config` / overlay merge: alias keys from the overlay should win over `max_ligands`
/// when more than one of the group is present; otherwise normalize to a single `max_ligands` key.
fn dedupe_grn_max_ligand_toml_keys_after_overlay(grn: &mut toml::map::Map<String, toml::Value>) {
    let ml = grn.remove("max_ligands");
    let lr = grn.remove("max_lr");
    let pairs = grn.remove("max_lr_pairs");
    let resolved = if lr.is_some() || pairs.is_some() {
        lr.or(pairs).or(ml)
    } else {
        ml
    };
    if let Some(v) = resolved {
        grn.insert("max_ligands".to_string(), v);
    }
}

/// Fills missing keys in `into` from `underlay_root` for `[data]`, `[spatial]`, … only where
/// `into` has no value yet (including nested tables). Existing values in `into` are never replaced.
pub fn merge_spaceship_underlay_into_toml(into: &mut toml::Value, underlay_root: &toml::Value) {
    let Some(into_t) = into.as_table_mut() else {
        return;
    };
    let Some(ul_t) = underlay_root.as_table() else {
        return;
    };
    for &sec in SPACESHIP_MERGE_SECTIONS {
        let Some(ul_sec) = ul_t.get(sec).and_then(|x| x.as_table()) else {
            continue;
        };
        let entry = into_t
            .entry(sec.to_string())
            .or_insert(toml::Value::Table(Default::default()));
        if let Some(bt) = entry.as_table_mut() {
            merge_toml_table_underlay_maps(bt, ul_sec);
        }
    }
    if let Some(grn) = into_t.get_mut("grn").and_then(|v| v.as_table_mut()) {
        dedupe_grn_max_ligand_toml_keys_after_underlay(grn);
    }
}

/// Mix a global per-run seed with a UTF-8 tag (e.g. target gene) using FNV-1a 64-bit.
/// Stable across processes (no `std` hasher randomization).
pub fn mix_execution_random_seed(base: u64, tag: &str) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut h = base ^ OFFSET;
    for &b in tag.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Merges `[data]`, `[spatial]`, … from `overlay_root` into a TOML document that will deserialize
/// as [`SpaceshipConfig`]. Unknown top-level keys in `overlay_root` are ignored.
pub fn merge_spaceship_overlay_into_toml(into: &mut toml::Value, overlay_root: &toml::Value) {
    let Some(into_t) = into.as_table_mut() else {
        return;
    };
    let Some(ov_t) = overlay_root.as_table() else {
        return;
    };
    for &sec in SPACESHIP_MERGE_SECTIONS {
        if let Some(ov_sec) = ov_t.get(sec).and_then(|x| x.as_table()) {
            let entry = into_t
                .entry(sec.to_string())
                .or_insert(toml::Value::Table(Default::default()));
            if let Some(bt) = entry.as_table_mut() {
                merge_toml_table_maps(bt, ov_sec);
            } else {
                *entry = toml::Value::Table(ov_sec.clone());
            }
        }
    }
    if let Some(grn) = into_t.get_mut("grn").and_then(|v| v.as_table_mut()) {
        dedupe_grn_max_ligand_toml_keys_after_overlay(grn);
    }
}

/// Canonical per-run TOML in the training output directory (full `SpaceshipConfig` as executed).
pub const RUN_REPRO_TOML_FILENAME: &str = "spacetravlr_run_repro.toml";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpaceshipConfig {
    #[serde(default)]
    pub data: DataConfig,
    #[serde(default)]
    pub preprocess: PreprocessConfig,
    #[serde(default)]
    pub spatial: SpatialConfig,
    #[serde(default)]
    pub grn: GrnConfig,
    #[serde(default)]
    pub cnn: CnnConfig,
    #[serde(default)]
    pub lasso: LassoConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    #[serde(default)]
    pub execution: ExecutionConfig,
    #[serde(default)]
    pub perturbation: PerturbationConfig,
    #[serde(default)]
    pub model_export: ModelExportConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataConfig {
    pub adata_path: String,
    pub layer: String,
    pub cluster_annot: String,
    pub condition: Option<String>,
    /// Optional path (tilde-expanded like `adata_path`): one AnnData `obs_names` value per line (`#` comments, blanks skipped).
    /// When set, perturbation loads only these rows (expression, spatial, betadata alignment). Results apply to this ROI only.
    pub perturb_obs_subset_file: Option<String>,
    /// Prior for heuristic `obsm['spatial']` → micron scaling during full Scanpy preprocess (`human` or `mouse`).
    /// Empty: infer from `var` gene symbol capitalization / Ensembl IDs (see [`crate::network::infer_species`]).
    #[serde(default = "default_data_spatial_species")]
    pub spatial_species: String,
    /// Override median k-NN target distance (µm) for spatial scaling; omit to use the species default.
    pub spatial_median_nn_target_um: Option<f64>,
}

/// Rust / training auto-prep pipeline ([`crate::rust_preprocess`]): QC, normalize, HVG, PCA, UMAP, Leiden, MAGIC.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PreprocessConfig {
    /// `sc.pp.filter_cells(min_genes=…)` — minimum detected genes per cell.
    pub min_genes: u32,
    /// `sc.pp.filter_genes(min_cells=…)` — minimum cells expressing each gene.
    pub min_cells: u32,
    /// `sc.pp.normalize_total(target_sum=…)` when `X` is raw counts.
    pub normalize_target_sum: u32,
    /// Max highly-variable genes when `n_vars` exceeds this (dispersion ranking skipped below).
    pub n_top_hvg: usize,
    pub n_pca_components: usize,
    /// RNG seed for randomized PCA (`single_rust` SVD).
    pub pca_random_seed: u32,
    /// UMAP / fuzzy graph KNN count.
    pub n_neighbors: usize,
    pub min_dist: f32,
    pub spread: f32,
    pub umap_learning_rate: f32,
    /// UMAP SGD epochs; omit for umap-rs default (data-size dependent).
    pub n_epochs: Option<usize>,
    /// HNSW `ef_construction` for PCA-space KNN.
    pub ef_construction: usize,
    /// Leiden resolution (`sc.tl.leiden` analogue).
    pub leiden_resolution: f64,
    pub leiden_max_iter: usize,
    /// MAGIC diffusion time `t` (Rust `magic-impute` path).
    pub magic_t: u32,
}

impl PreprocessConfig {
    pub fn to_rust_preprocess_params(&self) -> crate::rust_preprocess::RustPreprocessParams {
        crate::rust_preprocess::RustPreprocessParams {
            min_genes: self.min_genes,
            min_cells: self.min_cells,
            normalize_target_sum: self.normalize_target_sum,
            n_top_hvg: self.n_top_hvg,
            n_pca_components: self.n_pca_components,
            pca_random_seed: self.pca_random_seed,
            n_neighbors: self.n_neighbors,
            min_dist: self.min_dist,
            n_epochs: self.n_epochs,
            ef_construction: self.ef_construction,
            spread: self.spread,
            umap_learning_rate: self.umap_learning_rate,
            leiden_resolution: self.leiden_resolution,
            leiden_max_iter: self.leiden_max_iter,
            magic_t: self.magic_t,
        }
    }
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            min_genes: 100,
            min_cells: 3,
            normalize_target_sum: 10_000,
            n_top_hvg: 2000,
            n_pca_components: 50,
            pca_random_seed: 42,
            n_neighbors: 15,
            min_dist: 0.5,
            spread: 0.5,
            umap_learning_rate: 1.0,
            n_epochs: None,
            ef_construction: 30,
            leiden_resolution: 1.0,
            leiden_max_iter: 100,
            magic_t: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpatialConfig {
    pub radius: f64,
    pub spatial_dim: usize,
    pub contact_distance: f64,
    /// Multiplier on Gaussian kernel weights in received-ligand aggregation (`calculate_weighted_ligands`).
    #[serde(default = "default_one_f64")]
    pub weighted_ligand_scale_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GrnConfig {
    /// Directory containing `{mouse|human}_network.parquet`. Overrides `SPACETRAVLR_DATA_DIR` and
    /// built-in search (manifest / cwd walk). Tilde and `~/` expanded like `data.adata_path`.
    pub network_data_dir: Option<String>,
    /// Optional Feather/IPC file containing TF priors with columns:
    /// `source` (TF), `target` (gene), `cell_type` (obs.cell_type label).
    /// When omitted and `use_tf_modulators` is true, training runs CellOracle GRN inference and writes
    /// `{output_dir}/celloracle_tf_priors.feather` (reused on subsequent runs if present).
    pub tf_priors_feather: Option<String>,
    pub tf_ligand_cutoff: f64,
    /// Max p-value for CellOracle Bayesian ridge edges to enter the TF priors feather.
    /// Edges with `p > celloracle_p_max` are dropped before writing `celloracle_tf_priors.feather`.
    /// Default **0.05**. Only used during auto inference (not when loading a user-supplied feather).
    pub celloracle_p_max: f64,
    /// Keep only DB L–R pairs whose **ligand** is among the top `max_ligands` by mean expression
    /// (uses `[data].layer`, e.g. `imputed_count`). Requires per-gene mean map when training.
    /// Deserialize accepts TOML keys `max_lr` (matches CLI `--max-lr`) and legacy `max_lr_pairs`.
    #[serde(alias = "max_lr_pairs", alias = "max_lr")]
    pub max_ligands: Option<usize>,
    #[serde(default = "default_true")]
    pub use_tf_modulators: bool,
    #[serde(default = "default_true")]
    pub use_lr_modulators: bool,
    #[serde(default = "default_true")]
    pub use_tfl_modulators: bool,
    /// When set, replaces `use_tf_modulators`, `use_lr_modulators`, and `use_tfl_modulators` after
    /// load: comma-separated tokens `tf`, `lr`, `tfl` (alias `ltf`). Omitted means use those three
    /// booleans as written. Not written back to repro TOML (effective flags are serialized).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_modulators: Option<String>,
    /// Genes to add as raw-expression modulators (fourth Lasso group). Excludes target and any gene
    /// already used as TF / LR / TFL (see `resolve_extra_modulators_and_lr`).
    #[serde(default)]
    pub extra_modulators: Vec<String>,
    /// Optional file: one gene per line or comma-separated; `#` comments. Appended to `extra_modulators`.
    pub extra_modulators_file: Option<String>,
    /// Extra L–R pairs as `LIG$REC` strings (or `LIG,REC` per element). Merged after database LR selection.
    #[serde(default)]
    pub extra_lr: Vec<String>,
    /// Optional file: one pair per line (`LIG$REC` or `LIG,REC`); `#` comments.
    pub extra_lr_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LassoConfig {
    pub l1_reg: f64,
    pub group_reg: f64,
    pub n_iter: usize,
    pub tol: f64,
    #[serde(default = "default_true")]
    pub scale_modulators: bool,
    #[serde(default = "default_true")]
    pub unscale_betas_on_export: bool,
    /// When **true**, fit in-sample group Lasso **per cluster in parallel** (Rayon, up to a small fixed pool).
    /// Default **false** (sequential per cluster; does not affect gene-level `--parallel` workers).
    #[serde(default = "default_false")]
    pub parallel_lasso_clusters: bool,
    /// Group Lasso solve path: `Some(true)` = Gram-matrix FISTA; `Some(false)` = residual (full-data) gradients;
    /// `None` = auto (Gram when `n_rows > n_cols` on augmented design including intercept).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gram_override: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CnnTrainingMode {
    #[serde(alias = "minimal", alias = "seed-only")]
    #[default]
    Seed,
    Full,
}

fn default_true() -> bool {
    true
}

fn default_one_f64() -> f64 {
    1.0
}

fn default_training_mode_option() -> Option<CnnTrainingMode> {
    Some(CnnTrainingMode::Seed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// Kept for CLI/back-compat; full CNN still runs Lasso first per gene.
    pub seed_only: bool,
    #[serde(default = "default_training_mode_option")]
    pub mode: Option<CnnTrainingMode>,
    pub epochs: usize,
    pub learning_rate: f64,
    pub score_threshold: f64,
    /// Subset of AnnData `var` to train (`--genes`, `[training] genes`). Persisted in
    /// `spacetravlr_run_repro.toml` for `--join-output-dir`.
    #[serde(default)]
    pub genes: Option<Vec<String>>,
    /// Cap after the `genes` filter (`--max-genes`, `[training] max_genes`). Persisted for join.
    #[serde(default)]
    pub max_genes: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CnnOutputActivation {
    Identity,
    Sigmoid,
    Tanh,
    /// `2 * sigmoid(x)`, output in (0, 2) before anchor scaling (plain sigmoid is (0, 1)).
    #[default]
    SigmoidX2,
}

/// Learning-rate schedule for full-batch Adam CNN steps (per epoch).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CnnLrSchedule {
    /// Fixed `[training].learning_rate` every epoch (after warmup, if any).
    #[default]
    Constant,
    /// Cosine decay from `learning_rate` down to `learning_rate * cosine_lr_min_ratio` over post-warmup epochs.
    Cosine,
}

fn default_mean_beta_lasso_prior_weight() -> f64 {
    0.005
}

fn default_grad_clip_norm() -> Option<f64> {
    Some(3.0)
}

fn default_cosine_lr_min_ratio() -> f64 {
    0.01
}

fn default_early_stop_patience() -> usize {
    8
}

fn default_early_stop_min_epochs() -> usize {
    12
}

fn default_drop_cnn_if_insample_worse_than_lasso() -> bool {
    true
}

fn default_cnn_minibatch_size() -> usize {
    512
}

fn default_cnn_inference_batch_size() -> usize {
    512
}

fn default_cnn_max_cells_per_epoch() -> Option<usize> {
    Some(3000)
}

fn default_false() -> bool {
    false
}

fn default_execution_random_seed() -> u64 {
    42
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CnnConfig {
    pub adam_beta_1: f64,
    pub adam_beta_2: f64,
    pub adam_epsilon: f64,
    pub weight_decay: Option<f64>,
    #[serde(default = "default_grad_clip_norm")]
    pub grad_clip_norm: Option<f64>,
    pub spatial_feature_radius: f64,
    /// Applied after the final head linear (`mlp.l2`), before multiplying by lasso anchors.
    pub output_activation: CnnOutputActivation,
    /// When > 0, add this × MSE(`mean_batch(get_betas)`, lasso anchors) to the CNN loss (weak prior on mean effective coefficients).
    #[serde(default = "default_mean_beta_lasso_prior_weight")]
    pub mean_beta_lasso_prior_weight: f64,
    #[serde(default)]
    pub lr_schedule: CnnLrSchedule,
    #[serde(default = "default_cosine_lr_min_ratio")]
    /// Cosine floor: effective minimum LR is `learning_rate * cosine_lr_min_ratio` (only used with `lr_schedule = "cosine"`).
    pub cosine_lr_min_ratio: f64,
    /// Linear LR warmup: ramp from 0 to `learning_rate` over this many epochs (0 = off).
    pub lr_warmup_epochs: usize,
    #[serde(default = "default_early_stop_patience")]
    /// Stop early if train MSE does not improve for this many epochs (0 = disabled).
    pub cnn_early_stop_patience: usize,
    #[serde(default = "default_early_stop_min_epochs")]
    /// Minimum epochs before early stopping can trigger.
    pub cnn_early_stop_min_epochs: usize,
    /// Optional MSE(`y_pred`, `y_lasso`) weight; use with `lasso_pred_align_linear_decay` to avoid washing out spatial signal.
    pub lasso_pred_align_weight: f64,
    #[serde(default)]
    pub lasso_pred_align_linear_decay: bool,
    #[serde(default = "default_drop_cnn_if_insample_worse_than_lasso")]
    /// After training, remove CNN if in-sample `cnn_r2 + margin < lasso_r2` so exports fall back to Lasso.
    pub drop_cnn_if_insample_worse_than_lasso: bool,
    /// Margin added to `cnn_r2` when comparing to `lasso_r2` for [`Self::drop_cnn_if_insample_worse_than_lasso`].
    pub cnn_vs_lasso_arbitration_margin: f64,
    /// Per-optimizer-step batch size over cells within a cluster. `0` = full batch (one step per epoch).
    #[serde(default = "default_cnn_minibatch_size")]
    pub cnn_minibatch_size: usize,
    /// Cells per forward when exporting per-cell CNN betas and when computing post-training in-sample CNN R².
    /// Does not apply to training steps ([`Self::cnn_minibatch_size`]). `0` = full cluster in one forward.
    #[serde(default = "default_cnn_inference_batch_size")]
    pub cnn_inference_batch_size: usize,
    /// Maximum optimizer steps per epoch (`None` = sweep all cells each epoch).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cnn_max_batches_per_epoch: Option<usize>,
    /// Per-epoch **cell limit** for smart subsampling. When `cluster_n` exceeds this
    /// value, each epoch trains on a uniform random subset of `cnn_max_cells_per_epoch`
    /// cells (`ceil(cnn_max_cells_per_epoch / cnn_minibatch_size)` optimizer steps);
    /// otherwise the full cluster is used (the inner loop terminates naturally).
    /// Per-epoch reshuffle makes coverage uniform in expectation across epochs.
    /// Combined with [`Self::cnn_max_batches_per_epoch`] via `min`. `None` = no cell cap
    /// (legacy full-sweep behavior). Default `Some(3000)` — tune in `[cnn]` config.
    #[serde(default = "default_cnn_max_cells_per_epoch")]
    pub cnn_max_cells_per_epoch: Option<usize>,
    /// When > 0, skip CNN training for clusters with fewer than this many cells (Lasso is still fit
    /// when it passes the score threshold; exports use Lasso for those clusters). `0` = no minimum.
    #[serde(default)]
    pub min_cells_for_cnn: usize,
    /// When true, feed all cluster channels `[batch, n_clusters, H, W]` into conv1 (cross-cell-type layout).
    /// When false (default), only the focal cluster channel `[batch, 1, H, W]` (matches legacy behavior).
    #[serde(default = "default_false")]
    pub multi_channel_spatial_maps: bool,
    /// When true, each cell's inverse-distance grid is centered on that cell (translation-invariant local niche).
    #[serde(default = "default_false")]
    pub ego_center_spatial_maps: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionConfig {
    pub n_parallel: usize,
    pub output_dir: String,
    /// When true, write `spacetravlr_minimal_repro.h5ad` under the output directory (large I/O on big datasets).
    pub write_minimal_repro_h5ad: bool,
    /// If > 0, remove a gene `*.lock` file older than this many seconds before claiming the gene,
    /// and run a background sweep about every 10 minutes over the output directory (crash recovery on shared storage).
    pub stale_lock_secs: u64,
    /// RNG seed for Lasso (per target via [`mix_execution_random_seed`]) and CNN minibatch order.
    #[serde(default = "default_execution_random_seed")]
    pub random_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerturbationConfig {
    pub beta_scale_factor: f64,
    pub beta_cap: Option<f64>,
    pub n_propagation: usize,
    /// Grid spacing as a fraction of the Gaussian radius for approximate
    /// received-ligand computation.  E.g. 0.5 → spacing = radius/2.
    /// Smaller = more accurate, larger = faster.  Omit or comment out for
    /// exact O(N²) computation.
    pub ligand_grid_factor: Option<f64>,
    /// Default cells CSV path for `spacetravlr-perturb` export / TUI (relative to run TOML directory unless absolute).
    #[serde(default)]
    pub cells_csv: Option<String>,
    /// Default column in `cells_csv` (required in TOML when `cells_csv` is set).
    #[serde(default)]
    pub cells_csv_column: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelExportConfig {
    /// When true, export trained CNN weights for genes that run per-cell CNN refinement (default off).
    pub save_cnn_weights: bool,
    /// When true, write `{gene}_cnn_train_data.npz` plus `{gene}_cnn_train_meta.json` under
    /// [`Self::output_subdir`] for the Python reference trainer (`scripts/python_train_cnn.py`): same
    /// scaled `x`, `y`, spatial maps, and Lasso anchor init Rust used—no
    /// `SPACETRAVLR_DUMP_CNN_TRAIN_DATA` env var required.
    /// Still requires at least one trained CNN cluster (`est.models` non-empty).
    #[serde(default)]
    pub write_cnn_train_data_npz: bool,
    /// Write .npz with deflate compression (recommended).
    pub compressed_npz: bool,
    /// Subdirectory under [execution].output_dir for CNN `.npz` exports (default `CNN_weights`).
    pub output_subdir: String,
}

fn default_data_spatial_species() -> String {
    String::new()
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            adata_path: String::new(),
            layer: "imputed_count".into(),
            cluster_annot: "cell_type".into(),
            condition: None,
            perturb_obs_subset_file: None,
            spatial_species: default_data_spatial_species(),
            spatial_median_nn_target_um: None,
        }
    }
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            radius: 200.0,
            spatial_dim: 32,
            contact_distance: 50.0,
            weighted_ligand_scale_factor: 1.0,
        }
    }
}

impl Default for GrnConfig {
    fn default() -> Self {
        Self {
            network_data_dir: None,
            tf_priors_feather: None,
            tf_ligand_cutoff: 0.2,
            celloracle_p_max: 0.05,
            max_ligands: None,
            use_tf_modulators: true,
            use_lr_modulators: true,
            use_tfl_modulators: true,
            train_modulators: None,
            extra_modulators: Vec::new(),
            extra_modulators_file: None,
            extra_lr: Vec::new(),
            extra_lr_file: None,
        }
    }
}

/// Parse `[grn].train_modulators` / `--train-modulators`: comma- or whitespace-separated
/// `tf`, `lr`, `tfl` (alias `ltf`). Returns `(use_tf, use_lr, use_tfl)`.
pub fn parse_train_modulators_tokens(raw: &str) -> anyhow::Result<(bool, bool, bool)> {
    let mut tf = false;
    let mut lr = false;
    let mut tfl = false;
    for tok in raw
        .split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let t = tok.to_ascii_lowercase();
        match t.as_str() {
            "tf" => tf = true,
            "lr" => lr = true,
            "tfl" | "ltf" => tfl = true,
            _ => anyhow::bail!(
                "unknown train_modulators token {tok:?} (expected tf, lr, tfl or ltf; comma- or space-separated)"
            ),
        }
    }
    Ok((tf, lr, tfl))
}

pub type ResolvedExtraModulatorsAndLr = (Vec<String>, Vec<(String, String)>);

impl GrnConfig {
    /// Applies [`GrnConfig::train_modulators`] when set (overwrites the three `use_*_modulators`
    /// flags), then clears that field. Errors if no modulator family remains enabled.
    pub fn apply_train_modulators_shorthand(&mut self) -> anyhow::Result<()> {
        if let Some(raw) = self.train_modulators.take() {
            let t = raw.trim();
            if t.is_empty() {
                anyhow::bail!(
                    "[grn].train_modulators is empty; omit the key or use tokens: tf, lr, tfl (or ltf)"
                );
            }
            let (use_tf, use_lr, use_tfl) = parse_train_modulators_tokens(t)?;
            self.use_tf_modulators = use_tf;
            self.use_lr_modulators = use_lr;
            self.use_tfl_modulators = use_tfl;
        }
        if !self.use_tf_modulators && !self.use_lr_modulators && !self.use_tfl_modulators {
            anyhow::bail!(
                "at least one GRN modulator family must be enabled (TF targets, ligand–receptor, or TF–ligand / NicheNet-style); set [grn].train_modulators or the use_*_modulators flags"
            );
        }
        Ok(())
    }

    /// Merge TOML `extra_modulators` / `extra_lr` with optional files. Paths are expanded (`~`);
    /// relative paths resolve against `config_file_parent` when provided.
    pub fn resolve_extra_modulators_and_lr(
        &self,
        config_file_parent: Option<&Path>,
    ) -> anyhow::Result<ResolvedExtraModulatorsAndLr> {
        let resolve_path = |raw: &str| -> PathBuf {
            let exp = expand_user_path(raw.trim());
            let pb = Path::new(&exp);
            if pb.is_absolute() {
                pb.to_path_buf()
            } else if let Some(parent) = config_file_parent {
                parent.join(pb)
            } else {
                pb.to_path_buf()
            }
        };

        let mut genes: Vec<String> = Vec::new();
        let mut gene_seen: HashSet<String> = HashSet::new();
        for g in &self.extra_modulators {
            let t = g.trim();
            if t.is_empty() {
                continue;
            }
            let s = t.to_string();
            if gene_seen.insert(s.clone()) {
                genes.push(s);
            }
        }
        if let Some(ref f) = self.extra_modulators_file {
            let path = resolve_path(f);
            for g in crate::grn_extra::load_extra_modulators_file(&path)? {
                if gene_seen.insert(g.clone()) {
                    genes.push(g);
                }
            }
        }

        let mut pairs: Vec<(String, String)> = Vec::new();
        let mut pair_seen: HashSet<String> = HashSet::new();
        for s in &self.extra_lr {
            if let Some(p) = crate::grn_extra::parse_extra_lr_token(s) {
                let key = format!("{}${}", p.0, p.1);
                if pair_seen.insert(key.clone()) {
                    pairs.push(p);
                }
            }
        }
        if let Some(ref f) = self.extra_lr_file {
            let path = resolve_path(f);
            for p in crate::grn_extra::load_extra_lr_file(&path)? {
                let key = format!("{}${}", p.0, p.1);
                if pair_seen.insert(key) {
                    pairs.push(p);
                }
            }
        }

        Ok((genes, pairs))
    }
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            adam_beta_1: 0.9,
            adam_beta_2: 0.999,
            adam_epsilon: 1e-5,
            weight_decay: None,
            grad_clip_norm: Some(3.0),
            spatial_feature_radius: 100.0,
            output_activation: CnnOutputActivation::default(),
            mean_beta_lasso_prior_weight: 0.005,
            lr_schedule: CnnLrSchedule::Cosine,
            cosine_lr_min_ratio: 0.01,
            lr_warmup_epochs: 0,
            cnn_early_stop_patience: 8,
            cnn_early_stop_min_epochs: 12,
            lasso_pred_align_weight: 0.0,
            lasso_pred_align_linear_decay: false,
            drop_cnn_if_insample_worse_than_lasso: true,
            cnn_vs_lasso_arbitration_margin: 0.0,
            cnn_minibatch_size: 512,
            cnn_inference_batch_size: 512,
            cnn_max_batches_per_epoch: None,
            cnn_max_cells_per_epoch: default_cnn_max_cells_per_epoch(),
            min_cells_for_cnn: 0,
            multi_channel_spatial_maps: false,
            ego_center_spatial_maps: false,
        }
    }
}

impl Default for LassoConfig {
    fn default() -> Self {
        Self {
            l1_reg: 1e-5,
            group_reg: 1e-5,
            n_iter: 500,
            tol: 1e-4,
            scale_modulators: true,
            unscale_betas_on_export: true,
            parallel_lasso_clusters: false,
            gram_override: None,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            seed_only: true,
            mode: Some(CnnTrainingMode::Seed),
            epochs: 10,
            learning_rate: 1e-3,
            score_threshold: 0.2,
            genes: None,
            max_genes: None,
        }
    }
}

/// Intersect AnnData `var` names with an optional allow-list, preserving `all_var_names` order.
pub fn filter_training_var_names(
    all_var_names: &[String],
    gene_filter: Option<&[String]>,
) -> Vec<String> {
    let mut v = all_var_names.to_vec();
    if let Some(filter) = gene_filter {
        v.retain(|g| filter.contains(g));
    }
    v
}

/// Filter (see [`filter_training_var_names`]) then cap length at `max_genes` when set.
pub fn resolve_training_target_genes(
    all_var_names: &[String],
    gene_filter: Option<&[String]>,
    max_genes: Option<usize>,
) -> Vec<String> {
    let mut v = filter_training_var_names(all_var_names, gene_filter);
    if let Some(n) = max_genes {
        if v.len() > n {
            v.truncate(n);
        }
    }
    v
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            n_parallel: 1,
            output_dir: String::new(),
            write_minimal_repro_h5ad: false,
            stale_lock_secs: 0,
            random_seed: default_execution_random_seed(),
        }
    }
}

impl Default for PerturbationConfig {
    fn default() -> Self {
        Self {
            beta_scale_factor: 1.0,
            beta_cap: None,
            n_propagation: 4,
            ligand_grid_factor: None,
            cells_csv: None,
            cells_csv_column: None,
        }
    }
}

impl Default for ModelExportConfig {
    fn default() -> Self {
        Self {
            save_cnn_weights: false,
            write_cnn_train_data_npz: false,
            compressed_npz: true,
            output_subdir: "CNN_weights".into(),
        }
    }
}

/// Expand `~` / `~/` in a path string (HOME / USERPROFILE).
pub fn expand_user_path(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    if s == "~" {
        return std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| s.to_string());
    }
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(h) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return format!("{}/{}", h.trim_end_matches('/'), rest);
        }
    }
    s.to_string()
}

/// Resolve `spaceship_config.toml` next to prebuilt binaries (`…/data/`, same layout as
/// `install.sh`) or under `SPACETRAVLR_DATA_DIR`, then cwd (see [`crate::network::SPACETRAVLR_DATA_DIR_ENV`]).
pub fn resolve_spaceship_config_toml_path() -> Option<PathBuf> {
    const DATA_DIR_ENV: &str = "SPACETRAVLR_DATA_DIR";
    for name in ["spaceship_config.toml", "SpaceshipConfig.toml"] {
        if let Ok(dir) = std::env::var(DATA_DIR_ENV) {
            let dir = dir.trim();
            if !dir.is_empty() {
                let p = PathBuf::from(expand_user_path(dir)).join(name);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                for rel in ["data", "../data"] {
                    let p = parent.join(rel).join(name);
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
        }
        let cwd_rel = Path::new(name);
        if cwd_rel.is_file() {
            return Some(cwd_rel.to_path_buf());
        }
    }
    None
}

/// Resolve `malt_label_transfer.py` (MALT / `--map-labels`) under [`SPACETRAVLR_DATA_DIR_ENV`],
/// `…/data/` next to the executable, then cwd — same layout as [`resolve_spaceship_config_toml_path`]
/// and `scripts/install.sh`.
pub fn resolve_malt_label_transfer_py_path() -> Option<PathBuf> {
    const NAME: &str = "malt_label_transfer.py";
    const DATA_DIR_ENV: &str = "SPACETRAVLR_DATA_DIR";
    if let Ok(dir) = std::env::var(DATA_DIR_ENV) {
        let dir = dir.trim();
        if !dir.is_empty() {
            let p = PathBuf::from(expand_user_path(dir)).join(NAME);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for rel in ["data", "../data"] {
                let p = parent.join(rel).join(NAME);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    let data_cwd = Path::new("data").join(NAME);
    if data_cwd.is_file() {
        return Some(data_cwd);
    }
    let mut dir = std::env::current_dir().unwrap_or_default();
    for _ in 0..10 {
        let p = dir.join("data").join(NAME);
        if p.is_file() {
            return Some(p);
        }
        if !dir.pop() {
            break;
        }
    }
    let cwd_rel = Path::new(NAME);
    if cwd_rel.is_file() {
        return Some(cwd_rel.to_path_buf());
    }
    None
}

/// Strip a `file:` / `file://` URL prefix so pasted Finder / browser paths open correctly.
fn strip_file_url_prefix(s: &str) -> &str {
    let Some(rest) = s.strip_prefix("file:") else {
        return s;
    };
    let rest = rest.strip_prefix("//").unwrap_or(rest);
    if rest.is_empty() {
        return s;
    }
    if rest.starts_with('/') {
        return rest;
    }
    match rest.find('/') {
        Some(i) => &rest[i..],
        None => s,
    }
}

/// Normalize paths pasted into the spatial viewer (or similar UIs): trim, strip UTF-8 BOM,
/// optional wrapping quotes, `file://` URLs, then [`expand_user_path`].
pub fn normalize_ui_path(s: &str) -> String {
    let s = s.trim().trim_start_matches('\u{feff}').trim();
    let s = if s.len() >= 2 {
        let b = s.as_bytes();
        if (b[0] == b'"' && b[b.len() - 1] == b'"') || (b[0] == b'\'' && b[b.len() - 1] == b'\'') {
            s[1..s.len() - 1].trim()
        } else {
            s
        }
    } else {
        s
    };
    let s = strip_file_url_prefix(s.trim());
    expand_user_path(s)
}

#[cfg(test)]
mod normalize_ui_path_tests {
    use super::normalize_ui_path;

    #[test]
    fn file_triple_slash_unix() {
        assert_eq!(
            normalize_ui_path("file:///tmp/snrna_human_tonsil_v2.h5ad"),
            "/tmp/snrna_human_tonsil_v2.h5ad"
        );
    }

    #[test]
    fn file_localhost_unix() {
        assert_eq!(
            normalize_ui_path("file://localhost/tmp/a.h5ad"),
            "/tmp/a.h5ad"
        );
    }

    #[test]
    fn strips_wrapping_quotes() {
        assert_eq!(normalize_ui_path("  \"/tmp/x.h5ad\"  "), "/tmp/x.h5ad");
    }
}

/// Sanitized `.h5ad` stem for general output filenames (same character rules as
/// [`default_output_dir_for_adata_path`]).
pub fn canonical_adata_stem(adata_path: &std::path::Path) -> String {
    let stem = adata_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim();
    let stem = if stem.is_empty() {
        "spacetravlr_run"
    } else {
        stem
    };
    stem.chars()
        .map(|c| match c {
            '/' | '\\' | '\0' => '_',
            c => c,
        })
        .collect()
}

/// Stem for run artifacts (prep filenames under `output_dir/spacetravlr_prep/`, and CLI `{stem}_processed.h5ad`).
///
/// Trailing `_processed` segments (ASCII case-insensitive) are removed so a file named
/// `dataset_processed.h5ad` maps to `dataset_processed.h5ad`, not `dataset_processed_processed.h5ad`.
pub fn canonical_training_prep_stem(adata_path: &Path) -> String {
    const SUF: &str = "_processed";
    let mut s = canonical_adata_stem(adata_path);
    while s.len() > SUF.len() && s.to_lowercase().ends_with(SUF) {
        s.truncate(s.len() - SUF.len());
    }
    if s.is_empty() {
        "spacetravlr_run".to_string()
    } else {
        s
    }
}

/// Default training output directory: current working directory + `{stem}_{YYYY-MM-DD}`.
/// `stem` is [`canonical_training_prep_stem`] of the `.h5ad` path.
pub fn default_output_dir_for_adata_path(adata_path: impl AsRef<Path>) -> anyhow::Result<String> {
    let adata_path = adata_path.as_ref();
    let stem = canonical_training_prep_stem(adata_path);
    let date = chrono::Local::now().format("%Y-%m-%d");
    let dir_name = format!("{}_{}", stem, date);
    let cwd =
        std::env::current_dir().context("default output_dir: could not read current directory")?;
    Ok(cwd.join(dir_name).to_string_lossy().to_string())
}

#[cfg(test)]
mod canonical_training_prep_stem_tests {
    use super::{canonical_adata_stem, canonical_training_prep_stem};
    use std::path::Path;

    #[test]
    fn strips_one_processed_suffix() {
        assert_eq!(
            canonical_training_prep_stem(Path::new("/tmp/SlideTags_human_tonsil_processed.h5ad")),
            "SlideTags_human_tonsil"
        );
    }

    #[test]
    fn strips_chained_processed_suffixes() {
        assert_eq!(
            canonical_training_prep_stem(Path::new("/x/foo_processed_processed.h5ad")),
            "foo"
        );
    }

    #[test]
    fn plain_stem_unchanged() {
        assert_eq!(
            canonical_training_prep_stem(Path::new("/data/foo.h5ad")),
            canonical_adata_stem(Path::new("/data/foo.h5ad"))
        );
    }
}

impl SpaceshipConfig {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        let mut config: SpaceshipConfig = toml::from_str(&contents)?;
        config.grn.apply_train_modulators_shorthand()?;
        Ok(config)
    }

    /// Load a run repro TOML and merge overlay fragments (`[data]`, `[perturbation]`, …) from
    /// `overlay_root`. Scalar/array keys in overlay tables replace; nested tables merge recursively.
    pub fn from_file_merged(
        run_repro_path: impl AsRef<Path>,
        overlay_root: Option<&toml::Value>,
    ) -> anyhow::Result<Self> {
        let path = run_repro_path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut root: toml::Value = toml::from_str(&text)
            .with_context(|| format!("parse SpaceshipConfig TOML {}", path.display()))?;
        if let Some(ov) = overlay_root {
            merge_spaceship_overlay_into_toml(&mut root, ov);
        }
        let mut cfg = <SpaceshipConfig as Deserialize>::deserialize(root).with_context(|| {
            format!(
                "deserialize merged SpaceshipConfig from {} (after overlay)",
                path.display()
            )
        })?;
        cfg.grn.apply_train_modulators_shorthand()?;
        Ok(cfg)
    }

    /// Load `spacetravlr_run_repro.toml`-style document, fill missing keys from repo
    /// [`repo_spaceship_config_path`] when present, then merge `--config` overlay (overlay wins on conflict).
    pub fn from_run_repro_merged(
        repro_path: impl AsRef<Path>,
        config_overlay: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let path = repro_path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut root: toml::Value =
            toml::from_str(&text).with_context(|| format!("parse run repro {}", path.display()))?;

        let repo = Self::repo_spaceship_config_path();
        let underlay_path = if repo.is_file() {
            Some(repo)
        } else {
            resolve_spaceship_config_toml_path()
        };
        if let Some(ref up) = underlay_path {
            match std::fs::read_to_string(up) {
                Ok(repo_text) => match toml::from_str::<toml::Value>(&repo_text) {
                    Ok(repo_root) => {
                        merge_spaceship_underlay_into_toml(&mut root, &repo_root);
                    }
                    Err(e) => eprintln!(
                        "Warning: failed to parse spaceship underlay {}: {}",
                        up.display(),
                        e
                    ),
                },
                Err(e) => eprintln!("Warning: failed to read {}: {}", up.display(), e),
            }
        }

        if let Some(p) = config_overlay {
            let overlay_text =
                std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
            let overlay_root: toml::Value =
                toml::from_str(&overlay_text).with_context(|| format!("parse {}", p.display()))?;
            merge_spaceship_overlay_into_toml(&mut root, &overlay_root);
        }

        let mut cfg = <SpaceshipConfig as Deserialize>::deserialize(root).with_context(|| {
            format!(
                "deserialize SpaceshipConfig from {} (after repro merge)",
                path.display()
            )
        })?;
        cfg.grn.apply_train_modulators_shorthand()?;
        Ok(cfg)
    }

    pub fn to_toml_pretty(&self) -> anyhow::Result<String> {
        toml::to_string_pretty(self).map_err(|e| anyhow::anyhow!("serialize config to TOML: {e}"))
    }

    pub fn write_run_repro_toml(&self, output_dir: &Path) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(output_dir)?;
        let text = self.to_toml_pretty()?;
        let path = output_dir.join(RUN_REPRO_TOML_FILENAME);
        std::fs::write(&path, text.as_str())?;
        let _ = std::fs::remove_file(output_dir.join("spacetravlr_run_config.toml"));
        Ok(path)
    }

    /// Write `spacetravlr_run_repro.toml` only if missing (first trainer on a shared output directory).
    pub fn write_run_repro_toml_if_missing(
        &self,
        output_dir: &Path,
    ) -> anyhow::Result<Option<PathBuf>> {
        std::fs::create_dir_all(output_dir)?;
        let path = output_dir.join(RUN_REPRO_TOML_FILENAME);
        if path.is_file() {
            return Ok(None);
        }
        let text = self.to_toml_pretty()?;
        std::fs::write(&path, text.as_str())?;
        let _ = std::fs::remove_file(output_dir.join("spacetravlr_run_config.toml"));
        Ok(Some(path))
    }

    /// `spaceship_config.toml` next to the workspace `Cargo.toml` for this crate (the repo root when building `spacetravlr` from this tree).
    ///
    /// At runtime after `cargo install` from another machine, this path may not exist; callers fall back to [`resolve_spaceship_config_toml_path`] or [`SpaceshipConfig::default`].
    pub fn repo_spaceship_config_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("spaceship_config.toml")
    }

    pub fn discover_default_path() -> Option<PathBuf> {
        let repo = Self::repo_spaceship_config_path();
        if repo.is_file() {
            return Some(repo);
        }
        resolve_spaceship_config_toml_path()
    }

    fn read_config_base_document() -> anyhow::Result<(toml::Value, Option<PathBuf>)> {
        let repo = Self::repo_spaceship_config_path();
        if repo.is_file() {
            let text = std::fs::read_to_string(&repo)
                .with_context(|| format!("read {}", repo.display()))?;
            match toml::from_str::<toml::Value>(&text) {
                Ok(v) => return Ok((v, Some(repo))),
                Err(e) => eprintln!("Warning: failed to parse {}: {}", repo.display(), e),
            }
        }
        if let Some(p) = resolve_spaceship_config_toml_path() {
            let text =
                std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
            match toml::from_str::<toml::Value>(&text) {
                Ok(v) => return Ok((v, Some(p))),
                Err(e) => eprintln!("Warning: failed to parse {}: {}", p.display(), e),
            }
        }
        Ok((toml::Value::Table(Default::default()), None))
    }

    /// Load base TOML (repo [`repo_spaceship_config_path`], else [`resolve_spaceship_config_toml_path`], else empty + serde defaults),
    /// then merge `overlay` on top for keys in [`SPACESHIP_MERGE_SECTIONS`].
    pub fn try_load_merged(overlay: Option<&Path>) -> anyhow::Result<Self> {
        let (mut doc, base_path) = Self::read_config_base_document()?;
        if let Some(p) = overlay {
            let overlay_text =
                std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
            let overlay_root: toml::Value =
                toml::from_str(&overlay_text).with_context(|| format!("parse {}", p.display()))?;
            merge_spaceship_overlay_into_toml(&mut doc, &overlay_root);
            if let Some(ref bp) = base_path {
                eprintln!(
                    "Merged --config {} over defaults from {}.",
                    p.display(),
                    bp.display()
                );
            } else {
                eprintln!(
                    "Loaded --config {} (no base spaceship TOML; serde defaults for omitted keys).",
                    p.display()
                );
            }
        } else if let Some(bp) = base_path {
            eprintln!("Loaded config from {}", bp.display());
        }
        let mut cfg = <SpaceshipConfig as Deserialize>::deserialize(doc)
            .context("deserialize SpaceshipConfig after TOML merge")?;
        cfg.grn.apply_train_modulators_shorthand()?;
        Ok(cfg)
    }

    pub fn load() -> Self {
        match Self::try_load_merged(None) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: config load failed: {e:#}; using built-in defaults");
                Self::default()
            }
        }
    }

    pub fn resolved_cnn_mode(&self) -> CnnTrainingMode {
        self.training.mode.unwrap_or(CnnTrainingMode::Seed)
    }

    pub fn full_cnn(&self) -> bool {
        matches!(self.resolved_cnn_mode(), CnnTrainingMode::Full)
    }

    pub fn resolve_adata_path(&self) -> String {
        self.data.adata_path.trim().to_string()
    }

    /// Training output directory (contains `*_betadata.feather`): `[execution].output_dir`.
    ///
    /// Relative `output_dir` entries are resolved against the directory that contains
    /// `run_toml_path` so a copied or symlinked repro TOML still finds feathers. If `output_dir`
    /// is empty, returns the parent of `run_toml_path` (legacy layout: TOML next to feathers).
    pub fn resolve_training_output_dir(&self, run_toml_path: &Path) -> PathBuf {
        let toml_dir = run_toml_path.parent().unwrap_or_else(|| Path::new("."));
        let raw = self.execution.output_dir.trim();
        if raw.is_empty() {
            return toml_dir.to_path_buf();
        }
        let expanded = expand_user_path(raw);
        let p = Path::new(expanded.as_str());
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            toml_dir.join(p)
        }
    }
}

#[cfg(test)]
mod resolve_training_output_dir_tests {
    use super::SpaceshipConfig;
    use std::path::Path;

    #[test]
    fn repro_toml_serializes_tf_priors_feather_when_some() {
        let mut cfg = SpaceshipConfig::default();
        cfg.grn.tf_priors_feather = Some("/data/priors.feather".into());
        let s = cfg.to_toml_pretty().unwrap();
        assert!(
            s.contains("tf_priors_feather") && s.contains("/data/priors.feather"),
            "repro TOML should record grn.tf_priors_feather for join / viewer: {s}"
        );
    }

    #[test]
    fn empty_output_dir_uses_toml_parent() {
        let mut cfg = SpaceshipConfig::default();
        cfg.execution.output_dir = String::new();
        let p = Path::new("/configs/x/spacetravlr_run_repro.toml");
        assert_eq!(cfg.resolve_training_output_dir(p), Path::new("/configs/x"));
    }

    #[test]
    fn relative_output_dir_joined_to_toml_parent() {
        let mut cfg = SpaceshipConfig::default();
        cfg.execution.output_dir = "lasso_out".into();
        let p = Path::new("/home/u/notebook.toml");
        assert_eq!(
            cfg.resolve_training_output_dir(p),
            Path::new("/home/u/lasso_out")
        );
    }

    #[test]
    fn repro_toml_roundtrip_training_genes_and_max_genes() {
        let mut cfg = SpaceshipConfig::default();
        cfg.training.genes = Some(vec!["Actb".into(), "Gapdh".into()]);
        cfg.training.max_genes = Some(128);
        let s = cfg.to_toml_pretty().unwrap();
        let back: SpaceshipConfig = toml::from_str(&s).expect("deserialize repro TOML");
        assert_eq!(
            back.training.genes,
            Some(vec!["Actb".into(), "Gapdh".into()])
        );
        assert_eq!(back.training.max_genes, Some(128));
    }

    #[test]
    fn grn_max_lr_toml_alias_deserializes_to_max_ligands() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[grn]
max_lr = 42
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.grn.max_ligands, Some(42));
    }

    #[test]
    fn repro_toml_deserialize_without_training_genes_defaults_none() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 5
learning_rate = 0.001
score_threshold = 0.1
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert!(cfg.training.genes.is_none());
        assert!(cfg.training.max_genes.is_none());
    }
}

#[cfg(test)]
mod train_modulators_config_tests {
    use super::{SpaceshipConfig, parse_train_modulators_tokens};
    use std::fs;

    #[test]
    fn parse_accepts_ltf_alias_and_whitespace() {
        let (tf, lr, tfl) = parse_train_modulators_tokens(" TF \n ltf ").unwrap();
        assert!(tf);
        assert!(!lr);
        assert!(tfl);
    }

    #[test]
    fn parse_comma_combo() {
        let (tf, lr, tfl) = parse_train_modulators_tokens("tf,lr").unwrap();
        assert!(tf);
        assert!(lr);
        assert!(!tfl);
    }

    #[test]
    fn parse_unknown_token_errors() {
        assert!(parse_train_modulators_tokens("tf,x").is_err());
    }

    #[test]
    fn from_file_applies_train_modulators_shorthand() {
        let tmp = std::env::temp_dir().join(format!(
            "spacetravlr_train_mod_test_{}.toml",
            std::process::id()
        ));
        let body = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 1
learning_rate = 0.001
score_threshold = 0.1

[grn]
train_modulators = "lr"
use_tf_modulators = true
use_tfl_modulators = true
"#;
        fs::write(&tmp, body).unwrap();
        let cfg = SpaceshipConfig::from_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);
        assert!(!cfg.grn.use_tf_modulators);
        assert!(cfg.grn.use_lr_modulators);
        assert!(!cfg.grn.use_tfl_modulators);
        assert!(cfg.grn.train_modulators.is_none());
    }

    #[test]
    fn from_file_errors_when_no_modulators_enabled() {
        let tmp = std::env::temp_dir().join(format!(
            "spacetravlr_train_mod_none_{}.toml",
            std::process::id()
        ));
        let body = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 1
learning_rate = 0.001
score_threshold = 0.1

[grn]
use_tf_modulators = false
use_lr_modulators = false
use_tfl_modulators = false
"#;
        fs::write(&tmp, body).unwrap();
        assert!(SpaceshipConfig::from_file(&tmp).is_err());
        let _ = fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod training_target_genes_tests {
    use super::{filter_training_var_names, resolve_training_target_genes};

    fn vars() -> Vec<String> {
        vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()]
    }

    #[test]
    fn filter_none_keeps_order_and_len() {
        let v = vars();
        let out = filter_training_var_names(&v, None);
        assert_eq!(out, v);
    }

    #[test]
    fn filter_preserves_var_order() {
        let v = vars();
        let f = vec!["c".into(), "a".into()];
        let out = filter_training_var_names(&v, Some(&f));
        assert_eq!(out, vec!["a", "c"]);
    }

    #[test]
    fn filter_empty_list_yields_empty() {
        let v = vars();
        let f: Vec<String> = vec![];
        let out = filter_training_var_names(&v, Some(&f));
        assert!(out.is_empty());
    }

    #[test]
    fn resolve_cap_only_truncates_prefix_in_var_order() {
        let v = vars();
        let out = resolve_training_target_genes(&v, None, Some(3));
        assert_eq!(out, vec!["a", "b", "c"]);
    }

    #[test]
    fn resolve_filter_then_cap() {
        let v = vars();
        let f = vec!["e".into(), "b".into(), "a".into(), "d".into()];
        let out = resolve_training_target_genes(&v, Some(&f), Some(2));
        assert_eq!(out, vec!["a", "b"]);
    }

    #[test]
    fn resolve_cap_larger_than_filtered_no_op() {
        let v = vars();
        let f = vec!["b".into(), "c".into()];
        let out = resolve_training_target_genes(&v, Some(&f), Some(10));
        assert_eq!(out, vec!["b", "c"]);
    }

    #[test]
    fn resolve_matches_sequential_filter_and_truncate() {
        let v = vars();
        let f = vec!["d".into(), "b".into()];
        let mut manual = filter_training_var_names(&v, Some(&f));
        manual.truncate(1);
        let resolved = resolve_training_target_genes(&v, Some(&f), Some(1));
        assert_eq!(resolved, manual);
        assert_eq!(resolved, vec!["b"]);
    }
}

#[cfg(test)]
mod merge_spaceship_overlay_tests {
    use super::{SpaceshipConfig, merge_spaceship_overlay_into_toml};
    use std::path::PathBuf;

    fn tmp_run_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "stlr_cfg_merge_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn overlay_replaces_nested_perturbation_fields() {
        let base = r#"
[data]
adata_path = "/data/a.h5ad"
layer = "imputed_count"
cluster_annot = "ct"

[perturbation]
n_propagation = 2
beta_scale_factor = 1.0
"#;
        let overlay = r#"
[perturbation]
n_propagation = 9
"#;
        let mut root: toml::Value = toml::from_str(base).unwrap();
        let ov: toml::Value = toml::from_str(overlay).unwrap();
        merge_spaceship_overlay_into_toml(&mut root, &ov);
        let cfg: SpaceshipConfig = toml::from_str(&toml::to_string_pretty(&root).unwrap()).unwrap();
        assert_eq!(cfg.perturbation.n_propagation, 9);
        assert_eq!(cfg.perturbation.beta_scale_factor, 1.0);
        assert_eq!(cfg.data.layer, "imputed_count");
    }

    #[test]
    fn from_file_merged_matches_manual_merge() {
        let tmp = tmp_run_dir();
        let repro = tmp.join("spacetravlr_run_repro.toml");
        let body = r#"
[data]
adata_path = "/x.h5ad"
layer = "L0"
cluster_annot = "c0"

[perturbation]
n_propagation = 3
"#;
        std::fs::write(&repro, body).unwrap();
        let overlay: toml::Value = toml::from_str(
            r#"
[perturbation]
n_propagation = 7
[data]
layer = "L1"
"#,
        )
        .unwrap();
        let merged = SpaceshipConfig::from_file_merged(&repro, Some(&overlay)).unwrap();
        assert_eq!(merged.perturbation.n_propagation, 7);
        assert_eq!(merged.data.layer, "L1");
        assert_eq!(merged.data.cluster_annot, "c0");

        let round = SpaceshipConfig::from_file_merged(&repro, None).unwrap();
        assert_eq!(round.perturbation.n_propagation, 3);
        assert_eq!(round.data.layer, "L0");
    }

    #[test]
    fn resolve_training_output_dir_uses_merged_execution() {
        let tmp = tmp_run_dir();
        let repro = tmp.join("spacetravlr_run_repro.toml");
        std::fs::write(
            &repro,
            r#"
[data]
adata_path = "/d.h5ad"

[execution]
output_dir = "out_a"
"#,
        )
        .unwrap();
        let overlay: toml::Value = toml::from_str("[execution]\noutput_dir = \"out_b\"\n").unwrap();
        let cfg = SpaceshipConfig::from_file_merged(&repro, Some(&overlay)).unwrap();
        let dir = cfg.resolve_training_output_dir(repro.as_path());
        assert_eq!(dir, tmp.join("out_b"));
    }

    #[test]
    fn underlay_fills_missing_keys_only() {
        use super::merge_spaceship_underlay_into_toml;
        let mut base: toml::Value = toml::from_str(
            r#"
[data]
adata_path = "/repro.h5ad"
layer = "L0"
"#,
        )
        .unwrap();
        let fill: toml::Value = toml::from_str(
            r#"
[data]
cluster_annot = "c_repo"
layer = "L99"
"#,
        )
        .unwrap();
        merge_spaceship_underlay_into_toml(&mut base, &fill);
        let cfg: SpaceshipConfig = toml::from_str(&toml::to_string_pretty(&base).unwrap()).unwrap();
        assert_eq!(cfg.data.layer, "L0");
        assert_eq!(cfg.data.cluster_annot, "c_repo");
        assert_eq!(cfg.data.adata_path, "/repro.h5ad");
    }

    #[test]
    fn repro_underlay_repo_then_overlay_cli_order() {
        use super::{merge_spaceship_overlay_into_toml, merge_spaceship_underlay_into_toml};
        let mut root: toml::Value = toml::from_str(
            r#"
[data]
adata_path = "/repro.h5ad"
layer = "L0"
"#,
        )
        .unwrap();
        let repo_like: toml::Value = toml::from_str("[data]\ncluster_annot = \"c1\"\n").unwrap();
        merge_spaceship_underlay_into_toml(&mut root, &repo_like);
        let cli: toml::Value = toml::from_str("[data]\nlayer = \"LX\"\n").unwrap();
        merge_spaceship_overlay_into_toml(&mut root, &cli);
        let cfg: SpaceshipConfig = toml::from_str(&toml::to_string_pretty(&root).unwrap()).unwrap();
        assert_eq!(cfg.data.adata_path, "/repro.h5ad");
        assert_eq!(cfg.data.layer, "LX");
        assert_eq!(cfg.data.cluster_annot, "c1");
    }

    #[test]
    fn overlay_max_lr_with_repro_max_ligands_deserializes_join_style_merge() {
        let mut root: toml::Value = toml::from_str(
            r#"
[data]
adata_path = "/x.h5ad"
layer = "L"
cluster_annot = "c"

[grn]
max_ligands = 50
"#,
        )
        .unwrap();
        let ov: toml::Value = toml::from_str("[grn]\nmax_lr = 120\n").unwrap();
        merge_spaceship_overlay_into_toml(&mut root, &ov);
        let cfg: SpaceshipConfig = toml::from_str(&toml::to_string_pretty(&root).unwrap()).unwrap();
        assert_eq!(cfg.grn.max_ligands, Some(120));
    }
}

#[cfg(test)]
mod lasso_scaling_config_tests {
    use super::{LassoConfig, SpaceshipConfig};

    #[test]
    fn default_lasso_has_scale_modulators_true() {
        assert!(LassoConfig::default().scale_modulators);
    }

    #[test]
    fn default_lasso_has_unscale_betas_true() {
        assert!(LassoConfig::default().unscale_betas_on_export);
    }

    #[test]
    fn default_lasso_parallel_clusters_false() {
        assert!(!LassoConfig::default().parallel_lasso_clusters);
    }

    #[test]
    fn default_lasso_gram_override_none() {
        assert!(LassoConfig::default().gram_override.is_none());
    }

    #[test]
    fn toml_explicit_parallel_lasso_clusters_true_parsed() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 5
learning_rate = 0.001
score_threshold = 0.1

[lasso]
parallel_lasso_clusters = true
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert!(cfg.lasso.parallel_lasso_clusters);
    }

    #[test]
    fn toml_explicit_scale_modulators_false_parsed() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 5
learning_rate = 0.001
score_threshold = 0.1

[lasso]
scale_modulators = false
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert!(!cfg.lasso.scale_modulators);
        assert!(cfg.lasso.unscale_betas_on_export);
    }

    #[test]
    fn toml_explicit_unscale_betas_false_parsed() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 5
learning_rate = 0.001
score_threshold = 0.1

[lasso]
unscale_betas_on_export = false
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert!(cfg.lasso.scale_modulators);
        assert!(!cfg.lasso.unscale_betas_on_export);
    }

    #[test]
    fn toml_omitted_scaling_fields_use_true_defaults() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 5
learning_rate = 0.001
score_threshold = 0.1

[lasso]
l1_reg = 1e-9
group_reg = 1e-9
n_iter = 50
tol = 1e-4
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert!(cfg.lasso.scale_modulators);
        assert!(cfg.lasso.unscale_betas_on_export);
    }

    #[test]
    fn toml_gram_override_false_parsed() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 5
learning_rate = 0.001
score_threshold = 0.1

[lasso]
gram_override = false
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.lasso.gram_override, Some(false));
    }
}

#[cfg(test)]
mod preprocess_config_tests {
    use super::{PreprocessConfig, SpaceshipConfig, merge_spaceship_overlay_into_toml};

    #[test]
    fn default_preprocess_matches_rust_pipeline_defaults() {
        let p = PreprocessConfig::default();
        assert_eq!(p.min_genes, 100);
        assert_eq!(p.min_cells, 3);
        assert_eq!(p.normalize_target_sum, 10_000);
        assert_eq!(p.n_top_hvg, 2000);
        assert_eq!(p.n_pca_components, 50);
        assert_eq!(p.n_neighbors, 15);
        assert_eq!(p.leiden_resolution, 1.0);
        assert_eq!(p.magic_t, 3);
        let rust = p.to_rust_preprocess_params();
        assert_eq!(rust.n_top_hvg, p.n_top_hvg);
        assert_eq!(rust.n_pca_components, p.n_pca_components);
        assert_eq!(rust.min_genes, p.min_genes);
        assert_eq!(rust.magic_t, p.magic_t);
    }

    #[test]
    fn toml_preprocess_section_deserializes() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[preprocess]
min_genes = 50
n_top_hvg = 800
n_pca_components = 12
n_neighbors = 8
leiden_resolution = 0.8
magic_t = 2
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.preprocess.min_genes, 50);
        assert_eq!(cfg.preprocess.n_top_hvg, 800);
        assert_eq!(cfg.preprocess.n_pca_components, 12);
        assert_eq!(cfg.preprocess.n_neighbors, 8);
        assert!((cfg.preprocess.leiden_resolution - 0.8).abs() < 1e-9);
        assert_eq!(cfg.preprocess.magic_t, 2);
        assert_eq!(cfg.preprocess.min_cells, 3);
    }

    #[test]
    fn overlay_merges_preprocess_section() {
        let base = r#"
[data]
adata_path = "/x.h5ad"
layer = "L"
cluster_annot = "c"

[preprocess]
n_top_hvg = 1000
n_pca_components = 40
"#;
        let overlay = r#"
[preprocess]
n_pca_components = 25
n_neighbors = 20
"#;
        let mut root: toml::Value = toml::from_str(base).unwrap();
        let ov: toml::Value = toml::from_str(overlay).unwrap();
        merge_spaceship_overlay_into_toml(&mut root, &ov);
        let cfg: SpaceshipConfig = toml::from_str(&toml::to_string_pretty(&root).unwrap()).unwrap();
        assert_eq!(cfg.preprocess.n_top_hvg, 1000);
        assert_eq!(cfg.preprocess.n_pca_components, 25);
        assert_eq!(cfg.preprocess.n_neighbors, 20);
    }

    #[test]
    fn repro_toml_roundtrip_preprocess_fields() {
        let mut cfg = SpaceshipConfig::default();
        cfg.preprocess.n_top_hvg = 1500;
        cfg.preprocess.min_genes = 80;
        cfg.preprocess.magic_t = 4;
        let s = cfg.to_toml_pretty().unwrap();
        let back: SpaceshipConfig = toml::from_str(&s).expect("deserialize repro TOML");
        assert_eq!(back.preprocess.n_top_hvg, 1500);
        assert_eq!(back.preprocess.min_genes, 80);
        assert_eq!(back.preprocess.magic_t, 4);
    }
}
