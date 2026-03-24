use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceshipConfig {
    #[serde(default)]
    pub data: DataConfig,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpatialConfig {
    pub radius: f64,
    pub spatial_dim: usize,
    pub contact_distance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GrnConfig {
    /// Directory containing `{mouse|human}_network.parquet`. Overrides `SPACETRAVLR_DATA_DIR` and
    /// built-in search (manifest / cwd walk). Tilde and `~/` expanded like `data.adata_path`.
    pub network_data_dir: Option<String>,
    pub tf_ligand_cutoff: f64,
    /// Cap LR pairs in database order (no expression ranking).
    pub max_lr_pairs: Option<usize>,
    /// Keep only this many LR pairs with highest mean expression
    /// (average of ligand and receptor means across cells). Requires a full
    /// pass over the expression matrix at pipeline start. Ignores `max_lr_pairs`
    /// when set.
    pub top_lr_pairs_by_mean_expression: Option<usize>,
    #[serde(default = "default_true")]
    pub use_tf_modulators: bool,
    #[serde(default = "default_true")]
    pub use_lr_modulators: bool,
    #[serde(default = "default_true")]
    pub use_tfl_modulators: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LassoConfig {
    pub l1_reg: f64,
    pub group_reg: f64,
    pub n_iter: usize,
    pub tol: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CnnTrainingMode {
    #[serde(alias = "minimal", alias = "seed-only")]
    Seed,
    Full,
    #[default]
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HybridCnnGatingConfig {
    /// Smallest cluster cell count must be at least this for CNN (sample-complexity gate).
    pub min_cells_per_cluster_for_cnn: usize,
    /// If modulator count exceeds this, Moran's p must pass `moran_p_value_max_when_over_modulator_cap`.
    pub max_modulators_soft_for_cnn: Option<usize>,
    pub moran_k_neighbors: usize,
    pub moran_permutations: usize,
    pub moran_p_value_max: f64,
    pub moran_p_value_max_when_over_modulator_cap: Option<f64>,
    pub require_all_clusters_lasso_converged: bool,
    /// If None, `TrainingConfig.score_threshold` is used as the minimum mean lasso R² for CNN.
    pub min_mean_lasso_r2_for_cnn: Option<f64>,
    pub min_mean_target_expression_for_cnn: Option<f64>,
    pub hybrid_modulator_spatial_weight: f64,
    pub cnn_force_genes_file: Option<String>,
    pub cnn_skip_genes_file: Option<String>,
    /// If set, phase 1 only records candidates; phase 2 runs CNN for the top-K by `rank_score`.
    pub hybrid_cnn_top_k: Option<usize>,
    /// 0 = conservative (stricter Moran p and mean R² gates → fewer CNNs). 1 = permissive.
    /// 0.5 reproduces the effective thresholds implied by `moran_p_value_max` / mean R² alone (legacy behavior).
    #[serde(default = "default_hybrid_cnn_permissiveness")]
    pub hybrid_cnn_permissiveness: f64,
}

fn default_hybrid_cnn_permissiveness() -> f64 {
    0.5
}

fn default_true() -> bool {
    true
}

impl Default for HybridCnnGatingConfig {
    fn default() -> Self {
        Self {
            min_cells_per_cluster_for_cnn: 80,
            max_modulators_soft_for_cnn: Some(256),
            moran_k_neighbors: 8,
            moran_permutations: 99,
            moran_p_value_max: 0.05,
            moran_p_value_max_when_over_modulator_cap: Some(0.01),
            require_all_clusters_lasso_converged: true,
            min_mean_lasso_r2_for_cnn: None,
            min_mean_target_expression_for_cnn: None,
            hybrid_modulator_spatial_weight: 1.0,
            cnn_force_genes_file: None,
            cnn_skip_genes_file: None,
            hybrid_cnn_top_k: None,
            hybrid_cnn_permissiveness: default_hybrid_cnn_permissiveness(),
        }
    }
}

impl HybridCnnGatingConfig {
    fn permissiveness_t(&self) -> f64 {
        self.hybrid_cnn_permissiveness.clamp(0.0, 1.0)
    }

    pub fn effective_moran_p_max(&self) -> f64 {
        let t = self.permissiveness_t();
        let f = 0.3 + 1.4 * t;
        (self.moran_p_value_max * f).clamp(1e-12, 1.0)
    }

    pub fn effective_moran_p_strict(&self) -> f64 {
        let base = self
            .moran_p_value_max_when_over_modulator_cap
            .unwrap_or(self.moran_p_value_max);
        let t = self.permissiveness_t();
        let f = 0.3 + 1.4 * t;
        (base * f).clamp(1e-12, 1.0)
    }

    pub fn effective_min_mean_lasso_r2(&self, base_min_r2: f64) -> f64 {
        let t = self.permissiveness_t();
        let r2f = 1.4 - 0.8 * t;
        (base_min_r2 * r2f).max(0.0)
    }
}

fn default_training_mode_option() -> Option<CnnTrainingMode> {
    Some(CnnTrainingMode::Seed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// Kept for CLI/back-compat; hybrid runs lasso first per gene regardless.
    pub seed_only: bool,
    #[serde(default = "default_training_mode_option")]
    pub mode: Option<CnnTrainingMode>,
    pub epochs: usize,
    pub learning_rate: f64,
    pub score_threshold: f64,
    #[serde(default)]
    pub hybrid: HybridCnnGatingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CnnConfig {
    pub adam_beta_1: f64,
    pub adam_beta_2: f64,
    pub adam_epsilon: f64,
    pub weight_decay: Option<f64>,
    pub grad_clip_norm: Option<f64>,
    pub spatial_feature_radius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionConfig {
    pub n_parallel: usize,
    pub output_dir: String,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelExportConfig {
    /// Export trained CNN weights for genes that run per-cell CNN refinement.
    pub save_cnn_weights: bool,
    /// Write .npz with deflate compression (recommended).
    pub compressed_npz: bool,
    /// Subdirectory under [execution].output_dir for model artifacts.
    pub output_subdir: String,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            adata_path: String::new(),
            layer: "imputed_count".into(),
            cluster_annot: "cell_type_int".into(),
        }
    }
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            radius: 0.1,
            spatial_dim: 32,
            contact_distance: 0.05,
        }
    }
}

impl Default for GrnConfig {
    fn default() -> Self {
        Self {
            network_data_dir: None,
            tf_ligand_cutoff: 0.5,
            max_lr_pairs: None,
            top_lr_pairs_by_mean_expression: None,
            use_tf_modulators: true,
            use_lr_modulators: true,
            use_tfl_modulators: true,
        }
    }
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            adam_beta_1: 0.9,
            adam_beta_2: 0.999,
            adam_epsilon: 1e-5,
            weight_decay: None,
            grad_clip_norm: None,
            spatial_feature_radius: 100.0,
        }
    }
}

impl Default for LassoConfig {
    fn default() -> Self {
        Self {
            l1_reg: 1e-4,
            group_reg: 1e-4,
            n_iter: 100,
            tol: 1e-4,
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
            score_threshold: 0.0,
            hybrid: HybridCnnGatingConfig::default(),
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            n_parallel: 1,
            output_dir: "/tmp/training".into(),
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
        }
    }
}

impl Default for ModelExportConfig {
    fn default() -> Self {
        Self {
            save_cnn_weights: false,
            compressed_npz: true,
            output_subdir: "saved_models".into(),
        }
    }
}

impl Default for SpaceshipConfig {
    fn default() -> Self {
        Self {
            data: DataConfig::default(),
            spatial: SpatialConfig::default(),
            grn: GrnConfig::default(),
            cnn: CnnConfig::default(),
            lasso: LassoConfig::default(),
            training: TrainingConfig::default(),
            execution: ExecutionConfig::default(),
            perturbation: PerturbationConfig::default(),
            model_export: ModelExportConfig::default(),
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

impl SpaceshipConfig {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        let config: SpaceshipConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn load() -> Self {
        let candidates = ["spaceship_config.toml", "SpaceshipConfig.toml"];
        for name in &candidates {
            if Path::new(name).exists() {
                match Self::from_file(name) {
                    Ok(cfg) => {
                        eprintln!("Loaded config from {}", name);
                        return cfg;
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to parse {}: {}", name, e);
                    }
                }
            }
        }
        Self::default()
    }

    pub fn resolved_cnn_mode(&self) -> CnnTrainingMode {
        self.training
            .mode
            .unwrap_or(CnnTrainingMode::Seed)
    }

    pub fn full_cnn(&self) -> bool {
        matches!(self.resolved_cnn_mode(), CnnTrainingMode::Full)
    }

    pub fn min_mean_lasso_r2_for_hybrid_cnn(&self) -> f64 {
        self.training
            .hybrid
            .min_mean_lasso_r2_for_cnn
            .unwrap_or(self.training.score_threshold)
    }

    pub fn resolve_adata_path(&self) -> String {
        self.data.adata_path.trim().to_string()
    }
}
