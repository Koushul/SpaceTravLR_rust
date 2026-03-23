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
    pub tf_ligand_cutoff: f64,
    /// Cap LR pairs in database order (no expression ranking).
    pub max_lr_pairs: Option<usize>,
    /// Keep only this many LR pairs with highest mean expression
    /// (average of ligand and receptor means across cells). Requires a full
    /// pass over the expression matrix at pipeline start. Ignores `max_lr_pairs`
    /// when set.
    pub top_lr_pairs_by_mean_expression: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LassoConfig {
    pub l1_reg: f64,
    pub group_reg: f64,
    pub n_iter: usize,
    pub tol: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    pub seed_only: bool,
    pub epochs: usize,
    pub learning_rate: f64,
    pub score_threshold: f64,
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
            tf_ligand_cutoff: 0.5,
            max_lr_pairs: None,
            top_lr_pairs_by_mean_expression: None,
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
            epochs: 10,
            learning_rate: 1e-3,
            score_threshold: 0.0,
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
        }
    }
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

    pub fn full_cnn(&self) -> bool {
        !self.training.seed_only
    }

    pub fn resolve_adata_path(&self) -> String {
        if self.data.adata_path.is_empty() {
            std::env::var("SPACETRAVLR_H5AD").unwrap_or_else(|_| {
                "/ix/djishnu/shared/djishnu_kor11/training_data_2025/snrna_human_tonsil.h5ad"
                    .to_string()
            })
        } else {
            self.data.adata_path.clone()
        }
    }
}
