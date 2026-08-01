use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnTrainHyperparams {
    pub learning_rate: f64,
    pub epochs: u32,
    pub adam_beta_1: f32,
    pub adam_beta_2: f32,
    pub adam_epsilon: f32,
    pub weight_decay: Option<f32>,
    pub grad_clip_norm: Option<f32>,
    pub mean_beta_lasso_prior_weight: f32,
    pub lasso_pred_align_weight: f32,
    pub cnn_minibatch_size: u32,
    pub cnn_max_cells_per_epoch: Option<u32>,
    pub cnn_early_stop_patience: u32,
    pub cnn_early_stop_min_epochs: u32,
    pub lr_schedule_cosine: bool,
    pub cosine_lr_min_ratio: f64,
    pub lr_warmup_epochs: u32,
    /// 0=identity 1=sigmoid 2=tanh 3=sigmoid-x2
    pub output_activation: u8,
    pub shuffle_seed: u64,
}

impl Default for CnnTrainHyperparams {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 8,
            adam_beta_1: 0.9,
            adam_beta_2: 0.999,
            adam_epsilon: 1e-5,
            weight_decay: None,
            grad_clip_norm: Some(3.0),
            mean_beta_lasso_prior_weight: 0.005,
            lasso_pred_align_weight: 0.0,
            cnn_minibatch_size: 64,
            cnn_max_cells_per_epoch: Some(512),
            cnn_early_stop_patience: 0,
            cnn_early_stop_min_epochs: 0,
            lr_schedule_cosine: true,
            cosine_lr_min_ratio: 0.01,
            lr_warmup_epochs: 0,
            output_activation: 3,
            shuffle_seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnClusterPack {
    pub cluster_id: u32,
    pub n_cells: u32,
    pub n_modulators: u32,
    pub n_clusters: u32,
    pub spatial_h: u32,
    pub spatial_w: u32,
    pub vision_in_channels: u32,
    /// Row-major `[n_cells, channels, H, W]` f32
    pub spatial_maps: Vec<f32>,
    /// Row-major `[n_cells, n_modulators]` f32 (modulator-scaled X)
    pub x: Vec<f32>,
    /// Row-major `[n_cells, n_clusters]` f32
    pub spatial_features: Vec<f32>,
    pub y: Vec<f32>,
    /// Length `n_modulators + 1` (intercept + coefs)
    pub anchors: Vec<f32>,
    pub y_lasso: Option<Vec<f32>>,
    pub lasso_r2: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnGeneTrainPack {
    pub gene: String,
    pub hyperparams: CnnTrainHyperparams,
    pub clusters: Vec<CnnClusterPack>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTrainResult {
    pub cluster_id: u32,
    pub n_cells: u32,
    pub lasso_r2: f32,
    pub mse_epochs: Vec<f32>,
    pub diverged: bool,
    pub wall_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneTrainResult {
    pub gene: String,
    pub clusters: Vec<ClusterTrainResult>,
    pub wall_ms: u32,
}

pub fn encode_pack(pack: &CnnGeneTrainPack) -> Result<Vec<u8>, bincode::Error> {
    bincode::serialize(pack)
}

pub fn decode_pack(bytes: &[u8]) -> Result<CnnGeneTrainPack, bincode::Error> {
    bincode::deserialize(bytes)
}
