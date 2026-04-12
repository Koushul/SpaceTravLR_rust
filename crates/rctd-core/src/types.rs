use std::sync::Arc;

use ndarray::{Array1, Array2};

pub type BatchProgress = Arc<dyn Fn(usize) + Send + Sync>;

#[inline]
pub fn notify_batch(progress: &Option<BatchProgress>, units: usize) {
    if units == 0 {
        return;
    }
    if let Some(f) = progress {
        f(units);
    }
}

#[derive(Clone, Debug)]
pub struct RctdConfig {
    pub confidence_threshold: f64,
    pub doublet_threshold: f64,
    pub max_multi_types: usize,
    pub umi_min: i32,
    pub umi_min_sigma: i32,
    pub n_fit: usize,
    pub n_epoch: usize,
    pub k_val: i64,
}

impl Default for RctdConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 5.0,
            doublet_threshold: 20.0,
            max_multi_types: 4,
            umi_min: 100,
            umi_min_sigma: 300,
            n_fit: 100,
            n_epoch: 8,
            k_val: 1000,
        }
    }
}

pub const SPOT_CLASS_REJECT: i32 = 0;
pub const SPOT_CLASS_SINGLET: i32 = 1;
pub const SPOT_CLASS_DOUBLET_CERTAIN: i32 = 2;
pub const SPOT_CLASS_DOUBLET_UNCERTAIN: i32 = 3;

#[derive(Clone, Debug)]
pub struct DoubletResult {
    pub weights: Array2<f64>,
    pub weights_doublet: Array2<f32>,
    pub spot_class: Array1<i32>,
    pub first_type: Array1<i32>,
    pub second_type: Array1<i32>,
    pub first_class: Array1<bool>,
    pub second_class: Array1<bool>,
    pub min_score: Array1<f32>,
    pub singlet_score: Array1<f32>,
    pub cell_type_names: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct MultiResult {
    pub weights: Array2<f64>,
    pub sub_weights: Array2<f32>,
    pub cell_type_indices: Array2<i32>,
    pub n_types: Array1<i32>,
    pub conf_list: Array2<bool>,
    pub min_score: Array1<f32>,
    pub cell_type_names: Vec<String>,
}
