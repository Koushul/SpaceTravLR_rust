use ndarray::{Array1, Array2};

use crate::backend::RctdDevice;

use crate::doublet::run_doublet_mode;
use crate::full::{run_full_mode, FullResult};
use crate::multi::run_multi_mode;
use crate::types::{BatchProgress, DoubletResult, MultiResult, RctdConfig};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeconvMode {
    Full,
    Doublet,
    Multi,
}

pub struct PreparedData {
    pub spatial_counts: Array2<f64>,
    pub spatial_numi: Array1<f64>,
    pub norm_profiles: Array2<f64>,
    pub cell_type_names: Vec<String>,
    pub q_mat: Array2<f64>,
    pub sq_mat: Array2<f64>,
    pub x_vals: Array1<f64>,
}

pub enum DeconvolutionOutput {
    Full(FullResult),
    Doublet(DoubletResult),
    Multi(MultiResult),
}

pub fn run_deconvolution(
    data: &PreparedData,
    config: &RctdConfig,
    mode: DeconvMode,
    batch_size: usize,
    device: &RctdDevice,
    progress: Option<BatchProgress>,
) -> DeconvolutionOutput {
    match mode {
        DeconvMode::Full => DeconvolutionOutput::Full(run_full_mode(
            &data.spatial_counts,
            &data.spatial_numi,
            &data.norm_profiles,
            &data.q_mat,
            &data.sq_mat,
            &data.x_vals,
            batch_size,
            device,
            progress,
        )),
        DeconvMode::Doublet => DeconvolutionOutput::Doublet(run_doublet_mode(
            &data.spatial_counts,
            &data.spatial_numi,
            &data.norm_profiles,
            data.cell_type_names.clone(),
            &data.q_mat,
            &data.sq_mat,
            &data.x_vals,
            config,
            batch_size,
            device,
            progress,
        )),
        DeconvMode::Multi => DeconvolutionOutput::Multi(run_multi_mode(
            &data.spatial_counts,
            &data.spatial_numi,
            &data.norm_profiles,
            data.cell_type_names.clone(),
            &data.q_mat,
            &data.sq_mat,
            &data.x_vals,
            config,
            batch_size,
            device,
            progress,
        )),
    }
}
