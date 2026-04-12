//! RCTD core: Poisson–Lognormal likelihood + IRWLS (Burn tensors; CPU `NdArray<f64>` or optional GPU `Wgpu<f32>`).
#![recursion_limit = "256"]

pub mod backend;
pub mod calc_q;
pub mod doublet;
pub mod full;
pub mod io_npz;
pub mod irwls;
pub mod irwls_native;
pub mod likelihood_tables;
pub mod linalg;
pub mod multi;
pub mod normalize;
pub mod pipeline;
pub mod qp;
pub mod sigma;
pub mod simplex;
pub mod types;

#[cfg(feature = "wgpu")]
pub use backend::init_wgpu;
pub use backend::{
    default_device, fe, scalar_to_f64, slice_elems_to_f64, sync_device, FloatElem, RctdBackend,
    RctdDevice,
};
pub use calc_q::{calc_log_likelihood_batch, calc_q_all, device_cpu};
pub use doublet::run_doublet_mode;
pub use full::{run_full_mode, FullResult};
pub use io_npz::{load_q_matrices_npz, QMatrixMap};
pub use likelihood_tables::{build_x_vals, compute_q_matrix, compute_spline_coefficients};
pub use multi::run_multi_mode;
pub use pipeline::{run_deconvolution, DeconvMode, DeconvolutionOutput, PreparedData};
pub use sigma::{choose_sigma, SIGMA_ALL};
pub use types::{BatchProgress, DoubletResult, MultiResult, RctdConfig, notify_batch};
