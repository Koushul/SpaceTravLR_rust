use ndarray::{Array1, Array2};

use crate::backend::RctdDevice;
use crate::irwls_native::{solve_irwls_native, NativeSharedPrepared};
use crate::types::{notify_batch, BatchProgress};

pub struct FullResult {
    pub weights: Array2<f64>,
    pub converged: Array1<bool>,
}

/// Full-mode IRWLS over all pixels.
///
/// Always uses the native ndarray/rayon solver. The Burn tensor path repeatedly
/// round-trips through `into_data`/`from_data` for PSD, box-QP, and simplex each
/// iteration, so it is slower on CPU and not a true GPU win under `wgpu` either.
#[allow(clippy::too_many_arguments)]
pub fn run_full_mode(
    spatial_counts: &Array2<f64>,
    spatial_numi: &Array1<f64>,
    norm_profiles: &Array2<f64>,
    q_mat: &Array2<f64>,
    sq_mat: &Array2<f64>,
    x_vals: &Array1<f64>,
    batch_size: usize,
    _device: &RctdDevice,
    progress: Option<BatchProgress>,
) -> FullResult {
    let n = spatial_counts.nrows();
    let k = norm_profiles.ncols();
    if n == 0 {
        return FullResult {
            weights: Array2::zeros((0, k)),
            converged: Array1::default(0),
        };
    }
    let prep = NativeSharedPrepared::new(norm_profiles, q_mat, sq_mat, x_vals);
    let mut all_w = Vec::new();
    let mut all_c = Vec::new();
    for start in (0..n).step_by(batch_size.max(1)) {
        let end = (start + batch_size).min(n);
        let y_v = spatial_counts.slice(ndarray::s![start..end, ..]);
        let n_v = spatial_numi.slice(ndarray::s![start..end]);
        let (w, c) = solve_irwls_native(&prep, y_v, n_v, 50, 0.001, 0.3, false, false);
        all_w.push(w);
        all_c.push(c);
        notify_batch(&progress, end - start);
    }
    let mut weights = all_w[0].clone();
    for a in all_w.iter().skip(1) {
        weights.append(ndarray::Axis(0), a.view()).unwrap();
    }
    let mut converged = all_c[0].clone();
    for a in all_c.iter().skip(1) {
        converged.append(ndarray::Axis(0), a.view()).unwrap();
    }
    FullResult { weights, converged }
}
