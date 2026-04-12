use burn::tensor::Tensor;

use crate::backend::{
    scalar_to_f64, tensor1_from_f64, tensor2_from_f64, FloatElem, RctdBackend, RctdDevice,
};
use crate::irwls::solve_irwls_single_bulk;

/// Platform-effect normalization (`fit_bulk` + `get_norm_ref` logic).
pub fn fit_bulk(
    cell_type_profiles: &ndarray::Array2<f64>,
    spatial_counts: &ndarray::Array2<f64>,
    spatial_numi: &ndarray::Array1<f64>,
    device: &RctdDevice,
) -> (ndarray::Array1<f64>, ndarray::Array2<f64>) {
    let g = cell_type_profiles.nrows();
    let k = cell_type_profiles.ncols();
    let bulk_y: ndarray::Array1<f64> = spatial_counts.sum_axis(ndarray::Axis(0));
    let bulk_numi: f64 = spatial_numi.sum();
    let bulk_s = cell_type_profiles * bulk_numi;
    let s_t = tensor2_from(&bulk_s, device);
    let y_t = tensor1_from(bulk_y.view(), device);
    let w_t = solve_irwls_single_bulk(&s_t, &y_t, bulk_numi, device);
    let w_d = w_t.into_data();
    let w_sl = w_d.as_slice::<FloatElem>().unwrap();
    let mut bulk_weights = ndarray::Array1::zeros(k);
    for i in 0..k {
        bulk_weights[i] = scalar_to_f64(w_sl[i]).max(0.0);
    }
    let prop_sum: f64 = bulk_weights.sum().max(1e-10);
    let prop_n = &bulk_weights / prop_sum;
    let weight_avg = cell_type_profiles.dot(&prop_n);
    let target_means = &bulk_y / bulk_numi.max(1e-10);
    let gene_factor = &weight_avg / &target_means.mapv(|t: f64| t.max(1e-10));
    let mut norm = ndarray::Array2::zeros((g, k));
    for j in 0..k {
        let col = cell_type_profiles.column(j);
        norm.column_mut(j)
            .assign(&(&col / gene_factor.mapv(|f: f64| f.max(1e-10))));
    }
    (bulk_weights, norm)
}

fn tensor1_from(a: ndarray::ArrayView1<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 1> {
    tensor1_from_f64(&a.to_owned(), dev)
}

fn tensor2_from(a: &ndarray::Array2<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 2> {
    tensor2_from_f64(a, dev)
}
