use crate::backend::RctdDevice;
use crate::irwls_native::solve_irwls_single_bulk_native;

/// Platform-effect normalization (`fit_bulk` + `get_norm_ref` logic).
pub fn fit_bulk(
    cell_type_profiles: &ndarray::Array2<f64>,
    spatial_counts: &ndarray::Array2<f64>,
    spatial_numi: &ndarray::Array1<f64>,
    _device: &RctdDevice,
) -> (ndarray::Array1<f64>, ndarray::Array2<f64>) {
    let g = cell_type_profiles.nrows();
    let k = cell_type_profiles.ncols();
    let bulk_y: ndarray::Array1<f64> = spatial_counts.sum_axis(ndarray::Axis(0));
    let bulk_numi: f64 = spatial_numi.sum();
    let bulk_s = cell_type_profiles * bulk_numi;
    let mut bulk_weights = solve_irwls_single_bulk_native(&bulk_s, &bulk_y, bulk_numi);
    for w in bulk_weights.iter_mut() {
        *w = w.max(0.0);
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
