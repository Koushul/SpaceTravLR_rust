use ndarray::{Array1, Array2};

pub(crate) fn scale_columns_no_center(x: &mut Array2<f64>) -> Array1<f64> {
    let (n, p) = x.dim();
    let mut scales = Array1::<f64>::ones(p);
    for j in 0..p {
        let col = x.column(j);
        let m = col.sum() / n as f64;
        let var: f64 = col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
        let sc = var.sqrt();
        if sc > 1e-12 {
            scales[j] = sc;
            x.column_mut(j).mapv_inplace(|v| v / sc);
        }
    }
    scales
}

pub(crate) fn apply_modulator_scales_inplace(x: &mut Array2<f64>, scales: &Array1<f64>) {
    for j in 0..x.ncols().min(scales.len()) {
        let s = scales[j];
        if s != 1.0 {
            x.column_mut(j).mapv_inplace(|v| v / s);
        }
    }
}

pub(crate) fn unscale_betadata_columns_inplace(betas: &mut Array2<f64>, scales: &Array1<f64>) {
    for j in 1..betas.ncols().min(scales.len() + 1) {
        let s = scales[j - 1];
        if s != 1.0 {
            betas.column_mut(j).mapv_inplace(|v| v / s);
        }
    }
}
