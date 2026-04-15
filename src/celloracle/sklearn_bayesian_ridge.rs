//! sklearn-compatible `BayesianRidge` (MacKay / Tipping), `fit_intercept=True`.
//! Used for CellOracle edge statistics (coef + posterior variance diagonal).

use nalgebra::{DMatrix, DVector, SVD};

const ALPHA_1: f64 = 1e-6;
const ALPHA_2: f64 = 1e-6;
const LAMBDA_1: f64 = 1e-6;
const LAMBDA_2: f64 = 1e-6;
const MAX_ITER: usize = 300;
const TOL: f64 = 1e-3;

pub struct BayesianRidgeFit {
    pub coef: Vec<f64>,
    pub intercept: f64,
    pub sigma_diag: Vec<f64>,
    pub alpha: f64,
    pub lambda: f64,
    pub n_iter: usize,
}

fn y_variance_population(y: &DVector<f64>) -> f64 {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
}

fn preprocess_center(
    x_orig: &DMatrix<f64>,
    y_orig: &DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>, f64) {
    let n = x_orig.nrows();
    let p = x_orig.ncols();
    let mut x = x_orig.clone();
    let mut x_offset = DVector::zeros(p);
    for j in 0..p {
        x_offset[j] = (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64;
    }
    for j in 0..p {
        for i in 0..n {
            x[(i, j)] -= x_offset[j];
        }
    }
    let y_offset = y_orig.iter().sum::<f64>() / n as f64;
    let mut y = y_orig.clone();
    for yi in y.iter_mut() {
        *yi -= y_offset;
    }
    (x, y, x_offset, y_offset)
}

fn update_coef_n_gt_p(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    xt_y: &DVector<f64>,
    v_t: &DMatrix<f64>,
    eigen_vals: &[f64],
    alpha: f64,
    lambda: f64,
) -> (DVector<f64>, f64) {
    let p = xt_y.len();
    let k = eigen_vals.len();
    let mut scaled = v_t.clone();
    for i in 0..k {
        let d = eigen_vals[i] + lambda / alpha;
        for j in 0..p {
            scaled[(i, j)] /= d;
        }
    }
    let tmp = &scaled * xt_y;
    let coef = v_t.transpose() * tmp;
    let pred = x * &coef;
    let sse: f64 = (0..y.len()).map(|i| (y[i] - pred[i]).powi(2)).sum();
    (coef, sse)
}

fn posterior_sigma_diag(
    v_t: &DMatrix<f64>,
    eigen_full: &[f64],
    alpha: f64,
    lambda: f64,
) -> Vec<f64> {
    let p = v_t.ncols();
    let mut scaled = v_t.clone();
    for i in 0..p {
        let d = alpha * eigen_full[i] + lambda;
        for j in 0..p {
            scaled[(i, j)] /= d;
        }
    }
    let sigma = v_t.transpose() * &scaled;
    (0..p).map(|i| sigma[(i, i)]).collect()
}

#[allow(unused_assignments)]
pub fn bayesian_ridge_fit(x_celloracle: &DMatrix<f64>, y: &DVector<f64>) -> Option<BayesianRidgeFit> {
    let n_samples = x_celloracle.nrows();
    let n_features = x_celloracle.ncols();
    if n_samples <= n_features {
        return None;
    }
    assert_eq!(y.len(), n_samples);

    let y_var = y_variance_population(y);
    let eps = f64::EPSILON;
    let mut alpha = 1.0 / (y_var + eps);
    let mut lambda = 1.0_f64;

    let (x, y_c, x_offset, y_offset) = preprocess_center(x_celloracle, y);
    let xt_y = x.transpose() * &y_c;

    let svd = SVD::new(x.clone(), true, true);
    let v_t = svd.v_t.expect("SVD V^T");
    let s = svd.singular_values;
    let k = s.len();
    let mut eigen_vals = vec![0.0_f64; k];
    for i in 0..k {
        eigen_vals[i] = s[i] * s[i];
    }

    let mut eigen_full = vec![0.0_f64; n_features];
    let take = k.min(n_features);
    eigen_full[..take].copy_from_slice(&eigen_vals[..take]);

    let mut coef_old: Option<DVector<f64>> = None;
    let mut coef = DVector::zeros(n_features);
    let mut sse = 0.0;
    let mut iter_done = MAX_ITER;

    for it in 0..MAX_ITER {
        let (c, s_) = update_coef_n_gt_p(&x, &y_c, &xt_y, &v_t, &eigen_vals, alpha, lambda);
        coef = c;
        sse = s_;

        let gamma: f64 = (0..k)
            .map(|i| (alpha * eigen_vals[i]) / (lambda + alpha * eigen_vals[i]))
            .sum();
        lambda =
            (gamma + 2.0 * LAMBDA_1) / (coef.iter().map(|v| v * v).sum::<f64>() + 2.0 * LAMBDA_2);
        alpha = (n_samples as f64 - gamma + 2.0 * ALPHA_1) / (sse + 2.0 * ALPHA_2);

        if let Some(ref old) = coef_old {
            let delta: f64 = (0..n_features).map(|i| (old[i] - coef[i]).abs()).sum();
            if delta < TOL {
                iter_done = it + 1;
                break;
            }
        }
        coef_old = Some(coef.clone());
    }

    let (c, _) = update_coef_n_gt_p(&x, &y_c, &xt_y, &v_t, &eigen_vals, alpha, lambda);
    coef = c;
    let sigma_diag = posterior_sigma_diag(&v_t, &eigen_full, alpha, lambda);
    let dot_xo_c: f64 = (0..n_features).map(|j| x_offset[j] * coef[j]).sum();
    let intercept = y_offset - dot_xo_c;

    Some(BayesianRidgeFit {
        coef: coef.as_slice().to_vec(),
        intercept,
        sigma_diag,
        alpha,
        lambda,
        n_iter: iter_done,
    })
}
