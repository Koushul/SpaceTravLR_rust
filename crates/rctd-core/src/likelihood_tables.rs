//! Likelihood grid and Q-matrix construction (matches Python `rctd._likelihood`).

use nalgebra::DMatrix;
use ndarray::{s, Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;

fn normal_cdf(x: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().cdf(x)
}

/// Heavy-tailed PDF (normalized), ports `ht_pdf_norm` from R.
pub fn ht_pdf_norm(x: &Array1<f64>) -> Array1<f64> {
    let a = 4.0 / 9.0 * (-9.0_f64 / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let c = 7.0 / 3.0;
    let cap_c = 1.0 / ((a / (3.0 - c) - normal_cdf(-3.0)) * 2.0 + 1.0);
    let mut p = Array1::zeros(x.len());
    for (i, &xi) in x.iter().enumerate() {
        p[i] = if xi.abs() < 3.0 {
            cap_c / (2.0 * std::f64::consts::PI).sqrt() * (-(xi * xi) / 2.0).exp()
        } else {
            cap_c * a / (xi.abs() - c).powi(2)
        };
    }
    p
}

/// Reconstructs the X_vals grid used by spacexr (439 points).
pub fn build_x_vals() -> Array1<f64> {
    let delta = 1e-6;
    let max_l = 40000usize;
    let mut m_to_first_l = std::collections::HashMap::new();
    for l in 10..=max_l {
        let lf = l as f64;
        let m = (l - 9).min(40)
            + (((lf - 48.7499).max(0.0) * 4.0).sqrt().ceil() as i64 - 2).max(0) as usize;
        m_to_first_l.entry(m).or_insert(l);
    }
    let n_grid = 439usize;
    let mut x_vals = Array1::zeros(n_grid);
    for m in 1..n_grid {
        let l = *m_to_first_l.get(&m).expect("m in range");
        x_vals[m - 1] = (l as f64).powi(2) * delta;
    }
    x_vals[n_grid - 1] = 1600.0;
    x_vals
}

fn get_q_single(x_vals: &Array1<f64>, k: i32, sigma: f64) -> Array1<f64> {
    const N_Y: usize = 5000;
    let gamma_step = 4e-3;
    let n_x = x_vals.len();
    let mut y = Array1::zeros(2 * N_Y + 1);
    for j in 0..y.len() {
        y[j] = (j as isize - N_Y as isize) as f64 * gamma_step;
    }
    let ht_vals = ht_pdf_norm(&(&y / sigma)) / sigma;
    let log_p: Vec<f64> = ht_vals.iter().map(|&v| (v.max(1e-300)).ln()).collect();
    let lg_k_fact = ln_gamma((k + 1) as f64);
    let mut results = Array1::zeros(n_x);
    let batch = 100usize;
    let kf = k as f64;
    for b in (0..n_x).step_by(batch) {
        let end = (b + batch).min(n_x);
        let curr_x = x_vals.slice(s![b..end]);
        for (i, &xv) in curr_x.iter().enumerate() {
            let mut sum = 0.0;
            for j in 0..y.len() {
                let log_s = -y[j].exp() * xv + kf * y[j] + log_p[j] - lg_k_fact + kf * xv.ln();
                sum += log_s.exp();
            }
            results[b + i] = sum * gamma_step;
        }
    }
    results
}

/// Q_mat[k, i] = log P(Y=k | lambda=x_vals[i]).
pub fn compute_q_matrix(sigma: f64, x_vals: &Array1<f64>, k_val: usize) -> Array2<f64> {
    let n_rows = k_val + 3;
    let n_x = x_vals.len();
    let mut q_mat = Array2::zeros((n_rows, n_x));
    for k in 0..n_rows {
        let raw = get_q_single(x_vals, k as i32, sigma);
        for j in 0..n_x {
            q_mat[[k, j]] = (raw[j].max(1e-300)).ln();
        }
    }
    q_mat
}

fn tridiagonal_mi(x_vals: &Array1<f64>) -> Array2<f64> {
    let n = x_vals.len() - 1;
    let delta: Vec<f64> = x_vals
        .iter()
        .zip(x_vals.iter().skip(1))
        .map(|(a, b)| b - a)
        .collect();
    let inner = n - 1;
    let mut m = DMatrix::from_element(inner, inner, 0.0);
    for i in 0..inner {
        m[(i, i)] = 2.0 * (delta[i] + delta[i + 1]);
    }
    for i in 0..inner - 1 {
        let v = delta[i + 1];
        m[(i + 1, i)] = v;
        m[(i, i + 1)] = v;
    }
    let mi = m.try_inverse().expect("invert tridiagonal M");
    Array2::from_shape_fn((inner, inner), |(r, c)| mi[(r, c)])
}

/// Cubic spline second-derivative coefficients (SQ_mat).
pub fn compute_spline_coefficients(q_mat: &Array2<f64>, x_vals: &Array1<f64>) -> Array2<f64> {
    let delta = &x_vals.slice(s![1..]) - &x_vals.slice(s![..-1]);
    let mi = tridiagonal_mi(x_vals);
    let diff_q = &q_mat.slice(s![.., 1..]) - &q_mat.slice(s![.., ..-1]);
    let delta_row = delta.to_shape((1, delta.len())).unwrap();
    let f_b = diff_q / &delta_row;
    let f_bd = 6.0 * (&f_b.slice(s![.., 1..]) - &f_b.slice(s![.., ..-1]));
    let sq_inner = f_bd.dot(&mi.t());
    let k = q_mat.nrows();
    let nx = q_mat.ncols();
    let mut sq_mat = Array2::zeros((k, nx));
    sq_mat.slice_mut(s![.., 1..nx - 1]).assign(&sq_inner);
    sq_mat
}
