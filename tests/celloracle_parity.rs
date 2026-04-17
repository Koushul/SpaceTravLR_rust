//! Parity tests vs Python CellOracle / sklearn (golden fixtures under `tests/fixtures/celloracle_parity/golden/`).
//!
//! Golden JSON was produced by Python sklearn; regenerate manually if needed.

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use spacetravlr::celloracle::bayesian_ridge_fit;
use statrs::distribution::{ContinuousCDF, Normal};
use std::fs;
use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/celloracle_parity/golden")
}

fn read_json<T: for<'a> Deserialize<'a>>(name: &str) -> T {
    let p = golden_dir().join(format!("{name}.json"));
    let s = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

#[derive(Deserialize)]
struct StandardScalerFixture {
    #[allow(dead_code)]
    description: String,
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
    scale_: Vec<f64>,
    #[serde(rename = "X_transformed")]
    x_transformed: Vec<Vec<f64>>,
}

fn standard_scale_no_mean(x: &[Vec<f64>], scale: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let p = x[0].len();
    let mut out = vec![vec![0.0_f64; p]; n];
    for i in 0..n {
        for j in 0..p {
            out[i][j] = x[i][j] / scale[j];
        }
    }
    out
}

#[test]
fn parity_standard_scaler_no_mean() {
    let f: StandardScalerFixture = read_json("standard_scaler_no_mean");
    let got = standard_scale_no_mean(&f.x, &f.scale_);
    for (got_row, exp_row) in got.iter().zip(&f.x_transformed) {
        for (got_val, exp_val) in got_row.iter().zip(exp_row) {
            assert_relative_eq!(*got_val, *exp_val, max_relative = 1e-12);
        }
    }
}

#[derive(Deserialize)]
struct StatsBayesianFixture {
    #[allow(dead_code)]
    description: String,
    coef_mean: Vec<f64>,
    coef_variance: Vec<f64>,
    #[allow(dead_code)]
    coef_abs: Vec<f64>,
    p: Vec<f64>,
    neg_log_p: Vec<f64>,
}

fn stats_from_bayesian_ridge(coef_mean: &[f64], coef_variance: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let normal = Normal::new(0.0, 1.0).expect("std normal");
    let mut p = Vec::with_capacity(coef_mean.len());
    let mut neg = Vec::with_capacity(coef_mean.len());
    for i in 0..coef_mean.len() {
        let coef_abs = coef_mean[i].abs();
        let sig = coef_variance[i].sqrt();
        let tail = normal.cdf(-coef_abs / sig);
        let pi = 2.0 * tail;
        p.push(pi);
        neg.push(-pi.ln());
    }
    (p, neg)
}

#[test]
fn parity_stats_bayesian() {
    let f: StatsBayesianFixture = read_json("stats_bayesian");
    let (p, neg_log_p) = stats_from_bayesian_ridge(&f.coef_mean, &f.coef_variance);
    for i in 0..p.len() {
        assert_relative_eq!(p[i], f.p[i], max_relative = 1e-9, epsilon = 1e-14);
        assert_relative_eq!(
            neg_log_p[i],
            f.neg_log_p[i],
            max_relative = 1e-9,
            epsilon = 1e-14
        );
    }
}

#[derive(Deserialize)]
struct BayesianRidgeFixture {
    #[allow(dead_code)]
    description: String,
    #[allow(dead_code)]
    seed: u64,
    n_samples: usize,
    n_features: usize,
    #[serde(rename = "X_celloracle_scaled")]
    x_celloracle_scaled: Vec<Vec<f64>>,
    y: Vec<f64>,
    #[serde(rename = "coef_")]
    coef: Vec<f64>,
    #[serde(rename = "intercept_")]
    intercept: f64,
    #[serde(rename = "sigma_diag")]
    sigma_diag: Vec<f64>,
    #[serde(rename = "alpha_")]
    alpha: f64,
    #[serde(rename = "lambda_")]
    lambda: f64,
    #[allow(dead_code)]
    #[serde(rename = "n_iter_")]
    n_iter: usize,
}

fn matrix_from_rows(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let n = rows.len();
    let p = rows[0].len();
    DMatrix::from_fn(n, p, |i, j| rows[i][j])
}

fn assert_bayesian_ridge_parity(f: BayesianRidgeFixture) {
    let x = matrix_from_rows(&f.x_celloracle_scaled);
    let y = DVector::from_vec(f.y.clone());
    assert_eq!(x.nrows(), f.n_samples);
    assert_eq!(x.ncols(), f.n_features);
    let got = bayesian_ridge_fit(&x, &y).expect("bayesian_ridge_fit");
    assert_relative_eq!(
        got.intercept,
        f.intercept,
        max_relative = 1e-6,
        epsilon = 1e-9
    );
    assert_relative_eq!(got.alpha, f.alpha, max_relative = 1e-5, epsilon = 1e-9);
    assert_relative_eq!(got.lambda, f.lambda, max_relative = 1e-5, epsilon = 1e-9);
    for i in 0..f.n_features {
        assert_relative_eq!(got.coef[i], f.coef[i], max_relative = 1e-5, epsilon = 1e-8);
        assert_relative_eq!(
            got.sigma_diag[i],
            f.sigma_diag[i],
            max_relative = 1e-4,
            epsilon = 1e-8
        );
    }
}

#[test]
fn parity_bayesian_ridge_tiny() {
    let f: BayesianRidgeFixture = read_json("bayesian_ridge_tiny");
    assert_bayesian_ridge_parity(f);
}

#[test]
fn parity_bayesian_ridge_wide() {
    let f: BayesianRidgeFixture = read_json("bayesian_ridge_wide");
    assert_bayesian_ridge_parity(f);
}
