//! Parity tests vs Python CellOracle / sklearn (golden fixtures under `tests/fixtures/celloracle_parity/golden/`).
//!
//! Golden JSON was produced by Python sklearn; regenerate manually if needed.

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};
use ndarray::Array2;
use serde::Deserialize;
use spacetravlr::celloracle::{bayesian_ridge_fit, scale_gem_no_center, two_sided_p_celloracle};
use statrs::distribution::Normal;
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
    #[allow(dead_code)]
    scale_: Vec<f64>,
    #[serde(rename = "X_transformed")]
    x_transformed: Vec<Vec<f64>>,
}

#[test]
fn parity_standard_scaler_no_mean() {
    let f: StandardScalerFixture = read_json("standard_scaler_no_mean");
    let n = f.x.len();
    let p = f.x[0].len();
    let gem = Array2::from_shape_fn((n, p), |(i, j)| f.x[i][j]);
    let got = scale_gem_no_center(&gem);
    for i in 0..n {
        for j in 0..p {
            assert_relative_eq!(
                got[[i, j]],
                f.x_transformed[i][j],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
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

#[test]
fn parity_stats_bayesian() {
    let f: StatsBayesianFixture = read_json("stats_bayesian");
    let normal = Normal::new(0.0, 1.0).expect("std normal");
    for i in 0..f.coef_mean.len() {
        let (p, neg_log_p) = two_sided_p_celloracle(&normal, f.coef_mean[i], f.coef_variance[i]);
        assert_relative_eq!(p, f.p[i], max_relative = 1e-9, epsilon = 1e-14);
        assert_relative_eq!(
            neg_log_p,
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
