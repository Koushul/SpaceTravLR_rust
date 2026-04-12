//! Likelihood grid: Rust `build_x_vals` / `compute_q_matrix` match values shipped from Python in the parity fixture.

use std::fs::File;
use std::path::Path;

use approx::assert_abs_diff_eq;
use ndarray::Array1;
use ndarray_npy::NpzReader;
use rctd_core::{build_x_vals, compute_q_matrix, compute_spline_coefficients};

fn fixture_npz() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/parity_synthetic.npz"
    ))
}

#[test]
fn build_x_vals_shape_and_endpoint() {
    let x = build_x_vals();
    assert_eq!(x.len(), 439);
    assert_abs_diff_eq!(x[x.len() - 1], 1600.0, epsilon = 1e-12);
}

#[test]
fn compute_q_matrix_matches_python_exported_table() {
    let path = fixture_npz();
    assert!(
        path.is_file(),
        "missing {} — run: uv run --project python python/scripts/export_rust_parity_fixtures.py",
        path.display()
    );
    let mut npz = NpzReader::new(File::open(path).unwrap()).unwrap();
    let py_x: Array1<f64> = npz.by_name("x_vals").unwrap();
    let py_q: ndarray::Array2<f64> = npz.by_name("q_mat").unwrap();

    let x_r = build_x_vals();
    assert_eq!(x_r.len(), py_x.len());
    let max_x = x_r
        .iter()
        .zip(py_x.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert_abs_diff_eq!(max_x, 0.0, epsilon = 1e-14);

    let k_val = 100usize;
    let sigma = 100.0f64;
    let q_r = compute_q_matrix(sigma, &x_r, k_val);
    assert_eq!(q_r.dim(), py_q.dim());
    let max_q = q_r
        .iter()
        .zip(py_q.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert_abs_diff_eq!(max_q, 0.0, epsilon = 1e-9);

    let py_sq: ndarray::Array2<f64> = npz.by_name("sq_mat").unwrap();
    let sq_r = compute_spline_coefficients(&q_r, &x_r);
    assert_eq!(sq_r.dim(), py_sq.dim());
    let max_sq = sq_r
        .iter()
        .zip(py_sq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert_abs_diff_eq!(max_sq, 0.0, epsilon = 5e-4);
}
