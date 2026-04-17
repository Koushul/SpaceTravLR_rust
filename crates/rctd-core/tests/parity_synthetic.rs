//! Numerical parity vs Python (PyTorch) on `fixtures/parity_synthetic.npz`.
//!
//! Regenerate: `uv run --project python python/scripts/export_rust_parity_fixtures.py`

use std::fs::File;
use std::path::Path;

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use rctd_core::{device_cpu, run_doublet_mode, run_full_mode, run_multi_mode, RctdConfig};

type ParityInputs = (
    Array2<f64>,
    Array1<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array1<f64>,
);

fn fixture_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/parity_synthetic.npz"
    ))
}

fn load_parity() -> ParityInputs {
    let path = fixture_path();
    assert!(
        path.is_file(),
        "missing {} — run: uv run --project python python/scripts/export_rust_parity_fixtures.py",
        path.display()
    );
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut npz = NpzReader::new(f).expect("npz");
    let counts: Array2<f64> = npz.by_name("counts").expect("counts");
    let numi: Array1<f64> = npz.by_name("numi").expect("numi");
    let profiles: Array2<f64> = npz.by_name("profiles").expect("profiles");
    let q_mat: Array2<f64> = npz.by_name("q_mat").expect("q_mat");
    let sq_mat: Array2<f64> = npz.by_name("sq_mat").expect("sq_mat");
    let x_vals: Array1<f64> = npz.by_name("x_vals").expect("x_vals");
    (counts, numi, profiles, q_mat, sq_mat, x_vals)
}

fn k_names(k: usize) -> Vec<String> {
    (0..k).map(|i| format!("t{i}")).collect()
}

#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity; wgpu uses f32")]
#[test]
fn full_mode_matches_python_fixture() {
    let (counts, numi, profiles, q_mat, sq_mat, x_vals) = load_parity();
    let mut npz = NpzReader::new(File::open(fixture_path()).unwrap()).unwrap();
    let py_w: Array2<f64> = npz.by_name("py_full_weights").expect("py_full_weights");
    let device = device_cpu();
    let res = run_full_mode(
        &counts, &numi, &profiles, &q_mat, &sq_mat, &x_vals, 64, &device, None,
    );
    assert_eq!(res.weights.dim(), py_w.dim());
    let max_abs = res
        .weights
        .iter()
        .zip(py_w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert_abs_diff_eq!(max_abs, 0.0, epsilon = 5e-5);
}

#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity")]
#[test]
fn doublet_mode_matches_python_fixture() {
    let (counts, numi, profiles, q_mat, sq_mat, x_vals) = load_parity();
    let mut npz = NpzReader::new(File::open(fixture_path()).unwrap()).unwrap();
    let py_w: Array2<f64> = npz.by_name("py_doublet_weights").unwrap();
    let py_wd: ndarray::Array2<f32> = npz.by_name("py_doublet_weights_doublet").unwrap();
    let py_sc: ndarray::Array1<i32> = npz.by_name("py_spot_class").unwrap();
    let py_ft: ndarray::Array1<i32> = npz.by_name("py_first_type").unwrap();
    let py_st: ndarray::Array1<i32> = npz.by_name("py_second_type").unwrap();
    let py_fc: ndarray::Array1<i8> = npz.by_name("py_first_class").unwrap();
    let py_syc: ndarray::Array1<i8> = npz.by_name("py_second_class").unwrap();
    let py_ms: ndarray::Array1<f32> = npz.by_name("py_min_score").unwrap();
    let py_ss: ndarray::Array1<f32> = npz.by_name("py_singlet_score").unwrap();

    let k = profiles.ncols();
    let cfg = RctdConfig::default();
    let device = device_cpu();
    let res = run_doublet_mode(
        &counts,
        &numi,
        &profiles,
        k_names(k),
        &q_mat,
        &sq_mat,
        &x_vals,
        &cfg,
        64,
        &device,
        None,
    );

    assert_eq!(res.weights.dim(), py_w.dim());
    let max_w = res
        .weights
        .iter()
        .zip(py_w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert_abs_diff_eq!(max_w, 0.0, epsilon = 5e-5);

    assert_eq!(res.weights_doublet.dim(), py_wd.dim());
    let max_wd = res
        .weights_doublet
        .iter()
        .zip(py_wd.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_wd < 1e-3,
        "weights_doublet max abs diff {max_wd} >= 1e-3"
    );

    assert_eq!(
        res.spot_class.as_slice().unwrap(),
        py_sc.as_slice().unwrap()
    );
    assert_eq!(
        res.first_type.as_slice().unwrap(),
        py_ft.as_slice().unwrap()
    );
    assert_eq!(
        res.second_type.as_slice().unwrap(),
        py_st.as_slice().unwrap()
    );
    assert_eq!(res.first_class.len(), py_fc.len());
    for (a, &b) in res.first_class.iter().zip(py_fc.iter()) {
        assert_eq!(*a, b != 0);
    }
    assert_eq!(res.second_class.len(), py_syc.len());
    for (a, &b) in res.second_class.iter().zip(py_syc.iter()) {
        assert_eq!(*a, b != 0);
    }

    let max_ms = res
        .min_score
        .iter()
        .zip(py_ms.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_ms < 0.02, "min_score max abs diff {max_ms}");

    let max_ss = res
        .singlet_score
        .iter()
        .zip(py_ss.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_ss < 0.02, "singlet_score max abs diff {max_ss}");
}

#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity")]
#[test]
fn multi_mode_matches_python_fixture() {
    let (counts, numi, profiles, q_mat, sq_mat, x_vals) = load_parity();
    let mut npz = NpzReader::new(File::open(fixture_path()).unwrap()).unwrap();
    let py_w: Array2<f64> = npz.by_name("py_multi_weights").unwrap();
    let py_sw: ndarray::Array2<f32> = npz.by_name("py_multi_sub_weights").unwrap();
    let py_ct: ndarray::Array2<i32> = npz.by_name("py_multi_cell_type_indices").unwrap();
    let py_nt: ndarray::Array1<i32> = npz.by_name("py_multi_n_types").unwrap();
    let py_cf: ndarray::Array2<i8> = npz.by_name("py_multi_conf_list").unwrap();
    let py_mn: ndarray::Array1<f32> = npz.by_name("py_multi_min_score").unwrap();

    let k = profiles.ncols();
    let cfg = RctdConfig::default();
    let device = device_cpu();
    let res = run_multi_mode(
        &counts,
        &numi,
        &profiles,
        k_names(k),
        &q_mat,
        &sq_mat,
        &x_vals,
        &cfg,
        64,
        &device,
        None,
    );

    let max_w = res
        .weights
        .iter()
        .zip(py_w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert_abs_diff_eq!(max_w, 0.0, epsilon = 5e-5);

    assert_eq!(res.sub_weights.dim(), py_sw.dim());
    let max_sw = res
        .sub_weights
        .iter()
        .zip(py_sw.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_sw < 1e-3, "sub_weights max abs {max_sw}");

    assert_eq!(
        res.cell_type_indices.as_slice().unwrap(),
        py_ct.as_slice().unwrap()
    );
    assert_eq!(res.n_types.as_slice().unwrap(), py_nt.as_slice().unwrap());
    assert_eq!(res.conf_list.dim(), py_cf.dim());
    for (a, &b) in res.conf_list.iter().zip(py_cf.iter()) {
        assert_eq!(*a, b != 0);
    }

    let max_mn = res
        .min_score
        .iter()
        .zip(py_mn.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_mn < 0.02, "multi min_score max abs {max_mn}");
}

#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity")]
#[test]
fn full_mode_batch_sizes_agree_on_fixture() {
    let (counts, numi, profiles, q_mat, sq_mat, x_vals) = load_parity();
    let device = device_cpu();
    let a = run_full_mode(
        &counts, &numi, &profiles, &q_mat, &sq_mat, &x_vals, 16, &device, None,
    );
    let b = run_full_mode(
        &counts, &numi, &profiles, &q_mat, &sq_mat, &x_vals, 64, &device, None,
    );
    let max_abs = a
        .weights
        .iter()
        .zip(b.weights.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    assert_abs_diff_eq!(max_abs, 0.0, epsilon = 1e-9);
}
