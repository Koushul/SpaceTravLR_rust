mod common;

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, ArrayBase, Data, Dimension};
use rctd_core::types::{
    SPOT_CLASS_DOUBLET_CERTAIN, SPOT_CLASS_DOUBLET_UNCERTAIN, SPOT_CLASS_REJECT, SPOT_CLASS_SINGLET,
};
use rctd_core::{
    build_x_vals, compute_q_matrix, compute_spline_coefficients, device_cpu, run_doublet_mode,
    run_full_mode, run_multi_mode, RctdConfig,
};

use common::{k_names, synthetic_pixel_data};

fn tables() -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let x_vals = build_x_vals();
    let q_mat = compute_q_matrix(100.0, &x_vals, 100);
    let sq_mat = compute_spline_coefficients(&q_mat, &x_vals);
    (q_mat, sq_mat, x_vals)
}

fn assert_all_finite<S: Data<Elem = f64>, D: Dimension>(a: &ArrayBase<S, D>) {
    assert!(a.iter().all(|x| x.is_finite()));
}

fn assert_all_finite_f32<S: Data<Elem = f32>, D: Dimension>(a: &ArrayBase<S, D>) {
    assert!(a.iter().all(|x| x.is_finite()));
}

#[cfg_attr(feature = "wgpu", ignore = "NdArray f64 parity; wgpu backend uses f32")]
#[test]
fn full_mode_batch_sizes_match() {
    let (counts, numi, profiles) = synthetic_pixel_data(42);
    let (q_mat, sq_mat, x_vals) = tables();
    let device = device_cpu();
    let a = run_full_mode(
        &counts, &numi, &profiles, &q_mat, &sq_mat, &x_vals, 16, &device, None,
    );
    let b = run_full_mode(
        &counts, &numi, &profiles, &q_mat, &sq_mat, &x_vals, 64, &device, None,
    );
    assert_eq!(a.weights.dim(), b.weights.dim());
    let max_abs = a
        .weights
        .iter()
        .zip(b.weights.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert_abs_diff_eq!(max_abs, 0.0, epsilon = 1e-9);
    assert_all_finite(&a.weights);
}

#[cfg_attr(feature = "wgpu", ignore = "NdArray f64 parity; wgpu uses f32")]
#[test]
fn doublet_mode_runs_with_sane_outputs() {
    let (counts, numi, profiles) = synthetic_pixel_data(42);
    let k = profiles.ncols();
    let (q_mat, sq_mat, x_vals) = tables();
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
        32,
        &device,
        None,
    );

    let n = counts.nrows();
    assert_eq!(res.weights.nrows(), n);
    assert_eq!(res.weights.ncols(), k);
    assert_all_finite(&res.weights);

    for &sc in res.spot_class.iter() {
        assert!(
            sc == SPOT_CLASS_REJECT
                || sc == SPOT_CLASS_SINGLET
                || sc == SPOT_CLASS_DOUBLET_CERTAIN
                || sc == SPOT_CLASS_DOUBLET_UNCERTAIN,
            "unexpected spot_class {sc}"
        );
    }
    for i in 0..n {
        let ft = res.first_type[i];
        let st = res.second_type[i];
        assert!((0..k as i32).contains(&ft));
        assert!((0..k as i32).contains(&st));
    }
    assert_all_finite_f32(&res.min_score);
    assert_all_finite_f32(&res.singlet_score);
}

#[cfg_attr(feature = "wgpu", ignore = "NdArray f64 parity; wgpu uses f32")]
#[test]
fn multi_mode_runs_with_sane_outputs() {
    let (counts, numi, profiles) = synthetic_pixel_data(42);
    let k = profiles.ncols();
    let max_t = RctdConfig::default().max_multi_types;
    let (q_mat, sq_mat, x_vals) = tables();
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
        32,
        &device,
        None,
    );

    let n = counts.nrows();
    assert_eq!(res.weights.nrows(), n);
    assert_eq!(res.weights.ncols(), k);
    assert_all_finite(&res.weights);
    assert_eq!(res.sub_weights.ncols(), max_t);
    assert_eq!(res.cell_type_indices.ncols(), max_t);
    assert_eq!(res.conf_list.ncols(), max_t);

    for i in 0..n {
        let nt = res.n_types[i];
        assert!(nt >= 0 && nt as usize <= max_t);
    }
    assert_all_finite_f32(&res.min_score);
}
