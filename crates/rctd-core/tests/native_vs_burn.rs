use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use rctd_core::backend::{tensor1_from_f64, tensor2_from_f64};
use rctd_core::irwls::solve_irwls_batch_shared;
use rctd_core::irwls_native::{
    calc_q_d0_flat, solve_irwls_native, solve_irwls_single_bulk_native, NativeSharedPrepared,
};
use rctd_core::{
    build_x_vals, calc_q_all, compute_q_matrix, compute_spline_coefficients, device_cpu,
    slice_elems_to_f64, FloatElem,
};

mod common;

#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity; wgpu uses f32")]
#[test]
fn native_irwls_matches_burn_on_synthetic() {
    let (counts, numi, profiles) = common::synthetic_pixel_data(7);
    let x = build_x_vals();
    let q = compute_q_matrix(100.0, &x, 100);
    let sq = compute_spline_coefficients(&q, &x);
    let device = device_cpu();

    let prep = NativeSharedPrepared::new(&profiles, &q, &sq, &x);
    let (w_n, _) = solve_irwls_native(&prep, counts.view(), numi.view(), 50, 0.001, 0.3, false, false);
    let (w_b, _) = solve_irwls_batch_shared(
        &profiles, &counts, &numi, &q, &sq, &x, 50, 0.001, 0.3, false, false, &device,
    );

    assert_eq!(w_n.dim(), w_b.dim());
    for (a, b) in w_n.iter().zip(w_b.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-8);
    }
}

#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity; wgpu uses f32")]
#[test]
fn native_calc_q_d0_matches_burn() {
    let y = Array1::from(vec![0.0, 1.0, 2.0, 5.0, 10.0]);
    let lam = Array1::from(vec![0.5, 1.2, 2.5, 4.0, 8.0]);
    let x = build_x_vals();
    let q = compute_q_matrix(100.0, &x, 100);
    let sq = compute_spline_coefficients(&q, &x);
    let device = device_cpu();

    let d0_n = calc_q_d0_flat(
        y.as_slice().unwrap(),
        lam.as_slice().unwrap(),
        &q,
        &sq,
        &x,
        -1,
    );
    let (d0_b, _, _) = calc_q_all(
        tensor1_from_f64(&y, &device),
        tensor1_from_f64(&lam, &device),
        tensor2_from_f64(&q, &device),
        tensor2_from_f64(&sq, &device),
        tensor1_from_f64(&x, &device),
        -1,
    );
    let d0_b = slice_elems_to_f64(d0_b.into_data().as_slice::<FloatElem>().unwrap());
    assert_eq!(d0_n.len(), d0_b.len());
    for (a, b) in d0_n.iter().zip(d0_b.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-9);
    }
}

#[test]
fn native_bulk_single_finite() {
    let s = Array2::from_elem((20, 4), 0.05);
    let y = Array1::from_elem(20, 10.0);
    let w = solve_irwls_single_bulk_native(&s, &y, 200.0);
    assert_eq!(w.len(), 4);
    assert!(w.iter().all(|v| v.is_finite()));
}
