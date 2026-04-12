use std::path::Path;

use rctd_core::io_npz::load_q_matrices_npz;
use rctd_core::likelihood_tables::compute_spline_coefficients;
#[test]
#[ignore = "set RCTD_XENIUM_H5 and RCTD_REFERENCE_H5AD to real paths; optional subsample in test"]
fn xenium_liver_smoke_subsample() {
    let xenium = std::env::var("RCTD_XENIUM_H5").expect("RCTD_XENIUM_H5");
    let reference = std::env::var("RCTD_REFERENCE_H5AD").expect("RCTD_REFERENCE_H5AD");
    let q_npz = std::env::var("RCTD_Q_MATRICES_NPZ").expect("RCTD_Q_MATRICES_NPZ");
    assert!(Path::new(&xenium).exists());
    assert!(Path::new(&reference).exists());
    assert!(Path::new(&q_npz).exists());

    let _device = rctd_core::device_cpu();
    let (_q_map, _x_vals) = load_q_matrices_npz(Path::new(&q_npz)).expect("load q npz");
    let _sigma_key = "100";
    let _q = _q_map.get(_sigma_key).expect("sigma key");
    let _sq = compute_spline_coefficients(_q, &_x_vals);

    let _max_spots: usize = std::env::var("RCTD_SUBSAMPLE_SPOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    let _ = (xenium, reference);
}
