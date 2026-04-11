//! End-to-end sklearn PCA vs Rust (same sketch `omega.npy`, QR power iterations, gesdd on `B`).
//!
//! The thin SVD of `B` is verified separately in `magic_pca::tests::gesdd_matches_scipy_fixture`
//! (bit-level vs SciPy `lapack_driver="gesdd"` on the same `B_pca.npy`). Remaining score drift
//! vs sklearn is from **faer Householder QR** vs **scipy.linalg.qr** on the power iterations.

use std::path::PathBuf;

use approx::assert_relative_eq;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use spacetravlr::magic_pca::{SklearnRandomizedPcaConfig, fit_randomized_pca_sklearn};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pca_sklearn_qr")
}

#[test]
fn randomized_pca_matches_sklearn_qr_normalizer() {
    let dir = fixture_dir();
    let x: Array2<f64> = read_npy(dir.join("X.npy")).expect("X.npy");
    let mean_exp: Array1<f64> = read_npy(dir.join("mean.npy")).expect("mean.npy");
    let comp_exp: Array2<f64> = read_npy(dir.join("components.npy")).expect("components.npy");
    let scores_exp: Array2<f64> = read_npy(dir.join("scores.npy")).expect("scores.npy");
    let omega: Array2<f64> = read_npy(dir.join("omega.npy")).expect("omega.npy");

    let cfg = SklearnRandomizedPcaConfig {
        n_components: 5,
        n_oversamples: 10,
        n_iter: Some(4),
        omega: Some(omega),
    };
    let (scores, comp, mean) = fit_randomized_pca_sklearn(x.view(), &cfg).expect("fit");

    assert_eq!(mean.dim(), mean_exp.dim());
    assert_eq!(comp.dim(), comp_exp.dim());
    assert_eq!(scores.dim(), scores_exp.dim());

    for (a, b) in mean.iter().zip(mean_exp.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-12, max_relative = 1e-12);
    }
    for (a, b) in comp.iter().zip(comp_exp.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-2, max_relative = 2e-2);
    }
    for (a, b) in scores.iter().zip(scores_exp.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-2, max_relative = 2e-2);
    }
}
