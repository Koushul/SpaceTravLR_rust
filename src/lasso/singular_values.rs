//! Power-iteration based largest-singular-value estimator.
//!
//! This is a faithful port of `_singular_values.py`.
//! The algorithm iterates  v ← Xᵀ(Xv) / ‖Xᵀ(Xv)‖  until the relative
//! improvement in the estimated eigenvalue falls below `tol`.

use ndarray::{Array1, Array2, ArrayView2};
use rand_distr::{Distribution, StandardNormal};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use crate::lasso::subsampling::{get_row_indices, select_rows, SubsamplingScheme};

const LIPSCHITZ_MAXITS: usize = 20;
const LIPSCHITZ_TOL: f64 = 5e-3;

// ── Internal helpers ──────────────────────────────────────────────────────────

/// One step of the power iteration: v ← Xᵀ(Xv), returns (v_normalised, ‖v‖).
fn power_iteration(x: &ArrayView2<f64>, v: &Array1<f64>) -> (Array1<f64>, f64) {
    // Xv  (shape: n)
    let xv: Array1<f64> = x.dot(v);
    // Xᵀ(Xv)  (shape: p)
    let mut v_new: Array1<f64> = x.t().dot(&xv);
    let s = norm2(&v_new);
    if s > 0.0 {
        v_new.mapv_inplace(|x| x / s);
    }
    (v_new, s)
}

/// Power iteration on a random subsample of rows, with Lipschitz rescaling.
fn subsampled_power_iteration(
    x: &Array2<f64>,
    v: &Array1<f64>,
    scheme: &SubsamplingScheme,
    rng: &mut ChaCha8Rng,
) -> (Array1<f64>, f64) {
    let indices = get_row_indices(x.nrows(), scheme, rng);
    let x_sub = select_rows(&x.view(), &indices);
    let (v_new, s) = power_iteration(&x_sub.view(), v);
    let frac = indices.len() as f64 / x.nrows() as f64;
    (v_new, s / frac)
}

/// Euclidean norm of a 1-D array.
#[inline]
fn norm2(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Estimate the largest singular value of `x` via randomised power iteration.
///
/// # Arguments
/// * `x`      – data matrix (n × p)
/// * `seed`   – seed for the internal RNG
/// * `scheme` – subsampling scheme; use [`SubsamplingScheme::None`] for exact
/// * `maxits` – maximum number of iterations (default [`LIPSCHITZ_MAXITS`])
/// * `tol`    – relative convergence tolerance (default [`LIPSCHITZ_TOL`])
pub fn find_largest_singular_value(
    x: &Array2<f64>,
    seed: u64,
    scheme: &SubsamplingScheme,
    maxits: Option<usize>,
    tol: Option<f64>,
) -> f64 {
    let maxits = maxits.unwrap_or(LIPSCHITZ_MAXITS);
    let tol = tol.unwrap_or(LIPSCHITZ_TOL);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Initialise v ~ N(0,1), then normalise
    let normal = StandardNormal;
    let p = x.ncols();
    let raw: Vec<f64> = (0..p).map(|_| normal.sample(&mut rng)).collect();
    let mut v = Array1::from_vec(raw);
    let n0 = norm2(&v);
    v.mapv_inplace(|x| x / n0);

    let mut s = n0;     // current eigenvalue estimate (will be overwritten immediately)
    let mut s_best = s;

    for i in 0..maxits {
        let s_prev = s;
        let v_prev = v.clone();

        let (v_new, s_new) = subsampled_power_iteration(x, &v, scheme, &mut rng);
        v = v_new;
        s = s_new;

        let improvement = (s - s_prev).abs() / s.abs().max(s_prev.abs());
        if improvement < tol && i > 0 {
            return s_best.sqrt();
        }

        // Keep the best estimate so far (subsampling can cause regressions)
        if s > s_best {
            s_best = s;
        } else {
            // Revert to best seen
            s = s_prev;
            v = v_prev;
        }
    }

    s_best.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn singular_value_identity() {
        // σ_max of the 3×3 identity is 1.0
        let eye = Array2::<f64>::eye(3);
        let sv = find_largest_singular_value(&eye, 0, &SubsamplingScheme::None, None, None);
        assert_abs_diff_eq!(sv, 1.0, epsilon = 0.05);
    }

    #[test]
    fn singular_value_scaled_identity() {
        // σ_max of 5·I₃ is 5.0
        let x = Array2::<f64>::eye(3) * 5.0;
        let sv = find_largest_singular_value(&x, 1, &SubsamplingScheme::None, None, None);
        assert_abs_diff_eq!(sv, 5.0, epsilon = 0.1);
    }
}
