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
        let eye = Array2::<f64>::eye(3);
        let sv = find_largest_singular_value(&eye, 0, &SubsamplingScheme::None, None, None);
        assert_abs_diff_eq!(sv, 1.0, epsilon = 0.05);
    }

    #[test]
    fn singular_value_scaled_identity() {
        let x = Array2::<f64>::eye(3) * 5.0;
        let sv = find_largest_singular_value(&x, 1, &SubsamplingScheme::None, None, None);
        assert_abs_diff_eq!(sv, 5.0, epsilon = 0.1);
    }

    #[test]
    fn singular_value_diagonal() {
        // diag(1, 2, 7, 3) → σ_max = 7
        let mut d = Array2::<f64>::zeros((4, 4));
        d[[0, 0]] = 1.0;
        d[[1, 1]] = 2.0;
        d[[2, 2]] = 7.0;
        d[[3, 3]] = 3.0;
        let sv = find_largest_singular_value(&d, 42, &SubsamplingScheme::None, None, None);
        assert_abs_diff_eq!(sv, 7.0, epsilon = 0.2);
    }

    #[test]
    fn singular_value_rank1_matrix() {
        // rank-1: u v^T where u = [1,2,3], v = [1,1] → σ = ||u|| * ||v|| = √14 * √2
        // Actually σ_max of u*v^T = ||u||*||v||
        let u = ndarray::array![[1.0], [2.0], [3.0]];
        let v = ndarray::array![[1.0, 1.0]];
        let x = u.dot(&v); // 3×2 rank-1
        let expected = (1.0 + 4.0 + 9.0_f64).sqrt() * 2.0_f64.sqrt();
        let sv = find_largest_singular_value(&x, 7, &SubsamplingScheme::None, Some(50), None);
        assert_abs_diff_eq!(sv, expected, epsilon = 0.1);
    }

    #[test]
    fn singular_value_rectangular_tall() {
        // [[1,0],[0,2],[0,0]] → σ_max = 2
        let x = ndarray::array![[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]];
        let sv = find_largest_singular_value(&x, 0, &SubsamplingScheme::None, None, None);
        assert_abs_diff_eq!(sv, 2.0, epsilon = 0.1);
    }

    #[test]
    fn singular_value_rectangular_wide() {
        // [[3,0,0],[0,1,0]] → σ_max = 3
        let x = ndarray::array![[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let sv = find_largest_singular_value(&x, 0, &SubsamplingScheme::None, None, None);
        assert_abs_diff_eq!(sv, 3.0, epsilon = 0.15);
    }

    #[test]
    fn singular_value_positive_for_nonzero_matrix() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let sv = find_largest_singular_value(&x, 0, &SubsamplingScheme::None, None, None);
        assert!(sv > 0.0);
    }

    #[test]
    fn singular_value_deterministic_with_same_seed() {
        let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let sv1 = find_largest_singular_value(&x, 42, &SubsamplingScheme::None, None, None);
        let sv2 = find_largest_singular_value(&x, 42, &SubsamplingScheme::None, None, None);
        assert_abs_diff_eq!(sv1, sv2, epsilon = 1e-15);
    }

    #[test]
    fn norm2_known_values() {
        let v = Array1::from_vec(vec![3.0, 4.0]);
        assert_abs_diff_eq!(norm2(&v), 5.0, epsilon = 1e-15);
    }
}
