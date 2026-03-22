//! FISTA optimiser with backtracking line-search and adaptive restart.
//!
//! This is a faithful port of `_fista.py` (Beck & Teboulle 2009) augmented
//! with the gradient-based adaptive restart scheme from O'Donoghue & Candès
//! (2012).
//!
//! # Algorithm
//! 1. Gradient step on the smooth part.
//! 2. Proximal step on the non-smooth part.
//! 3. Nesterov momentum update.
//! 4. Adaptive restart: if the generalised gradient inner-product with the
//!    update direction exceeds the smooth loss, reset momentum to 1.
//! 5. Backtracking: double the Lipschitz estimate until the quadratic upper
//!    bound is satisfied (criterion from Beck & Teboulle eq. 2.5).

use ndarray::{Array2, Zip};

/// Convergence / iteration log entry produced by the optional callback.
pub struct IterInfo {
    pub iteration: usize,
    pub norm_change: f64,
    pub lipschitz: f64,
}

/// Trait that every problem plugged into FISTA must implement.
///
/// All methods receive coefficient matrices of shape `(num_features, num_targets)`.
pub trait FistaProblem {
    /// Smooth loss f(w).
    fn smooth_loss(&self, w: &Array2<f64>) -> f64;

    /// Gradient of the smooth loss ∇f(w).
    fn smooth_grad(&self, w: &Array2<f64>) -> Array2<f64>;

    /// Proximal operator: prox_{g/L}(w) for the non-smooth term g.
    fn prox(&self, w: &Array2<f64>, lipschitz: f64) -> Array2<f64>;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

#[inline]
fn next_momentum(t: f64) -> f64 {
    0.5 + 0.5 * (1.0 + 4.0 * t * t).sqrt()
}

/// Squared Frobenius norm of `a - b`.
#[inline]
fn sq_frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    Zip::from(a).and(b).fold(0.0, |acc, &x, &y| acc + (x - y).powi(2))
}

/// Frobenius norm of `a`.
#[inline]
pub fn frob_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Flat dot-product of two same-shape arrays (treats them as vectors).
#[inline]
fn flat_dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    Zip::from(a).and(b).fold(0.0, |acc, &x, &y| acc + x * y)
}

// ── Backtracking criterion (eq. 2.5 in Beck & Teboulle 2009) ─────────────────

/// Returns `true` if the Lipschitz constant should be doubled.
///
/// Condition: f(new_x) > f(y) + ⟨∇f(y), new_x − y⟩ + L/2.5 · ‖new_x − y‖²
fn continue_backtracking(
    problem: &impl FistaProblem,
    new_x: &Array2<f64>,
    y: &Array2<f64>,         // momentum point
    lipschitz: f64,
) -> bool {
    let f_new = problem.smooth_loss(new_x);
    let f_y   = problem.smooth_loss(y);
    let grad_y = problem.smooth_grad(y);

    let update = Array2::from_shape_fn(new_x.raw_dim(), |(r, c)| new_x[[r, c]] - y[[r, c]]);
    let update_dist = sq_frob_diff(new_x, y) * lipschitz / 2.5;
    let lin_imp = flat_dot(&grad_y, &update);

    f_new > (f_y + update_dist + lin_imp)
}

// ── One gradient + prox step ──────────────────────────────────────────────────

/// Perform one FISTA update step.
///
/// Returns `(new_x, new_momentum_x, new_momentum)`.
fn update_step(
    problem: &impl FistaProblem,
    x: &Array2<f64>,
    y: &Array2<f64>,   // momentum point
    t: f64,
    lipschitz: f64,
) -> (Array2<f64>, Array2<f64>, f64) {
    let grad = problem.smooth_grad(y);
    // Gradient step: y − (1/(2L)) · ∇f(y)
    let step = Array2::from_shape_fn(y.raw_dim(), |(r, c)| {
        y[[r, c]] - 0.5 * grad[[r, c]] / lipschitz
    });
    let new_x = problem.prox(&step, lipschitz);
    let new_t = next_momentum(t);

    // Nesterov momentum extrapolation
    let new_y = Array2::from_shape_fn(new_x.raw_dim(), |(r, c)| {
        new_x[[r, c]] + (new_x[[r, c]] - x[[r, c]]) * (t - 1.0) / new_t
    });

    (new_x, new_y, new_t)
}

// ── Public minimiser ──────────────────────────────────────────────────────────

/// Result returned by [`minimise`].
pub struct FistaResult {
    pub coef: Array2<f64>,
    /// Final Lipschitz estimate (may have grown due to backtracking).
    pub lipschitz: f64,
    /// Whether the algorithm converged within `n_iter` iterations.
    pub converged: bool,
}

/// Run FISTA to minimise `f(w) + g(w)` starting from `w0`.
///
/// # Arguments
/// * `problem`   – implements [`FistaProblem`]
/// * `w0`        – initial coefficient matrix
/// * `lipschitz` – initial Lipschitz estimate for the smooth part
/// * `n_iter`    – maximum number of iterations
/// * `tol`       – relative convergence tolerance `‖Δw‖/‖w‖ < tol`
/// * `callback`  – optional closure called after each iteration
pub fn minimise<P, F>(
    problem: &P,
    w0: Array2<f64>,
    lipschitz: f64,
    n_iter: usize,
    tol: f64,
    mut callback: Option<F>,
) -> FistaResult
where
    P: FistaProblem,
    F: FnMut(&IterInfo),
{
    let mut l = lipschitz;
    let mut x = w0;
    let mut y = x.clone();
    let mut t = 1.0_f64;

    for iter in 0..n_iter {
        let prev_x = x.clone();

        // ── Tentative update ────────────────────────────────────────────────
        let (mut new_x, mut new_y, mut new_t) = update_step(problem, &x, &y, t, l);

        // ── Adaptive restart (O'Donoghue & Candès 2012, eq. 12) ────────────
        // generalised_gradient = y − new_x
        // update_vector        = new_x − prev_x
        // Restart condition: ⟨gen_grad, update_vec⟩ > f(prev_x)
        {
            let gen_grad_dot_update = {
                let mut acc = 0.0_f64;
                for ((yv, nxv), pxv) in y.iter().zip(new_x.iter()).zip(prev_x.iter()) {
                    acc += (yv - nxv) * (nxv - pxv);
                }
                acc
            };
            if gen_grad_dot_update > problem.smooth_loss(&prev_x) {
                // Reset momentum and redo the step from prev_x
                y = prev_x.clone();
                t = 1.0;
                let res = update_step(problem, &prev_x, &y, t, l);
                new_x = res.0;
                new_y = res.1;
                new_t = res.2;
            }
        }

        // ── Backtracking line search ────────────────────────────────────────
        while continue_backtracking(problem, &new_x, &y, l) {
            l *= 2.0;
            let res = update_step(problem, &x, &y, t, l);
            new_x = res.0;
            new_y = res.1;
            new_t = res.2;
        }

        x = new_x;
        y = new_y;
        t = new_t;

        // ── Convergence check ───────────────────────────────────────────────
        let norm_change = frob_norm(&Array2::from_shape_fn(x.raw_dim(), |(r, c)| {
            x[[r, c]] - prev_x[[r, c]]
        }));
        let norm_x = frob_norm(&x) + 1e-16;
        let rel_change = norm_change / norm_x;

        if let Some(ref mut cb) = callback {
            cb(&IterInfo { iteration: iter, norm_change, lipschitz: l });
        }

        if rel_change < tol {
            return FistaResult { coef: x, lipschitz: l, converged: true };
        }
    }

    FistaResult { coef: x, lipschitz: l, converged: false }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Trivial problem: minimise 0.5 ‖w‖² (smooth) + 0 (prox = identity).
    struct QuadraticProblem;
    impl FistaProblem for QuadraticProblem {
        fn smooth_loss(&self, w: &Array2<f64>) -> f64 {
            0.5 * w.iter().map(|x| x * x).sum::<f64>()
        }
        fn smooth_grad(&self, w: &Array2<f64>) -> Array2<f64> {
            w.clone()
        }
        fn prox(&self, w: &Array2<f64>, _l: f64) -> Array2<f64> {
            w.clone()
        }
    }

    #[test]
    fn fista_converges_to_zero_for_quadratic() {
        let w0 = array![[5.0], [-3.0], [1.0]];
        let result = minimise(&QuadraticProblem, w0, 1.0, 500, 1e-8, None::<fn(&IterInfo)>);
        assert!(result.converged);
        for &v in result.coef.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-4);
        }
    }
}
