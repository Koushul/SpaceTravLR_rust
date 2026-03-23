//! Proximal operators used by the group-lasso solver.
//!
//! Three building blocks (matching the Python helpers in `_group_lasso.py`):
//!
//! * [`l1_prox`]       – soft-thresholding for the ℓ₁ penalty
//! * [`l2_prox`]       – block soft-thresholding for the ℓ₂ penalty  
//! * [`group_l2_prox`] – apply [`l2_prox`] independently to each group
//! * [`l1_l2_prox`]    – compose ℓ₁ then group-ℓ₂ (sparse-group lasso)

use ndarray::Array2;

// ── Scalar / column-wise helpers ─────────────────────────────────────────────

/// Soft-thresholding: sign(w) · max(0, |w| − reg).
#[inline]
pub fn l1_prox_scalar(w: f64, reg: f64) -> f64 {
    w.signum() * (w.abs() - reg).max(0.0)
}

/// Element-wise ℓ₁ proximal operator applied to every entry of `w`.
pub fn l1_prox(w: &Array2<f64>, reg: f64) -> Array2<f64> {
    w.mapv(|v| l1_prox_scalar(v, reg))
}

// ── ℓ₂ block proximal operator ───────────────────────────────────────────────

/// Proximal operator for `reg · ‖w‖₂` (not squared).
///
/// Shrinks the whole block towards zero; returns the zero vector when
/// `‖w‖₂ = 0`.
///
/// The block is given as a mutable sub-slice so the caller can work
/// in-place on a coefficient matrix without extra allocations.
pub fn l2_prox_inplace(block: &mut ndarray::ArrayViewMut1<f64>, reg: f64) {
    let norm: f64 = block.dot(block).sqrt();
    if norm == 0.0 {
        return; // already zero – nothing to do
    }
    let scale = (1.0_f64 - reg / norm).max(0.0);
    block.mapv_inplace(|v| v * scale);
}

// ── Group-wise ℓ₂ proximal operator ──────────────────────────────────────────

/// Apply the ℓ₂ proximal operator independently to each group of rows.
///
/// # Arguments
/// * `w`      – coefficient matrix **(num_features × num_targets)**
/// * `groups` – slice of boolean masks, each of length `num_features`;
///              `groups[k][i] == true` means feature `i` belongs to group `k`
/// * `regs`   – per-group regularisation strengths (same length as `groups`)
pub fn group_l2_prox(w: &Array2<f64>, groups: &[Vec<bool>], regs: &[f64]) -> Array2<f64> {
    assert_eq!(groups.len(), regs.len());
    let mut out = w.clone();
    for (mask, &reg) in groups.iter().zip(regs.iter()) {
        // Collect the row indices that belong to this group
        let row_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        if row_indices.is_empty() {
            continue;
        }

        // Apply l2_prox to each target column independently over this group's rows
        let num_targets = out.ncols();
        for col in 0..num_targets {
            // Compute the ℓ₂ norm of the group slice for this target
            let norm: f64 = row_indices
                .iter()
                .map(|&r| out[[r, col]].powi(2))
                .sum::<f64>()
                .sqrt();

            if norm == 0.0 {
                continue;
            }
            let scale = (1.0 - reg / norm).max(0.0);
            for &r in &row_indices {
                out[[r, col]] *= scale;
            }
        }
    }
    out
}

// ── Sparse-group lasso composite prox ────────────────────────────────────────

/// Composite proximal operator for the sparse-group lasso penalty:
/// ℓ₁ first, then group-ℓ₂.
pub fn l1_l2_prox(
    w: &Array2<f64>,
    l1_reg: f64,
    group_regs: &[f64],
    groups: &[Vec<bool>],
) -> Array2<f64> {
    let after_l1 = l1_prox(w, l1_reg);
    group_l2_prox(&after_l1, groups, group_regs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn l1_prox_shrinks_correctly() {
        let w = array![[2.0], [-0.5], [0.3]];
        let out = l1_prox(&w, 0.4);
        assert_abs_diff_eq!(out[[0, 0]], 1.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], -0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[2, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l1_prox_zero_reg_is_identity() {
        let w = array![[2.0, -1.0], [-0.5, 3.0]];
        let out = l1_prox(&w, 0.0);
        for (a, b) in out.iter().zip(w.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn l1_prox_large_reg_zeros_everything() {
        let w = array![[2.0], [-0.5], [0.3]];
        let out = l1_prox(&w, 100.0);
        for &v in out.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn l1_prox_preserves_sign() {
        let w = array![[5.0], [-5.0]];
        let out = l1_prox(&w, 2.0);
        assert!(out[[0, 0]] > 0.0);
        assert!(out[[1, 0]] < 0.0);
        assert_abs_diff_eq!(out[[0, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], -3.0, epsilon = 1e-10);
    }

    #[test]
    fn l1_prox_at_threshold_boundary() {
        let w = array![[0.5], [-0.5]];
        let out = l1_prox(&w, 0.5);
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[1, 0]], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn l1_prox_scalar_symmetry() {
        assert_abs_diff_eq!(l1_prox_scalar(3.0, 1.0), 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(l1_prox_scalar(-3.0, 1.0), -2.0, epsilon = 1e-15);
    }

    #[test]
    fn l2_prox_zero_stays_zero() {
        let w = array![[0.0], [0.0]];
        let groups = vec![vec![true, true]];
        let regs = vec![1.0];
        let out = group_l2_prox(&w, &groups, &regs);
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l2_prox_shrinks_unit_vector() {
        let w = array![[1.0], [0.0]];
        let groups = vec![vec![true, true]];
        let regs = vec![0.5];
        let out = group_l2_prox(&w, &groups, &regs);
        assert_abs_diff_eq!(out[[0, 0]], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn l2_prox_kills_small_block() {
        // ||[0.3, 0.4]|| = 0.5, reg = 1.0 → scale = max(0, 1 - 1.0/0.5) = 0 → zero
        let w = array![[0.3], [0.4]];
        let groups = vec![vec![true, true]];
        let regs = vec![1.0];
        let out = group_l2_prox(&w, &groups, &regs);
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l2_prox_preserves_direction() {
        let w = array![[3.0], [4.0]]; // norm = 5
        let groups = vec![vec![true, true]];
        let regs = vec![1.0]; // scale = 1 - 1/5 = 0.8
        let out = group_l2_prox(&w, &groups, &regs);
        assert_abs_diff_eq!(out[[0, 0]], 2.4, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 3.2, epsilon = 1e-10);
        let ratio = out[[0, 0]] / out[[1, 0]];
        assert_abs_diff_eq!(ratio, 3.0 / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn l2_prox_multiple_groups_independent() {
        let w = array![[3.0], [4.0], [0.1], [0.1]]; // group 0: norm=5, group 1: norm=√0.02
        let groups = vec![
            vec![true, true, false, false],
            vec![false, false, true, true],
        ];
        let regs = vec![1.0, 0.5];
        let out = group_l2_prox(&w, &groups, &regs);

        // Group 0: scale = 1 - 1/5 = 0.8
        assert_abs_diff_eq!(out[[0, 0]], 2.4, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 3.2, epsilon = 1e-10);

        // Group 1: norm=√0.02 ≈ 0.1414, reg=0.5 > norm → zeroed
        assert_abs_diff_eq!(out[[2, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[3, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l2_prox_multi_target() {
        let w = array![[3.0, 0.0], [4.0, 0.0]];
        let groups = vec![vec![true, true]];
        let regs = vec![1.0];
        let out = group_l2_prox(&w, &groups, &regs);
        // Column 0: norm=5, scale=0.8
        assert_abs_diff_eq!(out[[0, 0]], 2.4, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 3.2, epsilon = 1e-10);
        // Column 1: all zeros → stay zero
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l2_prox_inplace_basic() {
        let mut w = ndarray::array![3.0, 4.0]; // norm = 5
        l2_prox_inplace(&mut w.view_mut(), 1.0); // scale = 0.8
        assert_abs_diff_eq!(w[0], 2.4, epsilon = 1e-10);
        assert_abs_diff_eq!(w[1], 3.2, epsilon = 1e-10);
    }

    #[test]
    fn l1_l2_prox_composition_order() {
        // l1 first, then l2. With w = [2.0, -2.0], l1_reg = 0.5:
        //   after l1: [1.5, -1.5], group norm = 1.5√2
        //   after l2 with reg=0.5: scale = max(0, 1 - 0.5/(1.5√2))
        let w = array![[2.0], [-2.0]];
        let groups = vec![vec![true, true]];
        let group_regs = vec![0.5];
        let out = l1_l2_prox(&w, 0.5, &group_regs, &groups);

        let after_l1_0 = 1.5_f64;
        let after_l1_1 = -1.5_f64;
        let group_norm = (after_l1_0.powi(2) + after_l1_1.powi(2)).sqrt();
        let scale = (1.0 - 0.5 / group_norm).max(0.0);
        assert_abs_diff_eq!(out[[0, 0]], after_l1_0 * scale, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], after_l1_1 * scale, epsilon = 1e-10);
    }

    #[test]
    fn l1_l2_prox_heavy_reg_zeros_all() {
        let w = array![[1.0], [-1.0], [0.5]];
        let groups = vec![vec![true, true, true]];
        let out = l1_l2_prox(&w, 10.0, &[10.0], &groups);
        for &v in out.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn l1_l2_prox_zero_regs_is_identity() {
        let w = array![[1.0, -2.0], [3.0, 4.0]];
        let groups = vec![vec![true, true]];
        let out = l1_l2_prox(&w, 0.0, &[0.0], &groups);
        for (a, b) in out.iter().zip(w.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }
}
