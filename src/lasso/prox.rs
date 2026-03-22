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
        // ‖[1,0]‖ = 1, reg = 0.5  →  scale = 0.5
        let w = array![[1.0], [0.0]];
        let groups = vec![vec![true, true]];
        let regs = vec![0.5];
        let out = group_l2_prox(&w, &groups, &regs);
        assert_abs_diff_eq!(out[[0, 0]], 0.5, epsilon = 1e-10);
    }
}
