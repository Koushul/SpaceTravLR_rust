use ndarray::{Array2, Axis};
use rayon::prelude::*;

/// ∂R_i/∂L_j for one ligand species: `R_i = (1/N) Σ_k scale·exp(-d²_{ik}/2r²)·L_k` (same as
/// [`crate::ligand::calculate_weighted_ligands_with_cutoff`] but without multiplying by L_k).
///
/// Row `i`, column `j` is the derivative of received ligand at cell `i` w.r.t. ligand expression
/// at sender cell `j`, for a single ligand column (same radius / scale / cutoff as WL).
pub fn received_ligand_jacobian_one_ligand(
    xy: &Array2<f64>,
    radius: f64,
    scale_factor: f64,
    max_neighbor_distance: Option<f64>,
) -> Array2<f64> {
    let n = xy.nrows();
    let mut jacobian = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return jacobian;
    }
    let inv_2r2 = -1.0 / (2.0 * radius * radius);
    let d2_cut = max_neighbor_distance
        .filter(|m| m.is_finite() && *m > 0.0)
        .map(|m| m * m);
    let n_inv = 1.0 / n as f64;
    let scale_n = scale_factor * n_inv;

    jacobian
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let xi = xy[[i, 0]];
            let yi = xy[[i, 1]];
            for j in 0..n {
                let dx = xi - xy[[j, 0]];
                let dy = yi - xy[[j, 1]];
                let d2 = dx * dx + dy * dy;
                if d2_cut.is_some_and(|c| d2 > c) {
                    continue;
                }
                row[j] = scale_n * (d2 * inv_2r2).exp();
            }
        });

    jacobian
}

/// One row of [`received_ligand_jacobian_one_ligand`]: `out[j] = ∂WL(receiver_idx)/∂L_j`.
/// By symmetry of the Gaussian kernel, this equals `∂WL_i/∂L_sender` when `receiver_idx == sender`
/// is used as the fixed index for the other direction.
pub fn received_ligand_jacobian_row_for_cell(
    xy: &Array2<f64>,
    receiver_idx: usize,
    radius: f64,
    scale_factor: f64,
    max_neighbor_distance: Option<f64>,
) -> Vec<f64> {
    let n = xy.nrows();
    let mut out = vec![0.0_f64; n];
    if n == 0 || receiver_idx >= n {
        return out;
    }
    let inv_2r2 = -1.0 / (2.0 * radius * radius);
    let d2_cut = max_neighbor_distance
        .filter(|m| m.is_finite() && *m > 0.0)
        .map(|m| m * m);
    let n_inv = 1.0 / n as f64;
    let scale_n = scale_factor * n_inv;
    let xi = xy[[receiver_idx, 0]];
    let yi = xy[[receiver_idx, 1]];
    for j in 0..n {
        let dx = xi - xy[[j, 0]];
        let dy = yi - xy[[j, 1]];
        let d2 = dx * dx + dy * dy;
        if d2_cut.is_some_and(|c| d2 > c) {
            continue;
        }
        out[j] = scale_n * (d2 * inv_2r2).exp();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn jacobian_matches_closed_form() {
        let xy = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0]).unwrap();
        let r = 10.0_f64;
        let scale = 2.0_f64;
        let n = 3usize;
        let n_inv = 1.0 / n as f64;
        let jac = received_ligand_jacobian_one_ligand(&xy, r, scale, None);
        for i in 0..n {
            for j in 0..n {
                let dx = xy[[i, 0]] - xy[[j, 0]];
                let dy = xy[[i, 1]] - xy[[j, 1]];
                let d2 = dx * dx + dy * dy;
                let expected = scale * n_inv * (-d2 / (2.0 * r * r)).exp();
                assert_relative_eq!(jac[[i, j]], expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn jacobian_row_matches_matrix_row() {
        let xy = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 2.0, 1.0, -1.0, 3.0, 0.5, 0.5]).unwrap();
        let r = 30.0_f64;
        let scale = 1.5_f64;
        let jac = received_ligand_jacobian_one_ligand(&xy, r, scale, Some(100.0));
        for recv in 0..xy.nrows() {
            let row = received_ligand_jacobian_row_for_cell(&xy, recv, r, scale, Some(100.0));
            for j in 0..xy.nrows() {
                assert_relative_eq!(row[j], jac[[recv, j]], epsilon = 1e-12);
            }
        }
    }
}
