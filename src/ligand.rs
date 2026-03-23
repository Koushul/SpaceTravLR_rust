use ndarray::{Array2, Axis};
use rayon::prelude::*;
use polars::prelude::*;
use std::collections::HashMap;

/// Compute the amount of ligand received by each cell.
/// 
/// Corresponds to `compute_radius_weights_fast` in Python.
/// Optimized for $O(N \times L)$ memory by avoiding full $N \times N$ matrix storage.
pub fn calculate_weighted_ligands(
    xy: &Array2<f64>,
    lig_values: &Array2<f64>,
    radius: f64,
    scale_factor: f64,
) -> Array2<f64> {
    let n_cells = xy.nrows();
    let n_ligands = lig_values.ncols();
    let inv_2r2 = -1.0 / (2.0 * radius * radius);

    let mut result = Array2::zeros((n_cells, n_ligands));

    // Parallelize over target cells (rows of the result)
    result.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let xi = xy[[i, 0]];
            let yi = xy[[i, 1]];

            // For each target cell i, iterate over all source cells j
            for j in 0..n_cells {
                let dx = xi - xy[[j, 0]];
                let dy = yi - xy[[j, 1]];
                let d2 = dx * dx + dy * dy;
                let w = scale_factor * (d2 * inv_2r2).exp();

                // Aggregate weighted ligand values from source cell j
                for k in 0..n_ligands {
                    row[k] += w * lig_values[[j, k]];
                }
            }

            // Divide by N (simple mean as in Python np.mean(gauss_weights[j] * lig_df_values[:, i]))
            // Python: out[i, j] /= n
            for k in 0..n_ligands {
                row[k] /= n_cells as f64;
            }
        });

    result
}

/// Grid-approximated version of `calculate_weighted_ligands`.
///
/// Instead of O(N²) pairwise Gaussian kernel evaluations, places anchor points
/// on a regular grid (spacing = `radius * grid_factor`), computes exact received
/// ligands at each anchor in O(A × N), then bilinearly interpolates to each cell
/// in O(N). Total: O(A × N × L) where A ≪ N for dense spatial data.
///
/// The error is O(h²/r²) where h = grid spacing, so `grid_factor = 0.5` gives
/// ~3% relative error — well within the modeling uncertainty of the Gaussian
/// kernel itself.
pub fn calculate_weighted_ligands_grid(
    xy: &Array2<f64>,
    lig_values: &Array2<f64>,
    radius: f64,
    scale_factor: f64,
    grid_factor: f64,
) -> Array2<f64> {
    let n_cells = xy.nrows();
    let n_ligands = lig_values.ncols();
    let grid_spacing = radius * grid_factor;
    let inv_2r2 = -1.0 / (2.0 * radius * radius);

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for i in 0..n_cells {
        let x = xy[[i, 0]];
        let y = xy[[i, 1]];
        if x < x_min { x_min = x; }
        if x > x_max { x_max = x; }
        if y < y_min { y_min = y; }
        if y > y_max { y_max = y; }
    }

    x_min -= grid_spacing;
    y_min -= grid_spacing;
    x_max += grid_spacing;
    y_max += grid_spacing;

    let nx = ((x_max - x_min) / grid_spacing).ceil() as usize + 1;
    let ny = ((y_max - y_min) / grid_spacing).ceil() as usize + 1;
    let n_anchors = nx * ny;

    if n_anchors >= n_cells {
        return calculate_weighted_ligands(xy, lig_values, radius, scale_factor);
    }

    let lig_flat = lig_values.as_slice().unwrap();
    let n_inv = 1.0 / n_cells as f64;

    // Compute exact received ligands at each grid anchor: O(A × N × L)
    let mut anchor_vals = vec![0.0f64; n_anchors * n_ligands];

    anchor_vals
        .par_chunks_mut(n_ligands)
        .enumerate()
        .for_each(|(a, row)| {
            let gx = a % nx;
            let gy = a / nx;
            let ax = x_min + gx as f64 * grid_spacing;
            let ay = y_min + gy as f64 * grid_spacing;

            for j in 0..n_cells {
                let dx = ax - xy[[j, 0]];
                let dy = ay - xy[[j, 1]];
                let d2 = dx * dx + dy * dy;
                let w = scale_factor * (d2 * inv_2r2).exp();
                let base = j * n_ligands;
                for k in 0..n_ligands {
                    unsafe {
                        *row.get_unchecked_mut(k) += w * *lig_flat.get_unchecked(base + k);
                    }
                }
            }
            for k in 0..n_ligands {
                row[k] *= n_inv;
            }
        });

    // Bilinear interpolation for each cell: O(N × L)
    let mut result = Array2::zeros((n_cells, n_ligands));
    let res_flat = result.as_slice_mut().unwrap();

    res_flat
        .par_chunks_mut(n_ligands)
        .enumerate()
        .for_each(|(i, row)| {
            let gx_f = (xy[[i, 0]] - x_min) / grid_spacing;
            let gy_f = (xy[[i, 1]] - y_min) / grid_spacing;

            let gx0 = gx_f.floor() as usize;
            let gy0 = gy_f.floor() as usize;
            let gx1 = (gx0 + 1).min(nx - 1);
            let gy1 = (gy0 + 1).min(ny - 1);

            let fx = gx_f - gx0 as f64;
            let fy = gy_f - gy0 as f64;

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let a00 = (gy0 * nx + gx0) * n_ligands;
            let a10 = (gy0 * nx + gx1) * n_ligands;
            let a01 = (gy1 * nx + gx0) * n_ligands;
            let a11 = (gy1 * nx + gx1) * n_ligands;

            for k in 0..n_ligands {
                unsafe {
                    *row.get_unchecked_mut(k) =
                        w00 * *anchor_vals.get_unchecked(a00 + k)
                        + w10 * *anchor_vals.get_unchecked(a10 + k)
                        + w01 * *anchor_vals.get_unchecked(a01 + k)
                        + w11 * *anchor_vals.get_unchecked(a11 + k);
                }
            }
        });

    result
}

/// Computes received ligands for various radii as specified in lr_info.
/// 
/// Args:
///     xy: (n_cells, 2) array of coordinates.
///     ligands_df: DataFrame with ligand expression (genes as columns).
///     lr_info: DataFrame with 'ligand' and 'radius' columns.
pub fn compute_received_ligands(
    xy: &Array2<f64>,
    ligands_df: &DataFrame,
    lr_info: &DataFrame,
    scale_factor: f64,
) -> anyhow::Result<DataFrame> {
    
    // 1. Group lr_info by radius
    // We need 'ligand' and 'radius' columns
    let radius_col = lr_info.column("radius")?.f64()?;
    let ligand_col = lr_info.column("ligand")?.str()?;
    
    let mut radius_to_ligands: HashMap<u64, Vec<String>> = HashMap::new();
    for i in 0..lr_info.height() {
        if let (Some(r), Some(l)) = (radius_col.get(i), ligand_col.get(i)) {
            // Use u64 bits for hashmap key to avoid float issues
            let r_bits = r.to_bits();
            radius_to_ligands.entry(r_bits).or_default().push(l.to_string());
        }
    }

    let mut results_cols = Vec::new();

    // 2. Process each radius group
    for (r_bits, ligands) in radius_to_ligands {
        let radius = f64::from_bits(r_bits);
        
        // Filter ligands_df for these ligands
        // Ensure we only take ligands present in ligands_df
        let mut valid_ligands = Vec::new();
        let mut lig_indices = Vec::new();
        for (idx, name) in ligands_df.get_column_names().iter().enumerate() {
            if ligands.contains(&name.to_string()) {
                valid_ligands.push(name.to_string());
                lig_indices.push(idx);
            }
        }

        if valid_ligands.is_empty() {
            continue;
        }

        // Convert ligands_df subset to ndarray
        let sub_df = ligands_df.select(&valid_ligands)?;
        let lig_values = sub_df.to_ndarray::<Float64Type>(IndexOrder::C)?;

        // Compute weighted ligands
        let weighted = calculate_weighted_ligands(xy, &lig_values, radius, scale_factor);

        // Convert back to Polars columns
        for (i, name) in valid_ligands.into_iter().enumerate() {
            let col_data: Vec<f64> = weighted.column(i).to_vec();
            results_cols.push(Column::new(name.into(), col_data));
        }
    }

    // 3. Construct final DataFrame and reorder to match ligands_df columns
    let result_df = DataFrame::new(results_cols)?;
    
    // Sort columns to match ligands_df original order (as Python does)
    let original_col_names = ligands_df.get_column_names();
    let sorted_df = result_df.select(original_col_names.iter().map(|s| s.as_str()))?;

    Ok(sorted_df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn single_cell_self_contribution() {
        // One cell at (0,0) with ligand value 1.0.
        // d=0 → exp(0) = 1.0, so weight = scale_factor * 1.0
        // result = scale_factor * 1.0 / 1 (n_cells)
        let xy = array![[0.0, 0.0]];
        let lig = array![[1.0]];
        let result = calculate_weighted_ligands(&xy, &lig, 1.0, 1.0);
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn two_cells_symmetry() {
        // Two cells equidistant from each other → both receive same total
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let lig = array![[1.0], [1.0]];
        let result = calculate_weighted_ligands(&xy, &lig, 1.0, 1.0);
        assert_abs_diff_eq!(result[[0, 0]], result[[1, 0]], epsilon = 1e-10);
    }

    #[test]
    fn gaussian_decay_with_distance() {
        // Cell 0 at origin, cell 1 at (d, 0). Cell 1 has ligand 1.0, cell 0 has 0.
        // Cell 0 receives: scale * exp(-d²/(2r²)) * 1.0 / 2
        let d = 2.0;
        let r = 1.0;
        let xy = array![[0.0, 0.0], [d, 0.0]];
        let lig = array![[0.0], [1.0]];
        let result = calculate_weighted_ligands(&xy, &lig, r, 1.0);

        let expected = (-(d * d) / (2.0 * r * r)).exp() / 2.0;
        assert_abs_diff_eq!(result[[0, 0]], expected, epsilon = 1e-10);
    }

    #[test]
    fn scale_factor_multiplies_result() {
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let lig = array![[1.0], [1.0]];
        let r1 = calculate_weighted_ligands(&xy, &lig, 1.0, 1.0);
        let r2 = calculate_weighted_ligands(&xy, &lig, 1.0, 3.0);
        for i in 0..2 {
            assert_abs_diff_eq!(r2[[i, 0]], 3.0 * r1[[i, 0]], epsilon = 1e-10);
        }
    }

    #[test]
    fn large_radius_uniform_weights() {
        // Very large radius → all Gaussian weights ≈ 1 → result ≈ mean(lig)
        let xy = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let lig = array![[3.0], [6.0], [9.0]];
        let result = calculate_weighted_ligands(&xy, &lig, 1e6, 1.0);
        let mean = 6.0; // (3+6+9)/3
        for i in 0..3 {
            assert_abs_diff_eq!(result[[i, 0]], mean, epsilon = 0.01);
        }
    }

    #[test]
    fn small_radius_self_dominant() {
        // Very small radius → only self-contribution matters (exp(-d²/2r²) → 0 for d > 0)
        let xy = array![[0.0, 0.0], [100.0, 0.0]];
        let lig = array![[5.0], [10.0]];
        let result = calculate_weighted_ligands(&xy, &lig, 0.001, 1.0);
        // Self: exp(0) = 1, other: exp(-100²/(2*0.001²)) ≈ 0
        assert_abs_diff_eq!(result[[0, 0]], 5.0 / 2.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result[[1, 0]], 10.0 / 2.0, epsilon = 1e-3);
    }

    #[test]
    fn multiple_ligands() {
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let lig = array![[1.0, 2.0], [3.0, 4.0]];
        let result = calculate_weighted_ligands(&xy, &lig, 1.0, 1.0);
        assert_eq!(result.ncols(), 2);
        assert_eq!(result.nrows(), 2);
    }

    #[test]
    fn result_shape_matches_input() {
        let xy = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let lig = Array2::from_shape_fn((4, 3), |(i, j)| (i + j) as f64);
        let result = calculate_weighted_ligands(&xy, &lig, 1.0, 1.0);
        assert_eq!(result.shape(), &[4, 3]);
    }

    #[test]
    fn nonnegative_output_for_nonneg_input() {
        let xy = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]];
        let lig = array![[1.0], [2.0], [3.0]];
        let result = calculate_weighted_ligands(&xy, &lig, 1.0, 1.0);
        for &v in result.iter() {
            assert!(v >= 0.0, "Output should be non-negative for non-negative input");
        }
    }

    #[test]
    fn zero_ligand_values_give_zero() {
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let lig = array![[0.0], [0.0]];
        let result = calculate_weighted_ligands(&xy, &lig, 1.0, 1.0);
        assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn gaussian_formula_verification() {
        // Verify the exact Gaussian kernel weight for a known distance
        let d = 1.5_f64;
        let r = 2.0_f64;
        let xy = array![[0.0, 0.0], [d, 0.0]];
        let lig = array![[0.0], [1.0]];
        let result = calculate_weighted_ligands(&xy, &lig, r, 1.0);
        let expected_weight = (-d * d / (2.0 * r * r)).exp();
        assert_abs_diff_eq!(result[[0, 0]], expected_weight / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn grid_of_cells() {
        // 2×2 grid at (0,0), (1,0), (0,1), (1,1), all with ligand 1.0
        // Each cell should receive the same (by symmetry of uniform field)
        let xy = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let lig = array![[1.0], [1.0], [1.0], [1.0]];
        let result = calculate_weighted_ligands(&xy, &lig, 1.0, 1.0);
        let r0 = result[[0, 0]];
        let r3 = result[[3, 0]];
        assert_abs_diff_eq!(r0, r3, epsilon = 1e-10, );

        let r1 = result[[1, 0]];
        let r2 = result[[2, 0]];
        assert_abs_diff_eq!(r1, r2, epsilon = 1e-10);
    }
}
