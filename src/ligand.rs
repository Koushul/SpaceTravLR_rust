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
