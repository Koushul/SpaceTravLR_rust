use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use ndarray::{Array2, Zip};
use rayon::prelude::*;

use crate::betadata::{Betabase, GeneMatrix};
use crate::ligand::{calculate_weighted_ligands, calculate_weighted_ligands_grid};

#[derive(Clone)]
pub struct PerturbTarget {
    pub gene: String,
    pub desired_expr: f64,
    pub cell_indices: Option<Vec<usize>>,
}

pub struct PerturbConfig {
    pub n_propagation: usize,
    pub scale_factor: f64,
    pub beta_scale_factor: f64,
    pub beta_cap: Option<f64>,
    pub min_expression: f64,
    /// When set, approximate received ligands using a spatial grid.
    /// Value is grid_spacing / radius (smaller = more accurate, larger = faster).
    /// 0.5 gives ~3% error; 0.3 gives ~1%. None = exact O(N²) computation.
    pub ligand_grid_factor: Option<f64>,
}

impl Default for PerturbConfig {
    fn default() -> Self {
        Self {
            n_propagation: 4,
            scale_factor: 1.0,
            beta_scale_factor: 1.0,
            beta_cap: None,
            min_expression: 1e-9,
            ligand_grid_factor: None,
        }
    }
}

pub struct PerturbResult {
    pub simulated: Array2<f64>,
    pub delta: Array2<f64>,
}

/// Simulate gene perturbation and propagate effects through the spatial GRN.
///
/// Mirrors Python's `GeneFactory.perturb()`: each iteration computes splash
/// derivatives, recomputes spatially-weighted ligands for the updated expression,
/// swaps direct ligand deltas with received-ligand deltas, then applies
/// delta × splash to propagate effects to all downstream genes.
pub fn perturb(
    bb: &Betabase,
    gene_mtx: &Array2<f64>,
    gene_names: &[String],
    xy: &Array2<f64>,
    rw_ligands_init: &GeneMatrix,
    rw_tfligands_init: &GeneMatrix,
    targets: &[(String, f64)],
    config: &PerturbConfig,
    lr_radii: &HashMap<String, f64>,
) -> PerturbResult {
    let scoped_targets: Vec<PerturbTarget> = targets
        .iter()
        .map(|(gene, desired_expr)| PerturbTarget {
            gene: gene.clone(),
            desired_expr: *desired_expr,
            cell_indices: None,
        })
        .collect();
    perturb_with_targets(
        bb,
        gene_mtx,
        gene_names,
        xy,
        rw_ligands_init,
        rw_tfligands_init,
        &scoped_targets,
        config,
        lr_radii,
        None,
    )
}

pub fn perturb_with_targets(
    bb: &Betabase,
    gene_mtx: &Array2<f64>,
    gene_names: &[String],
    xy: &Array2<f64>,
    rw_ligands_init: &GeneMatrix,
    rw_tfligands_init: &GeneMatrix,
    targets: &[PerturbTarget],
    config: &PerturbConfig,
    lr_radii: &HashMap<String, f64>,
    job_progress: Option<&Arc<AtomicU32>>,
) -> PerturbResult {
    let n_cells = gene_mtx.nrows();
    let n_genes = gene_mtx.ncols();
    let gene_to_idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    // delta_input: desired − original, nonzero only at target genes
    let mut delta_input = Array2::zeros((n_cells, n_genes));
    for target in targets {
        if let Some(&idx) = gene_to_idx.get(target.gene.as_str()) {
            if let Some(cell_indices) = target.cell_indices.as_ref() {
                for &cell in cell_indices {
                    if cell < n_cells {
                        delta_input[[cell, idx]] = target.desired_expr - gene_mtx[[cell, idx]];
                    }
                }
            } else {
                for cell in 0..n_cells {
                    delta_input[[cell, idx]] = target.desired_expr - gene_mtx[[cell, idx]];
                }
            }
        }
    }
    let mut delta_simulated = delta_input.clone();

    // Indices of all ligand genes (LR ∪ TFL) in the full gene matrix
    let all_ligand_set: HashSet<&str> = bb
        .ligands_set
        .iter()
        .chain(bb.tfl_ligands_set.iter())
        .map(|s| s.as_str())
        .collect();
    let ligand_gene_indices: Vec<usize> = all_ligand_set
        .iter()
        .filter_map(|name| gene_to_idx.get(name).copied())
        .collect();

    // ligands_0: original ligand expression, zero-padded to (n_cells × n_genes)
    let mut ligands_0 = Array2::zeros((n_cells, n_genes));
    for &idx in &ligand_gene_indices {
        ligands_0.column_mut(idx).assign(&gene_mtx.column(idx));
    }

    // rw_max_0: element-wise max(rw_lr, rw_tfl) reindexed to (n_cells × n_genes)
    let rw_max_0 = scatter_max_to_full(
        rw_ligands_init,
        rw_tfligands_init,
        &gene_to_idx,
        n_cells,
        n_genes,
    );

    // Received ligands evolve through iterations; TFL stays fixed (Python behavior)
    let mut rw_lr_for_splash = GeneMatrix::new(
        rw_ligands_init.data.clone(),
        rw_ligands_init.col_names.clone(),
    );

    let mut gene_mtx_1 = gene_mtx.clone();

    let max_per_gene: Vec<f64> = (0..n_genes)
        .map(|j| gene_mtx.column(j).iter().cloned().fold(0.0f64, f64::max))
        .collect();

    let lr_ligands: Vec<String> = bb.ligands_set.iter().cloned().collect();
    let tfl_ligands: Vec<String> = bb.tfl_ligands_set.iter().cloned().collect();

    let n_prop = config.n_propagation.max(1);
    if let Some(p) = job_progress {
        p.store(40, Ordering::Relaxed);
    }

    for iter in 0..config.n_propagation {
        if job_progress.is_none() {
            eprintln!("  perturb iteration {}/{}", iter + 1, config.n_propagation);
        }
        if let Some(p) = job_progress {
            let v = ((iter + 1) as u32 * 860).saturating_div(n_prop as u32).min(860);
            p.store(v.max(40), Ordering::Relaxed);
        }

        // 1. Splash all trained genes (expression → f32 for splash / betabase RAM)
        let gex_filtered = gene_mtx_1.mapv(|v| if v > config.min_expression { v } else { 0.0 });
        let gex_gm = GeneMatrix::new(
            gex_filtered.mapv(|v| v as f32),
            gene_names.to_vec(),
        );
        let splashed = splash_all(
            bb,
            &rw_lr_for_splash,
            rw_tfligands_init,
            &gex_gm,
            config.beta_scale_factor as f32,
            config.beta_cap.map(|c| c as f32),
        );

        // 2. Update gene expression
        gene_mtx_1 = gene_mtx + &delta_simulated;

        // 3. Recompute weighted ligands
        let w_lr_new = recompute_weighted_ligands(
            &gene_mtx_1,
            &gene_to_idx,
            &lr_ligands,
            xy,
            lr_radii,
            config.scale_factor,
            config.min_expression,
            config.ligand_grid_factor,
        );
        let w_tfl_new = recompute_weighted_ligands(
            &gene_mtx_1,
            &gene_to_idx,
            &tfl_ligands,
            xy,
            lr_radii,
            config.scale_factor,
            config.min_expression,
            config.ligand_grid_factor,
        );

        // 4. Delta in received ligands
        let rw_max_1 = scatter_max_to_full(&w_lr_new, &w_tfl_new, &gene_to_idx, n_cells, n_genes);
        let delta_rw = &rw_max_1 - &rw_max_0;

        // rw_lr_for_splash becomes max-combined for next iteration (Python behavior)
        rw_lr_for_splash = GeneMatrix::new(rw_max_1.mapv(|v| v as f32), gene_names.to_vec());

        // 5. Delta in ligand expression
        let mut ligands_1 = Array2::zeros((n_cells, n_genes));
        for &idx in &ligand_gene_indices {
            ligands_1.column_mut(idx).assign(&gene_mtx_1.column(idx));
        }
        let delta_ligands = &ligands_1 - &ligands_0;

        // 6. Replace ligand deltas with received-ligand deltas
        delta_simulated = &delta_simulated + &delta_rw - &delta_ligands;

        // 7. Perturb all cells: delta_y = splash_derivatives · delta_x
        delta_simulated = perturb_all_cells(gene_names, bb, &splashed, &delta_simulated);

        // 8. Pin target genes to their perturbed values
        Zip::from(&mut delta_simulated)
            .and(&delta_input)
            .for_each(|d, &di| {
                if di != 0.0 {
                    *d = di;
                }
            });

        // 9. Clip to [0, max_observed]
        let gem = gene_mtx + &delta_simulated;
        let gem_flat = gem.as_slice().unwrap();
        let gmtx_flat = gene_mtx.as_slice().unwrap();
        let delta_flat = delta_simulated.as_slice_memory_order_mut().unwrap();
        for cell in 0..n_cells {
            let base = cell * n_genes;
            for gene in 0..n_genes {
                let idx = base + gene;
                let val = gem_flat[idx].max(0.0).min(max_per_gene[gene]);
                delta_flat[idx] = val - gmtx_flat[idx];
            }
        }
    }

    if let Some(p) = job_progress {
        p.store(900, Ordering::Relaxed);
    }

    let mut simulated = gene_mtx + &delta_simulated;
    for target in targets {
        if let Some(&idx) = gene_to_idx.get(target.gene.as_str()) {
            if let Some(cell_indices) = target.cell_indices.as_ref() {
                for &cell in cell_indices {
                    if cell < n_cells {
                        simulated[[cell, idx]] = target.desired_expr;
                    }
                }
            } else {
                for cell in 0..n_cells {
                    simulated[[cell, idx]] = target.desired_expr;
                }
            }
        }
    }

    PerturbResult {
        simulated,
        delta: delta_simulated,
    }
}

/// max(rw_lr, rw_tfl) scattered into a (n_cells × n_genes) dense array.
fn scatter_max_to_full(
    rw_lr: &GeneMatrix,
    rw_tfl: &GeneMatrix,
    gene_to_idx: &HashMap<&str, usize>,
    n_cells: usize,
    n_genes: usize,
) -> Array2<f64> {
    let mut result = Array2::zeros((n_cells, n_genes));
    for (j, name) in rw_lr.col_names.iter().enumerate() {
        if let Some(&gi) = gene_to_idx.get(name.as_str()) {
            for c in 0..n_cells {
                result[[c, gi]] = rw_lr.data[[c, j]] as f64;
            }
        }
    }
    for (j, name) in rw_tfl.col_names.iter().enumerate() {
        if let Some(&gi) = gene_to_idx.get(name.as_str()) {
            for c in 0..n_cells {
                result[[c, gi]] = result[[c, gi]].max(rw_tfl.data[[c, j]] as f64);
            }
        }
    }
    result
}

fn splash_all(
    bb: &Betabase,
    rw_ligands: &GeneMatrix,
    rw_tfligands: &GeneMatrix,
    gex_df: &GeneMatrix,
    beta_scale_factor: f32,
    beta_cap: Option<f32>,
) -> HashMap<String, GeneMatrix> {
    bb.data
        .iter()
        .map(|(gene_name, bf)| {
            let splash = bf.splash(
                rw_ligands,
                rw_tfligands,
                gex_df,
                beta_scale_factor,
                beta_cap,
            );
            (gene_name.clone(), splash)
        })
        .collect()
}

/// For each gene with a trained model:
///   result[cell, gene_idx] = Σ_k splash[cell, k] · delta[cell, mod_idx[k]]
fn perturb_all_cells(
    gene_names: &[String],
    bb: &Betabase,
    splashed: &HashMap<String, GeneMatrix>,
    delta_simulated: &Array2<f64>,
) -> Array2<f64> {
    let n_cells = delta_simulated.nrows();
    let n_genes = gene_names.len();

    struct GeneWork<'a> {
        gene_col: usize,
        splash_flat: &'a [f32],
        n_mods: usize,
        mod_indices: &'a [usize],
    }

    let work: Vec<GeneWork> = gene_names
        .iter()
        .enumerate()
        .filter_map(|(gene_idx, gene_name)| {
            let splash = splashed.get(gene_name)?;
            let bf = bb.data.get(gene_name)?;
            let mod_indices = bf.modulator_gene_indices.as_ref()?;
            Some(GeneWork {
                gene_col: gene_idx,
                splash_flat: splash.data.as_slice().unwrap(),
                n_mods: splash.data.ncols(),
                mod_indices: mod_indices.as_slice(),
            })
        })
        .collect();

    let delta_flat = delta_simulated.as_slice().unwrap();
    let mut result = vec![0.0f64; n_cells * n_genes];

    result
        .par_chunks_mut(n_genes)
        .enumerate()
        .for_each(|(cell, r)| {
            let delta_base = cell * n_genes;
            for w in &work {
                let splash_base = cell * w.n_mods;
                let mut sum = 0.0f64;
                for k in 0..w.n_mods {
                    unsafe {
                        sum += f64::from(*w.splash_flat.get_unchecked(splash_base + k))
                            * *delta_flat
                                .get_unchecked(delta_base + *w.mod_indices.get_unchecked(k));
                    }
                }
                r[w.gene_col] = sum;
            }
        });

    Array2::from_shape_vec((n_cells, n_genes), result).unwrap()
}

fn recompute_weighted_ligands(
    gene_mtx: &Array2<f64>,
    gene_to_idx: &HashMap<&str, usize>,
    ligand_names: &[String],
    xy: &Array2<f64>,
    lr_radii: &HashMap<String, f64>,
    scale_factor: f64,
    min_expression: f64,
    grid_factor: Option<f64>,
) -> GeneMatrix {
    let n_cells = gene_mtx.nrows();
    if ligand_names.is_empty() {
        return GeneMatrix::new(Array2::<f32>::zeros((n_cells, 0)), Vec::new());
    }

    let mut seen = HashSet::new();
    let unique_ligands: Vec<&String> = ligand_names
        .iter()
        .filter(|l| seen.insert(l.as_str()))
        .collect();

    let mut lig_names: Vec<String> = Vec::new();
    let mut col_data: Vec<Vec<f64>> = Vec::new();

    for &lig in &unique_ligands {
        if let Some(&gene_idx) = gene_to_idx.get(lig.as_str()) {
            lig_names.push(lig.clone());
            let col: Vec<f64> = (0..n_cells)
                .map(|i| {
                    let v = gene_mtx[[i, gene_idx]];
                    if v > min_expression { v } else { 0.0 }
                })
                .collect();
            col_data.push(col);
        }
    }

    if lig_names.is_empty() {
        return GeneMatrix::new(Array2::<f32>::zeros((n_cells, 0)), Vec::new());
    }

    let n_lig = lig_names.len();
    let mut lig_data = Array2::<f64>::zeros((n_cells, n_lig));
    for (j, col) in col_data.iter().enumerate() {
        for i in 0..n_cells {
            lig_data[[i, j]] = col[i];
        }
    }

    // Group by radius
    let mut radius_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (j, name) in lig_names.iter().enumerate() {
        if let Some(&radius) = lr_radii.get(name) {
            radius_groups.entry(radius.to_bits()).or_default().push(j);
        }
    }

    let mut result_data = Array2::<f32>::zeros((n_cells, n_lig));

    for (radius_bits, group_indices) in &radius_groups {
        let radius = f64::from_bits(*radius_bits);
        let mut sub = Array2::<f64>::zeros((n_cells, group_indices.len()));
        for (k, &j) in group_indices.iter().enumerate() {
            sub.column_mut(k).assign(&lig_data.column(j));
        }
        let weighted = match grid_factor {
            Some(gf) => calculate_weighted_ligands_grid(xy, &sub, radius, scale_factor, gf),
            None => calculate_weighted_ligands(xy, &sub, radius, scale_factor),
        };
        for (k, &j) in group_indices.iter().enumerate() {
            let col = weighted.column(k);
            for i in 0..n_cells {
                result_data[[i, j]] = col[i] as f32;
            }
        }
    }

    GeneMatrix::new(result_data, lig_names)
}
