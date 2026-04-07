use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ndarray::{Array2, Zip};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::betadata::{Betabase, GeneMatrix};
use crate::ligand::{
    calculate_weighted_ligands_grid_with_cutoff, calculate_weighted_ligands_with_cutoff,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct PerturbTarget {
    pub gene: String,
    pub desired_expr: f64,
    pub cell_indices: Option<Vec<usize>>,
}

#[derive(Clone, Serialize, Deserialize)]
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
    /// Hard cutoff on sender distance for received-ligand aggregation. None = full Gaussian support.
    #[serde(default)]
    pub contact_distance: Option<f64>,
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
            contact_distance: None,
        }
    }
}

/// Key for iteration-0 splash reuse across perturbations (same baseline expression / RW state).
#[derive(Clone, PartialEq)]
pub struct SplashCacheKey {
    pub beta_scale_factor: f32,
    pub beta_cap: Option<f32>,
    pub min_expression: f64,
}

pub struct CachedBaselineSplash {
    pub key: SplashCacheKey,
    pub splashed: Arc<HashMap<String, GeneMatrix>>,
}

#[derive(Default)]
pub struct PerturbTimings {
    pub entries: Vec<(String, std::time::Duration)>,
}

impl PerturbTimings {
    pub fn record(&mut self, label: impl Into<String>, d: std::time::Duration) {
        self.entries.push((label.into(), d));
    }
}

pub struct PerturbResult {
    pub simulated: Array2<f64>,
    pub delta: Array2<f64>,
}

/// Rebuild [`PerturbResult::simulated`] from baseline expression and final δ (matches the end of [`perturb_with_targets`]).
pub fn perturb_result_from_delta(
    gene_mtx: &Array2<f64>,
    delta: Array2<f64>,
    targets: &[PerturbTarget],
    gene_names: &[String],
) -> PerturbResult {
    let n_cells = gene_mtx.nrows();
    let gene_to_idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
    let mut simulated = gene_mtx + &delta;
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
    PerturbResult { simulated, delta }
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
    let mut no_timings: Option<PerturbTimings> = None;
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
        None,
        None,
        None,
        &mut no_timings,
    )
    .expect("cancel is only used by spatial viewer")
}

#[inline]
fn report_perturb_step(
    job_progress: Option<&Arc<AtomicU32>>,
    job_message: Option<&Arc<Mutex<String>>>,
    permille: u32,
    message: &str,
) {
    if let Some(p) = job_progress {
        p.store(permille.min(1000), Ordering::Relaxed);
    }
    if let Some(m) = job_message {
        if let Ok(mut g) = m.lock() {
            *g = message.to_string();
        }
    }
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
    job_message: Option<&Arc<Mutex<String>>>,
    cancel: Option<&AtomicBool>,
    baseline_splash_cache: Option<&Mutex<Option<CachedBaselineSplash>>>,
    timings: &mut Option<PerturbTimings>,
) -> Result<PerturbResult, ()> {
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

    let mut gene_mtx_work: Option<Array2<f64>> = None;
    let mut perturb_scratch: Vec<f64> = vec![0.0f64; n_cells * n_genes];

    let lr_ligands: Vec<String> = bb.ligands_set.iter().cloned().collect();
    let tfl_ligands: Vec<String> = bb.tfl_ligands_set.iter().cloned().collect();

    let n_prop = config.n_propagation.max(1);
    let n_prop_u = n_prop as u32;
    const PROP_LO: u32 = 25;
    const PROP_HI: u32 = 915;
    let span = ((PROP_HI - PROP_LO) / n_prop_u).max(1u32);

    report_perturb_step(
        job_progress,
        job_message,
        15,
        "GRN perturbation · building target δ…",
    );

    for iter in 0..n_prop {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(());
        }
        if job_progress.is_none() && job_message.is_none() {
            eprintln!("  perturb iteration {}/{}", iter + 1, n_prop);
        }
        let iter_u = iter as u32;
        let base = PROP_LO + iter_u * span;
        let msg_prefix = format!("GRN propagation {}/{}", iter + 1, n_prop);
        report_perturb_step(
            job_progress,
            job_message,
            base,
            &format!("{msg_prefix} · splash & derivatives"),
        );

        // 1. Splash all trained genes (expression → f32 for splash / betabase RAM)
        let t_splash = Instant::now();
        let splash_key = SplashCacheKey {
            beta_scale_factor: config.beta_scale_factor as f32,
            beta_cap: config.beta_cap.map(|c| c as f32),
            min_expression: config.min_expression,
        };
        let expr_for_splash: &Array2<f64> = gene_mtx_work.as_ref().map_or(gene_mtx, |m| m);
        let splashed: Arc<HashMap<String, GeneMatrix>> = if iter == 0 {
            if let Some(slot) = baseline_splash_cache {
                let mut guard = slot.lock().expect("baseline splash cache poisoned");
                if let Some(cached) = guard.as_ref() {
                    if cached.key == splash_key {
                        Arc::clone(&cached.splashed)
                    } else {
                        let gex_gm = gene_matrix_masked_f32_from_expr(
                            expr_for_splash,
                            config.min_expression,
                            gene_names,
                        );
                        let Some(map) = compute_splash_all_progress(
                            bb,
                            &rw_lr_for_splash,
                            rw_tfligands_init,
                            &gex_gm,
                            config.beta_scale_factor as f32,
                            config.beta_cap.map(|c| c as f32),
                            job_progress.map(|p| p.as_ref()),
                            cancel,
                        ) else {
                            drop(guard);
                            return Err(());
                        };
                        let arc = Arc::new(map);
                        *guard = Some(CachedBaselineSplash {
                            key: splash_key,
                            splashed: Arc::clone(&arc),
                        });
                        arc
                    }
                } else {
                    let gex_gm = gene_matrix_masked_f32_from_expr(
                        expr_for_splash,
                        config.min_expression,
                        gene_names,
                    );
                    let Some(map) = compute_splash_all_progress(
                        bb,
                        &rw_lr_for_splash,
                        rw_tfligands_init,
                        &gex_gm,
                        config.beta_scale_factor as f32,
                        config.beta_cap.map(|c| c as f32),
                        job_progress.map(|p| p.as_ref()),
                        cancel,
                    ) else {
                        drop(guard);
                        return Err(());
                    };
                    let arc = Arc::new(map);
                    *guard = Some(CachedBaselineSplash {
                        key: splash_key,
                        splashed: Arc::clone(&arc),
                    });
                    arc
                }
            } else {
                let gex_gm = gene_matrix_masked_f32_from_expr(
                    expr_for_splash,
                    config.min_expression,
                    gene_names,
                );
                let Some(map) = compute_splash_all_progress(
                    bb,
                    &rw_lr_for_splash,
                    rw_tfligands_init,
                    &gex_gm,
                    config.beta_scale_factor as f32,
                    config.beta_cap.map(|c| c as f32),
                    job_progress.map(|p| p.as_ref()),
                    cancel,
                ) else {
                    return Err(());
                };
                Arc::new(map)
            }
        } else {
            let gex_gm = gene_matrix_masked_f32_from_expr(
                expr_for_splash,
                config.min_expression,
                gene_names,
            );
            let Some(map) = compute_splash_all_progress(
                bb,
                &rw_lr_for_splash,
                rw_tfligands_init,
                &gex_gm,
                config.beta_scale_factor as f32,
                config.beta_cap.map(|c| c as f32),
                job_progress.map(|p| p.as_ref()),
                cancel,
            ) else {
                return Err(());
            };
            Arc::new(map)
        };
        if let Some(t) = timings.as_mut() {
            t.record(format!("iter{}/splash", iter + 1), t_splash.elapsed());
        }
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(());
        }
        report_perturb_step(
            job_progress,
            job_message,
            base + span.saturating_mul(1) / 5,
            &format!("{msg_prefix} · spatial ligands (LR)"),
        );

        // 2. Update gene expression
        let t_lr = Instant::now();
        gene_mtx_work = Some(gene_mtx + &delta_simulated);
        let gene_mtx_1 = gene_mtx_work.as_ref().expect("gene_mtx_work set above");

        // 3. Recompute weighted ligands
        let Some(w_lr_new) = recompute_weighted_ligands(
            gene_mtx_1,
            &gene_to_idx,
            &lr_ligands,
            xy,
            lr_radii,
            config.scale_factor,
            config.min_expression,
            config.ligand_grid_factor,
            config.contact_distance,
            cancel,
        ) else {
            return Err(());
        };
        if let Some(t) = timings.as_mut() {
            t.record(
                format!("iter{}/weighted_ligands_lr", iter + 1),
                t_lr.elapsed(),
            );
        }
        report_perturb_step(
            job_progress,
            job_message,
            base + span.saturating_mul(2) / 5,
            &format!("{msg_prefix} · spatial ligands (TFL)"),
        );
        let t_tfl = Instant::now();
        let Some(w_tfl_new) = recompute_weighted_ligands(
            gene_mtx_1,
            &gene_to_idx,
            &tfl_ligands,
            xy,
            lr_radii,
            config.scale_factor,
            config.min_expression,
            config.ligand_grid_factor,
            config.contact_distance,
            cancel,
        ) else {
            return Err(());
        };
        if let Some(t) = timings.as_mut() {
            t.record(
                format!("iter{}/weighted_ligands_tfl", iter + 1),
                t_tfl.elapsed(),
            );
        }

        // 4. Delta in received ligands
        let t_grn = Instant::now();
        let lr_col_names = w_lr_new.col_names.clone();
        let rw_max_1 = scatter_max_to_full(&w_lr_new, &w_tfl_new, &gene_to_idx, n_cells, n_genes);
        drop((w_lr_new, w_tfl_new));

        let delta_rw = &rw_max_1 - &rw_max_0;
        rw_lr_for_splash = gene_matrix_narrow_lr_from_full(&rw_max_1, &gene_to_idx, &lr_col_names);
        drop(rw_max_1);

        // 5–6. Replace direct ligand expression deltas with received-ligand deltas (Python parity):
        // δ ← δ + (rw₁−rw₀) − (lig_expr₁−lig_expr₀); non-ligand genes: only +Δrw.
        delta_simulated = &delta_simulated + &delta_rw;
        for &idx in &ligand_gene_indices {
            let l0 = ligands_0.column(idx);
            let l1 = gene_mtx_1.column(idx);
            let mut dcol = delta_simulated.column_mut(idx);
            Zip::from(&mut dcol)
                .and(&l1)
                .and(&l0)
                .for_each(|d, &v1, &v0| {
                    *d -= v1 - v0;
                });
        }

        report_perturb_step(
            job_progress,
            job_message,
            base + span.saturating_mul(3) / 5,
            &format!("{msg_prefix} · GRN step (δ → Δexpr)"),
        );
        // 7. Perturb all cells: delta_y = splash_derivatives · delta_x
        perturb_all_cells_into(
            gene_names,
            bb,
            splashed.as_ref(),
            &delta_simulated,
            &mut perturb_scratch,
        );
        delta_simulated
            .as_slice_memory_order_mut()
            .unwrap()
            .copy_from_slice(&perturb_scratch);
        if let Some(t) = timings.as_mut() {
            t.record(format!("iter{}/grn_propagate", iter + 1), t_grn.elapsed());
        }

        // 8. Pin target genes to their perturbed values (only target columns)
        let t_pin_nonneg = Instant::now();
        for target in targets {
            if let Some(&gi) = gene_to_idx.get(target.gene.as_str()) {
                if let Some(cell_indices) = target.cell_indices.as_ref() {
                    for &cell in cell_indices {
                        if cell < n_cells {
                            delta_simulated[[cell, gi]] = delta_input[[cell, gi]];
                        }
                    }
                } else {
                    delta_simulated
                        .column_mut(gi)
                        .assign(&delta_input.column(gi));
                }
            }
        }

        // 9. Enforce simulated expression ≥ 0 (zero-alloc, parallel)
        let delta_flat = delta_simulated.as_slice_memory_order_mut().unwrap();
        let gmtx_flat = gene_mtx.as_slice().unwrap();
        delta_flat
            .par_chunks_mut(n_genes)
            .enumerate()
            .for_each(|(cell, row)| {
                let base = cell * n_genes;
                for gene in 0..n_genes {
                    unsafe {
                        let orig = *gmtx_flat.get_unchecked(base + gene);
                        let val = (orig + *row.get_unchecked(gene)).max(0.0);
                        *row.get_unchecked_mut(gene) = val - orig;
                    }
                }
            });
        if let Some(t) = timings.as_mut() {
            t.record(format!("iter{}/pin_nonneg", iter + 1), t_pin_nonneg.elapsed());
        }
        report_perturb_step(
            job_progress,
            job_message,
            (base + span).saturating_sub(1).min(PROP_HI),
            &format!("{msg_prefix} · nonneg & sync"),
        );
    }

    report_perturb_step(
        job_progress,
        job_message,
        930,
        "GRN perturbation · assembling result…",
    );

    let out = perturb_result_from_delta(gene_mtx, delta_simulated, targets, gene_names);

    report_perturb_step(
        job_progress,
        job_message,
        1000,
        "GRN perturbation · complete",
    );

    Ok(out)
}

/// Copy **LR received-ligand** channels from a full `scatter_max` matrix into the narrow
/// layout expected by [`BetaFrame::splash`] (column lookup by ligand gene name only).
///
/// Building splash input with all `n_genes` columns wasted ~`n_cells × n_genes × 4` bytes per
/// propagation iteration and contributed to OOM on large atlases.
fn gene_matrix_narrow_lr_from_full(
    full: &Array2<f64>,
    gene_to_idx: &HashMap<&str, usize>,
    lr_col_names: &[String],
) -> GeneMatrix {
    let n_cells = full.nrows();
    if lr_col_names.is_empty() {
        return GeneMatrix::new(ndarray::Array2::<f32>::zeros((n_cells, 0)), Vec::new());
    }
    let mut data = ndarray::Array2::<f32>::zeros((n_cells, lr_col_names.len()));
    for (j, name) in lr_col_names.iter().enumerate() {
        if let Some(&gi) = gene_to_idx.get(name.as_str()) {
            let src = full.column(gi);
            let mut dst = data.column_mut(j);
            for i in 0..n_cells {
                dst[i] = src[i] as f32;
            }
        }
    }
    GeneMatrix::new(data, lr_col_names.to_vec())
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

fn gene_matrix_masked_f32_from_expr(
    expr: &Array2<f64>,
    min_expression: f64,
    gene_names: &[String],
) -> GeneMatrix {
    let n_cells = expr.nrows();
    let n_genes = expr.ncols();
    let mut out = ndarray::Array2::<f32>::zeros((n_cells, n_genes));
    Zip::from(&mut out).and(expr).for_each(|o, &v| {
        *o = if v > min_expression { v as f32 } else { 0.0 };
    });
    GeneMatrix::new(out, gene_names.to_vec())
}

/// Partial derivatives ∂(target)/∂(modulator) for every trained target (baseline WL + expression).
pub fn compute_splash_all(
    bb: &Betabase,
    rw_ligands: &GeneMatrix,
    rw_tfligands: &GeneMatrix,
    gex_df: &GeneMatrix,
    beta_scale_factor: f32,
    beta_cap: Option<f32>,
) -> HashMap<String, GeneMatrix> {
    compute_splash_all_progress(
        bb,
        rw_ligands,
        rw_tfligands,
        gex_df,
        beta_scale_factor,
        beta_cap,
        None,
        None,
    )
    .expect("compute_splash_all_progress without cancel must return Some")
}

/// Like [`compute_splash_all`], optionally reporting coarse progress on `progress` (permille 0–1000).
/// Updates are throttled (~≤30 calls) to avoid sync overhead on large target counts.
///
/// Returns `None` when `cancel` is set and becomes true during computation.
pub fn compute_splash_all_progress(
    bb: &Betabase,
    rw_ligands: &GeneMatrix,
    rw_tfligands: &GeneMatrix,
    gex_df: &GeneMatrix,
    beta_scale_factor: f32,
    beta_cap: Option<f32>,
    progress: Option<&std::sync::atomic::AtomicU32>,
    cancel: Option<&AtomicBool>,
) -> Option<HashMap<String, GeneMatrix>> {
    use std::sync::atomic::Ordering;
    let n = bb.data.len().max(1);
    let step = (n / 28).max(1);
    let mut out = HashMap::with_capacity(bb.data.len());
    for (i, (gene_name, bf)) in bb.data.iter().enumerate() {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return None;
        }
        let splash = bf.splash(
            rw_ligands,
            rw_tfligands,
            gex_df,
            beta_scale_factor,
            beta_cap,
        );
        out.insert(gene_name.clone(), splash);
        if let Some(p) = progress {
            if i % step == 0 || i + 1 == n {
                let v = 50u32 + ((i as u32 + 1) * 700 / n as u32);
                p.store(v.min(750), Ordering::Relaxed);
            }
        }
    }
    Some(out)
}

/// For each gene with a trained model:
///   out[cell, gene_idx] = Σ_k splash[cell, k] · delta[cell, mod_idx[k]]
///
/// `out_row_major` must have length `n_cells * n_genes` (row-major); it is zeroed then filled.
fn perturb_all_cells_into(
    gene_names: &[String],
    bb: &Betabase,
    splashed: &HashMap<String, GeneMatrix>,
    delta_simulated: &Array2<f64>,
    out_row_major: &mut [f64],
) {
    let n_cells = delta_simulated.nrows();
    let n_genes = gene_names.len();
    assert_eq!(out_row_major.len(), n_cells * n_genes);

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

    let delta_flat = delta_simulated.as_slice_memory_order().unwrap();
    out_row_major.fill(0.0);
    out_row_major
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
    contact_distance: Option<f64>,
    cancel: Option<&AtomicBool>,
) -> Option<GeneMatrix> {
    use std::sync::atomic::Ordering;

    let n_cells = gene_mtx.nrows();
    if ligand_names.is_empty() {
        return Some(GeneMatrix::new(
            Array2::<f32>::zeros((n_cells, 0)),
            Vec::new(),
        ));
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
        return Some(GeneMatrix::new(
            Array2::<f32>::zeros((n_cells, 0)),
            Vec::new(),
        ));
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
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return None;
        }
        let radius = f64::from_bits(*radius_bits);
        let mut sub = Array2::<f64>::zeros((n_cells, group_indices.len()));
        for (k, &j) in group_indices.iter().enumerate() {
            sub.column_mut(k).assign(&lig_data.column(j));
        }
        let weighted = match grid_factor {
            Some(gf) if gf.is_finite() && gf > 0.0 => {
                calculate_weighted_ligands_grid_with_cutoff(
                    xy,
                    &sub,
                    radius,
                    scale_factor,
                    gf,
                    contact_distance,
                )
            }
            _ => calculate_weighted_ligands_with_cutoff(
                xy,
                &sub,
                radius,
                scale_factor,
                contact_distance,
            ),
        };
        for (k, &j) in group_indices.iter().enumerate() {
            let col = weighted.column(k);
            for i in 0..n_cells {
                result_data[[i, j]] = col[i] as f32;
            }
        }
    }

    Some(GeneMatrix::new(result_data, lig_names))
}
