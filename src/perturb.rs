use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ndarray::{Array2, Zip};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::betadata::{Betabase, GeneMatrix, SplashGex};
use crate::ligand::{
    calculate_weighted_ligands_grid_with_cutoff, calculate_weighted_ligands_with_cutoff,
};

pub use crate::config::SplashMode;

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
    /// Lower clip for simulated gene expression after each propagation iteration (default `0.0` when omitted).
    #[serde(default)]
    pub perturbed_gene_min_bound: Option<f64>,
    /// Upper clip for simulated gene expression after each propagation iteration (omit for no upper bound).
    #[serde(default)]
    pub perturbed_gene_max_bound: Option<f64>,
    #[serde(default)]
    pub splash_mode: SplashMode,
    #[serde(default)]
    pub splash_jacobian_max_mb: Option<u64>,
}

#[allow(clippy::derivable_impls)]
impl Default for PerturbConfig {
    fn default() -> Self {
        Self {
            n_propagation: 4,
            scale_factor: 1.0,
            beta_scale_factor: 100.0,
            beta_cap: None,
            min_expression: 1e-9,
            ligand_grid_factor: None,
            contact_distance: None,
            perturbed_gene_min_bound: None,
            perturbed_gene_max_bound: None,
            splash_mode: SplashMode::Auto,
            splash_jacobian_max_mb: None,
        }
    }
}

/// Resolved per-gene expression clip applied during perturbation propagation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpressionBounds {
    pub min: f64,
    pub max: f64,
}

impl ExpressionBounds {
    pub fn from_config(config: &PerturbConfig) -> Self {
        Self {
            min: config.perturbed_gene_min_bound.unwrap_or(0.0),
            max: config.perturbed_gene_max_bound.unwrap_or(f64::INFINITY),
        }
    }

    pub fn clip_value(&self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }
}

const SPLASH_JACOBIAN_MIB: usize = 1024 * 1024;
const SPLASH_JACOBIAN_DEFAULT_CAP_MIB: usize = 2048;

/// Bytes for a fully materialized splash HashMap plus the f32 GEX copy used only on that path.
pub fn estimated_splash_jacobian_bytes(bb: &Betabase, n_cells: usize, n_genes: usize) -> usize {
    let jac = bb.data.values().fold(0usize, |acc, bf| {
        acc.saturating_add(
            n_cells
                .saturating_mul(bf.modulator_genes.len())
                .saturating_mul(4),
        )
    });
    let gex = n_cells.saturating_mul(n_genes).saturating_mul(4);
    jac.saturating_add(gex)
}

/// RAM budget for [`SplashMode::Auto`]: config MB, then env, then sysinfo, else 2048 MiB.
pub fn splash_jacobian_budget_bytes(config: &PerturbConfig) -> usize {
    if let Some(mb) = config.splash_jacobian_max_mb {
        if mb > 0 {
            return (mb as usize).saturating_mul(SPLASH_JACOBIAN_MIB);
        }
    }
    if let Ok(s) = std::env::var("SPACETRAVLR_SPLASH_JACOBIAN_MAX_MB") {
        if let Ok(mb) = s.parse::<u64>() {
            if mb > 0 {
                return (mb as usize).saturating_mul(SPLASH_JACOBIAN_MIB);
            }
        }
    }
    #[cfg(feature = "tui")]
    {
        use sysinfo::{MemoryRefreshKind, RefreshKind, System};
        let mut sys = System::new_with_specifics(
            RefreshKind::new().with_memory(MemoryRefreshKind::everything()),
        );
        sys.refresh_memory();
        let avail = sys.available_memory() as usize;
        let budget = ((avail as f64) * 0.4).round() as usize;
        if budget > 0 {
            return budget;
        }
    }
    SPLASH_JACOBIAN_DEFAULT_CAP_MIB.saturating_mul(SPLASH_JACOBIAN_MIB)
}

/// Resolved GRN path: `fused` or `materialize` (never `auto`).
pub fn resolve_splash_mode(config: &PerturbConfig, estimated_bytes: usize) -> SplashMode {
    match config.splash_mode {
        SplashMode::Fused => SplashMode::Fused,
        SplashMode::Materialize => SplashMode::Materialize,
        SplashMode::Auto => {
            if estimated_bytes > splash_jacobian_budget_bytes(config) {
                SplashMode::Fused
            } else {
                SplashMode::Materialize
            }
        }
    }
}

fn format_gib(bytes: usize) -> String {
    format!("{:.3}", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
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

/// Argument bundle for [`perturb`] (same fields as the previous nine-parameter API).
pub struct PerturbInputs<'a> {
    pub bb: &'a Betabase,
    pub gene_mtx: &'a Array2<f64>,
    pub gene_names: &'a [String],
    pub xy: &'a Array2<f64>,
    pub rw_ligands_init: &'a GeneMatrix,
    pub rw_tfligands_init: &'a GeneMatrix,
    pub targets: &'a [(String, f64)],
    pub config: &'a PerturbConfig,
    pub lr_radii: &'a HashMap<String, f64>,
}

pub struct PerturbWithTargetsInputs<'a> {
    pub bb: &'a Betabase,
    pub gene_mtx: &'a Array2<f64>,
    pub gene_names: &'a [String],
    pub xy: &'a Array2<f64>,
    pub rw_ligands_init: &'a GeneMatrix,
    pub rw_tfligands_init: &'a GeneMatrix,
    pub targets: &'a [PerturbTarget],
    pub config: &'a PerturbConfig,
    pub lr_radii: &'a HashMap<String, f64>,
    pub job_progress: Option<&'a Arc<AtomicU32>>,
    pub job_message: Option<&'a Arc<Mutex<String>>>,
    pub cancel: Option<&'a AtomicBool>,
    pub baseline_splash_cache: Option<&'a Mutex<Option<CachedBaselineSplash>>>,
}

/// Rebuild [`PerturbResult::simulated`] from baseline expression and final δ (matches the end of [`perturb_with_targets`]).
pub fn perturb_result_from_delta(
    gene_mtx: &Array2<f64>,
    delta: Array2<f64>,
    targets: &[PerturbTarget],
    gene_names: &[String],
    bounds: Option<ExpressionBounds>,
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
            let desired = bounds
                .map(|b| b.clip_value(target.desired_expr))
                .unwrap_or(target.desired_expr);
            if let Some(cell_indices) = target.cell_indices.as_ref() {
                for &cell in cell_indices {
                    if cell < n_cells {
                        simulated[[cell, idx]] = desired;
                    }
                }
            } else {
                for cell in 0..n_cells {
                    simulated[[cell, idx]] = desired;
                }
            }
        }
    }
    if let Some(bounds) = bounds {
        clip_simulated_matrix_in_place(&mut simulated, bounds);
    }
    PerturbResult { simulated, delta }
}

/// Simulate gene perturbation and propagate effects through the spatial GRN.
///
/// Mirrors Python's `GeneFactory.perturb()`: each iteration computes splash
/// derivatives, recomputes spatially-weighted ligands for the updated expression,
/// swaps direct ligand deltas with received-ligand deltas, then applies
/// delta × splash to propagate effects to all downstream genes.
pub fn perturb(inputs: PerturbInputs<'_>) -> PerturbResult {
    let scoped_targets: Vec<PerturbTarget> = inputs
        .targets
        .iter()
        .map(|(gene, desired_expr)| PerturbTarget {
            gene: gene.clone(),
            desired_expr: *desired_expr,
            cell_indices: None,
        })
        .collect();
    let mut no_timings: Option<PerturbTimings> = None;
    perturb_with_targets(
        &PerturbWithTargetsInputs {
            bb: inputs.bb,
            gene_mtx: inputs.gene_mtx,
            gene_names: inputs.gene_names,
            xy: inputs.xy,
            rw_ligands_init: inputs.rw_ligands_init,
            rw_tfligands_init: inputs.rw_tfligands_init,
            targets: &scoped_targets,
            config: inputs.config,
            lr_radii: inputs.lr_radii,
            job_progress: None,
            job_message: None,
            cancel: None,
            baseline_splash_cache: None,
        },
        &mut no_timings,
    )
    .expect("perturb_with_targets completes without cancel when job hooks are unset")
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
    inputs: &PerturbWithTargetsInputs<'_>,
    timings: &mut Option<PerturbTimings>,
) -> Option<PerturbResult> {
    let bb = inputs.bb;
    let gene_mtx = inputs.gene_mtx;
    let gene_names = inputs.gene_names;
    let xy = inputs.xy;
    let rw_ligands_init = inputs.rw_ligands_init;
    let rw_tfligands_init = inputs.rw_tfligands_init;
    let targets = inputs.targets;
    let config = inputs.config;
    let lr_radii = inputs.lr_radii;
    let expression_bounds = ExpressionBounds::from_config(config);
    let job_progress = inputs.job_progress;
    let job_message = inputs.job_message;
    let cancel = inputs.cancel;
    let baseline_splash_cache = inputs.baseline_splash_cache;
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

    let splash_est = estimated_splash_jacobian_bytes(bb, n_cells, n_genes);
    let splash_budget = splash_jacobian_budget_bytes(config);
    let splash_compute = resolve_splash_mode(config, splash_est);
    let use_fused = splash_compute == SplashMode::Fused;
    let splash_mode_msg = format!(
        "splash_mode={} estimated={} GiB budget={} GiB",
        splash_compute,
        format_gib(splash_est),
        format_gib(splash_budget)
    );
    if timings.is_some() {
        eprintln!("  {splash_mode_msg}");
    }
    report_perturb_step(
        job_progress,
        job_message,
        15,
        &format!("GRN perturbation · building target δ · {splash_mode_msg}"),
    );

    for iter in 0..n_prop {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return None;
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
        let prev_expr = gene_mtx_work.take();
        let expr_for_splash: &Array2<f64> = prev_expr.as_ref().unwrap_or(gene_mtx);
        let rw_lr_fused = if use_fused {
            Some(GeneMatrix::new(
                rw_lr_for_splash.data.clone(),
                rw_lr_for_splash.col_names.clone(),
            ))
        } else {
            None
        };
        let splashed: Option<Arc<HashMap<String, GeneMatrix>>> = if use_fused {
            None
        } else if iter == 0 {
            if let Some(slot) = baseline_splash_cache {
                let mut guard = slot.lock().expect("baseline splash cache poisoned");
                if let Some(cached) = guard.as_ref() {
                    if cached.key == splash_key {
                        Some(Arc::clone(&cached.splashed))
                    } else {
                        let gex_gm = gene_matrix_masked_f32_from_expr(
                            expr_for_splash,
                            config.min_expression,
                            gene_names,
                        );
                        let map = compute_splash_all_progress(ComputeSplashAllProgressArgs {
                            bb,
                            rw_ligands: &rw_lr_for_splash,
                            rw_tfligands: rw_tfligands_init,
                            gex_df: &gex_gm,
                            beta_scale_factor: config.beta_scale_factor as f32,
                            beta_cap: config.beta_cap.map(|c| c as f32),
                            progress: job_progress.map(|p| p.as_ref()),
                            cancel,
                        })?;
                        let arc = Arc::new(map);
                        *guard = Some(CachedBaselineSplash {
                            key: splash_key,
                            splashed: Arc::clone(&arc),
                        });
                        Some(arc)
                    }
                } else {
                    let gex_gm = gene_matrix_masked_f32_from_expr(
                        expr_for_splash,
                        config.min_expression,
                        gene_names,
                    );
                    let map = compute_splash_all_progress(ComputeSplashAllProgressArgs {
                        bb,
                        rw_ligands: &rw_lr_for_splash,
                        rw_tfligands: rw_tfligands_init,
                        gex_df: &gex_gm,
                        beta_scale_factor: config.beta_scale_factor as f32,
                        beta_cap: config.beta_cap.map(|c| c as f32),
                        progress: job_progress.map(|p| p.as_ref()),
                        cancel,
                    })?;
                    let arc = Arc::new(map);
                    *guard = Some(CachedBaselineSplash {
                        key: splash_key,
                        splashed: Arc::clone(&arc),
                    });
                    Some(arc)
                }
            } else {
                let gex_gm = gene_matrix_masked_f32_from_expr(
                    expr_for_splash,
                    config.min_expression,
                    gene_names,
                );
                let map = compute_splash_all_progress(ComputeSplashAllProgressArgs {
                    bb,
                    rw_ligands: &rw_lr_for_splash,
                    rw_tfligands: rw_tfligands_init,
                    gex_df: &gex_gm,
                    beta_scale_factor: config.beta_scale_factor as f32,
                    beta_cap: config.beta_cap.map(|c| c as f32),
                    progress: job_progress.map(|p| p.as_ref()),
                    cancel,
                })?;
                Some(Arc::new(map))
            }
        } else {
            let gex_gm = gene_matrix_masked_f32_from_expr(
                expr_for_splash,
                config.min_expression,
                gene_names,
            );
            let map = compute_splash_all_progress(ComputeSplashAllProgressArgs {
                bb,
                rw_ligands: &rw_lr_for_splash,
                rw_tfligands: rw_tfligands_init,
                gex_df: &gex_gm,
                beta_scale_factor: config.beta_scale_factor as f32,
                beta_cap: config.beta_cap.map(|c| c as f32),
                progress: job_progress.map(|p| p.as_ref()),
                cancel,
            })?;
            Some(Arc::new(map))
        };
        if !use_fused {
            if let Some(t) = timings.as_mut() {
                t.record(format!("iter{}/splash", iter + 1), t_splash.elapsed());
            }
        } else {
            let _ = t_splash;
        }
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return None;
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
        let w_lr_new = recompute_weighted_ligands(RecomputeWeightedLigandsArgs {
            gene_mtx: gene_mtx_1,
            gene_to_idx: &gene_to_idx,
            ligand_names: &lr_ligands,
            xy,
            lr_radii,
            scale_factor: config.scale_factor,
            min_expression: config.min_expression,
            grid_factor: config.ligand_grid_factor,
            contact_distance: config.contact_distance,
            cancel,
        })?;
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
        let w_tfl_new = recompute_weighted_ligands(RecomputeWeightedLigandsArgs {
            gene_mtx: gene_mtx_1,
            gene_to_idx: &gene_to_idx,
            ligand_names: &tfl_ligands,
            xy,
            lr_radii,
            scale_factor: config.scale_factor,
            min_expression: config.min_expression,
            grid_factor: config.ligand_grid_factor,
            contact_distance: config.contact_distance,
            cancel,
        })?;
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
        if use_fused {
            let t_fused = Instant::now();
            let rw = rw_lr_fused.as_ref().expect("fused rw snapshot");
            if !perturb_all_cells_fused_into(
                gene_names,
                bb,
                rw,
                rw_tfligands_init,
                expr_for_splash,
                config.min_expression,
                config.beta_scale_factor as f32,
                config.beta_cap.map(|c| c as f32),
                &delta_simulated,
                &mut perturb_scratch,
                cancel,
                job_progress.map(|p| p.as_ref()),
            ) {
                return None;
            }
            if let Some(t) = timings.as_mut() {
                t.record(format!("iter{}/splash", iter + 1), t_fused.elapsed());
            }
        } else {
            perturb_all_cells_into(
                gene_names,
                bb,
                splashed.as_ref().expect("materialized splash").as_ref(),
                &delta_simulated,
                &mut perturb_scratch,
            );
        }
        delta_simulated
            .as_slice_memory_order_mut()
            .unwrap()
            .copy_from_slice(&perturb_scratch);
        if let Some(t) = timings.as_mut() {
            t.record(format!("iter{}/grn_propagate", iter + 1), t_grn.elapsed());
        }

        // 8. Pin target genes to their perturbed values (only target columns)
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

        // 9. Clip simulated expression to configured bounds (zero-alloc, parallel)
        let t_clip = Instant::now();
        clip_simulated_delta_in_place(gene_mtx, &mut delta_simulated, expression_bounds);
        if let Some(t) = timings.as_mut() {
            t.record(
                format!("iter{}/clip_expression", iter + 1),
                t_clip.elapsed(),
            );
        }
        report_perturb_step(
            job_progress,
            job_message,
            (base + span).saturating_sub(1).min(PROP_HI),
            &format!("{msg_prefix} · clip & sync"),
        );
    }

    report_perturb_step(
        job_progress,
        job_message,
        930,
        "GRN perturbation · assembling result…",
    );

    let out = perturb_result_from_delta(
        gene_mtx,
        delta_simulated,
        targets,
        gene_names,
        Some(expression_bounds),
    );

    report_perturb_step(
        job_progress,
        job_message,
        1000,
        "GRN perturbation · complete",
    );

    Some(out)
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

fn clip_simulated_delta_in_place(
    gene_mtx: &Array2<f64>,
    delta: &mut Array2<f64>,
    bounds: ExpressionBounds,
) {
    let n_genes = gene_mtx.ncols();
    let delta_flat = delta.as_slice_memory_order_mut().unwrap();
    let gmtx_flat = gene_mtx.as_slice().unwrap();
    delta_flat
        .par_chunks_mut(n_genes)
        .enumerate()
        .for_each(|(cell, row)| {
            let base = cell * n_genes;
            for gene in 0..n_genes {
                unsafe {
                    let orig = *gmtx_flat.get_unchecked(base + gene);
                    let val = bounds.clip_value(orig + *row.get_unchecked(gene));
                    *row.get_unchecked_mut(gene) = val - orig;
                }
            }
        });
}

fn clip_simulated_matrix_in_place(simulated: &mut Array2<f64>, bounds: ExpressionBounds) {
    simulated.mapv_inplace(|v| bounds.clip_value(v));
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

/// Arguments for [`compute_splash_all_progress`].
pub struct ComputeSplashAllProgressArgs<'a> {
    pub bb: &'a Betabase,
    pub rw_ligands: &'a GeneMatrix,
    pub rw_tfligands: &'a GeneMatrix,
    pub gex_df: &'a GeneMatrix,
    pub beta_scale_factor: f32,
    pub beta_cap: Option<f32>,
    pub progress: Option<&'a AtomicU32>,
    pub cancel: Option<&'a AtomicBool>,
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
    compute_splash_all_progress(ComputeSplashAllProgressArgs {
        bb,
        rw_ligands,
        rw_tfligands,
        gex_df,
        beta_scale_factor,
        beta_cap,
        progress: None,
        cancel: None,
    })
    .expect("compute_splash_all_progress without cancel must return Some")
}

/// Like [`compute_splash_all`], optionally reporting coarse progress on `progress` (permille 0–1000).
/// Updates are throttled (~≤30 calls) to avoid sync overhead on large target counts.
///
/// Returns `None` when `cancel` is set and becomes true during computation.
pub fn compute_splash_all_progress(
    args: ComputeSplashAllProgressArgs<'_>,
) -> Option<HashMap<String, GeneMatrix>> {
    let ComputeSplashAllProgressArgs {
        bb,
        rw_ligands,
        rw_tfligands,
        gex_df,
        beta_scale_factor,
        beta_cap,
        progress,
        cancel,
    } = args;
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

struct FusedGeneWork<'a> {
    gene_col: usize,
    plan: crate::betadata::SplashPlan,
    splash_n_mods: usize,
    mod_indices: &'a [usize],
    bf: &'a crate::betadata::BetaFrame,
}

/// Same math as [`compute_splash_all`] + [`perturb_all_cells_into`] without Jacobian HashMaps.
/// Returns `false` if `cancel` is set.
#[allow(clippy::too_many_arguments)]
fn perturb_all_cells_fused_into(
    gene_names: &[String],
    bb: &Betabase,
    rw_ligands: &GeneMatrix,
    rw_tfligands: &GeneMatrix,
    expr: &Array2<f64>,
    min_expression: f64,
    beta_scale_factor: f32,
    beta_cap: Option<f32>,
    delta_simulated: &Array2<f64>,
    out_row_major: &mut [f64],
    cancel: Option<&AtomicBool>,
    progress: Option<&AtomicU32>,
) -> bool {
    let n_cells = delta_simulated.nrows();
    let n_genes = gene_names.len();
    assert_eq!(out_row_major.len(), n_cells * n_genes);
    let gex_index: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
    let work: Vec<FusedGeneWork<'_>> = gene_names
        .iter()
        .enumerate()
        .filter_map(|(gene_idx, gene_name)| {
            let bf = bb.data.get(gene_name)?;
            let mod_indices = bf.modulator_gene_indices.as_ref()?;
            if bf.modulator_genes.is_empty() {
                return None;
            }
            let plan = bf.splash_plan(
                rw_ligands,
                rw_tfligands,
                |n| gex_index.get(n).copied(),
                beta_scale_factor,
            );
            Some(FusedGeneWork {
                gene_col: gene_idx,
                splash_n_mods: plan.n_out,
                plan,
                mod_indices: mod_indices.as_slice(),
                bf,
            })
        })
        .collect();
    let max_mods = work.iter().map(|w| w.splash_n_mods).max().unwrap_or(0);
    let rw_flat = rw_ligands.data.as_slice().unwrap();
    let rw_nc = rw_ligands.data.ncols();
    let rw_tfl_flat = rw_tfligands.data.as_slice().unwrap();
    let rw_tfl_nc = rw_tfligands.data.ncols();
    let expr_flat = expr.as_slice().expect("expression matrix row-major");
    let expr_nc = expr.ncols();
    let gex = SplashGex::F64Masked {
        flat: expr_flat,
        ncols: expr_nc,
        min_expression,
    };
    let delta_flat = delta_simulated.as_slice_memory_order().unwrap();
    let cancelled = AtomicBool::new(false);
    let done_cells = AtomicU32::new(0);
    let step = (n_cells / 28).max(1) as u32;
    out_row_major.fill(0.0);
    out_row_major
        .par_chunks_mut(n_genes)
        .enumerate()
        .for_each(|(cell, r)| {
            if cancelled.load(Ordering::Relaxed) {
                return;
            }
            if cell % 256 == 0 && cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            let mut row = vec![0.0f32; max_mods];
            let delta_base = cell * n_genes;
            for w in &work {
                let n_out = w.splash_n_mods;
                let scratch = &mut row[..n_out];
                w.plan.fill_row(
                    w.bf,
                    cell,
                    scratch,
                    rw_flat,
                    rw_nc,
                    rw_tfl_flat,
                    rw_tfl_nc,
                    gex,
                );
                if let Some(cap) = beta_cap {
                    for v in scratch.iter_mut() {
                        *v = v.clamp(-cap, cap);
                    }
                }
                let mut sum = 0.0f64;
                for k in 0..n_out {
                    unsafe {
                        sum += f64::from(*scratch.get_unchecked(k))
                            * *delta_flat
                                .get_unchecked(delta_base + *w.mod_indices.get_unchecked(k));
                    }
                }
                r[w.gene_col] = sum;
            }
            let n = done_cells.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(p) = progress {
                if n % step == 0 || n as usize == n_cells {
                    let v = 50u32 + (n.saturating_mul(700) / n_cells.max(1) as u32);
                    p.store(v.min(750), Ordering::Relaxed);
                }
            }
        });
    !cancelled.load(Ordering::Relaxed)
}

struct RecomputeWeightedLigandsArgs<'a> {
    gene_mtx: &'a Array2<f64>,
    gene_to_idx: &'a HashMap<&'a str, usize>,
    ligand_names: &'a [String],
    xy: &'a Array2<f64>,
    lr_radii: &'a HashMap<String, f64>,
    scale_factor: f64,
    min_expression: f64,
    grid_factor: Option<f64>,
    contact_distance: Option<f64>,
    cancel: Option<&'a AtomicBool>,
}

fn recompute_weighted_ligands(args: RecomputeWeightedLigandsArgs<'_>) -> Option<GeneMatrix> {
    use std::sync::atomic::Ordering;

    let RecomputeWeightedLigandsArgs {
        gene_mtx,
        gene_to_idx,
        ligand_names,
        xy,
        lr_radii,
        scale_factor,
        min_expression,
        grid_factor,
        contact_distance,
        cancel,
    } = args;

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
            Some(gf) if gf.is_finite() && gf > 0.0 => calculate_weighted_ligands_grid_with_cutoff(
                xy,
                &sub,
                radius,
                scale_factor,
                gf,
                contact_distance,
                None,
            ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn clip_simulated_delta_matches_elementwise_reference() {
        let n_cells = 64;
        let n_genes = 40;
        let gene_mtx = Array2::from_shape_fn((n_cells, n_genes), |(c, g)| {
            0.5 + 0.1 * ((c * 7 + g * 13) % 17) as f64
        });
        let mut delta = Array2::from_shape_fn((n_cells, n_genes), |(c, g)| {
            -0.1 + 0.02 * ((c * 3 + g * 11) % 23) as f64
        });
        for &gi in &[0usize, 7, 39] {
            for c in 0..n_cells {
                delta[[c, gi]] = 0.0 - gene_mtx[[c, gi]];
            }
        }
        let bounds = ExpressionBounds {
            min: 0.0,
            max: f64::INFINITY,
        };
        let mut expected = delta.clone();
        for c in 0..n_cells {
            for g in 0..n_genes {
                let val = bounds.clip_value(gene_mtx[[c, g]] + expected[[c, g]]);
                expected[[c, g]] = val - gene_mtx[[c, g]];
            }
        }
        clip_simulated_delta_in_place(&gene_mtx, &mut delta, bounds);
        let max_diff = delta
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "clip vs reference max_diff={max_diff:.2e}"
        );
    }

    #[test]
    fn clip_simulated_delta_respects_bounds() {
        let gene_mtx = array![[1.0, 5.0], [2.0, -1.0]];
        let mut delta = array![[0.0, 6.0], [0.0, 2.0]];
        clip_simulated_delta_in_place(
            &gene_mtx,
            &mut delta,
            ExpressionBounds { min: 0.0, max: 8.0 },
        );
        assert!((delta[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((delta[[0, 1]] - 3.0).abs() < 1e-12);
        assert!((delta[[1, 0]] - 0.0).abs() < 1e-12);
        assert!((delta[[1, 1]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn perturb_result_from_delta_clamps_target_gene() {
        let gene_mtx = array![[2.0, 4.0]];
        let delta = array![[0.0, 0.0]];
        let targets = vec![PerturbTarget {
            gene: "G1".into(),
            desired_expr: 20.0,
            cell_indices: None,
        }];
        let gene_names = vec!["G1".into(), "G2".into()];
        let out = perturb_result_from_delta(
            &gene_mtx,
            delta,
            &targets,
            &gene_names,
            Some(ExpressionBounds {
                min: 0.0,
                max: 10.0,
            }),
        );
        assert!((out.simulated[[0, 0]] - 10.0).abs() < 1e-12);
        assert!((out.simulated[[0, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn expression_bounds_default_matches_nonneg_only() {
        let bounds = ExpressionBounds::from_config(&PerturbConfig::default());
        assert_eq!(bounds.min, 0.0);
        assert!(bounds.max.is_infinite());
    }

    #[test]
    fn estimated_splash_jacobian_bytes_scales_with_cells_and_mods() {
        let mut bf = crate::betadata::BetaFrame::from_parts(crate::betadata::BetaFrameFromParts {
            gene_name: "T".into(),
            row_labels: vec!["0".into()],
            intercepts: array![0.0],
            tf_betas: array![[1.0, 2.0]],
            tfs: vec!["A".into(), "B".into()],
            lr_betas: ndarray::Array2::zeros((1, 0)),
            ligands: vec![],
            receptors: vec![],
            tfl_betas: ndarray::Array2::zeros((1, 0)),
            tfl_ligands: vec![],
            tfl_regulators: vec![],
            cis_betas: ndarray::Array2::zeros((1, 0)),
            cis_left: vec![],
            cis_right: vec![],
        });
        let n_cells = 10usize;
        let obs: Vec<String> = (0..n_cells).map(|i| format!("c{i}")).collect();
        let keys: Vec<String> = vec!["0".to_string(); n_cells];
        let mapping = std::sync::Arc::new(
            crate::betadata::BetaFrame::compute_cell_mapping(&bf.row_labels, &obs, &keys).0,
        );
        bf.expand_to_cells(std::sync::Arc::new(obs), mapping);
        let n_mods = bf.modulator_genes.len();
        let mut data = HashMap::new();
        data.insert("T".to_string(), bf);
        let bb = Betabase {
            data,
            ligands_set: HashSet::new(),
            receptors_set: HashSet::new(),
            tfl_ligands_set: HashSet::new(),
            tfs_set: HashSet::new(),
        };
        let n_genes = 5usize;
        let bytes = estimated_splash_jacobian_bytes(&bb, n_cells, n_genes);
        assert_eq!(bytes, n_cells * n_mods * 4 + n_cells * n_genes * 4);
        let bytes2 = estimated_splash_jacobian_bytes(&bb, n_cells * 2, n_genes);
        assert_eq!(
            bytes2,
            2 * (n_cells * n_mods * 4) + (n_cells * 2) * n_genes * 4
        );
    }

    #[test]
    fn resolve_splash_mode_honors_explicit_fused_and_budget() {
        let mut cfg = PerturbConfig {
            splash_mode: SplashMode::Fused,
            splash_jacobian_max_mb: Some(1),
            ..Default::default()
        };
        assert_eq!(resolve_splash_mode(&cfg, 1), SplashMode::Fused);
        cfg.splash_mode = SplashMode::Materialize;
        assert_eq!(
            resolve_splash_mode(&cfg, usize::MAX),
            SplashMode::Materialize
        );
        cfg.splash_mode = SplashMode::Auto;
        cfg.splash_jacobian_max_mb = Some(1);
        let over = 2 * 1024 * 1024;
        assert_eq!(resolve_splash_mode(&cfg, over), SplashMode::Fused);
        assert_eq!(resolve_splash_mode(&cfg, 16), SplashMode::Materialize);
    }
}
