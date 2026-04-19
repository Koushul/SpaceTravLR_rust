use crate::config::{CnnTrainingMode, SpaceshipConfig};
use crate::estimator::{ClusterTrainingSummary, CnnEpochHudSlot};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct RunConfigSummary {
    pub config_source: String,
    pub compute_backend: String,
    pub compute_device_detail: String,
    pub compute_notice: String,
    pub layer: String,
    pub cluster_annot: String,
    pub spatial_radius: f64,
    pub spatial_dim: usize,
    pub contact_distance: f64,
    pub weighted_ligand_scale_factor: f64,
    pub tf_ligand_cutoff: f64,
    pub max_ligands: String,
    pub l1_reg: f64,
    pub group_reg: f64,
    pub n_iter: usize,
    pub tol: f64,
    pub learning_rate: f64,
    pub score_threshold: f64,
    pub epochs_per_gene: usize,
    pub gene_selection: String,
    pub cnn_training_mode: String,
    pub condition_split: String,
}

pub struct RunConfigSummaryBuildArgs<'a> {
    pub config_path: Option<&'a Path>,
    pub compute_backend: &'a str,
    pub compute_device_detail: &'a str,
    pub compute_notice: &'a str,
    pub cfg: &'a SpaceshipConfig,
    pub max_genes: Option<usize>,
    pub gene_filter: Option<&'a [String]>,
    pub condition_split: Option<&'a str>,
}

impl RunConfigSummary {
    pub fn build(args: RunConfigSummaryBuildArgs<'_>) -> Self {
        let RunConfigSummaryBuildArgs {
            config_path,
            compute_backend,
            compute_device_detail,
            compute_notice,
            cfg,
            max_genes,
            gene_filter,
            condition_split,
        } = args;
        let config_source = config_path
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| {
                "spaceship_config.toml (repo / install data/ base + optional --config overlay)"
                    .to_string()
            });

        let max_ligands = cfg
            .grn
            .max_ligands
            .map(|n| n.to_string())
            .unwrap_or_else(|| "—".to_string());

        let cnn_training_mode = match cfg.resolved_cnn_mode() {
            CnnTrainingMode::Seed => "seed",
            CnnTrainingMode::Full => "full",
            CnnTrainingMode::Hybrid => "hybrid",
        }
        .to_string();

        let gene_selection = match (gene_filter, max_genes) {
            (Some(genes), _) if !genes.is_empty() => {
                let take = 4usize.min(genes.len());
                let head: Vec<_> = genes.iter().take(take).cloned().collect();
                let mut s = head.join(", ");
                if genes.len() > take {
                    s.push_str(&format!(" (+{} more)", genes.len() - take));
                }
                s
            }
            (None, Some(n)) => format!("first {} genes (var order)", n),
            _ => "all genes (var order)".to_string(),
        };

        Self {
            config_source,
            compute_backend: compute_backend.to_string(),
            compute_device_detail: compute_device_detail.to_string(),
            compute_notice: compute_notice.to_string(),
            layer: cfg.data.layer.clone(),
            cluster_annot: cfg.data.cluster_annot.clone(),
            spatial_radius: cfg.spatial.radius,
            spatial_dim: cfg.spatial.spatial_dim,
            contact_distance: cfg.spatial.contact_distance,
            weighted_ligand_scale_factor: cfg.spatial.weighted_ligand_scale_factor,
            tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
            max_ligands,
            l1_reg: cfg.lasso.l1_reg,
            group_reg: cfg.lasso.group_reg,
            n_iter: cfg.lasso.n_iter,
            tol: cfg.lasso.tol,
            learning_rate: cfg.training.learning_rate,
            score_threshold: cfg.training.score_threshold,
            epochs_per_gene: cfg.training.epochs,
            gene_selection,
            cnn_training_mode,
            condition_split: condition_split.unwrap_or("—").to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingHudState {
    pub dataset_path: String,
    pub output_dir: String,
    pub run_config: RunConfigSummary,
    pub full_cnn: bool,
    pub genes_exported_seed_only: usize,
    pub genes_exported_cnn: usize,
    pub epochs_per_gene: usize,
    pub n_parallel: usize,
    pub total_genes: usize,
    pub genes_done: usize,
    pub genes_skipped: usize,
    pub genes_failed: usize,
    pub genes_orphan: usize,
    /// TF modulators disabled (`[grn].use_tf_modulators = false`) but target had TF-only GRN/prior
    /// support — written as `GENE.tf_ablated`, not counted as GRN orphan.
    pub genes_tf_ablated: usize,
    pub genes_rounds: usize,
    pub active_genes: HashMap<String, String>,
    /// Per-gene progress: parallel Lasso clusters done / total, then sequential CNN clusters done / total (TUI only).
    pub gene_lasso_cluster_progress: HashMap<String, (usize, usize)>,
    pub gene_cnn_epoch_slots: HashMap<String, Arc<CnnEpochHudSlot>>,
    pub n_cells: usize,
    pub n_clusters: usize,
    pub cell_type_counts: Vec<(String, usize)>,
    pub started: Instant,
    pub finished: Option<Result<(), String>>,
    pub cancel_requested: Arc<AtomicBool>,
    /// Per completed gene for the TUI best / worst list: `(gene, mean_lasso_r2, mean_cnn_r2, n_modulators)`.
    /// `mean_cnn_r2` is `Some` when the spatial CNN ran for that gene (mean over clusters with finite `cnn_r2`).
    /// Fourth field: count of `beta_*` columns in written betadata (non-zero across rows/cells);
    /// falls back to design-matrix width when not supplied (e.g. training demo).
    pub gene_r2_mean: Vec<(String, f64, Option<f64>, usize)>,
    pub perf_stats_generation: u64,
    pub gene_train_times: VecDeque<(String, f64)>,
    /// Human-readable obs value for the subset currently training (`--condition` mode).
    pub current_condition_value: Option<String>,
    /// `(1-based index, total splits)` for the active subset.
    pub condition_split_progress: Option<(usize, usize)>,
    /// `spacetravlr --demo`: synthetic run; TUI uses `genes_rounds` for progress (no output dir scan).
    pub is_demo: bool,
    /// CellOracle GRN inference: total Bayesian-ridge target fits (0 = not in CellOracle phase).
    pub celloracle_infer_total: usize,
    /// Completed target fits during CellOracle; updated from Rayon without holding the HUD mutex.
    pub celloracle_infer_done: Arc<AtomicUsize>,
}

impl TrainingHudState {
    pub fn new(
        dataset_path: String,
        output_dir: String,
        run_config: RunConfigSummary,
        full_cnn: bool,
        epochs_per_gene: usize,
        n_parallel: usize,
        cancel_requested: Arc<AtomicBool>,
    ) -> Self {
        Self {
            dataset_path,
            output_dir,
            run_config,
            full_cnn,
            genes_exported_seed_only: 0,
            genes_exported_cnn: 0,
            epochs_per_gene,
            n_parallel,
            total_genes: 0,
            genes_done: 0,
            genes_skipped: 0,
            genes_failed: 0,
            genes_orphan: 0,
            genes_tf_ablated: 0,
            genes_rounds: 0,
            active_genes: HashMap::new(),
            gene_lasso_cluster_progress: HashMap::new(),
            gene_cnn_epoch_slots: HashMap::new(),
            n_cells: 0,
            n_clusters: 0,
            cell_type_counts: Vec::new(),
            started: Instant::now(),
            finished: None,
            cancel_requested,
            gene_r2_mean: Vec::new(),
            perf_stats_generation: 0,
            gene_train_times: VecDeque::new(),
            current_condition_value: None,
            condition_split_progress: None,
            is_demo: false,
            celloracle_infer_total: 0,
            celloracle_infer_done: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn reset_for_new_split(
        &mut self,
        dataset_path: String,
        output_dir: String,
        condition_split: Option<(String, usize, usize)>,
    ) {
        self.dataset_path = dataset_path;
        self.output_dir = output_dir;
        match condition_split {
            Some((label, idx, total)) => {
                self.current_condition_value = Some(label);
                self.condition_split_progress = Some((idx, total));
            }
            None => {
                self.current_condition_value = None;
                self.condition_split_progress = None;
            }
        }
        self.genes_exported_seed_only = 0;
        self.genes_exported_cnn = 0;
        self.total_genes = 0;
        self.genes_done = 0;
        self.genes_skipped = 0;
        self.genes_failed = 0;
        self.genes_orphan = 0;
        self.genes_tf_ablated = 0;
        self.genes_rounds = 0;
        self.active_genes.clear();
        self.gene_lasso_cluster_progress.clear();
        self.gene_cnn_epoch_slots.clear();
        self.n_cells = 0;
        self.n_clusters = 0;
        self.cell_type_counts.clear();
        self.started = Instant::now();
        self.finished = None;
        self.gene_r2_mean.clear();
        self.perf_stats_generation = self.perf_stats_generation.wrapping_add(1);
        self.gene_train_times.clear();
        self.is_demo = false;
        self.celloracle_infer_total = 0;
        self.celloracle_infer_done.store(0, Ordering::Relaxed);
    }

    pub fn record_gene_time(&mut self, gene: &str, secs: f64) {
        const MAX: usize = 64;
        while self.gene_train_times.len() >= MAX {
            self.gene_train_times.pop_front();
        }
        self.gene_train_times.push_back((gene.to_string(), secs));
    }

    pub fn record_training_metrics(
        &mut self,
        gene: &str,
        summaries: &[ClusterTrainingSummary],
        n_betadata_beta_columns: Option<usize>,
    ) {
        if summaries.is_empty() {
            return;
        }
        let mean_lasso: f64 =
            summaries.iter().map(|s| s.lasso_r2).sum::<f64>() / summaries.len() as f64;
        let mut cnn_sum = 0.0_f64;
        let mut n_cnn = 0usize;
        for s in summaries {
            if s.cnn_r2.is_finite() {
                cnn_sum += s.cnn_r2;
                n_cnn += 1;
            }
        }
        let mean_cnn = (n_cnn > 0).then_some(cnn_sum / n_cnn as f64);
        let n_modulators = n_betadata_beta_columns
            .unwrap_or_else(|| summaries.iter().map(|s| s.n_modulators).max().unwrap_or(0));
        self.gene_r2_mean
            .push((gene.to_string(), mean_lasso, mean_cnn, n_modulators));
        self.perf_stats_generation = self.perf_stats_generation.wrapping_add(1);
    }

    pub fn record_gene_export_mode(&mut self, per_cell_cnn: bool) {
        if per_cell_cnn {
            self.genes_exported_cnn = self.genes_exported_cnn.saturating_add(1);
        } else {
            self.genes_exported_seed_only = self.genes_exported_seed_only.saturating_add(1);
        }
    }

    pub fn set_gene_status(&mut self, gene: &str, status: impl std::fmt::Display) {
        self.active_genes
            .insert(gene.to_string(), status.to_string());
    }

    pub fn ensure_gene_cnn_epoch_slot(
        &mut self,
        gene: &str,
        total_epochs: usize,
    ) -> Arc<CnnEpochHudSlot> {
        use std::collections::hash_map::Entry;
        let key = gene.to_string();
        match self.gene_cnn_epoch_slots.entry(key) {
            Entry::Occupied(e) => {
                let s = e.get().clone();
                s.reconfigure(total_epochs);
                s
            }
            Entry::Vacant(v) => {
                let s = CnnEpochHudSlot::new(total_epochs);
                v.insert(s.clone());
                s
            }
        }
    }

    pub fn clear_gene_cnn_epoch_slot(&mut self, gene: &str) {
        self.gene_cnn_epoch_slots.remove(gene);
    }

    pub fn set_gene_lasso_cluster_progress(&mut self, gene: &str, done: usize, total: usize) {
        if total == 0 {
            self.gene_lasso_cluster_progress.remove(gene);
            return;
        }
        match self.gene_lasso_cluster_progress.get_mut(gene) {
            Some(v) if v.0 == done && v.1 == total => {}
            Some(v) => *v = (done, total),
            None => {
                self.gene_lasso_cluster_progress
                    .insert(gene.to_string(), (done, total));
            }
        }
    }

    pub fn clear_gene_lasso_cluster_progress(&mut self, gene: &str) {
        self.gene_lasso_cluster_progress.remove(gene);
    }

    pub fn remove_gene(&mut self, gene: &str) {
        self.active_genes.remove(gene);
        self.gene_lasso_cluster_progress.remove(gene);
        self.gene_cnn_epoch_slots.remove(gene);
    }

    pub fn should_cancel(&self) -> bool {
        self.cancel_requested.load(Ordering::Relaxed)
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn mean_completed_gene_secs(&self) -> Option<f64> {
        let n = self.gene_train_times.len();
        if n == 0 {
            return None;
        }
        let sum: f64 = self.gene_train_times.iter().map(|(_, t)| *t).sum();
        let m = sum / n as f64;
        if m.is_finite() && m > 0.0 {
            Some(m)
        } else {
            None
        }
    }

    pub fn parallel_rate_genes_per_sec(&self) -> Option<f64> {
        let elapsed = self.elapsed_secs().max(0.001);
        if self.genes_rounds > 0 {
            let observed = self.genes_rounds as f64 / elapsed;
            if observed.is_finite() && observed > f64::EPSILON {
                return Some(observed);
            }
        }
        if let Some(single_gene_secs) = self.mean_completed_gene_secs() {
            let workers = self.n_parallel.max(1) as f64;
            let estimated = workers / single_gene_secs;
            if estimated.is_finite() && estimated > f64::EPSILON {
                return Some(estimated);
            }
        }
        None
    }

    pub fn parallel_wall_secs_per_gene(&self) -> Option<f64> {
        self.parallel_rate_genes_per_sec()
            .map(|rate| 1.0 / rate)
            .filter(|secs| secs.is_finite() && *secs > 0.0)
    }

    pub fn eta_secs(&self) -> Option<f64> {
        if self.total_genes == 0 {
            return None;
        }
        let remaining = self.total_genes.saturating_sub(self.genes_rounds);
        if remaining == 0 {
            return Some(0.0);
        }
        if let Some(rate) = self.parallel_rate_genes_per_sec() {
            let eta = remaining as f64 / rate;
            if eta.is_finite() && eta >= 0.0 {
                return Some(eta);
            }
        }
        None
    }
}

pub type TrainingHud = Arc<Mutex<TrainingHudState>>;

/// After a run with the dashboard, explain when nothing wrote betadata (TUI hides per-gene `println!`).
pub fn print_training_outcome_banner(hud: &Option<TrainingHud>) {
    let Some(h) = hud else {
        return;
    };
    let Ok(g) = h.lock() else {
        return;
    };
    if g.total_genes == 0 {
        return;
    }
    let exported = g.genes_exported_seed_only + g.genes_exported_cnn;
    if exported > 0 {
        return;
    }
    if g.genes_rounds < g.total_genes {
        return;
    }
    if g.genes_failed == 0
        && g.genes_orphan == 0
        && g.genes_tf_ablated == 0
        && g.genes_skipped >= g.total_genes
    {
        eprintln!(
            "\nNote: no new *_betadata.feather files were written — every gene was skipped (outputs already exist or another process holds a .lock)."
        );
        return;
    }
    eprintln!("\n=== No betadata Feather files were written this run ===");
    eprintln!("Genes queued: {}", g.total_genes);
    eprintln!("  skipped (existing CSV / lock): {}", g.genes_skipped);
    eprintln!(
        "  failed (init or fit — check {}/log/ for details): {}",
        g.output_dir, g.genes_failed
    );
    eprintln!(
        "  orphan (no modulators in GRN for that target): {}",
        g.genes_orphan
    );
    eprintln!(
        "  tf_ablated (TF modulators off; TF-only target per GRN/priors): {}",
        g.genes_tf_ablated
    );
    eprintln!(
        "Typical fixes: set [data].layer and [data].cluster_annot to match the .h5ad; ensure obsm has spatial / X_spatial / spatial_loc (≥2 cols); verify species/GRN covers your gene symbols; relax --genes filter."
    );
}

pub fn log_line(hud: &Option<TrainingHud>, msg: String) {
    if hud.is_none() {
        println!("{}", msg);
    }
}

pub fn pipeline_step_begin(_hud: &Option<TrainingHud>, _label: &str) -> Instant {
    Instant::now()
}

pub fn pipeline_step_end(hud: &Option<TrainingHud>, label: &str, started: Instant) {
    if hud.is_none() {
        let s = started.elapsed().as_secs_f64();
        println!("+ {:.1}s  {}", s, label);
    }
}
