use crate::config::{CnnTrainingMode, SpaceshipConfig};
use crate::estimator::ClusterTrainingSummary;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct RunConfigSummary {
    pub config_source: String,
    pub compute_backend: String,
    pub compute_notice: String,
    pub layer: String,
    pub cluster_annot: String,
    pub spatial_radius: f64,
    pub spatial_dim: usize,
    pub contact_distance: f64,
    pub tf_ligand_cutoff: f64,
    pub max_lr_pairs: String,
    pub top_lr_pairs: String,
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

impl RunConfigSummary {
    pub fn build(
        config_path: Option<&Path>,
        compute_backend: &str,
        compute_notice: &str,
        cfg: &SpaceshipConfig,
        max_genes: Option<usize>,
        gene_filter: Option<&[String]>,
        condition_split: Option<&str>,
    ) -> Self {
        let config_source = config_path
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "spaceship_config.toml (search path)".to_string());

        let max_lr_pairs = cfg
            .grn
            .max_lr_pairs
            .map(|n| n.to_string())
            .unwrap_or_else(|| "—".to_string());
        let top_lr_pairs = cfg
            .grn
            .top_lr_pairs_by_mean_expression
            .map(|n| format!("{}", n))
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
            compute_notice: compute_notice.to_string(),
            layer: cfg.data.layer.clone(),
            cluster_annot: cfg.data.cluster_annot.clone(),
            spatial_radius: cfg.spatial.radius,
            spatial_dim: cfg.spatial.spatial_dim,
            contact_distance: cfg.spatial.contact_distance,
            tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
            max_lr_pairs,
            top_lr_pairs,
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
    pub genes_rounds: usize,
    pub active_genes: HashMap<String, String>,
    /// Per-gene LASSO progress: clusters (cell types) completed / total, for TUI only.
    pub gene_lasso_cluster_progress: HashMap<String, (usize, usize)>,
    pub n_cells: usize,
    pub n_clusters: usize,
    pub cell_type_counts: Vec<(String, usize)>,
    pub started: Instant,
    pub finished: Option<Result<(), String>>,
    pub cancel_requested: Arc<AtomicBool>,
    /// Mean LASSO R² per completed gene (for TUI best / worst list only).
    pub gene_r2_mean: Vec<(String, f64)>,
    pub perf_stats_generation: u64,
    pub gene_train_times: VecDeque<(String, f64)>,
    /// Human-readable obs value for the subset currently training (`--condition` mode).
    pub current_condition_value: Option<String>,
    /// `(1-based index, total splits)` for the active subset.
    pub condition_split_progress: Option<(usize, usize)>,
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
            genes_rounds: 0,
            active_genes: HashMap::new(),
            gene_lasso_cluster_progress: HashMap::new(),
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
        self.genes_rounds = 0;
        self.active_genes.clear();
        self.gene_lasso_cluster_progress.clear();
        self.n_cells = 0;
        self.n_clusters = 0;
        self.cell_type_counts.clear();
        self.started = Instant::now();
        self.finished = None;
        self.gene_r2_mean.clear();
        self.perf_stats_generation = self.perf_stats_generation.wrapping_add(1);
        self.gene_train_times.clear();
    }

    pub fn record_gene_time(&mut self, gene: &str, secs: f64) {
        const MAX: usize = 64;
        while self.gene_train_times.len() >= MAX {
            self.gene_train_times.pop_front();
        }
        self.gene_train_times.push_back((gene.to_string(), secs));
    }

    pub fn record_training_metrics(&mut self, gene: &str, summaries: &[ClusterTrainingSummary]) {
        if summaries.is_empty() {
            return;
        }
        let mean: f64 = summaries.iter().map(|s| s.lasso_r2).sum::<f64>() / summaries.len() as f64;
        self.gene_r2_mean.push((gene.to_string(), mean));
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
    if g.genes_failed == 0 && g.genes_orphan == 0 && g.genes_skipped >= g.total_genes {
        eprintln!(
            "\nNote: no new *_betadata.feather files were written — every gene was skipped (outputs already exist or another process holds a .lock)."
        );
        return;
    }
    eprintln!("\n=== No betadata Feather files were written this run ===");
    eprintln!("Genes queued: {}", g.total_genes);
    eprintln!(
        "  skipped (existing CSV / lock): {}",
        g.genes_skipped
    );
    eprintln!(
        "  failed (init or fit — check {}/log/ for details): {}",
        g.output_dir, g.genes_failed
    );
    eprintln!(
        "  orphan (no modulators in GRN for that target): {}",
        g.genes_orphan
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

pub fn pipeline_step_begin(hud: &Option<TrainingHud>, label: &str) -> Instant {
    if hud.is_none() {
        println!("[pipeline] … {}", label);
    }
    Instant::now()
}

pub fn pipeline_step_end(hud: &Option<TrainingHud>, label: &str, started: Instant) {
    if hud.is_none() {
        println!(
            "[pipeline] done {} ({:.1}s)",
            label,
            started.elapsed().as_secs_f64()
        );
    }
}
