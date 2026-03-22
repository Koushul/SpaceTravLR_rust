use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TrainingHudState {
    pub dataset_path: String,
    pub output_dir: String,
    pub full_cnn: bool,
    pub epochs_per_gene: usize,
    pub n_parallel: usize,
    pub total_genes: usize,
    pub genes_done: usize,
    pub genes_skipped: usize,
    pub genes_failed: usize,
    pub genes_orphan: usize,
    pub genes_rounds: usize,
    pub active_genes: HashMap<String, String>,
    pub n_cells: usize,
    pub n_clusters: usize,
    pub started: Instant,
    pub finished: Option<Result<(), String>>,
    pub cancel_requested: Arc<AtomicBool>,
}

impl TrainingHudState {
    pub fn new(
        dataset_path: String,
        output_dir: String,
        full_cnn: bool,
        epochs_per_gene: usize,
        n_parallel: usize,
        cancel_requested: Arc<AtomicBool>,
    ) -> Self {
        Self {
            dataset_path,
            output_dir,
            full_cnn,
            epochs_per_gene,
            n_parallel,
            total_genes: 0,
            genes_done: 0,
            genes_skipped: 0,
            genes_failed: 0,
            genes_orphan: 0,
            genes_rounds: 0,
            active_genes: HashMap::new(),
            n_cells: 0,
            n_clusters: 0,
            started: Instant::now(),
            finished: None,
            cancel_requested,
        }
    }

    pub fn set_gene_status(&mut self, gene: &str, status: impl std::fmt::Display) {
        self.active_genes.insert(gene.to_string(), status.to_string());
    }

    pub fn remove_gene(&mut self, gene: &str) {
        self.active_genes.remove(gene);
    }

    pub fn should_cancel(&self) -> bool {
        self.cancel_requested.load(Ordering::Relaxed)
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn eta_secs(&self) -> Option<f64> {
        if self.total_genes == 0 {
            return None;
        }
        let remaining = self.total_genes.saturating_sub(self.genes_rounds);
        let rate = (self.genes_rounds as f64) / self.elapsed_secs().max(0.001);
        if rate <= f64::EPSILON || remaining == 0 {
            return None;
        }
        Some(remaining as f64 / rate)
    }
}

pub type TrainingHud = Arc<Mutex<TrainingHudState>>;

pub fn log_line(hud: &Option<TrainingHud>, msg: String) {
    if hud.is_none() {
        println!("{}", msg);
    }
    // In TUI mode the event log panel is not shown; messages go to stdout only in plain mode.
}
