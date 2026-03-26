use crate::estimator::ClusterTrainingSummary;
use crate::training_hud::TrainingHud;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const DEMO_LATENCY_SCALE: u64 = 25;

fn demo_delay_ms(base_plus_jitter: u64) -> Duration {
    Duration::from_millis(base_plus_jitter.saturating_mul(DEMO_LATENCY_SCALE))
}

const DEMO_GENES: &[&str] = &[
    "CD74", "MALAT1", "PAX5", "PTPRC", "TMSB4X", "CD3D", "MS4A1", "CD19", "FCER1G", "GNLY",
    "CD14", "LYZ", "EPCAM", "KRT8", "ACTA2", "COL1A1", "PECAM1", "VWF", "CD8A", "FOXP3", "IL7R",
    "CCR7", "IFNG", "TNF", "CD4", "CD68", "DCN", "POSTN", "MKI67", "TOP2A",
];

fn demo_gene_names(total: usize, filter: Option<&[String]>) -> Vec<String> {
    let mut base: Vec<String> = if let Some(f) = filter {
        if f.is_empty() {
            DEMO_GENES.iter().map(|s| (*s).to_string()).collect()
        } else {
            let set: std::collections::HashSet<_> = f.iter().cloned().collect();
            let v: Vec<String> = DEMO_GENES
                .iter()
                .filter(|g| set.iter().any(|s| s == *g))
                .map(|s| (*s).to_string())
                .collect();
            if v.is_empty() {
                f.to_vec()
            } else {
                v
            }
        }
    } else {
        DEMO_GENES.iter().map(|s| (*s).to_string()).collect()
    };

    if base.is_empty() {
        base = (0..total.max(1))
            .map(|i| format!("DEMO_GENE_{i}"))
            .collect();
    }

    let mut out = Vec::with_capacity(total);
    let mut i = 0usize;
    while out.len() < total {
        out.push(base[i % base.len()].clone());
        i += 1;
    }
    out
}

fn gene_hash(gene: &str) -> u32 {
    gene.bytes().fold(5381u32, |h, b| h.wrapping_mul(33).wrapping_add(b as u32))
}

fn fake_summaries(gene: &str, n_clusters: usize, full_cnn: bool) -> Vec<ClusterTrainingSummary> {
    let h = gene_hash(gene);
    let base = 0.35 + ((h % 1000) as f64 / 1000.0) * 0.55;
    let k = n_clusters.max(3).min(12);
    let mut v = Vec::with_capacity(k);
    for i in 0..k {
        let jitter = (i as f64 * 0.03 - (k as f64 / 2.0) * 0.03) * 0.5;
        let r2 = (base + jitter).clamp(0.08, 0.995);
        let epochs = if full_cnn { 8usize } else { 0usize };
        let cnn_train_mse_epochs = if full_cnn {
            (0..epochs)
                .map(|e| 0.5 - (e as f32) * 0.04 + ((h.wrapping_add(e as u32) % 7) as f32) * 0.01)
                .collect()
        } else {
            Vec::new()
        };
        v.push(ClusterTrainingSummary {
            cluster_id: i,
            n_cells: 320 + (h as usize % 200) + i * 41,
            n_modulators: 24 + (h as usize % 80) + i * 2,
            lasso_r2: r2,
            lasso_train_mse: 0.02 + (i as f64) * 0.001,
            lasso_fista_iters: 30 + (h as usize % 40) + i * 3,
            lasso_converged: true,
            cnn_train_mse_epochs,
        });
    }
    v
}

enum DemoOutcome {
    Success,
    Skip,
    Orphan,
    Fail,
}

fn classify_demo_outcome(gene: &str, idx: usize) -> DemoOutcome {
    let h = gene_hash(gene).wrapping_add(idx as u32);
    if h % 19 == 0 {
        DemoOutcome::Skip
    } else if h % 23 == 0 {
        DemoOutcome::Orphan
    } else if h % 47 == 0 {
        DemoOutcome::Fail
    } else {
        DemoOutcome::Success
    }
}

fn demo_worker(
    work: Arc<Mutex<VecDeque<(usize, String)>>>,
    hud: TrainingHud,
    run_full_cnn: bool,
    epochs_per_gene: usize,
) {
    loop {
        if hud.lock().map(|g| g.should_cancel()).unwrap_or(false) {
            return;
        }
        let job = {
            let mut q = work.lock().unwrap_or_else(|e| e.into_inner());
            q.pop_front()
        };
        let Some((idx, gene)) = job else {
            break;
        };
        let job_start = Instant::now();

        {
            let st = hud.lock().unwrap_or_else(|e| e.into_inner());
            if st.should_cancel() {
                return;
            }
        }

        let outcome = classify_demo_outcome(&gene, idx);
        if matches!(outcome, DemoOutcome::Skip) {
            thread::sleep(demo_delay_ms(50));
            if let Ok(mut g) = hud.lock() {
                g.record_gene_time(&gene, job_start.elapsed().as_secs_f64());
                g.genes_skipped = g.genes_skipped.saturating_add(1);
                g.genes_rounds = g.genes_rounds.saturating_add(1);
            }
            continue;
        }

        {
            let mut g = hud.lock().unwrap_or_else(|e| e.into_inner());
            g.set_gene_status(&gene, "estimator | ? mods".to_string());
        }
        thread::sleep(demo_delay_ms(120 + (gene_hash(&gene) % 140) as u64));

        if hud.lock().map(|g| g.should_cancel()).unwrap_or(true) {
            let _ = hud.lock().map(|mut g| g.remove_gene(&gene));
            return;
        }

        let n_mods = 12 + (gene_hash(&gene) % 140) as usize;
        let lasso_base_ms = 180 + (gene_hash(&gene) % 160) as u64;
        let n_ct = hud.lock().map(|g| g.n_clusters.max(1)).unwrap_or(8);
        {
            let mut g = hud.lock().unwrap_or_else(|e| e.into_inner());
            g.set_gene_status(&gene, format!("lasso | {n_mods} mods"));
            g.set_gene_lasso_cluster_progress(&gene, 0, n_ct);
        }
        let per_step_ms = (lasso_base_ms / n_ct as u64).max(5);
        for d in 1..=n_ct {
            thread::sleep(demo_delay_ms(per_step_ms));
            if hud.lock().map(|g| g.should_cancel()).unwrap_or(true) {
                let _ = hud.lock().map(|mut g| g.remove_gene(&gene));
                return;
            }
            if let Ok(mut g) = hud.lock() {
                g.set_gene_lasso_cluster_progress(&gene, d, n_ct);
            }
        }

        if matches!(outcome, DemoOutcome::Orphan) {
            if let Ok(mut g) = hud.lock() {
                g.record_gene_time(&gene, job_start.elapsed().as_secs_f64());
                g.genes_orphan = g.genes_orphan.saturating_add(1);
                g.remove_gene(&gene);
                g.genes_rounds = g.genes_rounds.saturating_add(1);
            }
            continue;
        }

        if matches!(outcome, DemoOutcome::Fail) {
            thread::sleep(demo_delay_ms(55));
            if let Ok(mut g) = hud.lock() {
                g.record_gene_time(&gene, job_start.elapsed().as_secs_f64());
                g.genes_failed = g.genes_failed.saturating_add(1);
                g.remove_gene(&gene);
                g.genes_rounds = g.genes_rounds.saturating_add(1);
            }
            continue;
        }

        if run_full_cnn {
            {
                let mut g = hud.lock().unwrap_or_else(|e| e.into_inner());
                g.set_gene_status(&gene, format!("lasso+cnn | {n_mods} mods"));
            }
            let ep = epochs_per_gene.max(1).min(32);
            for e in 1..=ep {
                if hud.lock().map(|g| g.should_cancel()).unwrap_or(true) {
                    let _ = hud.lock().map(|mut g| g.remove_gene(&gene));
                    return;
                }
                if let Ok(mut g) = hud.lock() {
                    g.set_gene_status(&gene, format!("CNN epoch {e}/{ep} | {n_mods} mods"));
                }
                thread::sleep(demo_delay_ms(
                    55 + (gene_hash(&gene).wrapping_add(e as u32) % 65) as u64,
                ));
            }
        } else if hud
            .lock()
            .map(|g| g.run_config.cnn_training_mode == "hybrid")
            .unwrap_or(false)
        {
            {
                let mut g = hud.lock().unwrap_or_else(|e| e.into_inner());
                g.set_gene_status(&gene, format!("hybrid gate | {n_mods} mods"));
            }
            thread::sleep(demo_delay_ms(140 + (gene_hash(&gene) % 120) as u64));
        }

        thread::sleep(demo_delay_ms(100 + (gene_hash(&gene) % 150) as u64));

        let n_clusters = hud
            .lock()
            .map(|g| g.n_clusters.max(3))
            .unwrap_or(8);
        let summaries = fake_summaries(&gene, n_clusters, run_full_cnn);
        if let Ok(mut g) = hud.lock() {
            g.record_gene_time(&gene, job_start.elapsed().as_secs_f64());
            g.genes_done = g.genes_done.saturating_add(1);
            g.genes_rounds = g.genes_rounds.saturating_add(1);
            g.record_gene_export_mode(run_full_cnn);
            g.record_training_metrics(&gene, &summaries);
            g.remove_gene(&gene);
        }
    }
}

pub fn run_demo_training(
    hud: TrainingHud,
    total_genes: usize,
    gene_filter: Option<Vec<String>>,
) -> anyhow::Result<()> {
    let total_genes = total_genes.max(1).min(512);
    let names = demo_gene_names(total_genes, gene_filter.as_deref());

    let (n_parallel, run_full_cnn, epochs_per_gene) = {
        let g = hud.lock().map_err(|e| anyhow::anyhow!("HUD lock poisoned: {}", e))?;
        (
            g.n_parallel.max(1).min(32),
            g.full_cnn,
            g.epochs_per_gene.max(1),
        )
    };

    {
        let mut g = hud.lock().map_err(|e| anyhow::anyhow!("HUD lock poisoned: {}", e))?;
        g.total_genes = total_genes;
        g.n_cells = 18_432;
        g.n_clusters = 14;
        g.cell_type_counts = vec![
            ("T cells".to_string(), 4_820usize),
            ("B cells".to_string(), 3_450usize),
            ("Monocytes".to_string(), 2_610usize),
            ("DCs".to_string(), 1_920usize),
            ("NK cells".to_string(), 1_280usize),
            ("Plasma cells".to_string(), 840usize),
            ("Epithelial cells".to_string(), 1_620usize),
            ("Stromal cells".to_string(), 1_892usize),
        ];
        g.genes_done = 0;
        g.genes_skipped = 0;
        g.genes_failed = 0;
        g.genes_orphan = 0;
        g.genes_rounds = 0;
        g.genes_exported_seed_only = 0;
        g.genes_exported_cnn = 0;
        g.active_genes.clear();
        g.gene_lasso_cluster_progress.clear();
        g.gene_train_times.clear();
        g.gene_r2_mean.clear();
        g.perf_stats_generation = 0;
        g.finished = None;
        g.started = std::time::Instant::now();
    }

    let queue: VecDeque<(usize, String)> = names.into_iter().enumerate().collect();
    let work = Arc::new(Mutex::new(queue));

    let mut handles = Vec::new();
    for _ in 0..n_parallel {
        let work = work.clone();
        let hud = hud.clone();
        let h = thread::spawn(move || demo_worker(work, hud, run_full_cnn, epochs_per_gene));
        handles.push(h);
    }

    for h in handles {
        h.join().map_err(|_| anyhow::anyhow!("demo worker thread panicked"))?;
    }

    let mut g = hud.lock().map_err(|e| anyhow::anyhow!("HUD lock poisoned: {}", e))?;
    if g.finished.is_none() {
        g.finished = Some(Ok(()));
    }
    Ok(())
}
