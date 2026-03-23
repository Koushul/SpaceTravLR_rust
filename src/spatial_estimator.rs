use crate::config::CnnConfig;
use crate::estimator::{ClusteredGCNNWR, finite_or_zero_f64};
use crate::lasso::GroupLassoParams;
use crate::training_hud::{TrainingHud, log_line};
use anndata::data::SelectInfoElem;
use anndata::{AnnData, AnnDataOp, ArrayElemOp, AxisArraysOp, Backend};
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

fn compute_gene_mean_expression<AnB: Backend>(
    adata: &AnnData<AnB>,
    layer: &str,
) -> anyhow::Result<HashMap<String, f64>> {
    let var_names = adata.var_names().into_vec();
    let n_obs = adata.n_obs();
    if n_obs == 0 {
        return Ok(HashMap::new());
    }
    let slice = [SelectInfoElem::full(), SelectInfoElem::full()];
    let data: Array2<f64> = if layer != "X" && !layer.is_empty() {
        if let Some(layer_elem) = adata.layers().get(layer) {
            layer_elem
                .slice(slice)?
                .ok_or_else(|| anyhow::anyhow!("Failed to slice layer {}", layer))?
        } else {
            let x_elem = adata.x();
            if x_elem.is_none() {
                return Err(anyhow::anyhow!(
                    "Layer '{}' not found and X is empty",
                    layer
                ));
            }
            x_elem
                .slice(slice)?
                .ok_or_else(|| anyhow::anyhow!("Failed to slice X"))?
        }
    } else {
        let x_elem = adata.x();
        if x_elem.is_none() {
            return Err(anyhow::anyhow!("X is empty"));
        }
        x_elem
            .slice(slice)?
            .ok_or_else(|| anyhow::anyhow!("Failed to slice X"))?
    };
    let inv_n = 1.0 / n_obs as f64;
    let mut out = HashMap::with_capacity(var_names.len());
    for (j, name) in var_names.iter().enumerate() {
        let sum: f64 = data.column(j).iter().sum();
        out.insert(name.clone(), sum * inv_n);
    }
    Ok(out)
}

pub struct SpatialCellularProgramsEstimator<AB: AutodiffBackend, AnB: Backend> {
    pub adata: Arc<AnnData<AnB>>,
    pub target_gene: String,
    pub spatial_dim: usize,
    pub cluster_annot: String,
    pub layer: String,
    pub radius: f64,
    pub contact_distance: f64,
    pub grn: Arc<crate::network::GeneNetwork>,
    pub tf_ligand_cutoff: f64,
    pub regulators: Vec<String>,
    pub ligands: Vec<String>,
    pub receptors: Vec<String>,
    pub tfl_ligands: Vec<String>,
    pub tfl_regulators: Vec<String>,
    pub lr_pairs: Vec<String>,
    pub tfl_pairs: Vec<String>,
    pub modulators_genes: Vec<String>,
    pub max_lr_pairs: Option<usize>,
    pub seed_only: bool,
    pub estimator: Option<ClusteredGCNNWR<AB>>,
    pub group_reg_vec: Option<Vec<f64>>,
}

impl<AB: AutodiffBackend, AnB: Backend> SpatialCellularProgramsEstimator<AB, AnB> {
    pub fn new_with_metadata(
        adata: Arc<AnnData<AnB>>,
        target_gene: String,
        radius: f64,
        spatial_dim: usize,
        contact_distance: f64,
        tf_ligand_cutoff: f64,
        max_lr_pairs: Option<usize>,
        top_lr_pairs_by_mean_expression: Option<usize>,
        gene_mean_expression: Option<Arc<HashMap<String, f64>>>,
        grn: Arc<crate::network::GeneNetwork>,
        layer: String,
    ) -> anyhow::Result<Self> {
        let target_gene_str = target_gene.to_string();
        let cluster_annot = "cell_type_int".to_string();

        let modulators = grn.get_modulators(
            &target_gene_str,
            tf_ligand_cutoff,
            max_lr_pairs,
            top_lr_pairs_by_mean_expression,
            gene_mean_expression.as_deref(),
        )?;
        let mut modulators_genes_ordered = modulators.regulators.clone();
        modulators_genes_ordered.extend(modulators.lr_pairs.clone());
        modulators_genes_ordered.extend(modulators.tfl_pairs.clone());

        Ok(Self {
            adata,
            target_gene,
            spatial_dim,
            cluster_annot,
            layer,
            radius,
            contact_distance,
            grn,
            tf_ligand_cutoff,
            regulators: modulators.regulators,
            ligands: modulators.ligands,
            receptors: modulators.receptors,
            tfl_ligands: modulators.tfl_ligands,
            tfl_regulators: modulators.tfl_regulators,
            lr_pairs: modulators.lr_pairs,
            tfl_pairs: modulators.tfl_pairs,
            modulators_genes: modulators_genes_ordered,
            max_lr_pairs,
            seed_only: false,
            estimator: None,
            group_reg_vec: None,
        })
    }

    pub fn new(
        adata: Arc<AnnData<AnB>>,
        target_gene: String,
        radius: f64,
        spatial_dim: usize,
        contact_distance: f64,
        tf_ligand_cutoff: f64,
        max_lr_pairs: Option<usize>,
    ) -> anyhow::Result<Self> {
        let adata_var_names = adata.var_names().into_vec();
        let species = crate::network::infer_species(&adata_var_names);
        let grn = Arc::new(crate::network::GeneNetwork::new(species, &adata_var_names)?);
        Self::new_with_metadata(
            adata,
            target_gene,
            radius,
            spatial_dim,
            contact_distance,
            tf_ligand_cutoff,
            max_lr_pairs,
            None,
            None,
            grn,
            "imputed_count".to_string(),
        )
    }
}

impl<AB: AutodiffBackend> SpatialCellularProgramsEstimator<AB, anndata_hdf5::H5> {
    pub fn fit_all_genes(
        adata_path: &str,
        radius: f64,
        spatial_dim: usize,
        contact_distance: f64,
        tf_ligand_cutoff: f64,
        max_lr_pairs: Option<usize>,
        top_lr_pairs_by_mean_expression: Option<usize>,
        layer: &str,
        cnn: &CnnConfig,
        epochs: usize,
        learning_rate: f64,
        score_threshold: f64,
        l1_reg: f64,
        group_reg: f64,
        n_iter: usize,
        tol: f64,
        full_cnn: bool,
        gene_filter: Option<Vec<String>>,
        max_genes: Option<usize>,
        n_parallel: usize,
        output_dir: &str,
        hud: Option<TrainingHud>,
        device: &AB::Device,
    ) -> anyhow::Result<()>
    where
        AB: Send + 'static,
        AB::Device: Clone + Send + 'static,
    {
        use anndata_hdf5::H5;
        use std::fs;
        use std::io::Write;
        use std::thread;

        let training_dir = output_dir;
        fs::create_dir_all(training_dir)?;
        fs::create_dir_all(format!("{training_dir}/log"))?;

        // ── Setup: build gene list and pre-cache shared metadata ──────────────
        let setup_adata = Arc::new(AnnData::<H5>::open(H5::open(adata_path)?)?);
        let all_var_names = setup_adata.var_names().into_vec();

        let mut target_genes = all_var_names.clone();
        if let Some(filter) = gene_filter {
            let msg = format!("Filtering for specific genes: {:?}", filter);
            log_line(&hud, msg.clone());
            if hud.is_none() {
                println!("{}", msg);
            }
            target_genes.retain(|g| filter.contains(g));
            let msg = format!("Retained {} genes for training", target_genes.len());
            log_line(&hud, msg.clone());
            if hud.is_none() {
                println!("{}", msg);
            }
        }
        if let Some(n) = max_genes {
            if target_genes.len() > n {
                target_genes.truncate(n);
                let preview: Vec<_> = target_genes.iter().take(5).cloned().collect();
                let msg = format!("Using first {} genes (preview: {:?})", n, preview);
                log_line(&hud, msg.clone());
                if hud.is_none() {
                    println!("{}", msg);
                }
            }
        }

        let obs_names = Arc::new(setup_adata.obs_names().into_vec());
        let species = crate::network::infer_species(&all_var_names);
        let global_grn = Arc::new(crate::network::GeneNetwork::new(species, &all_var_names)?);

        let total_genes = target_genes.len();

        let msg = "Pre-caching metadata and coordinates...";
        log_line(&hud, msg.to_string());
        if hud.is_none() {
            println!("{}", msg);
        }

        let cluster_annot = "cell_type_int".to_string();
        let obs_df = setup_adata.read_obs()?;
        let xy: Arc<Array2<f64>> = Arc::new(
            setup_adata
                .obsm()
                .get_item("spatial")?
                .ok_or_else(|| anyhow::anyhow!("obsm['spatial'] not found"))?,
        );
        let clusters_ser = obs_df.column(&cluster_annot)?;
        let clusters: Arc<Array1<usize>> = Arc::new(
            clusters_ser
                .as_materialized_series()
                .cast(&polars::prelude::DataType::Float64)?
                .f64()?
                .to_ndarray()?
                .mapv(|v| v as usize),
        );
        let num_clusters = clusters
            .iter()
            .copied()
            .max()
            .map(|m| m.saturating_add(1))
            .unwrap_or(1);

        let gene_mean_arc: Option<Arc<HashMap<String, f64>>> =
            if top_lr_pairs_by_mean_expression.is_some() {
                let msg = format!(
                    "Computing per-gene mean expression (layer: {}) for LR ranking...",
                    layer
                );
                log_line(&hud, msg.clone());
                if hud.is_none() {
                    println!("{}", msg);
                }
                Some(Arc::new(compute_gene_mean_expression(
                    setup_adata.as_ref(),
                    layer,
                )?))
            } else {
                None
            };

        let layer_for_workers = layer.to_string();
        let cnn_for_workers = cnn.clone();

        drop(setup_adata); // release; workers open their own handles

        if let Some(ref h) = hud {
            if let Ok(mut g) = h.lock() {
                g.total_genes = total_genes;
                g.n_cells = obs_names.len();
                g.n_clusters = num_clusters;
                g.init_cluster_perf_buckets(num_clusters);
            }
        }

        let pb_opt: Option<ProgressBar> = if hud.is_none() {
            let pb = ProgressBar::new(total_genes as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        // ── Shared work queue ─────────────────────────────────────────────────
        let work: Arc<Mutex<VecDeque<String>>> =
            Arc::new(Mutex::new(target_genes.into_iter().collect()));

        let n_workers = n_parallel.max(1);
        let mut handles: Vec<thread::JoinHandle<()>> = Vec::with_capacity(n_workers);

        for _worker in 0..n_workers {
            let work = work.clone();
            let xy = xy.clone();
            let clusters = clusters.clone();
            let obs_names = obs_names.clone();
            let global_grn = global_grn.clone();
            let hud = hud.clone();
            let pb = pb_opt.clone();
            let device = device.clone();
            let adata_path = adata_path.to_string();
            let training_dir = training_dir.to_string();

            // scalar params
            let (radius, spatial_dim, contact_distance, tf_ligand_cutoff) =
                (radius, spatial_dim, contact_distance, tf_ligand_cutoff);
            let max_lr_pairs = max_lr_pairs;
            let top_lr_pairs_by_mean_expression = top_lr_pairs_by_mean_expression;
            let gene_mean_arc = gene_mean_arc.clone();
            let layer_w = layer_for_workers.clone();
            let cnn_w = cnn_for_workers.clone();
            let (epochs, learning_rate, score_threshold, l1_reg, group_reg, n_iter, tol) = (
                epochs,
                learning_rate,
                score_threshold,
                l1_reg,
                group_reg,
                n_iter,
                tol,
            );
            let full_cnn = full_cnn;
            let num_clusters = num_clusters;

            let handle = thread::Builder::new()
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    let thread_adata = match H5::open(&adata_path)
                        .and_then(|f| AnnData::<H5>::open(f))
                    {
                        Ok(a) => Arc::new(a),
                        Err(e) => {
                            log_line(&hud, format!("ERROR: worker failed to open adata: {}", e));
                            return;
                        }
                    };

                    loop {
                        // Cancel check
                        if hud
                            .as_ref()
                            .and_then(|h| h.lock().ok())
                            .map(|g| g.should_cancel())
                            .unwrap_or(false)
                        {
                            log_line(&hud, ">> worker: cancel signal received".to_string());
                            break;
                        }

                        let gene = match work.lock() {
                            Ok(mut q) => q.pop_front(),
                            Err(_) => break,
                        };
                        let Some(gene) = gene else { break };

                        let csv_path = format!("{}/{}_betadata.csv", training_dir, gene);
                        let orphan_path = format!("{}/{}.orphan", training_dir, gene);
                        let lock_path = format!("{}/{}.lock", training_dir, gene);

                        // Skip already-done
                        if std::path::Path::new(&csv_path).exists()
                            || std::path::Path::new(&orphan_path).exists()
                        {
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.genes_skipped += 1;
                                }
                                log_line(&hud, format!(">> skip (cached) {}", gene));
                            }
                            if let Some(ref p) = pb {
                                p.inc(1);
                            }
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.genes_rounds += 1;
                                }
                            }
                            continue;
                        }

                        // Try to claim this gene via a lock file
                        if fs::OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(&lock_path)
                            .is_err()
                        {
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.genes_skipped += 1;
                                }
                                log_line(&hud, format!(">> skip (lock) {}", gene));
                            }
                            if let Some(ref p) = pb {
                                p.inc(1);
                            }
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.genes_rounds += 1;
                                }
                            }
                            continue;
                        }
                        struct LockGuard(String);
                        impl Drop for LockGuard {
                            fn drop(&mut self) {
                                let _ = fs::remove_file(&self.0);
                            }
                        }
                        let _guard = LockGuard(lock_path);

                        // Register as active
                        if let Some(ref h) = hud {
                            if let Ok(mut g) = h.lock() {
                                g.set_gene_status(&gene, "estimator | ? mods");
                            }
                        }

                        let mut estimator = match Self::new_with_metadata(
                            thread_adata.clone(),
                            gene.clone(),
                            radius,
                            spatial_dim,
                            contact_distance,
                            tf_ligand_cutoff,
                            max_lr_pairs,
                            top_lr_pairs_by_mean_expression,
                            gene_mean_arc.clone(),
                            global_grn.clone(),
                            layer_w.clone(),
                        )
                        .map(Box::new)
                        {
                            Ok(est) => est,
                            Err(e) => {
                                log_line(&hud, format!("❌ estimator init failed {}: {}", gene, e));
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_failed += 1;
                                        g.remove_gene(&gene);
                                    }
                                }
                                if let Some(ref p) = pb {
                                    p.inc(1);
                                }
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_rounds += 1;
                                    }
                                }
                                continue;
                            }
                        };

                        let n_mods = estimator.modulators_genes.len();
                        if let Some(ref h) = hud {
                            if let Ok(mut g) = h.lock() {
                                g.set_gene_status(&gene, format!("estimator | {} mods", n_mods));
                            }
                        }

                        if n_mods == 0 {
                            let _ = fs::File::create(format!("{}/{}.orphan", training_dir, gene));
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.genes_orphan += 1;
                                    g.remove_gene(&gene);
                                    g.genes_rounds += 1;
                                }
                            }
                            log_line(&hud, format!(">> orphan (no modulators) {}", gene));
                            if let Some(ref p) = pb {
                                p.inc(1);
                            }
                            continue;
                        }

                        estimator.seed_only = !full_cnn;
                        let phase_str = if full_cnn {
                            format!("lasso+cnn | {} mods", n_mods)
                        } else {
                            format!("lasso | {} mods", n_mods)
                        };
                        if let Some(ref h) = hud {
                            if let Ok(mut g) = h.lock() {
                                g.set_gene_status(&gene, &phase_str);
                            }
                        }

                        let fit_ok = estimator
                            .fit_with_cache(
                                &xy,
                                &clusters,
                                num_clusters,
                                epochs,
                                learning_rate,
                                score_threshold,
                                l1_reg,
                                group_reg,
                                n_iter,
                                tol,
                                "lasso",
                                &cnn_w,
                                &device,
                            )
                            .is_ok();

                        let mut wrote = false;
                        let mut orphan_zero_mod_betas = false;
                        if fit_ok {
                            if let Some(est_inner) = &estimator.estimator {
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.set_gene_status(
                                            &gene,
                                            format!("export | {} mods", n_mods),
                                        );
                                    }
                                }
                                let betadata_path =
                                    format!("{}/{}_betadata.csv", training_dir, gene);
                                let col_names: Vec<String> = std::iter::once("beta0".to_string())
                                    .chain(
                                        estimator
                                            .modulators_genes
                                            .iter()
                                            .map(|m| format!("beta_{}", m)),
                                    )
                                    .collect();

                                if full_cnn {
                                    let x_mock = Array2::<f64>::zeros((xy.nrows(), n_mods));
                                    let all_betas = est_inner.predict_betas(
                                        &x_mock,
                                        &xy,
                                        &clusters,
                                        num_clusters,
                                        &device,
                                    );

                                    let keep: Vec<usize> = (0..all_betas.ncols())
                                        .filter(|&j| {
                                            all_betas
                                                .column(j)
                                                .iter()
                                                .any(|&v| finite_or_zero_f64(v) != 0.0)
                                        })
                                        .collect();

                                    if !keep.iter().any(|&j| j >= 1) {
                                        let _ = fs::File::create(format!(
                                            "{}/{}.orphan",
                                            training_dir, gene
                                        ));
                                        orphan_zero_mod_betas = true;
                                        if let Some(ref h) = hud {
                                            if let Ok(mut g) = h.lock() {
                                                g.genes_orphan += 1;
                                            }
                                        }
                                        log_line(
                                            &hud,
                                            format!(
                                                ">> orphan (no non-zero modulator betas) {}",
                                                gene
                                            ),
                                        );
                                    } else if let Ok(mut f) = fs::File::create(&betadata_path) {
                                        let mut header = "CellID".to_string();
                                        for &j in &keep {
                                            header.push_str(&format!(",{}", col_names[j]));
                                        }
                                        let _ = writeln!(f, "{}", header);

                                        for (i, cell_id) in obs_names.iter().enumerate() {
                                            let mut row = format!("{}", cell_id);
                                            for &j in &keep {
                                                row.push_str(&format!(
                                                    ",{}",
                                                    finite_or_zero_f64(all_betas[[i, j]])
                                                ));
                                            }
                                            let _ = writeln!(f, "{}", row);
                                        }
                                        wrote = true;
                                    }
                                } else {
                                    let mut cluster_ids: Vec<usize> =
                                        est_inner.lasso_coefficients.keys().copied().collect();
                                    cluster_ids.sort();

                                    let rows: Vec<Vec<f64>> = cluster_ids
                                        .iter()
                                        .map(|&c_id| {
                                            let intercept = finite_or_zero_f64(
                                                est_inner
                                                    .lasso_intercepts
                                                    .get(&c_id)
                                                    .copied()
                                                    .unwrap_or(0.0),
                                            );
                                            let coefs = &est_inner.lasso_coefficients[&c_id];
                                            std::iter::once(intercept)
                                                .chain(
                                                    coefs
                                                        .column(0)
                                                        .iter()
                                                        .map(|&b| finite_or_zero_f64(b)),
                                                )
                                                .collect()
                                        })
                                        .collect();

                                    let n_cols = 1 + n_mods;
                                    let keep: Vec<usize> = (0..n_cols)
                                        .filter(|&j| rows.iter().any(|r| r[j] != 0.0))
                                        .collect();

                                    if !keep.iter().any(|&j| j >= 1) {
                                        let _ = fs::File::create(format!(
                                            "{}/{}.orphan",
                                            training_dir, gene
                                        ));
                                        orphan_zero_mod_betas = true;
                                        if let Some(ref h) = hud {
                                            if let Ok(mut g) = h.lock() {
                                                g.genes_orphan += 1;
                                            }
                                        }
                                        log_line(
                                            &hud,
                                            format!(
                                                ">> orphan (no non-zero modulator betas) {}",
                                                gene
                                            ),
                                        );
                                    } else if let Ok(mut f) = fs::File::create(&betadata_path) {
                                        let mut header = "Cluster".to_string();
                                        for &j in &keep {
                                            header.push_str(&format!(",{}", col_names[j]));
                                        }
                                        let _ = writeln!(f, "{}", header);

                                        for (row_vals, &c_id) in rows.iter().zip(cluster_ids.iter())
                                        {
                                            let mut row = format!("{}", c_id);
                                            for &j in &keep {
                                                row.push_str(&format!(",{}", row_vals[j]));
                                            }
                                            let _ = writeln!(f, "{}", row);
                                        }
                                        wrote = true;
                                    }
                                }
                            }
                        }

                        if wrote {
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.genes_done += 1;
                                }
                                log_line(&hud, format!(">> wrote {}", gene));
                            }
                        } else if !orphan_zero_mod_betas {
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.genes_failed += 1;
                                }
                                log_line(&hud, format!(">> fail (fit/export) {}", gene));
                            }
                        }

                        if n_mods > 0 {
                            if let Some(est) = estimator.estimator.as_ref() {
                                let safe_gene = gene.replace(['/', '\\'], "_");
                                let log_path = format!("{}/log/{}.log", training_dir, safe_gene);
                                let _ = crate::training_log::write_gene_training_log(
                                    std::path::Path::new(&log_path),
                                    &gene,
                                    !full_cnn,
                                    epochs,
                                    learning_rate,
                                    n_iter,
                                    tol,
                                    &est.cluster_training_summaries,
                                );
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        if !est.cluster_training_summaries.is_empty() {
                                            g.record_training_metrics(
                                                &gene,
                                                &est.cluster_training_summaries,
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Deregister from active, bump counter
                        if let Some(ref h) = hud {
                            if let Ok(mut g) = h.lock() {
                                g.remove_gene(&gene);
                                g.genes_rounds += 1;
                            }
                        }
                        if let Some(ref p) = pb {
                            p.inc(1);
                        }
                    }
                })
                .expect("failed to spawn worker thread");

            handles.push(handle);
        }

        for h in handles {
            let _ = h.join();
        }

        if let Some(ref h) = hud {
            if let Ok(mut g) = h.lock() {
                g.finished = Some(Ok(()));
            }
        }

        Ok(())
    }
}

impl<AB: AutodiffBackend, AnB: Backend> SpatialCellularProgramsEstimator<AB, AnB> {
    pub fn fit_with_cache(
        &mut self,
        xy: &Array2<f64>,
        clusters: &Array1<usize>,
        num_clusters: usize,
        epochs: usize,
        learning_rate: f64,
        _score_threshold: f64,
        l1_reg: f64,
        group_reg: f64,
        n_iter: usize,
        tol: f64,
        _estimator_type: &str,
        cnn: &CnnConfig,
        device: &AB::Device,
    ) -> anyhow::Result<()> {
        let target_expr = self.get_gene_expression(&self.target_gene)?;

        let mut all_unique_genes: HashSet<String> = HashSet::new();
        for g in &self.regulators {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.ligands {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.receptors {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.tfl_ligands {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.tfl_regulators {
            all_unique_genes.insert(g.clone());
        }

        let unique_genes_vec: Vec<String> = all_unique_genes.into_iter().collect::<Vec<_>>();
        let expr_matrix = self.get_multiple_gene_expressions(&unique_genes_vec)?;

        let mut gene_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, g) in unique_genes_vec.iter().enumerate() {
            gene_to_idx.insert(g.clone(), i);
        }

        let n_obs = self.adata.n_obs();
        let total_modulators = self.regulators.len() + self.lr_pairs.len() + self.tfl_pairs.len();
        let mut x_modulators = Array2::<f64>::zeros((n_obs, total_modulators));

        for (i, gene) in self.regulators.iter().enumerate() {
            let idx = gene_to_idx[gene];
            x_modulators.column_mut(i).assign(&expr_matrix.column(idx));
        }

        let offset_lr = self.regulators.len();
        for (i, pair) in self.lr_pairs.iter().enumerate() {
            let parts: Vec<&str> = pair.split('$').collect::<Vec<_>>();
            if parts.len() == 2 {
                let l_idx = gene_to_idx[&parts[0].to_string()];
                let r_idx = gene_to_idx[&parts[1].to_string()];
                let mut interaction = expr_matrix.column(l_idx).to_owned();
                interaction *= &expr_matrix.column(r_idx);
                x_modulators.column_mut(offset_lr + i).assign(&interaction);
            }
        }

        let offset_tfl = offset_lr + self.lr_pairs.len();
        for (i, pair) in self.tfl_pairs.iter().enumerate() {
            let parts: Vec<&str> = pair.split('#').collect::<Vec<_>>();
            if parts.len() == 2 {
                let l_idx = gene_to_idx[&parts[0].to_string()];
                let tf_idx = gene_to_idx[&parts[1].to_string()];
                let mut interaction = expr_matrix.column(l_idx).to_owned();
                interaction *= &expr_matrix.column(tf_idx);
                x_modulators.column_mut(offset_tfl + i).assign(&interaction);
            }
        }

        if self.estimator.is_none() {
            let mut groups = Vec::new();
            for _ in 0..self.regulators.len() {
                groups.push(0);
            }
            for _ in 0..self.lr_pairs.len() {
                groups.push(1);
            }
            for _ in 0..self.tfl_pairs.len() {
                groups.push(2);
            }

            let params = GroupLassoParams {
                l1_reg,
                group_reg,
                groups,
                n_iter,
                tol,
                ..Default::default()
            };
            let mut est =
                ClusteredGCNNWR::new(params, self.spatial_dim, cnn.spatial_feature_radius);
            est.group_reg_vec = self.group_reg_vec.clone();
            self.estimator = Some(est);
        }

        if let Some(est) = &mut self.estimator {
            est.fit(
                &x_modulators,
                &target_expr,
                xy,
                clusters,
                num_clusters,
                device,
                epochs,
                learning_rate,
                self.seed_only,
                cnn,
            );
        }
        Ok(())
    }

    pub fn fit(
        &mut self,
        epochs: usize,
        learning_rate: f64,
        score_threshold: f64,
        l1_reg: f64,
        group_reg: f64,
        n_iter: usize,
        tol: f64,
        estimator_type: &str,
        device: &AB::Device,
    ) -> anyhow::Result<()> {
        let obs_df = self.adata.read_obs()?;
        let xy: Array2<f64> = self
            .adata
            .obsm()
            .get_item("spatial")?
            .ok_or_else(|| anyhow::anyhow!("obsm['spatial'] not found"))?;
        let clusters_ser = obs_df.column(&self.cluster_annot)?;
        let clusters: Array1<usize> = clusters_ser
            .as_materialized_series()
            .cast(&polars::prelude::DataType::Float64)?
            .f64()?
            .to_ndarray()?
            .mapv(|v| v as usize);
        let num_clusters = clusters
            .iter()
            .copied()
            .max()
            .map(|m| m.saturating_add(1))
            .unwrap_or(1);

        let cnn = CnnConfig::default();
        self.fit_with_cache(
            &xy,
            &clusters,
            num_clusters,
            epochs,
            learning_rate,
            score_threshold,
            l1_reg,
            group_reg,
            n_iter,
            tol,
            estimator_type,
            &cnn,
            device,
        )
    }

    pub fn get_multiple_gene_expressions(&self, genes: &[String]) -> anyhow::Result<Array2<f64>> {
        let mut gene_indices: Vec<usize> = Vec::new();
        for gene in genes {
            let idx = self
                .adata
                .var_names()
                .get_index(gene)
                .ok_or_else(|| anyhow::anyhow!("Gene {} not found in var_names", gene))?;
            gene_indices.push(idx);
        }

        let slice = [SelectInfoElem::full(), SelectInfoElem::Index(gene_indices)];

        if self.layer != "X" && !self.layer.is_empty() {
            if let Some(layer_elem) = self.adata.layers().get(&self.layer) {
                let data: Array2<f64> = layer_elem
                    .slice(slice)?
                    .ok_or_else(|| anyhow::anyhow!("Failed to slice layer {}", self.layer))?;
                return Ok(data);
            }
        }

        let x_elem = self.adata.x();
        if x_elem.is_none() {
            return Err(anyhow::anyhow!(
                "X is empty and layer {} not found",
                self.layer
            ));
        }
        let data: Array2<f64> = x_elem
            .slice(slice)?
            .ok_or_else(|| anyhow::anyhow!("Failed to slice X"))?;
        Ok(data)
    }

    pub fn get_gene_expression(&self, gene: &str) -> anyhow::Result<Array1<f64>> {
        self.get_multiple_gene_expressions(&[gene.to_string()])
            .map(|data: Array2<f64>| data.column(0).to_owned())
    }
}
