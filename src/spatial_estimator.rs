use crate::lasso::GroupLassoParams;
use std::collections::{HashSet, HashMap};
use std::sync::Arc;
use anndata::data::SelectInfoElem;
use anndata::{AnnData, AnnDataOp, ArrayElemOp, AxisArraysOp, Backend};
use burn::tensor::backend::AutodiffBackend;
use ndarray::{Array1, Array2};
use indicatif::{ProgressBar, ProgressStyle};
use crate::estimator::{finite_or_zero_f64, ClusteredGCNNWR};

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
        grn: Arc<crate::network::GeneNetwork>,
    ) -> anyhow::Result<Self> {
        let target_gene_str = target_gene.to_string();
        let cluster_annot = "cell_type_int".to_string();

        let modulators = grn.get_modulators(&target_gene_str, tf_ligand_cutoff, max_lr_pairs)?;
        let mut modulators_genes_ordered = modulators.regulators.clone();
        modulators_genes_ordered.extend(modulators.lr_pairs.clone());
        modulators_genes_ordered.extend(modulators.tfl_pairs.clone());

        Ok(Self {
            adata,
            target_gene,
            spatial_dim,
            cluster_annot,
            layer: "imputed_count".to_string(),
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
        Self::new_with_metadata(adata, target_gene, radius, spatial_dim, contact_distance, tf_ligand_cutoff, max_lr_pairs, grn)
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
        device: &AB::Device,
    ) -> anyhow::Result<()> {
        use anndata_hdf5::H5;
        use std::fs;
        use std::io::Write;

        let training_dir = "/tmp/training";
        fs::create_dir_all(training_dir)?;

        let adata = Arc::new(AnnData::<H5>::open(H5::open(adata_path)?)?);
        let all_var_names = adata.var_names().into_vec();
        
        let mut target_genes = all_var_names.clone();
        if let Some(filter) = gene_filter {
            println!("🔍 Filtering for specific genes: {:?}", filter);
            target_genes.retain(|g| filter.contains(g));
            println!("✅ Retained {} genes for training: {:?}", target_genes.len(), target_genes);
        }

        if let Some(n) = max_genes {
            if target_genes.len() > n {
                target_genes.truncate(n);
                let preview: Vec<_> = target_genes.iter().take(5).cloned().collect();
                println!(
                    "✂️ Using first {} genes in AnnData var order (preview): {:?}",
                    n, preview
                );
            }
        }

        let obs_names = adata.obs_names().into_vec();
        let species = crate::network::infer_species(&all_var_names);
        // CRITICAL: Initialize GRN with ALL var_names so modulators can be found
        let global_grn = Arc::new(crate::network::GeneNetwork::new(species, &all_var_names)?);
        
        let total_genes = target_genes.len();

        println!("📂 Pre-caching metadata and coordinates...");
        let cluster_annot = "cell_type_int".to_string();
        let obs_df = adata.read_obs()?;

        let xy: Array2<f64> = adata.obsm().get_item("spatial")?
            .ok_or_else(|| anyhow::anyhow!("obsm['spatial'] not found"))?;

        let clusters_ser = obs_df.column(&cluster_annot)?;
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

        let pb = ProgressBar::new(total_genes as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
            .progress_chars("#>-"));

        target_genes.into_iter().for_each(|gene| {
            let csv_path = format!("{}/{}_betadata.csv", training_dir, gene);
            let orphan_path = format!("{}/{}.orphan", training_dir, gene);
            let lock_path = format!("{}/{}.lock", training_dir, gene);

            if std::path::Path::new(&csv_path).exists() || std::path::Path::new(&orphan_path).exists() {
                pb.inc(1);
                return;
            }

            if fs::OpenOptions::new().write(true).create_new(true).open(&lock_path).is_err() {
                pb.inc(1);
                return;
            }

            struct LockGuard(String);
            impl Drop for LockGuard {
                fn drop(&mut self) {
                    let _ = std::fs::remove_file(&self.0);
                }
            }
            let _guard = LockGuard(lock_path.clone());

            let mut estimator = match Self::new_with_metadata(
                adata.clone(),
                gene.clone(),
                radius,
                spatial_dim,
                contact_distance,
                tf_ligand_cutoff,
                max_lr_pairs,
                global_grn.clone(),
            ).map(Box::new) {
                Ok(est) => est,
                Err(e) => {
                    eprintln!("❌ Failed to create estimator for {}: {:?}", gene, e);
                    pb.inc(1);
                    return;
                }
            };

            let n_mods = estimator.modulators_genes.len();
            if n_mods == 0 {
                // Check if it's truly an orphan or just failed modulators
                if estimator.regulators.is_empty() {
                    let orphan_path = format!("{}/{}.orphan", training_dir, gene);
                    let _ = fs::File::create(orphan_path);
                }
                pb.inc(1);
                return;
            }

            estimator.seed_only = !full_cnn;
            
            if let Ok(_) = estimator.fit_with_cache(
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
                device
            ) {
                if let Some(est_inner) = &estimator.estimator {
                    let betadata_path = format!("{}/{}_betadata.csv", training_dir, gene);
                    if let Ok(mut f) = fs::File::create(betadata_path) {
                        if full_cnn {
                            let x_mock = Array2::<f64>::zeros((xy.nrows(), estimator.modulators_genes.len()));
                            let all_betas = est_inner.predict_betas(&x_mock, &xy, &clusters, num_clusters, device);
                            
                            let mut header = "CellID,beta0".to_string();
                            for m in &estimator.modulators_genes {
                                header.push_str(&format!(",{}", m));
                            }
                            let _ = writeln!(f, "{}", header);
                            
                            for (i, cell_id) in obs_names.iter().enumerate() {
                                let mut row = format!("{}", cell_id);
                                for j in 0..all_betas.ncols() {
                                    row.push_str(&format!(
                                        ",{}",
                                        finite_or_zero_f64(all_betas[[i, j]])
                                    ));
                                }
                                let _ = writeln!(f, "{}", row);
                            }
                        } else {
                            // Seed-Only Format: Cluster, beta0, Mod1, Mod2, ...
                            let mut header = "Cluster,beta0".to_string();
                            for m in &estimator.modulators_genes {
                                header.push_str(&format!(",{}", m));
                            }
                            let _ = writeln!(f, "{}", header);
                            
                            let mut cluster_ids: Vec<usize> = est_inner.lasso_coefficients.keys().copied().collect::<Vec<_>>();
                            cluster_ids.sort();
                            
                            for c_id in cluster_ids {
                                let coefs = &est_inner.lasso_coefficients[&c_id];
                                let intercept = finite_or_zero_f64(
                                    est_inner.lasso_intercepts.get(&c_id).copied().unwrap_or(0.0),
                                );

                                let mut row = format!("{},{}", c_id, intercept);
                                for &beta in coefs.column(0).iter() {
                                    row.push_str(&format!(",{}", finite_or_zero_f64(beta)));
                                }
                                let _ = writeln!(f, "{}", row);
                            }
                        }
                    }
                }
            }
            pb.inc(1);
        });

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
        _learning_rate: f64,
        _score_threshold: f64,
        l1_reg: f64,
        group_reg: f64,
        n_iter: usize,
        tol: f64,
        _estimator_type: &str,
        device: &AB::Device,
    ) -> anyhow::Result<()> {
        let target_expr = self.get_gene_expression(&self.target_gene)?;

        let mut all_unique_genes: HashSet<String> = HashSet::new();
        for g in &self.regulators { all_unique_genes.insert(g.clone()); }
        for g in &self.ligands { all_unique_genes.insert(g.clone()); }
        for g in &self.receptors { all_unique_genes.insert(g.clone()); }
        for g in &self.tfl_ligands { all_unique_genes.insert(g.clone()); }
        for g in &self.tfl_regulators { all_unique_genes.insert(g.clone()); }
        
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
            for _ in 0..self.regulators.len() { groups.push(0); }
            for _ in 0..self.lr_pairs.len() { groups.push(1); }
            for _ in 0..self.tfl_pairs.len() { groups.push(2); }

            let params = GroupLassoParams {
                l1_reg,
                group_reg,
                groups,
                n_iter,
                tol,
                ..Default::default()
            };
            let mut est = ClusteredGCNNWR::new(params, self.spatial_dim);
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
                self.seed_only,
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
        let xy: Array2<f64> = self.adata.obsm().get_item("spatial")?
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

        self.fit_with_cache(&xy, &clusters, num_clusters, epochs, learning_rate, score_threshold, l1_reg, group_reg, n_iter, tol, estimator_type, device)
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

        let slice = [
            SelectInfoElem::full(),
            SelectInfoElem::Index(gene_indices),
        ];

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
            return Err(anyhow::anyhow!("X is empty and layer {} not found", self.layer));
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
