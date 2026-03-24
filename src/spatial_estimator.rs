use crate::cnn_gating::{
    build_neighbors, decide_cnn_for_gene, load_gene_set_file, predict_lasso_y, CnnGateDecision,
};
use crate::config::{CnnConfig, CnnTrainingMode, HybridCnnGatingConfig, ModelExportConfig};
use crate::estimator::{ClusteredGCNNWR, finite_or_zero_f64};
use crate::lasso::GroupLassoParams;
use crate::training_hud::{TrainingHud, log_line};
use anndata::data::SelectInfoElem;
use anndata::{AnnData, AnnDataOp, ArrayElemOp, AxisArraysOp, Backend};
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2, Array4};
use ndarray_npy::NpzWriter;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::Path;
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

fn sanitize_filename(name: &str) -> String {
    name.replace(['/', '\\', ':', ' '], "_")
}

fn export_cnn_models_npz<AB: AutodiffBackend>(
    est: &ClusteredGCNNWR<AB>,
    gene: &str,
    training_dir: &str,
    model_export: &ModelExportConfig,
) -> anyhow::Result<Option<String>> {
    if est.models.is_empty() || !model_export.save_cnn_weights {
        return Ok(None);
    }
    let model_dir = format!("{}/saved_models", training_dir);
    std::fs::create_dir_all(&model_dir)?;
    let out_path = format!("{}/{}_cnn_weights.npz", model_dir, sanitize_filename(gene));
    let f = File::create(&out_path)?;
    let mut npz = if model_export.compressed_npz {
        NpzWriter::new_compressed(f)
    } else {
        NpzWriter::new(f)
    };

    let mut cluster_ids: Vec<usize> = est.models.keys().copied().collect();
    cluster_ids.sort_unstable();
    npz.add_array("cluster_ids", &Array1::from_vec(cluster_ids.iter().map(|&x| x as u32).collect()))?;

    for c in cluster_ids {
        let m = est
            .models
            .get(&c)
            .ok_or_else(|| anyhow::anyhow!("missing model for cluster {}", c))?;
        let p = format!("c{:04}_", c);

        let conv1_w = m.conv_layers.conv1.weight.to_data();
        let conv1_shape = m.conv_layers.conv1.weight.shape().dims::<4>();
        let conv1_arr = Array4::from_shape_vec(
            (
                conv1_shape[0],
                conv1_shape[1],
                conv1_shape[2],
                conv1_shape[3],
            ),
            conv1_w.as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec(),
        )?;
        npz.add_array(format!("{}conv1_weight", p), &conv1_arr)?;
        if let Some(b) = &m.conv_layers.conv1.bias {
            npz.add_array(
                format!("{}conv1_bias", p),
                &Array1::from_vec(b.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
            )?;
        }
        let conv2_w = m.conv_layers.conv2.weight.to_data();
        let conv2_shape = m.conv_layers.conv2.weight.shape().dims::<4>();
        let conv2_arr = Array4::from_shape_vec(
            (
                conv2_shape[0],
                conv2_shape[1],
                conv2_shape[2],
                conv2_shape[3],
            ),
            conv2_w.as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec(),
        )?;
        npz.add_array(format!("{}conv2_weight", p), &conv2_arr)?;
        if let Some(b) = &m.conv_layers.conv2.bias {
            npz.add_array(
                format!("{}conv2_bias", p),
                &Array1::from_vec(b.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
            )?;
        }
        let conv3_w = m.conv_layers.conv3.weight.to_data();
        let conv3_shape = m.conv_layers.conv3.weight.shape().dims::<4>();
        let conv3_arr = Array4::from_shape_vec(
            (
                conv3_shape[0],
                conv3_shape[1],
                conv3_shape[2],
                conv3_shape[3],
            ),
            conv3_w.as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec(),
        )?;
        npz.add_array(format!("{}conv3_weight", p), &conv3_arr)?;
        if let Some(b) = &m.conv_layers.conv3.bias {
            npz.add_array(
                format!("{}conv3_bias", p),
                &Array1::from_vec(b.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
            )?;
        }

        npz.add_array(
            format!("{}bn1_gamma", p),
            &Array1::from_vec(m.conv_layers.bn1.gamma.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
        )?;
        npz.add_array(
            format!("{}bn1_beta", p),
            &Array1::from_vec(m.conv_layers.bn1.beta.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
        )?;
        npz.add_array(
            format!("{}bn1_running_mean", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn1
                    .running_mean
                    .value()
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn1_running_var", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn1
                    .running_var
                    .value()
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}bn2_gamma", p),
            &Array1::from_vec(m.conv_layers.bn2.gamma.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
        )?;
        npz.add_array(
            format!("{}bn2_beta", p),
            &Array1::from_vec(m.conv_layers.bn2.beta.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
        )?;
        npz.add_array(
            format!("{}bn2_running_mean", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn2
                    .running_mean
                    .value()
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn2_running_var", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn2
                    .running_var
                    .value()
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}bn3_gamma", p),
            &Array1::from_vec(m.conv_layers.bn3.gamma.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
        )?;
        npz.add_array(
            format!("{}bn3_beta", p),
            &Array1::from_vec(m.conv_layers.bn3.beta.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
        )?;
        npz.add_array(
            format!("{}bn3_running_mean", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn3
                    .running_mean
                    .value()
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn3_running_var", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn3
                    .running_var
                    .value()
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}spatial_l1_weight", p),
            &Array2::from_shape_vec(
                (
                    m.spatial_features_mlp.l1.weight.shape().dims::<2>()[0],
                    m.spatial_features_mlp.l1.weight.shape().dims::<2>()[1],
                ),
                m.spatial_features_mlp
                    .l1
                    .weight
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}spatial_l1_bias", p),
            &Array1::from_vec(
                m.spatial_features_mlp
                    .l1
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing spatial_l1 bias"))?
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}spatial_l2_weight", p),
            &Array2::from_shape_vec(
                (
                    m.spatial_features_mlp.l2.weight.shape().dims::<2>()[0],
                    m.spatial_features_mlp.l2.weight.shape().dims::<2>()[1],
                ),
                m.spatial_features_mlp
                    .l2
                    .weight
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}spatial_l2_bias", p),
            &Array1::from_vec(
                m.spatial_features_mlp
                    .l2
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing spatial_l2 bias"))?
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}spatial_l3_weight", p),
            &Array2::from_shape_vec(
                (
                    m.spatial_features_mlp.l3.weight.shape().dims::<2>()[0],
                    m.spatial_features_mlp.l3.weight.shape().dims::<2>()[1],
                ),
                m.spatial_features_mlp
                    .l3
                    .weight
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}spatial_l3_bias", p),
            &Array1::from_vec(
                m.spatial_features_mlp
                    .l3
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing spatial_l3 bias"))?
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}head_l1_weight", p),
            &Array2::from_shape_vec(
                (m.mlp.l1.weight.shape().dims::<2>()[0], m.mlp.l1.weight.shape().dims::<2>()[1]),
                m.mlp.l1.weight.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}head_l1_bias", p),
            &Array1::from_vec(
                m.mlp
                    .l1
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing head_l1 bias"))?
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}head_l2_weight", p),
            &Array2::from_shape_vec(
                (m.mlp.l2.weight.shape().dims::<2>()[0], m.mlp.l2.weight.shape().dims::<2>()[1]),
                m.mlp.l2.weight.to_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}head_l2_bias", p),
            &Array1::from_vec(
                m.mlp
                    .l2
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing head_l2 bias"))?
                    .to_data()
                    .as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}anchors", p),
            &Array1::from_vec(m.anchors.clone().into_data().as_slice::<f32>().map_err(|e| anyhow::anyhow!("{:?}", e))?.to_vec()),
        )?;
    }

    npz.finish()?;
    Ok(Some(out_path))
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
    #[allow(clippy::too_many_arguments)]
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
        cnn_training_mode: CnnTrainingMode,
        hybrid_pass2_full_cnn: bool,
        hybrid_gating: &HybridCnnGatingConfig,
        min_mean_lasso_r2_for_cnn: f64,
        gene_filter: Option<Vec<String>>,
        max_genes: Option<usize>,
        n_parallel: usize,
        output_dir: &str,
        model_export: &ModelExportConfig,
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

        let compute_mean_for_hybrid = matches!(cnn_training_mode, CnnTrainingMode::Hybrid)
            && !hybrid_pass2_full_cnn
            && hybrid_gating.min_mean_target_expression_for_cnn.is_some();

        let gene_mean_arc: Option<Arc<HashMap<String, f64>>> =
            if top_lr_pairs_by_mean_expression.is_some() || compute_mean_for_hybrid {
                let msg = format!(
                    "Computing per-gene mean expression (layer: {})...",
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

        let neighbors = Arc::new(build_neighbors(
            xy.as_ref(),
            hybrid_gating.moran_k_neighbors.max(1),
        ));

        let (force_genes, skip_genes) =
            if matches!(cnn_training_mode, CnnTrainingMode::Hybrid) && !hybrid_pass2_full_cnn {
                let f = if let Some(ref p) = hybrid_gating.cnn_force_genes_file {
                    load_gene_set_file(Path::new(p))?
                } else {
                    HashSet::new()
                };
                let s = if let Some(ref p) = hybrid_gating.cnn_skip_genes_file {
                    load_gene_set_file(Path::new(p))?
                } else {
                    HashSet::new()
                };
                (Arc::new(f), Arc::new(s))
            } else {
                (Arc::new(HashSet::new()), Arc::new(HashSet::new()))
            };

        let hybrid_collect_top_k = matches!(cnn_training_mode, CnnTrainingMode::Hybrid)
            && !hybrid_pass2_full_cnn
            && hybrid_gating.hybrid_cnn_top_k.is_some();
        let cnn_candidates: Arc<Mutex<Vec<(String, f64, CnnGateDecision)>>> =
            Arc::new(Mutex::new(Vec::new()));

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
            let cnn_mode_w = cnn_training_mode;
            let hybrid_pass2 = hybrid_pass2_full_cnn;
            let hybrid_cfg = hybrid_gating.clone();
            let min_mean_r2 = min_mean_lasso_r2_for_cnn;
            let neighbors_w = neighbors.clone();
            let force_w = force_genes.clone();
            let skip_w = skip_genes.clone();
            let candidates_w = cnn_candidates.clone();
            let model_export_w = model_export.clone();
            let collect_top_k = hybrid_collect_top_k;
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

                        let worker_run_full_cnn = hybrid_pass2
                            || matches!(cnn_mode_w, CnnTrainingMode::Full);
                        estimator.seed_only = !worker_run_full_cnn;
                        if matches!(cnn_mode_w, CnnTrainingMode::Hybrid) && !hybrid_pass2 {
                            estimator.seed_only = true;
                        }
                        let phase_str =
                            if matches!(cnn_mode_w, CnnTrainingMode::Hybrid) && !hybrid_pass2 {
                                format!("hybrid gate | {} mods", n_mods)
                            } else if worker_run_full_cnn {
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

                        let mut export_per_cell = matches!(cnn_mode_w, CnnTrainingMode::Full)
                            || hybrid_pass2;
                        let mut gate_record: Option<CnnGateDecision> = None;

                        if fit_ok {
                            let hybrid_gate = matches!(cnn_mode_w, CnnTrainingMode::Hybrid)
                                && !hybrid_pass2
                                && n_mods > 0;
                            if hybrid_gate {
                                match estimator.build_x_modulators_and_target_y() {
                                    Ok((x_mat, y_vec)) => {
                                        let decision_opt = estimator.estimator.as_ref().map(|inn| {
                                            let summaries_snapshot =
                                                inn.cluster_training_summaries.clone();
                                            let yhat = predict_lasso_y(
                                                &inn.lasso_coefficients,
                                                &inn.lasso_intercepts,
                                                &x_mat,
                                                &clusters,
                                            );
                                            let residuals: Vec<f64> = y_vec
                                                .iter()
                                                .zip(yhat.iter())
                                                .map(|(a, b)| a - b)
                                                .collect();
                                            let mean_tgt = gene_mean_arc
                                                .as_ref()
                                                .and_then(|m| m.get(&gene).copied());
                                            decide_cnn_for_gene(
                                                &hybrid_cfg,
                                                min_mean_r2,
                                                &gene,
                                                &summaries_snapshot,
                                                estimator.regulators.len(),
                                                estimator.lr_pairs.len(),
                                                estimator.tfl_pairs.len(),
                                                &residuals,
                                                neighbors_w.as_ref(),
                                                &force_w,
                                                &skip_w,
                                                mean_tgt,
                                            )
                                        });
                                        if let Some(decision) = decision_opt {
                                            gate_record = Some(decision.clone());
                                            if collect_top_k {
                                                export_per_cell = false;
                                                if decision.use_cnn {
                                                    if let Ok(mut c) = candidates_w.lock() {
                                                        c.push((
                                                            gene.clone(),
                                                            decision.rank_score,
                                                            decision,
                                                        ));
                                                    }
                                                }
                                            } else if decision.use_cnn {
                                                if let Some(inn) = estimator.estimator.as_mut() {
                                                    inn.fit_cnn_refinement(
                                                        &x_mat,
                                                        &y_vec,
                                                        &xy,
                                                        &clusters,
                                                        num_clusters,
                                                        &device,
                                                        epochs,
                                                        learning_rate,
                                                        &cnn_w,
                                                    );
                                                }
                                                export_per_cell = true;
                                            } else {
                                                export_per_cell = false;
                                            }
                                        } else {
                                            export_per_cell = false;
                                        }
                                    }
                                    Err(e) => {
                                        log_line(
                                            &hud,
                                            format!("hybrid design matrix failed {}: {}", gene, e),
                                        );
                                        export_per_cell = false;
                                    }
                                }
                            }
                        }

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

                                if export_per_cell {
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
                                if wrote && export_per_cell && model_export_w.save_cnn_weights {
                                    match export_cnn_models_npz(
                                        est,
                                        &gene,
                                        &training_dir,
                                        &model_export_w,
                                    ) {
                                        Ok(Some(path)) => {
                                            log_line(
                                                &hud,
                                                format!(">> wrote cnn model {}", path),
                                            );
                                        }
                                        Ok(None) => {}
                                        Err(e) => {
                                            log_line(
                                                &hud,
                                                format!(
                                                    ">> warn (cnn model export) {}: {}",
                                                    gene, e
                                                ),
                                            );
                                        }
                                    }
                                }
                                let safe_gene = gene.replace(['/', '\\'], "_");
                                let log_path = format!("{}/log/{}.log", training_dir, safe_gene);
                                let _ = crate::training_log::write_gene_training_log(
                                    std::path::Path::new(&log_path),
                                    &gene,
                                    !export_per_cell,
                                    export_per_cell,
                                    epochs,
                                    learning_rate,
                                    n_iter,
                                    tol,
                                    &est.cluster_training_summaries,
                                    gate_record.as_ref(),
                                );
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        if wrote {
                                            g.record_gene_export_mode(export_per_cell);
                                        }
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

        if matches!(cnn_training_mode, CnnTrainingMode::Hybrid) && !hybrid_pass2_full_cnn {
            if let Some(k) = hybrid_gating.hybrid_cnn_top_k {
                let mut cand = cnn_candidates
                    .lock()
                    .map_err(|e| anyhow::anyhow!("hybrid candidate lock poisoned: {}", e))?;
                cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let picked: Vec<String> = cand
                    .iter()
                    .filter(|t| t.2.use_cnn)
                    .take(k)
                    .map(|t| t.0.clone())
                    .collect();
                drop(cand);
                if !picked.is_empty() {
                    log_line(
                        &hud,
                        format!(
                            "hybrid phase 2: re-training {} genes with full CNN (top-K)",
                            picked.len()
                        ),
                    );
                    if let Some(ref h) = hud {
                        if let Ok(mut g) = h.lock() {
                            let k = picked.len();
                            g.total_genes = g.total_genes.saturating_add(k);
                            g.genes_done = g.genes_done.saturating_sub(k);
                            g.genes_rounds = g.genes_rounds.saturating_sub(k);
                            g.genes_exported_seed_only =
                                g.genes_exported_seed_only.saturating_sub(k);
                        }
                    }
                    for g in &picked {
                        let pth = format!("{training_dir}/{g}_betadata.csv");
                        let _ = fs::remove_file(&pth);
                    }
                    return Self::fit_all_genes(
                        adata_path,
                        radius,
                        spatial_dim,
                        contact_distance,
                        tf_ligand_cutoff,
                        max_lr_pairs,
                        top_lr_pairs_by_mean_expression,
                        layer,
                        cnn,
                        epochs,
                        learning_rate,
                        score_threshold,
                        l1_reg,
                        group_reg,
                        n_iter,
                        tol,
                        cnn_training_mode,
                        true,
                        hybrid_gating,
                        min_mean_lasso_r2_for_cnn,
                        Some(picked),
                        None,
                        n_parallel,
                        output_dir,
                        model_export,
                        hud,
                        device,
                    );
                }
            }
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
    pub fn build_x_modulators_and_target_y(&self) -> anyhow::Result<(Array2<f64>, Array1<f64>)> {
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

        Ok((x_modulators, target_expr))
    }

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
        let (x_modulators, target_expr) = self.build_x_modulators_and_target_y()?;

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
