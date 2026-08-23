use crate::betadata::{
    Betabase, BetadataProgressPhase, BetadataUiProgress, GeneMatrix,
    betadata_cluster_keys_from_obs_dataframe, clusters_usize_from_obs_dataframe,
    obs_series_row_str, resolve_betadata_cluster_key_column, write_betadata_feather,
};
use crate::config::{SpaceshipConfig, expand_user_path};
use crate::ligand::{
    calculate_weighted_ligands_grid_with_cutoff, calculate_weighted_ligands_with_cutoff,
};
use crate::perturb::{
    CachedBaselineSplash, ExpressionBounds, PerturbConfig, PerturbTarget, PerturbTimings,
    PerturbWithTargetsInputs, perturb_with_targets,
};
use crate::spatial_estimator::{load_spatial_coords_f64, read_expression_matrix_dense_f64};

pub fn single_perturb_target(
    gene: &str,
    desired_expr: f64,
    gene_names: &[String],
) -> anyhow::Result<PerturbTarget> {
    if !gene_names.iter().any(|g| g == gene) {
        anyhow::bail!("Gene '{}' is not present in AnnData var_names.", gene);
    }
    Ok(PerturbTarget {
        gene: gene.to_string(),
        desired_expr,
        cell_indices: None,
    })
}
use anndata::data::SelectInfoElem;
use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::Context;
use ndarray::Array2;
use polars::prelude::{CsvReadOptions, DataType, SerReader};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

pub struct PerturbRuntime {
    pub run_toml_path: PathBuf,
    pub run_dir: PathBuf,
    pub cfg: SpaceshipConfig,
    pub gene_mtx: Array2<f64>,
    pub gene_names: Vec<String>,
    pub obs_names: Vec<String>,
    /// Betadata cluster key string per cell (same order as `obs_names` / training feathers).
    pub betadata_cluster_key: Vec<String>,
    /// Labels for TUI mean-expression preview: prefers `obs` column `cell_type` when present,
    /// otherwise matches [`Self::betadata_cluster_key`].
    pub expression_preview_labels: Vec<String>,
    pub cell_types: Vec<usize>,
    pub bb: Betabase,
    pub xy: Array2<f64>,
    pub rw_ligands_init: GeneMatrix,
    pub rw_tfligands_init: GeneMatrix,
    pub lr_radii: HashMap<String, f64>,
    pub perturb_cfg: PerturbConfig,
    /// Reuse iteration-0 `compute_splash_all` across perturbations with matching splash parameters.
    pub baseline_splash_cache: Mutex<Option<CachedBaselineSplash>>,
}

/// Resolve `obs_names` line-list file into sorted unique row indices (AnnData order).
pub fn perturb_obs_indices_from_file(
    path: &Path,
    obs_names_full: &[String],
) -> anyhow::Result<Vec<usize>> {
    let name_to_i: HashMap<&str, usize> = obs_names_full
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let text = fs::read_to_string(path)
        .with_context(|| format!("read perturb_obs_subset_file {}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let idx = *name_to_i
            .get(t)
            .ok_or_else(|| anyhow::anyhow!("obs name {:?} not in AnnData obs_names", t))?;
        out.push(idx);
    }
    out.sort_unstable();
    out.dedup();
    anyhow::ensure!(
        !out.is_empty(),
        "perturb obs subset file {} produced no rows",
        path.display()
    );
    Ok(out)
}

fn subset_xy_rows(xy: &Array2<f64>, row_idx: &[usize]) -> anyhow::Result<Array2<f64>> {
    let n = row_idx.len();
    let g = xy.ncols();
    let mut out = Array2::<f64>::zeros((n, g));
    for (ni, &oi) in row_idx.iter().enumerate() {
        anyhow::ensure!(oi < xy.nrows(), "row index {} out of bounds for xy", oi);
        out.row_mut(ni).assign(&xy.row(oi));
    }
    Ok(out)
}

fn sanitize_float(v: f64) -> String {
    format!("{:.6}", v).replace('-', "m").replace('.', "p")
}

fn request_output_dir(
    run_dir: &Path,
    selected: &[String],
    value: f64,
    n_propagation: usize,
    job_id: Option<u64>,
) -> PathBuf {
    let mut sorted = selected.to_vec();
    sorted.sort();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sorted.hash(&mut hasher);
    n_propagation.hash(&mut hasher);
    if let Some(j) = job_id {
        j.hash(&mut hasher);
    }
    let hash = hasher.finish();
    run_dir
        .join("perturbations")
        .join(format!("genes_{hash:016x}_value_{}", sanitize_float(value)))
}

pub struct ComputeInitialWeightedLigandsArgs<'a> {
    pub gene_mtx: &'a Array2<f64>,
    pub gene_names: &'a [String],
    pub ligand_names: &'a [String],
    pub xy: &'a Array2<f64>,
    pub lr_radii: &'a HashMap<String, f64>,
    pub weighted_ligand_scale: f64,
    pub min_expression: f64,
    pub grid_factor: Option<f64>,
    pub contact_distance: Option<f64>,
}

pub fn compute_initial_weighted_ligands(args: ComputeInitialWeightedLigandsArgs<'_>) -> GeneMatrix {
    let ComputeInitialWeightedLigandsArgs {
        gene_mtx,
        gene_names,
        ligand_names,
        xy,
        lr_radii,
        weighted_ligand_scale,
        min_expression,
        grid_factor,
        contact_distance,
    } = args;
    let n_cells = gene_mtx.nrows();
    let gene_to_idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    let mut seen = HashSet::new();
    let unique: Vec<&String> = ligand_names
        .iter()
        .filter(|l| seen.insert(l.as_str()))
        .collect();

    let mut lig_names = Vec::new();
    let mut lig_data_cols = Vec::new();
    for &lig in &unique {
        if let Some(&gi) = gene_to_idx.get(lig.as_str()) {
            lig_names.push(lig.clone());
            let col: Vec<f64> = (0..n_cells)
                .map(|i| {
                    let v = gene_mtx[[i, gi]];
                    if v > min_expression { v } else { 0.0 }
                })
                .collect();
            lig_data_cols.push(col);
        }
    }

    if lig_names.is_empty() {
        return GeneMatrix::new(Array2::<f32>::zeros((n_cells, 0)), Vec::new());
    }

    let n_lig = lig_names.len();
    let mut lig_data = Array2::<f64>::zeros((n_cells, n_lig));
    for (j, col) in lig_data_cols.iter().enumerate() {
        for i in 0..n_cells {
            lig_data[[i, j]] = col[i];
        }
    }

    let mut radius_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (j, name) in lig_names.iter().enumerate() {
        if let Some(&r) = lr_radii.get(name) {
            radius_groups.entry(r.to_bits()).or_default().push(j);
        }
    }

    let mut result = Array2::<f32>::zeros((n_cells, n_lig));
    for (rbits, group) in &radius_groups {
        let radius = f64::from_bits(*rbits);
        let mut sub = Array2::<f64>::zeros((n_cells, group.len()));
        for (k, &j) in group.iter().enumerate() {
            sub.column_mut(k).assign(&lig_data.column(j));
        }
        let weighted = match grid_factor {
            Some(gf) if gf.is_finite() && gf > 0.0 => calculate_weighted_ligands_grid_with_cutoff(
                xy,
                &sub,
                radius,
                weighted_ligand_scale,
                gf,
                contact_distance,
            ),
            _ => calculate_weighted_ligands_with_cutoff(
                xy,
                &sub,
                radius,
                weighted_ligand_scale,
                contact_distance,
            ),
        };
        for (k, &j) in group.iter().enumerate() {
            let col = weighted.column(k);
            for i in 0..n_cells {
                result[[i, j]] = col[i] as f32;
            }
        }
    }

    GeneMatrix::new(result, lig_names)
}

impl PerturbRuntime {
    pub fn from_run_toml(run_toml: &Path) -> anyhow::Result<Self> {
        Self::from_run_toml_with_config_overlay(run_toml, None)
    }

    pub fn from_run_toml_with_config_overlay(
        run_toml: &Path,
        config_overlay: Option<&toml::Value>,
    ) -> anyhow::Result<Self> {
        let dummy_ui = Arc::new(BetadataUiProgress::new());
        Self::from_run_toml_with_progress(run_toml, None, None, Some(dummy_ui), config_overlay)
    }

    /// Same as [`from_run_toml`], but reports coarse load progress in **permille** (0–1000) and
    /// optional status text for UIs (e.g. spatial viewer).
    pub fn from_run_toml_with_progress(
        run_toml: &Path,
        progress_permille: Option<Arc<AtomicU32>>,
        progress_message: Option<Arc<Mutex<String>>>,
        betadata_progress: Option<Arc<BetadataUiProgress>>,
        config_overlay: Option<&toml::Value>,
    ) -> anyhow::Result<Self> {
        let set_p = |v: u32| {
            if let Some(p) = &progress_permille {
                p.store(v.min(1000), Ordering::Relaxed);
            }
        };
        let set_msg = |s: &str| {
            if let Some(m) = &progress_message {
                if let Ok(mut g) = m.lock() {
                    *g = s.to_string();
                }
            }
        };

        let run_toml_path = run_toml.to_path_buf();
        set_msg("Reading run configuration…");
        set_p(20);
        let cfg = if let Some(ov) = config_overlay {
            SpaceshipConfig::from_file_merged(&run_toml_path, Some(ov))?
        } else {
            SpaceshipConfig::from_file(&run_toml_path)?
        };
        let run_dir = cfg.resolve_training_output_dir(run_toml);

        let adata_path = expand_user_path(cfg.resolve_adata_path().as_str());
        if adata_path.is_empty() {
            anyhow::bail!("data.adata_path is empty in run TOML");
        }
        set_msg("Opening AnnData…");
        set_p(40);
        let adata = AnnData::<H5>::open(H5::open(adata_path.as_str())?)?;
        let gene_names = adata.var_names().into_vec();
        let obs_names_full = adata.obs_names().into_vec();
        let row_idx: Vec<usize> = if let Some(rel) = cfg.data.perturb_obs_subset_file.as_deref() {
            let p = PathBuf::from(expand_user_path(rel));
            perturb_obs_indices_from_file(p.as_path(), &obs_names_full)?
        } else {
            (0..obs_names_full.len()).collect()
        };
        let obs_df = adata.read_obs()?;
        let betadata_key_col =
            resolve_betadata_cluster_key_column(&obs_df, cfg.data.cluster_annot.as_str());
        let cluster_keys_full =
            betadata_cluster_keys_from_obs_dataframe(&obs_df, betadata_key_col.as_str())?;
        let clusters_full =
            clusters_usize_from_obs_dataframe(&obs_df, cfg.data.cluster_annot.as_str())?;
        let obs_names: Vec<String> = row_idx.iter().map(|&i| obs_names_full[i].clone()).collect();
        let cluster_keys: Vec<String> = row_idx
            .iter()
            .map(|&i| cluster_keys_full[i].clone())
            .collect();
        let expression_preview_labels: Vec<String> = if obs_df.column("cell_type").is_ok() {
            let col = obs_df
                .column("cell_type")
                .with_context(|| "read obs column cell_type")?;
            let series = col.as_materialized_series();
            row_idx
                .iter()
                .map(|&i| obs_series_row_str(series, i))
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            cluster_keys.clone()
        };
        let clusters: Vec<usize> = row_idx.iter().map(|&i| clusters_full[i]).collect();
        let xy_full = load_spatial_coords_f64(&adata)?;
        let xy = subset_xy_rows(&xy_full, &row_idx)?;
        let slice = [
            SelectInfoElem::Index(row_idx.clone()),
            SelectInfoElem::full(),
        ];
        set_msg("Reading expression matrix…");
        set_p(80);
        let gene_mtx = read_expression_matrix_dense_f64(&adata, cfg.data.layer.as_str(), &slice)?;

        let gene2index: HashMap<String, usize> = gene_names
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        let betadata_dir = run_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("training output directory is not valid UTF-8"))?;
        set_msg("Loading betadata feathers…");
        let p_perm = progress_permille.clone();
        let ui_prog = betadata_progress.clone();
        let on_betadata: Option<Arc<dyn Fn(u32, BetadataProgressPhase) + Send + Sync>> =
            if progress_permille.is_some() || betadata_progress.is_some() {
                Some(Arc::new(move |sub: u32, phase: BetadataProgressPhase| {
                    if let Some(g) = &p_perm {
                        let v = 120u32.saturating_add(sub.saturating_mul(700) / 1000);
                        g.store(v.min(820), Ordering::Relaxed);
                    }
                    if let Some(c) = &ui_prog {
                        match phase {
                            BetadataProgressPhase::ReadingFeathers { done, total } => {
                                c.phase.store(1, Ordering::Relaxed);
                                c.done.store(done as u32, Ordering::Relaxed);
                                c.total.store(total as u32, Ordering::Relaxed);
                            }
                            BetadataProgressPhase::ExpandingToCells { done, total } => {
                                c.phase.store(2, Ordering::Relaxed);
                                c.done.store(done as u32, Ordering::Relaxed);
                                c.total.store(total as u32, Ordering::Relaxed);
                            }
                        }
                    }
                }))
            } else {
                None
            };
        let bb = Betabase::from_directory(
            betadata_dir,
            &obs_names,
            &cluster_keys,
            Some(&gene2index),
            on_betadata,
        )
        .with_context(|| {
            format!(
                "Failed to load *_betadata.feather from {}",
                run_dir.display()
            )
        })?;
        if let Some(ui) = &betadata_progress {
            ui.reset();
        }

        let mut lr_radii: HashMap<String, f64> = HashMap::new();
        for lig in bb.ligands_set.iter().chain(bb.tfl_ligands_set.iter()) {
            lr_radii.insert(lig.clone(), cfg.spatial.radius);
        }

        let min_expression = 1e-9;
        let grid = cfg.ligand_field.ligand_grid_factor;
        let wl_scale = cfg.ligand_field.weighted_ligand_scale_factor;
        let lr_ligands: Vec<String> = bb.ligands_set.iter().cloned().collect();
        let tfl_ligands: Vec<String> = bb.tfl_ligands_set.iter().cloned().collect();
        set_msg("Weighted ligand precomputation (LR)…");
        set_p(830);
        let rw_ligands_init = compute_initial_weighted_ligands(ComputeInitialWeightedLigandsArgs {
            gene_mtx: &gene_mtx,
            gene_names: &gene_names,
            ligand_names: &lr_ligands,
            xy: &xy,
            lr_radii: &lr_radii,
            weighted_ligand_scale: wl_scale,
            min_expression,
            grid_factor: grid,
            contact_distance: None,
        });
        set_msg("Weighted ligand precomputation (TFL)…");
        set_p(910);
        let rw_tfligands_init =
            compute_initial_weighted_ligands(ComputeInitialWeightedLigandsArgs {
                gene_mtx: &gene_mtx,
                gene_names: &gene_names,
                ligand_names: &tfl_ligands,
                xy: &xy,
                lr_radii: &lr_radii,
                weighted_ligand_scale: wl_scale,
                min_expression,
                grid_factor: grid,
                contact_distance: None,
            });

        let perturb_cfg = PerturbConfig {
            n_propagation: cfg.perturbation.n_propagation,
            scale_factor: wl_scale,
            beta_scale_factor: cfg.perturbation.beta_scale_factor,
            beta_cap: cfg.perturbation.beta_cap,
            min_expression,
            ligand_grid_factor: cfg.ligand_field.ligand_grid_factor,
            contact_distance: None,
            perturbed_gene_min_bound: cfg.perturbation.perturbed_gene_min_bound,
            perturbed_gene_max_bound: cfg.perturbation.perturbed_gene_max_bound,
        };
        let bounds = ExpressionBounds::from_config(&perturb_cfg);
        anyhow::ensure!(
            bounds.min.is_finite() && bounds.max.is_finite() && bounds.min <= bounds.max,
            "perturbed_gene_min_bound must be <= perturbed_gene_max_bound and both must be finite"
        );

        set_msg("Perturbation runtime ready.");
        set_p(1000);

        Ok(Self {
            run_toml_path,
            run_dir,
            cfg,
            gene_mtx,
            gene_names,
            obs_names,
            betadata_cluster_key: cluster_keys.clone(),
            expression_preview_labels,
            cell_types: clusters,
            bb,
            xy,
            rw_ligands_init,
            rw_tfligands_init,
            lr_radii,
            perturb_cfg,
            baseline_splash_cache: Mutex::new(None),
        })
    }
}

/// After a successful perturbation, check output shape and (for near-zero targets) that the KO gene is ~0 on scoped rows.
pub fn validate_perturb_simulated_matrix(
    gene_mtx: &Array2<f64>,
    gene_names: &[String],
    simulated: &Array2<f64>,
    target_gene: &str,
    desired_expr: f64,
    cell_indices: Option<&[usize]>,
) -> anyhow::Result<()> {
    let nrows = gene_mtx.nrows();
    let ngenes = gene_names.len();
    if simulated.nrows() != nrows || simulated.ncols() != ngenes {
        anyhow::bail!(
            "perturb output shape {:?} != expected ({nrows}, {ngenes})",
            simulated.dim()
        );
    }
    let g_col = gene_names
        .iter()
        .position(|g| g == target_gene)
        .ok_or_else(|| anyhow::anyhow!("validate: gene {:?} not in var names", target_gene))?;
    const KO_DESIRED_EPS: f64 = 1e-6;
    const KO_VALUE_TOL: f64 = 1e-4;
    if desired_expr.abs() <= KO_DESIRED_EPS {
        let mut max_dev = 0.0f64;
        let mut n_checked = 0usize;
        match cell_indices {
            None => {
                for r in 0..nrows {
                    max_dev = max_dev.max(simulated[[r, g_col]].abs());
                    n_checked += 1;
                }
            }
            Some(idxs) => {
                for &r in idxs {
                    if r >= nrows {
                        anyhow::bail!("validate: cell index {r} >= nrows {nrows}");
                    }
                    max_dev = max_dev.max(simulated[[r, g_col]].abs());
                    n_checked += 1;
                }
            }
        }
        if max_dev > KO_VALUE_TOL {
            anyhow::bail!(
                "KO check failed for gene {:?}: max |simulated| = {max_dev:.3e} over {n_checked} row(s) (tol {KO_VALUE_TOL:.1e})",
                target_gene
            );
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct PerturbRunSummary {
    run_toml_path: String,
    selected_genes: Vec<String>,
    target_value: f64,
    output_dir: String,
    n_propagation: usize,
    beta_scale_factor: f64,
    beta_cap: Option<f64>,
    ligand_grid_factor: Option<f64>,
    outputs: Vec<String>,
    selected_cell_types_per_gene: HashMap<String, Option<Vec<usize>>>,
}

pub type GeneCellTypeScopes = HashMap<String, Option<HashSet<usize>>>;

/// CSV with header row; each column lists AnnData `obs_names` (one per row). Parsed once into cell indices.
#[derive(Clone, Debug)]
pub struct ObsColumnsCsv {
    pub column_names: Vec<String>,
    columns: HashMap<String, Vec<usize>>,
}

impl ObsColumnsCsv {
    pub fn indices_for_column(&self, name: &str) -> Option<&[usize]> {
        self.columns.get(name).map(|v| v.as_slice())
    }

    pub fn is_empty(&self) -> bool {
        self.column_names.is_empty()
    }
}

pub fn build_obs_name_index_map(obs_names: &[String]) -> anyhow::Result<HashMap<String, usize>> {
    let mut m = HashMap::with_capacity(obs_names.len());
    for (i, name) in obs_names.iter().enumerate() {
        if m.insert(name.clone(), i).is_some() {
            anyhow::bail!(
                "Duplicate obs_name in AnnData: '{}' (ambiguous indices).",
                name
            );
        }
    }
    Ok(m)
}

fn parse_obs_column_values(
    series: &polars::prelude::Series,
    col_name: &str,
    obs_to_idx: &HashMap<String, usize>,
) -> anyhow::Result<Vec<usize>> {
    let string_series = series.cast(&DataType::String).with_context(|| {
        format!(
            "CSV column '{}': could not cast to string for validation",
            col_name
        )
    })?;
    let ca = string_series
        .str()
        .map_err(|e| anyhow::anyhow!("CSV column '{}': {}", col_name, e))?;
    let mut seen = HashSet::new();
    for row in 0..ca.len() {
        let Some(raw) = ca.get(row) else {
            continue;
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(&cell_idx) = obs_to_idx.get(trimmed) else {
            anyhow::bail!(
                "CSV column '{}', row {}: obs_name '{}' not found in AnnData obs",
                col_name,
                row + 2,
                trimmed
            );
        };
        seen.insert(cell_idx);
    }
    let mut v: Vec<usize> = seen.into_iter().collect();
    v.sort_unstable();
    Ok(v)
}

pub fn parse_obs_columns_csv(path: &Path, obs_names: &[String]) -> anyhow::Result<ObsColumnsCsv> {
    let obs_to_idx = build_obs_name_index_map(obs_names)?;
    let pb = path
        .to_path_buf()
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf());
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(pb.clone()))
        .with_context(|| format!("open CSV {}", path.display()))?
        .finish()
        .with_context(|| format!("parse CSV {}", path.display()))?;
    if df.width() == 0 {
        anyhow::bail!("CSV {} has no columns", path.display());
    }
    let column_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut columns = HashMap::with_capacity(column_names.len());
    for name in &column_names {
        let col = df
            .column(name)
            .with_context(|| format!("CSV column '{}'", name))?;
        let indices = parse_obs_column_values(col.as_materialized_series(), name, &obs_to_idx)?;
        columns.insert(name.clone(), indices);
    }
    Ok(ObsColumnsCsv {
        column_names,
        columns,
    })
}

fn escape_csv_field(s: &str) -> String {
    if s.contains(['"', ',', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Write a perturbation `cells.csv`: header = distinct `labels`, each column lists `obs_names`.
pub fn write_cells_csv_grouped_by_label(
    out_path: &Path,
    obs_names: &[String],
    labels: &[String],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        obs_names.len() == labels.len(),
        "obs_names and labels length mismatch ({} vs {})",
        obs_names.len(),
        labels.len()
    );
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, label) in obs_names.iter().zip(labels.iter()) {
        groups.entry(label.clone()).or_default().push(name.clone());
    }
    for cells in groups.values_mut() {
        cells.sort();
    }
    let column_names: Vec<&String> = groups.keys().collect();
    let max_rows = groups.values().map(|v| v.len()).max().unwrap_or(0);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create directory {}", parent.display()))?;
        }
    }
    let mut out = Vec::new();
    {
        let mut w = io::BufWriter::new(&mut out);
        let header: Vec<String> = column_names
            .iter()
            .map(|name| escape_csv_field(name))
            .collect();
        writeln!(w, "{}", header.join(",")).context("write cells.csv header")?;
        for row in 0..max_rows {
            let fields: Vec<String> = column_names
                .iter()
                .map(|name| {
                    groups
                        .get(*name)
                        .and_then(|cells| cells.get(row))
                        .map(|s| escape_csv_field(s))
                        .unwrap_or_default()
                })
                .collect();
            writeln!(w, "{}", fields.join(",")).context("write cells.csv row")?;
        }
        w.flush().context("flush cells.csv")?;
    }
    fs::write(out_path, &out).with_context(|| format!("write {}", out_path.display()))?;
    Ok(())
}

/// Build `cells.csv` in the training output directory from a finished run repro TOML.
///
/// Columns are the distinct values of `[data].cluster_annot` in AnnData `obs` (default column
/// name `cell_type`). Respects `[data].perturb_obs_subset_file` when set.
pub fn write_cells_csv_from_run_toml(
    run_toml: &Path,
    out_path: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    crate::ensure_process_env();
    let cfg = SpaceshipConfig::from_file(run_toml)?;
    let annot_col = cfg.data.cluster_annot.trim();
    anyhow::ensure!(
        !annot_col.is_empty(),
        "[data].cluster_annot must be non-empty in {}",
        run_toml.display()
    );
    let ctx = load_obs_for_collect_interactions(run_toml, annot_col)?;
    let output_dir = cfg.resolve_training_output_dir(run_toml);
    let out = out_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| output_dir.join("cells.csv"));
    write_cells_csv_grouped_by_label(&out, &ctx.obs_names, &ctx.cell_type_labels)?;
    eprintln!(
        "Wrote {} ({} columns, {} cells) using obs[{:?}] from {}",
        out.display(),
        ctx.cell_type_labels.iter().collect::<HashSet<_>>().len(),
        ctx.obs_names.len(),
        annot_col,
        cfg.resolve_adata_path()
    );
    Ok(out)
}

/// Combines optional CSV column indices with optional per–cell-type row index lists (intersection when both).
pub fn merge_csv_and_type_cell_indices(
    csv_indices: Option<&[usize]>,
    type_row_indices: Option<Vec<usize>>,
) -> Option<Vec<usize>> {
    match (csv_indices, type_row_indices) {
        (None, None) => None,
        (Some(c), None) => Some(c.to_vec()),
        (None, Some(mut t)) => {
            t.sort_unstable();
            t.dedup();
            Some(t)
        }
        (Some(c), Some(mut t)) => {
            t.sort_unstable();
            t.dedup();
            if t.is_empty() {
                return Some(Vec::new());
            }
            let tset: HashSet<usize> = t.iter().copied().collect();
            let mut out: Vec<usize> = c.iter().copied().filter(|i| tset.contains(i)).collect();
            out.sort_unstable();
            out.dedup();
            Some(out)
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct JointCellsCsvExportSummary {
    pub path: String,
    pub column: String,
    pub n_cells_per_target_gene: HashMap<String, usize>,
}

#[derive(Serialize)]
struct JointPerturbExportSummary {
    run_toml_path: String,
    selected_genes: Vec<String>,
    desired_expr: f64,
    output_dir: String,
    n_propagation: usize,
    beta_scale_factor: f64,
    beta_cap: Option<f64>,
    ligand_grid_factor: Option<f64>,
    export_kind: String,
    outputs: Vec<String>,
    selected_cell_types_per_gene: HashMap<String, Option<Vec<usize>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cells_csv: Option<JointCellsCsvExportSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    job_id: Option<u64>,
}

pub struct ExportJointPerturbArgs<'a> {
    pub runtime: &'a PerturbRuntime,
    pub simulated: &'a Array2<f64>,
    pub selected_genes: &'a [String],
    pub desired_expr: f64,
    pub n_propagation: usize,
    pub selected_cell_types_per_gene: &'a GeneCellTypeScopes,
    pub cells_csv_summary: Option<JointCellsCsvExportSummary>,
    pub job_id: Option<u64>,
    /// When set, writes `joint_perturb_expr.feather` to this path (parents created).
    /// Summary JSON is written alongside as `perturbation_run_summary.json`.
    pub joint_feather_path: Option<PathBuf>,
}

/// Writes full **joint** `simulated` matrix (one `perturb_with_targets` call with all genes),
/// unlike [`execute_marked_perturbations`] which runs separate jobs per gene.
pub fn export_joint_perturb_result(args: ExportJointPerturbArgs<'_>) -> anyhow::Result<PathBuf> {
    let ExportJointPerturbArgs {
        runtime,
        simulated,
        selected_genes,
        desired_expr,
        n_propagation,
        selected_cell_types_per_gene,
        cells_csv_summary,
        job_id,
        joint_feather_path,
    } = args;
    if selected_genes.is_empty() {
        anyhow::bail!("No selected genes to export.");
    }
    let mut selected: Vec<String> = selected_genes.to_vec();
    selected.sort();
    for g in &selected {
        if !runtime.gene_names.iter().any(|x| x == g) {
            anyhow::bail!("Gene '{}' is not present in AnnData var_names.", g);
        }
    }

    let (out_dir, feather_path, feather_name_for_summary) =
        if let Some(ref custom) = joint_feather_path {
            let feather_path = if custom.parent().is_none_or(|p| p.as_os_str().is_empty()) {
                let name = custom
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("export path has no file name"))?;
                runtime.run_dir.join(name)
            } else {
                custom.clone()
            };
            let parent = feather_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("export path has no parent directory"))?
                .to_path_buf();
            std::fs::create_dir_all(&parent)?;
            let fname = feather_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("joint_perturb_expr.feather")
                .to_string();
            (parent, feather_path, fname)
        } else {
            let out_dir = request_output_dir(
                runtime.run_dir.as_path(),
                &selected,
                desired_expr,
                n_propagation,
                job_id,
            );
            std::fs::create_dir_all(&out_dir)?;
            let feather_name = "joint_perturb_expr.feather";
            let out_path = out_dir.join(feather_name);
            (out_dir.clone(), out_path, feather_name.to_string())
        };

    write_betadata_feather(
        feather_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 output path"))?,
        "CellID",
        &runtime.obs_names,
        &runtime.gene_names,
        simulated,
    )?;

    let summary = JointPerturbExportSummary {
        run_toml_path: runtime.run_toml_path.display().to_string(),
        selected_genes: selected.clone(),
        desired_expr,
        output_dir: out_dir.display().to_string(),
        n_propagation,
        beta_scale_factor: runtime.perturb_cfg.beta_scale_factor,
        beta_cap: runtime.perturb_cfg.beta_cap,
        ligand_grid_factor: runtime.perturb_cfg.ligand_grid_factor,
        export_kind: "joint".into(),
        outputs: vec![feather_name_for_summary],
        selected_cell_types_per_gene: selected
            .iter()
            .map(|g| {
                let scope = selected_cell_types_per_gene
                    .get(g)
                    .and_then(|s| s.as_ref())
                    .map(|set| {
                        let mut v = set.iter().copied().collect::<Vec<_>>();
                        v.sort_unstable();
                        v
                    });
                (g.clone(), scope)
            })
            .collect(),
        cells_csv: cells_csv_summary,
        job_id,
    };
    let summary_path = out_dir.join("perturbation_run_summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    Ok(out_dir)
}

pub fn execute_marked_perturbations(
    runtime: &PerturbRuntime,
    selected_genes: &[String],
    selected_cell_types_per_gene: &GeneCellTypeScopes,
    value: f64,
) -> anyhow::Result<PathBuf> {
    if selected_genes.is_empty() {
        anyhow::bail!("No selected genes to perturb.");
    }
    let selected: Vec<String> = selected_genes.to_vec();
    for g in &selected {
        if !runtime.gene_names.iter().any(|x| x == g) {
            anyhow::bail!("Gene '{}' is not present in AnnData var_names.", g);
        }
    }

    let out_dir = request_output_dir(
        runtime.run_dir.as_path(),
        &selected,
        value,
        runtime.perturb_cfg.n_propagation,
        None,
    );
    std::fs::create_dir_all(&out_dir)?;

    let outputs = selected
        .par_iter()
        .map(|gene| -> anyhow::Result<PathBuf> {
            let selected_cells: Option<Vec<usize>> = selected_cell_types_per_gene
                .get(gene)
                .and_then(|scope| scope.as_ref())
                .map(|cell_types| {
                    runtime
                        .cell_types
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, ct)| {
                            if cell_types.contains(ct) {
                                Some(idx)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                });
            let targets = vec![PerturbTarget {
                gene: gene.clone(),
                desired_expr: value,
                cell_indices: selected_cells,
            }];
            let mut no_timings: Option<PerturbTimings> = None;
            let result = perturb_with_targets(
                &PerturbWithTargetsInputs {
                    bb: &runtime.bb,
                    gene_mtx: &runtime.gene_mtx,
                    gene_names: &runtime.gene_names,
                    xy: &runtime.xy,
                    rw_ligands_init: &runtime.rw_ligands_init,
                    rw_tfligands_init: &runtime.rw_tfligands_init,
                    targets: &targets,
                    config: &runtime.perturb_cfg,
                    lr_radii: &runtime.lr_radii,
                    job_progress: None,
                    job_message: None,
                    cancel: None,
                    baseline_splash_cache: Some(&runtime.baseline_splash_cache),
                },
                &mut no_timings,
            )
            .expect("perturb batch");
            let out_path = out_dir.join(format!("{}_perturb_expr.feather", gene));
            write_betadata_feather(
                out_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("non-utf8 output path"))?,
                "CellID",
                &runtime.obs_names,
                &runtime.gene_names,
                &result.simulated,
            )?;
            Ok(out_path)
        })
        .collect::<Vec<_>>();

    let mut output_paths = Vec::with_capacity(outputs.len());
    for path in outputs {
        output_paths.push(path?);
    }

    let summary = PerturbRunSummary {
        run_toml_path: runtime.run_toml_path.display().to_string(),
        selected_genes: selected.clone(),
        target_value: value,
        output_dir: out_dir.display().to_string(),
        n_propagation: runtime.perturb_cfg.n_propagation,
        beta_scale_factor: runtime.perturb_cfg.beta_scale_factor,
        beta_cap: runtime.perturb_cfg.beta_cap,
        ligand_grid_factor: runtime.perturb_cfg.ligand_grid_factor,
        outputs: output_paths
            .iter()
            .map(|p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            })
            .collect(),
        selected_cell_types_per_gene: selected
            .iter()
            .map(|g| {
                let scope = selected_cell_types_per_gene
                    .get(g)
                    .and_then(|s| s.as_ref())
                    .map(|set| {
                        let mut v = set.iter().copied().collect::<Vec<_>>();
                        v.sort_unstable();
                        v
                    });
                (g.clone(), scope)
            })
            .collect(),
    };

    let summary_path = out_dir.join("perturbation_run_summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    Ok(out_dir)
}

fn prompt_line(prompt: &str) -> anyhow::Result<String> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    Ok(s.trim().to_string())
}

pub fn interactive_run_toml_prompt() -> anyhow::Result<PathBuf> {
    loop {
        let raw = prompt_line("Path to spacetravlr_run_repro.toml: ")?;
        let expanded = expand_user_path(raw.as_str());
        let p = PathBuf::from(expanded);
        if p.is_file() {
            return Ok(p);
        }
        eprintln!("Not found: {}", p.display());
    }
}

pub fn run_interactive(runtime: PerturbRuntime) -> anyhow::Result<()> {
    let mut selected: HashSet<String> = HashSet::new();
    let mut selected_cell_types_per_gene: GeneCellTypeScopes = HashMap::new();
    let mut all_cell_types = runtime.cell_types.to_vec();
    all_cell_types.sort_unstable();
    all_cell_types.dedup();
    println!(
        "Perturbation mode loaded from {}",
        runtime.run_toml_path.display()
    );
    println!("Run directory: {}", runtime.run_dir.display());
    println!(
        "Loaded {} genes and {} cells.",
        runtime.gene_names.len(),
        runtime.obs_names.len()
    );
    println!(
        "Commands: list [N], search <query>, mark <gene> [all|ct1,ct2], scope <gene> <all|ct1,ct2>, unmark <gene>, show, run <value>, quit"
    );
    println!("Available cell_type values: {:?}", all_cell_types);

    fn parse_cell_type_scope(raw: &str) -> anyhow::Result<Option<HashSet<usize>>> {
        let cleaned = raw.trim();
        if cleaned.is_empty() || cleaned.eq_ignore_ascii_case("all") {
            return Ok(None);
        }
        let mut out = HashSet::new();
        for part in cleaned.split(',') {
            let v = part
                .trim()
                .parse::<usize>()
                .with_context(|| format!("Invalid cell_type '{}'", part.trim()))?;
            out.insert(v);
        }
        Ok(Some(out))
    }

    loop {
        let cmd = prompt_line("perturb> ")?;
        let mut parts = cmd.split_whitespace();
        let Some(head) = parts.next() else {
            continue;
        };

        match head {
            "list" => {
                let n = parts
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(40);
                for g in runtime.gene_names.iter().take(n) {
                    let mark = if selected.contains(g) { "[x]" } else { "[ ]" };
                    println!("{mark} {g}");
                }
                if runtime.gene_names.len() > n {
                    println!("... {} more", runtime.gene_names.len() - n);
                }
            }
            "search" => {
                let q = parts.collect::<Vec<_>>().join(" ");
                if q.is_empty() {
                    continue;
                }
                let q_lower = q.to_ascii_lowercase();
                let mut shown = 0usize;
                for g in &runtime.gene_names {
                    if g.to_ascii_lowercase().contains(&q_lower) {
                        let mark = if selected.contains(g) { "[x]" } else { "[ ]" };
                        println!("{mark} {g}");
                        shown += 1;
                        if shown >= 100 {
                            break;
                        }
                    }
                }
                if shown == 0 {
                    println!("No genes matched '{q}'.");
                }
            }
            "mark" => {
                let remaining = parts.collect::<Vec<_>>();
                if remaining.is_empty() {
                    println!("Usage: mark <gene> [all|ct1,ct2]");
                    continue;
                }
                let gene = remaining[0].to_string();
                if runtime.gene_names.iter().any(|g| g == &gene) {
                    selected.insert(gene.clone());
                    let scope = if remaining.len() > 1 {
                        parse_cell_type_scope(remaining[1])?
                    } else {
                        None
                    };
                    selected_cell_types_per_gene.insert(gene, scope);
                } else {
                    println!("Unknown gene.");
                }
            }
            "scope" => {
                let remaining = parts.collect::<Vec<_>>();
                if remaining.len() < 2 {
                    println!("Usage: scope <gene> <all|ct1,ct2>");
                    continue;
                }
                let gene = remaining[0].to_string();
                if !selected.contains(&gene) {
                    println!("Gene is not marked.");
                    continue;
                }
                let scope = parse_cell_type_scope(remaining[1])?;
                selected_cell_types_per_gene.insert(gene, scope);
            }
            "unmark" => {
                let gene = parts.collect::<Vec<_>>().join(" ");
                selected.remove(gene.as_str());
                selected_cell_types_per_gene.remove(gene.as_str());
            }
            "show" => {
                if selected.is_empty() {
                    println!("No genes selected.");
                } else {
                    let mut v = selected.iter().cloned().collect::<Vec<_>>();
                    v.sort();
                    println!("Selected {} genes:", v.len());
                    for g in v {
                        let scope = selected_cell_types_per_gene
                            .get(&g)
                            .and_then(|s| s.as_ref())
                            .map(|set| {
                                let mut vv = set.iter().copied().collect::<Vec<_>>();
                                vv.sort_unstable();
                                vv
                            });
                        match scope {
                            Some(vv) => println!("- {g} (cell_types={:?})", vv),
                            None => println!("- {g} (cell_types=all)"),
                        }
                    }
                }
            }
            "run" => {
                let value = parts
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("Usage: run <value>"))?
                    .parse::<f64>()
                    .with_context(|| "run value must be a floating number")?;
                let mut genes = selected.iter().cloned().collect::<Vec<_>>();
                genes.sort();
                let out = execute_marked_perturbations(
                    &runtime,
                    &genes,
                    &selected_cell_types_per_gene,
                    value,
                )?;
                println!("Finished. Outputs written under {}", out.display());
            }
            "quit" | "exit" => return Ok(()),
            _ => println!("Unknown command."),
        }
    }
}

pub struct CollectInteractionsObs {
    pub obs_names: Vec<String>,
    pub cluster_keys: Vec<String>,
    pub cell_type_labels: Vec<String>,
}

fn collect_interactions_row_indices(
    run_toml: &std::path::Path,
    obs_names_full: &[String],
) -> anyhow::Result<Vec<usize>> {
    let cfg = SpaceshipConfig::from_file(run_toml)?;
    if let Some(rel) = cfg.data.perturb_obs_subset_file.as_deref() {
        let p = PathBuf::from(expand_user_path(rel));
        perturb_obs_indices_from_file(p.as_path(), obs_names_full)
    } else {
        Ok((0..obs_names_full.len()).collect())
    }
}

/// One string per cell from `obs[column]` (e.g. `adata.obs['cluster']`), aligned with training obs order.
pub fn load_obs_column_for_collect_interactions(
    run_toml: &std::path::Path,
    column: &str,
) -> anyhow::Result<Vec<String>> {
    let cfg = SpaceshipConfig::from_file(run_toml)?;
    let adata_path = expand_user_path(cfg.resolve_adata_path().as_str());
    if adata_path.is_empty() {
        anyhow::bail!("data.adata_path is empty in run TOML");
    }
    let adata = AnnData::<H5>::open(H5::open(adata_path.as_str())?)?;
    let obs_names_full = adata.obs_names().into_vec();
    let row_idx = collect_interactions_row_indices(run_toml, &obs_names_full)?;
    let obs_df = adata.read_obs()?;
    let col = obs_df
        .column(column)
        .with_context(|| format!("obs column {:?} not found", column))?;
    let series = col.as_materialized_series();
    let mut out = Vec::with_capacity(row_idx.len());
    for &i in &row_idx {
        out.push(obs_series_row_str(series, i)?);
    }
    Ok(out)
}

pub fn load_obs_for_collect_interactions(
    run_toml: &std::path::Path,
    annot_col: &str,
) -> anyhow::Result<CollectInteractionsObs> {
    let cfg = SpaceshipConfig::from_file(run_toml)?;
    let adata_path = expand_user_path(cfg.resolve_adata_path().as_str());
    if adata_path.is_empty() {
        anyhow::bail!("data.adata_path is empty in run TOML");
    }
    let adata = AnnData::<H5>::open(H5::open(adata_path.as_str())?)?;
    let obs_names_full = adata.obs_names().into_vec();
    let row_idx = collect_interactions_row_indices(run_toml, &obs_names_full)?;
    let obs_df = adata.read_obs()?;
    let betadata_key_col =
        resolve_betadata_cluster_key_column(&obs_df, cfg.data.cluster_annot.as_str());
    let cluster_keys_full =
        betadata_cluster_keys_from_obs_dataframe(&obs_df, betadata_key_col.as_str())?;

    let col = obs_df.column(annot_col).with_context(|| {
        format!(
            "obs column {:?} not found (for cell-type grouping)",
            annot_col
        )
    })?;
    let series = col.as_materialized_series();

    let obs_names: Vec<String> = row_idx.iter().map(|&i| obs_names_full[i].clone()).collect();
    let cluster_keys: Vec<String> = row_idx
        .iter()
        .map(|&i| cluster_keys_full[i].clone())
        .collect();
    let mut cell_type_labels = Vec::with_capacity(row_idx.len());
    for &i in &row_idx {
        cell_type_labels.push(obs_series_row_str(series, i)?);
    }

    Ok(CollectInteractionsObs {
        obs_names,
        cluster_keys,
        cell_type_labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use polars::prelude::SerReader;

    #[test]
    fn single_perturb_target_unknown_gene() {
        assert!(single_perturb_target("Nope", 0.0, &["A".into()]).is_err());
    }

    #[test]
    fn single_perturb_target_ok() {
        let t = single_perturb_target("A", 1.5, &["A".into(), "B".into()]).unwrap();
        assert_eq!(t.gene, "A");
        assert_eq!(t.desired_expr, 1.5);
        assert!(t.cell_indices.is_none());
    }

    #[test]
    fn output_dir_is_deterministic() {
        let run_dir = PathBuf::from("/tmp/example");
        let genes = vec!["GZMB".to_string(), "CD74".to_string()];
        let a = request_output_dir(&run_dir, &genes, 0.0, 4, None);
        let b = request_output_dir(&run_dir, &genes, 0.0, 4, None);
        assert_eq!(a, b);
        let c = request_output_dir(&run_dir, &genes, 0.0, 4, Some(7));
        assert_ne!(a, c);
    }

    #[test]
    fn write_feather_shape_matches_matrix() {
        let temp =
            std::env::temp_dir().join(format!("spacetravlr_perturb_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp);
        let out = temp.join("matrix.feather");
        let obs = vec!["c1".to_string(), "c2".to_string()];
        let genes = vec!["g1".to_string(), "g2".to_string(), "g3".to_string()];
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        write_betadata_feather(out.to_str().unwrap(), "CellID", &obs, &genes, &data).unwrap();
        let f = std::fs::File::open(&out).unwrap();
        let df = polars::prelude::IpcReader::new(f).finish().unwrap();
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 4);
    }

    #[test]
    fn build_obs_name_index_map_rejects_duplicate() {
        let obs = vec!["a".into(), "b".into(), "a".into()];
        assert!(build_obs_name_index_map(&obs).is_err());
    }

    #[test]
    fn merge_csv_and_type_cell_indices_cases() {
        assert_eq!(merge_csv_and_type_cell_indices(None, None), None);
        let c = [1usize, 3, 5];
        assert_eq!(
            merge_csv_and_type_cell_indices(Some(&c), None),
            Some(vec![1, 3, 5])
        );
        assert_eq!(
            merge_csv_and_type_cell_indices(None, Some(vec![3usize, 1])),
            Some(vec![1, 3])
        );
        assert_eq!(
            merge_csv_and_type_cell_indices(Some(&c), Some(vec![3usize, 10])),
            Some(vec![3])
        );
        assert_eq!(
            merge_csv_and_type_cell_indices(Some(&c), Some(vec![])),
            Some(vec![])
        );
    }

    #[test]
    fn write_cells_csv_grouped_roundtrips_with_parser() {
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_make_cells_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("cells.csv");
        let obs = vec!["c1".into(), "c2".into(), "c3".into(), "c4".into()];
        let labels = vec!["B".into(), "A".into(), "A".into(), "B".into()];
        write_cells_csv_grouped_by_label(&p, &obs, &labels).unwrap();
        let body = std::fs::read_to_string(&p).unwrap();
        assert!(body.starts_with("A,B\n"));
        let parsed = parse_obs_columns_csv(&p, &obs).unwrap();
        assert_eq!(parsed.indices_for_column("A").unwrap(), &[1usize, 2]);
        assert_eq!(parsed.indices_for_column("B").unwrap(), &[0usize, 3]);
    }

    #[test]
    fn parse_obs_columns_csv_dedupe_and_unknown() {
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_csv_perturb_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("cells.csv");
        std::fs::write(&p, "col_a,col_b\nalpha,beta\ngamma,alpha\n  alpha  ,\n").unwrap();
        let obs = vec!["alpha".into(), "beta".into(), "gamma".into()];
        let parsed = parse_obs_columns_csv(&p, &obs).unwrap();
        assert_eq!(parsed.indices_for_column("col_a").unwrap(), &[0usize, 2]);
        assert_eq!(parsed.indices_for_column("col_b").unwrap(), &[0usize, 1]);
        std::fs::write(&p, "x\nunknown\n").unwrap();
        assert!(parse_obs_columns_csv(&p, &obs).is_err());
    }

    #[test]
    fn validate_perturb_ko_passes() {
        let g = array![[0.0, 1.0], [2.0, 3.0]];
        let names = vec!["A".into(), "B".into()];
        let sim = array![[1e-8, 1.0], [1e-9, 3.0]];
        validate_perturb_simulated_matrix(&g, &names, &sim, "A", 0.0, None).unwrap();
    }

    #[test]
    fn validate_perturb_ko_scoped_rows() {
        let g = array![[0.0, 1.0], [2.0, 3.0]];
        let names = vec!["A".into(), "B".into()];
        let sim = array![[1.0, 1.0], [1e-8, 3.0]];
        validate_perturb_simulated_matrix(&g, &names, &sim, "A", 0.0, Some(&[1])).unwrap();
    }

    #[test]
    fn validate_perturb_ko_fails_residual() {
        let g = array![[0.0, 1.0], [2.0, 3.0]];
        let names = vec!["A".into(), "B".into()];
        let sim = array![[0.2, 1.0], [2.0, 3.0]];
        assert!(validate_perturb_simulated_matrix(&g, &names, &sim, "A", 0.0, None).is_err());
    }

    #[test]
    fn validate_perturb_shape_mismatch() {
        let g = array![[0.0], [1.0]];
        let names = vec!["A".into()];
        let sim = array![[0.0, 1.0]];
        assert!(validate_perturb_simulated_matrix(&g, &names, &sim, "A", 0.0, None).is_err());
    }
}
