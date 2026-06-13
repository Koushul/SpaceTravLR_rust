use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::Deserialize;

use crate::betadata::{GeneMatrix, write_betadata_feather};
use crate::config::{SPACESHIP_MERGE_SECTIONS, expand_user_path};
use crate::perturb::{
    PerturbConfig, PerturbTarget, PerturbTimings, PerturbWithTargetsInputs, perturb_with_targets,
};
use crate::perturb_mode::{
    ComputeInitialWeightedLigandsArgs, PerturbRuntime, compute_initial_weighted_ligands,
    parse_obs_columns_csv, perturb_obs_indices_from_file, validate_perturb_simulated_matrix,
};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum GenesSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum F64OrVec {
    One(f64),
    Many(Vec<f64>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum UszOrVec {
    One(usize),
    Many(Vec<usize>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OutSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StrOrVec {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Default, Deserialize)]
pub struct PerturbBatchFile {
    #[serde(default)]
    pub gene: Option<String>,
    #[serde(default)]
    pub genes: Option<GenesSpec>,
    #[serde(default)]
    pub desired_expr: Option<F64OrVec>,
    #[serde(default)]
    pub n_propagation: Option<UszOrVec>,
    #[serde(default)]
    pub out: Option<OutSpec>,
    #[serde(default)]
    pub out_dir: Option<String>,
    #[serde(default)]
    pub parallelism: Option<usize>,
    #[serde(default)]
    pub cells_csv: Option<String>,
    #[serde(default)]
    pub cells_csv_column: Option<String>,
    /// Per-gene CSV column names (same length as `genes`, or length 1 broadcast). Empty string = perturb all cells.
    #[serde(default)]
    pub cells_csv_columns: Option<StrOrVec>,
    #[serde(default)]
    pub cells_obs_file: Option<String>,
    /// Override Gaussian radius for received-ligand aggregation (default: `[spatial].radius` from run TOML).
    #[serde(default)]
    pub radius: Option<f64>,
    /// Override `[perturbation].ligand_grid_factor` for this batch (omit = run default).
    #[serde(default)]
    pub ligand_grid_factor: Option<f64>,
    /// Hard max neighbor distance for received ligands (omit = no cutoff, matching historical behavior).
    #[serde(default)]
    pub contact_distance: Option<f64>,
    /// Per-gene ligand β scale (scalar, length-1 broadcast, or one per `genes`). Omit = run / `[perturbation]` default.
    #[serde(default)]
    pub beta_scale_factor: Option<F64OrVec>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedPerturbJob {
    pub gene: String,
    pub desired_expr: f64,
    pub n_propagation: usize,
    pub out_path: PathBuf,
    pub radius: Option<f64>,
    pub ligand_grid_factor: Option<f64>,
    pub contact_distance: Option<f64>,
    pub beta_scale_factor: f64,
    /// Resolved cell row indices for scoped perturbation; `None` = all cells.
    pub cell_indices: Option<Vec<usize>>,
}

pub fn default_worker_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(2)
}

pub fn effective_parallelism(
    file_parallelism: Option<usize>,
    cli_override: Option<usize>,
) -> usize {
    let n = cli_override
        .or(file_parallelism)
        .unwrap_or_else(default_worker_parallelism);
    n.max(1)
}

pub fn load_batch_file(path: &Path) -> anyhow::Result<PerturbBatchFile> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read batch TOML {}: {e}", path.display()))?;
    toml::from_str(&s).map_err(|e| anyhow::anyhow!("parse batch TOML {}: {e}", path.display()))
}

pub struct ParsedPerturbToml {
    /// From `run_toml` / `run_repro` / `spacetravlr_run_repro` in the file; omit when using `--run-toml` only.
    pub run_toml: Option<PathBuf>,
    pub overlay_source: toml::Value,
    pub batch_table: Option<toml::value::Table>,
}

pub fn batch_from_perturb_table(tbl: &toml::value::Table) -> anyhow::Result<PerturbBatchFile> {
    let v = toml::Value::Table(tbl.clone());
    PerturbBatchFile::deserialize(v).map_err(|e| anyhow::anyhow!("parse batch/job fields: {e}"))
}

/// Top-level keys in `--config` TOML that belong under `[perturbation]` for repro merge.
const PERTURBATION_SCALAR_HOIST_KEYS: &[&str] = &["beta_scale_factor", "beta_cap"];

fn hoist_perturbation_scalars_into_overlay(root: &mut toml::Value) {
    let Some(tbl) = root.as_table_mut() else {
        return;
    };
    let mut pert = tbl
        .get("perturbation")
        .and_then(|v| v.as_table())
        .cloned()
        .unwrap_or_default();
    let mut hoisted = false;
    for key in PERTURBATION_SCALAR_HOIST_KEYS {
        if let Some(v) = tbl.remove(*key) {
            // Arrays are per-gene batch fields; only hoist scalars into `[perturbation]`.
            if matches!(v, toml::Value::Array(_)) {
                tbl.insert(key.to_string(), v);
            } else {
                pert.entry(key.to_string()).or_insert(v);
                hoisted = true;
            }
        }
    }
    if hoisted || tbl.contains_key("perturbation") {
        tbl.insert("perturbation".to_string(), toml::Value::Table(pert));
    }
}

pub fn load_perturb_cli_toml(path: &Path) -> anyhow::Result<ParsedPerturbToml> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read perturb config {}: {e}", path.display()))?;
    let mut root: toml::Value = toml::from_str(&s)
        .map_err(|e| anyhow::anyhow!("parse perturb config {}: {e}", path.display()))?;
    hoist_perturbation_scalars_into_overlay(&mut root);
    let table = root
        .as_table()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "perturb config {}: document root must be a TOML table",
                path.display()
            )
        })?
        .clone();
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));

    let run_raw = table
        .get("run_toml")
        .or_else(|| table.get("run_repro"))
        .or_else(|| table.get("spacetravlr_run_repro"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let run_toml = run_raw.map(|r| {
        let exp = expand_user_path(r);
        let p = std::path::Path::new(&exp);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            parent.join(p)
        }
    });

    let mut batch_tbl = table;
    for &sec in SPACESHIP_MERGE_SECTIONS {
        batch_tbl.remove(sec);
    }
    for key in ["run_toml", "run_repro", "spacetravlr_run_repro"] {
        batch_tbl.remove(key);
    }
    let batch_table = if batch_tbl.is_empty() {
        None
    } else {
        Some(batch_tbl)
    };

    Ok(ParsedPerturbToml {
        run_toml,
        overlay_source: root,
        batch_table,
    })
}

/// Effective repro path when `--run-toml` and `--config` may each supply it (`--run-toml` wins).
pub fn resolve_effective_run_toml(
    cli_run_toml: Option<PathBuf>,
    config_run_toml: Option<PathBuf>,
    config_path: Option<&std::path::Path>,
) -> anyhow::Result<PathBuf> {
    if let Some(p) = cli_run_toml {
        return Ok(p);
    }
    if let Some(p) = config_run_toml {
        return Ok(p);
    }
    let hint = config_path
        .map(|p| format!("{}", p.display()))
        .unwrap_or_else(|| "config".into());
    anyhow::bail!("need a run repro TOML: pass --run-toml, or set run_toml in {hint}")
}

pub fn resolve_relative_to(batch_parent: &Path, rel: impl AsRef<Path>) -> PathBuf {
    let p = rel.as_ref();
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        batch_parent.join(p)
    }
}

fn normalize_gene_list(file: &PerturbBatchFile) -> anyhow::Result<Vec<String>> {
    let mut v = match (&file.gene, &file.genes) {
        (Some(_), Some(_)) => {
            anyhow::bail!("batch TOML: use either `gene` or `genes`, not both")
        }
        (Some(g), None) => vec![g.clone()],
        (None, Some(GenesSpec::One(g))) => vec![g.clone()],
        (None, Some(GenesSpec::Many(v))) => v.clone(),
        (None, None) => anyhow::bail!("batch TOML: missing `gene` or `genes`"),
    };
    for g in &mut v {
        *g = g.trim().to_string();
    }
    v.retain(|g| !g.is_empty());
    if v.is_empty() {
        anyhow::bail!("batch TOML: `genes` list is empty");
    }
    Ok(v)
}

fn broadcast_f64(spec: Option<&F64OrVec>, n: usize) -> anyhow::Result<Vec<f64>> {
    broadcast_f64_field(spec, n, 0.0, "desired_expr")
}

pub(crate) fn broadcast_f64_field(
    spec: Option<&F64OrVec>,
    n: usize,
    default: f64,
    field: &'static str,
) -> anyhow::Result<Vec<f64>> {
    match spec {
        None => Ok(vec![default; n]),
        Some(F64OrVec::One(x)) => Ok(vec![*x; n]),
        Some(F64OrVec::Many(v)) => {
            if v.len() == n {
                Ok(v.clone())
            } else if v.len() == 1 {
                Ok(vec![v[0]; n])
            } else {
                anyhow::bail!(
                    "batch TOML: `{field}` length {} must be 1, {n}, or omit (default {default})",
                    v.len(),
                )
            }
        }
    }
}

fn broadcast_str_cols(spec: Option<&StrOrVec>, n: usize) -> anyhow::Result<Vec<String>> {
    match spec {
        None => anyhow::bail!("internal: cells_csv_columns missing"),
        Some(StrOrVec::One(s)) => Ok(vec![s.clone(); n]),
        Some(StrOrVec::Many(v)) => {
            if v.len() == n {
                Ok(v.clone())
            } else if v.len() == 1 {
                Ok(vec![v[0].clone(); n])
            } else {
                anyhow::bail!(
                    "batch TOML: `cells_csv_columns` length {} must be 1 or {}",
                    v.len(),
                    n
                )
            }
        }
    }
}

pub(crate) fn broadcast_usize(
    spec: Option<&UszOrVec>,
    n: usize,
    default: usize,
) -> anyhow::Result<Vec<usize>> {
    match spec {
        None => Ok(vec![default; n]),
        Some(UszOrVec::One(x)) => Ok(vec![*x; n]),
        Some(UszOrVec::Many(v)) => {
            if v.len() == n {
                Ok(v.clone())
            } else if v.len() == 1 {
                Ok(vec![v[0]; n])
            } else {
                anyhow::bail!(
                    "batch TOML: `n_propagation` length {} must be 1, {}, or omit (use run / CLI default)",
                    v.len(),
                    n
                )
            }
        }
    }
}

pub fn sanitize_gene_for_filename(gene: &str) -> String {
    gene.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '\0' => '_',
            c => c,
        })
        .collect()
}

fn default_feather_name(gene: &str) -> String {
    format!("{}_perturb_expr.feather", sanitize_gene_for_filename(gene))
}

pub fn expand_prepared_jobs(
    file: &PerturbBatchFile,
    batch_parent: &Path,
    default_n_propagation: usize,
    default_beta_scale_factor: f64,
) -> anyhow::Result<Vec<PreparedPerturbJob>> {
    let genes = normalize_gene_list(file)?;
    let n = genes.len();
    let desired = broadcast_f64(file.desired_expr.as_ref(), n)?;
    let n_props = broadcast_usize(file.n_propagation.as_ref(), n, default_n_propagation)?;
    let beta_scales = broadcast_f64_field(
        file.beta_scale_factor.as_ref(),
        n,
        default_beta_scale_factor,
        "beta_scale_factor",
    )?;

    let out_paths: Vec<PathBuf> = match (&file.out, &file.out_dir) {
        (Some(_), Some(_)) => anyhow::bail!("batch TOML: set either `out` or `out_dir`, not both"),
        (Some(OutSpec::One(p)), None) => {
            if n != 1 {
                anyhow::bail!(
                    "batch TOML: scalar `out` is only valid when there is exactly one gene"
                );
            }
            vec![resolve_relative_to(batch_parent, p)]
        }
        (Some(OutSpec::Many(paths)), None) => {
            if paths.len() != n {
                anyhow::bail!(
                    "batch TOML: `out` array length {} must equal gene count {}",
                    paths.len(),
                    n
                );
            }
            paths
                .iter()
                .map(|p| resolve_relative_to(batch_parent, p))
                .collect()
        }
        (None, Some(dir)) => {
            let d = resolve_relative_to(batch_parent, dir);
            genes
                .iter()
                .map(|g| d.join(default_feather_name(g)))
                .collect()
        }
        (None, None) => {
            anyhow::bail!("batch TOML: set `out_dir` (default names per gene) or `out`")
        }
    };

    let n = genes.len();
    let radius = file.radius;
    let ligand_grid_factor = file.ligand_grid_factor;
    let contact_distance = file.contact_distance;

    Ok((0..n)
        .map(|i| PreparedPerturbJob {
            gene: genes[i].clone(),
            desired_expr: desired[i],
            n_propagation: n_props[i],
            out_path: out_paths[i].clone(),
            radius,
            ligand_grid_factor,
            contact_distance,
            beta_scale_factor: beta_scales[i],
            cell_indices: None,
        })
        .collect())
}

/// Fills [`PreparedPerturbJob::cell_indices`] for each job (`None` = all cells).
pub fn resolve_prepared_job_cell_indices(
    file: &PerturbBatchFile,
    batch_parent: &Path,
    obs_names: &[String],
    jobs: &mut [PreparedPerturbJob],
) -> anyhow::Result<()> {
    if file.cells_csv_columns.is_some() {
        let has_csv = file
            .cells_csv
            .as_ref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        if !has_csv {
            anyhow::bail!("batch TOML: `cells_csv_columns` requires `cells_csv`");
        }
    }
    let n = jobs.len();
    let csv = file
        .cells_csv
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty());
    let obs_f = file
        .cells_obs_file
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty());

    match (csv, obs_f) {
        (Some(_), Some(_)) => {
            anyhow::bail!("batch TOML: use either `cells_csv` or `cells_obs_file`, not both")
        }
        (None, Some(p)) => {
            let path = resolve_relative_to(batch_parent, p);
            let idx = perturb_obs_indices_from_file(&path, obs_names)?;
            for j in jobs.iter_mut() {
                j.cell_indices = Some(idx.clone());
            }
        }
        (Some(cs), None) => {
            if file.cells_csv_columns.is_some() && file.cells_csv_column.is_some() {
                anyhow::bail!(
                    "batch TOML: set either `cells_csv_column` or `cells_csv_columns`, not both"
                );
            }
            let path = resolve_relative_to(batch_parent, cs);
            let parsed = parse_obs_columns_csv(&path, obs_names)?;

            let col_per_job: Vec<String> = if let Some(ref spec) = file.cells_csv_columns {
                broadcast_str_cols(Some(spec), n)?
            } else if let Some(ref g) = file.cells_csv_column {
                vec![g.clone(); n]
            } else {
                anyhow::bail!(
                    "batch TOML: when `cells_csv` is set, provide `cells_csv_column` or `cells_csv_columns`"
                );
            };

            for (job, col_raw) in jobs.iter_mut().zip(col_per_job.iter()) {
                let col = col_raw.trim();
                if col.is_empty() {
                    job.cell_indices = None;
                } else {
                    let sl = parsed.indices_for_column(col).ok_or_else(|| {
                        anyhow::anyhow!(
                            "batch TOML: cells_csv column {:?} not found in CSV header",
                            col
                        )
                    })?;
                    job.cell_indices = Some(sl.to_vec());
                }
            }
        }
        (None, None) => {
            for j in jobs.iter_mut() {
                j.cell_indices = None;
            }
        }
    }
    Ok(())
}

pub fn validate_jobs_genes(
    jobs: &[PreparedPerturbJob],
    gene_names: &[String],
) -> anyhow::Result<()> {
    for j in jobs {
        if !gene_names.iter().any(|g| g == &j.gene) {
            anyhow::bail!(
                "batch TOML: gene {:?} is not present in AnnData var_names",
                j.gene
            );
        }
    }
    Ok(())
}

fn run_one_job(
    runtime: &PerturbRuntime,
    job: PreparedPerturbJob,
    verbose: bool,
) -> anyhow::Result<()> {
    let cell_indices = job.cell_indices.clone();
    let targets = vec![PerturbTarget {
        gene: job.gene.clone(),
        desired_expr: job.desired_expr,
        cell_indices,
    }];
    let ligand_grid = job
        .ligand_grid_factor
        .or(runtime.perturb_cfg.ligand_grid_factor);
    let contact = job
        .contact_distance
        .or(runtime.perturb_cfg.contact_distance);
    let mut cfg: PerturbConfig = runtime.perturb_cfg.clone();
    cfg.n_propagation = job.n_propagation;
    cfg.beta_scale_factor = job.beta_scale_factor;
    cfg.ligand_grid_factor = ligand_grid;
    cfg.contact_distance = contact;

    let spatial_override =
        job.radius.is_some() || job.ligand_grid_factor.is_some() || job.contact_distance.is_some();

    let rw_lr_store;
    let rw_tfl_store;
    let lr_store;
    let (rw_ligands_ref, rw_tfl_ref, lr_radii_ref): (
        &GeneMatrix,
        &GeneMatrix,
        &HashMap<String, f64>,
    ) = if spatial_override {
        let radius = job.radius.unwrap_or(runtime.cfg.spatial.radius);
        let mut lr_radii = HashMap::new();
        for lig in runtime
            .bb
            .ligands_set
            .iter()
            .chain(runtime.bb.tfl_ligands_set.iter())
        {
            lr_radii.insert(lig.clone(), radius);
        }
        lr_store = lr_radii;
        let lr_ligands: Vec<String> = runtime.bb.ligands_set.iter().cloned().collect();
        let tfl_ligands: Vec<String> = runtime.bb.tfl_ligands_set.iter().cloned().collect();
        rw_lr_store = compute_initial_weighted_ligands(ComputeInitialWeightedLigandsArgs {
            gene_mtx: &runtime.gene_mtx,
            gene_names: &runtime.gene_names,
            ligand_names: &lr_ligands,
            xy: &runtime.xy,
            lr_radii: &lr_store,
            weighted_ligand_scale: runtime.perturb_cfg.scale_factor,
            min_expression: runtime.perturb_cfg.min_expression,
            grid_factor: ligand_grid,
            contact_distance: contact,
        });
        rw_tfl_store = compute_initial_weighted_ligands(ComputeInitialWeightedLigandsArgs {
            gene_mtx: &runtime.gene_mtx,
            gene_names: &runtime.gene_names,
            ligand_names: &tfl_ligands,
            xy: &runtime.xy,
            lr_radii: &lr_store,
            weighted_ligand_scale: runtime.perturb_cfg.scale_factor,
            min_expression: runtime.perturb_cfg.min_expression,
            grid_factor: ligand_grid,
            contact_distance: contact,
        });
        (&rw_lr_store, &rw_tfl_store, &lr_store)
    } else {
        (
            &runtime.rw_ligands_init,
            &runtime.rw_tfligands_init,
            &runtime.lr_radii,
        )
    };

    let mut timings: Option<PerturbTimings> = if verbose {
        Some(PerturbTimings::default())
    } else {
        None
    };
    let baseline_cache = if spatial_override {
        None
    } else {
        Some(&runtime.baseline_splash_cache)
    };
    let result = perturb_with_targets(
        &PerturbWithTargetsInputs {
            bb: &runtime.bb,
            gene_mtx: &runtime.gene_mtx,
            gene_names: &runtime.gene_names,
            xy: &runtime.xy,
            rw_ligands_init: rw_ligands_ref,
            rw_tfligands_init: rw_tfl_ref,
            targets: &targets,
            config: &cfg,
            lr_radii: lr_radii_ref,
            job_progress: None,
            job_message: None,
            cancel: None,
            baseline_splash_cache: baseline_cache,
        },
        &mut timings,
    )
    .ok_or_else(|| anyhow::anyhow!("perturbation failed for gene {}", job.gene))?;
    let cell_scope = job.cell_indices.as_deref();
    validate_perturb_simulated_matrix(
        &runtime.gene_mtx,
        &runtime.gene_names,
        &result.simulated,
        &job.gene,
        job.desired_expr,
        cell_scope,
    )?;
    let p = job
        .out_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("output path must be UTF-8"))?;
    if let Some(parent) = job.out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_betadata_feather(
        p,
        "CellID",
        &runtime.obs_names,
        &runtime.gene_names,
        &result.simulated,
    )?;
    if verbose {
        eprintln!(
            "Wrote {} (gene={}, desired_expr={}, n_propagation={}, beta_scale_factor={})",
            job.out_path.display(),
            job.gene,
            job.desired_expr,
            job.n_propagation,
            job.beta_scale_factor
        );
    } else {
        eprintln!("Wrote {}", job.out_path.display());
    }
    Ok(())
}

pub fn run_batch_jobs(
    runtime: Arc<PerturbRuntime>,
    jobs: Vec<PreparedPerturbJob>,
    parallelism: usize,
    verbose: bool,
) -> anyhow::Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    let n_workers = parallelism.max(1).min(jobs.len());
    let queue = Arc::new(Mutex::new(VecDeque::from(jobs)));
    let mut handles = Vec::with_capacity(n_workers);
    for _ in 0..n_workers {
        let q = Arc::clone(&queue);
        let rt = Arc::clone(&runtime);
        handles.push(thread::spawn(move || -> anyhow::Result<()> {
            loop {
                let job = { q.lock().expect("batch queue poisoned").pop_front() };
                let Some(job) = job else {
                    break;
                };
                run_one_job(rt.as_ref(), job, verbose)?;
            }
            Ok(())
        }));
    }
    for h in handles {
        h.join()
            .map_err(|_| anyhow::anyhow!("batch perturb worker thread panicked"))??;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hoist_top_level_beta_scale_into_perturbation_section() {
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_hoist_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("perturb.toml");
        std::fs::write(
            &path,
            r#"
run_toml = "run.toml"
gene = "G"
beta_scale_factor = 42.0
"#,
        )
        .unwrap();
        let parsed = load_perturb_cli_toml(&path).unwrap();
        let pert = parsed
            .overlay_source
            .get("perturbation")
            .and_then(|v| v.as_table())
            .expect("perturbation table");
        assert_eq!(
            pert.get("beta_scale_factor").and_then(|v| v.as_float()),
            Some(42.0)
        );
        assert!(parsed.batch_table.as_ref().unwrap().contains_key("gene"));
        assert!(
            !parsed
                .batch_table
                .as_ref()
                .unwrap()
                .contains_key("beta_scale_factor")
        );
    }

    #[test]
    fn genes_scalar_string() {
        let s = r#"
genes = "SOX2"
out_dir = "o"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        let jobs = expand_prepared_jobs(&f, Path::new("/proj/root"), 5, 1.0).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].gene, "SOX2");
        assert_eq!(
            jobs[0].out_path,
            PathBuf::from("/proj/root/o/SOX2_perturb_expr.feather")
        );
    }

    #[test]
    fn normalize_gene_alias() {
        let s = r#"
gene = "SOX2"
out = "solo.feather"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        let jobs = expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].gene, "SOX2");
        assert_eq!(jobs[0].n_propagation, 4);
        assert!((jobs[0].desired_expr - 0.0).abs() < 1e-9);
        assert_eq!(jobs[0].out_path, PathBuf::from("/tmp/solo.feather"));
    }

    #[test]
    fn broadcast_desired_and_nprop() {
        let s = r#"
genes = ["SOX2", "PAX6"]
out_dir = "panel"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        let jobs = expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).unwrap();
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| (j.desired_expr - 0.0).abs() < 1e-9));
        assert!(jobs.iter().all(|j| j.n_propagation == 4));
        assert_eq!(
            jobs[0].out_path,
            PathBuf::from("/tmp/panel/SOX2_perturb_expr.feather")
        );
        assert_eq!(
            jobs[1].out_path,
            PathBuf::from("/tmp/panel/PAX6_perturb_expr.feather")
        );
    }

    #[test]
    fn zip_lists() {
        let s = r#"
genes = ["SOX2", "PAX6"]
desired_expr = [0.0, 0.5]
n_propagation = [2, 3]
beta_scale_factor = [1.0, 100.0]
out_dir = "panel"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        let jobs = expand_prepared_jobs(&f, Path::new("/tmp"), 99, 1.0).unwrap();
        assert_eq!(jobs[0].desired_expr, 0.0);
        assert_eq!(jobs[1].desired_expr, 0.5);
        assert_eq!(jobs[0].n_propagation, 2);
        assert_eq!(jobs[1].n_propagation, 3);
        assert!((jobs[0].beta_scale_factor - 1.0).abs() < 1e-9);
        assert!((jobs[1].beta_scale_factor - 100.0).abs() < 1e-9);
    }

    #[test]
    fn hoist_scalar_beta_scale_not_array() {
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_hoist_arr_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("perturb.toml");
        std::fs::write(
            &path,
            r#"
run_toml = "run.toml"
genes = ["G1", "G2"]
beta_scale_factor = [1.0, 2.0]
out_dir = "out"
"#,
        )
        .unwrap();
        let parsed = load_perturb_cli_toml(&path).unwrap();
        assert!(
            parsed
                .batch_table
                .as_ref()
                .unwrap()
                .contains_key("beta_scale_factor"),
            "per-gene array must stay in batch table"
        );
        assert!(
            parsed
                .overlay_source
                .get("perturbation")
                .and_then(|v| v.get("beta_scale_factor"))
                .is_none(),
            "array must not be hoisted into [perturbation]"
        );
        let batch = batch_from_perturb_table(parsed.batch_table.as_ref().unwrap()).unwrap();
        let jobs = expand_prepared_jobs(&batch, dir.as_path(), 2, 1.0).unwrap();
        assert!((jobs[0].beta_scale_factor - 1.0).abs() < 1e-9);
        assert!((jobs[1].beta_scale_factor - 2.0).abs() < 1e-9);
    }

    #[test]
    fn broadcast_len_one_arrays() {
        let s = r#"
genes = ["SOX2", "PAX6"]
desired_expr = [0.25]
n_propagation = [7]
out_dir = "panel"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        let jobs = expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).unwrap();
        assert!(jobs.iter().all(|j| (j.desired_expr - 0.25).abs() < 1e-9));
        assert!(jobs.iter().all(|j| j.n_propagation == 7));
    }

    #[test]
    fn out_array_paths() {
        let s = r#"
genes = ["SOX2", "PAX6"]
out = ["a.feather", "b.feather"]
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        let jobs = expand_prepared_jobs(&f, Path::new("/batch/parent"), 4, 1.0).unwrap();
        assert_eq!(jobs[0].out_path, PathBuf::from("/batch/parent/a.feather"));
        assert_eq!(jobs[1].out_path, PathBuf::from("/batch/parent/b.feather"));
    }

    #[test]
    fn errors_gene_and_genes() {
        let s = r#"
gene = "X"
genes = ["Y"]
out_dir = "o"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).is_err());
    }

    #[test]
    fn errors_scalar_out_multi_gene() {
        let s = r#"
genes = ["SOX2", "PAX6"]
out = "x.feather"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).is_err());
    }

    #[test]
    fn errors_out_out_dir_both() {
        let s = r#"
genes = ["SOX2", "PAX6"]
out = ["a.feather", "b.feather"]
out_dir = "q"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).is_err());
    }

    #[test]
    fn errors_mismatched_desired_len() {
        let s = r#"
genes = ["SOX2", "PAX6"]
desired_expr = [0.0, 0.5, 1.0]
out_dir = "panel"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).is_err());
    }

    #[test]
    fn errors_mismatched_beta_scale_len() {
        let s = r#"
genes = ["SOX2", "PAX6"]
beta_scale_factor = [1.0, 2.0, 3.0]
out_dir = "panel"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).is_err());
    }

    #[test]
    fn errors_missing_out_and_out_dir() {
        let s = r#"genes = ["SOX2", "PAX6"]"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4, 1.0).is_err());
    }

    #[test]
    fn parse_roundtrip_toml() {
        let s = r#"
genes = ["SOX2", "PAX6"]
desired_expr = [0.0, 1.0]
n_propagation = 3
out_dir = "out/panel"
parallelism = 2
"#;
        let file: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(matches!(
            file.genes,
            Some(GenesSpec::Many(ref v)) if v.len() == 2
        ));
        let jobs = expand_prepared_jobs(&file, Path::new("/proj"), 99, 1.0).unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(
            jobs[0].out_path,
            PathBuf::from("/proj/out/panel/SOX2_perturb_expr.feather")
        );
    }

    #[test]
    fn sanitize_gene_slash() {
        assert_eq!(default_feather_name("a/b"), "a_b_perturb_expr.feather");
    }

    #[test]
    fn effective_parallelism_prefers_cli() {
        assert_eq!(effective_parallelism(Some(3), Some(7)), 7);
        assert_eq!(
            effective_parallelism(None, None),
            default_worker_parallelism().max(1)
        );
    }

    #[test]
    fn cells_csv_columns_zip_and_empty_all_cells() {
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_batch_cells_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let csv_path = dir.join("lists.csv");
        std::fs::write(&csv_path, "col_a,col_b,col_c\nalpha,beta,gamma\n,,delta\n").unwrap();
        let obs = vec![
            "alpha".into(),
            "beta".into(),
            "gamma".into(),
            "delta".into(),
        ];
        let batch_toml = format!(
            r#"
genes = ["G1", "G2", "G3"]
out_dir = "out"
cells_csv = "{}"
cells_csv_columns = ["col_a", "col_b", ""]
"#,
            csv_path.display()
        );
        let f: PerturbBatchFile = toml::from_str(&batch_toml).unwrap();
        let parent = dir.as_path();
        let mut jobs = expand_prepared_jobs(&f, parent, 4, 1.0).unwrap();
        assert_eq!(jobs.len(), 3);
        resolve_prepared_job_cell_indices(&f, parent, &obs, &mut jobs).unwrap();
        assert_eq!(jobs[0].cell_indices.as_ref().unwrap().as_slice(), &[0usize]);
        assert_eq!(jobs[1].cell_indices.as_ref().unwrap().as_slice(), &[1usize]);
        assert!(jobs[2].cell_indices.is_none());
    }

    #[test]
    fn cells_csv_columns_broadcast_one() {
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_batch_cells2_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let csv_path = dir.join("x.csv");
        std::fs::write(&csv_path, "only\nbeta\n").unwrap();
        let obs = vec!["alpha".into(), "beta".into()];
        let batch_toml = format!(
            r#"
genes = ["A", "B"]
out_dir = "o"
cells_csv = "{}"
cells_csv_columns = ["only"]
"#,
            csv_path.display()
        );
        let f: PerturbBatchFile = toml::from_str(&batch_toml).unwrap();
        let parent = dir.as_path();
        let mut jobs = expand_prepared_jobs(&f, parent, 4, 1.0).unwrap();
        resolve_prepared_job_cell_indices(&f, parent, &obs, &mut jobs).unwrap();
        assert_eq!(jobs[0].cell_indices, jobs[1].cell_indices);
        assert_eq!(jobs[0].cell_indices.as_ref().unwrap().as_slice(), &[1usize]);
    }

    #[test]
    fn errors_cells_csv_column_and_columns_together() {
        let dir = std::env::temp_dir().join(format!("spacetravlr_batch_e_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let csv_path = dir.join("c.csv");
        std::fs::write(&csv_path, "a\nalpha\n").unwrap();
        let s = format!(
            r#"
genes = ["A"]
out_dir = "o"
cells_csv = "{}"
cells_csv_column = "a"
cells_csv_columns = ["a"]
"#,
            csv_path.display()
        );
        let f: PerturbBatchFile = toml::from_str(&s).unwrap();
        let obs = vec!["alpha".into()];
        let mut jobs = expand_prepared_jobs(&f, dir.as_path(), 4, 1.0).unwrap();
        assert!(resolve_prepared_job_cell_indices(&f, dir.as_path(), &obs, &mut jobs).is_err());
    }
}
