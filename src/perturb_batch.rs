use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::Deserialize;

use crate::betadata::{GeneMatrix, write_betadata_feather};
use crate::perturb::{PerturbConfig, PerturbTarget, PerturbTimings, perturb_with_targets};
use crate::perturb_mode::{
    PerturbRuntime, compute_initial_weighted_ligands, parse_obs_columns_csv,
    perturb_obs_indices_from_file,
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
    match spec {
        None => Ok(vec![0.0f64; n]),
        Some(F64OrVec::One(x)) => Ok(vec![*x; n]),
        Some(F64OrVec::Many(v)) => {
            if v.len() == n {
                Ok(v.clone())
            } else if v.len() == 1 {
                Ok(vec![v[0]; n])
            } else {
                anyhow::bail!(
                    "batch TOML: `desired_expr` length {} must be 1, {}, or omit (default 0)",
                    v.len(),
                    n
                )
            }
        }
    }
}

fn broadcast_usize(
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
) -> anyhow::Result<Vec<PreparedPerturbJob>> {
    let genes = normalize_gene_list(file)?;
    let n = genes.len();
    let desired = broadcast_f64(file.desired_expr.as_ref(), n)?;
    let n_props = broadcast_usize(file.n_propagation.as_ref(), n, default_n_propagation)?;

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
        })
        .collect())
}

pub fn resolve_batch_cell_indices(
    file: &PerturbBatchFile,
    batch_parent: &Path,
    obs_names: &[String],
) -> anyhow::Result<Option<Vec<usize>>> {
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
        (Some(_), None) => {
            let col = file.cells_csv_column.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "batch TOML: `cells_csv_column` is required when `cells_csv` is set"
                )
            })?;
            let path = resolve_relative_to(batch_parent, csv.unwrap());
            let parsed = parse_obs_columns_csv(&path, obs_names)?;
            let sl = parsed.indices_for_column(col).ok_or_else(|| {
                anyhow::anyhow!(
                    "batch TOML: cells_csv column {:?} not found in CSV header",
                    col
                )
            })?;
            Ok(Some(sl.to_vec()))
        }
        (None, Some(p)) => {
            let path = resolve_relative_to(batch_parent, p);
            Ok(Some(perturb_obs_indices_from_file(&path, obs_names)?))
        }
        (None, None) => Ok(None),
    }
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
    cell_indices: Option<Vec<usize>>,
    verbose: bool,
) -> anyhow::Result<()> {
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
    cfg.ligand_grid_factor = ligand_grid;
    cfg.contact_distance = contact;

    let spatial_override = job.radius.is_some()
        || job.ligand_grid_factor.is_some()
        || job.contact_distance.is_some();

    let rw_lr_store;
    let rw_tfl_store;
    let lr_store;
    let (rw_ligands_ref, rw_tfl_ref, lr_radii_ref): (&GeneMatrix, &GeneMatrix, &HashMap<String, f64>) =
        if spatial_override {
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
            rw_lr_store = compute_initial_weighted_ligands(
                &runtime.gene_mtx,
                &runtime.gene_names,
                &lr_ligands,
                &runtime.xy,
                &lr_store,
                runtime.perturb_cfg.scale_factor,
                runtime.perturb_cfg.min_expression,
                ligand_grid,
                contact,
            );
            rw_tfl_store = compute_initial_weighted_ligands(
                &runtime.gene_mtx,
                &runtime.gene_names,
                &tfl_ligands,
                &runtime.xy,
                &lr_store,
                runtime.perturb_cfg.scale_factor,
                runtime.perturb_cfg.min_expression,
                ligand_grid,
                contact,
            );
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
        &runtime.bb,
        &runtime.gene_mtx,
        &runtime.gene_names,
        &runtime.xy,
        rw_ligands_ref,
        rw_tfl_ref,
        &targets,
        &cfg,
        lr_radii_ref,
        None,
        None,
        None,
        baseline_cache,
        &mut timings,
    )
    .map_err(|_| anyhow::anyhow!("perturbation failed for gene {}", job.gene))?;
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
            "Wrote {} (gene={}, desired_expr={}, n_propagation={})",
            job.out_path.display(),
            job.gene,
            job.desired_expr,
            job.n_propagation
        );
    } else {
        eprintln!("Wrote {}", job.out_path.display());
    }
    Ok(())
}

pub fn run_batch_jobs(
    runtime: Arc<PerturbRuntime>,
    jobs: Vec<PreparedPerturbJob>,
    cell_indices: Option<Vec<usize>>,
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
        let cells = cell_indices.clone();
        handles.push(thread::spawn(move || -> anyhow::Result<()> {
            loop {
                let job = { q.lock().expect("batch queue poisoned").pop_front() };
                let Some(job) = job else {
                    break;
                };
                run_one_job(rt.as_ref(), job, cells.clone(), verbose)?;
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
    fn genes_scalar_string() {
        let s = r#"
genes = "SOX2"
out_dir = "o"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        let jobs = expand_prepared_jobs(&f, Path::new("/proj/root"), 5).unwrap();
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
        let jobs = expand_prepared_jobs(&f, Path::new("/tmp"), 4).unwrap();
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
        let jobs = expand_prepared_jobs(&f, Path::new("/tmp"), 4).unwrap();
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
out_dir = "panel"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        let jobs = expand_prepared_jobs(&f, Path::new("/tmp"), 99).unwrap();
        assert_eq!(jobs[0].desired_expr, 0.0);
        assert_eq!(jobs[1].desired_expr, 0.5);
        assert_eq!(jobs[0].n_propagation, 2);
        assert_eq!(jobs[1].n_propagation, 3);
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
        let jobs = expand_prepared_jobs(&f, Path::new("/tmp"), 4).unwrap();
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
        let jobs = expand_prepared_jobs(&f, Path::new("/batch/parent"), 4).unwrap();
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
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4).is_err());
    }

    #[test]
    fn errors_scalar_out_multi_gene() {
        let s = r#"
genes = ["SOX2", "PAX6"]
out = "x.feather"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4).is_err());
    }

    #[test]
    fn errors_out_out_dir_both() {
        let s = r#"
genes = ["SOX2", "PAX6"]
out = ["a.feather", "b.feather"]
out_dir = "q"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4).is_err());
    }

    #[test]
    fn errors_mismatched_desired_len() {
        let s = r#"
genes = ["SOX2", "PAX6"]
desired_expr = [0.0, 0.5, 1.0]
out_dir = "panel"
"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4).is_err());
    }

    #[test]
    fn errors_missing_out_and_out_dir() {
        let s = r#"genes = ["SOX2", "PAX6"]"#;
        let f: PerturbBatchFile = toml::from_str(s).unwrap();
        assert!(expand_prepared_jobs(&f, Path::new("/tmp"), 4).is_err());
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
        let jobs = expand_prepared_jobs(&file, Path::new("/proj"), 99).unwrap();
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
}
