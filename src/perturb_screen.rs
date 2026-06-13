use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::perturb_batch::{
    PerturbBatchFile, batch_from_perturb_table, effective_parallelism, resolve_prepared_job_cell_indices,
    resolve_relative_to, run_batch_jobs, sanitize_gene_for_filename, validate_jobs_genes,
    PreparedPerturbJob,
};
use crate::perturb_mode::PerturbRuntime;

pub const SCREEN_PERTURBATIONS_SUBDIR: &str = "perturbations";

fn screen_feather_name(gene: &str) -> String {
    format!("{}_KO.feather", sanitize_gene_for_filename(gene))
}

/// Union of TFs, LR ligands/receptors, TFL ligands, `[grn].extra_modulators`, and genes from `extra_lr`.
pub fn collect_screen_genes(runtime: &PerturbRuntime) -> anyhow::Result<Vec<String>> {
    let bb = &runtime.bb;
    let mut genes: HashSet<String> = HashSet::new();
    genes.extend(bb.tfs_set.iter().cloned());
    genes.extend(bb.ligands_set.iter().cloned());
    genes.extend(bb.receptors_set.iter().cloned());
    genes.extend(bb.tfl_ligands_set.iter().cloned());

    let run_parent = runtime
        .run_toml_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (extra_modulators, extra_lr) = runtime
        .cfg
        .grn
        .resolve_extra_modulators_and_lr(Some(run_parent))?;
    genes.extend(extra_modulators);
    for (lig, rec) in extra_lr {
        genes.insert(lig);
        genes.insert(rec);
    }

    let var: HashSet<&str> = runtime.gene_names.iter().map(String::as_str).collect();
    let mut out: Vec<String> = genes
        .into_iter()
        .filter(|g| var.contains(g.as_str()))
        .collect();
    out.sort();
    Ok(out)
}

pub fn resolve_screen_output_root(
    batch_file: &PerturbBatchFile,
    batch_parent: &Path,
    run_dir: &Path,
) -> PathBuf {
    batch_file
        .out_dir
        .as_ref()
        .map(|d| resolve_relative_to(batch_parent, d))
        .unwrap_or_else(|| run_dir.to_path_buf())
}

pub fn expand_screen_jobs(
    genes: &[String],
    perturbations_dir: &Path,
    batch_file: &PerturbBatchFile,
    default_n_propagation: usize,
    default_beta_scale_factor: f64,
) -> anyhow::Result<Vec<PreparedPerturbJob>> {
    if genes.is_empty() {
        anyhow::bail!("screen: no perturbable modulator genes found in AnnData var_names");
    }

    let n = genes.len();
    let desired = crate::perturb_batch::broadcast_f64_field(
        batch_file.desired_expr.as_ref(),
        n,
        0.0,
        "desired_expr",
    )?;
    let n_props = crate::perturb_batch::broadcast_usize(
        batch_file.n_propagation.as_ref(),
        n,
        default_n_propagation,
    )?;
    let beta_scales = crate::perturb_batch::broadcast_f64_field(
        batch_file.beta_scale_factor.as_ref(),
        n,
        default_beta_scale_factor,
        "beta_scale_factor",
    )?;

    let radius = batch_file.radius;
    let ligand_grid_factor = batch_file.ligand_grid_factor;
    let contact_distance = batch_file.contact_distance;

    Ok(genes
        .iter()
        .enumerate()
        .map(|(i, gene)| PreparedPerturbJob {
            gene: gene.clone(),
            desired_expr: desired[i],
            n_propagation: n_props[i],
            out_path: perturbations_dir.join(screen_feather_name(gene)),
            radius,
            ligand_grid_factor,
            contact_distance,
            beta_scale_factor: beta_scales[i],
            cell_indices: None,
        })
        .collect())
}

pub struct RunPerturbScreenArgs<'a> {
    pub run_toml: PathBuf,
    pub config_path: &'a Path,
    pub overlay: Option<&'a toml::Value>,
    pub n_propagation_cli: Option<usize>,
    pub parallelism_cli: Option<usize>,
    pub verbose: bool,
}

pub fn run_perturb_screen(args: RunPerturbScreenArgs<'_>) -> anyhow::Result<()> {
    let RunPerturbScreenArgs {
        run_toml,
        config_path,
        overlay,
        n_propagation_cli,
        parallelism_cli,
        verbose,
    } = args;

    let parsed = crate::perturb_batch::load_perturb_cli_toml(config_path)?;
    let batch_file = match parsed.batch_table.as_ref() {
        Some(tbl) => batch_from_perturb_table(tbl)?,
        None => PerturbBatchFile::default(),
    };
    let batch_parent = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let t_load = Instant::now();
    let mut runtime =
        PerturbRuntime::from_run_toml_with_config_overlay(run_toml.as_path(), overlay)?;
    let load_elapsed = t_load.elapsed();

    if let Some(n) = n_propagation_cli {
        runtime.perturb_cfg.n_propagation = n;
    }
    let default_n_prop = runtime.perturb_cfg.n_propagation;
    let default_beta_scale = runtime.perturb_cfg.beta_scale_factor;

    let mut batch_file = batch_file;
    if batch_file
        .cells_csv
        .as_ref()
        .map(|s| s.trim().is_empty())
        .unwrap_or(true)
        && batch_file.cells_obs_file.is_none()
    {
        if let Some(ref rel) = runtime.cfg.perturbation.cells_csv {
            if !rel.trim().is_empty() {
                let run_parent = run_toml
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."));
                let exp = crate::config::expand_user_path(rel.trim());
                let pb = Path::new(&exp);
                let resolved = if pb.is_absolute() {
                    pb.to_path_buf()
                } else {
                    run_parent.join(pb)
                };
                batch_file.cells_csv = Some(resolved.to_string_lossy().into_owned());
                if batch_file.cells_csv_column.is_none() {
                    batch_file.cells_csv_column =
                        runtime.cfg.perturbation.cells_csv_column.clone();
                }
            }
        }
    }

    let genes = collect_screen_genes(&runtime)?;
    if verbose {
        eprintln!(
            "screen: {} modulator genes to KO (TF / ligand / receptor / extras)",
            genes.len()
        );
    }

    let output_root = resolve_screen_output_root(&batch_file, batch_parent, &runtime.run_dir);
    let perturbations_dir = output_root.join(SCREEN_PERTURBATIONS_SUBDIR);
    std::fs::create_dir_all(&perturbations_dir)?;

    let mut jobs = expand_screen_jobs(
        &genes,
        &perturbations_dir,
        &batch_file,
        default_n_prop,
        default_beta_scale,
    )?;
    validate_jobs_genes(&jobs, &runtime.gene_names)?;
    resolve_prepared_job_cell_indices(&batch_file, batch_parent, &runtime.obs_names, &mut jobs)?;

    let parallelism = effective_parallelism(batch_file.parallelism, parallelism_cli);
    let rt = Arc::new(runtime);
    let t_batch = Instant::now();
    run_batch_jobs(Arc::clone(&rt), jobs, parallelism, verbose)?;
    let batch_elapsed = t_batch.elapsed();

    eprintln!(
        "Wrote {} KO feathers under {}",
        genes.len(),
        perturbations_dir.display()
    );
    if verbose {
        eprintln!("--- spacetravlr-perturb screen timings ---");
        eprintln!("  load_runtime: {load_elapsed:?}");
        eprintln!("  screen_batch_total: {batch_elapsed:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betadata::Betabase;
    use crate::config::SpaceshipConfig;
    use std::collections::HashMap;

    fn minimal_runtime(
        gene_names: &[&str],
        tfs: &[&str],
        ligands: &[&str],
        receptors: &[&str],
        extra_modulators: &[&str],
        extra_lr: &[&str],
    ) -> PerturbRuntime {
        let bb = Betabase {
            data: HashMap::new(),
            ligands_set: ligands.iter().map(|s| s.to_string()).collect(),
            receptors_set: receptors.iter().map(|s| s.to_string()).collect(),
            tfl_ligands_set: HashSet::new(),
            tfs_set: tfs.iter().map(|s| s.to_string()).collect(),
        };

        let mut cfg = SpaceshipConfig::default();
        cfg.grn.extra_modulators = extra_modulators.iter().map(|s| s.to_string()).collect();
        cfg.grn.extra_lr = extra_lr.iter().map(|s| s.to_string()).collect();

        PerturbRuntime {
            run_toml_path: PathBuf::from("/tmp/spacetravlr_run_repro.toml"),
            run_dir: PathBuf::from("/tmp/run"),
            cfg,
            gene_mtx: ndarray::Array2::zeros((1, gene_names.len())),
            gene_names: gene_names.iter().map(|s| s.to_string()).collect(),
            obs_names: vec!["c0".into()],
            betadata_cluster_key: vec!["k0".into()],
            expression_preview_labels: vec!["k0".into()],
            cell_types: vec![0],
            bb,
            xy: ndarray::Array2::zeros((1, 2)),
            rw_ligands_init: crate::betadata::GeneMatrix::new(
                ndarray::Array2::<f32>::zeros((1, 0)),
                vec![],
            ),
            rw_tfligands_init: crate::betadata::GeneMatrix::new(
                ndarray::Array2::<f32>::zeros((1, 0)),
                vec![],
            ),
            lr_radii: HashMap::new(),
            perturb_cfg: crate::perturb::PerturbConfig::default(),
            baseline_splash_cache: std::sync::Mutex::new(None),
        }
    }

    #[test]
    fn collect_screen_genes_unions_modulator_families() {
        let rt = minimal_runtime(
            &["TF1", "L1", "R1", "X1", "MISSING"],
            &["TF1"],
            &["L1"],
            &["R1"],
            &["X1"],
            &["L2$R2"],
        );
        let genes = collect_screen_genes(&rt).unwrap();
        assert_eq!(genes, vec!["L1", "R1", "TF1", "X1"]);
    }

    #[test]
    fn expand_screen_jobs_ko_feather_names() {
        let batch: PerturbBatchFile = toml::from_str("parallelism = 2").unwrap();
        let dir = Path::new("/out/root/perturbations");
        let jobs = expand_screen_jobs(&["SOX2".into(), "A/B".into()], dir, &batch, 4, 1.0).unwrap();
        assert_eq!(jobs.len(), 2);
        assert!((jobs[0].desired_expr).abs() < 1e-9);
        assert_eq!(jobs[0].n_propagation, 4);
        assert_eq!(
            jobs[0].out_path,
            PathBuf::from("/out/root/perturbations/SOX2_KO.feather")
        );
        assert_eq!(
            jobs[1].out_path,
            PathBuf::from("/out/root/perturbations/A_B_KO.feather")
        );
    }

    #[test]
    fn resolve_screen_output_root_prefers_out_dir() {
        let batch: PerturbBatchFile = toml::from_str(r#"out_dir = "panel""#).unwrap();
        let root = resolve_screen_output_root(&batch, Path::new("/cfg"), Path::new("/run"));
        assert_eq!(root, PathBuf::from("/cfg/panel"));
    }
}
