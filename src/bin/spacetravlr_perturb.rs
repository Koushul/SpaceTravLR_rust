use clap::Parser;
use spacetravlr::betadata::write_betadata_feather;
use spacetravlr::config::expand_user_path;
use spacetravlr::perturb::{
    PerturbTarget, PerturbTimings, PerturbWithTargetsInputs, perturb_with_targets,
};
use spacetravlr::perturb_batch::{
    PerturbBatchFile, batch_from_perturb_table, effective_parallelism, expand_prepared_jobs,
    load_batch_file, load_perturb_cli_toml, resolve_effective_run_toml,
    resolve_prepared_job_cell_indices, run_batch_jobs, validate_jobs_genes,
};
use spacetravlr::perturb_mode::{
    PerturbRuntime, parse_obs_columns_csv, validate_perturb_simulated_matrix,
};
#[cfg(not(feature = "tui"))]
use spacetravlr::perturb_mode::{interactive_run_toml_prompt, run_interactive};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "spacetravlr-perturb",
    version,
    about = "SpaceTravLR perturbation: Ratatui UI (default) or --export/--out batch mode or --batch-toml. Same run TOML + betadata loading model as spatial_viewer.",
    after_long_help = r#"Use --config PATH (or pass PATH as the first argument) for a single TOML: `run_toml` (path to spacetravlr_run_repro.toml) plus optional `[data]` / `[perturbation]` / … sections that override the repro file. `--run-toml` overrides `run_toml` in that file when both are set.

Batch mode (fully non-interactive) uses --export PATH or --out PATH (same flag), or --batch-toml PATH for many single-gene jobs. Batch keys can live in --config instead of a separate batch file.

Single-job batch: requires a repro TOML (--run-toml or run_toml in --config), --gene, --export/--out. Optional: --desired-expr (default 0), --n-propagation, --cells-csv + --cells-csv-column, --verbose.

Batch TOML: repro path + --batch-toml (gene lists, zips, out_dir or out; parallelism inside the file or --batch-parallelism). Do not combine with --gene, --export/--out, or --cells-*.

Example:
  spacetravlr-perturb \
    --run-toml /path/to/spacetravlr_run_repro.toml \
    --out /tmp/simulated.feather \
    --gene SOX2 \
    --desired-expr 0 \
    --n-propagation 4 \
    --cells-csv /path/to/cells.csv \
    --cells-csv-column selected \
    --verbose

If --cells-csv is set, --cells-csv-column is required in single-job batch mode."#
)]
struct Cli {
    #[arg(
        long = "config",
        visible_alias = "perturb-toml",
        value_name = "PATH",
        help = "Perturbation TOML: `run_toml`, optional section overrides vs. repro, optional batch/job fields. See --help long help."
    )]
    config: Option<PathBuf>,

    #[arg(index = 1, value_name = "CONFIG", help = "Same as --config.")]
    config_positional: Option<PathBuf>,

    #[arg(
        long = "run-toml",
        value_name = "PATH",
        help = "Path to spacetravlr_run_repro.toml. Overrides run_toml in --config. If omitted and not in --config: TUI prompts; without TUI, stdin prompt."
    )]
    run_toml: Option<PathBuf>,

    #[arg(
        long = "export",
        visible_alias = "out",
        value_name = "PATH",
        help = "Write simulated expression as feather (rows = cells, columns = CellID + genes); exit. Same as --out. Requires --gene and a repro path (--run-toml and/or run_toml in --config); not for multi-job batch."
    )]
    export: Option<PathBuf>,

    #[arg(
        long = "gene",
        help = "Gene to perturb (single-job --export / --out). For multi-job output, use batch keys in --config or --batch-toml."
    )]
    gene: Option<String>,

    #[arg(
        long = "desired-expr",
        default_value_t = 0.0,
        help = "Batch: target expression. TUI: initial desired_expr."
    )]
    desired_expr: f64,

    #[arg(
        long = "n-propagation",
        help = "Override [perturbation].n_propagation from the TOML (batch or TUI initial value)."
    )]
    n_propagation: Option<usize>,

    #[arg(
        long,
        help = "Batch: print load and perturb timings (stderr). TUI: start with per-step timings (toggle Ctrl+V)."
    )]
    verbose: bool,

    #[arg(
        long = "cells-csv",
        value_name = "PATH",
        help = "Optional CSV (header row); each column lists obs_names from AnnData. TUI: pick column with Ctrl+O. Batch: if set, requires --cells-csv-column."
    )]
    cells_csv: Option<PathBuf>,

    #[arg(
        long = "cells-csv-column",
        value_name = "NAME",
        help = "Column name in --cells-csv (required in single-job batch mode when --cells-csv is set)."
    )]
    cells_csv_column: Option<String>,

    #[arg(
        long = "batch-toml",
        value_name = "PATH",
        help = "Batch perturbation spec (TOML): multiple single-gene runs with shared runtime load. Needs repro path (--run-toml or run_toml in --config). Incompatible with batch keys inside --config, --gene, --export/--out, --cells-csv/--cells-csv-column."
    )]
    batch_toml: Option<PathBuf>,

    #[arg(
        long = "batch-parallelism",
        value_name = "N",
        help = "Override batch TOML parallelism (max concurrent perturb threads)."
    )]
    batch_parallelism: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    spacetravlr::ensure_hdf5_no_file_locking();
    let cli = Cli::parse();

    let perturb_config_path = cli.config.clone().or(cli.config_positional.clone());
    let parsed_opt = if let Some(ref p) = perturb_config_path {
        Some(load_perturb_cli_toml(p)?)
    } else {
        None
    };
    let overlay_ref: Option<&toml::Value> = parsed_opt.as_ref().map(|p| &p.overlay_source);

    let run_batch_branch = |batch_file: PerturbBatchFile,
                            batch_parent: &Path,
                            run_toml_eff: PathBuf|
     -> anyhow::Result<()> {
        if cli.export.is_some() {
            anyhow::bail!(
                "--export/--out cannot be used in multi-job batch mode (outputs come from the batch spec)."
            );
        }
        if cli.gene.is_some() {
            anyhow::bail!("--gene cannot be used in batch mode.");
        }
        if cli.cells_csv.is_some() || cli.cells_csv_column.is_some() {
            anyhow::bail!(
                "--cells-csv / --cells-csv-column cannot be used in batch mode (set cells in the batch TOML if needed)."
            );
        }
        let t_load = Instant::now();
        let mut runtime =
            PerturbRuntime::from_run_toml_with_config_overlay(run_toml_eff.as_path(), overlay_ref)?;
        let load_elapsed = t_load.elapsed();
        let default_n_prop = if let Some(n) = cli.n_propagation {
            runtime.perturb_cfg.n_propagation = n;
            n
        } else {
            runtime.perturb_cfg.n_propagation
        };
        let default_beta_scale = runtime.perturb_cfg.beta_scale_factor;

        let mut jobs = expand_prepared_jobs(
            &batch_file,
            batch_parent,
            default_n_prop,
            default_beta_scale,
        )?;
        validate_jobs_genes(&jobs, &runtime.gene_names)?;
        resolve_prepared_job_cell_indices(
            &batch_file,
            batch_parent,
            &runtime.obs_names,
            &mut jobs,
        )?;

        let parallelism = effective_parallelism(batch_file.parallelism, cli.batch_parallelism);
        let rt = Arc::new(runtime);
        let t_batch = Instant::now();
        run_batch_jobs(Arc::clone(&rt), jobs, parallelism, cli.verbose)?;
        let batch_elapsed = t_batch.elapsed();
        if cli.verbose {
            eprintln!("--- spacetravlr-perturb batch timings ---");
            eprintln!("  load_runtime: {load_elapsed:?}");
            eprintln!("  perturb_batch_total: {batch_elapsed:?}");
        }
        Ok(())
    };

    if let Some(batch_path) = cli.batch_toml.as_ref() {
        if parsed_opt
            .as_ref()
            .and_then(|p| p.batch_table.as_ref())
            .is_some()
        {
            anyhow::bail!(
                "do not combine --batch-toml with batch/job keys (gene, out, …) inside --config"
            );
        }
        let run_toml_eff = resolve_effective_run_toml(
            cli.run_toml.clone(),
            parsed_opt.as_ref().and_then(|p| p.run_toml.clone()),
            perturb_config_path.as_deref(),
        )?;
        let batch_file = load_batch_file(batch_path.as_path())?;
        let batch_parent = batch_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        return run_batch_branch(batch_file, batch_parent, run_toml_eff);
    }

    if let Some(tbl) = parsed_opt.as_ref().and_then(|p| p.batch_table.as_ref()) {
        let run_toml_eff = resolve_effective_run_toml(
            cli.run_toml.clone(),
            parsed_opt.as_ref().and_then(|p| p.run_toml.clone()),
            perturb_config_path.as_deref(),
        )?;
        let batch_file = batch_from_perturb_table(tbl)?;
        let batch_parent = perturb_config_path
            .as_ref()
            .and_then(|p| p.parent())
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        return run_batch_branch(batch_file, batch_parent, run_toml_eff);
    }

    let run_for_interactive = cli
        .run_toml
        .clone()
        .or_else(|| parsed_opt.as_ref().and_then(|p| p.run_toml.clone()));

    if cli.export.is_none() {
        #[cfg(feature = "tui")]
        {
            let opts = spacetravlr::perturb_tui::PerturbTuiOptions {
                run_toml: run_for_interactive.clone(),
                default_desired_expr: cli.desired_expr,
                n_propagation_initial: cli.n_propagation,
                verbose: cli.verbose,
                toml_path_hint_for_error: run_for_interactive
                    .as_ref()
                    .map(|p| p.display().to_string()),
                cells_csv: cli.cells_csv.clone(),
                cells_csv_column: cli.cells_csv_column.clone(),
                config_merge_overlay: parsed_opt.as_ref().map(|p| p.overlay_source.clone()),
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            return rt.block_on(spacetravlr::perturb_tui::run(opts));
        }
        #[cfg(not(feature = "tui"))]
        {
            let run_toml = match run_for_interactive {
                Some(p) => p,
                None => interactive_run_toml_prompt()?,
            };
            let mut runtime =
                PerturbRuntime::from_run_toml_with_config_overlay(run_toml.as_path(), overlay_ref)?;
            if let Some(n) = cli.n_propagation {
                runtime.perturb_cfg.n_propagation = n;
            }
            return run_interactive(runtime);
        }
    }

    let run_toml = resolve_effective_run_toml(
        cli.run_toml.clone(),
        parsed_opt.as_ref().and_then(|p| p.run_toml.clone()),
        perturb_config_path.as_deref(),
    )?;

    let export_path = cli.export.as_ref().unwrap();
    let gene = cli
        .gene
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--gene is required with --export / --out"))?;
    let t_load = Instant::now();
    let mut runtime =
        PerturbRuntime::from_run_toml_with_config_overlay(run_toml.as_path(), overlay_ref)?;
    let load_elapsed = t_load.elapsed();
    if let Some(n) = cli.n_propagation {
        runtime.perturb_cfg.n_propagation = n;
    }
    if !runtime.gene_names.iter().any(|g| g == gene) {
        anyhow::bail!("Gene '{}' is not present in AnnData var_names.", gene);
    }

    let run_parent = run_toml
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let mut csv_path = cli.cells_csv.clone();
    let mut csv_column = cli.cells_csv_column.clone();
    if csv_path.is_none() && csv_column.is_none() {
        if let Some(ref rel) = runtime.cfg.perturbation.cells_csv {
            if !rel.trim().is_empty() {
                let exp = expand_user_path(rel.trim());
                let pb = Path::new(&exp);
                csv_path = Some(if pb.is_absolute() {
                    pb.to_path_buf()
                } else {
                    run_parent.join(pb)
                });
                csv_column = runtime.cfg.perturbation.cells_csv_column.clone();
            }
        }
    } else if csv_path.is_some() && csv_column.is_none() {
        csv_column = runtime.cfg.perturbation.cells_csv_column.clone();
    }

    if csv_path.is_some() && csv_column.is_none() {
        anyhow::bail!(
            "cells CSV column missing: use --cells-csv-column or set [perturbation].cells_csv_column in the run TOML"
        );
    }

    let cell_indices_batch = match (&csv_path, &csv_column) {
        (Some(csv_path), Some(col)) => {
            let parsed = parse_obs_columns_csv(csv_path.as_path(), &runtime.obs_names)?;
            let sl = parsed.indices_for_column(col.as_str()).ok_or_else(|| {
                anyhow::anyhow!("cells_csv column {:?} not found in CSV header", col)
            })?;
            Some(sl.to_vec())
        }
        (None, None) => None,
        _ => anyhow::bail!("internal: inconsistent cells CSV path / column state"),
    };

    let targets = vec![PerturbTarget {
        gene: gene.to_string(),
        desired_expr: cli.desired_expr,
        cell_indices: cell_indices_batch,
    }];
    let mut timings: Option<PerturbTimings> = if cli.verbose {
        Some(PerturbTimings::default())
    } else {
        None
    };
    let t_perturb = Instant::now();
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
        &mut timings,
    )
    .ok_or_else(|| anyhow::anyhow!("perturbation failed"))?;
    validate_perturb_simulated_matrix(
        &runtime.gene_mtx,
        &runtime.gene_names,
        &result.simulated,
        gene,
        cli.desired_expr,
        targets[0].cell_indices.as_deref(),
    )?;
    let perturb_elapsed = t_perturb.elapsed();
    let p = export_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("export path must be UTF-8"))?;
    write_betadata_feather(
        p,
        "CellID",
        &runtime.obs_names,
        &runtime.gene_names,
        &result.simulated,
    )?;
    eprintln!(
        "Wrote {} ({} cells × {} genes, n_propagation={})",
        export_path.display(),
        runtime.obs_names.len(),
        runtime.gene_names.len(),
        runtime.perturb_cfg.n_propagation
    );
    if cli.verbose {
        eprintln!("--- spacetravlr-perturb timings ---");
        eprintln!("  load_runtime (PerturbRuntime::from_run_toml): {load_elapsed:?}");
        eprintln!("  perturb_total (perturb_with_targets): {perturb_elapsed:?}");
        if let Some(t) = timings.as_ref() {
            eprintln!("  per-step (within propagation loop):");
            for (label, d) in &t.entries {
                eprintln!("    {label}: {d:?}");
            }
            let sum_suffix = |suf: &str| -> Duration {
                t.entries
                    .iter()
                    .filter(|(k, _)| k.ends_with(suf))
                    .map(|(_, d)| *d)
                    .sum()
            };
            eprintln!("  sums over iterations:");
            eprintln!("    splash: {:?}", sum_suffix("/splash"));
            eprintln!(
                "    weighted_ligands_lr: {:?}",
                sum_suffix("/weighted_ligands_lr")
            );
            eprintln!(
                "    weighted_ligands_tfl: {:?}",
                sum_suffix("/weighted_ligands_tfl")
            );
            eprintln!("    grn_propagate: {:?}", sum_suffix("/grn_propagate"));
            eprintln!("    pin_nonneg: {:?}", sum_suffix("/pin_nonneg"));
        }
    }
    Ok(())
}
