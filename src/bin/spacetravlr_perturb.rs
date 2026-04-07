use clap::Parser;
use space_trav_lr_rust::betadata::write_betadata_feather;
use space_trav_lr_rust::perturb::{PerturbTarget, PerturbTimings, perturb_with_targets};
use space_trav_lr_rust::perturb_batch::{
    effective_parallelism, expand_prepared_jobs, load_batch_file, resolve_batch_cell_indices,
    run_batch_jobs, validate_jobs_genes,
};
use space_trav_lr_rust::perturb_mode::{PerturbRuntime, parse_obs_columns_csv};
#[cfg(not(feature = "tui"))]
use space_trav_lr_rust::perturb_mode::{interactive_run_toml_prompt, run_interactive};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "spacetravlr-perturb",
    version,
    about = "SpaceTravLR perturbation: Ratatui UI (default) or --export/--out batch mode or --batch-toml. Same run TOML + betadata loading model as spatial_viewer.",
    after_long_help = r#"Batch mode (fully non-interactive) uses --export PATH or --out PATH (same flag), or --batch-toml PATH for many single-gene jobs.

Single-job batch: requires --run-toml, --gene, --export/--out. Optional: --desired-expr (default 0), --n-propagation, --cells-csv + --cells-csv-column, --verbose.

Batch TOML: requires --run-toml and --batch-toml (gene lists, zips, out_dir or out; parallelism inside the file or --batch-parallelism). Do not combine with --gene, --export/--out, or --cells-*.

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
        long = "run-toml",
        value_name = "PATH",
        help = "Path to spacetravlr_run_repro.toml. If omitted: TUI prompts for a path; without TUI feature, stdin prompt is used."
    )]
    run_toml: Option<PathBuf>,

    #[arg(
        long = "export",
        visible_alias = "out",
        value_name = "PATH",
        help = "Write simulated expression as feather (rows = cells, columns = CellID + genes); exit. Same as --out. Requires --run-toml and --gene."
    )]
    export: Option<PathBuf>,

    #[arg(
        long = "gene",
        help = "Gene to perturb (use with batch: --export / --out)"
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
        help = "Batch perturbation spec (TOML): multiple single-gene runs with shared runtime load. Requires --run-toml. Incompatible with --gene, --export/--out, --cells-csv/--cells-csv-column."
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
    let cli = Cli::parse();

    if let Some(batch_path) = cli.batch_toml.as_ref() {
        let run_toml = cli
            .run_toml
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--run-toml is required with --batch-toml"))?;
        if cli.export.is_some() {
            anyhow::bail!(
                "--export/--out cannot be used with --batch-toml (outputs come from the batch file)."
            );
        }
        if cli.gene.is_some() {
            anyhow::bail!("--gene cannot be used with --batch-toml.");
        }
        if cli.cells_csv.is_some() || cli.cells_csv_column.is_some() {
            anyhow::bail!(
                "--cells-csv / --cells-csv-column cannot be used with --batch-toml (set cells in the batch TOML if needed)."
            );
        }
        let batch_file = load_batch_file(batch_path.as_path())?;
        let batch_parent = batch_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));

        let t_load = Instant::now();
        let mut runtime = PerturbRuntime::from_run_toml(run_toml.as_path())?;
        let load_elapsed = t_load.elapsed();
        let default_n_prop = if let Some(n) = cli.n_propagation {
            runtime.perturb_cfg.n_propagation = n;
            n
        } else {
            runtime.perturb_cfg.n_propagation
        };

        let jobs = expand_prepared_jobs(&batch_file, batch_parent, default_n_prop)?;
        validate_jobs_genes(&jobs, &runtime.gene_names)?;
        let cell_indices =
            resolve_batch_cell_indices(&batch_file, batch_parent, &runtime.obs_names)?;

        let parallelism = effective_parallelism(batch_file.parallelism, cli.batch_parallelism);
        let rt = Arc::new(runtime);
        let t_batch = Instant::now();
        run_batch_jobs(
            Arc::clone(&rt),
            jobs,
            cell_indices,
            parallelism,
            cli.verbose,
        )?;
        let batch_elapsed = t_batch.elapsed();
        if cli.verbose {
            eprintln!("--- spacetravlr-perturb batch timings ---");
            eprintln!("  load_runtime: {load_elapsed:?}");
            eprintln!("  perturb_batch_total: {batch_elapsed:?}");
        }
        return Ok(());
    }

    if cli.export.is_none() {
        #[cfg(feature = "tui")]
        {
            let opts = space_trav_lr_rust::perturb_tui::PerturbTuiOptions {
                run_toml: cli.run_toml.clone(),
                default_desired_expr: cli.desired_expr,
                n_propagation_initial: cli.n_propagation,
                verbose: cli.verbose,
                toml_path_hint_for_error: cli.run_toml.as_ref().map(|p| p.display().to_string()),
                cells_csv: cli.cells_csv.clone(),
                cells_csv_column: cli.cells_csv_column.clone(),
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            return rt.block_on(space_trav_lr_rust::perturb_tui::run(opts));
        }
        #[cfg(not(feature = "tui"))]
        {
            let run_toml = match &cli.run_toml {
                Some(p) => p.clone(),
                None => interactive_run_toml_prompt()?,
            };
            let mut runtime = PerturbRuntime::from_run_toml(run_toml.as_path())?;
            if let Some(n) = cli.n_propagation {
                runtime.perturb_cfg.n_propagation = n;
            }
            return run_interactive(runtime);
        }
    }

    let run_toml = cli
        .run_toml
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--run-toml is required with --export / --out"))?;

    if cli.cells_csv.is_some() && cli.cells_csv_column.is_none() {
        anyhow::bail!("--cells-csv-column is required when --cells-csv is set (batch export).");
    }

    let export_path = cli.export.as_ref().unwrap();
    let gene = cli
        .gene
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--gene is required with --export / --out"))?;
    let t_load = Instant::now();
    let mut runtime = PerturbRuntime::from_run_toml(run_toml.as_path())?;
    let load_elapsed = t_load.elapsed();
    if let Some(n) = cli.n_propagation {
        runtime.perturb_cfg.n_propagation = n;
    }
    if !runtime.gene_names.iter().any(|g| g == gene) {
        anyhow::bail!("Gene '{}' is not present in AnnData var_names.", gene);
    }
    let cell_indices_batch = match (&cli.cells_csv, &cli.cells_csv_column) {
        (Some(csv_path), Some(col)) => {
            let parsed = parse_obs_columns_csv(csv_path, &runtime.obs_names)?;
            let sl = parsed.indices_for_column(col.as_str()).ok_or_else(|| {
                anyhow::anyhow!("--cells-csv-column '{}' not found in CSV header", col)
            })?;
            Some(sl.to_vec())
        }
        (None, None) => None,
        _ => unreachable!("validated above"),
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
        &runtime.bb,
        &runtime.gene_mtx,
        &runtime.gene_names,
        &runtime.xy,
        &runtime.rw_ligands_init,
        &runtime.rw_tfligands_init,
        &targets,
        &runtime.perturb_cfg,
        &runtime.lr_radii,
        None,
        None,
        None,
        Some(&runtime.baseline_splash_cache),
        &mut timings,
    )
    .map_err(|_| anyhow::anyhow!("perturbation failed"))?;
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
            eprintln!("    pin_clip: {:?}", sum_suffix("/pin_clip"));
        }
    }
    Ok(())
}
