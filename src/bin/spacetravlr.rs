mod compute_backend;

use clap::Parser;
use compute_backend::{
    ComputeChoice, FitAllGenesParams, compute_hardware_details, fit_all_genes_dispatch,
    select_compute_backend,
};
use space_trav_lr_rust::config::{CnnTrainingMode, SpaceshipConfig};
use space_trav_lr_rust::training_hud::RunConfigSummary;
#[cfg(feature = "tui")]
use space_trav_lr_rust::training_hud::TrainingHudState;
#[cfg(feature = "tui")]
use space_trav_lr_rust::training_tui::{TrainingDashboardExit, run_adata_path_prompt, run_training_dashboard};
use std::path::{Path, PathBuf};
#[cfg(feature = "tui")]
use std::sync::atomic::AtomicBool;
#[cfg(feature = "tui")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "tui")]
use std::thread;

#[derive(Parser, Debug)]
#[command(
    name = "spacetravlr",
    version,
    about = "SpaceTravLR — spatial GRN training from single-cell spatial AnnData (.h5ad).",
    after_long_help = "Load spaceship_config.toml (or pass --config), then apply CLI overrides. Use --plain for line-oriented logs instead of the dashboard."
)]
struct Cli {
    #[arg(
        short = 'c',
        long,
        value_name = "PATH",
        help = "spaceship_config.toml path"
    )]
    config: Option<PathBuf>,

    #[arg(
        long,
        value_name = "PATH",
        help = "AnnData .h5ad path (overrides config data.adata_path)"
    )]
    h5ad: Option<PathBuf>,

    #[arg(long, help = "CNN spatial mode (per-cell betadata export)")]
    full: bool,

    #[arg(long, help = "Seed-only: cluster-level betas (default)")]
    seed_only: bool,

    #[arg(
        long,
        value_name = "MODE",
        help = "minimal | full | hybrid — per-gene CNN gating when hybrid (overrides --full / --seed-only)"
    )]
    training_mode: Option<String>,

    #[arg(long, value_name = "N", help = "CNN fine-tuning epochs per gene")]
    epochs: Option<usize>,

    #[arg(long, value_name = "N", help = "Parallel gene workers")]
    parallel: Option<usize>,

    #[arg(long, value_name = "N", help = "Train at most N genes (var order)")]
    max_genes: Option<usize>,

    #[arg(
        long,
        value_name = "LIST",
        help = "Comma-separated target genes (filters the gene list)"
    )]
    genes: Option<String>,

    #[arg(long, value_name = "DIR", help = "Output directory for *_betadata.csv")]
    output_dir: Option<PathBuf>,

    #[arg(long, value_name = "F", help = "L1 regularization (lasso)")]
    l1_reg: Option<f64>,

    #[arg(long, value_name = "F", help = "Group regularization (lasso)")]
    group_reg: Option<f64>,

    #[arg(long, value_name = "F", help = "Adam learning rate (CNN phase)")]
    lr: Option<f64>,

    #[arg(long, value_name = "N", help = "Max FISTA iterations")]
    n_iter: Option<usize>,

    #[arg(long, value_name = "F", help = "FISTA convergence tolerance")]
    tol: Option<f64>,

    #[arg(
        long,
        help = "Line-oriented logs on stdout (no full-screen dashboard when built with the `tui` feature)"
    )]
    plain: bool,
}

fn apply_cli_to_config(cli: &Cli, cfg: &mut SpaceshipConfig) {
    if cli.full {
        cfg.training.seed_only = false;
        cfg.training.mode = Some(CnnTrainingMode::Full);
    }
    if cli.seed_only {
        cfg.training.seed_only = true;
        cfg.training.mode = Some(CnnTrainingMode::Minimal);
    }
    if let Some(v) = cli.epochs {
        cfg.training.epochs = v;
    }
    if let Some(v) = cli.parallel {
        cfg.execution.n_parallel = v.max(1);
    }
    if let Some(p) = &cli.output_dir {
        cfg.execution.output_dir = p.display().to_string();
    }
    if let Some(v) = cli.l1_reg {
        cfg.lasso.l1_reg = v;
    }
    if let Some(v) = cli.group_reg {
        cfg.lasso.group_reg = v;
    }
    if let Some(v) = cli.lr {
        cfg.training.learning_rate = v;
    }
    if let Some(v) = cli.n_iter {
        cfg.lasso.n_iter = v;
    }
    if let Some(v) = cli.tol {
        cfg.lasso.tol = v;
    }
    if let Some(p) = &cli.h5ad {
        cfg.data.adata_path = p.display().to_string();
    }
    if let Some(ref m) = cli.training_mode {
        let m = m.to_ascii_lowercase();
        match m.as_str() {
            "minimal" | "seed" | "seed-only" => {
                cfg.training.mode = Some(CnnTrainingMode::Minimal);
                cfg.training.seed_only = true;
            }
            "full" | "cnn" => {
                cfg.training.mode = Some(CnnTrainingMode::Full);
                cfg.training.seed_only = false;
            }
            "hybrid" => {
                cfg.training.mode = Some(CnnTrainingMode::Hybrid);
                cfg.training.seed_only = true;
            }
            _ => {}
        }
    }
}

fn parse_gene_filter(cli: &Cli) -> Option<Vec<String>> {
    let genes = cli
        .genes
        .as_ref()?
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if genes.is_empty() { None } else { Some(genes) }
}

fn compute_notice_text(compute: &ComputeChoice) -> String {
    let details = compute_hardware_details(compute);
    match compute {
        ComputeChoice::Wgpu(_) => format!("Using WGPU compute backend: {}", details),
        ComputeChoice::NdArray(_) => {
            let forced = std::env::var("SPACETRAVLR_FORCE_CPU")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if forced {
                format!(
                    "SPACETRAVLR_FORCE_CPU: using CPU (NdArray) backend: {}",
                    details
                )
            } else {
                format!(
                    "No WGPU adapter found; using CPU (NdArray) backend: {}",
                    details
                )
            }
        }
    }
}

fn print_compute_notice(compute: &ComputeChoice) {
    println!("{}", compute_notice_text(compute));
}

fn print_plain_preamble(
    summary: &RunConfigSummary,
    dataset: &str,
    output_dir: &str,
    mode: &str,
    n_parallel: usize,
) {
    println!(
        "SpaceTravLR  |  {}  |  {} workers  |  {} epochs/gene",
        mode, n_parallel, summary.epochs_per_gene
    );
    println!("Compute:     {}", summary.compute_backend);
    println!("Config:      {}", summary.config_source);
    println!("Dataset:     {}", dataset);
    println!("Output:      {}", output_dir);
    println!(
        "Layer:       {}  |  obs: {}",
        summary.layer, summary.cluster_annot
    );
    println!(
        "Spatial:     r={}  dim={}  contact={}",
        summary.spatial_radius, summary.spatial_dim, summary.contact_distance
    );
    println!(
        "Lasso:       l1={}  group={}  n_iter={}  tol={:.1e}",
        summary.l1_reg, summary.group_reg, summary.n_iter, summary.tol
    );
    println!(
        "Training:    mode={}  lr={}  score≥{}",
        summary.cnn_training_mode, summary.learning_rate, summary.score_threshold
    );
    println!(
        "GRN:         tf_lig≥{}  max_lr={}  top_lr={}",
        summary.tf_ligand_cutoff, summary.max_lr_pairs, summary.top_lr_pairs
    );
    println!("Genes:       {}", summary.gene_selection);
    println!("{}", "—".repeat(60));
}

fn expand_user_path(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    if s == "~" {
        return std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| s.to_string());
    }
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(h) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return format!("{}/{}", h.trim_end_matches('/'), rest);
        }
    }
    s.to_string()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut cfg = match &cli.config {
        Some(path) => SpaceshipConfig::from_file(path)?,
        None => SpaceshipConfig::load(),
    };

    apply_cli_to_config(&cli, &mut cfg);

    let max_genes = cli.max_genes;
    let gene_filter = parse_gene_filter(&cli);

    let use_dashboard = cfg!(feature = "tui") && !cli.plain;
    let compute = select_compute_backend();

    if cfg.resolve_adata_path().is_empty() {
        #[cfg(feature = "tui")]
        {
            if use_dashboard {
                print_compute_notice(&compute);
                match run_adata_path_prompt()? {
                    None => {
                        eprintln!("No dataset path; exiting.");
                        return Ok(());
                    }
                    Some(p) => {
                        cfg.data.adata_path = p;
                    }
                }
            } else {
                anyhow::bail!(
                    "No AnnData path. Use --h5ad, set data.adata_path in config, or omit --plain for an interactive path prompt."
                );
            }
        }
        #[cfg(not(feature = "tui"))]
        {
            anyhow::bail!(
                "No AnnData path. Use --h5ad or set data.adata_path in spaceship_config.toml."
            );
        }
    }

    let path = expand_user_path(&cfg.data.adata_path);
    cfg.data.adata_path = path.clone();

    if !Path::new(&path).exists() {
        anyhow::bail!("Dataset not found at {}.", path);
    }

    let mode_label = match cfg.resolved_cnn_mode() {
        CnnTrainingMode::Minimal => "Seed-only lasso",
        CnnTrainingMode::Full => "CNN spatial",
        CnnTrainingMode::Hybrid => "Hybrid gated CNN",
    };
    #[cfg(feature = "tui")]
    let full_cnn = cfg.full_cnn();
    let epochs = cfg.training.epochs;
    let n_parallel = cfg.execution.n_parallel;
    let output_dir = cfg.execution.output_dir.clone();

    let _ = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global();

    let config_path_ref = cli.config.as_deref();
    let run_summary = RunConfigSummary::build(
        config_path_ref,
        compute.label(),
        &compute_notice_text(&compute),
        &cfg,
        max_genes,
        gene_filter.as_deref(),
    );

    if !use_dashboard {
        print_compute_notice(&compute);
        print_plain_preamble(&run_summary, &path, &output_dir, mode_label, n_parallel);
        let params = FitAllGenesParams {
            path: &path,
            radius: cfg.spatial.radius,
            spatial_dim: cfg.spatial.spatial_dim,
            contact_distance: cfg.spatial.contact_distance,
            tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
            max_lr_pairs: cfg.grn.max_lr_pairs,
            top_lr_pairs_by_mean_expression: cfg.grn.top_lr_pairs_by_mean_expression,
            layer: &cfg.data.layer,
            cnn: &cfg.cnn,
            epochs,
            learning_rate: cfg.training.learning_rate,
            score_threshold: cfg.training.score_threshold,
            l1_reg: cfg.lasso.l1_reg,
            group_reg: cfg.lasso.group_reg,
            n_iter: cfg.lasso.n_iter,
            tol: cfg.lasso.tol,
            cnn_training_mode: cfg.resolved_cnn_mode(),
            hybrid_pass2_full_cnn: false,
            hybrid_gating: &cfg.training.hybrid,
            min_mean_lasso_r2_for_cnn: cfg.min_mean_lasso_r2_for_hybrid_cnn(),
            gene_filter: gene_filter.clone(),
            max_genes,
            n_parallel,
            output_dir: &output_dir,
            model_export: &cfg.model_export,
            hud: None,
        };
        fit_all_genes_dispatch(&params, &compute)?;
        println!("Finished.");
        return Ok(());
    }

    #[cfg(feature = "tui")]
    {
        print_compute_notice(&compute);

        let cancel = Arc::new(AtomicBool::new(false));
        let hud = Arc::new(Mutex::new(TrainingHudState::new(
            path.clone(),
            output_dir.clone(),
            run_summary,
            full_cnn,
            epochs,
            n_parallel,
            cancel.clone(),
        )));

        let hud_worker = hud.clone();
        let compute_thread = compute.clone();

        let handle = thread::spawn(move || {
            let params = FitAllGenesParams {
                path: &path,
                radius: cfg.spatial.radius,
                spatial_dim: cfg.spatial.spatial_dim,
                contact_distance: cfg.spatial.contact_distance,
                tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
                max_lr_pairs: cfg.grn.max_lr_pairs,
                top_lr_pairs_by_mean_expression: cfg.grn.top_lr_pairs_by_mean_expression,
                layer: &cfg.data.layer,
                cnn: &cfg.cnn,
                epochs,
                learning_rate: cfg.training.learning_rate,
                score_threshold: cfg.training.score_threshold,
                l1_reg: cfg.lasso.l1_reg,
                group_reg: cfg.lasso.group_reg,
                n_iter: cfg.lasso.n_iter,
                tol: cfg.lasso.tol,
                cnn_training_mode: cfg.resolved_cnn_mode(),
                hybrid_pass2_full_cnn: false,
                hybrid_gating: &cfg.training.hybrid,
                min_mean_lasso_r2_for_cnn: cfg.min_mean_lasso_r2_for_hybrid_cnn(),
                gene_filter,
                max_genes,
                n_parallel,
                output_dir: &output_dir,
                model_export: &cfg.model_export,
                hud: Some(hud_worker),
            };
            fit_all_genes_dispatch(&params, &compute_thread)
        });

        match run_training_dashboard(hud.clone())? {
            TrainingDashboardExit::ForceQuit => {
                eprintln!("Aborted (Shift+Q).");
                std::process::exit(130);
            }
            TrainingDashboardExit::Completed => {}
        }

        match handle.join() {
            Ok(r) => r?,
            Err(_) => anyhow::bail!("training thread panicked"),
        }

        println!("Finished.");
    }

    Ok(())
}
