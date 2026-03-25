mod compute_backend;

use clap::{Parser, Subcommand};
use serde_json::Value;
use compute_backend::{
    ComputeChoice, FitAllGenesParams, compute_hardware_details, fit_all_genes_dispatch,
    select_compute_backend,
};
use space_trav_lr_rust::config::{
    default_output_dir_for_adata_path, expand_user_path, CnnTrainingMode, SpaceshipConfig,
};
use space_trav_lr_rust::{RunSummaryParams, write_run_summary_html};
use space_trav_lr_rust::training_hud::RunConfigSummary;
#[cfg(feature = "tui")]
use space_trav_lr_rust::training_hud::TrainingHudState;
#[cfg(feature = "tui")]
use space_trav_lr_rust::training_demo::run_demo_training;
#[cfg(feature = "tui")]
use space_trav_lr_rust::training_tui::{TrainingDashboardExit, run_dataset_paths_prompt, run_training_dashboard};
use std::path::{Path, PathBuf};
#[cfg(feature = "tui")]
use std::sync::atomic::AtomicBool;
#[cfg(feature = "tui")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "tui")]
use std::thread;

#[derive(clap::ValueEnum, Clone, Debug)]
enum TrainingModeArg {
    Full,
    Seed,
    Hybrid,
}

impl From<TrainingModeArg> for CnnTrainingMode {
    fn from(value: TrainingModeArg) -> Self {
        match value {
            TrainingModeArg::Full => CnnTrainingMode::Full,
            TrainingModeArg::Seed => CnnTrainingMode::Seed,
            TrainingModeArg::Hybrid => CnnTrainingMode::Hybrid,
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Write spacetravlr_run_summary.html (AnnData summary + config / optional manifest).
    RunSummary(RunSummaryCli),
}

#[derive(Parser, Debug, Clone)]
struct RunSummaryCli {
    #[arg(long, value_name = "PATH", help = "AnnData .h5ad (default: data.adata_path)")]
    h5ad: Option<PathBuf>,
    #[arg(
        long,
        value_name = "DIR",
        help = "Training output directory (default: cwd/{adata_stem}_YYYY-MM-DD when unset in config)"
    )]
    output_dir: Option<PathBuf>,
    #[arg(
        short = 'c',
        long,
        value_name = "PATH",
        help = "spaceship_config.toml (defaults to cwd discovery if omitted)"
    )]
    config: Option<PathBuf>,
    #[arg(
        long,
        help = "obs column for cluster count (default: data.cluster_annot)"
    )]
    cluster_key: Option<String>,
    #[arg(long, help = "documented in the report only")]
    layer: Option<String>,
    #[arg(long, help = "override run id (default: manifest or AnnData stem)")]
    run_id: Option<String>,
    #[arg(
        long,
        value_name = "PATH",
        help = "optional JSON manifest from training"
    )]
    manifest: Option<PathBuf>,
    #[arg(
        long,
        default_value = "*_betadata.feather",
        help = "glob for counting betadata Feather files in the output directory"
    )]
    betadata_pattern: String,
}

#[derive(Parser, Debug)]
#[command(
    name = "spacetravlr",
    version,
    about = "SpaceTravLR — spatial GRN training from single-cell spatial AnnData (.h5ad).",
    after_long_help = "Load spaceship_config.toml (or pass --config), then apply CLI overrides. Use --plain for line-oriented logs instead of the dashboard. Subcommand `run-summary` writes the HTML report without training."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
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

    #[arg(
        long,
        value_name = "MODE",
        value_enum,
        help = "full | seed | hybrid (default: seed)"
    )]
    training_mode: Option<TrainingModeArg>,

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

    #[arg(
        long,
        value_name = "DIR",
        help = "Output for *_betadata.feather (default: cwd/{adata_stem}_YYYY-MM-DD when unset in config)"
    )]
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

    #[arg(
        long,
        help = "Simulated training dashboard only: fake workers, gene progress, and R² stats — no AnnData I/O, no exports, no GPU (omit --plain)"
    )]
    demo: bool,
}

fn apply_cli_to_config(cli: &Cli, cfg: &mut SpaceshipConfig) {
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
        cfg.training.mode = Some(m.clone().into());
        cfg.training.seed_only = !matches!(cfg.training.mode, Some(CnnTrainingMode::Full));
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

fn grn_modulator_label(cfg: &SpaceshipConfig) -> String {
    let mut parts = Vec::new();
    if cfg.grn.use_tf_modulators {
        parts.push("TF");
    }
    if cfg.grn.use_lr_modulators {
        parts.push("LR");
    }
    if cfg.grn.use_tfl_modulators {
        parts.push("TFL");
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("+")
    }
}

fn print_plain_preamble(
    summary: &RunConfigSummary,
    cfg: &SpaceshipConfig,
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
        "GRN:         tf_lig≥{}  max_lr={}  top_lr={}  mods={}",
        summary.tf_ligand_cutoff,
        summary.max_lr_pairs,
        summary.top_lr_pairs,
        grn_modulator_label(cfg)
    );
    println!("Genes:       {}", summary.gene_selection);
    println!("{}", "—".repeat(60));
}

fn run_run_summary(cli: &Cli, rs: &RunSummaryCli) -> anyhow::Result<()> {
    let cfg = match rs.config.as_ref().or(cli.config.as_ref()) {
        Some(p) => SpaceshipConfig::from_file(p)?,
        None => SpaceshipConfig::load(),
    };

    let adata_path = rs
        .h5ad
        .clone()
        .or_else(|| {
            let p = expand_user_path(&cfg.resolve_adata_path());
            if p.is_empty() {
                None
            } else {
                Some(PathBuf::from(p))
            }
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No AnnData path: pass --h5ad or set data.adata_path in spaceship_config.toml."
            )
        })?;

    let output_dir = if let Some(p) = rs.output_dir.clone() {
        p
    } else {
        let d = expand_user_path(cfg.execution.output_dir.trim());
        if !d.is_empty() {
            PathBuf::from(d)
        } else {
            PathBuf::from(default_output_dir_for_adata_path(&adata_path)?)
        }
    };

    if !Path::new(&adata_path).exists() {
        anyhow::bail!("AnnData not found at {}.", adata_path.display());
    }

    let manifest: Option<Value> = rs
        .manifest
        .as_ref()
        .map(|p| {
            let s = std::fs::read_to_string(p)?;
            let v: Value = serde_json::from_str(&s)?;
            Ok::<_, anyhow::Error>(v)
        })
        .transpose()?;

    let path = write_run_summary_html(RunSummaryParams {
        adata_path: &adata_path,
        output_dir: &output_dir,
        cfg: &cfg,
        cluster_key: rs.cluster_key.as_deref(),
        layer_override: rs.layer.as_deref(),
        run_id: rs.run_id.as_deref(),
        manifest: manifest.as_ref(),
        betadata_pattern: rs.betadata_pattern.as_str(),
    })?;
    println!("{}", path.display());
    Ok(())
}

#[cfg(feature = "tui")]
fn run_demo_mode(cli: &Cli) -> anyhow::Result<()> {
    if cli.plain {
        anyhow::bail!("--demo is for the full-screen dashboard; omit --plain.");
    }

    let mut cfg = match &cli.config {
        Some(path) => SpaceshipConfig::from_file(path)?,
        None => SpaceshipConfig::load(),
    };
    apply_cli_to_config(cli, &mut cfg);

    let gene_filter = parse_gene_filter(cli);
    let demo_total = cli.max_genes.unwrap_or(24).max(1).min(512);

    let config_path_ref = cli.config.as_deref();
    let run_summary = RunConfigSummary::build(
        config_path_ref,
        "demo",
        "Demo mode — simulated genes/workers only; no AnnData load, no betadata export, no training backend.",
        &cfg,
        Some(demo_total),
        gene_filter.as_deref(),
    );

    let full_cnn = cfg.full_cnn();
    let epochs = cfg.training.epochs;
    let n_parallel = cfg.execution.n_parallel;
    let cancel = Arc::new(AtomicBool::new(false));
    let hud = Arc::new(Mutex::new(TrainingHudState::new(
        "(demo) simulated_visium.h5ad".to_string(),
        "(demo — no disk writes)".to_string(),
        run_summary,
        full_cnn,
        epochs,
        n_parallel,
        cancel.clone(),
    )));

    println!("SpaceTravLR --demo: opening dashboard (Shift+Q to exit immediately).");

    let hud_worker = hud.clone();
    let filter_for_demo = gene_filter.clone();
    let handle = thread::spawn(move || run_demo_training(hud_worker, demo_total, filter_for_demo));

    match run_training_dashboard(hud.clone())? {
        TrainingDashboardExit::ForceQuit => {
            eprintln!("Aborted (Shift+Q).");
            std::process::exit(130);
        }
        TrainingDashboardExit::Completed => {}
    }

    match handle.join() {
        Ok(r) => r?,
        Err(_) => anyhow::bail!("demo thread panicked"),
    }

    println!("Demo finished.");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(Commands::RunSummary(rs)) = &cli.command {
        return run_run_summary(&cli, rs);
    }

    if cli.demo {
        #[cfg(not(feature = "tui"))]
        anyhow::bail!(
            "This binary was built without the `tui` feature; rebuild with default features to use --demo."
        );
        #[cfg(feature = "tui")]
        return run_demo_mode(&cli);
    }

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
                match run_dataset_paths_prompt(cfg.execution.output_dir.trim())? {
                    None => {
                        eprintln!("No dataset path; exiting.");
                        return Ok(());
                    }
                    Some((h5ad, out_dir)) => {
                        cfg.data.adata_path = h5ad;
                        cfg.execution.output_dir = out_dir;
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

    let network_data_dir: Option<String> = cfg
        .grn
        .network_data_dir
        .as_ref()
        .map(|s| expand_user_path(s.trim()))
        .filter(|s| !s.is_empty());
    let tf_priors_feather: Option<String> = cfg
        .grn
        .tf_priors_feather
        .as_ref()
        .map(|s| expand_user_path(s.trim()))
        .filter(|s| !s.is_empty());

    if !Path::new(&path).exists() {
        anyhow::bail!("Dataset not found at {}.", path);
    }

    if cfg.execution.output_dir.trim().is_empty() {
        cfg.execution.output_dir = default_output_dir_for_adata_path(Path::new(&path))?;
    }

    let mode_label = match cfg.resolved_cnn_mode() {
        CnnTrainingMode::Seed => "seed",
        CnnTrainingMode::Full => "full",
        CnnTrainingMode::Hybrid => "hybrid",
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
        print_plain_preamble(&run_summary, &cfg, &path, &output_dir, mode_label, n_parallel);
        let params = FitAllGenesParams {
            path: &path,
            radius: cfg.spatial.radius,
            spatial_dim: cfg.spatial.spatial_dim,
            contact_distance: cfg.spatial.contact_distance,
            tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
            max_lr_pairs: cfg.grn.max_lr_pairs,
            top_lr_pairs_by_mean_expression: cfg.grn.top_lr_pairs_by_mean_expression,
            use_tf_modulators: cfg.grn.use_tf_modulators,
            use_lr_modulators: cfg.grn.use_lr_modulators,
            use_tfl_modulators: cfg.grn.use_tfl_modulators,
            layer: &cfg.data.layer,
            cluster_annot: &cfg.data.cluster_annot,
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
            network_data_dir: network_data_dir.clone(),
            tf_priors_feather: tf_priors_feather.clone(),
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
        let network_data_dir_thread = network_data_dir.clone();

        let handle = thread::spawn(move || {
            let params = FitAllGenesParams {
                path: &path,
                radius: cfg.spatial.radius,
                spatial_dim: cfg.spatial.spatial_dim,
                contact_distance: cfg.spatial.contact_distance,
                tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
                max_lr_pairs: cfg.grn.max_lr_pairs,
                top_lr_pairs_by_mean_expression: cfg.grn.top_lr_pairs_by_mean_expression,
                use_tf_modulators: cfg.grn.use_tf_modulators,
                use_lr_modulators: cfg.grn.use_lr_modulators,
                use_tfl_modulators: cfg.grn.use_tfl_modulators,
                layer: &cfg.data.layer,
                cluster_annot: &cfg.data.cluster_annot,
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
                network_data_dir: network_data_dir_thread,
                tf_priors_feather: tf_priors_feather.clone(),
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
