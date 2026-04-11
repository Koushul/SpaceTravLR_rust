mod compute_backend;

use clap::builder::styling::AnsiColor;
use clap::builder::Styles;
use anyhow::Context;
use clap::{ArgAction, ColorChoice, Parser, Subcommand};
use compute_backend::{
    ComputeChoice, FitAllGenesParams, compute_hardware_details, fit_all_genes_dispatch,
    select_compute_backend,
};
use serde_json::Value;
use spacetravlr::condition_split::{prepare_condition_splits, scan_condition_status};
use spacetravlr::config::{
    CnnOutputActivation, CnnTrainingMode, RUN_REPRO_TOML_FILENAME, SpaceshipConfig,
    canonical_adata_stem, default_output_dir_for_adata_path, expand_user_path,
};
use spacetravlr::grn_extra;
#[cfg(feature = "tui")]
use spacetravlr::training_demo::{
    prepare_demo_hud, run_demo_training, DEMO_KIDNEY_SLIDETAGS_H5AD, DEMO_OUTPUT_DIR_LABEL,
};
use spacetravlr::training_hud::RunConfigSummary;
#[cfg(feature = "tui")]
use spacetravlr::training_hud::TrainingHudState;
#[cfg(feature = "tui")]
use spacetravlr::training_tui::{
    TrainingDashboardExit, run_dataset_paths_prompt, run_training_dashboard,
};
use spacetravlr::{RunSummaryParams, write_run_summary_html};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "tui")]
use std::sync::Mutex;
#[cfg(feature = "tui")]
use std::sync::atomic::AtomicBool;
#[cfg(feature = "tui")]
use std::thread;

const SPACETRAVLR_LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (target ",
    env!("SPACETRAVLR_TARGET_TRIPLE"),
    ", git ",
    env!("SPACETRAVLR_GIT_SHA"),
    ")"
);

const SPACETRAVLR_HELP_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Blue.on_default().bold())
    .usage(AnsiColor::Cyan.on_default().bold())
    .literal(AnsiColor::Green.on_default().bold())
    .placeholder(AnsiColor::BrightCyan.on_default())
    .valid(AnsiColor::Green.on_default())
    .invalid(AnsiColor::Yellow.on_default())
    .error(AnsiColor::Red.on_default().bold());

const SPACETRAVLR_LONG_ABOUT: &str = r#"Spatial gene regulatory network (GRN) training from Visium-style spatial AnnData (.h5ad).

• Load spaceship_config.toml (or pass --config), then apply CLI overrides.
• Use --plain for line-oriented logs instead of the full-screen dashboard (when built with `tui`).
• Subcommand run-summary writes the HTML report without training."#;

const SPACETRAVLR_AFTER_LONG_HELP: &str = r#"

Multi-host / shared storage
  Start a leader run (writes spacetravlr_run_repro.toml early), then use --join-output-dir DIR on other hosts with --parallel set per machine.

Condition splits
  With --condition, --join-output-dir points to the parent output directory; conditions/<group>/ subdirectories are auto-discovered from the repro TOML."#;

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

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum CnnOutputActivationArg {
    Identity,
    Sigmoid,
    Tanh,
    SigmoidX2,
}

impl From<CnnOutputActivationArg> for CnnOutputActivation {
    fn from(value: CnnOutputActivationArg) -> Self {
        match value {
            CnnOutputActivationArg::Identity => CnnOutputActivation::Identity,
            CnnOutputActivationArg::Sigmoid => CnnOutputActivation::Sigmoid,
            CnnOutputActivationArg::Tanh => CnnOutputActivation::Tanh,
            CnnOutputActivationArg::SigmoidX2 => CnnOutputActivation::SigmoidX2,
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Generate spacetravlr_run_summary.html (AnnData summary, config, optional manifest).
    RunSummary(RunSummaryCli),
}

#[derive(Parser, Debug, Clone)]
struct RunSummaryCli {
    #[arg(
        long,
        value_name = "PATH",
        help = "AnnData .h5ad (default: data.adata_path)"
    )]
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
    version = env!("CARGO_PKG_VERSION"),
    long_version = SPACETRAVLR_LONG_VERSION,
    about = "Spatial GRN training from spatial AnnData (.h5ad).",
    long_about = SPACETRAVLR_LONG_ABOUT,
    after_long_help = SPACETRAVLR_AFTER_LONG_HELP,
    styles = SPACETRAVLR_HELP_STYLES,
    color = ColorChoice::Auto,
    next_line_help = true,
    propagate_version = true,
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        help_heading = "Install",
        help = "Download the latest release and replace spacetravlr, spacetravlr-perturb, and spatial_viewer next to this executable (opt-in; uses the network only when you pass this flag). Requires build with `self-update`."
    )]
    update: bool,

    #[arg(
        long = "update-version",
        value_name = "TAG",
        help_heading = "Install",
        help = "With --update: install a specific release tag (e.g. v0.2.0) instead of latest"
    )]
    update_version: Option<String>,

    #[arg(
        short = 'c',
        long,
        value_name = "PATH",
        help_heading = "Input",
        help = "spaceship_config.toml (searched nearby if omitted)"
    )]
    config: Option<PathBuf>,

    #[arg(
        long,
        value_name = "PATH",
        help_heading = "Input",
        help = "Spatial AnnData .h5ad — overrides [data].adata_path"
    )]
    h5ad: Option<PathBuf>,

    #[arg(
        long = "skip-auto-adata-prep",
        action = ArgAction::SetTrue,
        help_heading = "Input",
        help = "Do not auto-run Scanpy / imputation when AnnData lacks cell_type or layers[\"imputed_count\"]"
    )]
    skip_auto_adata_prep: bool,

    #[arg(
        long = "tf-prior",
        value_name = "PATH",
        help_heading = "Input",
        help = "Feather with TF priors (source, target, cell_type) — overrides [grn].tf_priors_feather"
    )]
    tf_priors_feather: Option<PathBuf>,

    #[arg(
        long,
        value_name = "LIST",
        help_heading = "Gene list & GRN extras",
        help = "Train only these targets — comma-separated symbols, same style as a single-line gene list"
    )]
    genes: Option<String>,

    #[arg(
        long,
        value_name = "N",
        help_heading = "Gene list & GRN extras",
        help = "Stop after N genes (AnnData var order, after --genes filter)"
    )]
    max_genes: Option<usize>,

    #[arg(
        long = "max-ligands",
        value_name = "N",
        help_heading = "Gene list & GRN extras",
        help = "Keep only DB L–R pairs whose ligand ranks in the top N by mean expression ([data].layer)"
    )]
    max_ligands: Option<usize>,

    #[arg(
        long = "extra-modulators",
        value_name = "GENES",
        help_heading = "Gene list & GRN extras",
        help = "Comma-separated genes added as an extra Lasso modulator block — merged with [grn].extra_modulators / *_file"
    )]
    extra_modulators: Option<String>,

    #[arg(
        long = "extra-lr",
        value_name = "PAIRS",
        help_heading = "Gene list & GRN extras",
        help = "Extra ligand→receptor pairs, merged with [grn].extra_lr / *_file. Forms: L1$R1,L2$R2  or  L1,R1;L2,R2  or  single L1,R1"
    )]
    extra_lr: Option<String>,

    #[arg(
        long,
        value_name = "MODE",
        value_enum,
        help_heading = "Training",
        help = "seed | full | hybrid CNN (default from config, usually seed)"
    )]
    training_mode: Option<TrainingModeArg>,

    #[arg(
        long,
        value_name = "N",
        help_heading = "Training",
        help = "CNN epochs per gene when CNN runs"
    )]
    epochs: Option<usize>,

    #[arg(
        long,
        value_name = "N",
        help_heading = "Training",
        help = "Parallel worker threads (one gene per worker at a time)"
    )]
    parallel: Option<usize>,

    #[arg(
        long,
        value_name = "F",
        help_heading = "Training",
        help = "L1 penalty for Lasso (element-wise)"
    )]
    l1_reg: Option<f64>,

    #[arg(
        long,
        value_name = "F",
        help_heading = "Training",
        help = "Group penalty for Lasso (per modulator group)"
    )]
    group_reg: Option<f64>,

    #[arg(
        long,
        value_name = "F",
        help_heading = "Training",
        help = "Adam learning rate for CNN fine-tuning"
    )]
    lr: Option<f64>,

    #[arg(
        long = "cnn-output-activation",
        value_enum,
        value_name = "MODE",
        help_heading = "Training",
        help = "CNN head nonlinearity before Lasso-anchor scaling: identity | sigmoid | tanh | sigmoid-x2"
    )]
    cnn_output_activation: Option<CnnOutputActivationArg>,

    #[arg(
        long,
        value_name = "N",
        help_heading = "Training",
        help = "Max FISTA iterations for Lasso"
    )]
    n_iter: Option<usize>,

    #[arg(
        long,
        value_name = "F",
        help_heading = "Training",
        help = "FISTA relative tolerance"
    )]
    tol: Option<f64>,

    #[arg(
        long = "weighted-ligand-scale-factor",
        value_name = "F",
        help_heading = "Training",
        help = "Scales Gaussian weights when aggregating received ligands — overrides [spatial].weighted_ligand_scale_factor"
    )]
    weighted_ligand_scale_factor: Option<f64>,

    #[arg(
        long,
        value_name = "DIR",
        help_heading = "Output",
        help = "Directory for *_betadata.feather and logs (default: dated folder from stem of .h5ad)"
    )]
    output_dir: Option<PathBuf>,

    #[arg(
        long,
        value_name = "OBS_COLUMN",
        help_heading = "Output",
        help = "Split training by this obs column (one subdirectory per value under output_dir/conditions/). With `--process-h5ad` or `--impute`, also selects this `adata.obs` column as the MAGIC batch axis (unless `--magic-batch-obs` is set)"
    )]
    condition: Option<String>,

    #[arg(
        long = "join-output-dir",
        value_name = "DIR",
        help_heading = "Output",
        help = "Resume/join a shared run: read DIR/spacetravlr_run_repro.toml; claim unfinished genes via locks. Hyperparameters come from the repro file (not --config)"
    )]
    join_output_dir: Option<PathBuf>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        help_heading = "Output",
        help = "Write spacetravlr_minimal_repro.h5ad into the run directory (large I/O)"
    )]
    write_minimal_repro_h5ad: bool,

    #[arg(
        long = "save-cnn-weights",
        action = ArgAction::SetTrue,
        help_heading = "Output",
        help = "Save CNN weights as .npz under the run directory"
    )]
    save_cnn_weights: bool,

    #[arg(
        long,
        help_heading = "Interface",
        help = "Print line-oriented logs instead of the full-screen dashboard (when built with `tui`)"
    )]
    plain: bool,

    #[arg(
        long,
        help_heading = "Interface",
        help = "Fake training dashboard only — no AnnData, no disk exports, no accelerator"
    )]
    demo: bool,

    #[arg(
        long = "plot-h5ad",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Print terminal spatial scatter (obsm spatial, obs colored by cluster column) and exit"
    )]
    plot_h5ad: bool,

    #[arg(
        long = "process-h5ad",
        alias = "process_h5ad",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Full pipeline: uv Scanpy (QC → UMAP/Leiden) + clusterwise magic-impute → `<stem>_processed.h5ad` (requires `--h5ad`)"
    )]
    process_h5ad: bool,

    #[arg(
        long = "process-output-dir",
        value_name = "DIR",
        help_heading = "Utility",
        help = "With `--process-h5ad` / `--impute`: write `<stem>_processed.h5ad` or `<stem>_imputed.h5ad` here (default: current directory)"
    )]
    process_output_dir: Option<PathBuf>,

    #[arg(
        long,
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Imputation only: clusterwise magic-impute on `layers[\"normalized_count\"]` (same step as `--process-h5ad`) → `<stem>_imputed.h5ad` (requires `--h5ad`; needs `cell_type` or `leiden`)"
    )]
    impute: bool,

    #[arg(
        long = "magic-batch-obs",
        value_name = "OBS_COLUMN",
        help_heading = "Utility",
        help = "With `--process-h5ad` / `--impute`: run MAGIC once per (cell_type or Leiden) × this `adata.obs` column. When omitted but `--condition` is set, that column is used as the batch axis"
    )]
    magic_batch_obs: Option<String>,
}

fn apply_cli_join_overrides(cli: &Cli, cfg: &mut SpaceshipConfig) -> anyhow::Result<()> {
    if let Some(v) = cli.parallel {
        cfg.execution.n_parallel = v.max(1);
    }
    if cli.save_cnn_weights {
        cfg.model_export.save_cnn_weights = true;
    }
    if cli.write_minimal_repro_h5ad {
        cfg.execution.write_minimal_repro_h5ad = true;
    }
    if let Some(p) = &cli.h5ad {
        cfg.data.adata_path = expand_user_path(p.to_string_lossy().as_ref());
    }
    if let Some(p) = &cli.tf_priors_feather {
        cfg.grn.tf_priors_feather = Some(expand_user_path(p.to_string_lossy().as_ref()));
    }
    if let Some(ref c) = cli.condition {
        let t = c.trim();
        if !t.is_empty() {
            cfg.data.condition = Some(t.to_string());
        }
    }
    if let Some(ref raw) = cli.extra_modulators {
        cfg.grn
            .extra_modulators
            .extend(grn_extra::parse_extra_modulators_cli(raw));
    }
    if let Some(ref raw) = cli.extra_lr {
        cfg.grn.extra_lr.extend(grn_extra::parse_extra_lr_cli(raw)?);
    }
    Ok(())
}

fn apply_cli_to_config(cli: &Cli, cfg: &mut SpaceshipConfig) -> anyhow::Result<()> {
    if let Some(v) = cli.epochs {
        cfg.training.epochs = v;
    }
    if let Some(v) = cli.parallel {
        cfg.execution.n_parallel = v.max(1);
    }
    if let Some(v) = cli.max_ligands {
        cfg.grn.max_ligands = Some(v.max(1));
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
    if let Some(a) = cli.cnn_output_activation {
        cfg.cnn.output_activation = a.into();
    }
    if let Some(v) = cli.n_iter {
        cfg.lasso.n_iter = v;
    }
    if let Some(v) = cli.tol {
        cfg.lasso.tol = v;
    }
    if let Some(v) = cli.weighted_ligand_scale_factor {
        cfg.spatial.weighted_ligand_scale_factor = v;
    }
    if let Some(p) = &cli.h5ad {
        cfg.data.adata_path = expand_user_path(p.to_string_lossy().as_ref());
    }
    if let Some(p) = &cli.tf_priors_feather {
        cfg.grn.tf_priors_feather = Some(expand_user_path(p.to_string_lossy().as_ref()));
    }
    if let Some(ref m) = cli.training_mode {
        cfg.training.mode = Some(m.clone().into());
        cfg.training.seed_only = !matches!(cfg.training.mode, Some(CnnTrainingMode::Full));
    }
    if cli.write_minimal_repro_h5ad {
        cfg.execution.write_minimal_repro_h5ad = true;
    }
    if cli.save_cnn_weights {
        cfg.model_export.save_cnn_weights = true;
    }
    if let Some(ref c) = cli.condition {
        let t = c.trim();
        if !t.is_empty() {
            cfg.data.condition = Some(t.to_string());
        }
    }
    if let Some(ref raw) = cli.extra_modulators {
        cfg.grn
            .extra_modulators
            .extend(grn_extra::parse_extra_modulators_cli(raw));
    }
    if let Some(ref raw) = cli.extra_lr {
        cfg.grn.extra_lr.extend(grn_extra::parse_extra_lr_cli(raw)?);
    }
    if let Some(genes) = parse_gene_filter(cli) {
        cfg.training.genes = Some(genes);
    }
    if let Some(n) = cli.max_genes {
        cfg.training.max_genes = Some(n);
    }
    Ok(())
}

fn load_config_for_main(cli: &Cli) -> anyhow::Result<(SpaceshipConfig, bool)> {
    if let Some(j) = cli.join_output_dir.as_ref() {
        let jexp = expand_user_path(j.to_string_lossy().as_ref());
        let repro = Path::new(&jexp).join(RUN_REPRO_TOML_FILENAME);
        if !repro.is_file() {
            anyhow::bail!(
                "--join-output-dir: missing run config {} (start a leader run on this directory first, or copy the TOML from the primary host)",
                repro.display()
            );
        }
        let mut cfg = SpaceshipConfig::from_file(&repro)?;
        if let Some(cli_k) = cli.max_ligands {
            let expected = cli_k.max(1);
            if cfg.grn.max_ligands != Some(expected) {
                anyhow::bail!(
                    "--join-output-dir: --max-ligands {} does not match [grn].max_ligands ({:?}) in {}.\n\
                     Join training uses the repro TOML as the single source of truth; omit --max-ligands, or set [grn].max_ligands the same on the leader run.",
                    expected,
                    cfg.grn.max_ligands,
                    repro.display()
                );
            }
        }
        let repro_file_condition = cfg.data.condition.clone();
        if let Some(cli_raw) = cli.condition.as_deref() {
            let cli_c = cli_raw.trim();
            if !cli_c.is_empty() {
                if let Some(ref file_c) = repro_file_condition {
                    if !cli_c.eq_ignore_ascii_case(file_c.trim()) {
                        anyhow::bail!(
                            "--condition {:?} does not match [data].condition = {:?} in {}; omit --condition to use the file, or fix the mismatch.",
                            cli_c,
                            file_c,
                            repro.display()
                        );
                    }
                }
            }
        }
        cfg.execution.output_dir = jexp;
        apply_cli_join_overrides(cli, &mut cfg)?;
        if cli.config.is_some() {
            eprintln!(
                "Note: --join-output-dir ignores --config for training settings (using repro TOML)."
            );
        }
        if cli.max_genes.is_some() || cli.genes.is_some() {
            eprintln!(
                "Note: --join-output-dir uses [training] genes / max_genes from {}; --genes and --max-genes on this command are ignored.",
                repro.display()
            );
        }
        if cli.epochs.is_some()
            || cli.lr.is_some()
            || cli.l1_reg.is_some()
            || cli.group_reg.is_some()
            || cli.n_iter.is_some()
            || cli.tol.is_some()
            || cli.training_mode.is_some()
            || cli.output_dir.is_some()
            || cli.cnn_output_activation.is_some()
            || cli.weighted_ligand_scale_factor.is_some()
        {
            eprintln!(
                "Note: --join-output-dir ignores hyperparameter/output CLI flags except --parallel (using repro TOML)."
            );
        }
        Ok((cfg, true))
    } else {
        let mut cfg = match &cli.config {
            Some(path) => SpaceshipConfig::from_file(path)?,
            None => SpaceshipConfig::load(),
        };
        apply_cli_to_config(cli, &mut cfg)?;
        Ok((cfg, false))
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
        ComputeChoice::Wgpu(_) => format!("Using WebGPU compute backend: {}", details),
        ComputeChoice::NdArray(_) => {
            let forced_cpu = std::env::var("SPACETRAVLR_FORCE_CPU")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let disable_wgpu = std::env::var("SPACETRAVLR_DISABLE_WGPU")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if forced_cpu || disable_wgpu {
                format!(
                    "Using CPU (NdArray) backend (SPACETRAVLR_FORCE_CPU / SPACETRAVLR_DISABLE_WGPU): {}",
                    details
                )
            } else {
                format!(
                    "No GPU backend available; using CPU (NdArray) backend: {}",
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
    println!(
        "Compute:     {} — {}",
        summary.compute_backend, summary.compute_device_detail
    );
    println!("Config:      {}", summary.config_source);
    println!("Dataset:     {}", dataset);
    println!("Output:      {}", output_dir);
    println!(
        "Layer:       {}  |  obs: {}",
        summary.layer, summary.cluster_annot
    );
    println!(
        "Spatial:     r={}  dim={}  contact={}  weighted_ligand_scale={}",
        summary.spatial_radius,
        summary.spatial_dim,
        summary.contact_distance,
        summary.weighted_ligand_scale_factor
    );
    println!(
        "Lasso:       l1={:.3e}  group={:.3e}  n_iter={}  tol={:.1e}",
        summary.l1_reg, summary.group_reg, summary.n_iter, summary.tol
    );
    println!(
        "Training:    mode={}  lr={:.3e}  score≥{}",
        summary.cnn_training_mode, summary.learning_rate, summary.score_threshold
    );
    println!(
        "GRN:         tf_lig≥{}  max_ligands={}  mods={}",
        summary.tf_ligand_cutoff,
        summary.max_ligands,
        grn_modulator_label(cfg)
    );
    println!("Genes:       {}", summary.gene_selection);
    println!(
        "Minimal repro: {}",
        if cfg.execution.write_minimal_repro_h5ad {
            "on (spacetravlr_minimal_repro.h5ad)"
        } else {
            "off"
        }
    );
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

    let config_source_run: Option<PathBuf> = rs
        .config
        .as_ref()
        .or(cli.config.as_ref())
        .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())))
        .or_else(SpaceshipConfig::discover_default_path);

    cfg.write_run_repro_toml(&output_dir)?;

    let path = write_run_summary_html(RunSummaryParams {
        adata_path: &adata_path,
        output_dir: &output_dir,
        cfg: &cfg,
        cluster_key: rs.cluster_key.as_deref(),
        layer_override: rs.layer.as_deref(),
        run_id: rs.run_id.as_deref(),
        manifest: manifest.as_ref(),
        betadata_pattern: rs.betadata_pattern.as_str(),
        config_source_path: config_source_run.as_deref(),
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
    apply_cli_to_config(cli, &mut cfg)?;

    let gene_filter = cfg.training.genes.clone();
    let demo_total = cfg.training.max_genes.unwrap_or(24).clamp(1, 512);

    let config_path_ref = cli.config.as_deref();
    let run_summary = RunConfigSummary::build(
        config_path_ref,
        "demo",
        "— (demo; no accelerator)",
        "Demo mode — kidney slide-tags path is display-only; obsm['spatial'] from embedded cache; simulated genes/workers; no AnnData load, no betadata export, no training backend.",
        &cfg,
        Some(demo_total),
        gene_filter.as_deref(),
        None,
    );

    let full_cnn = cfg.full_cnn();
    let epochs = cfg.training.epochs;
    let n_parallel = cfg.execution.n_parallel;
    let cancel = Arc::new(AtomicBool::new(false));
    let hud = Arc::new(Mutex::new(TrainingHudState::new(
        DEMO_KIDNEY_SLIDETAGS_H5AD.to_string(),
        DEMO_OUTPUT_DIR_LABEL.to_string(),
        run_summary,
        full_cnn,
        epochs,
        n_parallel,
        cancel.clone(),
    )));

    prepare_demo_hud(&hud, demo_total, gene_filter.as_deref())?;

    println!(
        "SpaceTravLR --demo: opening dashboard (Shift+Q exit · t cycles theme). Dataset path is display-only; spatial panel uses embedded kidney obsm cache (no .h5ad read)."
    );

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

fn run_process_h5ad(cli: &Cli) -> anyhow::Result<()> {
    let h5ad = cli
        .h5ad
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--process-h5ad requires `--h5ad PATH`"))?;
    let h5ad = PathBuf::from(expand_user_path(h5ad.to_string_lossy().as_ref()));
    if !h5ad.is_file() {
        anyhow::bail!("AnnData not found at {}.", h5ad.display());
    }
    let out_dir = match &cli.process_output_dir {
        Some(p) => PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())),
        None => std::env::current_dir().context("process-output-dir default (cwd)")?,
    };
    std::fs::create_dir_all(&out_dir)?;
    let stem = canonical_adata_stem(&h5ad);
    let dest =
        spacetravlr::scanpy_preprocess::training_processed_h5ad_path(&out_dir, &stem);
    let batch_owned = spacetravlr::scanpy_preprocess::resolve_magic_batch_obs_column(
        cli.magic_batch_obs.as_deref(),
        cli.condition.as_deref(),
    );
    let batch = batch_owned.as_deref();
    let (out, log) = spacetravlr::scanpy_preprocess::full_preprocess_maybe_log(
        &h5ad,
        &dest,
        true,
        batch,
    )?;
    if let Some(l) = log {
        eprint!("{l}");
    }
    eprintln!("Wrote {}", out.display());
    Ok(())
}

fn run_impute(cli: &Cli) -> anyhow::Result<()> {
    use spacetravlr::scanpy_preprocess::{magic_impute_and_attach_batch, training_imputed_h5ad_path};

    let h5ad = cli.h5ad.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--impute requires `--h5ad PATH`")
    })?;
    let h5ad = PathBuf::from(expand_user_path(h5ad.to_string_lossy().as_ref()));
    if !h5ad.is_file() {
        anyhow::bail!("AnnData not found at {}.", h5ad.display());
    }
    let out_dir = match &cli.process_output_dir {
        Some(p) => PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())),
        None => std::env::current_dir().context("process-output-dir default (cwd)")?,
    };
    std::fs::create_dir_all(&out_dir)?;
    let stem = canonical_adata_stem(&h5ad);
    let out = training_imputed_h5ad_path(&out_dir, &stem);
    let batch_owned = spacetravlr::scanpy_preprocess::resolve_magic_batch_obs_column(
        cli.magic_batch_obs.as_deref(),
        cli.condition.as_deref(),
    );
    let log = magic_impute_and_attach_batch(&h5ad, &out, batch_owned.as_deref(), true)?;
    eprint!("{log}");
    eprintln!("Wrote {}", out.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.update {
        #[cfg(feature = "self-update")]
        return spacetravlr::self_update::run(cli.update_version.as_deref());
        #[cfg(not(feature = "self-update"))]
        anyhow::bail!(
            "This binary was built without the `self-update` feature. Upgrade with:\n\
             curl -fsSL https://raw.githubusercontent.com/Koushul/SpaceTravLR_rust/refs/tags/v1.1.0/scripts/install.sh -o install-spacetravlr.sh && sh install-spacetravlr.sh && rm -f install-spacetravlr.sh\n\
             See https://github.com/Koushul/SpaceTravLR_rust/blob/main/install.md"
        );
    }

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

    if cli.process_h5ad {
        return run_process_h5ad(&cli);
    }

    if cli.impute {
        return run_impute(&cli);
    }

    let (mut cfg, join_training) = load_config_for_main(&cli)?;

    let config_source_path: Option<PathBuf> = if join_training {
        Some(
            PathBuf::from(expand_user_path(cfg.execution.output_dir.trim()))
                .join(RUN_REPRO_TOML_FILENAME),
        )
    } else {
        cli.config
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())))
            .or_else(SpaceshipConfig::discover_default_path)
    };

    let max_genes = cfg.training.max_genes;
    let gene_filter = cfg.training.genes.clone();
    let condition_column = cli
        .condition
        .clone()
        .or_else(|| cfg.data.condition.clone())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if join_training
        && condition_column.is_none()
        && Path::new(&cfg.execution.output_dir)
            .join(spacetravlr::condition_split::CONDITION_RUNS_SUBDIR)
            .is_dir()
    {
        eprintln!(
            "Warning: --join-output-dir points at a run with a `conditions/` subtree, but neither --condition nor [data].condition in the repro TOML is set; training will use a single output directory (not per-condition). Pass --condition <obs_column> if you meant to resume condition splits."
        );
    }

    let use_dashboard = cfg!(feature = "tui") && !cli.plain;
    let compute = select_compute_backend();

    if cli.plot_h5ad && cfg.resolve_adata_path().is_empty() {
        let can_prompt = cfg!(feature = "tui") && !cli.plain;
        if !can_prompt {
            anyhow::bail!(
                "--plot-h5ad requires --h5ad or data.adata_path in spaceship_config.toml."
            );
        }
    }

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

    let mut path = expand_user_path(&cfg.data.adata_path);
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
    cfg.grn.tf_priors_feather = tf_priors_feather.clone();

    if !Path::new(&path).exists() {
        anyhow::bail!("Dataset not found at {}.", path);
    }

    let adata_path_for_stem = path.clone();

    if cfg.execution.output_dir.trim().is_empty() {
        cfg.execution.output_dir =
            default_output_dir_for_adata_path(Path::new(&adata_path_for_stem))?;
    }
    let output_dir_pb = PathBuf::from(expand_user_path(cfg.execution.output_dir.trim()));
    std::fs::create_dir_all(&output_dir_pb)?;
    cfg.execution.output_dir = output_dir_pb.to_string_lossy().to_string();

    if !cli.skip_auto_adata_prep
        && Path::new(&path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("h5ad"))
            .unwrap_or(false)
    {
        let magic_batch = spacetravlr::scanpy_preprocess::resolve_magic_batch_obs_column(
            None,
            cfg.data.condition.as_deref(),
        );
        spacetravlr::scanpy_preprocess::ensure_training_adata_ready(
            &mut cfg.data.adata_path,
            &output_dir_pb,
            Path::new(&adata_path_for_stem),
            magic_batch.as_deref(),
        )?;
        path = expand_user_path(&cfg.data.adata_path);
        cfg.data.adata_path = path.clone();
        if !Path::new(&path).exists() {
            anyhow::bail!("Dataset not found at {} after auto-prep.", path);
        }
    }

    if Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("h5ad"))
        .unwrap_or(false)
    {
        spacetravlr::spatial_estimator::materialize_canonical_training_adata(
            &mut path,
            &output_dir_pb,
            Path::new(&adata_path_for_stem),
            &cfg,
            network_data_dir.as_deref(),
        )?;
        cfg.data.adata_path = path.clone();
    }

    if cli.plot_h5ad {
        let color_by = spacetravlr::adata_terminal_scatter::resolve_plot_h5ad_color_column(
            Path::new(&path),
            cfg.data.cluster_annot.as_str(),
        )?;
        return spacetravlr::adata_terminal_scatter::print_h5ad_scatter(
            Path::new(&path),
            color_by.as_str(),
        );
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
        &compute_hardware_details(&compute),
        &compute_notice_text(&compute),
        &cfg,
        max_genes,
        gene_filter.as_deref(),
        condition_column.as_deref(),
    );

    if !use_dashboard {
        print_compute_notice(&compute);
        print_plain_preamble(
            &run_summary,
            &cfg,
            &path,
            &output_dir,
            mode_label,
            n_parallel,
        );
        if join_training {
            if condition_column.is_some() {
                println!(
                    "Join mode (condition): shared parent directory {}; each conditions/<group>/ uses .lock coordination",
                    output_dir
                );
            } else {
                println!(
                    "Join mode: shared directory {}; unfinished genes claimed via .lock; existing *_betadata.feather skipped",
                    output_dir
                );
            }
        }
        if let Some(condition_col) = condition_column.as_deref() {
            if !join_training {
                cfg.write_run_repro_toml_if_missing(Path::new(&output_dir))?;
            }
            let splits =
                prepare_condition_splits(&path, &output_dir, condition_col, join_training)?;
            println!(
                "Condition split: obs.{:?} -> {} groups (betadata under {}/conditions/<group>/)",
                condition_col,
                splits.len(),
                output_dir.trim_end_matches('/')
            );
            if join_training {
                let dir_status = scan_condition_status(&output_dir)?;
                if !dir_status.is_empty() {
                    println!("Condition status (from filesystem):");
                    for cs in &dir_status {
                        let status = if cs.n_locks > 0 {
                            "in progress"
                        } else if cs.n_done() > 0 {
                            "has results"
                        } else {
                            "not started"
                        };
                        println!(
                            "  {}: {} done ({} feather + {} orphan), {} active locks [{}]",
                            cs.label,
                            cs.n_done(),
                            cs.n_feathers,
                            cs.n_orphans,
                            cs.n_locks,
                            status,
                        );
                    }
                }
            }
            for split in splits {
                let split_output_dir = split.output_dir.display().to_string();
                let obs_subset = Arc::from(split.obs_indices.into_boxed_slice());
                println!(
                    "Running split '{}' ({} cells) -> {}",
                    split.label, split.n_obs, split_output_dir
                );
                let params = FitAllGenesParams {
                    path: &path,
                    obs_row_subset: Some(obs_subset),
                    radius: cfg.spatial.radius,
                    spatial_dim: cfg.spatial.spatial_dim,
                    contact_distance: cfg.spatial.contact_distance,
                    tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
                    max_ligands: cfg.grn.max_ligands,
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
                    output_dir: &split_output_dir,
                    model_export: &cfg.model_export,
                    hud: None,
                    network_data_dir: network_data_dir.clone(),
                    tf_priors_feather: tf_priors_feather.clone(),
                    write_minimal_repro_h5ad: cfg.execution.write_minimal_repro_h5ad,
                    spaceship_config: &cfg,
                    config_source_path: config_source_path.clone(),
                    join_training,
                };
                fit_all_genes_dispatch(&params, &compute)?;
            }
        } else {
            let params = FitAllGenesParams {
                path: &path,
                obs_row_subset: None,
                radius: cfg.spatial.radius,
                spatial_dim: cfg.spatial.spatial_dim,
                contact_distance: cfg.spatial.contact_distance,
                tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
                max_ligands: cfg.grn.max_ligands,
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
                write_minimal_repro_h5ad: cfg.execution.write_minimal_repro_h5ad,
                spaceship_config: &cfg,
                config_source_path: config_source_path.clone(),
                join_training,
            };
            fit_all_genes_dispatch(&params, &compute)?;
        }
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
        let condition_column_thread = condition_column.clone();
        let config_source_for_training = config_source_path.clone();

        let handle = thread::spawn(move || {
            if let Some(condition_col) = condition_column_thread {
                if !join_training {
                    cfg.write_run_repro_toml_if_missing(Path::new(&output_dir))?;
                }
                let splits =
                    prepare_condition_splits(&path, &output_dir, &condition_col, join_training)?;
                let n_splits = splits.len();
                for (si, split) in splits.into_iter().enumerate() {
                    let split_output_dir = split.output_dir.display().to_string();
                    let obs_subset = Arc::from(split.obs_indices.into_boxed_slice());
                    if let Ok(mut state) = hud_worker.lock() {
                        state.reset_for_new_split(
                            path.clone(),
                            split_output_dir.clone(),
                            Some((split.label.clone(), si + 1, n_splits)),
                        );
                    }
                    let params = FitAllGenesParams {
                        path: &path,
                        obs_row_subset: Some(obs_subset),
                        radius: cfg.spatial.radius,
                        spatial_dim: cfg.spatial.spatial_dim,
                        contact_distance: cfg.spatial.contact_distance,
                        tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
                        max_ligands: cfg.grn.max_ligands,
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
                        output_dir: &split_output_dir,
                        model_export: &cfg.model_export,
                        hud: Some(hud_worker.clone()),
                        network_data_dir: network_data_dir_thread.clone(),
                        tf_priors_feather: tf_priors_feather.clone(),
                        write_minimal_repro_h5ad: cfg.execution.write_minimal_repro_h5ad,
                        spaceship_config: &cfg,
                        config_source_path: config_source_for_training.clone(),
                        join_training,
                    };
                    fit_all_genes_dispatch(&params, &compute_thread)?;
                }
                Ok(())
            } else {
                let params = FitAllGenesParams {
                    path: &path,
                    obs_row_subset: None,
                    radius: cfg.spatial.radius,
                    spatial_dim: cfg.spatial.spatial_dim,
                    contact_distance: cfg.spatial.contact_distance,
                    tf_ligand_cutoff: cfg.grn.tf_ligand_cutoff,
                    max_ligands: cfg.grn.max_ligands,
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
                    write_minimal_repro_h5ad: cfg.execution.write_minimal_repro_h5ad,
                    spaceship_config: &cfg,
                    config_source_path: config_source_for_training,
                    join_training,
                };
                fit_all_genes_dispatch(&params, &compute_thread)
            }
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
