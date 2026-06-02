mod compute_backend;

use anyhow::Context;
use clap::builder::Styles;
use clap::builder::styling::AnsiColor;
use clap::{ArgAction, ColorChoice, Parser, Subcommand, ValueEnum};
use compute_backend::{
    ComputeChoice, FitAllGenesParams, compute_hardware_details, fit_all_genes_dispatch,
    select_compute_backend,
};
use serde_json::Value;
use spacetravlr::condition_split::{prepare_condition_splits, scan_condition_status};
use spacetravlr::config::{
    CnnOutputActivation, CnnTrainingMode, RUN_REPRO_TOML_FILENAME, SpaceshipConfig,
    canonical_adata_stem, canonical_training_prep_stem, default_output_dir_for_adata_path,
    expand_user_path,
};
use spacetravlr::grn_extra;
#[cfg(feature = "tui")]
use spacetravlr::training_demo::{
    DEMO_KIDNEY_SLIDETAGS_H5AD, DEMO_OUTPUT_DIR_LABEL, prepare_demo_hud, run_demo_training,
};
#[cfg(feature = "tui")]
use spacetravlr::training_hud::TrainingHudState;
use spacetravlr::training_hud::{RunConfigSummary, RunConfigSummaryBuildArgs};
#[cfg(feature = "tui")]
use spacetravlr::training_tui::{
    TrainingDashboardExit, run_dataset_paths_prompt, run_training_dashboard,
};
use spacetravlr::{
    BetadataCollectAggregate, RunSummaryParams, betadata_collect_interactions_all_cell_types,
    betadata_collect_interactions_all_cell_types_full, load_obs_column_for_collect_interactions,
    load_obs_for_collect_interactions, write_collected_interactions_feather,
    write_collected_interactions_full_feather, write_run_summary_html,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
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

• Load spaceship_config.toml from the repo build, install data/ next to the binary, or pass --config, then apply CLI overrides.
• Use --plain for compact line-oriented logs instead of the full-screen dashboard (when built with `tui`).
• Subcommand run-summary writes the HTML report without training.
• Subcommand collect-interactions builds a multi–cell-type interaction database from *_betadata.feather files.
• Subcommand gui runs `npm run build` in web/umap_lab, then starts the UMAP lab server and prints the URL.
• Use --map-labels with --reference and --query for MALT label transfer (requires uv on PATH; may download PyTorch on first run).
• Use --peek PATH (e.g. .h5ad or 10x .h5; alias --peak) for a compact summary: wrapped lines to terminal width, obs/var names in a small grid, human-only file size. Add --obs COL for value_counts on AnnData.
• Use --verify for a smoke test: download tonsil .h5ad (or local path), strip prep layers to force Rust full preprocess + MAGIC, parallel-2 full-mode train on AICDA and CD74, require WebGPU CNN backend unless SPACETRAVLR_VERIFY_ALLOW_CPU=1; confirms two betadata feathers; writes a plain-text log (hardware + checklist). Override log path with SPACETRAVLR_VERIFY_LOG. Needs curl and spaceship_config.toml (see --help)."#;

const SPACETRAVLR_AFTER_LONG_HELP: &str = r#"

Multi-host / shared storage
  Start a leader run (writes spacetravlr_run_repro.toml early), then use --join-output-dir DIR on other hosts with --parallel set per machine.
  Per-gene mean_lasso_r2 (and mean_cnn_r2 when used) are merged into spacetravlr_gene_performance.feather in the output directory under an advisory flock (no single-host-only step).

Condition splits
  With --condition, --join-output-dir points to the parent output directory; conditions/<group>/ subdirectories are auto-discovered from the repro TOML."#;

#[derive(clap::ValueEnum, Clone, Debug)]
enum TrainingModeArg {
    Full,
    Seed,
}

impl From<TrainingModeArg> for CnnTrainingMode {
    fn from(value: TrainingModeArg) -> Self {
        match value {
            TrainingModeArg::Full => CnnTrainingMode::Full,
            TrainingModeArg::Seed => CnnTrainingMode::Seed,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
enum MapLabelsExpressionMode {
    #[default]
    Auto,
    Counts,
    Lognorm,
}

impl MapLabelsExpressionMode {
    fn as_str(self) -> &'static str {
        match self {
            MapLabelsExpressionMode::Auto => "auto",
            MapLabelsExpressionMode::Counts => "counts",
            MapLabelsExpressionMode::Lognorm => "lognorm",
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, Eq, PartialEq)]
enum PlotUmapBackend {
    #[default]
    Rust,
    Scanpy,
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

#[cfg(feature = "rctd")]
use spacetravlr_rctd::{DeconvMode, RctdCliArgs, run_rctd};

#[cfg(feature = "rctd")]
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum RctdModeArg {
    Full,
    Doublet,
    Multi,
}

#[cfg(feature = "rctd")]
impl From<RctdModeArg> for DeconvMode {
    fn from(value: RctdModeArg) -> Self {
        match value {
            RctdModeArg::Full => DeconvMode::Full,
            RctdModeArg::Doublet => DeconvMode::Doublet,
            RctdModeArg::Multi => DeconvMode::Multi,
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Generate spacetravlr_run_summary.html (AnnData summary, config, optional manifest).
    RunSummary(RunSummaryCli),
    /// Scan *_betadata.feather under a run directory; aggregate β per modulator × target × cell type.
    CollectInteractions(CollectInteractionsCli),
    /// UMAP lab: build the web UI, start the API + static server, print the URL.
    Gui(GuiCli),
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
        help = "spaceship_config.toml overlay (merged on top of repo or install data/ base; omitted keys keep base values)"
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

#[derive(Parser, Debug, Clone)]
struct CollectInteractionsCli {
    #[arg(
        long,
        value_name = "PATH",
        help = "spacetravlr_run_repro.toml from the finished training run"
    )]
    run_toml: PathBuf,
    #[arg(
        long,
        default_value = "cell_type",
        help = "obs column for cell-type grouping"
    )]
    annot: String,
    #[arg(
        long,
        value_name = "COL",
        help = "obs column (e.g. cluster) — collect interactions independently per value; output includes a cluster column with mean/min/max/sum/positive/negative"
    )]
    cluster_col: Option<String>,
    #[arg(
        long,
        default_value = "mean",
        help = "mean|min|max|sum|positive|negative (ignored when --cluster-col is set)"
    )]
    aggregate: String,
    #[arg(
        long,
        value_name = "PATH",
        help = "Output .feather path (default: <[execution].output_dir>/plucked_feathers.feather)"
    )]
    out: Option<PathBuf>,
}

#[derive(Parser, Debug, Clone)]
struct GuiCli {
    #[arg(
        long,
        default_value = "127.0.0.1",
        help = "Listen address for the UMAP lab HTTP server"
    )]
    bind: String,
    #[arg(long, default_value_t = 8765, help = "TCP port")]
    port: u16,
    #[arg(
        long,
        action = ArgAction::SetTrue,
        help = "Do not run npm (use existing web/umap_lab/dist)"
    )]
    skip_npm: bool,
    #[arg(
        long,
        value_name = "PATH",
        help = "Passed through to umap_lab --static-dir (default: web/umap_lab/dist)"
    )]
    static_dir: Option<PathBuf>,
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
        help = "Download the latest release and replace spacetravlr, spacetravlr-perturb, and spatial_viewer next to this executable; also refreshes data/spaceship_config.toml and data/malt_label_transfer.py from GitHub (same layout as install.sh). Opt-in; uses the network only when you pass this flag. Requires build with `self-update`."
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
        long,
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Smoke test: curl SlideTags_human_tonsil.h5ad from GitHub (override with SPACETRAVLR_VERIFY_H5AD=…), copy with stripped normalized_count/imputed_count layers so training auto-prep runs Rust full preprocess (QC → normalize → HVG → PCA → UMAP → Leiden → MAGIC). Verify sets SPACETRAVLR_FORCE_KEEP_GENES=AICDA,CD74 on the subprocess so the target genes survive dispersion HVG. Full-mode training on AICDA and CD74 with --parallel 2 (2 epochs, spatial_dim 8 — minimum safe for three 2×2 max-pools). Captures training stderr: requires `CNN/compute backend = WebGPU` unless SPACETRAVLR_VERIFY_ALLOW_CPU=1. Confirms each gene’s *_betadata.feather with real β values. DB ligand cap defaults to 256 (`--max-lr`); set SPACETRAVLR_VERIFY_MAX_LR to override. SPACETRAVLR_VERIFY_SKIP_PREP_STRIP=1 skips the layer strip and the Rust prep/MAGIC log checks (uses raw .h5ad as-is). Writes a plain-text log (host CPU/RAM/swap, wgpu adapter, SPACETRAVLR_* env, checklist); default log under $TMPDIR, or SPACETRAVLR_VERIFY_LOG=/path/verify.log. Uses spaceship_config.toml under SPACETRAVLR_ROOT or the crate manifest dir; needs curl."
    )]
    verify: bool,

    #[arg(
        short = 'c',
        long,
        value_name = "PATH",
        help_heading = "Input",
        help = "spaceship_config.toml overlay (merged on top of repo or install data/ base; omitted keys keep base values)"
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
        long = "peek",
        visible_alias = "peak",
        value_name = "PATH",
        help_heading = "Input",
        help = "Peek: path/size/shape (wrapped); obs & var column names in a grid; other keys wrapped. --obs COL adds value_counts. HDF5 metadata only"
    )]
    peek: Option<PathBuf>,

    #[arg(
        long = "obs",
        value_name = "COLUMN",
        help_heading = "Input",
        help = "With --peek: load only this obs column and print value_counts (rank, count, %, category). With --plot-umap: color the terminal UMAP by this obs column (overrides `--leiden` default coloring; without `--obs`, defaults are auto cell_type/leiden, or `leiden` when `--plot-umap --leiden`)."
    )]
    obs: Option<String>,

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
        visible_alias = "max-lr",
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
        long = "train-modulators",
        value_name = "LIST",
        help_heading = "Gene list & GRN extras",
        help = "Ablation / subset: which GRN modulator groups to train (comma-separated). Tokens: tf (TF→target), lr (ligand–receptor), tfl or ltf (TF–ligand / NicheNet-style). Overrides [grn].use_*_modulators when set (same as [grn].train_modulators in TOML)"
    )]
    train_modulators: Option<String>,

    #[arg(
        long,
        value_name = "MODE",
        value_enum,
        help_heading = "Training",
        help = "seed | full CNN (default from config, usually seed)"
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
        long = "random-seed",
        value_name = "U64",
        help_heading = "Training",
        help = "Global RNG seed — Lasso (per target) and CNN minibatch shuffles ([execution].random_seed)"
    )]
    random_seed: Option<u64>,

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
        long = "mean-beta-lasso-prior-weight",
        value_name = "F",
        help_heading = "Training",
        help = "CNN auxiliary loss: weight on MSE(mean batch betas, lasso anchors) — overrides [cnn].mean_beta_lasso_prior_weight"
    )]
    mean_beta_lasso_prior_weight: Option<f64>,

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
        long = "spatial_dim",
        value_name = "N",
        help_heading = "Training",
        help = "CNN spatial map grid edge length (square H=W) — overrides [spatial].spatial_dim"
    )]
    spatial_dim: Option<usize>,

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
        long = "clean-output-dir",
        action = ArgAction::SetTrue,
        help_heading = "Output",
        help = "Remove the training output directory if it exists, then start a new run (cannot be used with --join-output-dir)"
    )]
    clean_output_dir: bool,

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
        long = "write-cnn-train-data-npz",
        action = ArgAction::SetTrue,
        help_heading = "Output",
        help = "Write {gene}_cnn_train_data.npz + _cnn_train_meta.json under CNN_weights/ for scripts/python_train_cnn.py (no SPACETRAVLR_DUMP_CNN_TRAIN_DATA env var)"
    )]
    write_cnn_train_data_npz: bool,

    #[arg(
        long,
        help_heading = "Interface",
        help = "Compact line-oriented logs instead of the full-screen dashboard (when built with `tui`)"
    )]
    plain: bool,

    #[arg(
        short = 'v',
        long,
        action = ArgAction::SetTrue,
        help_heading = "Interface",
        help = "Per gene, print each cluster's Lasso R² and CNN R² (before zeroing failed Lasso rows), plus pass / CNN weight-export skip flags vs [training].score_threshold"
    )]
    verbose: bool,

    #[arg(
        long,
        help_heading = "Interface",
        help = "Fake training dashboard only — no AnnData, no disk exports, no accelerator"
    )]
    demo: bool,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd",
        action = ArgAction::SetTrue,
        help_heading = "RCTD",
        help = "RCTD spatial deconvolution (GPL-3.0-or-later linked code); exits before training. Needs `--h5ad` (spatial) and `--ref-adata`."
    )]
    rctd: bool,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "ref-adata",
        value_name = "PATH",
        help_heading = "RCTD",
        help = "Reference .h5ad (single-cell or K×G with --ref-rows-are-types) or .rds"
    )]
    rctd_reference: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-obs-subset-file",
        value_name = "PATH",
        help_heading = "RCTD",
        help = "Optional text file: one spatial barcode per line (# comments OK). Restricts RCTD to this puck (e.g. barcodes after spacexr::create.RCTD)."
    )]
    rctd_obs_subset_file: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-genes-file",
        value_name = "PATH",
        help_heading = "RCTD",
        help = "One gene per line (# comments OK). For spacexr parity use internal_vars$gene_list_reg (RCTD regression genes), not necessarily all puck rows."
    )]
    rctd_genes_file: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-spatial-numi-tsv",
        value_name = "PATH",
        help_heading = "RCTD",
        help = "TSV with header obs<TAB>nUMI; one row per spatial barcode (spacexr puck@nUMI). Matches R when counts are gene-subset but nUMI is total per spot."
    )]
    rctd_spatial_numi_tsv: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-sigma-file",
        value_name = "PATH",
        help_heading = "RCTD",
        help = "Single float σ per line: build Q/SQ at runtime (matches spacexr internal_vars$sigma; no q_matrices.npz needed)."
    )]
    rctd_sigma_file: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-q-tsv",
        value_name = "PATH",
        help_heading = "RCTD",
        help = "Tab-separated Q_mat from spacexr (internal_vars$Q_mat); overrides --sigma / q_matrices.npz and --rctd-sigma-file."
    )]
    rctd_q_tsv: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-x-vals-tsv",
        value_name = "PATH",
        help_heading = "RCTD",
        help = "One X grid value per line (internal_vars$X_vals); required to match R Q_mat column count if it differs from the built-in grid."
    )]
    rctd_x_vals_tsv: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-skip-profile-normalize",
        action = ArgAction::SetTrue,
        help_heading = "RCTD",
        help = "Use reference profiles as-is (no per-type column L1 normalize). Use with spacexr cell_type_info$renorm[[1]] (K×G export) for parity."
    )]
    rctd_skip_profile_normalize: bool,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-k-val",
        default_value_t = 1000,
        value_name = "K",
        help_heading = "RCTD",
        help = "Poisson tail K for Q matrix rows (spacexr config K_val; use 100 to match default spacexr Reference)."
    )]
    rctd_k_val: i64,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "cell-type-col",
        value_name = "NAME",
        help_heading = "RCTD",
        default_value = "cell_type",
        help = "obs column with cell type labels (.h5ad single-cell reference only)"
    )]
    rctd_cell_type_col: String,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "ref-rows-are-types",
        action = ArgAction::SetTrue,
        help_heading = "RCTD",
        help = "Reference matrix rows are cell types (K×G), obs_names are type labels"
    )]
    rctd_ref_rows_are_types: bool,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "ref-cell-min",
        default_value_t = 25,
        value_name = "N",
        help_heading = "RCTD",
        help = "Minimum cells per type when aggregating single-cell reference"
    )]
    rctd_ref_cell_min: usize,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "ref-min-umi",
        default_value_t = 100,
        value_name = "N",
        help_heading = "RCTD",
        help = "Minimum mean UMI per type when aggregating reference"
    )]
    rctd_ref_min_umi: u32,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "ref-max-cells-per-type",
        default_value_t = 10000,
        value_name = "N",
        help_heading = "RCTD",
        help = "Max cells sampled per type for reference profiles"
    )]
    rctd_ref_max_cells_per_type: usize,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "q-matrices",
        value_name = "PATH",
        help_heading = "RCTD",
        help = "q_matrices.npz (default: ~/.cache/rctd/q_matrices.npz, downloaded on first run)"
    )]
    rctd_q_matrices: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "sigma",
        default_value_t = 100,
        value_name = "N",
        help_heading = "RCTD",
        help = "Q-matrix key (integer)"
    )]
    rctd_sigma: i32,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-mode",
        value_enum,
        default_value_t = RctdModeArg::Full,
        help_heading = "RCTD",
        help = "Deconvolution mode"
    )]
    rctd_mode: RctdModeArg,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-batch-size",
        default_value_t = 4096,
        value_name = "N",
        help_heading = "RCTD",
        help = "Batch size (spots per batch for outer RCTD loops)"
    )]
    rctd_batch_size: usize,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "rctd-output",
        value_name = "PREFIX",
        help_heading = "RCTD",
        help = "Optional output path prefix; writes PREFIX.weights.csv when set"
    )]
    rctd_output: Option<PathBuf>,

    #[cfg(feature = "rctd")]
    #[arg(
        long = "gpu",
        action = ArgAction::SetTrue,
        help_heading = "RCTD",
        help = "Use wgpu for RCTD (f32; not bit-identical vs CPU). Build with `--features rctd,rctd-wgpu`."
    )]
    rctd_gpu: bool,

    #[arg(
        long = "infer-species",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Print inferred species (human or mouse) from var gene symbols and exit (requires `--h5ad`)"
    )]
    infer_species: bool,

    #[arg(
        long = "plot-h5ad",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Print terminal spatial scatter (obsm spatial, obs colored by cluster column) and exit"
    )]
    plot_h5ad: bool,

    #[arg(
        long = "plot-umap",
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = "",
        help_heading = "Utility",
        help = "Auto-preprocess if needed, then print terminal UMAP scatter (obsm X_umap; obs coloring defaults to cell_type/leiden when present) and exit. Pass optional .h5ad on this flag (`--plot-umap data.h5ad`) or `--plot-umap` with `--h5ad data.h5ad`. If PATH is given on `--plot-umap`, that file is used (`--h5ad` is ignored for this command). With `--leiden`, runs Rust UMAP + Leiden in memory when UMAP or `obs['leiden']` is missing (no disk writes; default `--plot-umap-backend rust`); colors by `leiden` unless `--obs` is set (which wins). Missing UMAP: see `--plot-umap-backend`."
    )]
    plot_umap: Option<String>,

    #[arg(
        long = "plot-umap-backend",
        value_enum,
        default_value_t = PlotUmapBackend::Rust,
        help_heading = "Utility",
        help = "When obsm has no X_umap: `rust` (default) runs Rust QC→normalize_total(1e4)+log1p→HVG→PCA→UMAP; `scanpy` is legacy (embedded full_preprocess) — prefer `--process-h5ad` for Scanpy"
    )]
    plot_umap_backend: PlotUmapBackend,

    #[arg(
        long = "umap",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Rust: run QC → normalize_total(1e4)+log1p → HVG → PCA → UMAP (obsm['X_umap'], X_pca). Combine with `--leiden` / `--rust-magic`. Writes only when `--output PATH.h5ad` is set; otherwise in-memory only."
    )]
    prep_umap: bool,

    #[arg(
        long = "leiden",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Rust: Leiden on the UMAP fuzzy graph → obs['leiden'] (implies embedding). Often used with `--umap`. With `--plot-umap`, participates in the in-memory Rust prep path and defaults UMAP coloring to `leiden` when `--obs` is omitted. Writes only when `--output PATH.h5ad` is set."
    )]
    prep_leiden: bool,

    #[arg(
        long = "rust-magic",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Rust: clusterwise MAGIC → layers['imputed_count'] (implies UMAP graph). For Scanpy-only impute on existing `layers['normalized_count']` use `--impute`. Writes only when `--output PATH.h5ad` is set."
    )]
    prep_rust_magic: bool,

    #[arg(
        long = "map-labels",
        action = ArgAction::SetTrue,
        help_heading = "Map labels",
        help = "MALT: transfer labels from reference .h5ad to query .h5ad (writes obs['malt_label'], optional obs['leiden']/obs['leiden_celltype'], malt_labels.csv indexed by obs_name, plots, JSON under --map-labels-outdir; requires `uv` on PATH)."
    )]
    map_labels: bool,

    #[arg(
        short = 'r',
        long = "reference",
        value_name = "PATH",
        help_heading = "Map labels",
        help = "Reference AnnData .h5ad with label column in obs; use with `--map-labels`"
    )]
    reference: Option<PathBuf>,

    #[arg(
        short = 'q',
        long = "query",
        value_name = "PATH",
        help_heading = "Map labels",
        help = "Query AnnData .h5ad to receive predicted labels; use with `--map-labels`"
    )]
    query: Option<PathBuf>,

    #[arg(
        long = "map-labels-outdir",
        value_name = "DIR",
        default_value = "malt_out",
        help_heading = "Map labels",
        help = "Directory for labeled query .h5ad, marker_genes.json, run_meta.json, and figures"
    )]
    map_labels_outdir: PathBuf,

    #[arg(
        long = "map-labels-groupby",
        short = 'g',
        value_name = "OBS_COLUMN",
        action = ArgAction::Append,
        help_heading = "Map labels",
        help = "Reference obs column(s) for labels; comma-separated in one flag (e.g. -g cell_type,cell_type_fine) and/or repeat -g for multiple independent MALT runs (suffixed obs + CSV when >1 column). Omit for a single inferred column (cell_type, final_annotation, …)"
    )]
    map_labels_groupby: Vec<String>,

    #[arg(
        long = "map-labels-output-query",
        value_name = "NAME",
        default_value = "query_labeled.h5ad",
        help_heading = "Map labels",
        help = "Output filename under --map-labels-outdir for the labeled query"
    )]
    map_labels_output_query: String,

    #[arg(
        long = "map-labels-extra-markers",
        value_name = "GENES",
        help_heading = "Map labels",
        help = "Comma-separated gene symbols appended to dotplots when present in shared genes"
    )]
    map_labels_extra_markers: Option<String>,

    #[arg(
        long = "map-labels-expression-mode",
        value_enum,
        default_value_t = MapLabelsExpressionMode::Auto,
        help_heading = "Map labels",
        help = "auto | counts | lognorm — how MALT resolves counts vs log-normalized expression"
    )]
    map_labels_expression_mode: MapLabelsExpressionMode,

    #[arg(
        long = "map-labels-counts-layer",
        value_name = "LAYER",
        help_heading = "Map labels",
        help = "Use this adata.layers matrix as raw counts before normalize_total+log1p"
    )]
    map_labels_counts_layer: Option<String>,

    #[arg(
        long = "map-labels-prefer-raw-counts",
        action = ArgAction::SetTrue,
        help_heading = "Map labels",
        help = "When resolving counts, try AnnData.raw after standard count layers"
    )]
    map_labels_prefer_raw_counts: bool,

    #[arg(
        long = "map-labels-no-leiden",
        action = ArgAction::SetTrue,
        help_heading = "Map labels",
        help = "Skip adaptive Leiden + leiden_celltype mapping after MALT (passes --no-leiden-map to the script)"
    )]
    map_labels_no_leiden: bool,

    #[arg(
        long = "map-labels-reference-gene-list",
        value_name = "PATH",
        help_heading = "Map labels",
        help = "One gene symbol per line (count = reference n_vars); passed as --reference-gene-list to MALT when reference var_names are placeholders"
    )]
    map_labels_reference_gene_list: Option<PathBuf>,

    #[arg(
        long = "map-labels-spatial",
        action = ArgAction::SetTrue,
        help_heading = "Map labels",
        help = "Enable spatial MALT: dotplot-selected transfer genes, SpaceTravLR betadata priors, and spatial-neighbor smoothing."
    )]
    map_labels_spatial: bool,

    #[arg(
        long = "map-labels-reference-betadata-dir",
        value_name = "DIR",
        help_heading = "Map labels",
        help = "Reference SpaceTravLR output directory containing seed *_betadata.feather files for selected genes."
    )]
    map_labels_reference_betadata_dir: Option<PathBuf>,

    #[arg(
        long = "map-labels-query-betadata-dir",
        value_name = "DIR",
        help_heading = "Map labels",
        help = "Query SpaceTravLR output directory containing seed *_betadata.feather files for selected genes."
    )]
    map_labels_query_betadata_dir: Option<PathBuf>,

    #[arg(
        long = "map-labels-benchmark-truth",
        value_name = "OBS_COLUMN",
        help_heading = "Map labels",
        help = "Optional query obs column with ground-truth labels; writes accuracy/ARI for KNN, MALT, and spatial MALT."
    )]
    map_labels_benchmark_truth: Option<String>,

    #[arg(
        long = "map-labels-spatial-genes-per-type",
        value_name = "N",
        default_value_t = 6,
        help_heading = "Map labels",
        help = "Number of compact dotplot-optimized training genes to select per reference cell type for spatial MALT."
    )]
    map_labels_spatial_genes_per_type: usize,

    #[arg(
        long = "rust-process-h5ad",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Rust preprocessing (QC → normalize_total(1e4)+log1p → HVG → PCA → HNSW KNN → UMAP → Leiden if needed → MAGIC if needed) → `<stem>_rust_processed.h5ad` (requires `--h5ad`). For modular outputs use `--umap` / `--leiden` / `--rust-magic`."
    )]
    rust_process_h5ad: bool,

    #[arg(
        long = "rust-n-top-hvg",
        value_name = "N",
        help_heading = "Utility",
        help = "With `--rust-process-h5ad`: number of highly variable genes (default 2000)."
    )]
    rust_n_top_hvg: Option<usize>,

    #[arg(
        long = "rust-n-neighbors",
        value_name = "N",
        help_heading = "Utility",
        help = "With `--rust-process-h5ad`: UMAP / KNN n_neighbors (default 15)."
    )]
    rust_n_neighbors: Option<usize>,

    #[arg(
        long = "process-h5ad",
        alias = "process_h5ad",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Full pipeline: uv Scanpy (QC → UMAP/Leiden) + clusterwise magic-impute → `<stem>_processed.h5ad` (requires `--h5ad`)."
    )]
    process_h5ad: bool,

    #[arg(
        long = "process-output-dir",
        value_name = "DIR",
        help_heading = "Utility",
        help = "With `--process-h5ad` / `--impute` / `--rust-process-h5ad`: directory for the derived `<stem>_*.h5ad` (default: cwd). Rust convenience flags (`--umap` / `--leiden` / `--rust-magic`) do not write here unless you pass `--output`. `--plot-umap` with default `--plot-umap-backend rust` stays in memory (no temp `.h5ad`); legacy `--plot-umap-backend scanpy` may write under this directory or the system temp dir."
    )]
    process_output_dir: Option<PathBuf>,

    #[arg(
        short = 'o',
        long = "output",
        value_name = "PATH",
        help_heading = "Utility",
        help = "With `--umap` / `--leiden` / `--rust-magic`: write the preprocessed AnnData to this `.h5ad` path (parents created). Omit to run in memory only (no prep `.h5ad` on disk; `--process-output-dir` is not used for that mode)."
    )]
    rust_prep_output: Option<PathBuf>,

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

    #[arg(
        long = "skip-spatial-microns",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "With `--process-h5ad`: skip heuristic scaling of obsm spatial coordinates to microns"
    )]
    skip_spatial_microns: bool,

    #[arg(
        long = "spatial-species",
        value_name = "SPECIES",
        help_heading = "Utility",
        help = "With `--process-h5ad` / `--celloracle` preprocess: `human` or `mouse` for spatial → µm prior; omit to infer from var gene symbols (see infer_species)"
    )]
    spatial_species: Option<String>,

    #[arg(
        long = "spatial-microns-target-um",
        value_name = "UM",
        help_heading = "Utility",
        help = "With `--process-h5ad`: override assumed median k-NN distance in µm (otherwise species default)"
    )]
    spatial_microns_target_um: Option<f64>,

    #[arg(
        long = "celloracle",
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = "",
        help_heading = "Utility",
        help = "CellOracle-style TF prior inference only: run Bayesian ridge GRN (SpaceTravLR priors), write TF-prior Feather (source, target, cell_type), and exit. Pass optional .h5ad path, or omit PATH and use `--h5ad` for the same file"
    )]
    celloracle: Option<String>,

    #[arg(
        long = "celloracle-output",
        value_name = "PATH",
        help_heading = "Utility",
        help = "Output Feather path for `--celloracle` (default: <output-dir>/<adata_stem>_celloracle_tf_priors.feather)"
    )]
    celloracle_output: Option<PathBuf>,

    #[arg(
        long = "celloracle-output-dir",
        value_name = "DIR",
        help_heading = "Utility",
        help = "Directory for default `--celloracle` Feather name and for auto preprocess (imputed .h5ad); defaults to cwd or `--process-output-dir` when set"
    )]
    celloracle_output_dir: Option<PathBuf>,

    #[arg(
        long = "celloracle-layer",
        value_name = "NAME",
        default_value = "imputed_count",
        help_heading = "Utility",
        help = "AnnData layer for expression (`X` if the layer is missing and X is dense/CSR)"
    )]
    celloracle_layer: String,

    #[arg(
        long = "celloracle-skip-preprocess",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Do not auto-impute / patch AnnData before reading (use raw path as-is)"
    )]
    celloracle_skip_preprocess: bool,

    #[arg(
        long = "celloracle-per-cluster",
        action = ArgAction::SetTrue,
        help_heading = "Utility",
        help = "Infer edges per cell type (or `--celloracle-obs-key` column) instead of one global model"
    )]
    celloracle_per_cluster: bool,

    #[arg(
        long = "celloracle-obs-key",
        value_name = "NAME",
        default_value = "cell_type",
        help_heading = "Utility",
        help = "obs column for `--celloracle-per-cluster`"
    )]
    celloracle_obs_key: String,

    #[arg(
        long = "celloracle-species",
        value_name = "SPECIES",
        help_heading = "Utility",
        help = "GRN species (`human` / `mouse`); default: infer from gene symbols"
    )]
    celloracle_species: Option<String>,

    #[arg(
        long = "celloracle-network-data-dir",
        value_name = "DIR",
        help_heading = "Utility",
        help = "Directory with `{species}_network.parquet` (overrides config / SPACETRAVLR_DATA_DIR)"
    )]
    celloracle_network_data_dir: Option<PathBuf>,

    #[arg(
        long = "celloracle-p-max",
        value_name = "P",
        help_heading = "Utility",
        help = "Keep only edges with p ≤ this threshold before writing Feather"
    )]
    celloracle_p_max: Option<f64>,

    #[arg(
        long = "celloracle-threads",
        value_name = "N",
        help_heading = "Utility",
        help = "Rayon thread count for `--celloracle` (omit for default)"
    )]
    celloracle_threads: Option<usize>,
}

fn apply_cli_join_overrides(cli: &Cli, cfg: &mut SpaceshipConfig) -> anyhow::Result<()> {
    if let Some(v) = cli.parallel {
        cfg.execution.n_parallel = v.max(1);
    }
    if let Some(s) = cli.random_seed {
        cfg.execution.random_seed = s;
    }
    if cli.save_cnn_weights {
        cfg.model_export.save_cnn_weights = true;
    }
    if cli.write_cnn_train_data_npz {
        cfg.model_export.write_cnn_train_data_npz = true;
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
    if let Some(s) = cli.random_seed {
        cfg.execution.random_seed = s;
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
    if let Some(v) = cli.mean_beta_lasso_prior_weight {
        cfg.cnn.mean_beta_lasso_prior_weight = v;
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
    if let Some(v) = cli.spatial_dim {
        cfg.spatial.spatial_dim = v.max(1);
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
    if cli.write_cnn_train_data_npz {
        cfg.model_export.write_cnn_train_data_npz = true;
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
    if let Some(raw) = cli.train_modulators.as_deref() {
        cfg.grn.train_modulators = Some(raw.trim().to_string());
    }
    cfg.grn.apply_train_modulators_shorthand()?;
    Ok(())
}

fn validate_join_cli_against_repro(
    cli: &Cli,
    cfg: &SpaceshipConfig,
    repro: &Path,
    err_prefix: &str,
) -> anyhow::Result<()> {
    if let Some(cli_k) = cli.max_ligands {
        let expected = cli_k.max(1);
        if cfg.grn.max_ligands != Some(expected) {
            anyhow::bail!(
                "{err_prefix} --max-ligands / --max-lr {} does not match [grn].max_ligands ({:?}) in {}.\n\
                 Join-style training uses the repro TOML as the single source of truth; omit those flags, or set [grn].max_ligands the same on the leader run.",
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
                        "{err_prefix} --condition {:?} does not match [data].condition = {:?} in {}; omit --condition to use the file, or fix the mismatch.",
                        cli_c,
                        file_c,
                        repro.display()
                    );
                }
            }
        }
    }
    Ok(())
}

fn eprint_join_style_resume_cli_notes(cli: &Cli, repro: &Path, explicit_join_flag: bool) {
    let via = if explicit_join_flag {
        "--join-output-dir"
    } else {
        "existing output directory"
    };
    if cli.config.is_some() {
        eprintln!(
            "Note: {via}: primary training contract is {}; --config overlays it; repo or install data/spaceship_config.toml fills keys missing from the repro.",
            repro.display()
        );
    }
    if cli.max_genes.is_some() || cli.genes.is_some() {
        eprintln!(
            "Note: {via}: [training] genes / max_genes come from {}; --genes and --max-genes are ignored.",
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
        || cli.spatial_dim.is_some()
        || cli.train_modulators.is_some()
    {
        eprintln!(
            "Note: {via}: hyperparameter / output CLI flags are ignored except --parallel (using repro TOML)."
        );
    }
}

fn load_config_for_main(cli: &Cli) -> anyhow::Result<(SpaceshipConfig, bool)> {
    if cli.clean_output_dir && cli.join_output_dir.is_some() {
        anyhow::bail!("--clean-output-dir cannot be used with --join-output-dir.");
    }
    if let Some(j) = cli.join_output_dir.as_ref() {
        let jexp = expand_user_path(j.to_string_lossy().as_ref());
        let repro = Path::new(&jexp).join(RUN_REPRO_TOML_FILENAME);
        if !repro.is_file() {
            anyhow::bail!(
                "--join-output-dir: missing run config {} (start a leader run on this directory first, or copy the TOML from the primary host)",
                repro.display()
            );
        }
        let mut cfg = SpaceshipConfig::from_run_repro_merged(&repro, cli.config.as_deref())?;
        validate_join_cli_against_repro(cli, &cfg, &repro, "--join-output-dir:")?;
        cfg.execution.output_dir = jexp;
        apply_cli_join_overrides(cli, &mut cfg)?;
        eprint_join_style_resume_cli_notes(cli, &repro, true);
        Ok((cfg, true))
    } else {
        let mut cfg = SpaceshipConfig::try_load_merged(cli.config.as_deref())?;
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

fn plain_trim_chars(s: &str, max: usize) -> String {
    let t = s.trim();
    let n = t.chars().count();
    if n <= max {
        return t.to_string();
    }
    let keep = max.saturating_sub(1);
    t.chars().take(keep).collect::<String>() + "…"
}

fn print_plain_preamble(
    summary: &RunConfigSummary,
    cfg: &SpaceshipConfig,
    dataset: &str,
    output_dir: &str,
    mode: &str,
    n_parallel: usize,
) {
    let dev = plain_trim_chars(&summary.compute_device_detail, 88);
    println!(
        "plain  {}  {}×w  {}ep  {}  {}",
        mode, n_parallel, summary.epochs_per_gene, summary.compute_backend, dev
    );
    println!("data {}", dataset);
    println!("out  {}", output_dir);
    let repro = if cfg.execution.write_minimal_repro_h5ad {
        "repro:h5ad"
    } else {
        "repro:off"
    };
    let cfg_line = plain_trim_chars(&summary.config_source, 96);
    println!(
        "cfg {}  {}  layer={} obs={}  r={} dim={} cd={} wlig={}  l1={:.0e} g={:.0e} n={} tol={:.0e}  CNN={} lr={:.0e} thr={}  GRN tf≤{} L={} mods={}  genes: {}",
        cfg_line,
        repro,
        summary.layer,
        summary.cluster_annot,
        summary.spatial_radius,
        summary.spatial_dim,
        summary.contact_distance,
        summary.weighted_ligand_scale_factor,
        summary.l1_reg,
        summary.group_reg,
        summary.n_iter,
        summary.tol,
        summary.cnn_training_mode,
        summary.learning_rate,
        summary.score_threshold,
        summary.tf_ligand_cutoff,
        summary.max_ligands,
        grn_modulator_label(cfg),
        summary.gene_selection,
    );
}

fn run_run_summary(cli: &Cli, rs: &RunSummaryCli) -> anyhow::Result<()> {
    let cfg = SpaceshipConfig::try_load_merged(
        rs.config
            .as_ref()
            .or(cli.config.as_ref())
            .map(|p| p.as_path()),
    )?;

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

fn run_collect_interactions(ci: &CollectInteractionsCli) -> anyhow::Result<()> {
    let cfg = SpaceshipConfig::from_file(&ci.run_toml)?;
    let annot_col = ci.annot.trim();
    anyhow::ensure!(
        !annot_col.is_empty(),
        "--annot must be a non-empty obs column name"
    );
    let ctx = load_obs_for_collect_interactions(ci.run_toml.as_path(), annot_col)?;
    let run_output_dir = cfg.resolve_training_output_dir(ci.run_toml.as_path());
    let dir_s = run_output_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("training output directory path must be UTF-8"))?;
    let out_path = ci
        .out
        .clone()
        .unwrap_or_else(|| run_output_dir.join("plucked_feathers.feather"));
    let out_s = out_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("output path must be UTF-8"))?;

    if let Some(ref cluster_col) = ci.cluster_col {
        let col = cluster_col.trim();
        anyhow::ensure!(!col.is_empty(), "--cluster-col must be non-empty");
        let cluster_obs =
            load_obs_column_for_collect_interactions(ci.run_toml.as_path(), col)?;
        let rows = betadata_collect_interactions_all_cell_types_full(
            dir_s,
            &ctx.obs_names,
            &ctx.cluster_keys,
            &ctx.cell_type_labels,
            None,
            Some(cluster_obs.as_slice()),
        )?;
        write_collected_interactions_full_feather(out_s, &rows)?;
        eprintln!(
            "Wrote {} rows (per-cluster β, column {:?}) to {}",
            rows.len(),
            col,
            out_path.display()
        );
        return Ok(());
    }

    let mode = BetadataCollectAggregate::parse(ci.aggregate.trim()).ok_or_else(|| {
        anyhow::anyhow!(
            "aggregate must be mean|min|max|sum|positive|negative (got {:?})",
            ci.aggregate
        )
    })?;
    let rows = betadata_collect_interactions_all_cell_types(
        dir_s,
        &ctx.obs_names,
        &ctx.cluster_keys,
        &ctx.cell_type_labels,
        mode,
        None,
    )?;
    write_collected_interactions_feather(out_s, &rows)?;
    eprintln!("Wrote {} rows to {}", rows.len(), out_path.display());
    Ok(())
}

#[cfg(feature = "tui")]
fn run_demo_mode(cli: &Cli) -> anyhow::Result<()> {
    if cli.plain {
        anyhow::bail!("--demo is for the full-screen dashboard; omit --plain.");
    }

    let mut cfg = SpaceshipConfig::try_load_merged(cli.config.as_deref())?;
    apply_cli_to_config(cli, &mut cfg)?;
    if matches!(cfg.resolved_cnn_mode(), CnnTrainingMode::Seed) {
        cfg.training.mode = Some(CnnTrainingMode::Full);
    }

    let gene_filter = cfg.training.genes.clone();
    let demo_total = cfg.training.max_genes.unwrap_or(16).clamp(1, 512);

    let config_path_ref = cli.config.as_deref();
    let run_summary = RunConfigSummary::build(RunConfigSummaryBuildArgs {
        config_path: config_path_ref,
        compute_backend: "demo",
        compute_device_detail: "— (demo; no accelerator)",
        compute_notice: "Demo mode — kidney slide-tags path is display-only; obsm['spatial'] from embedded cache; simulated genes/workers; no AnnData load, no betadata export, no training backend.",
        cfg: &cfg,
        max_genes: Some(demo_total),
        gene_filter: gene_filter.as_deref(),
        condition_split: None,
    });

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

fn expand_map_labels_groupby_columns(columns: &[String]) -> Vec<String> {
    columns
        .iter()
        .flat_map(|s| {
            s.split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
        })
        .collect()
}

fn run_map_labels(cli: &Cli) -> anyhow::Result<()> {
    let reference = cli
        .reference
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--map-labels requires `--reference PATH` (or `-r`)"))?;
    let query = cli
        .query
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--map-labels requires `--query PATH` (or `-q`)"))?;
    let reference = PathBuf::from(expand_user_path(reference.to_string_lossy().as_ref()));
    let query = PathBuf::from(expand_user_path(query.to_string_lossy().as_ref()));
    let outdir = PathBuf::from(expand_user_path(
        cli.map_labels_outdir.to_string_lossy().as_ref(),
    ));
    std::fs::create_dir_all(&outdir)?;
    if !reference.is_file() {
        anyhow::bail!("Reference AnnData not found at {}.", reference.display());
    }
    if !query.is_file() {
        anyhow::bail!("Query AnnData not found at {}.", query.display());
    }
    eprintln!(
        "spacetravlr: map-labels (MALT) via uv; writing under {}",
        outdir.display()
    );
    let groupby_expanded = expand_map_labels_groupby_columns(&cli.map_labels_groupby);
    let reference_betadata_dir = cli
        .map_labels_reference_betadata_dir
        .as_ref()
        .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())));
    let query_betadata_dir = cli
        .map_labels_query_betadata_dir
        .as_ref()
        .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())));
    spacetravlr::malt_label_transfer::run_map_labels(
        spacetravlr::malt_label_transfer::MapLabelsParams {
            reference: &reference,
            query: &query,
            outdir: &outdir,
            groupby: &groupby_expanded,
            output_query: &cli.map_labels_output_query,
            extra_markers: cli.map_labels_extra_markers.as_deref(),
            expression_mode: cli.map_labels_expression_mode.as_str(),
            counts_layer: cli.map_labels_counts_layer.as_deref(),
            prefer_raw_counts: cli.map_labels_prefer_raw_counts,
            leiden_map: !cli.map_labels_no_leiden,
            reference_gene_list: cli.map_labels_reference_gene_list.as_deref(),
            spatial: cli.map_labels_spatial,
            reference_betadata_dir: reference_betadata_dir.as_deref(),
            query_betadata_dir: query_betadata_dir.as_deref(),
            benchmark_truth: cli.map_labels_benchmark_truth.as_deref(),
            spatial_genes_per_type: Some(cli.map_labels_spatial_genes_per_type),
        },
    )?;
    Ok(())
}

fn resolve_rust_preprocess_params(cli: &Cli) -> spacetravlr::rust_preprocess::RustPreprocessParams {
    let mut params = spacetravlr::rust_preprocess::RustPreprocessParams::default();
    if let Some(n) = cli.rust_n_top_hvg {
        params.n_top_hvg = n;
    }
    if let Some(n) = cli.rust_n_neighbors {
        params.n_neighbors = n;
    }
    params
}

fn run_rust_prep_convenience(cli: &Cli) -> anyhow::Result<()> {
    let h5ad_ref = cli.h5ad.as_ref().ok_or_else(|| {
        anyhow::anyhow!("`--umap` / `--leiden` / `--rust-magic` require `--h5ad PATH`")
    })?;
    if let Some(ref raw_out) = cli.rust_prep_output {
        let dest = PathBuf::from(expand_user_path(raw_out.to_string_lossy().as_ref()));
        if !dest
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("h5ad"))
            .unwrap_or(false)
        {
            anyhow::bail!(
                "`--output` / `-o` must be a path ending in .h5ad (got {})",
                dest.display()
            );
        }
    }
    let h5ad = PathBuf::from(expand_user_path(h5ad_ref.to_string_lossy().as_ref()));
    if !h5ad.is_file() {
        anyhow::bail!("AnnData not found at {}.", h5ad.display());
    }
    let params = resolve_rust_preprocess_params(cli);
    let steps = spacetravlr::rust_preprocess::RustPreprocessSteps::from_convenience_flags(
        cli.prep_umap,
        cli.prep_leiden,
        cli.prep_rust_magic,
    );
    if let Some(ref raw_out) = cli.rust_prep_output {
        let dest = PathBuf::from(expand_user_path(raw_out.to_string_lossy().as_ref()));
        if let Some(parent) = dest.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create parent dirs for {}", dest.display()))?;
            }
        }
        spacetravlr::rust_preprocess::rust_preprocess_h5ad_with_steps(
            &h5ad,
            Some(dest.as_path()),
            &params,
            &steps,
        )?;
        eprintln!("spacetravlr: wrote {}", dest.display());
    } else {
        spacetravlr::rust_preprocess::rust_preprocess_h5ad_with_steps(
            &h5ad, None, &params, &steps,
        )?;
        eprintln!("spacetravlr: no --output (-o); skipped writing .h5ad");
    }
    Ok(())
}

fn run_rust_process_h5ad(cli: &Cli) -> anyhow::Result<()> {
    let h5ad = cli
        .h5ad
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--rust-process-h5ad requires `--h5ad PATH`"))?;
    let h5ad = PathBuf::from(expand_user_path(h5ad.to_string_lossy().as_ref()));
    if !h5ad.is_file() {
        anyhow::bail!("AnnData not found at {}.", h5ad.display());
    }
    let out_dir = match &cli.process_output_dir {
        Some(p) => PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())),
        None => std::env::current_dir().context("process-output-dir default (cwd)")?,
    };
    std::fs::create_dir_all(&out_dir)?;
    let stem = canonical_training_prep_stem(&h5ad);
    let dest = out_dir.join(format!("{stem}_rust_processed.h5ad"));

    let params = resolve_rust_preprocess_params(cli);

    spacetravlr::rust_preprocess::rust_preprocess_h5ad(&h5ad, &dest, &params)?;
    eprintln!("spacetravlr: wrote {}", dest.display());
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
    let stem = canonical_training_prep_stem(&h5ad);
    let dest = spacetravlr::scanpy_preprocess::training_processed_h5ad_path(&out_dir, &stem);
    let batch_owned = spacetravlr::scanpy_preprocess::resolve_magic_batch_obs_column(
        cli.magic_batch_obs.as_deref(),
        cli.condition.as_deref(),
    );
    let batch = batch_owned.as_deref();
    let spatial_microns = spacetravlr::scanpy_preprocess::SpatialMicronsOptions {
        skip: cli.skip_spatial_microns,
        species: cli
            .spatial_species
            .as_deref()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .unwrap_or_default(),
        target_median_nn_um: cli.spatial_microns_target_um,
    };
    // #region agent log
    spacetravlr::scanpy_preprocess::agent_debug_ndjson(
        "B",
        "spacetravlr.rs:run_process_h5ad",
        "CLI --process-h5ad spatial microns before full_preprocess",
        "preprocess",
        serde_json::json!({
            "h5ad": h5ad.to_string_lossy(),
            "spatial_species_cli": cli.spatial_species.as_deref().unwrap_or(""),
            "spatial_microns_skip": cli.skip_spatial_microns,
            "spatial_microns_target_um_cli": cli.spatial_microns_target_um,
        }),
    );
    // #endregion
    if spacetravlr::scanpy_preprocess::prepared_training_output_is_reusable(&h5ad, &dest)? {
        eprintln!(
            "spacetravlr: reusing existing {} (>= mtime of {})",
            dest.display(),
            h5ad.display()
        );
        return Ok(());
    }
    let (_out, log) = spacetravlr::scanpy_preprocess::full_preprocess_maybe_log(
        &h5ad,
        &dest,
        true,
        batch,
        spatial_microns,
        true,
    )?;
    if let Some(l) = log {
        eprint!("{l}");
    }
    Ok(())
}

fn run_impute(cli: &Cli) -> anyhow::Result<()> {
    use spacetravlr::scanpy_preprocess::{
        magic_impute_and_attach_batch, training_imputed_h5ad_path,
    };

    let h5ad = cli
        .h5ad
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--impute requires `--h5ad PATH`"))?;
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
    let log = magic_impute_and_attach_batch(&h5ad, &out, batch_owned.as_deref(), true, true)?;
    eprint!("{log}");
    Ok(())
}

fn resolve_plot_umap_h5ad(cli: &Cli) -> anyhow::Result<PathBuf> {
    let raw = cli.plot_umap.as_deref().unwrap_or("");
    let trimmed = raw.trim();
    if !trimmed.is_empty() {
        let p = PathBuf::from(expand_user_path(trimmed));
        if !p.is_file() {
            anyhow::bail!(
                "`--plot-umap` PATH is not an existing file: {}",
                p.display()
            );
        }
        return Ok(p);
    }
    let cand = cli
        .h5ad
        .as_ref()
        .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())));
    match cand {
        Some(p) if p.is_file() => Ok(p),
        Some(p) => anyhow::bail!(
            "`--plot-umap` without PATH requires `--h5ad` to an existing .h5ad file; not found: {}",
            p.display()
        ),
        None => anyhow::bail!(
            "`--plot-umap` requires PATH on the flag (`--plot-umap data.h5ad`) or `--h5ad PATH` to an existing .h5ad file."
        ),
    }
}

fn run_plot_umap(cli: &Cli) -> anyhow::Result<()> {
    let color_obs = cli.obs.as_deref().map(str::trim).filter(|s| !s.is_empty());
    if cli.obs.is_some() && color_obs.is_none() {
        anyhow::bail!("--obs must be a non-empty column name when used with --plot-umap");
    }

    if cli.prep_leiden && matches!(cli.plot_umap_backend, PlotUmapBackend::Scanpy) {
        anyhow::bail!(
            "--plot-umap --leiden requires the Rust in-memory path: use default `--plot-umap-backend rust` (omit scanpy)."
        );
    }

    let h5ad = resolve_plot_umap_h5ad(cli)?;

    fn h5ad_obsm_has_umap(path: &Path) -> anyhow::Result<bool> {
        if !path.is_file() {
            return Ok(false);
        }
        use anndata::{AnnDataOp, AxisArraysOp, Backend};
        let a = anndata::AnnData::<anndata_hdf5::H5>::open(anndata_hdf5::H5::open(path)?)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let found = a.obsm().keys().iter().any(|k| k == "X_umap" || k == "umap");
        a.close()?;
        Ok(found)
    }

    let has_umap = h5ad_obsm_has_umap(&h5ad)?;

    let need_mem_preprocess_for_leiden = cli.prep_leiden
        && color_obs.is_none()
        && !spacetravlr::adata_terminal_scatter::h5ad_obs_column_exists(&h5ad, "leiden")?;
    let plot_from_disk = has_umap && !need_mem_preprocess_for_leiden;

    if plot_from_disk {
        eprintln!(
            "obsm already has UMAP in {}; plotting directly.",
            h5ad.display()
        );
        let disk_color = match color_obs {
            Some(c) => Some(c),
            None if cli.prep_leiden => Some("leiden"),
            None => None,
        };
        return spacetravlr::adata_terminal_scatter::print_h5ad_umap_scatter(&h5ad, disk_color);
    }

    match cli.plot_umap_backend {
        PlotUmapBackend::Rust => {
            let phase = if need_mem_preprocess_for_leiden {
                "UMAP + Leiden (in memory; file had UMAP but no obs['leiden'])"
            } else if cli.prep_leiden {
                "UMAP + Leiden"
            } else {
                "UMAP"
            };
            eprintln!(
                "No usable on-disk UMAP plot path — running Rust preprocess ({phase}) on {} …",
                h5ad.display()
            );
            let params = resolve_rust_preprocess_params(cli);
            let steps = spacetravlr::rust_preprocess::RustPreprocessSteps::from_convenience_flags(
                true,
                cli.prep_leiden,
                false,
            );
            let adata = spacetravlr::rust_preprocess::rust_preprocess_h5ad_to_memory(
                &h5ad, &params, &steps,
            )?;

            use anndata::ArrayData;
            use anndata::data::ArrayConvert;
            let umap_elem = adata
                .obsm()
                .get_array("X_umap")
                .map_err(|e| anyhow::anyhow!("obsm X_umap: {e}"))?;
            let umap_data = umap_elem
                .get_data()
                .map_err(|e| anyhow::anyhow!("obsm X_umap data: {e}"))?;
            let umap_coords: ndarray::Array2<f64> = match umap_data {
                ArrayData::Array(d) => d
                    .try_convert()
                    .map_err(|e| anyhow::anyhow!("convert X_umap to Array2<f64>: {e}"))?,
                _ => anyhow::bail!("obsm['X_umap'] is not a dense array after preprocessing"),
            };

            let obs_df = adata.obs().get_data();
            let try_cols: &[&str] = if cli.prep_leiden && color_obs.is_none() {
                &["leiden", "cell_type"]
            } else {
                &["cell_type", "leiden"]
            };
            let obs_labels: Option<(String, Vec<String>)> = if let Some(col) = color_obs {
                let series = obs_df
                    .column(col)
                    .map_err(|_| anyhow::anyhow!("obs column {:?} not found", col))?
                    .as_materialized_series();
                let vals: Vec<String> = (0..series.len())
                    .map(|i| spacetravlr::betadata::obs_series_row_str(series, i))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                Some((col.to_string(), vals))
            } else {
                let mut found = None;
                for col_name in try_cols {
                    if let Ok(c) = obs_df.column(col_name) {
                        let series = c.as_materialized_series();
                        if let Ok(vals) = (0..series.len())
                            .map(|i| spacetravlr::betadata::obs_series_row_str(series, i))
                            .collect::<anyhow::Result<Vec<_>>>()
                        {
                            found = Some((col_name.to_string(), vals));
                            break;
                        }
                    }
                }
                found
            };

            let color_arg = obs_labels
                .as_ref()
                .map(|(name, vals)| (name.as_str(), vals.as_slice()));
            spacetravlr::adata_terminal_scatter::print_umap_scatter_from_arrays(
                &umap_coords,
                color_arg,
                &h5ad.display().to_string(),
            )
        }
        PlotUmapBackend::Scanpy => {
            eprintln!(
                "warning: --plot-umap-backend scanpy is legacy; use `--process-h5ad` for full Scanpy, or omit this flag for Rust (default)."
            );
            eprintln!(
                "No UMAP in obsm — running full Scanpy preprocess on {} …",
                h5ad.display()
            );
            let out_dir = match &cli.process_output_dir {
                Some(p) => PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())),
                None => std::env::temp_dir(),
            };
            std::fs::create_dir_all(&out_dir)?;
            let stem = canonical_training_prep_stem(&h5ad);
            let dest =
                spacetravlr::scanpy_preprocess::training_processed_h5ad_path(&out_dir, &stem);
            let spatial_microns = spacetravlr::scanpy_preprocess::SpatialMicronsOptions {
                skip: cli.skip_spatial_microns,
                species: cli
                    .spatial_species
                    .as_deref()
                    .map(|s| s.trim().to_lowercase())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_default(),
                target_median_nn_um: cli.spatial_microns_target_um,
            };
            let batch_owned = spacetravlr::scanpy_preprocess::resolve_magic_batch_obs_column(
                cli.magic_batch_obs.as_deref(),
                cli.condition.as_deref(),
            );
            let plot_path = if spacetravlr::scanpy_preprocess::prepared_training_output_is_reusable(
                &h5ad, &dest,
            )? && h5ad_obsm_has_umap(&dest)?
            {
                eprintln!(
                    "spacetravlr: reusing existing {} (>= mtime of {})",
                    dest.display(),
                    h5ad.display()
                );
                dest
            } else {
                let (written, log) = spacetravlr::scanpy_preprocess::full_preprocess_maybe_log(
                    &h5ad,
                    &dest,
                    true,
                    batch_owned.as_deref(),
                    spatial_microns,
                    true,
                )?;
                if let Some(l) = log {
                    eprint!("{l}");
                }
                written
            };
            let _ = spacetravlr::scanpy_preprocess::strip_heavy_training_artifacts_from_h5ad(
                &plot_path,
            );
            spacetravlr::adata_terminal_scatter::print_h5ad_umap_scatter(&plot_path, color_obs)
        }
    }
}

fn resolve_celloracle_network_data_dir(cli: &Cli) -> anyhow::Result<Option<String>> {
    if let Some(p) = cli.celloracle_network_data_dir.as_ref() {
        return Ok(Some(expand_user_path(p.to_string_lossy().as_ref())));
    }
    if let Some(p) = cli.config.as_ref() {
        let cfg = SpaceshipConfig::try_load_merged(Some(p.as_path()))
            .with_context(|| format!("load merged spaceship config {}", p.display()))?;
        return Ok(cfg.grn.network_data_dir);
    }
    if let Ok(cfg) = SpaceshipConfig::try_load_merged(None) {
        return Ok(cfg.grn.network_data_dir);
    }
    Ok(None)
}

fn run_celloracle(cli: &Cli, h5ad_input: &std::path::Path) -> anyhow::Result<()> {
    use spacetravlr::celloracle::{
        filter_links_p_max, infer_grn_per_cluster, infer_grn_whole, scale_gem_no_center,
        write_links_as_tf_priors_feather,
    };
    use spacetravlr::network::{GeneNetwork, infer_species};
    use spacetravlr::scanpy_preprocess::{
        SpatialMicronsOptions, ensure_training_adata_ready, resolve_magic_batch_obs_column,
    };
    use spacetravlr::{
        read_h5ad_expression_dense_f64, read_h5ad_obs_column_str, read_h5ad_var_names,
    };
    use std::path::Path;

    let h5ad_expanded = expand_user_path(h5ad_input.to_string_lossy().as_ref());

    let out_base = if let Some(p) = cli
        .celloracle_output_dir
        .as_ref()
        .or(cli.process_output_dir.as_ref())
    {
        PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()))
    } else {
        std::env::current_dir().context("celloracle output dir (cwd)")?
    };
    std::fs::create_dir_all(&out_base).with_context(|| format!("mkdir {:?}", out_base))?;

    let mut adata_path = h5ad_expanded.clone();

    if !cli.celloracle_skip_preprocess {
        let magic_batch = resolve_magic_batch_obs_column(cli.magic_batch_obs.as_deref(), None);
        let species_trim = cli
            .spatial_species
            .as_deref()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .unwrap_or_default();
        let spatial_microns = SpatialMicronsOptions {
            skip: cli.skip_spatial_microns,
            species: species_trim,
            target_median_nn_um: cli.spatial_microns_target_um,
        };
        ensure_training_adata_ready(
            &mut adata_path,
            &out_base,
            Path::new(&h5ad_expanded),
            magic_batch.as_deref(),
            spatial_microns,
        )?;
    }

    let adata_in = PathBuf::from(expand_user_path(adata_path.trim()));
    if !adata_in.is_file() {
        anyhow::bail!("AnnData not found at {}", adata_in.display());
    }

    let layer = cli.celloracle_layer.trim();
    let var_names = read_h5ad_var_names(&adata_in).context("read var_names")?;
    let gem = read_h5ad_expression_dense_f64(&adata_in, layer)
        .with_context(|| format!("read expression layer {:?}", layer))?;
    anyhow::ensure!(
        gem.ncols() == var_names.len(),
        "expression shape {:?} vs len(var_names) {}",
        gem.dim(),
        var_names.len()
    );

    let species = match cli.celloracle_species.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => s.to_string(),
        None => infer_species(&var_names)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "could not infer GRN species from var_names; pass --celloracle-species human or mouse"
                )
            })?
            .to_string(),
    };

    let network_data_dir = resolve_celloracle_network_data_dir(cli)?;
    let network = GeneNetwork::new(species.trim(), &var_names, network_data_dir.as_deref())?;
    let tf_by_target = network.grn_celloracle_tf_regulators_by_target()?;

    let gem_scaled = scale_gem_no_center(&gem);

    let run_infer = || {
        if cli.celloracle_per_cluster {
            let key = cli.celloracle_obs_key.trim();
            anyhow::ensure!(!key.is_empty(), "--celloracle-obs-key must not be empty");
            let obs = read_h5ad_obs_column_str(&adata_in, key)
                .with_context(|| format!("read obs[{key}]"))?;
            infer_grn_per_cluster(
                &gem,
                &gem_scaled,
                &var_names,
                &tf_by_target,
                &obs,
                true,
                None,
            )
        } else {
            infer_grn_whole(&gem, &gem_scaled, &var_names, &tf_by_target, true, None)
        }
    };

    let mut links = if let Some(n) = cli.celloracle_threads.filter(|n| *n > 0) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .context("rayon ThreadPool")?;
        pool.install(run_infer)?
    } else {
        run_infer()?
    };

    let p_max = cli.celloracle_p_max.unwrap_or(0.05);
    let n_before = links.len();
    links = filter_links_p_max(links, p_max);
    if links.len() < n_before {
        eprintln!(
            "CellOracle p-filter: {} → {} edges (p ≤ {})",
            n_before,
            links.len(),
            p_max
        );
    }

    let feather_out = match &cli.celloracle_output {
        Some(p) => PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())),
        None => {
            let stem = canonical_adata_stem(&adata_in);
            out_base.join(format!("{stem}_celloracle_tf_priors.feather"))
        }
    };
    if let Some(parent) = feather_out.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {:?}", parent))?;
    }
    write_links_as_tf_priors_feather(&feather_out, &links)
        .with_context(|| format!("write {:?}", feather_out))?;
    eprintln!(
        "Wrote CellOracle TF priors ({} edges) to {}",
        links.len(),
        feather_out.display()
    );
    Ok(())
}

#[cfg(feature = "rctd")]
fn run_rctd_from_cli(cli: &Cli) -> anyhow::Result<()> {
    let h5ad = cli.h5ad.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "--rctd requires `--h5ad PATH` (spatial AnnData .h5ad or .rds via same path)"
        )
    })?;
    let spatial = PathBuf::from(expand_user_path(h5ad.to_string_lossy().as_ref()));
    if !spatial.is_file() {
        anyhow::bail!("spatial input not found at {}.", spatial.display());
    }
    let reference = cli
        .rctd_reference
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--rctd requires `--ref-adata PATH`"))?;
    let reference = PathBuf::from(expand_user_path(reference.to_string_lossy().as_ref()));
    if !reference.is_file() {
        anyhow::bail!("reference not found at {}.", reference.display());
    }
    let sigma_float: Option<f64> = if let Some(ref p) = cli.rctd_sigma_file {
        let path = PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()));
        let t = std::fs::read_to_string(&path)
            .with_context(|| format!("read --rctd-sigma-file {}", path.display()))?;
        let line = t
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim())
            .find(|l| !l.is_empty())
            .context("--rctd-sigma-file: no non-empty line")?;
        Some(
            line.parse::<f64>()
                .with_context(|| format!("parse --rctd-sigma-file value {:?}", line))?,
        )
    } else {
        None
    };
    run_rctd(RctdCliArgs {
        spatial,
        reference,
        spatial_obs_subset_file: cli
            .rctd_obs_subset_file
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()))),
        gene_subset_file: cli
            .rctd_genes_file
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()))),
        spatial_numi_tsv: cli
            .rctd_spatial_numi_tsv
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()))),
        sigma_float: if cli.rctd_q_tsv.is_some() {
            None
        } else {
            sigma_float
        },
        q_matrix_tsv: cli
            .rctd_q_tsv
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()))),
        x_vals_tsv: cli
            .rctd_x_vals_tsv
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()))),
        k_val: cli.rctd_k_val,
        skip_profile_column_normalize: cli.rctd_skip_profile_normalize,
        cell_type_col: cli.rctd_cell_type_col.clone(),
        ref_rows_are_types: cli.rctd_ref_rows_are_types,
        ref_cell_min: cli.rctd_ref_cell_min,
        ref_min_umi: cli.rctd_ref_min_umi,
        ref_max_cells_per_type: cli.rctd_ref_max_cells_per_type,
        q_matrices: cli
            .rctd_q_matrices
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()))),
        sigma: cli.rctd_sigma,
        mode: cli.rctd_mode.into(),
        batch_size: cli.rctd_batch_size,
        output_prefix: cli
            .rctd_output
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref()))),
        gpu: cli.rctd_gpu,
    })
}

fn umap_lab_executable_name() -> &'static str {
    #[cfg(windows)]
    {
        "umap_lab.exe"
    }
    #[cfg(not(windows))]
    {
        "umap_lab"
    }
}

fn spacetravlr_workspace_root() -> PathBuf {
    std::env::var_os("SPACETRAVLR_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn resolve_umap_lab_binary(workspace: &Path) -> anyhow::Result<PathBuf> {
    let name = umap_lab_executable_name();
    if let Ok(me) = std::env::current_exe() {
        if let Some(dir) = me.parent() {
            let cand = dir.join(name);
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    let mut profiles = vec!["release", "debug"];
    if let Ok(me) = std::env::current_exe() {
        let s = me.to_string_lossy();
        if s.contains("target/debug") {
            profiles = vec!["debug", "release"];
        }
    }
    for p in &profiles {
        let cand = workspace.join("target").join(p).join(name);
        if cand.is_file() {
            return Ok(cand);
        }
    }
    eprintln!(
        "spacetravlr gui: umap_lab not found; building with cargo (release, --features umap-lab) …"
    );
    let st = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--features",
            "umap-lab",
            "--bin",
            "umap_lab",
        ])
        .current_dir(workspace)
        .status()
        .context("spawn cargo to build umap_lab")?;
    anyhow::ensure!(
        st.success(),
        "cargo build --release --features umap-lab --bin umap_lab failed with status {:?}",
        st.code()
    );
    let cand = workspace.join("target/release").join(name);
    anyhow::ensure!(
        cand.is_file(),
        "expected umap_lab at {} after cargo build",
        cand.display()
    );
    Ok(cand)
}

fn run_spacetravlr_gui(gui: &GuiCli) -> anyhow::Result<()> {
    let root = spacetravlr_workspace_root();
    let web = root.join("web/umap_lab");
    anyhow::ensure!(
        web.join("package.json").is_file(),
        "missing {}; set SPACETRAVLR_ROOT to the SpaceTravLR_rust repo root (contains web/umap_lab/package.json). Current root: {}",
        web.join("package.json").display(),
        root.display()
    );

    if !gui.skip_npm {
        if !web.join("node_modules").is_dir() {
            eprintln!("spacetravlr gui: npm install (first-time dependencies) …");
            let st = Command::new("npm")
                .arg("install")
                .current_dir(&web)
                .status()
                .context("spawn npm install")?;
            anyhow::ensure!(
                st.success(),
                "npm install failed with status {:?}",
                st.code()
            );
        }
        eprintln!("spacetravlr gui: npm run build …");
        let st = Command::new("npm")
            .args(["run", "build"])
            .current_dir(&web)
            .status()
            .context("spawn npm run build")?;
        anyhow::ensure!(
            st.success(),
            "npm run build failed with status {:?}",
            st.code()
        );
    }

    let umap_bin = resolve_umap_lab_binary(&root)?;
    let bind = gui.bind.trim();
    anyhow::ensure!(!bind.is_empty(), "--bind must not be empty");
    let url = format!("http://{bind}:{}/", gui.port);
    println!("{url}");

    let mut cmd = Command::new(&umap_bin);
    cmd.arg("--bind").arg(bind).arg("--port").arg(gui.port.to_string());
    cmd.current_dir(&root);
    if let Some(sd) = gui.static_dir.as_ref() {
        cmd.arg("--static-dir").arg(sd);
    }
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let st = cmd
        .status()
        .with_context(|| format!("failed to run {}", umap_bin.display()))?;
    std::process::exit(st.code().unwrap_or(1));
}

fn main() -> anyhow::Result<()> {
    spacetravlr::ensure_process_env();
    let cli = Cli::parse();

    if cli.verify {
        if cli.command.is_some() {
            anyhow::bail!("--verify cannot be combined with a subcommand");
        }
        if cli.update {
            anyhow::bail!("--verify cannot be combined with --update");
        }
        return spacetravlr::verify_bundle::run_spacetravlr_verify();
    }

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

    match &cli.command {
        Some(Commands::RunSummary(rs)) => return run_run_summary(&cli, rs),
        Some(Commands::CollectInteractions(ci)) => return run_collect_interactions(ci),
        Some(Commands::Gui(g)) => return run_spacetravlr_gui(g),
        None => {}
    }

    if cli.obs.is_some() && cli.peek.is_none() && cli.plot_umap.is_none() {
        anyhow::bail!(
            "--obs requires --peek PATH (or --peak PATH), or use --plot-umap to color the UMAP"
        );
    }

    if let Some(peek_path) = &cli.peek {
        if let Some(s) = &cli.obs {
            if s.trim().is_empty() {
                anyhow::bail!("--obs must be a non-empty column name");
            }
        }
        let p = PathBuf::from(expand_user_path(peek_path.to_string_lossy().as_ref()));
        if !p.is_file() {
            anyhow::bail!("--peek: not a file: {}", p.display());
        }
        return spacetravlr::print_h5ad_peek(p.as_path(), cli.obs.as_deref().map(str::trim));
    }

    if cli.demo {
        #[cfg(not(feature = "tui"))]
        anyhow::bail!(
            "This binary was built without the `tui` feature; rebuild with default features to use --demo."
        );
        #[cfg(feature = "tui")]
        return run_demo_mode(&cli);
    }

    if cli.infer_species {
        let h5ad = cli
            .h5ad
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())))
            .filter(|p| p.is_file());
        let h5ad = match h5ad {
            Some(p) => p,
            None => {
                anyhow::bail!(
                    "--infer-species requires --h5ad PATH pointing to an existing .h5ad file."
                );
            }
        };
        let var_names = spacetravlr::read_h5ad_var_names(&h5ad)
            .with_context(|| format!("read var_names from {}", h5ad.display()))?;
        let n = var_names.len();
        match spacetravlr::network::infer_species(&var_names) {
            Some(species) => {
                println!("{species}");
                eprintln!(
                    "inferred species={species} from {n} var_names in {}",
                    h5ad.display()
                );
            }
            None => {
                eprintln!(
                    "could not determine species from {n} var_names in {} — set --spatial-species or [data].spatial_species explicitly",
                    h5ad.display()
                );
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    if cli.map_labels {
        return run_map_labels(&cli);
    }

    if cli.prep_umap || cli.prep_rust_magic || (cli.prep_leiden && cli.plot_umap.is_none()) {
        return run_rust_prep_convenience(&cli);
    }

    if cli.rust_process_h5ad {
        return run_rust_process_h5ad(&cli);
    }

    if cli.process_h5ad {
        return run_process_h5ad(&cli);
    }

    if cli.impute {
        return run_impute(&cli);
    }

    if let Some(raw) = cli.celloracle.as_deref() {
        let h5ad_path = if raw.trim().is_empty() {
            cli.h5ad.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "`--celloracle` without PATH requires `--h5ad PATH` to the AnnData .h5ad"
                )
            })?
        } else {
            std::path::PathBuf::from(expand_user_path(raw.trim()))
        };
        return run_celloracle(&cli, h5ad_path.as_path());
    }

    #[cfg(feature = "rctd")]
    if cli.rctd {
        return run_rctd_from_cli(&cli);
    }

    if cli.plot_h5ad {
        let h5ad = cli
            .h5ad
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())))
            .filter(|p| p.is_file());
        let h5ad = match h5ad {
            Some(p) => p,
            None => {
                anyhow::bail!(
                    "--plot-h5ad requires --h5ad PATH pointing to an existing .h5ad file."
                );
            }
        };
        return spacetravlr::adata_terminal_scatter::print_h5ad_scatter(&h5ad, "cell_type");
    }

    if cli.plot_umap.is_some() {
        return run_plot_umap(&cli);
    }

    let (mut cfg, mut join_training) = load_config_for_main(&cli)?;

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

    let mut path = expand_user_path(&cfg.data.adata_path);
    cfg.data.adata_path = path.clone();

    let mut network_data_dir: Option<String> = cfg
        .grn
        .network_data_dir
        .as_ref()
        .map(|s| expand_user_path(s.trim()))
        .filter(|s| !s.is_empty());
    let mut tf_priors_feather: Option<String> = cfg
        .grn
        .tf_priors_feather
        .as_ref()
        .map(|s| expand_user_path(s.trim()))
        .filter(|s| !s.is_empty());
    cfg.grn.tf_priors_feather = tf_priors_feather.clone();

    if !Path::new(&path).exists() {
        anyhow::bail!("Dataset not found at {}.", path);
    }

    let mut adata_path_for_stem = path.clone();

    if cfg.execution.output_dir.trim().is_empty() {
        cfg.execution.output_dir =
            default_output_dir_for_adata_path(Path::new(&adata_path_for_stem))?;
    }
    let output_dir_pb = PathBuf::from(expand_user_path(cfg.execution.output_dir.trim()));
    if cli.clean_output_dir && output_dir_pb.exists() {
        std::fs::remove_dir_all(&output_dir_pb).with_context(|| {
            format!(
                "--clean-output-dir: could not remove {}",
                output_dir_pb.display()
            )
        })?;
    }
    std::fs::create_dir_all(&output_dir_pb)?;
    cfg.execution.output_dir = output_dir_pb.to_string_lossy().to_string();

    if !join_training {
        let repro_pb = output_dir_pb.join(RUN_REPRO_TOML_FILENAME);
        if repro_pb.is_file() {
            eprintln!(
                "Note: {} already exists under {} — loading it (same training contract as --join-output-dir).",
                RUN_REPRO_TOML_FILENAME,
                output_dir_pb.display()
            );
            join_training = true;
            cfg = SpaceshipConfig::from_run_repro_merged(&repro_pb, cli.config.as_deref())?;
            cfg.execution.output_dir = output_dir_pb.to_string_lossy().to_string();
            validate_join_cli_against_repro(&cli, &cfg, &repro_pb, "Resume:")?;
            apply_cli_join_overrides(&cli, &mut cfg)?;
            eprint_join_style_resume_cli_notes(&cli, &repro_pb, false);
            path = expand_user_path(cfg.data.adata_path.trim());
            cfg.data.adata_path = path.clone();
            if !Path::new(&path).exists() {
                anyhow::bail!("Dataset not found at {}.", path);
            }
            adata_path_for_stem = path.clone();
            network_data_dir = cfg
                .grn
                .network_data_dir
                .as_ref()
                .map(|s| expand_user_path(s.trim()))
                .filter(|s| !s.is_empty());
            tf_priors_feather = cfg
                .grn
                .tf_priors_feather
                .as_ref()
                .map(|s| expand_user_path(s.trim()))
                .filter(|s| !s.is_empty());
            cfg.grn.tf_priors_feather = tf_priors_feather.clone();
        }
    }

    let max_genes = cfg.training.max_genes;
    let gene_filter = cfg.training.genes.clone();
    let condition_column = cli
        .condition
        .clone()
        .or_else(|| cfg.data.condition.clone())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let config_source_path: Option<PathBuf> = if join_training {
        Some(output_dir_pb.join(RUN_REPRO_TOML_FILENAME))
    } else {
        cli.config
            .as_ref()
            .map(|p| PathBuf::from(expand_user_path(p.to_string_lossy().as_ref())))
            .or_else(SpaceshipConfig::discover_default_path)
    };

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
        let spatial_microns = spacetravlr::scanpy_preprocess::SpatialMicronsOptions {
            skip: false,
            species: {
                let s = cfg.data.spatial_species.trim().to_lowercase();
                if s.is_empty() { String::new() } else { s }
            },
            target_median_nn_um: cfg.data.spatial_median_nn_target_um,
        };
        // #region agent log
        spacetravlr::scanpy_preprocess::agent_debug_ndjson(
            "A",
            "spacetravlr.rs:training_auto_adata_prep",
            "[data].spatial_species from config -> SpatialMicronsOptions",
            "preprocess",
            serde_json::json!({
                "cfg_data_spatial_species_raw": cfg.data.spatial_species,
                "resolved_species_for_scanpy": spatial_microns.species,
                "spatial_median_nn_target_um": cfg.data.spatial_median_nn_target_um,
                "adata_path": cfg.data.adata_path,
            }),
        );
        // #endregion
        spacetravlr::scanpy_preprocess::ensure_training_adata_ready(
            &mut cfg.data.adata_path,
            &output_dir_pb,
            Path::new(&adata_path_for_stem),
            magic_batch.as_deref(),
            spatial_microns,
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

    let mode_label = match cfg.resolved_cnn_mode() {
        CnnTrainingMode::Seed => "seed",
        CnnTrainingMode::Full => "full",
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
    let run_summary = RunConfigSummary::build(RunConfigSummaryBuildArgs {
        config_path: config_path_ref,
        compute_backend: compute.label(),
        compute_device_detail: &compute_hardware_details(&compute),
        compute_notice: &compute_notice_text(&compute),
        cfg: &cfg,
        max_genes,
        gene_filter: gene_filter.as_deref(),
        condition_split: condition_column.as_deref(),
    });

    let verbose = cli.verbose;

    if !use_dashboard {
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
                    "join+conditions  parent={}  locks per conditions/<group>/",
                    output_dir.trim_end_matches('/')
                );
            } else {
                println!(
                    "join  out={}  skip existing *_betadata.feather · claim genes via .lock",
                    output_dir.trim_end_matches('/')
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
                "split  obs.{}  ·  {} groups  ·  {}/conditions/<name>/",
                condition_col,
                splits.len(),
                output_dir.trim_end_matches('/')
            );
            if join_training {
                let dir_status = scan_condition_status(&output_dir)?;
                if !dir_status.is_empty() {
                    let mut parts = Vec::with_capacity(dir_status.len());
                    for cs in &dir_status {
                        let st = if cs.n_locks > 0 {
                            "run"
                        } else if cs.n_done() > 0 {
                            "ok"
                        } else {
                            "·"
                        };
                        parts.push(format!(
                            "{}:d={}/fe={}/o={}/ta={}/L={}/{}",
                            cs.label,
                            cs.n_done(),
                            cs.n_feathers,
                            cs.n_orphans,
                            cs.n_tf_ablated,
                            cs.n_locks,
                            st
                        ));
                    }
                    println!("join {}", parts.join("  "));
                }
            }
            for split in splits {
                let split_output_dir = split.output_dir.display().to_string();
                let obs_subset = Arc::from(split.obs_indices.into_boxed_slice());
                println!(
                    "→  '{}'  {} cells  {}",
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
                    verbose,
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
                verbose,
            };
            fit_all_genes_dispatch(&params, &compute)?;
        }
        println!("done.");
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
                        verbose,
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
                    verbose,
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
