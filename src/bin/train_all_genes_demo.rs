use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn_autodiff::Autodiff;
use space_trav_lr_rust::config::SpaceshipConfig;
use space_trav_lr_rust::spatial_estimator::SpatialCellularProgramsEstimator;
use space_trav_lr_rust::training_hud::TrainingHudState;
use space_trav_lr_rust::training_tui::{run_training_dashboard, TrainingDashboardExit};
use std::env;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread;

type AB = Autodiff<Wgpu>;

fn next_usize(args: &[String], flag: &str) -> Option<usize> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1)?.parse().ok()
}

fn next_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).map(|s| s.as_str())
}

fn next_f64(args: &[String], flag: &str) -> Option<f64> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1)?.parse().ok()
}

fn print_usage() {
    eprintln!(
        "\
Usage: train_all_genes_demo [OPTIONS]

Options:
  --config PATH       Path to spaceship_config.toml       (default: ./spaceship_config.toml)
  --full              CNN spatial mode — per-cell betadata export
  --seed-only         Seed-only mode — cluster-level betas (default)
  --epochs N          CNN fine-tuning epochs per gene
  --parallel N        Number of parallel gene workers
  --max-genes N       Cap to first N genes in var order
  --genes A,B,C       Comma-separated target gene names
  --output-dir PATH   Directory to write *_betadata.csv files
  --l1-reg F          L1 regularization coefficient
  --group-reg F       Group regularization coefficient
  --lr F              Learning rate
  --n-iter N          Max FISTA iterations
  --tol F             Convergence tolerance
  --plain             Simple progress bar, no sci-fi TUI
  -h, --help          This message

All hyperparameters are read from spaceship_config.toml first;
CLI flags override individual values.

Environment:
  SPACETRAVLR_H5AD    Fallback path to .h5ad when data.adata_path is empty

Examples:
  # seed-only, 4 workers, 50 genes, custom output dir
  cargo run --release --bin train_all_genes_demo -- --parallel 4 --max-genes 50 --output-dir /data/betas

  # CNN mode, 2 workers, 5 epochs
  cargo run --release --bin train_all_genes_demo -- --full --epochs 5 --parallel 2 --max-genes 20

  # custom config file
  cargo run --release --bin train_all_genes_demo -- --config my_experiment.toml --max-genes 100
"
    );
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    // ── Load config (file first, then CLI overrides) ─────────────────────────
    let mut cfg = if let Some(path) = next_str(&args, "--config") {
        SpaceshipConfig::from_file(path)?
    } else {
        SpaceshipConfig::load()
    };

    let plain = args.contains(&"--plain".to_string());

    if args.contains(&"--full".to_string()) {
        cfg.training.seed_only = false;
    }
    if args.contains(&"--seed-only".to_string()) {
        cfg.training.seed_only = true;
    }
    if let Some(v) = next_usize(&args, "--epochs")     { cfg.training.epochs = v; }
    if let Some(v) = next_usize(&args, "--parallel")    { cfg.execution.n_parallel = v.max(1); }
    if let Some(v) = next_str(&args, "--output-dir")    { cfg.execution.output_dir = v.to_string(); }
    if let Some(v) = next_f64(&args, "--l1-reg")        { cfg.lasso.l1_reg = v; }
    if let Some(v) = next_f64(&args, "--group-reg")     { cfg.lasso.group_reg = v; }
    if let Some(v) = next_f64(&args, "--lr")            { cfg.training.learning_rate = v; }
    if let Some(v) = next_usize(&args, "--n-iter")      { cfg.lasso.n_iter = v; }
    if let Some(v) = next_f64(&args, "--tol")           { cfg.lasso.tol = v; }

    let max_genes = next_usize(&args, "--max-genes");

    let mut gene_filter = None;
    if let Some(pos) = args.iter().position(|r| r == "--genes") {
        if pos + 1 < args.len() {
            let genes = args[pos + 1]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();
            gene_filter = Some(genes);
        }
    }

    let path = cfg.resolve_adata_path();

    if !std::path::Path::new(&path).exists() {
        anyhow::bail!(
            "Dataset not found at {}. Set data.adata_path in config or SPACETRAVLR_H5AD env.",
            path
        );
    }

    let full_cnn    = cfg.full_cnn();
    let epochs      = cfg.training.epochs;
    let n_parallel  = cfg.execution.n_parallel;
    let output_dir  = cfg.execution.output_dir.clone();

    let _ = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global();

    let mode_label = if full_cnn { "Full CNN" } else { "Seed-Only (lasso)" };

    if plain {
        println!("🚀 SpaceTravLR  |  {}  |  {} worker(s)  |  {} epochs/gene", mode_label, n_parallel, epochs);
        println!("Dataset:    {}", path);
        println!("Output dir: {}", output_dir);
        println!("Lasso:      l1={} group={} n_iter={} tol={}", cfg.lasso.l1_reg, cfg.lasso.group_reg, cfg.lasso.n_iter, cfg.lasso.tol);
        println!("Spatial:    radius={} dim={} contact={}", cfg.spatial.radius, cfg.spatial.spatial_dim, cfg.spatial.contact_distance);
        if let Some(n) = max_genes  { println!("Gene cap:   {}", n); }
        if let Some(ref g) = gene_filter { println!("Filter:     {:?}", g); }
        println!("--------------------------------------------------");

        let device: WgpuDevice = Default::default();
        SpatialCellularProgramsEstimator::<AB, anndata_hdf5::H5>::fit_all_genes(
            &path,
            cfg.spatial.radius,
            cfg.spatial.spatial_dim,
            cfg.spatial.contact_distance,
            cfg.grn.tf_ligand_cutoff,
            cfg.grn.max_lr_pairs,
            cfg.grn.top_lr_pairs_by_mean_expression,
            &cfg.data.layer,
            &cfg.cnn,
            epochs,
            cfg.training.learning_rate,
            cfg.training.score_threshold,
            cfg.lasso.l1_reg,
            cfg.lasso.group_reg,
            cfg.lasso.n_iter,
            cfg.lasso.tol,
            full_cnn, gene_filter, max_genes, n_parallel, &output_dir, None, &device,
        )?;
        println!("\n✨ Done!");
        return Ok(());
    }

    // ── TUI mode ──────────────────────────────────────────────────────────────
    let cancel = Arc::new(AtomicBool::new(false));
    let hud = Arc::new(Mutex::new(TrainingHudState::new(
        path.clone(),
        output_dir.clone(),
        full_cnn,
        epochs,
        n_parallel,
        cancel.clone(),
    )));

    let hud_worker = hud.clone();

    let handle = thread::spawn(move || {
        let device: WgpuDevice = Default::default();
        SpatialCellularProgramsEstimator::<AB, anndata_hdf5::H5>::fit_all_genes(
            &path,
            cfg.spatial.radius,
            cfg.spatial.spatial_dim,
            cfg.spatial.contact_distance,
            cfg.grn.tf_ligand_cutoff,
            cfg.grn.max_lr_pairs,
            cfg.grn.top_lr_pairs_by_mean_expression,
            &cfg.data.layer,
            &cfg.cnn,
            epochs,
            cfg.training.learning_rate,
            cfg.training.score_threshold,
            cfg.lasso.l1_reg,
            cfg.lasso.group_reg,
            cfg.lasso.n_iter,
            cfg.lasso.tol,
            full_cnn, gene_filter, max_genes, n_parallel, &output_dir,
            Some(hud_worker), &device,
        )
    });

    match run_training_dashboard(hud.clone())? {
        TrainingDashboardExit::ForceQuit => {
            eprintln!("\nAborted (Shift+Q).");
            std::process::exit(130);
        }
        TrainingDashboardExit::Completed => {}
    }

    match handle.join() {
        Ok(r)  => r?,
        Err(_) => anyhow::bail!("training thread panicked"),
    }

    println!("\n✨ Done!");
    Ok(())
}
