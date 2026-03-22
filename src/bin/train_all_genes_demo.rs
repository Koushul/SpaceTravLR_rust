use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn_autodiff::Autodiff;
use space_trav_lr_rust::spatial_estimator::SpatialCellularProgramsEstimator;
use space_trav_lr_rust::training_hud::TrainingHudState;
use space_trav_lr_rust::training_tui::run_training_dashboard;
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

fn print_usage() {
    eprintln!(
        "\
Usage: train_all_genes_demo [OPTIONS]

Options:
  --full              CNN spatial mode — per-cell betadata export
  --epochs N          CNN fine-tuning epochs per gene            (default: 10)
  --parallel N        Number of parallel gene workers            (default: 1)
  --max-genes N       Cap to first N genes in var order
  --genes A,B,C       Comma-separated target gene names
  --output-dir PATH   Directory to write *_betadata.csv files   (default: /tmp/training)
  --plain             Simple progress bar, no sci-fi TUI
  -h, --help          This message

Environment:
  SPACETRAVLR_H5AD    Path to .h5ad (default: built-in path)

TUI: CPU / RAM, per-worker active-gene list, event log.
     Press [q] to drain and stop after current genes finish.

Examples:
  # seed-only, 4 workers, 50 genes, custom output dir
  cargo run --release --bin train_all_genes_demo -- --parallel 4 --max-genes 50 --output-dir /data/betas

  # CNN mode, 2 workers, 5 epochs
  cargo run --release --bin train_all_genes_demo -- --full --epochs 5 --parallel 2 --max-genes 20

  # plain text output (CI / no TTY)
  cargo run --release --bin train_all_genes_demo -- --plain --max-genes 20 --output-dir /tmp/run1
"
    );
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let plain       = args.contains(&"--plain".to_string());
    let full_cnn    = args.contains(&"--full".to_string());
    let epochs      = next_usize(&args, "--epochs").unwrap_or(10);
    let n_parallel  = next_usize(&args, "--parallel").unwrap_or(1).max(1);
    let max_genes   = next_usize(&args, "--max-genes");
    let output_dir  = next_str(&args, "--output-dir").unwrap_or("/tmp/training").to_string();

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

    let path = std::env::var("SPACETRAVLR_H5AD").unwrap_or_else(|_| {
        "/ix/djishnu/shared/djishnu_kor11/training_data_2025/snrna_human_tonsil.h5ad".to_string()
    });

    if !std::path::Path::new(&path).exists() {
        anyhow::bail!(
            "Dataset not found at {}. Set SPACETRAVLR_H5AD or place the h5ad there.",
            path
        );
    }

    let _ = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global();

    let mode_label = if full_cnn { "Full CNN" } else { "Seed-Only (lasso)" };

    if plain {
        println!("🚀 SpaceTravLR  |  {}  |  {} worker(s)  |  {} epochs/gene", mode_label, n_parallel, epochs);
        println!("Dataset:    {}", path);
        println!("Output dir: {}", output_dir);
        if let Some(n) = max_genes  { println!("Gene cap:   {}", n); }
        if let Some(ref g) = gene_filter { println!("Filter:     {:?}", g); }
        println!("--------------------------------------------------");

        let device: WgpuDevice = Default::default();
        SpatialCellularProgramsEstimator::<AB, anndata_hdf5::H5>::fit_all_genes(
            &path, 0.1, 32, 0.05, 0.5, None,
            epochs, 1e-3, 0.0, 1e-4, 1e-4, 100, 1e-4,
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

    let path_worker        = path.clone();
    let gene_filter_worker = gene_filter.clone();
    let output_dir_worker  = output_dir.clone();
    let hud_worker         = hud.clone();

    let handle = thread::spawn(move || {
        let device: WgpuDevice = Default::default();
        SpatialCellularProgramsEstimator::<AB, anndata_hdf5::H5>::fit_all_genes(
            &path_worker, 0.1, 32, 0.05, 0.5, None,
            epochs, 1e-3, 0.0, 1e-4, 1e-4, 100, 1e-4,
            full_cnn, gene_filter_worker, max_genes, n_parallel, &output_dir_worker,
            Some(hud_worker), &device,
        )
    });

    run_training_dashboard(hud.clone())?;

    match handle.join() {
        Ok(r)  => r?,
        Err(_) => anyhow::bail!("training thread panicked"),
    }

    println!("\n✨ Done!");
    Ok(())
}
