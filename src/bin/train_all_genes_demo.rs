use burn::backend::Wgpu;
use burn_autodiff::Autodiff;
use space_trav_lr_rust::spatial_estimator::SpatialCellularProgramsEstimator;
use std::env;

type AB = Autodiff<Wgpu>;

fn next_usize(args: &[String], flag: &str) -> Option<usize> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1)?.parse().ok()
}

fn print_usage() {
    eprintln!(
        "\
Usage: train_all_genes_demo [OPTIONS]

CNN (spatial) mode requires --full.

Options:
  --full                 Train the CNN head and export per-cell betas (not seed-only / cluster betas)
  --epochs N             CNN fine-tuning epochs per gene (default: 10)
  --max-genes N          Train on the first N genes in the h5ad var order (after --genes filter, if any)
  --genes A,B,C          Comma-separated target gene names (optional)
  -h, --help             This message

Environment:
  SPACETRAVLR_H5AD       Path to .h5ad (default: built-in demo path)

Outputs under /tmp/training: <GENE>_betadata.csv

Example (5 epochs, 20 genes, CNN mode):
  cargo run --release --bin train_all_genes_demo -- --full --epochs 5 --max-genes 20
"
    );
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let full_cnn = args.contains(&"--full".to_string());

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

    let epochs = next_usize(&args, "--epochs").unwrap_or(10);
    let max_genes = next_usize(&args, "--max-genes");

    let path = std::env::var("SPACETRAVLR_H5AD").unwrap_or_else(|_| {
        "/Users/koush/Downloads/snrna_human_tonsil.h5ad".to_string()
    });

    if !std::path::Path::new(&path).exists() {
        anyhow::bail!(
            "Dataset not found at {}. Set SPACETRAVLR_H5AD or place the h5ad there.",
            path
        );
    }

    let device = Default::default();

    let _ = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global();

    println!("🚀 Starting All-Gene Training Demo");
    println!("Dataset: {}", path);
    println!(
        "Mode: {}",
        if full_cnn {
            "Full CNN (Cell-Level Export)"
        } else {
            "Seed-Only (Cluster-Level Export)"
        }
    );
    println!("Epochs per gene: {}", epochs);
    if let Some(n) = max_genes {
        println!("Gene cap: first {}", n);
    }
    if let Some(ref filter) = gene_filter {
        println!("🔍 Gene filter: {:?}", filter);
    }
    println!("--------------------------------------------------");

    SpatialCellularProgramsEstimator::<AB, anndata_hdf5::H5>::fit_all_genes(
        path.as_str(),
        0.1,
        32,
        0.05,
        0.5,
        Some(100),
        epochs,
        1e-3,
        0.0,
        1e-4,
        1e-4,
        100,
        1e-4,
        full_cnn,
        gene_filter,
        max_genes,
        &device,
    )?;

    println!("\n✨ Demo completed successfully!");
    Ok(())
}
