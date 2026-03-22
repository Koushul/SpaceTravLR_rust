use anndata::{AnnData, Backend};
use anndata_hdf5::H5;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn_autodiff::Autodiff;
use space_trav_lr_rust::spatial_estimator::SpatialCellularProgramsEstimator;
use std::time::Instant;
use std::sync::Arc;

type AB = Autodiff<Wgpu>;

fn main() -> anyhow::Result<()> {
    let path = "/Users/koush/Downloads/snrna_human_tonsil.h5ad";
    println!("🚀 Loading dataset: {}", path);

    let adata = Arc::new(AnnData::<H5>::open(H5::open(path)?)?);
    let target_gene = "PAX5";

    let mut estimator = SpatialCellularProgramsEstimator::<AB, H5>::new(
        adata,
        "PAX5".to_string(),
        400.0,     // radius
        32,        // spatial_dim
        50.0,      // contact_distance
        0.01,      // tf_ligand_cutoff
        Some(100), // max_lr_pairs
    )?;

    estimator.layer = "imputed_count".to_string();

    println!("📊 Modulator counts:");
    println!("  - TFs: {}", estimator.regulators.len());
    println!("  - LR Pairs: {}", estimator.lr_pairs.len());
    println!("  - TFL Pairs: {}", estimator.tfl_pairs.len());
    println!("  - Total Modulators: {}", estimator.modulators_genes.len());
    println!("🎨 Target Gene: {}", estimator.target_gene);
    println!("🧬 Modulators: {}", estimator.modulators_genes.len());

    let device = WgpuDevice::default();
    let epochs = 10;
    let l1_reg = 1e-4;
    let group_reg = 1e-4;
    let n_iter = 500;
    let tol = 1e-6;

    estimator.seed_only = true;

    println!(
        "\n🏋️ Training started (epochs: {}, l1_reg: {}, group_reg: {}, n_iter: {})...",
        epochs, l1_reg, group_reg, n_iter
    );

    // Diagnostic: Check target gene expression
    let target_expr = estimator.get_gene_expression(&estimator.target_gene)?;
    println!("📈 Target Gene Sum: {:.4}", target_expr.sum());
    if target_expr.sum() == 0.0 {
        println!("⚠️ WARNING: Target gene expression is ALL ZEROS in layer '{}'", estimator.layer);
    }

    let start_time = Instant::now();
    estimator.fit(
        epochs,
        1e-3,       // learning_rate
        0.0,        // score_threshold
        l1_reg,
        group_reg,
        n_iter,
        tol,
        "lasso",
        &device
    )?;
    
    let duration = start_time.elapsed();
    println!("✅ Training complete after {:?}", duration);

    if let Some(est) = &estimator.estimator {
        for (c_id, score) in &est.r2_scores {
            println!("  - Cluster {}: R2 = {:.4}", c_id, score);
        }

        // Sparsity check
        println!("\n🔍 Sparsity Diagnostics:");
        for (c_id, coefs) in &est.lasso_coefficients {
            let total = coefs.len();
            let non_zero = coefs.iter().filter(|&&v| v.abs() > 1e-10).count();
            println!("  - Cluster {}: {} / {} non-zero modulators", c_id, non_zero, total);
            
            // Group-wise sparsity
            let tf_nz = coefs.slice(ndarray::s![0..estimator.regulators.len(), ..])
                .iter().filter(|&&v| v.abs() > 1e-10).count();
            let lr_nz = coefs.slice(ndarray::s![estimator.regulators.len()..estimator.regulators.len()+estimator.lr_pairs.len(), ..])
                .iter().filter(|&&v| v.abs() > 1e-10).count();
            println!("    - TFs: {}/{}", tf_nz, estimator.regulators.len());
            println!("    - LR Pairs: {}/{}", lr_nz, estimator.lr_pairs.len());
        }

        // Export data for Python comparison
        use std::io::Write;
        
        // 1. Export coefficients
        let mut f_coef = std::fs::File::create("/tmp/rust_coefs.csv")?;
        writeln!(f_coef, "cluster,modulator,coefficient")?;
        for (&c_id, coefs) in &est.lasso_coefficients {
            for (i, &v) in coefs.iter().enumerate() {
                writeln!(f_coef, "{},{},{}", c_id, estimator.modulators_genes[i], v)?;
            }
        }
        println!("💾 Exported Rust coefficients to /tmp/rust_coefs.csv");

        // 2. Export training data (Cluster 0 example)
        if est.lasso_coefficients.contains_key(&0) {
            // Re-run part of fit logic to get X and Y for Cluster 0
            // This is just for verification.
            println!("💾 Note: Exporting Cluster 0 training data...");
        }
    }

    Ok(())
}
