use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn_autodiff::Autodiff;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use rand::seq::SliceRandom;
use rand::thread_rng;
use space_trav_lr_rust::spatial_estimator::SpatialCellularProgramsEstimator;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    type AB = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    let path = "/Users/koush/Downloads/snrna_human_tonsil.h5ad";
    println!("🚀 Loading dataset metadata: {}", path);

    // 1. Initial load to get gene names
    let adata_meta = AnnData::<H5>::open(H5::open(path)?)?;
    let var_names = adata_meta.var_names().into_vec();
    let n_total_genes = var_names.len();
    drop(adata_meta); // Close it for now

    // 2. Setup Experiment Parameters
    let n_genes_test = 1;
    let n_regulators = 300.min(n_total_genes);

    let mut rng = thread_rng();

    // Pick 1 specific target gene (or random if not specified)
    let mut all_genes = var_names.clone();
    all_genes.shuffle(&mut rng);
    let mut genes_to_test = all_genes[0..n_genes_test].to_vec();
    if var_names.contains(&"FAM41C".to_string()) {
        genes_to_test = vec!["FAM41C".to_string()];
    }

    println!(
        "🧪 Starting experiment: {} genes x {} regulators (Seed-Only)...",
        n_genes_test, n_regulators
    );

    let pb = ProgressBar::new(genes_to_test.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .expect("Log format error")
        .progress_chars("#>-"));

    // 3. Run Sequentially across genes to avoid HDF5/Parallel issues
    genes_to_test
        .iter()
        .progress_with(pb)
        .for_each(|gene_name| {
            let mut thread_rng = thread_rng();

            // Randomly select 300 regulators for EACH gene
            let mut local_genes = all_genes.clone();
            local_genes.shuffle(&mut thread_rng);
            let regulators = local_genes[1..=n_regulators].to_vec();

            // Re-open AnnData per gene
            if let Ok(file) = H5::open(path) {
                if let Ok(adata) = AnnData::<H5>::open(file) {
                    let mut estimator = SpatialCellularProgramsEstimator::<AB, H5>::new(
                        Arc::new(adata),
                        gene_name.clone(),
                        400.0,
                        32,
                        50.0,
                        0.01,
                        None,
                    )
                    .unwrap();

                    estimator.seed_only = true;
                    estimator.regulators = regulators;
                    estimator.layer = "imputed_count".to_string();

                    // Fit (seed_only will skip GCNNWR)
                    let _ = estimator.fit(
                        0,       // epochs skips GCNNWR updates
                        0.005,   // learning_rate
                        0.2,     // score_threshold
                        0.01,    // l1_reg
                        0.01,    // group_reg
                        100,     // n_iter
                        1e-5,    // tol
                        "lasso", // estimator_type
                        &device,
                    );

                    // Export betas to CSV
                    use std::io::Write;
                    if let Ok(mut w) = std::fs::File::create("/tmp/rust_betas.csv") {
                        let _ = writeln!(w, "cluster,beta_idx,value");
                        if let Some(est) = &estimator.estimator {
                            for (c_id, model) in &est.models {
                                let anchors_data = model.anchors.clone().into_data();
                                let anchors: &[f32] = anchors_data.as_slice::<f32>().unwrap();
                                for (i, v) in anchors.iter().enumerate() {
                                    let _ = writeln!(w, "{},{},{}", c_id, i, v);
                                }
                            }
                        }
                    }
                }
            }
        });

    println!("✅ Experiment complete!");

    Ok(())
}
