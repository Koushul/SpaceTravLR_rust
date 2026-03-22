use ndarray::Array2;
use ndarray_npy::NpzReader;
use std::fs::File;
use space_trav_lr_rust::ligand::calculate_weighted_ligands;
use std::time::Instant;
use ndarray::ArrayD;

fn main() -> anyhow::Result<()> {
    let path = "/tmp/ligand_ref.npz";
    println!("📂 Loading reference data from: {}", path);
    let mut npz = NpzReader::new(File::open(path)?)?;

    let xy: Array2<f64> = npz.by_name("xy.npy")?;
    let lig_values: Array2<f64> = npz.by_name("lig_values.npy")?;
    let radius_arr: ArrayD<f64> = npz.by_name("radius.npy")?;
    let radius = *radius_arr.first().unwrap();
    let expected: Array2<f64> = npz.by_name("expected.npy")?;

    println!("🧪 Data loaded: {} cells, {} ligands, radius={}", xy.nrows(), lig_values.ncols(), radius);

    println!("🚀 Running Rust implementation...");
    let start = Instant::now();
    let result = calculate_weighted_ligands(&xy, &lig_values, radius, 1.0);
    let duration = start.elapsed();
    println!("⏱️ Rust calculation took: {:?}", duration);

    println!("🧐 Verifying numerical accuracy...");
    assert_eq!(result.shape(), expected.shape());
    
    let mut max_diff = 0.0;
    let mut count = 0;
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            let diff = (result[[i, j]] - expected[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-10 {
                count += 1;
            }
        }
    }

    println!("📊 Max difference: {:.2e}", max_diff);
    println!("📊 Number of elements with diff > 1e-10: {}", count);

    if max_diff < 1e-10 {
        println!("✅ Success! Numerical match confirmed.");
    } else {
        println!("⚠️ Warning: Numerical differences detected.");
    }

    Ok(())
}
