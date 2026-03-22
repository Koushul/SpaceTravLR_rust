use space_trav_lr_rust::ligand::calculate_weighted_ligands;
use std::time::Instant;
use anndata_hdf5::H5;
use anndata::{AnnData, AnnDataOp, AxisArraysOp, Backend, ArrayElemOp};
use ndarray::Array2;
fn main() -> anyhow::Result<()> {
    let path = "/Users/koush/Downloads/snrna_human_tonsil.h5ad";
    println!("📂 Loading full dataset: {}", path);
    
    // Explicitly use H5 backend traits
    let file = H5::open(path)?;
    let adata = AnnData::<H5>::open(file)?;

    // Access spatial coordinates
    let xy: Array2<f64> = adata.obsm().get_item("spatial")?.unwrap();
    
    // Pick 300 ligands (first 300 genes)
    let n_ligands = 300;
    println!("🧪 Picking {} ligands...", n_ligands);
    let slice = [anndata::data::SelectInfoElem::full(), anndata::data::SelectInfoElem::full()];
    let x_raw: Array2<f64> = adata.x().slice(slice)?.unwrap();
    let lig_values = x_raw.slice(ndarray::s![.., ..n_ligands]).mapv(|v| v);
    
    let radius = 100.0;
    
    println!("🧪 Data ready: {} cells, {} ligands", xy.nrows(), lig_values.ncols());

    println!("🚀 Running Rust implementation in Release mode...");
    let start = Instant::now();
    let _result = calculate_weighted_ligands(&xy, &lig_values, radius, 1.0);
    let duration = start.elapsed();
    
    println!("⏱️ Rust calculation took: {:?}", duration);
    
    // Calculate total operations: N * N * L
    let n = xy.nrows() as u64;
    let l = lig_values.ncols() as u64;
    let ops = n * n * l;
    println!("📊 Total inner loop iterations: {} (approx {:.2} Billion)", ops, ops as f64 / 1e9);
    
    Ok(())
}
