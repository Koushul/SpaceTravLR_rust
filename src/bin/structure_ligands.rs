//! Build / inspect tissue-structure references and run ligand-recovery checks.
//!
//! Examples:
//! ```text
//! cargo run --release --bin structure_ligands -- build \
//!   --adata data.h5ad --out ref.json --radius 200
//!
//! cargo run --release --bin structure_ligands -- validate-self \
//!   --adata data.h5ad --radius 200 --n-ligands 30 --max-cells 3000
//! ```

use anndata::{AnnData, AnnDataOp};
use anndata_hdf5::H5;
use clap::{Parser, Subcommand};
use ndarray::Array2;
use spacetravlr::ligand::calculate_weighted_ligands;
use spacetravlr::spatial_estimator::{
    load_spatial_coords_f64, read_h5ad_expression_dense_f64, read_h5ad_obs_column_str,
    read_h5ad_var_names,
};
use spacetravlr::structure::{
    StructureBuildArgs, abundance_baseline_weight_mass, build_tissue_structure_ref,
    column_pearson, compute_cell_structure_weights, infer_received_ligands_from_structure,
    infer_received_with_weight_matrix, matrix_error_metrics, mean_composition_cosine,
    received_ligands_type_mean_oracle, type_mean_expression,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "structure_ligands", about = "Tissue-structure received ligands")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build a JSON tissue-structure reference from a spatial AnnData.
    Build {
        #[arg(long)]
        adata: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "cell_type")]
        cluster_annot: String,
        #[arg(long, default_value_t = 200.0)]
        radius: f64,
        #[arg(long, default_value_t = 1.0)]
        scale: f64,
        #[arg(long)]
        hard_radius: Option<f64>,
    },
    /// Compare structure-inferred ligands to spatial Gaussian truth on one dataset.
    ValidateSelf {
        #[arg(long)]
        adata: PathBuf,
        #[arg(long, default_value = "cell_type")]
        cluster_annot: String,
        #[arg(long, default_value_t = 200.0)]
        radius: f64,
        #[arg(long, default_value_t = 1.0)]
        scale: f64,
        #[arg(long, default_value_t = 30)]
        n_ligands: usize,
        #[arg(long)]
        max_cells: Option<usize>,
        #[arg(long, default_value = "X")]
        layer: String,
    },
}

fn column_variance(col: ndarray::ArrayView1<f64>) -> f64 {
    let mean = col.mean().unwrap_or(0.0);
    let mut acc = 0.0;
    for &v in col {
        let d = v - mean;
        acc += d * d;
    }
    if col.len() == 0 {
        0.0
    } else {
        acc / col.len() as f64
    }
}

fn pick_ligand_indices(expr: &Array2<f64>, n: usize) -> Vec<usize> {
    let n_genes = expr.ncols();
    let mut scores: Vec<(usize, f64)> = (0..n_genes)
        .map(|j| {
            let col = expr.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var = column_variance(col);
            (j, mean * (1.0 + var.sqrt()))
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.into_iter().take(n.min(n_genes)).map(|(j, _)| j).collect()
}

fn subsample_indices(n: usize, max_cells: Option<usize>) -> Vec<usize> {
    match max_cells {
        Some(m) if m < n => {
            // Even stride keeps spatial coverage better than a contiguous head slice.
            let step = (n as f64 / m as f64).floor().max(1.0) as usize;
            (0..n).step_by(step).take(m).collect()
        }
        _ => (0..n).collect(),
    }
}

fn main() -> anyhow::Result<()> {
    spacetravlr::ensure_process_env();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Build {
            adata,
            out,
            cluster_annot,
            radius,
            scale,
            hard_radius,
        } => {
            let a = AnnData::<H5>::open(H5::open(&adata)?)?;
            let xy = load_spatial_coords_f64(&a)?;
            a.close()?;
            let types = read_h5ad_obs_column_str(&adata, &cluster_annot)?;
            let st = build_tissue_structure_ref(StructureBuildArgs {
                xy: &xy,
                cell_types: &types,
                radius,
                scale_factor: scale,
                hard_radius,
            })?;
            st.save_json(&out)?;
            println!(
                "Wrote {} ({} types, n_ref={})",
                out.display(),
                st.cell_types.len(),
                st.n_ref_cells
            );
            for (i, t) in st.cell_types.iter().enumerate() {
                let total_soft: f64 = st.mean_soft_counts.row(i).sum();
                let total_hard: f64 = st.mean_hard_counts.row(i).sum();
                println!(
                    "  {t}: soft_neighbors≈{total_soft:.2} hard_neighbors≈{total_hard:.2} (n={})",
                    st.ref_type_counts[i]
                );
            }
        }
        Cmd::ValidateSelf {
            adata,
            cluster_annot,
            radius,
            scale,
            n_ligands,
            max_cells,
            layer,
        } => {
            let types_full = read_h5ad_obs_column_str(&adata, &cluster_annot)?;
            let a = AnnData::<H5>::open(H5::open(&adata)?)?;
            let xy_full = load_spatial_coords_f64(&a)?;
            a.close()?;
            let expr_full = if layer == "X" || layer.is_empty() {
                // read_h5ad_expression_dense_f64 expects a layer name; empty/"X" → try layer then X via helper
                match read_h5ad_expression_dense_f64(&adata, "X") {
                    Ok(m) => m,
                    Err(_) => read_h5ad_expression_dense_f64(&adata, "")?,
                }
            } else {
                read_h5ad_expression_dense_f64(&adata, &layer)?
            };
            let genes = read_h5ad_var_names(&adata)?;
            let idx = subsample_indices(xy_full.nrows(), max_cells);
            let n = idx.len();
            let mut xy = Array2::<f64>::zeros((n, 2));
            let mut types = Vec::with_capacity(n);
            for (ii, &i) in idx.iter().enumerate() {
                xy[[ii, 0]] = xy_full[[i, 0]];
                xy[[ii, 1]] = xy_full[[i, 1]];
                types.push(types_full[i].clone());
            }
            let lig_idx = pick_ligand_indices(&expr_full, n_ligands);
            let mut lig = Array2::<f64>::zeros((n, lig_idx.len()));
            for (k, &j) in lig_idx.iter().enumerate() {
                for (ii, &i) in idx.iter().enumerate() {
                    lig[[ii, k]] = expr_full[[i, j]];
                }
            }
            let truth = calculate_weighted_ligands(&xy, &lig, radius, scale);
            let st = build_tissue_structure_ref(StructureBuildArgs {
                xy: &xy,
                cell_types: &types,
                radius,
                scale_factor: scale,
                hard_radius: Some(radius),
            })?;
            let means = type_mean_expression(&lig, &types, &st.cell_types)?;
            let cell = compute_cell_structure_weights(StructureBuildArgs {
                xy: &xy,
                cell_types: &types,
                radius,
                scale_factor: scale,
                hard_radius: Some(radius),
            })?;
            let oracle = received_ligands_type_mean_oracle(&cell.weight_mass, &means)?;
            let pooled = infer_received_ligands_from_structure(&st, &types, &means)?;
            let abund_w = abundance_baseline_weight_mass(&st);
            let abund =
                infer_received_with_weight_matrix(&abund_w, &st.cell_types, &types, &means)?;
            let (soft_pred, _) =
                spacetravlr::structure::infer_neighbor_composition(&st, &types)?;

            for (name, pred) in [
                ("type_mean_oracle", &oracle),
                ("structure_pooled", &pooled),
                ("abundance_baseline", &abund),
            ] {
                let (mae, rmse, rel) = matrix_error_metrics(pred, &truth);
                let pears = column_pearson(pred, &truth);
                let pearson_mean = pears.mean().unwrap_or(f64::NAN);
                println!(
                    "{name}: pearson_mean={pearson_mean:.4} mae={mae:.6} rmse={rmse:.6} rel_mae={rel:.4}"
                );
            }
            let soft_cos = mean_composition_cosine(&soft_pred, &cell.soft_counts);
            println!(
                "neighbor soft-composition cosine (pooled vs true): {soft_cos:.4}"
            );
            println!(
                "ligands={} cells={} genes_sample={:?}",
                lig_idx.len(),
                n,
                lig_idx
                    .iter()
                    .take(5)
                    .map(|&j| genes.get(j).cloned().unwrap_or_default())
                    .collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}
