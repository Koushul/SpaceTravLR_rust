use anndata::{AnnData, AnnDataOp, ArrayElemOp, Backend};
use anyhow::{bail, Context, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array2, Axis};
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;

pub fn x_to_dense_f64<B: Backend>(ad: &AnnData<B>) -> Result<Array2<f64>> {
    let elem = ad.x();
    if elem.is_none() {
        bail!("AnnData has no X matrix");
    }
    if let Ok(Some(arr)) = elem.get::<Array2<f64>>() {
        return Ok(arr);
    }
    if let Ok(Some(arr)) = elem.get::<Array2<f32>>() {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(Some(arr)) = elem.get::<Array2<i32>>() {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(Some(arr)) = elem.get::<Array2<i64>>() {
        return Ok(arr.mapv(|v| v as f64));
    }
    if let Ok(Some(csr)) = elem.get::<CsrMatrix<f64>>() {
        return Ok(csr_to_dense_f64_csr(&csr));
    }
    if let Ok(Some(csr)) = elem.get::<CsrMatrix<f32>>() {
        return Ok(csr_to_dense_f32_csr(&csr));
    }
    bail!(
        "unsupported X layout / dtype {:?}; need dense numeric X or CSR float matrix",
        elem.dtype()
    );
}

fn csr_to_dense_f64_csr(csr: &CsrMatrix<f64>) -> Array2<f64> {
    let mut out = Array2::zeros((csr.nrows(), csr.ncols()));
    for r in 0..csr.nrows() {
        let row = csr.row(r);
        for (c, v) in row.col_indices().iter().zip(row.values().iter()) {
            out[[r, *c]] = *v;
        }
    }
    out
}

fn csr_to_dense_f32_csr(csr: &CsrMatrix<f32>) -> Array2<f64> {
    let mut out = Array2::zeros((csr.nrows(), csr.ncols()));
    for r in 0..csr.nrows() {
        let row = csr.row(r);
        for (c, v) in row.col_indices().iter().zip(row.values().iter()) {
            out[[r, *c]] = *v as f64;
        }
    }
    out
}

fn series_to_labels(series: &Series) -> Result<Vec<String>> {
    let s = series
        .cast(&DataType::String)
        .with_context(|| "cast obs column to string")?;
    let ca = s.str().with_context(|| "obs column as Utf8")?;
    Ok((0..ca.len())
        .map(|i| ca.get(i).unwrap_or("").to_string())
        .collect())
}

/// Mean normalized expression per cell type (matches Python `Reference` profile construction).
/// `x` is **cells × genes**; `labels` is one cell type label per row.
/// Returns matrix **K × G** (one row per cell type, same row order as returned names).
pub fn single_cell_reference_profiles_from_arrays(
    x: &Array2<f64>,
    labels: &[String],
    cell_min: usize,
    min_umi: f64,
    n_max_cells: usize,
) -> Result<(Array2<f64>, Vec<String>)> {
    let n = x.nrows();
    let g = x.ncols();
    if labels.len() != n {
        bail!("labels length {} != matrix rows {}", labels.len(), n);
    }

    let mut umi: Vec<f64> = Vec::with_capacity(n);
    for r in 0..n {
        umi.push(x.row(r).sum());
    }

    let mut type_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (i, lab) in labels.iter().enumerate() {
        if umi[i] >= min_umi {
            *type_counts.entry(lab.clone()).or_insert(0) += 1;
        }
    }
    let mut unique_types: Vec<String> = type_counts
        .iter()
        .filter(|(_, c)| **c >= cell_min)
        .map(|(t, _)| t.clone())
        .collect();
    unique_types.sort();

    if unique_types.len() < 2 {
        bail!(
            "fewer than 2 cell types with ≥{cell_min} cells after UMI≥{min_umi} filter; lower --ref-cell-min or --ref-min-umi"
        );
    }

    let valid: std::collections::HashSet<String> = unique_types.iter().cloned().collect();
    let mut keep = vec![false; n];
    for i in 0..n {
        keep[i] = umi[i] >= min_umi && valid.contains(&labels[i]);
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut pick: Vec<usize> = Vec::new();
    for ct in &unique_types {
        let mut idx: Vec<usize> = (0..n).filter(|&i| keep[i] && labels[i] == *ct).collect();
        if idx.len() > n_max_cells {
            let smp = sample(&mut rng, idx.len(), n_max_cells);
            idx = smp.iter().map(|j| idx[j]).collect();
        } else {
            idx.sort();
        }
        pick.extend(idx);
    }
    pick.sort_unstable();
    pick.dedup();

    let k = unique_types.len();
    let mut profiles = Array2::<f64>::zeros((k, g));
    for (ti, ct) in unique_types.iter().enumerate() {
        let cells: Vec<usize> = pick.iter().copied().filter(|&i| labels[i] == *ct).collect();
        if cells.is_empty() {
            bail!("no cells for type {ct:?} after subsampling");
        }
        for gene in 0..g {
            let mut acc = 0.0f64;
            for &i in &cells {
                let u = umi[i].max(1e-12);
                acc += x[[i, gene]] / u;
            }
            profiles[[ti, gene]] = acc / cells.len() as f64;
        }
    }

    Ok((profiles, unique_types))
}

/// Mean normalized expression per cell type (matches Python `Reference` profile construction).
/// Returns matrix **K × G** (one row per cell type, same row order as `cell_type_names`).
pub fn single_cell_reference_profiles<B: Backend>(
    ad: &AnnData<B>,
    cell_type_col: &str,
    cell_min: usize,
    min_umi: f64,
    n_max_cells: usize,
) -> Result<(Array2<f64>, Vec<String>)> {
    let df = ad.read_obs().with_context(|| "read reference obs")?;
    let col = df
        .column(cell_type_col)
        .with_context(|| format!("obs column '{cell_type_col}' not found"))?;
    let labels = series_to_labels(col.as_materialized_series())?;
    let x = x_to_dense_f64(ad)?;
    single_cell_reference_profiles_from_arrays(&x, &labels, cell_min, min_umi, n_max_cells)
}

pub fn normalize_columns(mat: &Array2<f64>) -> Array2<f64> {
    let mut out = mat.clone();
    for mut col in out.axis_iter_mut(Axis(1)) {
        let s: f64 = col.sum();
        if s > 0.0 {
            col /= s;
        }
    }
    out
}
