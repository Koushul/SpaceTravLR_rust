use std::collections::{HashMap, HashSet};

use anndata::data::SelectInfoElem;
use anndata::{AnnData, AnnDataOp, AxisArraysOp, Backend};
use anndata_hdf5::H5;
use ndarray::{s, Array2};
use polars::datatypes::AnyValue;
use polars::prelude::*;

use crate::spatial_estimator::{
    load_spatial_coords_f64, obsm_get_dense_matrix_f64, read_expression_matrix_dense_f64,
};

pub fn open_adata(path: &str) -> anyhow::Result<AnnData<H5>> {
    AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))
}

pub fn spatial_xy(adata: &AnnData<H5>) -> anyhow::Result<Array2<f64>> {
    load_spatial_coords_f64(adata)
}

pub fn spatial_obsm_key_used(adata: &AnnData<H5>) -> anyhow::Result<String> {
    const KEYS: [&str; 3] = ["spatial", "X_spatial", "spatial_loc"];
    for key in KEYS {
        if let Some(arr) = obsm_get_dense_matrix_f64(adata, key)? {
            if arr.nrows() > 0 && arr.ncols() >= 2 {
                return Ok(key.to_string());
            }
        }
    }
    let keys = adata.obsm().keys();
    anyhow::bail!(
        "No usable 2D spatial coordinates in obsm (tried {:?}). Keys: {:?}",
        KEYS.as_slice(),
        keys
    )
}

/// Scanpy-style embedding: `obsm["X_umap"]` or `obsm["umap"]`, first two columns.
/// Returns `None` if missing or row count ≠ `n_obs`.
pub fn try_umap_xy(
    adata: &AnnData<H5>,
    n_obs: usize,
) -> anyhow::Result<Option<(String, Array2<f64>)>> {
    const KEYS: [&str; 2] = ["X_umap", "umap"];
    for key in KEYS {
        if let Some(arr) = obsm_get_dense_matrix_f64(adata, key)? {
            if arr.nrows() == n_obs && arr.ncols() >= 2 {
                let xy = arr.slice(s![.., ..2]).to_owned();
                return Ok(Some((key.to_string(), xy)));
            }
        }
    }
    Ok(None)
}

pub fn gene_expression_f32(
    adata: &AnnData<H5>,
    layer: &str,
    gene: &str,
) -> anyhow::Result<Vec<f32>> {
    let idx = adata
        .var_names()
        .get_index(gene)
        .ok_or_else(|| anyhow::anyhow!("gene {:?} not found in var_names", gene))?;
    let slice = [
        SelectInfoElem::full(),
        SelectInfoElem::Index(vec![idx]),
    ];
    let data = read_expression_matrix_dense_f64(adata, layer, &slice)?;
    Ok(data.column(0).iter().map(|v| *v as f32).collect())
}

pub fn obs_names(adata: &AnnData<H5>) -> Vec<String> {
    adata.obs_names().into_vec()
}

pub fn var_names(adata: &AnnData<H5>) -> Vec<String> {
    adata.var_names().into_vec()
}

pub fn genes_with_prefix(adata: &AnnData<H5>, prefix: &str, limit: usize) -> Vec<String> {
    let names = adata.var_names().into_vec();
    if prefix.is_empty() {
        names.into_iter().take(limit).collect()
    } else {
        names
            .into_iter()
            .filter(|n| n.starts_with(prefix))
            .take(limit)
            .collect()
    }
}

pub fn clusters_from_obs_column(
    adata: &AnnData<H5>,
    cluster_annot: &str,
) -> anyhow::Result<Vec<usize>> {
    let obs = adata.read_obs()?;
    let col = obs
        .column(cluster_annot)
        .map_err(|_| {
            let preview: Vec<String> = obs
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .take(25)
                .collect();
            anyhow::anyhow!(
                "obs column {:?} not found. First columns: {:?}",
                cluster_annot,
                preview
            )
        })?;
    let f = col.cast(&DataType::Float64).map_err(|e| {
        anyhow::anyhow!(
            "obs column {:?} must be numeric (cluster ids): {}",
            cluster_annot,
            e
        )
    })?;
    let ca = f.f64()?;
    Ok(ca
        .into_iter()
        .map(|v| v.unwrap_or(0.0).round() as i64 as usize)
        .collect())
}

pub fn spatial_xy_f32_interleaved(xy: &Array2<f64>) -> Vec<f32> {
    let n = xy.nrows();
    let mut v = Vec::with_capacity(n * 2);
    for i in 0..n {
        v.push(xy[[i, 0]] as f32);
        v.push(xy[[i, 1]] as f32);
    }
    v
}

pub fn f32_vec_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Little-endian `u32` per cell, for the spatial viewer (cluster ids from obs).
pub fn clusters_as_u32_le_bytes(clusters: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(clusters.len() * 4);
    for &c in clusters {
        let u = u32::try_from(c).unwrap_or(u32::MAX);
        out.extend_from_slice(&u.to_le_bytes());
    }
    out
}

fn resolve_cell_type_column(obs: &DataFrame) -> Option<String> {
    let names = obs.get_column_names();
    const PREFERRED: &[&str] = &["cell_type", "cell_types", "celltype", "major_cell_type"];
    for p in PREFERRED {
        if let Some(n) = names.iter().find(|n| n.to_string().as_str() == *p) {
            return Some(n.to_string());
        }
    }
    for n in names {
        let s = n.to_string();
        if s.eq_ignore_ascii_case("cell_type") {
            return Some(s);
        }
    }
    None
}

fn any_value_to_cell_type_str(v: AnyValue<'_>) -> String {
    match v {
        AnyValue::Null => String::new(),
        AnyValue::String(s) => s.to_string(),
        AnyValue::StringOwned(s) => s.to_string(),
        _ => v.to_string(),
    }
}

/// Resolved `obs` column name, sorted category list, and per-obs category index (`u16::MAX` = unknown).
pub fn cell_type_encoding(
    adata: &AnnData<H5>,
) -> anyhow::Result<Option<(String, Vec<String>, Vec<u16>)>> {
    let obs = adata.read_obs()?;
    let Some(col_name) = resolve_cell_type_column(&obs) else {
        return Ok(None);
    };
    let cell_col = obs
        .column(&col_name)
        .map_err(|_| anyhow::anyhow!("obs column {:?} missing", col_name))?;
    let series = cell_col.as_materialized_series();
    let mut uniq: HashSet<String> = HashSet::new();
    for v in series.iter() {
        let s = any_value_to_cell_type_str(v);
        let t = s.trim();
        if t.is_empty() || t.eq_ignore_ascii_case("null") {
            continue;
        }
        uniq.insert(t.to_string());
    }
    let mut categories: Vec<String> = uniq.into_iter().collect();
    categories.sort();
    if categories.len() >= u16::MAX as usize {
        anyhow::bail!("too many distinct cell_type labels");
    }
    let cat_to_id: std::collections::HashMap<String, u16> = categories
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u16))
        .collect();
    let mut codes = Vec::with_capacity(series.len());
    for v in series.iter() {
        let s = any_value_to_cell_type_str(v);
        let t = s.trim();
        if t.is_empty() || t.eq_ignore_ascii_case("null") {
            codes.push(u16::MAX);
        } else {
            codes.push(*cat_to_id.get(t).unwrap_or(&u16::MAX));
        }
    }
    Ok(Some((col_name, categories, codes)))
}

pub fn u16_vec_to_le_bytes(codes: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.len() * 2);
    for &c in codes {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out
}

pub fn cell_expression_map(
    adata: &AnnData<H5>,
    layer: &str,
    cell_idx: usize,
) -> anyhow::Result<HashMap<String, f64>> {
    let vn = var_names(adata);
    let slice = [
        SelectInfoElem::Index(vec![cell_idx]),
        SelectInfoElem::full(),
    ];
    let data = read_expression_matrix_dense_f64(adata, layer, &slice)?;
    anyhow::ensure!(data.nrows() == 1, "expected one row, got {}", data.nrows());
    let row = data.row(0);
    let mut out = HashMap::with_capacity(vn.len());
    for (j, name) in vn.iter().enumerate() {
        out.insert(name.clone(), row[j]);
    }
    Ok(out)
}

/// Mean expression of `gene` over `cell_indices` (empty slice → 0). Missing gene → error.
pub fn mean_expression_over_cells(
    adata: &AnnData<H5>,
    layer: &str,
    cell_indices: &[usize],
    gene: &str,
    var_names: &[String],
) -> anyhow::Result<f64> {
    if cell_indices.is_empty() {
        return Ok(0.0);
    }
    let j = var_names
        .iter()
        .position(|v| v == gene)
        .ok_or_else(|| anyhow::anyhow!("gene {:?} not in var_names", gene))?;
    let slice = [
        SelectInfoElem::Index(cell_indices.to_vec()),
        SelectInfoElem::Index(vec![j]),
    ];
    let data = read_expression_matrix_dense_f64(adata, layer, &slice)?;
    let mut s = 0.0_f64;
    for i in 0..data.nrows() {
        s += data[[i, 0]];
    }
    Ok(s / data.nrows() as f64)
}

pub fn expression_profiles_for_cells(
    adata: &AnnData<H5>,
    layer: &str,
    cell_indices: &[usize],
    genes: &[String],
    var_names: &[String],
) -> anyhow::Result<HashMap<usize, HashMap<String, f64>>> {
    let name_to_j: HashMap<&str, usize> = var_names
        .iter()
        .enumerate()
        .map(|(j, s)| (s.as_str(), j))
        .collect();
    let mut col_idx = Vec::new();
    let mut gene_order: Vec<String> = Vec::new();
    for g in genes {
        if let Some(&j) = name_to_j.get(g.as_str()) {
            col_idx.push(j);
            gene_order.push(g.clone());
        }
    }
    let mut out: HashMap<usize, HashMap<String, f64>> = HashMap::new();
    if cell_indices.is_empty() {
        return Ok(out);
    }
    if col_idx.is_empty() {
        for &ci in cell_indices {
            out.insert(ci, HashMap::new());
        }
        return Ok(out);
    }
    let slice = [
        SelectInfoElem::Index(cell_indices.to_vec()),
        SelectInfoElem::Index(col_idx),
    ];
    let mat = read_expression_matrix_dense_f64(adata, layer, &slice)?;
    for (ri, &ci) in cell_indices.iter().enumerate() {
        let mut m = HashMap::with_capacity(gene_order.len());
        for (cj, g) in gene_order.iter().enumerate() {
            m.insert(g.clone(), mat[[ri, cj]]);
        }
        out.insert(ci, m);
    }
    Ok(out)
}
