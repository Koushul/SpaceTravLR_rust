use anyhow::Context;
use crate::betadata::write_betadata_feather;
use crate::cnn_gating::{
    CnnGateDecision, build_neighbors, decide_cnn_for_gene, load_gene_set_file, predict_lasso_y,
};
use crate::config::{
    CnnConfig, CnnTrainingMode, HybridCnnGatingConfig, ModelExportConfig, RUN_REPRO_TOML_FILENAME,
    SpaceshipConfig, expand_user_path,
};
use crate::estimator::{CachedSpatialData, ClusteredGCNNWR, finite_or_zero_f64};
use crate::lasso::GroupLassoParams;
use crate::ligand::{calculate_weighted_ligands, calculate_weighted_ligands_grid};
use crate::modulator_scale::{
    apply_modulator_scales_inplace, scale_columns_no_center, unscale_betadata_columns_inplace,
};
use crate::run_summary_html::{RunSummaryParams, write_run_summary_html};
use crate::training_hud::{
    TrainingHud, log_line, pipeline_step_begin, pipeline_step_end, print_training_outcome_banner,
};
use anndata::data::{ArrayConvert, SelectInfoElem};
use anndata::{
    AnnData, AnnDataOp, ArrayData, ArrayElemOp, AxisArraysOp, Backend, ElemCollectionOp,
};
use anndata_hdf5::H5;
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra_sparse::{csc::CscMatrix, csr::CsrMatrix};
use ndarray::{Array1, Array2, Array4};
use ndarray_npy::NpzWriter;
use polars::prelude::{DataFrame, NamedFrom, Series};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

const LARGE_DATASET_GRID_AUTO_CELLS: usize = 15_000;
const DEFAULT_LIGAND_GRID_FACTOR: f64 = 0.5;

/// Remove `*.lock` files in `dir` whose modification time is at least `stale_secs` old.
/// Returns how many files were deleted. No-op if `stale_secs == 0` or `dir` is not a directory.
pub(crate) fn remove_stale_lock_files_in_dir(dir: &Path, stale_secs: u64) -> usize {
    if stale_secs == 0 || !dir.is_dir() {
        return 0;
    }
    let now = SystemTime::now();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0usize;
    for e in entries.flatten() {
        let Ok(meta) = e.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let name = e.file_name();
        let Some(n) = name.to_str() else {
            continue;
        };
        if !n.ends_with(".lock") {
            continue;
        }
        let path = e.path();
        let Ok(modified) = meta.modified() else {
            continue;
        };
        let Ok(age) = now.duration_since(modified) else {
            continue;
        };
        if age.as_secs() >= stale_secs && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

pub(crate) fn compute_gene_mean_expression<AnB: Backend>(
    adata: &AnnData<AnB>,
    layer: &str,
    obs_row_subset: Option<&[usize]>,
) -> anyhow::Result<HashMap<String, f64>> {
    let var_names = adata.var_names().into_vec();
    let row_sel = match obs_row_subset {
        Some(rows) => SelectInfoElem::Index(rows.to_vec()),
        None => SelectInfoElem::full(),
    };
    let slice = [row_sel, SelectInfoElem::full()];
    let data = read_expression_matrix_dense_f64(adata, layer, &slice)?;
    let n_obs = data.nrows();
    if n_obs == 0 {
        return Ok(HashMap::new());
    }
    let inv_n = 1.0 / n_obs as f64;
    let mut out = HashMap::with_capacity(var_names.len());
    for (j, name) in var_names.iter().enumerate() {
        let sum: f64 = data.column(j).iter().sum();
        out.insert(name.clone(), sum * inv_n);
    }
    Ok(out)
}

fn csr_to_dense_f64(csr: &CsrMatrix<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((csr.nrows(), csr.ncols()));
    for (r, c, v) in csr.triplet_iter() {
        out[[r, c]] = *v;
    }
    out
}

fn csc_to_dense_f64(csc: &CscMatrix<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((csc.nrows(), csc.ncols()));
    for (r, c, v) in csc.triplet_iter() {
        out[[r, c]] = *v;
    }
    out
}

fn array_data_to_dense_f64(data: ArrayData) -> anyhow::Result<Array2<f64>> {
    match data {
        ArrayData::Array(d) => d.try_convert(),
        ArrayData::CsrMatrix(csr) => {
            let csr_f64: CsrMatrix<f64> = csr.try_convert()?;
            Ok(csr_to_dense_f64(&csr_f64))
        }
        ArrayData::CscMatrix(csc) => {
            let csc_f64: CscMatrix<f64> = csc.try_convert()?;
            Ok(csc_to_dense_f64(&csc_f64))
        }
        ArrayData::CsrNonCanonical(non) => match non.canonicalize() {
            Ok(csr) => {
                let csr_f64: CsrMatrix<f64> = csr.try_convert()?;
                Ok(csr_to_dense_f64(&csr_f64))
            }
            Err(_) => anyhow::bail!("Failed to canonicalize non-canonical CSR matrix."),
        },
        ArrayData::DataFrame(_) => anyhow::bail!("Expected matrix array data, found dataframe."),
    }
}

fn expression_array_data_for_slice<AnB: Backend>(
    adata: &AnnData<AnB>,
    layer: &str,
    slice: &[SelectInfoElem],
) -> anyhow::Result<ArrayData> {
    let data: ArrayData = if layer != "X" && !layer.is_empty() {
        if let Some(layer_elem) = adata.layers().get(layer) {
            layer_elem
                .slice(slice)?
                .ok_or_else(|| anyhow::anyhow!("Failed to slice layer {}", layer))?
        } else {
            let x_elem = adata.x();
            if x_elem.is_none() {
                return Err(anyhow::anyhow!(
                    "Layer '{}' not found and X is empty",
                    layer
                ));
            }
            x_elem
                .slice(slice)?
                .ok_or_else(|| anyhow::anyhow!("Failed to slice X"))?
        }
    } else {
        let x_elem = adata.x();
        if x_elem.is_none() {
            return Err(anyhow::anyhow!("X is empty"));
        }
        x_elem
            .slice(slice)?
            .ok_or_else(|| anyhow::anyhow!("Failed to slice X"))?
    };
    Ok(data)
}

pub(crate) fn read_expression_matrix_dense_f64<AnB: Backend>(
    adata: &AnnData<AnB>,
    layer: &str,
    slice: &[SelectInfoElem],
) -> anyhow::Result<Array2<f64>> {
    let data = expression_array_data_for_slice(adata, layer, slice)?;
    array_data_to_dense_f64(data)
}

/// Open an `.h5ad` and read all of `X` or a named layer as a dense `f64` matrix (obs × var).
pub fn read_h5ad_expression_dense_f64(path: &Path, layer: &str) -> anyhow::Result<Array2<f64>> {
    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    let slice = [SelectInfoElem::full(), SelectInfoElem::full()];
    read_expression_matrix_dense_f64(&adata, layer, &slice)
}

/// Gene symbols in AnnData `var` order (same columns as [`read_h5ad_expression_dense_f64`]).
pub fn read_h5ad_var_names(path: &Path) -> anyhow::Result<Vec<String>> {
    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(adata.var_names().into_vec())
}

/// String values for one `adata.obs` column (e.g. `cell_type`, `leiden`).
pub fn read_h5ad_obs_column_str(path: &Path, key: &str) -> anyhow::Result<Vec<String>> {
    use polars::prelude::*;
    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    let obs = adata.read_obs().map_err(|e| anyhow::anyhow!("{}", e))?;
    let col = obs.column(key).map_err(|_| {
        anyhow::anyhow!("obs column {:?} missing", key)
    })?;
    let cast_col = col.cast(&DataType::String)?;
    let s = cast_col.str().map_err(|_| {
        anyhow::anyhow!("obs column {:?} could not be cast to string", key)
    })?;
    let mut out = Vec::with_capacity(s.len());
    for i in 0..s.len() {
        out.push(s.get(i).unwrap_or("").to_string());
    }
    adata.close()?;
    Ok(out)
}

/// When the sliced `X`/layer is canonical CSR in AnnData, returns it without densifying (useful for
/// future column-wise pipelines). Dense or CSC layouts return [`None`] — use [`read_expression_matrix_dense_f64`].
#[allow(dead_code)]
pub(crate) fn try_read_expression_csr_f64<AnB: Backend>(
    adata: &AnnData<AnB>,
    layer: &str,
    slice: &[SelectInfoElem],
) -> anyhow::Result<Option<CsrMatrix<f64>>> {
    let data = expression_array_data_for_slice(adata, layer, slice)?;
    match data {
        ArrayData::CsrMatrix(csr) => Ok(Some(csr.try_convert()?)),
        ArrayData::CsrNonCanonical(non) => match non.canonicalize() {
            Ok(csr) => Ok(Some(csr.try_convert()?)),
            Err(_) => anyhow::bail!("Failed to canonicalize non-canonical CSR matrix."),
        },
        ArrayData::Array(_) | ArrayData::CscMatrix(_) | ArrayData::DataFrame(_) => Ok(None),
    }
}

/// Read `obsm[key]` as a dense matrix in `f64`. HDF5 may store the same logical array as `f32` or
/// `f64`; requesting the wrong Rust element type returns `Err` from anndata, so we try both.
pub(crate) fn obsm_get_dense_matrix_f64<AnB: Backend>(
    adata: &AnnData<AnB>,
    key: &str,
) -> anyhow::Result<Option<Array2<f64>>> {
    if let Ok(Some(arr)) = adata.obsm().get_item::<Array2<f32>>(key) {
        return Ok(Some(arr.mapv(|v| v as f64)));
    }
    if let Ok(Some(arr)) = adata.obsm().get_item::<Array2<f64>>(key) {
        return Ok(Some(arr));
    }
    Ok(None)
}

pub(crate) fn load_spatial_coords_f64<AnB: Backend>(
    adata: &AnnData<AnB>,
) -> anyhow::Result<Array2<f64>> {
    const KEYS: [&str; 3] = ["spatial", "X_spatial", "spatial_loc"];
    for key in KEYS {
        if let Some(arr) = obsm_get_dense_matrix_f64(adata, key)? {
            if arr.nrows() > 0 && arr.ncols() >= 2 {
                return Ok(arr);
            }
        }
    }
    let obsm_keys = adata.obsm().keys();
    anyhow::bail!(
        "No usable 2D spatial coordinates in obsm (tried {:?}, need ≥2 columns). obsm keys: {:?}.",
        KEYS.as_slice(),
        obsm_keys
    );
}

fn ensure_expression_layer_readable<AnB: Backend>(
    adata: &AnnData<AnB>,
    layer: &str,
) -> anyhow::Result<()> {
    let slice = [SelectInfoElem::full(), SelectInfoElem::full()];
    if layer != "X" && !layer.is_empty() && adata.layers().get(layer).is_none() {
        let keys = adata.layers().keys();
        let preview: Vec<String> = keys.into_iter().take(20).collect();
        anyhow::bail!(
            "Expression layer {:?} is missing from AnnData.layers. Known keys (first 20): {:?}. Fix [data].layer in spaceship_config.toml or use \"X\".",
            layer,
            preview
        );
    }
    let data = read_expression_matrix_dense_f64(adata, layer, &slice)?;
    if data.nrows() != adata.n_obs() {
        anyhow::bail!(
            "Expression matrix has {} rows but n_obs is {}; check layer / AnnData integrity.",
            data.nrows(),
            adata.n_obs()
        );
    }
    Ok(())
}

fn validate_training_inputs<AnB: Backend>(
    adata: &AnnData<AnB>,
    cluster_annot: &str,
    layer: &str,
    n_targets: usize,
) -> anyhow::Result<DataFrame> {
    if adata.n_obs() == 0 {
        anyhow::bail!("AnnData has 0 cells (n_obs=0); nothing to train.");
    }
    if adata.var_names().into_vec().is_empty() {
        anyhow::bail!("AnnData has 0 genes (n_vars=0); nothing to train.");
    }
    if n_targets == 0 {
        anyhow::bail!(
            "Gene list is empty after --genes / max_genes filters; widen the filter or remove --genes."
        );
    }
    let obs = adata.read_obs()?;
    if obs.column(cluster_annot).is_err() {
        let names: Vec<String> = obs
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .take(25)
            .collect();
        anyhow::bail!(
            "obs column {:?} not found (needed for per-cluster training). First obs columns: {:?}. Set [data].cluster_annot in spaceship_config.toml.",
            cluster_annot,
            names
        );
    }
    let _ = clusters_array1_from_obs_column(&obs, cluster_annot)?;
    ensure_expression_layer_readable(adata, layer)?;
    Ok(obs)
}

/// Per-cell cluster indices for `cluster_annot`: numeric, categorical codes, or stable indices from strings.
fn clusters_array1_from_obs_column(
    obs_df: &DataFrame,
    cluster_annot: &str,
) -> anyhow::Result<Array1<usize>> {
    use polars::prelude::DataType;
    let col = obs_df.column(cluster_annot)?;
    let s = col.as_materialized_series();
    let n = obs_df.height();
    let v: Vec<usize> = match s.dtype() {
        DataType::String => {
            let ca = s
                .str()
                .map_err(|e| anyhow::anyhow!("obs column {:?} as string: {}", cluster_annot, e))?;
            let mut seen = HashSet::<String>::new();
            for opt in ca.into_iter() {
                if let Some(x) = opt {
                    seen.insert(x.to_string());
                }
            }
            let mut uniq: Vec<String> = seen.into_iter().collect();
            uniq.sort();
            let map: HashMap<String, usize> =
                uniq.into_iter().enumerate().map(|(i, k)| (k, i)).collect();
            ca.into_iter()
                .map(|o| {
                    o.and_then(|x| map.get(x).copied())
                        .unwrap_or(0)
                })
                .collect()
        }
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let phys = s.cast(&DataType::UInt32).map_err(|e| {
                anyhow::anyhow!(
                    "obs column {:?} (categorical) could not be read as codes: {}",
                    cluster_annot,
                    e
                )
            })?;
            let ca = phys.u32().map_err(|e| anyhow::anyhow!("{}", e))?;
            ca.into_iter()
                .map(|o| o.unwrap_or(0) as usize)
                .collect()
        }
        _ => {
            let f_series = s.cast(&DataType::Float64).map_err(|e| {
                anyhow::anyhow!(
                    "obs column {:?} must be numeric, string, or categorical for clusters: {}",
                    cluster_annot,
                    e
                )
            })?;
            let f = f_series
                .f64()
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            f.into_iter()
                .map(|o| o.unwrap_or(0.0).round() as usize)
                .collect()
        }
    };
    anyhow::ensure!(
        v.len() == n,
        "cluster column {:?} length {} != n_obs {}",
        cluster_annot,
        v.len(),
        n
    );
    Ok(Array1::from_vec(v))
}

fn resolve_obs_cell_type_label_column(obs_df: &DataFrame) -> Option<String> {
    let names = obs_df.get_column_names();
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

fn cell_type_label_counts_from_obs(obs_df: &DataFrame) -> Vec<(String, usize)> {
    let Some(ct_name) = resolve_obs_cell_type_label_column(obs_df) else {
        return Vec::new();
    };
    let Ok(cell_col) = obs_df.column(&ct_name) else {
        return Vec::new();
    };
    let series = cell_col.as_materialized_series();
    let mut map: HashMap<String, usize> = HashMap::new();
    for v in series.iter() {
        let key = v.to_string();
        if key != "null" && !key.trim().is_empty() {
            *map.entry(key).or_insert(0) += 1;
        }
    }
    let mut v: Vec<(String, usize)> = map.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v
}

fn build_cluster_to_cell_type_map(
    obs_df: &DataFrame,
    cluster_annot: &str,
) -> anyhow::Result<HashMap<usize, String>> {
    let Some(ct_name) = resolve_obs_cell_type_label_column(obs_df) else {
        return Ok(HashMap::new());
    };
    let cell_col = match obs_df.column(&ct_name) {
        Ok(c) => c,
        Err(_) => return Ok(HashMap::new()),
    };
    let cluster_ids = clusters_array1_from_obs_column(obs_df, cluster_annot)?;
    let cell_ser = cell_col.as_materialized_series();

    let mut counts: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    for (i, ct) in cell_ser.iter().enumerate() {
        if i >= cluster_ids.len() {
            break;
        }
        let cid = cluster_ids[i];
        let cell_t = ct.to_string();
        if cell_t != "null" && !cell_t.trim().is_empty() {
            *counts.entry(cid).or_default().entry(cell_t).or_insert(0) += 1;
        }
    }

    let mut out = HashMap::new();
    for (cid, m) in counts {
        if let Some((best, _)) = m.into_iter().max_by_key(|(_, n)| *n) {
            out.insert(cid, best);
        }
    }
    Ok(out)
}

fn detect_spatial_obsm_key<AnB: Backend>(adata: &AnnData<AnB>) -> anyhow::Result<String> {
    for key in ["spatial", "X_spatial", "spatial_loc"] {
        if let Some(arr) = obsm_get_dense_matrix_f64(adata, key)? {
            if arr.nrows() > 0 && arr.ncols() >= 2 {
                return Ok(key.to_string());
            }
        }
    }
    anyhow::bail!("No spatial coordinates found in obsm for minimal reproducibility export.");
}

pub fn dense_to_csr_f64(arr: &Array2<f64>) -> anyhow::Result<CsrMatrix<f64>> {
    let nrows = arr.nrows();
    let ncols = arr.ncols();
    let mut indptr = Vec::with_capacity(nrows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for i in 0..nrows {
        for j in 0..ncols {
            let v = arr[[i, j]];
            if v != 0.0 {
                indices.push(j);
                data.push(v);
            }
        }
        indptr.push(indices.len());
    }
    CsrMatrix::try_from_csr_data(nrows, ncols, indptr, indices, data)
        .map_err(|e| anyhow::anyhow!("build CSR from dense failed: {}", e))
}

fn ensure_sparse_array_data(data: ArrayData) -> anyhow::Result<ArrayData> {
    match data {
        ArrayData::CsrMatrix(_) | ArrayData::CscMatrix(_) | ArrayData::CsrNonCanonical(_) => {
            Ok(data)
        }
        ArrayData::Array(d) => {
            let dense: Array2<f64> = d.try_convert()?;
            Ok(ArrayData::from(dense_to_csr_f64(&dense)?))
        }
        ArrayData::DataFrame(_) => anyhow::bail!("Expected matrix data, found dataframe."),
    }
}

fn build_spatial_maps_flat_csr(
    spatial_maps: &Array4<f32>,
    clusters: &Array1<usize>,
    num_clusters: usize,
    spatial_dim: usize,
) -> anyhow::Result<CsrMatrix<f64>> {
    let n_cells = spatial_maps.shape()[0];
    let hw = spatial_dim * spatial_dim;
    let total_cols = num_clusters * hw;
    let mut indptr = Vec::with_capacity(n_cells + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);

    for cell in 0..n_cells {
        let c = clusters[cell];
        if c >= num_clusters {
            indptr.push(indices.len());
            continue;
        }
        let base = c * hw;
        for i in 0..spatial_dim {
            for j in 0..spatial_dim {
                let v = spatial_maps[[cell, c, i, j]] as f64;
                if v != 0.0 {
                    indices.push(base + i * spatial_dim + j);
                    data.push(v);
                }
            }
        }
        indptr.push(indices.len());
    }
    CsrMatrix::try_from_csr_data(n_cells, total_cols, indptr, indices, data)
        .map_err(|e| anyhow::anyhow!("build CSR for spatial maps failed: {}", e))
}

fn export_minimal_repro_adata_with_cache(
    src: &AnnData<H5>,
    output_dir: &str,
    layer: &str,
    cluster_annot: &str,
    xy: &Array2<f64>,
    clusters: &Array1<usize>,
    num_clusters: usize,
    spatial_dim: usize,
    spatial_feature_radius: f64,
) -> anyhow::Result<String> {
    let out_path = format!("{}/spacetravlr_minimal_repro.h5ad", output_dir);
    let dst = AnnData::<H5>::new(&out_path)?;

    dst.set_obs_names(src.obs_names())?;
    dst.set_var_names(src.var_names())?;

    let obs = src.read_obs()?;
    let cell_type = obs
        .column("cell_type")
        .map_err(|_| anyhow::anyhow!("obs.cell_type is required for minimal reproducibility file"))?
        .clone();
    let obs_min = DataFrame::new(vec![cell_type])?;
    dst.set_obs(obs_min)?;

    let spatial_key = detect_spatial_obsm_key(src)?;
    let spatial_xy = obsm_get_dense_matrix_f64(src, &spatial_key)?.ok_or_else(|| {
        anyhow::anyhow!(
            "obsm[{:?}] not readable as dense f32/f64 matrix",
            spatial_key
        )
    })?;
    dst.obsm().add(&spatial_key, spatial_xy)?;

    let x_data = if let Some(xe) = src.layers().get("imputed_count") {
        xe.get::<ArrayData>()?
            .ok_or_else(|| anyhow::anyhow!("imputed_count exists but cannot be read"))?
    } else {
        src.x()
            .get::<ArrayData>()?
            .ok_or_else(|| anyhow::anyhow!("source X is empty"))?
    };
    dst.set_x(ensure_sparse_array_data(x_data)?)?;

    for layer_name in ["imputed_count", "normalized_count", "raw_count"] {
        if let Some(elem) = src.layers().get(layer_name) {
            let data: ArrayData = match elem.get::<ArrayData>() {
                Ok(Some(v)) => v,
                Ok(None) => continue,
                Err(e) => {
                    if layer_name == layer {
                        return Err(e);
                    }
                    continue;
                }
            };
            dst.layers()
                .add(layer_name, ensure_sparse_array_data(data)?)?;
        }
    }

    if dst.layers().get(layer).is_none() && layer != "X" {
        anyhow::bail!(
            "Configured layer {:?} is missing from minimal reproducibility AnnData.",
            layer
        );
    }

    let spatial_features = crate::estimator::create_spatial_features(
        xy,
        clusters,
        num_clusters,
        spatial_feature_radius,
    );
    let spatial_maps =
        crate::estimator::xyc2spatial_fast(xy, clusters, num_clusters, spatial_dim, spatial_dim);
    let spatial_features_csr = dense_to_csr_f64(&spatial_features)?;
    let spatial_maps_flat_csr =
        build_spatial_maps_flat_csr(&spatial_maps, clusters, num_clusters, spatial_dim)?;

    dst.obsm()
        .add("spacetravlr_spatial_features", spatial_features_csr)?;
    dst.obsm()
        .add("spacetravlr_spatial_maps_flat", spatial_maps_flat_csr)?;
    dst.uns()
        .add("spacetravlr_cache_cluster_annot", cluster_annot.to_string())?;
    dst.uns()
        .add("spacetravlr_cache_spatial_dim", spatial_dim as i64)?;
    dst.uns()
        .add("spacetravlr_cache_num_clusters", num_clusters as i64)?;
    dst.uns().add(
        "spacetravlr_cache_spatial_feature_radius",
        spatial_feature_radius,
    )?;

    dst.close()?;
    Ok(out_path)
}

fn verify_minimal_repro_adata_loadable(
    path: &str,
    layer: &str,
    cluster_annot: &str,
) -> anyhow::Result<()> {
    let adata = AnnData::<H5>::open(H5::open(path)?)?;
    let obs = adata.read_obs()?;
    if obs.column("cell_type").is_err() {
        anyhow::bail!("Minimal AnnData missing required obs column cell_type.");
    }
    if obs.column(cluster_annot).is_err() {
        anyhow::bail!(
            "Minimal AnnData missing cluster annotation column {:?}.",
            cluster_annot
        );
    }
    if adata.layers().get("imputed_count").is_none() {
        anyhow::bail!("Minimal AnnData missing layer imputed_count.");
    }
    if layer != "X" && adata.layers().get(layer).is_none() {
        anyhow::bail!(
            "Minimal AnnData missing configured training layer {:?}.",
            layer
        );
    }
    let x_dtype = adata
        .x()
        .dtype()
        .ok_or_else(|| anyhow::anyhow!("Minimal AnnData X is empty"))?;
    if !matches!(
        x_dtype,
        anndata::backend::DataType::CsrMatrix(_) | anndata::backend::DataType::CscMatrix(_)
    ) {
        anyhow::bail!("X is not sparse in minimal AnnData: {:?}", x_dtype);
    }
    for layer_name in ["imputed_count", "normalized_count", "raw_count"] {
        if let Some(elem) = adata.layers().get(layer_name) {
            let dtype = elem
                .dtype()
                .ok_or_else(|| anyhow::anyhow!("layer {:?} is empty", layer_name))?;
            if !matches!(
                dtype,
                anndata::backend::DataType::CsrMatrix(_) | anndata::backend::DataType::CscMatrix(_)
            ) {
                anyhow::bail!("Layer {:?} is not sparse: {:?}", layer_name, dtype);
            }
        }
    }
    validate_training_inputs(&adata, cluster_annot, layer, 1)?;
    let _xy = load_spatial_coords_f64(&adata)?;
    adata.close()?;
    Ok(())
}

fn sanitize_filename(name: &str) -> String {
    name.replace(['/', '\\', ':', ' '], "_")
}

fn export_cnn_models_npz<AB: AutodiffBackend>(
    est: &ClusteredGCNNWR<AB>,
    gene: &str,
    training_dir: &str,
    model_export: &ModelExportConfig,
    excluded_clusters: Option<&HashSet<usize>>,
) -> anyhow::Result<Option<String>> {
    if est.models.is_empty() || !model_export.save_cnn_weights {
        return Ok(None);
    }
    let sub = model_export.output_subdir.trim().trim_matches('/');
    if sub.is_empty() {
        anyhow::bail!("model_export.output_subdir is empty");
    }
    let model_dir = Path::new(training_dir).join(sub);

    let mut cluster_ids: Vec<usize> = est
        .models
        .keys()
        .copied()
        .filter(|c| {
            !excluded_clusters
                .map(|excluded| excluded.contains(c))
                .unwrap_or(false)
        })
        .collect();
    cluster_ids.sort_unstable();
    if cluster_ids.is_empty() {
        return Ok(None);
    }

    std::fs::create_dir_all(&model_dir)?;
    let out_pb = model_dir.join(format!("{}_cnn_weights.npz", sanitize_filename(gene)));
    let out_str = out_pb
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF8 path {:?}", out_pb))?;
    let f = File::create(out_str)?;
    let mut npz = if model_export.compressed_npz {
        NpzWriter::new_compressed(f)
    } else {
        NpzWriter::new(f)
    };
    npz.add_array(
        "cluster_ids",
        &Array1::from_vec(cluster_ids.iter().map(|&x| x as u32).collect()),
    )?;

    let first_c = *cluster_ids.first().unwrap();
    let m0 = est
        .models
        .get(&first_c)
        .ok_or_else(|| anyhow::anyhow!("missing model for cluster {}", first_c))?;
    let sc1 = m0.spatial_features_mlp.l1.weight.shape().dims::<2>();
    let n_clusters_meta = sc1[1] as u32;
    let hl2 = m0.mlp.l2.weight.shape().dims::<2>();
    let n_mods_meta = hl2[0].saturating_sub(1) as u32;
    npz.add_array(
        "meta_spatial_dim",
        &Array1::from_vec(vec![est.spatial_dim as u32]),
    )?;
    npz.add_array("meta_n_clusters", &Array1::from_vec(vec![n_clusters_meta]))?;
    npz.add_array("meta_n_modulators", &Array1::from_vec(vec![n_mods_meta]))?;
    npz.add_array(
        "meta_cnn_output_activation",
        &Array1::from_vec(vec![m0.output_activation as u32]),
    )?;

    for c in cluster_ids {
        let m = est
            .models
            .get(&c)
            .ok_or_else(|| anyhow::anyhow!("missing model for cluster {}", c))?;
        let p = format!("c{:04}_", c);

        let conv1_w = m.conv_layers.conv1.weight.to_data();
        let conv1_shape = m.conv_layers.conv1.weight.shape().dims::<4>();
        let conv1_arr = Array4::from_shape_vec(
            (
                conv1_shape[0],
                conv1_shape[1],
                conv1_shape[2],
                conv1_shape[3],
            ),
            conv1_w
                .as_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .to_vec(),
        )?;
        npz.add_array(format!("{}conv1_weight", p), &conv1_arr)?;
        if let Some(b) = &m.conv_layers.conv1.bias {
            npz.add_array(
                format!("{}conv1_bias", p),
                &Array1::from_vec(
                    b.to_data()
                        .as_slice::<f32>()
                        .map_err(|e| anyhow::anyhow!("{:?}", e))?
                        .to_vec(),
                ),
            )?;
        }
        let conv2_w = m.conv_layers.conv2.weight.to_data();
        let conv2_shape = m.conv_layers.conv2.weight.shape().dims::<4>();
        let conv2_arr = Array4::from_shape_vec(
            (
                conv2_shape[0],
                conv2_shape[1],
                conv2_shape[2],
                conv2_shape[3],
            ),
            conv2_w
                .as_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .to_vec(),
        )?;
        npz.add_array(format!("{}conv2_weight", p), &conv2_arr)?;
        if let Some(b) = &m.conv_layers.conv2.bias {
            npz.add_array(
                format!("{}conv2_bias", p),
                &Array1::from_vec(
                    b.to_data()
                        .as_slice::<f32>()
                        .map_err(|e| anyhow::anyhow!("{:?}", e))?
                        .to_vec(),
                ),
            )?;
        }
        let conv3_w = m.conv_layers.conv3.weight.to_data();
        let conv3_shape = m.conv_layers.conv3.weight.shape().dims::<4>();
        let conv3_arr = Array4::from_shape_vec(
            (
                conv3_shape[0],
                conv3_shape[1],
                conv3_shape[2],
                conv3_shape[3],
            ),
            conv3_w
                .as_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .to_vec(),
        )?;
        npz.add_array(format!("{}conv3_weight", p), &conv3_arr)?;
        if let Some(b) = &m.conv_layers.conv3.bias {
            npz.add_array(
                format!("{}conv3_bias", p),
                &Array1::from_vec(
                    b.to_data()
                        .as_slice::<f32>()
                        .map_err(|e| anyhow::anyhow!("{:?}", e))?
                        .to_vec(),
                ),
            )?;
        }

        npz.add_array(
            format!("{}bn1_gamma", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn1
                    .gamma
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn1_beta", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn1
                    .beta
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn1_running_mean", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn1
                    .running_mean
                    .value()
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn1_running_var", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn1
                    .running_var
                    .value()
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}bn2_gamma", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn2
                    .gamma
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn2_beta", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn2
                    .beta
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn2_running_mean", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn2
                    .running_mean
                    .value()
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn2_running_var", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn2
                    .running_var
                    .value()
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}bn3_gamma", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn3
                    .gamma
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn3_beta", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn3
                    .beta
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn3_running_mean", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn3
                    .running_mean
                    .value()
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}bn3_running_var", p),
            &Array1::from_vec(
                m.conv_layers
                    .bn3
                    .running_var
                    .value()
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}spatial_l1_weight", p),
            &Array2::from_shape_vec(
                (
                    m.spatial_features_mlp.l1.weight.shape().dims::<2>()[0],
                    m.spatial_features_mlp.l1.weight.shape().dims::<2>()[1],
                ),
                m.spatial_features_mlp
                    .l1
                    .weight
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}spatial_l1_bias", p),
            &Array1::from_vec(
                m.spatial_features_mlp
                    .l1
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing spatial_l1 bias"))?
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}spatial_l2_weight", p),
            &Array2::from_shape_vec(
                (
                    m.spatial_features_mlp.l2.weight.shape().dims::<2>()[0],
                    m.spatial_features_mlp.l2.weight.shape().dims::<2>()[1],
                ),
                m.spatial_features_mlp
                    .l2
                    .weight
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}spatial_l2_bias", p),
            &Array1::from_vec(
                m.spatial_features_mlp
                    .l2
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing spatial_l2 bias"))?
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}spatial_l3_weight", p),
            &Array2::from_shape_vec(
                (
                    m.spatial_features_mlp.l3.weight.shape().dims::<2>()[0],
                    m.spatial_features_mlp.l3.weight.shape().dims::<2>()[1],
                ),
                m.spatial_features_mlp
                    .l3
                    .weight
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}spatial_l3_bias", p),
            &Array1::from_vec(
                m.spatial_features_mlp
                    .l3
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing spatial_l3 bias"))?
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}head_l1_weight", p),
            &Array2::from_shape_vec(
                (
                    m.mlp.l1.weight.shape().dims::<2>()[0],
                    m.mlp.l1.weight.shape().dims::<2>()[1],
                ),
                m.mlp
                    .l1
                    .weight
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}head_l1_bias", p),
            &Array1::from_vec(
                m.mlp
                    .l1
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing head_l1 bias"))?
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
        npz.add_array(
            format!("{}head_l2_weight", p),
            &Array2::from_shape_vec(
                (
                    m.mlp.l2.weight.shape().dims::<2>()[0],
                    m.mlp.l2.weight.shape().dims::<2>()[1],
                ),
                m.mlp
                    .l2
                    .weight
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            )?,
        )?;
        npz.add_array(
            format!("{}head_l2_bias", p),
            &Array1::from_vec(
                m.mlp
                    .l2
                    .bias
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing head_l2 bias"))?
                    .to_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;

        npz.add_array(
            format!("{}anchors", p),
            &Array1::from_vec(
                m.anchors
                    .clone()
                    .into_data()
                    .as_slice::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?
                    .to_vec(),
            ),
        )?;
    }

    npz.finish()?;
    Ok(Some(out_str.to_string()))
}

const RECEIVED_LIGANDS_UNS_DF: &str = "spacetravlr_received_ligands";
const RECEIVED_LIGANDS_UNS_META: &str = "spacetravlr_received_ligands_meta";
const RECEIVED_LIGANDS_LEGACY_OBSM: &str = "spacetravlr_received_ligands";

fn strip_spacetravlr_received_ligand_keys(path: &Path) -> anyhow::Result<()> {
    let adata = AnnData::<H5>::open(H5::open_rw(path)?)?;
    let _ = adata.obsm().remove(RECEIVED_LIGANDS_LEGACY_OBSM);
    let _ = adata.uns().remove(RECEIVED_LIGANDS_UNS_DF);
    let _ = adata.uns().remove(RECEIVED_LIGANDS_UNS_META);
    adata.close()?;
    Ok(())
}

fn compute_received_ligands_uns_payload(
    adata: &AnnData<H5>,
    xy: &Array2<f64>,
    layer: &str,
    union_ligands_sorted: &[String],
    radius: f64,
    weighted_ligand_scale: f64,
    ligand_grid_factor: Option<f64>,
) -> anyhow::Result<(DataFrame, String)> {
    anyhow::ensure!(
        xy.nrows() == adata.n_obs(),
        "xy rows {} != adata n_obs {}",
        xy.nrows(),
        adata.n_obs()
    );
    let mut lig_unique: Vec<String> = union_ligands_sorted.to_vec();
    lig_unique.sort_unstable();
    lig_unique.dedup();

    let var_names = adata.var_names().into_vec();
    let gene_to_idx: HashMap<String, usize> = var_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();
    let lig_present: Vec<String> = lig_unique
        .into_iter()
        .filter(|g| gene_to_idx.contains_key(g.as_str()))
        .filter(|g| g.as_str() != "_index")
        .collect();
    if lig_present.is_empty() {
        return Ok((DataFrame::new(vec![])?, String::new()));
    }
    let slice = [SelectInfoElem::full(), SelectInfoElem::full()];
    let full = read_expression_matrix_dense_f64(adata, layer, &slice)?;
    let n = xy.nrows();
    let mut lig_expr = Array2::<f64>::zeros((n, lig_present.len()));
    for (k, lig) in lig_present.iter().enumerate() {
        let idx = gene_to_idx[lig];
        lig_expr.column_mut(k).assign(&full.column(idx));
    }
    let received = match ligand_grid_factor {
        Some(gf) if gf.is_finite() && gf > 0.0 => calculate_weighted_ligands_grid(
            xy,
            &lig_expr,
            radius,
            weighted_ligand_scale,
            gf,
        ),
        _ => calculate_weighted_ligands(xy, &lig_expr, radius, weighted_ligand_scale),
    };
    let obs_names: Vec<String> = adata.obs_names().into_iter().collect();
    anyhow::ensure!(
        obs_names.len() == n,
        "obs_names len {} != expression rows {}",
        obs_names.len(),
        n
    );
    let mut cols: Vec<_> = vec![Series::new("_index".into(), obs_names).into()];
    cols.extend(lig_present.iter().enumerate().map(|(k, name)| {
        Series::new(
            name.as_str().into(),
            received.column(k).to_vec(),
        )
        .into()
    }));
    let df = DataFrame::new(cols)?;
    let meta = serde_json::json!({
        "schema": "spacetravlr_received_ligands/v1",
        "radius": radius,
        "weighted_ligand_scale_factor": weighted_ligand_scale,
        "ligand_grid_factor": ligand_grid_factor,
        "layer": layer,
        "n_obs": n,
        "n_ligands": lig_present.len(),
        "ligands": lig_present,
    });
    let meta_str = serde_json::to_string(&meta)?;
    Ok((df, meta_str))
}

fn write_received_ligands_uns_payload(
    h5ad_path: &Path,
    df: DataFrame,
    meta_json: &str,
) -> anyhow::Result<()> {
    let adata_rw = AnnData::<H5>::open(H5::open_rw(h5ad_path)?)?;
    let _ = adata_rw.obsm().remove(RECEIVED_LIGANDS_LEGACY_OBSM);
    let _ = adata_rw.uns().remove(RECEIVED_LIGANDS_UNS_DF);
    let _ = adata_rw.uns().remove(RECEIVED_LIGANDS_UNS_META);
    adata_rw.uns().add(RECEIVED_LIGANDS_UNS_DF, df)?;
    adata_rw.uns().add(RECEIVED_LIGANDS_UNS_META, meta_json.to_string())?;
    adata_rw.close()?;
    Ok(())
}

/// Writes `uns['spacetravlr_received_ligands']` (+ `_meta`) on a processed `.h5ad` using
/// `[grn]`, `[spatial]`, `[data].layer`, and `[perturbation].ligand_grid_factor` from `cfg`
/// (same rules as training: [`GeneNetwork::dataset_union_lr_ligands`], Gaussian / grid aggregation).
///
/// Returns `Ok(true)` if a non-empty dataframe was written. `Ok(false)` when LR and TFL modulators
/// are both off, the expression layer name is empty, the GRN yields no union ligands, or the
/// filtered ligand set is empty after matching `var`.
pub fn cache_received_ligands_uns_for_processed_h5ad(
    processed_h5ad: &Path,
    cfg: &SpaceshipConfig,
) -> anyhow::Result<bool> {
    if !cfg.grn.use_lr_modulators && !cfg.grn.use_tfl_modulators {
        return Ok(false);
    }
    let layer = cfg.data.layer.trim();
    if layer.is_empty() {
        return Ok(false);
    }
    let network_dd = cfg
        .grn
        .network_data_dir
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(expand_user_path);

    let adata = AnnData::<H5>::open(H5::open(processed_h5ad)?)?;
    let all_var_names = adata.var_names().into_vec();
    let species = crate::network::infer_species(&all_var_names).with_context(|| {
        format!(
            "could not infer human vs mouse from var_names in {}; set [data].spatial_species or use explicit GRN paths",
            processed_h5ad.display()
        )
    })?;
    let grn = crate::network::GeneNetwork::new(
        species,
        &all_var_names,
        network_dd.as_deref(),
    )?;
    let max_lig = cfg.grn.max_ligands;
    let gene_mean = if max_lig.map_or(false, |k| k > 0) {
        Some(compute_gene_mean_expression(&adata, layer, None)?)
    } else {
        None
    };
    let mut union_ligs =
        grn.dataset_union_lr_ligands(&all_var_names, max_lig, gene_mean.as_ref())?;
    if union_ligs.is_empty() {
        adata.close()?;
        return Ok(false);
    }
    union_ligs.sort();
    let xy = load_spatial_coords_f64(&adata)?;
    let grid_eff = cfg.perturbation.ligand_grid_factor.or_else(|| {
        if xy.nrows() > LARGE_DATASET_GRID_AUTO_CELLS {
            Some(DEFAULT_LIGAND_GRID_FACTOR)
        } else {
            None
        }
    });
    let (df, meta) = compute_received_ligands_uns_payload(
        &adata,
        &xy,
        layer,
        &union_ligs,
        cfg.spatial.radius,
        cfg.spatial.weighted_ligand_scale_factor,
        grid_eff,
    )?;
    adata.close()?;
    if df.width() == 0 {
        return Ok(false);
    }
    write_received_ligands_uns_payload(processed_h5ad, df, &meta)?;
    Ok(true)
}

/// Parallel-safe per-gene mean Lasso R² (see [`patch_adata_var_mean_lasso_r2`]).
#[derive(Clone)]
pub struct MeanLassoR2Accum {
    pub gene_to_idx: Arc<HashMap<String, usize>>,
    pub scores: Arc<Vec<AtomicU64>>,
}

fn init_mean_lasso_r2_accum(all_var_names: &[String]) -> MeanLassoR2Accum {
    let mut gene_to_idx = HashMap::new();
    for (i, s) in all_var_names.iter().enumerate() {
        gene_to_idx.entry(s.clone()).or_insert(i);
    }
    let nan = f64::NAN.to_bits();
    let scores = Arc::new(
        (0..all_var_names.len())
            .map(|_| AtomicU64::new(nan))
            .collect::<Vec<_>>(),
    );
    MeanLassoR2Accum {
        gene_to_idx: Arc::new(gene_to_idx),
        scores,
    }
}

pub fn patch_adata_var_mean_lasso_r2(
    path: &Path,
    accum: &MeanLassoR2Accum,
) -> anyhow::Result<()> {
    let adata = AnnData::<H5>::open(H5::open_rw(path)?)?;
    let mut df = adata.read_var()?;
    let n = accum.scores.len();
    anyhow::ensure!(
        df.height() == n,
        "var rows {} != training n_vars {}",
        df.height(),
        n
    );
    let mut col: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let bits = accum.scores[i].load(Ordering::Acquire);
        col.push(f64::from_bits(bits));
    }
    let s = Series::new("mean_lasso_r2".into(), col);
    if df.column("mean_lasso_r2").is_ok() {
        df = df.drop("mean_lasso_r2")?;
    }
    df.with_column(s)?;
    adata.set_var(df)?;
    adata.close()?;
    Ok(())
}

/// Ensures the training `.h5ad` is the canonical `{output_dir}/{stem}_processed.h5ad` with CSR `X`/layers.
/// Drops any persisted `spacetravlr_received_ligands` / `_meta` keys so a stale on-disk cache is not reused;
/// [`SpatialCellularProgramsEstimator::fit_all_genes`] recomputes and writes them to `adata.uns` when training starts (full-obs runs only).
pub fn materialize_canonical_training_adata(
    adata_path: &mut String,
    output_dir: &Path,
    original_input_for_stem: &Path,
    _cfg: &SpaceshipConfig,
    _network_data_dir: Option<&str>,
) -> anyhow::Result<()> {
    let expanded = expand_user_path(adata_path.trim());
    let current = PathBuf::from(&expanded);
    if !current.is_file() {
        anyhow::bail!("AnnData not found at {}.", current.display());
    }
    if !current
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("h5ad"))
        .unwrap_or(false)
    {
        *adata_path = expanded;
        return Ok(());
    }
    let stem = crate::config::canonical_adata_stem(original_input_for_stem);
    let canonical = crate::scanpy_preprocess::training_processed_h5ad_path(output_dir, &stem);
    if current != canonical {
        let _ = std::fs::remove_file(&canonical);
        if crate::scanpy_preprocess::adata_x_and_layers_are_csr(&current)? {
            std::fs::copy(&current, &canonical)?;
        } else {
            crate::scanpy_preprocess::ensure_h5ad_csr_layers_on_path(&current, &canonical, false)?;
        }
        *adata_path = expand_user_path(canonical.to_string_lossy().as_ref());
    } else if !crate::scanpy_preprocess::adata_x_and_layers_are_csr(&current)? {
        crate::scanpy_preprocess::ensure_h5ad_csr_layers_on_path(&current, &current, false)?;
    }
    let path = PathBuf::from(expand_user_path(adata_path.trim()));
    strip_spacetravlr_received_ligand_keys(&path)?;
    Ok(())
}

pub struct SpatialCellularProgramsEstimator<AB: AutodiffBackend, AnB: Backend> {
    pub adata: Arc<AnnData<AnB>>,
    pub target_gene: String,
    pub spatial_dim: usize,
    pub cluster_annot: String,
    pub layer: String,
    pub radius: f64,
    pub contact_distance: f64,
    pub grn: Arc<crate::network::GeneNetwork>,
    pub tf_ligand_cutoff: f64,
    pub regulators: Vec<String>,
    pub ligands: Vec<String>,
    pub receptors: Vec<String>,
    pub tfl_ligands: Vec<String>,
    pub tfl_regulators: Vec<String>,
    pub lr_pairs: Vec<String>,
    pub tfl_pairs: Vec<String>,
    /// User-requested genes in the fourth Lasso group (raw expression), after filtering vs target/occupied.
    pub extra_modulators: Vec<String>,
    pub modulators_genes: Vec<String>,
    pub max_ligands: Option<usize>,
    pub regulator_masks_by_cluster: Option<HashMap<usize, Vec<bool>>>,
    pub seed_only: bool,
    pub estimator: Option<ClusteredGCNNWR<AB>>,
    pub group_reg_vec: Option<Vec<f64>>,
    pub ligand_grid_factor: Option<f64>,
    pub weighted_ligand_scale_factor: f64,
    /// When set, only these obs rows are read from the backing AnnData (read-only; no disk writes).
    pub obs_row_subset: Option<Arc<[usize]>>,
    pub modulator_scales: Option<Array1<f64>>,
}

impl<AB: AutodiffBackend, AnB: Backend> SpatialCellularProgramsEstimator<AB, AnB> {
    pub fn new_with_metadata(
        adata: Arc<AnnData<AnB>>,
        target_gene: String,
        radius: f64,
        spatial_dim: usize,
        contact_distance: f64,
        tf_ligand_cutoff: f64,
        max_ligands: Option<usize>,
        gene_mean_expression: Option<Arc<HashMap<String, f64>>>,
        use_tf_modulators: bool,
        use_lr_modulators: bool,
        use_tfl_modulators: bool,
        grn: Arc<crate::network::GeneNetwork>,
        tf_priors: Option<Arc<crate::network::TfPriors>>,
        cluster_to_cell_type: Option<Arc<HashMap<usize, String>>>,
        cluster_annot: String,
        layer: String,
        ligand_grid_factor: Option<f64>,
        weighted_ligand_scale_factor: f64,
        obs_row_subset: Option<Arc<[usize]>>,
        extra_lr_pairs: &[(String, String)],
        extra_modulator_candidates: &[String],
    ) -> anyhow::Result<Self> {
        let target_gene_str = target_gene.to_string();

        let var_set: HashSet<String> = adata.var_names().into_vec().into_iter().collect();

        let modulators = grn
            .get_modulators(
                &target_gene_str,
                tf_ligand_cutoff,
                max_ligands,
                gene_mean_expression.as_deref(),
            )?
            .apply_modulator_mask(use_tf_modulators, use_lr_modulators, use_tfl_modulators);

        let mut ligands = modulators.ligands.clone();
        let mut receptors = modulators.receptors.clone();
        let mut lr_pairs = modulators.lr_pairs.clone();
        if use_lr_modulators && !extra_lr_pairs.is_empty() {
            crate::grn_extra::merge_extra_lr_into(
                &mut ligands,
                &mut receptors,
                &mut lr_pairs,
                extra_lr_pairs,
                &target_gene_str,
                &var_set,
            );
        }

        let mut regulators = modulators.regulators.clone();
        let mut tfl_ligands = modulators.tfl_ligands.clone();
        let mut tfl_regulators = modulators.tfl_regulators.clone();
        let mut tfl_pairs = modulators.tfl_pairs.clone();
        let mut regulator_masks_by_cluster: Option<HashMap<usize, Vec<bool>>> = None;

        if use_tf_modulators {
            if let Some(priors) = tf_priors.as_ref() {
                if let Some(prior_union) = priors.tfs_for_target_any(&target_gene_str) {
                    if !prior_union.is_empty() {
                        regulators = prior_union.clone();
                        let allowed: HashSet<&str> =
                            regulators.iter().map(|s| s.as_str()).collect();
                        let mut filtered_tfl_l = Vec::new();
                        let mut filtered_tfl_r = Vec::new();
                        let mut filtered_tfl_p = Vec::new();
                        for ((lig, reg), pair) in tfl_ligands
                            .iter()
                            .zip(tfl_regulators.iter())
                            .zip(tfl_pairs.iter())
                        {
                            if allowed.contains(reg.as_str()) {
                                filtered_tfl_l.push(lig.clone());
                                filtered_tfl_r.push(reg.clone());
                                filtered_tfl_p.push(pair.clone());
                            }
                        }
                        tfl_ligands = filtered_tfl_l;
                        tfl_regulators = filtered_tfl_r;
                        tfl_pairs = filtered_tfl_p;

                        if let Some(cluster_map) = cluster_to_cell_type.as_ref() {
                            let mut masks = HashMap::new();
                            for (cluster_id, cell_type) in cluster_map.iter() {
                                if let Some(tf_ct) =
                                    priors.tfs_for_target_cell_type(&target_gene_str, cell_type)
                                {
                                    let allowed_ct: HashSet<&str> =
                                        tf_ct.iter().map(|s| s.as_str()).collect();
                                    let mask = regulators
                                        .iter()
                                        .map(|tf| allowed_ct.contains(tf.as_str()))
                                        .collect::<Vec<bool>>();
                                    masks.insert(*cluster_id, mask);
                                }
                            }
                            if !masks.is_empty() {
                                regulator_masks_by_cluster = Some(masks);
                            }
                        }
                    }
                }
            }
        }

        let mut occupied: HashSet<String> = HashSet::new();
        occupied.insert(target_gene_str.clone());
        for g in regulators
            .iter()
            .chain(ligands.iter())
            .chain(receptors.iter())
            .chain(tfl_ligands.iter())
            .chain(tfl_regulators.iter())
        {
            occupied.insert(g.clone());
        }
        let extra_modulators_accepted = crate::grn_extra::filter_extra_modulators(
            extra_modulator_candidates,
            &occupied,
            &var_set,
        );

        crate::grn_extra::verify_modulator_invariants(
            &target_gene_str,
            &regulators,
            &lr_pairs,
            &extra_modulators_accepted,
        )?;

        let mut modulators_genes_ordered = regulators.clone();
        modulators_genes_ordered.extend(lr_pairs.iter().cloned());
        modulators_genes_ordered.extend(tfl_pairs.iter().cloned());
        modulators_genes_ordered.extend(extra_modulators_accepted.iter().cloned());

        Ok(Self {
            adata,
            target_gene,
            spatial_dim,
            cluster_annot,
            layer,
            radius,
            contact_distance,
            grn,
            tf_ligand_cutoff,
            regulators,
            ligands,
            receptors,
            tfl_ligands,
            tfl_regulators,
            lr_pairs,
            tfl_pairs,
            extra_modulators: extra_modulators_accepted,
            modulators_genes: modulators_genes_ordered,
            max_ligands,
            regulator_masks_by_cluster,
            seed_only: false,
            estimator: None,
            group_reg_vec: None,
            ligand_grid_factor,
            weighted_ligand_scale_factor,
            obs_row_subset,
            modulator_scales: None,
        })
    }

    pub fn new(
        adata: Arc<AnnData<AnB>>,
        target_gene: String,
        radius: f64,
        spatial_dim: usize,
        contact_distance: f64,
        tf_ligand_cutoff: f64,
        max_ligands: Option<usize>,
    ) -> anyhow::Result<Self> {
        let adata_var_names = adata.var_names().into_vec();
        let species = crate::network::infer_species(&adata_var_names).ok_or_else(|| {
            anyhow::anyhow!(
                "could not infer human vs mouse from AnnData var_names (set explicit species if needed)"
            )
        })?;
        let grn = Arc::new(crate::network::GeneNetwork::new(
            species,
            &adata_var_names,
            None,
        )?);
        Self::new_with_metadata(
            adata,
            target_gene,
            radius,
            spatial_dim,
            contact_distance,
            tf_ligand_cutoff,
            max_ligands,
            None,
            true,
            true,
            true,
            grn,
            None,
            None,
            "cell_type".to_string(),
            "imputed_count".to_string(),
            None,
            1.0,
            None,
            &[],
            &[],
        )
    }
}

impl<AB: AutodiffBackend> SpatialCellularProgramsEstimator<AB, anndata_hdf5::H5> {
    #[allow(clippy::too_many_arguments)]
    pub fn fit_all_genes(
        adata_path: &str,
        obs_row_subset: Option<Arc<[usize]>>,
        radius: f64,
        spatial_dim: usize,
        contact_distance: f64,
        tf_ligand_cutoff: f64,
        max_ligands: Option<usize>,
        use_tf_modulators: bool,
        use_lr_modulators: bool,
        use_tfl_modulators: bool,
        layer: &str,
        cluster_annot: &str,
        cnn: &CnnConfig,
        epochs: usize,
        learning_rate: f64,
        score_threshold: f64,
        l1_reg: f64,
        group_reg: f64,
        n_iter: usize,
        tol: f64,
        cnn_training_mode: CnnTrainingMode,
        hybrid_pass2_full_cnn: bool,
        hybrid_gating: &HybridCnnGatingConfig,
        min_mean_lasso_r2_for_cnn: f64,
        gene_filter: Option<Vec<String>>,
        max_genes: Option<usize>,
        n_parallel: usize,
        output_dir: &str,
        model_export: &ModelExportConfig,
        hud: Option<TrainingHud>,
        network_data_dir: Option<&str>,
        tf_priors_feather: Option<&str>,
        write_minimal_repro_h5ad: bool,
        spaceship_config: &SpaceshipConfig,
        config_source_path: Option<PathBuf>,
        join_training: bool,
        mean_r2_accum_parent: Option<MeanLassoR2Accum>,
        device: &AB::Device,
    ) -> anyhow::Result<()>
    where
        AB: Send + 'static,
        AB::Device: Clone + Send + 'static,
    {
        let hud_for_done = hud.clone();
        let result = (|| -> anyhow::Result<()> {
            use anndata_hdf5::H5;
            use std::fs;
            use std::thread;

            let training_dir = output_dir;
            fs::create_dir_all(training_dir)?;
            fs::create_dir_all(format!("{training_dir}/log"))?;

            // Per-condition splits use `output_dir = .../conditions/<group>/`. The canonical
            // `spacetravlr_run_repro.toml` lives at the parent output root (written by the CLI before
            // splits). Do not write a second repro into each condition subfolder — that confuses
            // `--join-output-dir` and duplicates config.
            if obs_row_subset.is_none() {
                let repro_path = Path::new(training_dir).join(RUN_REPRO_TOML_FILENAME);
                match spaceship_config.write_run_repro_toml_if_missing(Path::new(training_dir)) {
                    Ok(Some(p)) => log_line(
                        &hud,
                        format!(
                            "Wrote {} (shared run config for additional trainers: use --join-output-dir on other hosts)",
                            p.display()
                        ),
                    ),
                    Ok(None) => {
                        if join_training {
                            log_line(
                                &hud,
                                format!(
                                    "Join mode: using existing run config {}",
                                    repro_path.display()
                                ),
                            );
                        }
                    }
                    Err(e) => log_line(
                        &hud,
                        format!("Could not write run repro TOML (early): {}", e),
                    ),
                }
            } else if join_training {
                log_line(
                    &hud,
                    "Join mode (condition split): canonical spacetravlr_run_repro.toml is under the parent output directory, not this conditions/<group>/ folder.".to_string(),
                );
            }

            // ── Setup: build gene list and pre-cache shared metadata ──────────────
            let t_open = pipeline_step_begin(&hud, "open AnnData (HDF5)");
            let setup_adata = Arc::new(AnnData::<H5>::open(H5::open(adata_path)?)?);
            pipeline_step_end(&hud, "open AnnData (HDF5)", t_open);
            if let Some(rows) = obs_row_subset.as_ref() {
                log_line(
                    &hud,
                    format!(
                        "Condition subset: {} / {} cells (read-only; .h5ad file untouched)",
                        rows.len(),
                        setup_adata.n_obs()
                    ),
                );
            }
            let all_var_names = setup_adata.var_names().into_vec();
            let mean_r2_accum = match mean_r2_accum_parent {
                Some(a) => a,
                None => init_mean_lasso_r2_accum(&all_var_names),
            };

            let mut target_genes =
                crate::config::filter_training_var_names(&all_var_names, gene_filter.as_deref());
            if let Some(ref filter) = gene_filter {
                log_line(&hud, format!("Filtering for specific genes: {:?}", filter));
                log_line(
                    &hud,
                    format!("Retained {} genes for training", target_genes.len()),
                );
            }
            if let Some(n) = max_genes {
                if target_genes.len() > n {
                    target_genes.truncate(n);
                    let preview: Vec<_> = target_genes.iter().take(5).cloned().collect();
                    log_line(
                        &hud,
                        format!("Using first {} genes (preview: {:?})", n, preview),
                    );
                }
            }

            let total_genes = target_genes.len();

            let t_val = pipeline_step_begin(
                &hud,
                "validate expression layer & read obs (full matrix check)",
            );
            let obs_df_full =
                validate_training_inputs(setup_adata.as_ref(), cluster_annot, layer, total_genes)?;
            pipeline_step_end(
                &hud,
                "validate expression layer & read obs (full matrix check)",
                t_val,
            );

            let (obs_names, obs_df) = if let Some(rows) = obs_row_subset.as_ref() {
                let full_names = setup_adata.obs_names().into_vec();
                let filtered_names: Vec<String> =
                    rows.iter().map(|&i| full_names[i].clone()).collect();
                use polars::prelude::NamedFrom;
                let idx_vec: Vec<u32> = rows.iter().map(|&i| i as u32).collect();
                let idx_ca = polars::prelude::IdxCa::new("".into(), &idx_vec);
                let filtered_df = obs_df_full.take(&idx_ca)?;
                (Arc::new(filtered_names), filtered_df)
            } else {
                (Arc::new(setup_adata.obs_names().into_vec()), obs_df_full)
            };

            let t_cl = pipeline_step_begin(&hud, "build cluster labels & cell-type map");
            let cell_type_counts = cell_type_label_counts_from_obs(&obs_df);
            let clusters: Arc<Array1<usize>> = Arc::new(clusters_array1_from_obs_column(
                &obs_df,
                cluster_annot,
            )?);
            let num_clusters = clusters
                .iter()
                .copied()
                .max()
                .map(|m| m.saturating_add(1))
                .unwrap_or(1);
            let cluster_to_cell_type =
                Arc::new(build_cluster_to_cell_type_map(&obs_df, cluster_annot)?);
            pipeline_step_end(&hud, "build cluster labels & cell-type map", t_cl);
            if tf_priors_feather.is_some() && cluster_to_cell_type.is_empty() {
                log_line(
                &hud,
                "TF priors provided, but no cell-type label column (cell_type / cell_types / celltype); using target-level TF priors without per-cell_type masking.".to_string(),
            );
            }

            if let Some(ref h) = hud {
                if let Ok(mut g) = h.lock() {
                    g.total_genes = total_genes;
                    g.n_cells = obs_names.len();
                    g.n_clusters = num_clusters;
                    g.cell_type_counts = cell_type_counts;
                }
            }

            let t_grn = pipeline_step_begin(&hud, "load GRN (network parquet / priors)");
            let species = crate::network::infer_species(&all_var_names).ok_or_else(|| {
                anyhow::anyhow!(
                    "could not infer human vs mouse from var_names; set [data].spatial_species or ensure var uses HGNC or MGI-style symbols"
                )
            })?;
            let global_grn = Arc::new(crate::network::GeneNetwork::new(
                species,
                &all_var_names,
                network_data_dir,
            )?);
            pipeline_step_end(&hud, "load GRN (network parquet / priors)", t_grn);

            let tf_priors = if let Some(path) = tf_priors_feather {
                let t_tf = pipeline_step_begin(&hud, "load TF priors (feather)");
                let loaded = crate::network::TfPriors::from_feather(path, &all_var_names)?;
                pipeline_step_end(&hud, "load TF priors (feather)", t_tf);
                log_line(&hud, format!("Loaded TF priors from {}", path));
                Some(Arc::new(loaded))
            } else if use_tf_modulators {
                let co_path = Path::new(training_dir).join("celloracle_tf_priors.feather");
                let co_path_str = co_path
                    .to_str()
                    .with_context(|| {
                        format!(
                            "celloracle TF priors path must be UTF-8: {}",
                            co_path.display()
                        )
                    })?;
                let t_co = pipeline_step_begin(&hud, "CellOracle GRN inference (auto TF priors)");
                if !co_path.is_file() {
                    let gem = read_h5ad_expression_dense_f64(Path::new(adata_path), layer)?;
                    let gem_scaled = crate::celloracle::scale_gem_no_center(&gem);
                    let tf_by_target = global_grn.grn_regulators_by_target()?;
                    let celloracle_terminal_progress = hud.is_none();
                    let links = if cluster_to_cell_type.is_empty() {
                        crate::celloracle::infer_grn_whole(
                            &gem,
                            &gem_scaled,
                            &all_var_names,
                            &tf_by_target,
                            celloracle_terminal_progress,
                        )?
                    } else {
                        let n = clusters.len();
                        let mut obs_cluster: Vec<String> = Vec::with_capacity(n);
                        for i in 0..n {
                            let cid = clusters[i];
                            let ct = cluster_to_cell_type
                                .get(&cid)
                                .cloned()
                                .unwrap_or_else(|| "unknown".to_string());
                            obs_cluster.push(ct);
                        }
                        crate::celloracle::infer_grn_per_cluster(
                            &gem,
                            &gem_scaled,
                            &all_var_names,
                            &tf_by_target,
                            &obs_cluster,
                            celloracle_terminal_progress,
                        )?
                    };
                    crate::celloracle::write_links_as_tf_priors_feather(&co_path, &links)?;
                    log_line(
                        &hud,
                        format!(
                            "Wrote CellOracle TF priors ({} edges) to {}",
                            links.len(),
                            co_path.display()
                        ),
                    );
                } else {
                    log_line(
                        &hud,
                        format!(
                            "Reusing cached CellOracle TF priors: {}",
                            co_path.display()
                        ),
                    );
                }
                let loaded = crate::network::TfPriors::from_feather(co_path_str, &all_var_names)?;
                pipeline_step_end(&hud, "CellOracle GRN inference (auto TF priors)", t_co);
                log_line(
                    &hud,
                    format!("Loaded TF priors from {}", co_path.display()),
                );
                Some(Arc::new(loaded))
            } else {
                None
            };

            let t_xy = pipeline_step_begin(&hud, "load spatial coordinates (obsm)");
            let xy_full = load_spatial_coords_f64(setup_adata.as_ref())?;
            let xy: Arc<Array2<f64>> = Arc::new(if let Some(rows) = obs_row_subset.as_ref() {
                xy_full.select(ndarray::Axis(0), rows)
            } else {
                xy_full
            });
            pipeline_step_end(&hud, "load spatial coordinates (obsm)", t_xy);

            if write_minimal_repro_h5ad && obs_row_subset.is_some() {
                anyhow::bail!(
                    "--condition splits are not compatible with write_minimal_repro_h5ad; \
                 set execution.write_minimal_repro_h5ad = false in your config."
                );
            }
            let repro_label = if write_minimal_repro_h5ad {
                "write spacetravlr_minimal_repro.h5ad"
            } else {
                "skip repro .h5ad (workers use input file)"
            };
            let t_repro = pipeline_step_begin(&hud, repro_label);
            let worker_adata_path: String = if write_minimal_repro_h5ad {
                if cluster_annot != "cell_type" {
                    anyhow::bail!(
                        "Minimal reproducibility export keeps only obs.cell_type; set [data].cluster_annot = \"cell_type\"."
                    );
                }
                let out = export_minimal_repro_adata_with_cache(
                    setup_adata.as_ref(),
                    training_dir,
                    layer,
                    cluster_annot,
                    xy.as_ref(),
                    clusters.as_ref(),
                    num_clusters,
                    spatial_dim,
                    cnn.spatial_feature_radius,
                )?;
                verify_minimal_repro_adata_loadable(&out, layer, cluster_annot)?;
                log_line(
                    &hud,
                    format!(
                        "Wrote minimal reproducibility AnnData with sparse X/layers and cached spatial tensors: {}",
                        out
                    ),
                );
                out
            } else {
                log_line(
                &hud,
                "Skipping spacetravlr_minimal_repro.h5ad (execution.write_minimal_repro_h5ad = false); using input AnnData for training.".to_string(),
            );
                adata_path.to_string()
            };
            pipeline_step_end(&hud, repro_label, t_repro);

            let obs_row_subset_for_workers = if write_minimal_repro_h5ad {
                None
            } else {
                obs_row_subset.clone()
            };

            let t_sp = pipeline_step_begin(&hud, "precompute shared spatial feature tensors");
            let cached_spatial = Arc::new(CachedSpatialData {
                spatial_features: crate::estimator::create_spatial_features(
                    xy.as_ref(),
                    clusters.as_ref(),
                    num_clusters,
                    cnn.spatial_feature_radius,
                ),
                spatial_maps: crate::estimator::xyc2spatial_fast(
                    xy.as_ref(),
                    clusters.as_ref(),
                    num_clusters,
                    spatial_dim,
                    spatial_dim,
                ),
            });
            pipeline_step_end(&hud, "precompute shared spatial feature tensors", t_sp);

            let compute_mean_for_hybrid = matches!(cnn_training_mode, CnnTrainingMode::Hybrid)
                && !hybrid_pass2_full_cnn
                && hybrid_gating.min_mean_target_expression_for_cnn.is_some();

            let gene_mean_arc: Option<Arc<HashMap<String, f64>>> =
                if max_ligands.map_or(false, |k| k > 0) || compute_mean_for_hybrid {
                    let t_m =
                        pipeline_step_begin(&hud, "per-gene mean expression (full matrix pass)");
                    let gm = compute_gene_mean_expression(
                        setup_adata.as_ref(),
                        layer,
                        obs_row_subset.as_deref(),
                    )?;
                    pipeline_step_end(&hud, "per-gene mean expression (full matrix pass)", t_m);
                    Some(Arc::new(gm))
                } else {
                    None
                };

            let cfg_parent = config_source_path.as_deref().and_then(Path::parent);
            let (resolved_ex_mod, resolved_ex_lr) = spaceship_config
                .grn
                .resolve_extra_modulators_and_lr(cfg_parent)?;
            if !resolved_ex_mod.is_empty() || !resolved_ex_lr.is_empty() {
                log_line(
                    &hud,
                    format!(
                        "GRN extras from config: {} extra modulator gene(s), {} extra L–R pair(s)",
                        resolved_ex_mod.len(),
                        resolved_ex_lr.len()
                    ),
                );
            }
            let extra_mod_arc = Arc::new(resolved_ex_mod);
            let extra_lr_arc = Arc::new(resolved_ex_lr);

            let neighbors: Arc<Vec<Vec<usize>>> =
                if matches!(cnn_training_mode, CnnTrainingMode::Hybrid) && !hybrid_pass2_full_cnn {
                    let n_cells = xy.nrows();
                    let k = hybrid_gating.moran_k_neighbors.max(1);
                    let mut msg = format!(
                        "Moran kNN graph ({} cells, k={}; KD-tree build + queries)",
                        n_cells, k
                    );
                    if n_cells > 8_000 {
                        msg.push_str(" — can take a while at very large n");
                    }
                    let t_nb = pipeline_step_begin(&hud, &msg);
                    let nb = build_neighbors(xy.as_ref(), k);
                    pipeline_step_end(&hud, &msg, t_nb);
                    Arc::new(nb)
                } else {
                    Arc::new(Vec::new())
                };

            let (force_genes, skip_genes) =
                if matches!(cnn_training_mode, CnnTrainingMode::Hybrid) && !hybrid_pass2_full_cnn {
                    let f = if let Some(ref p) = hybrid_gating.cnn_force_genes_file {
                        load_gene_set_file(Path::new(p))?
                    } else {
                        HashSet::new()
                    };
                    let s = if let Some(ref p) = hybrid_gating.cnn_skip_genes_file {
                        load_gene_set_file(Path::new(p))?
                    } else {
                        HashSet::new()
                    };
                    (Arc::new(f), Arc::new(s))
                } else {
                    (Arc::new(HashSet::new()), Arc::new(HashSet::new()))
                };

            let hybrid_collect_top_k = matches!(cnn_training_mode, CnnTrainingMode::Hybrid)
                && !hybrid_pass2_full_cnn
                && hybrid_gating.hybrid_cnn_top_k.is_some();
            let cnn_candidates: Arc<Mutex<Vec<(String, f64, CnnGateDecision)>>> =
                Arc::new(Mutex::new(Vec::new()));

            let layer_for_workers = layer.to_string();
            let cnn_for_workers = cnn.clone();
            let ligand_grid_factor = spaceship_config.perturbation.ligand_grid_factor;
            let weighted_ligand_scale_factor =
                spaceship_config.spatial.weighted_ligand_scale_factor;

            if obs_row_subset.is_none() && (use_lr_modulators || use_tfl_modulators) {
                let t_rl = pipeline_step_begin(&hud, "cache received ligands → adata.uns");
                let grid_eff = ligand_grid_factor.or_else(|| {
                    if xy.nrows() > LARGE_DATASET_GRID_AUTO_CELLS {
                        Some(DEFAULT_LIGAND_GRID_FACTOR)
                    } else {
                        None
                    }
                });
                let received_payload: anyhow::Result<Option<(DataFrame, String)>> = (|| {
                    let mut union_ligs = global_grn.dataset_union_lr_ligands(
                        &all_var_names,
                        max_ligands,
                        gene_mean_arc.as_deref(),
                    )?;
                    if union_ligs.is_empty() {
                        return Ok(None);
                    }
                    union_ligs.sort();
                    let (df, meta) = compute_received_ligands_uns_payload(
                        setup_adata.as_ref(),
                        xy.as_ref(),
                        layer,
                        &union_ligs,
                        radius,
                        weighted_ligand_scale_factor,
                        grid_eff,
                    )?;
                    if df.width() == 0 {
                        return Ok(None);
                    }
                    Ok(Some((df, meta)))
                })();
                drop(setup_adata);
                match received_payload {
                    Ok(Some((df, meta))) => {
                        let rw_path = PathBuf::from(&worker_adata_path);
                        let n_l = df.width();
                        let n_c = df.height();
                        match write_received_ligands_uns_payload(&rw_path, df, &meta) {
                            Ok(()) => log_line(
                                &hud,
                                format!(
                                    "Wrote uns['{}'] ({} ligands × {} cells) → {}",
                                    RECEIVED_LIGANDS_UNS_DF,
                                    n_l,
                                    n_c,
                                    rw_path.display()
                                ),
                            ),
                            Err(e) => log_line(
                                &hud,
                                format!("Could not write {}: {}", RECEIVED_LIGANDS_UNS_DF, e),
                            ),
                        }
                    }
                    Ok(None) => {}
                    Err(e) => log_line(
                        &hud,
                        format!("Could not build {}: {}", RECEIVED_LIGANDS_UNS_DF, e),
                    ),
                }
                pipeline_step_end(&hud, "cache received ligands → adata.uns", t_rl);
            } else {
                if obs_row_subset.is_some() && (use_lr_modulators || use_tfl_modulators) {
                    log_line(
                        &hud,
                        "Skipping spacetravlr_received_ligands cache (condition split: obs subset vs full .h5ad)."
                            .to_string(),
                    );
                }
                drop(setup_adata);
            }

            let pb_opt: Option<ProgressBar> = if hud.is_none() {
                let pb = ProgressBar::new(total_genes as u64);
                pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
                    .progress_chars("#>-"),
            );
                Some(pb)
            } else {
                None
            };

            // ── Shared work queue ─────────────────────────────────────────────────
            let work: Arc<Mutex<VecDeque<String>>> =
                Arc::new(Mutex::new(target_genes.into_iter().collect()));

            let n_workers = n_parallel.max(1);
            let mut handles: Vec<thread::JoinHandle<()>> = Vec::with_capacity(n_workers);
            let stale_lock_secs = spaceship_config.execution.stale_lock_secs;
            let scale_modulators_w = spaceship_config.lasso.scale_modulators;
            let unscale_betas_on_export_w = spaceship_config.lasso.unscale_betas_on_export;

            let janitor_stop = Arc::new(AtomicBool::new(false));
            let janitor_handle = if stale_lock_secs > 0 {
                let training_dir_j = training_dir.to_string();
                let hud_j = hud.clone();
                let stop = janitor_stop.clone();
                let secs = stale_lock_secs;
                Some(thread::spawn(move || {
                    const TICK: Duration = Duration::from_secs(1);
                    const SWEEP_INTERVAL_SECS: u64 = 600;
                    let mut acc = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        thread::sleep(TICK);
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        if let Some(ref hh) = hud_j {
                            if hh.lock().map(|g| g.should_cancel()).unwrap_or(true) {
                                break;
                            }
                        }
                        acc += 1;
                        if acc < SWEEP_INTERVAL_SECS {
                            continue;
                        }
                        acc = 0;
                        let n = remove_stale_lock_files_in_dir(Path::new(&training_dir_j), secs);
                        if n > 0 {
                            log_line(
                                &hud_j,
                                format!(">> periodic sweep: removed {n} stale .lock file(s)"),
                            );
                        }
                    }
                }))
            } else {
                None
            };

            log_line(
                &hud,
                format!(
                    "Spawning {} worker threads for {} genes (each opens HDF5 once)…",
                    n_workers, total_genes
                ),
            );
            let t_workers = pipeline_step_begin(&hud, "per-gene training (workers running)");
            let cluster_annot_for_workers = cluster_annot.to_string();
            for _worker in 0..n_workers {
                let work = work.clone();
                let xy = xy.clone();
                let clusters = clusters.clone();
                let obs_names = obs_names.clone();
                let global_grn = global_grn.clone();
                let tf_priors = tf_priors.clone();
                let cluster_to_cell_type = cluster_to_cell_type.clone();
                let cluster_annot_w = cluster_annot_for_workers.clone();
                let hud = hud.clone();
                let pb = pb_opt.clone();
                let device = device.clone();
                let adata_path = worker_adata_path.clone();
                let training_dir = training_dir.to_string();
                let cached_spatial = cached_spatial.clone();
                let obs_subset = obs_row_subset_for_workers.clone();

                let gene_mean_arc = gene_mean_arc.clone();
                let extra_mod_arc_w = extra_mod_arc.clone();
                let extra_lr_arc_w = extra_lr_arc.clone();
                let layer_w = layer_for_workers.clone();
                let cnn_w = cnn_for_workers.clone();
                let cnn_mode_w = cnn_training_mode;
                let hybrid_pass2 = hybrid_pass2_full_cnn;
                let hybrid_cfg = hybrid_gating.clone();
                let min_mean_r2 = min_mean_lasso_r2_for_cnn;
                let neighbors_w = neighbors.clone();
                let force_w = force_genes.clone();
                let skip_w = skip_genes.clone();
                let candidates_w = cnn_candidates.clone();
                let model_export_w = model_export.clone();
                let collect_top_k = hybrid_collect_top_k;
                let mean_r2_accum_w = mean_r2_accum.clone();

                let handle = thread::Builder::new()
                    .stack_size(8 * 1024 * 1024)
                    .spawn(move || {
                        let thread_adata = match H5::open(&adata_path).and_then(AnnData::<H5>::open)
                        {
                            Ok(a) => Arc::new(a),
                            Err(e) => {
                                log_line(
                                    &hud,
                                    format!("ERROR: worker failed to open adata: {}", e),
                                );
                                return;
                            }
                        };

                        let n_samples = xy.nrows();
                        let n_lasso_total = (0..num_clusters)
                            .filter(|&c_id| (0..n_samples).any(|i| clusters[i] == c_id))
                            .count();

                        loop {
                            // Cancel check
                            if hud
                                .as_ref()
                                .and_then(|h| h.lock().ok())
                                .map(|g| g.should_cancel())
                                .unwrap_or(false)
                            {
                                log_line(&hud, ">> worker: cancel signal received".to_string());
                                break;
                            }

                            let gene = match work.lock() {
                                Ok(mut q) => q.pop_front(),
                                Err(_) => break,
                            };
                            let Some(gene) = gene else { break };

                            let feather_path =
                                format!("{}/{}_betadata.feather", training_dir, gene);
                            let orphan_path = format!("{}/{}.orphan", training_dir, gene);
                            let lock_path = format!("{}/{}.lock", training_dir, gene);

                            // Skip already-done
                            if std::path::Path::new(&feather_path).exists()
                                || std::path::Path::new(&orphan_path).exists()
                            {
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_skipped += 1;
                                    }
                                    log_line(&hud, format!(">> skip (cached) {}", gene));
                                }
                                if let Some(ref p) = pb {
                                    p.inc(1);
                                }
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_rounds += 1;
                                    }
                                }
                                continue;
                            }

                            if stale_lock_secs > 0 {
                                let lp = Path::new(&lock_path);
                                if lp.is_file() {
                                    if let Ok(meta) = fs::metadata(lp) {
                                        if let Ok(modified) = meta.modified() {
                                            if let Ok(age) =
                                                SystemTime::now().duration_since(modified)
                                            {
                                                if age.as_secs() >= stale_lock_secs {
                                                    let _ = fs::remove_file(lp);
                                                    log_line(
                                                        &hud,
                                                        format!(
                                                            ">> removed stale lock {} ({:.0}s old)",
                                                            gene,
                                                            age.as_secs_f64()
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Try to claim this gene via a lock file (exclusive with other hosts/processes on shared storage)
                            if fs::OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(&lock_path)
                                .is_err()
                            {
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_skipped += 1;
                                    }
                                    log_line(&hud, format!(">> skip (lock) {}", gene));
                                }
                                if let Some(ref p) = pb {
                                    p.inc(1);
                                }
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_rounds += 1;
                                    }
                                }
                                continue;
                            }
                            struct LockGuard(String);
                            impl Drop for LockGuard {
                                fn drop(&mut self) {
                                    let _ = fs::remove_file(&self.0);
                                }
                            }
                            let _guard = LockGuard(lock_path);
                            let gene_start = std::time::Instant::now();

                            // Register as active
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.set_gene_status(&gene, "estimator | ? mods");
                                    if n_lasso_total > 0 {
                                        g.set_gene_lasso_cluster_progress(&gene, 0, n_lasso_total);
                                    }
                                }
                            }

                            let mut estimator = match Self::new_with_metadata(
                                thread_adata.clone(),
                                gene.clone(),
                                radius,
                                spatial_dim,
                                contact_distance,
                                tf_ligand_cutoff,
                                max_ligands,
                                gene_mean_arc.clone(),
                                use_tf_modulators,
                                use_lr_modulators,
                                use_tfl_modulators,
                                global_grn.clone(),
                                tf_priors.clone(),
                                Some(cluster_to_cell_type.clone()),
                                cluster_annot_w.clone(),
                                layer_w.clone(),
                                ligand_grid_factor,
                                weighted_ligand_scale_factor,
                                obs_subset.clone(),
                                extra_lr_arc_w.as_slice(),
                                extra_mod_arc_w.as_slice(),
                            )
                            .map(Box::new)
                            {
                                Ok(est) => est,
                                Err(e) => {
                                    log_line(
                                        &hud,
                                        format!("❌ estimator init failed {}: {}", gene, e),
                                    );
                                    if let Some(ref h) = hud {
                                        if let Ok(mut g) = h.lock() {
                                            g.genes_failed += 1;
                                            g.remove_gene(&gene);
                                        }
                                    }
                                    if let Some(ref p) = pb {
                                        p.inc(1);
                                    }
                                    if let Some(ref h) = hud {
                                        if let Ok(mut g) = h.lock() {
                                            g.genes_rounds += 1;
                                        }
                                    }
                                    continue;
                                }
                            };

                            let n_mods = estimator.modulators_genes.len();
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.set_gene_status(
                                        &gene,
                                        format!("estimator | {} mods", n_mods),
                                    );
                                }
                            }

                            if n_mods == 0 {
                                let _ =
                                    fs::File::create(format!("{}/{}.orphan", training_dir, gene));
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_orphan += 1;
                                        g.remove_gene(&gene);
                                        g.genes_rounds += 1;
                                    }
                                }
                                log_line(&hud, format!(">> orphan (no modulators) {}", gene));
                                if let Some(ref p) = pb {
                                    p.inc(1);
                                }
                                continue;
                            }

                            let worker_run_full_cnn =
                                hybrid_pass2 || matches!(cnn_mode_w, CnnTrainingMode::Full);
                            estimator.seed_only = !worker_run_full_cnn;
                            if matches!(cnn_mode_w, CnnTrainingMode::Hybrid) && !hybrid_pass2 {
                                estimator.seed_only = true;
                            }
                            let phase_str =
                                if matches!(cnn_mode_w, CnnTrainingMode::Hybrid) && !hybrid_pass2 {
                                    format!("hybrid gate | {} mods", n_mods)
                                } else if worker_run_full_cnn {
                                    format!("lasso+cnn | {} mods", n_mods)
                                } else {
                                    format!("lasso | {} mods", n_mods)
                                };
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.set_gene_status(&gene, &phase_str);
                                }
                            }

                            let mut on_lasso_progress = |done: usize, total: usize| {
                                if let Some(hh) = hud.as_ref() {
                                    if let Ok(mut g) = hh.lock() {
                                        g.set_gene_lasso_cluster_progress(&gene, done, total);
                                    }
                                }
                            };
                            let fit_ok = estimator
                                .fit_with_cache(
                                    &xy,
                                    &clusters,
                                    num_clusters,
                                    epochs,
                                    learning_rate,
                                    score_threshold,
                                    l1_reg,
                                    group_reg,
                                    n_iter,
                                    tol,
                                    scale_modulators_w,
                                    "lasso",
                                    &cnn_w,
                                    &device,
                                    Some(cached_spatial.as_ref()),
                                    &mut on_lasso_progress,
                                )
                                .is_ok();
                            if !fit_ok {
                                if let Some(hh) = hud.as_ref() {
                                    if let Ok(mut g) = hh.lock() {
                                        g.clear_gene_lasso_cluster_progress(&gene);
                                    }
                                }
                            }

                            let mut export_per_cell =
                                matches!(cnn_mode_w, CnnTrainingMode::Full) || hybrid_pass2;
                            let mut gate_record: Option<CnnGateDecision> = None;

                            if fit_ok {
                                let hybrid_gate = matches!(cnn_mode_w, CnnTrainingMode::Hybrid)
                                    && !hybrid_pass2
                                    && n_mods > 0;
                                if hybrid_gate {
                                    match estimator.build_x_modulators_and_target_y(&xy) {
                                        Ok((mut x_mat, y_vec)) => {
                                            if scale_modulators_w {
                                                if let Some(ref scales) =
                                                    estimator.modulator_scales
                                                {
                                                    apply_modulator_scales_inplace(
                                                        &mut x_mat, scales,
                                                    );
                                                }
                                            }
                                            let decision_opt =
                                                estimator.estimator.as_ref().map(|inn| {
                                                    let summaries_snapshot =
                                                        inn.cluster_training_summaries.clone();
                                                    let yhat = predict_lasso_y(
                                                        &inn.lasso_coefficients,
                                                        &inn.lasso_intercepts,
                                                        &x_mat,
                                                        &clusters,
                                                    );
                                                    let residuals: Vec<f64> = y_vec
                                                        .iter()
                                                        .zip(yhat.iter())
                                                        .map(|(a, b)| a - b)
                                                        .collect();
                                                    let mean_tgt = gene_mean_arc
                                                        .as_ref()
                                                        .and_then(|m| m.get(&gene).copied());
                                                    decide_cnn_for_gene(
                                                        &hybrid_cfg,
                                                        min_mean_r2,
                                                        &gene,
                                                        &summaries_snapshot,
                                                        estimator.regulators.len(),
                                                        estimator.lr_pairs.len(),
                                                        estimator.tfl_pairs.len(),
                                                        &residuals,
                                                        neighbors_w.as_ref(),
                                                        &force_w,
                                                        &skip_w,
                                                        mean_tgt,
                                                    )
                                                });
                                            if let Some(decision) = decision_opt {
                                                gate_record = Some(decision.clone());
                                                if collect_top_k {
                                                    export_per_cell = false;
                                                    if decision.use_cnn {
                                                        if let Ok(mut c) = candidates_w.lock() {
                                                            c.push((
                                                                gene.clone(),
                                                                decision.rank_score,
                                                                decision,
                                                            ));
                                                        }
                                                    }
                                                } else if decision.use_cnn {
                                                    if let Some(inn) = estimator.estimator.as_mut()
                                                    {
                                                        inn.fit_cnn_refinement(
                                                            &x_mat,
                                                            &y_vec,
                                                            &xy,
                                                            &clusters,
                                                            num_clusters,
                                                            &device,
                                                            epochs,
                                                            learning_rate,
                                                            &cnn_w,
                                                            Some(cached_spatial.as_ref()),
                                                        );
                                                    }
                                                    export_per_cell = true;
                                                } else {
                                                    export_per_cell = false;
                                                }
                                            } else {
                                                export_per_cell = false;
                                            }
                                        }
                                        Err(e) => {
                                            log_line(
                                                &hud,
                                                format!(
                                                    "hybrid design matrix failed {}: {}",
                                                    gene, e
                                                ),
                                            );
                                            export_per_cell = false;
                                        }
                                    }
                                }
                            }

                            let mut wrote = false;
                            let mut orphan_zero_mod_betas = false;
                            let mut bad_r2_clusters: HashSet<usize> = HashSet::new();
                            let mut n_betadata_beta_columns: Option<usize> = None;
                            if fit_ok {
                                if let Some(est_inner) = estimator.estimator.as_mut() {
                                    for s in &mut est_inner.cluster_training_summaries {
                                        if !s.lasso_r2.is_finite() || s.lasso_r2 < score_threshold {
                                            bad_r2_clusters.insert(s.cluster_id);
                                            s.lasso_r2 = 0.0;
                                        }
                                    }
                                    for &cid in &bad_r2_clusters {
                                        est_inner.r2_scores.insert(cid, 0.0);
                                        est_inner.lasso_intercepts.insert(cid, 0.0);
                                        if let Some(coef) =
                                            est_inner.lasso_coefficients.get_mut(&cid)
                                        {
                                            coef.fill(0.0);
                                        }
                                    }
                                    if let Some(ref h) = hud {
                                        if let Ok(mut g) = h.lock() {
                                            g.clear_gene_lasso_cluster_progress(&gene);
                                            g.set_gene_status(
                                                &gene,
                                                format!("export | {} mods", n_mods),
                                            );
                                        }
                                    }
                                    let betadata_path =
                                        format!("{}/{}_betadata.feather", training_dir, gene);
                                    let col_names: Vec<String> =
                                        std::iter::once("beta0".to_string())
                                            .chain(
                                                estimator
                                                    .modulators_genes
                                                    .iter()
                                                    .map(|m| format!("beta_{}", m)),
                                            )
                                            .collect();

                                    if export_per_cell {
                                        let x_mock = Array2::<f64>::zeros((xy.nrows(), n_mods));
                                        let mut all_betas = est_inner.predict_betas(
                                            &x_mock,
                                            &xy,
                                            &clusters,
                                            num_clusters,
                                            &device,
                                            Some(cached_spatial.as_ref()),
                                        );
                                        if !bad_r2_clusters.is_empty() {
                                            for i in 0..all_betas.nrows() {
                                                if bad_r2_clusters.contains(&clusters[i]) {
                                                    all_betas.row_mut(i).fill(0.0);
                                                }
                                            }
                                        }
                                        if unscale_betas_on_export_w {
                                            if let Some(ref scales) = estimator.modulator_scales {
                                                unscale_betadata_columns_inplace(
                                                    &mut all_betas,
                                                    scales,
                                                );
                                            }
                                        }

                                        let keep: Vec<usize> = (0..all_betas.ncols())
                                            .filter(|&j| {
                                                all_betas
                                                    .column(j)
                                                    .iter()
                                                    .any(|&v| finite_or_zero_f64(v) != 0.0)
                                            })
                                            .collect();
                                        n_betadata_beta_columns =
                                            Some(keep.iter().filter(|&&j| j >= 1).count());

                                        if !keep.iter().any(|&j| j >= 1) {
                                            let _ = fs::File::create(format!(
                                                "{}/{}.orphan",
                                                training_dir, gene
                                            ));
                                            orphan_zero_mod_betas = true;
                                            if let Some(ref h) = hud {
                                                if let Ok(mut g) = h.lock() {
                                                    g.genes_orphan += 1;
                                                }
                                            }
                                            log_line(
                                                &hud,
                                                format!(
                                                    ">> orphan (no non-zero modulator betas) {}",
                                                    gene
                                                ),
                                            );
                                        } else {
                                            let n_rows = obs_names.len();
                                            let n_keep = keep.len();
                                            let mut mat = Array2::<f64>::zeros((n_rows, n_keep));
                                            for (new_j, &j) in keep.iter().enumerate() {
                                                for i in 0..n_rows {
                                                    mat[[i, new_j]] =
                                                        finite_or_zero_f64(all_betas[[i, j]]);
                                                }
                                            }
                                            let data_cols: Vec<String> = keep
                                                .iter()
                                                .map(|&j| col_names[j].clone())
                                                .collect();
                                            if write_betadata_feather(
                                                &betadata_path,
                                                "CellID",
                                                obs_names.as_ref(),
                                                &data_cols,
                                                &mat,
                                            )
                                            .is_ok()
                                            {
                                                wrote = true;
                                            }
                                        }
                                    } else {
                                        let mut cluster_ids: Vec<usize> =
                                            est_inner.lasso_coefficients.keys().copied().collect();
                                        cluster_ids.sort();

                                        let rows: Vec<Vec<f64>> = cluster_ids
                                            .iter()
                                            .map(|&c_id| {
                                                if bad_r2_clusters.contains(&c_id) {
                                                    return vec![0.0; 1 + n_mods];
                                                }
                                                let intercept = finite_or_zero_f64(
                                                    est_inner
                                                        .lasso_intercepts
                                                        .get(&c_id)
                                                        .copied()
                                                        .unwrap_or(0.0),
                                                );
                                                let coefs = &est_inner.lasso_coefficients[&c_id];
                                                let mut row = Vec::with_capacity(1 + n_mods);
                                                row.push(intercept);
                                                for j in 0..coefs.nrows() {
                                                    let mut b =
                                                        finite_or_zero_f64(coefs[[j, 0]]);
                                                    if unscale_betas_on_export_w {
                                                        if let Some(ref s) =
                                                            estimator.modulator_scales
                                                        {
                                                            if j < s.len() {
                                                                let sj = s[j];
                                                                if sj != 1.0 {
                                                                    b /= sj;
                                                                }
                                                            }
                                                        }
                                                    }
                                                    row.push(b);
                                                }
                                                row
                                            })
                                            .collect();

                                        let n_cols = 1 + n_mods;
                                        let keep: Vec<usize> = (0..n_cols)
                                            .filter(|&j| rows.iter().any(|r| r[j] != 0.0))
                                            .collect();
                                        n_betadata_beta_columns =
                                            Some(keep.iter().filter(|&&j| j >= 1).count());

                                        if !keep.iter().any(|&j| j >= 1) {
                                            let _ = fs::File::create(format!(
                                                "{}/{}.orphan",
                                                training_dir, gene
                                            ));
                                            orphan_zero_mod_betas = true;
                                            if let Some(ref h) = hud {
                                                if let Ok(mut g) = h.lock() {
                                                    g.genes_orphan += 1;
                                                }
                                            }
                                            log_line(
                                                &hud,
                                                format!(
                                                    ">> orphan (no non-zero modulator betas) {}",
                                                    gene
                                                ),
                                            );
                                        } else {
                                            let n_rows = rows.len();
                                            let n_keep = keep.len();
                                            let mut mat = Array2::<f64>::zeros((n_rows, n_keep));
                                            for (i, row_vals) in rows.iter().enumerate() {
                                                for (new_j, &j) in keep.iter().enumerate() {
                                                    mat[[i, new_j]] = row_vals[j];
                                                }
                                            }
                                            let ids: Vec<String> =
                                                cluster_ids.iter().map(|c| c.to_string()).collect();
                                            let data_cols: Vec<String> = keep
                                                .iter()
                                                .map(|&j| col_names[j].clone())
                                                .collect();
                                            if write_betadata_feather(
                                                &betadata_path,
                                                "Cluster",
                                                &ids,
                                                &data_cols,
                                                &mat,
                                            )
                                            .is_ok()
                                            {
                                                wrote = true;
                                            }
                                        }
                                    }
                                }
                            }

                            if wrote {
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_done += 1;
                                    }
                                    log_line(&hud, format!(">> wrote {}", gene));
                                }
                            } else if !orphan_zero_mod_betas {
                                if let Some(ref h) = hud {
                                    if let Ok(mut g) = h.lock() {
                                        g.genes_failed += 1;
                                    }
                                    log_line(&hud, format!(">> fail (fit/export) {}", gene));
                                }
                            }

                            if n_mods > 0 {
                                if let Some(est) = estimator.estimator.as_ref() {
                                    if wrote && export_per_cell && model_export_w.save_cnn_weights {
                                        match export_cnn_models_npz(
                                            est,
                                            &gene,
                                            &training_dir,
                                            &model_export_w,
                                            Some(&bad_r2_clusters),
                                        ) {
                                            Ok(Some(path)) => {
                                                log_line(
                                                    &hud,
                                                    format!(">> wrote cnn model {}", path),
                                                );
                                            }
                                            Ok(None) => {}
                                            Err(e) => {
                                                log_line(
                                                    &hud,
                                                    format!(
                                                        ">> warn (cnn model export) {}: {}",
                                                        gene, e
                                                    ),
                                                );
                                            }
                                        }
                                    }
                                    let safe_gene = gene.replace(['/', '\\'], "_");
                                    let log_path =
                                        format!("{}/log/{}.log", training_dir, safe_gene);
                                    let _ = crate::training_log::write_gene_training_log(
                                        std::path::Path::new(&log_path),
                                        &gene,
                                        !export_per_cell,
                                        export_per_cell,
                                        epochs,
                                        learning_rate,
                                        n_iter,
                                        tol,
                                        &est.cluster_training_summaries,
                                        gate_record.as_ref(),
                                    );
                                    if let Some(ref h) = hud {
                                        if let Ok(mut g) = h.lock() {
                                            if wrote {
                                                g.record_gene_export_mode(export_per_cell);
                                            }
                                            if !est.cluster_training_summaries.is_empty() {
                                                g.record_training_metrics(
                                                    &gene,
                                                    &est.cluster_training_summaries,
                                                    n_betadata_beta_columns,
                                                );
                                            }
                                        }
                                    }
                                    if wrote && !est.cluster_training_summaries.is_empty() {
                                        if let Some(&idx) = mean_r2_accum_w.gene_to_idx.get(&gene) {
                                            let mean_r2: f64 = est
                                                .cluster_training_summaries
                                                .iter()
                                                .map(|s| s.lasso_r2)
                                                .sum::<f64>()
                                                / est.cluster_training_summaries.len() as f64;
                                            mean_r2_accum_w.scores[idx].store(
                                                mean_r2.to_bits(),
                                                Ordering::Release,
                                            );
                                        }
                                    }
                                }
                            }

                            // Deregister from active, bump counter
                            if let Some(ref h) = hud {
                                if let Ok(mut g) = h.lock() {
                                    g.record_gene_time(&gene, gene_start.elapsed().as_secs_f64());
                                    g.remove_gene(&gene);
                                    g.genes_rounds += 1;
                                }
                            }
                            if let Some(ref p) = pb {
                                p.inc(1);
                            }
                        }
                    })
                    .expect("failed to spawn worker thread");

                handles.push(handle);
            }

            for h in handles {
                let _ = h.join();
            }
            janitor_stop.store(true, Ordering::Relaxed);
            if let Some(h) = janitor_handle {
                let _ = h.join();
            }
            pipeline_step_end(&hud, "per-gene training (workers running)", t_workers);

            if matches!(cnn_training_mode, CnnTrainingMode::Hybrid) && !hybrid_pass2_full_cnn {
                if let Some(k) = hybrid_gating.hybrid_cnn_top_k {
                    let mut cand = cnn_candidates
                        .lock()
                        .map_err(|e| anyhow::anyhow!("hybrid candidate lock poisoned: {}", e))?;
                    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let picked: Vec<String> = cand
                        .iter()
                        .filter(|t| t.2.use_cnn)
                        .take(k)
                        .map(|t| t.0.clone())
                        .collect();
                    drop(cand);
                    if !picked.is_empty() {
                        log_line(
                            &hud,
                            format!(
                                "hybrid phase 2: re-training {} genes with full CNN (top-K)",
                                picked.len()
                            ),
                        );
                        if let Some(ref h) = hud {
                            if let Ok(mut g) = h.lock() {
                                let k = picked.len();
                                g.total_genes = g.total_genes.saturating_add(k);
                                g.genes_done = g.genes_done.saturating_sub(k);
                                g.genes_rounds = g.genes_rounds.saturating_sub(k);
                                g.genes_exported_seed_only =
                                    g.genes_exported_seed_only.saturating_sub(k);
                            }
                        }
                        for g in &picked {
                            let pth = format!("{training_dir}/{g}_betadata.feather");
                            let _ = fs::remove_file(&pth);
                        }
                        return Self::fit_all_genes(
                            &worker_adata_path,
                            obs_row_subset.clone(),
                            radius,
                            spatial_dim,
                            contact_distance,
                            tf_ligand_cutoff,
                            max_ligands,
                            use_tf_modulators,
                            use_lr_modulators,
                            use_tfl_modulators,
                            layer,
                            cluster_annot,
                            cnn,
                            epochs,
                            learning_rate,
                            score_threshold,
                            l1_reg,
                            group_reg,
                            n_iter,
                            tol,
                            cnn_training_mode,
                            true,
                            hybrid_gating,
                            min_mean_lasso_r2_for_cnn,
                            Some(picked),
                            None,
                            n_parallel,
                            output_dir,
                            model_export,
                            hud,
                            network_data_dir,
                            tf_priors_feather,
                            false,
                            spaceship_config,
                            config_source_path.clone(),
                            join_training,
                            Some(mean_r2_accum.clone()),
                            device,
                        );
                    }
                }
            }

            if !join_training && obs_row_subset.is_none() {
                match patch_adata_var_mean_lasso_r2(Path::new(adata_path), &mean_r2_accum) {
                    Ok(()) => log_line(
                        &hud,
                        "Updated var['mean_lasso_r2'] on training AnnData.".to_string(),
                    ),
                    Err(e) => log_line(
                        &hud,
                        format!("Could not write var['mean_lasso_r2']: {}", e),
                    ),
                }
            } else if join_training {
                log_line(
                    &hud,
                    "Join mode: skipped var['mean_lasso_r2'] (multi-writer); use a single-host run for that column."
                        .to_string(),
                );
            }

            if join_training {
                log_line(
                &hud,
                "Join mode: did not overwrite spacetravlr_run_repro.toml (canonical copy from leader run)"
                    .to_string(),
            );
            } else if obs_row_subset.is_none() {
                match spaceship_config.write_run_repro_toml(Path::new(training_dir)) {
                    Ok(p) => log_line(&hud, format!("Wrote run repro {}", p.display())),
                    Err(e) => log_line(&hud, format!("Run repro TOML not written: {}", e)),
                }
            }

            match write_run_summary_html(RunSummaryParams {
                adata_path: Path::new(&worker_adata_path),
                output_dir: Path::new(training_dir),
                cfg: spaceship_config,
                cluster_key: None,
                layer_override: None,
                run_id: None,
                manifest: None,
                betadata_pattern: "*_betadata.feather",
                config_source_path: config_source_path.as_deref(),
            }) {
                Ok(p) => log_line(&hud, format!("Wrote run summary {}", p.display())),
                Err(e) => log_line(&hud, format!("Run summary HTML failed: {}", e)),
            }

            print_training_outcome_banner(&hud);

            Ok(())
        })();

        if let Some(ref h) = hud_for_done {
            if let Ok(mut g) = h.lock() {
                g.finished = Some(match &result {
                    Ok(()) => Ok(()),
                    Err(e) => Err(e.to_string()),
                });
            }
        }

        result
    }
}

impl<AB: AutodiffBackend, AnB: Backend> SpatialCellularProgramsEstimator<AB, AnB> {
    pub fn build_x_modulators_and_target_y(
        &self,
        xy: &Array2<f64>,
    ) -> anyhow::Result<(Array2<f64>, Array1<f64>)> {
        let target_expr = self.get_gene_expression(&self.target_gene)?;

        let mut all_unique_genes: HashSet<String> = HashSet::new();
        for g in &self.regulators {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.ligands {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.receptors {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.tfl_ligands {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.tfl_regulators {
            all_unique_genes.insert(g.clone());
        }
        for g in &self.extra_modulators {
            all_unique_genes.insert(g.clone());
        }

        let unique_genes_vec: Vec<String> = all_unique_genes.into_iter().collect::<Vec<_>>();
        let expr_matrix = self.get_multiple_gene_expressions(&unique_genes_vec)?;

        let mut gene_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, g) in unique_genes_vec.iter().enumerate() {
            gene_to_idx.insert(g.clone(), i);
        }

        // Collect unique ligand genes from LR and TFL pairs for received-ligand computation
        let mut unique_lig_genes: Vec<String> = Vec::new();
        let mut lig_seen: HashSet<String> = HashSet::new();
        for pair in &self.lr_pairs {
            let parts: Vec<&str> = pair.split('$').collect();
            if parts.len() == 2 {
                let lig = parts[0].to_string();
                if lig_seen.insert(lig.clone()) {
                    unique_lig_genes.push(lig);
                }
            }
        }
        for pair in &self.tfl_pairs {
            let parts: Vec<&str> = pair.split('#').collect();
            if parts.len() == 2 {
                let lig = parts[0].to_string();
                if lig_seen.insert(lig.clone()) {
                    unique_lig_genes.push(lig);
                }
            }
        }

        // Compute spatially-weighted received ligands via Gaussian kernel
        let mut received_map: HashMap<String, Array1<f64>> = HashMap::new();
        if !unique_lig_genes.is_empty() {
            let n = xy.nrows();
            let mut lig_expr = Array2::<f64>::zeros((n, unique_lig_genes.len()));
            for (k, lig) in unique_lig_genes.iter().enumerate() {
                let idx = gene_to_idx[lig];
                lig_expr.column_mut(k).assign(&expr_matrix.column(idx));
            }
            let grid_factor = self.ligand_grid_factor.or({
                if n > LARGE_DATASET_GRID_AUTO_CELLS {
                    Some(DEFAULT_LIGAND_GRID_FACTOR)
                } else {
                    None
                }
            });
            let received = match grid_factor {
                Some(gf) if gf.is_finite() && gf > 0.0 => calculate_weighted_ligands_grid(
                    xy,
                    &lig_expr,
                    self.radius,
                    self.weighted_ligand_scale_factor,
                    gf,
                ),
                _ => calculate_weighted_ligands(
                    xy,
                    &lig_expr,
                    self.radius,
                    self.weighted_ligand_scale_factor,
                ),
            };
            for (k, lig) in unique_lig_genes.iter().enumerate() {
                received_map.insert(lig.clone(), received.column(k).to_owned());
            }
        }

        let n_obs = match &self.obs_row_subset {
            Some(rows) => rows.len(),
            None => self.adata.n_obs(),
        };
        let total_modulators = self.regulators.len()
            + self.lr_pairs.len()
            + self.tfl_pairs.len()
            + self.extra_modulators.len();
        let mut x_modulators = Array2::<f64>::zeros((n_obs, total_modulators));

        for (i, gene) in self.regulators.iter().enumerate() {
            let idx = gene_to_idx[gene];
            x_modulators.column_mut(i).assign(&expr_matrix.column(idx));
        }

        let offset_lr = self.regulators.len();
        for (i, pair) in self.lr_pairs.iter().enumerate() {
            let parts: Vec<&str> = pair.split('$').collect::<Vec<_>>();
            if parts.len() == 2 {
                let lig_name = parts[0].to_string();
                let r_idx = gene_to_idx[&parts[1].to_string()];
                let mut interaction = received_map[&lig_name].clone();
                interaction *= &expr_matrix.column(r_idx);
                x_modulators.column_mut(offset_lr + i).assign(&interaction);
            }
        }

        let offset_tfl = offset_lr + self.lr_pairs.len();
        for (i, pair) in self.tfl_pairs.iter().enumerate() {
            let parts: Vec<&str> = pair.split('#').collect::<Vec<_>>();
            if parts.len() == 2 {
                let lig_name = parts[0].to_string();
                let tf_idx = gene_to_idx[&parts[1].to_string()];
                let mut interaction = received_map[&lig_name].clone();
                interaction *= &expr_matrix.column(tf_idx);
                x_modulators.column_mut(offset_tfl + i).assign(&interaction);
            }
        }

        let offset_extra = offset_tfl + self.tfl_pairs.len();
        for (i, gene) in self.extra_modulators.iter().enumerate() {
            let idx = gene_to_idx[gene];
            x_modulators
                .column_mut(offset_extra + i)
                .assign(&expr_matrix.column(idx));
        }

        Ok((x_modulators, target_expr))
    }

    pub fn fit_with_cache<F: FnMut(usize, usize)>(
        &mut self,
        xy: &Array2<f64>,
        clusters: &Array1<usize>,
        num_clusters: usize,
        epochs: usize,
        learning_rate: f64,
        _score_threshold: f64,
        l1_reg: f64,
        group_reg: f64,
        n_iter: usize,
        tol: f64,
        scale_modulators: bool,
        _estimator_type: &str,
        cnn: &CnnConfig,
        device: &AB::Device,
        cached_spatial: Option<&CachedSpatialData>,
        lasso_progress: F,
    ) -> anyhow::Result<()> {
        let (mut x_modulators, target_expr) = self.build_x_modulators_and_target_y(xy)?;
        if scale_modulators {
            let scales = scale_columns_no_center(&mut x_modulators);
            self.modulator_scales = Some(scales);
        } else {
            self.modulator_scales = None;
        }

        if self.estimator.is_none() {
            let mut groups = Vec::new();
            for _ in 0..self.regulators.len() {
                groups.push(0);
            }
            for _ in 0..self.lr_pairs.len() {
                groups.push(1);
            }
            for _ in 0..self.tfl_pairs.len() {
                groups.push(2);
            }
            for _ in 0..self.extra_modulators.len() {
                groups.push(3);
            }

            let params = GroupLassoParams {
                l1_reg,
                group_reg,
                groups,
                n_iter,
                tol,
                ..Default::default()
            };
            let mut est =
                ClusteredGCNNWR::new(params, self.spatial_dim, cnn.spatial_feature_radius);
            est.group_reg_vec = self.group_reg_vec.clone();
            est.regulator_masks_by_cluster = self.regulator_masks_by_cluster.clone();
            self.estimator = Some(est);
        }

        if let Some(est) = &mut self.estimator {
            est.fit(
                &x_modulators,
                &target_expr,
                xy,
                clusters,
                num_clusters,
                device,
                epochs,
                learning_rate,
                self.seed_only,
                cnn,
                cached_spatial,
                lasso_progress,
            );
        }
        Ok(())
    }

    pub fn fit(
        &mut self,
        epochs: usize,
        learning_rate: f64,
        score_threshold: f64,
        l1_reg: f64,
        group_reg: f64,
        n_iter: usize,
        tol: f64,
        estimator_type: &str,
        device: &AB::Device,
    ) -> anyhow::Result<()> {
        let obs_df = self.adata.read_obs()?;
        let xy: Array2<f64> = self
            .adata
            .obsm()
            .get_item("spatial")?
            .ok_or_else(|| anyhow::anyhow!("obsm['spatial'] not found"))?;
        let clusters: Array1<usize> =
            clusters_array1_from_obs_column(&obs_df, self.cluster_annot.as_str())?;
        let num_clusters = clusters
            .iter()
            .copied()
            .max()
            .map(|m| m.saturating_add(1))
            .unwrap_or(1);

        let cnn = CnnConfig::default();
        let scale_mod = crate::config::LassoConfig::default().scale_modulators;
        self.fit_with_cache(
            &xy,
            &clusters,
            num_clusters,
            epochs,
            learning_rate,
            score_threshold,
            l1_reg,
            group_reg,
            n_iter,
            tol,
            scale_mod,
            estimator_type,
            &cnn,
            device,
            None,
            |_, _| {},
        )
    }

    pub fn get_multiple_gene_expressions(&self, genes: &[String]) -> anyhow::Result<Array2<f64>> {
        let mut gene_indices: Vec<usize> = Vec::new();
        for gene in genes {
            let idx = self
                .adata
                .var_names()
                .get_index(gene)
                .ok_or_else(|| anyhow::anyhow!("Gene {} not found in var_names", gene))?;
            gene_indices.push(idx);
        }

        let row_sel = match &self.obs_row_subset {
            Some(indices) => SelectInfoElem::Index(indices.to_vec()),
            None => SelectInfoElem::full(),
        };
        let slice = [row_sel, SelectInfoElem::Index(gene_indices)];

        read_expression_matrix_dense_f64(&self.adata, &self.layer, &slice)
    }

    pub fn get_gene_expression(&self, gene: &str) -> anyhow::Result<Array1<f64>> {
        self.get_multiple_gene_expressions(&[gene.to_string()])
            .map(|data: Array2<f64>| data.column(0).to_owned())
    }
}

#[cfg(test)]
mod scale_columns_tests {
    use crate::lasso::{GroupLasso, GroupLassoParams};
    use crate::modulator_scale::{scale_columns_no_center, unscale_betadata_columns_inplace};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    fn pop_std(col: ndarray::ArrayView1<f64>) -> f64 {
        let n = col.len() as f64;
        let m = col.sum() / n;
        (col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt()
    }

    #[test]
    fn scale_columns_matches_celloracle_standard() {
        let x0 = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 4.0, 2.0, 4.0, 6.0, 3.0, 6.0, 8.0, 4.0, 8.0, 10.0],
        )
        .unwrap();
        let mut x = x0.clone();
        let scales = scale_columns_no_center(&mut x);
        for j in 0..3 {
            let s = pop_std(x0.column(j));
            let exp_s = if s > 1e-12 { s } else { 1.0 };
            assert_abs_diff_eq!(scales[j], exp_s, epsilon = 1e-9);
            for i in 0..4 {
                assert_abs_diff_eq!(x[[i, j]], x0[[i, j]] / exp_s, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn scale_columns_zero_std_passthrough() {
        let mut x = Array2::from_elem((3, 2), 5.0f64);
        let scales = scale_columns_no_center(&mut x);
        assert_abs_diff_eq!(scales[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(scales[1], 1.0, epsilon = 1e-12);
        assert!(x.iter().all(|&v| (v - 5.0).abs() < 1e-12));
    }

    #[test]
    fn scale_columns_no_mean_subtraction() {
        let mut x = Array2::from_elem((5, 2), 3.0f64);
        x[[0, 0]] = 10.0;
        let _ = scale_columns_no_center(&mut x);
        assert!(x.column(0).sum().abs() > 1e-6);
    }

    #[test]
    fn unscale_betadata_columns_inplace_skips_intercept() {
        let mut betas = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 4.0, 1.0, 6.0, 8.0]).unwrap();
        let scales = Array1::from_vec(vec![0.5, 2.0]);
        unscale_betadata_columns_inplace(&mut betas, &scales);
        assert_abs_diff_eq!(betas[[0, 0]], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(betas[[0, 1]], 4.0, epsilon = 1e-9);
        assert_abs_diff_eq!(betas[[0, 2]], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn unscale_roundtrip_matches_predict() {
        let n = 50usize;
        let x_orig = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                i as f64 * 10.0
            } else {
                (n - i) as f64 * 0.01
            }
        });
        let y = Array2::from_shape_fn((n, 1), |(i, _)| {
            1.5 * x_orig[[i, 0]] + 2.0 * x_orig[[i, 1]]
        });
        let mut x_scaled = x_orig.clone();
        let scales = scale_columns_no_center(&mut x_scaled);
        let params = GroupLassoParams {
            groups: vec![0, 1],
            group_reg: 1e-8,
            l1_reg: 1e-8,
            n_iter: 800,
            tol: 1e-10,
            fit_intercept: true,
            ..Default::default()
        };
        let mut model = GroupLasso::new(params);
        let _ = model.fit(&x_scaled, &y, None);
        let fitted = model.fitted.as_ref().unwrap();
        let intercept = fitted.intercept[[0, 0]];
        let pred_scaled = model.predict(&x_scaled).unwrap();
        let mut coef_orig = fitted.coef.column(0).to_owned();
        for j in 0..coef_orig.len() {
            if scales[j] != 1.0 {
                coef_orig[j] /= scales[j];
            }
        }
        for i in 0..n {
            let mut v = intercept;
            for j in 0..2 {
                v += x_orig[[i, j]] * coef_orig[j];
            }
            assert_abs_diff_eq!(v, pred_scaled[[i, 0]], epsilon = 1e-5);
        }
    }
}

#[cfg(test)]
mod mean_lasso_r2_patch_tests {
    use super::{
        AnnData, AnnDataOp, ArrayData, Backend, MeanLassoR2Accum, dense_to_csr_f64,
        patch_adata_var_mean_lasso_r2,
    };
    use anndata_hdf5::H5;
    use ndarray::Array2;
    use polars::prelude::{DataFrame, NamedFrom, Series};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Once;
    use std::thread;
    use std::time::Duration;

    static PATCH_TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(0);
    static HDF5_TEST_LOCK_ENV: Once = Once::new();

    #[test]
    fn patch_var_mean_lasso_r2_column() {
        HDF5_TEST_LOCK_ENV.call_once(|| {
            // macOS / parallel `cargo test`: HDF5 single-writer flock can return EAGAIN (errno 35)
            // between close and reopen; disabling file locking is the documented workaround.
            unsafe {
                std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
            }
        });

        const MAX_ATTEMPTS: usize = 12;
        const BACKOFF: Duration = Duration::from_millis(40);

        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..MAX_ATTEMPTS {
            if attempt > 0 {
                thread::sleep(BACKOFF);
            }

            let seq = PATCH_TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "spacetravlr_var_patch_{}_{}",
                std::process::id(),
                seq
            ));
            let _ = std::fs::remove_dir_all(&dir);
            if std::fs::create_dir_all(&dir).is_err() {
                continue;
            }
            let p = dir.join("mini.h5ad");

            let step = (|| -> anyhow::Result<()> {
                {
                    let a = AnnData::<H5>::new(&p)?;
                    a.set_obs_names(vec!["c0".into(), "c1".into()].into())?;
                    a.set_var_names(vec!["G0".into(), "G1".into()].into())?;
                    let obs = DataFrame::new(vec![Series::new(
                        "cell_type".into(),
                        vec!["x".to_string(), "x".to_string()],
                    )
                    .into()])?;
                    a.set_obs(obs)?;
                    let var =
                        DataFrame::new(vec![Series::new("gene_ids".into(), vec!["g0", "g1"]).into()])?;
                    a.set_var(var)?;
                    let dense = Array2::from_elem((2, 2), 0.5f64);
                    let csr = dense_to_csr_f64(&dense)?;
                    a.set_x(ArrayData::from(csr))?;
                    a.close()?;
                }
                thread::sleep(Duration::from_millis(5));

                let mut m: HashMap<String, usize> = HashMap::new();
                m.insert("G0".into(), 0);
                m.insert("G1".into(), 1);
                let scores = Arc::new(vec![
                    AtomicU64::new(0.25f64.to_bits()),
                    AtomicU64::new(f64::NAN.to_bits()),
                ]);
                let accum = MeanLassoR2Accum {
                    gene_to_idx: Arc::new(m),
                    scores,
                };
                patch_adata_var_mean_lasso_r2(&p, &accum)?;

                let a2 = AnnData::<H5>::open(H5::open(&p)?)?;
                let v = a2.read_var()?;
                let r2 = v.column("mean_lasso_r2")?.f64()?;
                anyhow::ensure!((r2.get(0).unwrap() - 0.25).abs() < 1e-9);
                anyhow::ensure!(r2.get(1).unwrap().is_nan());
                a2.close()?;
                Ok(())
            })();

            match step {
                Ok(()) => {
                    let _ = std::fs::remove_dir_all(&dir);
                    return;
                }
                Err(e) => {
                    last_err = Some(e);
                    let _ = std::fs::remove_dir_all(&dir);
                }
            }
        }

        panic!(
            "patch_var_mean_lasso_r2_column failed after {MAX_ATTEMPTS} attempts: {:?}",
            last_err
        );
    }
}
