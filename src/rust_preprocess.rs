//! Rust-native Scanpy-style preprocessing: filter → normalize → log1p → HVG → PCA →
//! HNSW KNN → UMAP. Writes `.h5ad` via `anndata-memory` + `convert_to_new_backed_h5`.
//! Does not run Leiden or MAGIC (use the Python `--process-h5ad` path for those).

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anndata::data::SelectInfoElem;
use anndata::data::array::DynArray;
use anndata::data::{DynCscMatrix, DynCsrMatrix};
use anndata::{ArrayData};
use anndata_memory::{
    convert_to_new_backed_h5, load_h5ad_fast, IMAnnData, IMArrayElement, IMAxisArrays,
};
use anyhow::{Context, Result, anyhow, bail};
use instant_distance::{Builder, Hnsw, Search};
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray_umap::Array2 as Array2Umap;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use polars::prelude::{Column, DataFrame, DataType};
use rayon::prelude::*;
use sprs::CsMatI;

use single_algebra::dimred::pca::{PowerIterationNormalizer, SVDMethod};
use nalgebra_sparse::CsrMatrix;
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use single_rust::memory::processing::dimred::pca::run_pca_sparse_masked;
use single_rust::memory::processing::filtering::{mark_filter_cells, mark_filter_genes};
use single_rust::memory::processing::{
    compute_highly_variable_genes, log1p_expression, normalize_expression,
};
use single_rust::shared::HVGParams;
use single_utilities::types::Direction;
use umap_rs::{
    EuclideanMetric, GraphParams, ManifoldParams, MetricType, OptimizationParams, Optimizer,
    Umap, UmapConfig,
};

type FuzzyGraph = CsMatI<f32, u32, usize>;

const INIT_NOISE_STD: f32 = 1e-4;

#[derive(Clone, Debug)]
pub struct RustPreprocessParams {
    pub n_top_hvg: usize,
    pub n_pca_components: usize,
    pub n_neighbors: usize,
    pub min_dist: f32,
    pub n_epochs: Option<usize>,
    pub ef_construction: usize,
}

impl Default for RustPreprocessParams {
    fn default() -> Self {
        Self {
            n_top_hvg: 2000,
            n_pca_components: 50,
            n_neighbors: 15,
            min_dist: 0.5,
            n_epochs: None,
            ef_construction: 200,
        }
    }
}

#[derive(Clone)]
struct PcaVec(Vec<f32>);

impl instant_distance::Point for PcaVec {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }
}

thread_local! {
    static HNSW_SEARCH: RefCell<Search> = RefCell::new(Search::default());
}

fn mask_to_indices(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .collect()
}

fn pca_to_points_f32(pca: &ndarray::Array2<f64>, dim: usize) -> Vec<PcaVec> {
    let n = pca.nrows();
    let mut out = Vec::with_capacity(n);
    for row in pca.outer_iter() {
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            v.push(*row.get(j).unwrap_or(&0.0) as f32);
        }
        out.push(PcaVec(v));
    }
    out
}

fn knn_find_components(
    knn_idx: &Array2Umap<u32>,
    n: usize,
    n_neighbors: usize,
) -> (Vec<usize>, usize) {
    let mut head = vec![u32::MAX; n];
    let mut next = vec![u32::MAX; n * (n_neighbors - 1) * 2];
    let mut target = vec![0u32; n * (n_neighbors - 1) * 2];
    let mut edge_cnt = 0usize;

    let mut push = |from: usize, to: usize| {
        target[edge_cnt] = to as u32;
        next[edge_cnt] = head[from];
        head[from] = edge_cnt as u32;
        edge_cnt += 1;
    };
    for i in 0..n {
        for j in 1..n_neighbors {
            let nb = knn_idx[(i, j)] as usize;
            push(i, nb);
            push(nb, i);
        }
    }

    let mut comp = vec![usize::MAX; n];
    let mut n_comps = 0usize;
    let mut queue = std::collections::VecDeque::new();
    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        comp[start] = n_comps;
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            let mut e = head[v];
            while e != u32::MAX {
                let nb = target[e as usize] as usize;
                if comp[nb] == usize::MAX {
                    comp[nb] = n_comps;
                    queue.push_back(nb);
                }
                e = next[e as usize];
            }
        }
        n_comps += 1;
    }
    (comp, n_comps)
}

fn bridge_knn_components(
    knn_idx: &mut Array2Umap<u32>,
    knn_dist: &mut Array2Umap<f32>,
    points: &[PcaVec],
    n_neighbors: usize,
    hnsw: &Hnsw<PcaVec>,
    pid_to_orig: &[u32],
    ef_search_bridge: usize,
) {
    let n = points.len();

    let sort_row =
        |knn_idx: &mut Array2Umap<u32>, knn_dist: &mut Array2Umap<f32>, row: usize| {
            let mut pairs: Vec<(u32, f32)> = (1..n_neighbors)
                .map(|j| (knn_idx[(row, j)], knn_dist[(row, j)]))
                .collect();
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (j, (idx, d)) in pairs.into_iter().enumerate() {
                knn_idx[(row, j + 1)] = idx;
                knn_dist[(row, j + 1)] = d;
            }
        };

    loop {
        let (comp, n_comps) = knn_find_components(knn_idx, n, n_neighbors);
        if n_comps == 1 {
            eprintln!("  KNN graph: fully connected (1 component)");
            break;
        }

        let mut comp_sizes = vec![0usize; n_comps];
        for &c in &comp {
            comp_sizes[c] += 1;
        }
        let main_id = comp_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| *s)
            .unwrap()
            .0;
        let mut s = comp_sizes.clone();
        s.sort_unstable_by(|a, b| b.cmp(a));
        eprintln!(
            "  KNN graph: {n_comps} components, sizes: {:?}",
            &s[..s.len().min(10)]
        );

        let mut added = 0usize;
        for comp_id in 0..n_comps {
            if comp_id == main_id {
                continue;
            }
            let comp_pts: Vec<usize> = (0..n).filter(|&i| comp[i] == comp_id).collect();
            let sample: Vec<usize> = if comp_pts.len() <= 50 {
                comp_pts.clone()
            } else {
                let step = (comp_pts.len() / 50).max(1);
                comp_pts.iter().step_by(step).take(50).copied().collect()
            };

            let mut best: Option<(usize, usize, f32)> = None;
            'outer: for &p in &sample {
                let results: Vec<(u32, f32)> = HNSW_SEARCH.with_borrow_mut(|search| {
                    hnsw
                        .search(&points[p], search)
                        .take(ef_search_bridge)
                        .map(|item| {
                            (
                                pid_to_orig[item.pid.into_inner() as usize],
                                item.distance,
                            )
                        })
                        .collect()
                });
                for (q_orig, d) in results {
                    if comp[q_orig as usize] == main_id {
                        if best.map_or(true, |(_, _, bd)| d < bd) {
                            best = Some((p, q_orig as usize, d));
                        }
                        break 'outer;
                    }
                }
            }

            if let Some((p, q, d)) = best {
                knn_idx[(p, n_neighbors - 1)] = q as u32;
                knn_dist[(p, n_neighbors - 1)] = d;
                sort_row(knn_idx, knn_dist, p);
                knn_idx[(q, n_neighbors - 1)] = p as u32;
                knn_dist[(q, n_neighbors - 1)] = d;
                sort_row(knn_idx, knn_dist, q);
                added += 1;
            }
        }
        eprintln!("  bridged {added} component(s)");
        if added == 0 {
            break;
        }
    }
}

fn knn_indices_dists(
    points: &[PcaVec],
    n_neighbors: usize,
    ef_construction: usize,
) -> (Array2Umap<u32>, Array2Umap<f32>) {
    let n = points.len();
    let ef_search = (n_neighbors * 3).max(50);
    let ef_bridge = (n_neighbors * 20).max(300).min(n);

    eprintln!("  building HNSW (ef_construction={ef_construction}, ef_search={ef_search})…");
    let (hnsw, pids) = Builder::default()
        .ef_construction(ef_construction)
        .ef_search(ef_search)
        .seed(42)
        .build_hnsw(points.to_vec());

    let mut pid_to_orig = vec![0u32; n];
    for (orig, pid) in pids.iter().enumerate() {
        pid_to_orig[pid.into_inner() as usize] = orig as u32;
    }

    let search_k = n_neighbors + 8;
    eprintln!("  querying {n} points (HNSW ef_search={ef_search})…");
    let mut rows: Vec<(usize, Vec<u32>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let results: Vec<(u32, f32)> = HNSW_SEARCH.with_borrow_mut(|search| {
                hnsw
                    .search(&points[i], search)
                    .take(search_k)
                    .map(|item| {
                        (
                            pid_to_orig[item.pid.into_inner() as usize],
                            item.distance,
                        )
                    })
                    .collect()
            });

            let mut idx_row = Vec::with_capacity(n_neighbors);
            let mut dist_row = Vec::with_capacity(n_neighbors);
            idx_row.push(i as u32);
            dist_row.push(0.0f32);
            for (nb, d) in results {
                if nb == i as u32 {
                    continue;
                }
                if idx_row.len() >= n_neighbors {
                    break;
                }
                idx_row.push(nb);
                dist_row.push(d);
            }
            while idx_row.len() < n_neighbors {
                idx_row.push(i as u32);
                dist_row.push(0.0f32);
            }
            (i, idx_row, dist_row)
        })
        .collect();
    rows.sort_by_key(|(i, _, _)| *i);

    let mut idx = Array2Umap::<u32>::zeros((n, n_neighbors));
    let mut dist = Array2Umap::<f32>::zeros((n, n_neighbors));
    for (i, ir, dr) in rows.into_iter() {
        for (j, (&ci, &cd)) in ir.iter().zip(dr.iter()).enumerate() {
            idx[(i, j)] = ci;
            dist[(i, j)] = cd;
        }
    }

    eprintln!("  checking KNN graph connectivity…");
    bridge_knn_components(
        &mut idx,
        &mut dist,
        points,
        n_neighbors,
        &hnsw,
        &pid_to_orig,
        ef_bridge,
    );

    (idx, dist)
}

fn norm_affinity_matvec(graph: &FuzzyGraph, d_inv_sqrt: &[f32], x: &[f32]) -> Vec<f32> {
    let n = graph.rows();
    let scaled: Vec<f32> = x
        .iter()
        .zip(d_inv_sqrt.iter())
        .map(|(&xi, &di)| xi * di)
        .collect();

    (0..n)
        .into_par_iter()
        .map(|i| {
            let row = graph.outer_view(i).unwrap();
            let mut sum = 0.0f32;
            for (j_idx, &val) in row.indices().iter().zip(row.data().iter()) {
                sum += val * scaled[*j_idx as usize];
            }
            sum * d_inv_sqrt[i]
        })
        .collect()
}

fn mgs_orthonormalize(cols: &mut [Vec<f32>]) {
    let r = cols.len();
    for i in 0..r {
        for j in 0..i {
            let dot: f32 = cols[i]
                .iter()
                .zip(cols[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            for k in 0..cols[i].len() {
                cols[i][k] -= dot * cols[j][k];
            }
        }
        let norm: f32 = cols[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in cols[i].iter_mut() {
                *x /= norm;
            }
        }
    }
}

fn deflate_constant(cols: &mut [Vec<f32>], v1: &[f32]) {
    for col in cols.iter_mut() {
        let dot: f32 = col.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
        for k in 0..col.len() {
            col[k] -= dot * v1[k];
        }
    }
}

fn spectral_init_2d(graph: &FuzzyGraph, n_components: usize, seed: u64) -> Array2Umap<f32> {
    let n = graph.rows();
    let n_oversamples = 8;
    let r = n_components + n_oversamples;
    let n_iter = 8;

    let d: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| graph.outer_view(i).unwrap().data().iter().sum())
        .collect();
    let d_inv_sqrt: Vec<f32> = d
        .iter()
        .map(|&x| if x > 1e-12 { 1.0 / x.sqrt() } else { 0.0 })
        .collect();

    let total: f32 = d.iter().sum();
    let total_sqrt = total.sqrt().max(1e-30);
    let v1: Vec<f32> = d.iter().map(|&x| x.sqrt() / total_sqrt).collect();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega: Vec<Vec<f32>> = (0..r)
        .map(|_| (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    deflate_constant(&mut omega, &v1);
    mgs_orthonormalize(&mut omega);

    for _ in 0..n_iter {
        let mut y: Vec<Vec<f32>> = omega
            .par_iter()
            .map(|col| norm_affinity_matvec(graph, &d_inv_sqrt, col))
            .collect();
        deflate_constant(&mut y, &v1);
        mgs_orthonormalize(&mut y);
        omega = y;
    }

    let aq: Vec<Vec<f32>> = omega
        .par_iter()
        .map(|col| {
            let mut col_a = norm_affinity_matvec(graph, &d_inv_sqrt, col);
            let dot: f32 = col_a.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
            for k in 0..col_a.len() {
                col_a[k] -= dot * v1[k];
            }
            col_a
        })
        .collect();

    let mut b = DMatrix::<f32>::zeros(r, r);
    for i in 0..r {
        for j in 0..r {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += omega[i][k] * aq[j][k];
            }
            b[(i, j)] = sum;
        }
    }
    let b_sym = (b.clone() + b.transpose()) * 0.5f32;

    let eigen = SymmetricEigen::new(b_sym);
    let mut eig_pairs: Vec<(f32, usize)> = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    eig_pairs.sort_by(|a, c| c.0.partial_cmp(&a.0).unwrap());

    eprintln!(
        "  spectral eigenvalues (top {} in deflated subspace): {:?}",
        eig_pairs.len().min(5),
        eig_pairs
            .iter()
            .take(5)
            .map(|(v, _)| *v)
            .collect::<Vec<_>>()
    );

    let mut init = Array2Umap::<f32>::zeros((n, n_components));
    let mut noise = StdRng::seed_from_u64(seed.wrapping_add(1));
    for (out_idx, &(_eig_val, eig_idx)) in eig_pairs.iter().take(n_components).enumerate() {
        let v_small = eigen.eigenvectors.column(eig_idx);
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..r {
                s += omega[j][i] * v_small[j];
            }
            init[(i, out_idx)] = s + noise.gen_range(-INIT_NOISE_STD..INIT_NOISE_STD);
        }
    }
    init
}

fn run_umap(
    pca: &ndarray::Array2<f64>,
    params: &RustPreprocessParams,
    log: &mut Vec<(String, f64)>,
) -> Result<Array2Umap<f32>> {
    let n = pca.nrows();
    let dim = params.n_pca_components;
    if pca.ncols() < dim {
        anyhow::bail!(
            "PCA has {} columns but n_pca_components is {}",
            pca.ncols(),
            dim
        );
    }

    let t0 = Instant::now();
    eprintln!(">>> umap KNN (HNSW)");
    let points = pca_to_points_f32(pca, dim);
    let (knn_idx, knn_dist) = knn_indices_dists(&points, params.n_neighbors, params.ef_construction);
    eprintln!("<<< umap KNN (HNSW): {:.2} s", t0.elapsed().as_secs_f64());
    log.push(("umap KNN (HNSW)".to_string(), t0.elapsed().as_secs_f64()));

    let data = Array2Umap::from_shape_vec(
        (n, dim),
        points.into_iter().flat_map(|p| p.0).collect(),
    )
    .map_err(|e| anyhow!("UMAP data shape: {e}"))?;

    let n_epochs = params
        .n_epochs
        .unwrap_or_else(|| if n <= 10_000 { 500 } else { 200 });

    let config = UmapConfig {
        n_components: 2,
        manifold: ManifoldParams {
            min_dist: params.min_dist,
            spread: 1.0,
            ..Default::default()
        },
        graph: GraphParams {
            n_neighbors: params.n_neighbors,
            symmetrize: true,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: Some(n_epochs),
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
        },
    };

    let t1 = Instant::now();
    eprintln!(">>> umap learn_manifold (fuzzy graph)");
    let umap = Umap::new(config.clone());
    let manifold = umap.learn_manifold(data.view(), knn_idx.view(), knn_dist.view());
    eprintln!(
        "<<< umap learn_manifold (fuzzy graph): {:.2} s",
        t1.elapsed().as_secs_f64()
    );
    log.push((
        "umap learn_manifold (fuzzy graph)".to_string(),
        t1.elapsed().as_secs_f64(),
    ));

    let t2 = Instant::now();
    eprintln!(">>> umap spectral init");
    let init = spectral_init_2d(manifold.graph(), 2, 42);
    eprintln!("<<< umap spectral init: {:.2} s", t2.elapsed().as_secs_f64());
    log.push(("umap spectral init".to_string(), t2.elapsed().as_secs_f64()));

    let t3 = Instant::now();
    eprintln!(">>> umap optimize (umap-rs)");
    let mut opt = Optimizer::new(manifold, init, n_epochs, &config, MetricType::Euclidean);
    opt.step_epochs(n_epochs, &EuclideanMetric);
    let fitted = opt.into_fitted(config);
    eprintln!(
        "<<< umap optimize (umap-rs): {:.2} s",
        t3.elapsed().as_secs_f64()
    );
    log.push(("umap optimize (umap-rs)".to_string(), t3.elapsed().as_secs_f64()));

    Ok(fitted.into_embedding())
}

fn ensure_x_csr_for_pca(adata: &IMAnnData) -> Result<()> {
    let x = adata.x().get_data()?;
    match x {
        ArrayData::CsrMatrix(_) => Ok(()),
        ArrayData::CscMatrix(csc) => {
            let csr = match csc {
                DynCscMatrix::F32(m) => ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&m))),
                DynCscMatrix::F64(m) => ArrayData::CsrMatrix(DynCsrMatrix::F64(CsrMatrix::from(&m))),
                _ => bail!(
                    "rust_preprocess: X CSC matrix must be F32 or F64 for conversion to CSR"
                ),
            };
            adata.x().set_data(csr)?;
            Ok(())
        }
        _ => bail!(
            "rust_preprocess: X must be CSR or CSC sparse matrix for PCA (got other layout)"
        ),
    }
}

fn clear_uns_for_hdf5_export(adata: &IMAnnData) -> Result<()> {
    let keys = adata.uns().keys().context("uns.keys")?;
    for k in keys {
        adata.uns().remove_data(&k)?;
    }
    Ok(())
}

fn polars_series_dtype_ann_dataframe_hdf5(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Boolean
            | DataType::String
    )
}

fn strip_supplemental_axis_arrays_for_h5_export(adata: &IMAnnData) -> Result<()> {
    const KEEP_OBSM: &[&str] = &["X_pca", "X_umap"];
    let obsm = adata.obsm();
    for k in obsm.keys() {
        if !KEEP_OBSM.contains(&k.as_str()) {
            obsm.remove_array(&k)?;
        }
    }
    for ax in [adata.obsp(), adata.varm(), adata.varp()] {
        for k in ax.keys() {
            ax.remove_array(&k)?;
        }
    }
    Ok(())
}

fn temp_h5ad_path(output: &Path) -> Result<PathBuf> {
    let dir = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let name = output
        .file_name()
        .ok_or_else(|| anyhow!("output path has no file name"))?;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(dir.join(format!(
        ".{}.{t}.part.h5ad",
        name.to_string_lossy()
    )))
}

fn dataframe_hdf5_safe(df: DataFrame) -> Result<DataFrame> {
    let mut cols: Vec<Column> = Vec::new();
    for s in df.iter() {
        let dt = s.dtype();
        let column = match dt {
            DataType::List(_) | DataType::Array(_, _) | DataType::Struct(_) => continue,
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                let s2 = s
                    .clone()
                    .cast(&DataType::String)
                    .with_context(|| format!("cast obs/var column {} to String", s.name()))?;
                Column::from(s2)
            }
            dt if polars_series_dtype_ann_dataframe_hdf5(dt) => Column::from(s.clone()),
            _ => {
                let s2 = s.clone().cast(&DataType::String).with_context(|| {
                    format!(
                        "cast obs/var column {} from {dt:?} to String for HDF5 export",
                        s.name()
                    )
                })?;
                Column::from(s2)
            }
        };
        cols.push(column);
    }
    if cols.is_empty() {
        bail!(
            "after HDF5 sanitization, obs/var has no exportable columns (all dropped or failed cast)"
        );
    }
    DataFrame::new(cols).map_err(|e| anyhow!("{e}"))
}

fn read_var_hvg_mask(adata: &IMAnnData) -> Result<Vec<bool>> {
    let column = adata
        .var()
        .get_column_from_df("highly_variable")
        .context("var has no highly_variable column")?;
    let series = column.as_materialized_series();
    Ok(series
        .bool()
        .map_err(|e| anyhow!("highly_variable not bool: {e:?}"))?
        .into_iter()
        .map(|o: Option<bool>| o.unwrap_or(false))
        .collect())
}

fn axis_replace_array(axis: &IMAxisArrays, key: &str, data: ArrayData) -> Result<()> {
    let _ = axis.remove_array(key);
    axis.add_array(key.to_string(), IMArrayElement::new(data))?;
    Ok(())
}

fn layer_replace_if_present(adata: &IMAnnData, key: &str, data: ArrayData) -> Result<()> {
    let _ = adata.layers().remove_array(key);
    adata.layers().add_array(key.to_string(), IMArrayElement::new(data))?;
    Ok(())
}

/// Full Rust preprocessing pipeline; writes `output` as a backed HDF5 AnnData.
pub fn rust_preprocess_h5ad(
    input: &Path,
    output: &Path,
    params: &RustPreprocessParams,
) -> Result<PathBuf> {
    let mut log: Vec<(String, f64)> = Vec::new();

    let t0 = Instant::now();
    eprintln!(">>> read_h5ad");
    let mut adata = load_h5ad_fast(input).context("load_h5ad_fast")?;
    eprintln!(
        "  loaded shape=({}, {})",
        adata.n_obs(),
        adata.n_vars()
    );
    eprintln!(
        "<<< read_h5ad: {:.2} s",
        t0.elapsed().as_secs_f64()
    );
    log.push(("read_h5ad".to_string(), t0.elapsed().as_secs_f64()));

    let t = Instant::now();
    eprintln!(">>> filter_genes(min_cells=3)");
    let gene_mask =
        mark_filter_genes::<u32, f64>(&adata, Some(3u32), None, None, None, None, None)
            .map_err(|e| anyhow!("mark_filter_genes: {e:?}"))?;
    eprintln!(
        "<<< filter_genes(min_cells=3): {:.2} s",
        t.elapsed().as_secs_f64()
    );
    log.push(("filter_genes(min_cells=3)".to_string(), t.elapsed().as_secs_f64()));

    let t = Instant::now();
    eprintln!(">>> filter_cells(min_genes=100)");
    let cell_mask =
        mark_filter_cells::<u32, f64>(&adata, Some(100u32), None, None, None, None, None)
            .map_err(|e| anyhow!("mark_filter_cells: {e:?}"))?;
    eprintln!(
        "<<< filter_cells(min_genes=100): {:.2} s",
        t.elapsed().as_secs_f64()
    );
    log.push((
        "filter_cells(min_genes=100)".to_string(),
        t.elapsed().as_secs_f64(),
    ));

    let t = Instant::now();
    eprintln!(">>> apply masks (subset)");
    let cell_idx = mask_to_indices(&cell_mask);
    let gene_idx = mask_to_indices(&gene_mask);
    let obs_sel = SelectInfoElem::from(cell_idx.clone());
    let var_sel = SelectInfoElem::from(gene_idx.clone());
    adata = adata
        .subset(&[&obs_sel, &var_sel])
        .map_err(|e| anyhow!("subset: {e:?}"))?;
    eprintln!(
        "  shape after filter: ({}, {})",
        adata.n_obs(),
        adata.n_vars()
    );
    eprintln!(
        "<<< apply masks (subset): {:.2} s",
        t.elapsed().as_secs_f64()
    );
    log.push(("apply masks (subset)".to_string(), t.elapsed().as_secs_f64()));

    let t = Instant::now();
    eprintln!(">>> normalize_total");
    normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)
        .map_err(|e| anyhow!("normalize_expression: {e:?}"))?;
    let norm_data = adata.x().get_data().context("x after normalize")?;
    layer_replace_if_present(&adata, "normalized_count", norm_data)?;
    eprintln!("<<< normalize_total: {:.2} s", t.elapsed().as_secs_f64());
    log.push(("normalize_total".to_string(), t.elapsed().as_secs_f64()));

    let t = Instant::now();
    eprintln!(">>> log1p");
    log1p_expression(&adata.x(), None).map_err(|e| anyhow!("log1p_expression: {e:?}"))?;
    let log_data = adata.x().get_data().context("x after log1p")?;
    layer_replace_if_present(&adata, "log1p", log_data)?;
    eprintln!("<<< log1p: {:.2} s", t.elapsed().as_secs_f64());
    log.push(("log1p".to_string(), t.elapsed().as_secs_f64()));

    let hvg_target = params
        .n_top_hvg
        .min(adata.n_vars().saturating_sub(50).max(1));
    let t = Instant::now();
    eprintln!(">>> highly_variable_genes({hvg_target})");
    compute_highly_variable_genes(
        &adata,
        Some(HVGParams {
            n_top_genes: Some(hvg_target),
            ..Default::default()
        }),
    )
    .map_err(|e| anyhow!("compute_highly_variable_genes: {e:?}"))?;
    eprintln!(
        "<<< highly_variable_genes({hvg_target}): {:.2} s",
        t.elapsed().as_secs_f64()
    );
    log.push((
        format!("highly_variable_genes({hvg_target})"),
        t.elapsed().as_secs_f64(),
    ));

    let t = Instant::now();
    eprintln!(">>> subset HVG & drop mt");
    let hvg_mask = read_var_hvg_mask(&adata)?;
    let var_names = adata.var_names();
    let combined_mask: Vec<bool> = hvg_mask
        .iter()
        .zip(var_names.iter())
        .map(|(&hv, name)| hv && !name.to_lowercase().starts_with("mt"))
        .collect();
    eprintln!(
        "<<< subset HVG & drop mt: {:.2} s",
        t.elapsed().as_secs_f64()
    );
    log.push(("subset HVG & drop mt".to_string(), t.elapsed().as_secs_f64()));

    let t = Instant::now();
    eprintln!(">>> convert X to CSR (for PCA)");
    ensure_x_csr_for_pca(&adata)?;
    eprintln!(
        "<<< convert X to CSR: {:.2} s",
        t.elapsed().as_secs_f64()
    );
    log.push(("convert X to CSR".to_string(), t.elapsed().as_secs_f64()));

    let t = Instant::now();
    eprintln!(">>> pca");
    let pca_res = run_pca_sparse_masked::<f64>(
        &adata.x(),
        Some(FeatureSelectionMethod::HighlyVariableSelection(combined_mask)),
        Some(true),
        Some(false),
        Some(params.n_pca_components),
        None,
        Some(42),
        Some(SVDMethod::Random {
            n_oversamples: 10,
            n_power_iterations: 4,
            normalizer: PowerIterationNormalizer::QR,
        }),
    )
    .map_err(|e| anyhow!("PCA: {e:?}"))?;
    let pca = pca_res.transformed;
    eprintln!("  PCA shape: {:?}", pca.shape());
    eprintln!("<<< pca: {:.2} s", t.elapsed().as_secs_f64());
    log.push(("pca".to_string(), t.elapsed().as_secs_f64()));

    let emb_umap = run_umap(&pca, params, &mut log)?;

    let n = emb_umap.nrows();
    let pca_f64 = ndarray::Array2::<f64>::from_shape_fn((pca.nrows(), pca.ncols()), |(i, j)| {
        pca[(i, j)]
    });
    let umap_f64 =
        ndarray::Array2::<f64>::from_shape_fn((n, 2), |(i, j)| emb_umap[(i, j)] as f64);

    axis_replace_array(
        &adata.obsm(),
        "X_pca",
        ArrayData::Array(DynArray::from(pca_f64)),
    )?;
    axis_replace_array(
        &adata.obsm(),
        "X_umap",
        ArrayData::Array(DynArray::from(umap_f64)),
    )?;

    let t = Instant::now();
    eprintln!(">>> write_h5ad {}", output.display());
    if output.exists() {
        std::fs::remove_file(output).map_err(|e| {
            anyhow!("cannot remove existing output {}: {e}", output.display())
        })?;
    }
    strip_supplemental_axis_arrays_for_h5_export(&adata)
        .context("strip obsp/varm/varp and non-embedding obsm for HDF5 export")?;
    let obs_safe = dataframe_hdf5_safe(adata.obs().get_data()).context("sanitize obs for HDF5")?;
    adata
        .obs()
        .set_data(obs_safe)
        .context("obs.set_data after sanitize")?;
    let var_safe = dataframe_hdf5_safe(adata.var().get_data()).context("sanitize var for HDF5")?;
    adata
        .var()
        .set_data(var_safe)
        .context("var.set_data after sanitize")?;
    clear_uns_for_hdf5_export(&adata).context("clear uns for HDF5")?;

    let tmp = temp_h5ad_path(output).context("temp output path for HDF5")?;
    if tmp.exists() {
        let _ = std::fs::remove_file(&tmp);
    }
    let write_result = (|| -> Result<()> {
        let written =
            convert_to_new_backed_h5(&adata, &tmp).context("convert_to_new_backed_h5 (write)")?;
        written
            .close()
            .context("close AnnData H5 (HDF5 finalize)")?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    write_result.context(
        "rust_preprocess: HDF5 export failed; partial temp file was removed if present",
    )?;
    std::fs::rename(&tmp, output).with_context(|| {
        format!(
            "rename temp HDF5 {:?} -> {:?}",
            tmp.display(),
            output.display()
        )
    })?;
    eprintln!("<<< write_h5ad: {:.2} s", t.elapsed().as_secs_f64());
    log.push(("write_h5ad".to_string(), t.elapsed().as_secs_f64()));

    let total: f64 = log.iter().map(|(_, s)| s).sum();
    eprintln!("rust_preprocess: TOTAL (sum of steps) {total:.2} s");
    for (name, dt) in &log {
        eprintln!("  {name}: {dt:.2} s");
    }

    Ok(output.to_path_buf())
}
