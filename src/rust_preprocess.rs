//! Rust-native Scanpy-style preprocessing: optional QC, **`normalize_total` (target_sum=10_000) +
//! `log1p` on `X`** when `X` looks like raw counts (aligned with `sc.pp.normalize_total` /
//! `sc.pp.log1p` via `single_rust`). Dense **`X`** is converted to **CSR `f64`** when needed because
//! `normalize_expression` row-sum helpers only support CSR/CSC. If `uns['log1p']` exists or **`_infer_x_is_log1p`-style**
//! heuristics match Scanpy’s embedded preprocess, **`normalize_total` / `log1p` are skipped** and
//! `X` is copied into `layers['normalized_count']` and `layers['log1p']` unchanged. Writable `.h5ad` files are patched before load to **drop `uns/**` datasets whose `encoding-type` is the literal `null`** (e.g. Scanpy `uns['log1p']['base']`), which **anndata-rs 0.6** cannot parse. If **`var` index** is digit-like (`"0"`…`"n"`) but symbols live in **`var` columns** (e.g. `feature_name`, `gene_symbols`), those are copied onto the variable index after load. Optional
//! **HVG** (dispersion-based when `n_vars > n_top_hvg`, else all non-MT genes) → **gene subset to that mask**
//! → PCA → UMAP → Leiden / MAGIC via [`RustPreprocessSteps`]. Subsetting runs **before** PCA and MAGIC so
//! saved objects only carry HVG columns in `X` and `layers`. HDF5 export (when an output path is provided)
//! writes `.h5ad` via `anndata-memory` + `AnnData::<H5>` with **`X` and all `layers`** coerced to CSR `f64` for on-disk layout.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anndata::backend::{AttributeOp, DataContainer, DatasetOp, GroupOp};
use anndata::data::array::DynArray;
use anndata::data::index::Interval;
use anndata::data::{ArrayConvert, DataFrameIndex, SelectInfoElem};
use anndata::data::{DynCscMatrix, DynCsrMatrix};
use anndata::{AnnData, AnnDataOp, ArrayData, AxisArraysOp, Backend, Readable};
use anndata_hdf5::{H5, H5File};
use anndata_memory::{IMAnnData, IMArrayElement, IMAxisArrays, load_h5ad_fast};
use anyhow::{Context, Result, anyhow, bail};
use instant_distance::{Builder, Hnsw, PointId, Search};
use leiden::leiden::Leiden;
use leiden::{Clustering, Graph, Network, SimpleClustering};
use magic_impute::{CsrF64, ImputeConfig, impute_magic_f32};
use nalgebra::{DMatrix, SymmetricEigen};
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::{CsrMatrix, SparseEntry};
use ndarray::{Array2, s};
use ndarray_umap::Array2 as Array2Umap;
use polars::prelude::{Column, DataFrame, DataType, NamedFrom, Series};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use sprs::{CsMatI, TriMatI};
use std::convert::TryInto;

use crate::betadata::obs_series_row_str;
use single_algebra::dimred::pca::{PowerIterationNormalizer, SVDMethod};
use single_rust::memory::processing::dimred::FeatureSelectionMethod;
use single_rust::memory::processing::dimred::pca::run_pca_sparse_masked;
use single_rust::memory::processing::filtering::{mark_filter_cells, mark_filter_genes};
use single_rust::memory::processing::{
    compute_highly_variable_genes, log1p_expression, normalize_expression,
};
use single_rust::shared::HVGParams;
use single_utilities::types::Direction;
use umap_rs::{
    EuclideanMetric, GraphParams, ManifoldParams, MetricType, OptimizationParams, Optimizer, Umap,
    UmapConfig,
};

pub type FuzzyGraph = CsMatI<f32, u32, usize>;

const INIT_NOISE_STD: f32 = 1e-4;

#[derive(Clone, Debug)]
pub struct RustPreprocessParams {
    /// `filter_cells(min_genes=…)` when QC is enabled.
    pub min_genes: u32,
    /// `filter_genes(min_cells=…)` when QC is enabled.
    pub min_cells: u32,
    /// `normalize_total(target_sum=…)` when `X` is raw counts.
    pub normalize_target_sum: u32,
    /// Max highly-variable genes when `n_vars` exceeds this; when `n_vars <= n_top_hvg`, dispersion
    /// ranking is skipped and **all non-MT genes** are used (nothing to trim to a smaller top-N).
    pub n_top_hvg: usize,
    pub n_pca_components: usize,
    pub pca_random_seed: u32,
    pub n_neighbors: usize,
    pub min_dist: f32,
    pub n_epochs: Option<usize>,
    pub ef_construction: usize,
    /// UMAP manifold `spread` (paired with `min_dist` in umap-rs).
    pub spread: f32,
    /// UMAP SGD learning rate passed to [`OptimizationParams::learning_rate`].
    pub umap_learning_rate: f32,
    /// Leiden resolution passed to [`leiden_labels_from_graph`].
    pub leiden_resolution: f64,
    pub leiden_max_iter: usize,
    /// MAGIC diffusion time `t` (Rust `magic-impute` path).
    pub magic_t: u32,
}

impl Default for RustPreprocessParams {
    fn default() -> Self {
        crate::config::PreprocessConfig::default().to_rust_preprocess_params()
    }
}

/// Reusable k-nearest neighbors for UMAP when only manifold or optimization parameters change.
#[derive(Clone, Debug)]
pub struct UmapLabKnnCache {
    pub knn_idx: Array2Umap<u32>,
    pub knn_dist: Array2Umap<f32>,
    pub n_neighbors: usize,
    pub ef_construction: usize,
    pub n_pca_components: usize,
}

impl UmapLabKnnCache {
    fn matches(&self, pca: &ndarray::Array2<f64>, params: &RustPreprocessParams) -> bool {
        self.knn_idx.nrows() == pca.nrows()
            && self.knn_idx.ncols() == params.n_neighbors
            && self.n_neighbors == params.n_neighbors
            && self.ef_construction == params.ef_construction
            && self.n_pca_components == params.n_pca_components
    }
}

/// umap-rs manifold calibration requires `min_dist <= spread`. Returns a pair that satisfies
/// this by taking `min_dist = min(min_dist, spread)` and `spread = max(spread, min_dist)` after
/// clamping away from zero.
pub fn clamp_umap_min_dist_spread(min_dist: f32, spread: f32) -> (f32, f32) {
    let md = min_dist.max(1e-6f32);
    let sp = spread.max(1e-6f32);
    let md2 = md.min(sp);
    let sp2 = sp.max(md);
    (md2, sp2)
}

#[derive(Clone, Debug)]
pub struct RustPreprocessSteps {
    pub qc_filter: bool,
    pub normalize_log1p: bool,
    pub hvg_pca: bool,
    pub run_umap_and_graph: bool,
    pub write_leiden: bool,
    pub run_magic_impute: bool,
}

impl RustPreprocessSteps {
    pub const FULL: Self = Self {
        qc_filter: true,
        normalize_log1p: true,
        hvg_pca: true,
        run_umap_and_graph: true,
        write_leiden: false,
        run_magic_impute: true,
    };

    /// Training auto-prep when `normalized_count` and `imputed_count` exist but neither `leiden` nor `cell_type`.
    /// No QC subsetting (row alignment with existing layers); HVG (or all non-MT if `n_vars <= n_top_hvg`)
    /// → **gene subset** → PCA → UMAP → Leiden → `cell_type` from labels.
    pub const TRAINING_LAYERS_LEIDEN_ANNOTATE: Self = Self {
        qc_filter: false,
        normalize_log1p: true,
        hvg_pca: true,
        run_umap_and_graph: true,
        write_leiden: true,
        run_magic_impute: false,
    };

    pub fn from_convenience_flags(umap: bool, leiden: bool, rust_magic: bool) -> Self {
        let graph = umap || leiden || rust_magic;
        Self {
            qc_filter: true,
            normalize_log1p: true,
            hvg_pca: graph,
            run_umap_and_graph: graph,
            write_leiden: leiden,
            run_magic_impute: rust_magic,
        }
    }

    /// Normalize → HVG → PCA only (no UMAP / Leiden / MAGIC). Used by the UMAP lab UI to build `X_pca` once.
    pub const UMAP_LAB_PCA_ONLY: Self = Self {
        qc_filter: false,
        normalize_log1p: true,
        hvg_pca: true,
        run_umap_and_graph: false,
        write_leiden: false,
        run_magic_impute: false,
    };
}

#[derive(Clone)]
struct PcaVec(Vec<f32>);

impl instant_distance::Point for PcaVec {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
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

/// Env-driven escape hatch: gene symbols listed in **`SPACETRAVLR_FORCE_KEEP_GENES`** (comma-separated,
/// case-insensitive, optional whitespace) are OR-ed into the HVG ∩ ¬MT mask before subsetting so
/// downstream callers (notably **`spacetravlr --verify`**) can guarantee that target symbols survive
/// dispersion HVG even when they fall below the top-N. Returns the count actually flipped from
/// `false` to `true`.
fn apply_force_keep_genes_env(adata: &IMAnnData, mask: &mut [bool]) -> usize {
    let raw = match std::env::var("SPACETRAVLR_FORCE_KEEP_GENES") {
        Ok(v) => v,
        Err(_) => return 0,
    };
    let wanted: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if wanted.is_empty() {
        return 0;
    }
    let var_names = adata.var_names();
    let mut added = 0usize;
    let mut hit_names: Vec<String> = Vec::new();
    for (i, name) in var_names.iter().enumerate() {
        if i >= mask.len() {
            break;
        }
        if mask[i] {
            continue;
        }
        let lc = name.to_ascii_lowercase();
        if wanted.iter().any(|w| w == &lc) {
            mask[i] = true;
            added += 1;
            hit_names.push(name.clone());
        }
    }
    if added > 0 {
        eprintln!(
            "rust_preprocess: SPACETRAVLR_FORCE_KEEP_GENES preserved {added} gene(s) through HVG ∩ ¬MT subset: {}",
            hit_names.join(", ")
        );
    } else {
        eprintln!(
            "rust_preprocess: SPACETRAVLR_FORCE_KEEP_GENES set ({} symbol(s)) but no new matches in var_names (already kept or absent)",
            wanted.len()
        );
    }
    added
}

const VAR_SYMBOL_COLUMN_CANDIDATES: &[&str] = &[
    "gene_symbols",
    "gene_symbol",
    "feature_name",
    "feature_names",
    "names",
    "name",
    "symbol",
    "genesymbol",
    "gene_name",
    "gene",
    "genes",
    "gene_ids",
];

fn var_name_is_digit_placeholder(s: &str) -> bool {
    let t = s.trim();
    !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())
}

fn var_names_placeholder_ratio(names: &[String]) -> f64 {
    let n = names.len();
    if n == 0 {
        return 0.0;
    }
    names
        .iter()
        .filter(|s| var_name_is_digit_placeholder(s))
        .count() as f64
        / n as f64
}

fn find_var_df_column_ci(df: &DataFrame, want: &str) -> Option<String> {
    df.get_column_names()
        .into_iter()
        .find(|c| c.as_str().eq_ignore_ascii_case(want))
        .map(|c| c.to_string())
}

fn series_non_digit_label_ratio(series: &Series) -> Result<f64> {
    let n = series.len();
    if n == 0 {
        return Ok(0.0);
    }
    let mut good = 0usize;
    for i in 0..n {
        let s = obs_series_row_str(series, i)?.trim().to_string();
        if s.is_empty() || s.eq_ignore_ascii_case("nan") || s.eq_ignore_ascii_case("null") {
            continue;
        }
        if var_name_is_digit_placeholder(&s) {
            continue;
        }
        good += 1;
    }
    Ok(good as f64 / n as f64)
}

fn pick_var_symbol_column(df: &DataFrame) -> Option<(String, f64)> {
    let mut best: Option<(String, f64)> = None;
    for cand in VAR_SYMBOL_COLUMN_CANDIDATES {
        let Some(col) = find_var_df_column_ci(df, cand) else {
            continue;
        };
        let Ok(col_handle) = df.column(&col) else {
            continue;
        };
        let series = col_handle.as_materialized_series();
        let Ok(score) = series_non_digit_label_ratio(series) else {
            continue;
        };
        if score < 0.5 {
            continue;
        }
        match &best {
            None => best = Some((col, score)),
            Some((_, s0)) if score > *s0 => best = Some((col, score)),
            _ => {}
        }
    }
    best
}

fn dedupe_var_names_scanpy_style(names: Vec<String>) -> Vec<String> {
    let mut counts = HashMap::<String, usize>::new();
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        let c = counts.entry(n.clone()).or_insert(0);
        let label = if *c == 0 {
            n.clone()
        } else {
            format!("{}-{}", n, *c)
        };
        *c += 1;
        out.push(label);
    }
    out
}

/// When `var_names` look like `"0"`…`"n"` but symbols live in `var` (e.g. `feature_name`), return
/// deduplicated names for the AnnData variable index.
fn restore_var_names_if_placeholder(names: &[String], df: &DataFrame) -> Result<Option<Vec<String>>> {
    const MIN_PH: f64 = 0.9;
    if var_names_placeholder_ratio(names) < MIN_PH {
        return Ok(None);
    }
    let Some((col_name, score)) = pick_var_symbol_column(df) else {
        eprintln!(
            "rust_preprocess: var index ~{:.0}% digit-like; no suitable symbol column in `var`",
            var_names_placeholder_ratio(names) * 100.0
        );
        return Ok(None);
    };
    let series = df
        .column(&col_name)
        .with_context(|| format!("var column {col_name}"))?
        .as_materialized_series();
    let mut new_names = Vec::with_capacity(names.len());
    for i in 0..names.len() {
        new_names.push(obs_series_row_str(series, i)?.trim().to_string());
    }
    if new_names.len() != names.len() {
        bail!("rust_preprocess: symbol column row count mismatch");
    }
    if var_names_placeholder_ratio(&new_names) >= MIN_PH {
        eprintln!(
            "rust_preprocess: skipped var restore from var[{col_name}] (still mostly digit-like)"
        );
        return Ok(None);
    }
    let new_names = dedupe_var_names_scanpy_style(new_names);
    eprintln!(
        "rust_preprocess: var index was digit-like ({:.0}%); restored from var[{col_name}] ({:.0}% non-digit labels)",
        var_names_placeholder_ratio(names) * 100.0,
        score * 100.0
    );
    Ok(Some(new_names))
}

fn maybe_restore_var_names_in_memory(adata: &IMAnnData) -> Result<()> {
    let cur = adata.var_names();
    let df = adata.var().get_data();
    if let Some(names) = restore_var_names_if_placeholder(&cur, &df)? {
        let idx: DataFrameIndex = names.into();
        adata
            .var()
            .set_index(idx)
            .context("var.set_index after symbol restore")?;
    }
    Ok(())
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
    n_neighbors: usize,
    hnsw: &Hnsw<PcaVec>,
    pids: &[PointId],
    pid_to_orig: &[u32],
    ef_search_bridge: usize,
) {
    let n = knn_idx.nrows();

    let sort_row = |knn_idx: &mut Array2Umap<u32>, knn_dist: &mut Array2Umap<f32>, row: usize| {
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
                    hnsw.search(&hnsw[pids[p]], search)
                        .take(ef_search_bridge)
                        .map(|item| {
                            (
                                pid_to_orig[item.pid.into_inner() as usize],
                                item.distance.sqrt(),
                            )
                        })
                        .collect()
                });
                for (q_orig, d) in results {
                    if comp[q_orig as usize] == main_id {
                        if best.as_ref().is_none_or(|(_, _, bd)| d < *bd) {
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
    points: Vec<PcaVec>,
    n_neighbors: usize,
    ef_construction: usize,
    log: &mut Vec<(String, f64)>,
) -> (Array2Umap<u32>, Array2Umap<f32>) {
    let n = points.len();
    let ef_search = (n_neighbors * 3).max(50);
    let ef_bridge = (n_neighbors * 20).max(300).min(n);

    eprintln!("  building HNSW (ef_construction={ef_construction}, ef_search={ef_search})…");
    let t_build = Instant::now();
    let (hnsw, pids) = Builder::default()
        .ef_construction(ef_construction)
        .ef_search(ef_search)
        .seed(42)
        .build_hnsw(points);
    let dt_build = t_build.elapsed().as_secs_f64();
    eprintln!("  HNSW build: {dt_build:.2} s");
    log.push(("umap KNN: HNSW build".to_string(), dt_build));

    let mut pid_to_orig = vec![0u32; n];
    for (orig, pid) in pids.iter().enumerate() {
        pid_to_orig[pid.into_inner() as usize] = orig as u32;
    }

    let search_k = n_neighbors + 8;
    eprintln!("  querying {n} points (HNSW ef_search={ef_search})…");
    let t_query = Instant::now();
    let mut rows: Vec<(usize, Vec<u32>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let results: Vec<(u32, f32)> = HNSW_SEARCH.with_borrow_mut(|search| {
                hnsw.search(&hnsw[pids[i]], search)
                    .take(search_k)
                    .map(|item| {
                        (
                            pid_to_orig[item.pid.into_inner() as usize],
                            item.distance.sqrt(),
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
    let dt_query = t_query.elapsed().as_secs_f64();
    eprintln!("  HNSW query all points: {dt_query:.2} s");
    log.push(("umap KNN: HNSW query".to_string(), dt_query));
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
    let t_bridge = Instant::now();
    bridge_knn_components(
        &mut idx,
        &mut dist,
        n_neighbors,
        &hnsw,
        &pids,
        &pid_to_orig,
        ef_bridge,
    );
    let dt_bridge = t_bridge.elapsed().as_secs_f64();
    eprintln!("  KNN connectivity / bridge: {dt_bridge:.2} s");
    log.push(("umap KNN: connectivity".to_string(), dt_bridge));

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
            let dot: f32 = cols[i].iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
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

pub fn run_umap_on_pca(
    pca: &ndarray::Array2<f64>,
    params: &RustPreprocessParams,
    log: &mut Vec<(String, f64)>,
    knn_cache_in: Option<&UmapLabKnnCache>,
) -> Result<(Array2Umap<f32>, FuzzyGraph, UmapLabKnnCache)> {
    let n = pca.nrows();
    let dim = params.n_pca_components;
    if pca.ncols() < dim {
        anyhow::bail!(
            "PCA has {} columns but n_pca_components is {}",
            pca.ncols(),
            dim
        );
    }

    let (knn_idx, knn_dist, knn_cache) = match knn_cache_in {
        Some(cache) if cache.matches(pca, params) => {
            let t0 = Instant::now();
            eprintln!(">>> umap KNN (cached HNSW graph)");
            eprintln!("<<< umap KNN (cached): {:.3} s", t0.elapsed().as_secs_f64());
            log.push(("umap KNN (cached)".to_string(), t0.elapsed().as_secs_f64()));
            (cache.knn_idx.clone(), cache.knn_dist.clone(), cache.clone())
        }
        _ => {
            let t0 = Instant::now();
            eprintln!(">>> umap KNN (HNSW)");
            let t_pts = Instant::now();
            let points = pca_to_points_f32(pca, dim);
            log.push((
                "umap KNN: PCA→f32 points".to_string(),
                t_pts.elapsed().as_secs_f64(),
            ));
            let mut knn_log = Vec::new();
            let (knn_idx, knn_dist) = knn_indices_dists(
                points,
                params.n_neighbors,
                params.ef_construction,
                &mut knn_log,
            );
            log.extend(knn_log);
            let dt_total = t0.elapsed().as_secs_f64();
            eprintln!("<<< umap KNN (HNSW): {dt_total:.2} s");
            log.push(("umap KNN (HNSW total)".to_string(), dt_total));
            let knn_cache = UmapLabKnnCache {
                knn_idx: knn_idx.clone(),
                knn_dist: knn_dist.clone(),
                n_neighbors: params.n_neighbors,
                ef_construction: params.ef_construction,
                n_pca_components: params.n_pca_components,
            };
            (knn_idx, knn_dist, knn_cache)
        }
    };

    let mut data_vec = Vec::with_capacity(n * dim);
    for row in pca.outer_iter() {
        for j in 0..dim {
            data_vec.push(*row.get(j).unwrap_or(&0.0) as f32);
        }
    }
    let data = Array2Umap::from_shape_vec((n, dim), data_vec)
        .map_err(|e| anyhow!("UMAP data shape: {e}"))?;

    let n_epochs = params
        .n_epochs
        .unwrap_or(if n <= 10_000 { 500 } else { 200 });

    let (min_dist, spread) = clamp_umap_min_dist_spread(params.min_dist, params.spread);

    let config = UmapConfig {
        n_components: 2,
        manifold: ManifoldParams {
            min_dist,
            spread,
            ..Default::default()
        },
        graph: GraphParams {
            n_neighbors: params.n_neighbors,
            symmetrize: true,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: Some(n_epochs),
            learning_rate: params.umap_learning_rate,
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
    let fuzzy_graph = manifold.graph().clone();
    eprintln!(
        "<<< umap spectral init: {:.2} s",
        t2.elapsed().as_secs_f64()
    );
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
    log.push((
        "umap optimize (umap-rs)".to_string(),
        t3.elapsed().as_secs_f64(),
    ));

    Ok((fitted.into_embedding(), fuzzy_graph, knn_cache))
}

fn im_obsm_dense_matrix_f64(adata: &IMAnnData, key: &str) -> Result<Option<Array2<f64>>> {
    let elem = match adata.obsm().get_array(key) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };
    let data = match elem.get_data() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    match data {
        ArrayData::Array(d) => {
            let arr_f64: Result<Array2<f64>, _> = d.clone().try_convert();
            if let Ok(a) = arr_f64 {
                return Ok(Some(a));
            }
            let a32: Array2<f32> = d
                .try_convert()
                .map_err(|e| anyhow!("obsm[{key}] dense matrix: {e}"))?;
            Ok(Some(a32.mapv(|v| v as f64)))
        }
        _ => Ok(None),
    }
}

fn pca_from_im_after_preprocess(adata: &IMAnnData) -> Result<Array2<f64>> {
    let elem = adata
        .obsm()
        .get_array("X_pca")
        .context("obsm['X_pca'] missing after preprocess")?;
    let data = elem.get_data().context("X_pca data")?;
    let arr: Array2<f64> = match data {
        ArrayData::Array(d) => d
            .try_convert()
            .map_err(|e| anyhow!("X_pca to f64 matrix: {e}"))?,
        _ => bail!("X_pca must be a dense array"),
    };
    Ok(arr)
}

fn umap_lab_color_labels_from_obs(adata: &IMAnnData) -> Result<(Option<String>, Vec<String>)> {
    let n = adata.n_obs();
    let obs = adata.obs().get_data();
    for col in ["leiden", "cell_type", "louvain"] {
        if let Some(v) = obs_column_as_strings(&obs, col)? {
            if v.len() == n && n > 0 {
                return Ok((Some(col.to_string()), v));
            }
        }
    }
    Ok((None, Vec::new()))
}

fn umap_lab_obs_meta(adata: &IMAnnData) -> (Vec<String>, Vec<String>) {
    let obs = adata.obs().get_data();
    let obs_columns: Vec<String> = obs
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let obs_names: Vec<String> = adata.obs_names().to_vec();
    (obs_names, obs_columns)
}

pub fn umap_lab_read_obs_column(path: &Path, column: &str) -> Result<Vec<String>> {
    prepare_h5ad_path_for_anndata_memory_load(path);
    let h5 = H5::open(path).context("H5::open for obs column")?;
    let obs_container = DataContainer::open(&h5, "obs").context("open obs")?;
    let obs_df: DataFrame = ArrayData::read(&obs_container)?
        .try_into()
        .map_err(|e| anyhow!("obs DataFrame: {e}"))?;
    obs_column_as_strings(&obs_df, column)?
        .ok_or_else(|| anyhow!("obs column {column:?} not found"))
}

/// PCA matrix and optional per-cell labels (for coloring) after loading an `.h5ad`.
#[derive(Clone, Debug)]
pub struct UmapLabLoaded {
    pub pca: Array2<f64>,
    pub color_column: Option<String>,
    pub color_labels: Vec<String>,
    pub obs_names: Vec<String>,
    pub obs_columns: Vec<String>,
    pub var_names: Vec<String>,
    /// When `obsm['spatial']` (or `X_spatial` / `spatial_loc`) exists with ≥2 columns and `n_obs` rows.
    pub spatial: Option<(String, Vec<f32>, Vec<f32>)>,
}

/// Load PCA from `obsm['X_pca']` when present and valid; otherwise runs
/// [`rust_preprocess_h5ad_to_memory`] with [`RustPreprocessSteps::UMAP_LAB_PCA_ONLY`], which
/// writes `obsm['X_pca']` whenever HVG+PCA runs.
/// The returned `pca` contains all available components (use `params.n_pca_components` when running UMAP).
pub fn umap_lab_load_pca_session(
    input: &Path,
    params: &RustPreprocessParams,
) -> Result<UmapLabLoaded> {
    if let Some(loaded) = umap_lab_try_headless_umap_loaded(input)? {
        return Ok(loaded);
    }
    prepare_h5ad_path_for_anndata_memory_load(input);
    let adata = load_h5ad_fast(input).map_err(|e| {
        anyhow!(
            "full in-memory .h5ad load failed (often OOM on huge dense X). \
             If `obsm['X_pca']` exists, umap_lab uses a headless path; otherwise convert X to CSR in Python or add X_pca. \
             Caused by: {:#}",
            e
        )
    })?;
    maybe_restore_var_names_in_memory(&adata).context("restore var_names from var columns")?;
    let n_obs = adata.n_obs();
    let pca_opt = im_obsm_dense_matrix_f64(&adata, "X_pca")?;
    let (pca, (color_column, color_labels), (obs_names, obs_columns)) = if let Some(pca) = pca_opt {
        if pca.nrows() != n_obs {
            bail!(
                "obsm['X_pca'] rows {} do not match n_obs {}",
                pca.nrows(),
                n_obs
            );
        }
        if pca.ncols() < 2 {
            bail!(
                "obsm['X_pca'] must have at least 2 columns (got {})",
                pca.ncols()
            );
        }
        eprintln!(
            "umap_lab: using existing obsm['X_pca'] ({}×{})",
            pca.nrows(),
            pca.ncols()
        );
        let colors = umap_lab_color_labels_from_obs(&adata)?;
        let meta = umap_lab_obs_meta(&adata);
        (pca, colors, meta)
    } else {
        drop(adata);
        eprintln!("umap_lab: no usable X_pca; running normalization + HVG + PCA …");
        let adata2 =
            rust_preprocess_h5ad_to_memory(input, params, &RustPreprocessSteps::UMAP_LAB_PCA_ONLY)?;
        let pca = pca_from_im_after_preprocess(&adata2)?;
        let colors = umap_lab_color_labels_from_obs(&adata2)?;
        let meta = umap_lab_obs_meta(&adata2);
        (pca, colors, meta)
    };
    let h5_meta = H5::open(input).context("H5::open (var names / spatial)")?;
    let (_, _, _, var_names) = umap_lab_h5_read_obs_var_dataframes(&h5_meta)?;
    let spatial = umap_lab_h5_read_obsm_spatial_xy(&h5_meta, pca.nrows())?;
    Ok(UmapLabLoaded {
        pca,
        color_column,
        color_labels,
        obs_names,
        obs_columns,
        var_names,
        spatial,
    })
}

/// Run UMAP on `pca` (same umap-rs + HNSW path as [`rust_preprocess_h5ad_to_memory`]).
///
/// Pass `knn_cache_in` from a previous run when only manifold or optimization parameters changed;
/// the neighbor graph is reused without rebuilding HNSW.
pub fn umap_lab_run_embedding(
    pca: &Array2<f64>,
    params: &RustPreprocessParams,
    knn_cache_in: Option<&UmapLabKnnCache>,
) -> Result<(Array2<f32>, FuzzyGraph, Vec<(String, f64)>, UmapLabKnnCache)> {
    let mut log = Vec::new();
    let (emb_umap, graph, knn_cache) = run_umap_on_pca(pca, params, &mut log, knn_cache_in)?;
    let n = emb_umap.nrows();
    let m = emb_umap.ncols();
    let emb = Array2::<f32>::from_shape_fn((n, m), |(i, j)| emb_umap[(i, j)]);
    Ok((emb, graph, log, knn_cache))
}

fn csr_col_to_f32(csr: &CsrMatrix<f64>, col: usize, n_obs: usize) -> Vec<f32> {
    (0..n_obs)
        .map(|i| match csr.get_entry(i, col) {
            Some(SparseEntry::NonZero(v)) => *v as f32,
            Some(SparseEntry::Zero) | None => 0.0,
        })
        .collect()
}

fn csr_col_f32_to_f32(csr: &CsrMatrix<f32>, col: usize, n_obs: usize) -> Vec<f32> {
    (0..n_obs)
        .map(|i| match csr.get_entry(i, col) {
            Some(SparseEntry::NonZero(v)) => *v,
            Some(SparseEntry::Zero) | None => 0.0,
        })
        .collect()
}

fn x_column_as_f32(x: &ArrayData, col: usize, n_obs: usize, n_vars: usize) -> Result<Vec<f32>> {
    if col >= n_vars {
        bail!("gene column index {col} out of range (n_vars={n_vars})");
    }
    Ok(match x {
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => csr_col_to_f32(m, col, n_obs),
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => csr_col_f32_to_f32(m, col, n_obs),
        ArrayData::CscMatrix(d) => {
            let csr = csc_dyn_to_csr_f64(d.clone())?;
            csr_col_to_f32(&csr, col, n_obs)
        }
        ArrayData::CsrNonCanonical(non) => {
            let d = non
                .clone()
                .canonicalize()
                .map_err(|e| anyhow!("non-canonical CSR X: {e:?}"))?;
            let csr = csr_dyn_to_csr_f64(d)?;
            csr_col_to_f32(&csr, col, n_obs)
        }
        ArrayData::Array(d) => {
            let dense: Array2<f64> = d.clone().try_convert().context("X dense to f64")?;
            if dense.nrows() != n_obs || dense.ncols() != n_vars {
                bail!(
                    "dense X shape {}×{} does not match n_obs×n_vars {}×{}",
                    dense.nrows(),
                    dense.ncols(),
                    n_obs,
                    n_vars
                );
            }
            dense.column(col).iter().map(|v| *v as f32).collect()
        }
        _ => bail!("X must be CSR, CSC, or dense matrix to color by gene"),
    })
}

fn gene_display_bounds(values: &[f32]) -> (f32, f32) {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (0.0, 1.0);
    }
    if v.len() == 1 {
        let a = v[0];
        let pad = a.abs() * 0.05 + 0.05;
        return (a - pad, a + pad);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let lo_i = ((n - 1) as f64 * 0.02).round() as usize;
    let hi_i = ((n - 1) as f64 * 0.98).round() as usize;
    let lo_i = lo_i.min(n - 1);
    let hi_i = hi_i.min(n - 1).max(lo_i);
    let mut lo = v[lo_i];
    let mut hi = v[hi_i];
    if !(lo < hi) {
        lo = v[0];
        hi = v[n - 1];
        if lo >= hi {
            let pad = lo.abs() * 0.05 + 0.05;
            hi = lo + pad;
        }
    }
    (lo, hi)
}

/// Adapted from anndata-memory `read_dataframe_index` (MIT) for umap_lab HDF5 headless loads.
fn umap_lab_h5_read_dataframe_index(container: &DataContainer<H5>) -> Result<DataFrameIndex> {
    let index_name: String = container.get_attr("_index")?;
    let dataset = container.as_group()?.open_dataset(&index_name)?;
    let index_ty = match dataset.get_attr::<String>("index_type") {
        Ok(s) => s,
        Err(_) => "list".to_string(),
    };
    match index_ty.as_str() {
        "list" => {
            let data = dataset.read_array()?;
            let mut index: DataFrameIndex = data.to_vec().into();
            index.index_name = index_name;
            Ok(index)
        }
        "intervals" => {
            let keys: Vec<String> = dataset.get_attr("names")?;
            let values: Vec<Vec<u64>> = dataset.get_attr("intervals")?;
            Ok(keys
                .into_iter()
                .zip(values.into_iter().map(|row| Interval {
                    start: row[0] as usize,
                    end: row[1] as usize,
                    size: row[2] as usize,
                    step: row[3] as usize,
                }))
                .collect())
        }
        "range" => {
            let start: u64 = dataset.get_attr("start")?;
            let end: u64 = dataset.get_attr("end")?;
            Ok((start as usize..end as usize).into())
        }
        x => bail!("Unknown obs/var index type: {}", x),
    }
}

fn umap_lab_h5_read_obs_var_dataframes(
    h5: &H5File,
) -> Result<(DataFrame, Vec<String>, DataFrame, Vec<String>)> {
    let obs_container = DataContainer::open(h5, "obs").context("open obs")?;
    let obs_df: DataFrame = ArrayData::read(&obs_container)?
        .try_into()
        .map_err(|e| anyhow!("obs DataFrame: {e}"))?;
    let obs_names = umap_lab_h5_read_dataframe_index(&obs_container)?.into_vec();

    let var_container = DataContainer::open(h5, "var").context("open var")?;
    let var_df: DataFrame = ArrayData::read(&var_container)?
        .try_into()
        .map_err(|e| anyhow!("var DataFrame: {e}"))?;
    let mut var_names = umap_lab_h5_read_dataframe_index(&var_container)?.into_vec();
    if let Some(new_names) = restore_var_names_if_placeholder(&var_names, &var_df)? {
        var_names = new_names;
    }

    Ok((obs_df, obs_names, var_df, var_names))
}

fn umap_lab_h5_read_obsm_x_pca_matrix(h5: &H5File) -> Result<Option<Array2<f64>>> {
    if !h5.exists("obsm")? {
        return Ok(None);
    }
    let obsm = h5.open_group("obsm")?;
    let names = obsm.list()?;
    if !names.iter().any(|n| n == "X_pca") {
        return Ok(None);
    }
    let cont = DataContainer::open(&obsm, "X_pca").context("open obsm/X_pca")?;
    let data = ArrayData::read(&cont)?;
    match data {
        ArrayData::Array(d) => match ArrayConvert::<Array2<f64>>::try_convert(d.clone()) {
            Ok(a) => Ok(Some(a)),
            Err(_) => {
                let a32: Array2<f32> = ArrayConvert::try_convert(d)
                    .map_err(|e| anyhow!("obsm['X_pca'] to f32 matrix: {e}"))?;
                Ok(Some(a32.mapv(|v| v as f64)))
            }
        },
        _ => Ok(None),
    }
}

fn umap_lab_h5_read_obsm_spatial_xy(
    h5: &H5File,
    n_obs: usize,
) -> Result<Option<(String, Vec<f32>, Vec<f32>)>> {
    if !h5.exists("obsm")? {
        return Ok(None);
    }
    let obsm = h5.open_group("obsm")?;
    for key in ["spatial", "X_spatial", "spatial_loc"] {
        if !obsm.exists(key)? {
            continue;
        }
        let cont = match DataContainer::open(&obsm, key) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let data = match ArrayData::read(&cont) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let arr = match data {
            ArrayData::Array(d) => {
                if let Ok(a) = ArrayConvert::<Array2<f64>>::try_convert(d.clone()) {
                    a
                } else if let Ok(a32) = ArrayConvert::<Array2<f32>>::try_convert(d) {
                    a32.mapv(|v| v as f64)
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        if arr.nrows() != n_obs || arr.ncols() < 2 {
            continue;
        }
        let sx: Vec<f32> = (0..n_obs).map(|i| arr[[i, 0]] as f32).collect();
        let sy: Vec<f32> = (0..n_obs).map(|i| arr[[i, 1]] as f32).collect();
        return Ok(Some((key.to_string(), sx, sy)));
    }
    Ok(None)
}

fn umap_lab_try_headless_umap_loaded(path: &Path) -> Result<Option<UmapLabLoaded>> {
    let h5 = H5::open(path).context("H5::open (headless umap_lab)")?;
    let Some(pca) = umap_lab_h5_read_obsm_x_pca_matrix(&h5)? else {
        return Ok(None);
    };
    let (obs_df, obs_names, var_df, var_names) = umap_lab_h5_read_obs_var_dataframes(&h5)?;
    let n_obs = obs_names.len();
    if pca.nrows() != n_obs || pca.ncols() < 2 {
        return Ok(None);
    }
    let n_vars = var_names.len();
    if n_vars == 0 {
        return Ok(None);
    }
    let obs_columns: Vec<String> = obs_df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let coo = CooMatrix::new(n_obs, n_vars);
    let csr = CsrMatrix::from(&coo);
    let x_data = ArrayData::CsrMatrix(DynCsrMatrix::F64(csr));
    let adata =
        IMAnnData::new_extended(x_data, obs_names.clone(), var_names.clone(), obs_df, var_df)
            .context("IMAnnData::new_extended (placeholder X)")?;
    let colors = umap_lab_color_labels_from_obs(&adata)?;
    eprintln!(
        "umap_lab: headless HDF5 load using obsm['X_pca'] ({}×{}) — skipped materializing full X",
        pca.nrows(),
        pca.ncols()
    );
    let spatial = umap_lab_h5_read_obsm_spatial_xy(&h5, n_obs)?;
    Ok(Some(UmapLabLoaded {
        pca,
        color_column: colors.0,
        color_labels: colors.1,
        obs_names,
        obs_columns,
        var_names,
        spatial,
    }))
}

fn umap_lab_h5_root_x_is_dense_dataset(path: &Path) -> Result<bool> {
    use hdf5_metno::{File as MetH5File, LocationType};
    let f = MetH5File::open(path)?;
    let t = f
        .loc_type_by_name("X")
        .map_err(|e| anyhow!("inspect X link: {e}"))?;
    Ok(t == LocationType::Dataset)
}

fn umap_lab_h5_root_x_is_sparse_group(path: &Path) -> Result<bool> {
    use hdf5_metno::{File as MetH5File, LocationType};
    let f = MetH5File::open(path)?;
    let t = f
        .loc_type_by_name("X")
        .map_err(|e| anyhow!("inspect X link: {e}"))?;
    Ok(t == LocationType::Group)
}

fn h5_read_1d_usize(ds: &hdf5_metno::Dataset) -> Result<Vec<usize>> {
    let sh = ds.shape();
    anyhow::ensure!(sh.len() == 1, "expected 1d dataset, got shape {:?}", sh);
    if let Ok(a) = ds.read_1d::<i64>() {
        return Ok(a.iter().map(|&v| v.max(0) as usize).collect());
    }
    if let Ok(a) = ds.read_1d::<u64>() {
        return Ok(a.iter().map(|&v| v as usize).collect());
    }
    if let Ok(a) = ds.read_1d::<i32>() {
        return Ok(a.iter().map(|&v| v.max(0) as usize).collect());
    }
    if let Ok(a) = ds.read_1d::<u32>() {
        return Ok(a.iter().map(|&v| v as usize).collect());
    }
    bail!("unsupported integer dtype for sparse X indptr/indices")
}

fn h5_read_1d_f64_as_vec(ds: &hdf5_metno::Dataset) -> Result<Vec<f64>> {
    let sh = ds.shape();
    anyhow::ensure!(sh.len() == 1, "expected 1d data array");
    if let Ok(a) = ds.read_1d::<f64>() {
        return Ok(a.to_vec());
    }
    if let Ok(a) = ds.read_1d::<f32>() {
        return Ok(a.iter().map(|&v| v as f64).collect());
    }
    if let Ok(a) = ds.read_1d::<i32>() {
        return Ok(a.iter().map(|&v| v as f64).collect());
    }
    if let Ok(a) = ds.read_1d::<i64>() {
        return Ok(a.iter().map(|&v| v as f64).collect());
    }
    bail!("unsupported dtype for sparse X data")
}

/// Read one gene column from on-disk CSR/CSC `X` without requiring canonical sorted lanes
/// (avoids `Minor indices are not monotonically increasing` when loading the full AnnData).
fn umap_lab_h5_sparse_x_column_f32(
    path: &Path,
    col: usize,
    n_obs: usize,
    n_vars: usize,
) -> Result<Vec<f32>> {
    use hdf5_metno::{File as H5F, LocationType};
    let f = H5F::open(path)?;
    anyhow::ensure!(
        f.loc_type_by_name("X")? == LocationType::Group,
        "X is not an HDF5 group"
    );
    let g = f.group("X")?;
    let shape_a = g
        .attr("shape")
        .context("sparse X: missing shape attribute")?;
    let (nr, nc) = if let Ok(v) = shape_a.read_1d::<u64>() {
        anyhow::ensure!(v.len() >= 2, "sparse X shape attr");
        (v[0] as usize, v[1] as usize)
    } else if let Ok(v) = shape_a.read_1d::<i64>() {
        anyhow::ensure!(v.len() >= 2, "sparse X shape attr");
        (v[0] as usize, v[1] as usize)
    } else if let Ok(v) = shape_a.read_1d::<u32>() {
        anyhow::ensure!(v.len() >= 2, "sparse X shape attr");
        (v[0] as usize, v[1] as usize)
    } else {
        bail!("sparse X: unsupported shape attribute dtype");
    };
    anyhow::ensure!(
        nr == n_obs && nc == n_vars,
        "sparse X shape {}×{} does not match obs×var {}×{}",
        nr,
        nc,
        n_obs,
        n_vars
    );
    anyhow::ensure!(
        col < n_vars,
        "gene column {col} out of range (n_vars={n_vars})"
    );

    let indptr = h5_read_1d_usize(&g.dataset("indptr").context("sparse X indptr")?)?;
    let indices = h5_read_1d_usize(&g.dataset("indices").context("sparse X indices")?)?;
    let data = h5_read_1d_f64_as_vec(&g.dataset("data").context("sparse X data")?)?;
    anyhow::ensure!(
        indices.len() == data.len(),
        "sparse X indices/data length mismatch"
    );

    let enc = g
        .attr("encoding-type")
        .ok()
        .and_then(|a| h5ad_attr_encoding_type_string(&a))
        .unwrap_or_default();

    let is_csr = enc == "csr_matrix" || (enc.is_empty() && indptr.len() == n_obs + 1);
    let is_csc = enc == "csc_matrix" || (enc.is_empty() && indptr.len() == n_vars + 1);
    if is_csr {
        anyhow::ensure!(indptr.len() == n_obs + 1, "csr indptr len");
        let mut out = vec![0f32; n_obs];
        for row in 0..n_obs {
            let s = indptr[row];
            let e = indptr[row + 1];
            let mut acc = 0f32;
            for k in s..e {
                if indices[k] == col {
                    acc += data[k] as f32;
                }
            }
            out[row] = acc;
        }
        return Ok(out);
    }
    if is_csc {
        anyhow::ensure!(indptr.len() == n_vars + 1, "csc indptr len");
        let mut out = vec![0f32; n_obs];
        let s = indptr[col];
        let e = indptr[col + 1];
        for k in s..e {
            let row = indices[k];
            if row < n_obs {
                out[row] += data[k] as f32;
            }
        }
        return Ok(out);
    }
    bail!(
        "X sparse group: encoding-type {enc:?}, indptr len {} (expected csr n_obs+1 or csc n_vars+1)",
        indptr.len()
    )
}

fn h5ad_attr_encoding_type_string(attr: &hdf5_metno::Attribute) -> Option<String> {
    use hdf5_metno::types::{VarLenAscii, VarLenUnicode};
    if let Ok(v) = attr.read_scalar::<VarLenUnicode>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<VarLenAscii>() {
        return Some(v.to_string());
    }
    None
}

fn h5ad_strip_uns_datasets_encoding_null(path: &Path) -> Result<usize> {
    use hdf5_metno::{Dataset, File as H5File, Group, LocationType};

    fn strip_in_group(g: &Group, removed: &mut usize) -> Result<()> {
        let names = g.member_names()?;
        for sub in names {
            match g.loc_type_by_name(&sub)? {
                LocationType::Group => strip_in_group(&g.group(&sub)?, removed)?,
                LocationType::Dataset => {
                    let ds: Dataset = g.dataset(&sub)?;
                    let is_null = ds
                        .attr("encoding-type")
                        .ok()
                        .and_then(|a| h5ad_attr_encoding_type_string(&a))
                        .is_some_and(|s| s == "null");
                    if is_null {
                        g.unlink(&sub)?;
                        *removed += 1;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    let f = H5File::open_rw(path).with_context(|| {
        format!(
            "open {} read-write to remove uns datasets with encoding-type=null (anndata-rs cannot read them)",
            path.display()
        )
    })?;
    if !f.link_exists("uns") {
        f.close().context("HDF5 close")?;
        return Ok(0);
    }
    let uns = f.group("uns")?;
    let mut removed = 0usize;
    strip_in_group(&uns, &mut removed)?;
    if removed > 0 {
        f.flush()
            .context("HDF5 flush after stripping encoding-type=null uns datasets")?;
    }
    f.close().context("HDF5 close after uns patch")?;
    Ok(removed)
}

fn prepare_h5ad_path_for_anndata_memory_load(path: &Path) {
    match h5ad_strip_uns_datasets_encoding_null(path) {
        Ok(0) => {}
        Ok(n) => eprintln!(
            "rust_preprocess: removed {n} uns dataset(s) with encoding-type=null (e.g. Scanpy uns['log1p']['base']) so anndata-rs can load {}",
            path.display()
        ),
        Err(e) => eprintln!(
            "rust_preprocess: warning: could not patch encoding-type=null entries in {}: {:#}. \
             Open the file read-write once, or re-save in Python without null-encoded uns leaves.",
            path.display(),
            e
        ),
    }
    match h5ad_strip_obsp_for_preprocess_load(path) {
        Ok(false) => {}
        Ok(true) => eprintln!(
            "rust_preprocess: removed obsp (Scanpy neighbor graphs) from {} — full prep recomputes UMAP/Leiden",
            path.display()
        ),
        Err(e) => eprintln!(
            "rust_preprocess: warning: could not strip obsp from {}: {:#}. \
             Loading may still work if obsp CSR is canonical.",
            path.display(),
            e
        ),
    }
}

fn h5ad_strip_obsp_for_preprocess_load(path: &Path) -> Result<bool> {
    use hdf5_metno::File as H5File;

    let f = match H5File::open_rw(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    if !f.link_exists("obsp") {
        f.close().context("HDF5 close")?;
        return Ok(false);
    }
    f.unlink("obsp").context("unlink obsp group")?;
    f.flush().context("HDF5 flush after obsp strip")?;
    f.close().context("HDF5 close after obsp strip")?;
    Ok(true)
}

fn umap_lab_h5_dense_x_column_f32(
    path: &Path,
    col: usize,
    n_obs: usize,
    n_vars: usize,
) -> Result<Vec<f32>> {
    use hdf5_metno::File as H5File;
    let f = H5File::open(path)?;
    let ds = f.dataset("X").context("HDF5 dataset X")?;
    let sh = ds.shape();
    anyhow::ensure!(sh.len() == 2, "X: expected 2D dataset, got shape {:?}", sh);
    anyhow::ensure!(
        sh[0] as usize == n_obs && sh[1] as usize == n_vars,
        "X shape {:?} does not match obs×var {}×{}",
        sh,
        n_obs,
        n_vars
    );
    anyhow::ensure!(
        col < n_vars,
        "gene column {} out of range (n_vars={})",
        col,
        n_vars
    );
    if let Ok(slab) = ds.read_slice_2d::<f32, _>(s![.., col..col + 1]) {
        return Ok(slab.iter().copied().collect());
    }
    let slab = ds
        .read_slice_2d::<f64, _>(s![.., col..col + 1])
        .map_err(|e| anyhow!("read X column as f64: {e}"))?;
    Ok(slab.iter().map(|v| *v as f32).collect())
}

/// Load `X` from disk and return per-cell values for one gene (`var_names` match, case-insensitive fallback).
pub fn umap_lab_gene_expression_from_h5ad(
    path: &Path,
    gene_query: &str,
) -> Result<(String, Vec<f32>, f32, f32)> {
    let q = gene_query.trim();
    if q.is_empty() {
        bail!("gene name is empty");
    }

    let h5 = H5::open(path).context("H5::open (gene)")?;
    let var_container = DataContainer::open(&h5, "var").context("open var")?;
    let var_names = umap_lab_h5_read_dataframe_index(&var_container)?.into_vec();
    let n_vars = var_names.len();

    let obs_container = DataContainer::open(&h5, "obs").context("open obs")?;
    let n_obs = umap_lab_h5_read_dataframe_index(&obs_container)?
        .into_vec()
        .len();

    let idx = var_names
        .iter()
        .position(|n| n.as_str() == q)
        .or_else(|| {
            let ql = q.to_lowercase();
            var_names.iter().position(|n| n.to_lowercase() == ql)
        })
        .with_context(|| format!("gene {q:?} not found (n_vars={n_vars})"))?;
    let resolved = var_names
        .get(idx)
        .with_context(|| "var_names index")?
        .clone();

    if umap_lab_h5_root_x_is_dense_dataset(path)? {
        if let Ok(col) = umap_lab_h5_dense_x_column_f32(path, idx, n_obs, n_vars) {
            let (vmin, vmax) = gene_display_bounds(&col);
            return Ok((resolved, col, vmin, vmax));
        }
    } else if umap_lab_h5_root_x_is_sparse_group(path)? {
        if let Ok(col) = umap_lab_h5_sparse_x_column_f32(path, idx, n_obs, n_vars) {
            let (vmin, vmax) = gene_display_bounds(&col);
            return Ok((resolved, col, vmin, vmax));
        }
    }

    prepare_h5ad_path_for_anndata_memory_load(path);
    let adata = load_h5ad_fast(path).map_err(|e| {
        anyhow!(
            "loading .h5ad for gene expression failed. Caused by: {:#}",
            e
        )
    })?;
    maybe_restore_var_names_in_memory(&adata).context("restore var_names from var columns")?;
    anyhow::ensure!(
        adata.n_obs() == n_obs && adata.n_vars() == n_vars,
        "obs/var shape mismatch after full load (expected {}×{})",
        n_obs,
        n_vars
    );
    let x = adata.x().get_data().context("read X")?;
    let col = x_column_as_f32(&x, idx, n_obs, n_vars)?;
    let (vmin, vmax) = gene_display_bounds(&col);
    Ok((resolved, col, vmin, vmax))
}

fn umap_lab_ensure_normalized_count_for_magic(adata: &IMAnnData) -> Result<()> {
    let n_obs = adata.n_obs();
    let n_vars = adata.n_vars();
    let mut shape_ok = false;
    if let Ok(elem) = adata.layers().get_array("normalized_count") {
        if let Ok(d) = elem.get_data() {
            if let Ok(csr) = normalized_layer_to_csr_f64(d.clone()) {
                if csr.nrows() == n_obs && csr.ncols() == n_vars {
                    shape_ok = true;
                }
            }
        }
    }
    if shape_ok {
        return Ok(());
    }

    ensure_x_csr_for_pca(adata).context("prepare X as CSR for MAGIC normalized_count")?;
    let log_like = infer_x_is_log1p_space(adata)?;
    if log_like {
        let x = adata.x().get_data().context("x for normalized_count")?;
        layer_replace_if_present(adata, "normalized_count", x.clone())?;
        return Ok(());
    }

    let t0 = Instant::now();
    normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)
        .map_err(|e| anyhow!("normalize_expression: {e:?}"))?;
    let norm_data = adata.x().get_data().context("x after normalize")?;
    layer_replace_if_present(adata, "normalized_count", norm_data)?;
    log1p_expression(&adata.x(), None).map_err(|e| anyhow!("log1p_expression: {e:?}"))?;
    eprintln!(
        "umap_lab MAGIC prep: built layers['normalized_count'] in {:.2}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Leiden-cluster-wise MAGIC (same as [`add_magic_imputed_count`]) on the AnnData at `h5ad_path`,
/// writing a full `.h5ad` copy (including `layers['normalized_count']` and `layers['imputed_count']`)
/// to `out_h5ad_path`. Requires `layers['normalized_count']` as CSR/CSC matching `n_obs × n_vars`,
/// or builds it from `X` when missing.
pub fn umap_lab_run_magic_imputed_leiden(
    h5ad_path: &Path,
    graph: &FuzzyGraph,
    leiden_labels: &[String],
    out_h5ad_path: &Path,
) -> Result<()> {
    prepare_h5ad_path_for_anndata_memory_load(h5ad_path);
    let adata = load_h5ad_fast(h5ad_path).map_err(|e| {
        anyhow!(
            "loading .h5ad for MAGIC failed. Caused by: {:#}",
            e
        )
    })?;
    maybe_restore_var_names_in_memory(&adata).context("restore var_names from var columns")?;
    anyhow::ensure!(
        leiden_labels.len() == adata.n_obs(),
        "Leiden labels length {} does not match n_obs {}",
        leiden_labels.len(),
        adata.n_obs()
    );

    umap_lab_ensure_normalized_count_for_magic(&adata)?;
    let _ = adata.layers().remove_array("imputed_count");
    let mut log = Vec::new();
    add_magic_imputed_count(&adata, graph, leiden_labels, RustPreprocessParams::default().magic_t, &mut log)?;
    write_adata_h5ad(&adata, out_h5ad_path)?;
    Ok(())
}

fn umap_lab_h5_sparse_layer_column_f32(
    path: &Path,
    layer: &str,
    col: usize,
    n_obs: usize,
    n_vars: usize,
) -> Result<Vec<f32>> {
    use hdf5_metno::{File as H5F, LocationType};
    let f = H5F::open(path)?;
    let layers = f
        .group("layers")
        .with_context(|| format!("HDF5 missing layers group (layer {layer:?})"))?;
    anyhow::ensure!(
        layers.loc_type_by_name(layer)? == LocationType::Group,
        "layers/{layer} is not a sparse matrix group"
    );
    let g = layers.group(layer)?;
    let shape_a = g
        .attr("shape")
        .context("sparse layer: missing shape attribute")?;
    let (nr, nc) = if let Ok(v) = shape_a.read_1d::<u64>() {
        anyhow::ensure!(v.len() >= 2, "sparse layer shape attr");
        (v[0] as usize, v[1] as usize)
    } else if let Ok(v) = shape_a.read_1d::<i64>() {
        anyhow::ensure!(v.len() >= 2, "sparse layer shape attr");
        (v[0] as usize, v[1] as usize)
    } else if let Ok(v) = shape_a.read_1d::<u32>() {
        anyhow::ensure!(v.len() >= 2, "sparse layer shape attr");
        (v[0] as usize, v[1] as usize)
    } else {
        bail!("sparse layer: unsupported shape attribute dtype");
    };
    anyhow::ensure!(
        nr == n_obs && nc == n_vars,
        "sparse layer {layer} shape {}×{} does not match obs×var {}×{}",
        nr,
        nc,
        n_obs,
        n_vars
    );
    anyhow::ensure!(
        col < n_vars,
        "gene column {col} out of range (n_vars={n_vars})"
    );

    let indptr = h5_read_1d_usize(&g.dataset("indptr").context("sparse layer indptr")?)?;
    let indices = h5_read_1d_usize(&g.dataset("indices").context("sparse layer indices")?)?;
    let data = h5_read_1d_f64_as_vec(&g.dataset("data").context("sparse layer data")?)?;
    anyhow::ensure!(
        indices.len() == data.len(),
        "sparse layer indices/data length mismatch"
    );

    let enc = g
        .attr("encoding-type")
        .ok()
        .and_then(|a| h5ad_attr_encoding_type_string(&a))
        .unwrap_or_default();

    let is_csr = enc == "csr_matrix" || (enc.is_empty() && indptr.len() == n_obs + 1);
    let is_csc = enc == "csc_matrix" || (enc.is_empty() && indptr.len() == n_vars + 1);
    if is_csr {
        anyhow::ensure!(indptr.len() == n_obs + 1, "csr indptr len");
        let mut out = vec![0f32; n_obs];
        for row in 0..n_obs {
            let s = indptr[row];
            let e = indptr[row + 1];
            let mut acc = 0f32;
            for k in s..e {
                if indices[k] == col {
                    acc += data[k] as f32;
                }
            }
            out[row] = acc;
        }
        return Ok(out);
    }
    if is_csc {
        anyhow::ensure!(indptr.len() == n_vars + 1, "csc indptr len");
        let mut out = vec![0f32; n_obs];
        let s = indptr[col];
        let e = indptr[col + 1];
        for k in s..e {
            let row = indices[k];
            if row < n_obs {
                out[row] += data[k] as f32;
            }
        }
        return Ok(out);
    }
    bail!(
        "layers/{layer} sparse: encoding-type {enc:?}, indptr len {} (expected csr n_obs+1 or csc n_vars+1)",
        indptr.len()
    )
}

fn umap_lab_h5_dense_layer_column_f32(
    path: &Path,
    layer: &str,
    col: usize,
    n_obs: usize,
    n_vars: usize,
) -> Result<Vec<f32>> {
    use hdf5_metno::File as H5File;
    let f = H5File::open(path)?;
    let layers = f.group("layers").context("HDF5 missing layers group")?;
    let ds = layers
        .dataset(layer)
        .with_context(|| format!("layers/{layer} dataset"))?;
    let sh = ds.shape();
    anyhow::ensure!(sh.len() == 2, "layer {layer}: expected 2D dataset, got shape {:?}", sh);
    anyhow::ensure!(
        sh[0] as usize == n_obs && sh[1] as usize == n_vars,
        "layer {layer} shape {:?} does not match obs×var {}×{}",
        sh,
        n_obs,
        n_vars
    );
    anyhow::ensure!(
        col < n_vars,
        "gene column {col} out of range (n_vars={n_vars})"
    );
    if let Ok(slab) = ds.read_slice_2d::<f32, _>(s![.., col..col + 1]) {
        return Ok(slab.iter().copied().collect());
    }
    let slab = ds
        .read_slice_2d::<f64, _>(s![.., col..col + 1])
        .map_err(|e| anyhow!("read layer column as f64: {e}"))?;
    Ok(slab.iter().map(|v| *v as f32).collect())
}

/// Read one gene column from `layers/{layer}` (CSR/CSC group or dense dataset), with full-load fallback.
pub fn umap_lab_gene_expression_from_h5ad_layer(
    path: &Path,
    layer: &str,
    gene_query: &str,
) -> Result<(String, Vec<f32>, f32, f32)> {
    let q = gene_query.trim();
    if q.is_empty() {
        bail!("gene name is empty");
    }

    let h5 = H5::open(path).context("H5::open (gene layer)")?;
    let var_container = DataContainer::open(&h5, "var").context("open var")?;
    let var_names = umap_lab_h5_read_dataframe_index(&var_container)?.into_vec();
    let n_vars = var_names.len();

    let obs_container = DataContainer::open(&h5, "obs").context("open obs")?;
    let n_obs = umap_lab_h5_read_dataframe_index(&obs_container)?
        .into_vec()
        .len();

    let idx = var_names
        .iter()
        .position(|n| n.as_str() == q)
        .or_else(|| {
            let ql = q.to_lowercase();
            var_names.iter().position(|n| n.to_lowercase() == ql)
        })
        .with_context(|| format!("gene {q:?} not found (n_vars={n_vars})"))?;
    let resolved = var_names
        .get(idx)
        .with_context(|| "var_names index")?
        .clone();

    use hdf5_metno::{File as H5F, LocationType};
    let f = H5F::open(path)?;
    if f.link_exists("layers") {
        let layers_g = f.group("layers")?;
        if layers_g.link_exists(layer) {
            match layers_g.loc_type_by_name(layer).context("layers entry type")? {
                LocationType::Dataset => {
                    if let Ok(colv) =
                        umap_lab_h5_dense_layer_column_f32(path, layer, idx, n_obs, n_vars)
                    {
                        let (vmin, vmax) = gene_display_bounds(&colv);
                        return Ok((resolved, colv, vmin, vmax));
                    }
                }
                LocationType::Group => {
                    if let Ok(colv) =
                        umap_lab_h5_sparse_layer_column_f32(path, layer, idx, n_obs, n_vars)
                    {
                        let (vmin, vmax) = gene_display_bounds(&colv);
                        return Ok((resolved, colv, vmin, vmax));
                    }
                }
                _ => {}
            }
        }
    }

    prepare_h5ad_path_for_anndata_memory_load(path);
    let adata = load_h5ad_fast(path).map_err(|e| {
        anyhow!(
            "loading .h5ad for layer {layer} expression failed. Caused by: {:#}",
            e
        )
    })?;
    maybe_restore_var_names_in_memory(&adata).context("restore var_names from var columns")?;
    anyhow::ensure!(
        adata.n_obs() == n_obs && adata.n_vars() == n_vars,
        "obs/var shape mismatch after full load (expected {}×{})",
        n_obs,
        n_vars
    );
    let layer_data = adata
        .layers()
        .get_array(layer)
        .with_context(|| format!("layers[{layer}] missing"))?
        .get_data()
        .with_context(|| format!("read layers[{layer}]"))?;
    let col = x_column_as_f32(&layer_data, idx, n_obs, n_vars)?;
    let (vmin, vmax) = gene_display_bounds(&col);
    Ok((resolved, col, vmin, vmax))
}

/// `source`: `"x"` (root `X` on `primary_h5ad`), `"normalized_count"`, or `"imputed_count"`.
/// `magic_artifact` is the temp `.h5ad` written by [`umap_lab_run_magic_imputed_leiden`]; required for `imputed_count`.
pub fn umap_lab_gene_expression_from_h5ad_source(
    primary_h5ad: &Path,
    magic_artifact: Option<&Path>,
    source: &str,
    gene_query: &str,
) -> Result<(String, Vec<f32>, f32, f32)> {
    let s = source.trim();
    match s {
        "" | "x" => umap_lab_gene_expression_from_h5ad(primary_h5ad, gene_query),
        "normalized_count" => {
            let p = magic_artifact.unwrap_or(primary_h5ad);
            umap_lab_gene_expression_from_h5ad_layer(p, "normalized_count", gene_query)
        }
        "imputed_count" => {
            let p = magic_artifact
                .ok_or_else(|| anyhow!("imputed_count requires MAGIC; run cluster-wise MAGIC first"))?;
            umap_lab_gene_expression_from_h5ad_layer(p, "imputed_count", gene_query)
        }
        _ => bail!("unknown gene expression source {s:?}; expected x, normalized_count, or imputed_count"),
    }
}

fn ensure_x_csr_for_pca(adata: &IMAnnData) -> Result<()> {
    let x = adata.x().get_data()?;
    match x {
        ArrayData::CsrMatrix(DynCsrMatrix::F64(_)) | ArrayData::CsrMatrix(DynCsrMatrix::F32(_)) => {
            Ok(())
        }
        ArrayData::CscMatrix(csc) => {
            let csr = match csc {
                DynCscMatrix::F32(m) => {
                    ArrayData::CsrMatrix(DynCsrMatrix::F32(CsrMatrix::from(&m)))
                }
                DynCscMatrix::F64(m) => {
                    ArrayData::CsrMatrix(DynCsrMatrix::F64(CsrMatrix::from(&m)))
                }
                _ => {
                    bail!("rust_preprocess: X CSC matrix must be F32 or F64 for conversion to CSR")
                }
            };
            adata.x().set_data(csr)?;
            Ok(())
        }
        ArrayData::CsrNonCanonical(non) => {
            let csr_dyn = non
                .clone()
                .canonicalize()
                .map_err(|e| anyhow!("rust_preprocess: X non-canonical CSR: {e:?}"))?;
            let csr = csr_dyn_to_csr_f64(csr_dyn)?;
            adata
                .x()
                .set_data(ArrayData::CsrMatrix(DynCsrMatrix::F64(csr)))?;
            Ok(())
        }
        ArrayData::Array(d) => {
            let dense: Array2<f64> = d
                .try_convert()
                .context("rust_preprocess: dense X must convert to f64 for CSR layout")?;
            let nrows = dense.nrows();
            let ncols = dense.ncols();
            eprintln!(
                "rust_preprocess: converting dense X ({nrows}×{ncols}) to CSR f64 (required for normalize_expression / PCA)"
            );
            let csr = dense_ndarray_to_csr_f64(&dense)?;
            adata
                .x()
                .set_data(ArrayData::CsrMatrix(DynCsrMatrix::F64(csr)))?;
            Ok(())
        }
        ArrayData::CsrMatrix(other) => {
            let csr = csr_dyn_to_csr_f64(other)?;
            adata
                .x()
                .set_data(ArrayData::CsrMatrix(DynCsrMatrix::F64(csr)))?;
            Ok(())
        }
        ArrayData::DataFrame(_) => {
            bail!("rust_preprocess: X as DataFrame is not supported for preprocessing")
        }
    }
}

const INFER_LOG1P_MAX_SAMPLES: usize = 100_000;

fn infer_x_is_log1p_space(adata: &IMAnnData) -> Result<bool> {
    let keys = adata.uns().keys().context("uns.keys")?;
    if keys.iter().any(|k| k == "log1p") {
        return Ok(true);
    }
    let x = adata.x().get_data().context("X for log1p inference")?;
    let mut rng = StdRng::seed_from_u64(0);
    let sample = sample_matrix_values_for_log1p_infer(&x, INFER_LOG1P_MAX_SAMPLES, &mut rng)?;
    if sample.is_empty() {
        return Ok(false);
    }
    let mx = sample.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = sample.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[sorted.len() / 2];
    let frac_int = sample
        .iter()
        .filter(|&&v| (v - v.round()).abs() < 1e-5)
        .count() as f64
        / sample.len() as f64;
    if mx > 30.0 {
        return Ok(false);
    }
    if frac_int > 0.72 && mx > 12.0 {
        return Ok(false);
    }
    if mx <= 12.0 && med <= 3.5 && frac_int < 0.55 {
        return Ok(true);
    }
    Ok(false)
}

fn reservoir_sample_f64_from_iter<I>(iter: I, limit: usize, rng: &mut StdRng) -> Vec<f64>
where
    I: Iterator<Item = f64>,
{
    let mut out: Vec<f64> = Vec::new();
    for (i, v) in iter.enumerate() {
        let n = i + 1;
        if out.len() < limit {
            out.push(v);
        } else {
            let j = rng.gen_range(0..n);
            if j < limit {
                out[j] = v;
            }
        }
    }
    out
}

fn sample_matrix_values_for_log1p_infer(
    x: &ArrayData,
    limit: usize,
    rng: &mut StdRng,
) -> Result<Vec<f64>> {
    Ok(match x {
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => {
            reservoir_sample_f64_from_iter(m.triplet_iter().map(|(_, _, v)| *v), limit, rng)
        }
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => {
            reservoir_sample_f64_from_iter(m.triplet_iter().map(|(_, _, v)| *v as f64), limit, rng)
        }
        ArrayData::CscMatrix(DynCscMatrix::F64(m)) => {
            reservoir_sample_f64_from_iter(m.triplet_iter().map(|(_, _, v)| *v), limit, rng)
        }
        ArrayData::CscMatrix(DynCscMatrix::F32(m)) => {
            reservoir_sample_f64_from_iter(m.triplet_iter().map(|(_, _, v)| *v as f64), limit, rng)
        }
        ArrayData::CsrNonCanonical(non) => match non.clone().canonicalize() {
            Ok(csr_dyn) => match csr_dyn {
                DynCsrMatrix::F64(m) => {
                    reservoir_sample_f64_from_iter(m.triplet_iter().map(|(_, _, v)| *v), limit, rng)
                }
                DynCsrMatrix::F32(m) => reservoir_sample_f64_from_iter(
                    m.triplet_iter().map(|(_, _, v)| *v as f64),
                    limit,
                    rng,
                ),
                _ => Vec::new(),
            },
            Err(_) => Vec::new(),
        },
        ArrayData::Array(d) => {
            let dense: Array2<f64> = d.clone().try_convert()?;
            reservoir_sample_f64_from_iter(dense.iter().copied(), limit, rng)
        }
        ArrayData::DataFrame(_) => Vec::new(),
        _ => Vec::new(),
    })
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
    /// Embeddings for UMAP/PCA training plus coordinate `obsm` used by spatial viewers and
    /// [`crate::adata_query::spatial_xy_from_obsm`] (anything else in `obsm` is dropped to shrink exports).
    const KEEP_OBSM: &[&str] = &[
        "X_pca",
        "X_umap",
        "umap",
        "spatial",
        "X_spatial",
        "spatial_loc",
        "unscaled_spatial",
    ];
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
    Ok(dir.join(format!(".{}.{t}.part.h5ad", name.to_string_lossy())))
}

fn attach_dataframe_index(df: DataFrame, names: &[String]) -> Result<DataFrame> {
    let mut cols: Vec<Column> = Vec::with_capacity(df.width() + 1);
    cols.push(Column::from(Series::new("_index".into(), names.to_vec())));
    for s in df.iter() {
        let n = s.name().as_str();
        if n == "_index" || n == "index" {
            continue;
        }
        cols.push(Column::from(s.clone()));
    }
    if cols.len() == 1 && names.is_empty() {
        bail!("attach_dataframe_index: empty axis index and no data columns");
    }
    DataFrame::new(cols).map_err(|e| anyhow!("attach_dataframe_index: {e}"))
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

fn mark_all_var_highly_variable(adata: &IMAnnData) -> Result<()> {
    let var = adata.var();
    if var.get_data().column("highly_variable").is_ok() {
        var.remove_column_from_df("highly_variable")?;
    }
    let n = adata.n_vars();
    var.attach_column_to_df(Series::new(
        "highly_variable".into(),
        vec![true; n],
    ))?;
    Ok(())
}

fn axis_replace_array(axis: &IMAxisArrays, key: &str, data: ArrayData) -> Result<()> {
    let _ = axis.remove_array(key);
    axis.add_array(key.to_string(), IMArrayElement::new(data))?;
    Ok(())
}

fn layer_replace_if_present(adata: &IMAnnData, key: &str, data: ArrayData) -> Result<()> {
    let _ = adata.layers().remove_array(key);
    adata
        .layers()
        .add_array(key.to_string(), IMArrayElement::new(data))?;
    Ok(())
}

const MAGIC_MAX_DENSE_FLOATS: usize = 3_000_000_000;

fn csr_f32_to_csr_f64(m: &CsrMatrix<f32>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(m.nrows(), m.ncols());
    for (r, c, v) in m.triplet_iter() {
        coo.push(r, c, *v as f64);
    }
    CsrMatrix::from(&coo)
}

fn dense_ndarray_to_csr_f64(arr: &Array2<f64>) -> Result<CsrMatrix<f64>> {
    let nrows = arr.nrows();
    let ncols = arr.ncols();
    let mut coo = CooMatrix::new(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            let v = arr[(i, j)];
            if v != 0.0 {
                coo.push(i, j, v);
            }
        }
    }
    Ok(CsrMatrix::from(&coo))
}

fn csr_dyn_to_csr_f64(d: DynCsrMatrix) -> Result<CsrMatrix<f64>> {
    match d {
        DynCsrMatrix::F64(m) => Ok(m),
        DynCsrMatrix::F32(m) => Ok(csr_f32_to_csr_f64(&m)),
        DynCsrMatrix::I8(m) => Ok(csr_typed_to_f64_csr(&m, |v| *v as f64)),
        DynCsrMatrix::I16(m) => Ok(csr_typed_to_f64_csr(&m, |v| *v as f64)),
        DynCsrMatrix::I32(m) => Ok(csr_typed_to_f64_csr(&m, |v| *v as f64)),
        DynCsrMatrix::I64(m) => Ok(csr_typed_to_f64_csr(&m, |v| *v as f64)),
        DynCsrMatrix::U8(m) => Ok(csr_typed_to_f64_csr(&m, |v| *v as f64)),
        DynCsrMatrix::U16(m) => Ok(csr_typed_to_f64_csr(&m, |v| *v as f64)),
        DynCsrMatrix::U32(m) => Ok(csr_typed_to_f64_csr(&m, |v| *v as f64)),
        DynCsrMatrix::U64(m) => Ok(csr_typed_to_f64_csr(&m, |v| *v as f64)),
        DynCsrMatrix::Bool(_) | DynCsrMatrix::String(_) => {
            bail!("HDF5 export: CSR matrix has bool/string dtype; cannot coerce to f64 CSR")
        }
    }
}

fn csr_typed_to_f64_csr<T: Copy>(m: &CsrMatrix<T>, as_f64: impl Fn(&T) -> f64) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(m.nrows(), m.ncols());
    for (r, c, v) in m.triplet_iter() {
        let vf = as_f64(v);
        if vf != 0.0 {
            coo.push(r, c, vf);
        }
    }
    CsrMatrix::from(&coo)
}

fn csc_dyn_to_csr_f64(d: DynCscMatrix) -> Result<CsrMatrix<f64>> {
    match d {
        DynCscMatrix::F64(m) => Ok(CsrMatrix::from(&m)),
        DynCscMatrix::F32(m) => Ok(csr_f32_to_csr_f64(&CsrMatrix::from(&m))),
        DynCscMatrix::I8(m) => Ok(csr_typed_to_f64_csr(&CsrMatrix::from(&m), |v| *v as f64)),
        DynCscMatrix::I16(m) => Ok(csr_typed_to_f64_csr(&CsrMatrix::from(&m), |v| *v as f64)),
        DynCscMatrix::I32(m) => Ok(csr_typed_to_f64_csr(&CsrMatrix::from(&m), |v| *v as f64)),
        DynCscMatrix::I64(m) => Ok(csr_typed_to_f64_csr(&CsrMatrix::from(&m), |v| *v as f64)),
        DynCscMatrix::U8(m) => Ok(csr_typed_to_f64_csr(&CsrMatrix::from(&m), |v| *v as f64)),
        DynCscMatrix::U16(m) => Ok(csr_typed_to_f64_csr(&CsrMatrix::from(&m), |v| *v as f64)),
        DynCscMatrix::U32(m) => Ok(csr_typed_to_f64_csr(&CsrMatrix::from(&m), |v| *v as f64)),
        DynCscMatrix::U64(m) => Ok(csr_typed_to_f64_csr(&CsrMatrix::from(&m), |v| *v as f64)),
        DynCscMatrix::Bool(_) | DynCscMatrix::String(_) => {
            bail!("HDF5 export: CSC matrix has bool/string dtype; cannot coerce to f64 CSR")
        }
    }
}

fn array_data_to_csr_f64_for_h5_export(data: ArrayData) -> Result<ArrayData> {
    let csr = match data {
        ArrayData::CsrMatrix(d) => csr_dyn_to_csr_f64(d)?,
        ArrayData::CscMatrix(d) => csc_dyn_to_csr_f64(d)?,
        ArrayData::CsrNonCanonical(non) => {
            let d = non.clone().canonicalize().map_err(|_| {
                anyhow!("HDF5 export: non-canonical CSR could not be canonicalized")
            })?;
            csr_dyn_to_csr_f64(d)?
        }
        ArrayData::Array(d) => {
            let dense: Array2<f64> = d
                .try_convert()
                .context("HDF5 export: dense matrix to f64")?;
            dense_ndarray_to_csr_f64(&dense)?
        }
        ArrayData::DataFrame(_) => {
            bail!("HDF5 export: expected matrix for X or layer, got DataFrame")
        }
    };
    Ok(ArrayData::CsrMatrix(DynCsrMatrix::F64(csr)))
}

fn normalized_layer_to_csr_f64(nc: ArrayData) -> Result<CsrMatrix<f64>> {
    match nc {
        ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) => Ok(m),
        ArrayData::CsrMatrix(DynCsrMatrix::F32(m)) => Ok(csr_f32_to_csr_f64(&m)),
        ArrayData::CscMatrix(DynCscMatrix::F64(m)) => Ok(CsrMatrix::from(&m)),
        ArrayData::CscMatrix(DynCscMatrix::F32(m)) => Ok(csr_f32_to_csr_f64(&CsrMatrix::from(&m))),
        ArrayData::CsrNonCanonical(non) => {
            let csr_dyn = non.clone().canonicalize().map_err(|_| {
                anyhow!("normalized_count: non-canonical CSR could not be canonicalized")
            })?;
            match csr_dyn {
                DynCsrMatrix::F64(m) => Ok(m),
                DynCsrMatrix::F32(m) => Ok(csr_f32_to_csr_f64(&m)),
                _ => bail!("normalized_count: unsupported CSR scalar type for MAGIC"),
            }
        }
        ArrayData::Array(_) | ArrayData::DataFrame(_) => bail!(
            "layers['normalized_count'] must be CSR/CSC sparse for --rust-magic (avoids full dense materialization of the count matrix)"
        ),
        _ => bail!(
            "layers['normalized_count'] must be CSR/CSC f32/f64 for --rust-magic (unsupported scalar type)"
        ),
    }
}

fn csr_fill_cols_f32_block(
    csr: &CsrMatrix<f64>,
    block: &mut ndarray::ArrayViewMut2<f32>,
    col0: usize,
    col1: usize,
) {
    block.fill(0.0f32);
    for (r, c, v) in csr.triplet_iter() {
        if c >= col0 && c < col1 {
            block[[r, c - col0]] = *v as f32;
        }
    }
}

fn obs_column_as_strings(df: &DataFrame, key: &str) -> Result<Option<Vec<String>>> {
    let Ok(column) = df.column(key) else {
        return Ok(None);
    };
    let series = column.as_materialized_series();
    let mut values = Vec::with_capacity(series.len());
    for i in 0..series.len() {
        values.push(obs_series_row_str(series, i)?);
    }
    Ok(Some(values))
}

/// Upper-triangular induced subgraph on `subset_global` (unique global row indices in visit order).
/// Only edges with both endpoints in the subset and `gi < gj` are retained (same convention as
/// [`leiden_labels_from_graph`]).
pub fn fuzzy_graph_induced_subgraph(
    graph: &FuzzyGraph,
    subset_global: &[usize],
) -> Result<FuzzyGraph> {
    let n = graph.rows();
    if subset_global.is_empty() {
        bail!("fuzzy_graph_induced_subgraph: empty subset");
    }
    let mut g2l: Vec<Option<usize>> = vec![None; n];
    for (li, &gi) in subset_global.iter().enumerate() {
        if gi >= n {
            bail!(
                "fuzzy_graph_induced_subgraph: index {} out of range (n={})",
                gi,
                n
            );
        }
        if g2l[gi].is_some() {
            bail!("fuzzy_graph_induced_subgraph: duplicate index {}", gi);
        }
        g2l[gi] = Some(li);
    }
    let k = subset_global.len();
    let mut tri = TriMatI::new((k, k));
    for &gi in subset_global {
        if let Some(row) = graph.outer_view(gi) {
            for (&jj, &w) in row.indices().iter().zip(row.data().iter()) {
                let gj = jj as usize;
                if w <= 0.0 || gi >= gj {
                    continue;
                }
                let Some(li) = g2l[gi] else {
                    continue;
                };
                let Some(lj) = g2l[gj] else {
                    continue;
                };
                if li < lj {
                    tri.add_triplet(li, lj, w);
                }
            }
        }
    }
    Ok(tri.to_csr())
}

/// Re-run Leiden on the fuzzy subgraph of cells in `parent_code`, then merge labels:
/// cells outside the parent keep their current category string; cells inside become
/// `{parent_name}/{local_leiden_id}`.
pub fn leiden_labels_subcluster_into(
    graph: &FuzzyGraph,
    base_codes: &[u32],
    categories: &[String],
    parent_code: u32,
    resolution: f64,
    max_iter: usize,
) -> Result<Vec<String>> {
    let n = graph.rows();
    if base_codes.len() != n {
        bail!(
            "leiden_labels_subcluster_into: base_codes len {} != graph rows {}",
            base_codes.len(),
            n
        );
    }
    let pc = parent_code as usize;
    if pc >= categories.len() {
        bail!(
            "leiden_labels_subcluster_into: parent_code {} out of range ({} categories)",
            parent_code,
            categories.len()
        );
    }
    let parent_name = &categories[pc];
    let subset: Vec<usize> = base_codes
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| (c == parent_code).then_some(i))
        .collect();
    if subset.is_empty() {
        bail!(
            "leiden_labels_subcluster_into: no cells with code {}",
            parent_code
        );
    }
    let subgraph = fuzzy_graph_induced_subgraph(graph, &subset)?;
    let local = leiden_labels_from_graph(&subgraph, resolution, max_iter);
    if local.len() != subset.len() {
        bail!(
            "leiden_labels_subcluster_into: local labels len {} != subset {}",
            local.len(),
            subset.len()
        );
    }
    let mut out = Vec::with_capacity(n);
    let mut li = 0usize;
    for global_i in 0..n {
        if base_codes[global_i] == parent_code {
            out.push(format!("{}/{}", parent_name, local[li]));
            li += 1;
        } else {
            let c = base_codes[global_i] as usize;
            out.push(categories.get(c).cloned().unwrap_or_else(|| c.to_string()));
        }
    }
    Ok(out)
}

/// Leiden clustering matching Scanpy's `sc.tl.leiden` (RBConfigurationVertexPartition).
///
/// Scanpy's quality increment: `ΔQ = e_{j→c} - γ·k_j·K_c / (2m)`
/// Rust leiden crate's formula: `ΔQ = e_{j→c} - w_j·W_c·r`
/// Match when node weight = weighted degree and `r = γ / (2m)`.
pub fn leiden_labels_from_graph(
    graph: &FuzzyGraph,
    resolution: f64,
    max_iter: usize,
) -> Vec<String> {
    let n = graph.rows();

    let mut degrees = vec![0.0f32; n];
    let mut total_edge_weight = 0.0f64;
    for i in 0..n {
        if let Some(row) = graph.outer_view(i) {
            for (&j, &w) in row.indices().iter().zip(row.data().iter()) {
                let j = j as usize;
                if i < j && w > 0.0 {
                    degrees[i] += w;
                    degrees[j] += w;
                    total_edge_weight += w as f64;
                }
            }
        }
    }

    let mut g = Graph::with_capacity(n, graph.nnz() * 2);
    for i in 0..n {
        g.add_node(degrees[i]);
    }
    for i in 0..n {
        if let Some(row) = graph.outer_view(i) {
            for (&j, &w) in row.indices().iter().zip(row.data().iter()) {
                let j = j as usize;
                if i < j && w > 0.0 {
                    g.add_edge((i as u32).into(), (j as u32).into(), w);
                }
            }
        }
    }

    let scaled_res = if total_edge_weight > 0.0 {
        resolution / (2.0 * total_edge_weight)
    } else {
        resolution
    };

    let network = Network::new_from_graph(g);
    let mut clustering = SimpleClustering::init_different_clusters(network.nodes());
    let mut leiden = Leiden::new(scaled_res, 0.01, Some(42));
    for _ in 0..max_iter {
        if !leiden.iterate(&network, &mut clustering) {
            break;
        }
    }
    (0..network.nodes())
        .map(|i| clustering.get(i).to_string())
        .collect()
}

fn sync_labels_after_embedding(
    adata: &IMAnnData,
    graph: &FuzzyGraph,
    write_leiden: bool,
    run_magic: bool,
    params: &RustPreprocessParams,
    log: &mut Vec<(String, f64)>,
) -> Result<Vec<String>> {
    let obs = adata.obs().get_data();
    let had_cell_type = obs_column_as_strings(&obs, "cell_type")?.is_some();
    let had_leiden = obs_column_as_strings(&obs, "leiden")?.is_some();

    if write_leiden || (run_magic && !had_cell_type && !had_leiden) {
        let t = Instant::now();
        eprintln!(">>> leiden-rs");
        let labels = leiden_labels_from_graph(
            graph,
            params.leiden_resolution,
            params.leiden_max_iter,
        );
        eprintln!("<<< leiden-rs: {:.2} s", t.elapsed().as_secs_f64());
        log.push(("leiden-rs".to_string(), t.elapsed().as_secs_f64()));

        let mut patched = adata.obs().get_data();
        let leiden_ids: Vec<i32> = labels
            .iter()
            .map(|s| s.parse::<i32>())
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| {
                format!(
                    "rust_preprocess: leiden cluster ids must parse as i32 (first labels: {:?})",
                    labels.iter().take(8).collect::<Vec<_>>()
                )
            })?;
        patched.with_column(Series::new("leiden".into(), leiden_ids))?;
        if !had_cell_type {
            patched.with_column(Series::new("cell_type".into(), labels.clone()))?;
        }
        adata
            .obs()
            .set_data(patched)
            .context("obs after leiden-rs")?;
    }

    if run_magic {
        let obs2 = adata.obs().get_data();
        if let Some(ct) = obs_column_as_strings(&obs2, "cell_type")? {
            return Ok(ct);
        }
        if let Some(l) = obs_column_as_strings(&obs2, "leiden")? {
            let mut patched = obs2;
            patched.with_column(Series::new("cell_type".into(), l.clone()))?;
            adata
                .obs()
                .set_data(patched)
                .context("obs cell_type from leiden")?;
            return Ok(l);
        }
        bail!("impute: expected cell_type or leiden in obs after clustering");
    }

    Ok(Vec::new())
}

fn diffusion_for_subset(graph: &FuzzyGraph, rows: &[usize]) -> CsrF64 {
    let n = rows.len();
    let mut local = vec![usize::MAX; graph.rows()];
    for (i, &global) in rows.iter().enumerate() {
        local[global] = i;
    }

    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0_i32);
    for &global_i in rows {
        let mut row = Vec::<(usize, f64)>::new();
        if let Some(view) = graph.outer_view(global_i) {
            for (&global_j, &w) in view.indices().iter().zip(view.data().iter()) {
                let local_j = local[global_j as usize];
                if local_j != usize::MAX && w > 0.0 {
                    row.push((local_j, w as f64));
                }
            }
        }
        let self_j = local[global_i];
        if !row.iter().any(|(j, _)| *j == self_j) {
            row.push((self_j, 1.0));
        }
        let sum: f64 = row.iter().map(|(_, w)| *w).sum();
        let denom = if sum > 0.0 { sum } else { 1.0 };
        row.sort_by_key(|(j, _)| *j);
        for (j, w) in row {
            data.push(w / denom);
            indices.push(j as i32);
        }
        indptr.push(data.len() as i32);
    }
    CsrF64::from_parts(data, indices, indptr, n, n)
}

fn add_magic_imputed_count(
    adata: &IMAnnData,
    graph: &FuzzyGraph,
    labels: &[String],
    magic_t: u32,
    log: &mut Vec<(String, f64)>,
) -> Result<()> {
    if adata.layers().get_array("imputed_count").is_ok() {
        eprintln!("rust_preprocess: layers['imputed_count'] present; skipping MAGIC");
        return Ok(());
    }

    let t = Instant::now();
    eprintln!(">>> MAGIC per cell_type (gene-blocked CSR, f32; no full-matrix densify)");
    let nc = adata
        .layers()
        .get_array("normalized_count")
        .context("layers['normalized_count'] missing before MAGIC")?
        .get_data()
        .context("read normalized_count")?;
    let csr = normalized_layer_to_csr_f64(nc)?;
    let n = csr.nrows();
    let p = csr.ncols();
    let np = n
        .checked_mul(p)
        .ok_or_else(|| anyhow!("rust-magic: n_obs * n_vars overflow"))?;
    if np > MAGIC_MAX_DENSE_FLOATS {
        bail!(
            "rust-magic: n_obs * n_vars = {np} exceeds limit {MAGIC_MAX_DENSE_FLOATS} (dense f32 imputed buffer); subsample or omit --rust-magic"
        );
    }

    let mut groups = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for (i, label) in labels.iter().enumerate() {
        groups.entry(label.clone()).or_default().push(i);
    }

    let cfg = ImputeConfig {
        threads: None,
        gene_block_size: 128,
    };
    let gene_strip = cfg.gene_block_size.max(256);

    let mut out = ndarray::Array2::<f32>::zeros((n, p));
    for kc in (0..p).step_by(gene_strip) {
        let ke = (kc + gene_strip).min(p);
        let mut block = ndarray::Array2::<f32>::zeros((n, ke - kc));
        csr_fill_cols_f32_block(&csr, &mut block.view_mut(), kc, ke);
        for rows in groups.values() {
            if rows.len() < 2 {
                continue;
            }
            let diff = diffusion_for_subset(graph, rows);
            let sub = ndarray::Array2::<f32>::from_shape_fn((rows.len(), ke - kc), |(i, j)| {
                block[[rows[i], j]]
            });
            let imp = impute_magic_f32(&diff, &sub, magic_t, &cfg);
            for (i, &gi) in rows.iter().enumerate() {
                for j in 0..(ke - kc) {
                    block[[gi, j]] = imp[[i, j]];
                }
            }
        }
        out.slice_mut(s![.., kc..ke]).assign(&block);
    }

    let out_f64 = out.mapv(|v| v as f64);
    layer_replace_if_present(
        adata,
        "imputed_count",
        ArrayData::Array(DynArray::from(out_f64)),
    )?;
    eprintln!(
        "<<< MAGIC per cell_type: {:.2} s (n_obs * n_vars = {np})",
        t.elapsed().as_secs_f64()
    );
    log.push(("MAGIC per cell_type".to_string(), t.elapsed().as_secs_f64()));
    Ok(())
}

/// Core in-memory preprocessing: loads h5ad, runs the requested pipeline steps, returns the
/// `IMAnnData` without any disk writes. Used by `rust_preprocess_h5ad_with_steps` and callers
/// that need the processed data in memory (e.g. `--plot-umap`).
pub fn rust_preprocess_h5ad_to_memory(
    input: &Path,
    params: &RustPreprocessParams,
    steps: &RustPreprocessSteps,
) -> Result<IMAnnData> {
    let mut log: Vec<(String, f64)> = Vec::new();

    let t0 = Instant::now();
    eprintln!(">>> read_h5ad");
    prepare_h5ad_path_for_anndata_memory_load(input);
    let mut adata = load_h5ad_fast(input).context("load_h5ad_fast")?;
    maybe_restore_var_names_in_memory(&adata).context("restore var_names from var columns")?;
    eprintln!("  loaded shape=({}, {})", adata.n_obs(), adata.n_vars());
    eprintln!("<<< read_h5ad: {:.2} s", t0.elapsed().as_secs_f64());
    log.push(("read_h5ad".to_string(), t0.elapsed().as_secs_f64()));

    if steps.qc_filter {
        let t = Instant::now();
        eprintln!(">>> filter_genes(min_cells={})", params.min_cells);
        let gene_mask = mark_filter_genes::<u32, f64>(
            &adata,
            Some(params.min_cells),
            None,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| anyhow!("mark_filter_genes: {e:?}"))?;
        eprintln!(
            "<<< filter_genes(min_cells={}): {:.2} s",
            params.min_cells,
            t.elapsed().as_secs_f64()
        );
        log.push((
            format!("filter_genes(min_cells={})", params.min_cells),
            t.elapsed().as_secs_f64(),
        ));

        let t = Instant::now();
        eprintln!(">>> filter_cells(min_genes={})", params.min_genes);
        let cell_mask = mark_filter_cells::<u32, f64>(
            &adata,
            Some(params.min_genes),
            None,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| anyhow!("mark_filter_cells: {e:?}"))?;
        eprintln!(
            "<<< filter_cells(min_genes={}): {:.2} s",
            params.min_genes,
            t.elapsed().as_secs_f64()
        );
        log.push((
            format!("filter_cells(min_genes={})", params.min_genes),
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
        log.push((
            "apply masks (subset)".to_string(),
            t.elapsed().as_secs_f64(),
        ));
    } else {
        eprintln!("rust_preprocess: skipping QC filters (steps.qc_filter=false)");
    }

    let needs_expr_pipeline = steps.normalize_log1p
        || steps.hvg_pca
        || steps.run_umap_and_graph
        || steps.run_magic_impute;
    if !needs_expr_pipeline {
        bail!(
            "rust_preprocess: at least one of normalize_log1p, hvg_pca, run_umap_and_graph, run_magic_impute must be true"
        );
    }

    if steps.normalize_log1p {
        let log_like = infer_x_is_log1p_space(&adata)?;
        if log_like {
            eprintln!(
                "rust_preprocess: X classified as log-normalized (uns['log1p'] or Scanpy-style heuristic); skip normalize_total + log1p; copy X → layers['normalized_count'] / ['log1p']"
            );
            let x = adata.x().get_data().context("x for layer copy")?;
            layer_replace_if_present(&adata, "normalized_count", x.clone())?;
            layer_replace_if_present(&adata, "log1p", x)?;
            log.push(("skip_normalize_log1p_log_space".to_string(), 0.0));
        } else {
            ensure_x_csr_for_pca(&adata).context("prepare X as CSR for normalize_total")?;
            let t = Instant::now();
            eprintln!(
                ">>> normalize_total (target_sum={}, Scanpy-equivalent)",
                params.normalize_target_sum
            );
            normalize_expression(
                &adata.x(),
                params.normalize_target_sum,
                &Direction::ROW,
                None,
            )
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
        }
    } else if steps.hvg_pca || steps.run_umap_and_graph || steps.run_magic_impute {
        bail!(
            "rust_preprocess: HVG/PCA/UMAP/impute require normalize_log1p=true (Scanpy normalize_total + log1p on X)"
        );
    }

    let mut pca = ndarray::Array2::<f64>::zeros((0, 0));
    if steps.hvg_pca {
        let n_total = adata.n_vars();
        let skip_dispersion_hvg = n_total <= params.n_top_hvg;

        let mut combined_mask: Vec<bool> = if skip_dispersion_hvg {
            eprintln!(
                "rust_preprocess: n_vars={n_total} <= n_top_hvg={} — skipping dispersion HVG; using all non-MT genes",
                params.n_top_hvg
            );
            let var_names = adata.var_names();
            var_names
                .iter()
                .map(|name| !name.to_lowercase().starts_with("mt"))
                .collect()
        } else {
            let hvg_target = params
                .n_top_hvg
                .min(n_total.saturating_sub(50).max(1));
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

            let hvg_mask = read_var_hvg_mask(&adata)?;
            let var_names = adata.var_names();
            hvg_mask
                .iter()
                .zip(var_names.iter())
                .map(|(&hv, name)| hv && !name.to_lowercase().starts_with("mt"))
                .collect()
        };

        let _force_kept = apply_force_keep_genes_env(&adata, &mut combined_mask);

        let n_keep = combined_mask.iter().filter(|&&x| x).count();
        if n_keep == 0 {
            bail!("rust_preprocess: after HVG / MT filter, zero genes remain");
        }

        if n_keep < n_total {
            let t = Instant::now();
            eprintln!(
                ">>> subset AnnData to {n_keep} genes (HVG ∩ ¬MT of {n_total}) before PCA / MAGIC"
            );
            let gene_idx = mask_to_indices(&combined_mask);
            let obs_idx: Vec<usize> = (0..adata.n_obs()).collect();
            adata = adata
                .subset(&[&SelectInfoElem::from(obs_idx), &SelectInfoElem::from(gene_idx)])
                .map_err(|e| anyhow!("HVG gene subset: {e:?}"))?;
            eprintln!(
                "<<< subset genes: {:.2} s (shape now {} × {})",
                t.elapsed().as_secs_f64(),
                adata.n_obs(),
                adata.n_vars()
            );
            log.push(("subset HVG genes".to_string(), t.elapsed().as_secs_f64()));
        } else {
            eprintln!(
                "rust_preprocess: retaining all {n_keep} genes (no column subset; matches HVG mask)"
            );
        }

        mark_all_var_highly_variable(&adata).context("var highly_variable after HVG subset")?;

        let t = Instant::now();
        eprintln!(">>> convert X to CSR (for PCA)");
        ensure_x_csr_for_pca(&adata)?;
        eprintln!("<<< convert X to CSR: {:.2} s", t.elapsed().as_secs_f64());
        log.push(("convert X to CSR".to_string(), t.elapsed().as_secs_f64()));

        let pca_feature_mask = vec![true; adata.n_vars()];
        let t = Instant::now();
        eprintln!(">>> pca");
        let pca_res = run_pca_sparse_masked::<f64>(
            &adata.x(),
            Some(FeatureSelectionMethod::HighlyVariableSelection(
                pca_feature_mask,
            )),
            Some(true),
            Some(false),
            Some(params.n_pca_components),
            None,
            Some(params.pca_random_seed),
            Some(SVDMethod::Random {
                n_oversamples: 10,
                n_power_iterations: 4,
                normalizer: PowerIterationNormalizer::QR,
            }),
        )
        .map_err(|e| anyhow!("PCA: {e:?}"))?;
        pca = pca_res.transformed;
        eprintln!("  PCA shape: {:?}", pca.shape());
        eprintln!("<<< pca: {:.2} s", t.elapsed().as_secs_f64());
        log.push(("pca".to_string(), t.elapsed().as_secs_f64()));
    } else if steps.run_umap_and_graph {
        bail!("rust_preprocess: UMAP requires hvg_pca=true");
    }

    if pca.nrows() > 0 {
        let pca_f64 =
            ndarray::Array2::<f64>::from_shape_fn((pca.nrows(), pca.ncols()), |(i, j)| pca[(i, j)]);
        axis_replace_array(
            &adata.obsm(),
            "X_pca",
            ArrayData::Array(DynArray::from(pca_f64)),
        )?;
    }

    if steps.run_umap_and_graph {
        let (emb_umap, fuzzy_graph, _knn_cache) = run_umap_on_pca(&pca, params, &mut log, None)?;

        let n = emb_umap.nrows();
        let umap_f64 =
            ndarray::Array2::<f64>::from_shape_fn((n, 2), |(i, j)| emb_umap[(i, j)] as f64);

        axis_replace_array(
            &adata.obsm(),
            "X_umap",
            ArrayData::Array(DynArray::from(umap_f64)),
        )?;

        let labels = sync_labels_after_embedding(
            &adata,
            &fuzzy_graph,
            steps.write_leiden,
            steps.run_magic_impute,
            params,
            &mut log,
        )?;
        if steps.run_magic_impute {
            add_magic_imputed_count(
                &adata,
                &fuzzy_graph,
                &labels,
                params.magic_t,
                &mut log,
            )?;
        }
    }

    let total: f64 = log.iter().map(|(_, s)| s).sum();
    eprintln!("rust_preprocess: TOTAL (sum of steps) {total:.2} s");
    for (name, dt) in &log {
        eprintln!("  {name}: {dt:.2} s");
    }

    Ok(adata)
}

/// Scanpy-style pipeline on AnnData: optional QC, then either **`normalize_total` (target_sum=10_000) + `log1p`**
/// on `X` or a **log-space skip** matching Scanpy `full_preprocess` rules for `X`, then optional HVG → PCA → UMAP
/// and optional Leiden / MAGIC per [`RustPreprocessSteps`]. When `output` is `None`, runs in memory and skips the
/// HDF5 write. When `output` is `Some(path)`, writes `path` with **`X` and every `layers` matrix stored as CSR f64**.
pub fn rust_preprocess_h5ad_with_steps(
    input: &Path,
    output: Option<&Path>,
    params: &RustPreprocessParams,
    steps: &RustPreprocessSteps,
) -> Result<Option<PathBuf>> {
    let adata = rust_preprocess_h5ad_to_memory(input, params, steps)?;

    let Some(output) = output else {
        eprintln!("rust_preprocess: no output path; skipped HDF5 write");
        return Ok(None);
    };

    write_adata_h5ad(&adata, output)?;
    Ok(Some(output.to_path_buf()))
}

fn write_adata_h5ad(adata: &IMAnnData, output: &Path) -> Result<()> {
    let t = Instant::now();
    eprintln!(">>> write_h5ad {}", output.display());
    if output.exists() {
        std::fs::remove_file(output)
            .map_err(|e| anyhow!("cannot remove existing output {}: {e}", output.display()))?;
    }
    strip_supplemental_axis_arrays_for_h5_export(adata)
        .context("strip obsp/varm/varp and non-embedding obsm for HDF5 export")?;
    let obs_safe = dataframe_hdf5_safe(adata.obs().get_data()).context("sanitize obs for HDF5")?;
    adata
        .obs()
        .set_data(obs_safe)
        .context("obs.set_data after sanitize")?;
    clear_uns_for_hdf5_export(adata).context("clear uns for HDF5")?;

    let tmp = temp_h5ad_path(output).context("temp output path for HDF5")?;
    if tmp.exists() {
        let _ = std::fs::remove_file(&tmp);
    }
    let write_result = (|| -> Result<()> {
        let written = AnnData::<H5>::new(&tmp).context("create temp h5ad")?;

        let obs_with_index = attach_dataframe_index(adata.obs().get_data(), &adata.obs_names())
            .context("attach _index to obs before HDF5 write")?;
        written
            .set_obs(obs_with_index)
            .context("write obs (with _index)")?;

        let var_with_index = attach_dataframe_index(adata.var().get_data(), &adata.var_names())
            .context("attach _index to var before HDF5 write")?;
        written
            .set_var(var_with_index)
            .context("write var (with _index)")?;

        let x_export =
            array_data_to_csr_f64_for_h5_export(adata.x().get_data().context("read X for export")?)
                .context("coerce X to CSR f64 for HDF5 export")?;
        written.set_x(x_export).context("write X")?;
        for key in adata.obsm().keys() {
            let elem = adata
                .obsm()
                .get_array(&key)
                .with_context(|| format!("read obsm[{key}]"))?;
            written
                .obsm()
                .add(&key, elem.get_data()?)
                .with_context(|| format!("write obsm[{key}]"))?;
        }
        for key in adata.layers().keys() {
            let elem = adata
                .layers()
                .get_array(&key)
                .with_context(|| format!("read layers[{key}]"))?;
            let layer_export = array_data_to_csr_f64_for_h5_export(elem.get_data()?)
                .with_context(|| format!("coerce layers[{key}] to CSR f64 for HDF5 export"))?;
            written
                .layers()
                .add(&key, layer_export)
                .with_context(|| format!("write layers[{key}]"))?;
        }
        written.close().context("close AnnData H5")?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    write_result
        .context("rust_preprocess: HDF5 export failed; partial temp file was removed if present")?;
    std::fs::rename(&tmp, output).with_context(|| {
        format!(
            "rename temp HDF5 {:?} -> {:?}",
            tmp.display(),
            output.display()
        )
    })?;
    eprintln!("<<< write_h5ad: {:.2} s", t.elapsed().as_secs_f64());
    Ok(())
}

pub fn rust_preprocess_h5ad(
    input: &Path,
    output: &Path,
    params: &RustPreprocessParams,
) -> Result<Option<PathBuf>> {
    rust_preprocess_h5ad_with_steps(input, Some(output), params, &RustPreprocessSteps::FULL)
}

#[cfg(test)]
mod preprocess_tests {
    use super::*;

    #[test]
    fn var_restore_digit_index_uses_feature_name() {
        let names: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        let s = Series::new(
            "feature_name".into(),
            vec!["TP53", "EGFR", "MALAT1", "X", "Y"],
        );
        let df = DataFrame::new(vec![s.into()]).expect("dataframe");
        let got = restore_var_names_if_placeholder(&names, &df)
            .expect("restore")
            .expect("expected restore");
        assert_eq!(got, vec!["TP53", "EGFR", "MALAT1", "X", "Y"]);
    }

    #[test]
    fn var_restore_skips_when_index_already_symbolic() {
        let names = vec!["A".to_string(), "B".to_string()];
        let s = Series::new("feature_name".into(), vec!["x", "y"]);
        let df = DataFrame::new(vec![s.into()]).unwrap();
        assert!(restore_var_names_if_placeholder(&names, &df)
            .unwrap()
            .is_none());
    }

    #[test]
    fn var_restore_dedupes_duplicate_symbols() {
        let names: Vec<String> = (0..3).map(|i| i.to_string()).collect();
        let s = Series::new("feature_name".into(), vec!["G", "G", "H"]);
        let df = DataFrame::new(vec![s.into()]).unwrap();
        let got = restore_var_names_if_placeholder(&names, &df)
            .unwrap()
            .unwrap();
        assert_eq!(got, vec!["G", "G-1", "H"]);
    }

    #[test]
    fn var_restore_respects_column_priority_feature_name_before_gene_ids() {
        let names: Vec<String> = (0..2).map(|i| i.to_string()).collect();
        let sym = Series::new("feature_name".into(), vec!["RealA", "RealB"]);
        let ens = Series::new("gene_ids".into(), vec!["ENS1", "ENS2"]);
        let df = DataFrame::new(vec![ens.into(), sym.into()]).unwrap();
        let got = restore_var_names_if_placeholder(&names, &df)
            .unwrap()
            .unwrap();
        assert_eq!(got, vec!["RealA", "RealB"]);
    }
}
