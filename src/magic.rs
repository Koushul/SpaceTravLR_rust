use std::collections::HashMap;
use std::path::Path;

use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::{Context, Result, bail};
use polars::prelude::{DataFrame, DataType};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::magic_pca::{SklearnRandomizedPcaConfig, fit_randomized_pca_sklearn};

#[derive(Clone, Debug)]
pub struct MagicGraphParams {
    pub knn: usize,
    pub knn_max: usize,
    pub decay: f64,
    pub thresh: f64,
    pub bandwidth_scale: f64,
    pub search_multiplier: usize,
}

impl Default for MagicGraphParams {
    fn default() -> Self {
        Self {
            knn: 5,
            knn_max: 15,
            decay: 1.0,
            thresh: 1e-4,
            bandwidth_scale: 1.0,
            search_multiplier: 6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Csr {
    pub data: Vec<f64>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
    pub nrows: usize,
    pub ncols: usize,
}

impl Csr {
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn from_scipy_npz(path: &Path) -> Result<Self> {
        use ndarray::Array1;
        use ndarray_npy::NpzReader;
        use std::fs::File;

        let read_usize_vec = |name: &str| -> Result<Vec<usize>> {
            let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let mut r = NpzReader::new(f)?;
            let try_i32: Result<Array1<i32>, _> = r.by_name(name);
            match try_i32 {
                Ok(a) => Ok(a.iter().map(|&x| x as usize).collect()),
                Err(_) => {
                    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
                    let mut r = NpzReader::new(f)?;
                    let a: Array1<i64> = r.by_name(name)?;
                    Ok(a.iter().map(|&x| x as usize).collect())
                }
            }
        };

        let data: Array1<f64> = {
            let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let mut r = NpzReader::new(f)?;
            r.by_name("data")?
        };
        let indices = read_usize_vec("indices")?;
        let indptr = read_usize_vec("indptr")?;
        let shape: Array1<i64> = {
            let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let mut r = NpzReader::new(f)?;
            r.by_name("shape")?
        };
        let nrows = shape[0] as usize;
        let ncols = shape[1] as usize;
        Ok(Self {
            data: data.to_vec(),
            indices,
            indptr,
            nrows,
            ncols,
        })
    }

    pub fn symmetrize_additive(&self) -> Csr {
        let mut acc: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..self.nrows {
            let s = self.indptr[i];
            let e = self.indptr[i + 1];
            for k in s..e {
                let j = self.indices[k];
                let v = self.data[k];
                *acc.entry((i, j)).or_insert(0.0) += v;
            }
        }
        let summed: Vec<((usize, usize), f64)> = acc.iter().map(|(&k, &v)| (k, v)).collect();
        for ((i, j), v) in summed {
            *acc.entry((j, i)).or_insert(0.0) += v;
        }
        for v in acc.values_mut() {
            *v *= 0.5;
        }
        coo_to_csr_sorted(self.nrows, self.ncols, acc)
    }

    pub fn row_normalize_l1(&self) -> Csr {
        let mut data = self.data.clone();
        for i in 0..self.nrows {
            let s = self.indptr[i];
            let e = self.indptr[i + 1];
            let sum: f64 = data[s..e].iter().sum();
            if sum > 0.0 {
                for k in s..e {
                    data[k] /= sum;
                }
            }
        }
        Csr {
            data,
            indices: self.indices.clone(),
            indptr: self.indptr.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    pub fn spmm_dense(&self, x: ArrayView2<f64>) -> Array2<f64> {
        assert_eq!(self.ncols, x.nrows(), "P.cols must match X.rows");
        let n = self.nrows;
        let g = x.ncols();
        let mut y = Array2::<f64>::zeros((n, g));
        y.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row_y)| {
                let s = self.indptr[i];
                let e = self.indptr[i + 1];
                for k in s..e {
                    let j = self.indices[k];
                    let a = self.data[k];
                    let xrow = x.row(j);
                    for c in 0..g {
                        row_y[c] += a * xrow[c];
                    }
                }
            });
        y
    }
}

fn coo_to_csr_sorted(nrows: usize, ncols: usize, acc: HashMap<(usize, usize), f64>) -> Csr {
    let mut entries: Vec<(usize, usize, f64)> = acc
        .into_iter()
        .filter(|(_, v)| *v != 0.0)
        .map(|((i, j), v)| (i, j, v))
        .collect();
    entries.sort_by_key(|&(i, j, _)| (i, j));
    let mut indptr = vec![0usize; nrows + 1];
    let mut indices = Vec::with_capacity(entries.len());
    let mut data = Vec::with_capacity(entries.len());
    for (i, j, v) in entries {
        indptr[i + 1] += 1;
        indices.push(j);
        data.push(v);
    }
    for i in 1..=nrows {
        indptr[i] += indptr[i - 1];
    }
    debug_assert_eq!(indptr[nrows], data.len());
    Csr {
        data,
        indices,
        indptr,
        nrows,
        ncols,
    }
}

fn knn_brute(
    data_nu: ArrayView2<f64>,
    n_neighbors: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let n = data_nu.nrows();
    let norms: Array1<f64> = data_nu
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|x| x * x).sum::<f64>())
        .collect();

    (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = data_nu.row(i);
            let mut dists: Vec<(usize, f64)> = Vec::with_capacity(n);
            let ni = norms[i];
            for j in 0..n {
                let dot: f64 = xi.dot(&data_nu.row(j));
                let dist_sq = (ni + norms[j] - 2.0 * dot).max(0.0);
                dists.push((j, dist_sq.sqrt()));
            }
            dists.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            dists.truncate(n_neighbors);
            let idx: Vec<usize> = dists.iter().map(|x| x.0).collect();
            let dist: Vec<f64> = dists.iter().map(|x| x.1).collect();
            (idx, dist)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .unzip()
}

fn affinity_row(
    neigh_idx: &[usize],
    neigh_dist: &[f64],
    bw: f64,
    decay: f64,
    thresh: f64,
) -> Vec<(usize, f64)> {
    let mut out = Vec::new();
    for (&j, &dist) in neigh_idx.iter().zip(neigh_dist.iter()) {
        let scaled = dist / bw;
        let mut a = (-scaled.powf(decay)).exp();
        if a.is_nan() || a.is_infinite() {
            a = 1.0;
        }
        if a >= thresh {
            out.push((j, a));
        }
    }
    if out.len() > 1 {
        out.sort_by_key(|&(j, _)| j);
    }
    out
}

pub fn build_magic_kernel_graphtools_style(
    data_nu: ArrayView2<f64>,
    p: &MagicGraphParams,
) -> Result<Csr> {
    let n = data_nu.nrows();
    if p.knn == 0 {
        bail!("knn must be positive");
    }
    let internal_k = p.knn + 1;
    let knn_max_cap = p.knn_max + 1;
    let search_knn = (internal_k * p.search_multiplier)
        .min(knn_max_cap)
        .max(internal_k)
        .min(n);

    let (indices, distances) = knn_brute(data_nu, search_knn);
    let eps = f64::EPSILON;

    let mut acc: HashMap<(usize, usize), f64> = HashMap::new();
    for i in 0..n {
        let bw = distances[i][internal_k - 1] * p.bandwidth_scale;
        let bw = bw.max(eps);
        let row = affinity_row(&indices[i], &distances[i], bw, p.decay, p.thresh);
        for (j, v) in row {
            *acc.entry((i, j)).or_insert(0.0) += v;
        }
    }

    Ok(coo_to_csr_sorted(n, n, acc))
}

pub fn diffusion_operator_from_affinity(k: &Csr) -> Csr {
    k.symmetrize_additive().row_normalize_l1()
}

pub fn impute_markov(p: &Csr, x: ArrayView2<f64>, t: usize) -> Array2<f64> {
    assert_eq!(p.nrows, x.nrows());
    let mut cur = x.to_owned();
    for _ in 0..t {
        cur = p.spmm_dense(cur.view());
    }
    cur
}

pub fn preprocess_library_size_sqrt(x: ArrayView2<f64>) -> Array2<f64> {
    let mut out = x.to_owned();
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let s: f64 = row.iter().copied().sum::<f64>().max(1e-12);
            let scale = 1e4 / s;
            for v in row.iter_mut() {
                *v = (*v * scale).sqrt();
            }
        });
    out
}

#[derive(Clone, Debug)]
pub struct MagicMarkovParams {
    pub graph: MagicGraphParams,
    pub n_pca: usize,
    pub t: usize,
    pub pca: SklearnRandomizedPcaConfig,
}

impl Default for MagicMarkovParams {
    fn default() -> Self {
        Self {
            graph: MagicGraphParams::default(),
            n_pca: 100,
            t: 3,
            pca: SklearnRandomizedPcaConfig::new(100),
        }
    }
}

pub fn magic_impute_from_embedding(
    data_nu: ArrayView2<f64>,
    x: ArrayView2<f64>,
    t: usize,
    graph: &MagicGraphParams,
) -> Result<Array2<f64>> {
    let k = build_magic_kernel_graphtools_style(data_nu, graph)?;
    let p = diffusion_operator_from_affinity(&k);
    Ok(impute_markov(&p, x, t))
}

pub fn magic_impute_preprocessed(x: ArrayView2<f64>, params: &MagicMarkovParams) -> Result<Array2<f64>> {
    let n_comp = params
        .n_pca
        .min(x.nrows())
        .min(x.ncols());
    if n_comp == 0 {
        bail!("empty or degenerate expression matrix");
    }
    let mut pca_cfg = params.pca.clone();
    pca_cfg.n_components = n_comp;
    let (data_nu, _, _) = fit_randomized_pca_sklearn(x, &pca_cfg)?;
    magic_impute_from_embedding(data_nu.view(), x, params.t, &params.graph)
}

pub fn magic_impute_h5ad_raw_counts(
    path: &Path,
    layer: &str,
    params: &MagicMarkovParams,
) -> anyhow::Result<Array2<f64>> {
    let raw = crate::spatial_estimator::read_h5ad_expression_dense_f64(path, layer)?;
    let x = preprocess_library_size_sqrt(raw.view());
    magic_impute_preprocessed(x.view(), params).map_err(|e| anyhow::anyhow!("{}", e))
}

fn rows_subset(x: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((idx.len(), x.ncols()));
    for (i, &r) in idx.iter().enumerate() {
        out.row_mut(i).assign(&x.row(r));
    }
    out
}

fn cluster_annot_column(obs: &DataFrame) -> anyhow::Result<&'static str> {
    if obs.column("cell_type").is_ok() {
        return Ok("cell_type");
    }
    if obs.column("leiden").is_ok() {
        return Ok("leiden");
    }
    anyhow::bail!(
        "clusterwise MAGIC needs obs column 'cell_type' or 'leiden' (e.g. after Leiden in preprocess)"
    )
}

fn obs_column_strings(obs: &DataFrame, name: &str) -> anyhow::Result<Vec<String>> {
    let col = obs.column(name).with_context(|| format!("obs column {name:?}"))?;
    let series = col.as_materialized_series();
    let as_str = series.cast(&DataType::String)?;
    Ok(as_str
        .str()?
        .into_iter()
        .map(|o| o.map(str::to_string).unwrap_or_else(|| "NA".into()))
        .collect())
}

/// Clusterwise Markov imputation on **`normalized_count`** (library-size normalized linear counts).
/// Groups cells by **`cell_type`** when present in `obs`, otherwise **`leiden`**. Single-cell clusters
/// are copied unchanged. Graph `knn` / `knn_max` are capped per cluster size.
pub fn magic_impute_clusterwise_normalized_count_layer(
    path: &Path,
    params: &MagicMarkovParams,
) -> anyhow::Result<Array2<f64>> {
    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    let obs = adata.read_obs().map_err(|e| anyhow::anyhow!("{}", e))?;
    adata.close()?;

    let col = cluster_annot_column(&obs)?;
    let labels = obs_column_strings(&obs, col)?;
    let x = crate::spatial_estimator::read_h5ad_expression_dense_f64(path, "normalized_count")?;
    if x.nrows() != labels.len() {
        anyhow::bail!(
            "normalized_count rows {} != obs height {}",
            x.nrows(),
            labels.len()
        );
    }

    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, lab) in labels.into_iter().enumerate() {
        groups.entry(lab).or_default().push(i);
    }

    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort();

    let mut out = Array2::<f64>::zeros(x.dim());
    for k in keys {
        let idx = groups.get(&k).expect("cluster key");
        let n_sub = idx.len();
        if n_sub < 2 {
            for &r in idx.iter() {
                out.row_mut(r).assign(&x.row(r));
            }
            continue;
        }
        let sub = rows_subset(&x, idx);
        let mut p = params.clone();
        p.graph.knn = p.graph.knn.min(n_sub - 1).max(1);
        p.graph.knn_max = p.graph.knn_max.min(n_sub - 1).max(p.graph.knn);
        let im = magic_impute_preprocessed(sub.view(), &p).map_err(|e| anyhow::anyhow!("{}", e))?;
        for (i, &r) in idx.iter().enumerate() {
            out.row_mut(r).assign(&im.row(i));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markov_two_step_matches_dense() {
        let n = 4;
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
            data[i * n + ((i + 1) % n)] = 0.5;
            data[i * n + ((i + n - 1) % n)] = 0.5;
        }
        let mut row_sums = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                row_sums[i] += data[i * n + j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] /= row_sums[i];
            }
        }
        let mut indptr = vec![0usize; n + 1];
        let mut indices = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let v: f64 = data[i * n + j];
                if v.abs() > 1e-15 {
                    indices.push(j);
                    vals.push(v);
                }
            }
            indptr[i + 1] = indices.len();
        }
        let csr = Csr {
            data: vals,
            indices,
            indptr,
            nrows: n,
            ncols: n,
        };
        let x =
            Array2::from_shape_vec((n, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y1 = impute_markov(&csr, x.view(), 2);
        let pd = ndarray::Array2::from_shape_vec((n, n), data).unwrap();
        let y2 = pd.dot(&pd).dot(&x);
        let d = (&y1 - &y2).mapv(f64::abs).iter().copied().fold(0.0_f64, f64::max);
        assert!(d < 1e-10, "max abs diff {d}");
    }
}
