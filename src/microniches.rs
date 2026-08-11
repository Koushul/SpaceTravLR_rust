//! Discover functional microniches from a finished SpaceTravLR run.
//!
//! Pipeline:
//! 1. Load `spacetravlr_run_repro.toml` → AnnData + `*_betadata.feather`
//! 2. Optional cell-type subset
//! 3. Spatially filter β features (Moran's I × spatial η², FDR, decorrelation)
//! 4. Z-score → dense PCA
//! 5. Fuzzy kNN graph on PCA → Leiden
//! 6. By default, sweep Leiden resolution and pick the best mean silhouette

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::{Context, bail};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::Serialize;

use crate::betadata::{
    betadata_cluster_keys_from_obs_dataframe, betadata_feather_per_cell_column,
    betadata_feather_plottable_columns, obs_series_row_str, resolve_betadata_cluster_key_column,
    write_betadata_feather,
};
use crate::config::{SpaceshipConfig, expand_user_path};
use crate::rust_preprocess::{FuzzyGraph, fuzzy_graph_from_pca, leiden_labels_from_graph};
use crate::spatial_estimator::load_spatial_coords_f64;

const DEFAULT_ANNOT: &str = "cell_type";
const DEFAULT_N_NEIGHBORS: usize = 15;
const DEFAULT_N_PCS: usize = 40;
const DEFAULT_EF: usize = 30;
const DEFAULT_SPATIAL_K: usize = 8;
const DEFAULT_MORAN_PERM: usize = 49;
const DEFAULT_Q_MAX: f64 = 0.05;
const DEFAULT_CORR_MAX: f64 = 0.85;
const DEFAULT_RES_MIN: f64 = 0.2;
const DEFAULT_RES_MAX: f64 = 2.0;
const DEFAULT_RES_STEP: f64 = 0.1;
const DEFAULT_LEIDEN_MAX_ITER: usize = 100;
const DEFAULT_GRID: usize = 8;
const DEFAULT_SEED: u64 = 0;

#[derive(Clone, Debug)]
pub struct MicronichesParams {
    pub annot_col: String,
    pub cell_type: Option<String>,
    pub n_neighbors: usize,
    pub n_pcs: usize,
    pub ef_construction: usize,
    pub leiden_resolution: Option<f64>,
    pub resolution_min: f64,
    pub resolution_max: f64,
    pub resolution_step: f64,
    pub leiden_max_iter: usize,
    pub spatial_k: usize,
    pub moran_n_perm: usize,
    pub q_bh_max: f64,
    pub corr_max: f64,
    pub spatial_grid: usize,
    pub max_features: Option<usize>,
    pub features_csv: Option<PathBuf>,
    pub max_genes: Option<usize>,
    pub seed: u64,
}

impl Default for MicronichesParams {
    fn default() -> Self {
        Self {
            annot_col: DEFAULT_ANNOT.into(),
            cell_type: None,
            n_neighbors: DEFAULT_N_NEIGHBORS,
            n_pcs: DEFAULT_N_PCS,
            ef_construction: DEFAULT_EF,
            leiden_resolution: None,
            resolution_min: DEFAULT_RES_MIN,
            resolution_max: DEFAULT_RES_MAX,
            resolution_step: DEFAULT_RES_STEP,
            leiden_max_iter: DEFAULT_LEIDEN_MAX_ITER,
            spatial_k: DEFAULT_SPATIAL_K,
            moran_n_perm: DEFAULT_MORAN_PERM,
            q_bh_max: DEFAULT_Q_MAX,
            corr_max: DEFAULT_CORR_MAX,
            spatial_grid: DEFAULT_GRID,
            max_features: None,
            features_csv: None,
            max_genes: None,
            seed: DEFAULT_SEED,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct KeptBetaFeature {
    pub gene: String,
    pub feature: String,
    pub moran_i: f64,
    pub eta2: f64,
    pub q_bh: f64,
    pub spatial_score: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResolutionSweepRow {
    pub resolution: f64,
    pub silhouette: f64,
    pub n_clusters: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MicronichesSummary {
    pub n_cells: usize,
    pub n_kept_features: usize,
    pub n_pcs: usize,
    pub n_clusters: usize,
    pub chosen_resolution: f64,
    pub silhouette: f64,
    pub cell_type_subset: Option<String>,
    pub annot_col: String,
    pub optimized_by_silhouette: bool,
    pub resolution_sweep: Vec<ResolutionSweepRow>,
    pub output_dir: String,
}

#[derive(Clone, Debug)]
pub struct MicronichesResult {
    pub cell_ids: Vec<String>,
    pub labels: Vec<String>,
    pub pca: Array2<f64>,
    pub spatial: Array2<f64>,
    pub kept_features: Vec<KeptBetaFeature>,
    pub summary: MicronichesSummary,
}

#[derive(Clone, Debug)]
struct FeatureCandidate {
    gene: String,
    feature: String,
    values: Vec<f64>,
    #[allow(dead_code)]
    mad: f64,
    moran_i: f64,
    eta2: f64,
    p_perm: f64,
    q_bh: f64,
    spatial_score: f64,
}

/// Resolve AnnData path relative to common locations around the run TOML.
pub fn resolve_run_adata_path(cfg: &SpaceshipConfig, run_toml: &Path) -> anyhow::Result<PathBuf> {
    let raw = expand_user_path(cfg.resolve_adata_path().as_str());
    if raw.trim().is_empty() {
        bail!("data.adata_path is empty in {}", run_toml.display());
    }
    let p = PathBuf::from(&raw);
    if p.is_file() {
        return Ok(p);
    }
    let toml_dir = run_toml.parent().unwrap_or_else(|| Path::new("."));
    let candidates = [
        toml_dir.join(&raw),
        toml_dir
            .parent()
            .unwrap_or(toml_dir)
            .join(&raw),
        PathBuf::from(&raw),
    ];
    for c in &candidates {
        if c.is_file() {
            return Ok(c.clone());
        }
    }
    bail!(
        "could not find AnnData {:?} (tried cwd and paths relative to {})",
        raw,
        toml_dir.display()
    )
}

fn list_betadata_feathers(output_dir: &Path) -> anyhow::Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(output_dir)
        .with_context(|| format!("read output dir {}", output_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(gene) = name.strip_suffix("_betadata.feather") {
            if !gene.is_empty() && path.is_file() {
                out.push((gene.to_string(), path));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn spatial_knn_indices(spatial: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    let n = spatial.nrows();
    let kk = k.min(n.saturating_sub(1));
    (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = spatial.row(i);
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d = (xi[0] - spatial[(j, 0)]).hypot(xi[1] - spatial[(j, 1)]);
                    (d, j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.into_iter().take(kk).map(|(_, j)| j).collect()
        })
        .collect()
}

fn moran_i(values: &[f64], knn: &[Vec<usize>]) -> f64 {
    let n = values.len();
    if n < 3 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut wsum = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let zi = values[i] - mean;
        den += zi * zi;
        for &j in &knn[i] {
            num += zi * (values[j] - mean);
            wsum += 1.0;
        }
    }
    if den <= 1e-18 || wsum <= 0.0 {
        return 0.0;
    }
    (n as f64 / wsum) * (num / den)
}

fn eta2_spatial_grid(values: &[f64], spatial: &Array2<f64>, grid: usize) -> f64 {
    let n = values.len();
    if n < 2 || grid < 2 {
        return 0.0;
    }
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for i in 0..n {
        xmin = xmin.min(spatial[(i, 0)]);
        xmax = xmax.max(spatial[(i, 0)]);
        ymin = ymin.min(spatial[(i, 1)]);
        ymax = ymax.max(spatial[(i, 1)]);
    }
    let dx = (xmax - xmin).max(1e-9);
    let dy = (ymax - ymin).max(1e-9);
    let mut buckets: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    for i in 0..n {
        let gx = (((spatial[(i, 0)] - xmin) / dx) * grid as f64).floor() as usize;
        let gy = (((spatial[(i, 1)] - ymin) / dy) * grid as f64).floor() as usize;
        let gx = gx.min(grid - 1);
        let gy = gy.min(grid - 1);
        buckets.entry((gx, gy)).or_default().push(values[i]);
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let sst = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    if sst <= 1e-18 {
        return 0.0;
    }
    let mut ssb = 0.0;
    for vals in buckets.values() {
        if vals.is_empty() {
            continue;
        }
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        ssb += vals.len() as f64 * (m - mean).powi(2);
    }
    (ssb / sst).clamp(0.0, 1.0)
}

fn mad(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = v[v.len() / 2];
    let mut abs: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    abs[abs.len() / 2]
}

fn bh_fdr(pvals: &[f64]) -> Vec<f64> {
    let n = pvals.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pvals[a].partial_cmp(&pvals[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut q = vec![0.0; n];
    let mut running = f64::INFINITY;
    for (rank_rev, &idx) in order.iter().enumerate().rev() {
        let rank = n - rank_rev;
        let val = (pvals[idx] * n as f64 / rank as f64).min(1.0);
        running = running.min(val);
        q[idx] = running;
    }
    q
}

fn permute_moran_p(
    values: &[f64],
    knn: &[Vec<usize>],
    observed: f64,
    n_perm: usize,
    seed: u64,
) -> f64 {
    if n_perm == 0 {
        // Heuristic: treat high Moran's I as significant for ranking when perms disabled.
        return if observed > 0.1 { 0.01 } else { 0.5 };
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut buf = values.to_vec();
    let mut ge = 0usize;
    for _ in 0..n_perm {
        for i in (1..buf.len()).rev() {
            let j = rng.gen_range(0..=i);
            buf.swap(i, j);
        }
        let m = moran_i(&buf, knn);
        if m >= observed {
            ge += 1;
        }
    }
    (1 + ge) as f64 / (1 + n_perm) as f64
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n != b.len() || n < 2 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n as f64;
    let mb = b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da <= 1e-18 || db <= 1e-18 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn greedy_decorrelate(mut cands: Vec<FeatureCandidate>, corr_max: f64) -> Vec<FeatureCandidate> {
    cands.sort_by(|a, b| {
        b.spatial_score
            .partial_cmp(&a.spatial_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<FeatureCandidate> = Vec::new();
    for c in cands {
        let redundant = kept
            .iter()
            .any(|k| pearson(&k.values, &c.values).abs() >= corr_max);
        if !redundant {
            kept.push(c);
        }
    }
    kept
}

fn load_features_csv(path: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut rdr = csv::Reader::from_reader(f);
    let headers = rdr.headers()?.clone();
    let gene_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("gene"))
        .context("features CSV needs a gene column")?;
    let feat_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("feature"))
        .context("features CSV needs a feature column")?;
    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        out.push((rec[gene_idx].to_string(), rec[feat_idx].to_string()));
    }
    Ok(out)
}

fn zscore_columns(mut x: Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    if n == 0 {
        return x;
    }
    for j in 0..x.ncols() {
        let col = x.column(j);
        let mean = col.mean().unwrap_or(0.0);
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt().max(1e-8);
        for i in 0..n {
            x[(i, j)] = (x[(i, j)] - mean) / sd;
        }
    }
    x
}

/// Dense PCA via covariance eigendecomposition (feature Gram matrix).
pub fn dense_pca(x: &Array2<f64>, n_comps: usize) -> anyhow::Result<Array2<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    anyhow::ensure!(n >= 2 && p >= 1, "PCA needs n>=2 and p>=1");
    let k = n_comps.min(p).min(n - 1).max(1);
    let mut centered = x.clone();
    for j in 0..p {
        let mean = centered.column(j).mean().unwrap_or(0.0);
        for i in 0..n {
            centered[(i, j)] -= mean;
        }
    }
    // C = X^T X  (p × p)
    let mut c = Array2::<f64>::zeros((p, p));
    for a in 0..p {
        for b in a..p {
            let mut s = 0.0;
            for i in 0..n {
                s += centered[(i, a)] * centered[(i, b)];
            }
            c[(a, b)] = s;
            c[(b, a)] = s;
        }
    }
    // Power iteration with deflation for top-k eigenvectors of C.
    let mut components = Array2::<f64>::zeros((p, k));
    let mut work = c.clone();
    let mut rng = StdRng::seed_from_u64(42);
    for comp in 0..k {
        let mut v = Array1::<f64>::from_shape_fn(p, |_| rng.r#gen::<f64>() - 0.5);
        let mut norm = v.dot(&v).sqrt().max(1e-12);
        v /= norm;
        for _ in 0..80 {
            let mut w = Array1::<f64>::zeros(p);
            for i in 0..p {
                let mut s = 0.0;
                for j in 0..p {
                    s += work[(i, j)] * v[j];
                }
                w[i] = s;
            }
            // orthogonalize against previous
            for prev in 0..comp {
                let mut dot = 0.0;
                for i in 0..p {
                    dot += w[i] * components[(i, prev)];
                }
                for i in 0..p {
                    w[i] -= dot * components[(i, prev)];
                }
            }
            norm = w.dot(&w).sqrt().max(1e-12);
            v = w / norm;
        }
        for i in 0..p {
            components[(i, comp)] = v[i];
        }
        // deflate: C <- C - λ v v^T
        let mut cv = Array1::<f64>::zeros(p);
        for i in 0..p {
            let mut s = 0.0;
            for j in 0..p {
                s += work[(i, j)] * v[j];
            }
            cv[i] = s;
        }
        let lambda = v.dot(&cv);
        for i in 0..p {
            for j in 0..p {
                work[(i, j)] -= lambda * v[i] * v[j];
            }
        }
    }
    // scores = centered * components
    let mut scores = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for comp in 0..k {
            let mut s = 0.0;
            for j in 0..p {
                s += centered[(i, j)] * components[(j, comp)];
            }
            scores[(i, comp)] = s;
        }
    }
    Ok(scores)
}

fn mean_silhouette(pca: &Array2<f64>, labels: &[String]) -> anyhow::Result<f64> {
    let n = pca.nrows();
    anyhow::ensure!(n == labels.len(), "pca/labels length mismatch");
    let mut map: HashMap<&str, usize> = HashMap::new();
    let mut codes = Vec::with_capacity(n);
    for lab in labels {
        let next = map.len();
        let code = *map.entry(lab.as_str()).or_insert(next);
        codes.push(code);
    }
    let k = map.len();
    if k < 2 {
        return Ok(0.0);
    }
    // Subsample for O(n²) silhouette on large n.
    let idx: Vec<usize> = if n > 1200 {
        let mut rng = StdRng::seed_from_u64(0);
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..idx.len()).rev() {
            let j = rng.gen_range(0..=i);
            idx.swap(i, j);
        }
        idx.truncate(1200);
        idx.sort_unstable();
        idx
    } else {
        (0..n).collect()
    };
    let mut total = 0.0;
    let mut counted = 0usize;
    for &ii in &idx {
        let ci = codes[ii];
        let mut sum_same = 0.0;
        let mut n_same = 0usize;
        let mut best_other = f64::INFINITY;
        let mut other_sums = vec![0.0; k];
        let mut other_ns = vec![0usize; k];
        for &jj in &idx {
            if ii == jj {
                continue;
            }
            let mut d2 = 0.0;
            for c in 0..pca.ncols() {
                let d = pca[(ii, c)] - pca[(jj, c)];
                d2 += d * d;
            }
            let d = d2.sqrt();
            let cj = codes[jj];
            if cj == ci {
                sum_same += d;
                n_same += 1;
            } else {
                other_sums[cj] += d;
                other_ns[cj] += 1;
            }
        }
        let a = if n_same == 0 {
            0.0
        } else {
            sum_same / n_same as f64
        };
        for c in 0..k {
            if c == ci || other_ns[c] == 0 {
                continue;
            }
            best_other = best_other.min(other_sums[c] / other_ns[c] as f64);
        }
        if !best_other.is_finite() {
            continue;
        }
        let denom = a.max(best_other).max(1e-12);
        total += (best_other - a) / denom;
        counted += 1;
    }
    if counted == 0 {
        Ok(0.0)
    } else {
        Ok(total / counted as f64)
    }
}

fn count_unique(labels: &[String]) -> usize {
    let mut s = std::collections::BTreeSet::new();
    for l in labels {
        s.insert(l.as_str());
    }
    s.len()
}

fn resolution_grid(min: f64, max: f64, step: f64) -> Vec<f64> {
    let mut out = Vec::new();
    if step <= 0.0 || max < min {
        return out;
    }
    let mut r = min;
    while r <= max + 1e-9 {
        out.push((r * 1000.0).round() / 1000.0);
        r += step;
    }
    if out.is_empty() {
        out.push(min);
    }
    out
}

fn pick_labels_by_silhouette(
    graph: &FuzzyGraph,
    pca: &Array2<f64>,
    params: &MicronichesParams,
) -> anyhow::Result<(Vec<String>, f64, f64, Vec<ResolutionSweepRow>)> {
    if let Some(res) = params.leiden_resolution {
        let labels = leiden_labels_from_graph(graph, res, params.leiden_max_iter);
        let sil = mean_silhouette(pca, &labels)?;
        let n_clusters = count_unique(&labels);
        return Ok((
            labels,
            res,
            sil,
            vec![ResolutionSweepRow {
                resolution: res,
                silhouette: sil,
                n_clusters,
            }],
        ));
    }
    let grid = resolution_grid(
        params.resolution_min,
        params.resolution_max,
        params.resolution_step,
    );
    let mut best_labels = Vec::new();
    let mut best_sil = f64::NEG_INFINITY;
    let mut best_res = grid[0];
    let mut sweep = Vec::new();
    for &res in &grid {
        let labels = leiden_labels_from_graph(graph, res, params.leiden_max_iter);
        let n_clusters = count_unique(&labels);
        let sil = if n_clusters < 2 {
            0.0
        } else {
            mean_silhouette(pca, &labels)?
        };
        sweep.push(ResolutionSweepRow {
            resolution: res,
            silhouette: sil,
            n_clusters,
        });
        if sil > best_sil || (sil == best_sil && n_clusters > count_unique(&best_labels)) {
            best_sil = sil;
            best_res = res;
            best_labels = labels;
        }
    }
    if best_labels.is_empty() {
        best_labels = leiden_labels_from_graph(graph, best_res, params.leiden_max_iter);
        best_sil = mean_silhouette(pca, &best_labels)?;
    }
    Ok((best_labels, best_res, best_sil, sweep))
}

fn score_feature_matrix(
    feathers: &[(String, PathBuf)],
    obs_names: &[String],
    cluster_keys: &[String],
    spatial: &Array2<f64>,
    params: &MicronichesParams,
) -> anyhow::Result<Vec<FeatureCandidate>> {
    let knn = spatial_knn_indices(spatial, params.spatial_k);
    let mut scored: Vec<FeatureCandidate> = Vec::new();

    for (gene, path) in feathers {
        let cols = betadata_feather_plottable_columns(path.to_str().unwrap_or_default())?;
        for feature in cols {
            let vals_f32 = betadata_feather_per_cell_column(
                path.to_str().unwrap_or_default(),
                &feature,
                obs_names,
                cluster_keys,
            )?;
            let values: Vec<f64> = vals_f32.iter().map(|v| *v as f64).collect();
            let m = mad(&values);
            if m <= 1e-12 {
                continue;
            }
            let mi = moran_i(&values, &knn);
            let e2 = eta2_spatial_grid(&values, spatial, params.spatial_grid);
            let seed = params
                .seed
                .wrapping_add(gene.len() as u64)
                .wrapping_mul(31)
                .wrapping_add(feature.len() as u64);
            let p = permute_moran_p(&values, &knn, mi, params.moran_n_perm, seed);
            scored.push(FeatureCandidate {
                gene: gene.clone(),
                feature,
                values,
                mad: m,
                moran_i: mi,
                eta2: e2,
                p_perm: p,
                q_bh: 1.0,
                spatial_score: mi.max(0.0) * e2,
            });
        }
    }

    let pvals: Vec<f64> = scored.iter().map(|c| c.p_perm).collect();
    let q = bh_fdr(&pvals);
    for (c, qq) in scored.iter_mut().zip(q) {
        c.q_bh = qq;
    }
    let mut kept: Vec<_> = scored
        .iter()
        .filter(|c| c.q_bh <= params.q_bh_max && c.spatial_score > 0.0)
        .cloned()
        .collect();
    if kept.is_empty() {
        // Tiny / noisy runs: keep top spatial scores regardless of FDR.
        let mut ranked = scored;
        ranked.sort_by(|a, b| {
            b.spatial_score
                .partial_cmp(&a.spatial_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        kept = ranked.into_iter().take(32).collect();
    }
    if kept.is_empty() {
        bail!("no spatially informative β features found");
    }
    kept = greedy_decorrelate(kept, params.corr_max);
    if let Some(max_f) = params.max_features {
        if kept.len() > max_f {
            kept.truncate(max_f);
        }
    }
    Ok(kept)
}

fn load_specified_features(
    feathers: &[(String, PathBuf)],
    wanted: &[(String, String)],
    obs_names: &[String],
    cluster_keys: &[String],
) -> anyhow::Result<Vec<FeatureCandidate>> {
    let by_gene: HashMap<&str, &PathBuf> = feathers.iter().map(|(g, p)| (g.as_str(), p)).collect();
    let mut out = Vec::new();
    for (gene, feature) in wanted {
        let Some(path) = by_gene.get(gene.as_str()) else {
            continue;
        };
        let vals_f32 = betadata_feather_per_cell_column(
            path.to_str().unwrap_or_default(),
            feature,
            obs_names,
            cluster_keys,
        )?;
        let values: Vec<f64> = vals_f32.iter().map(|v| *v as f64).collect();
        let feature_mad = mad(&values);
        if feature_mad <= 1e-12 {
            continue;
        }
        out.push(FeatureCandidate {
            gene: gene.clone(),
            feature: feature.clone(),
            values,
            mad: feature_mad,
            moran_i: 0.0,
            eta2: 0.0,
            p_perm: 0.0,
            q_bh: 0.0,
            spatial_score: 0.0,
        });
    }
    if out.is_empty() {
        bail!("none of the requested features were found in betadata feathers");
    }
    Ok(out)
}

fn matrix_from_features(feats: &[FeatureCandidate]) -> Array2<f64> {
    let n = feats.first().map(|f| f.values.len()).unwrap_or(0);
    let p = feats.len();
    let mut x = Array2::<f64>::zeros((n, p));
    for (j, f) in feats.iter().enumerate() {
        for i in 0..n {
            x[(i, j)] = f.values[i];
        }
    }
    x
}

/// Run the microniche discovery pipeline from a training repro TOML.
pub fn run_microniches(
    run_toml: &Path,
    params: &MicronichesParams,
    out_dir: Option<&Path>,
) -> anyhow::Result<MicronichesResult> {
    crate::ensure_process_env();
    let cfg = SpaceshipConfig::from_file(run_toml)?;
    let output_dir = cfg.resolve_training_output_dir(run_toml);
    let adata_path = resolve_run_adata_path(&cfg, run_toml)?;
    let adata = AnnData::<H5>::open(H5::open(
        adata_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("AnnData path must be UTF-8"))?,
    )?)?;

    let all_obs = adata.obs_names().into_vec();
    let spatial_full = load_spatial_coords_f64(&adata)?;
    let obs_df = adata.read_obs()?;
    let annot_series = obs_df
        .column(params.annot_col.as_str())
        .with_context(|| format!("obs column {:?} not found", params.annot_col))?
        .as_materialized_series();
    let mut annot = Vec::with_capacity(all_obs.len());
    for i in 0..all_obs.len() {
        annot.push(obs_series_row_str(annot_series, i)?);
    }
    let betadata_key_col =
        resolve_betadata_cluster_key_column(&obs_df, cfg.data.cluster_annot.as_str());
    let cluster_keys_full =
        betadata_cluster_keys_from_obs_dataframe(&obs_df, betadata_key_col.as_str())?;

    let subset: Vec<usize> = match &params.cell_type {
        Some(ct) => annot
            .iter()
            .enumerate()
            .filter(|(_, a)| a.as_str() == ct.as_str())
            .map(|(i, _)| i)
            .collect(),
        None => (0..all_obs.len()).collect(),
    };
    if subset.is_empty() {
        bail!(
            "no cells matched {}={:?}",
            params.annot_col,
            params.cell_type
        );
    }

    let obs_names: Vec<String> = subset.iter().map(|&i| all_obs[i].clone()).collect();
    let cluster_keys: Vec<String> = subset
        .iter()
        .map(|&i| cluster_keys_full[i].clone())
        .collect();
    let mut spatial = Array2::<f64>::zeros((subset.len(), 2));
    for (row, &i) in subset.iter().enumerate() {
        spatial[(row, 0)] = spatial_full[(i, 0)];
        spatial[(row, 1)] = spatial_full[(i, 1)];
    }

    let mut feathers = list_betadata_feathers(&output_dir)?;
    if let Some(max_g) = params.max_genes {
        feathers.truncate(max_g);
    }
    if feathers.is_empty() {
        bail!(
            "no *_betadata.feather files under {}",
            output_dir.display()
        );
    }

    eprintln!(
        "get-microniches: {} cells · {} betadata genes · {}…",
        obs_names.len(),
        feathers.len(),
        if params.features_csv.is_some() {
            "loading feature CSV"
        } else {
            "spatial β filter"
        }
    );

    let kept = if let Some(csv) = &params.features_csv {
        let wanted = load_features_csv(csv)?;
        load_specified_features(&feathers, &wanted, &obs_names, &cluster_keys)?
    } else {
        score_feature_matrix(&feathers, &obs_names, &cluster_keys, &spatial, params)?
    };
    eprintln!("get-microniches: kept {} β features", kept.len());

    let x = zscore_columns(matrix_from_features(&kept));
    let n_pcs = params.n_pcs.min(x.ncols()).min(x.nrows().saturating_sub(1)).max(1);
    let pca = dense_pca(&x, n_pcs)?;
    eprintln!(
        "get-microniches: PCA {}×{} → fuzzy graph (k={})…",
        pca.nrows(),
        pca.ncols(),
        params.n_neighbors
    );
    let graph = fuzzy_graph_from_pca(
        &pca,
        pca.ncols(),
        params.n_neighbors,
        params.ef_construction,
    )?;
    let optimized = params.leiden_resolution.is_none();
    let (labels, chosen_res, sil, sweep) = pick_labels_by_silhouette(&graph, &pca, params)?;
    let n_clusters = count_unique(&labels);
    eprintln!(
        "get-microniches: {} niches · resolution={:.3} · silhouette={:.3}",
        n_clusters, chosen_res, sil
    );

    let dest = out_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| output_dir.join("microniches"));
    std::fs::create_dir_all(&dest)?;

    let kept_features: Vec<KeptBetaFeature> = kept
        .iter()
        .map(|f| KeptBetaFeature {
            gene: f.gene.clone(),
            feature: f.feature.clone(),
            moran_i: f.moran_i,
            eta2: f.eta2,
            q_bh: f.q_bh,
            spatial_score: f.spatial_score,
        })
        .collect();

    let summary = MicronichesSummary {
        n_cells: obs_names.len(),
        n_kept_features: kept_features.len(),
        n_pcs: pca.ncols(),
        n_clusters,
        chosen_resolution: chosen_res,
        silhouette: sil,
        cell_type_subset: params.cell_type.clone(),
        annot_col: params.annot_col.clone(),
        optimized_by_silhouette: optimized,
        resolution_sweep: sweep,
        output_dir: dest.display().to_string(),
    };

    write_outputs(&dest, &obs_names, &labels, &spatial, &pca, &kept_features, &summary)?;

    Ok(MicronichesResult {
        cell_ids: obs_names,
        labels,
        pca,
        spatial,
        kept_features,
        summary,
    })
}

fn write_outputs(
    dest: &Path,
    cell_ids: &[String],
    labels: &[String],
    spatial: &Array2<f64>,
    pca: &Array2<f64>,
    kept: &[KeptBetaFeature],
    summary: &MicronichesSummary,
) -> anyhow::Result<()> {
    {
        let mut w = csv::Writer::from_path(dest.join("microniche_labels.csv"))?;
        w.write_record(["cell_id", "microniche", "x", "y"])?;
        for i in 0..cell_ids.len() {
            w.write_record([
                cell_ids[i].as_str(),
                labels[i].as_str(),
                &spatial[(i, 0)].to_string(),
                &spatial[(i, 1)].to_string(),
            ])?;
        }
        w.flush()?;
    }
    {
        let mut w = csv::Writer::from_path(dest.join("kept_beta_features.csv"))?;
        w.write_record([
            "gene",
            "feature",
            "moran_I",
            "eta2",
            "q_bh",
            "spatial_score",
        ])?;
        for f in kept {
            w.write_record([
                f.gene.as_str(),
                f.feature.as_str(),
                &format!("{:.6}", f.moran_i),
                &format!("{:.6}", f.eta2),
                &format!("{:.6}", f.q_bh),
                &format!("{:.6}", f.spatial_score),
            ])?;
        }
        w.flush()?;
    }
    {
        let mut w = csv::Writer::from_path(dest.join("resolution_sweep.csv"))?;
        w.write_record(["resolution", "silhouette", "n_clusters"])?;
        for r in &summary.resolution_sweep {
            w.write_record([
                &format!("{:.4}", r.resolution),
                &format!("{:.6}", r.silhouette),
                &r.n_clusters.to_string(),
            ])?;
        }
        w.flush()?;
    }
    {
        let mut f = File::create(dest.join("summary.json"))?;
        serde_json::to_writer_pretty(&mut f, summary)?;
        f.write_all(b"\n")?;
    }
    // Save PCA as a lightweight feather for downstream use.
    let pc_names: Vec<String> = (0..pca.ncols()).map(|i| format!("PC{i}")).collect();
    write_betadata_feather(
        dest.join("microniche_pca.feather").to_str().unwrap_or("microniche_pca.feather"),
        "CellID",
        cell_ids,
        &pc_names,
        pca,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::data::ArrayData;
    use anndata::AxisArraysOp;
    use polars::prelude::{DataFrame, NamedFrom, Series};

    fn write_toy_run(dir: &Path) -> anyhow::Result<PathBuf> {
        crate::ensure_process_env();
        std::fs::create_dir_all(dir)?;
        let h5ad = dir.join("toy.h5ad");
        let a = AnnData::<H5>::new(&h5ad)?;
        let n = 60usize;
        let obs_names: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
        a.set_obs_names(obs_names.clone().into())?;
        a.set_var_names(vec!["G1".into()].into())?;
        let mut cell_types = Vec::with_capacity(n);
        let mut xy = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let niche = i / 20; // 0,1,2
            cell_types.push(if i < 40 { "Alpha" } else { "Beta" }.to_string());
            xy[(i, 0)] = (niche as f64) * 10.0 + (i % 20) as f64 * 0.1;
            xy[(i, 1)] = (i % 20) as f64 * 0.1;
        }
        let obs = DataFrame::new(vec![Series::new("cell_type".into(), cell_types).into()])?;
        a.set_obs(obs)?;
        a.obsm().add("spatial", ArrayData::from(xy))?;
        let x = Array2::<f64>::from_elem((n, 1), 1.0);
        a.set_x(ArrayData::from(x))?;
        a.close()?;

        // Three spatially structured β features + one noise feature.
        let mut data = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let niche = (i / 20) as f64;
            data[(i, 0)] = niche + 0.01 * (i as f64);
            data[(i, 1)] = (2.0 - niche) + 0.01 * (i as f64);
            data[(i, 2)] = ((i / 20) == 1) as i32 as f64;
            data[(i, 3)] = ((i * 17) % 7) as f64; // noise
        }
        write_betadata_feather(
            dir.join("GENE1_betadata.feather").to_str().unwrap(),
            "CellID",
            &obs_names,
            &[
                "beta_L1$R1".into(),
                "beta_L2$R2".into(),
                "beta_TF".into(),
                "beta_noise".into(),
            ],
            &data,
        )?;

        let repro = dir.join("spacetravlr_run_repro.toml");
        let mut f = File::create(&repro)?;
        write!(
            f,
            r#"
[data]
adata_path = "{h5ad}"
layer = "X"
cluster_annot = "cell_type"

[execution]
output_dir = "{out}"
n_parallel = 1
write_minimal_repro_h5ad = false
stale_lock_secs = 0
"#,
            h5ad = h5ad.display(),
            out = dir.display()
        )?;
        Ok(repro)
    }

    #[test]
    fn dense_pca_reduces_dims() {
        let x = Array2::from_shape_fn((30, 5), |(i, j)| (i * j) as f64 * 0.1);
        let pca = dense_pca(&x, 3).unwrap();
        assert_eq!(pca.nrows(), 30);
        assert_eq!(pca.ncols(), 3);
    }

    #[test]
    fn moran_high_for_spatial_block() {
        let mut spatial = Array2::<f64>::zeros((40, 2));
        let mut values = vec![0.0; 40];
        for i in 0..40 {
            spatial[(i, 0)] = (i / 20) as f64 * 5.0;
            spatial[(i, 1)] = (i % 20) as f64;
            values[i] = (i / 20) as f64;
        }
        let knn = spatial_knn_indices(&spatial, 4);
        let mi = moran_i(&values, &knn);
        assert!(mi > 0.5, "expected high Moran, got {mi}");
    }

    #[test]
    fn end_to_end_toy_microniches() {
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_microniches_toy_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let repro = write_toy_run(&dir).unwrap();
        let mut params = MicronichesParams::default();
        params.cell_type = Some("Alpha".into());
        params.moran_n_perm = 19;
        params.resolution_min = 0.3;
        params.resolution_max = 1.2;
        params.resolution_step = 0.3;
        params.n_neighbors = 8;
        params.n_pcs = 3;
        params.q_bh_max = 0.2;
        let out = dir.join("out_micro");
        let res = run_microniches(&repro, &params, Some(&out)).expect("run microniches");
        assert_eq!(res.summary.n_cells, 40);
        assert!(res.summary.n_kept_features >= 1);
        assert!(res.summary.n_clusters >= 2);
        assert!(out.join("microniche_labels.csv").is_file());
        assert!(out.join("summary.json").is_file());
        assert!(out.join("kept_beta_features.csv").is_file());
        assert!(res.summary.optimized_by_silhouette);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fixed_resolution_skips_sweep_flag() {
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_microniches_fixed_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let repro = write_toy_run(&dir).unwrap();
        let mut params = MicronichesParams::default();
        params.leiden_resolution = Some(0.6);
        params.moran_n_perm = 9;
        params.q_bh_max = 0.25;
        params.n_neighbors = 8;
        let res = run_microniches(&repro, &params, Some(&dir.join("fixed"))).unwrap();
        assert!(!res.summary.optimized_by_silhouette);
        assert!((res.summary.chosen_resolution - 0.6).abs() < 1e-9);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
