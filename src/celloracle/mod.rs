//! CellOracle-style GRN inference: TF-only priors from SpaceTravLR `*_network.parquet`,
//! sklearn-compatible Bayesian ridge for coefficients and edge statistics.

mod sklearn_bayesian_ridge;

use anyhow::Context;
use ndarray::Array2;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::sync::Arc;

pub use sklearn_bayesian_ridge::{bayesian_ridge_fit, BayesianRidgeFit};

#[derive(Clone, Debug)]
pub struct LinkRow {
    pub source: String,
    pub target: String,
    pub cluster: Arc<str>,
    pub coef_mean: f64,
    pub coef_abs: f64,
    pub coef_variance: f64,
    pub p: f64,
    pub neg_log_p: f64,
}

pub fn scale_gem_no_center(gem: &Array2<f64>) -> Array2<f64> {
    let (n, p) = gem.dim();
    let mut out = gem.clone();
    for j in 0..p {
        let col = gem.column(j);
        let m = col.sum() / n as f64;
        let var = col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
        let sc = var.sqrt();
        let scale = if sc < 1e-12 { 1.0 } else { sc };
        for i in 0..n {
            out[[i, j]] /= scale;
        }
    }
    out
}

pub fn build_coef_matrix(var_names: &[String], links: &[LinkRow]) -> Array2<f64> {
    let n = var_names.len();
    let gene_to_idx: HashMap<&str, usize> = var_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut mat = Array2::zeros((n, n));
    for r in links {
        let Some(&ri) = gene_to_idx.get(r.source.as_str()) else {
            continue;
        };
        let Some(&ci) = gene_to_idx.get(r.target.as_str()) else {
            continue;
        };
        mat[[ri, ci]] = r.coef_mean;
    }
    mat
}

pub fn write_coef_matrix_exports(
    path_prefix: &std::path::Path,
    var_names: &[String],
    mat: &Array2<f64>,
) -> anyhow::Result<()> {
    use ndarray_npy::write_npy;

    let base = path_prefix.to_string_lossy();
    let genes_p = format!("{base}_genes.json");
    let npy_p = format!("{base}_coef_matrix.npy");
    std::fs::write(&genes_p, serde_json::to_string_pretty(var_names)?)
        .with_context(|| format!("write {genes_p}"))?;
    write_npy(&npy_p, mat).with_context(|| format!("write {npy_p}"))?;
    Ok(())
}

pub fn filter_links_p_max(rows: Vec<LinkRow>, p_max: f64) -> Vec<LinkRow> {
    rows.into_iter().filter(|r| r.p <= p_max).collect()
}

fn two_sided_p_celloracle(normal: &Normal, coef_mean: f64, coef_variance: f64) -> (f64, f64) {
    let coef_abs = coef_mean.abs();
    let sig = coef_variance.sqrt().max(1e-300);
    let p = 2.0 * normal.cdf(-coef_abs / sig);
    let neg = -p.ln();
    (p, neg)
}

fn subset_rows(gem: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    let p = gem.ncols();
    let mut out = Array2::zeros((rows.len(), p));
    for (oi, &ri) in rows.iter().enumerate() {
        for j in 0..p {
            out[[oi, j]] = gem[[ri, j]];
        }
    }
    out
}

pub fn infer_grn_whole(
    gem: &Array2<f64>,
    gem_scaled: &Array2<f64>,
    var_names: &[String],
    tf_by_target: &HashMap<String, Vec<String>>,
) -> anyhow::Result<Vec<LinkRow>> {
    infer_grn_subset(gem, gem_scaled, var_names, tf_by_target, None, "all")
}

pub fn infer_grn_subset(
    gem: &Array2<f64>,
    gem_scaled: &Array2<f64>,
    var_names: &[String],
    tf_by_target: &HashMap<String, Vec<String>>,
    row_idx: Option<&[usize]>,
    cluster_label: &str,
) -> anyhow::Result<Vec<LinkRow>> {
    let n_cells = gem.nrows();
    anyhow::ensure!(
        gem_scaled.nrows() == n_cells && gem.ncols() == gem_scaled.ncols(),
        "gem / gem_scaled shape mismatch"
    );
    let gene_to_idx: HashMap<&str, usize> = var_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let gem_s: Cow<'_, Array2<f64>> = match row_idx {
        Some(idx) => Cow::Owned(subset_rows(gem, idx)),
        None => Cow::Borrowed(gem),
    };
    let gem_sc: Cow<'_, Array2<f64>> = match row_idx {
        Some(idx) => Cow::Owned(subset_rows(gem_scaled, idx)),
        None => Cow::Borrowed(gem_scaled),
    };

    let n_sub = gem_s.nrows();
    let var_set: HashSet<&str> = var_names.iter().map(|s| s.as_str()).collect();

    let targets: Vec<String> = tf_by_target
        .keys()
        .filter(|t| var_set.contains(t.as_str()))
        .cloned()
        .collect();

    let pb = indicatif::ProgressBar::new(targets.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} genes {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let cluster: Arc<str> = Arc::from(cluster_label);
    let normal = Normal::new(0.0, 1.0).expect("std normal");

    let links: Vec<LinkRow> = targets
        .par_iter()
        .flat_map_iter(|target| {
            let out = infer_one_target(
                &gene_to_idx,
                tf_by_target,
                var_names,
                &gem_s,
                &gem_sc,
                n_sub,
                target,
                &cluster,
                &normal,
            );
            pb.inc(1);
            out.into_iter()
        })
        .collect();

    pb.finish_with_message("done");
    Ok(links)
}

fn infer_one_target(
    gene_to_idx: &HashMap<&str, usize>,
    tf_by_target: &HashMap<String, Vec<String>>,
    var_names: &[String],
    gem_s: &Cow<'_, Array2<f64>>,
    gem_sc: &Cow<'_, Array2<f64>>,
    n_sub: usize,
    target: &String,
    cluster: &Arc<str>,
    normal: &Normal,
) -> Vec<LinkRow> {
    let Some(&ti) = gene_to_idx.get(target.as_str()) else {
        return Vec::new();
    };
    let Some(regs) = tf_by_target.get(target) else {
        return Vec::new();
    };
    let mut reggenes: Vec<usize> = regs
        .iter()
        .filter_map(|g| {
            if g == target {
                return None;
            }
            gene_to_idx.get(g.as_str()).copied()
        })
        .collect();
    reggenes.sort_unstable();
    reggenes.dedup();
    if reggenes.is_empty() || n_sub <= reggenes.len() {
        return Vec::new();
    }

    let p_feat = reggenes.len();
    let x = nalgebra::DMatrix::from_fn(n_sub, p_feat, |i, j| gem_sc[[i, reggenes[j]]]);
    let yv = nalgebra::DVector::from_iterator(n_sub, (0..n_sub).map(|i| gem_s[[i, ti]]));

    let Some(fit) = sklearn_bayesian_ridge::bayesian_ridge_fit(&x, &yv) else {
        return Vec::new();
    };

    let mut rows = Vec::with_capacity(reggenes.len());
    for (j, &gj) in reggenes.iter().enumerate() {
        let coef_mean = fit.coef[j];
        let v = fit.sigma_diag[j].max(0.0);
        let coef_abs = coef_mean.abs();
        let (p, neg_log_p) = two_sided_p_celloracle(normal, coef_mean, v);
        rows.push(LinkRow {
            source: var_names[gj].clone(),
            target: target.clone(),
            cluster: cluster.clone(),
            coef_mean,
            coef_abs,
            coef_variance: v,
            p,
            neg_log_p,
        });
    }
    rows
}

pub fn infer_grn_per_cluster(
    gem: &Array2<f64>,
    gem_scaled: &Array2<f64>,
    var_names: &[String],
    tf_by_target: &HashMap<String, Vec<String>>,
    obs_cluster: &[String],
) -> anyhow::Result<Vec<LinkRow>> {
    anyhow::ensure!(
        obs_cluster.len() == gem.nrows(),
        "obs_cluster length must match n_cells"
    );
    let mut by_label: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, lab) in obs_cluster.iter().enumerate() {
        by_label.entry(lab.clone()).or_default().push(i);
    }
    let pairs: Vec<(String, Vec<usize>)> = by_label.into_iter().collect();
    let results: Vec<anyhow::Result<Vec<LinkRow>>> = pairs
        .into_par_iter()
        .map(|(cluster, rows)| {
            infer_grn_subset(
                gem,
                gem_scaled,
                var_names,
                tf_by_target,
                Some(rows.as_slice()),
                cluster.as_str(),
            )
        })
        .collect();
    let mut all = Vec::new();
    for r in results {
        all.extend(r?);
    }
    Ok(all)
}

fn sorted_link_rows(rows: &[LinkRow]) -> Vec<LinkRow> {
    let mut v: Vec<LinkRow> = rows.iter().cloned().collect();
    v.sort_by(|a, b| {
        (&a.source, &a.target, a.cluster.as_ref()).cmp(&(&b.source, &b.target, b.cluster.as_ref()))
    });
    v
}

fn links_dataframe(rows: &[LinkRow]) -> anyhow::Result<polars::prelude::DataFrame> {
    use polars::prelude::*;

    let rows = sorted_link_rows(rows);
    let n = rows.len();
    let mut source = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);
    let mut cluster = Vec::with_capacity(n);
    let mut coef_mean = Vec::with_capacity(n);
    let mut coef_abs = Vec::with_capacity(n);
    let mut coef_variance = Vec::with_capacity(n);
    let mut p = Vec::with_capacity(n);
    let mut neg_log_p = Vec::with_capacity(n);
    for r in &rows {
        source.push(r.source.as_str());
        target.push(r.target.as_str());
        cluster.push(r.cluster.as_ref());
        coef_mean.push(r.coef_mean);
        coef_abs.push(r.coef_abs);
        coef_variance.push(r.coef_variance);
        p.push(r.p);
        neg_log_p.push(r.neg_log_p);
    }
    DataFrame::new(vec![
        Series::new("source".into(), source).into(),
        Series::new("target".into(), target).into(),
        Series::new("cluster".into(), cluster).into(),
        Series::new("coef_mean".into(), coef_mean).into(),
        Series::new("coef_abs".into(), coef_abs).into(),
        Series::new("coef_variance".into(), coef_variance).into(),
        Series::new("p".into(), p).into(),
        Series::new("neg_log_p".into(), neg_log_p).into(),
    ])
    .map_err(Into::into)
}

pub fn write_links_parquet(path: &std::path::Path, rows: &[LinkRow]) -> anyhow::Result<()> {
    use polars::prelude::*;

    let mut df = links_dataframe(rows)?;
    let mut f = std::fs::File::create(path).with_context(|| format!("create {:?}", path))?;
    ParquetWriter::new(&mut f).finish(&mut df)?;
    Ok(())
}

pub fn write_links_csv(path: &std::path::Path, rows: &[LinkRow]) -> anyhow::Result<()> {
    use polars::prelude::*;

    let mut df = links_dataframe(rows)?;
    let mut f = std::fs::File::create(path).with_context(|| format!("create {:?}", path))?;
    CsvWriter::new(&mut f)
        .include_header(true)
        .finish(&mut df)?;
    Ok(())
}

pub fn write_links_as_tf_priors_feather(path: &std::path::Path, rows: &[LinkRow]) -> anyhow::Result<()> {
    use polars::prelude::*;

    let rows = sorted_link_rows(rows);
    let n = rows.len();
    let mut source = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);
    let mut cell_type = Vec::with_capacity(n);
    for r in &rows {
        source.push(r.source.as_str());
        target.push(r.target.as_str());
        cell_type.push(r.cluster.as_ref());
    }
    let mut df = DataFrame::new(vec![
        Series::new("source".into(), source).into(),
        Series::new("target".into(), target).into(),
        Series::new("cell_type".into(), cell_type).into(),
    ])?;
    let f = File::create(path).with_context(|| format!("create {:?}", path))?;
    let mut w = IpcWriter::new(f).with_compression(Some(IpcCompression::LZ4));
    w.finish(&mut df)?;
    Ok(())
}
