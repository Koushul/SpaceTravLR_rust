//! Tissue-structure references for non-spatial received-ligand inference.
//!
//! SpaceTravLR's spatial received ligands are
//! `received[i,l] = (1/N) Σ_j scale·exp(-d(i,j)² / 2r²)·expr[j,l]`.
//! Grouping senders by cell type yields
//! `received[i,l] = Σ_t S[i,t] · (weighted mixture of type-t expression)`,
//! where `S[i,t] = (1/N) Σ_{j∈t} scale·exp(-d(i,j)² / 2r²)`.
//!
//! A [`TissueStructureRef`] stores type-conditional expectations of those
//! Gaussian weight masses (and soft/hard neighbor counts) learned from a
//! spatial reference. Non-spatial query cells of type `c` then receive
//! `Σ_t Ŝ[c,t] · μ_query[t,l]` using query type-mean ligand expression.

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Learned tissue neighborhood structure for a matched tissue type.
#[derive(Debug, Clone)]
pub struct TissueStructureRef {
    pub cell_types: Vec<String>,
    /// Mean Gaussian weight mass `Ŝ[receiver, sender]` including `(1/N_ref)·scale·exp(...)`.
    pub mean_weight_mass: Array2<f64>,
    /// Mean soft neighbor mass `Σ_{j∈sender} scale·exp(...)` (no `1/N`).
    pub mean_soft_counts: Array2<f64>,
    /// Mean hard neighbor counts within `hard_radius` (default = Gaussian `radius`).
    pub mean_hard_counts: Array2<f64>,
    pub radius: f64,
    pub scale_factor: f64,
    pub hard_radius: f64,
    pub n_ref_cells: usize,
    /// Per-receiver-type cell counts in the reference.
    pub ref_type_counts: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TissueStructureRefJson {
    cell_types: Vec<String>,
    mean_weight_mass: Vec<Vec<f64>>,
    mean_soft_counts: Vec<Vec<f64>>,
    mean_hard_counts: Vec<Vec<f64>>,
    radius: f64,
    scale_factor: f64,
    hard_radius: f64,
    n_ref_cells: usize,
    ref_type_counts: Vec<usize>,
}

fn array2_to_vec(a: &Array2<f64>) -> Vec<Vec<f64>> {
    a.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn vec_to_array2(v: &[Vec<f64>]) -> anyhow::Result<Array2<f64>> {
    if v.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }
    let nrows = v.len();
    let ncols = v[0].len();
    let mut out = Array2::zeros((nrows, ncols));
    for (i, row) in v.iter().enumerate() {
        if row.len() != ncols {
            anyhow::bail!("ragged matrix row {i}");
        }
        for (j, &val) in row.iter().enumerate() {
            out[[i, j]] = val;
        }
    }
    Ok(out)
}

impl TissueStructureRef {
    fn to_json_value(&self) -> TissueStructureRefJson {
        TissueStructureRefJson {
            cell_types: self.cell_types.clone(),
            mean_weight_mass: array2_to_vec(&self.mean_weight_mass),
            mean_soft_counts: array2_to_vec(&self.mean_soft_counts),
            mean_hard_counts: array2_to_vec(&self.mean_hard_counts),
            radius: self.radius,
            scale_factor: self.scale_factor,
            hard_radius: self.hard_radius,
            n_ref_cells: self.n_ref_cells,
            ref_type_counts: self.ref_type_counts.clone(),
        }
    }

    fn from_json_value(v: TissueStructureRefJson) -> anyhow::Result<Self> {
        Ok(Self {
            cell_types: v.cell_types,
            mean_weight_mass: vec_to_array2(&v.mean_weight_mass)?,
            mean_soft_counts: vec_to_array2(&v.mean_soft_counts)?,
            mean_hard_counts: vec_to_array2(&v.mean_hard_counts)?,
            radius: v.radius,
            scale_factor: v.scale_factor,
            hard_radius: v.hard_radius,
            n_ref_cells: v.n_ref_cells,
            ref_type_counts: v.ref_type_counts,
        })
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(&self.to_json_value())?;
        std::fs::write(path.as_ref(), text)?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let v: TissueStructureRefJson = serde_json::from_str(&text)?;
        Self::from_json_value(v)
    }
}

/// Per-cell structure features before type pooling.
#[derive(Debug, Clone)]
pub struct CellStructureWeights {
    /// `S[i, t]` with `(1/N)·scale·exp` normalization matching `calculate_weighted_ligands`.
    pub weight_mass: Array2<f64>,
    /// Soft counts without `1/N`.
    pub soft_counts: Array2<f64>,
    /// Hard neighbor counts within cutoff (excludes self).
    pub hard_counts: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct StructureBuildArgs<'a> {
    pub xy: &'a Array2<f64>,
    pub cell_types: &'a [String],
    pub radius: f64,
    pub scale_factor: f64,
    pub hard_radius: Option<f64>,
}

fn type_index_map(cell_types: &[String]) -> (Vec<String>, Vec<usize>) {
    // BTreeSet key order ⇒ deterministic alphabetical type indexing for transfer.
    let names: Vec<String> = cell_types
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let remap: HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let indices: Vec<usize> = cell_types
        .iter()
        .map(|t| remap[t.as_str()])
        .collect();
    (names, indices)
}

/// Per-cell Gaussian type weights / neighbor counts on a spatial reference.
pub fn compute_cell_structure_weights(args: StructureBuildArgs<'_>) -> anyhow::Result<CellStructureWeights> {
    let n = args.xy.nrows();
    if n == 0 {
        anyhow::bail!("compute_cell_structure_weights: empty coordinates");
    }
    if args.cell_types.len() != n {
        anyhow::bail!(
            "cell_types length {} != n_cells {}",
            args.cell_types.len(),
            n
        );
    }
    if !(args.radius.is_finite() && args.radius > 0.0) {
        anyhow::bail!("radius must be finite and > 0");
    }
    let (type_names, type_idx) = type_index_map(args.cell_types);
    let n_types = type_names.len();
    let inv_2r2 = -1.0 / (2.0 * args.radius * args.radius);
    let hard_r = args
        .hard_radius
        .unwrap_or(args.radius)
        .max(0.0);
    let hard_r2 = hard_r * hard_r;
    let n_inv = 1.0 / n as f64;
    let scale = args.scale_factor;

    let mut weight_mass = Array2::<f64>::zeros((n, n_types));
    let mut soft_counts = Array2::<f64>::zeros((n, n_types));
    let mut hard_counts = Array2::<f64>::zeros((n, n_types));

    weight_mass
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(soft_counts.axis_iter_mut(Axis(0)).into_par_iter())
        .zip(hard_counts.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate()
        .for_each(|(i, ((mut wrow, mut srow), mut hrow))| {
            let xi = args.xy[[i, 0]];
            let yi = args.xy[[i, 1]];
            for j in 0..n {
                let dx = xi - args.xy[[j, 0]];
                let dy = yi - args.xy[[j, 1]];
                let d2 = dx * dx + dy * dy;
                let t = type_idx[j];
                let w = scale * (d2 * inv_2r2).exp();
                wrow[t] += w;
                srow[t] += w;
                if i != j && d2 <= hard_r2 {
                    hrow[t] += 1.0;
                }
            }
            for t in 0..n_types {
                wrow[t] *= n_inv;
            }
        });

    Ok(CellStructureWeights {
        weight_mass,
        soft_counts,
        hard_counts,
    })
}

/// Build a type-conditional tissue structure reference from spatial data.
pub fn build_tissue_structure_ref(args: StructureBuildArgs<'_>) -> anyhow::Result<TissueStructureRef> {
    let n = args.xy.nrows();
    let (type_names, type_idx) = type_index_map(args.cell_types);
    let n_types = type_names.len();
    let hard_radius = args.hard_radius.unwrap_or(args.radius);
    let cell = compute_cell_structure_weights(StructureBuildArgs {
        hard_radius: Some(hard_radius),
        ..args
    })?;

    let mut mean_weight_mass = Array2::<f64>::zeros((n_types, n_types));
    let mut mean_soft_counts = Array2::<f64>::zeros((n_types, n_types));
    let mut mean_hard_counts = Array2::<f64>::zeros((n_types, n_types));
    let mut ref_type_counts = vec![0usize; n_types];

    for i in 0..n {
        ref_type_counts[type_idx[i]] += 1;
    }

    for recv in 0..n_types {
        let mut count = 0.0;
        for i in 0..n {
            if type_idx[i] != recv {
                continue;
            }
            count += 1.0;
            for send in 0..n_types {
                mean_weight_mass[[recv, send]] += cell.weight_mass[[i, send]];
                mean_soft_counts[[recv, send]] += cell.soft_counts[[i, send]];
                mean_hard_counts[[recv, send]] += cell.hard_counts[[i, send]];
            }
        }
        if count > 0.0 {
            let inv = 1.0 / count;
            for send in 0..n_types {
                mean_weight_mass[[recv, send]] *= inv;
                mean_soft_counts[[recv, send]] *= inv;
                mean_hard_counts[[recv, send]] *= inv;
            }
        }
    }

    Ok(TissueStructureRef {
        cell_types: type_names,
        mean_weight_mass,
        mean_soft_counts,
        mean_hard_counts,
        radius: args.radius,
        scale_factor: args.scale_factor,
        hard_radius,
        n_ref_cells: n,
        ref_type_counts,
    })
}

/// Mean ligand expression per cell type (`types × ligands`).
pub fn type_mean_expression(
    expr: &Array2<f64>,
    cell_types: &[String],
    type_names: &[String],
) -> anyhow::Result<Array2<f64>> {
    let n = expr.nrows();
    let n_lig = expr.ncols();
    if cell_types.len() != n {
        anyhow::bail!("type_mean_expression: length mismatch");
    }
    let idx_map: HashMap<&str, usize> = type_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut sums = Array2::<f64>::zeros((type_names.len(), n_lig));
    let mut counts = vec![0.0f64; type_names.len()];
    for i in 0..n {
        let Some(&t) = idx_map.get(cell_types[i].as_str()) else {
            continue;
        };
        counts[t] += 1.0;
        for k in 0..n_lig {
            sums[[t, k]] += expr[[i, k]];
        }
    }
    for t in 0..type_names.len() {
        if counts[t] > 0.0 {
            let inv = 1.0 / counts[t];
            for k in 0..n_lig {
                sums[[t, k]] *= inv;
            }
        }
    }
    Ok(sums)
}

fn map_query_types(
    query_types: &[String],
    ref_types: &[String],
) -> anyhow::Result<Vec<Option<usize>>> {
    let map: HashMap<&str, usize> = ref_types
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    Ok(query_types
        .iter()
        .map(|t| map.get(t.as_str()).copied())
        .collect())
}

/// Infer per-cell structure weights by matching query expression to reference cells.
///
/// For each query cell of type `c`, find the `k` nearest reference cells (preferring the
/// same type) in cosine fingerprint space and average their Gaussian weight-mass rows.
/// This recovers within-type niche variation when transcriptional state correlates with
/// local neighborhood composition.
pub fn infer_weight_mass_expression_matched(
    ref_weight_mass: &Array2<f64>,
    ref_type_idx: &[usize],
    ref_fingerprint: &Array2<f64>,
    query_fingerprint: &Array2<f64>,
    query_type_idx: &[Option<usize>],
    k: usize,
) -> anyhow::Result<Array2<f64>> {
    let n_ref = ref_weight_mass.nrows();
    let n_types = ref_weight_mass.ncols();
    let n_q = query_fingerprint.nrows();
    if ref_fingerprint.nrows() != n_ref {
        anyhow::bail!("ref fingerprint / weight_mass row mismatch");
    }
    if ref_fingerprint.ncols() != query_fingerprint.ncols() {
        anyhow::bail!("fingerprint gene dimension mismatch");
    }
    if query_type_idx.len() != n_q {
        anyhow::bail!("query_type_idx length mismatch");
    }
    if ref_type_idx.len() != n_ref {
        anyhow::bail!("ref_type_idx length mismatch");
    }
    let kk = k.max(1).min(n_ref);
    let mut out = Array2::<f64>::zeros((n_q, n_types));

    // Pre-normalize fingerprints for cosine similarity.
    let mut ref_norm = ref_fingerprint.clone();
    for mut row in ref_norm.axis_iter_mut(Axis(0)) {
        let nrm = row.dot(&row).sqrt().max(1e-12);
        row.mapv_inplace(|v| v / nrm);
    }
    let mut query_norm = query_fingerprint.clone();
    for mut row in query_norm.axis_iter_mut(Axis(0)) {
        let nrm = row.dot(&row).sqrt().max(1e-12);
        row.mapv_inplace(|v| v / nrm);
    }

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let q = query_norm.row(i);
            let prefer = query_type_idx[i];
            let mut scored: Vec<(f64, usize)> = Vec::with_capacity(n_ref);
            for j in 0..n_ref {
                let sim = q.dot(&ref_norm.row(j));
                let boost = if prefer == Some(ref_type_idx[j]) {
                    1.0
                } else {
                    0.0
                };
                scored.push((sim + boost, j));
            }
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let take = kk.min(scored.len());
            if take == 0 {
                return;
            }
            let inv = 1.0 / take as f64;
            for &(_, j) in scored.iter().take(take) {
                for t in 0..n_types {
                    row[t] += ref_weight_mass[[j, t]] * inv;
                }
            }
        });
    Ok(out)
}

/// Apply per-cell weight mass to type-mean ligand expression.
pub fn received_from_weight_mass(
    weight_mass: &Array2<f64>,
    type_means: &Array2<f64>,
) -> anyhow::Result<Array2<f64>> {
    received_ligands_type_mean_oracle(weight_mass, type_means)
}

/// Infer received ligands for non-spatial cells using a tissue structure reference.
///
/// `type_means` must be ordered as `structure.cell_types` (rows) × ligands (cols).
/// Missing query types (not in the reference) receive zeros.
pub fn infer_received_ligands_from_structure(
    structure: &TissueStructureRef,
    query_cell_types: &[String],
    type_means: &Array2<f64>,
) -> anyhow::Result<Array2<f64>> {
    let n = query_cell_types.len();
    let n_lig = type_means.ncols();
    if type_means.nrows() != structure.cell_types.len() {
        anyhow::bail!(
            "type_means rows {} != structure types {}",
            type_means.nrows(),
            structure.cell_types.len()
        );
    }
    let mapped = map_query_types(query_cell_types, &structure.cell_types)?;
    let mut out = Array2::<f64>::zeros((n, n_lig));
    let n_types = structure.cell_types.len();

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let Some(recv) = mapped[i] else {
                return;
            };
            for send in 0..n_types {
                let w = structure.mean_weight_mass[[recv, send]];
                if w == 0.0 {
                    continue;
                }
                for k in 0..n_lig {
                    row[k] += w * type_means[[send, k]];
                }
            }
        });
    Ok(out)
}

/// Oracle type-mean approximation using per-cell spatial structure weights.
///
/// `received_hat[i,l] = Σ_t S[i,t] · μ[t,l]` — isolates expression heterogeneity
/// error from structure-pooling error.
pub fn received_ligands_type_mean_oracle(
    cell_weights: &Array2<f64>,
    type_means: &Array2<f64>,
) -> anyhow::Result<Array2<f64>> {
    let n = cell_weights.nrows();
    let n_types = cell_weights.ncols();
    let n_lig = type_means.ncols();
    if type_means.nrows() != n_types {
        anyhow::bail!("type_means / cell_weights type dimension mismatch");
    }
    let mut out = Array2::<f64>::zeros((n, n_lig));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for t in 0..n_types {
                let w = cell_weights[[i, t]];
                if w == 0.0 {
                    continue;
                }
                for k in 0..n_lig {
                    row[k] += w * type_means[[t, k]];
                }
            }
        });
    Ok(out)
}

/// Infer expected soft/hard neighbor counts by type for query cells.
pub fn infer_neighbor_composition(
    structure: &TissueStructureRef,
    query_cell_types: &[String],
) -> anyhow::Result<(Array2<f64>, Array2<f64>)> {
    let n = query_cell_types.len();
    let n_types = structure.cell_types.len();
    let mapped = map_query_types(query_cell_types, &structure.cell_types)?;
    let mut soft = Array2::<f64>::zeros((n, n_types));
    let mut hard = Array2::<f64>::zeros((n, n_types));
    for (i, recv) in mapped.into_iter().enumerate() {
        let Some(r) = recv else { continue };
        for t in 0..n_types {
            soft[[i, t]] = structure.mean_soft_counts[[r, t]];
            hard[[i, t]] = structure.mean_hard_counts[[r, t]];
        }
    }
    Ok((soft, hard))
}

/// Abundance-only baseline: replace spatial structure with global type frequencies
/// times the mean total Gaussian mass of the matched receiver type.
pub fn abundance_baseline_weight_mass(structure: &TissueStructureRef) -> Array2<f64> {
    let n_types = structure.cell_types.len();
    let total_n: f64 = structure.ref_type_counts.iter().map(|c| *c as f64).sum();
    let freqs: Vec<f64> = structure
        .ref_type_counts
        .iter()
        .map(|c| *c as f64 / total_n.max(1.0))
        .collect();
    let mut out = Array2::<f64>::zeros((n_types, n_types));
    for recv in 0..n_types {
        let total_mass: f64 = structure.mean_weight_mass.row(recv).sum();
        for send in 0..n_types {
            out[[recv, send]] = total_mass * freqs[send];
        }
    }
    out
}

/// Infer ligands with an arbitrary receiver×sender weight matrix (for baselines).
pub fn infer_received_with_weight_matrix(
    weight_mass: &Array2<f64>,
    type_names: &[String],
    query_cell_types: &[String],
    type_means: &Array2<f64>,
) -> anyhow::Result<Array2<f64>> {
    let tmp = TissueStructureRef {
        cell_types: type_names.to_vec(),
        mean_weight_mass: weight_mass.clone(),
        mean_soft_counts: Array2::zeros(weight_mass.raw_dim()),
        mean_hard_counts: Array2::zeros(weight_mass.raw_dim()),
        radius: 1.0,
        scale_factor: 1.0,
        hard_radius: 1.0,
        n_ref_cells: 0,
        ref_type_counts: vec![0; type_names.len()],
    };
    infer_received_ligands_from_structure(&tmp, query_cell_types, type_means)
}

/// Align a reference onto a query type vocabulary (intersection). Types missing
/// from either side are dropped. Returns remapped structure + shared type names.
pub fn restrict_structure_to_types(
    structure: &TissueStructureRef,
    keep_types: &[String],
) -> anyhow::Result<TissueStructureRef> {
    let keep: Vec<String> = keep_types
        .iter()
        .filter(|t| structure.cell_types.iter().any(|s| s == *t))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    if keep.is_empty() {
        anyhow::bail!("no overlapping cell types between structure and keep_types");
    }
    let old_idx: HashMap<&str, usize> = structure
        .cell_types
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let n = keep.len();
    let mut mean_weight_mass = Array2::zeros((n, n));
    let mut mean_soft_counts = Array2::zeros((n, n));
    let mut mean_hard_counts = Array2::zeros((n, n));
    let mut ref_type_counts = vec![0usize; n];
    for (ni, rt) in keep.iter().enumerate() {
        let oi = old_idx[rt.as_str()];
        ref_type_counts[ni] = structure.ref_type_counts[oi];
        for (nj, st) in keep.iter().enumerate() {
            let oj = old_idx[st.as_str()];
            mean_weight_mass[[ni, nj]] = structure.mean_weight_mass[[oi, oj]];
            mean_soft_counts[[ni, nj]] = structure.mean_soft_counts[[oi, oj]];
            mean_hard_counts[[ni, nj]] = structure.mean_hard_counts[[oi, oj]];
        }
    }
    Ok(TissueStructureRef {
        cell_types: keep,
        mean_weight_mass,
        mean_soft_counts,
        mean_hard_counts,
        radius: structure.radius,
        scale_factor: structure.scale_factor,
        hard_radius: structure.hard_radius,
        n_ref_cells: structure.n_ref_cells,
        ref_type_counts,
    })
}

/// Row-wise Pearson correlation between prediction and truth matrices.
pub fn column_pearson(pred: &Array2<f64>, truth: &Array2<f64>) -> Array1<f64> {
    assert_eq!(pred.raw_dim(), truth.raw_dim());
    let n_lig = pred.ncols();
    let mut out = Array1::zeros(n_lig);
    for k in 0..n_lig {
        out[k] = pearson(pred.column(k).as_slice().unwrap(), truth.column(k).as_slice().unwrap());
    }
    out
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den == 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

/// MAE / RMSE / mean relative absolute error across all entries.
pub fn matrix_error_metrics(pred: &Array2<f64>, truth: &Array2<f64>) -> (f64, f64, f64) {
    assert_eq!(pred.raw_dim(), truth.raw_dim());
    let mut mae = 0.0;
    let mut mse = 0.0;
    let mut rel = 0.0;
    let mut n = 0.0;
    for (p, t) in pred.iter().zip(truth.iter()) {
        let e = (p - t).abs();
        mae += e;
        mse += e * e;
        rel += e / t.abs().max(1e-8);
        n += 1.0;
    }
    if n == 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    (mae / n, (mse / n).sqrt(), rel / n)
}

/// Mean cosine similarity of per-cell neighbor-composition vectors.
pub fn mean_composition_cosine(pred: &Array2<f64>, truth: &Array2<f64>) -> f64 {
    assert_eq!(pred.raw_dim(), truth.raw_dim());
    let n = pred.nrows();
    if n == 0 {
        return f64::NAN;
    }
    let mut acc = 0.0;
    for i in 0..n {
        let p = pred.row(i);
        let t = truth.row(i);
        let mut num = 0.0;
        let mut dp = 0.0;
        let mut dt = 0.0;
        for k in 0..p.len() {
            num += p[k] * t[k];
            dp += p[k] * p[k];
            dt += t[k] * t[k];
        }
        let den = (dp * dt).sqrt();
        if den > 0.0 {
            acc += num / den;
        }
    }
    acc / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ligand::calculate_weighted_ligands;
    use ndarray::array;

    #[test]
    fn structure_inference_matches_type_mean_spatial_when_homogeneous() {
        // Two types on a line; expression constant within type ⇒ structure
        // inference with true type means equals exact spatial received ligands.
        let xy = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
        ];
        let types = vec![
            "A".into(),
            "A".into(),
            "B".into(),
            "B".into(),
        ];
        let lig = array![
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 5.0],
            [0.0, 5.0],
        ];
        let radius = 2.0;
        let scale = 1.0;
        let truth = calculate_weighted_ligands(&xy, &lig, radius, scale);
        let st = build_tissue_structure_ref(StructureBuildArgs {
            xy: &xy,
            cell_types: &types,
            radius,
            scale_factor: scale,
            hard_radius: Some(radius),
        })
        .unwrap();
        let means = type_mean_expression(&lig, &types, &st.cell_types).unwrap();
        let pred = infer_received_ligands_from_structure(&st, &types, &means).unwrap();
        let cell = compute_cell_structure_weights(StructureBuildArgs {
            xy: &xy,
            cell_types: &types,
            radius,
            scale_factor: scale,
            hard_radius: Some(radius),
        })
        .unwrap();
        let oracle = received_ligands_type_mean_oracle(&cell.weight_mass, &means).unwrap();
        for i in 0..4 {
            for k in 0..2 {
                assert!(
                    (oracle[[i, k]] - truth[[i, k]]).abs() < 1e-9,
                    "oracle mismatch at [{i},{k}]"
                );
            }
        }
        // Type-pooled structure should be close (identical types share neighborhoods loosely).
        let (mae, _, _) = matrix_error_metrics(&pred, &truth);
        assert!(mae < 0.05, "mae={mae}");
    }

    #[test]
    fn hard_counts_exclude_self_and_respect_radius() {
        let xy = array![[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]];
        let types = vec!["A".into(), "B".into(), "A".into()];
        let st = build_tissue_structure_ref(StructureBuildArgs {
            xy: &xy,
            cell_types: &types,
            radius: 10.0,
            scale_factor: 1.0,
            hard_radius: Some(2.0),
        })
        .unwrap();
        // Cell 0 (A): neighbor cell 1 (B) within 2, cell 2 far.
        let cell = compute_cell_structure_weights(StructureBuildArgs {
            xy: &xy,
            cell_types: &types,
            radius: 10.0,
            scale_factor: 1.0,
            hard_radius: Some(2.0),
        })
        .unwrap();
        let b = st.cell_types.iter().position(|t| t == "B").unwrap();
        let a = st.cell_types.iter().position(|t| t == "A").unwrap();
        assert!((cell.hard_counts[[0, b]] - 1.0).abs() < 1e-12);
        assert!((cell.hard_counts[[0, a]] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn restrict_structure_intersection() {
        let xy = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let types = vec!["A".into(), "B".into(), "C".into()];
        let st = build_tissue_structure_ref(StructureBuildArgs {
            xy: &xy,
            cell_types: &types,
            radius: 5.0,
            scale_factor: 1.0,
            hard_radius: None,
        })
        .unwrap();
        let sub = restrict_structure_to_types(&st, &["C".into(), "A".into(), "Z".into()]).unwrap();
        assert_eq!(sub.cell_types, vec!["A".to_string(), "C".to_string()]);
        assert_eq!(sub.mean_weight_mass.nrows(), 2);
    }
}
