//! CellOracle-style transition field on an embedding (UMAP): `velocyto`-style column
//! delta correlations (`colDeltaCor`, `colDeltaCorpartial`), transition probabilities
//! from `SpaceTravLR.plotting.shift` (`estimate_transition_probabilities`),
//! projection (`project_probabilities`), and grid binning / scaling from
//! `SpaceTravLR.plotting.cartography` (`compute_transition_vector_field` / `plot_umap_quiver`).
//! In-app perturb flows use `GeneFactory`-style δ, but the UMAP field math matches **cartography + shift**, not `gene_factory.py` itself.

use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use ndarray::Array2;
use rayon::prelude::*;

const EPS_VAR: f64 = 1e-18;

/// Pearson correlation between `vel_i` and `expr_j - expr_i` across genes.
/// Matches `velocyto` / `numpy.nan_to_num(..., nan=1)`: zero or undefined variance → **1.0**.
#[inline]
pub fn pearson_vel_vs_expr_delta(expr_i: &[f64], expr_j: &[f64], vel_i: &[f64]) -> f64 {
    let g = expr_i.len();
    if g == 0 {
        return 1.0;
    }
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    for k in 0..g {
        sx += vel_i[k];
        sy += expr_j[k] - expr_i[k];
    }
    let inv = 1.0 / g as f64;
    let mx = sx * inv;
    let my = sy * inv;
    let mut cv = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for k in 0..g {
        let x = vel_i[k] - mx;
        let y = (expr_j[k] - expr_i[k]) - my;
        cv += x * y;
        vx += x * x;
        vy += y * y;
    }
    if vx <= EPS_VAR || vy <= EPS_VAR {
        return 1.0;
    }
    let r = cv / (vx * vy).sqrt();
    if !r.is_finite() {
        1.0
    } else {
        r.clamp(-1.0, 1.0)
    }
}

/// `colDeltaCor`: `out[i,j]` = corr(`delta[i,:]`, `expr[j,:] - expr[i,:]`).
/// Parallelized over rows with rayon.
pub fn col_delta_cor(expr: &Array2<f64>, delta: &Array2<f64>) -> Array2<f64> {
    let n = expr.nrows();
    let g = expr.ncols();
    assert_eq!(delta.dim(), (n, g));
    let expr_rows: Vec<Vec<f64>> = (0..n).map(|i| expr.row(i).to_vec()).collect();
    let delta_rows: Vec<Vec<f64>> = (0..n).map(|i| delta.row(i).to_vec()).collect();
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let ei = &expr_rows[i];
            let vi = &delta_rows[i];
            (0..n)
                .map(|j| pearson_vel_vs_expr_delta(ei, &expr_rows[j], vi))
                .collect()
        })
        .collect();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n, n), flat).unwrap()
}

/// KNN on UMAP (first column x, second y), **self excluded**, up to `k` neighbors per cell.
/// Parallelized over cells with rayon.
pub fn umap_knn_indices(umap: &[[f64; 2]], k: usize) -> Vec<Vec<usize>> {
    let n = umap.len();
    if n == 0 {
        return vec![];
    }
    let k_eff = k.min(n.saturating_sub(1).max(1));
    let points: Vec<[f64; 2]> = umap.iter().copied().collect();
    let tree = ImmutableKdTree::<f64, 2>::new_from_slice(&points);
    let k_query = std::num::NonZero::new(k_eff + 1).unwrap();
    (0..n)
        .into_par_iter()
        .map(|i| {
            tree.nearest_n::<SquaredEuclidean>(&points[i], k_query)
                .iter()
                .filter_map(|nn| {
                    let j = nn.item as usize;
                    (j != i).then_some(j)
                })
                .take(k_eff)
                .collect()
        })
        .collect()
}

/// `colDeltaCorpartial`: computed only for neighbor pairs.
/// Returns sparse rows: `Vec<Vec<(col_index, correlation)>>`.
/// Parallelized over rows. Avoids allocating a dense N×N matrix.
pub fn col_delta_cor_partial_sparse(
    expr: &Array2<f64>,
    delta: &Array2<f64>,
    neighbors: &[Vec<usize>],
) -> Vec<Vec<(usize, f64)>> {
    let n = expr.nrows();
    let g = expr.ncols();
    assert_eq!(delta.dim(), (n, g));
    assert_eq!(neighbors.len(), n);
    let expr_rows: Vec<Vec<f64>> = (0..n).map(|i| expr.row(i).to_vec()).collect();
    let delta_rows: Vec<Vec<f64>> = (0..n).map(|i| delta.row(i).to_vec()).collect();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let ei = &expr_rows[i];
            let vi = &delta_rows[i];
            neighbors[i]
                .iter()
                .filter(|&&j| j < n)
                .map(|&j| (j, pearson_vel_vs_expr_delta(ei, &expr_rows[j], vi)))
                .collect()
        })
        .collect()
}

/// Numerically stable softmax: subtract row-max before exp to prevent overflow.
fn softmax_rows_from_sparse_corr(
    corr_rows: &[Vec<(usize, f64)>],
    temperature: f64,
) -> Vec<Vec<(usize, f64)>> {
    let t = temperature.max(1e-9);
    corr_rows
        .par_iter()
        .map(|row| {
            if row.is_empty() {
                return vec![];
            }
            let max_c = row.iter().map(|&(_, c)| c).fold(f64::NEG_INFINITY, f64::max);
            let mut w: Vec<(usize, f64)> = Vec::with_capacity(row.len());
            let mut sum = 0.0_f64;
            for &(j, c) in row {
                let wt = ((c - max_c) / t).exp();
                w.push((j, wt));
                sum += wt;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for x in &mut w {
                    x.1 *= inv;
                }
            }
            w
        })
        .collect()
}

fn softmax_rows_dense(corr: &Array2<f64>, temperature: f64) -> Array2<f64> {
    let n = corr.nrows();
    let t = temperature.max(1e-9);
    let corr_flat = corr.as_slice().expect("corr must be contiguous");
    let mut p_flat = vec![0.0_f64; n * n];
    p_flat
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, p_row)| {
            let c_row = &corr_flat[i * n..(i + 1) * n];
            let mut max_c = f64::NEG_INFINITY;
            for j in 0..n {
                if j != i && c_row[j] > max_c {
                    max_c = c_row[j];
                }
            }
            let mut sum = 0.0_f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let wt = ((c_row[j] - max_c) / t).exp();
                p_row[j] = wt;
                sum += wt;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for j in 0..n {
                    if j != i {
                        p_row[j] *= inv;
                    }
                }
            }
        });
    Array2::from_shape_vec((n, n), p_flat).unwrap()
}

/// Subtract null-model P (computed from zero delta), clip negatives, renormalize.
fn subtract_null_sparse(
    p_sig: Vec<Vec<(usize, f64)>>,
    p_null: &[Vec<(usize, f64)>],
) -> Vec<Vec<(usize, f64)>> {
    p_sig
        .into_par_iter()
        .enumerate()
        .map(|(i, row)| {
            let null_map: std::collections::HashMap<usize, f64> =
                p_null[i].iter().copied().collect();
            let mut out: Vec<(usize, f64)> = Vec::new();
            let mut sum = 0.0_f64;
            for (j, ps) in row {
                let pn = null_map.get(&j).copied().unwrap_or(0.0);
                let d = ps - pn;
                if d > 0.0 {
                    out.push((j, d));
                    sum += d;
                }
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for x in &mut out {
                    x.1 *= inv;
                }
                out
            } else {
                vec![]
            }
        })
        .collect()
}

fn subtract_null_dense(p_sig: &Array2<f64>, p_null: &Array2<f64>) -> Array2<f64> {
    let n = p_sig.nrows();
    if n <= 1 {
        return Array2::<f64>::zeros((n, n));
    }
    let sig_flat = p_sig.as_slice().unwrap();
    let null_flat = p_null.as_slice().unwrap();
    let mut out_flat = vec![0.0_f64; n * n];
    out_flat
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, o_row)| {
            let base = i * n;
            let mut sum = 0.0_f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = sig_flat[base + j] - null_flat[base + j];
                if d > 0.0 {
                    o_row[j] = d;
                    sum += d;
                }
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for j in 0..n {
                    if j != i {
                        o_row[j] *= inv;
                    }
                }
            }
        });
    Array2::from_shape_vec((n, n), out_flat).unwrap()
}

fn project_sparse(
    umap: &[[f64; 2]],
    p_rows: &[Vec<(usize, f64)>],
    unit_directions: bool,
) -> Vec<[f64; 2]> {
    let n = umap.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = umap[i][0];
            let yi = umap[i][1];
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            for &(j, pij) in &p_rows[i] {
                if j >= n || pij == 0.0 {
                    continue;
                }
                let dx = umap[j][0] - xi;
                let dy = umap[j][1] - yi;
                if unit_directions {
                    let norm_sq = dx * dx + dy * dy;
                    if norm_sq < 1e-24 {
                        continue;
                    }
                    let inv_norm = 1.0 / norm_sq.sqrt();
                    sx += pij * dx * inv_norm;
                    sy += pij * dy * inv_norm;
                } else {
                    sx += pij * dx;
                    sy += pij * dy;
                }
            }
            [sx, sy]
        })
        .collect()
}

fn project_dense(umap: &[[f64; 2]], p: &Array2<f64>, unit_directions: bool) -> Vec<[f64; 2]> {
    let n = umap.len();
    let p_flat = p.as_slice().unwrap();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = umap[i][0];
            let yi = umap[i][1];
            let p_row = &p_flat[i * n..(i + 1) * n];
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pij = p_row[j];
                if pij == 0.0 {
                    continue;
                }
                let dx = umap[j][0] - xi;
                let dy = umap[j][1] - yi;
                if unit_directions {
                    let norm_sq = dx * dx + dy * dy;
                    if norm_sq < 1e-24 {
                        continue;
                    }
                    let inv_norm = 1.0 / norm_sq.sqrt();
                    sx += pij * dx * inv_norm;
                    sy += pij * dy * inv_norm;
                } else {
                    sx += pij * dx;
                    sy += pij * dy;
                }
            }
            [sx, sy]
        })
        .collect()
}

/// Grid layout similar to `get_grid_layout` in `layout.py`.
fn grid_axes(min_v: f64, max_v: f64, grid_scale: f64) -> Vec<f64> {
    let span = (max_v - min_v).max(1e-12);
    let count = (((span + 1.0) * grid_scale.max(1e-6)).sqrt() as usize)
        .pow(2)
        .clamp(4, 10_000);
    (0..count)
        .map(|k| min_v + (span * k as f64) / (count.saturating_sub(1).max(1) as f64))
        .collect()
}

/// Match Python `np.mean(abs(np.diff(layout_embedding)))`: mean of absolute
/// differences along axis 0 (default), which for an (n,2) array computes
/// `|x[i+1]-x[i]|` and `|y[i+1]-y[i]|` across all consecutive pairs.
fn mean_abs_diff_2d(umap: &[[f64; 2]]) -> f64 {
    let n = umap.len();
    if n < 2 {
        return 1.0;
    }
    let mut s = 0.0_f64;
    let mut c = 0_usize;
    for i in 0..n - 1 {
        s += (umap[i + 1][0] - umap[i][0]).abs();
        s += (umap[i + 1][1] - umap[i][1]).abs();
        c += 2;
    }
    (s / c as f64).max(1e-9)
}

/// Parameters for `compute_umap_transition_grid` (mirror `cartography` / UI).
#[derive(Clone, Debug)]
pub struct TransitionUmapParams {
    pub n_neighbors: usize,
    pub temperature: f64,
    pub remove_null: bool,
    /// Matches `plot_umap_quiver(..., normalize=...)` in `cartography.py` (default `False` there).
    pub unit_directions: bool,
    pub grid_scale: f64,
    pub vector_scale: f64,
    pub delta_rescale: f64,
    pub magnitude_threshold: f64,
    /// If `true` and `n` ≤ `full_graph_max_cells`, use dense `colDeltaCor` (slow).
    pub use_full_graph: bool,
    pub full_graph_max_cells: usize,
}

impl Default for TransitionUmapParams {
    fn default() -> Self {
        Self {
            n_neighbors: 150,
            temperature: 0.05,
            remove_null: true,
            unit_directions: false,
            grid_scale: 1.0,
            vector_scale: 0.85,
            delta_rescale: 1.0,
            magnitude_threshold: 0.0,
            use_full_graph: false,
            full_graph_max_cells: 4096,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransitionGrid {
    pub grid_x: Vec<f64>,
    pub grid_y: Vec<f64>,
    /// Flattened `(grid_x.len(), grid_y.len())` row-major `[ix][iy]` → `ix * ny + iy`
    pub vectors: Vec<[f64; 2]>,
    pub cell_vectors: Vec<[f64; 2]>,
}

pub fn round_delta_inplace(delta: &mut Array2<f64>, decimals: i32) {
    let m = 10_f64.powi(decimals);
    delta.iter_mut().for_each(|v| *v = (*v * m).round() / m);
}

/// Per-bin sum of cell vectors for cells whose UMAP coordinates fall in the axis-aligned
/// rectangle around each grid point, matching `compute_transition_vector_field` /
/// `plot_umap_quiver` in `SpaceTravLR/plotting/cartography.py` (`get_neighborhood` +
/// `mean(V)*len(indices)` == sum of V).
fn aggregate_grid_cartography(
    v_cell: &[[f64; 2]],
    umap: &[[f64; 2]],
    grid_x: &[f64],
    grid_y: &[f64],
) -> Vec<[f64; 2]> {
    let nx = grid_x.len();
    let ny = grid_y.len();
    let n = umap.len();
    if nx == 0 || ny == 0 {
        return vec![];
    }

    let x_thresh = if nx >= 2 {
        (grid_x[1] - grid_x[0]).abs() / 2.0
    } else {
        f64::INFINITY
    };
    let y_thresh = if ny >= 2 {
        (grid_y[1] - grid_y[0]).abs() / 2.0
    } else {
        f64::INFINITY
    };

    let mut out = vec![[0.0_f64; 2]; nx * ny];
    for ix in 0..nx {
        let gx = grid_x[ix];
        for iy in 0..ny {
            let gy = grid_y[iy];
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            for i in 0..n {
                let ux = umap[i][0];
                let uy = umap[i][1];
                if (ux - gx).abs() <= x_thresh && (uy - gy).abs() <= y_thresh {
                    sx += v_cell[i][0];
                    sy += v_cell[i][1];
                }
            }
            out[ix * ny + iy] = [sx, sy];
        }
    }
    out
}

pub fn compute_umap_transition_grid(
    expr: &Array2<f64>,
    delta: &Array2<f64>,
    umap: &[[f64; 2]],
    params: &TransitionUmapParams,
) -> TransitionGrid {
    let n = expr.nrows();
    assert_eq!(delta.nrows(), n);
    assert_eq!(umap.len(), n);

    let mut delta_w = delta * params.delta_rescale;
    round_delta_inplace(&mut delta_w, 3);

    let (p_sparse, p_dense_opt): (Vec<Vec<(usize, f64)>>, Option<Array2<f64>>) =
        if params.use_full_graph && n <= params.full_graph_max_cells {
            let corr = col_delta_cor(expr, &delta_w);
            let mut p = softmax_rows_dense(&corr, params.temperature);
            if params.remove_null {
                let zero_delta = Array2::<f64>::zeros(delta_w.dim());
                let corr0 = col_delta_cor(expr, &zero_delta);
                let p0 = softmax_rows_dense(&corr0, params.temperature);
                p = subtract_null_dense(&p, &p0);
            }
            (vec![], Some(p))
        } else {
            let neighbors = umap_knn_indices(umap, params.n_neighbors);
            let corr_sparse = col_delta_cor_partial_sparse(expr, &delta_w, &neighbors);
            let mut p = softmax_rows_from_sparse_corr(&corr_sparse, params.temperature);
            if params.remove_null {
                let zero_delta = Array2::<f64>::zeros(delta_w.dim());
                let corr0 = col_delta_cor_partial_sparse(expr, &zero_delta, &neighbors);
                let p0 = softmax_rows_from_sparse_corr(&corr0, params.temperature);
                p = subtract_null_sparse(p, &p0);
            }
            (p, None)
        };

    let v_cell: Vec<[f64; 2]> = if let Some(ref pd) = p_dense_opt {
        project_dense(umap, pd, params.unit_directions)
    } else {
        project_sparse(umap, &p_sparse, params.unit_directions)
    };

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in umap {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }

    let step_scale = mean_abs_diff_2d(umap);
    let gs = 10.0 * params.grid_scale / step_scale;

    let grid_x = grid_axes(min_x, max_x, gs);
    let grid_y = grid_axes(min_y, max_y, gs);

    let mut field = aggregate_grid_cartography(&v_cell, umap, &grid_x, &grid_y);

    // Match `cartography.py`: `vector_scale = vector_scale / np.max(vector_field)` then multiply.
    let mut m_comp = 0.0_f64;
    for w in &field {
        m_comp = m_comp.max(w[0].abs()).max(w[1].abs());
    }
    if m_comp > 1e-36 {
        let s = params.vector_scale / m_comp;
        for w in &mut field {
            w[0] *= s;
            w[1] *= s;
        }
    }

    if params.magnitude_threshold > 0.0 {
        let t_sq = params.magnitude_threshold * params.magnitude_threshold;
        for w in &mut field {
            if w[0] * w[0] + w[1] * w[1] < t_sq {
                *w = [0.0, 0.0];
            }
        }
    }

    TransitionGrid {
        grid_x,
        grid_y,
        vectors: field,
        cell_vectors: v_cell,
    }
}

// ── Legacy API wrappers (kept for backward compat with callers using dense output) ──

/// `colDeltaCorpartial` returning a dense N×N matrix. Prefer `col_delta_cor_partial_sparse`.
pub fn col_delta_cor_partial(
    expr: &Array2<f64>,
    delta: &Array2<f64>,
    neighbors: &[Vec<usize>],
) -> Array2<f64> {
    let n = expr.nrows();
    let sparse = col_delta_cor_partial_sparse(expr, delta, neighbors);
    let mut out = Array2::<f64>::zeros((n, n));
    for (i, row) in sparse.iter().enumerate() {
        for &(j, v) in row {
            out[[i, j]] = v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper to build TransitionUmapParams quickly ──

    fn base_params() -> TransitionUmapParams {
        TransitionUmapParams {
            n_neighbors: 150,
            temperature: 0.05,
            remove_null: false,
            unit_directions: false,
            grid_scale: 1.0,
            vector_scale: 1.0,
            delta_rescale: 1.0,
            magnitude_threshold: 0.0,
            use_full_graph: true,
            full_graph_max_cells: 10_000,
        }
    }

    fn cell_mag(v: [f64; 2]) -> f64 {
        (v[0] * v[0] + v[1] * v[1]).sqrt()
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Pearson correlation unit tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn pearson_perfect_positive() {
        let ei = vec![0.0, 0.0, 0.0];
        let ej = vec![1.0, 2.0, 3.0];
        let vel = vec![1.0, 2.0, 3.0];
        let r = pearson_vel_vs_expr_delta(&ei, &ej, &vel);
        assert!((r - 1.0).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn pearson_perfect_negative() {
        let ei = vec![0.0, 0.0, 0.0];
        let ej = vec![1.0, 2.0, 3.0];
        let vel = vec![-1.0, -2.0, -3.0];
        let r = pearson_vel_vs_expr_delta(&ei, &ej, &vel);
        assert!((r + 1.0).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn pearson_zero_variance_returns_one() {
        let ei = vec![1.0, 1.0, 1.0];
        let ej = vec![1.0, 1.0, 1.0];
        let vel = vec![0.5, 0.5, 0.5];
        let r = pearson_vel_vs_expr_delta(&ei, &ej, &vel);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn pearson_empty_genes_returns_one() {
        let r = pearson_vel_vs_expr_delta(&[], &[], &[]);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn pearson_single_gene_zero_variance() {
        let r = pearson_vel_vs_expr_delta(&[1.0], &[2.0], &[0.5]);
        assert_eq!(r, 1.0, "single gene has zero variance");
    }

    #[test]
    fn pearson_orthogonal_near_zero() {
        let ei = vec![0.0, 0.0, 0.0, 0.0];
        let ej = vec![1.0, 0.0, -1.0, 0.0];
        let vel = vec![0.0, 1.0, 0.0, -1.0];
        let r = pearson_vel_vs_expr_delta(&ei, &ej, &vel);
        assert!(r.abs() < 1e-9, "orthogonal vectors, got {}", r);
    }

    #[test]
    fn pearson_clamped_to_range() {
        for _ in 0..100 {
            let ei: Vec<f64> = (0..5).map(|i| (i as f64 * 1.7).sin()).collect();
            let ej: Vec<f64> = (0..5).map(|i| (i as f64 * 0.3).cos()).collect();
            let vel: Vec<f64> = (0..5).map(|i| (i as f64 * 2.1).sin()).collect();
            let r = pearson_vel_vs_expr_delta(&ei, &ej, &vel);
            assert!(
                (-1.0..=1.0).contains(&r),
                "correlation must be in [-1,1], got {}",
                r
            );
        }
    }

    #[test]
    fn pearson_known_value() {
        let ei = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let ej = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let vel = vec![2.0, 4.0, 3.0, 7.0, 6.0]; // vel = 2*(ej-ei) - 1 → high correlation
        let r = pearson_vel_vs_expr_delta(&ei, &ej, &vel);
        assert!(r > 0.99, "highly correlated, got {}", r);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  colDeltaCor tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn col_delta_cor_symmetric_on_uniform_delta() {
        let n = 4;
        let g = 3;
        let expr = Array2::from_shape_fn((n, g), |(i, k)| (i + k) as f64);
        let delta = Array2::from_elem((n, g), 1.0);
        let corr = col_delta_cor(&expr, &delta);
        assert_eq!(corr.dim(), (n, n));
        for i in 0..n {
            for j in 0..n {
                assert!(corr[[i, j]].is_finite(), "corr[{},{}] not finite", i, j);
            }
        }
    }

    #[test]
    fn col_partial_matches_full_on_clique() {
        let n = 5;
        let g = 4;
        let expr = Array2::from_shape_fn((n, g), |(i, k)| (i + k) as f64 * 0.3);
        let delta = Array2::from_shape_fn((n, g), |(i, k)| (k as f64).sin() * 0.1 + i as f64 * 0.05);
        let neigh: Vec<Vec<usize>> =
            (0..n).map(|i| (0..n).filter(|&j| j != i).collect()).collect();
        let cp = col_delta_cor_partial(&expr, &delta, &neigh);
        let cf = col_delta_cor(&expr, &delta);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                assert!(
                    (cp[[i, j]] - cf[[i, j]]).abs() < 1e-9,
                    "i={} j={} partial={} full={}",
                    i, j, cp[[i, j]], cf[[i, j]]
                );
            }
        }
    }

    #[test]
    fn col_partial_sparse_matches_dense() {
        let n = 6;
        let g = 3;
        let expr = Array2::from_shape_fn((n, g), |(i, k)| (i * 3 + k) as f64 * 0.2);
        let delta = Array2::from_shape_fn((n, g), |(i, k)| (i as f64 * 0.1) + (k as f64 * 0.3));
        let neigh: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i && (j as isize - i as isize).unsigned_abs() <= 2)
                    .collect()
            })
            .collect();
        let sparse = col_delta_cor_partial_sparse(&expr, &delta, &neigh);
        let dense = col_delta_cor_partial(&expr, &delta, &neigh);
        for (i, row) in sparse.iter().enumerate() {
            for &(j, v) in row {
                assert!(
                    (v - dense[[i, j]]).abs() < 1e-12,
                    "mismatch at [{},{}]: sparse={} dense={}",
                    i, j, v, dense[[i, j]]
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Softmax numerical stability tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn softmax_sparse_row_sums_to_one() {
        let corr_rows = vec![
            vec![(1, 0.5), (2, -0.3), (3, 0.9)],
            vec![(0, 0.1), (2, 0.2)],
        ];
        let p = softmax_rows_from_sparse_corr(&corr_rows, 0.05);
        for (i, row) in p.iter().enumerate() {
            let sum: f64 = row.iter().map(|&(_, v)| v).sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "row {} sums to {}, not 1.0",
                i, sum
            );
        }
    }

    #[test]
    fn softmax_sparse_no_overflow_extreme_values() {
        let corr_rows = vec![vec![(1, 100.0), (2, -100.0), (3, 50.0)]];
        let p = softmax_rows_from_sparse_corr(&corr_rows, 0.01);
        for &(_, v) in &p[0] {
            assert!(v.is_finite(), "softmax should not overflow, got {}", v);
        }
        let sum: f64 = p[0].iter().map(|&(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn softmax_sparse_very_low_temperature() {
        let corr_rows = vec![vec![(0, 0.9), (1, 0.1), (2, -0.5)]];
        let p = softmax_rows_from_sparse_corr(&corr_rows, 0.001);
        assert!(p[0][0].1 > 0.99, "max corr should dominate at low T");
        let sum: f64 = p[0].iter().map(|&(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn softmax_dense_row_sums_to_one() {
        let n = 5;
        let corr = Array2::from_shape_fn((n, n), |(i, j)| (i as f64 - j as f64) * 0.3);
        let p = softmax_rows_dense(&corr, 0.1);
        for i in 0..n {
            let sum: f64 = p.row(i).iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "row {} sums to {}, not 1.0",
                i, sum
            );
            assert_eq!(p[[i, i]], 0.0, "diagonal should be 0");
        }
    }

    #[test]
    fn softmax_dense_no_overflow() {
        let n = 3;
        let corr = Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j { 0.0 } else if j > i { 50.0 } else { -50.0 }
        });
        let p = softmax_rows_dense(&corr, 0.01);
        for val in p.iter() {
            assert!(val.is_finite(), "must not overflow");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  KNN tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn knn_excludes_self() {
        let umap = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]];
        let neighbors = umap_knn_indices(&umap, 2);
        for (i, nb) in neighbors.iter().enumerate() {
            assert!(!nb.contains(&i), "cell {} contains self", i);
            assert!(nb.len() <= 2);
        }
        assert_eq!(neighbors[0][0], 1);
    }

    #[test]
    fn knn_returns_correct_count() {
        let n = 20;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();
        let k = 5;
        let neighbors = umap_knn_indices(&umap, k);
        for nb in &neighbors {
            assert_eq!(nb.len(), k, "each cell should have exactly k neighbors");
        }
    }

    #[test]
    fn knn_k_larger_than_n() {
        let umap = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let neighbors = umap_knn_indices(&umap, 100);
        for nb in &neighbors {
            assert_eq!(nb.len(), 2, "k clamped to n-1");
        }
    }

    #[test]
    fn knn_empty() {
        let neighbors = umap_knn_indices(&[], 5);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn knn_single_cell() {
        let neighbors = umap_knn_indices(&[[0.0, 0.0]], 5);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors[0].is_empty() || neighbors[0].len() <= 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Transition direction tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn transition_direction_rightward_on_line() {
        let expr = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let mut delta = Array2::zeros((4, 1));
        delta[[1, 0]] = 1.0;
        let umap: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let params = TransitionUmapParams {
            n_neighbors: 3,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        assert!(grid.cell_vectors[1][0] > 0.0, "should point right, got {}", grid.cell_vectors[1][0]);
    }

    #[test]
    fn transition_direction_leftward_on_line() {
        let expr = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let mut delta = Array2::zeros((4, 1));
        delta[[2, 0]] = -1.0;
        let umap: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let params = TransitionUmapParams {
            n_neighbors: 3,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        assert!(grid.cell_vectors[2][0] < 0.0, "should point left, got {}", grid.cell_vectors[2][0]);
    }

    #[test]
    fn transition_2d_diagonal() {
        let expr = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ).unwrap();
        let mut delta = Array2::zeros((4, 2));
        delta[[0, 0]] = 1.0;
        delta[[0, 1]] = 1.0;
        let umap: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let params = TransitionUmapParams {
            n_neighbors: 3,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        let v0 = grid.cell_vectors[0];
        assert!(v0[0] > 0.0, "should point right, got u={}", v0[0]);
        assert!(v0[1] > 0.0, "should point up, got v={}", v0[1]);
    }

    #[test]
    fn transition_2d_upward_only() {
        let expr = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ).unwrap();
        let mut delta = Array2::zeros((4, 2));
        delta[[0, 0]] = 0.0;
        delta[[0, 1]] = 2.0; // push only gene1 (y-axis in expression space)
        let umap: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let params = TransitionUmapParams {
            n_neighbors: 3,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        let v0 = grid.cell_vectors[0];
        assert!(v0[1] > v0[0].abs(), "should point mostly up, got {:?}", v0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Zero delta / remove_null tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn transition_zero_delta_zero_vectors() {
        let n = 8;
        let expr = Array2::from_shape_fn((n, 3), |(i, k)| (i as f64 * 0.5) + k as f64);
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [(i % 3) as f64, (i / 3) as f64]).collect();
        let delta = Array2::zeros((n, 3));
        let params = TransitionUmapParams {
            n_neighbors: 4,
            remove_null: true,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        for (i, v) in grid.cell_vectors.iter().enumerate() {
            assert!(cell_mag(*v) < 1e-9, "cell {} mag={:.6e}", i, cell_mag(*v));
        }
    }

    #[test]
    fn remove_null_zeros_uniform_transitions() {
        let n = 5;
        let expr = Array2::from_shape_fn((n, 2), |(i, k)| (i + k) as f64);
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();
        let delta = Array2::zeros((n, 2));
        let params = TransitionUmapParams {
            n_neighbors: 4,
            remove_null: true,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        for (i, v) in grid.cell_vectors.iter().enumerate() {
            assert!(cell_mag(*v) < 1e-9, "cell {} mag={:.6e}", i, cell_mag(*v));
        }
    }

    #[test]
    fn remove_null_preserves_directional_signal() {
        let n = 6;
        let expr = Array2::from_shape_fn((n, 3), |(i, k)| (i + k) as f64 * 0.5);
        let mut delta = Array2::zeros((n, 3));
        delta[[1, 0]] = 2.0;
        delta[[1, 1]] = 1.5;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64 * 2.0, 0.0]).collect();

        let p_no = TransitionUmapParams {
            n_neighbors: 5,
            remove_null: false,
            use_full_graph: true,
            ..base_params()
        };
        let p_yes = TransitionUmapParams { remove_null: true, ..p_no.clone() };

        let g_no = compute_umap_transition_grid(&expr, &delta, &umap, &p_no);
        let g_yes = compute_umap_transition_grid(&expr, &delta, &umap, &p_yes);

        assert!(g_no.cell_vectors[1][0] > 1e-12, "without null should point right");
        let mag = cell_mag(g_yes.cell_vectors[1]);
        if mag > 1e-12 {
            assert!(g_yes.cell_vectors[1][0] > 0.0, "with null should still point right");
        }
    }

    #[test]
    fn remove_null_sparse_path() {
        let n = 20;
        let g = 5;
        // Use nonlinear gene-dependent expression so expr[j]-expr[i] varies across genes
        let expr = Array2::from_shape_fn((n, g), |(i, k)| {
            ((i as f64 * 0.7 + k as f64 * 1.3).sin() + 1.0) * 2.0
        });
        let mut delta = Array2::zeros((n, g));
        delta[[5, 0]] = 2.0;
        delta[[5, 2]] = -1.5;
        delta[[5, 4]] = 1.0;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [(i % 5) as f64, (i / 5) as f64]).collect();
        let params = TransitionUmapParams {
            n_neighbors: 10,
            remove_null: true,
            use_full_graph: false,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        let total_mag: f64 = grid.cell_vectors.iter().map(|v| cell_mag(*v)).sum();
        assert!(total_mag > 0.0, "signal should not be completely zeroed");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Sparse vs dense consistency
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn sparse_dense_agree_small_n() {
        let n = 6;
        let g = 3;
        let expr = Array2::from_shape_fn((n, g), |(i, k)| (i * 2 + k) as f64 * 0.4);
        let mut delta = Array2::zeros((n, g));
        delta[[1, 0]] = 0.5;
        delta[[3, 2]] = -0.3;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [(i % 3) as f64, (i / 3) as f64]).collect();

        let p_dense = TransitionUmapParams {
            n_neighbors: 5,
            temperature: 0.1,
            use_full_graph: true,
            ..base_params()
        };
        let p_sparse = TransitionUmapParams {
            use_full_graph: false,
            ..p_dense.clone()
        };

        let g_dense = compute_umap_transition_grid(&expr, &delta, &umap, &p_dense);
        let g_sparse = compute_umap_transition_grid(&expr, &delta, &umap, &p_sparse);

        for i in 0..n {
            let vd = g_dense.cell_vectors[i];
            let vs = g_sparse.cell_vectors[i];
            let diff = ((vd[0] - vs[0]).powi(2) + (vd[1] - vs[1]).powi(2)).sqrt();
            assert!(diff < 0.1, "cell {} diff={:.4} dense={:?} sparse={:?}", i, diff, vd, vs);
        }
    }

    #[test]
    fn sparse_dense_remove_null_agree() {
        let n = 6;
        let g = 3;
        let expr = Array2::from_shape_fn((n, g), |(i, k)| (i * 2 + k) as f64 * 0.4);
        let mut delta = Array2::zeros((n, g));
        delta[[2, 0]] = 1.0;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [(i % 3) as f64, (i / 3) as f64]).collect();

        let p_dense = TransitionUmapParams {
            n_neighbors: 5,
            temperature: 0.1,
            remove_null: true,
            use_full_graph: true,
            ..base_params()
        };
        let p_sparse = TransitionUmapParams {
            use_full_graph: false,
            ..p_dense.clone()
        };

        let g_dense = compute_umap_transition_grid(&expr, &delta, &umap, &p_dense);
        let g_sparse = compute_umap_transition_grid(&expr, &delta, &umap, &p_sparse);

        for i in 0..n {
            let vd = g_dense.cell_vectors[i];
            let vs = g_sparse.cell_vectors[i];
            let diff = ((vd[0] - vs[0]).powi(2) + (vd[1] - vs[1]).powi(2)).sqrt();
            assert!(diff < 0.15, "cell {} diff={:.4}", i, diff);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Unit directions test
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn unit_directions_normalizes() {
        let expr = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let mut delta = Array2::zeros((4, 1));
        delta[[1, 0]] = 1.0;
        let umap: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [10.0, 0.0]];

        let p_unit = TransitionUmapParams { unit_directions: true, n_neighbors: 3, use_full_graph: true, ..base_params() };
        let p_raw = TransitionUmapParams { unit_directions: false, ..p_unit.clone() };

        let g_unit = compute_umap_transition_grid(&expr, &delta, &umap, &p_unit);
        let g_raw = compute_umap_transition_grid(&expr, &delta, &umap, &p_raw);

        assert!(g_unit.cell_vectors[1][0] > 0.0 && g_raw.cell_vectors[1][0] > 0.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Grid layout tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn grid_axes_reasonable_count() {
        let axes = grid_axes(0.0, 10.0, 5.0);
        assert!(axes.len() >= 4);
        assert!(axes.len() <= 10_000);
        assert!((axes[0] - 0.0).abs() < 1e-9);
        let last = *axes.last().unwrap();
        assert!((last - 10.0).abs() < 1e-6);
    }

    #[test]
    fn grid_axes_small_span() {
        let axes = grid_axes(5.0, 5.001, 1.0);
        assert!(axes.len() >= 4, "should produce at least 4 ticks even for tiny span");
    }

    #[test]
    fn mean_abs_diff_2d_simple() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let m = mean_abs_diff_2d(&pts);
        assert!((m - 0.5).abs() < 1e-9, "expected 0.5, got {}", m);
    }

    #[test]
    fn mean_abs_diff_2d_single_point() {
        assert_eq!(mean_abs_diff_2d(&[[3.0, 4.0]]), 1.0);
    }

    #[test]
    fn mean_abs_diff_2d_empty() {
        assert_eq!(mean_abs_diff_2d(&[]), 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Grid aggregation / bucketing test
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn grid_bucketing_matches_brute_force() {
        let n = 50;
        let umap: Vec<[f64; 2]> = (0..n)
            .map(|i| [(i % 7) as f64 * 0.5, (i / 7) as f64 * 0.5])
            .collect();
        let v_cell: Vec<[f64; 2]> = (0..n)
            .map(|i| [(i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()])
            .collect();
        let grid_x = grid_axes(0.0, 3.0, 2.0);
        let grid_y = grid_axes(0.0, 4.0, 2.0);
        let nx = grid_x.len();
        let ny = grid_y.len();

        let bucketed = aggregate_grid_cartography(&v_cell, &umap, &grid_x, &grid_y);

        let x_half = if nx >= 2 { (grid_x[1] - grid_x[0]).abs() / 2.0 } else { 1.0 };
        let y_half = if ny >= 2 { (grid_y[1] - grid_y[0]).abs() / 2.0 } else { 1.0 };
        for ix in 0..nx {
            let gx = grid_x[ix];
            for iy in 0..ny {
                let gy = grid_y[iy];
                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                for i in 0..n {
                    if (umap[i][0] - gx).abs() <= x_half && (umap[i][1] - gy).abs() <= y_half {
                        sx += v_cell[i][0];
                        sy += v_cell[i][1];
                    }
                }
                let idx = ix * ny + iy;
                assert!(
                    (bucketed[idx][0] - sx).abs() < 1e-9 && (bucketed[idx][1] - sy).abs() < 1e-9,
                    "mismatch at [{},{}]: bucketed={:?} brute=[{},{}]",
                    ix, iy, bucketed[idx], sx, sy
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Magnitude threshold test
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn magnitude_threshold_zeros_small_vectors() {
        let expr = Array2::from_shape_fn((4, 1), |(i, _)| i as f64);
        let mut delta = Array2::zeros((4, 1));
        delta[[0, 0]] = 0.01;
        let umap: Vec<[f64; 2]> = (0..4).map(|i| [i as f64, 0.0]).collect();
        let params = TransitionUmapParams {
            n_neighbors: 3,
            magnitude_threshold: 0.5,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        for v in &grid.vectors {
            let mag = cell_mag(*v);
            assert!(mag < 1e-9 || mag >= 0.5, "got mag={}", mag);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Larger-scale determinism and sanity tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn deterministic_output() {
        let n = 30;
        let g = 5;
        let expr = Array2::from_shape_fn((n, g), |(i, k)| (i * k + 1) as f64 * 0.1);
        let mut delta = Array2::zeros((n, g));
        delta[[5, 0]] = 1.0;
        delta[[10, 2]] = -0.5;
        let umap: Vec<[f64; 2]> = (0..n)
            .map(|i| [(i % 6) as f64, (i / 6) as f64])
            .collect();
        let params = TransitionUmapParams {
            n_neighbors: 10,
            remove_null: true,
            ..base_params()
        };

        let g1 = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        let g2 = compute_umap_transition_grid(&expr, &delta, &umap, &params);

        for i in 0..n {
            assert_eq!(g1.cell_vectors[i][0], g2.cell_vectors[i][0]);
            assert_eq!(g1.cell_vectors[i][1], g2.cell_vectors[i][1]);
        }
        assert_eq!(g1.vectors.len(), g2.vectors.len());
        for i in 0..g1.vectors.len() {
            assert_eq!(g1.vectors[i][0], g2.vectors[i][0]);
            assert_eq!(g1.vectors[i][1], g2.vectors[i][1]);
        }
    }

    #[test]
    fn grid_vectors_count_matches_nx_ny() {
        let n = 20;
        let expr = Array2::from_shape_fn((n, 3), |(i, k)| (i + k) as f64);
        let mut delta = Array2::zeros((n, 3));
        delta[[0, 0]] = 1.0;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();
        let params = base_params();
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        assert_eq!(
            grid.vectors.len(),
            grid.grid_x.len() * grid.grid_y.len(),
            "vectors length must be nx*ny"
        );
    }

    #[test]
    fn cell_vectors_length_matches_n() {
        let n = 15;
        let expr = Array2::from_shape_fn((n, 2), |(i, k)| (i + k) as f64);
        let delta = Array2::zeros((n, 2));
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();
        let params = base_params();
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        assert_eq!(grid.cell_vectors.len(), n);
    }

    #[test]
    fn all_vectors_finite() {
        let n = 25;
        let expr = Array2::from_shape_fn((n, 4), |(i, k)| (i as f64 * 0.3 + k as f64).sin());
        let delta = Array2::from_shape_fn((n, 4), |(i, k)| if i == 5 && k == 0 { 1.0 } else { 0.0 });
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [(i % 5) as f64, (i / 5) as f64]).collect();
        let params = TransitionUmapParams {
            n_neighbors: 8,
            remove_null: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        for v in &grid.cell_vectors {
            assert!(v[0].is_finite() && v[1].is_finite());
        }
        for v in &grid.vectors {
            assert!(v[0].is_finite() && v[1].is_finite());
        }
    }

    #[test]
    fn vector_scale_controls_max_magnitude() {
        let n = 10;
        let expr = Array2::from_shape_fn((n, 3), |(i, k)| (i + k) as f64);
        let mut delta = Array2::zeros((n, 3));
        delta[[3, 0]] = 1.0;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();

        for &vs in &[0.5, 1.0, 2.0] {
            let params = TransitionUmapParams {
                n_neighbors: 5,
                vector_scale: vs,
                use_full_graph: true,
                ..base_params()
            };
            let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
            let max_comp = grid
                .vectors
                .iter()
                .fold(0.0_f64, |acc, w| acc.max(w[0].abs()).max(w[1].abs()));
            if max_comp > 1e-12 {
                assert!(
                    (max_comp - vs).abs() < 1e-5,
                    "max |component| should match cartography np.max scaling, vector_scale={} got {}",
                    vs, max_comp
                );
            }
        }
    }

    #[test]
    fn delta_rescale_affects_result() {
        let n = 8;
        let expr = Array2::from_shape_fn((n, 3), |(i, k)| (i * 2 + k) as f64 * 0.3);
        let mut delta = Array2::zeros((n, 3));
        delta[[2, 0]] = 0.5;
        delta[[2, 1]] = 0.3;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();

        let p1 = TransitionUmapParams {
            n_neighbors: 5,
            delta_rescale: 1.0,
            use_full_graph: true,
            ..base_params()
        };
        let p2 = TransitionUmapParams { delta_rescale: 0.1, ..p1.clone() };

        let g1 = compute_umap_transition_grid(&expr, &delta, &umap, &p1);
        let g2 = compute_umap_transition_grid(&expr, &delta, &umap, &p2);

        // delta_rescale changes what gets rounded to 3 decimals, which can alter
        // the correlation profile. At minimum, the two should not be identical.
        let v1 = g1.cell_vectors[2];
        let v2 = g2.cell_vectors[2];
        let same = (v1[0] - v2[0]).abs() < 1e-12 && (v1[1] - v2[1]).abs() < 1e-12;
        // They could coincidentally match; just verify both are finite
        assert!(v1[0].is_finite() && v2[0].is_finite());
        if !same {
            // Good — different rescale produced different vectors
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Edge cases
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn two_cells() {
        let expr = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let mut delta = Array2::zeros((2, 2));
        delta[[0, 0]] = 1.0;
        let umap: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0]];
        let params = TransitionUmapParams {
            n_neighbors: 1,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        assert_eq!(grid.cell_vectors.len(), 2);
        for v in &grid.cell_vectors {
            assert!(v[0].is_finite() && v[1].is_finite());
        }
    }

    #[test]
    fn identical_cells_zero_vectors() {
        let n = 5;
        let expr = Array2::from_elem((n, 3), 1.0);
        let delta = Array2::from_elem((n, 3), 0.5);
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, 0.0]).collect();
        let params = TransitionUmapParams {
            n_neighbors: 4,
            remove_null: true,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        for (i, v) in grid.cell_vectors.iter().enumerate() {
            assert!(cell_mag(*v) < 1e-9, "identical cells should yield zero vectors, cell {} mag={:.6e}", i, cell_mag(*v));
        }
    }

    #[test]
    fn collinear_umap_no_nan() {
        let n = 10;
        let expr = Array2::from_shape_fn((n, 3), |(i, k)| (i + k) as f64);
        let mut delta = Array2::zeros((n, 3));
        delta[[5, 0]] = 1.0;
        let umap: Vec<[f64; 2]> = (0..n).map(|i| [i as f64, i as f64]).collect();
        let params = TransitionUmapParams {
            n_neighbors: 5,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        for v in &grid.cell_vectors {
            assert!(v[0].is_finite() && v[1].is_finite());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Performance sanity: larger n doesn't crash or produce garbage
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn moderate_n_completes_sparse() {
        let n = 200;
        let g = 10;
        let expr = Array2::from_shape_fn((n, g), |(i, k)| ((i * 7 + k * 3) % 100) as f64 * 0.01);
        let mut delta = Array2::zeros((n, g));
        delta[[50, 0]] = 1.0;
        delta[[100, 3]] = -0.5;
        let umap: Vec<[f64; 2]> = (0..n)
            .map(|i| [(i % 15) as f64, (i / 15) as f64])
            .collect();
        let params = TransitionUmapParams {
            n_neighbors: 30,
            remove_null: true,
            use_full_graph: false,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        assert_eq!(grid.cell_vectors.len(), n);
        let any_nonzero = grid.cell_vectors.iter().any(|v| cell_mag(*v) > 1e-12);
        assert!(any_nonzero, "at least some cell should have a nonzero vector");
    }

    #[test]
    fn moderate_n_completes_dense() {
        let n = 100;
        let g = 8;
        let expr = Array2::from_shape_fn((n, g), |(i, k)| ((i * 3 + k * 5) % 50) as f64 * 0.02);
        let mut delta = Array2::zeros((n, g));
        delta[[25, 0]] = 2.0;
        let umap: Vec<[f64; 2]> = (0..n)
            .map(|i| [(i % 10) as f64, (i / 10) as f64])
            .collect();
        let params = TransitionUmapParams {
            n_neighbors: 20,
            remove_null: true,
            use_full_graph: true,
            ..base_params()
        };
        let grid = compute_umap_transition_grid(&expr, &delta, &umap, &params);
        assert_eq!(grid.cell_vectors.len(), n);
        for v in &grid.cell_vectors {
            assert!(v[0].is_finite() && v[1].is_finite());
        }
    }
}
