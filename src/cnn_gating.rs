use crate::config::HybridCnnGatingConfig;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CnnGateDecision {
    pub use_cnn: bool,
    pub reason: String,
    pub min_cells_per_cluster: usize,
    pub n_modulators: usize,
    pub n_lr_pairs: usize,
    pub n_tfl_pairs: usize,
    pub modulator_spatial_fraction: f64,
    pub mean_lasso_r2: f64,
    pub all_lasso_converged: bool,
    pub moran_i: f64,
    pub moran_p_value: f64,
    pub moran_permutations: usize,
    pub forced_by_allowlist: bool,
    pub blocked_by_denylist: bool,
    pub mean_target_expression: Option<f64>,
    pub rank_score: f64,
}

pub fn load_gene_set_file(path: &Path) -> anyhow::Result<HashSet<String>> {
    let s = fs::read_to_string(path)?;
    let mut out = HashSet::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        for part in t.split(',') {
            let g = part.trim();
            if !g.is_empty() {
                out.insert(g.to_string());
            }
        }
    }
    Ok(out)
}

pub fn predict_lasso_y(
    lasso_coefs: &HashMap<usize, ndarray::Array2<f64>>,
    intercepts: &HashMap<usize, f64>,
    x: &Array2<f64>,
    clusters: &Array1<usize>,
) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut yhat = Array1::<f64>::zeros(n);
    let default_intercept = 0.0;
    for i in 0..n {
        let c = clusters[i];
        let intercept = intercepts.get(&c).copied().unwrap_or(default_intercept);
        let coef = lasso_coefs.get(&c);
        let mut v = intercept;
        if let Some(b) = coef {
            let col = b.column(0);
            for j in 0..p.min(col.len()) {
                v += x[[i, j]] * col[j];
            }
        }
        yhat[i] = crate::estimator::finite_or_zero_f64(v);
    }
    yhat
}

fn symmetrized_knn_indices(xy: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    let n = xy.nrows();
    if n == 0 {
        return Vec::new();
    }
    let k_eff = k.min(n.saturating_sub(1).max(1));
    let mut directed: Vec<Vec<usize>> = vec![Vec::new(); n];
    let x0 = xy.column(0);
    let x1 = xy.column(1);
    for i in 0..n {
        let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n);
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = x0[i] - x0[j];
            let dy = x1[i] - x1[j];
            dists.push((dx * dx + dy * dy, j));
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for &( _, j) in dists.iter().take(k_eff) {
            directed[i].push(j);
        }
    }
    let mut adj: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::new()).collect();
    for i in 0..n {
        for &j in &directed[i] {
            adj[i].insert(j);
            adj[j].insert(i);
        }
    }
    adj.into_iter().map(|s| s.into_iter().collect()).collect()
}

pub fn morans_i_binary_residuals(e: &[f64], neighbors: &[Vec<usize>]) -> f64 {
    let n = e.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = e.iter().sum::<f64>() / n as f64;
    let z: Vec<f64> = e.iter().map(|&x| x - mean).collect();
    let v: f64 = z.iter().map(|&x| x * x).sum();
    if v <= 1e-18 {
        return 0.0;
    }
    let mut s0 = 0.0_f64;
    let mut cross = 0.0_f64;
    for i in 0..n {
        for &j in &neighbors[i] {
            if j < n {
                cross += z[i] * z[j];
                s0 += 1.0;
            }
        }
    }
    if s0 <= 0.0 {
        return 0.0;
    }
    (n as f64 / s0) * (cross / v)
}

fn moran_permutation_p_value(
    e_obs: &[f64],
    neighbors: &[Vec<usize>],
    n_perm: usize,
    rng: &mut impl rand::Rng,
) -> (f64, f64) {
    let i_obs = morans_i_binary_residuals(e_obs, neighbors);
    if n_perm == 0 {
        return (i_obs, 1.0);
    }
    let mut e_work = e_obs.to_vec();
    let mut extreme = 0usize;
    let abs_obs = i_obs.abs();
    for _ in 0..n_perm {
        e_work.shuffle(rng);
        let i_p = morans_i_binary_residuals(&e_work, neighbors);
        if i_p.abs() >= abs_obs - 1e-15 {
            extreme += 1;
        }
    }
    let p = (1 + extreme) as f64 / (1 + n_perm) as f64;
    (i_obs, p)
}

pub fn decide_cnn_for_gene(
    cfg: &HybridCnnGatingConfig,
    min_mean_lasso_r2: f64,
    gene: &str,
    summaries: &[crate::estimator::ClusterTrainingSummary],
    n_regulators: usize,
    n_lr_pairs: usize,
    n_tfl_pairs: usize,
    residuals: &[f64],
    neighbors: &[Vec<usize>],
    force_genes: &HashSet<String>,
    skip_genes: &HashSet<String>,
    mean_target_expression: Option<f64>,
) -> CnnGateDecision {
    decide_cnn_for_gene_with_rng(
        cfg,
        min_mean_lasso_r2,
        gene,
        summaries,
        n_regulators,
        n_lr_pairs,
        n_tfl_pairs,
        residuals,
        neighbors,
        force_genes,
        skip_genes,
        mean_target_expression,
        &mut thread_rng(),
    )
}

pub(crate) fn decide_cnn_for_gene_with_rng(
    cfg: &HybridCnnGatingConfig,
    min_mean_lasso_r2: f64,
    gene: &str,
    summaries: &[crate::estimator::ClusterTrainingSummary],
    n_regulators: usize,
    n_lr_pairs: usize,
    n_tfl_pairs: usize,
    residuals: &[f64],
    neighbors: &[Vec<usize>],
    force_genes: &HashSet<String>,
    skip_genes: &HashSet<String>,
    mean_target_expression: Option<f64>,
    rng: &mut impl Rng,
) -> CnnGateDecision {
    let min_r2_gate = cfg.effective_min_mean_lasso_r2(min_mean_lasso_r2);
    let moran_p_max = cfg.effective_moran_p_max();
    let moran_p_strict = cfg.effective_moran_p_strict();
    let n_mods = n_regulators + n_lr_pairs + n_tfl_pairs;
    let frac = if n_mods > 0 {
        (n_lr_pairs + n_tfl_pairs) as f64 / n_mods as f64
    } else {
        0.0
    };

    let mut blocked = false;
    let mut forced = false;

    if skip_genes.contains(gene) {
        blocked = true;
        return finish_decision(
            false,
            "blocked_by_skip_list".to_string(),
            summaries,
            n_mods,
            n_lr_pairs,
            n_tfl_pairs,
            frac,
            0.0,
            1.0,
            cfg.moran_permutations,
            forced,
            blocked,
            mean_target_expression,
            cfg,
        );
    }

    let (i_obs, p_val) =
        moran_permutation_p_value(residuals, neighbors, cfg.moran_permutations, rng);

    if force_genes.contains(gene) {
        forced = true;
        return finish_decision(
            true,
            "forced_by_allowlist".to_string(),
            summaries,
            n_mods,
            n_lr_pairs,
            n_tfl_pairs,
            frac,
            i_obs,
            p_val,
            cfg.moran_permutations,
            forced,
            blocked,
            mean_target_expression,
            cfg,
        );
    }

    let min_cells = summaries
        .iter()
        .map(|s| s.n_cells)
        .min()
        .unwrap_or(0);
    if min_cells < cfg.min_cells_per_cluster_for_cnn {
        let reason = format!(
            "min_cells_per_cluster {} < {}",
            min_cells, cfg.min_cells_per_cluster_for_cnn
        );
        return finish_decision(
            false,
            reason,
            summaries,
            n_mods,
            n_lr_pairs,
            n_tfl_pairs,
            frac,
            i_obs,
            p_val,
            cfg.moran_permutations,
            forced,
            blocked,
            mean_target_expression,
            cfg,
        );
    }

    let all_conv = summaries.iter().all(|s| s.lasso_converged);
    if cfg.require_all_clusters_lasso_converged && !all_conv {
        let reason = "lasso_not_converged_all_clusters".to_string();
        return finish_decision(
            false,
            reason,
            summaries,
            n_mods,
            n_lr_pairs,
            n_tfl_pairs,
            frac,
            i_obs,
            p_val,
            cfg.moran_permutations,
            forced,
            blocked,
            mean_target_expression,
            cfg,
        );
    }

    let mean_r2: f64 = if summaries.is_empty() {
        0.0
    } else {
        summaries.iter().map(|s| s.lasso_r2).sum::<f64>() / summaries.len() as f64
    };
    if mean_r2 < min_r2_gate {
        let reason = format!(
            "mean_lasso_r2 {:.4} < {:.4} (gate)",
            mean_r2, min_r2_gate
        );
        return finish_decision(
            false,
            reason,
            summaries,
            n_mods,
            n_lr_pairs,
            n_tfl_pairs,
            frac,
            i_obs,
            p_val,
            cfg.moran_permutations,
            forced,
            blocked,
            mean_target_expression,
            cfg,
        );
    }

    if let Some(min_expr) = cfg.min_mean_target_expression_for_cnn {
        if let Some(m) = mean_target_expression {
            if m < min_expr {
                let reason = format!("mean_target_expr {:.4} < {:.4}", m, min_expr);
                return finish_decision(
                    false,
                    reason,
                    summaries,
                    n_mods,
                    n_lr_pairs,
                    n_tfl_pairs,
                    frac,
                    i_obs,
                    p_val,
                    cfg.moran_permutations,
                    forced,
                    blocked,
                    mean_target_expression,
                    cfg,
                );
            }
        }
    }

    let max_mod_ok = match cfg.max_modulators_soft_for_cnn {
        None => true,
        Some(max) => n_mods <= max,
    };

    let moran_ok = p_val <= moran_p_max;
    if !max_mod_ok {
        if p_val > moran_p_strict {
            let reason = format!(
                "n_modulators {} > cap {:?} and moran_p {:.4} > strict {:.4}",
                n_mods, cfg.max_modulators_soft_for_cnn, p_val, moran_p_strict
            );
            return finish_decision(
                false,
                reason,
                summaries,
                n_mods,
                n_lr_pairs,
                n_tfl_pairs,
                frac,
                i_obs,
                p_val,
                cfg.moran_permutations,
                forced,
                blocked,
                mean_target_expression,
                cfg,
            );
        }
    } else if !moran_ok {
        let reason = format!(
            "moran_p {:.4} > {:.4} (I={:.6})",
            p_val, moran_p_max, i_obs
        );
        return finish_decision(
            false,
            reason,
            summaries,
            n_mods,
            n_lr_pairs,
            n_tfl_pairs,
            frac,
            i_obs,
            p_val,
            cfg.moran_permutations,
            forced,
            blocked,
            mean_target_expression,
            cfg,
        );
    }

    let reason = if max_mod_ok {
        format!("moran_ok p={:.4} I={:.6}", p_val, i_obs)
    } else {
        format!(
            "moran_ok_strict_mod_cap p={:.4} I={:.6} n_mod={}",
            p_val, i_obs, n_mods
        )
    };
    finish_decision(
        true,
        reason,
        summaries,
        n_mods,
        n_lr_pairs,
        n_tfl_pairs,
        frac,
        i_obs,
        p_val,
        cfg.moran_permutations,
        forced,
        blocked,
        mean_target_expression,
        cfg,
    )
}

fn finish_decision(
    use_cnn: bool,
    reason: String,
    summaries: &[crate::estimator::ClusterTrainingSummary],
    n_mods: usize,
    n_lr: usize,
    n_tfl: usize,
    frac: f64,
    moran_i: f64,
    moran_p: f64,
    moran_perm: usize,
    forced: bool,
    blocked: bool,
    mean_target_expression: Option<f64>,
    cfg: &HybridCnnGatingConfig,
) -> CnnGateDecision {
    let mean_r2: f64 = if summaries.is_empty() {
        0.0
    } else {
        summaries.iter().map(|s| s.lasso_r2).sum::<f64>() / summaries.len() as f64
    };
    let all_conv = summaries.iter().all(|s| s.lasso_converged);
    let min_cells = summaries
        .iter()
        .map(|s| s.n_cells)
        .min()
        .unwrap_or(0);
    let rank_score = (-moran_p.max(1e-300).ln()) + cfg.hybrid_modulator_spatial_weight * frac + mean_r2;
    CnnGateDecision {
        use_cnn,
        reason,
        min_cells_per_cluster: min_cells,
        n_modulators: n_mods,
        n_lr_pairs: n_lr,
        n_tfl_pairs: n_tfl,
        modulator_spatial_fraction: frac,
        mean_lasso_r2: mean_r2,
        all_lasso_converged: all_conv,
        moran_i,
        moran_p_value: moran_p,
        moran_permutations: moran_perm,
        forced_by_allowlist: forced,
        blocked_by_denylist: blocked,
        mean_target_expression,
        rank_score,
    }
}

pub fn build_neighbors(xy: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    symmetrized_knn_indices(xy, k.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::ClusterTrainingSummary;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn summary(r2: f64, n_cells: usize) -> ClusterTrainingSummary {
        ClusterTrainingSummary {
            cluster_id: 0,
            n_cells,
            n_modulators: 5,
            lasso_r2: r2,
            lasso_train_mse: 0.0,
            lasso_fista_iters: 10,
            lasso_converged: true,
            cnn_train_mse_epochs: vec![],
        }
    }

    fn cfg_moran_disabled_for_tests() -> HybridCnnGatingConfig {
        HybridCnnGatingConfig {
            moran_permutations: 0,
            moran_p_value_max: 4.0,
            moran_p_value_max_when_over_modulator_cap: Some(4.0),
            require_all_clusters_lasso_converged: false,
            ..HybridCnnGatingConfig::default()
        }
    }

    #[test]
    fn effective_permissiveness_scales_moran_and_r2_gates() {
        let mut c = HybridCnnGatingConfig::default();
        c.hybrid_cnn_permissiveness = 0.0;
        assert!((c.effective_moran_p_max() - 0.05 * 0.3).abs() < 1e-9);
        assert!((c.effective_min_mean_lasso_r2(0.1) - 0.14).abs() < 1e-9);
        c.hybrid_cnn_permissiveness = 0.5;
        assert!((c.effective_moran_p_max() - 0.05).abs() < 1e-9);
        assert!((c.effective_min_mean_lasso_r2(0.1) - 0.1).abs() < 1e-9);
        c.hybrid_cnn_permissiveness = 1.0;
        assert!((c.effective_moran_p_max() - 0.05 * 1.7).abs() < 1e-9);
        assert!((c.effective_min_mean_lasso_r2(0.1) - 0.06).abs() < 1e-9);
    }

    #[test]
    fn skip_list_blocks_cnn() {
        let cfg = cfg_moran_disabled_for_tests();
        let mut skip = HashSet::new();
        skip.insert("X".to_string());
        let xy = Array2::from_shape_fn((8, 2), |(i, j)| if j == 0 { i as f64 } else { 0.0 });
        let nb = build_neighbors(&xy, 3);
        let res = vec![0.0_f64; 8];
        let d = decide_cnn_for_gene_with_rng(
            &cfg,
            0.0,
            "X",
            &[summary(0.5, 100)],
            1,
            1,
            1,
            &res,
            &nb,
            &HashSet::new(),
            &skip,
            None,
            &mut StdRng::seed_from_u64(0),
        );
        assert!(!d.use_cnn);
        assert!(d.blocked_by_denylist);
    }

    #[test]
    fn allow_list_forces_cnn() {
        let cfg = cfg_moran_disabled_for_tests();
        let mut force = HashSet::new();
        force.insert("Y".to_string());
        let xy = Array2::from_shape_fn((8, 2), |(i, j)| if j == 0 { i as f64 } else { 0.0 });
        let nb = build_neighbors(&xy, 3);
        let res = vec![0.0_f64; 8];
        let d = decide_cnn_for_gene_with_rng(
            &cfg,
            1.0,
            "Y",
            &[summary(0.0, 100)],
            1,
            1,
            1,
            &res,
            &nb,
            &force,
            &HashSet::new(),
            None,
            &mut StdRng::seed_from_u64(0),
        );
        assert!(d.use_cnn);
        assert!(d.forced_by_allowlist);
    }

    #[test]
    fn permissiveness_toggles_mean_r2_gate_with_moran_bypassed() {
        let mut cfg = cfg_moran_disabled_for_tests();
        let xy = Array2::from_shape_fn((12, 2), |(i, j)| if j == 0 { i as f64 } else { 0.0 });
        let nb = build_neighbors(&xy, 4);
        let res = vec![0.0_f64; 12];
        let summaries = vec![summary(0.11, 100)];
        let mut rng = StdRng::seed_from_u64(42);

        cfg.hybrid_cnn_permissiveness = 0.0;
        let d_strict = decide_cnn_for_gene_with_rng(
            &cfg,
            0.1,
            "G",
            &summaries,
            5,
            2,
            1,
            &res,
            &nb,
            &HashSet::new(),
            &HashSet::new(),
            None,
            &mut rng,
        );
        assert!(!d_strict.use_cnn, "expected mean_r2 gate fail: {:?}", d_strict.reason);

        cfg.hybrid_cnn_permissiveness = 1.0;
        let mut rng = StdRng::seed_from_u64(42);
        let d_loose = decide_cnn_for_gene_with_rng(
            &cfg,
            0.1,
            "G",
            &summaries,
            5,
            2,
            1,
            &res,
            &nb,
            &HashSet::new(),
            &HashSet::new(),
            None,
            &mut rng,
        );
        assert!(d_loose.use_cnn, "expected pass: {:?}", d_loose.reason);
    }

    #[test]
    fn morans_i_binary_residuals_constant_is_zero() {
        let n = 6;
        let e = vec![1.0_f64; n];
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| vec![(i + 1) % n, (i + n - 1) % n])
            .collect();
        let i = morans_i_binary_residuals(&e, &neighbors);
        assert!(i.abs() < 1e-9);
    }
}
