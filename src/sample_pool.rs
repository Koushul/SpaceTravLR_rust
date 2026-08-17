//! Helpers for pooled-lasso training across independent spatial samples.

use crate::ligand::calculate_weighted_ligands;
use crate::modulator_scale::scale_columns_no_center;
use ndarray::{Array1, Array2, Axis, s};
use std::collections::{HashMap, HashSet};

/// Sorted unique labels → integer ids, then encode `labels` with that map.
/// Factorizing on the full set first keeps ids stable when a subset is missing a label.
pub fn encode_cluster_ids(labels: &[String]) -> (Vec<usize>, HashMap<String, usize>) {
    let mut seen = HashSet::<String>::new();
    for l in labels {
        seen.insert(l.clone());
    }
    let mut uniq: Vec<String> = seen.into_iter().collect();
    uniq.sort();
    let map: HashMap<String, usize> = uniq.into_iter().enumerate().map(|(i, k)| (k, i)).collect();
    let ids: Vec<usize> = labels
        .iter()
        .map(|l| map.get(l).copied().unwrap_or(0))
        .collect();
    (ids, map)
}

pub fn encode_cluster_ids_with_map(labels: &[String], map: &HashMap<String, usize>) -> Vec<usize> {
    labels
        .iter()
        .map(|l| map.get(l).copied().unwrap_or(0))
        .collect()
}

pub fn vstack_rows(mats: &[Array2<f64>]) -> anyhow::Result<Array2<f64>> {
    anyhow::ensure!(!mats.is_empty(), "vstack_rows: no matrices");
    let ncols = mats[0].ncols();
    for m in mats {
        anyhow::ensure!(
            m.ncols() == ncols,
            "vstack_rows: column mismatch {} vs {}",
            m.ncols(),
            ncols
        );
    }
    let views: Vec<_> = mats.iter().map(|m| m.view()).collect();
    ndarray::concatenate(Axis(0), &views).map_err(|e| anyhow::anyhow!("vstack_rows: {e}"))
}

pub fn concat_vec1(parts: &[Array1<f64>]) -> anyhow::Result<Array1<f64>> {
    anyhow::ensure!(!parts.is_empty(), "concat_vec1: no vectors");
    let views: Vec<_> = parts.iter().map(|v| v.view()).collect();
    ndarray::concatenate(Axis(0), &views).map_err(|e| anyhow::anyhow!("concat_vec1: {e}"))
}

pub fn concat_usize(parts: &[Array1<usize>]) -> anyhow::Result<Array1<usize>> {
    anyhow::ensure!(!parts.is_empty(), "concat_usize: no vectors");
    let n: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend(p.iter().copied());
    }
    Ok(Array1::from_vec(out))
}

/// Column-std scale on the concatenated design matrix (no mean centering).
pub fn joint_scale_columns(parts: &mut [Array2<f64>]) -> anyhow::Result<Array1<f64>> {
    let mut pooled = vstack_rows(parts)?;
    let scales = scale_columns_no_center(&mut pooled);
    let mut offset = 0usize;
    for part in parts.iter_mut() {
        let n = part.nrows();
        part.assign(&pooled.slice(s![offset..offset + n, ..]));
        offset += n;
    }
    Ok(scales)
}

/// Received-ligand rows for `xy` / `lig` restricted to `row_indices` (spatial neighbors
/// cannot see cells outside the subset).
pub fn weighted_ligands_for_rows(
    xy: &Array2<f64>,
    lig: &Array2<f64>,
    row_indices: &[usize],
    radius: f64,
    scale_factor: f64,
) -> Array2<f64> {
    let xy_s = xy.select(Axis(0), row_indices);
    let lig_s = lig.select(Axis(0), row_indices);
    calculate_weighted_ligands(&xy_s, &lig_s, radius, scale_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lasso::{GroupLasso, GroupLassoParams};
    use crate::modulator_scale::scale_columns_no_center;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn joint_scale_matches_vstack_then_scale() {
        let a = Array2::from_shape_fn((4, 2), |(i, j)| (i + 1) as f64 * (j + 1) as f64);
        let b = Array2::from_shape_fn((3, 2), |(i, j)| 10.0 * (i + 1) as f64 * (j + 1) as f64);
        let mut stacked = vstack_rows(&[a.clone(), b.clone()]).unwrap();
        let expected = scale_columns_no_center(&mut stacked);
        let mut parts = [a, b];
        let got = joint_scale_columns(&mut parts).unwrap();
        assert_abs_diff_eq!(got[0], expected[0], epsilon = 1e-12);
        assert_abs_diff_eq!(got[1], expected[1], epsilon = 1e-12);
        let restack = vstack_rows(&parts).unwrap();
        for (a, b) in restack.iter().zip(stacked.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn per_sample_scales_differ_when_second_is_10x() {
        let a = Array2::from_shape_fn((6, 2), |(i, j)| (i + 1) as f64 + j as f64);
        let b = &a * 10.0;
        let mut a1 = a.clone();
        let mut b1 = b.clone();
        let sa = scale_columns_no_center(&mut a1);
        let sb = scale_columns_no_center(&mut b1);
        assert!((sa[0] - sb[0]).abs() > 1.0);
        let mut parts = [a, b];
        let pooled = joint_scale_columns(&mut parts).unwrap();
        assert!((pooled[0] - sa[0]).abs() > 1e-6);
        assert!((pooled[0] - sb[0]).abs() > 1e-6);
    }

    #[test]
    fn ligands_blocked_by_sample_ignore_overlapping_coords() {
        // Two slides of 4 cells sharing the same 0–3 x-coordinates (y = 0).
        let xy = Array2::from_shape_fn((8, 2), |(i, j)| {
            let local = i % 4;
            if j == 0 { local as f64 } else { 0.0 }
        });
        let mut lig = Array2::<f64>::zeros((8, 1));
        for i in 0..4 {
            lig[[i, 0]] = 1.0;
        }
        for i in 4..8 {
            lig[[i, 0]] = 50.0;
        }
        let s1: Vec<usize> = (0..4).collect();
        let blocked = weighted_ligands_for_rows(&xy, &lig, &s1, 2.0, 1.0);
        let xy_s1 = xy.select(Axis(0), &s1);
        let lig_s1 = lig.select(Axis(0), &s1);
        let isolated = calculate_weighted_ligands(&xy_s1, &lig_s1, 2.0, 1.0);
        for (a, b) in blocked.iter().zip(isolated.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
        let naive = calculate_weighted_ligands(&xy, &lig, 2.0, 1.0);
        for i in 0..4 {
            assert!(
                (blocked[[i, 0]] - naive[[i, 0]]).abs() > 1e-6,
                "row {i}: blocked {} should differ from naive {}",
                blocked[[i, 0]],
                naive[[i, 0]]
            );
        }
        let blocked_again = weighted_ligands_for_rows(&xy, &lig, &s1, 2.0, 1.0);
        for (a, b) in blocked.iter().zip(blocked_again.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn global_cluster_ids_stable_when_subset_missing_a_type() {
        let all = vec!["ct_a".into(), "ct_b".into(), "ct_a".into(), "ct_b".into()];
        let (ids_all, map) = encode_cluster_ids(&all);
        let s1_only = vec!["ct_a".into(), "ct_a".into()];
        let (ids_local, _) = encode_cluster_ids(&s1_only);
        let ids_global = encode_cluster_ids_with_map(&s1_only, &map);
        assert_eq!(ids_local, vec![0, 0]);
        assert_eq!(ids_global, vec![ids_all[0], ids_all[2]]);
        assert_eq!(map.get("ct_a").copied(), Some(0));
        assert_eq!(map.get("ct_b").copied(), Some(1));
    }

    #[test]
    fn pooled_lasso_matches_single_when_samples_are_iid_copies() {
        let n = 40usize;
        let x0 = Array2::from_shape_fn((n, 2), |(i, j)| ((i + 1) as f64) * (0.1 + j as f64));
        let y0 = Array2::from_shape_fn((n, 1), |(i, _)| 1.5 * x0[[i, 0]] - 0.4 * x0[[i, 1]]);
        let x_pool = vstack_rows(&[x0.clone(), x0.clone()]).unwrap();
        let y_pool = vstack_rows(&[y0.clone(), y0.clone()]).unwrap();
        let params = GroupLassoParams {
            groups: vec![0, 1],
            group_reg: 1e-8,
            l1_reg: 1e-8,
            n_iter: 400,
            tol: 1e-10,
            fit_intercept: true,
            seed: 7,
            ..Default::default()
        };
        let mut single = GroupLasso::new(params.clone());
        let mut pooled = GroupLasso::new(params);
        let _ = single.fit(&x0, &y0, None);
        let _ = pooled.fit(&x_pool, &y_pool, None);
        let c1 = single.fitted.as_ref().expect("single fitted").coef.clone();
        let c2 = pooled.fitted.as_ref().expect("pooled fitted").coef.clone();
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-4);
        }
    }

    #[test]
    fn pooled_lasso_differs_from_s1_when_s2_is_scaled() {
        let n = 30usize;
        let x1 = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 + 1.0) * (j as f64 + 0.5));
        let y1 = Array2::from_shape_fn((n, 1), |(i, _)| x1[[i, 0]] + 0.2 * x1[[i, 1]]);
        let x2 = &x1 * 8.0;
        let mut x1_scaled = x1.clone();
        let _ = scale_columns_no_center(&mut x1_scaled);
        let mut parts = [x1.clone(), x2];
        let _ = joint_scale_columns(&mut parts).unwrap();
        let x_pool = vstack_rows(&parts).unwrap();
        let y_pool = vstack_rows(&[y1.clone(), y1.clone()]).unwrap();
        let params = GroupLassoParams {
            groups: vec![0, 1],
            group_reg: 1e-6,
            l1_reg: 1e-6,
            n_iter: 300,
            tol: 1e-8,
            seed: 3,
            ..Default::default()
        };
        let mut s1 = GroupLasso::new(params.clone());
        let mut pooled = GroupLasso::new(params);
        let _ = s1.fit(&x1_scaled, &y1, None);
        let _ = pooled.fit(&x_pool, &y_pool, None);
        let c1 = s1.fitted.as_ref().expect("s1 fitted").coef.clone();
        let c2 = pooled.fitted.as_ref().expect("pooled fitted").coef.clone();
        let max_diff = c1
            .iter()
            .zip(c2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-4,
            "expected pooled coefs to differ, max_diff={max_diff}"
        );
    }
}
