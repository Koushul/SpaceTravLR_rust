use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::backend::RctdDevice;
use crate::irwls_native::{
    calc_q_d0_flat, solve_irwls_native, NativeSharedPrepared,
};
use crate::likelihood_tables::compute_spline_coefficients;

pub static SIGMA_ALL: &[i32] = &[
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92,
    94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130,
    132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168,
    170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200,
];

#[allow(clippy::too_many_arguments)]
pub fn choose_sigma(
    spatial_counts: &Array2<f64>,
    spatial_numi: &Array1<f64>,
    norm_profiles: &Array2<f64>,
    q_matrices: &HashMap<String, Array2<f64>>,
    x_vals: &Array1<f64>,
    sq_matrices: Option<HashMap<String, Array2<f64>>>,
    sigma_init: i32,
    min_umi: i32,
    n_fit: usize,
    n_epoch: usize,
    k_val: i64,
    seed: u64,
    _device: &RctdDevice,
) -> i32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let valid_idx: Vec<usize> = spatial_numi
        .iter()
        .enumerate()
        .filter(|(_, u)| **u > min_umi as f64)
        .map(|(i, _)| i)
        .collect();
    assert!(
        !valid_idx.is_empty(),
        "No pixels with UMI > min_umi for sigma"
    );
    let n_samples = n_fit.min(valid_idx.len());
    let mut pick = valid_idx;
    pick.shuffle(&mut rng);
    let fit_idx: Vec<usize> = pick.into_iter().take(n_samples).collect();

    let mut sq_map: HashMap<String, Array2<f64>> = sq_matrices.unwrap_or_default();
    if sq_map.is_empty() {
        for (k, q) in q_matrices {
            sq_map.insert(k.clone(), compute_spline_coefficients(q, x_vals));
        }
    }

    let fit_counts = Array2::from_shape_fn((fit_idx.len(), spatial_counts.ncols()), |(i, g)| {
        spatial_counts[[fit_idx[i], g]]
    });
    let fit_numi: Array1<f64> = fit_idx.iter().map(|&i| spatial_numi[i]).collect();

    let mut sigma = sigma_init;
    for _epoch in 0..n_epoch {
        let sigma_use = nearest_sigma_key(sigma, q_matrices);
        let q_cur = q_matrices.get(&sigma_use.to_string()).expect("q mat");
        let sq_cur = sq_map.get(&sigma_use.to_string()).expect("sq mat");

        let prep = NativeSharedPrepared::new(norm_profiles, q_cur, sq_cur, x_vals);
        let (mut weights, _) = solve_irwls_native(
            &prep,
            fit_counts.view(),
            fit_numi.view(),
            50,
            0.001,
            0.3,
            false,
            false,
        );
        weights.mapv_inplace(|w| w.max(0.0));

        let mut prediction = Array2::zeros((weights.nrows(), norm_profiles.nrows()));
        for i in 0..weights.nrows() {
            let row = weights.row(i);
            let p = norm_profiles.dot(&row) * fit_numi[i];
            for g in 0..p.len() {
                prediction[[i, g]] = p[g].max(1e-4);
            }
        }

        let si_idx = sigma_index(sigma);
        let start = si_idx.saturating_sub(8);
        let end = (si_idx + 8 + 1).min(SIGMA_ALL.len());
        let valid_cands: Vec<i32> = SIGMA_ALL[start..end]
            .iter()
            .copied()
            .filter(|s| q_matrices.contains_key(&s.to_string()))
            .collect();
        if valid_cands.is_empty() {
            break;
        }

        let y_flat: Vec<f64> = fit_counts.iter().cloned().collect();
        let y_len = y_flat.len();

        let mut best_sigma = sigma;
        let mut best_score = f64::INFINITY;
        for &s in &valid_cands {
            let q_s = q_matrices.get(&s.to_string()).unwrap();
            let sq_s = sq_map.get(&s.to_string()).unwrap();
            let mut min_over_fac = f64::INFINITY;
            for fac_num in 8_i32..=12 {
                let fac = fac_num as f64 / 10.0;
                let lam_flat: Vec<f64> = prediction.iter().map(|p| (p * fac).max(1e-4)).collect();
                let d0 = calc_q_d0_flat(&y_flat, &lam_flat, q_s, sq_s, x_vals, k_val);
                debug_assert_eq!(d0.len(), y_len);
                let score = -d0.iter().sum::<f64>();
                min_over_fac = min_over_fac.min(score);
            }
            if min_over_fac < best_score {
                best_score = min_over_fac;
                best_sigma = s;
            }
        }
        let prev = sigma;
        sigma = best_sigma;
        if sigma == prev {
            break;
        }
    }
    sigma
}

fn nearest_sigma_key(sigma: i32, q: &HashMap<String, Array2<f64>>) -> i32 {
    if q.contains_key(&sigma.to_string()) {
        return sigma;
    }
    *SIGMA_ALL
        .iter()
        .min_by_key(|&&s| (s - sigma).abs())
        .unwrap()
}

fn sigma_index(sigma: i32) -> usize {
    SIGMA_ALL
        .iter()
        .position(|&s| s == sigma)
        .unwrap_or_else(|| {
            SIGMA_ALL
                .iter()
                .enumerate()
                .min_by_key(|(_, &s)| (s - sigma).abs())
                .unwrap()
                .0
        })
}
