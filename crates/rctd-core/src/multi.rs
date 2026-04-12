use std::collections::{BTreeMap, BTreeSet, HashMap};

use ndarray::{Array1, Array2};

use crate::backend::RctdDevice;
use crate::full::run_full_mode;
use crate::irwls_native::{calc_neg_loglik_row_sums, solve_irwls_batch_s_ndarray};
use crate::types::{BatchProgress, MultiResult, RctdConfig, notify_batch};

type ConfTask = (usize, Vec<usize>, usize, usize);

#[allow(clippy::too_many_arguments)]
fn run_batched_scoring(
    tasks: &[(usize, Vec<usize>)],
    spatial_numi: &Array1<f64>,
    spatial_counts: &Array2<f64>,
    norm_profiles: &Array2<f64>,
    q_mat: &Array2<f64>,
    sq_mat: &Array2<f64>,
    x_vals: &Array1<f64>,
    batch_size: usize,
    max_iter: usize,
    progress: &Option<BatchProgress>,
) -> Vec<f64> {
    if tasks.is_empty() {
        return Vec::new();
    }
    let k_sub = tasks[0].1.len();
    debug_assert!(tasks.iter().all(|t| t.1.len() == k_sub));
    let g = norm_profiles.nrows();
    let m = tasks.len();
    let mut all_scores = Vec::with_capacity(m);

    for start in (0..m).step_by(batch_size.max(1)) {
        let end = (start + batch_size).min(m);
        let bs = end - start;
        let mut s = ndarray::Array3::<f64>::zeros((bs, g, k_sub));
        let y_b = Array2::from_shape_fn((bs, g), |(i, gg)| {
            let n = tasks[start + i].0;
            spatial_counts[[n, gg]]
        });
        let n_b: Array1<f64> = (0..bs).map(|i| spatial_numi[tasks[start + i].0]).collect();

        for i in 0..bs {
            let n = tasks[start + i].0;
            let u = spatial_numi[n];
            for j in 0..k_sub {
                let t = tasks[start + i].1[j];
                for gg in 0..g {
                    s[[i, gg, j]] = u * norm_profiles[[gg, t]];
                }
            }
        }

        let (w, _) = solve_irwls_batch_s_ndarray(
            &s, &y_b, &n_b, q_mat, sq_mat, x_vals, max_iter, 0.001, 0.3, false, false,
        );

        let mut lam = Array2::<f64>::zeros((bs, g));
        for i in 0..bs {
            for gg in 0..g {
                let mut e = 0.0f64;
                for j in 0..k_sub {
                    e += s[[i, gg, j]] * w[[i, j]];
                }
                lam[[i, gg]] = e.max(1e-4);
            }
        }

        let sl = calc_neg_loglik_row_sums(&y_b, &lam, q_mat, sq_mat, x_vals, -1);
        all_scores.extend(sl);
        notify_batch(progress, end - start);
    }
    all_scores
}

#[allow(clippy::too_many_arguments)]
fn run_batched_weights(
    tasks: &[(usize, Vec<usize>)],
    spatial_numi: &Array1<f64>,
    spatial_counts: &Array2<f64>,
    norm_profiles: &Array2<f64>,
    q_mat: &Array2<f64>,
    sq_mat: &Array2<f64>,
    x_vals: &Array1<f64>,
    batch_size: usize,
    max_iter: usize,
    progress: &Option<BatchProgress>,
) -> Vec<Array1<f64>> {
    if tasks.is_empty() {
        return Vec::new();
    }
    let k_sub = tasks[0].1.len();
    debug_assert!(tasks.iter().all(|t| t.1.len() == k_sub));
    let g = norm_profiles.nrows();
    let m = tasks.len();
    let mut out: Vec<Array1<f64>> = Vec::with_capacity(m);

    for start in (0..m).step_by(batch_size.max(1)) {
        let end = (start + batch_size).min(m);
        let bs = end - start;
        let mut s = ndarray::Array3::<f64>::zeros((bs, g, k_sub));
        let y_b = Array2::from_shape_fn((bs, g), |(i, gg)| {
            let n = tasks[start + i].0;
            spatial_counts[[n, gg]]
        });
        let n_b: Array1<f64> = (0..bs).map(|i| spatial_numi[tasks[start + i].0]).collect();

        for i in 0..bs {
            let n = tasks[start + i].0;
            let u = spatial_numi[n];
            for j in 0..k_sub {
                let t = tasks[start + i].1[j];
                for gg in 0..g {
                    s[[i, gg, j]] = u * norm_profiles[[gg, t]];
                }
            }
        }

        let (w, _) = solve_irwls_batch_s_ndarray(
            &s, &y_b, &n_b, q_mat, sq_mat, x_vals, max_iter, 0.001, 0.3, false, false,
        );

        for i in 0..bs {
            out.push(w.row(i).to_owned());
        }
        notify_batch(progress, end - start);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn run_multi_mode(
    spatial_counts: &Array2<f64>,
    spatial_numi: &Array1<f64>,
    norm_profiles: &Array2<f64>,
    cell_type_names: Vec<String>,
    q_mat: &Array2<f64>,
    sq_mat: &Array2<f64>,
    x_vals: &Array1<f64>,
    config: &RctdConfig,
    batch_size: usize,
    device: &RctdDevice,
    progress: Option<BatchProgress>,
) -> MultiResult {
    let n = spatial_counts.nrows();
    let max_t = config.max_multi_types;
    let full = run_full_mode(
        spatial_counts,
        spatial_numi,
        norm_profiles,
        q_mat,
        sq_mat,
        x_vals,
        batch_size,
        device,
        progress.clone(),
    );
    let weights_out = full.weights.clone();
    let w_sel = full.weights.mapv(|x| (x * 1e8).round() / 1e8);

    let mut candidates_list: Vec<BTreeSet<usize>> = Vec::with_capacity(n);
    for row in w_sel.rows() {
        let mut hs: BTreeSet<usize> = BTreeSet::new();
        for (i, &val) in row.iter().enumerate() {
            if val > 0.01 {
                hs.insert(i);
            }
        }
        if hs.is_empty() {
            let mut best = 0usize;
            let mut best_v = row[0];
            for (i, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best = i;
                    best_v = v;
                }
            }
            hs.insert(best);
        }
        candidates_list.push(hs);
    }

    let inf = 1e18f64;
    let mut current_scores = vec![inf; n];
    let mut cell_type_lists: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut active_pixels: BTreeSet<usize> = (0..n).collect();

    for _ in 1..=max_t {
        if active_pixels.is_empty() {
            break;
        }

        let mut tasks: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut task_info: Vec<(usize, usize)> = Vec::new();

        for &pix in &active_pixels {
            for &cand in &candidates_list[pix] {
                let mut cur = cell_type_lists[pix].clone();
                cur.push(cand);
                tasks.push((pix, cur));
                task_info.push((pix, cand));
            }
        }

        let scores = run_batched_scoring(
            &tasks,
            spatial_numi,
            spatial_counts,
            norm_profiles,
            q_mat,
            sq_mat,
            x_vals,
            batch_size,
            25,
            &progress,
        );

        let mut best_cand_for_pixel: HashMap<usize, usize> = HashMap::new();
        let mut best_score_for_pixel: HashMap<usize, f64> = HashMap::new();
        for (i, &(pix, cand)) in task_info.iter().enumerate() {
            let sc = scores[i];
            match best_score_for_pixel.get(&pix) {
                None => {
                    best_score_for_pixel.insert(pix, sc);
                    best_cand_for_pixel.insert(pix, cand);
                }
                Some(&prev) if sc < prev => {
                    best_score_for_pixel.insert(pix, sc);
                    best_cand_for_pixel.insert(pix, cand);
                }
                _ => {}
            }
        }

        let mut new_active: BTreeSet<usize> = BTreeSet::new();
        for &pix in &active_pixels {
            let Some(&min_score) = best_score_for_pixel.get(&pix) else {
                continue;
            };
            let best_cand = best_cand_for_pixel[&pix];
            let thresh = config.doublet_threshold;

            if min_score > current_scores[pix] - thresh {
                if cell_type_lists[pix].is_empty() {
                    cell_type_lists[pix].push(best_cand);
                    candidates_list[pix].remove(&best_cand);
                    current_scores[pix] = min_score;
                }
            } else {
                cell_type_lists[pix].push(best_cand);
                candidates_list[pix].remove(&best_cand);
                current_scores[pix] = min_score;
                new_active.insert(pix);
            }
        }
        active_pixels = new_active;
    }

    let mut conf_lists: Vec<HashMap<usize, bool>> = vec![HashMap::new(); n];
    for pix in 0..n {
        for &t in &cell_type_lists[pix] {
            conf_lists[pix].insert(t, true);
        }
    }

    let mut conf_tasks: BTreeMap<usize, Vec<ConfTask>> = BTreeMap::new();
    for pix in 0..n {
        let ct_list = &cell_type_lists[pix];
        if ct_list.is_empty() {
            continue;
        }
        let k_sub = ct_list.len();
        for &t in ct_list {
            for &newtype in &candidates_list[pix] {
                let cur_list: Vec<usize> = ct_list
                    .iter()
                    .copied()
                    .filter(|&x| x != t)
                    .chain([newtype])
                    .collect();
                conf_tasks
                    .entry(k_sub)
                    .or_default()
                    .push((pix, cur_list, t, newtype));
            }
        }
    }

    for t_list in conf_tasks.values() {
        if t_list.is_empty() {
            continue;
        }
        let k_sub_conf = t_list[0].1.len();
        debug_assert!(t_list.iter().all(|x| x.1.len() == k_sub_conf));
        let base_tasks: Vec<(usize, Vec<usize>)> =
            t_list.iter().map(|x| (x.0, x.1.clone())).collect();
        let scores = run_batched_scoring(
            &base_tasks,
            spatial_numi,
            spatial_counts,
            norm_profiles,
            q_mat,
            sq_mat,
            x_vals,
            batch_size,
            25,
            &progress,
        );
        for (i, (pix, _, t, _)) in t_list.iter().enumerate() {
            if !conf_lists[*pix].get(t).copied().unwrap_or(true) {
                continue;
            }
            if scores[i] < current_scores[*pix] + config.confidence_threshold {
                conf_lists[*pix].insert(*t, false);
            }
        }
    }

    let mut final_tasks: BTreeMap<usize, Vec<(usize, Vec<usize>)>> = BTreeMap::new();
    for (pix, ct_list) in cell_type_lists.iter_mut().enumerate() {
        let mut k_sub = ct_list.len();
        if k_sub == 0 {
            k_sub = 1;
            let row = w_sel.row(pix);
            let mut best = 0usize;
            let mut best_v = row[0];
            for (i, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best = i;
                    best_v = v;
                }
            }
            *ct_list = vec![best];
        }
        let k_sub = ct_list.len();
        final_tasks
            .entry(k_sub)
            .or_default()
            .push((pix, ct_list.clone()));
    }

    let mut final_weights: HashMap<usize, Array1<f64>> = HashMap::new();
    for t_list in final_tasks.values() {
        if t_list.is_empty() {
            continue;
        }
        let w_rows = run_batched_weights(
            t_list,
            spatial_numi,
            spatial_counts,
            norm_profiles,
            q_mat,
            sq_mat,
            x_vals,
            batch_size,
            50,
            &progress,
        );
        for (i, (pix, _)) in t_list.iter().enumerate() {
            let mut w_norm = w_rows[i].clone();
            let s: f64 = w_norm.sum();
            if s > 0.0 {
                w_norm.mapv_inplace(|x| x / s);
            }
            final_weights.insert(*pix, w_norm);
        }
    }

    let mut sub_weights = ndarray::Array2::<f32>::zeros((n, max_t));
    let mut cell_type_indices = ndarray::Array2::<i32>::from_elem((n, max_t), -1);
    let mut n_types = Array1::<i32>::zeros(n);
    let mut conf_list_arr = ndarray::Array2::<bool>::from_elem((n, max_t), false);
    let min_score_arr = Array1::from_vec(current_scores.iter().map(|&s| s as f32).collect());

    for pix in 0..n {
        let ct_list = &cell_type_lists[pix];
        let k_sub = ct_list.len() as i32;
        n_types[pix] = k_sub;
        let fw = final_weights.get(&pix).expect("final weights");
        for (i, &t) in ct_list.iter().enumerate() {
            cell_type_indices[[pix, i]] = t as i32;
            sub_weights[[pix, i]] = fw[i] as f32;
            conf_list_arr[[pix, i]] = conf_lists[pix].get(&t).copied().unwrap_or(true);
        }
    }

    MultiResult {
        weights: weights_out,
        sub_weights,
        cell_type_indices,
        n_types,
        conf_list: conf_list_arr,
        min_score: min_score_arr,
        cell_type_names,
    }
}
