use itertools::Itertools;
use ndarray::{Array1, Array2, Array3};

use crate::backend::RctdDevice;
use crate::full::run_full_mode;
use crate::irwls_native::{calc_neg_loglik_row_sums, solve_irwls_batch_s_ndarray};
use crate::types::{
    BatchProgress, DoubletResult, RctdConfig, SPOT_CLASS_DOUBLET_CERTAIN,
    SPOT_CLASS_DOUBLET_UNCERTAIN, SPOT_CLASS_REJECT, SPOT_CLASS_SINGLET, notify_batch,
};

fn s_pair_batch(
    p: &Array2<f64>,
    pix: &[usize],
    t1i: &[usize],
    t2i: &[usize],
    numi: &Array1<f64>,
) -> Array3<f64> {
    let bs = pix.len();
    let g = p.nrows();
    let mut s = Array3::zeros((bs, g, 2));
    for i in 0..bs {
        let u = numi[pix[i]];
        for gg in 0..g {
            s[[i, gg, 0]] = u * p[[gg, t1i[i]]];
            s[[i, gg, 1]] = u * p[[gg, t2i[i]]];
        }
    }
    s
}

fn s_single_batch(p: &Array2<f64>, pix: &[usize], ti: &[usize], numi: &Array1<f64>) -> Array3<f64> {
    let bs = pix.len();
    let g = p.nrows();
    let mut s = Array3::zeros((bs, g, 1));
    for i in 0..bs {
        let u = numi[pix[i]];
        for gg in 0..g {
            s[[i, gg, 0]] = u * p[[gg, ti[i]]];
        }
    }
    s
}

#[allow(clippy::too_many_arguments)]
pub fn run_doublet_mode(
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
) -> DoubletResult {
    let n = spatial_counts.nrows();
    let k = norm_profiles.ncols();
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

    let mut candidates_list: Vec<Vec<usize>> = Vec::with_capacity(n);
    for row in w_sel.rows() {
        let mut cands: Vec<usize> = Vec::new();
        for (i, &val) in row.iter().enumerate() {
            if val > 0.01 {
                cands.push(i);
            }
        }
        if cands.is_empty() {
            cands = (0..k.min(3)).collect();
        } else if cands.len() == 1 {
            if cands[0] == 0 {
                cands.push(1);
            } else {
                cands.push(0);
            }
        }
        candidates_list.push(cands);
    }

    let mut triples: Vec<(usize, usize, usize)> = Vec::new();
    for (n_idx, cands) in candidates_list.iter().enumerate() {
        for pair in cands.iter().copied().combinations(2) {
            let t1 = pair[0];
            let t2 = pair[1];
            triples.push((n_idx, t1, t2));
        }
    }

    let mut pair_log_l: std::collections::HashMap<(usize, usize, usize), f64> =
        std::collections::HashMap::new();
    let mut pair_weights: std::collections::HashMap<(usize, usize, usize), [f64; 2]> =
        std::collections::HashMap::new();

    for start in (0..triples.len()).step_by(batch_size.max(1)) {
        let end = (start + batch_size).min(triples.len());
        let tr = &triples[start..end];
        let pix: Vec<usize> = tr.iter().map(|t| t.0).collect();
        let t1v: Vec<usize> = tr.iter().map(|t| t.1).collect();
        let t2v: Vec<usize> = tr.iter().map(|t| t.2).collect();
        let s_b = s_pair_batch(norm_profiles, &pix, &t1v, &t2v, spatial_numi);
        let y_b = Array2::from_shape_fn((tr.len(), spatial_counts.ncols()), |(i, g)| {
            spatial_counts[[pix[i], g]]
        });
        let n_b: Array1<f64> = pix.iter().map(|&i| spatial_numi[i]).collect();
        let (w_b, _) = solve_irwls_batch_s_ndarray(
            &s_b, &y_b, &n_b, q_mat, sq_mat, x_vals, 25, 0.001, 0.3, false, false,
        );
        let mut exp = Array2::zeros((tr.len(), spatial_counts.ncols()));
        for i in 0..tr.len() {
            for g in 0..spatial_counts.ncols() {
                exp[[i, g]] =
                    (s_b[[i, g, 0]] * w_b[[i, 0]] + s_b[[i, g, 1]] * w_b[[i, 1]]).max(1e-4);
            }
        }
        let scs = calc_neg_loglik_row_sums(&y_b, &exp, q_mat, sq_mat, x_vals, -1);
        for (i, t) in tr.iter().enumerate() {
            pair_log_l.insert((t.0, t.1, t.2), scs[i]);
            pair_weights.insert((t.0, t.1, t.2), [w_b[[i, 0]], w_b[[i, 1]]]);
        }
        notify_batch(&progress, end - start);
    }

    let mut singles: Vec<(usize, usize)> = Vec::new();
    for (n_idx, cands) in candidates_list.iter().enumerate() {
        for &t in cands {
            singles.push((n_idx, t));
        }
    }
    singles.sort_unstable();
    singles.dedup();

    let mut singlet_log_l: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();
    for start in (0..singles.len()).step_by(batch_size.max(1)) {
        let end = (start + batch_size).min(singles.len());
        let sg = &singles[start..end];
        let pix: Vec<usize> = sg.iter().map(|t| t.0).collect();
        let tv: Vec<usize> = sg.iter().map(|t| t.1).collect();
        let s_b = s_single_batch(norm_profiles, &pix, &tv, spatial_numi);
        let y_b = Array2::from_shape_fn((sg.len(), spatial_counts.ncols()), |(i, g)| {
            spatial_counts[[pix[i], g]]
        });
        let n_b: Array1<f64> = pix.iter().map(|&i| spatial_numi[i]).collect();
        let (w_b, _) = solve_irwls_batch_s_ndarray(
            &s_b, &y_b, &n_b, q_mat, sq_mat, x_vals, 25, 0.001, 0.3, false, false,
        );
        let mut exp = Array2::zeros((sg.len(), spatial_counts.ncols()));
        for i in 0..sg.len() {
            for g in 0..spatial_counts.ncols() {
                exp[[i, g]] = (s_b[[i, g, 0]] * w_b[[i, 0]]).max(1e-4);
            }
        }
        let scs = calc_neg_loglik_row_sums(&y_b, &exp, q_mat, sq_mat, x_vals, -1);
        for (i, t) in sg.iter().enumerate() {
            singlet_log_l.insert(*t, scs[i]);
        }
        notify_batch(&progress, end - start);
    }

    const INF: f64 = 1e18;
    let mut weights_doublet = ndarray::Array2::<f32>::zeros((n, 2));
    let mut spot_class = ndarray::Array1::<i32>::zeros(n);
    let mut first_type = ndarray::Array1::<i32>::zeros(n);
    let mut second_type = ndarray::Array1::<i32>::zeros(n);
    let mut first_class = ndarray::Array1::<bool>::from_elem(n, false);
    let mut second_class = ndarray::Array1::<bool>::from_elem(n, false);
    let mut min_score = ndarray::Array1::<f32>::zeros(n);
    let mut singlet_score_res = ndarray::Array1::<f32>::zeros(n);

    for n_idx in 0..n {
        let cands = &candidates_list[n_idx];
        let c = cands.len();
        let sing_scores: std::collections::HashMap<usize, f64> = cands
            .iter()
            .map(|&t| (t, *singlet_log_l.get(&(n_idx, t)).unwrap_or(&INF)))
            .collect();

        let mut score_mat: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        let mut min_p_score = INF;
        let (mut best_t1, mut best_t2) = (cands[0], if c > 1 { cands[1] } else { cands[0] });
        for i in 0..c {
            for j in (i + 1)..c {
                let t1 = cands[i];
                let t2 = cands[j];
                let sc = *pair_log_l.get(&(n_idx, t1, t2)).unwrap_or(&INF);
                score_mat.insert((t1, t2), sc);
                score_mat.insert((t2, t1), sc);
                if sc < min_p_score {
                    min_p_score = sc;
                    best_t1 = t1;
                    best_t2 = t2;
                }
            }
        }

        let check_pairs = |my_type: usize| -> (bool, f64) {
            let mut all_pairs = true;
            for i in 0..c {
                for j in 0..c {
                    if i == j {
                        continue;
                    }
                    let t1 = cands[i];
                    let t2 = cands[j];
                    let sc = *score_mat.get(&(t1, t2)).unwrap_or(&INF);
                    if sc < min_p_score + config.confidence_threshold
                        && t1 != my_type
                        && t2 != my_type
                    {
                        all_pairs = false;
                    }
                }
            }
            (all_pairs, *sing_scores.get(&my_type).unwrap_or(&INF))
        };

        let (type1_all_pairs, type1_sing) = check_pairs(best_t1);
        let (type2_all_pairs, type2_sing) = check_pairs(best_t2);
        let mut s_class;
        let s_score;
        let mut f_class = false;
        let mut sc_class = false;

        if !type1_all_pairs && !type2_all_pairs {
            s_class = SPOT_CLASS_REJECT;
            s_score = min_p_score + 2.0 * config.doublet_threshold;
        } else if type1_all_pairs && !type2_all_pairs {
            s_class = SPOT_CLASS_DOUBLET_UNCERTAIN;
            s_score = type1_sing;
        } else if !type1_all_pairs && type2_all_pairs {
            s_class = SPOT_CLASS_DOUBLET_UNCERTAIN;
            std::mem::swap(&mut best_t1, &mut best_t2);
            s_score = type2_sing;
        } else {
            s_class = SPOT_CLASS_DOUBLET_CERTAIN;
            s_score = type1_sing.min(type2_sing);
            if type2_sing < type1_sing {
                std::mem::swap(&mut best_t1, &mut best_t2);
            }
        }

        if s_score - min_p_score < config.doublet_threshold {
            s_class = SPOT_CLASS_SINGLET;
        }

        let mut dw = if let Some(w) = pair_weights.get(&(n_idx, best_t1, best_t2)) {
            *w
        } else if let Some(w) = pair_weights.get(&(n_idx, best_t2, best_t1)) {
            let mut a = *w;
            a.reverse();
            a
        } else {
            [
                weights_out[[n_idx, best_t1]] as f64,
                weights_out[[n_idx, best_t2]] as f64,
            ]
        };
        let ssum: f64 = dw.iter().sum();
        if ssum > 0.0 {
            dw[0] /= ssum;
            dw[1] /= ssum;
        } else {
            dw = [0.5, 0.5];
        }

        let mut first_t = best_t1;
        let mut second_t = best_t2;
        if s_class == SPOT_CLASS_SINGLET {
            let best_sing_type = *sing_scores
                .iter()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            first_t = best_sing_type;
            second_t = if best_t1 == best_sing_type {
                best_t2
            } else {
                best_t1
            };
            f_class = false;
            sc_class = false;
        }

        spot_class[n_idx] = s_class;
        first_type[n_idx] = first_t as i32;
        second_type[n_idx] = second_t as i32;
        min_score[n_idx] = min_p_score as f32;
        singlet_score_res[n_idx] = s_score as f32;
        first_class[n_idx] = f_class;
        second_class[n_idx] = sc_class;
        weights_doublet[[n_idx, 0]] = dw[0] as f32;
        weights_doublet[[n_idx, 1]] = dw[1] as f32;
    }

    let mut final_t1 = first_type.clone();
    let mut final_t2 = second_type.clone();
    for n_idx in 0..n {
        if spot_class[n_idx] == SPOT_CLASS_SINGLET {
            final_t1[n_idx] = first_type[n_idx];
            final_t2[n_idx] = second_type[n_idx];
        }
    }

    for start in (0..n).step_by(batch_size.max(1)) {
        let end = (start + batch_size).min(n);
        let pix: Vec<usize> = (start..end).collect();
        let t1v: Vec<usize> = pix.iter().map(|&i| final_t1[i] as usize).collect();
        let t2v: Vec<usize> = pix.iter().map(|&i| final_t2[i] as usize).collect();
        let s_b = s_pair_batch(norm_profiles, &pix, &t1v, &t2v, spatial_numi);
        let y_b = spatial_counts.slice(ndarray::s![start..end, ..]).to_owned();
        let n_b = spatial_numi.slice(ndarray::s![start..end]).to_owned();
        let (w_b, _) = solve_irwls_batch_s_ndarray(
            &s_b, &y_b, &n_b, q_mat, sq_mat, x_vals, 50, 0.001, 0.3, false, false,
        );
        for i in 0..pix.len() {
            let row_sum: f32 = w_b.row(i).iter().map(|x| *x as f32).sum();
            let norm = row_sum.max(1e-10);
            weights_doublet[[start + i, 0]] = w_b[[i, 0]] as f32 / norm;
            weights_doublet[[start + i, 1]] = w_b[[i, 1]] as f32 / norm;
        }
        notify_batch(&progress, end - start);
    }

    DoubletResult {
        weights: weights_out,
        weights_doublet,
        spot_class,
        first_type,
        second_type,
        first_class,
        second_class,
        min_score,
        singlet_score: singlet_score_res,
        cell_type_names,
    }
}
