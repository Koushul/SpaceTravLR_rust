//! Pure ndarray + rayon IRWLS solver for CPU f64.
//! Bypasses Burn tensor overhead: fused calc_q, parallel psd+qp+simplex,
//! direct f64 slices with no FloatElem conversion.

use nalgebra::{linalg::SymmetricEigen, DMatrix};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use rayon::prelude::*;

pub struct NativeSharedPrepared {
    p: Array2<f64>,
    p_t: Array2<f64>,
    p_outer: Array2<f64>,
    q_flat: Vec<f64>,
    sq_flat: Vec<f64>,
    x_vals: Vec<f64>,
    nk: usize,
    nx: usize,
    pub(crate) k: usize,
    k_val_clamp: f64,
    x_max: f64,
}

impl NativeSharedPrepared {
    pub fn new(
        p: &Array2<f64>,
        q_mat: &Array2<f64>,
        sq_mat: &Array2<f64>,
        x_vals: &Array1<f64>,
    ) -> Self {
        let k = p.ncols();
        let g = p.nrows();
        let nk = q_mat.nrows();
        let nx = q_mat.ncols();
        let p_t = p.t().to_owned();
        let mut p_outer = Array2::zeros((g, k * k));
        for gi in 0..g {
            for a in 0..k {
                for b in 0..k {
                    p_outer[[gi, a * k + b]] = p[[gi, a]] * p[[gi, b]];
                }
            }
        }
        let q_flat: Vec<f64> = q_mat.as_standard_layout().iter().copied().collect();
        let sq_flat: Vec<f64> = sq_mat.as_standard_layout().iter().copied().collect();
        let x_vals_vec: Vec<f64> = x_vals.iter().copied().collect();
        let x_max = x_vals_vec[nx - 1];
        let k_val_clamp = (nk as i64 - 3) as f64;
        Self {
            p: p.to_owned(),
            p_t,
            p_outer,
            q_flat,
            sq_flat,
            x_vals: x_vals_vec,
            nk,
            nx,
            k,
            k_val_clamp,
            x_max,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn calc_q_chunk(
    y_chunk: &[f64],
    pred_chunk: &[f64],
    d0_chunk: &mut [f64],
    d1_chunk: &mut [f64],
    d2_chunk: &mut [f64],
    q_flat: &[f64],
    sq_flat: &[f64],
    x_vals: &[f64],
    nk: usize,
    nx: usize,
    x_max: f64,
) {
    let nkg = nk * nx;
    let eps = 1e-4f64;
    let delta = 1e-6f64;
    for idx in 0..y_chunk.len() {
        let yi = y_chunk[idx];
        let li = pred_chunk[idx].clamp(eps, x_max - eps);
        let l_f = (li / delta).sqrt().floor();
        let l_i = l_f as i64;
        let m_raw =
            (l_i - 9).min(40) + (((l_f - 48.7499).max(0.0) * 4.0).sqrt().ceil() as i64 - 2).max(0);
        let m = m_raw.max(0).min(nx as i64 - 1) as usize;
        let m1 = (m_raw - 1).max(0).min(nx as i64 - 1) as usize;
        let ti1 = x_vals[m1];
        let ti = x_vals[m];
        let hi = ti - ti1;
        let y_idx = (yi as i64).max(0).min(nk as i64 - 1) as usize;
        let lin1 = (y_idx * nx + m1).min(nkg - 1);
        let lin2 = (y_idx * nx + m).min(nkg - 1);
        let fti1 = q_flat[lin1];
        let fti = q_flat[lin2];
        let zi1 = sq_flat[lin1];
        let zi = sq_flat[lin2];
        let diff1 = li - ti1;
        let diff2 = ti - li;
        let zdi = zi / hi;
        let zdi1 = zi1 / hi;
        let diff3 = fti / hi - zi * hi / 6.0;
        let diff4 = fti1 / hi - zi1 * hi / 6.0;
        d0_chunk[idx] =
            zdi * diff1.powi(3) / 6.0 + zdi1 * diff2.powi(3) / 6.0 + diff3 * diff1 + diff4 * diff2;
        d1_chunk[idx] = zdi * diff1 * diff1 * 0.5 - zdi1 * diff2 * diff2 * 0.5 + diff3 - diff4;
        d2_chunk[idx] = zdi * diff1 + zdi1 * diff2;
    }
}

fn psd_normalize_qp(
    hess: &[f64],
    grad: &[f64],
    solution: &[f64],
    dw: &mut [f64],
    k: usize,
    epsilon: f64,
) {
    let kk = k * k;
    let mut h_psd = vec![0.0f64; kk];
    let max_eig;
    if k == 1 {
        h_psd[0] = hess[0].max(epsilon);
        max_eig = h_psd[0];
    } else if k == 2 {
        let a = hess[0];
        let b = hess[1];
        let d = hess[3];
        let half_trace = (a + d) * 0.5;
        let disc = (((a - d) * 0.5).powi(2) + b * b).sqrt();
        let lam1 = (half_trace - disc).max(epsilon);
        let lam2 = (half_trace + disc).max(epsilon);
        max_eig = lam2;
        let safe_disc = disc.max(1e-30);
        let cos2t = (a - d) * 0.5 / safe_disc;
        let sin2t = b / safe_disc;
        let half_sum = (lam1 + lam2) * 0.5;
        let half_diff = (lam2 - lam1) * 0.5;
        h_psd[0] = half_sum + half_diff * cos2t;
        h_psd[1] = half_diff * sin2t;
        h_psd[2] = h_psd[1];
        h_psd[3] = half_sum - half_diff * cos2t;
    } else {
        let mut mat = DMatrix::from_row_slice(k, k, hess);
        if mat.iter().any(|v: &f64| v.is_nan()) {
            mat.fill(0.0);
            for j in 0..k {
                mat[(j, j)] = epsilon;
            }
        }
        let se = SymmetricEigen::new(mat);
        let mut evals = se.eigenvalues.clone();
        for e in evals.iter_mut() {
            *e = e.max(epsilon);
        }
        let v = &se.eigenvectors;
        let reconstructed = v * DMatrix::from_diagonal(&evals) * v.transpose();
        max_eig = *evals
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        for r in 0..k {
            for c in 0..k {
                h_psd[r * k + c] = reconstructed[(r, c)];
            }
        }
    }

    let nf = max_eig.max(1e-10);

    if k == 2 {
        let d00 = h_psd[0] / nf + 1e-7;
        let d01 = h_psd[1] / nf;
        let d11 = h_psd[3] / nf + 1e-7;
        let di0 = -grad[0] / nf;
        let di1 = -grad[1] / nf;
        let lb0 = -solution[0];
        let lb1 = -solution[1];
        let det = d00 * d11 - d01 * d01;
        let det_sign = det.signum();
        let det_abs = det.abs().max(1e-30) * det_sign;
        let mut x1 = (d00 * di1 - d01 * di0) / det_abs;
        x1 = x1.max(lb1);
        let x0 = ((di0 - d01 * x1) / d00).max(lb0);
        let x1 = ((di1 - d01 * x0) / d11).max(lb1);
        dw[0] = x0;
        dw[1] = x1;
    } else {
        let mut d_mat = vec![0.0f64; kk];
        let mut d_vec = vec![0.0f64; k];
        let mut lb = vec![0.0f64; k];
        for r in 0..k {
            d_vec[r] = -grad[r] / nf;
            lb[r] = -solution[r];
            for c in 0..k {
                d_mat[r * k + c] = h_psd[r * k + c] / nf;
                if r == c {
                    d_mat[r * k + c] += 1e-7;
                }
            }
        }
        for j in 0..k {
            dw[j] = (d_vec[j] / d_mat[j * k + j]).max(lb[j]);
        }
        for _ in 0..50 {
            for j in 0..k {
                let djj = d_mat[j * k + j];
                let mut dot = 0.0;
                for t in 0..k {
                    dot += d_mat[j * k + t] * dw[t];
                }
                dw[j] = ((d_vec[j] - dot + djj * dw[j]) / djj).max(lb[j]);
            }
        }
    }
}

fn simplex_project_slice(v: &mut [f64]) {
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut cssv = 0.0f64;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        if ui * (i + 1) as f64 > cssv - 1.0 {
            rho = i + 1;
        }
    }
    let theta = if rho == 0 {
        0.0
    } else {
        (u[..rho].iter().sum::<f64>() - 1.0) / rho as f64
    };
    for vi in v.iter_mut() {
        *vi = (*vi - theta).max(0.0);
    }
}

fn weight_derivatives(d1s: &mut [f64], d2s: &mut [f64], numi_act: &[f64], g: usize) {
    for (i, &nu) in numi_act.iter().enumerate() {
        let base = i * g;
        for j in 0..g {
            d1s[base + j] *= -nu;
            d2s[base + j] *= -(nu * nu);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn solve_irwls_native(
    prep: &NativeSharedPrepared,
    y: ArrayView2<f64>,
    numi: ArrayView1<f64>,
    max_iter: usize,
    min_change: f64,
    step_size: f64,
    constrain: bool,
    bulk_mode: bool,
) -> (Array2<f64>, Array1<bool>) {
    let n = y.nrows();
    let g = y.ncols();
    let k = prep.k;
    if n == 0 {
        return (Array2::zeros((0, k)), Array1::default(0));
    }
    let kk = k * k;

    let y_data: Array2<f64> = if bulk_mode {
        y.to_owned()
    } else {
        y.mapv(|v| v.min(prep.k_val_clamp))
    };

    let threshold: Array1<f64> = numi.mapv(|u| (u * 1e-7).max(1e-4));
    let numi_owned: Vec<f64> = numi.iter().copied().collect();
    let threshold_owned: Vec<f64> = threshold.iter().copied().collect();

    let mut w = Array2::from_elem((n, k), 1.0 / k as f64);
    let mut converged = Array1::from_elem(n, false);
    let mut active: Vec<usize> = (0..n).collect();

    for _it in 0..max_iter {
        let n_act = active.len();
        if n_act == 0 {
            break;
        }

        let mut solution = Array2::zeros((n_act, k));
        let mut y_act = Array2::zeros((n_act, g));
        let mut numi_act = vec![0.0f64; n_act];
        let mut thresh_act = vec![0.0f64; n_act];
        {
            let w_sl = w.as_slice().unwrap();
            let y_sl = y_data.as_slice().unwrap();
            let sol_sl = solution.as_slice_mut().unwrap();
            let ya_sl = y_act.as_slice_mut().unwrap();
            for (j, &ai) in active.iter().enumerate() {
                for t in 0..k {
                    sol_sl[j * k + t] = w_sl[ai * k + t].max(0.0);
                }
                ya_sl[j * g..(j + 1) * g].copy_from_slice(&y_sl[ai * g..(ai + 1) * g]);
                numi_act[j] = numi_owned[ai];
                thresh_act[j] = threshold_owned[ai];
            }
        }

        // pred = solution @ P^T, then apply |pred * numi| clamped to threshold
        let mut pred = solution.dot(&prep.p_t);
        {
            let ps = pred.as_slice_mut().unwrap();
            for i in 0..n_act {
                let (nu, th) = (numi_act[i], thresh_act[i]);
                for j in 0..g {
                    let idx = i * g + j;
                    ps[idx] = (ps[idx] * nu).abs().max(th);
                }
            }
        }

        // Derivatives → grad (n_act, k) and hess (n_act, k²)
        let (grad, hess) = if bulk_mode {
            let mut d1 = Array2::zeros((n_act, g));
            let mut d2 = Array2::zeros((n_act, g));
            {
                let d1s = d1.as_slice_mut().unwrap();
                let d2s = d2.as_slice_mut().unwrap();
                let ys = y_act.as_slice().unwrap();
                let ps = pred.as_slice().unwrap();
                for idx in 0..n_act * g {
                    let p = ps[idx];
                    let yv = ys[idx] + 1e-10;
                    d1s[idx] = (p.ln() - yv.ln()) / p * -2.0;
                    d2s[idx] = (1.0 - p.ln() + yv.ln()) / (p * p) * -2.0;
                }
            }
            weight_derivatives(
                d1.as_slice_mut().unwrap(),
                d2.as_slice_mut().unwrap(),
                &numi_act,
                g,
            );
            (d1.dot(&prep.p), d2.dot(&prep.p_outer))
        } else {
            let mut d0 = Array2::zeros((n_act, g));
            let mut d1 = Array2::zeros((n_act, g));
            let mut d2 = Array2::zeros((n_act, g));
            {
                let d0s = d0.as_slice_mut().unwrap();
                let d1s = d1.as_slice_mut().unwrap();
                let d2s = d2.as_slice_mut().unwrap();
                let ys = y_act.as_slice().unwrap();
                let ps = pred.as_slice().unwrap();
                d0s.par_chunks_mut(g)
                    .zip(d1s.par_chunks_mut(g))
                    .zip(d2s.par_chunks_mut(g))
                    .zip(ys.par_chunks(g))
                    .zip(ps.par_chunks(g))
                    .for_each(|((((d0c, d1c), d2c), yc), pc)| {
                        calc_q_chunk(
                            yc,
                            pc,
                            d0c,
                            d1c,
                            d2c,
                            &prep.q_flat,
                            &prep.sq_flat,
                            &prep.x_vals,
                            prep.nk,
                            prep.nx,
                            prep.x_max,
                        );
                    });
            }
            weight_derivatives(
                d1.as_slice_mut().unwrap(),
                d2.as_slice_mut().unwrap(),
                &numi_act,
                g,
            );
            (d1.dot(&prep.p), d2.dot(&prep.p_outer))
        };

        // Parallel PSD + normalize + box-QP
        let mut delta_w = vec![0.0f64; n_act * k];
        {
            let gs = grad.as_slice().unwrap();
            let hs = hess.as_slice().unwrap();
            let ss = solution.as_slice().unwrap();
            delta_w.par_chunks_mut(k).enumerate().for_each(|(i, dw)| {
                psd_normalize_qp(
                    &hs[i * kk..(i + 1) * kk],
                    &gs[i * k..(i + 1) * k],
                    &ss[i * k..(i + 1) * k],
                    dw,
                    k,
                    1e-3,
                );
            });
        }

        // w_new = solution + delta_w * step_size
        let sol_sl = solution.as_slice().unwrap();
        let mut w_new = vec![0.0f64; n_act * k];
        for i in 0..n_act * k {
            w_new[i] = sol_sl[i] + delta_w[i] * step_size;
        }

        if constrain {
            w_new.par_chunks_mut(k).for_each(simplex_project_slice);
        }

        // Scatter results and compact active list
        let mut next = Vec::new();
        for (j, &ai) in active.iter().enumerate() {
            let mut change = 0.0f64;
            for t in 0..k {
                change += (w_new[j * k + t] - w[[ai, t]]).abs();
                w[[ai, t]] = w_new[j * k + t];
            }
            if change <= min_change {
                converged[ai] = true;
            } else {
                next.push(ai);
            }
        }
        if next.is_empty() {
            break;
        }
        active = next;
    }

    (w, converged)
}

#[allow(clippy::too_many_arguments)]
pub fn solve_irwls_batch_s_ndarray(
    s: &Array3<f64>,
    y: &Array2<f64>,
    numi: &Array1<f64>,
    q_mat: &Array2<f64>,
    sq_mat: &Array2<f64>,
    x_vals: &Array1<f64>,
    max_iter: usize,
    min_change: f64,
    step_size: f64,
    constrain: bool,
    bulk_mode: bool,
) -> (Array2<f64>, Array1<bool>) {
    let n = s.shape()[0];
    let g = s.shape()[1];
    let k = s.shape()[2];
    if n == 0 {
        return (Array2::zeros((0, k)), Array1::default(0));
    }

    let nk = q_mat.nrows();
    let nx = q_mat.ncols();
    let q_flat: Vec<f64> = q_mat.as_standard_layout().iter().copied().collect();
    let sq_flat: Vec<f64> = sq_mat.as_standard_layout().iter().copied().collect();
    let x_vals_vec: Vec<f64> = x_vals.iter().copied().collect();
    let x_max = x_vals_vec[nx - 1];
    let k_val_clamp = (nk as i64 - 3) as f64;

    let y_data: Array2<f64> = if bulk_mode {
        y.to_owned()
    } else {
        y.mapv(|v| v.min(k_val_clamp))
    };

    let threshold: Array1<f64> = numi.mapv(|u| (u * 1e-7).max(1e-4));
    let mut w = Array2::from_elem((n, k), 1.0 / k as f64);
    let mut converged = Array1::from_elem(n, false);

    for _it in 0..max_iter {
        if converged.iter().all(|&c| c) {
            break;
        }

        let solution = w.mapv(|x| x.max(0.0));
        let mut pred = Array2::<f64>::zeros((n, g));
        for i in 0..n {
            let th = threshold[i];
            for gg in 0..g {
                let mut sum = 0.0f64;
                for t in 0..k {
                    sum += s[[i, gg, t]] * solution[[i, t]];
                }
                pred[[i, gg]] = sum.abs().max(th);
            }
        }

        let mut d1 = Array2::zeros((n, g));
        let mut d2 = Array2::zeros((n, g));
        if bulk_mode {
            let d1s = d1.as_slice_mut().unwrap();
            let d2s = d2.as_slice_mut().unwrap();
            let ys = y_data.as_slice().unwrap();
            let ps = pred.as_slice().unwrap();
            for idx in 0..n * g {
                let p = ps[idx];
                let yv = ys[idx] + 1e-10;
                d1s[idx] = (p.ln() - yv.ln()) / p * -2.0;
                d2s[idx] = (1.0 - p.ln() + yv.ln()) / (p * p) * -2.0;
            }
        } else {
            let mut d0 = Array2::zeros((n, g));
            {
                let d0s = d0.as_slice_mut().unwrap();
                let d1s = d1.as_slice_mut().unwrap();
                let d2s = d2.as_slice_mut().unwrap();
                let ys = y_data.as_slice().unwrap();
                let ps = pred.as_slice().unwrap();
                d0s.par_chunks_mut(g)
                    .zip(d1s.par_chunks_mut(g))
                    .zip(d2s.par_chunks_mut(g))
                    .zip(ys.par_chunks(g))
                    .zip(ps.par_chunks(g))
                    .for_each(|((((d0c, d1c), d2c), yc), pc)| {
                        calc_q_chunk(
                            yc,
                            pc,
                            d0c,
                            d1c,
                            d2c,
                            &q_flat,
                            &sq_flat,
                            &x_vals_vec,
                            nk,
                            nx,
                            x_max,
                        );
                    });
            }
        }

        let mut grad = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for t in 0..k {
                let mut acc = 0.0f64;
                for gg in 0..g {
                    acc += d1[[i, gg]] * s[[i, gg, t]];
                }
                grad[[i, t]] = -acc;
            }
        }

        let kk = k * k;
        let mut delta_w = vec![0.0f64; n * k];
        delta_w.par_chunks_mut(k).enumerate().for_each(|(i, dw)| {
            let mut hess = vec![0.0f64; kk];
            for t in 0..k {
                for u in 0..k {
                    let mut h = 0.0f64;
                    for gg in 0..g {
                        h += s[[i, gg, t]] * (-d2[[i, gg]]) * s[[i, gg, u]];
                    }
                    hess[t * k + u] = h;
                }
            }
            let gi = grad.row(i);
            let si = solution.row(i);
            psd_normalize_qp(
                &hess,
                gi.as_slice().unwrap(),
                si.as_slice().unwrap(),
                dw,
                k,
                1e-3,
            );
        });

        let sol_sl = solution.as_slice().unwrap();
        for i in 0..n {
            if converged[i] {
                continue;
            }
            let mut w_new = vec![0.0f64; k];
            for t in 0..k {
                w_new[t] = sol_sl[i * k + t] + step_size * delta_w[i * k + t];
            }
            if constrain {
                simplex_project_slice(&mut w_new);
            }
            let mut change = 0.0f64;
            for t in 0..k {
                change += (w_new[t] - w[[i, t]]).abs();
                w[[i, t]] = w_new[t];
            }
            if change <= min_change {
                converged[i] = true;
            }
        }
    }

    (w, converged)
}

/// Per-row negative log-likelihood sums (same definition as `calc_log_likelihood_batch` / PyTorch).
pub fn calc_neg_loglik_row_sums(
    y: &Array2<f64>,
    lam: &Array2<f64>,
    q_mat: &Array2<f64>,
    sq_mat: &Array2<f64>,
    x_vals: &Array1<f64>,
    k_val: i64,
) -> Vec<f64> {
    let n = y.nrows();
    let g = y.ncols();
    if n == 0 {
        return Vec::new();
    }
    let nk = q_mat.nrows();
    let nx = q_mat.ncols();
    let k_eff = if k_val < 0 { nk as i64 - 3 } else { k_val };
    let yc = y.mapv(|v| v.clamp(0.0, k_eff as f64));
    let q_flat: Vec<f64> = q_mat.as_standard_layout().iter().copied().collect();
    let sq_flat: Vec<f64> = sq_mat.as_standard_layout().iter().copied().collect();
    let x_vals_vec: Vec<f64> = x_vals.iter().copied().collect();
    let x_max = x_vals_vec[nx - 1];

    let mut d0 = Array2::<f64>::zeros((n, g));
    let mut d1 = Array2::<f64>::zeros((n, g));
    let mut d2 = Array2::<f64>::zeros((n, g));
    {
        let d0s = d0.as_slice_mut().unwrap();
        let d1s = d1.as_slice_mut().unwrap();
        let d2s = d2.as_slice_mut().unwrap();
        let ys = yc.as_slice().unwrap();
        let ls = lam.as_slice().unwrap();
        d0s.par_chunks_mut(g)
            .zip(d1s.par_chunks_mut(g))
            .zip(d2s.par_chunks_mut(g))
            .zip(ys.par_chunks(g))
            .zip(ls.par_chunks(g))
            .for_each(|((((d0c, d1c), d2c), y_row), l_row)| {
                calc_q_chunk(
                    y_row,
                    l_row,
                    d0c,
                    d1c,
                    d2c,
                    &q_flat,
                    &sq_flat,
                    &x_vals_vec,
                    nk,
                    nx,
                    x_max,
                );
            });
    }

    let d0s = d0.as_slice().unwrap();
    (0..n)
        .map(|i| -d0s[i * g..(i + 1) * g].iter().sum::<f64>())
        .collect()
}
