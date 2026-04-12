use burn::tensor::{Int, Tensor};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::backend::{
    fe, scalar_to_f64, slice_elems_to_f64, tensor1_from_view, tensor2_from_f64, tensor2_from_view,
    tensor2_from_view_clamped_y, tensor3_from_f64, FloatElem, RctdBackend, RctdDevice,
};
use crate::calc_q::calc_q_all;
use crate::linalg::{psd_batch, spectral_norm_upper};
use crate::qp::solve_box_qp_batch;
use crate::simplex::project_simplex_batch;

fn tensor2(a: &Array2<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 2> {
    tensor2_from_f64(a, dev)
}

fn tensor1(a: &Array1<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 1> {
    crate::backend::tensor1_from_f64(a, dev)
}

fn select_rows_f2(
    t: Tensor<RctdBackend, 2>,
    idx: &[usize],
    dev: &RctdDevice,
) -> Tensor<RctdBackend, 2> {
    let v: Vec<i32> = idx.iter().map(|&i| i as i32).collect();
    let ind = Tensor::<RctdBackend, 1, Int>::from_ints(v.as_slice(), dev);
    t.select(0, ind)
}

fn select_rows_f1(
    t: Tensor<RctdBackend, 1>,
    idx: &[usize],
    dev: &RctdDevice,
) -> Tensor<RctdBackend, 1> {
    let v: Vec<i32> = idx.iter().map(|&i| i as i32).collect();
    let ind = Tensor::<RctdBackend, 1, Int>::from_ints(v.as_slice(), dev);
    t.select(0, ind)
}

fn p_outer_from_profiles(p: &Array2<f64>) -> Array2<f64> {
    let g = p.nrows();
    let k = p.ncols();
    let mut po = Array2::zeros((g, k * k));
    for gi in 0..g {
        for a in 0..k {
            for b in 0..k {
                po[[gi, a * k + b]] = p[[gi, a]] * p[[gi, b]];
            }
        }
    }
    po
}

pub struct IrwlsSharedPrepared {
    p_gpu: Tensor<RctdBackend, 2>,
    p_t: Tensor<RctdBackend, 2>,
    q_gpu: Tensor<RctdBackend, 2>,
    sq_gpu: Tensor<RctdBackend, 2>,
    x_gpu: Tensor<RctdBackend, 1>,
    p_outer: Tensor<RctdBackend, 2>,
    eye: Tensor<RctdBackend, 2>,
    k_val_clamp: f64,
    k: usize,
}

impl IrwlsSharedPrepared {
    pub fn new(
        p: &Array2<f64>,
        q_mat: &Array2<f64>,
        sq_mat: &Array2<f64>,
        x_vals: &Array1<f64>,
        device: &RctdDevice,
    ) -> Self {
        let k = p.ncols();
        let p_gpu = tensor2(p, device);
        let p_t = p_gpu.clone().transpose();
        let q_gpu = tensor2(q_mat, device);
        let sq_gpu = tensor2(sq_mat, device);
        let x_gpu = tensor1(x_vals, device);
        let p_outer = tensor2(&p_outer_from_profiles(p), device);
        let eye = Tensor::<RctdBackend, 2>::eye(k, device);
        let k_val_clamp = (q_mat.nrows() as i64 - 3) as f64;
        Self {
            p_gpu,
            p_t,
            q_gpu,
            sq_gpu,
            x_gpu,
            p_outer,
            eye,
            k_val_clamp,
            k,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn solve_irwls_batch_shared_prepared(
    prep: &IrwlsSharedPrepared,
    y: ArrayView2<f64>,
    numi: ArrayView1<f64>,
    max_iter: usize,
    min_change: f64,
    step_size: f64,
    constrain: bool,
    bulk_mode: bool,
    device: &RctdDevice,
) -> (Array2<f64>, Array1<bool>) {
    let n = y.nrows();
    let g = y.ncols();
    let k = prep.k;
    if n == 0 {
        return (Array2::zeros((0, k)), Array1::default(0));
    }

    let p_gpu = prep.p_gpu.clone();
    let p_t = prep.p_t.clone();
    let q_gpu = prep.q_gpu.clone();
    let sq_gpu = prep.sq_gpu.clone();
    let x_gpu = prep.x_gpu.clone();
    let p_outer = prep.p_outer.clone();
    let eye = prep.eye.clone();

    let y_data = if bulk_mode {
        tensor2_from_view(y, device)
    } else {
        tensor2_from_view_clamped_y(y, prep.k_val_clamp, device)
    };
    let numi_gpu = tensor1_from_view(numi, device);

    let mut w = Array2::from_elem((n, k), 1.0 / k as f64);
    let mut converged = Array1::from_elem(n, false);

    let threshold = numi.mapv(|u| (u * 1e-7).max(1e-4));
    let thresh_gpu_full = tensor1(&threshold, device);

    let mut active: Vec<usize> = (0..n).collect();
    let mut w_act = tensor2(&w, device);
    let mut y_act = y_data.clone();
    let mut numi_act = numi_gpu.clone();
    let mut thresh_act = thresh_gpu_full.clone();

    for _it in 0..max_iter {
        let n_act = active.len();
        if n_act == 0 {
            break;
        }

        let solution = w_act.clone().clamp_min(fe(0.0));
        let pred = solution
            .clone()
            .matmul(p_t.clone())
            .mul(numi_act.clone().reshape([n_act, 1]))
            .abs();
        let pred = pred.max_pair(thresh_act.clone().reshape([n_act, 1]));

        let (grad, hess_t) = if bulk_mode {
            let d1 =
                (pred.clone().log() - (y_act.clone() + fe(1e-10)).log()) / pred.clone() * fe(-2.0);
            let d2 = (pred.ones_like() - pred.clone().log() + (y_act.clone() + fe(1e-10)).log())
                / pred.clone().powf_scalar(2.0)
                * fe(-2.0);
            let grad = d1
                .clone()
                .mul(numi_act.clone().reshape([n_act, 1]))
                .matmul(p_gpu.clone())
                * fe(-1.0);
            let d2_w = (d2 * fe(-1.0)).mul(numi_act.clone().reshape([n_act, 1]).powf_scalar(2.0));
            let hess = d2_w.matmul(p_outer.clone()).reshape([n_act, k, k]);
            (grad, hess)
        } else {
            let yf = y_act.clone().reshape([n_act * g]);
            let pf = pred.reshape([n_act * g]);
            let (_d0, d1, d2) =
                calc_q_all(yf, pf, q_gpu.clone(), sq_gpu.clone(), x_gpu.clone(), -1);
            let d1_vec = d1.reshape([n_act, g]);
            let d2_vec = d2.reshape([n_act, g]);
            let grad = d1_vec
                .mul(numi_act.clone().reshape([n_act, 1]))
                .matmul(p_gpu.clone())
                * fe(-1.0);
            let d2_w = d2_vec * fe(-1.0) * numi_act.clone().reshape([n_act, 1]).powf_scalar(2.0);
            let hess = d2_w.matmul(p_outer.clone()).reshape([n_act, k, k]);
            (grad, hess)
        };

        let (hess_psd, max_eig) = psd_batch(hess_t, 1e-3, device);
        let norm_factor = spectral_norm_upper(hess_psd.clone(), max_eig, device);
        let d_mat = hess_psd / norm_factor.clone().reshape([n_act, 1, 1])
            + eye.clone().unsqueeze_dim::<3>(0).mul_scalar(fe(1e-7));
        let d_vec = grad * fe(-1.0) / norm_factor.reshape([n_act, 1]);

        let delta_w = solve_box_qp_batch(d_mat, d_vec, solution.clone() * fe(-1.0), 50, device);
        let mut w_new = solution + delta_w * fe(step_size);
        if constrain {
            w_new = project_simplex_batch(w_new, device);
        }

        let change_sl = slice_elems_to_f64(
            (w_new.clone() - w_act.clone())
                .abs()
                .sum_dim(1)
                .reshape([n_act])
                .into_data()
                .as_slice::<FloatElem>()
                .unwrap(),
        );
        let mut newly = vec![false; n_act];
        for i in 0..n_act {
            newly[i] = change_sl[i] <= min_change;
        }

        let w_new_data = slice_elems_to_f64(w_new.into_data().as_slice::<FloatElem>().unwrap());
        for (j, &ai) in active.iter().enumerate() {
            for t in 0..k {
                w[[ai, t]] = w_new_data[j * k + t];
            }
            if newly[j] {
                converged[ai] = true;
            }
        }

        let mut next = Vec::new();
        for (j, &ai) in active.iter().enumerate() {
            if !newly[j] {
                next.push(ai);
            }
        }
        if next.is_empty() {
            break;
        }
        active = next;
        w_act = tensor2(
            &Array2::from_shape_fn((active.len(), k), |(i, t)| w[[active[i], t]]),
            device,
        );
        y_act = select_rows_f2(y_data.clone(), &active, device);
        numi_act = select_rows_f1(numi_gpu.clone(), &active, device);
        thresh_act = select_rows_f1(thresh_gpu_full.clone(), &active, device);
    }

    (w, converged)
}

#[allow(clippy::too_many_arguments)]
pub fn solve_irwls_batch_shared(
    p: &Array2<f64>,
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
    device: &RctdDevice,
) -> (Array2<f64>, Array1<bool>) {
    let prep = IrwlsSharedPrepared::new(p, q_mat, sq_mat, x_vals, device);
    solve_irwls_batch_shared_prepared(
        &prep,
        y.view(),
        numi.view(),
        max_iter,
        min_change,
        step_size,
        constrain,
        bulk_mode,
        device,
    )
}

fn tensor3(a: &ndarray::Array3<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 3> {
    tensor3_from_f64(a, dev)
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn solve_irwls_batch(
    s: &ndarray::Array3<f64>,
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
    device: &RctdDevice,
) -> (Array2<f64>, Array1<bool>) {
    let n = y.nrows();
    let g = y.ncols();
    let k = s.dim().2;
    if n == 0 {
        return (Array2::zeros((0, k)), Array1::default(0));
    }
    let s_gpu = tensor3(s, device);
    let q_gpu = tensor2(q_mat, device);
    let sq_gpu = tensor2(sq_mat, device);
    let x_gpu = tensor1(x_vals, device);
    let eye = Tensor::<RctdBackend, 2>::eye(k, device);
    let k_val = q_mat.nrows() as i64 - 3;
    let y_data = if bulk_mode {
        tensor2(y, device)
    } else {
        let ym = y.mapv(|v| v.min(k_val as f64));
        tensor2(&ym, device)
    };
    let mut w = Array2::from_elem((n, k), 1.0 / k as f64);
    let mut converged = Array1::from_elem(n, false);
    let threshold = numi.mapv(|u| (u * 1e-7).max(1e-4));
    let thresh_full = tensor1(&threshold, device);
    for _it in 0..max_iter {
        if converged.iter().all(|&c| c) {
            break;
        }
        let w_t = tensor2(&w, device);
        let solution = w_t.clone().clamp_min(fe(0.0));
        let pred = s_gpu
            .clone()
            .matmul(solution.clone().reshape([n, k, 1]))
            .reshape([n, g])
            .abs();
        let pred = pred.max_pair(thresh_full.clone().reshape([n, 1]));
        let (grad, hess_t) = if bulk_mode {
            let d1 =
                (pred.clone().log() - (y_data.clone() + fe(1e-10)).log()) / pred.clone() * fe(-2.0);
            let d2 = (pred.ones_like() - pred.clone().log() + (y_data.clone() + fe(1e-10)).log())
                / pred.clone().powf_scalar(2.0)
                * fe(-2.0);
            let grad = -(d1.clone().reshape([n, 1, g]).matmul(s_gpu.clone())).reshape([n, k]);
            let sw = s_gpu.clone() * (d2 * fe(-1.0)).reshape([n, g, 1]);
            let hess = sw.swap_dims(1, 2).matmul(s_gpu.clone());
            (grad, hess)
        } else {
            let yf = y_data.clone().reshape([n * g]);
            let pf = pred.reshape([n * g]);
            let (_d0, d1f, d2f) =
                calc_q_all(yf, pf, q_gpu.clone(), sq_gpu.clone(), x_gpu.clone(), -1);
            let d1_vec = d1f.reshape([n, g]);
            let d2_vec = d2f.reshape([n, g]);
            let grad = -(d1_vec.clone().reshape([n, 1, g]).matmul(s_gpu.clone())).reshape([n, k]);
            let sw = s_gpu.clone() * (d2_vec * fe(-1.0)).reshape([n, g, 1]);
            let hess = sw.swap_dims(1, 2).matmul(s_gpu.clone());
            (grad, hess)
        };
        let (hess_psd, max_eig) = psd_batch(hess_t, 1e-3, device);
        let norm_factor = spectral_norm_upper(hess_psd.clone(), max_eig, device);
        let d_mat = hess_psd / norm_factor.clone().reshape([n, 1, 1])
            + eye.clone().unsqueeze_dim::<3>(0).mul_scalar(fe(1e-7));
        let d_vec = grad * fe(-1.0) / norm_factor.reshape([n, 1]);
        let delta_w = solve_box_qp_batch(d_mat, d_vec, solution.clone() * fe(-1.0), 50, device);
        let mut w_new_t = solution + delta_w * fe(step_size);
        if constrain {
            w_new_t = project_simplex_batch(w_new_t, device);
        }
        let change = slice_elems_to_f64(
            (w_new_t.clone() - w_t.clone())
                .abs()
                .sum_dim(1)
                .reshape([n])
                .into_data()
                .as_slice::<FloatElem>()
                .unwrap(),
        );
        let w_new_sl = slice_elems_to_f64(w_new_t.into_data().as_slice::<FloatElem>().unwrap());
        for i in 0..n {
            if converged[i] {
                continue;
            }
            for t in 0..k {
                w[[i, t]] = w_new_sl[i * k + t];
            }
            if change[i] <= min_change {
                converged[i] = true;
            }
        }
    }
    (w, converged)
}

pub fn solve_irwls_single_bulk(
    s: &Tensor<RctdBackend, 2>,
    y: &Tensor<RctdBackend, 1>,
    numi: f64,
    device: &RctdDevice,
) -> Tensor<RctdBackend, 1> {
    let g = s.dims()[0];
    let k = s.dims()[1];
    let mut w = Tensor::<RctdBackend, 1>::ones([k], device) / fe(k as f64);
    let eye = Tensor::<RctdBackend, 2>::eye(k, device);
    let mut change = 1.0f64;
    for _ in 0..100 {
        if change <= 0.001 {
            break;
        }
        let solution = w.clone().clamp_min(fe(0.0));
        let prediction = s
            .clone()
            .matmul(solution.clone().reshape([k, 1]))
            .reshape([g])
            .abs();
        let thr = (1e-4f64).max(numi * 1e-7);
        let prediction = prediction.clamp_min(fe(thr));
        let d1 = (prediction.clone().log() - (y.clone() + fe(1e-10)).log()) / prediction.clone()
            * fe(-2.0);
        let d2 = (prediction.ones_like() - prediction.clone().log()
            + (y.clone() + fe(1e-10)).log())
            / prediction.clone().powf_scalar(2.0)
            * fe(-2.0);
        let grad = -(d1.clone().reshape([1, g]).matmul(s.clone())).reshape([k]);
        let neg_d2 = (d2 * fe(-1.0)).reshape([g, 1]);
        let weighted = s.clone() * neg_d2;
        let hess = s.clone().transpose().matmul(weighted);
        let (hess_psd, max_eig) = psd_batch(hess.unsqueeze_dim::<3>(0), 1e-3, device);
        let h2 = hess_psd.reshape([k, k]);
        let nf0 = scalar_to_f64(max_eig.into_scalar());
        let d_mat = h2 / fe(nf0) + eye.clone().mul_scalar(fe(1e-7));
        let d_vec = grad * fe(-1.0) / fe(nf0);
        let delta_w = solve_box_qp_batch(
            d_mat.unsqueeze_dim::<3>(0),
            d_vec.reshape([1, k]),
            solution.clone().reshape([1, k]) * fe(-1.0),
            50,
            device,
        )
        .reshape([k]);
        let w_new = solution + delta_w * fe(0.3);
        change = slice_elems_to_f64(
            (w_new.clone() - w.clone())
                .abs()
                .into_data()
                .as_slice::<FloatElem>()
                .unwrap(),
        )
        .iter()
        .sum();
        w = w_new;
    }
    w
}
