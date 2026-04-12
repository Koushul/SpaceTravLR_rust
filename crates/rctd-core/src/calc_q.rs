//! Cubic spline `calc_q_all` on Burn tensors (eager, matches `_calc_q_all_impl`).

use burn::tensor::Tensor;

use crate::backend::{fe, scalar_to_f64, tensor1_from_f64, FloatElem, RctdDevice};

pub use crate::backend::RctdBackend;

pub type F1 = Tensor<RctdBackend, 1>;
pub type F2 = Tensor<RctdBackend, 2>;

/// Returns (d0, d1, d2) log-likelihood derivatives per gene (same length as `y` / `lam`).
pub fn calc_q_all(y: F1, lam: F1, q_mat: F2, sq_mat: F2, x_vals: F1, k_val: i64) -> (F1, F1, F1) {
    let k_eff = if k_val < 0 {
        q_mat.shape().dims[0] as i64 - 3
    } else {
        k_val
    };
    let y = y.clamp(fe(0.0), fe(k_eff as f64));
    let nx = x_vals.dims()[0];
    let x_data = x_vals.clone().into_data();
    let x_sl = x_data.as_slice::<FloatElem>().unwrap();
    let x_max_s: f64 = scalar_to_f64(x_sl[nx - 1]);
    let eps = 1e-4f64;
    let lam = lam.clamp(fe(eps), fe(x_max_s - eps));

    let delta = 1e-6f64;
    let l = (lam.clone() / fe(delta)).sqrt().floor().int();
    let l_f = l.clone().float();
    let m = (l.clone() - 9).clamp_max(40)
        + (((l_f - fe(48.7499)).clamp_min(fe(0.0)) * fe(4.0))
            .sqrt()
            .ceil()
            .int()
            - 2)
        .clamp_min(0);
    let m_col = m.clone().clamp(0, (nx as i64) - 1);
    let m1 = (m.clone() - 1).clamp(0, (nx as i64) - 1);

    let ti1 = x_vals.clone().select(0, m1.clone());
    let ti = x_vals.clone().select(0, m_col.clone());
    let hi = ti.clone() - ti1.clone();

    let nk = q_mat.shape().dims[0];
    let y_idx = y.int().clamp(0, (nk as i64) - 1);
    let q_flat = q_mat.clone().reshape([nk * nx]);
    let sq_flat = sq_mat.clone().reshape([nk * nx]);

    let lin1 = (y_idx.clone() * (nx as i64) + m1).clamp(0, (nk * nx) as i64 - 1);
    let lin2 = (y_idx * (nx as i64) + m_col).clamp(0, (nk * nx) as i64 - 1);

    let fti1 = q_flat.clone().gather(0, lin1.clone());
    let fti = q_flat.gather(0, lin2.clone());
    let zi1 = sq_flat.clone().gather(0, lin1.clone());
    let zi = sq_flat.gather(0, lin2.clone());

    let diff1 = lam.clone() - ti1.clone();
    let diff2 = ti.clone() - lam.clone();
    let diff3 = fti.clone() / hi.clone() - zi.clone() * hi.clone() / fe(6.0);
    let diff4 = fti1.clone() / hi.clone() - zi1.clone() * hi.clone() / fe(6.0);
    let zdi = zi.clone() / hi.clone();
    let zdi1 = zi1 / hi.clone();

    let d0_vec = zdi.clone() * diff1.clone().powf_scalar(3.0) / fe(6.0)
        + zdi1.clone() * diff2.clone().powf_scalar(3.0) / fe(6.0)
        + diff3.clone() * diff1.clone()
        + diff4.clone() * diff2.clone();
    let d1_vec = zdi.clone() * diff1.clone().powf_scalar(2.0) / fe(2.0)
        - zdi1.clone() * diff2.clone().powf_scalar(2.0) / fe(2.0)
        + diff3
        - diff4;
    let d2_vec = zdi * diff1 + zdi1 * diff2;

    (d0_vec, d1_vec, d2_vec)
}

pub fn calc_log_likelihood_batch(
    y: Tensor<RctdBackend, 2>,
    lam: Tensor<RctdBackend, 2>,
    q_mat: F2,
    sq_mat: F2,
    x_vals: F1,
    k_val: i64,
) -> Tensor<RctdBackend, 1> {
    let [n, g] = y.dims();
    let yf = y.reshape([n * g]);
    let lf = lam.reshape([n * g]);
    let (d0, _, _) = calc_q_all(yf, lf, q_mat, sq_mat, x_vals, k_val);
    let d0 = d0.reshape([n, g]);
    (-d0.sum_dim(1)).reshape([n])
}

pub fn device_cpu() -> RctdDevice {
    #[cfg(not(feature = "wgpu"))]
    {
        use burn_ndarray::NdArrayDevice;
        NdArrayDevice::Cpu
    }
    #[cfg(feature = "wgpu")]
    {
        use burn_wgpu::WgpuDevice;
        WgpuDevice::default()
    }
}

pub fn x_vals_tensor(x_vals: &ndarray::Array1<f64>, dev: &RctdDevice) -> F1 {
    tensor1_from_f64(x_vals, dev)
}
