//! PSD projection (eigen clamp). Burn 0.17 has no `eigh` on tensors; use nalgebra on CPU data.

use burn::tensor::{Tensor, TensorData};
use nalgebra::linalg::SymmetricEigen;
use nalgebra::DMatrix;

use crate::backend::{
    f64_slice_to_elems, fe, slice_elems_to_f64, FloatElem, RctdBackend, RctdDevice,
};

pub fn psd_2x2(
    h: Tensor<RctdBackend, 3>,
    epsilon: f64,
    device: &RctdDevice,
) -> Tensor<RctdBackend, 3> {
    let [n, _, _] = h.dims();
    let data = h.into_data();
    let s = slice_elems_to_f64(data.as_slice::<FloatElem>().unwrap());
    let mut out = vec![0f64; n * 4];
    for i in 0..n {
        let base = i * 4;
        let a = s[base];
        let b = s[base + 1];
        let d = s[base + 3];
        let half_trace = (a + d) * 0.5;
        let disc = (((a - d) * 0.5).powi(2) + b * b).sqrt();
        let lam1 = (half_trace - disc).max(epsilon);
        let lam2 = (half_trace + disc).max(epsilon);
        let safe_disc = disc.max(1e-30);
        let cos2t = (a - d) * 0.5 / safe_disc;
        let sin2t = b / safe_disc;
        let half_lam_sum = (lam1 + lam2) * 0.5;
        let half_lam_diff = (lam2 - lam1) * 0.5;
        out[base] = half_lam_sum + half_lam_diff * cos2t;
        out[base + 1] = half_lam_diff * sin2t;
        out[base + 2] = out[base + 1];
        out[base + 3] = half_lam_sum - half_lam_diff * cos2t;
    }
    Tensor::from_data(TensorData::new(f64_slice_to_elems(&out), [n, 2, 2]), device)
}

pub fn psd_batch(
    h: Tensor<RctdBackend, 3>,
    epsilon: f64,
    device: &RctdDevice,
) -> (Tensor<RctdBackend, 3>, Tensor<RctdBackend, 1>) {
    let [n, k, k2] = h.dims();
    assert_eq!(k, k2);
    if k == 1 {
        let data = h.into_data();
        let s = slice_elems_to_f64(data.as_slice::<FloatElem>().unwrap());
        let mut clamped = s;
        for x in &mut clamped {
            *x = x.max(epsilon);
        }
        let max_eig = clamped.clone();
        return (
            Tensor::from_data(
                TensorData::new(f64_slice_to_elems(&clamped), [n, 1, 1]),
                device,
            ),
            Tensor::from_data(TensorData::new(f64_slice_to_elems(&max_eig), [n]), device),
        );
    }
    if k == 2 {
        let h_psd = psd_2x2(h, epsilon, device);
        let data = h_psd.clone().into_data();
        let s = slice_elems_to_f64(data.as_slice::<FloatElem>().unwrap());
        let mut max_e = vec![0f64; n];
        for i in 0..n {
            let a = s[i * 4];
            let b = s[i * 4 + 1];
            let d = s[i * 4 + 3];
            let half_trace = (a + d) * 0.5;
            let disc = (((a - d) * 0.5).powi(2) + b * b).sqrt();
            max_e[i] = (half_trace + disc).max(epsilon);
        }
        return (
            h_psd,
            Tensor::from_data(TensorData::new(f64_slice_to_elems(&max_e), [n]), device),
        );
    }

    let data = h.into_data();
    let s = slice_elems_to_f64(data.as_slice::<FloatElem>().unwrap());
    let mut out = vec![0f64; n * k * k];
    let mut max_eig = vec![0f64; n];
    for i in 0..n {
        let mut mat: DMatrix<f64> = DMatrix::from_row_slice(k, k, &s[i * k * k..(i + 1) * k * k]);
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
        let v = se.eigenvectors.clone();
        let h_psd = &v * DMatrix::from_diagonal(&evals) * v.transpose();
        max_eig[i] = *evals
            .iter()
            .max_by(|a: &&f64, b| a.partial_cmp(b).unwrap())
            .unwrap();
        for r in 0..k {
            for c in 0..k {
                out[i * k * k + r * k + c] = h_psd[(r, c)];
            }
        }
    }
    (
        Tensor::from_data(TensorData::new(f64_slice_to_elems(&out), [n, k, k]), device),
        Tensor::from_data(TensorData::new(f64_slice_to_elems(&max_eig), [n]), device),
    )
}

pub fn spectral_norm_upper(
    h_psd: Tensor<RctdBackend, 3>,
    max_eig: Tensor<RctdBackend, 1>,
    _device: &RctdDevice,
) -> Tensor<RctdBackend, 1> {
    let _ = h_psd;
    max_eig.clamp_min(fe(1e-10f64))
}
