//! Simplex projection (Duchi et al. 2008), batched via ndarray (matches `_simplex.py`).

use burn::tensor::Tensor;

use crate::backend::{f64_slice_to_elems, slice_elems_to_f64, FloatElem, RctdBackend, RctdDevice};

pub fn project_simplex_batch(
    v: Tensor<RctdBackend, 2>,
    device: &RctdDevice,
) -> Tensor<RctdBackend, 2> {
    let [n, k] = v.dims();
    let data = v.into_data();
    let sl = slice_elems_to_f64(data.as_slice::<FloatElem>().expect("float tensor"));
    let mut out = vec![0f64; n * k];
    for i in 0..n {
        let row: Vec<f64> = (0..k).map(|j| sl[i * k + j]).collect();
        let p = project_simplex_slice(&row);
        for j in 0..k {
            out[i * k + j] = p[j];
        }
    }
    Tensor::<RctdBackend, 2>::from_data(
        burn::tensor::TensorData::new(f64_slice_to_elems(&out), [n, k]),
        device,
    )
}

fn project_simplex_slice(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut cssv = vec![0.0; n];
    for i in 0..n {
        cssv[i] = if i == 0 { u[0] } else { cssv[i - 1] + u[i] };
    }
    let mut rho = 0usize;
    for i in 0..n {
        let ind = (i + 1) as f64;
        if u[i] * ind > (cssv[i] - 1.0) {
            rho = i + 1;
        }
    }
    let theta = if rho == 0 {
        0.0
    } else {
        (cssv[rho - 1] - 1.0) / rho as f64
    };
    v.iter().map(|&x| (x - theta).max(0.0)).collect()
}
