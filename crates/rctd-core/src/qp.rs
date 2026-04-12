use burn::tensor::{Tensor, TensorData};

use crate::backend::{f64_slice_to_elems, slice_elems_to_f64, FloatElem, RctdBackend, RctdDevice};

pub fn solve_box_qp_2(
    d: Tensor<RctdBackend, 3>,
    di: Tensor<RctdBackend, 2>,
    lb: Tensor<RctdBackend, 2>,
    device: &RctdDevice,
) -> Tensor<RctdBackend, 2> {
    let [n, _, _] = d.dims();
    let ds = slice_elems_to_f64(d.into_data().as_slice::<FloatElem>().unwrap());
    let dis = slice_elems_to_f64(di.into_data().as_slice::<FloatElem>().unwrap());
    let lbs = slice_elems_to_f64(lb.into_data().as_slice::<FloatElem>().unwrap());
    let mut out = vec![0f64; n * 2];
    for i in 0..n {
        let d00 = ds[i * 4];
        let d01 = ds[i * 4 + 1];
        let d11 = ds[i * 4 + 3];
        let det = d00 * d11 - d01 * d01;
        let det_sign = det.signum();
        let det_abs = det.abs().max(1e-30) * det_sign;
        let mut x1 = (d00 * dis[i * 2 + 1] - d01 * dis[i * 2]) / det_abs;
        x1 = x1.max(lbs[i * 2 + 1]);
        let x0 = ((dis[i * 2] - d01 * x1) / d00).max(lbs[i * 2]);
        let x1 = ((dis[i * 2 + 1] - d01 * x0) / d11).max(lbs[i * 2 + 1]);
        out[i * 2] = x0;
        out[i * 2 + 1] = x1;
    }
    Tensor::from_data(TensorData::new(f64_slice_to_elems(&out), [n, 2]), device)
}

pub fn solve_box_qp_batch(
    d: Tensor<RctdBackend, 3>,
    di: Tensor<RctdBackend, 2>,
    lb: Tensor<RctdBackend, 2>,
    n_sweeps: usize,
    device: &RctdDevice,
) -> Tensor<RctdBackend, 2> {
    let [n, k, k2] = d.dims();
    assert_eq!(k, k2);
    if k == 2 {
        return solve_box_qp_2(d, di, lb, device);
    }
    let ds = slice_elems_to_f64(d.into_data().as_slice::<FloatElem>().unwrap());
    let dis = slice_elems_to_f64(di.into_data().as_slice::<FloatElem>().unwrap());
    let lbs = slice_elems_to_f64(lb.into_data().as_slice::<FloatElem>().unwrap());
    let mut x = vec![0f64; n * k];
    for i in 0..n {
        for j in 0..k {
            let djj = ds[i * k * k + j * k + j];
            x[i * k + j] = (dis[i * k + j] / djj).max(lbs[i * k + j]);
        }
    }
    for _ in 0..n_sweeps {
        for j in 0..k {
            for i in 0..n {
                let djj = ds[i * k * k + j * k + j];
                let mut dot = 0.0;
                for t in 0..k {
                    dot += ds[i * k * k + j * k + t] * x[i * k + t];
                }
                let residual = dis[i * k + j] - dot + djj * x[i * k + j];
                let xi = residual / djj;
                x[i * k + j] = xi.max(lbs[i * k + j]);
            }
        }
    }
    Tensor::from_data(TensorData::new(f64_slice_to_elems(&x), [n, k]), device)
}
