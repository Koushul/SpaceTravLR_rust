use burn::tensor::{Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use ndarray::Array2;

type B = NdArray<f64>;

#[test]
fn burn_matmul_matches_ndarray_f64() {
    let dev = NdArrayDevice::Cpu;
    let a = Array2::from_shape_fn((5, 4), |(i, j)| (i * 10 + j) as f64 + 0.25);
    let b = Array2::from_shape_fn((4, 3), |(i, j)| (i + 1) as f64 * 0.5 + j as f64);
    let expect = a.dot(&b);

    let ta = Tensor::<B, 2>::from_data(TensorData::new(a.iter().cloned().collect(), [5, 4]), &dev);
    let tb = Tensor::<B, 2>::from_data(TensorData::new(b.iter().cloned().collect(), [4, 3]), &dev);
    let tc = ta.matmul(tb);
    let got: Vec<f64> = tc.into_data().as_slice::<f64>().unwrap().to_vec();
    for (i, (&e, g)) in expect.iter().zip(got.iter()).enumerate() {
        assert!((e - g).abs() < 1e-12, "idx {i}: expected {e}, got {g}");
    }
}
