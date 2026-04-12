//! Burn backend: CPU (`NdArray<f64>`) by default, optional GPU via `wgpu` (Metal on Apple Silicon,
//! Vulkan/DX12 elsewhere — the same WebGPU/wgpu stack; native macOS uses Metal, not browser WebGPU).

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[cfg(not(feature = "wgpu"))]
pub use burn_ndarray::{NdArray, NdArrayDevice};

#[cfg(not(feature = "wgpu"))]
pub type RctdBackend = NdArray<f64>;
#[cfg(not(feature = "wgpu"))]
pub type RctdDevice = NdArrayDevice;

#[cfg(feature = "wgpu")]
pub use burn_wgpu::{Wgpu, WgpuDevice};

#[cfg(feature = "wgpu")]
pub type RctdBackend = Wgpu<f32, i32, u32>;
#[cfg(feature = "wgpu")]
pub type RctdDevice = WgpuDevice;

pub type FloatElem = <RctdBackend as Backend>::FloatElem;

#[inline]
pub fn fe(x: f64) -> FloatElem {
    FloatElem::from_elem(x)
}

#[inline]
pub fn scalar_to_f64<E: ElementConversion>(x: E) -> f64 {
    x.elem::<f64>()
}

pub fn tensor1_from_f64(a: &Array1<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 1> {
    tensor1_from_view(a.view(), dev)
}

pub fn tensor1_from_view(a: ArrayView1<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 1> {
    let n = a.len();
    let v: Vec<FloatElem> = a.iter().map(|&x| FloatElem::from_elem(x)).collect();
    Tensor::from_data(TensorData::new(v, [n]), dev)
}

pub fn tensor2_from_f64(a: &Array2<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 2> {
    tensor2_from_view(a.view(), dev)
}

pub fn tensor2_from_view(a: ArrayView2<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 2> {
    let (r, c) = a.dim();
    let v: Vec<FloatElem> = a.iter().map(|&x| FloatElem::from_elem(x)).collect();
    Tensor::from_data(TensorData::new(v, [r, c]), dev)
}

pub fn tensor2_from_view_clamped_y(
    y: ArrayView2<f64>,
    k_val: f64,
    dev: &RctdDevice,
) -> Tensor<RctdBackend, 2> {
    let (r, c) = y.dim();
    let v: Vec<FloatElem> = y
        .iter()
        .map(|&x| FloatElem::from_elem(x.min(k_val)))
        .collect();
    Tensor::from_data(TensorData::new(v, [r, c]), dev)
}

pub fn tensor3_from_f64(a: &ndarray::Array3<f64>, dev: &RctdDevice) -> Tensor<RctdBackend, 3> {
    let (n, g, k) = a.dim();
    let v: Vec<FloatElem> = a.iter().map(|&x| FloatElem::from_elem(x)).collect();
    Tensor::from_data(TensorData::new(v, [n, g, k]), dev)
}

pub fn slice_elems_to_f64(s: &[FloatElem]) -> Vec<f64> {
    s.iter().copied().map(scalar_to_f64).collect()
}

pub fn f64_slice_to_elems(v: &[f64]) -> Vec<FloatElem> {
    v.iter().map(|&x| FloatElem::from_elem(x)).collect()
}

#[cfg(not(feature = "wgpu"))]
pub fn default_device() -> RctdDevice {
    NdArrayDevice::Cpu
}

#[cfg(feature = "wgpu")]
pub fn default_device() -> RctdDevice {
    WgpuDevice::default()
}

#[cfg(feature = "wgpu")]
pub fn init_wgpu(device: &WgpuDevice) {
    use burn_wgpu::graphics::AutoGraphicsApi;
    burn_wgpu::init_setup::<AutoGraphicsApi>(device, Default::default());
}

#[cfg(feature = "wgpu")]
pub fn sync_device(device: &RctdDevice) {
    RctdBackend::sync(device);
}

#[cfg(not(feature = "wgpu"))]
pub fn sync_device(_device: &RctdDevice) {}
