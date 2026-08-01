//! SpaceTravLR CellularNicheNetwork CNN trainer for WebAssembly.
//! Prefers Burn **WebGPU** when available; falls back to **NdArray** (CPU WASM).

pub mod backend;
pub mod model;
pub mod pack;
pub mod train;

pub use pack::{
    ClusterTrainResult, CnnClusterPack, CnnGeneTrainPack, CnnTrainHyperparams, GeneTrainResult,
    decode_pack, encode_pack,
};
#[cfg(not(target_arch = "wasm32"))]
pub use train::{train_gene_pack, train_gene_pack_bytes};
pub use train::{train_gene_pack_async, train_gene_pack_bytes_async};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

/// Current compute backend name: `"webgpu"` or `"ndarray"`.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn active_backend_name() -> String {
    backend::backend_name(backend::active_backend()).to_string()
}

/// Whether `navigator.gpu` is present (does not fully initialize the device).
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn webgpu_available() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        backend::navigator_gpu_present()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// Try to initialize Burn WebGPU. On success, subsequent [`train_pack`] uses the GPU.
/// Falls back is left to the caller — this returns an error string if WebGPU init fails.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub async fn init_webgpu() -> Result<String, JsValue> {
    match backend::wgpu_init::init_webgpu_runtime().await {
        Ok(()) => {
            backend::set_active_backend(backend::BACKEND_WEBGPU);
            Ok("webgpu".into())
        }
        Err(e) => {
            backend::set_active_backend(backend::BACKEND_NDARRAY);
            Err(JsValue::from_str(&e))
        }
    }
}

/// Force NdArray (CPU WASM) backend for subsequent training.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn use_ndarray_backend() {
    backend::set_active_backend(backend::BACKEND_NDARRAY);
}

/// Train CNN clusters from a bincode [`CnnGeneTrainPack`]. Returns JSON [`GeneTrainResult`].
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub async fn train_pack(pack_bytes: &[u8]) -> Result<JsValue, JsValue> {
    let result = match backend::active_backend() {
        backend::BACKEND_WEBGPU => {
            use burn::backend::wgpu::WgpuDevice;
            use burn::backend::Wgpu;
            use burn_autodiff::Autodiff;
            type B = Autodiff<Wgpu<f32, i32>>;
            let device = WgpuDevice::default();
            train::train_gene_pack_bytes_on_async::<B>(pack_bytes, &device, "webgpu").await
        }
        _ => train::train_gene_pack_bytes_async(pack_bytes).await,
    }
    .map_err(|e| JsValue::from_str(&e))?;
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
fn smoke_pack() -> CnnGeneTrainPack {
    let n = 32usize;
    let h = 8usize;
    let n_mod = 4usize;
    let n_clust = 3usize;
    CnnGeneTrainPack {
        gene: "SMOKE".into(),
        hyperparams: CnnTrainHyperparams {
            epochs: 2,
            cnn_minibatch_size: 16,
            cnn_max_cells_per_epoch: Some(32),
            cnn_early_stop_patience: 0,
            ..Default::default()
        },
        clusters: vec![CnnClusterPack {
            cluster_id: 0,
            n_cells: n as u32,
            n_modulators: n_mod as u32,
            n_clusters: n_clust as u32,
            spatial_h: h as u32,
            spatial_w: h as u32,
            vision_in_channels: 1,
            spatial_maps: vec![0.1f32; n * 1 * h * h],
            x: vec![0.05f32; n * n_mod],
            spatial_features: vec![0.02f32; n * n_clust],
            y: vec![0.0f32; n],
            anchors: {
                let mut a = vec![0.5f32];
                a.extend(std::iter::repeat_n(0.1f32, n_mod));
                a
            },
            y_lasso: None,
            lasso_r2: 0.5,
        }],
    }
}

/// Minimal WebGPU probe: create a tiny tensor, run one matmul-ish op, async-read a scalar.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub async fn webgpu_probe() -> Result<String, JsValue> {
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::Wgpu;
    use burn::tensor::{Distribution, ElementConversion, Tensor};
    use burn_autodiff::Autodiff;

    if !backend::navigator_gpu_present() {
        return Err(JsValue::from_str("navigator.gpu missing"));
    }
    match backend::wgpu_init::init_webgpu_runtime().await {
        Ok(()) => {}
        Err(e) => return Err(JsValue::from_str(&e)),
    }
    backend::set_active_backend(backend::BACKEND_WEBGPU);

    type B = Autodiff<Wgpu<f32, i32>>;
    let device = WgpuDevice::default();
    let a = Tensor::<B, 2>::random([4, 4], Distribution::Default, &device);
    let b = Tensor::<B, 2>::random([4, 4], Distribution::Default, &device);
    let c = a.matmul(b).sum();
    let s = c.into_scalar_async().await;
    let v = s.elem::<f32>();
    Ok(format!("webgpu_probe ok scalar={v}"))
}

/// Smoke-test on the active backend. Returns wall time ms.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub async fn smoke_train_ms() -> u32 {
    let pack = smoke_pack();
    match backend::active_backend() {
        backend::BACKEND_WEBGPU => {
            use burn::backend::wgpu::WgpuDevice;
            use burn::backend::Wgpu;
            use burn_autodiff::Autodiff;
            type B = Autodiff<Wgpu<f32, i32>>;
            let device = WgpuDevice::default();
            train::train_gene_pack_on_async::<B>(&pack, &device, "webgpu")
                .await
                .wall_ms
        }
        _ => train::train_gene_pack_async(&pack).await.wall_ms,
    }
}
