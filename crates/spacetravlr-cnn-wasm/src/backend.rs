//! Runtime compute backend selection: WebGPU when available, else NdArray.

use std::sync::atomic::{AtomicU8, Ordering};

pub const BACKEND_NDARRAY: u8 = 0;
pub const BACKEND_WEBGPU: u8 = 1;

static ACTIVE: AtomicU8 = AtomicU8::new(BACKEND_NDARRAY);

pub fn active_backend() -> u8 {
    ACTIVE.load(Ordering::SeqCst)
}

pub fn set_active_backend(v: u8) {
    ACTIVE.store(v, Ordering::SeqCst);
}

pub fn backend_name(v: u8) -> &'static str {
    match v {
        BACKEND_WEBGPU => "webgpu",
        _ => "ndarray",
    }
}

/// True when the JS host exposes `navigator.gpu` (browser WebGPU).
#[cfg(target_arch = "wasm32")]
pub fn navigator_gpu_present() -> bool {
    let global = js_sys::global();
    let Ok(navigator) = js_sys::Reflect::get(&global, &js_sys::JsString::from("navigator")) else {
        return false;
    };
    if navigator.is_undefined() || navigator.is_null() {
        return false;
    }
    let Ok(gpu) = js_sys::Reflect::get(&navigator, &js_sys::JsString::from("gpu")) else {
        return false;
    };
    !gpu.is_undefined() && !gpu.is_null()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn navigator_gpu_present() -> bool {
    false
}

#[cfg(any(target_arch = "wasm32", feature = "webgpu"))]
pub mod wgpu_init {
    use burn::backend::wgpu::{
        init_setup_async, AutoGraphicsApi, RuntimeOptions, WgpuDevice,
    };

    /// Initialize Burn WGPU (WebGPU on wasm32, Metal/Vulkan/DX elsewhere).
    pub async fn init_webgpu_runtime() -> Result<(), String> {
        #[cfg(target_arch = "wasm32")]
        {
            if !super::navigator_gpu_present() {
                return Err(
                    "WebGPU unavailable: navigator.gpu is missing (use a browser with WebGPU)"
                        .into(),
                );
            }
        }
        let device = WgpuDevice::default();
        let _setup = init_setup_async::<AutoGraphicsApi>(&device, RuntimeOptions::default()).await;
        Ok(())
    }
}
