use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use burn_autodiff::Autodiff;
use spacetravlr::config::{CnnConfig, CnnTrainingMode, ModelExportConfig, SpaceshipConfig};
use spacetravlr::spatial_estimator::SpatialCellularProgramsEstimator;
use spacetravlr::training_hud::TrainingHud;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub(crate) enum ComputeChoice {
    Wgpu(WgpuDevice),
    NdArray(NdArrayDevice),
}

impl ComputeChoice {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            ComputeChoice::Wgpu(_) => "WebGPU",
            ComputeChoice::NdArray(_) => "CPU (NdArray)",
        }
    }
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Discrete / integrated / virtual GPUs can run Burn WebGPU. CPU and `Other` adapters
/// (llvmpipe, SwiftShader) are not a reliable CNN path — NdArray is used instead.
pub(crate) fn wgpu_device_type_ok(device_type: wgpu::DeviceType) -> bool {
    matches!(
        device_type,
        wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::VirtualGpu
    )
}

/// Burn **WebGPU** when a real GPU adapter is present and `WgpuDevice` initializes;
/// otherwise Burn **NdArray** on CPU. Never panics if wgpu/Vulkan/Metal is missing.
/// `SPACETRAVLR_FORCE_CPU` / `SPACETRAVLR_DISABLE_WGPU` skip the adapter probe.
pub(crate) fn select_compute_backend() -> ComputeChoice {
    let choice = if env_truthy("SPACETRAVLR_FORCE_CPU") || env_truthy("SPACETRAVLR_DISABLE_WGPU") {
        ComputeChoice::NdArray(NdArrayDevice::Cpu)
    } else {
        match wgpu_adapter_probe_cached().as_ref() {
            Some(info) if wgpu_device_type_ok(info.device_type) => match try_wgpu_device() {
                Some(device) => ComputeChoice::Wgpu(device),
                None => ComputeChoice::NdArray(NdArrayDevice::Cpu),
            },
            Some(_) | None => ComputeChoice::NdArray(NdArrayDevice::Cpu),
        }
    };
    log_compute_backend_choice(&choice);
    choice
}

fn try_wgpu_device() -> Option<WgpuDevice> {
    catch_unwind(AssertUnwindSafe(WgpuDevice::default)).ok()
}

fn log_compute_backend_choice(choice: &ComputeChoice) {
    if env_truthy("SPACETRAVLR_QUIET_COMPUTE") {
        return;
    }
    match choice {
        ComputeChoice::Wgpu(_) => {
            if let Some(info) = wgpu_adapter_probe_cached().as_ref() {
                eprintln!(
                    "spacetravlr: CNN/compute backend = WebGPU (adapter `{}`, {:?})",
                    info.name, info.device_type
                );
            } else {
                eprintln!("spacetravlr: CNN/compute backend = WebGPU");
            }
        }
        ComputeChoice::NdArray(_) => {
            if env_truthy("SPACETRAVLR_FORCE_CPU") || env_truthy("SPACETRAVLR_DISABLE_WGPU") {
                eprintln!(
                    "spacetravlr: CNN/compute backend = CPU (NdArray) — SPACETRAVLR_FORCE_CPU or SPACETRAVLR_DISABLE_WGPU is set"
                );
            } else if let Some(info) = wgpu_adapter_probe_cached().as_ref() {
                if wgpu_device_type_ok(info.device_type) {
                    eprintln!(
                        "spacetravlr: CNN/compute backend = CPU (NdArray) — WebGPU device init failed after adapter `{}` ({:?}); CNN training will be much slower than WebGPU",
                        info.name, info.device_type
                    );
                } else {
                    eprintln!(
                        "spacetravlr: CNN/compute backend = CPU (NdArray) — wgpu adapter `{}` is {:?} (software/CPU); using NdArray (CNN training will be much slower than a GPU)",
                        info.name, info.device_type
                    );
                }
            } else {
                eprintln!(
                    "spacetravlr: CNN/compute backend = CPU (NdArray) — no usable wgpu GPU adapter (CNN training will be much slower than WebGPU)"
                );
            }
        }
    }
}

static WGPU_ADAPTER_PROBE: OnceLock<Option<wgpu::AdapterInfo>> = OnceLock::new();

fn wgpu_adapter_probe_cached() -> &'static Option<wgpu::AdapterInfo> {
    WGPU_ADAPTER_PROBE.get_or_init(preferred_wgpu_adapter_info)
}

fn probe_wgpu_adapter_info() -> Option<wgpu::AdapterInfo> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;
        Some(adapter.get_info())
    })
}

fn preferred_wgpu_adapter_info() -> Option<wgpu::AdapterInfo> {
    catch_unwind(AssertUnwindSafe(probe_wgpu_adapter_info))
        .ok()
        .flatten()
}

pub(crate) fn compute_hardware_details(choice: &ComputeChoice) -> String {
    match choice {
        ComputeChoice::Wgpu(_) => {
            if let Some(info) = wgpu_adapter_probe_cached().as_ref() {
                format!(
                    "{} ({:?}, {} backend)",
                    info.name, info.device_type, info.backend
                )
            } else {
                "adapter details unavailable".to_string()
            }
        }
        ComputeChoice::NdArray(_) => {
            let arch = std::env::consts::ARCH;
            let os = std::env::consts::OS;
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            format!("{} {} CPU ({} threads)", os, arch, threads)
        }
    }
}

pub(crate) struct FitAllGenesParams<'a> {
    pub path: &'a str,
    pub obs_row_subset: Option<std::sync::Arc<[usize]>>,
    pub radius: f64,
    pub spatial_dim: usize,
    pub contact_distance: f64,
    pub tf_ligand_cutoff: f64,
    pub max_ligands: Option<usize>,
    pub use_tf_modulators: bool,
    pub use_lr_modulators: bool,
    pub use_tfl_modulators: bool,
    pub layer: &'a str,
    pub cluster_annot: &'a str,
    pub cnn: &'a CnnConfig,
    pub epochs: usize,
    pub learning_rate: f64,
    pub score_threshold: f64,
    pub l1_reg: f64,
    pub group_reg: f64,
    pub n_iter: usize,
    pub tol: f64,
    pub cnn_training_mode: CnnTrainingMode,
    pub gene_filter: Option<Vec<String>>,
    pub max_genes: Option<usize>,
    pub n_parallel: usize,
    pub output_dir: &'a str,
    pub model_export: &'a ModelExportConfig,
    pub hud: Option<TrainingHud>,
    pub network_data_dir: Option<String>,
    pub tf_priors_feather: Option<String>,
    pub write_minimal_repro_h5ad: bool,
    pub spaceship_config: &'a SpaceshipConfig,
    pub config_source_path: Option<std::path::PathBuf>,
    /// Loaded from shared `spacetravlr_run_repro.toml` (`--join-output-dir`); skips overwriting that file at end.
    pub join_training: bool,
    pub verbose: bool,
}

macro_rules! dispatch_fit_all_genes {
    ($backend:ty, $p:expr, $device:expr) => {
        SpatialCellularProgramsEstimator::<Autodiff<$backend>, anndata_hdf5::H5>::fit_all_genes(
            $p.path,
            $p.obs_row_subset.clone(),
            $p.radius,
            $p.spatial_dim,
            $p.contact_distance,
            $p.tf_ligand_cutoff,
            $p.max_ligands,
            $p.use_tf_modulators,
            $p.use_lr_modulators,
            $p.use_tfl_modulators,
            $p.layer,
            $p.cluster_annot,
            $p.cnn,
            $p.epochs,
            $p.learning_rate,
            $p.score_threshold,
            $p.l1_reg,
            $p.group_reg,
            $p.n_iter,
            $p.tol,
            $p.cnn_training_mode,
            $p.gene_filter.clone(),
            $p.max_genes,
            $p.n_parallel,
            $p.output_dir,
            $p.model_export,
            $p.hud.clone(),
            $p.network_data_dir.as_deref(),
            $p.tf_priors_feather.as_deref(),
            $p.write_minimal_repro_h5ad,
            $p.spaceship_config,
            $p.config_source_path.clone(),
            $p.join_training,
            $p.verbose,
            None,
            $device,
        )
    };
}

pub(crate) fn fit_all_genes_dispatch(
    p: &FitAllGenesParams<'_>,
    choice: &ComputeChoice,
) -> anyhow::Result<()> {
    match choice {
        ComputeChoice::Wgpu(device) => dispatch_fit_all_genes!(Wgpu, p, device),
        ComputeChoice::NdArray(device) => {
            dispatch_fit_all_genes!(NdArray<f32, i32>, p, device)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::wgpu_device_type_ok;
    use wgpu::DeviceType;

    #[test]
    fn gpu_device_types_are_accepted() {
        assert!(wgpu_device_type_ok(DeviceType::DiscreteGpu));
        assert!(wgpu_device_type_ok(DeviceType::IntegratedGpu));
        assert!(wgpu_device_type_ok(DeviceType::VirtualGpu));
    }

    #[test]
    fn cpu_and_other_adapters_are_rejected() {
        assert!(!wgpu_device_type_ok(DeviceType::Cpu));
        assert!(!wgpu_device_type_ok(DeviceType::Other));
    }
}
