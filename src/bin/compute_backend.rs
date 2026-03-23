use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use burn_autodiff::Autodiff;
use space_trav_lr_rust::config::CnnConfig;
use space_trav_lr_rust::spatial_estimator::SpatialCellularProgramsEstimator;
use space_trav_lr_rust::training_hud::TrainingHud;

#[derive(Clone, Debug)]
pub(crate) enum ComputeChoice {
    Wgpu(WgpuDevice),
    NdArray(NdArrayDevice),
}

impl ComputeChoice {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            ComputeChoice::Wgpu(_) => "WGPU",
            ComputeChoice::NdArray(_) => "CPU (NdArray)",
        }
    }
}

pub(crate) fn select_compute_backend() -> ComputeChoice {
    if std::env::var("SPACETRAVLR_FORCE_CPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return ComputeChoice::NdArray(NdArrayDevice::Cpu);
    }
    if preferred_wgpu_adapter_info().is_some() {
        ComputeChoice::Wgpu(WgpuDevice::default())
    } else {
        ComputeChoice::NdArray(NdArrayDevice::Cpu)
    }
}

fn preferred_wgpu_adapter_info() -> Option<wgpu::AdapterInfo> {
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

pub(crate) fn compute_hardware_details(choice: &ComputeChoice) -> String {
    match choice {
        ComputeChoice::Wgpu(_) => {
            if let Some(info) = preferred_wgpu_adapter_info() {
                format!(
                    "{} ({:?}, {} backend)",
                    info.name, info.device_type, info.backend
                )
            } else {
                "WGPU adapter (details unavailable)".to_string()
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
    pub radius: f64,
    pub spatial_dim: usize,
    pub contact_distance: f64,
    pub tf_ligand_cutoff: f64,
    pub max_lr_pairs: Option<usize>,
    pub top_lr_pairs_by_mean_expression: Option<usize>,
    pub layer: &'a str,
    pub cnn: &'a CnnConfig,
    pub epochs: usize,
    pub learning_rate: f64,
    pub score_threshold: f64,
    pub l1_reg: f64,
    pub group_reg: f64,
    pub n_iter: usize,
    pub tol: f64,
    pub full_cnn: bool,
    pub gene_filter: Option<Vec<String>>,
    pub max_genes: Option<usize>,
    pub n_parallel: usize,
    pub output_dir: &'a str,
    pub hud: Option<TrainingHud>,
}

pub(crate) fn fit_all_genes_dispatch(
    p: &FitAllGenesParams<'_>,
    choice: &ComputeChoice,
) -> anyhow::Result<()> {
    match choice {
        ComputeChoice::Wgpu(device) => {
            SpatialCellularProgramsEstimator::<Autodiff<Wgpu>, anndata_hdf5::H5>::fit_all_genes(
                p.path,
                p.radius,
                p.spatial_dim,
                p.contact_distance,
                p.tf_ligand_cutoff,
                p.max_lr_pairs,
                p.top_lr_pairs_by_mean_expression,
                p.layer,
                p.cnn,
                p.epochs,
                p.learning_rate,
                p.score_threshold,
                p.l1_reg,
                p.group_reg,
                p.n_iter,
                p.tol,
                p.full_cnn,
                p.gene_filter.clone(),
                p.max_genes,
                p.n_parallel,
                p.output_dir,
                p.hud.clone(),
                device,
            )
        }
        ComputeChoice::NdArray(device) => SpatialCellularProgramsEstimator::<
            Autodiff<NdArray<f32, i32>>,
            anndata_hdf5::H5,
        >::fit_all_genes(
            p.path,
            p.radius,
            p.spatial_dim,
            p.contact_distance,
            p.tf_ligand_cutoff,
            p.max_lr_pairs,
            p.top_lr_pairs_by_mean_expression,
            p.layer,
            p.cnn,
            p.epochs,
            p.learning_rate,
            p.score_threshold,
            p.l1_reg,
            p.group_reg,
            p.n_iter,
            p.tol,
            p.full_cnn,
            p.gene_filter.clone(),
            p.max_genes,
            p.n_parallel,
            p.output_dir,
            p.hud.clone(),
            device,
        ),
    }
}
