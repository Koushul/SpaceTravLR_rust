use std::path::{Path, PathBuf};

/// Process-wide defaults for batch/HPC nodes (NFS HDF5 locking, headless XDG runtime).
/// Safe to call more than once.
pub fn ensure_process_env() {
    ensure_hdf5_no_file_locking();
    ensure_xdg_runtime_dir();
}

/// Ensure HDF5 file locking is disabled so `.h5ad` opens succeed on network
/// filesystems (NFS, Lustre, GPFS) that do not support POSIX advisory locks.
/// Safe to call more than once; only the first call sets the variable.
pub fn ensure_hdf5_no_file_locking() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("HDF5_USE_FILE_LOCKING").is_none() {
            unsafe {
                std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
            }
        }
    });
}

/// On batch/HPC nodes `XDG_RUNTIME_DIR` is often unset or points at a missing path (no
/// systemd user session). Wayland, DBus, and some Python stacks log noisy errors when it
/// is invalid; set a private writable runtime dir before those libraries initialize.
pub fn ensure_xdg_runtime_dir() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if xdg_runtime_dir_is_usable(std::env::var_os("XDG_RUNTIME_DIR")) {
            return;
        }
        for dir in xdg_runtime_dir_candidates() {
            if install_xdg_runtime_dir(&dir) {
                unsafe {
                    std::env::set_var("XDG_RUNTIME_DIR", dir.as_os_str());
                }
                return;
            }
        }
    });
}

fn xdg_runtime_dir_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    #[cfg(unix)]
    if let Some(uid) = unix_effective_uid() {
        out.push(PathBuf::from(format!("/run/user/{uid}")));
    }
    let tmp = std::env::var_os("TMPDIR")
        .or_else(|| std::env::var_os("TEMP"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    #[cfg(unix)]
    let suffix = unix_effective_uid()
        .map(|u| u.to_string())
        .unwrap_or_else(|| std::process::id().to_string());
    #[cfg(not(unix))]
    let suffix = std::process::id().to_string();
    out.push(tmp.join(format!("spacetravlr-xdg-runtime-{suffix}")));
    out
}

fn xdg_runtime_dir_is_usable(val: Option<std::ffi::OsString>) -> bool {
    let Some(val) = val.filter(|v| !v.is_empty()) else {
        return false;
    };
    let path = Path::new(&val);
    if !path.is_dir() {
        return false;
    }
    let probe = path.join(format!(".spacetravlr_xdg_probe_{}", std::process::id()));
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
    {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}


#[cfg(unix)]
fn unix_effective_uid() -> Option<u32> {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
}

#[cfg(not(unix))]
fn unix_effective_uid() -> Option<u32> {
    None
}

fn install_xdg_runtime_dir(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }
    xdg_runtime_dir_is_usable(Some(std::ffi::OsString::from(dir.as_os_str())))
}

#[cfg(test)]
mod env_tests {
    use super::*;

    #[test]
    fn xdg_runtime_dir_usable_when_writable() {
        let dir = std::env::temp_dir().join(format!(
            "st_xdg_ok_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(install_xdg_runtime_dir(&dir));
        assert!(xdg_runtime_dir_is_usable(Some(dir.as_os_str().to_os_string())));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn xdg_runtime_dir_not_usable_when_missing() {
        assert!(!xdg_runtime_dir_is_usable(Some("/nonexistent/spacetravlr/xdg".into())));
    }
}

#[cfg(feature = "spatial-viewer")]
pub mod adata_query;
pub mod adata_terminal_scatter;
pub mod betadata;
#[cfg(feature = "spatial-viewer")]
pub mod betadata_view;
pub mod celloracle;
pub mod condition_split;
pub mod config;
pub mod estimator;
#[cfg(feature = "spatial-viewer")]
pub mod foyer_perturb_cache;
pub mod grn_extra;
pub mod h5ad_peek;
pub mod lasso;
pub mod ligand;
pub mod malt_label_transfer;
pub mod model;
mod modulator_scale;
pub mod network;
pub mod perturb;
pub use perturb::{ComputeSplashAllProgressArgs, PerturbWithTargetsInputs};
pub mod perturb_batch;
pub mod perturb_mode;
pub use perturb_mode::{
    CollectInteractionsObs, load_obs_column_for_collect_interactions,
    load_obs_for_collect_interactions,
};
#[cfg(feature = "tui")]
pub mod perturb_tui;
pub mod run_summary_html;
pub mod scanpy_preprocess;
pub mod rust_preprocess;
pub use rust_preprocess::{
    FuzzyGraph, UmapLabKnnCache, RustPreprocessParams, RustPreprocessSteps, UmapLabLoaded,
    clamp_umap_min_dist_spread, fuzzy_graph_induced_subgraph, leiden_labels_from_graph,
    leiden_labels_subcluster_into, rust_preprocess_h5ad, rust_preprocess_h5ad_to_memory,
    rust_preprocess_h5ad_with_steps, umap_lab_gene_expression_from_h5ad,
    umap_lab_gene_expression_from_h5ad_source, umap_lab_load_pca_session, umap_lab_read_obs_column,
    umap_lab_run_embedding, umap_lab_run_magic_imputed_leiden,
};
#[cfg(feature = "self-update")]
pub mod self_update;
pub mod spatial_estimator;
pub mod verify_bundle;
#[cfg(feature = "tui")]
pub mod training_demo;
pub mod training_hud;
pub mod training_log;
#[cfg(feature = "tui")]
pub mod training_tui;
pub mod transition_umap;
#[cfg(feature = "tui")]
pub mod tui_theme;
pub use betadata::{
    BetaAggregates, BetaFrame, BetaFrameFromParts, Betabase, BetadataCollectAggregate,
    CollectedInteraction, CollectedInteractionFull, CollectedInteractionRow,
    CollectedInteractionRowFull, GeneMatrix, TopBetaCoefficient,
    betadata_collect_interactions_all_cell_types,
    betadata_collect_interactions_all_cell_types_full,
    betadata_collect_interactions_all_cell_types_one_gene,
    betadata_collect_interactions_all_cell_types_one_gene_full,
    betadata_collect_interactions_parallel, betadata_collect_interactions_parallel_full,
    write_betadata_feather, write_betadata_feather_to_writer, write_collected_interactions_feather,
    write_collected_interactions_full_feather,
};
pub use config::{
    CnnConfig, CnnLrSchedule, CnnOutputActivation, CnnTrainingMode, RUN_REPRO_TOML_FILENAME,
    SpaceshipConfig, canonical_adata_stem, canonical_training_prep_stem,
    default_output_dir_for_adata_path, expand_user_path, filter_training_var_names,
    mix_execution_random_seed, normalize_ui_path, resolve_malt_label_transfer_py_path,
    resolve_spaceship_config_toml_path, resolve_training_target_genes,
};
pub use estimator::{
    CachedSpatialData, ClusterTrainingSummary, ClusteredGCNNWR, ClusteredGcnNwrCnnRefineInputs,
    ClusteredGcnNwrFitInputs, CnnEpochHudSlot, FittedClusterResult, PredictBetasInput,
    run_benchmark_mock_cluster_cnn_training, train_cluster_cnn_epochs, TrainClusterCnnEpochsInput,
};
pub use model::{CellularNicheNetwork, CellularNicheNetworkConfig};
pub use h5ad_peek::print_h5ad_peek;
pub use run_summary_html::{RunSummaryParams, write_run_summary_html};
pub use spatial_estimator::{
    GENE_PERFORMANCE_FEATHER_NAME, SpatialCellularProgramsEstimator, gene_performance_feather_path,
    materialize_canonical_training_adata, read_h5ad_expression_dense_f64, read_h5ad_obs_column_str,
    read_h5ad_var_names,
};
pub use training_hud::RunConfigSummary;
