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
#[cfg(feature = "tui")]
pub mod perturb_tui;
pub mod run_summary_html;
pub mod scanpy_preprocess;
#[cfg(feature = "self-update")]
pub mod self_update;
pub mod spatial_estimator;
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
    BetaFrame, BetaFrameFromParts, Betabase, BetadataCollectAggregate, CollectedInteraction,
    GeneMatrix, TopBetaCoefficient, betadata_collect_interactions_parallel, write_betadata_feather,
    write_betadata_feather_to_writer,
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
    ClusteredGcnNwrFitInputs, CnnEpochHudSlot, FittedClusterResult,
    run_benchmark_mock_cluster_cnn_training, train_cluster_cnn_epochs,
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
