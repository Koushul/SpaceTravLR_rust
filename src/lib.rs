#[cfg(feature = "spatial-viewer")]
pub mod adata_query;
pub mod adata_terminal_scatter;
pub mod betadata;
#[cfg(feature = "spatial-viewer")]
pub mod betadata_view;
pub mod cnn_gating;
pub mod celloracle;
pub mod condition_split;
pub mod config;
pub mod estimator;
#[cfg(feature = "spatial-viewer")]
pub mod foyer_perturb_cache;
pub mod grn_extra;
pub mod lasso;
pub mod ligand;
pub mod model;
pub mod network;
pub mod perturb;
pub mod perturb_batch;
pub mod perturb_mode;
#[cfg(feature = "tui")]
pub mod perturb_tui;
#[cfg(feature = "self-update")]
pub mod self_update;
pub mod run_summary_html;
pub mod scanpy_preprocess;
pub mod spatial_estimator;
#[cfg(feature = "tui")]
pub mod training_demo;
pub mod training_hud;
pub mod training_log;
#[cfg(feature = "tui")]
pub mod training_tui;
#[cfg(feature = "tui")]
pub mod tui_theme;
pub mod transition_umap;
pub use betadata::{
    BetaFrame, Betabase, BetadataCollectAggregate, CollectedInteraction, GeneMatrix,
    TopBetaCoefficient, betadata_collect_interactions_parallel, write_betadata_feather,
    write_betadata_feather_to_writer,
};
pub use cnn_gating::CnnGateDecision;
pub use config::{
    CnnConfig, CnnOutputActivation, CnnTrainingMode, HybridCnnGatingConfig,
    RUN_REPRO_TOML_FILENAME, SpaceshipConfig, canonical_adata_stem,
    default_output_dir_for_adata_path, expand_user_path, filter_training_var_names,
    normalize_ui_path, resolve_training_target_genes,
};
pub use estimator::{CachedSpatialData, ClusterTrainingSummary, ClusteredGCNNWR};
pub use model::{CellularNicheNetwork, CellularNicheNetworkConfig};
pub use run_summary_html::{RunSummaryParams, write_run_summary_html};
pub use spatial_estimator::{
    cache_received_ligands_uns_for_processed_h5ad, materialize_canonical_training_adata,
    read_h5ad_expression_dense_f64, read_h5ad_obs_column_str, read_h5ad_var_names,
    SpatialCellularProgramsEstimator,
};
pub use training_hud::RunConfigSummary;
