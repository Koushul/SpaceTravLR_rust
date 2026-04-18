//! `spacetravlr-niche` — per-cell **microniche** detection that learns a CNN
//! over the per-cell *gene × gene* (target × modulator) splash Jacobian.
//!
//! Pipeline
//! --------
//! 1. **`image`** — turn `compute_splash_all` output (one
//!    `(n_cells, n_modulators)` matrix per trained target, with per-target
//!    modulator alphabets) into a single per-cell **`(n_targets, n_modulators)`
//!    image** with a unified, alphabetically sorted target axis and a unified,
//!    alphabetically sorted modulator axis. Missing entries are 0.
//! 2. **`model`** — a small Conv2d/BatchNorm encoder over those images that
//!    produces a `D`-dim embedding per cell.
//! 3. **`train`** — composite loss
//!      * `lambda_recon` × MSE reconstruction of the input image (autoencoder)
//!      * `lambda_func`  × functional head MSE: predict each cell's
//!        normalized **program-activity vector** computed from
//!        `‖J[c, :, m]‖₁` summed over modulators belonging to each
//!        unsupervised "program" cluster of co-active modulators
//!      * `lambda_spatial` × neighbour-coherence loss that pulls embeddings of
//!        spatial neighbours together (cosine similarity)
//! 4. K-means on embeddings → integer niche labels per cell.
//! 5. **`io`** — write `niche_labels.feather` + `.csv` with embedding columns.
//!
//! Every step is independently testable — see `tests/test_niche.rs` for an
//! end-to-end run on `synth::make_synthetic_run`.

pub mod image;
pub mod io;
pub mod kmeans;
pub mod metrics;
pub mod model;
pub mod runtime;
pub mod synth;
pub mod train;

pub use image::{NicheImageStack, build_niche_image_stack};
pub use io::{NicheLabels, write_niche_labels_csv, write_niche_labels_feather};
pub use kmeans::{KMeansResult, kmeans_lloyd};
pub use metrics::{adjusted_rand_index, normalized_mutual_info, spatial_purity_knn};
pub use model::{NicheEncoder, NicheEncoderConfig, NicheHeads};
pub use runtime::{NicheRuntime, NicheRuntimeBuilder, NicheRuntimeOutputs};
pub use synth::{SyntheticNicheRun, make_synthetic_run};
pub use train::{NicheTrainConfig, NicheTrainOutputs, train_niche_encoder};
