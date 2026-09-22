#![cfg(feature = "umap-lab")]

use std::path::{Path, PathBuf};

#[test]
#[ignore = "local tonsil h5ad; set SPACETRAVLR_UMAP_LAB_H5AD or keep the SSD path and run cargo test -- --ignored"]
fn snrna_human_tonsil_v2_h5ad_loads_for_umap_lab() {
    let path = std::env::var("SPACETRAVLR_UMAP_LAB_H5AD").map_or_else(
        |_| PathBuf::from("/Volumes/SSD/training_data/snrna_human_tonsil_v2.h5ad"),
        PathBuf::from,
    );
    assert!(
        path.is_file(),
        "tonsil h5ad missing at {} (set SPACETRAVLR_UMAP_LAB_H5AD to a real file)",
        path.display()
    );
    let params = spacetravlr::RustPreprocessParams::default();
    let loaded = spacetravlr::umap_lab_load_pca_session(Path::new(&path), &params)
        .unwrap_or_else(|e| panic!("umap_lab_load_pca_session: {e:#}"));
    assert_eq!(loaded.pca.nrows(), 5778, "n_obs");
    assert!(
        loaded.pca.ncols() >= 2,
        "PCA should have at least 2 components, got {}",
        loaded.pca.ncols()
    );
}
