#![cfg(feature = "umap-lab")]

use std::path::Path;

#[test]
fn snrna_human_tonsil_v2_h5ad_loads_for_umap_lab() {
    let path = Path::new("/Volumes/SSD/training_data/snrna_human_tonsil_v2.h5ad");
    if !path.is_file() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    let params = spacetravlr::RustPreprocessParams::default();
    let loaded = spacetravlr::umap_lab_load_pca_session(path, &params)
        .unwrap_or_else(|e| panic!("umap_lab_load_pca_session: {e:#}"));
    assert_eq!(loaded.pca.nrows(), 5778, "n_obs");
    assert!(
        loaded.pca.ncols() >= 2,
        "PCA should have at least 2 components, got {}",
        loaded.pca.ncols()
    );
}
