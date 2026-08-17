//! Preprocess parameter sensitivity: different [`RustPreprocessParams`] values yield expected shapes.

mod common;

use std::process::Command;

use common::uv_python::uv_available;
use spacetravlr::rust_preprocess::{
    RustPreprocessParams, RustPreprocessSteps, rust_preprocess_h5ad_to_memory,
};

fn write_toy_h5ad(path: &std::path::Path, py_body: &str) -> bool {
    Command::new(common::uv_python::uv_bin())
        .env_remove("PYTHONPATH")
        .env("PYTHONNOUSERSITE", "1")
        .args([
            "run",
            "--isolated",
            "--with",
            "numpy<2",
            "--with",
            "anndata>=0.11",
        ])
        .arg("python")
        .arg("-c")
        .arg(py_body)
        .arg(path.to_str().expect("utf-8 path"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn standard_toy_h5ad(path: &std::path::Path) -> bool {
    let py = r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

p = Path(sys.argv[1])
n_obs, n_var = 60, 200
rng = np.random.default_rng(11)
x = rng.poisson(4, size=(n_obs, n_var)).astype(np.float64)
a = ad.AnnData(X=x)
a.obs_names = [f"cell{i}" for i in range(n_obs)]
a.var_names = [f"g{i}" for i in range(n_var)]
a.write_h5ad(p)
"#;
    write_toy_h5ad(path, py)
}

fn qc_variable_h5ad(path: &std::path::Path) -> bool {
    let py = r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

p = Path(sys.argv[1])
n_obs, n_var = 40, 120
rng = np.random.default_rng(3)
x = np.zeros((n_obs, n_var), dtype=np.float64)
# dense cells
x[:20, :] = rng.poisson(5, size=(20, n_var))
# sparse cells (few genes)
for i in range(20, n_obs):
    x[i, :8] = rng.poisson(3, size=8)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"g{i}" for i in range(n_var)]
a.write_h5ad(p)
"#;
    write_toy_h5ad(path, py)
}

fn run_pca_only(h5: &std::path::Path, params: &RustPreprocessParams) -> (usize, usize, usize) {
    let adata = rust_preprocess_h5ad_to_memory(h5, params, &RustPreprocessSteps::UMAP_LAB_PCA_ONLY)
        .expect("preprocess");
    let pca = adata
        .obsm()
        .get_array("X_pca")
        .expect("X_pca")
        .get_shape()
        .expect("shape");
    (adata.n_obs(), adata.n_vars(), pca[1])
}

#[test]
#[ignore = "requires uv/python (isolated `uv run`); default off — run `cargo test -- --ignored`"]
fn n_pca_components_sets_x_pca_width() {
    if !uv_available() {
        eprintln!("skip: uv not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!("prep_params_pca_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h5 = dir.join("toy.h5ad");
    assert!(standard_toy_h5ad(&h5), "uv toy h5ad failed");

    let low = RustPreprocessParams {
        n_top_hvg: 100,
        n_pca_components: 6,
        ..Default::default()
    };

    let high = RustPreprocessParams {
        n_top_hvg: 100,
        n_pca_components: 18,
        ..Default::default()
    };

    let (_, _, ncol_low) = run_pca_only(&h5, &low);
    let (_, _, ncol_high) = run_pca_only(&h5, &high);

    assert_eq!(ncol_low, 6, "expected X_pca ncol == n_pca_components (low)");
    assert_eq!(
        ncol_high, 18,
        "expected X_pca ncol == n_pca_components (high)"
    );
    assert_ne!(ncol_low, ncol_high);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires uv/python (isolated `uv run`); default off — run `cargo test -- --ignored`"]
fn n_top_hvg_caps_gene_subset() {
    if !uv_available() {
        eprintln!("skip: uv not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!("prep_params_hvg_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h5 = dir.join("toy.h5ad");
    assert!(standard_toy_h5ad(&h5), "uv toy h5ad failed");

    let small = RustPreprocessParams {
        n_top_hvg: 40,
        n_pca_components: 8,
        ..Default::default()
    };

    let large = RustPreprocessParams {
        n_top_hvg: 120,
        n_pca_components: 8,
        ..Default::default()
    };

    let (_, nvar_small, _) = run_pca_only(&h5, &small);
    let (_, nvar_large, _) = run_pca_only(&h5, &large);

    assert!(
        nvar_small <= 40,
        "n_vars should respect small n_top_hvg, got {nvar_small}"
    );
    assert!(
        nvar_large <= 120,
        "n_vars should respect large n_top_hvg, got {nvar_large}"
    );
    assert!(
        nvar_small < nvar_large,
        "larger n_top_hvg should retain more genes ({nvar_small} vs {nvar_large})"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires uv/python (isolated `uv run`); default off — run `cargo test -- --ignored`"]
fn min_genes_filters_sparse_cells() {
    if !uv_available() {
        eprintln!("skip: uv not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!("prep_params_qc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h5 = dir.join("qc.h5ad");
    assert!(qc_variable_h5ad(&h5), "uv qc h5ad failed");

    let permissive = RustPreprocessParams {
        min_genes: 5,
        min_cells: 1,
        n_top_hvg: 80,
        n_pca_components: 6,
        ..Default::default()
    };

    let strict = RustPreprocessParams {
        min_genes: 50,
        min_cells: 1,
        n_top_hvg: 80,
        n_pca_components: 6,
        ..Default::default()
    };

    let steps = RustPreprocessSteps {
        qc_filter: true,
        normalize_log1p: true,
        hvg_pca: true,
        run_umap_and_graph: false,
        write_leiden: false,
        run_magic_impute: false,
    };

    let adata_perm =
        rust_preprocess_h5ad_to_memory(&h5, &permissive, &steps).expect("permissive preprocess");
    let adata_strict =
        rust_preprocess_h5ad_to_memory(&h5, &strict, &steps).expect("strict preprocess");

    assert_eq!(
        adata_perm.n_obs(),
        40,
        "permissive QC should keep all cells"
    );
    assert!(
        adata_strict.n_obs() < adata_perm.n_obs(),
        "strict min_genes=50 should drop sparse cells (got {} vs {})",
        adata_strict.n_obs(),
        adata_perm.n_obs()
    );
    assert_eq!(
        adata_strict.n_obs(),
        20,
        "expected only the 20 dense cells after min_genes=50"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn config_preprocess_converts_to_rust_params() {
    use spacetravlr::PreprocessConfig;

    let cfg = PreprocessConfig {
        n_top_hvg: 777,
        n_pca_components: 9,
        min_genes: 42,
        ..Default::default()
    };
    let rust = cfg.to_rust_preprocess_params();
    assert_eq!(rust.n_top_hvg, 777);
    assert_eq!(rust.n_pca_components, 9);
    assert_eq!(rust.min_genes, 42);
}
