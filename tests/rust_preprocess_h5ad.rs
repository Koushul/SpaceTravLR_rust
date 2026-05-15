//! Integration: `rust_preprocess` loads real `.h5ad` from disk (Python-written) and runs the
//! in-memory pipeline — including digit-like `var_names` repair from `var['feature_name']`.

use std::process::Command;

use spacetravlr::rust_preprocess::{
    rust_preprocess_h5ad_to_memory, rust_preprocess_h5ad_with_steps, RustPreprocessParams,
    RustPreprocessSteps,
};

fn uv_ok() -> bool {
    Command::new(std::env::var_os("UV_BIN").unwrap_or_else(|| "uv".into()))
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn write_h5ad_digit_var_with_feature_name(path: &std::path::Path) -> std::process::ExitStatus {
    let py = r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

p = Path(sys.argv[1])
n_obs, n_var = 40, 150
rng = np.random.default_rng(7)
x = rng.poisson(3, size=(n_obs, n_var)).astype(np.float32).astype(np.float64)
a = ad.AnnData(X=x)
a.obs_names = [f"cell{i}" for i in range(n_obs)]
a.var_names = [str(i) for i in range(n_var)]
a.var["feature_name"] = [f"GeneSym{k}" for k in range(n_var)]
a.write_h5ad(p)
"#;
    Command::new(std::env::var_os("UV_BIN").unwrap_or_else(|| "uv".into()))
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
        .arg(py)
        .arg(path.to_str().expect("utf-8 path"))
        .status()
        .expect("spawn uv")
}

#[test]
fn rust_preprocess_memory_repairs_digit_var_index_from_feature_name() {
    if !uv_ok() {
        eprintln!("skip: uv not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "rust_preprocess_digit_var_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h5 = dir.join("digit_var.h5ad");
    assert!(
        write_h5ad_digit_var_with_feature_name(&h5).success(),
        "uv toy h5ad failed"
    );

    let params = RustPreprocessParams {
        n_top_hvg: 60,
        n_pca_components: 8,
        ..Default::default()
    };
    let adata = rust_preprocess_h5ad_to_memory(
        &h5,
        &params,
        &RustPreprocessSteps::UMAP_LAB_PCA_ONLY,
    )
    .expect("rust_preprocess_h5ad_to_memory");

    assert_eq!(adata.n_obs(), 40, "obs count");
    assert!(
        adata.n_vars() <= 60 && adata.n_vars() > 0,
        "expected HVG subset ≤ n_top_hvg, got n_vars={}",
        adata.n_vars()
    );
    for (i, name) in adata.var_names().iter().enumerate() {
        let t = name.trim();
        let digit_only = !t.is_empty() && t.chars().all(|c| c.is_ascii_digit());
        assert!(
            !digit_only,
            "var {i} name should not be digit-only after restore: {name:?}"
        );
        assert!(
            name.starts_with("GeneSym"),
            "var {i} expected GeneSym* from feature_name, got {name:?}"
        );
    }

    let pca = adata
        .obsm()
        .get_array("X_pca")
        .expect("X_pca after PCA-only preprocess");
    let sh = pca.get_shape().expect("X_pca shape");
    assert_eq!(sh[0], 40);
    assert_eq!(sh[1], 8);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rust_preprocess_write_roundtrip_keeps_symbolic_var_index() {
    if !uv_ok() {
        eprintln!("skip: uv not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "rust_preprocess_write_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h5_in = dir.join("in.h5ad");
    assert!(
        write_h5ad_digit_var_with_feature_name(&h5_in).success(),
        "uv toy h5ad failed"
    );
    let h5_out = dir.join("out.h5ad");

    let params = RustPreprocessParams {
        n_top_hvg: 55,
        n_pca_components: 6,
        ..Default::default()
    };
    let out = rust_preprocess_h5ad_with_steps(
        &h5_in,
        Some(h5_out.as_path()),
        &params,
        &RustPreprocessSteps::UMAP_LAB_PCA_ONLY,
    )
    .expect("rust_preprocess_h5ad_with_steps");
    assert!(out.is_some());
    assert!(h5_out.is_file(), "missing {}", h5_out.display());

    let adata2 = anndata_memory::load_h5ad_fast(&h5_out).expect("reload written h5ad");
    assert_eq!(adata2.n_obs(), 40);
    assert!(adata2.n_vars() <= 55 && adata2.n_vars() > 0);
    for name in adata2.var_names() {
        let t = name.trim();
        let digit_only = !t.is_empty() && t.chars().all(|c| c.is_ascii_digit());
        assert!(!digit_only, "reloaded var name must not be digit-only: {name:?}");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
