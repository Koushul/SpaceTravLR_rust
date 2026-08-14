//! Integration: `rust_preprocess` loads real `.h5ad` from disk (Python-written) and runs the
//! in-memory pipeline — including digit-like `var_names` repair from `var['feature_name']`.

mod common;

use std::process::Command;

use common::uv_python::uv_available;
use spacetravlr::rust_preprocess::{
    RustPreprocessParams, RustPreprocessSteps, rust_preprocess_h5ad_to_memory,
    rust_preprocess_h5ad_with_steps,
};

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
        .arg(py)
        .arg(path.to_str().expect("utf-8 path"))
        .status()
        .expect("spawn uv")
}

#[test]
#[ignore = "requires uv/python (isolated `uv run`); default off — run `cargo test -- --ignored`"]
fn rust_preprocess_memory_repairs_digit_var_index_from_feature_name() {
    if !uv_available() {
        eprintln!("skip: uv not on PATH");
        return;
    }
    let dir =
        std::env::temp_dir().join(format!("rust_preprocess_digit_var_{}", std::process::id()));
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
    let adata =
        rust_preprocess_h5ad_to_memory(&h5, &params, &RustPreprocessSteps::UMAP_LAB_PCA_ONLY)
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
#[ignore = "requires uv/python (isolated `uv run`); default off — run `cargo test -- --ignored`"]
fn rust_preprocess_write_roundtrip_keeps_symbolic_var_index() {
    if !uv_available() {
        eprintln!("skip: uv not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join(format!("rust_preprocess_write_{}", std::process::id()));
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
        assert!(
            !digit_only,
            "reloaded var name must not be digit-only: {name:?}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires uv/python (isolated `uv run`); default off — run `cargo test -- --ignored`"]
fn load_h5ad_tolerates_scanpy_obsp_distances_unsorted_columns() {
    if !uv_available() {
        eprintln!("skip: uv not on PATH");
        return;
    }
    let py = r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad
import scipy.sparse as sp

p = Path(sys.argv[1])
n = 32
rng = np.random.default_rng(0)
x = rng.poisson(2, size=(n, n)).astype(np.float32)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(n)]
a.var_names = [f"g{i}" for i in range(n)]
rows, cols = [], []
data = []
for i in range(n):
    nbrs = rng.choice(n, size=8, replace=False)
    dists = rng.random(8)
    order = np.argsort(dists)
    for j in order:
        rows.append(i)
        cols.append(int(nbrs[j]))
        data.append(float(dists[j]))
dist = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
conn = sp.csr_matrix((np.ones(len(data)), (rows, cols)), shape=(n, n))
conn.sort_indices()
a.obsp["distances"] = dist
a.obsp["connectivities"] = conn
a.write_h5ad(p)
"#;
    let dir = std::env::temp_dir().join(format!("scanpy_obsp_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h5 = dir.join("obsp.h5ad");
    let st = Command::new(common::uv_python::uv_bin())
        .env_remove("PYTHONPATH")
        .env("PYTHONNOUSERSITE", "1")
        .args([
            "run",
            "--isolated",
            "--with",
            "numpy<2",
            "--with",
            "anndata>=0.11",
            "--with",
            "scipy",
        ])
        .arg("python")
        .arg("-c")
        .arg(py)
        .arg(h5.to_str().expect("utf-8"))
        .status()
        .expect("spawn uv");
    assert!(st.success(), "uv scanpy obsp h5ad failed");

    let adata = anndata_memory::load_h5ad_fast(&h5).expect("load h5ad with scanpy obsp");
    assert_eq!(adata.n_obs(), 32);
    let obsp_keys = adata.obsp().keys();
    assert!(obsp_keys.iter().any(|k| k == "distances"));
    assert!(obsp_keys.iter().any(|k| k == "connectivities"));

    let _ = std::fs::remove_dir_all(&dir);
}
