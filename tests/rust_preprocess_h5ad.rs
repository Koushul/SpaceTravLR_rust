//! Integration: `rust_preprocess` loads real `.h5ad` from disk (Python-written) and runs the
//! in-memory pipeline — including digit-like `var_names` repair from `var['feature_name']`.

mod common;

use std::process::Command;

use common::uv_python::require_uv;
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
    require_uv();
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
    require_uv();
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
    require_uv();
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

fn recompress_f64_dataset_lzf(group: &hdf5_metno::Group, name: &str) {
    use std::str::FromStr;
    use hdf5_metno::types::{VarLenAscii, VarLenUnicode};
    let ds = group.dataset(name).expect(name);
    let shape = ds.shape();
    let values: Vec<f64> = ds.read_raw().unwrap_or_else(|e| panic!("read {name}: {e}"));
    let mut attrs = Vec::new();
    for attr_name in ds.attr_names().expect("attr names") {
        let attr = ds.attr(&attr_name).expect("attr");
        if let Ok(v) = attr.read_scalar::<VarLenUnicode>() {
            attrs.push((attr_name, v.to_string()));
        } else if let Ok(v) = attr.read_scalar::<VarLenAscii>() {
            attrs.push((attr_name, v.to_string()));
        }
    }
    drop(ds);
    group.unlink(name).expect("unlink");
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values).expect("shape");
    let created = group
        .new_dataset_builder()
        .with_data(&arr)
        .lzf()
        .create(name)
        .expect("lzf dataset");
    assert!(
        created.filters().iter().any(|f| f.id() == 32000),
        "filter id on {name}: {:?}",
        created.filters()
    );
    for (attr_name, value) in attrs {
        let unicode = VarLenUnicode::from_str(&value).expect("unicode attr");
        created
            .new_attr::<VarLenUnicode>()
            .create(attr_name.as_str())
            .expect("attr")
            .write_scalar(&unicode)
            .expect("write attr");
    }
}

#[test]
fn hdf5_lzf_roundtrip_numeric_chunk() {
    assert!(hdf5_metno::filters::lzf_available());
    let path = std::env::temp_dir().join(format!("spacetravlr_lzf_rt_{}.h5", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let data: Vec<f64> = (0..4096).map(|i| (i % 17) as f64).collect();
    {
        let file = hdf5_metno::File::create(&path).expect("create");
        let ds = file
            .new_dataset_builder()
            .with_data(&data)
            .lzf()
            .create("x_centroid")
            .expect("write lzf");
        assert!(
            ds.filters().iter().any(|f| f.id() == 32000),
            "filters: {:?}",
            ds.filters()
        );
        file.close().ok();
    }
    let file = hdf5_metno::File::open(&path).expect("reopen");
    let got: Vec<f64> = file
        .dataset("x_centroid")
        .expect("dataset")
        .read_raw()
        .expect("read");
    assert_eq!(got, data);
    file.close().ok();
    let _ = std::fs::remove_file(&path);
}

fn write_small_h5ad(path: &std::path::Path, with_obsp: bool) {
    use anndata::data::ArrayData;
    use anndata::{AnnData, AnnDataOp};
    use anndata_hdf5::H5;
    use ndarray::Array2;
    use polars::prelude::{DataFrame, NamedFrom, Series};

    spacetravlr::ensure_process_env();
    let n = 24usize;
    let n_vars = 12usize;
    let a = AnnData::<H5>::new(path).expect("create h5ad");
    let obs_names: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
    let var_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();
    a.set_obs_names(obs_names.into()).expect("obs names");
    a.set_var_names(var_names.into()).expect("var names");
    let x = Array2::<f64>::from_shape_fn((n, n_vars), |(i, j)| ((i * 3 + j) % 11) as f64);
    a.set_x(ArrayData::from(x)).expect("set x");
    let centroids: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let obs = DataFrame::new(vec![Series::new("x_centroid".into(), centroids).into()]).expect("obs");
    a.set_obs(obs).expect("set obs");
    if with_obsp {
        let dist = Array2::<f64>::eye(n);
        a.set_obsp([("distances".to_string(), ArrayData::from(dist))])
            .expect("set obsp");
    }
    a.close().expect("close");
}

#[test]
fn load_h5ad_fast_reads_lzf_x_and_obs_centroid() {
    let path = std::env::temp_dir().join(format!(
        "spacetravlr_lzf_h5ad_{}.h5ad",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    write_small_h5ad(&path, false);
    {
        let file = hdf5_metno::File::open_rw(&path).expect("rw");
        recompress_f64_dataset_lzf(&file, "X");
        let obs = file.group("obs").expect("obs");
        recompress_f64_dataset_lzf(&obs, "x_centroid");
        file.close().ok();
    }
    let adata = anndata_memory::load_h5ad_fast(&path).expect("load lzf h5ad");
    assert_eq!(adata.n_obs(), 24);
    assert_eq!(adata.n_vars(), 12);
    let obs = adata.obs().get_data();
    let col = obs.column("x_centroid").expect("x_centroid");
    assert_eq!(col.len(), 24);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn preprocess_leaves_source_obsp_in_place() {
    use spacetravlr::rust_preprocess::testing_begin_preprocess_timing_capture;
    use spacetravlr::rust_preprocess::testing_take_preprocess_timing_capture;
    use spacetravlr::rust_preprocess::PREPROCESS_TIMING_LABELS;

    let path = std::env::temp_dir().join(format!(
        "spacetravlr_obsp_keep_{}.h5ad",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    write_small_h5ad(&path, true);
    let params = RustPreprocessParams {
        n_top_hvg: 8,
        n_pca_components: 4,
        ..Default::default()
    };
    testing_begin_preprocess_timing_capture();
    rust_preprocess_h5ad_to_memory(&path, &params, &RustPreprocessSteps::UMAP_LAB_PCA_ONLY)
        .expect("preprocess");
    let summary = testing_take_preprocess_timing_capture().expect("timing summary");
    for label in PREPROCESS_TIMING_LABELS {
        assert!(
            summary.contains(&format!("{label}:")),
            "missing {label} in:\n{summary}"
        );
    }
    assert!(summary.contains("QC: not run"));
    assert!(summary.contains("Scale: not run"));
    let file = hdf5_metno::File::open(&path).expect("reopen source");
    assert!(file.link_exists("obsp"), "source obsp was removed");
    file.close().ok();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn compress_h5ad_makes_x_and_layers_gzip_csr() {
    use anndata::data::ArrayData;
    use anndata::{AnnData, AnnDataOp, AxisArraysOp, Backend};
    use anndata_hdf5::H5;
    use hdf5_metno::filters::Filter;
    use ndarray::Array2;

    let path = std::env::temp_dir().join(format!(
        "spacetravlr_compress_{}.h5ad",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    write_small_h5ad(&path, true);
    {
        let a = AnnData::<H5>::open(H5::open_rw(&path).expect("rw")).expect("open");
        let layer = Array2::<f64>::from_elem((24, 12), 1.0);
        a.layers()
            .add("counts", ArrayData::from(layer))
            .expect("layer");
        a.close().ok();
    }
    spacetravlr::compress_h5ad_inplace(&path).expect("compress");
    let file = hdf5_metno::File::open(&path).expect("reopen");
    assert!(file.link_exists("obsp"), "obsp copied");
    let x = file.group("X").expect("X group");
    let enc = x
        .attr("encoding-type")
        .expect("encoding")
        .read_scalar::<hdf5_metno::types::VarLenUnicode>()
        .expect("encoding str");
    assert_eq!(enc.to_string(), "csr_matrix");
    let filters = x.dataset("data").expect("X/data").filters();
    assert!(
        filters.iter().any(|f| matches!(f, Filter::Deflate(_))),
        "X/data filters: {filters:?}"
    );
    let counts = file.group("layers/counts").expect("layer");
    let layer_filters = counts.dataset("data").expect("layer data").filters();
    assert!(
        layer_filters.iter().any(|f| matches!(f, Filter::Deflate(_))),
        "layer filters: {layer_filters:?}"
    );
    file.close().ok();
    let adata = anndata_memory::load_h5ad_fast(&path).expect("reload");
    assert_eq!(adata.n_obs(), 24);
    assert_eq!(adata.n_vars(), 12);
    assert!(adata.layers().keys().iter().any(|k| k == "counts"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn failed_load_still_prints_load_and_total() {
    use spacetravlr::rust_preprocess::testing_begin_preprocess_timing_capture;
    use spacetravlr::rust_preprocess::testing_take_preprocess_timing_capture;

    let path = std::env::temp_dir().join(format!(
        "spacetravlr_missing_{}.h5ad",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    testing_begin_preprocess_timing_capture();
    let err = match rust_preprocess_h5ad_to_memory(
        &path,
        &RustPreprocessParams::default(),
        &RustPreprocessSteps::UMAP_LAB_PCA_ONLY,
    ) {
        Err(e) => e,
        Ok(_) => panic!("expected missing file to fail"),
    };
    let summary = testing_take_preprocess_timing_capture().expect("timing summary");
    assert!(
        summary.contains("Load:") && !summary.contains("Load: not run"),
        "{summary}\nerr: {err:#}"
    );
    assert!(
        summary.contains("Total:") && !summary.contains("Total: not run"),
        "{summary}"
    );
}
