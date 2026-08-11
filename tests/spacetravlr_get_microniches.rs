//! `spacetravlr get-microniches` CLI smoke tests.

use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp, AxisArraysOp};
use anndata_hdf5::H5;
use ndarray::Array2;
use polars::prelude::{DataFrame, NamedFrom, Series};
use spacetravlr::write_betadata_feather;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn spacetravlr_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spacetravlr"))
}

fn write_toy_run(dir: &Path) -> anyhow::Result<PathBuf> {
    spacetravlr::ensure_process_env();
    std::fs::create_dir_all(dir)?;
    let h5ad = dir.join("toy.h5ad");
    let a = AnnData::<H5>::new(&h5ad)?;
    let n = 48usize;
    let obs_names: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
    a.set_obs_names(obs_names.clone().into())?;
    a.set_var_names(vec!["G1".into()].into())?;
    let mut cell_types = Vec::with_capacity(n);
    let mut xy = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let niche = i / 16;
        cell_types.push("Alpha".to_string());
        xy[(i, 0)] = (niche as f64) * 8.0 + (i % 16) as f64 * 0.05;
        xy[(i, 1)] = (i % 16) as f64 * 0.05;
    }
    a.set_obs(DataFrame::new(vec![
        Series::new("cell_type".into(), cell_types).into(),
    ])?)?;
    a.obsm().add("spatial", ArrayData::from(xy))?;
    a.set_x(ArrayData::from(Array2::<f64>::from_elem((n, 1), 1.0)))?;
    a.close()?;

    let mut data = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let niche = (i / 16) as f64;
        data[(i, 0)] = niche;
        data[(i, 1)] = 2.0 - niche;
        data[(i, 2)] = ((i * 13) % 5) as f64;
    }
    write_betadata_feather(
        dir.join("GENE1_betadata.feather").to_str().unwrap(),
        "CellID",
        &obs_names,
        &["beta_A".into(), "beta_B".into(), "beta_noise".into()],
        &data,
    )?;

    let repro = dir.join("spacetravlr_run_repro.toml");
    let mut f = std::fs::File::create(&repro)?;
    write!(
        f,
        r#"
[data]
adata_path = "{h5ad}"
layer = "X"
cluster_annot = "cell_type"

[execution]
output_dir = "{out}"
n_parallel = 1
write_minimal_repro_h5ad = false
stale_lock_secs = 0
"#,
        h5ad = h5ad.display(),
        out = dir.display()
    )?;
    Ok(repro)
}

#[test]
fn help_lists_get_microniches() {
    let out = Command::new(spacetravlr_exe())
        .arg("get-microniches")
        .arg("--help")
        .output()
        .expect("spawn help");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("--run-toml"));
    assert!(s.contains("--cell-type"));
    assert!(s.contains("silhouette") || s.contains("resolution"));
}

#[test]
fn cli_writes_microniche_outputs() {
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_get_microniches_cli_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let repro = write_toy_run(&dir).unwrap();
    let out_dir = dir.join("cli_out");
    let out = Command::new(spacetravlr_exe())
        .args([
            "get-microniches",
            "--run-toml",
            repro.to_str().unwrap(),
            "--cell-type",
            "Alpha",
            "--out",
            out_dir.to_str().unwrap(),
            "--moran-n-perm",
            "9",
            "--q-bh-max",
            "0.25",
            "--resolution-min",
            "0.3",
            "--resolution-max",
            "1.0",
            "--resolution-step",
            "0.35",
            "--n-neighbors",
            "8",
            "--n-pcs",
            "2",
        ])
        .output()
        .expect("spawn get-microniches");
    assert!(
        out.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_dir.join("microniche_labels.csv").is_file());
    assert!(out_dir.join("summary.json").is_file());
    assert!(out_dir.join("kept_beta_features.csv").is_file());
    assert!(out_dir.join("resolution_sweep.csv").is_file());
    let summary = std::fs::read_to_string(out_dir.join("summary.json")).unwrap();
    assert!(summary.contains("optimized_by_silhouette"));
    let _ = std::fs::remove_dir_all(&dir);
}
