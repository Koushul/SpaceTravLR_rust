use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp};
use anndata_hdf5::H5;
use ndarray::Array2;
use polars::prelude::{DataFrame, NamedFrom, ParquetWriter, SerReader, Series};
use std::path::PathBuf;
use std::process::Command;

fn spacetravlr_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spacetravlr"))
}

fn write_minimal_mouse_grn_parquet(dir: &std::path::Path) -> anyhow::Result<()> {
    let mut df = DataFrame::new(vec![
        Series::new("source".into(), vec!["Reg1", "Tgt1", "Tgt1", "Reg2"]).into(),
        Series::new("target".into(), vec!["Tgt1", "Reg1", "Reg2", "Tgt1"]).into(),
        Series::new("edge_type".into(), vec!["grn", "grn", "grn", "grn"]).into(),
        Series::new("weight".into(), vec![1.0_f64, 1.0, 1.0, 1.0]).into(),
    ])?;
    let path = dir.join("mouse_network.parquet");
    let f = std::fs::File::create(&path)?;
    ParquetWriter::new(f).finish(&mut df)?;
    Ok(())
}

fn write_tiny_mouse_h5ad(path: &std::path::Path) -> anyhow::Result<()> {
    let a = AnnData::<H5>::new(path)?;
    a.set_obs_names(vec!["c0".into(), "c1".into(), "c2".into()].into())?;
    a.set_var_names(vec!["Reg1".into(), "Tgt1".into(), "Reg2".into()].into())?;
    let obs = DataFrame::new(vec![
        Series::new(
            "cell_type".into(),
            vec!["t".to_string(), "t".to_string(), "t".to_string()],
        )
        .into(),
    ])?;
    a.set_obs(obs)?;
    let var = DataFrame::new(vec![
        Series::new("gene_ids".into(), vec!["Reg1", "Tgt1", "Reg2"]).into(),
    ])?;
    a.set_var(var)?;
    let mut gem = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        let t = i as f64 * 0.2;
        gem[[i, 0]] = 0.5 + t.sin();
        gem[[i, 2]] = 0.3 + t.cos();
        gem[[i, 1]] = 0.2 * gem[[i, 0]] + 0.3 * gem[[i, 2]] + 0.01 * (i as f64);
    }
    a.set_x(ArrayData::from(gem))?;
    a.close()?;
    Ok(())
}

#[test]
fn help_lists_celloracle() {
    let out = Command::new(spacetravlr_exe())
        .arg("--help")
        .output()
        .expect("spawn spacetravlr --help");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        s.contains("--celloracle"),
        "expected --celloracle in help:\n{s}"
    );
    assert!(
        s.contains("--celloracle-output"),
        "expected --celloracle-output in help:\n{s}"
    );
}

#[test]
fn celloracle_writes_feather() {
    let dir =
        std::env::temp_dir().join(format!("spacetravlr_celloracle_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_minimal_mouse_grn_parquet(&dir).unwrap();
    let h5ad = dir.join("tiny.h5ad");
    write_tiny_mouse_h5ad(&h5ad).unwrap();
    let out_feather = dir.join("priors.feather");

    let status = Command::new(spacetravlr_exe())
        .args([
            "--celloracle",
            h5ad.to_str().unwrap(),
            "--celloracle-skip-preprocess",
            "--celloracle-layer",
            "X",
            "--celloracle-species",
            "mouse",
            "--celloracle-network-data-dir",
            dir.to_str().unwrap(),
            "--celloracle-output",
            out_feather.to_str().unwrap(),
        ])
        .status()
        .expect("spawn");
    assert!(status.success(), "celloracle CLI failed");

    assert!(out_feather.is_file(), "feather not written");
    let df = polars::prelude::IpcReader::new(std::fs::File::open(&out_feather).unwrap())
        .finish()
        .unwrap();
    let cols = df.get_column_names();
    let has = |name: &str| cols.iter().any(|c| c.as_str() == name);
    assert!(
        has("source") && has("target") && has("cell_type"),
        "unexpected columns: {cols:?}"
    );
    assert!(df.height() > 0, "empty feather");

    let _ = std::fs::remove_dir_all(&dir);
}
