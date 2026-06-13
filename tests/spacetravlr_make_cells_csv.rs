//! `spacetravlr --make-cells-csv --run-toml` writes a perturbation-ready cells.csv.

use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp};
use anndata_hdf5::H5;
use ndarray::Array2;
use polars::prelude::{DataFrame, NamedFrom, Series};
use spacetravlr::perturb_mode::{parse_obs_columns_csv, write_cells_csv_from_run_toml};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn spacetravlr_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spacetravlr"))
}

fn spacetravlr_cmd() -> Command {
    Command::new(spacetravlr_exe())
}

fn write_toy_h5ad(path: &Path) -> anyhow::Result<()> {
    spacetravlr::ensure_process_env();
    let a = AnnData::<H5>::new(path)?;
    let obs_names: Vec<String> = (0..6).map(|i| format!("cell_{i}")).collect();
    a.set_obs_names(obs_names.into())?;
    a.set_var_names(vec!["G1".into(), "G2".into()].into())?;
    let cell_types: Vec<String> = vec![
        "Alpha".into(),
        "Alpha".into(),
        "Beta".into(),
        "Beta".into(),
        "Gamma".into(),
        "Gamma".into(),
    ];
    let obs = DataFrame::new(vec![Series::new("cell_type".into(), cell_types).into()])?;
    a.set_obs(obs)?;
    let x = Array2::<f64>::from_elem((6, 2), 1.0);
    a.set_x(ArrayData::from(x))?;
    a.close()?;
    Ok(())
}

fn write_run_repro_toml(dir: &Path, h5ad: &Path) -> PathBuf {
    let repro = dir.join("spacetravlr_run_repro.toml");
    let mut f = std::fs::File::create(&repro).unwrap();
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
    )
    .unwrap();
    repro
}

#[test]
fn help_lists_make_cells_csv() {
    let out = spacetravlr_cmd()
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
        s.contains("--make-cells-csv"),
        "expected --make-cells-csv in help:\n{s}"
    );
    assert!(
        s.contains("--run-toml"),
        "expected --run-toml in help:\n{s}"
    );
}

#[test]
fn make_cells_csv_requires_run_toml() {
    let out = spacetravlr_cmd()
        .arg("--make-cells-csv")
        .output()
        .expect("spawn spacetravlr --make-cells-csv");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--run-toml"),
        "expected missing --run-toml error:\n{stderr}"
    );
}

#[test]
fn make_cells_csv_cli_writes_grouped_obs_names() {
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_make_cells_csv_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let h5ad = dir.join("toy.h5ad");
    write_toy_h5ad(&h5ad).unwrap();
    let repro = write_run_repro_toml(&dir, &h5ad);
    let expected_csv = dir.join("cells.csv");

    let out = spacetravlr_cmd()
        .arg("--make-cells-csv")
        .arg("--run-toml")
        .arg(&repro)
        .output()
        .expect("spawn spacetravlr --make-cells-csv");
    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(printed, expected_csv.display().to_string());
    assert!(expected_csv.is_file(), "missing {}", expected_csv.display());

    let obs: Vec<String> = (0..6).map(|i| format!("cell_{i}")).collect();
    let parsed = parse_obs_columns_csv(&expected_csv, &obs).unwrap();
    assert_eq!(
        parsed.indices_for_column("Alpha").unwrap(),
        &[0usize, 1]
    );
    assert_eq!(
        parsed.indices_for_column("Beta").unwrap(),
        &[2usize, 3]
    );
    assert_eq!(
        parsed.indices_for_column("Gamma").unwrap(),
        &[4usize, 5]
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn write_cells_csv_from_run_toml_library_matches_cli() {
    spacetravlr::ensure_process_env();
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_make_cells_lib_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let h5ad = dir.join("toy.h5ad");
    write_toy_h5ad(&h5ad).unwrap();
    let repro = write_run_repro_toml(&dir, &h5ad);

    let out = write_cells_csv_from_run_toml(repro.as_path(), None).unwrap();
    assert_eq!(out, dir.join("cells.csv"));

    let body = std::fs::read_to_string(&out).unwrap();
    assert!(body.starts_with("Alpha,Beta,Gamma\n"));
    assert!(body.contains("cell_0"));
    assert!(body.contains("cell_5"));

    let obs: Vec<String> = (0..6).map(|i| format!("cell_{i}")).collect();
    let parsed = parse_obs_columns_csv(&out, &obs).unwrap();
    assert_eq!(parsed.column_names, vec!["Alpha", "Beta", "Gamma"]);

    let _ = std::fs::remove_dir_all(&dir);
}
