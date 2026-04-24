use std::path::PathBuf;
use std::process::Command;

fn spacetravlr_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spacetravlr"))
}

fn spacetravlr_cmd() -> Command {
    let mut c = Command::new(spacetravlr_exe());
    c.env("SPACETRAVLR_UV_ALLOW_CACHE", "1");
    c
}

#[test]
fn map_labels_requires_reference_and_query() {
    let out = spacetravlr_cmd()
        .args(["--map-labels", "--query", "/no/ref.h5ad"])
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("reference") || stderr.contains("--reference"),
        "stderr: {stderr}"
    );

    let out2 = spacetravlr_cmd()
        .args(["--map-labels", "--reference", "/no/ref.h5ad"])
        .output()
        .expect("spawn");
    assert!(!out2.status.success());
    let stderr2 = String::from_utf8_lossy(&out2.stderr);
    assert!(
        stderr2.contains("query") || stderr2.contains("--query"),
        "stderr: {stderr2}"
    );
}

#[test]
fn map_labels_end_to_end_toy_h5ad() {
    if std::env::var_os("SPACETRAVLR_MAP_LABELS_E2E").is_none() {
        eprintln!(
            "skip: set SPACETRAVLR_MAP_LABELS_E2E=1 to run map-labels e2e (uv + torch download)"
        );
        return;
    }
    let uv = std::env::var_os("UV_BIN").unwrap_or_else(|| "uv".into());
    if !Command::new(&uv)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        eprintln!("skip: uv not on PATH");
        return;
    }

    let dir = std::env::temp_dir().join(format!("spacetravlr_map_labels_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let ref_path = dir.join("ref.h5ad");
    let q_path = dir.join("query.h5ad");
    let outdir = dir.join("malt_out");
    let ref_str = ref_path.to_str().expect("utf-8");
    let q_str = q_path.to_str().expect("utf-8");

    let toy = r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

rng = np.random.default_rng(42)
n_var = 160
genes = [f"G{i}" for i in range(n_var)]
n_ref = 48
x_ref = rng.poisson(2, (n_ref, n_var)).astype(np.float32)
for i in range(25):
    x_ref[:24, i] += rng.poisson(18, 24).astype(np.float32)
for i in range(90, 125):
    x_ref[24:, i] += rng.poisson(18, 24).astype(np.float32)
ct = np.array(["TypeA"] * 24 + ["TypeB"] * 24, dtype=object)
ref = ad.AnnData(X=x_ref)
ref.obs_names = [f"r{i}" for i in range(n_ref)]
ref.var_names = genes
ref.obs["cell_type"] = ct
ref.write_h5ad(sys.argv[1])

n_q = 16
x_q = rng.poisson(4, (n_q, n_var)).astype(np.float32)
x_q[:, :25] += rng.poisson(6, (n_q, 25)).astype(np.float32)
q = ad.AnnData(X=x_q)
q.obs_names = [f"q{i}" for i in range(n_q)]
q.var_names = genes
q.write_h5ad(sys.argv[2])
"#;

    let st = Command::new(&uv)
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
        .arg(toy)
        .arg(ref_str)
        .arg(q_str)
        .status()
        .expect("uv toy");
    assert!(st.success(), "uv toy h5ad failed: {st}");

    let outdir_str = outdir.to_str().expect("utf-8");
    let out_bin = spacetravlr_cmd()
        .args([
            "--map-labels",
            "--reference",
            ref_str,
            "--query",
            q_str,
            "--map-labels-outdir",
            outdir_str,
            "--map-labels-groupby",
            "cell_type",
        ])
        .output()
        .expect("spawn map-labels");
    if !out_bin.status.success() {
        panic!(
            "map-labels failed: {}\nstdout:\n{}\nstderr:\n{}",
            out_bin.status,
            String::from_utf8_lossy(&out_bin.stdout),
            String::from_utf8_lossy(&out_bin.stderr)
        );
    }

    let labeled = outdir.join("query_labeled.h5ad");
    assert!(labeled.is_file(), "missing {}", labeled.display());
    let marker_json = outdir.join("marker_genes.json");
    assert!(marker_json.is_file());

    let labels = spacetravlr::read_h5ad_obs_column_str(&labeled, "malt_label").expect("malt_label");
    assert_eq!(labels.len(), 16, "one label per query cell");
}
