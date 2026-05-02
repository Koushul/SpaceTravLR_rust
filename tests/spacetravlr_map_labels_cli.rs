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
    let labels_csv = outdir.join("malt_labels.csv");
    assert!(labels_csv.is_file(), "missing {}", labels_csv.display());
    let csv_head = std::fs::read_to_string(&labels_csv).expect("read malt_labels.csv");
    assert!(
        csv_head.starts_with("obs_name,"),
        "expected obs_name index column in csv header, got: {:?}",
        csv_head.lines().next()
    );
    let marker_json = outdir.join("marker_genes.json");
    assert!(marker_json.is_file());

    let labels = spacetravlr::read_h5ad_obs_column_str(&labeled, "malt_label").expect("malt_label");
    assert_eq!(labels.len(), 16, "one label per query cell");
}

#[test]
fn map_labels_spatial_toy_h5ad_with_seed_betadata() {
    if std::env::var_os("SPACETRAVLR_MAP_LABELS_E2E").is_none() {
        eprintln!(
            "skip: set SPACETRAVLR_MAP_LABELS_E2E=1 to run spatial map-labels e2e (uv + torch download)"
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

    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_spatial_map_labels_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let ref_path = dir.join("ref.h5ad");
    let q_path = dir.join("query.h5ad");
    let ref_beta = dir.join("ref_beta");
    let query_beta = dir.join("query_beta");
    let outdir = dir.join("malt_spatial_out");

    let toy = r#"
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

rng = np.random.default_rng(7)
root = Path(sys.argv[1])
ref_path = Path(sys.argv[2])
q_path = Path(sys.argv[3])
ref_beta = Path(sys.argv[4])
query_beta = Path(sys.argv[5])
ref_beta.mkdir()
query_beta.mkdir()

marker_genes = ["M_NEU", "M_MES", "M_EPI"]
spatial_genes = ["NICH_NEU", "NICH_MES", "NICH_EPI"]
genes = marker_genes + spatial_genes + [f"G{i}" for i in range(84)]
cell_types = np.array(["Neural"] * 24 + ["Mesenchyme"] * 24 + ["Epithelial"] * 24, dtype=object)
n_ref = len(cell_types)
x_ref = rng.poisson(2, (n_ref, len(genes))).astype(np.float32)
for gi, mask in [(0, cell_types == "Neural"), (1, cell_types == "Mesenchyme"), (2, cell_types == "Epithelial")]:
    x_ref[mask, gi] += rng.poisson(14, mask.sum()).astype(np.float32)
    x_ref[mask, 3 + gi] += rng.poisson(4, mask.sum()).astype(np.float32)
xy_ref = np.vstack([
    rng.normal([0, 0], 0.25, (24, 2)),
    rng.normal([3, 0], 0.25, (24, 2)),
    rng.normal([0, 3], 0.25, (24, 2)),
]).astype(np.float32)
ref = ad.AnnData(X=x_ref)
ref.obs_names = [f"r{i}" for i in range(n_ref)]
ref.var_names = genes
ref.obs["cell_type"] = cell_types
ref.obsm["spatial"] = xy_ref
ref.write_h5ad(ref_path)

q_labels = np.array(["Neural"] * 10 + ["Mesenchyme"] * 10 + ["Epithelial"] * 10, dtype=object)
n_q = len(q_labels)
x_q = rng.poisson(2, (n_q, len(genes))).astype(np.float32)
for gi, mask in [(0, q_labels == "Neural"), (1, q_labels == "Mesenchyme"), (2, q_labels == "Epithelial")]:
    x_q[mask, gi] += rng.poisson(10, mask.sum()).astype(np.float32)
    x_q[mask, 3 + gi] += rng.poisson(3, mask.sum()).astype(np.float32)
xy_q = np.vstack([
    rng.normal([0.1, 0.0], 0.22, (10, 2)),
    rng.normal([3.1, 0.0], 0.22, (10, 2)),
    rng.normal([0.1, 3.0], 0.22, (10, 2)),
]).astype(np.float32)
q = ad.AnnData(X=x_q)
q.obs_names = [f"q{i}" for i in range(n_q)]
q.var_names = genes
q.obs["truth"] = q_labels
q.obsm["spatial"] = xy_q
q.write_h5ad(q_path)

beta_cols = ["Cluster", "beta0", "beta_TF1", "beta_LIG1_RECA", "beta_LIG2_TFB"]
for gene in spatial_genes:
    rows = []
    for ct in ["Neural", "Mesenchyme", "Epithelial"]:
        v = {"Cluster": ct, "beta0": 0.0, "beta_TF1": 0.0, "beta_LIG1_RECA": 0.0, "beta_LIG2_TFB": 0.0}
        if gene == "NICH_NEU":
            v["beta_TF1"] = 2.0 if ct == "Neural" else -0.5
        elif gene == "NICH_MES":
            v["beta_LIG1_RECA"] = 2.0 if ct == "Mesenchyme" else -0.5
        else:
            v["beta_LIG2_TFB"] = 2.0 if ct == "Epithelial" else -0.5
        rows.append(v)
    pd.DataFrame(rows, columns=beta_cols).to_feather(ref_beta / f"{gene}_betadata.feather")
    qrows = []
    for obs, ct in zip(q.obs_names, q_labels):
        v = {"CellID": obs, "beta0": 0.0, "beta_TF1": 0.0, "beta_LIG1_RECA": 0.0, "beta_LIG2_TFB": 0.0}
        if gene == "NICH_NEU":
            v["beta_TF1"] = 2.0 if ct == "Neural" else -0.5
        elif gene == "NICH_MES":
            v["beta_LIG1_RECA"] = 2.0 if ct == "Mesenchyme" else -0.5
        else:
            v["beta_LIG2_TFB"] = 2.0 if ct == "Epithelial" else -0.5
        qrows.append(v)
    pd.DataFrame(qrows).to_feather(query_beta / f"{gene}_betadata.feather")
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
            "pandas",
            "--with",
            "anndata>=0.11",
            "--with",
            "pyarrow",
        ])
        .arg("python")
        .arg("-c")
        .arg(toy)
        .arg(&dir)
        .arg(&ref_path)
        .arg(&q_path)
        .arg(&ref_beta)
        .arg(&query_beta)
        .status()
        .expect("uv toy spatial");
    assert!(st.success(), "uv toy spatial h5ad failed: {st}");

    let out_bin = spacetravlr_cmd()
        .args([
            "--map-labels",
            "--map-labels-spatial",
            "--reference",
            ref_path.to_str().expect("utf-8"),
            "--query",
            q_path.to_str().expect("utf-8"),
            "--map-labels-outdir",
            outdir.to_str().expect("utf-8"),
            "--map-labels-groupby",
            "cell_type",
            "--map-labels-reference-betadata-dir",
            ref_beta.to_str().expect("utf-8"),
            "--map-labels-query-betadata-dir",
            query_beta.to_str().expect("utf-8"),
            "--map-labels-benchmark-truth",
            "truth",
            "--map-labels-no-leiden",
        ])
        .output()
        .expect("spawn spatial map-labels");
    if !out_bin.status.success() {
        panic!(
            "spatial map-labels failed: {}\nstdout:\n{}\nstderr:\n{}",
            out_bin.status,
            String::from_utf8_lossy(&out_bin.stdout),
            String::from_utf8_lossy(&out_bin.stderr)
        );
    }

    assert!(outdir.join("spatial_malt_training_genes.txt").is_file());
    let run_meta = std::fs::read_to_string(outdir.join("run_meta.json")).expect("run_meta");
    assert!(run_meta.contains("\"spatial\""));
    assert!(run_meta.contains("\"spatial_malt\""));
    let labels = spacetravlr::read_h5ad_obs_column_str(
        &outdir.join("query_labeled.h5ad"),
        "spatial_malt_label",
    )
    .expect("spatial_malt_label");
    assert_eq!(labels.len(), 30);
}
