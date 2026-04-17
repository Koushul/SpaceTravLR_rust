use std::path::PathBuf;
use std::process::{Command, ExitStatus};

fn uv_status_retry_no_cache(mut build: impl FnMut(bool) -> Command) -> std::io::Result<ExitStatus> {
    let s = build(false).status()?;
    if s.success() {
        return Ok(s);
    }
    build(true).status()
}

fn spacetravlr_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spacetravlr"))
}

fn spacetravlr_cmd() -> Command {
    let mut c = Command::new(spacetravlr_exe());
    c.env("SPACETRAVLR_UV_ALLOW_CACHE", "1");
    c
}

#[test]
fn help_lists_process_h5ad_and_impute() {
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
        s.contains("--process-h5ad"),
        "expected --process-h5ad in help:\n{s}"
    );
    assert!(s.contains("--impute"), "expected --impute in help:\n{s}");
    assert!(
        s.contains("--process-output-dir"),
        "expected --process-output-dir in help:\n{s}"
    );
    assert!(
        s.contains("--magic-batch-obs"),
        "expected --magic-batch-obs in help:\n{s}"
    );
}

#[test]
fn process_h5ad_requires_h5ad_flag() {
    let out = spacetravlr_cmd()
        .arg("--process-h5ad")
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--h5ad") || stderr.contains("h5ad"),
        "stderr: {stderr}"
    );
}

#[test]
fn process_h5ad_errors_on_missing_file() {
    let out = spacetravlr_cmd()
        .args([
            "--process-h5ad",
            "--h5ad",
            "/no/such/spacetravlr_cli_missing.h5ad",
        ])
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("AnnData"),
        "stderr: {stderr}"
    );
}

#[test]
fn process_h5ad_hidden_alias_still_works() {
    let out = spacetravlr_cmd()
        .arg("--process_h5ad")
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--h5ad") || stderr.contains("h5ad"),
        "stderr: {stderr}"
    );
}

#[test]
fn process_h5ad_end_to_end_writes_processed_sibling() {
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
        "spacetravlr_cli_process_h5ad_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let in_path = dir.join("toy_cli.h5ad");
    let in_str = in_path.to_str().expect("utf-8 path");

    let toy_status = uv_status_retry_no_cache(|no_cache| {
        let mut c = Command::new(&uv);
        c.env_remove("PYTHONPATH").env("PYTHONNOUSERSITE", "1");
        if no_cache {
            c.arg("--no-cache");
        }
        c.args([
            "run",
            "--isolated",
            "--with",
            "numpy<2",
            "--with",
            "anndata>=0.11",
        ])
        .arg("python")
        .arg("-c")
        .arg(
            r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

p = Path(sys.argv[1])
n_obs, n_var = 80, 800
rng = np.random.default_rng(42)
x = np.full((n_obs, n_var), 20.0, dtype=np.float32)
x += rng.normal(0.0, 1.5, size=x.shape).astype(np.float32)
x = np.clip(x, 0.0, None)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"G{i}" for i in range(n_var)]
a.write_h5ad(p)
"#,
        )
        .arg(in_str);
        c
    })
    .expect("uv toy h5ad");
    assert!(toy_status.success(), "uv toy h5ad failed: {toy_status}");

    let dir_str = dir.to_str().expect("utf-8 dir");
    let out_bin = spacetravlr_cmd()
        .args([
            "--process-h5ad",
            "--h5ad",
            in_str,
            "--spatial-species",
            "human",
            "--process-output-dir",
            dir_str,
        ])
        .output()
        .expect("spawn spacetravlr --process-h5ad");
    if !out_bin.status.success() {
        panic!(
            "spacetravlr --process-h5ad failed: {}\nstdout:\n{}\nstderr:\n{}",
            out_bin.status,
            String::from_utf8_lossy(&out_bin.stdout),
            String::from_utf8_lossy(&out_bin.stderr)
        );
    }

    let expected = dir.join(format!(
        "{}_processed.h5ad",
        in_path.file_stem().unwrap().to_str().unwrap()
    ));
    assert!(expected.is_file(), "missing {}", expected.display());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn impute_writes_imputed_sibling_after_process_h5ad() {
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

    let dir = std::env::temp_dir().join(format!("spacetravlr_cli_impute_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let raw_path = dir.join("chain.h5ad");
    let raw_str = raw_path.to_str().expect("utf-8");

    let toy_status = uv_status_retry_no_cache(|no_cache| {
        let mut c = Command::new(&uv);
        c.env_remove("PYTHONPATH").env("PYTHONNOUSERSITE", "1");
        if no_cache {
            c.arg("--no-cache");
        }
        c.args([
            "run",
            "--isolated",
            "--with",
            "numpy<2",
            "--with",
            "anndata>=0.11",
        ])
        .arg("python")
        .arg("-c")
        .arg(
            r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

p = Path(sys.argv[1])
n_obs, n_var = 80, 800
rng = np.random.default_rng(99)
x = np.full((n_obs, n_var), 22.0, dtype=np.float32)
x += rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
x = np.clip(x, 0.0, None)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"G{i}" for i in range(n_var)]
a.write_h5ad(p)
"#,
        )
        .arg(raw_str);
        c
    })
    .expect("uv toy");
    assert!(toy_status.success(), "uv toy failed: {toy_status}");

    let dir_str = dir.to_str().expect("utf-8");
    let proc_out = spacetravlr_cmd()
        .args([
            "--process-h5ad",
            "--h5ad",
            raw_str,
            "--spatial-species",
            "human",
            "--process-output-dir",
            dir_str,
        ])
        .output()
        .expect("process-h5ad");
    assert!(
        proc_out.status.success(),
        "process-h5ad failed: {}",
        String::from_utf8_lossy(&proc_out.stderr)
    );

    let processed = dir.join("chain_processed.h5ad");
    assert!(processed.is_file(), "missing {}", processed.display());

    let imp_out = spacetravlr_cmd()
        .args([
            "--impute",
            "--h5ad",
            processed.to_str().unwrap(),
            "--process-output-dir",
            dir_str,
        ])
        .output()
        .expect("impute");
    assert!(
        imp_out.status.success(),
        "impute failed: {}",
        String::from_utf8_lossy(&imp_out.stderr)
    );

    let imputed = dir.join("chain_processed_imputed.h5ad");
    assert!(imputed.is_file(), "missing {}", imputed.display());

    let _ = std::fs::remove_dir_all(&dir);
}
