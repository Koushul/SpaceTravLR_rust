use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn perturb_bin() -> PathBuf {
    if let Some(p) = std::env::var_os("CARGO_BIN_EXE_spacetravlr_perturb") {
        return PathBuf::from(p);
    }
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR") {
        root = PathBuf::from(dir);
    } else if cfg!(debug_assertions) {
        root.push("target/debug");
    } else {
        root.push("target/release");
    }
    root.join("spacetravlr-perturb")
}

#[test]
fn batch_toml_rejects_with_gene_flag() {
    let dir = temp_batch_dir();
    let batch = dir.join("batch.toml");
    fs::write(
        &batch,
        r#"genes = ["SOX2"]
out_dir = "o"
"#,
    )
    .unwrap();
    let out = Command::new(perturb_bin())
        .args([
            "--batch-toml",
            batch.to_str().unwrap(),
            "--run-toml",
            "/nonexistent/run.toml",
            "--gene",
            "SOX2",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot be used in batch mode")
            || stderr.contains("cannot be used with --batch-toml")
            || stderr.contains("--batch-toml"),
        "stderr: {stderr}"
    );
}

#[test]
fn batch_toml_parse_error_is_nonzero() {
    let dir = temp_batch_dir();
    let bad = dir.join("bad.toml");
    fs::write(&bad, "this is not valid toml [[[").unwrap();
    let out = Command::new(perturb_bin())
        .args([
            "--batch-toml",
            bad.to_str().unwrap(),
            "--run-toml",
            "/nonexistent/run.toml",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

fn temp_batch_dir() -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "stlr_perturb_batch_cli_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}
