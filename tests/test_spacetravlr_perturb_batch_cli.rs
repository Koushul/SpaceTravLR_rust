use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn perturb_bin() -> PathBuf {
    if let Some(p) = option_env!("CARGO_BIN_EXE_spacetravlr_perturb") {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::var_os("CARGO_BIN_EXE_spacetravlr_perturb") {
        return PathBuf::from(p);
    }
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let mut root = if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR") {
        PathBuf::from(dir)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target")
    };
    root.push(profile);
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

#[test]
fn pair_genes_comma_repeat_broadcast_and_errors() {
    let zipped = spacetravlr::perturb_batch::pair_genes_with_desired_expr(
        &["BCL6,IL21".into()],
        &[2.0, 0.0],
    )
    .unwrap();
    assert_eq!(zipped, vec![("BCL6".into(), 2.0), ("IL21".into(), 0.0)]);

    let repeated = spacetravlr::perturb_batch::pair_genes_with_desired_expr(
        &["BCL6".into(), "IL21".into()],
        &[0.0],
    )
    .unwrap();
    assert_eq!(repeated, vec![("BCL6".into(), 0.0), ("IL21".into(), 0.0)]);

    let mismatch = spacetravlr::perturb_batch::pair_genes_with_desired_expr(
        &["A".into(), "B".into()],
        &[0.0, 1.0, 2.0],
    )
    .unwrap_err()
    .to_string();
    assert!(mismatch.contains("desired-expr"), "{mismatch}");

    let dup =
        spacetravlr::perturb_batch::pair_genes_with_desired_expr(&["A".into(), "A".into()], &[])
            .unwrap_err()
            .to_string();
    assert!(dup.contains("duplicate"), "{dup}");
}

#[test]
fn cli_rejects_mismatched_desired_before_run_toml() {
    let out = Command::new(perturb_bin())
        .args([
            "--run-toml",
            "/nonexistent/run.toml",
            "--gene",
            "A,B",
            "--desired-expr",
            "0,1,2",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("desired-expr"),
        "stderr should reject the length mismatch before loading a run: {stderr}"
    );
    assert!(
        !stderr.contains("No such file") && !stderr.contains("spacetravlr_run_repro"),
        "stderr should not open the run toml: {stderr}"
    );
}

#[test]
fn cli_rejects_duplicate_gene_before_run_toml() {
    let out = Command::new(perturb_bin())
        .args([
            "--run-toml",
            "/nonexistent/run.toml",
            "--gene",
            "A",
            "--gene",
            "A",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("duplicate"), "stderr: {stderr}");
}

#[test]
fn cli_rejects_joint_toml_with_disagreeing_hops_before_run_toml() {
    let dir = temp_batch_dir();
    let batch = dir.join("joint.toml");
    fs::write(
        &batch,
        r#"
joint = true
genes = ["A", "B"]
desired_expr = [0.0, 2.0]
n_propagation = [2, 3]
out = "combo.feather"
"#,
    )
    .unwrap();
    let out = Command::new(perturb_bin())
        .args([
            "--batch-toml",
            batch.to_str().unwrap(),
            "--run-toml",
            "/nonexistent/run.toml",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("n_propagation"),
        "stderr should name n_propagation: {stderr}"
    );
    assert!(
        !stderr.contains("No such file"),
        "stderr should fail before opening the run toml: {stderr}"
    );
}
