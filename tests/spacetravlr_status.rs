use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn spacetravlr_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_spacetravlr"))
}

fn spacetravlr_cmd() -> Command {
    Command::new(spacetravlr_exe())
}

static SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_dir() -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("spacetravlr_status_cli_{}_{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

fn lock_body(host: &str, pid: u32, n_parallel: usize) -> String {
    format!("spacetravlr_gene 1\nhost={host}\npid={pid}\nn_parallel={n_parallel}\nstarted_unix=1\n")
}

fn status_output(arg: impl AsRef<Path>) -> (bool, String, String) {
    let out = spacetravlr_cmd()
        .arg("--status")
        .arg(arg.as_ref())
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn spacetravlr --status");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn status_ok(dir: &Path) -> String {
    let (ok, stdout, stderr) = status_output(dir);
    assert!(ok, "stdout:\n{stdout}\nstderr:\n{stderr}");
    stdout
}

#[test]
fn help_lists_status() {
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
    assert!(s.contains("--status"), "expected --status in help:\n{s}");
}

#[test]
fn status_missing_dir_fails() {
    let (ok, _, stderr) = status_output("/no/such/spacetravlr_status_dir");
    assert!(!ok);
    assert!(
        stderr.contains("not found") || stderr.contains("status"),
        "stderr:\n{stderr}"
    );
}

#[test]
fn status_rejects_plain_file() {
    let dir = tmp_dir();
    let f = dir.join("readme.txt");
    fs::write(&f, "not a run").unwrap();
    let (ok, _, stderr) = status_output(&f);
    assert!(!ok, "expected failure for plain file");
    assert!(
        stderr.contains("expected a training output directory"),
        "stderr:\n{stderr}"
    );
    cleanup(&dir);
}

#[test]
fn status_cannot_combine_with_subcommand() {
    let dir = tmp_dir();
    let out = spacetravlr_cmd()
        .arg("--status")
        .arg(&dir)
        .arg("run-summary")
        .output()
        .expect("spawn spacetravlr --status + subcommand");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot be combined with a subcommand"),
        "stderr:\n{stderr}"
    );
    cleanup(&dir);
}

#[test]
fn status_empty_dir_warns() {
    let dir = tmp_dir();
    let stdout = status_ok(&dir);
    assert!(
        stdout.contains("may not be a training output directory"),
        "stdout:\n{stdout}"
    );
    cleanup(&dir);
}

#[test]
fn status_in_progress_with_lock_payload() {
    let dir = tmp_dir();
    fs::write(
        dir.join("spacetravlr_target_genes.txt"),
        "SOX2\nACTB\nMALAT1\n",
    )
    .unwrap();
    fs::write(
        dir.join("spacetravlr_run_repro.toml"),
        "[training]\nmode = \"full\"\nepochs = 8\n",
    )
    .unwrap();
    fs::write(dir.join("SOX2_betadata.feather"), vec![0u8; 4096]).unwrap();
    fs::write(dir.join("ACTB.lock"), lock_body("testhost", 4242, 8)).unwrap();

    let stdout = status_ok(&dir);
    assert!(stdout.contains("1/3"), "progress 1/3:\n{stdout}");
    assert!(stdout.contains("testhost"), "lock host:\n{stdout}");
    assert!(stdout.contains("n_parallel=8"), "n_parallel:\n{stdout}");
    assert!(stdout.contains("In flight"), "in flight:\n{stdout}");
    assert!(
        stdout.contains("1 genes") || stdout.contains("In flight"),
        "{stdout}"
    );
    assert!(stdout.contains("Remaining"), "{stdout}");
    assert!(stdout.contains("Feathers"), "{stdout}");
    assert!(stdout.contains("KiB") || stdout.contains("B"), "{stdout}");
    assert!(stdout.contains("Last feather"), "{stdout}");
    assert!(stdout.contains("SOX2_betadata.feather"), "{stdout}");
    assert!(stdout.contains("Last lock"), "{stdout}");
    assert!(stdout.contains("ACTB.lock"), "{stdout}");

    let stdout2 = status_ok(&dir.join("spacetravlr_run_repro.toml"));
    assert!(stdout2.contains("1/3"), "repro path:\n{stdout2}");
    cleanup(&dir);
}

#[test]
fn status_complete_run() {
    let dir = tmp_dir();
    fs::write(dir.join("spacetravlr_target_genes.txt"), "A\nB\nC\n").unwrap();
    fs::write(dir.join("A_betadata.feather"), vec![1u8; 100]).unwrap();
    fs::write(dir.join("B_betadata.feather"), vec![1u8; 100]).unwrap();
    fs::write(dir.join("C.orphan"), b"").unwrap();
    let stdout = status_ok(&dir);
    assert!(stdout.contains("3/3"), "complete:\n{stdout}");
    assert!(stdout.contains("100.0%"), "{stdout}");
    assert!(
        stdout.contains("In flight") && stdout.contains("0 genes"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Remaining") && stdout.contains("0"),
        "{stdout}"
    );
    cleanup(&dir);
}

#[test]
fn status_join_workers_two_hosts() {
    let dir = tmp_dir();
    fs::write(
        dir.join("spacetravlr_target_genes.txt"),
        "G1\nG2\nG3\nG4\nG5\n",
    )
    .unwrap();
    fs::write(dir.join("G1_betadata.feather"), b"done").unwrap();
    fs::write(dir.join("G2.lock"), lock_body("hostA", 100, 8)).unwrap();
    fs::write(dir.join("G3.lock"), lock_body("hostA", 100, 8)).unwrap();
    fs::write(dir.join("G4.lock"), lock_body("hostB", 200, 4)).unwrap();
    fs::write(dir.join("LEGACY.lock"), b"").unwrap();

    let stdout = status_ok(&dir);
    assert!(stdout.contains("hostA"), "{stdout}");
    assert!(stdout.contains("hostB"), "{stdout}");
    assert!(stdout.contains("12 worker slots"), "{stdout}");
    assert!(stdout.contains("unlabelled lock"), "{stdout}");
    assert!(stdout.contains("n_parallel=8"), "{stdout}");
    assert!(stdout.contains("n_parallel=4"), "{stdout}");
    cleanup(&dir);
}

#[test]
fn status_condition_split() {
    let dir = tmp_dir();
    fs::write(
        dir.join("spacetravlr_run_repro.toml"),
        "[training]\ngenes = [\"G1\", \"G2\"]\nmode = \"full\"\n\n[data]\ncondition = \"batch\"\n",
    )
    .unwrap();
    let a = dir.join("conditions/groupA");
    let b = dir.join("conditions/groupB");
    fs::create_dir_all(&a).unwrap();
    fs::create_dir_all(&b).unwrap();
    fs::write(a.join("condition_label.txt"), "tumor").unwrap();
    fs::write(b.join("condition_label.txt"), "normal").unwrap();
    fs::write(a.join("spacetravlr_target_genes.txt"), "G1\nG2\n").unwrap();
    fs::write(b.join("spacetravlr_target_genes.txt"), "G1\nG2\n").unwrap();
    fs::write(a.join("G1_betadata.feather"), vec![0u8; 50]).unwrap();
    fs::write(a.join("G2.lock"), lock_body("condhost", 9, 2)).unwrap();
    fs::write(b.join("G1_betadata.feather"), vec![0u8; 50]).unwrap();

    let stdout = status_ok(&dir);
    assert!(stdout.contains("Conditions"), "{stdout}");
    assert!(stdout.contains("tumor"), "{stdout}");
    assert!(stdout.contains("normal"), "{stdout}");
    assert!(stdout.contains("2 groups"), "{stdout}");
    assert!(stdout.contains("condhost"), "{stdout}");
    cleanup(&dir);
}

#[test]
fn status_pool_lasso_parent_locks() {
    let dir = tmp_dir();
    fs::write(
        dir.join("spacetravlr_run_repro.toml"),
        "[training]\npool_lasso = true\ngenes = [\"G1\", \"G2\"]\n",
    )
    .unwrap();
    fs::write(dir.join("spacetravlr_target_genes.txt"), "G1\nG2\n").unwrap();
    fs::write(dir.join("G1.done"), b"").unwrap();
    fs::write(dir.join("G2.lock"), lock_body("poolhost", 3, 2)).unwrap();
    let s1 = dir.join("conditions/slide1");
    let s2 = dir.join("conditions/slide2");
    fs::create_dir_all(&s1).unwrap();
    fs::create_dir_all(&s2).unwrap();
    fs::write(s1.join("condition_label.txt"), "slide1").unwrap();
    fs::write(s2.join("condition_label.txt"), "slide2").unwrap();
    fs::write(s1.join("G1_betadata.feather"), vec![7u8; 80]).unwrap();
    fs::write(s2.join("G1_betadata.feather"), vec![7u8; 80]).unwrap();

    let stdout = status_ok(&dir);
    assert!(
        stdout.contains("1/2"),
        "one parent .done of two genes:\n{stdout}"
    );
    assert!(stdout.contains("pool-lasso"), "{stdout}");
    assert!(stdout.contains("poolhost"), "{stdout}");
    assert!(stdout.contains("Samples"), "{stdout}");
    assert!(stdout.contains("slide1"), "{stdout}");
    assert!(stdout.contains("slide2"), "{stdout}");
    assert!(
        !stdout.contains("Conditions"),
        "pool-lasso should list Samples, not condition groups:\n{stdout}"
    );
    assert!(stdout.contains("2 files"), "two sample feathers:\n{stdout}");

    let stdout_from_sample = status_ok(&s1);
    assert!(
        stdout_from_sample.contains("1/2"),
        "status from sample dir should walk up:\n{stdout_from_sample}"
    );
    cleanup(&dir);
}

#[test]
fn status_setup_preparing() {
    let dir = tmp_dir();
    fs::write(
        dir.join("spacetravlr_setup.flock"),
        b"spacetravlr_setup 1\n",
    )
    .unwrap();
    let stdout = status_ok(&dir);
    assert!(stdout.contains("still preparing"), "{stdout}");
    cleanup(&dir);
}

#[test]
fn status_setup_ready_and_orphans() {
    let dir = tmp_dir();
    fs::write(dir.join("spacetravlr_setup.ready"), b"ok\n").unwrap();
    fs::write(dir.join("spacetravlr_target_genes.txt"), "A\nB\nC\n").unwrap();
    fs::write(dir.join("A.orphan"), b"").unwrap();
    fs::write(dir.join("B.tf_ablated"), b"").unwrap();
    fs::write(dir.join("C_betadata.feather"), b"ccc").unwrap();
    fs::create_dir_all(dir.join("log")).unwrap();
    fs::write(dir.join("log/A.log"), b"log").unwrap();
    let stdout = status_ok(&dir);
    assert!(stdout.contains("3/3"), "{stdout}");
    assert!(stdout.contains("orphans"), "{stdout}");
    assert!(stdout.contains("tf_ablated"), "{stdout}");
    assert!(
        stdout.contains("Setup") && stdout.contains("ready"),
        "{stdout}"
    );
    cleanup(&dir);
}

#[test]
fn status_unknown_remaining_without_gene_list() {
    let dir = tmp_dir();
    fs::write(dir.join("SOX2_betadata.feather"), b"abc").unwrap();
    let stdout = status_ok(&dir);
    assert!(stdout.contains("unknown"), "{stdout}");
    assert!(stdout.contains("1 done"), "{stdout}");
    cleanup(&dir);
}
