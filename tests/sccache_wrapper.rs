//! The rustc wrapper must compile without sccache on PATH, and through sccache when present.

use std::path::PathBuf;
use std::process::Command;

fn which_cmd(name: &str) -> Option<PathBuf> {
    Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
        .filter(|p| p.is_file())
}

fn wrapper() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/spacetravlr-rustc-wrapper")
}

#[test]
fn wrapper_file_is_executable_script() {
    let w = wrapper();
    assert!(w.is_file(), "missing {}", w.display());
    let text = std::fs::read_to_string(&w).expect("read wrapper");
    assert!(text.contains("sccache"), "{text}");
}

#[test]
fn wrapper_invokes_rustc_without_sccache_on_path() {
    let rustc = which_cmd("rustc").expect("rustc on PATH");
    let out = Command::new(wrapper())
        .arg(&rustc)
        .arg("--version")
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("run wrapper");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "wrapper rustc --version failed: {stdout}{stderr}"
    );
    assert!(
        stdout.contains("rustc") || stderr.contains("rustc"),
        "expected rustc version, got stdout={stdout:?} stderr={stderr:?}"
    );
}

#[test]
fn sccache_records_a_rustc_cache_hit() {
    let sccache = match Command::new("sccache").arg("--version").output() {
        Ok(o) if o.status.success() => "sccache",
        _ => {
            eprintln!("sccache not on PATH; skip cache-hit check");
            return;
        }
    };

    let dir = std::env::temp_dir().join(format!("spacetravlr_sccache_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let src = dir.join("probe.rs");
    std::fs::write(&src, "#![crate_type = \"lib\"]\npub fn n() -> u8 { 1 }\n").expect("write");
    let obj1 = dir.join("probe1.rlib");
    let obj2 = dir.join("probe2.rlib");

    let _ = Command::new(sccache).arg("--start-server").output();
    assert!(
        Command::new(sccache)
            .arg("-z")
            .status()
            .expect("sccache -z")
            .success(),
        "sccache -z"
    );

    for dest in [&obj1, &obj2] {
        let st = Command::new(wrapper())
            .args([
                "rustc",
                "--crate-type",
                "lib",
                "--edition",
                "2021",
                src.to_str().expect("utf8"),
                "-o",
                dest.to_str().expect("utf8"),
            ])
            .status()
            .expect("wrapper rustc");
        assert!(st.success(), "wrapper rustc {}", dest.display());
    }

    let stats = Command::new(sccache)
        .arg("--show-stats")
        .output()
        .expect("show-stats");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&stats.stdout),
        String::from_utf8_lossy(&stats.stderr)
    );
    assert!(
        text.contains("Cache hits") && !text.contains("Cache hits                    0"),
        "expected a cache hit after compiling the same crate twice:\n{text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
