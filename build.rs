fn emit_spacetravlr_build_metadata() {
    let sha = std::env::var("SPACETRAVLR_GIT_SHA").unwrap_or_else(|_| {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        std::process::Command::new("git")
            .args(["-C", &dir, "rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    });
    let triple = std::env::var("SPACETRAVLR_TARGET_TRIPLE")
        .or_else(|_| std::env::var("TARGET"))
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=SPACETRAVLR_GIT_SHA={sha}");
    println!("cargo:rustc-env=SPACETRAVLR_TARGET_TRIPLE={triple}");
    println!("cargo:rerun-if-env-changed=SPACETRAVLR_GIT_SHA");
    println!("cargo:rerun-if-env-changed=SPACETRAVLR_TARGET_TRIPLE");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads");
}

fn main() {
    emit_spacetravlr_build_metadata();
}
