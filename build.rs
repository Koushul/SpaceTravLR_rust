fn main() {
    println!("cargo:rerun-if-env-changed=SPACETRAVLR_GIT_SHA");
    println!("cargo:rerun-if-env-changed=GITHUB_SHA");

    let sha = std::env::var("SPACETRAVLR_GIT_SHA")
        .or_else(|_| std::env::var("GITHUB_SHA"))
        .map(|s| {
            if s.len() > 7 {
                s[..7].to_string()
            } else {
                s
            }
        })
        .unwrap_or_else(|_| "dev".to_string());
    println!("cargo:rustc-env=SPACETRAVLR_GIT_SHA={sha}");

    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=SPACETRAVLR_TARGET_TRIPLE={target}");
}
