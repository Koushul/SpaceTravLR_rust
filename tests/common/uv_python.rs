//! Helpers for tests that spawn `uv run --isolated` + Python. Those tests use
//! `#[ignore = "requires uv/python (isolated `uv run`); default off — run `cargo test -- --ignored`]`
//! and stay off the default `cargo test` path. Run with:
//!
//! ```text
//! cargo test -- --ignored
//! ```

use std::ffi::OsString;
use std::process::Command;

pub fn uv_bin() -> OsString {
    std::env::var_os("UV_BIN").unwrap_or_else(|| "uv".into())
}

pub fn uv_available() -> bool {
    Command::new(uv_bin())
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Fail the test when `uv` is missing. Ignored uv tests must not return Ok.
pub fn require_uv() {
    assert!(
        uv_available(),
        "uv is required (install uv or set UV_BIN); this test must not pass without it"
    );
}
