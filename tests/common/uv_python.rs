//! Tests that spawn `uv run --isolated` + Python should use [`UV_PYTHON_IGNORE`] and stay off
//! the default `cargo test` path. Run with:
//!
//! ```text
//! cargo test -- --ignored
//! ```

use std::ffi::OsString;
use std::process::Command;

/// Message for `#[ignore = UV_PYTHON_IGNORE]` on uv/Python integration tests.
pub const UV_PYTHON_IGNORE: &str =
    "requires uv/python (isolated `uv run`); default off — run `cargo test -- --ignored`";

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
