# Installing SpaceTravLR (Rust)

Prebuilt binaries include **`spacetravlr`** (training), **`spacetravlr-perturb`** (perturbation / batch), and **`spatial_viewer`** (local web viewer + cache). All three are built together for each supported platform.

## Supported platforms (prebuilt)

GitHub Releases publish tarballs for:

| Target triple | Typical systems |
|---------------|-----------------|
| `x86_64-unknown-linux-gnu` | 64-bit Linux (glibc) |
| `aarch64-unknown-linux-gnu` | Linux ARM64 — **not** in default release matrix (needs public repo + `ubuntu-24.04-arm` in workflow or build from source) |
| `x86_64-apple-darwin` | Intel Mac |
| `aarch64-apple-darwin` | Apple Silicon Mac |

Anything else: use **WSL2** on Windows, or **build from source** (see below).

## Quick install (macOS / Linux)

Recommended one-liner pins the install script to **[v1.1.0](https://github.com/Koushul/SpaceTravLR_rust/releases/tag/v1.1.0)**. **Use `-o` then `sh`** — `curl … | sh` can fail if piped output is truncated (some IDEs, proxies, or buffer limits). Track `main` instead by using `refs/heads/main` in the URL.

```bash
curl -fsSL https://raw.githubusercontent.com/Koushul/SpaceTravLR_rust/refs/tags/v1.1.0/scripts/install.sh -o install-spacetravlr.sh && sh install-spacetravlr.sh && rm -f install-spacetravlr.sh
```

- **Install location:** `$HOME/.local/bin` by default. Override with `SPACETRAVLR_INSTALL_DIR`.
- **PATH:** If `spacetravlr` is not found afterward, add that directory to `PATH` (the installer prints a hint).

Preview what would happen (no download):

```bash
curl -fsSL https://raw.githubusercontent.com/Koushul/SpaceTravLR_rust/refs/tags/v1.1.0/scripts/install.sh -o install-spacetravlr.sh && env INSTALL_DRY_RUN=1 sh install-spacetravlr.sh && rm -f install-spacetravlr.sh
```

Quiet logs (errors still print):

```bash
curl -fsSL https://raw.githubusercontent.com/Koushul/SpaceTravLR_rust/refs/tags/v1.1.0/scripts/install.sh -o install-spacetravlr.sh && sh install-spacetravlr.sh --quiet && rm -f install-spacetravlr.sh
```

## Manual install from a release

1. Open **[Releases](https://github.com/Koushul/SpaceTravLR_rust/releases)** (current: **[v1.1.0](https://github.com/Koushul/SpaceTravLR_rust/releases/tag/v1.1.0)**).
2. Download **`spacetravlr-<tag>-<target>.tar.gz`** for your machine (triple from the table above).
3. Verify **`SHA256SUMS`** from the same release (recommended).
4. Extract and move the three binaries to a directory on your `PATH`, then `chmod +x` each.

## Updates (opt-in, not automatic)

**Nothing upgrades by itself.** When you want to update:

```bash
spacetravlr --update
```

This replaces **`spacetravlr`**, **`spacetravlr-perturb`**, and **`spatial_viewer`** next to the `spacetravlr` you run (keep them in the same directory). It requires a build that includes the `self-update` feature (prebuilt release assets do).

Pin a tag:

```bash
spacetravlr --update --update-version v0.2.0
```

If your binary was built without `self-update`, reinstall with the curl installer or from source.

## Compute: WebGPU vs CPU

Same as a source build: the CLIs may use **WebGPU** when available. For a stable **CPU** path:

- `SPACETRAVLR_FORCE_CPU=1` or `SPACETRAVLR_DISABLE_WGPU=1`  
 See the main [README](README.md) for details.

## macOS: Gatekeeper / quarantine

Downloaded binaries may be quarantined. If execution is blocked:

```bash
xattr -dr com.apple.quarantine /path/to/spacetravlr /path/to/spacetravlr-perturb /path/to/spatial_viewer
```

## Windows

There are **no** official Windows `.exe` artifacts in this repo today. Use **WSL2** (treat as Linux) or install Rust and build from source.

## Install from crates.io

If the crate is [published on crates.io](https://rust-cli.github.io/book/tutorial/packaging.html#quickest-cargo-publish), install with Cargo (compiles locally; Rust ≥ **1.86**):

```bash
cargo install spacetravlr --locked
```

Default **`cargo install`** includes the training/perturb TUIs and **`self-update`** (so **`spacetravlr --update`** works). Add **`spatial_viewer`** to match release tarballs:

```bash
cargo install spacetravlr --locked --features spatial-viewer
```

The package name is **`spacetravlr`**; binaries are **`spacetravlr`**, **`spacetravlr-perturb`**, and **`spatial_viewer`** when **`spatial-viewer`** is enabled.

## Install from source (developers)

Requires **Rust ≥ 1.86** (edition 2024).

```bash
git clone https://github.com/Koushul/SpaceTravLR_rust.git
cd SpaceTravLR_rust
```

Training + perturb TUI (default):

```bash
cargo install --path . --locked
```

That installs **`spacetravlr`** and **`spacetravlr-perturb`** with TUIs and **`self-update`**. To also install **`spatial_viewer`**:

```bash
cargo install --path . --locked --features spatial-viewer
```

Lean build (no Ratatui dashboard or self-update):

```bash
cargo install --path . --locked --no-default-features
```

To add **`self-update`** back on a lean build:

```bash
cargo install --path . --locked --no-default-features --features self-update
```

(Full parity with release tarballs: default features plus **`spatial-viewer`** — the line above with **`--features spatial-viewer`**.)

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Wrong architecture | Download the tarball matching your triple; on Mac, check Apple Silicon vs Intel. |
| `spacetravlr: command not found` | Add `$HOME/.local/bin` (or your install dir) to `PATH`. |
| Unwritable install directory | Choose a user-owned `SPACETRAVLR_INSTALL_DIR`, or fix permissions. |
| GitHub API rate limit | Rare for install/self-update; retry later or download manually from Releases. |
| SHA256 mismatch | Re-download; verify you picked the tarball for the correct tag and triple. |

## Scripts in this repo

- **[`scripts/install.sh`](scripts/install.sh)** — installer (also available via `curl` raw URL above).
- Tests: `bats scripts/install.bats` (see CI).
