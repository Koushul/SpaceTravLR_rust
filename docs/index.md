# SpaceTravLR {: .st-brand }
Spatially perturbing Transcription factors, Ligands & Receptors.


## Quick Install (Recommended)

```bash
curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh
```
This will download the latest precompiled binaries from github.
SpaceTravLR comes with two binaries: `spacetravlr` and `spacetravlr-perturb`.

For updates:

```bash
spacetravlr --update
```


## Compile from source
Alternatively, you can download and install SpaceTravLR from the source code using the [Rust toolchain](https://rustup.rs/).

First, install the Rust compiler.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then clone the repo and compile the code. This will install the binaries in `~/.cargo/bin` by default.

```bash
git clone https://github.com/Koushul/SpaceTravLR_rust.git
cd SpaceTravLR_rust
cargo install --path . --locked
```


## Verify installation
To confirm that SpaceTravLR was properly installed and is compatible with the current hardware, you can run

```bash
spacetravlr --verify
```

This will download a small .h5ad from github and train two genes in parallel to confirm that SpaceTravLR is able to see and use any CPUs and GPUs.


