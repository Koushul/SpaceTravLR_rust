# SpaceTravLR {: .st-brand }
Spatially perturbing Transcription factors, Ligands & Receptors.

## Quick Install (Recommended)
```bash
curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh
```
This will download the latest precompiled binaries and config files from github. Default install path: `\$HOME/.local/bin`

> SpaceTravLR works straight out of the box and requires **no virtual environment** or extra dependencies setup. And no CUDA. :)


SpaceTravLR ships with two binaries: `spacetravlr` and `spacetravlr-perturb`. Prebuilt binaries are available for `x86_64-unknown-linux-gnu` and `aarch64-apple-darwin`.




For updates:

```bash
spacetravlr --update
```


## Compile from source
Alternatively, you can download and install SpaceTravLR from the source code using the [Rust toolchain](https://rustup.rs/). This is platform agnostic and can help fix ldd/gcc or WebGPU compatibility issues with older hardware.

First, install the Rust compiler.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Or load it from the cluster env
```bash
module load cargo
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

This will download a tiny .h5ad from github and train two genes in parallel to confirm that SpaceTravLR is able to see and use any CPUs and GPUs.

## Documentation site

Public HTML docs are built on [Read the Docs](https://readthedocs.org/) from [`mkdocs.yml`](../mkdocs.yml) and [`.readthedocs.yaml`](../.readthedocs.yaml) (no `koushul` in the URL — e.g. `https://spacetravlr.readthedocs.io/`).

**One-time setup (maintainers):**

1. Sign in at [readthedocs.org](https://readthedocs.org/) and **Import a Project** → GitHub → `Koushul/SpaceTravLR_rust`.
2. Set the project **slug** to `spacetravlr` (Admin → Settings → name/slug) so the URL matches the links above.
3. Enable **Build pull requests** if you want preview builds on PRs.

Local build:

```bash
python3 -m venv .venv-docs && source .venv-docs/bin/activate
pip install -r docs/requirements.txt
mkdocs serve
```


