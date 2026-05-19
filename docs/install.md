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

| Host | URL | Status |
|------|-----|--------|
| **GitHub Pages** | https://koushul.github.io/SpaceTravLR_rust/ | Live (deployed from `main` via [`.github/workflows/docs.yml`](../.github/workflows/docs.yml)) |
| **Read the Docs** | https://spacetravlr-rust.readthedocs.io/en/latest/ | **404 until you import the project** (see below). Do **not** use `spacetravlr.readthedocs.io` — that slug is another project (“SpaceDocs”). |

Pushing [`.readthedocs.yaml`](../.readthedocs.yaml) to GitHub does **not** create the RTD site by itself. You must import the repo once on [readthedocs.org](https://readthedocs.org/).

**One-time Read the Docs setup (maintainers):**

1. Sign in at [readthedocs.org](https://readthedocs.org/) → **Import a Project** → **Import manually** or GitHub → select **`Koushul/SpaceTravLR_rust`**.
2. Set **Project slug** to **`spacetravlr-rust`** (Admin → Settings). The URL must be exactly `https://spacetravlr-rust.readthedocs.io/`.
3. **Repository URL** must be `https://github.com/Koushul/SpaceTravLR_rust` (no `/tree/main` suffix).
4. **Versions → `latest` (or default)** → set **Git ref** / branch to **`main`**. This repo has **no `master` branch**; if RTD builds `master`, you get *“Config file not found at default path”* even though [`.readthedocs.yaml`](../.readthedocs.yaml) exists on `main`.
5. Admin → **Settings → Advanced settings** → enable **“Use .readthedocs.yaml configuration file”** (Config file v2). Leave **Configuration file** empty or set exactly `.readthedocs.yaml` (repo root, not `docs/`).
6. Click **Build** for the `main` / `latest` version. A green build means https://spacetravlr-rust.readthedocs.io/en/latest/ will work.
7. Optional: Admin → **Integrations** → enable GitHub webhooks so every push to `main` rebuilds.

**“Config file not found at default path”** — checklist:

| Check | Expected |
|-------|----------|
| File on GitHub `main` | https://github.com/Koushul/SpaceTravLR_rust/blob/main/.readthedocs.yaml |
| RTD version branch | **`main`**, not `master` |
| Config file path in RTD UI | empty (default) or `.readthedocs.yaml` |
| Repository | `Koushul/SpaceTravLR_rust` |

Other build failures: open the RTD log (e.g. missing deps → this repo uses `pip install -r docs/requirements.txt` in `.readthedocs.yaml`).

Local build:

```bash
python3 -m venv .venv-docs && source .venv-docs/bin/activate
pip install -r docs/requirements.txt
mkdocs serve
```


