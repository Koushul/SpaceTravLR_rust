# Run SpaceTravLR on your own data

End-to-end path from install through training to an interaction table you can analyze in Python or Polars.

!!! tip "What you need"
    A machine with a **GPU** (WebGPU-capable; see [verify](#2-verify-installation) below), an `.h5ad` with expression in `.X` and coordinates in `.obsm['spatial']`, and disk space for per-gene `*_betadata.feather` files.

---

## 1. Install

**Recommended** — prebuilt binaries (Linux x86_64, macOS Apple Silicon):

```bash
curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh
```

This installs `spacetravlr` and `spacetravlr-perturb` on your `PATH`. No virtual environment or CUDA toolkit is required.

To update an existing install:

```bash
spacetravlr --update
```

**From source** (any platform with Rust 1.86+):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/Koushul/SpaceTravLR_rust.git
cd SpaceTravLR_rust
cargo install --path . --locked
```

More detail: [Installation](../install.md).

---

## 2. Verify installation

Confirm binaries and GPU training work on your hardware:

```bash
spacetravlr --verify
```

This downloads a small reference `.h5ad`, runs preprocessing and parallel training on two genes, and writes a log. Fix WebGPU or driver issues before pointing at a large dataset.

---

## 3. Prepare your `.h5ad`

SpaceTravLR expects:

| Requirement | Where |
|-------------|--------|
| Gene expression counts | `.X` |
| 2D spatial coordinates | `.obsm['spatial']` |
| Cell type labels (recommended) | `.obs['cell_type']` or another column you pass at train time |

If `cell_type` is missing, SpaceTravLR can cluster with Leiden before training. Raw counts are fine — normalization and QC can run automatically when needed.

Quick sanity check without loading the full object:

```bash
spacetravlr --peek /path/to/adata.h5ad
spacetravlr --peek /path/to/adata.h5ad --obs cell_type
```

Optional Rust-only preprocessing for very large objects:

```bash
spacetravlr --rust-process-h5ad --h5ad adata.h5ad
```

Input and preprocessing options: [Usage — Running SpaceTravLR](../usage.md#run-spacetravlr).

---

## 4. Train

Pick an output directory and start a run. The leader writes `spacetravlr_run_repro.toml` early so workers and downstream tools can find the same config.

**Interactive** (prompts for paths and options):

```bash
spacetravlr
```

**Non-interactive** (typical batch / cluster job):

```bash
spacetravlr --h5ad /storage/tissues/adata.h5ad \
  --output-dir /storage/outputs/spacetravlr_run
```

**Config file** (reproducible settings — [example `runme.toml`](../runme.toml)):

```bash
spacetravlr -c runme.toml
```

While training, each completed gene produces `{GENE}_betadata.feather` under the output directory. See [Output files](../output.md) for the full tree, locks, and condition splits.

### Multiple jobs on a cluster (shared output directory)

Large atlases are trained **one target gene at a time**, but genes are independent. SpaceTravLR scales horizontally by splitting that gene queue across many processes, each with its own GPU, all writing into the **same** output folder on shared storage (NFS, Lustre, etc.).

**How it works**

1. **Leader run** — You start training once with `--h5ad` and `--output-dir`. SpaceTravLR preprocesses the data, picks the gene list, and writes **`spacetravlr_run_repro.toml`** early (full effective config: paths, hyperparameters, which genes to train). That TOML is the contract every other process reads.
2. **Worker runs** — Each additional job uses **`--join-output-dir`** pointing at the **same** directory. Workers do not re-preprocess; they load the repro TOML and compete for genes that are not finished yet.
3. **Claiming a gene** — Before training gene `G`, a process creates **`G.lock`** in the output tree. Other processes skip genes that already have a finished **`G_betadata.feather`** or an active lock. When training finishes, the lock is removed and the feather remains.
4. **Crash recovery** — Locks left behind after a node failure can be cleared automatically after **`stale_lock_secs`** (default 3600 s in config) so another worker can reclaim the gene.

So you are not manually sharding gene lists in your job script: you submit **N identical worker jobs** (or one leader plus N−1 workers). Throughput scales roughly with the number of GPUs until shared filesystem or the gene queue becomes the bottleneck.

**Typical workflow on SLURM**

| Step | What to submit | Command |
|------|----------------|---------|
| 1 | One **leader** job (preprocess + start queue) | `spacetravlr --h5ad … --output-dir …` |
| 2 | Many **worker** jobs (same output path) | `spacetravlr --join-output-dir … --plain` |

Use **`--plain`** on workers so the process does not try to open the full-screen TUI inside a batch allocation.

After the leader has written `spacetravlr_run_repro.toml`, array-submit or manually launch as many GPU jobs as you want, all with the same `--join-output-dir`. Example worker script (one GPU per task):

```bash
#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=SpaceTravLR
#SBATCH --mem=300G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

spacetravlr --join-output-dir /storage/tissues/output --plain
```

Replace the path with your run’s **`output_dir`** (the folder that contains `spacetravlr_run_repro.toml`). Submit this script many times—or as a job array—to add GPUs; each task trains whichever genes are still unclaimed.

**Leader example** (run once before workers):

```bash
spacetravlr --h5ad /path/to/adata.h5ad \
  --output-dir /storage/tissues/output
```

You can also run the leader interactively on a login node or a single long GPU job, wait until `spacetravlr_run_repro.toml` exists, then flood the cluster with join jobs.

!!! note "Same config everywhere"
    Workers inherit hyperparameters and the gene list from the repro TOML. Do not mix different `--config` files or output directories across jobs in the same run.

Hyperparameters (`n_parallel`, `stale_lock_secs`, etc.): [Parameters](../params.md).

---

## 5. Collect interactions

After training, aggregate β coefficients across cell types (and optionally spatial clusters) into one table:

```bash
spacetravlr collect-interactions \
  --run-toml /storage/tissues/output/spacetravlr_run_repro.toml
```

Defaults:

- Reads `*_betadata.feather` from the run’s output directory (from the repro TOML).
- Groups by `--annot` (default `cell_type`).
- Aggregates with `--aggregate mean` (also `min`, `max`, `sum`, `positive`, `negative`).
- Writes `plucked_feathers.feather` next to the repro TOML unless you set `--out`.

**Per-cluster breakdown** (e.g. Leiden or sample id in `obs`):

```bash
spacetravlr collect-interactions \
  --run-toml /storage/outputs/spacetravlr_run/spacetravlr_run_repro.toml \
  --cluster-col cluster \
  --out /storage/outputs/spacetravlr_run/interactions_by_cluster.feather
```

Load the result in Python:

```python
import pandas as pd
betadf = pd.read_feather("/storage/tissues/output/plucked_feathers.feather")
```
