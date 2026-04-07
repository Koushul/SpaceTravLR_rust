---
name: spacetravlr-cli
description: Documents SpaceTravLR_rust training (spacetravlr) and in-silico perturbation (spacetravlr-perturb) CLIs—config TOML, run repro, gene subsets, GRN options, condition splits, multi-host join, compute backends, and batch/TUI perturb flows. Use when the user asks how to train, join a run, export CNN weights, run perturbations from the terminal, or interpret spaceship_config / spacetravlr_run_repro.toml for these binaries.
---

# SpaceTravLR CLI (training and perturbation)

## When to use

Apply when helping with:

- **`spacetravlr`** — spatial GRN training from `.h5ad`, config overrides, `run-summary`, `--join-output-dir`, `--condition`
- **`spacetravlr-perturb`** — load a finished run via `spacetravlr_run_repro.toml` + `*_betadata.feather`, TUI or **`--export` / `--out`** batch mode
- Choosing **config files**, **artifacts** (feathers, repro TOML, locks, optional CNN `.npz`), or **environment variables** for compute

**Source of truth in repo:** [`src/bin/spacetravlr.rs`](src/bin/spacetravlr.rs), [`src/bin/spacetravlr_perturb.rs`](src/bin/spacetravlr_perturb.rs), [`spaceship_config.toml`](spaceship_config.toml), [`src/config.rs`](src/config.rs). For HTTP/UI perturbation and MCP, see [.agents/skills/spatial-viewer-mcp/SKILL.md](spatial-viewer-mcp/SKILL.md).

---

## Binaries

| Binary | Default crate feature | Role |
|--------|------------------------|------|
| `spacetravlr` | `tui` | Train all targets; TUI dashboard unless `--plain`; optional `run-summary` subcommand |
| `spacetravlr-perturb` | `tui` | Perturbation only; TUI by default, **`--export` / `--out`** single-gene batch, or **`--batch-toml`** multi-job batch |

Install: `cargo install --path . --locked` or `cargo run --bin …`.

---

## Configuration files

### `spaceship_config.toml` (or `-c` / `--config`)

Typical sections:

- **`[data]`** — `adata_path`, `layer` (e.g. `imputed_count`), `cluster_annot`, optional `condition`, optional **`perturb_obs_subset_file`** (one `obs_names` per line; **perturbation** loads only those rows for smaller RAM)
- **`[spatial]`** — `radius`, `spatial_dim`, `contact_distance`, `weighted_ligand_scale_factor`
- **`[grn]`** — `network_data_dir`, `tf_priors_feather`, `tf_ligand_cutoff`, `max_ligands` (serde alias `max_lr_pairs`), modulator toggles, `extra_modulators`, `extra_lr`, file variants
- **`[cnn]`**, **`[lasso]`**, **`[training]`**, **`[training.hybrid]`** — CNN/Lasso/hybrid gating (hybrid only when `mode = "hybrid"`)
- **`[execution]`** — `n_parallel`, `output_dir`, `write_minimal_repro_h5ad`, `stale_lock_secs` (join lock recovery)
- **`[perturbation]`** — `beta_scale_factor`, optional `beta_cap`, **`n_propagation`**, **`ligand_grid_factor`** (grid vs exact received ligands), optional **`cells_csv`** / **`cells_csv_column`** (defaults for `spacetravlr-perturb` **`--export`** and TUI cell-scope CSV when CLI does not pass **`--cells-csv`**; paths relative to the run TOML’s directory unless absolute)
- **`[model_export]`** — `save_cnn_weights`, `compressed_npz`, `output_subdir`

CLI flags override many of these when **not** in `--join-output-dir` mode.

### `spacetravlr_run_repro.toml`

Written under **`[execution].output_dir`** (leader run). Records the **effective** `SpaceshipConfig` after CLI merges. **`--join-output-dir`** hosts read this file; hyperparameters and gene lists come from the repro, not from `--config` on the join command. **`--max-ligands`** on a join host must match `[grn].max_ligands` in the repro or the process errors.

---

## `spacetravlr` — training

### Modes

- **TUI (default)** — full-screen dashboard (`--plain` disables it → line logs)
- **`--demo`** — fake dashboard only; no AnnData / no exports
- **`run-summary`** subcommand — HTML report without training (`spacetravlr run-summary --help`)

### Input

- **`--config`** — TOML path; if omitted, searches for `spaceship_config.toml`
- **`--h5ad`** — overrides `[data].adata_path`
- **`--tf-prior`** — overrides `[grn].tf_priors_feather`

### Gene list and GRN extras

- **`--genes`** — comma-separated target symbols
- **`--max-genes`** — cap after `--genes`, AnnData `var` order
- **`--max-ligands`** — cap **database** LR pairs by top ligand mean expression (**`[data].layer`**); `extra_lr` not capped
- **`--extra-modulators`**, **`--extra-lr`** — merge with `[grn]` lists (see `--help` for pair syntax)

### Training hyperparameters (override TOML)

- **`--training-mode`** — `seed` | `full` | `hybrid`
- **`--epochs`**, **`--parallel`**, **`--l1-reg`**, **`--group-reg`**, **`--lr`** (CNN Adam), **`--n-iter`**, **`--tol`**
- **`--cnn-output-activation`** — `identity` | `sigmoid` | `tanh` | `sigmoid-x2`
- **`--weighted-ligand-scale-factor`**

### Output and multi-run

- **`--output-dir`** — run directory (`*_betadata.feather`, logs, repro TOML)
- **`--condition`** — `obs` column; training splits into `output_dir/conditions/<value>/` per group
- **`--join-output-dir DIR`** — load `DIR/spacetravlr_run_repro.toml`; claim genes via `.lock`; only **`--parallel`** and a few join-safe overrides apply; **`--max-ligands`** must match repro if passed
- **`--write-minimal-repro-h5ad`**, **`--save-cnn-weights`**

### Compute backend

Selection is in [`src/bin/compute_backend.rs`](src/bin/compute_backend.rs). **`SPACETRAVLR_FORCE_CPU=1`** or **`SPACETRAVLR_DISABLE_WGPU=1`** forces the NdArray CPU path instead of WebGPU.

### Artifacts

- **`{gene}_betadata.feather`** under output (or under `conditions/.../` when split)
- **`spacetravlr_run_repro.toml`** — canonical config for join / viewer / perturb
- Optional **`saved_models/`** (or `[model_export].output_subdir`) CNN `.npz` when enabled
- Lock files per gene during parallel/join training

### GRN data

Parquet networks resolve via **`[grn].network_data_dir`**, then **`SPACETRAVLR_DATA_DIR`**, then manifest / cwd walk (see [`src/network.rs`](src/network.rs)).

---

## `spacetravlr-perturb` — perturbation

Loads **`PerturbRuntime::from_run_toml`**: same AnnData path, layer, cluster column semantics, and **`output_dir`** as training (feathers + config). Does **not** auto-subset by **`[data].condition`**; use a subset `.h5ad`, **`perturb_obs_subset_file`** in the repro, or batch **`--cells-csv`**.

### TUI (default when no `--export` / `--out`)

Ratatui UI: pick gene, **`desired_expr`**, **`n_propagation`**, optional cell column from CSV (**Ctrl+O**). Initial defaults can be set from CLI (`--desired-expr`, `--n-propagation`, `--verbose`, `--cells-csv`, `--cells-csv-column`, `--run-toml`). If **`--cells-csv`** is not passed, the TUI loads **`[perturbation].cells_csv`** / **`cells_csv_column`** from the run TOML after **`PerturbRuntime`** loads so **Ctrl+O** works without extra flags. The right panel shows **mean expression of the highlighted gene by betadata cluster key** (Unicode bar preview of the input **`[data].layer`** matrix).

**Without `tui` feature:** legacy stdin flow / prompts (see `#[cfg(not(feature = "tui"))]` in [`spacetravlr_perturb.rs`](src/bin/spacetravlr_perturb.rs)).

### Batch (non-interactive)

Requires **`--run-toml`**, **`--gene`**, and **`--export PATH`** or **`--out PATH`** (same flag).

Optional:

- **`--desired-expr`** (default `0`)
- **`--n-propagation`** (else `[perturbation].n_propagation` from TOML)
- **`--cells-csv`** + **`--cells-csv-column`** — CSV must have a **header**; each column lists **`obs_names`** strings from the AnnData; chosen column = cells whose expression is scoped for the perturbation; omit CSV → perturb **all** cells. If the CLI omits **`--cells-csv`**, the binary falls back to **`[perturbation].cells_csv`** / **`cells_csv_column`** in the run TOML when set (requires both in TOML for CSV scope); if only **`--cells-csv`** is passed, **`[perturbation].cells_csv_column`** can supply the column name.
- **`--verbose`** — timings on stderr

After a successful run, the export path validates output shape and, for near-zero **`desired_expr`** (KO), checks that the target gene’s simulated values are ~0 on the scoped rows (fails loudly on mismatch).

Output: single Feather matrix (**`CellID`** + all genes), written via [`write_betadata_feather`](src/betadata.rs).

Full example: run **`spacetravlr-perturb --help`** (see `after_long_help`).

### Batch TOML (`--batch-toml`)

For many **single-gene** perturbations with one `PerturbRuntime::from_run_toml` load and bounded parallel workers (see [`perturb_batch.rs`](src/perturb_batch.rs)):

- **CLI:** `--run-toml` + `--batch-toml PATH`. Optional `--batch-parallelism` overrides the file’s `parallelism`. Optional `--n-propagation` sets the default `n_propagation` for every job before per-job values from the file. **`--gene`, `--out`/`--export`, `--cells-csv` are not allowed** (cell scope lives in the batch file if needed).
- **Unifying shape:** `gene` *or* `genes` (string or list). `desired_expr` and `n_propagation` may be scalars (broadcast) or lists of length **N** (zip with genes) or length **1** (broadcast). **`out_dir`** → default `{gene}_perturb_expr.feather` per gene; **or** `out` as one path (only if **N = 1**) or an array of **N** paths. Relative paths resolve against the batch TOML’s parent directory.
- Optional **`cells_csv`** with either **`cells_csv_column`** (default column for every job) **or** **`cells_csv_columns`** — a string or list of length **N** (zip with **`genes`**) or length **1** (broadcast). Each entry is a **CSV header name**; **`""`** or whitespace-only means **all cells** for that gene. Do not set both **`cells_csv_column`** and **`cells_csv_columns`**. **`cells_csv_columns` requires `cells_csv`**. Or use **`cells_obs_file`** (same line-list semantics as **`perturb_obs_subset_file`**, shared by all jobs; mutually exclusive with **`cells_csv`**).
- Each written feather is validated (shape + KO target check when **`desired_expr`** ≈ 0), same as single-job export.
- Optional spatial/received-ligand overrides (root keys, applied to every job in the file): **`radius`** (per-ligand Gaussian radius map is rebuilt with this single value for all ligands), **`ligand_grid_factor`** (grid vs exact received ligands during propagation; merges with the run’s `[perturbation]` default), **`contact_distance`** (hard neighbor cutoff in distance units; `None` when omitted). When any of these differ from the loaded `PerturbRuntime`, initial weighted-ligand matrices are recomputed and baseline splash cache is bypassed for that job so results stay consistent with the override.

### Perturbation physics (from TOML)

[`[perturbation]`](spaceship_config.toml): **`beta_scale_factor`**, **`n_propagation`**, **`ligand_grid_factor`** (grid-approx vs exact received ligands), optional **`beta_cap`**.

---

## Quick reference commands

```bash
# Train with plain logs
spacetravlr --plain --config spaceship_config.toml --h5ad /path/data.h5ad --output-dir /path/run

# Full CNN + explicit genes + max ligands + join-friendly repro in output_dir
spacetravlr --plain --training-mode full --genes ACTB,MALAT1 --max-ligands 100 --parallel 8 --output-dir /path/run --h5ad /path/data.h5ad

# Join workers (repro must exist; omit conflicting hyperparameter flags)
spacetravlr --join-output-dir /path/run --parallel 16 --plain

# Batch perturbation
spacetravlr-perturb --run-toml /path/run/spacetravlr_run_repro.toml --out /tmp/sim.feather --gene SOX2 --desired-expr 0 --n-propagation 4 --verbose

# Many single-gene jobs (batch TOML)
spacetravlr-perturb --run-toml /path/run/spacetravlr_run_repro.toml --batch-toml /path/batch_perturb.toml --verbose

# HTML run summary (no training)
spacetravlr run-summary --config spaceship_config.toml --h5ad /path/data.h5ad --output-dir /path/run
```

---

## Related

- Interactive spatial viewer + REST perturb APIs: **spatial_viewer** binary (feature `spatial-viewer`), documented in the spatial-viewer-mcp skill and [README.md](../../README.md).
