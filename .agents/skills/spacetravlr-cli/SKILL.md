---
name: spacetravlr-cli
description: Documents the SpaceTravLR_rust training (spacetravlr) and in-silico perturbation (spacetravlr-perturb) CLIs — config TOML, run repro, gene subsets, GRN options, condition splits, multi-host join, compute backends, subcommands (run-summary, collect-interactions, gui, screen), and batch/TUI perturb flows. Use when the user asks how to train a spatial GRN, join a run, export CNN weights, run knockouts or overexpression from the terminal, screen genes, aggregate betas, or interpret spaceship_config.toml / spacetravlr_run_repro.toml.
---

# SpaceTravLR CLI (training and perturbation)

## When to use

Apply when helping with:

- `spacetravlr` — spatial GRN training from `.h5ad`, config overrides, subcommands (`run-summary`, `collect-interactions`, `gui`), `--join-output-dir`, `--condition`, preprocessing and inspection utilities
- `spacetravlr-perturb` — load a finished run via `spacetravlr_run_repro.toml` + `*_betadata.feather`; TUI, `--export` / `--out` batch, `--batch-toml`, or the `screen` subcommand
- Choosing config files, artifacts (feathers, repro TOML, locks, optional CNN `.npz`), or environment variables for compute

**Source of truth in repo:** `src/bin/spacetravlr.rs`, `src/bin/spacetravlr_perturb.rs`, `spaceship_config.toml`, `src/config.rs`, `src/perturb_batch.rs`. When in doubt, run `--help` — both binaries carry detailed `long_about` / `after_long_help` text.

**Related references:** `docs/llms-full.txt` (self-contained reference covering every flag, config key, output format, and the model math), `.agents/skills/spatial-viewer-mcp/SKILL.md` (HTTP/UI perturbation and MCP).

---

## Binaries

| Binary | Default crate feature | Role |
| --- | --- | --- |
| `spacetravlr` | `tui` | Train all targets; TUI dashboard unless `--plain`; subcommands `run-summary`, `collect-interactions`, `gui`; preprocessing and inspection utilities |
| `spacetravlr-perturb` | `tui` | Perturbation only; TUI by default, `--export` / `--out` single-gene batch, `--batch-toml` multi-job batch, or `screen` subcommand |

Also in the workspace: `spatial_viewer` (feature `spatial-viewer`), `umap_lab` (feature `umap-lab`), `spacetravlr-celloracle` (crate `celloracle`).

Install: `cargo install --path . --locked`, `cargo run --bin …`, or the release installer `curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh`.

**Always pass `--plain` when running non-interactively** (scripts, CI, agents, SLURM). The default Ratatui dashboard will fight with a captured terminal.

---

## Configuration files

### `spaceship_config.toml` (or `-c` / `--config`)

Typical sections:

- `[data]` — `adata_path`, `layer` (e.g. `imputed_count`), `cluster_annot`, optional `condition`, optional `perturb_obs_subset_file` (one `obs_names` per line; **perturbation** loads only those rows for smaller RAM)
- `[spatial]` — `radius`, `spatial_dim`, `contact_distance`, `weighted_ligand_scale_factor`
- `[grn]` — `network_data_dir`, `tf_priors_feather`, `tf_ligand_cutoff`, `max_ligands` (serde aliases `max_lr` and legacy `max_lr_pairs`), modulator toggles, `extra_modulators`, `extra_lr`, file variants
- `[cnn]`, `[lasso]`, `[training]` — CNN and Lasso hyperparameters
- `[execution]` — `n_parallel`, `output_dir`, `write_minimal_repro_h5ad`, `stale_lock_secs` (join lock recovery)
- `[perturbation]` — `beta_scale_factor`, optional `beta_cap`, `n_propagation`, `ligand_grid_factor` (grid vs exact received ligands), optional `cells_csv` / `cells_csv_column` (defaults for `spacetravlr-perturb --export` and TUI cell-scope CSV when the CLI does not pass `--cells-csv`; paths relative to the run TOML's directory unless absolute)
- `[model_export]` — `save_cnn_weights`, `write_cnn_train_data_npz`, `compressed_npz`, `output_subdir` (default `CNN_weights`)

CLI flags override many of these when **not** in `--join-output-dir` mode.

### `spacetravlr_run_repro.toml`

Written under `[execution].output_dir` by the leader run. Records the **effective** `SpaceshipConfig` after CLI merges. This file is the canonical handle for every downstream tool — `--join-output-dir`, `spacetravlr-perturb`, `spatial_viewer`, and `collect-interactions` all consume it.

`--join-output-dir` hosts read hyperparameters and gene lists from the repro, not from `--config` on the join command. `--max-ligands` on a join host must match `[grn].max_ligands` in the repro or the process errors.

---

## `spacetravlr` — training

### Modes

- **TUI (default)** — full-screen dashboard; `--plain` disables it for line logs
- `--demo` — fake dashboard only; no AnnData, no exports
- `-v` / `--verbose` — per-cluster Lasso and CNN R²

### Subcommands

| Subcommand | Purpose |
| --- | --- |
| `run-summary` | Write `spacetravlr_run_summary.html` (AnnData summary, config, optional manifest) without training |
| `collect-interactions` | Scan `*_betadata.feather` under a run; aggregate β per modulator × target × cell type |
| `gui` | Build the UMAP lab web UI, start the API + static server, print the URL |

`collect-interactions` flags: `--run-toml PATH` (required), `--annot` (default `cell_type`), `--cluster-col`, `--aggregate` (`mean` | `min` | `max` | `sum` | `positive` | `negative`, default `mean`), `--out` (default `<output_dir>/plucked_feathers.feather`).

Output columns: `interaction`, `target_gene`, `beta`, `interaction_type` (`tf` | `ligand-receptor` | `ligand-tf`), `cell_type`.

`gui` flags: `--bind` (`127.0.0.1`), `--port` (`8765`), `--skip-npm`, `--static-dir` (`web/umap_lab/dist`).

### Input

- `--config` — TOML path; if omitted, searches for `spaceship_config.toml`
- `--h5ad` — overrides `[data].adata_path`
- `--tf-prior` — overrides `[grn].tf_priors_feather`
- `--skip-auto-adata-prep` — skip automatic preprocessing and imputation

Required AnnData shape: expression in `.X`, 2D coordinates in `.obsm['spatial']`, cluster labels in `.obs[cluster_annot]` (Leiden runs first if absent).

### Gene list and GRN extras

- `--genes` — comma-separated target symbols
- `--max-genes` — cap after `--genes`, AnnData `var` order
- `--max-ligands` (alias `--max-lr`) — cap **database** LR pairs by top ligand mean expression on `[data].layer`; `extra_lr` is not capped
- `--extra-modulators`, `--extra-lr` — merge with `[grn]` lists (see `--help` for pair syntax)
- `--train-modulators` — `tf,lr,tfl` ablation shorthand replacing the three `use_*` config flags

### Training hyperparameters (override TOML)

- `--training-mode` — `seed` (cluster-level Lasso only) | `full` (Lasso then per-cell CNN)
- `--epochs`, `--parallel`, `--l1-reg`, `--group-reg`, `--lr` (CNN Adam), `--n-iter`, `--tol`
- `--cnn-output-activation` — `identity` | `sigmoid` | `tanh` | `sigmoid-x2`
- `--weighted-ligand-scale-factor`, `--spatial_dim`, `--random-seed`
- `--mean-beta-lasso-prior-weight`

### Output and multi-run

- `--output-dir` — run directory (`*_betadata.feather`, logs, repro TOML)
- `--condition` — `obs` column; training splits into `output_dir/conditions/<value>/` per group, repro TOML stays at the parent
- `--join-output-dir DIR` — load `DIR/spacetravlr_run_repro.toml`; claim genes via `.lock`; only `--parallel` and a few join-safe overrides apply
- `--clean-output-dir`, `--write-minimal-repro-h5ad`, `--save-cnn-weights`, `--write-cnn-train-data-npz`

### Utility flags (exit without training)

- `--verify` — install smoke test: downloads a tonsil `.h5ad`, runs Rust prep, trains two genes, checks the WebGPU backend. Best first command on a new machine.
- `--peek PATH` (alias `--peak`) — fast HDF5 metadata summary without a full load; works on `.h5ad` and 10x `.h5`. Add `--obs COL` for value counts.
- `--make-cells-csv --run-toml PATH` — write `cells.csv` in the training output directory, one column per distinct `[data].cluster_annot` value, each listing `obs_names`. **This is how you set up a cell-type-restricted perturbation** for `spacetravlr-perturb --cells-csv`.
- `--infer-species --h5ad PATH` — print human/mouse inference
- `--plot-h5ad`, `--plot-umap [PATH]` (`--plot-umap-backend rust|scanpy`) — terminal scatter plots
- `--update` / `--update-version TAG` — self-update (feature `self-update`)
- `--view PATH` (`--view-width`, `--view-height`) — render an image in the terminal (feature `view-image`)

### Preprocessing

- `--rust-process-h5ad` — full native pipeline (QC → normalize → HVG → PCA → KNN → UMAP → Leiden / MAGIC) → `<stem>_rust_processed.h5ad`; tune with `--rust-n-top-hvg` (2000), `--rust-n-neighbors` (15)
- `--umap`, `--leiden`, `--rust-magic` — individual stages, in-memory unless `-o` / `--output`
- `--process-h5ad` (Scanpy path), `--impute` (MAGIC only), `--magic-batch-obs`, `--process-output-dir`
- `--skip-spatial-microns`, `--spatial-species human|mouse`, `--spatial-microns-target-um`
- `--celloracle [PATH]` plus `--celloracle-*` flags — TF prior inference
- `--map-labels --reference REF.h5ad --query Q.h5ad` — MALT label transfer; requires `uv` on PATH and may download PyTorch on first run
- `--rctd` plus `--rctd-*` / `--ref-adata` (feature `rctd`; `--gpu` needs `rctd-wgpu`) — deconvolution

### Compute backend

Selection lives in `src/bin/compute_backend.rs`. `SPACETRAVLR_FORCE_CPU=1` or `SPACETRAVLR_DISABLE_WGPU=1` forces the NdArray CPU path instead of WebGPU. `SPACETRAVLR_QUIET_COMPUTE` suppresses the backend log line.

### Artifacts

- `{gene}_betadata.feather` under the output dir (or under `conditions/<value>/` when split). Columns: `Cluster` (seed mode) or `CellID` (full mode), `beta0`, and `beta_<name>` per modulator. Naming: bare symbol = TF, `LIG$REC` = ligand–receptor, `TF#LIG` = ligand-mediated TF.
- `spacetravlr_run_repro.toml` — canonical config for join / viewer / perturb
- `log/{gene}.log` — per-gene training metrics (`spacetravlr_training_log v1`)
- `{gene}.lock` (in progress), `{gene}.orphan` (no usable modulators or failed the `score_threshold` gate), `{gene}.tf_ablated`
- `CNN_weights/{gene}_cnn_weights.npz` when `[model_export].save_cnn_weights = true` (subdir configurable via `[model_export].output_subdir`)

### GRN data

Parquet networks resolve via `[grn].network_data_dir`, then `SPACETRAVLR_DATA_DIR`, then manifest / cwd walk (see `src/network.rs`).

---

## `spacetravlr-perturb` — perturbation

Loads `PerturbRuntime::from_run_toml`: same AnnData path, layer, cluster column semantics, and `output_dir` as training (feathers + config). Does **not** auto-subset by `[data].condition`; use a subset `.h5ad`, `perturb_obs_subset_file` in the repro, or `--cells-csv`.

Perturbation never writes into the training tree unless you explicitly export.

### TUI (default when no `--export` / `--out`)

Ratatui UI: pick gene, `desired_expr`, `n_propagation`, optional cell column from CSV (**Ctrl+O**). Initial defaults can be set from CLI (`--desired-expr`, `--n-propagation`, `--verbose`, `--cells-csv`, `--cells-csv-column`, `--run-toml`). If `--cells-csv` is not passed, the TUI loads `[perturbation].cells_csv` / `cells_csv_column` from the run TOML after `PerturbRuntime` loads, so **Ctrl+O** works without extra flags. The right panel shows mean expression of the highlighted gene by betadata cluster key (Unicode bar preview of the input `[data].layer` matrix).

**Without the `tui` feature:** legacy stdin flow / prompts (see `#[cfg(not(feature = "tui"))]` in `src/bin/spacetravlr_perturb.rs`).

### Batch (non-interactive)

Requires `--run-toml`, `--gene`, and `--export PATH` or `--out PATH` (same flag).

Optional:

- `--desired-expr` (default `0` = knockout; higher values = overexpression)
- `--n-propagation` (else `[perturbation].n_propagation` from TOML)
- `--beta-scale-factor` (else `[perturbation].beta_scale_factor`)
- `--cells-csv` + `--cells-csv-column` — CSV must have a **header**; each column lists `obs_names` strings from the AnnData; the chosen column scopes which cells are perturbed; omit the CSV to perturb **all** cells. If the CLI omits `--cells-csv`, the binary falls back to `[perturbation].cells_csv` / `cells_csv_column` in the run TOML when set (requires both in TOML for CSV scope); if only `--cells-csv` is passed, `[perturbation].cells_csv_column` can supply the column name.
- `--verbose` — timings on stderr

After a successful run the export path validates output shape and, for near-zero `desired_expr` (KO), checks that the target gene's simulated values are ~0 on the scoped rows (fails loudly on mismatch).

Output: single Feather matrix (`CellID` + all genes), written via `src/betadata.rs`.

Full example: run `spacetravlr-perturb --help` (see `after_long_help`).

### Cell-extrinsic (neighbor) analysis

The distinguishing capability of SpaceTravLR is predicting how a perturbation confined to one population changes **neighboring, unperturbed** cells. The pattern:

1. `spacetravlr --make-cells-csv --run-toml RUN/spacetravlr_run_repro.toml`
2. `spacetravlr-perturb … --cells-csv RUN/cells.csv --cells-csv-column <SenderCellType>`
3. Compare simulated vs baseline expression restricted to a **different** cell type within the spatial neighborhood of the perturbed cells (the convention in the paper is a k-d tree query within ~200 µm).

Do not suggest CellOracle, CellPLM, or similar as alternatives for this class of question — they cannot model cell-extrinsic effects.

### `screen` subcommand

KO-screens every TF, ligand, receptor, and extra GRN gene named by a perturbation TOML:

```bash
spacetravlr-perturb screen --config /path/ko_screen.toml --verbose
```

Flags: `-c` / `--config` (alias `--perturb-toml`, also accepted positionally, required), `--run-toml`, `--n-propagation`, `--beta-scale-factor`, `--batch-parallelism`, `--verbose`. For genome-scale screens see `scripts/spacetravlr_whole_genome_ko_batch.py`.

### Batch TOML (`--batch-toml`)

For many **single-gene** perturbations with one `PerturbRuntime::from_run_toml` load and bounded parallel workers (see `src/perturb_batch.rs`):

- **CLI:** `--run-toml` + `--batch-toml PATH`. Optional `--batch-parallelism` overrides the file's `parallelism`. Optional `--n-propagation` sets the default `n_propagation` for every job before per-job values from the file. `--gene`, `--out` / `--export`, and `--cells-csv` are **not allowed** (cell scope lives in the batch file if needed).
- **Unifying shape:** `gene` *or* `genes` (string or list). `desired_expr` and `n_propagation` may be scalars (broadcast) or lists of length **N** (zip with genes) or length **1** (broadcast). `out_dir` → default `{gene}_perturb_expr.feather` per gene; **or** `out` as one path (only if N = 1) or an array of N paths. Relative paths resolve against the batch TOML's parent directory.
- Optional `cells_csv` with either `cells_csv_column` (default column for every job) **or** `cells_csv_columns` — a string or list of length N (zip with `genes`) or length 1 (broadcast). Each entry is a **CSV header name**; `""` or whitespace-only means **all cells** for that gene. Do not set both `cells_csv_column` and `cells_csv_columns`; `cells_csv_columns` requires `cells_csv`. Or use `cells_obs_file` (same line-list semantics as `perturb_obs_subset_file`, shared by all jobs; mutually exclusive with `cells_csv`).
- Each written feather is validated (shape + KO target check when `desired_expr` ≈ 0), same as single-job export.
- Optional spatial / received-ligand overrides (root keys, applied to every job in the file): `radius` (the per-ligand Gaussian radius map is rebuilt with this single value for all ligands), `ligand_grid_factor` (grid vs exact received ligands during propagation; merges with the run's `[perturbation]` default), `contact_distance` (hard neighbor cutoff in distance units; `None` when omitted). When any of these differ from the loaded `PerturbRuntime`, initial weighted-ligand matrices are recomputed and the baseline splash cache is bypassed for that job so results stay consistent with the override.

### Perturbation physics (from TOML)

`[perturbation]` in `spaceship_config.toml`: `beta_scale_factor`, `n_propagation`, `ligand_grid_factor` (grid-approx vs exact received ligands), optional `beta_cap`, optional `perturbed_gene_min_bound` / `perturbed_gene_max_bound` (per-step clipping).

**Accuracy caveat:** simulated overexpression beyond roughly twice the highest observed expression for that gene produces aberrant fold changes and loses distance dependency. Keep `desired_expr` near the observed distribution, and consider `perturbed_gene_max_bound` as a guard.

---

## Quick reference commands

```bash
# Confirm the install and the compute backend
spacetravlr --verify

# Inspect an input before committing to a run
spacetravlr --peek /path/data.h5ad --obs cell_type

# Train with plain logs
spacetravlr --plain --config spaceship_config.toml --h5ad /path/data.h5ad --output-dir /path/run

# Full CNN + explicit genes + max ligands + join-friendly repro in output_dir
spacetravlr --plain --training-mode full --genes ACTB,MALAT1 --max-ligands 100 --parallel 8 --output-dir /path/run --h5ad /path/data.h5ad

# Join workers (repro must exist; omit conflicting hyperparameter flags)
spacetravlr --join-output-dir /path/run --parallel 16 --plain

# Batch perturbation (whole slide knockout)
spacetravlr-perturb --run-toml /path/run/spacetravlr_run_repro.toml --out /tmp/sim.feather --gene SOX2 --desired-expr 0 --n-propagation 4 --verbose

# Knockout restricted to one cell type (cell-extrinsic setup)
spacetravlr --make-cells-csv --run-toml /path/run/spacetravlr_run_repro.toml
spacetravlr-perturb --run-toml /path/run/spacetravlr_run_repro.toml --out /tmp/mif_ko.feather \
  --gene MIF --desired-expr 0 --cells-csv /path/run/cells.csv --cells-csv-column Epithelial

# Many single-gene jobs (batch TOML)
spacetravlr-perturb --run-toml /path/run/spacetravlr_run_repro.toml --batch-toml /path/batch_perturb.toml --verbose

# KO screen every TF / ligand / receptor
spacetravlr-perturb screen --config /path/ko_screen.toml --verbose

# Aggregate betas into an interaction table
spacetravlr collect-interactions --run-toml /path/run/spacetravlr_run_repro.toml

# HTML run summary (no training)
spacetravlr run-summary --config spaceship_config.toml --h5ad /path/data.h5ad --output-dir /path/run
```

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Scripted output garbled | Missing `--plain` |
| `{gene}.orphan` files | No usable modulators or failed `[training].score_threshold`. Lower the threshold, raise `max_ligands`, or check `[data].layer` exists and is non-zero. |
| Join host errors | Join hosts must not override repro hyperparameters. Pass only `--join-output-dir`, `--parallel`, `--plain`. |
| Locks left after a crash | Set `[execution].stale_lock_secs` (3600 is reasonable on NFS) |
| GPU unavailable / wgpu issues | `SPACETRAVLR_FORCE_CPU=1`; confirm with `spacetravlr --verify` |
| GRN parquet not found | Set `SPACETRAVLR_DATA_DIR` or `[grn].network_data_dir` |
| Out of memory | Lower `[execution].n_parallel`, `[cnn].cnn_max_cells_per_epoch`, `cnn_minibatch_size`, or `[spatial].spatial_dim` |
| Perturbation slow on large slides | Set `[perturbation].ligand_grid_factor` ≈ `0.5`; raise `--batch-parallelism` |

---

## Related

- Interactive spatial viewer + REST perturb APIs: `spatial_viewer` binary (feature `spatial-viewer`), documented in `.agents/skills/spatial-viewer-mcp/SKILL.md`
- Full reference for agents: `docs/llms-full.txt`, published at `https://spacetravlr-rust.readthedocs.io/en/latest/llms-full.txt`
