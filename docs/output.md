# Output files

SpaceTravLR writes artifacts under **`[execution].output_dir`** (or a dated default next to your `.h5ad`). The canonical pointer for downstream tools is **`spacetravlr_run_repro.toml`** in that directory — it records the effective config after CLI overrides and is what **`--join-output-dir`**, **`spatial_viewer`**, and **`spacetravlr-perturb`** load.

---

## Training run layout

Typical single-run tree (no condition split):

```
output_dir/
├── spacetravlr_run_repro.toml    # reproducible configs
├── spacetravlr_run_summary.html  # HTML training summary
│
├── GENE_betadata.feather         # cell x modulator coefficients
├── OTHER_GENE_betadata.feather
├── …
│
├── GENE.lock                 # transient: gene in progress
├── FAILED_GENE.orphan        # no usable modulators / failed gate (no feather)
├── SOME_GENE.tf_ablated      # TF modulators off or all-zero TF support
│
├── log/
│   ├── GENE.log              # per-gene training metrics
│   └── …
│
├── CNN_weights/              # only if [model_export].save_cnn_weights = true
│   ├── GENE_cnn_weights.npz
│   └── …
```

With **`[data].condition`** (or `--condition`), each group trains under a subdirectory; the repro TOML stays at the **parent** `output_dir/`:

```
output_dir/
├── spacetravlr_run_repro.toml
├── spacetravlr_run_summary.html
│
└── conditions/
    ├── batch_A/                        # sanitized obs value (folder name)
    │   ├── condition_label.txt         # human-readable condition value
    │   ├── GENE_betadata.feather
    │   ├── GENE.lock
    │   ├── log/GENE.log
    │   └── …
    └── batch_B/
        └── …
```

**Multi-host training:** additional machines use `spacetravlr --join-output-dir output_dir`. They read the same repro TOML, claim genes via `{gene}.lock`, and write feathers into the same tree (or into `conditions/<group>/` when splitting). Stale locks can be removed after `[execution].stale_lock_secs`.

---

## `{gene}_betadata.feather`

Primary training product: learned coefficients for predicting target gene **`GENE`** from modulators (TFs, ligand–receptor pairs, TF–ligand links, extras).

| Column | Meaning |
|--------|---------|
| **`Cluster`** or **`CellID`** | Row key. **Seed-only** (`[training].mode = seed`): one row per cluster label from `[data].cluster_annot`. **Full / CNN** (`mode = full`): one row per cell (`obs_names`). |
| **`beta0`** | Intercept. |
| **`beta_<name>`** | Coefficient for modulator `<name>`. Naming follows the GRN: plain symbol = TF; **`LIG$REC`** = ligand–receptor; **`TF#LIG`** = TF–ligand (NicheNet-style). |

Feather files use Arrow IPC with LZ4 compression. The spatial viewer and perturbation tools join rows to AnnData cells via **`Cluster`** (cluster id / `cell_type` name) or **`CellID`** (per-cell export).

**Markers when no feather is written:**

| File | Meaning |
|------|---------|
| `{gene}.orphan` | Empty file: gene skipped (no modulators, failed Lasso gate, all-zero betas, etc.). |
| `{gene}.tf_ablated` | TF channel ablated; no standard feather (distinct from generic orphan). |
| `{gene}.lock` | Training in progress on this host (removed when done). |

---

## `log/{gene}.log`

Tab-separated training log (`format spacetravlr_training_log v1`) parsed by **`spacetravlr run-summary`** and the HTML report:

- Global: `seed_only`, `per_cell_cnn_export`, CNN epochs, learning rate, Lasso iterations/tolerance.
- Per cluster: `lasso_r2`, FISTA iters, convergence, CNN epoch MSE trace, in-sample `cnn_r2`.

---

## `spacetravlr_run_repro.toml`

Snapshot of **`SpaceshipConfig`** as executed (paths, layer, cluster column, GRN limits, Lasso/CNN hyperparameters, gene list for join hosts). **Do not** pass a different `--genes` / `--max-genes` on join than what is stored here.

Key fields for finding other outputs:

| Section | Field | Points to |
|---------|-------|-----------|
| `[data]` | `adata_path`, `layer`, `cluster_annot` | Input AnnData |
| `[execution]` | `output_dir` | Directory containing `*_betadata.feather` |
| `[model_export]` | `output_subdir` | CNN weight subdirectory (default `CNN_weights`) |

---

## Optional CNN exports

When **`[model_export].save_cnn_weights = true`** (or `--save-cnn-weights`):

```
output_dir/CNN_weights/   # or [model_export].output_subdir
└── GENE_cnn_weights.npz  # Burn CNN state for that gene (per-cluster models bundled)
```

Optional parity dumps (`write_cnn_train_data_npz`): `{gene}_cnn_train_data.npz` and `{gene}_cnn_train_meta.json` for external Python reference training.

---

## `collect-interactions` output

Aggregates β across cells / cell types from all `*_betadata.feather` under a run:

```bash
spacetravlr collect-interactions \
  --run-toml output_dir/spacetravlr_run_repro.toml \
  --out output_dir/plucked_feathers.feather   # default if --out omitted
```

Default output: **`plucked_feathers.feather`** in the run directory. Columns include **`interaction`**, **`target_gene`**, **`beta`**, **`interaction_type`** (`tf` | `ligand-receptor` | `ligand-tf`), and **`cell_type`** (from `--annot`, default `cell_type`).

---

## Perturbation outputs

Perturbation does **not** add files to the training tree unless you export. It reads **`spacetravlr_run_repro.toml`** + training feathers and writes **simulated expression** matrices.

### Single-gene batch (`spacetravlr-perturb --export`)

```
/path/you/choose.feather          # or --out (same flag)
```

| Column | Meaning |
|--------|---------|
| **`CellID`** | `obs_names` for each row (all cells or CSV-scoped subset). |
| **`<gene>`** | Simulated expression after perturbation for every gene in the AnnData. |

Default batch name pattern: **`{gene}_perturb_expr.feather`** when using `--batch-toml` with `out_dir`.

---

## Quick reference: what to open

| Goal | File / directory |
|------|------------------|
| Resume training on another machine | `spacetravlr_run_repro.toml` + `--join-output-dir` |
| Inspect coefficients for one gene | `{gene}_betadata.feather` |
| Browse spatially in the UI | `spacetravlr_run_repro.toml` + same `adata_path` as in repro |
| Run in-silico KO / OE | `spacetravlr-perturb --run-toml … --export …` |
| Screen interactions across cell types | `plucked_feathers.feather` (after `collect-interactions`) |
| Audit training quality | `spacetravlr_run_summary.html` or `log/{gene}.log` |
