# IL21 / BCL6 ablation experiment — report

This document records a **reproducible ablation** on **human tonsil snRNA** (`snrna_human_tonsil.h5ad`, 5778 × 3333), training only **IL21** and **BCL6**, then running **knockout (KO)** perturbations under different GRN modulator masks and **seed vs full** training modes.

## What was run

| Setting | Meaning |
|--------|---------|
| **Modulator ablation** | `[grn]` toggles: `tf_only`, `tf_lr`, `tf_lr_ltfl`, `lr_only`, `ltfl_only` (see overlay TOMLs under `overlays/`). |
| **Spatial / beta form** | `seed` = cluster-level lasso betas; `full` = per-cell CNN path (12 epochs in this harness). |
| **Gene cap** | `max_ligands = 40` on DB LR pairs. |
| **Cluster key** | `cluster_annot = cell_type_int` so betadata `Cluster` keys match obs (string `cell_type` labels do **not** match numeric cluster ids and break perturb mapping). |
| **Perturbation** | `n_propagation = 4`; outputs under each run’s `perturb_feathers/` (see below). |

Training and large intermediates live under `runs/` (gitignored). Overlay configs and small analysis tables are in the repo.

## Saved KO feather files

All **KO-related** perturb outputs (single-gene KO, joint KO, and asymmetric rows where one gene is KO) are **hardlinked** into:

`experiments/il21_bcl6_ablation/ko_feathers_export/`

Naming: `{run_name}__{original_filename}.feather`  
Example: `tf_lr__seed__IL21_KO_perturb_expr.feather`

This directory is **gitignored** (large matrices; same bytes as `runs/.../perturb_feathers/`). To regenerate links:

```bash
EXP=experiments/il21_bcl6_ablation
KOEXP="$EXP/ko_feathers_export"
rm -rf "$KOEXP" && mkdir -p "$KOEXP"
for d in "$EXP"/runs/*; do
  [[ -d "$d/perturb_feathers" ]] || continue
  name=$(basename "$d")
  for f in "$d"/perturb_feathers/*KO*.feather; do
    [[ -f "$f" ]] || continue
    ln "$f" "$KOEXP/${name}__$(basename "$f")"
  done
done
```

## Branched pseudotime (notebook parity)

Alignment uses the same **Palantir branched pseudotime** as `embeds2perturb.ipynb` (cell 9):

- `annot = cell_type_2`
- `source_cell_type = Naive CD4 T`, `n_source_cells = 1`
- Pairs: `(Naive CD4 T, T_follicular_helper)`, `(Naive CD4 T, Th1)`, `(Naive CD4 T, Th2)`

Script: `scripts/compute_branched_pseudotime_notebook.py`  
Output: `analysis/branched_pseudotime_notebook.csv` (`obs_name`, `pseudotime`, **5778 rows** — cells outside the union of those subgraphs get pseudotime `0` after reindexing to full `obs`, matching `adata.obs.join` semantics).

**Dependency:** `pip install palantir` (ScanPy external).

## Alignment metric (Rust)

`spacetravlr-alignment` reads the h5ad baseline expression + UMAP, each perturb feather, and the branched pseudotime CSV; it computes the **VirtualTissue-style** transition field and cosine alignment vs the pseudotime gradient reference, then **mean alignment per `cell_type_2`** and a normal-approximation two-sided Wilcoxon vs a **fixed** random-delta null field.

KO-only manifest and results:

- `analysis/manifest_ko_only.csv`
- `analysis/alignment_ko_only_celltype2.csv`

## Other analysis artifacts

| File | Content |
|------|---------|
| `analysis/training_mean_lasso_r2.csv` | Mean lasso R² for IL21/BCL6 per run (from processed h5ad `var['mean_lasso_r2']`). |
| `analysis/l2_delta_summary.csv` | L2 norm of (perturbed − baseline) over **all genes** per feather. |
| `analyze_perturb_identity.py` | Sanity: compares `IL21_KO_perturb_expr.feather` across runs (max abs diff vs first). |

## Results (high level)

### 1. Training quality differs by modulator mask

From `training_mean_lasso_r2.csv`, **LR-containing** settings (`tf_lr`, `tf_lr_ltfl`, `lr_only`) give **much higher** mean lasso R² for IL21/BCL6 than **TF-only**. **`ltfl_only`** yields **NaN** R² (orphan targets: no usable modulators for these genes in that mode).

So the **betadata / lasso fit** does respond to the GRN ablation switches.

### 2. Forward KO perturbations were identical across all ablations (this panel)

For every run and every KO scenario checked, **`IL21_KO_perturb_expr.feather` is byte-identical** (and `analyze_perturb_identity.py` reports **max abs diff 0** vs the first run). The same holds for **BCL6 KO** and for **seed vs full** on this two-gene training setup.

**Interpretation:** with only **IL21** and **BCL6** trained, the **propagated in-silico expression** after KO does **not** separate TF vs LR vs LR+TFL in this experiment. Use **training metrics / betas** for ablation conclusions here, or expand the trained gene set so perturbation trajectories can diverge.

### 3. UMAP alignment (KO-only, `cell_type_2`, branched pseudotime)

For **IL21 single KO** on **`T_follicular_helper`**, **all 10 runs** produced the **same** `mean_alignment` (example value ≈ **−0.457** in this run), because the underlying perturb matrices are identical.

Wilcoxon *p*-values vs the random null are still computed per cell type; where distributions are identical, *p* can collapse to **0** or **1** depending on ties — treat **effect size (mean_alignment)** as the primary readout when perturb fields are identical.

## Reproduce end-to-end

```bash
cd /path/to/SpaceTravLR_rust
export SPACETRAVLR_FORCE_CPU=1   # optional

cargo build --release --bin spacetravlr --bin spacetravlr-perturb --bin spacetravlr-alignment

# Full train + perturb + analysis (long; writes runs/)
bash experiments/il21_bcl6_ablation/run_pipeline.sh
```

Or stepwise: `generate_overlays.py` → train each overlay → perturb jobs → `compute_branched_pseudotime_notebook.py` → `spacetravlr-alignment` → `summarize_deltas.py` / `collect_training_metrics.py`.

## Caveats

1. **Palantir** is required for notebook-matched pseudotime; the Rust alignment binary invokes `python3` for that step unless you pass `--pseudotime-csv`.
2. **Cell counts**: tonsil has **5778** cells; branched subgraphs cover **612** cells before full-obs reindex; remaining cells have pseudotime **0**.
3. **Publication claims** about “LR improves perturb biology” should **not** rest on these identical KO feathers alone without retraining a broader target panel or inspecting `splash` / intermediate states.

---

*Generated as part of the `experiments/il21_bcl6_ablation` harness; large `runs/` and `ko_feathers_export/` are not committed to git.*
