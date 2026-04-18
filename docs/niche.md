# `spacetravlr-niche` — CNN microniche detection from per-cell splash

`spacetravlr-niche` is a new [`[[bin]]`](../Cargo.toml) and library module
that learns a per-cell **microniche embedding** from the per-cell *gene × gene*
splash Jacobian produced by a finished SpaceTravLR training run. The
embeddings are then clustered with k-means to assign every cell to one of
`k` microniches.

The whole pipeline is implemented in Rust, runs on CPU (Burn `NdArray`) or
WebGPU (Burn `Wgpu`), and ships with an in-memory synthetic generator so it
can be benchmarked without any external `.h5ad` data.

---

## Why splash images?

For each trained target gene `g`, [`compute_splash_all`](../src/perturb.rs)
produces a `(n_cells × n_modulators_g)` matrix whose `[c, m]` entry is the
partial derivative of `E[g, c]` with respect to modulator gene `m` evaluated
at the current expression + spatial-ligand state:

```
∂E[g] / ∂TF      = β_TF
∂E[g] / ∂R_lr    = β_LR · wL[L]      (where gex[R] > 0)
∂E[g] / ∂L_lr    = β_LR · gex[R]
∂E[g] / ∂L_tfl   = β_TFL · gex[reg]
∂E[g] / ∂TF_tfl  = β_TFL · wL_tfl
```

Stacking those matrices over all trained targets gives, for every cell, a
*per-cell* `(n_targets × n_modulators)` Jacobian. The columns therefore
encode **how each cell would respond to a perturbation of every modulator
in its current spatial signalling environment** — which is exactly the
information needed to define a *functional* microniche.

[`spacetravlr::niche::image::build_niche_image_stack`](../src/niche/image.rs)
takes the dictionary returned by `compute_splash_all` and pads every
target's modulator alphabet to the global union (sorted), so each cell ends
up with a fixed-shape matrix that the CNN can consume as a 1-channel image.

---

## Architecture (`src/niche/model.rs`)

`NicheEncoder` is a 3-block CNN backbone followed by an MLP that projects
to a `D`-dim embedding, plus three heads that train at the same time:

| Head           | Purpose                                                                                                    |
|----------------|------------------------------------------------------------------------------------------------------------|
| `embed`        | The `D`-dim niche embedding used for k-means clustering downstream.                                        |
| `functional`   | Predicts a per-cell **program activity vector** (see below). Forces the embedding to encode *which programs are active*. |
| `recon`        | Reconstructs a low-rank avg-pooled summary of the input image. Forces the embedding to retain Jacobian information. |
| `projection`   | A small projection used by the spatial-coherence loss (cosine similarity between a cell and its neighbours). |

The composite loss is

```
L_total = lambda_recon * MSE(recon, avg_pool(input))
        + lambda_func  * MSE(functional, program_activity)
        + lambda_spatial * (1 - cos(z_i, mean_j∈N(i) z_j))
```

with `lambda_recon=1.0`, `lambda_func=2.0`, `lambda_spatial=1.0` by default.

### How the functional head's targets are built

Without supervision we extract "programs" by k-means on **modulator
co-activity profiles** across cells × targets (every modulator gets a
hard program assignment). The functional target for a cell `c` is then

```
program_activity[c, p] = Σ_{t,m: P[m]=p} |J[c, t, m]|     (then L1-normalized per cell)
```

so the head is asking "tell me which programs this cell's signalling
neighbourhood is firing". That is the property a *functional* niche must
have, and it is the regularizer that pushes the embedding to be more than
just spatially smooth.

---

## CLI

`spacetravlr-niche` has two subcommands.

### From a trained run

```
spacetravlr-niche from-run \
    --run-toml /path/run/spacetravlr_run_repro.toml \
    --out-dir  /path/run/niche \
    --n-clusters 12 \
    --epochs 60
```

Output (under `--out-dir`):

* `niche_labels.feather` — `CellID, niche, z00…zD-1` (Polars/Arrow IPC, LZ4)
* `niche_labels.csv`     — same content, plain CSV

### Synthetic — known microniches

```
spacetravlr-niche synthetic \
    --cells-per-niche 80 \
    --n-niches 5 \
    --out-dir /tmp/synth_niches \
    --n-clusters 5
```

Adds `niche_metrics.json` with ARI / NMI / spatial purity vs the known
ground-truth, plus per-epoch loss summaries.

### Compute backend

`spacetravlr-niche` uses the same `Wgpu` / `NdArray` selection convention as
the rest of the toolchain:

* `SPACETRAVLR_FORCE_CPU=1` or `SPACETRAVLR_DISABLE_WGPU=1` forces CPU
* otherwise `Wgpu` is used when the wgpu adapter probe succeeds.

---

## Synthetic benchmark

[`tests/test_niche.rs`](../tests/test_niche.rs) builds a 400-cell × 5-niche
synthetic where:

* every cell is one of 3 cell types (TFs + receptors driven by cell type)
* each niche owns its own 4 LR pairs (ligands fire **only** inside that niche)
* cell type is **independent of niche** — so raw expression k-means
  cannot recover niche

Three baselines are included for comparison:

| baseline             | features                                                |
|----------------------|---------------------------------------------------------|
| `expression_kmeans`  | k-means on log1p(expression)                            |
| `banksy_like`        | k-means on `[expr, mean(expr over k spatial NN)]`       |
| `splash_pca`         | k-means on the top-16 PCs of the flattened splash image |

Result on the synthetic (`SPACETRAVLR_FORCE_CPU=1`, 30 epochs, ~7 min):

| method                | ARI       | NMI       | spatial purity (k=10) |
|-----------------------|-----------|-----------|------------------------|
| `expression_kmeans`   | 0.095     | 0.246     | 0.323                  |
| `banksy_like`         | 0.088     | 0.247     | 0.325                  |
| `splash_pca`          | 0.114     | 0.266     | 0.350                  |
| **`cnn_niche`**       | **0.975** | **0.969** | **0.965**              |

Functional fidelity (max per-cluster variance of dominant-program index)
is asserted to be `< 1.0` so each predicted niche concentrates on a single
signalling program — i.e. the niches are *functional*, not just spatially
contiguous.

---

## File layout

```
src/niche/
├── mod.rs            # public re-exports
├── image.rs          # build_niche_image_stack(..) + standardization modes
├── model.rs          # NicheEncoder + NicheHeads (Burn modules)
├── train.rs          # composite-loss training loop + program-membership k-means
├── kmeans.rs         # tiny pure-Rust k-means++ for label assignment
├── metrics.rs        # ARI / NMI / spatial purity (k-NN)
├── synth.rs          # in-memory synthetic spatial run with known niches
├── runtime.rs        # high-level orchestrator
└── io.rs             # Feather + CSV writers for niche labels + embeddings
src/bin/spacetravlr_niche.rs   # CLI binary
tests/test_niche.rs            # end-to-end benchmark vs baselines
```
