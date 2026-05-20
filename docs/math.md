# How it works

SpaceTravLR learns spatial coefficients for each cell for each gene using single-cell spatial transcriptomics.
These coefficients are then used to compute gene-gene partial derivatives to allow perturbations to propagate both through the gene regulatory network (cell intrinsict effects) and spatially via cell-cell communication (cell extrinsict effects).

---

## Step 1 — Computing received ligands

Given \(N\) cells at coordinates \(\mathbf{x}_i \in \mathbb{R}^2\) and ligand
expression \(L_{jk}\) (cell \(j\), ligand \(k\)), the "received ligand" for
cell \(i\) is an isotropic Gaussian kernel average:

\[
\widetilde{L}_{ik}
  = \frac{s}{N}\sum_{j=1}^{N}
    \exp\!\Bigl(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2r^2}\Bigr)
    \, L_{jk}
\]

where \(r\) is a learned-per-LR-pair or configured radius and \(s\) is a
global scale factor. An optional hard cutoff \(d_{\max}\) drops terms where
\(\|\mathbf{x}_i - \mathbf{x}_j\| > d_{\max}\).

```rust
// src/ligand.rs — O(N²) kernel, parallelized over receiver cells
result.axis_iter_mut(Axis(0))
    .into_par_iter()
    .enumerate()
    .for_each(|(i, mut row)| {
        let xi = xy[[i, 0]];
        let yi = xy[[i, 1]];
        for j in 0..n_cells {
            let dx = xi - xy[[j, 0]];
            let dy = yi - xy[[j, 1]];
            let d2 = dx * dx + dy * dy;
            if d2_cut.is_some_and(|c| d2 > c) { continue; }
            let w = scale_factor * (d2 * inv_2r2).exp();
            for k in 0..n_ligands {
                row[k] += w * lig_values[[j, k]];
            }
        }
        for k in 0..n_ligands { row[k] *= n_inv; }
    });
```

---

## Step 2 — Spatial Proximity maps

Each cell \(s\) needs a fixed-size pixel representation of its spatial
context so the Convolutional Neural Network can learn location-specific coefficients. The spatial proximity grid is a 4D tensor of shape
\((N,\, C,\, m,\, n)\) — one \(m \times n\) inverse-distance map per
cluster channel per cell.

The bounding box of all cell coordinates is divided into
an \(m \times n\) regular grid. Each pixel \((i, j)\) has a centre:

\[
g^x_j = x_{\min} + \bigl(j + \tfrac{1}{2}\bigr)\,\Delta_x,
\qquad
g^y_i = y_{\max} - \bigl(i + \tfrac{1}{2}\bigr)\,\Delta_y
\]

where \(\Delta_x = (x_{\max} - x_{\min})/n\) and
\(\Delta_y = (y_{\max} - y_{\min})/m\).

Each pixel stores the reciprocal Euclidean distance from the
cell to the pixel centre:

\[
\mathbf{S}_{s,\,c_s,\,i,\,j}
  = \frac{1}{\max\!\bigl(\sqrt{(x_s - g^x_j)^2 + (y_s - g^y_i)^2},\;\epsilon\bigr)}
\]

with \(\epsilon = 10^{-6}\) preventing division by zero. This produces a
smooth radial falloff centred on the cell's position, encoding both the local and global tissue structure.

```rust
//src/model.rs

let mut spatial_maps = Array4::<f32>::zeros((num_cells, num_clusters, m, n));

spatial_maps
    .axis_iter_mut(Axis(0))
    .into_par_iter()
    .enumerate()
    .for_each(|(s, mut cell_maps)| {
        let cluster_s = clusters[s];
        if cluster_s >= num_clusters {
            return;
        }
        let x_s = xy[[s, 0]] as f32;
        let y_s = xy[[s, 1]] as f32;
        if !x_s.is_finite() || !y_s.is_finite() {
            return;
        }

        let cx_ego = ego_center.then(|| {
            let half_x = span_x * 0.5;
            (0..n)
                .map(|j| x_s - half_x + (j as f32 + 0.5) * cell_width)
                .collect::<Vec<f32>>()
        });
        let cy_ego = ego_center.then(|| {
            let half_y = span_y * 0.5;
            let top_y = y_s + half_y;
            (0..m)
                .map(|i| top_y - (i as f32 + 0.5) * cell_height)
                .collect::<Vec<f32>>()
        });
        let cx_grid: &[f32] = cx_ego.as_deref().unwrap_or(&cx_global);
        let cy_grid: &[f32] = cy_ego.as_deref().unwrap_or(&cy_global);
    });

```

---

## Step 3 — Sparse Group Lasso
SpaceTravLR uses a custom sparse group lasso implementation written in pure Rust.

For each target gene \(g\) and each cell type \(c\), a regularized linear model predicts the spatial
expression profile from the modulator design matrix \(\mathbf{X}_c\) (columns =
TFs + received ligands × receptors + TF–ligand terms):

\[
\hat{\boldsymbol{\beta}}^{(g)}_c
  = \arg\min_{\boldsymbol{\beta}}
    \frac{1}{2n_c}
    \bigl\|\mathbf{y}_c - \mathbf{X}_c \boldsymbol{\beta}\bigr\|_2^2
    + \lambda_1 \|\boldsymbol{\beta}\|_1
    + \lambda_G \sum_{\ell} \|\boldsymbol{\beta}_{G_\ell}\|_2
\]

The \(\ell_1\) sparsifies within groups; the group norm \(\|\cdot\|_2\)
drops entire regulator groups \(G_\ell\) jointly (e.g., all LR pairs for
one receptor). Optimization is done using the FISTA algorithm with proximal l1/l2 steps. Fit quality
is measured by:

\[
R^2_c = 1 - \frac{\sum_{i\in c}(y_i - \hat{y}_i)^2}{\sum_{i\in c}(y_i - \bar{y}_c)^2}
\]

To prevent bad coefficients from polluting the network, cell types below a threshold performance are masked; with their coefficients set to zero.

---

## Step 4 — Convolutional Neural Networks

In `full` mode, clusters that pass the Lasso \(R^2\) gate are refined with a
**CellularNicheNetwork**: a small CNN on inverse-distance spatial maps plus an MLP
over cluster context features. The network outputs **effective coefficients**
\(\boldsymbol{\beta}_i\) per cell by scaling Lasso **anchors**; expression is then
read out linearly from modulators \(\mathbf{x}_i\):

\[
\hat{y}_i = \beta_{i,0} + \sum_{j>0} \beta_{i,j}\, x_{i,j}.
\]

### Loss function

Training minimizes **mean squared error (MSE)** on the target expression layer
(default `imputed_count`), plus optional weak terms that keep the CNN near Lasso:

\[
\mathcal{L}
  = \underbrace{\mathrm{MSE}(\hat{y}, y)}_{\text{primary}}
  + \lambda_{\mathrm{prior}}\,
    \mathrm{MSE}\!\left(\mathbb{E}_{\text{batch}}[\boldsymbol{\beta}],\,
      \boldsymbol{\beta}^{\mathrm{lasso}}\right)
  + \lambda_{\mathrm{align}}\,
    \mathrm{MSE}(\hat{y}, \hat{y}^{\mathrm{lasso}}).
\]

| Term | Config key | Default |
|------|------------|---------|
| Primary fit | — | always on |
| Mean-β prior | `mean_beta_lasso_prior_weight` | off (`null`) |
| Lasso alignment | `lasso_pred_align_weight` | `0.05` (ramps down if `lasso_pred_align_linear_decay`) |

Optimization uses **Adam** on CNN weights. After training, in-sample \(R^2\) is
compared to Lasso; the CNN export is dropped when it loses to Lasso unless
`drop_cnn_if_insample_worse_than_lasso` is disabled (see [Parameters](params.md)).

```rust
// src/estimator.rs — CNN training loss (Burn MseLoss, mean reduction)
let y_pred = CellularNicheNetwork::linear_readout_y(betas.clone(), b.x_tensor);
let y_loss = mse_loss.forward(y_pred.clone(), b.y_tensor, Reduction::Mean);
let mut total = y_loss.clone();
// optional: prior on batch-mean betas vs anchors; align y_pred to y_lasso
```

---

## Step 5 — Perturbation propagation

Given trained betadata \(\hat{\boldsymbol{\beta}}\), a perturbation sets
gene \(g^\star\) to a desired level and propagates through the GRN:

\[
\Delta^{(0)}_{c,g} =
  \begin{cases}
    y^{\mathrm{des}}_c - y^{\mathrm{base}}_c & g = g^\star \\
    0 & \text{otherwise}
  \end{cases}
\]

Each propagation iteration \(t = 1 \ldots T\) has three substeps.

### 5a. `Splash` — local Jacobian of the GRN
This function is affectionately named `splash` for the ligands from neighbors splashing on the cells.

For each trained gene, partial derivatives of predicted expression w.r.t.
modulators form a sparse per-cell matrix \(J\). The rules (stored
in betadata) include TF, LR, and TF–ligand channels:

\[
\frac{\partial \hat{y}_g}{\partial x_{\mathrm{TF}}} = \beta_{\mathrm{TF}},
\quad
\frac{\partial \hat{y}_g}{\partial x_R}
  = \beta_{\mathrm{LR}} \cdot \widetilde{L} \cdot \mathbb{1}[x_R > 0] \cdot s,
\quad
\frac{\partial \hat{y}_g}{\partial x_L}
  = \beta_{\mathrm{LR}} \cdot x_R \cdot s
\]

```rust
// src/betadata.rs — splash, row-parallel, flat indexed, L1-hot
result.par_chunks_mut(n_out).enumerate().for_each(|(i, r)| {
    let beta_row = map[i];
    // TF: direct copy
    for j in 0..n_tfs {
        r[tf_oi[j]] += tf_flat[beta_row * n_tfs + j];
    }
    // LR: β * wL * s (receptor), β * gex_R * s (ligand)
    for lw in &lr_work {
        let beta = lr_flat[beta_row * n_lr + lw.beta_col];
        let wl   = rw_flat[i * rw_nc + lw.wl_col];
        let gex  = gex_flat[i * gex_nc + lw.gex_col];
        if gex > 0.0 { r[lw.rec_oi] += beta * wl * scale_factor; }
        r[lw.lig_oi] += beta * gex * scale_factor;
    }
});
```

### 5b. Recompute spatial ligands and adjust Δ

Updated expression \(\mathbf{y}^{(t)} = \mathbf{y}^{\mathrm{base}} + \boldsymbol{\Delta}^{(t)}\)
feeds back into the Gaussian kernel in Step 1 to get new received ligands
\(\widetilde{\mathbf{L}}^{(t)}\). The delta bookkeeping swaps direct ligand
expression change for the received-ligand change (same logic as Python
`GeneFactory.perturb`):

\[
\Delta_{c,\ell} \;\leftarrow\; \Delta_{c,\ell}
  + \bigl(\widetilde{L}^{(t)}_{c,\ell} - \widetilde{L}^{(0)}_{c,\ell}\bigr)
  - \bigl(y^{(t)}_{c,\ell} - y^{\mathrm{base}}_{c,\ell}\bigr)
\]

### 5c. Linear push + nonneg projection

The sparse multiply propagates modulator deltas to target expression:

\[
\Delta^{(t+1)}_{c,g}
  = \sum_{k=1}^{M_g} J^{(t)}_{c,g,k} \cdot \Delta^{(t)}_{c,\,\mathrm{mod}_k(g)}
\]

followed by pinning targets and projecting to \(\mathbb{R}_{\ge 0}\):

```rust
// src/perturb.rs — inner product per gene per cell, fully parallel
out_flat.par_chunks_mut(n_genes).enumerate().for_each(|(cell, r)| {
    let delta_base = cell * n_genes;
    for w in &work {
        let splash_base = cell * w.n_mods;
        let mut sum = 0.0f64;
        for k in 0..w.n_mods {
            sum += f64::from(w.splash_flat[splash_base + k])
                 * delta_flat[delta_base + w.mod_indices[k]];
        }
        r[w.gene_col] = sum;
    }
});

// nonneg clip: simulated = max(base + Δ, 0), then fold back
delta_flat.par_chunks_mut(n_genes).enumerate().for_each(|(cell, row)| {
    for gene in 0..n_genes {
        let orig = base_flat[cell * n_genes + gene];
        row[gene] = (orig + row[gene]).max(0.0) - orig;
    }
});
```

---

## Why Rust

SpaceTravLR was ported from Python/NumPy to Rust because the inner loops
(Gaussian kernel, splash multiply, FISTA proximal steps) dominate runtime. 


### Zero-cost parallelism (Rayon)

Every cell-level computation (`into_par_iter` / `par_chunks_mut`) is data-parallel
over cells with no GIL, no multiprocessing overhead, no pickle serialization.
The thread pool scales to available cores automatically:

```
Python (GIL-bound)             Rust (Rayon work-stealing)
┌────────────────┐             ┌────────────────────────────────┐
│ thread 0       │             │ thread 0 ■■■■■                 │
│ ████████████   │             │ thread 1 ■■■■■                 │
│                │             │ thread 2 ■■■■■                 │
│ (GIL blocks    │             │ thread 3 ■■■■■                 │
│  all others)   │             │ thread 4 ■■■■■                 │
│                │             │ thread 5 ■■■■■                 │
│                │             │ ...                            │
│                │             │ thread N ■■■■■                 │
│  wall: T       │             │  wall: T/(N·η)                 │
└────────────────┘             └────────────────────────────────┘
```

The splash multiply, spatial kernel, nonneg projection, and betadata
expansion all use this pattern with zero cost synchronization.

### Cache-friendly flat layouts

NumPy stores ndarray as a single buffer but Python iteration adds
per-element dispatch. In Rust, inner loops index flat `&[f64]` slices
directly with known strides, keeping data in L1/L2 cache throughout a row:

```
Memory layout for splash (n_cells × n_mods, row-major f32):
┌───────────────────────────────────────────────────────┐
│ cell 0  [mod0 mod1 mod2 ... modM] ◄─ fits in L1 line  │
│ cell 1  [mod0 mod1 mod2 ... modM]                     │
│ ...                                                   │
│ cell N  [mod0 mod1 mod2 ... modM]                     │
└───────────────────────────────────────────────────────┘
                       │
       ┌───────────────┘
       ▼
  for k in 0..n_mods {
      sum += splash_flat[cell * n_mods + k]    ◄─ sequential read
           * delta_flat[cell * n_genes + mod_idx[k]];
  }
```

Compare to Python:

```python
# Python equivalent — 3 layers of indirection per element
for k in range(n_mods):
    sum += splash_df.iloc[cell, k] * delta_df.iloc[cell, mod_idx[k]]
```

Even vectorized NumPy (`np.einsum`, `@`) cannot express the per-gene
sparse-columns structure without materializing a dense 3D tensor
or falling back to a Python loop over genes.

### Grid-accelerated ligand computation

The naïve \(O(N^2)\) kernel dominates for large datasets. The grid
approximation reduces this to \(O(A \cdot N)\) where \(A \ll N\) is the
number of anchor points, followed by bilinear interpolation:

```
Exact kernel: O(N²)              Grid + interpolation: O(A·N + N)
                                 ┌───────────────────────────────────┐
  every cell ──► every cell      │  grid anchors ──► every cell      │
  N² pairs, N > 15k → slow       │  A = (span/h)², h = r·factor      │
                                 │                                   │
                                 │  each anchor:  O(N) Gaussian sum  │
                                 │  each cell:    bilinear from 4    │
                                 │                anchor neighbors   │
                                 │                                   │
                                 │  total: O(A·N·L + N·L)            │
                                 │  error: O(h²/r²) ≈ 3% at h=r/2    │
                                 └───────────────────────────────────┘
```

```rust
// src/ligand.rs — grid anchor computation (parallel over anchors)
anchor_vals.par_chunks_mut(n_ligands).enumerate().for_each(|(a, row)| {
    let ax = x_min + (a % nx) as f64 * grid_spacing;
    let ay = y_min + (a / nx) as f64 * grid_spacing;
    for j in 0..n_cells {
        let d2 = (ax - xy[[j,0]]).powi(2) + (ay - xy[[j,1]]).powi(2);
        if d2_cut.is_some_and(|c| d2 > c) { continue; }
        let w = scale_factor * (d2 * inv_2r2).exp();
        for (slot, &lv) in row.iter_mut().zip(&lig_flat[j*n_ligands..(j+1)*n_ligands]) {
            *slot += w * lv;
        }
    }
    for v in row.iter_mut() { *v *= n_inv; }
});

// bilinear interpolation to cell positions (parallel over cells)
res_flat.par_chunks_mut(n_ligands).enumerate().for_each(|(i, row)| {
    let fx = (xy[[i,0]] - x_min) / grid_spacing;
    let fy = (xy[[i,1]] - y_min) / grid_spacing;
    let (gx0, gy0) = (fx.floor() as usize, fy.floor() as usize);
    let (w00, w10, w01, w11) = bilinear_weights(fx - gx0 as f64, fy - gy0 as f64);
    for k in 0..n_ligands {
        row[k] = w00 * anchors[idx00 + k]
                + w10 * anchors[idx10 + k]
                + w01 * anchors[idx01 + k]
                + w11 * anchors[idx11 + k];
    }
});
```

### WebGPU CNN backend (Burn)

CNN training and inference run on the GPU via Burn's WebGPU backend when
available, falling back to NdArray on CPU. No CUDA install is required —
the same binary works on Metal (macOS), Vulkan (Linux), and DX12 (Windows):

```rust
// src/bin/compute_backend.rs — runtime backend selection
pub(crate) fn select_compute_backend() -> ComputeChoice {
    if env_truthy("SPACETRAVLR_FORCE_CPU") {
        ComputeChoice::NdArray(NdArrayDevice::Cpu)
    } else {
        match wgpu_adapter_probe_cached() {
            Some(_) => ComputeChoice::Wgpu(WgpuDevice::default()),
            None    => ComputeChoice::NdArray(NdArrayDevice::Cpu),
        }
    }
}
```

### Unsafe hot paths with safety invariants

The innermost loops (splash, nonneg clip) use `unsafe` to elide bounds
checks on indices known to be valid from construction. This is roughly
10–15% faster on the tight per-cell loops because the compiler can
vectorize without branch-per-element. Safety invariants are guaranteed
by array shape checks at entry:

```rust
// Bounds are proven: gene_mtx is (n_cells × n_genes), delta is same shape,
// and all mod_indices[k] < n_genes (validated at Betabase load).
unsafe {
    let orig = *base_flat.get_unchecked(cell * n_genes + gene);
    let val  = (orig + *row.get_unchecked(gene)).max(0.0);
    *row.get_unchecked_mut(gene) = val - orig;
}
```

### Summary: Python → Rust wall-time

| Operation | Python (NumPy) | Rust | Speedup |
|-----------|---------------|------|---------|
| Spatial kernel (N=20k, exact) | ~12 s | ~0.8 s | 15× |
| Spatial kernel (N=20k, grid) | — | ~0.05 s | 240× vs Python exact |
| Perturbation propagation (4 iters) | ~45 s | ~3 s | 15× |
| Full training (200 genes, 1 host) | hours | minutes | 10–30× |

(Numbers vary with hardware; Rayon utilizes all cores while Python's NumPy
is often single-threaded in the inner loops that SpaceTravLR hits.)
