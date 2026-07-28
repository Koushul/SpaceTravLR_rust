# CellChat × SpaceTravLR hybrid

CellChat (Jin et al., *Nat Commun* 2021) estimates a **mean-field** communication probability \(P_{i\to j}^k\) from group-averaged ligand/receptor abundance with a Hill / mass-action kernel. SpaceTravLR instead regresses **per-cell** coefficients on spatially received ligand × receptor features.

This hybrid runs CellChat-style probabilities first, then builds **spatially local** LR design columns for the usual sparse group Lasso.

## Why not use \(P\) alone as Lasso features?

SpaceTravLR fits **per cluster**. A feature that is constant within a cluster (group-level \(P\)) is absorbed by the intercept. Probabilities therefore **select and weight** spatial LR terms rather than replace them.

## Pipeline

1. **CellChatDB** — load `cellchat_{mouse|human}.csv` (complexes kept as multi-subunit).
2. **Expression prep** — per-group **trimean** \((Q_1+2Q_2+Q_3)/4\); drop groups with `< min_cells`.
3. **Probability** — for ligand complex \(L_i\) and receptor complex \(R_j\) (geometric means of subunits):

\[
P_{i\to j}^k \propto \frac{(L_i R_j)^n}{K_h^n + (L_i R_j)^n}
\]

Optional label-permutation p-values (`n_perm > 0`).

4. **LR terms** (`[cellchat].lr_mode`):

| Mode | Per-cell feature |
|------|------------------|
| `weighted_spatial` (default) | \(X_{c,k}=\sum_s P_{s\to\mathrm{type}(c)}^k\cdot\widetilde{L}_{c\leftarrow s}^{(k)}\cdot R_c^{(k)}\) |
| `hill_spatial` | \(X_{c,k}=(\widetilde{L}_c R_c)/(K_h+\widetilde{L}_c R_c)\) on CellChat-selected pairs |
| `spatial_product` | Classic \(\widetilde{L}_c\cdot R_c\), pair set from CellChat filters |

\(\widetilde{L}_{c\leftarrow s}\) is the Gaussian received-ligand field using **only** cells of type \(s\).

5. **Lasso** — same sparse group Lasso / optional CNN path as usual.

## Config

```toml
[cellchat]
enabled = true
lr_mode = "weighted_spatial"
kh = 0.5
hill_coef = 1.0
min_cells = 10
n_perm = 0
replace_lr_pairs = true
max_interactions = 200
# db_path = "data/cellchat_mouse.csv"
# signaling_types = ["Secreted Signaling"]
```

When `enabled = true`, training writes `cellchat_commun_prob.csv` under the run output directory and replaces (or weights) LR modulators accordingly.

## CLI preview

```bash
spacetravlr cellchat --h5ad path/to/data.h5ad --enable --species mouse --out probs.csv
```

## Contrast with stock SpaceTravLR

| | CellChat | SpaceTravLR (stock) | Hybrid |
|--|----------|---------------------|--------|
| Currency | Probability \(P\) | β on \(L\times R\) | β on \(P\)-weighted spatial \(L\times R\) |
| Locality | Mean-field over types | Gaussian neighborhoods | Both |
| Significance | Label permutation | Rank/Wilcoxon on β (downstream) | CellChat filter → Lasso β |
