# CellChat × SpaceTravLR hybrid

CellChat (Jin et al., *Nat Commun* 2021) estimates a **mean-field** communication probability \(P_{i\to j}^k\) from group-averaged ligand/receptor abundance with a Hill / mass-action kernel. SpaceTravLR instead regresses **per-cell** coefficients on spatially received ligand × receptor features.

This hybrid runs CellChat-style probabilities first, then builds **per-cell** LR design columns for the usual sparse group Lasso.

## Why not use \(P\) alone as Lasso features?

SpaceTravLR fits **per cluster**. A feature that is constant within a cluster (group-level \(P\)) is absorbed by the intercept. Probabilities therefore **select** (and optionally soft-gate) LR terms rather than replace them.

## Pipeline

1. **CellChatDB** — load `cellchat_{mouse|human}.csv` **as intact complexes** (multi-subunit rows kept together).
2. **Expression prep** — CellChat scale `expr / max(expr)`, then per-group **trimean** \((Q_1+2Q_2+Q_3)/4\); drop groups with `< min_cells`. Complex ligand/receptor levels use the **geometric mean** of subunits (zeros propagate), matching CellChat.
3. **Probability** — for each DB interaction \(k\) with complex levels \(L_i,R_j\):

\[
P_{i\to j}^k = \frac{(L_i R_j)^n}{K_h^n + (L_i R_j)^n}
\]

Optional population-size weights and label-permutation p-values (`n_perm > 0`).

4. **Expand for Lasso** — after filtering/selection, expand each retained complex into independent SpaceTravLR `Lig$Rec` units (cartesian product of subunits). Each child **inherits** the parent \(P_{s\to t}\) tensor. Example: `TGFB1` × `TGFBR1_TGFBR2` → `TGFB1$TGFBR1`, `TGFB1$TGFBR2`.

5. **LR terms** (`[cellchat].lr_mode`):

| Mode | Per-cell feature |
|------|------------------|
| `weighted_spatial` (default) | \(X_{c,k}=\sum_s P_{s\to\mathrm{type}(c)}^k\cdot\widetilde{L}_{c\leftarrow s}^{(k)}\cdot R_c^{(k)}\) (soft gate; \(P\) already depends on group \(L,R\)) |
| `hill_spatial` | \(X_{c,k}=(\widetilde{L}_c R_c)/(K_h+\widetilde{L}_c R_c)\) on CellChat-selected pairs |
| `spatial_product` | Classic \(\widetilde{L}_c\cdot R_c\), pair set from CellChat filters |
| `meanfield` | Flat-kernel received L: \(X_{c,k}=\bar{L}^{(k)}\,R_c^{(k)}\) with global mean ligand (no \(P\) in the product) |

\(\widetilde{L}_{c\leftarrow s}\) is the Gaussian received-ligand field using **only** cells of type \(s\).

### Fair A/B (received L)

Fix the CellChat pair set (`replace_lr_pairs`, `max_interactions`, filters) and compare only:

- `meanfield` — global / flat-kernel \(\bar{L}\,R\)
- `spatial_product` — Gaussian \(\widetilde{L}\,R\)

Do **not** put \(P\) into the meanfield product; \(P\) already embeds group \(L\) and \(R\).

## Not implemented (vs full CellChat)

- Agonist / antagonist / co-receptor cofactor tables
- Region-distance spatial constraints (`P.spatial`) used in CellChat’s imaging mode
- These affect absolute \(P\) calibration, not the meanfield vs spatial received-L contrast when the pair set is held fixed

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

`max_interactions` caps **complex-level** rows before expansion to `Lig$Rec` columns.

When `enabled = true`, training writes `cellchat_commun_prob.csv` under the run output directory and replaces (or weights) LR modulators accordingly.

## CLI preview

```bash
spacetravlr cellchat --h5ad path/to/data.h5ad --enable --species mouse --out probs.csv
```

## Contrast with stock SpaceTravLR

| | CellChat | SpaceTravLR (stock) | Hybrid |
|--|----------|---------------------|--------|
| Currency | Probability \(P\) | β on \(L\times R\) | β on CellChat-selected (and optionally \(P\)-gated) spatial \(L\times R\) |
| Complexes | Geom. mean of subunits | Single-gene `Lig$Rec` | P on complexes → expand for Lasso |
| Locality | Mean-field over types | Gaussian neighborhoods | Both |
| Significance | Label permutation | Rank/Wilcoxon on β (downstream) | CellChat filter → Lasso β |
