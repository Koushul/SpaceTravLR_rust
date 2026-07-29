# CellChat × SpaceTravLR hybrid

CellChat (Jin et al., *Nat Commun* 2021) estimates a **mean-field** communication probability \(P_{i\to j}^k\) from group-averaged ligand/receptor abundance with a Hill / mass-action kernel. SpaceTravLR regresses **per-cell** coefficients on received ligand × receptor features.

This hybrid uses CellChat probabilities to **select** LR pairs, then builds per-cell design columns with one of two received-ligand definitions.

## Why not use \(P\) alone as Lasso features?

SpaceTravLR fits **per cluster**. A feature that is constant within a cluster (group-level \(P\)) is absorbed by the intercept. \(P\) therefore selects pairs; received ligand supplies within-cluster variation (for `spatial`) or a flat ligand scale (for `meanfield`).

## Pipeline

1. **CellChatDB** — load `cellchat_{mouse|human}.csv` **as intact complexes**.
2. **Expression prep** — scale `expr / max(expr)`, group **trimean**, geometric-mean complexes.
3. **Probability** — \(P_{i\to j}^k = (L_i R_j)^n / (K_h^n + (L_i R_j)^n)\); filter / cap interactions.
4. **Expand for Lasso** — selected complexes → independent `Lig$Rec` units (children inherit parent \(P\)).
5. **LR terms** (`[cellchat].lr_mode`) — only two modes:

| Mode | Received ligand | Per-cell feature |
|------|-----------------|------------------|
| `meanfield` | Global mean \(\bar{L}\) (flat kernel) | \(X=\bar{L}\,R_c\) |
| `spatial` (default) | Gaussian neighborhood \(\widetilde{L}_c\) | \(X=\widetilde{L}_c\,R_c\) |

Same CellChat pair set for both; only the ligand aggregator differs.

## Not implemented (vs full CellChat)

- Agonist / antagonist / co-receptor cofactor tables
- Imaging-mode region-distance `P.spatial`

## Config

```toml
[cellchat]
enabled = true
lr_mode = "spatial"          # or "meanfield"
kh = 0.5
hill_coef = 1.0
min_cells = 10
n_perm = 0
replace_lr_pairs = true
max_interactions = 200
# signaling_types = ["Secreted Signaling"]
```

`max_interactions` caps **complex-level** rows before expansion.

## CLI preview

```bash
spacetravlr cellchat --h5ad path/to/data.h5ad --enable --species human --out probs.csv
```

## Fair A/B

```toml
# arm A
lr_mode = "meanfield"
# arm B
lr_mode = "spatial"
```

Hold all other `[cellchat]` / training settings fixed.
