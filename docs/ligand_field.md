# Ligand field

Group-level communication probabilities (CellChat-style; Jin et al., *Nat Commun* 2021) can **select** LR pairs. SpaceTravLR then builds **per-cell** design columns from a received-ligand field × local receptor.

## Why not use \(P\) alone as Lasso features?

SpaceTravLR fits **per cluster**. A feature that is constant within a cluster (group-level \(P\)) is absorbed by the intercept. \(P\) therefore selects pairs; the ligand field supplies within-cluster variation (for `spatial`) or a flat ligand scale (for `meanfield`).

## Pipeline

1. **Interaction DB** — load `cellchat_{mouse|human}.csv` **as intact complexes**.
2. **Expression prep** — scale `expr / max(expr)`, group **trimean**, geometric-mean complexes.
3. **Probability** — \(P_{i\to j}^k = (L_i R_j)^n / (K_h^n + (L_i R_j)^n)\); filter / cap interactions.
4. **Expand for Lasso** — selected complexes → independent `Lig$Rec` units (children inherit parent \(P\)).
5. **LR terms** (`[ligand_field].mode`) — only two modes:

| Mode | Received ligand | Per-cell feature |
|------|-----------------|------------------|
| `meanfield` | Global mean \(\bar{L}\) (flat kernel) | \(X=\bar{L}\,R_c\) |
| `spatial` (default) | Gaussian neighborhood \(\widetilde{L}_c\) | \(X=\widetilde{L}_c\,R_c\) |

Same pair set for both; only the ligand aggregator differs. Spatial aggregation uses `[ligand_field].weighted_ligand_scale_factor` and optional `ligand_grid_factor`.

## Not implemented (vs full CellChat)

- Agonist / antagonist / co-receptor cofactor tables
- Imaging-mode region-distance `P.spatial`

## Config

```toml
[ligand_field]
enabled = true
mode = "spatial"                 # or "meanfield" (alias: lr_mode)
weighted_ligand_scale_factor = 1.0
# ligand_grid_factor = 0.2        # approximate field (omit = exact)
kh = 0.5
hill_coef = 1.0
min_cells = 10
n_perm = 0
replace_lr_pairs = true
max_interactions = 200
# signaling_types = ["Secreted Signaling"]
```

`max_interactions` caps **complex-level** rows before expansion.

Legacy `[cellchat]` sections and `[spatial].weighted_ligand_scale_factor` / `[perturbation].ligand_grid_factor` are still accepted and migrated into `[ligand_field]`.

## CLI preview

```bash
spacetravlr ligand-field --h5ad path/to/data.h5ad --enable --species human --out probs.csv
```

(`cellchat` remains a CLI alias.)

## Fair A/B

```toml
# arm A
mode = "meanfield"
# arm B
mode = "spatial"
```

Hold all other `[ligand_field]` / training settings fixed.
