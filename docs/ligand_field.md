# Ligand field

When LR modulators are enabled, SpaceTravLR selects pairs via group-level communication probabilities (CellChat-style; Jin et al., *Nat Commun* 2021) and builds **per-cell** design columns from a received-ligand field × local receptor.

Set **`mode`** and related params under `[ligand_field]` — there is no on/off switch. Default mode is **`spatial`** (Gaussian neighborhood).

## Why not use \(P\) alone as Lasso features?

SpaceTravLR fits **per cluster**. A feature that is constant within a cluster (group-level \(P\)) is absorbed by the intercept. \(P\) therefore selects pairs; the ligand field supplies within-cluster variation (for `spatial`) or a flat ligand scale (for `meanfield`).

## Pipeline

1. **Interaction DB** — load `cellchat_{mouse|human}.csv` **as intact complexes**.
2. **Expression prep** — scale `expr / max(expr)`, group **trimean**, geometric-mean complexes.
3. **Pair selection** — `pair_selection = "prob"` uses \(P\); `"expressed"` keeps all present LR pairs ranked by mean \(L\times R\).
4. **Expand for Lasso** — selected complexes → independent `Lig$Rec` units.
5. **Received ligand** (`mode` + `received_ligand_norm`):

| Mode | Norm | Received ligand | Feature |
|------|------|-----------------|---------|
| `spatial` | `global_n` (default) | `(1/N)Σ w_{ij} L_j` | \(X=\widetilde L_c R_c\) |
| `spatial` | `kernel_mass` | \(Σ w L / Σ w\) | same; fair vs meanfield |
| `meanfield` | either | global \(\bar L\) | \(X=\bar L R_c\) |

Unique-ligand fields are **precomputed once** per run and shared across gene workers. Diagnostics CSV: `ligand_field_L_diagnostics.csv`.

## Config

```toml
[ligand_field]
mode = "spatial"                 # or "meanfield"
pair_selection = "prob"          # or "expressed"
received_ligand_norm = "global_n" # or "kernel_mass"
weighted_ligand_scale_factor = 1.0
max_interactions = 200
write_ligand_diagnostics = true
signaling_types = ["Secreted Signaling"]
```

## Fair A/B (recommended)

```toml
pair_selection = "expressed"
received_ligand_norm = "kernel_mass"
# arm A
mode = "meanfield"
# arm B
mode = "spatial"
```

Hold all other settings fixed. Compare **scaled** betas (`*_betadata_scaled.feather`) when `unscale_betas_on_export = true`.

## Fast meanfield screen

Use `mode = "meanfield"` for a quick screen, then re-run `mode = "spatial"` on selected genes / pairs.

Legacy `[cellchat]` / `enabled` keys are ignored or migrated; prefer `[ligand_field].mode`.
