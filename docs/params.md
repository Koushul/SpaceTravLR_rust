# Parameters

SpaceTravLR reads **`spaceship_config.toml`** at the repo root (or under `SPACETRAVLR_DATA_DIR`). CLI flags and `--config` overlays override these values for a given run. The executed config is saved as **`spacetravlr_run_repro.toml`** in the output directory — use that file with `--join-output-dir`, the spatial viewer, and `spacetravlr-perturb`.

See also [How it works](math.md) for the math behind spatial and GRN terms, and [Output files](output.md) for artifacts.

---

## Repository template

Values below match the checked-in [`spaceship_config.toml`](https://github.com/Koushul/SpaceTravLR_rust/blob/main/spaceship_config.toml).

<!-- --8<-- "docs/snippets/spaceship_config.html"

??? tip "Raw TOML"
    Copy the template from the repo or pass `--config path/to/override.toml` to merge only the sections you need.

--- -->

## `[data]` — AnnData inputs

| Parameter | Template | What it does | Increase / enable | Decrease / disable |
|-----------|----------|--------------|-------------------|---------------------|
| `adata_path` |  | Path to the spatial `.h5ad`. Usually set via CLI `--h5ad` instead. | — | — |
| `layer` | `imputed_count` | Expression matrix used for means, Lasso targets, and ligand ranking. | Use denoised / imputed layers when raw counts are sparse. | Use `X` or normalized layers only if you intentionally want raw-scale fitting. |
| `cluster_annot` | `cell_type` | `obs` column for cluster labels. **Seed mode**: one β row per label. **Full mode**: still used for cluster-stratified CNN and Lasso. | Finer labels → more rows per gene, more spatial heterogeneity captured per type. | Coarser labels → stabler Lasso, fewer CNN runs, less resolution. |
| `condition` |  | Split training into `output_dir/conditions/<value>/` per unique `obs` value. | Use for batches, patients, or slides you must not pool. | Omit when a single joint model is OK. |

---

## `[spatial]` — distances and CNN grids

Coordinate units should match `obsm['spatial']` (µm after prep). These settings affect **received ligands**, **contact signaling**, and the **spatial proximity maps** fed to the CNN (see [How it works](math.md)).

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `radius` | `300` | Gaussian σ for **secreted** ligand reception: neighbors farther than ~`radius` contribute little. Also used as a default L–R radius in perturbation when per-pair radii are absent. | Wider paracrine neighborhoods; smoother ligand fields; more long-range coupling. | Tighter local niches; less spillover from distant expressors. |
| `spatial_dim` | `64` | Side length of the per-cell spatial map (`spatial_dim × spatial_dim`). Must stay large enough for the CNN’s pooling stack (use ≥ `8` in practice). | Finer spatial context in the CNN; more parameters and memory. | Faster training; coarser niche geometry (risk missing fine structure). |
| `contact_distance` | `30` | Hard cutoff for **juxtacrine / contact** signaling (pairs must be within this distance). | Stricter “touching” definition; fewer contact edges. | More permissive contact graph. |

---

## `[grn]` — modulators and priors

Each target gene is predicted from modulator groups: **TFs**, **ligand–receptor** (`LIG$REC`), **TF–ligand** (`TF#LIG`), and optional **extras**.

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `network_data_dir` |  | Folder with `mouse_network.parquet` / `human_network.parquet`. Overrides `SPACETRAVLR_DATA_DIR`. | Point at a custom curated network bundle. | Rely on install / env search path. |
| `tf_priors_feather` |  | Precomputed TF→target links (`source`, `target`, `cell_type`). Skips auto CellOracle inference when set. | Reuse stable priors across runs. | Let training infer `celloracle_tf_priors.feather` once per output dir. |
| `tf_ligand_cutoff` | `0.1` | Minimum NicheNet-style score for a TF to regulate via ligand mediation. | More TF–ligand links; richer but noisier TFL channel. | Stricter TFL edges; sparser, more conservative graph. |
| `celloracle_p_max` | `0.05` | Max p-value for auto-inferred CellOracle ridge edges in the TF prior feather. | (Not usually increased.) | Lower (e.g. `0.01`) → fewer inferred TF edges. |
| `max_lr` | `200` | Keep only L–R pairs whose **ligand** is in the top *N* by mean expression on `[data].layer`. | More receptor channels; slower, more collinearity. | Fewer, high-abundance ligand axes; faster, sparser LR block. |
| `use_tf_modulators` | `true` | Include TF targets from priors. | — | Set `false` or use `train_modulators = "lr"` for LR-only ablation. |
| `use_lr_modulators` | `true` | Include `LIG$REC` columns. | — | Disable for intracellular-only models. |
| `use_tfl_modulators` | `true` | Include `TF#LIG` columns. | — | Disable to drop ligand-mediated TF terms. |
| `train_modulators` |  | Shorthand: `"tf,lr,tfl"` replaces the three `use_*` flags. | Combine only the families you need for an ablation. | Must leave at least one family enabled. |
| `extra_modulators` / `extra_modulators_file` |  | Add raw-expression predictors (fourth Lasso group). | Force inclusion of known covariates (e.g. ambient RNA proxies). | — |
| `extra_lr` / `extra_lr_file` |  | Add `LIG$REC` pairs beyond the database screen. | Hypothesis-driven pairs (e.g. `CXCL13$CXCR5`). | — |

---

## `[ligand_field]` — received ligand + communication-prob pair selection

Used whenever LR modulators are on (`use_lr_modulators` / `train_modulators` includes `lr`). Pick **`mode`** and params; there is no `enabled` flag. Default is Gaussian spatial. See [Ligand field](ligand_field.md).

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `db_path` |  | Path to `cellchat_{species}.csv`. | Custom curated DB. | Auto-resolve from `data/` / `SPACETRAVLR_DATA_DIR`. |
| `mode` | `spatial` | Received-ligand aggregator (`lr_mode` alias). | `meanfield` = global mean \(L\times R\). | `spatial` = Gaussian \(\widetilde{L}R\) (needs `obsm` spatial coords). |
| `pair_selection` | `prob` | How LR pairs are chosen before Lasso. | `expressed` = all present pairs ranked by mean \(L\times R\`. | `prob` = CellChat \(P\) rank/filter. |
| `received_ligand_norm` | `global_n` | How spatial weights reduce to \(\widetilde L\). | `kernel_mass` = \(Σ w L/Σ w\) (fair vs meanfield). | `global_n` = legacy `(1/N)Σ w L`. |
| `weighted_ligand_scale_factor` | `1.0` | Linear multiplier on Gaussian weights in received-ligand aggregation. | Stronger effective paracrine input per neighbor. | Weaker paracrine signal (before normalization). |
| `ligand_grid_factor` |  | Grid spacing as fraction of `[spatial].radius` for approximate received ligands. Ignored for `kernel_mass` training path. | Larger (e.g. `0.5`) → faster, ~few % error. | Smaller or omit → exact, slower on 5k+ cells. |
| `kh` | `0.5` | Half-saturation in the Hill term. | Harder to saturate. | Easier saturation. |
| `hill_coef` | `1.0` | Hill coefficient \(n\). | Steeper switch. | Near-linear mass action. |
| `min_cells` | `10` | Drop groups smaller than this. | More groups kept. | Stricter group filter. |
| `population_size_weight` | `false` | Multiply \(P\) by sender×receiver fractions. | Emphasize abundant populations. | Expression-only \(P\). |
| `n_perm` | `0` | Label-shuffle permutations for p-values (`0` = skip). | e.g. `100` for significance testing. | Keep `0` for speed. |
| `p_threshold` | `0.05` | Keep interactions with min p ≤ this (needs `n_perm > 0`). | More pairs. | Stricter significance. |
| `min_prob` | `0.0` | Drop interactions whose max \(P\) is below this (ignored when `pair_selection = expressed`). | Stronger-only edges. | Keep weak links. |
| `replace_lr_pairs` | `true` | Replace GRN `lr` edges with probability-selected `Lig$Rec` units (complexes pre-expanded). | Force DB pair set. | `false` to keep GRN pairs (probs still written for inspection). |
| `max_interactions` | `200` | Cap after ranking by max \(P\) or mean \(L\times R\). | Broader LR block. | Faster, sparser. |
| `write_ligand_diagnostics` | `true` | Write `ligand_field_L_diagnostics.csv` (meanfield vs spatial \(L\)). | Keep on for A/B audits. | `false` to skip I/O. |
| `signaling_types` |  | Restrict to classes (e.g. `Secreted Signaling`). | Focus paracrine. | Empty = all classes. |
| `random_seed` | `42` | Permutation RNG seed. | — | — |

---

## `[lasso]` — cluster-wise group Lasso

Lasso runs **per target gene × cluster** (or per cell in export semantics) before any CNN step in full mode. It sets **anchors** and a **quality gate** via `score_threshold`.

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `l1_reg` | `1e-4` | Element-wise L1 on coefficients. | More sparsity; fewer modulators survive. | Denser fits; risk overfitting small clusters. |
| `group_reg` | `1e-5` | Group L2 across modulator families (TF / LR / TFL / extra). | Entire groups dropped together; cleaner ablations. | Weaker group shrinkage. |
| `n_iter` | `500` | Max FISTA iterations. | Use if optimization stalls before `tol`. | Faster but may stop early. |
| `tol` | `1e-4` | Relative convergence tolerance. | Tighter convergence. | Looser, faster stop. |
| `scale_modulators` | `true` | Column-wise ÷ std (no mean centering), CellOracle-style. | Keep on when modulator scales differ widely. | `false` if you want raw-scale optimization. |
| `unscale_betas_on_export` | `false` | Divide exported β by column std to return to expression units. | `true` for interpretable coefficients in feather files. | `false` to keep scaled-space β (compare to CellOracle exports). |
| `export_scaled_betas` | `true` | Also write `{gene}_betadata_scaled.feather` when unscaling on export. | Keep on for meanfield vs spatial coeff A/B. | `false` to skip the extra file. |
| `parallel_lasso_clusters` | `false` | Rayon parallel over clusters within a gene (independent of `n_parallel`). | `true` when many clusters × one gene is slow. | `false` for lowest memory / simplest logs. |
| `gram_override` | `true` | `true` = Gram-matrix FISTA; `false` = residual gradients; omit key for auto when `n > p`. | Force Gram on large `n` for speed. | `false` when Gram path is unstable. |

---

## `[training]` — mode, schedule, and quality gate

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `mode` | `full` | `seed`: cluster-level β only (Lasso). `full`: Lasso then per-cell CNN refinement. | `full` for spatially varying coefficients. | `seed` for fastest baseline microniche maps. |
| `epochs` | `100` | CNN training epochs per cluster (full mode). | More time to fit spatial residuals; watch overfit. | Quicker runs; may underfit complex niches. |
| `learning_rate` | `4e-4` | Adam step size for CNN. | Faster convergence if loss is smooth; may diverge. | Safer optimization; slower training. |
| `score_threshold` | `0.1` | Minimum in-sample R² for a cluster to **keep** Lasso (and CNN in full mode). Failed clusters export as zeros / orphans. | More clusters kept (noisier fits included). | Stricter gate; fewer but higher-confidence rows. |
| `genes` |  | Subset of `var` to train (persisted in repro TOML for join). | Focused screens. | — |
| `max_genes` |  | Cap count after `genes` filter (order preserved). | — | Limit cost on pilot runs. |

---

## `[cnn]` — spatial refinement (full mode)

Only used when `mode = full`. The CNN predicts a **spatial map of multipliers** on Lasso anchors; see [How it works](math.md) Steps 2–4.

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `adam_beta_1` / `adam_beta_2` | `0.9` / `0.999` | Adam momentum decay rates. | Rarely changed; lower `beta_2` if loss is non-stationary. | Standard deep-learning defaults. |
| `adam_epsilon` | `1e-5` | Adam numerical stabilizer. | — | — |
| `weight_decay` |  | L2 on CNN weights. | Mild regularization against overfit. | Omit for maximum flexibility. |
| `grad_clip_norm` |  | Clip gradient L2 norm each step; set `null` to disable. | Tighter clip if loss spikes. | Looser or off if training is stable. |
| `spatial_feature_radius` | `300` | Gaussian radius for **CNN input** spatial feature maps (can differ from `[spatial].radius`). | Match biological niche size in coordinate units. | Smaller local context in CNN patches. |
| `output_activation` | `sigmoidx2` | Maps CNN logits before anchor scaling: `identity`, `sigmoid`, `tanh`, `sigmoidx2` (0–2). | `sigmoidx2` bounds effective multipliers. | `identity` if you need unconstrained scale (riskier). |
| `cnn_minibatch_size` | `256` | Cells per optimizer step within a cluster; `0` = full batch each epoch. | Smaller batches → noisier gradients, more steps. | Larger batches → smoother, fewer updates. |
| `cnn_inference_batch_size` | `512` | Chunk size for **export** and post-train R² only. | Lower if GPU/RAM limited during export. | Larger for faster inference passes. |
| `cnn_max_batches_per_epoch` |  | Cap optimizer steps per epoch. | Limit wall time on huge clusters. | Omit for full pass each epoch. |
| `cnn_max_cells_per_epoch` | `1000` | Random cell subsample per epoch when cluster is larger. | Faster epochs; still explores space over many epochs. | Set `null` for full-cluster sweeps each epoch (slower). |
| `min_cells_for_cnn` |  | Skip CNN when cluster size is below this (Lasso kept if it passed). | Require more cells per CNN fit. | Set `0` to always attempt CNN. |
| `mean_beta_lasso_prior_weight` |  | Pull batch-mean effective β toward Lasso anchors. | Stronger tie to cluster-average Lasso. | Weaker prior → more per-cell deviation. |
| `lr_schedule` |  | `constant` or `cosine` decay after warmup. | Cosine for long runs. | `constant` for short pilots. |
| `lasso_pred_align_weight` | `0.05` | Extra loss: match CNN prediction to Lasso prediction. | Keeps CNN close to Lasso early; use small values. | `0` to let spatial terms dominate immediately. |
| `lasso_pred_align_linear_decay` | `true` | Ramp alignment weight down over epochs. | Recommended when `lasso_pred_align_weight > 0`. | Fixed alignment pressure all epochs. |
| `drop_cnn_if_insample_worse_than_lasso` | `true` | Revert to Lasso export when CNN in-sample R² loses to Lasso. | — | `false` to always keep CNN outputs. |
| `cnn_vs_lasso_arbitration_margin` | `0.2` | CNN kept if `cnn_r2 + margin ≥ lasso_r2`. | More lenient toward CNN (margin helps CNN win ties). | `0` for strict comparison. |

---

## `[execution]` — parallelism and outputs

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `random_seed` |  | Base seed; mixed per target gene for Lasso and CNN shuffles. | Change to reproduce a different subsample. | Fixed seed for reproducibility. |
| `n_parallel` | `9` | Concurrent **genes** trained in parallel. | Faster wall clock if CPU/GPU allows. | `1` for debugging or limited RAM. |
| `output_dir` |  | Where `*_betadata.feather` and repro TOML are written. Empty → `{adata_stem}_{date}` in cwd. | Set explicit path for shared storage. | — |
| `write_minimal_repro_h5ad` |  | Write a slim `.h5ad` for offline replay (heavy I/O). | Enable for archival reproducibility. | Leave off on huge slides. |
| `stale_lock_secs` | `3600` | Multi-host: delete gene `*.lock` older than this before claiming. | Safer crash recovery on NFS (`3600` = 1 h). | `0` = never auto-remove locks. |

---

## `[perturbation]` — in silico splash defaults

Used by `spacetravlr-perturb` and the spatial viewer unless overridden at runtime.

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `beta_scale_factor` | `100.0` | Global multiplier on all β before propagation. | Stronger perturbation amplitudes. | Weaker systemic response. |
| `beta_cap` |  | Clamp β to `[-cap, cap]` after scaling. | Prevent extreme coefficients from dominating. | Omit for uncapped dynamics. |
| `n_propagation` | `4` | Rounds of ligand / GRN signal propagation in `splash()`. | Deeper equilibration; more neighbor feedback. | Shallower, more local effects. |
| `cells_csv` / `cells_csv_column` |  | Default ROI for perturb export. | Restrict splash to a cell list. | — |
| `perturbed_gene_min_bound` | `0.0` (when omitted) | Lower clip on simulated expression after each propagation step. | Keep predictions non-negative or in a biologically plausible range. | Set explicitly to allow negative values (unusual). |
| `perturbed_gene_max_bound` |  | Upper clip on simulated expression after each propagation step. | Cap OOD overexpression from linear propagation. | Omit for no upper bound. |

---

## `[model_export]` — optional CNN checkpoints

| Parameter | Template | What it does | Turn up | Turn down |
|-----------|----------|--------------|---------|-----------|
| `save_cnn_weights` | `false` | Write `{gene}_cnn_weights.npz` under `output_subdir`. | Enable for external analysis or Python trainer parity. | Off to save disk. |
| `compressed_npz` | `true` | Deflate compression on `.npz`. | — | `false` for slightly faster writes. |
| `output_subdir` | `CNN_weights` | Subfolder under `output_dir`. | — | — |

---
