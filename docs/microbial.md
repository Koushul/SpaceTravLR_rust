# Microbial secretion sources for SpaceTravLR

**Branch:** `cursor/microbial-secretion-lr-a580`  
**Status:** design / preliminary — not wired into training yet  
**Motivating data:** Stereo-seq + in situ PAP host–gut microbiome maps ([Ntekas et al., Nat Microbiol](https://www.nature.com/articles/s41564-026-02286-7)); local pilot under `spacetravlr_microbiome/`

---

## 1. Idea in one sentence

Treat **bacterial spatial loci as ligand-like secretion sources**: microbial signal at location \(b\) decays to host cell surfaces \(i\), multiplies by a **host receptor**, and enters the SpaceTravLR model as a new modulator family (`bact$HostRec`), parallel to classical CellChat `Lig$Rec`.

This is **not** “put genus counts in `extra_modulators`.” That skips spatial decay and mixes sender geometry into a per-cell covariate. The LR path already has the right geometry; we extend *who can send*.

---

## 2. Why the current LR term is almost right

Training today (see [math.md](math.md) Step 1):

\[
\widetilde{L}_{ik}
  = \frac{s}{N}\sum_{j=1}^{N}
    \exp\!\Bigl(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2r^2}\Bigr)
    \, L_{jk},
\qquad
x^{\mathrm{LR}}_{i,\,k\rightarrow\rho}
  = \widetilde{L}_{ik}\cdot R_{i\rho}
\]

Senders \(j\) are **host cells**; \(L_{jk}\) is **host ligand gene expression**. Bacterial molecules are invisible unless they happen to share a gene symbol with a host ligand (they do not).

CellChat mouse parquet has almost no PAMP→PRR edges (e.g. only host ligands → `Tlr4`). A separate **bacterial→host** interaction table is required.

---

## 3. Conceptual model

```text
                    lumen / mucosa interface
   ●  ●   ●     bacterial loci b  (bins, colonies, or centroids)
    \  |  /
     \ | /   Gaussian (or contact) kernel w(i,b; r_k)
      \|/
   ─── host cell i ───  receptor R_ρ on surface
            │
            ▼
   feature  received(S_k)·R_ρ   →  β_{S_k$R_ρ}
```

| Role | Classical CCC | Microbial extension |
|------|---------------|---------------------|
| Sender | Host cell \(j\) | Bacterial locus \(b\) |
| Payload | Host ligand gene \(L_k\) | Microbial signal \(S_k\) (PAMP / metabolite / proxy) |
| Propagation | Gaussian on cell–cell distance | Gaussian on cell–locus distance (possibly different \(r_k\)) |
| Receiver gate | Host receptor \(R_\rho\) | Same — **must be in host transcriptome** |
| Pair ID | `Tgfb1$Tgfbr1` | `Lps$Tlr4`, `Flagellin$Tlr5`, … |
| Lasso group | LR = 1 | New group **BR = 4** (or reuse 1 with `edge_type=bact_lr`) |

**Host–host LR stays unchanged.** Microbial terms are additive received fields, not replacements.

---

## 4. Math

### 4.1 Received microbial signal

For each microbial signal channel \(k\) and host cell \(i\):

\[
\widetilde{S}_{ik}
  = s_k\sum_{b=1}^{B}
    \kappa_k\!\bigl(\|\mathbf{x}_i - \mathbf{y}_b\|\bigr)\,
    A_{bk}
\]

with kernel (default isotropic Gaussian, optional contact cutoff \(d_{\max,k}\)):

\[
\kappa_k(d)
  = \exp\!\Bigl(-\frac{d^2}{2r_k^2}\Bigr)
  \cdot \mathbf{1}\![d \le d_{\max,k}]
\]

**Normalization (important):** do **not** blindly reuse \(1/N_{\mathrm{host}}\). Options:

| Scheme | Formula factor | When |
|--------|----------------|------|
| `none` | \(s_k\) only | Preferred v0 — \(s_k\) absorbs scale; amount \(A_{bk}\) already carries biomass |
| `n_senders` | \(1/B\) | Stable when \(B\) varies across slides |
| `density` | \(1/\sum_b \kappa_k(\|\mathbf{x}_i-\mathbf{y}_b\|)\) | Local density-normalized “exposure” |

Recommend **`none` + global \(s_k\)** for v0, tuned so \(\mathrm{median}(\widetilde{S})\) is O(host \(\widetilde{L}\)).

### 4.2 Interaction feature

\[
x^{\mathrm{BR}}_{i,\,k\rightarrow\rho}
  = \widetilde{S}_{ik}\cdot R_{i\rho}
\]

Same product structure as host LR → same splash rules in perturbation:

\[
\frac{\partial H}{\partial R_\rho} += \beta\cdot\widetilde{S}_k,
\qquad
\frac{\partial H}{\partial S_k}\ \text{only if } S_k \text{ is host-controllable (usually not)}
\]

For microbial channels, **perturb host receptors / downstream TFs**, not bacterial loci (unless we later add antibiotic / taxon-ablation experiments that zero \(A_{bk}\)).

### 4.3 Composite received field (optional)

Some analyses may want a host ligand that is *also* boosted by microbes (e.g. analog). v0 keeps channels **disjoint**: microbial `S_k` never share names with host `L_k`.

---

## 5. What is a “bacterial cell” in Stereo-seq?

True single-bacterium segmentation is unavailable. Operational senders (choose one, config-selectable):

| Mode | Definition of \(\mathbf{y}_b, A_{bk}\) | Pros / cons |
|------|----------------------------------------|-------------|
| **`bin`** | Every Stereo-seq bin with microbial UMI; \(A\) = genus/signal proxy counts | Dense, noisy, matches raw unmap H5AD |
| **`colony`** | Cluster bins (Moran / DBSCAN / Ripley); centroid + summed \(A\) | Matches paper’s colony sizes (∼10–30 µm); fewer senders |
| **`soft_field`** | Rasterize genus maps; sample or integrate continuously | Smooth; blurs colony structure |

**v0 recommendation:** `colony` for biology, `bin` for debugging. Radius prior: **contact / short-range PAMPs ∼10–30 µm**; metabolites / vesicles ∼**50–150 µm** (separate \(r_k\) in DB).

Host cells remain the only **receivers** with transcriptomes. Lumen-only microbial bins never become rows of the AnnData used for TF/LR regression — they only appear as sender tables.

---

## 6. Bacterial→host interaction database

Classical CellChat answers “which **host** ligands hit which **host** receptors.”  
We need “which **microbial signals** hit which **host** receptors,” plus “which **taxa** can emit which signals” (proxy layer for RNA-only data).

### 6.1 Schema — `bact_host_interactions`

| column | meaning |
|--------|---------|
| `signal_id` | Stable ID, e.g. `Lps`, `Flagellin`, `Scfa_butyrate` |
| `signal_name` | Display name |
| `receptor` | Host gene symbol (mouse Title / human UPPER) — must exist in `var` after QC |
| `receptor_complex` | Optional partners (`Ly96`, `Cd14`) for documentation; v0 uses primary receptor only |
| `pathway` | `TLR`, `NLR`, `GPCR_SCFA`, `FPR`, … |
| `signaling_class` | `Secreted` \| `Contact` \| `Vesicle` (sets default \(r_k\), \(d_{\max}\)) |
| `default_radius_um` | Kernel radius |
| `weight` | Prior strength (1.0 default) |
| `evidence` | Short note / PMID |
| `species` | `mouse` \| `human` \| `both` |

Shipped sketch: [`data/microbial/bact_host_interactions.v0.csv`](../data/microbial/bact_host_interactions.v0.csv)

### 6.2 Schema — `taxon_signal_priors`

Maps observed taxa (or Gram stain / phylum) → emitable signals. Stereo-seq gives **RNA taxonomy**, not measured LPS molecules — priors convert abundance → \(A_{bk}\).

| column | meaning |
|--------|---------|
| `taxon_level` | `genus` \| `family` \| `phylum` \| `gram` |
| `taxon` | e.g. `Clostridium`, `Bacillota`, `Gram_negative` |
| `signal_id` | FK into interactions table |
| `emission_weight` | Relative capacity (0–1+) |
| `notes` | |

Shipped sketch: [`data/microbial/taxon_signal_priors.v0.csv`](../data/microbial/taxon_signal_priors.v0.csv)

### 6.3 Building \(A_{bk}\) from unmap H5AD

1. Filter `superkingdom == Bacteria`; drop `Mus`/`Homo` leakage.  
2. Aggregate counts to genus (or species).  
3. For each signal \(k\):  
   \(A_{bk} = \sum_{g \in \mathcal{G}(k)} w_{g\rightarrow k}\,\mathrm{UMI}_{b,g}\)  
   with \(\mathcal{G}(k)\) from taxon priors (and Gram fallback).  
4. Optional: normalize per-bin or per-section; clip outliers.  
5. Optional colony clustering → replace bins with centroids.

Functional bacterial genes (e.g. `groL`, metabolic CDS from the paper) can later redefine \(A_{bk}\) more specifically than taxonomy priors.

---

## 7. Where this plugs into Rust

Current seams (from `src/ligand.rs`, `spatial_estimator.rs`, `network.rs`, `perturb.rs`):

1. **`ligand.rs`** — generalize weighted sum to  
   `receiver_xy` + `sender_xy` + `sender_amounts`  
   (host path = special case `sender_xy == receiver_xy`).

2. **`build_x_modulators_and_target_y`** — after host `received_map`, compute `received_bact`, form `S$R` columns; assign lasso group **4**.

3. **`merge_extra_lr_into` / modulator assembly** — allow ligand/signal IDs **not** in `var` if marked `bact`; receptor **must** remain in `var`.

4. **Config** (sketch):

```toml
[microbial]
enabled = true
sender_mode = "colony"          # bin | colony | soft_field
interactions = "data/microbial/bact_host_interactions.v0.csv"
taxon_priors = "data/microbial/taxon_signal_priors.v0.csv"
sender_table = "path/to/bact_senders.parquet"  # y_b, A_bk columns
normalization = "none"
scale_factor = 1.0
# optional overrides
# radius_um_override = 50.0
```

5. **Perturb** — keep bacterial \(\widetilde{S}\) **fixed** under host KO/OE; support experimental `zero_signal=Lps` / `zero_taxon=Fusobacterium` as ablation of \(A_{bk}\).

6. **Export** — `beta_Lps$Tlr4` etc. in betadata; collect-interactions ranks BR next to LR.

**Do not** implement microbial CCC only as `extra_modulators` (no spatial kernel). That remains a niche covariate fallback for ultra-sparse pilots.

---

## 8. Relation to Stereo-seq QC (local pilot)

From `spacetravlr_microbiome` QC:

- Host median UMI ∼60–80 → **impute / restrict receptor panel** before trusting \(\beta\).  
- PRR genes are sparsely detected → BR pairs will be few until imputation; start with receptors that actually appear (`Cd74`, adhesion genes today; TLR/NOD after impute).  
- Microbial neighborhoods are **dense** (median hundreds–thousands of nearby bacterial UMIs) → sender side is **not** the bottleneck; host receptor detection is.  
- Prefer **tumour boundary / mucosa** cells as receivers; lumen bins as senders only.

Pilot gene panel for BR (post-impute):  
`Tlr2, Tlr4, Tlr5, Tlr9, Cd14, Ly96, Nod1, Nod2, Ffar2, Ffar3, Hcar2, Fpr2, Myd88, Nfkb1, Rela`  
plus epithelial contact pairs already spatial-positive (`Cdh1`, `F11r`, GUCA) as **host–host** controls in the same run.

---

## 9. Biological scope & honesty bars

| Claim we can support | Claim we should not make |
|----------------------|---------------------------|
| Spatial microbial RNA/PAMP-*proxy* fields associate with host receptor programs | Measured LPS protein at each locus |
| In silico receptor KO under fixed microbial field | Causal “bacteria secrete X in vivo” without orthogonal assay |
| Taxon-prior proxies for Gram+/− / flagellin / SCFA capacity | Full bacterial secretome |
| Colony-scale secretion geometry (10–50 µm) | True single-bacterium CCC |

The Nat Microbiol paper measures **RNA biogeography**, not secreted metabolites. Our \(S_k\) are **mechanistic hypotheses encoded as channels**, analogous to how CellChat encodes ligand identity from gene expression.

---

## 10. Implementation milestones

| Milestone | Deliverable |
|-----------|-------------|
| **M0** (this branch) | Design doc + v0 CSV databases + config sketch |
| **M1** | `BactSenderTable` loader (parquet/CSV); unit tests for received-from-external-XY |
| **M2** | Generalize `calculate_weighted_ligands*`; BR features in estimator; group id 4 |
| **M3** | Wire Stereo-seq colony builder from unmap H5AD → sender parquet |
| **M4** | Perturb: fixed microbial field + taxon/signal ablation |
| **M5** | Tutorial on GSM9456850 tumour section; compare host-only vs host+BR niches |

---

## 11. Open questions

1. **Complex receptors** (`Tlr2`+`Tlr6`): product of partners vs primary only?  
2. **Inhibitory / decoy receptors** and antagonism — ignore in v0?  
3. **Should BR share LR lasso group** (compete with host ligands) or separate group (always available)? Lean **separate group**.  
4. **Human vs mouse** receptor symbol mapping for non-gut tissues (brain CosMx 16S paper).  
5. **Double counting** if we also put genus totals in `extra_modulators` — pick one spatial representation.  
6. **Diffusion anisotropy** along lumen vs tissue — isotropic Gaussian is wrong near boundaries; optional reflecting boundary later.

---

## 12. Summary recommendation

Ship the extension as **external secretion sources + bact→host pair DB**, not as a hack of CellChat host ligands. Use Stereo-seq microbial bins/colonies as senders, host PRRs as receptors, and keep host–host LR intact. v0 databases in `data/microbial/` are intentionally small and editable so biology can outrun engineering.
