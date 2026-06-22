# T Cell Mitochondrial Transfer Analysis — GSE280315 P1 CRC

**Tumor and macrophage donors → T cell receivers; effects on activation, exhaustion, and cytotoxic state**

| Field | Value |
|-------|-------|
| **Dataset** | GSE280315 P1 human colorectal cancer, Visium HD 8 µm bins |
| **T cell bins** | 23,766 (CD4: 3,837; CD8: 582; unsupervised T: 10,005) |
| **Tumor donors** | 121,378 bins |
| **Macrophage donors** | 14,538 bins (SELENOP⁺: 629; SPP1⁺: 166 annotated subtypes) |
| **Method** | MERCI-inspired RNA rank + LOO-SVR receiver calling (2,500 T cell subsample) |
| **Analysis date** | June 2026 |

---

## Executive summary

T cells in P1 CRC show **moderate mitochondrial transfer scores** from both tumor and macrophage donors (median ≈ 0.50). Tumor- and macrophage-derived transfer are **nearly interchangeable at the genome-wide rank level** (Spearman *r* = 0.96), but their **spatial patterns differ**: macrophage transfer drops sharply with distance, while tumor-associated transfer is **low inside tumor nests** and **elevated at the invasive front**.

Receiving mitochondria is **not associated with T cell exhaustion**. Exhaustion scores show no correlation with transfer (*r* ≈ 0.009, *p* ≈ 0.15) and no difference between high- and low-transfer quartiles (*p* ≈ 0.35). Instead, transfer correlates **weakly positively with activation and memory signatures** and **negatively with cytotoxic gene programs** (GZMB, PRF1, NKG7).

---

## 1. T cell identification

T cells were defined broadly to maximize sensitivity in 8 µm bin data:

- `DeconvolutionLabel1` or `Label2` = CD4/CD8 T cell, **or**
- `UnsupervisedL2` starts with `Tcells`

Subtypes:

| Subtype | *n* | Definition |
|---------|-----|------------|
| CD4 | 3,837 | Label1 = CD4 T cell |
| CD8 | 582 | Label1 = CD8 T cell |
| T_unsup | 10,005 | Unsupervised T cluster, no CD4/CD8 label |

---

## 2. Mitochondrial transfer from tumor vs macrophage

### 2.1 Genome-wide transfer scores

| Donor type | Median transfer to T cells | LOO receiver rate (all T cells)¹ |
|------------|---------------------------|----------------------------------|
| Tumor | 0.505 | 2.5% |
| Macrophage | 0.504 | 2.2% |
| Tumor + mac combined | 0.505 | — |

¹ LOO-SVR was run on 2,500 T cell bins; receiver labels are sparse across the full 23,766 bins. Among LOO-tested bins (*n* = 2,432), receiver rates are **24.1%** (tumor donor) and **21.1%** (macrophage donor).

Tumor and macrophage transfer scores are **highly correlated** across T cells (Spearman *r* = 0.96, *p* < 10⁻³⁰⁰), indicating that bins with high tumor-donor signature also carry high macrophage-donor signature — consistent with shared stromal/interface microniches rather than distinct donor pathways.

### 2.2 By T cell subtype

| Subtype | *n* | Median transfer (tumor) | Median transfer (mac) |
|---------|-----|---------------------------|------------------------|
| CD4 | 3,837 | 0.562 | 0.563 |
| CD8 | 582 | 0.570 | 0.572 |
| T_unsup | 10,005 | 0.533 | 0.542 |

CD4 and CD8 T cells show slightly higher transfer than unsupervised T clusters. CD8 cells also occupy the invasive front more often (33.8% in top transfer quartile vs 25.7% for T_unsup).

![Transfer by subtype](/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/human_visiumhd_merci/P1-CRC/figures/tcell_analysis/tcell_transfer_by_subtype.png)

---

## 3. Spatial context: where transfer happens

### 3.1 Distance to tumor

**Important caveat:** 2,159 T cell bins have **distance-to-tumor = 0** because they co-occupy tumor-labeled spatial bins (Periphery = Tumor). These embedded T cells show **low** transfer (median 0.29), distinct from infiltrating T cells at the interface.

| Spatial context | *n* | Median tumor transfer | Median mac transfer |
|-----------------|-----|----------------------|---------------------|
| Periphery = Tumor (embedded) | 2,134 | 0.291 | 0.266 |
| Periphery = 50 µm (invasive front) | 1,518 | **0.579** | **0.586** |
| Periphery = Tissue (stroma) | 20,114 | 0.535 | 0.537 |
| 0 < dist ≤ 50 µm (excluding co-occupancy) | 428 | 0.554 | 0.557 |
| > 200 µm from tumor | 18,416 | 0.542 | 0.544 |

**Interpretation:** Mitochondrial transfer to T cells is **enriched at the tumor invasive front**, not inside tumor nests. T cells spatially embedded in tumor regions may express predominantly endogenous mitochondrial programs, suppressing donor-signature detection.

![Transfer vs tumor distance](/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/human_visiumhd_merci/P1-CRC/figures/tcell_analysis/tcell_transfer_vs_tumor_distance.png)

### 3.2 Distance to macrophage

Macrophage transfer is **strongly distance-dependent**:

| Distance to nearest macrophage | *n* | Median mac transfer |
|-------------------------------|-----|---------------------|
| 0–50 µm | 5,916 | 0.524 |
| 50–100 µm | 8,828 | 0.515 |
| 100–200 µm | 5,434 | 0.477 |
| 200–500 µm | 918 | 0.375 |
| > 500 µm | 67 | 0.331 |

Spearman *r* = −0.057 between distance and mac transfer (*p* ≈ 2.4×10⁻¹⁸): **closer to macrophages → higher transfer**.

![Transfer vs macrophage distance](/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/human_visiumhd_merci/P1-CRC/figures/tcell_analysis/tcell_transfer_vs_mac_distance.png)

### 3.3 Spatial map

![Spatial transfer](/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/human_visiumhd_merci/P1-CRC/figures/tcell_analysis/tcell_transfer_spatial.png)

---

## 4. Effect on T cell state

Gene-set scores were computed with `scanpy.tl.score_genes`:

| Score | Genes |
|-------|-------|
| Activation | CD69, IL2RA, TNF, IFNG, ICOS, CD44, TNFRSF9, CD28 |
| Exhaustion | PDCD1, CTLA4, HAVCR2, LAG3, TIGIT, TOX, ENTPD1, CXCL13 |
| Cytotoxic | GZMB, PRF1, NKG7, GNLY, GZMA |
| Memory | IL7R, CCR7, SELL, LEF1 |
| Treg | FOXP3, IKZF2, CTLA4, IL2RA |

### 4.1 Correlation with transfer (Spearman)

| Transfer metric | State | *r* | FDR |
|-----------------|-------|-----|-----|
| From macrophage | Activation | +0.027 | 2.6×10⁻⁴ |
| From macrophage | Memory | +0.024 | 6.9×10⁻⁴ |
| From tumor | Memory | +0.024 | 6.9×10⁻⁴ |
| From tumor | Activation | +0.019 | 7.3×10⁻³ |
| From tumor | **Cytotoxic** | **−0.024** | 6.9×10⁻⁴ |
| From macrophage | **Cytotoxic** | **−0.028** | 1.4×10⁻⁴ |
| From tumor | **Exhaustion** | +0.009 | 0.23 (n.s.) |

![Correlation barplot](/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/human_visiumhd_merci/P1-CRC/figures/tcell_analysis/tcell_transfer_state_correlation.png)

### 4.2 High vs low transfer quartiles (top 25% vs bottom 25%)

| State score | High transfer median | Low transfer median | Mann–Whitney *p* |
|-------------|---------------------|---------------------|------------------|
| Activation | −0.043 | −0.044 | **0.009** |
| Memory | −0.022 | −0.022 | **0.015** |
| Cytotoxic | −0.017 | −0.016 | **1.4×10⁻⁴** |
| **Exhaustion** | −0.014 | −0.014 | **0.35 (n.s.)** |

### 4.3 Individual genes (high vs low transfer)

At Visium HD bin resolution, most exhaustion markers are sparse (median 0 UMIs). No significant exhaustion-gene differences were detected. Cytotoxic genes were **lower** in high-transfer T cells:

| Gene | High median | Low median | *p* |
|------|-------------|------------|-----|
| GZMB | 0 | 0 | 4.7×10⁻⁷ |
| PRF1 | 0 | 0 | 0.013 |
| NKG7 | 0 | 0 | 0.019 |

### 4.4 MERCI LOO receivers vs non-receivers

Among LOO-tested T cells, formal MERCI receivers (*n* ≈ 585 tumor-donor) show **lower activation** than non-receivers (median −0.055 vs −0.047, *p* ≈ 5.6×10⁻⁵) with **no exhaustion difference** (*p* ≈ 0.22). This contrasts with the weak positive rank-level activation correlation and may reflect the stricter SVR-based receiver definition capturing a distinct subset.

![Activation vs exhaustion](/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/human_visiumhd_merci/P1-CRC/figures/tcell_analysis/tcell_activation_exhaustion_by_transfer.png)

![State vs transfer scatter](/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/human_visiumhd_merci/P1-CRC/figures/tcell_analysis/tcell_state_vs_transfer.png)

---

## 5. Biological interpretation

### 5.1 Tumor vs macrophage as donors

At the RNA-signature level, tumor and macrophage mitochondrial transfer to T cells are **largely co-localized phenomena** in this CRC sample. Both peak at the **invasive front** and in **macrophage-proximal niches**. This aligns with the broader MERCI niche analysis showing transfer enrichment at the 50 µm tumor–stroma interface and near immune infiltrates.

Macrophage subtypes (SELENOP⁺ vs SPP1⁺) are too sparse at bin resolution for robust subtype-stratified donor analysis here.

### 5.2 Exhaustion

**Receiving mitochondria does not mark exhausted T cells** in P1 CRC. Exhaustion scores are flat across transfer levels, and PDCD1/CTLA4/HAVCR2/LAG3/TIGIT/TOX show no enrichment in high-transfer bins. If anything, T cells at the invasive front (high transfer) show **marginally lower** exhaustion than distant stromal T cells (*p* ≈ 0.036 for 0 < dist ≤ 50 µm vs > 200 µm).

This contrasts with hypotheses that mitochondrial uptake drives dysfunction; here, transfer associates with **interface-localized, slightly more activated/memory-polarized** T cells rather than terminally exhausted ones.

### 5.3 Activation and cytotoxicity

The weak positive activation/memory correlation suggests T cells receiving donor mitochondria may occupy **recently stimulated or tissue-resident memory-like** states at the tumor border. The **negative cytotoxic correlation** implies high-transfer T cells are **not** the GZMB⁺ effector population — consistent with CD4-skewed or helper-like infiltrates at the front rather than cytotoxic CD8 killers.

T cells embedded inside tumor nests (low transfer, dist = 0) show the **lowest activation** (median −0.065), possibly reflecting suppressed or excluded infiltrates within epithelial regions.

---

## 6. Limitations

1. **8 µm bins, not single cells** — T cell / tumor co-occupancy of bins (dist = 0) confounds proximity analysis; always stratify by Periphery.
2. **DNA rank = expression proxy** — no BAM / MERCI-mtSNP; transfer scores reflect mitochondrial gene-expression similarity, not confirmed physical transfer.
3. **Sparse exhaustion genes** — Visium HD UMI depth limits per-gene exhaustion marker analysis.
4. **Broad T cell definition** — includes 10,005 unsupervised T bins; CD8 *n* = 582 limits subtype-specific conclusions.
5. **LOO subsample** — receiver calling on 2,500 T cells; genome-wide ranks cover all 23,766.

---

## 7. Output files

| File | Description |
|------|-------------|
| `P1-CRC/results/tcell_analysis/tcell_summary.json` | Summary statistics |
| `P1-CRC/results/tcell_analysis/tcell_transfer_and_state.csv` | Per-bin transfer + state scores |
| `P1-CRC/results/tcell_analysis/tcell_transfer_state_correlations.csv` | Full correlation table |
| `P1-CRC/results/tcell_analysis/tcell_subtype_summary.csv` | Subtype-stratified medians |
| `P1-CRC/figures/tcell_analysis/*.png` | Seven figures |

**Script:** `run_tcell_mt_analysis.py`

---

## 8. Key takeaways

| Question | Answer |
|----------|--------|
| Do T cells receive mt from tumor or macrophages? | Yes — moderate scores (~0.50) from both; highly correlated |
| Which donor dominates? | Neither; spatial co-localization at invasive front |
| Does transfer increase near tumor? | **Yes at interface** (Periphery 50 µm); **no inside tumor nests** |
| Does transfer increase near macrophages? | **Yes** — strong distance decay |
| Is transfer linked to exhaustion? | **No** — flat exhaustion across transfer levels |
| Is transfer linked to activation? | **Weakly yes** — slight positive correlation |
| Effect on cytotoxic program? | **Negative** — high transfer ↔ lower GZMB/PRF1/NKG7 |
