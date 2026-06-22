# Refined Cell Types and Tumor Mitochondrial Receivers — P1 CRC

**GSE280315 Patient 1, Visium HD 8 µm bins**

## Annotation refinement

Official `DeconvolutionLabel1` was harmonized with `UnsupervisedL1/L2` and marker scores (`score_tumor`, `score_immune`, `score_myeloid`, `score_stroma`, `score_epithelial`):

1. **Agree** — Label1 lineage matches UnsupervisedL1 → keep Label1 fine type.
2. **Conflict** — marker-score tie-break when scores support one lineage; otherwise prefer UnsupervisedL1.
3. **Fine type** — resolve mixed bins using Label2 and unsupervised clusters (e.g. `Tcells-*` → `T cell (unsupervised)`).

| Metric | Value |
|--------|-------|
| Total singlet bins | 235,530 |
| Tumor donor bins | 117,393 |
| Non-tumor receiver bins | 118,137 |
| Bins with refined label ≠ naive Label1 lineage | 28,679 (12.2%) |

Refined annotations: `P1-CRC/results/celltype_mt_receivers/refined_celltype_annotations.csv`

---

## MERCI: tumor mitochondria → all non-tumor cell types

| Parameter | Value |
|-----------|-------|
| Donors | All tumor bins (`lineage_refined == tumor`) |
| Receivers | All non-tumor bins |
| Sampling | Up to **500 bins per refined cell type** (stratified) |
| Tested receivers | 11,914 |
| MERCI receiver calls (tested) | **2,547 (22.2%)** |
| Estimated receivers (extrapolated) | **~24,483** |
| DNA rank | Tumor mt-expression proxy (no BAM) |
| RNA rank | LOO-SVR, top 50% DNA+RNA overlap |

---

## Table: receivers by refined cell type

Sorted by estimated receiver count. **Rate** = tested receivers / tested bins. **Estimated receivers** extrapolates rate to all bins of that type (exact for fully tested rare types).

| Cell type | Lineage | Total bins | Tested | Receivers (tested) | Rate | Est. receivers |
|-----------|---------|------------|--------|-------------------|------|----------------|
| Goblet | epithelial | 35,920 | 500 | 104 | 20.8% | **7,471** |
| CAF | stroma | 25,877 | 475 | 87 | 18.3% | **4,740** |
| Endothelial | endothelial | 10,064 | 483 | 95 | 19.7% | **1,979** |
| Enterocyte | epithelial | 7,549 | 500 | 129 | 25.8% | **1,948** |
| Mature B | immune | 5,719 | 498 | 94 | 18.9% | **1,079** |
| vSM | stroma | 4,177 | 487 | 95 | 19.5% | **815** |
| CD4 T cell | immune | 4,254 | 485 | 90 | 18.6% | **789** |
| Pericytes | stroma | 2,813 | 485 | 117 | 24.1% | **679** |
| Fibroblast | stroma | 2,947 | 442 | 99 | 22.4% | **660** |
| Macrophage | myeloid | 2,829 | 480 | 111 | 23.1% | **654** |
| Proliferating Fibroblast | stroma | 2,107 | 493 | 122 | 24.7% | **521** |
| T cell (unsupervised) | immune | 1,762 | 490 | 128 | 26.1% | **460** |
| Plasma | immune | 2,110 | 460 | 95 | 20.7% | **436** |
| Myofibroblast | stroma | 2,099 | 498 | 101 | 20.3% | **426** |
| Proliferating Macrophages | myeloid | 1,244 | 493 | 129 | 26.2% | **326** |
| Neutrophil | myeloid | 1,272 | 466 | 113 | 24.2% | **308** |
| Lymphatic Endothelial | endothelial | 1,033 | 459 | 92 | 20.0% | **207** |
| Fibroblast (unsupervised) | stroma | 1,031 | 500 | 99 | 19.8% | **204** |
| Proliferating Immune II | immune | 710 | 487 | 118 | 24.2% | **172** |
| CD8 T cell | immune | 773 | 479 | 91 | 19.0% | **147** |
| Enteric Glial | neuronal | 489 | 460 | 126 | 27.4% | **134** |
| Smooth Muscle | stroma | 430 | 377 | 85 | 22.5% | **97** |
| cDC I | myeloid | 147 | 143 | 39 | 27.3% | **40** |
| Neuroendocrine | epithelial | 157 | 157 | 33 | 21.0% | **33** |
| Myeloid (unsupervised) | myeloid | 124 | 124 | 24 | 19.4% | **24** |
| Mast | myeloid | 96 | 91 | 20 | 22.0% | **21** |
| mRegDC | myeloid | 99 | 96 | 19 | 19.8% | **20** |
| Tuft | epithelial | 43 | 43 | 18 | 41.9% | **18** |
| Memory B | immune | 100 | 100 | 17 | 17.0% | **17** |
| Stromal (unsupervised) | stroma | 68 | 62 | 14 | 22.6% | **15** |
| Epithelial | epithelial | 37 | 36 | 14 | 38.9% | **14** |
| Unknown III (SM) | unknown | 26 | 25 | 8 | 32.0% | **8** |
| Vascular Fibroblast | stroma | 29 | 29 | 5 | 17.2% | **5** |
| SM Stress Response | stroma | 19 | 18 | 4 | 22.2% | **4** |
| Intestinal epithelial (unsup.) | epithelial | 14 | 14 | 4 | 28.6% | **4** |
| pDC | myeloid | 16 | 16 | 4 | 25.0% | **4** |
| Adipocyte | epithelial | 5 | 5 | 2 | 40.0% | **2** |
| Neuronal (unsupervised) | neuronal | 4 | 4 | 2 | 50.0% | **2** |
| NK | immune | 11 | 11 | 0 | 0.0% | **0** |

### By lineage (estimated)

| Lineage | Total bins | Est. receivers | Rate |
|---------|------------|----------------|------|
| Epithelial | 43,735 | ~9,490 | 21.7% |
| Stroma | 38,598 | ~8,166 | 21.2% |
| Immune | 16,439 | ~3,100 | 18.9% |
| Endothelial | 11,097 | ~2,186 | 19.7% |
| Myeloid | 5,827 | ~1,377 | 23.6% |
| Neuronal | 493 | ~136 | 27.6% |
| Unknown | 26 | ~8 | 30.8% |

---

## Key observations

1. **Goblet and CAF dominate absolute receiver counts** because of abundance, not the highest per-bin rates.
2. **Highest rates** among well-powered types: Proliferating Macrophages (26.2%), T cell unsupervised (26.1%), Enterocyte (25.8%), Enteric Glial (27.4%).
3. **Myeloid cells** (macrophages, neutrophils, DCs) show **~20–26%** receiver rates — consistent with interface transfer.
4. **T cells** (CD4, CD8, unsupervised) are **~19–26%** — similar to stroma and epithelium.
5. **Tumor bins are donors only** and excluded from the receiver table.

---

## Output files

| File | Description |
|------|-------------|
| `P1-CRC/results/celltype_mt_receivers/tumor_mt_receivers_by_refined_celltype.csv` | Main table |
| `P1-CRC/results/celltype_mt_receivers/tumor_mt_receivers_by_label1.csv` | Original Label1 comparison |
| `P1-CRC/results/celltype_mt_receivers/refined_celltype_annotations.csv` | Per-bin refined labels |
| `P1-CRC/results/celltype_mt_receivers/merci_tumor_to_receiver_predictions.csv` | Per-bin MERCI calls |
| `P1-CRC/figures/celltype_mt_receivers/*.png` | Bar plots |

**Scripts:** `celltype_annotation.py`, `run_celltype_mt_receiver_table.py`
