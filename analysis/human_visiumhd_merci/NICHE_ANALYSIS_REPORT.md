# Spatial Niche Analysis — Mitochondrial Transfer (P1 CRC)

**Dataset:** GSE280315 P1 human colorectal cancer, Visium HD 8 µm bins  
**Metric:** Donor mitochondrial transfer score (normalized DNA-proxy rank across 121,378 tumor singlet bins)  
**Script:** `run_spatial_niche_analysis.py`

## Executive summary

Mitochondrial transfer in P1 CRC is **not uniform** across the tumor. Transfer is strongly enriched at the **tumor invasive front** and in **immune-adjacent microniches**, with clear molecular and spatial structure.

---

## 1. Tumor core vs invasive front (strongest spatial axis)

| Zone | Bins | Median transfer score |
|------|------|----------------------|
| **50 µm periphery band** (tumor–stroma interface) | 1,755 | **0.721** |
| **Tumor core** | 119,456 | 0.496 |

Mann–Whitney p ≈ **7.7×10⁻¹⁰⁷** (front > core)

The official `Periphery` annotation from the Nature Genetics study marks a 50 µm band at the tumor edge. Mitochondrial transfer is **~45% higher** in this invasive front than in the bulk tumor interior. This is the single clearest spatial niche effect.

![Core vs front](P1-CRC/figures/niche_analysis/niche_core_vs_front.png)

---

## 2. Distance to immune/myeloid cells (gradient niche)

Immune and myeloid bins concentrate in the 50 µm band (1,941 bins vs 0 in tumor core). Transfer decays with distance from the nearest immune/myeloid bin:

| Distance to immune (µm) | Bins | Median transfer |
|-------------------------|------|-----------------|
| **0–50** | 1,958 | **0.740** |
| 50–100 | 8,826 | 0.583 |
| 100–200 | 28,445 | 0.490 |
| 200–400 | 49,472 | 0.481 |
| 400–1000 | 31,988 | 0.507 |

Transfer is highest **immediately adjacent to immune infiltrates** and falls to baseline beyond ~100–200 µm. This supports a model where mitochondrial transfer requires proximity to donor immune/myeloid cells.

**Local immune density (200 µm radius):** tumor bins in the top quartile of nearby immune density have median transfer 0.521 vs 0.491 in the bottom quartile (p ≈ **7.5×10⁻³⁶**).

---

## 3. Tumor molecular sub-niches (deconvolution)

Within tumor bins, transfer varies by deconvolved subtype:

| Subtype (Label1) | Bins | Median transfer | vs Tumor II |
|------------------|------|-----------------|-------------|
| **Tumor III** | 548 | **0.787** | +58% |
| **Tumor V** | 1,280 | **0.700** | +41% |
| Tumor I | 65 | 0.962 | (small n) |
| **Tumor II** (bulk) | 119,456 | 0.496 | baseline |

Tumor V > Tumor II: p ≈ **1.7×10⁻⁶⁶**

Label2 niches with highest transfer among tumor-assigned bins include **proliferating macrophage-adjacent** and **CAF-adjacent** regions:

| Label2 niche | Median transfer |
|--------------|-----------------|
| Proliferating Macrophages | 0.744 |
| Macrophage | 0.713 |
| CAF | 0.785 |
| Proliferating Fibroblast | 0.747 |

MERCI receiver calls (subsampled LOO-SVR) agree: **Tumor V** receiver rate 35% vs **Tumor II** 19%; **50 µm band** 32% vs core 19%.

---

## 4. Spatial hotspot clusters

K-means on tumor bin coordinates (k=12) identifies regional hotspots:

| Cluster | Bins | Centroid (x, y) | Median transfer |
|---------|------|-----------------|-----------------|
| **C7** | 1,134 | (12,882, 19,806) | **0.652** |
| C11 | 12,405 | (21,857, 35,245) | 0.555 |
| C10 | 11,790 | (22,671, 27,923) | 0.536 |
| C6 | 11,261 | (13,508, 29,623) | 0.529 |

Cluster **C7** is a compact high-transfer microniche (likely a tumor–stroma junction). Clusters 5, 6, 10, 11 form a contiguous high-transfer region in one sector of the section.

---

## 5. Biological interpretation

### Niches **more likely** to have mitochondrial transfer

1. **Invasive front (50 µm band)** — tumor cells at the leading edge facing stroma
2. **Immune-adjacent zones** — within 50–100 µm of macrophages, T cells, neutrophils
3. **Tumor III / Tumor V molecular states** — minor clonal/transcriptional programs enriched for transfer
4. **Proliferating macrophage neighborhoods** — highest MERCI receiver rates (~46%)
5. **Spatial sector C7** — focal hotspot, possibly a necrotic/inflammatory margin

### Niches **less likely**

1. **Tumor II bulk interior** — dominant population, baseline transfer
2. **>200 µm from immune infiltrate** — transfer drops to baseline
3. **Adipocyte-adjacent regions** (Label2) — depleted transfer (median 0.34)

### Proposed microniche model

```
Stroma (CAF, macrophages, T cells)
    ↓ proximity / contact
50 µm invasive front  ←── HIGH transfer zone
    ↓
Tumor II core interior ←── LOW transfer (baseline)
```

Transfer appears to be a **periphery-driven, immune-proximity-dependent** process rather than a uniform property of all tumor cells. This aligns with the Nature Genetics finding of distinct macrophage subpopulations at the tumor periphery.

---

## Output files

| File | Description |
|------|-------------|
| `P1-CRC/results/niche_analysis/niche_summary.json` | Machine-readable summary |
| `P1-CRC/results/niche_analysis/niche_enrichment_stats.csv` | All niche enrichment tests |
| `P1-CRC/results/niche_analysis/transfer_by_immune_distance.csv` | Distance gradient |
| `P1-CRC/results/niche_analysis/spatial_hotspot_clusters.csv` | Regional clusters |
| `P1-CRC/results/niche_analysis/tumor_bins_with_transfer_scores.csv` | Per-bin scores |
| `P1-CRC/figures/niche_analysis/*.png` | Spatial maps and barplots |

## Re-run

```bash
export PYTHONUSERBASE=.../mc38_visiumhd/pyuser
export PYTHONPATH=".../human_visiumhd_merci:$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH"
python3 run_spatial_niche_analysis.py --data-dir P1-CRC
```

## Caveats

- Transfer score uses **DNA-proxy** (donor-mt expression signature), not true mtSNV counts
- Analysis is at **8 µm bin** resolution, not single-cell
- Tumor I / small-n niches should be interpreted cautiously
- Causal direction (immune proximity → transfer vs transfer → immune recruitment) cannot be resolved from spatial data alone
