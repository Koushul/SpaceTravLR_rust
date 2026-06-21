# Human Visium HD MERCI Analysis Report (P1 CRC)

**Dataset:** GSE280315 P1 human colorectal cancer, Visium HD 8 µm bins ([Nature Genetics 2025](https://www.nature.com/articles/s41588-025-02193-3)).

**Note:** The originally recommended TNBC BAM ([EGAD50000002284](https://ega-archive.org/datasets/EGAD50000002284)) requires controlled EGA access. This analysis uses the public P1 CRC Visium HD sample as the accessible human cancer substitute on the same platform.

## Methods

- **Units:** 8 µm spatial bins with `DeconvolutionClass == singlet` in tumor periphery (`Periphery` ∈ {Tumor, 50 micron})
- **Donors:** immune + myeloid bins (T cells, B cells, macrophages, neutrophils)
- **Receivers:** tumor bins (Tumor I–V deconvolution labels)
- **MERCI RNA:** LOO linear SVR on MT gene expression (`organism=human`)
- **MERCI DNA:** donor mitochondrial expression signature proxy (BAM unavailable — SRA download blocked on cluster TLS; MERCI-mtSNP not run)
- **Subsample:** 3,000 tumor bins for LOO-SVR runtime

## Key results

| Metric | Value |
|--------|-------|
| Tumor bins analyzed | 3,000 |
| MERCI receivers | 566 (18.9%) |
| Rcm at 10% cutoff | 2.83 (>1, enrichment signal) |
| Rcm at 30% cutoff | 0.75 (<1, weak at broader cutoffs) |
| Median %MT (receiver) | 3.94% |
| Median %MT (non-receiver) | 4.32% |
| Stress gene score (MW p) | 0.69 (not significant) |

### Receiver enrichment by tumor subtype (deconvolution)

- **Tumor II:** 549 / 2,947 receiver calls (18.6%)
- **Tumor V:** 12 / 34 (35.3%)
- **Tumor III:** 5 / 19 (26.3%)

### Spatial context

- Receivers concentrated in core tumor (`Periphery = Tumor`: 549/566)
- 17/566 receivers at 50 µm periphery band

## Biological interpretation

1. **Mitochondrial transfer signal is detectable** in human CRC Visium HD data: Rcm > 1 at strict rank cutoffs indicates concordance between DNA-proxy and RNA-based donor-mt ranks among top candidates.
2. **~19% of tumor periphery bins** show receiver-like mitochondrial profiles — comparable in magnitude to the preliminary MC38 SPAC-seq analysis (~39% at cell level with imperfect guide mapping).
3. **Tumor II** dominates the spatial landscape and contributes most receiver calls; smaller tumor subclusters (III, V) show higher receiver fractions but low counts.
4. **No stress-response difference** between receivers and non-receivers (HSPA1A/B, ATF4, DDIT3, BAX), suggesting transferred mtDNA may not immediately trigger a canonical stress program at the bin level.

## Limitations

1. **No BAM / MERCI-mtSNP:** DNA rank uses expression signature proxy, not true mtSNV counts. For publication-grade MERCI, obtain TNBC EGA BAM or regenerate BAM from SRA FASTQs with `spaceranger count --create-bam=true`.
2. **Bin-level not cell-level:** GEO P1 deposit provides 8 µm bins, not segmented cells. Segmented outputs are on OSF (`VisiumHD_HumanColon_Oliveira`) and 10x full bundle.
3. **Human not mouse:** Cannot directly compare to MC38 SPAC-seq perturbation clones without a separate human perturbation dataset.

## Next steps

1. Apply for EGA TNBC BAM (EGAD50000002284) and re-run with `run_merci_mtsnp.py --use-mtsnp`
2. Download OSF segmented P1 outputs for cell-resolution MERCI
3. Integrate with Xenium / scRNA-seq from GSE280318 super-series for donor cell state characterization

## Output files

- `P1-CRC/results/merci_receiver_predictions.csv`
- `P1-CRC/results/biological_summary.json`
- `P1-CRC/figures/spatial_mt_receivers.png`
- `P1-CRC/figures/spatial_donor_mt_frac.png`
- `P1-CRC/figures/immune_proximity_receiver.png`
