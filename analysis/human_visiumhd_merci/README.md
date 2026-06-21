# Human Visium HD MERCI — Mitochondrial Transfer Analysis

Pipeline for **human cancer Visium HD** mitochondrial transfer using MERCI (mtSNV DNA rank + LOO-SVR RNA rank).

## Dataset

**Primary target:** TNBC Visium HD BAM on EGA ([EGAD50000002284](https://ega-archive.org/datasets/EGAD50000002284)) — requires DAC approval; not publicly downloadable.

**Analysis run:** Public substitute — **GSE280315 P1 human colorectal cancer** (`GSM8594567`), Nature Genetics 2025 Visium HD cohort. Same platform (Visium HD FFPE, 8 µm bins) with official deconvolution labels.

| Item | Value |
|------|--------|
| GEO | [GSE280315](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280315) |
| Sample | P1 Human Colon Cancer (`GSM8594567`) |
| SRA | `SRX26500647` (4 lanes, ~80 GB) |
| Resolution | 8 µm bins, singlet deconvolution |

## Quick start

```bash
export PYTHONUSERBASE=/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/mc38_visiumhd/pyuser
export PYTHONPATH="/ix1/ylee/kor11/tools/SpaceTravLR_rust/analysis/human_visiumhd_merci:$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH"

cd analysis/human_visiumhd_merci
python3 run_human_merci_analysis.py --data-dir P1-CRC --max-receivers 3000
```

## Pipeline steps

1. Load 8 µm bin matrix + official `Metadata.parquet` deconvolution
2. Filter: in-tissue, singlet, tumor periphery (`Periphery` = Tumor / 50 µm)
3. Donors = immune/myeloid bins; receivers = tumor bins
4. **RNA rank:** `MERCI_LOO_MT_est` (Python port)
5. **DNA rank:** MERCI-mtSNP from BAM if available; otherwise donor-mt expression signature proxy
6. Spatial microniche: immune proximity, nhood enrichment, DE receiver vs non-receiver

## MERCI-mtSNP (when BAM is available)

```bash
python3 run_merci_mtsnp.py \
  --bam P1-CRC/spaceranger/outs/possorted_genome_bam.bam \
  --barcodes-dir P1-CRC/barcodes \
  --genome-fa /path/to/genome.fa \
  --sample-id P1CRC \
  --out-dir P1-CRC/merci_mtsnp

python3 run_human_merci_analysis.py --data-dir P1-CRC --use-mtsnp
```

Regenerate BAM from SRA FASTQs with Space Ranger v4+ and `--create-bam=true`.

## Reports

- **[MITOCHONDRIAL_TRANSFER_REPORT.md](MITOCHONDRIAL_TRANSFER_REPORT.md)** — Full report with plots, stats, and biological interpretation
- [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) — Initial MERCI analysis summary
- [NICHE_ANALYSIS_REPORT.md](NICHE_ANALYSIS_REPORT.md) — Spatial niche analysis summary

## Outputs

Under `P1-CRC/results/` and `P1-CRC/figures/`:

- `merci_receiver_predictions.csv`
- `merci_rcm_statistics.csv`
- `biological_summary.json`
- Spatial maps: donor MT fraction, receiver calls, cell types

## TNBC EGA access

To use the original TNBC BAM (patient 1D525, 60.6 GB):

1. Apply at [EGAD50000002284](https://ega-archive.org/datasets/EGAD50000002284)
2. Contact DAC: skleeman@cshl.edu, janowit@cshl.edu
3. Re-run this pipeline with human segmented/cell-level data from that sample
