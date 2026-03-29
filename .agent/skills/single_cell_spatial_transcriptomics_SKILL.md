# Single-Cell & Spatial Transcriptomics Analysis — Comprehensive Reference

> **Purpose**: This document is a grounded, end-to-end reference for LLMs and computational biologists processing single-cell RNA-seq (scRNA-seq) and spatial transcriptomics data. It covers biological motivation, platform-specific considerations, standard and advanced analytical pipelines, mathematical foundations, code implementations, common pitfalls, and best practices. All guidance is rooted in the biology of gene expression and tissue organization.

---

## Table of Contents

1. [Biological Foundations](#1-biological-foundations)
2. [Experimental Platforms & Technologies](#2-experimental-platforms--technologies)
3. [Raw Data Processing & Quantification](#3-raw-data-processing--quantification)
4. [Quality Control & Filtering](#4-quality-control--filtering)
5. [Normalization & Transformation](#5-normalization--transformation)
6. [Feature Selection & Dimensionality Reduction](#6-feature-selection--dimensionality-reduction)
7. [Batch Effects & Data Integration](#7-batch-effects--data-integration)
8. [Clustering & Cell Type Annotation](#8-clustering--cell-type-annotation)
9. [Differential Expression Analysis](#9-differential-expression-analysis)
10. [Trajectory & RNA Velocity Analysis](#10-trajectory--rna-velocity-analysis)
11. [Gene Regulatory Network Inference](#11-gene-regulatory-network-inference)
12. [Cell-Cell Communication](#12-cell-cell-communication)
13. [Spatial Transcriptomics — Platform-Specific Processing](#13-spatial-transcriptomics--platform-specific-processing)
14. [Spatial Analysis Methods](#14-spatial-analysis-methods)
15. [Multi-Modal & Multi-Omics Integration](#15-multi-modal--multi-omics-integration)
16. [Perturbation & CRISPR Screen Analysis](#16-perturbation--crispr-screen-analysis)
17. [Scalability, Formats & Infrastructure](#17-scalability-formats--infrastructure)
18. [Foundation Models & Deep Learning](#18-foundation-models--deep-learning)
19. [Reproducibility & Reporting](#19-reproducibility--reporting)
20. [Quick-Reference Decision Trees](#20-quick-reference-decision-trees)

---

## 1. Biological Foundations

### 1.1 Why Single-Cell Resolution Matters

Bulk RNA-seq measures the average expression across millions of cells, collapsing heterogeneity. Tissues are composed of dozens to hundreds of transcriptionally distinct cell types and states. A tumor biopsy contains malignant cells, fibroblasts, endothelial cells, and diverse immune populations — bulk profiling obscures the biology of each. Single-cell resolution enables:

- **Cell type discovery**: Identifying novel or rare populations (e.g., a new T-cell subset in the tumor microenvironment).
- **State characterization**: Distinguishing activated vs. resting macrophages, cycling vs. quiescent stem cells.
- **Lineage reconstruction**: Tracing differentiation trajectories from progenitors to terminal fates.
- **Regulatory logic**: Inferring gene regulatory networks operating within specific cell types rather than across a tissue average.

### 1.2 Why Spatial Context Matters

Dissociation-based single-cell methods destroy tissue architecture. But biology is spatial: paracrine signaling requires physical proximity, morphogen gradients pattern developing tissues, and the tumor-immune interface is defined by spatial organization. Spatial transcriptomics preserves coordinate information, enabling:

- **Niche identification**: Defining the microenvironments (e.g., perivascular niche, germinal center) that support specific cell states.
- **Ligand-receptor modeling in situ**: Asking whether cells expressing a ligand are physically adjacent to cells expressing its receptor.
- **Tissue morphology–transcriptome links**: Correlating histological features (fibrosis, necrosis) with local gene expression programs.
- **Spatial gradients**: Detecting zonation patterns (e.g., hepatocyte zonation along the portal-central axis of the liver lobule).

### 1.3 The Central Dogma in a Single-Cell Context

Gene expression in individual cells is inherently stochastic (transcriptional bursting). scRNA-seq captures a snapshot of mRNA molecules present at the moment of lysis. Key biological realities that shape analysis:

- **Sparsity is biological, not just technical**: Many genes are expressed in bursts; a zero in the count matrix may reflect a gene that is "off" in that cell, a transcript that was not captured (dropout), or genuinely low expression below the detection limit.
- **Splicing kinetics matter**: Unspliced (nascent) and spliced (mature) mRNA have different abundances. RNA velocity exploits the ratio of these to infer the direction of transcriptional change.
- **Cell cycle is a confounder**: Proliferating cells express a conserved program (e.g., TOP2A, MKI67, PCNA) that can dominate variation. Regression or explicit annotation is often necessary.
- **Mitochondrial content signals stress**: High mitochondrial transcript fractions (e.g., >15–20% for many tissues) often indicate dying or damaged cells whose cytoplasmic mRNA has leaked, enriching for mitochondrially-encoded transcripts.

### 1.4 Mathematical Representation of scRNA-seq Data

The fundamental data object is a **count matrix** `X` of shape `(n_cells × n_genes)` where entry `X[i, j]` represents the number of unique molecular identifiers (UMIs) for gene `j` detected in cell `i`.

For UMI-based data, the generative model is approximately:

```
X[i,j] ~ Poisson(s_i * μ_j * θ_ij)
```

where:
- `s_i` = size factor (sequencing depth) for cell `i`
- `μ_j` = baseline expression rate for gene `j`
- `θ_ij` = cell-specific, gene-specific biological variation

In practice, overdispersion (variance > mean) is common, motivating the **negative binomial** model:

```
X[i,j] ~ NB(μ_ij, r_j)

where:
  E[X[i,j]] = μ_ij
  Var[X[i,j]] = μ_ij + μ_ij² / r_j
  r_j = inverse dispersion parameter for gene j (higher r → less overdispersion → approaches Poisson)
```

This is the foundation for tools like DESeq2, scVI, and SCTransform.

---

## 2. Experimental Platforms & Technologies

### 2.1 Droplet-Based scRNA-seq

| Platform | Chemistry | Throughput | UMI | Capture | Notes |
|---|---|---|---|---|---|
| **10x Genomics Chromium (3' v3/v3.1)** | Gel bead-in-emulsion (GEM) | 500–10,000 cells/lane | Yes | 3' polyA | Industry standard; CellRanger pipeline |
| **10x Genomics Chromium (5')** | GEM | Similar | Yes | 5' end | Enables TCR/BCR enrichment for immune profiling |
| **10x Genomics Chromium Single Cell Multiome** | GEM | ~500–10,000 nuclei | Yes | 3' RNA + ATAC | Joint transcriptome + chromatin accessibility from same nucleus |
| **10x Genomics Flex (Fixed RNA Profiling)** | Probe-based on fixed cells | Up to 128 samples multiplexed | Yes | Targeted probe panel or whole transcriptome | FFPE-compatible; uses probe ligation |
| **Drop-seq** | Microfluidic droplets | 10,000+ cells | Yes | 3' polyA | Lower cost, higher noise; uses barcoded beads in drops |
| **inDrop** | Hydrogel droplets | Similar to Drop-seq | Yes | 3' polyA | Academic alternative |
| **BD Rhapsody** | Microwell cartridge | Up to 40,000 cells | Yes | 3' polyA or targeted panels | Deterministic barcoding; good for targeted panels |

**Key biology note**: Droplet methods capture the 3' or 5' end of transcripts, not full-length. This means isoform-level resolution is limited. The 3' bias also means that genes with long 3' UTRs may appear more highly expressed.

### 2.2 Plate-Based / Full-Length scRNA-seq

| Platform | Chemistry | Throughput | UMI | Capture | Notes |
|---|---|---|---|---|---|
| **Smart-seq2** | Template switching + PCR | 96–384 cells/run | No | Full-length | Gold standard for sensitivity; detects more genes per cell |
| **Smart-seq3** | Template switching + UMIs | 96–384 cells/run | Yes (5' UMIs) | Full-length with UMIs | Combines full-length with UMI counting |
| **VASA-seq** | Total RNA | 384 cells/run | Yes | Full-length total RNA | Captures non-polyadenylated RNA (histones, some lncRNAs) |
| **CEL-Seq2** | Linear amplification by IVT | 96–384 cells | Yes | 3' end | Lower amplification bias than PCR-based |
| **FLASH-seq** | Fast Smart-seq variant | 384 cells | No | Full-length | 3-hour protocol; good sensitivity |

**When to use plate-based**: When you need isoform resolution, maximum sensitivity per cell, allele-specific expression, SNV calling from transcripts, or when studying rare populations where every cell matters. Trade-off is throughput.

### 2.3 Combinatorial Indexing (Split-Pool)

| Platform | Chemistry | Throughput | UMI | Notes |
|---|---|---|---|---|
| **SPLiT-seq** | Split-and-pool barcoding | 100,000+ nuclei | Yes | No special equipment; works on fixed cells/nuclei |
| **sci-RNA-seq3** | Combinatorial indexing | >1 million cells | Yes | Ultra-high throughput; lower sensitivity per cell |
| **SHARE-seq** | Combinatorial indexing | ~100,000 | Yes | Joint RNA + ATAC |
| **Parse Biosciences (Evercode)** | Combinatorial barcoding | 100,000+ cells/nuclei | Yes | Commercial split-pool; no microfluidics needed |

**Biology note**: Combinatorial indexing typically works on nuclei, not whole cells. Nuclear transcriptomes are enriched for nascent/unspliced transcripts and may underrepresent cytoplasmic mRNAs. This matters for genes with fast mRNA turnover.

### 2.4 Spatial Transcriptomics Platforms

#### 2.4.1 Imaging-Based (Single-Molecule FISH Family)

| Platform | Method | Genes | Resolution | Notes |
|---|---|---|---|---|
| **MERFISH** (Vizgen MERSCOPE) | Combinatorial FISH + error correction | 100–1,000+ genes | Subcellular (~100 nm) | Targeted panel; excellent sensitivity per transcript |
| **seqFISH+** | Sequential hybridization | 10,000+ genes | Subcellular | Transcriptome-scale imaging; technically demanding |
| **10x Genomics Xenium** | Padlock probe + rolling circle amplification | 100–5,000 genes (v2 panels) | Subcellular (~200 nm) | Commercial; growing panel sizes |
| **CosMx (NanoString)** | ISH with optical barcoding | 1,000–6,000 genes | Subcellular | Integrated with protein co-detection (up to 64 proteins) |
| **STARmap / STARmap PLUS** | In situ sequencing | 1,000+ genes | Subcellular | Hydrogel-tissue chemistry for thick sections |

#### 2.4.2 Sequencing-Based (Capture Array)

| Platform | Method | Resolution | Transcriptome | Notes |
|---|---|---|---|---|
| **10x Visium** | Oligonucleotide-spotted array | 55 µm spots | Whole transcriptome | Each spot covers ~1–10 cells; NOT single-cell |
| **10x Visium HD** | High-density array | 2 µm bins (aggregated to 8 µm) | Whole transcriptome | Near-single-cell; binning strategy matters |
| **Slide-seq / Slide-seqV2** | DNA-barcoded beads on glass | ~10 µm | Whole transcriptome | High resolution; lower sensitivity per bead |
| **Stereo-seq** (BGI STOmics) | DNA nanoball patterned array | 500 nm (aggregated to bins) | Whole transcriptome | Largest capture areas; bin size is user-defined |
| **Seq-Scope** | Repurposed Illumina flow cell | Subcellular | Whole transcriptome | Novel approach |
| **PIXEL-seq** | Polony barcoded array | ~1 µm | Whole transcriptome | Near-subcellular |

#### 2.4.3 Spatial Proteomics & Multi-Modal Spatial

| Platform | Modality | Notes |
|---|---|---|
| **CODEX / PhenoCycler (Akoya)** | Protein (40–100 markers) | Cyclic antibody staining; spatial protein expression |
| **MIBI-TOF** | Protein (~40 markers) | Mass spectrometry imaging with metal-tagged antibodies |
| **10x Visium CytAssist** | H&E + spatial transcriptomics | FFPE-compatible; maps transcriptome to histology |
| **CosMx** | RNA + protein (64 proteins) | Simultaneous RNA + protein spatial |

---

## 3. Raw Data Processing & Quantification

### 3.1 10x Genomics Chromium → Cell Ranger

#### Command-Line Usage

```bash
# Step 1: Demultiplex BCL files to FASTQ
cellranger mkfastq \
    --id=sample_demux \
    --run=/path/to/sequencer/output \
    --csv=samplesheet.csv

# Step 2: Alignment + counting (standard scRNA-seq, whole cells)
cellranger count \
    --id=sample_count \
    --transcriptome=/path/to/refdata-gex-GRCh38-2024-A \
    --fastqs=/path/to/fastqs \
    --sample=SampleName \
    --expect-cells=5000 \
    --localcores=16 \
    --localmem=64

# Step 2 (alt): Single-nucleus RNA-seq — MUST include introns
cellranger count \
    --id=sample_snrnaseq \
    --transcriptome=/path/to/refdata-gex-GRCh38-2024-A \
    --fastqs=/path/to/fastqs \
    --sample=SampleName \
    --include-introns=true \
    --expect-cells=8000

# Step 2 (alt): Multiplexed experiment (CellPlex / hashtag / CRISPR / Flex)
cellranger multi \
    --id=sample_multi \
    --csv=multi_config.csv

# Aggregation across samples (crude normalization — prefer manual aggregation downstream)
cellranger aggr \
    --id=aggregated \
    --csv=aggr_libraries.csv \
    --normalize=mapped
```

#### multi_config.csv example (for Flex or multiplexed)

```csv
[gene-expression]
reference,/path/to/refdata-gex-GRCh38-2024-A
probe-set,/path/to/Chromium_Human_Transcriptome_Probe_Set_v1.0.1_GRCh38-2024-A.csv

[libraries]
fastq_id,fastqs,feature_types
Sample1,/path/to/fastqs,Gene Expression

[samples]
sample_id,probe_barcode_ids,description
SampleA,BC001,Treatment A
SampleB,BC002,Treatment B
```

#### Cell Ranger Read Structure (3' v3)

```
Read 1 (28 bp):  [Cell Barcode (16 bp)][UMI (12 bp)]
Read 2 (91 bp):  [cDNA insert → aligned to transcriptome]
Index (8 bp):    [Sample index for demultiplexing]
```

#### Cell Ranger Output Structure

```
sample_count/outs/
├── filtered_feature_bc_matrix/       # Cells called by the algorithm
│   ├── barcodes.tsv.gz
│   ├── features.tsv.gz
│   └── matrix.mtx.gz
├── raw_feature_bc_matrix/            # ALL barcodes (including empty drops)
│   ├── barcodes.tsv.gz
│   ├── features.tsv.gz
│   └── matrix.mtx.gz
├── filtered_feature_bc_matrix.h5     # HDF5 format of filtered matrix
├── raw_feature_bc_matrix.h5          # HDF5 format of raw matrix
├── possorted_genome_bam.bam          # Aligned reads
├── possorted_genome_bam.bam.bai
├── molecule_info.h5                  # Per-molecule info (for aggr)
├── metrics_summary.csv               # QC metrics
├── web_summary.html                  # Interactive QC report
└── cloupe.cloupe                     # Loupe Browser file
```

#### Cell Calling Algorithm (EmptyDrops-inspired)

The algorithm distinguishes cell-containing droplets from empty droplets:

1. Rank all barcodes by total UMI count (descending).
2. Identify the "knee" — the inflection point in the log-rank vs. log-UMI curve.
3. Barcodes above the knee are confidently cells.
4. For barcodes below the knee but above the ambient threshold, test against the ambient RNA profile using a Dirichlet-multinomial likelihood ratio:

```
H₀: barcode expression profile ~ Dirichlet-Multinomial(ambient profile)
H₁: barcode expression profile deviates from ambient

Test statistic: log-likelihood ratio
p-value via Monte Carlo simulation
```

#### Building a Custom Reference (e.g., adding GFP transgene)

```bash
# Add GFP to genome FASTA
cat genome.fa gfp.fa > genome_gfp.fa

# Add GFP annotation to GTF
cat genes.gtf gfp.gtf > genes_gfp.gtf

# Build Cell Ranger reference
cellranger mkref \
    --genome=GRCh38_GFP \
    --fasta=genome_gfp.fa \
    --genes=genes_gfp.gtf
```

### 3.2 STARsolo — Fast, Open-Source Alternative

```bash
# Standard 10x Chromium 3' v3 processing
STAR --runMode alignReads \
    --genomeDir /path/to/STAR_index \
    --readFilesIn Read2.fastq.gz Read1.fastq.gz \
    --readFilesCommand zcat \
    --soloType CB_UMI_Simple \
    --soloCBwhitelist /path/to/3M-february-2018.txt \
    --soloCBstart 1 --soloCBlen 16 \
    --soloUMIstart 17 --soloUMIlen 12 \
    --soloFeatures Gene Velocyto \
    --outSAMtype BAM SortedByCoordinate \
    --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM \
    --runThreadN 16 \
    --outFileNamePrefix ./starsolo_output/

# Key flags:
# --soloFeatures Gene Velocyto  → outputs spliced/unspliced/ambiguous counts
#   (Gene = standard count matrix; Velocyto = S/U/A matrices for RNA velocity)
# --soloCBwhitelist              → 10x barcode whitelist file
# --soloType CB_UMI_Simple       → standard cell barcode + UMI layout
```

**STARsolo output**: `Solo.out/Gene/filtered/` contains MEX-format matrices identical in structure to Cell Ranger output. `Solo.out/Velocyto/filtered/` contains spliced, unspliced, and ambiguous matrices.

### 3.3 Alevin-fry (Salmon-Based Pseudoalignment)

```bash
# Step 1: Build splici (spliced + intronic) reference with simpleaf
pip install simpleaf

simpleaf index \
    --output /path/to/af_index \
    --fasta genome.fa \
    --gtf genes.gtf \
    --rlen 91 \
    --threads 16

# Step 2: Map and quantify
simpleaf quant \
    --reads1 Read1.fastq.gz \
    --reads2 Read2.fastq.gz \
    --index /path/to/af_index/index \
    --chemistry 10xv3 \
    --resolution cr-like \
    --expected-ori fw \
    --t2g-map /path/to/af_index/index/t2g_3col.tsv \
    --output /path/to/af_output \
    --threads 16

# The splici index generates counts for:
#   S (spliced/exonic), U (unspliced/intronic), A (ambiguous)
# These are natively compatible with RNA velocity pipelines
```

#### Loading Alevin-fry Output into Python

```python
import pyroe

# Load USA (unspliced/spliced/ambiguous) counts
adata = pyroe.load_fry("/path/to/af_output", output_format="scRNA")
# output_format options: "scRNA" (S+A as counts), "snRNA" (S+U+A), "velocity" (separate layers)

# For RNA velocity, load with separate layers:
adata = pyroe.load_fry("/path/to/af_output", output_format={
    "X": ["S", "A"],        # Main count matrix = spliced + ambiguous
    "unspliced": ["U"],      # Unspliced layer
    "spliced": ["S"],        # Spliced-only layer
    "ambiguous": ["A"]       # Ambiguous layer
})
```

### 3.4 Kallisto | BUStools

```bash
# Install
pip install kb-python

# Build reference (standard)
kb ref \
    -i index.idx \
    -g t2g.txt \
    -f1 cdna.fa \
    --workflow standard \
    genome.fa genes.gtf

# Build reference for RNA velocity (nascent/ambiguous/committed)
kb ref \
    -i index.idx \
    -g t2g.txt \
    -f1 cdna.fa -f2 intron.fa \
    --workflow nac \
    genome.fa genes.gtf

# Count (standard)
kb count \
    -i index.idx \
    -g t2g.txt \
    -x 10xv3 \
    -o output_dir \
    --workflow standard \
    Read1.fastq.gz Read2.fastq.gz

# Count (RNA velocity)
kb count \
    -i index.idx \
    -g t2g.txt \
    -x 10xv3 \
    -o output_dir \
    --workflow nac \
    Read1.fastq.gz Read2.fastq.gz
```

### 3.5 velocyto CLI (for RNA velocity from Cell Ranger BAM)

```bash
# Run velocyto on Cell Ranger output
velocyto run10x \
    /path/to/cellranger/sample_count \
    /path/to/genes.gtf

# For non-10x data:
velocyto run \
    -b filtered_barcodes.tsv \
    -o output_dir \
    -m repeat_mask.gtf \
    possorted_genome_bam.bam \
    genes.gtf

# Output: sample.loom file with layers:
#   spliced, unspliced, ambiguous
```

### 3.6 Loading Raw Data into Analysis Frameworks

#### Python (Scanpy / AnnData)

```python
import scanpy as sc
import anndata as ad
import numpy as np
import scipy.sparse as sp

# --- From Cell Ranger output ---
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")
# or from MEX format:
adata = sc.read_10x_mtx("filtered_feature_bc_matrix/")

# --- From multiple samples: manual aggregation (preferred over cellranger aggr) ---
samples = {
    "control_1": "path/to/ctrl1/filtered_feature_bc_matrix.h5",
    "control_2": "path/to/ctrl2/filtered_feature_bc_matrix.h5",
    "treated_1": "path/to/treat1/filtered_feature_bc_matrix.h5",
    "treated_2": "path/to/treat2/filtered_feature_bc_matrix.h5",
}

adatas = {}
for sample_id, path in samples.items():
    a = sc.read_10x_h5(path)
    a.obs["sample"] = sample_id
    a.obs["condition"] = "control" if "control" in sample_id else "treated"
    a.var_names_make_unique()
    adatas[sample_id] = a

adata = ad.concat(adatas, label="sample", keys=adatas.keys(), index_unique="-")

# Ensure count matrix is sparse (critical for memory efficiency)
if not sp.issparse(adata.X):
    adata.X = sp.csr_matrix(adata.X)

print(f"Combined: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"Matrix density: {adata.X.nnz / np.prod(adata.X.shape):.3f}")

# --- From loom (velocyto output) ---
import scvelo as scv
adata = scv.read("sample.loom", cache=True)

# --- Store raw counts before any processing ---
adata.layers["counts"] = adata.X.copy()
```

#### R (Seurat)

```r
library(Seurat)
library(Matrix)

# From Cell Ranger output
data <- Read10X(data.dir = "filtered_feature_bc_matrix/")
seurat_obj <- CreateSeuratObject(counts = data, project = "my_project", min.cells = 3, min.features = 200)

# From multiple samples
samples <- c("ctrl1", "ctrl2", "treat1", "treat2")
paths <- c("path/to/ctrl1/filtered_feature_bc_matrix/",
           "path/to/ctrl2/filtered_feature_bc_matrix/",
           "path/to/treat1/filtered_feature_bc_matrix/",
           "path/to/treat2/filtered_feature_bc_matrix/")

seurat_list <- lapply(seq_along(samples), function(i) {
    data <- Read10X(data.dir = paths[i])
    obj <- CreateSeuratObject(counts = data, project = samples[i])
    obj$sample <- samples[i]
    obj$condition <- ifelse(grepl("ctrl", samples[i]), "control", "treated")
    return(obj)
})
seurat_obj <- merge(seurat_list[[1]], y = seurat_list[2:4], add.cell.ids = samples)
```

---

## 4. Quality Control & Filtering

### 4.1 Cell-Level QC Metrics — Code

#### Python (Scanpy)

```python
import scanpy as sc
import numpy as np

# Annotate mitochondrial, ribosomal, and hemoglobin genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")       # Human mitochondrial
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))  # Ribosomal proteins
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")  # Hemoglobin (not HBP)

# Compute QC metrics
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt", "ribo", "hb"],
    percent_top=None,
    log1p=True,    # Also compute log1p of total_counts for visualization
    inplace=True
)

# Key columns now in adata.obs:
#   n_genes_by_counts  — number of genes with ≥1 count
#   total_counts        — total UMI count
#   pct_counts_mt       — % mitochondrial
#   pct_counts_ribo     — % ribosomal
#   pct_counts_hb       — % hemoglobin

# Visualize distributions
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
             jitter=0.4, multi_panel=True)
sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", color="pct_counts_mt")
```

#### R (Seurat)

```r
library(Seurat)

# Compute % mitochondrial
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
seurat_obj[["percent.ribo"]] <- PercentageFeatureSet(seurat_obj, pattern = "^RP[SL]")
seurat_obj[["percent.hb"]] <- PercentageFeatureSet(seurat_obj, pattern = "^HB[^(P)]")

# Visualize
VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3, pt.size = 0)
FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA") +
    geom_smooth(method = "lm")
```

### 4.2 Adaptive / MAD-Based QC Thresholds

Rather than fixed thresholds, compute per-sample median absolute deviations (MADs):

```
MAD = median(|x_i - median(x)|)

Outlier if: x_i < median(x) - nmads * MAD   (for n_genes, total_counts)
            x_i > median(x) + nmads * MAD   (for pct_counts_mt)

Typical nmads = 3-5
```

#### Python Implementation

```python
import numpy as np

def mad_outlier(adata, metric, nmads=3, direction="both"):
    """Flag outliers based on MAD for a given QC metric."""
    vals = adata.obs[metric].values
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    # Avoid zero MAD
    mad = max(mad, 1e-6)

    if direction == "lower":
        return vals < med - nmads * mad
    elif direction == "upper":
        return vals > med + nmads * mad
    else:
        return (vals < med - nmads * mad) | (vals > med + nmads * mad)

# Per-sample adaptive filtering
outlier_flags = np.zeros(adata.n_obs, dtype=bool)
for sample in adata.obs["sample"].unique():
    mask = adata.obs["sample"] == sample
    sub = adata[mask]
    outlier_flags[mask] |= mad_outlier(sub, "log1p_total_counts", nmads=5, direction="both")
    outlier_flags[mask] |= mad_outlier(sub, "log1p_n_genes_by_counts", nmads=5, direction="both")
    outlier_flags[mask] |= mad_outlier(sub, "pct_counts_mt", nmads=3, direction="upper")

print(f"Flagged {outlier_flags.sum()} / {adata.n_obs} cells as outliers ({100*outlier_flags.mean():.1f}%)")
adata = adata[~outlier_flags].copy()
```

#### R (scater)

```r
library(scater)

# Automatic MAD-based outlier detection
qc_results <- perCellQCFilters(sce,
    sub.fields = list(mito = which(rowData(sce)$is_mito)),
    nmads = 3)
# Returns logical columns: low_lib_size, low_n_features, high_subsets_mito_percent
sce <- sce[, !qc_results$discard]
```

### 4.3 Ambient RNA Removal

#### SoupX

```r
library(SoupX)

# Load raw (all barcodes) and filtered (called cells) matrices
sc <- load10X("path/to/cellranger/sample_count/outs/")
# SoupX needs clusters — provide pre-computed clusters or let it estimate
sc <- autoEstCont(sc)    # Automatically estimate contamination fraction
out <- adjustCounts(sc)  # Corrected count matrix

# Use corrected counts for downstream analysis
seurat_obj <- CreateSeuratObject(counts = out)
```

**SoupX math**: Estimates contamination fraction `ρ` per cell. The corrected count for gene `j` in cell `i` is:

```
X_corrected[i,j] = X_observed[i,j] - ρ_i * s_i * p_j

where:
  ρ_i = contamination fraction for cell i
  s_i = total counts for cell i
  p_j = fraction of gene j in the ambient RNA profile
```

#### CellBender

```bash
# Install
pip install cellbender

# Run on raw (unfiltered) Cell Ranger output
cellbender remove-background \
    --input raw_feature_bc_matrix.h5 \
    --output cellbender_output.h5 \
    --expected-cells 5000 \
    --total-droplets-included 25000 \
    --fpr 0.01 \
    --epochs 150 \
    --cuda   # GPU acceleration
```

**CellBender model**: Uses a variational autoencoder (VAE) that jointly models:
- Cell-containing droplets: expression = cell signal + ambient contamination
- Empty droplets: expression = ambient contamination only
- Barcode swapping artifacts (index hopping)

```python
# Load CellBender output
adata = sc.read_h5("cellbender_output_filtered.h5")
```

### 4.4 Doublet Detection

#### Scrublet (Python)

```python
import scrublet as scr

# Run per sample (critical — never run on merged data from multiple samples)
scrub = scr.Scrublet(adata_sample.X, expected_doublet_rate=0.06)
doublet_scores, predicted_doublets = scrub.scrub_doublets(
    min_counts=2,
    min_cells=3,
    min_gene_variability_pctl=85,
    n_prin_comps=30
)

# Visualize
scrub.plot_histogram()

# Store results
adata_sample.obs["doublet_score"] = doublet_scores
adata_sample.obs["predicted_doublet"] = predicted_doublets

# Filter
adata_sample = adata_sample[~adata_sample.obs["predicted_doublet"]].copy()
```

**Scrublet math**: Simulates `N_sim` synthetic doublets by averaging random pairs of observed cells. Computes a KNN graph in PC space over observed + simulated cells. For each observed cell:

```
doublet_score_i = (k_sim_i / k) / (N_sim / (N_obs + N_sim))

where:
  k_sim_i = number of simulated doublets among cell i's k nearest neighbors
  k = total neighbors
  N_sim = number of simulated doublets
  N_obs = number of observed cells
```

#### scDblFinder (R)

```r
library(scDblFinder)

# Run on SingleCellExperiment object
sce <- scDblFinder(sce, samples = "sample")  # Run per sample

# Results in colData(sce)$scDblFinder.class ("singlet" or "doublet")
# and colData(sce)$scDblFinder.score (continuous score)
sce <- sce[, sce$scDblFinder.class == "singlet"]
```

### 4.5 Cell Cycle Scoring

#### Python (Scanpy)

```python
# Tirosh et al. 2015 cell cycle gene sets
s_genes = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
           "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
           "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
           "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
           "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
           "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
g2m_genes = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
             "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
             "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
             "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
             "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
             "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
             "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
             "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]

sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
# Adds: adata.obs['S_score'], adata.obs['G2M_score'], adata.obs['phase'] (S/G2M/G1)

# Optional regression (only if cell cycle is a confounder):
sc.pp.regress_out(adata, ["S_score", "G2M_score"])
```

#### R (Seurat)

```r
# Built-in gene lists
s.genes <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes

seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- CellCycleScoring(seurat_obj, s.features = s.genes, g2m.features = g2m.genes)

# Optional: compute difference score for regression without losing proliferation signal
seurat_obj$CC.Difference <- seurat_obj$S.Score - seurat_obj$G2M.Score
seurat_obj <- ScaleData(seurat_obj, vars.to.regress = "CC.Difference")
```

### 4.6 Complete QC Pipeline Example (Python)

```python
import scanpy as sc
import scrublet as scr
import numpy as np

# 1. Load
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.layers["counts"] = adata.X.copy()  # Preserve raw counts

# 2. Gene-level filtering
sc.pp.filter_genes(adata, min_cells=3)  # Remove genes in <3 cells

# 3. QC metrics
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True, log1p=True)

# 4. Filter cells (adaptive or fixed)
sc.pp.filter_cells(adata, min_genes=200)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

# 5. Doublet detection
scrub = scr.Scrublet(adata.X, expected_doublet_rate=0.06)
adata.obs["doublet_score"], adata.obs["predicted_doublet"] = scrub.scrub_doublets()
adata = adata[~adata.obs["predicted_doublet"]].copy()

print(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes")
```

---

## 5. Normalization & Transformation

### 5.1 Library Size (Log) Normalization

The simplest and most common approach.

**Math**:

```
For cell i, gene j:

1. Compute size factor:  s_i = total_counts_i / target_sum
   (target_sum typically = median(total_counts) or 10,000)

2. Normalize:  X_norm[i,j] = X[i,j] / s_i

3. Log transform:  X_log[i,j] = log(X_norm[i,j] + 1)
   (pseudocount of 1 avoids log(0))
```

The log transform has two effects: variance stabilization (reduces the mean-variance relationship) and making the data more symmetric/Gaussian-like.

#### Python

```python
# Standard log-normalization
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize to 10,000 counts per cell
sc.pp.log1p(adata)  # Natural log(x + 1)

# Store normalized counts
adata.layers["log_normalized"] = adata.X.copy()
```

#### R

```r
seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
```

### 5.2 Scran Pooling-Based Normalization

Addresses composition bias by pooling cells and deconvolving size factors.

**Math**: Groups cells into pools of size ~20-100, sums their counts, estimates pool-level size factors using a linear system, then deconvolves to per-cell size factors:

```
For pool P containing cells {c1, c2, ..., cn}:
  Pooled counts: Y_P[j] = Σ X[ck, j] for ck in P
  Pool size factor: Σ s_ck = f(Y_P)

Solve the system of linear equations across overlapping pools
to recover individual s_i for each cell.
```

#### R (scran)

```r
library(scran)
library(scater)

# Pre-cluster for pool construction (rough clusters suffice)
clusters <- quickCluster(sce)
sce <- computeSumFactors(sce, clusters = clusters, min.mean = 0.1)
sce <- logNormCounts(sce)

# Check size factors
summary(sizeFactors(sce))
plot(sizeFactors(sce), colSums(counts(sce)), log = "xy",
     xlab = "Size factor", ylab = "Library size")
```

### 5.3 SCTransform (Regularized Negative Binomial Regression)

Models each gene's expression as a function of sequencing depth using a negative binomial regression, then uses Pearson residuals as the normalized expression.

**Math**:

```
For each gene j, fit:

  X[i,j] ~ NB(μ_ij, θ_j)
  log(μ_ij) = β₀_j + β₁_j * log(total_counts_i)

where θ_j = inverse dispersion parameter

Then regularize: instead of gene-specific (β₀_j, β₁_j, θ_j),
fit smooth functions of mean expression:
  β₁(m) = kernel regression of β₁_j on log(mean_j)
  θ(m) = kernel regression of θ_j on log(mean_j)

Pearson residual:
  r[i,j] = (X[i,j] - μ_ij) / sqrt(μ_ij + μ_ij² / θ_j)

Residuals are clipped to [-sqrt(n), +sqrt(n)] (where n = number of cells)
```

#### R (Seurat)

```r
# SCTransform v2 (improved regularization)
seurat_obj <- SCTransform(seurat_obj, vst.flavor = "v2", verbose = TRUE,
                          vars.to.regress = "percent.mt")  # Optional regression
# This replaces NormalizeData + FindVariableFeatures + ScaleData
```

#### Python

```python
# Analytic Pearson residuals (fast approximation of SCTransform)
import scanpy.experimental as sce_exp
sce_exp.pp.normalize_pearson_residuals(adata)
```

### 5.4 Scaling (Z-Score per Gene)

Performed **after** normalization and **before** PCA. Ensures each gene contributes equally to the principal components.

**Math**:

```
For each gene j across all cells:
  X_scaled[i,j] = (X_log[i,j] - mean_j) / sd_j

Optionally clip extreme values (e.g., max_value=10) to reduce outlier influence.
```

```python
sc.pp.scale(adata, max_value=10)
```

```r
seurat_obj <- ScaleData(seurat_obj, features = VariableFeatures(seurat_obj),
                         vars.to.regress = c("percent.mt", "nCount_RNA"))
```

**Important**: Scaling is for PCA input only. Never use scaled values for DE testing, gene-expression visualization, or biological interpretation.

---

## 6. Feature Selection & Dimensionality Reduction

### 6.1 Highly Variable Gene (HVG) Selection

**Math (Seurat v3/vst method)**:

```
For each gene j with mean expression μ_j and variance σ²_j:

1. Fit a LOESS (local polynomial) regression: log(σ²) = f(log(μ))
2. Expected variance: σ²_expected_j = exp(f(log(μ_j)))
3. Standardized variance: SV_j = σ²_j / σ²_expected_j
4. Select top N genes by SV (typically N = 2000)
```

```python
# Scanpy — operate on raw counts for flavor='seurat_v3'
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor="seurat_v3",    # Requires raw counts in adata.X
    batch_key="sample",    # Compute HVGs per batch, then combine
    subset=False           # Don't subset; just mark adata.var['highly_variable']
)

# Inspect
print(f"Selected {adata.var['highly_variable'].sum()} HVGs")
sc.pl.highly_variable_genes(adata)

# For downstream analysis, subset to HVGs:
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
```

```r
# Seurat
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)

# Visualize
top10 <- head(VariableFeatures(seurat_obj), 10)
VariableFeaturePlot(seurat_obj) %>% LabelPoints(points = top10)
```

### 6.2 PCA

PCA finds the directions of maximum variance in the scaled gene expression matrix.

**Math**:

```
Given scaled matrix Z (n_cells × n_genes):

1. Compute covariance matrix: C = (1/n) Z^T Z  (n_genes × n_genes)
   (In practice, use SVD: Z = U Σ V^T; PCs = U Σ; loadings = V)

2. PC scores for cell i: pc_i = Z[i, :] · V[:, 1:k]
   where V[:, 1:k] are the top k eigenvectors (loadings)

3. Variance explained by PC d: λ_d / Σ λ_d
   where λ_d is the d-th eigenvalue
```

```python
# Scanpy PCA
sc.tl.pca(adata, n_comps=50, svd_solver="arpack")

# Result stored in:
#   adata.obsm["X_pca"]    — cell × PC matrix (n_cells × 50)
#   adata.varm["PCs"]      — gene × PC loadings (n_genes × 50)
#   adata.uns["pca"]["variance_ratio"]  — fraction of variance explained per PC

# Elbow plot
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)

# Inspect top loadings for a PC
import pandas as pd
loadings = pd.DataFrame(
    adata.varm["PCs"][:, :5],
    index=adata.var_names,
    columns=[f"PC{i+1}" for i in range(5)]
)
for pc in loadings.columns:
    top = loadings[pc].abs().nlargest(10)
    print(f"\n{pc} top loadings:")
    print(top)
```

```r
# Seurat
seurat_obj <- ScaleData(seurat_obj, features = VariableFeatures(seurat_obj))
seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(seurat_obj), npcs = 50)
ElbowPlot(seurat_obj, ndims = 50)

# JackStraw test for PC significance
seurat_obj <- JackStraw(seurat_obj, num.replicate = 100)
seurat_obj <- ScoreJackStraw(seurat_obj, dims = 1:50)
JackStrawPlot(seurat_obj, dims = 1:50)
```

### 6.3 Neighbor Graph Construction

**Math (KNN graph)**:

```
For each cell i in PC space:
1. Find the k nearest neighbors by Euclidean distance: N_k(i)
2. Build adjacency matrix A: A[i,j] = 1 if j ∈ N_k(i)
3. Optionally make symmetric: A_sym = A + A^T (mutual KNN)

Shared Nearest Neighbor (SNN) graph:
  SNN[i,j] = |N_k(i) ∩ N_k(j)| / |N_k(i) ∪ N_k(j)|
  (Jaccard overlap of neighbor sets)
```

```python
sc.pp.neighbors(
    adata,
    n_neighbors=15,     # k (number of neighbors)
    n_pcs=30,           # Use top 30 PCs
    method="umap",      # Use UMAP's fast approximate NN algorithm
    metric="euclidean"
)
# Stores: adata.obsp["distances"], adata.obsp["connectivities"] (sparse matrices)
```

```r
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30, k.param = 20)
```

### 6.4 UMAP

**Math (high-level)**:

```
1. High-dimensional fuzzy graph:
   w_high(i,j) = exp(-(d(i,j) - ρ_i) / σ_i)
   where ρ_i = distance to nearest neighbor, σ_i chosen so that
   Σ_j w_high(i,j) = log₂(n_neighbors)

2. Symmetrize: w_sym(i,j) = w_high(i,j) + w_high(j,i) - w_high(i,j) * w_high(j,i)

3. Low-dimensional embedding: optimize positions y_1, ..., y_n to minimize:
   CE = Σ_{edges} [w_sym * log(w_sym / w_low) + (1 - w_sym) * log((1-w_sym)/(1-w_low))]

   where w_low(i,j) = 1 / (1 + a * ||y_i - y_j||^(2b))
   (a, b determined by min_dist parameter)
```

```python
sc.tl.umap(adata, min_dist=0.3, spread=1.0, n_components=2)
# Result: adata.obsm["X_umap"]
sc.pl.umap(adata, color=["leiden", "n_genes_by_counts", "pct_counts_mt"], ncols=3)

# t-SNE alternative
sc.tl.tsne(adata, n_pcs=30, perplexity=30)
```

```r
seurat_obj <- RunUMAP(seurat_obj, dims = 1:30, min.dist = 0.3, n.neighbors = 30)
DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters")
```

### 6.5 Diffusion Maps

**Math**:

```
1. Build kernel matrix: K(i,j) = exp(-d(i,j)² / (2σ²))

2. Normalize to transition matrix: T(i,j) = K(i,j) / Σ_k K(i,k)

3. Eigen-decompose T: T v_k = λ_k v_k

4. Diffusion components = eigenvectors v_1, v_2, ..., v_d
   (excluding v_0 = constant vector with λ_0 = 1)

5. Diffusion pseudotime from root cell c:
   DPT(i) = ||ψ(i) - ψ(c)||
   where ψ(i) = (λ_1^t v_1(i), λ_2^t v_2(i), ...)
```

```python
sc.tl.diffmap(adata, n_comps=15)
# Result: adata.obsm["X_diffmap"]

# Diffusion pseudotime
adata.uns["iroot"] = np.flatnonzero(adata.obs["cell_type"] == "HSC")[0]
sc.tl.dpt(adata)
# Result: adata.obs["dpt_pseudotime"]

sc.pl.scatter(adata, basis="diffmap", color=["dpt_pseudotime", "cell_type"], components="1,2")
```

---

## 7. Batch Effects & Data Integration

### 7.1 Harmony

**Math**: Iterates between soft-clustering in PC space and correcting cluster centroids.

```
1. Initialize: Z = PCA embedding (n_cells × d_PCs)
2. Soft-cluster: assign cell i to cluster k with probability:
   P(k|i) ∝ exp(-||Z_i - c_k||² / (2σ²)) * (1 - diversity_penalty(k, batch_i))
3. Correct: for each cluster k, compute batch-specific centroids and shift
   cells toward the global centroid.
4. Repeat until convergence.
```

```python
import scanpy.external as sce

sce.pp.harmony_integrate(adata, key="sample", basis="X_pca", adjusted_basis="X_pca_harmony")

# Use corrected embedding for downstream
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=15)
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

```r
library(harmony)

seurat_obj <- RunHarmony(seurat_obj, group.by.vars = "sample", reduction = "pca",
                          dims.use = 1:30, assay.use = "RNA")
seurat_obj <- RunUMAP(seurat_obj, reduction = "harmony", dims = 1:30)
seurat_obj <- FindNeighbors(seurat_obj, reduction = "harmony", dims = 1:30)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.8)
```

### 7.2 scVI (Variational Autoencoder Integration)

**Math (generative model)**:

```
For cell i from batch b_i:

1. Latent variable:    z_i ~ Normal(0, I)     (d-dimensional, typically d=10)
2. Library size:       l_i ~ LogNormal(μ_l, σ²_l)  (per-batch parameters)

3. Decoder:
   ρ_j = softmax(f_θ(z_i, b_i))_j          (gene proportions via neural network f_θ)
   μ_ij = l_i * ρ_j                          (expected count)
   θ_j = learned dispersion per gene

4. Observation model:
   X[i,j] ~ NB(μ_ij, θ_j)                   or   ZINB(μ_ij, θ_j, π_ij)

5. Inference (encoder):
   q(z_i | X_i, b_i) = Normal(μ_z(X_i, b_i), σ_z(X_i, b_i))

6. Loss = ELBO = E_q[log p(X|z,l,b)] - KL[q(z|X,b) || p(z)]
```

```python
import scvi

# Prepare AnnData (raw counts required)
adata.layers["counts"] = adata.X.copy()
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="sample",
    categorical_covariate_keys=["sample"],
    continuous_covariate_keys=["pct_counts_mt"]
)

# Train model
model = scvi.model.SCVI(
    adata,
    n_latent=10,
    n_layers=2,
    n_hidden=128,
    gene_likelihood="nb",
    dispersion="gene"
)
model.train(max_epochs=400, early_stopping=True)

# Get latent representation
adata.obsm["X_scVI"] = model.get_latent_representation()

# Use for downstream
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Get denoised expression
adata.layers["scvi_normalized"] = model.get_normalized_expression(library_size=1e4)

# Bayesian differential expression
de_results = model.differential_expression(
    groupby="cell_type",
    group1="CD4_T",
    group2="CD8_T"
)
```

### 7.3 Seurat v5 Integration (CCA + Anchors)

```r
library(Seurat)

seurat_list <- SplitObject(seurat_obj, split.by = "sample")
seurat_list <- lapply(seurat_list, function(x) {
    x <- NormalizeData(x)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
    return(x)
})

features <- SelectIntegrationFeatures(object.list = seurat_list, nfeatures = 2000)
anchors <- FindIntegrationAnchors(object.list = seurat_list, anchor.features = features,
                                   dims = 1:30, reduction = "cca")
seurat_integrated <- IntegrateData(anchorset = anchors, dims = 1:30)

DefaultAssay(seurat_integrated) <- "integrated"
seurat_integrated <- ScaleData(seurat_integrated)
seurat_integrated <- RunPCA(seurat_integrated, npcs = 30)
seurat_integrated <- RunUMAP(seurat_integrated, dims = 1:30)
seurat_integrated <- FindNeighbors(seurat_integrated, dims = 1:30)
seurat_integrated <- FindClusters(seurat_integrated, resolution = 0.8)
```

### 7.4 BBKNN

```python
import bbknn
bbknn.bbknn(adata, batch_key="sample", n_pcs=30, neighbors_within_batch=3)
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

### 7.5 Integration Quality Metrics

```python
from sklearn.metrics import silhouette_score

sil_batch = silhouette_score(adata.obsm["X_scVI"], adata.obs["sample"])
sil_celltype = silhouette_score(adata.obsm["X_scVI"], adata.obs["cell_type"])
print(f"Batch silhouette: {sil_batch:.3f} (want near 0)")
print(f"Cell type silhouette: {sil_celltype:.3f} (want near 1)")

# scib benchmarking suite
# import scib
# scib.metrics.ilisi_knn(adata, batch_key="sample", label_key="cell_type",
#                         type_="embed", use_rep="X_scVI")
```

---

## 8. Clustering & Cell Type Annotation

### 8.1 Leiden Clustering

**Math**: Optimizes the Constant Potts Model (CPM) or modularity on the SNN graph.

```
Modularity Q = (1/2m) Σ_ij [A_ij - γ(k_i * k_j)/(2m)] δ(c_i, c_j)

where:
  A_ij = edge weight between i and j
  k_i = degree of node i
  m = total edge weight
  c_i = community assignment of cell i
  δ = 1 if c_i = c_j, 0 otherwise
  γ = resolution parameter (higher → more, smaller clusters)
```

```python
for res in [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]:
    sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")

sc.pl.umap(adata, color=[f"leiden_{r}" for r in [0.2, 0.5, 1.0, 2.0]], ncols=2)
sc.tl.leiden(adata, resolution=0.8, key_added="leiden")
```

```r
seurat_obj <- FindClusters(seurat_obj, resolution = c(0.2, 0.5, 0.8, 1.0, 1.5), algorithm = 4)

library(clustree)
clustree(seurat_obj, prefix = "RNA_snn_res.")
```

### 8.2 Marker Gene Detection

```python
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", pts=True)
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, standard_scale="var")

# Access results as DataFrame
marker_df = sc.get.rank_genes_groups_df(adata, group=None)

# Visualization with known markers
sc.pl.dotplot(adata, var_names={
    "T cells": ["CD3D", "CD3E", "IL7R", "CD8A"],
    "B cells": ["CD79A", "MS4A1", "CD19"],
    "Monocytes": ["CD14", "LYZ", "S100A8"],
    "NK cells": ["NKG7", "GNLY", "KLRD1"],
    "Fibroblasts": ["COL1A1", "DCN", "LUM"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18"],
}, groupby="leiden", standard_scale="var")
```

```r
all_markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
top5 <- all_markers %>% group_by(cluster) %>% slice_max(n = 5, order_by = avg_log2FC)
DotPlot(seurat_obj, features = unique(top5$gene)) + RotatedAxis()
```

### 8.3 Automated Annotation

#### CellTypist (Python)

```python
import celltypist
from celltypist import models

models.download_models(force_update=True)
model = models.Model.load(model="Immune_All_Low.pkl")
predictions = celltypist.annotate(adata, model=model, majority_voting=True)
adata.obs["celltypist_label"] = predictions.predicted_labels["majority_voting"]
sc.pl.umap(adata, color="celltypist_label")
```

#### SingleR (R)

```r
library(SingleR)
library(celldex)

ref <- celldex::HumanPrimaryCellAtlasData()
results <- SingleR(test = as.SingleCellExperiment(seurat_obj),
                    ref = ref, labels = ref$label.main)
seurat_obj$SingleR_label <- results$labels
plotScoreHeatmap(results)
```

#### Azimuth (R)

```r
library(Azimuth)
seurat_obj <- RunAzimuth(seurat_obj, reference = "pbmcref")
DimPlot(seurat_obj, group.by = "predicted.celltype.l2")
```

### 8.4 Sub-clustering

```python
t_cells = adata[adata.obs["cell_type"] == "T_cell"].copy()
sc.pp.highly_variable_genes(t_cells, n_top_genes=1500, flavor="seurat_v3", layer="counts")
sc.tl.pca(t_cells)
sc.pp.neighbors(t_cells, n_pcs=20)
sc.tl.leiden(t_cells, resolution=0.5, key_added="t_cell_subtype")
sc.tl.rank_genes_groups(t_cells, groupby="t_cell_subtype", method="wilcoxon")
sc.pl.rank_genes_groups_dotplot(t_cells, n_genes=5)
```

---

## 9. Differential Expression Analysis

### 9.1 The Pseudobulk Paradigm (Critical)

**The single most important methodological point**: Cells from the same donor are NOT independent. Using cell-level tests inflates significance.

**Math (pseudobulk)**:

```
For cell type T, sample s, gene j:

  Y[s,j] = Σ_{i ∈ cells(T,s)} X[i,j]    (sum raw counts)

Now Y is a (n_samples × n_genes) matrix — standard bulk RNA-seq.
Apply standard DE models (DESeq2, edgeR, limma) with samples as replicates.
```

#### Python Pseudobulk + DESeq2 Pipeline

```python
import decoupler as dc
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# 1. Create pseudobulk
pseudobulk = dc.get_pseudobulk(
    adata,
    sample_col="sample",
    groups_col="cell_type",
    layer="counts",
    min_cells=10,
    min_counts=1000
)

# 2. Run DESeq2 on a specific cell type
ct = "Macrophage"
pb_ct = pseudobulk[pseudobulk.obs["cell_type"] == ct].copy()

dds = DeseqDataSet(
    counts=pd.DataFrame(
        pb_ct.X.toarray() if hasattr(pb_ct.X, 'toarray') else pb_ct.X,
        columns=pb_ct.var_names,
        index=pb_ct.obs_names
    ),
    metadata=pb_ct.obs[["condition"]],
    design_factors="condition"
)
dds.deseq2()
stat_res = DeseqStats(dds, contrast=["condition", "treated", "control"])
stat_res.summary()
results_df = stat_res.results_df

# 3. Volcano plot
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
sig = (results_df["padj"] < 0.05) & (results_df["log2FoldChange"].abs() > 1)
ax.scatter(results_df.loc[~sig, "log2FoldChange"], -np.log10(results_df.loc[~sig, "padj"]),
           s=3, color="grey", alpha=0.5)
ax.scatter(results_df.loc[sig, "log2FoldChange"], -np.log10(results_df.loc[sig, "padj"]),
           s=5, color="red", alpha=0.7)
ax.set_xlabel("log2 Fold Change"); ax.set_ylabel("-log10 adjusted p-value")
ax.set_title(f"DE: {ct} treated vs control")
ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.5)
ax.axvline(-1, color="black", linestyle="--", linewidth=0.5)
ax.axvline(1, color="black", linestyle="--", linewidth=0.5)
plt.tight_layout()
```

#### R Pseudobulk Pipeline (muscat)

```r
library(muscat)
library(DESeq2)

pb <- aggregateData(sce, assay = "counts", by = c("cell_type", "sample_id"), fun = "sum")
res <- pbDS(pb, method = "DESeq2",
            design = model.matrix(~ condition, data = colData(pb)),
            coef = "conditiontreated")
```

### 9.2 MAST (Cell-Level DE with Covariates)

**Math**: Two-part hurdle model:

```
Part 1 (Discrete): P(X[i,j] > 0) modeled by logistic regression
  logit(P(X[i,j] > 0)) = α_0j + α_1j * condition_i + α_2j * CDR_i

Part 2 (Continuous): E[X[i,j] | X[i,j] > 0] modeled by Gaussian regression
  E[log(X[i,j] + 1) | X[i,j] > 0] = β_0j + β_1j * condition_i + β_2j * CDR_i

where CDR = cellular detection rate (fraction of genes detected in cell i)

Joint p-value via likelihood ratio test combining both parts.
```

```r
library(MAST)
sca <- SceToSingleCellAssay(as.SingleCellExperiment(seurat_obj))
colData(sca)$CDR <- scale(colSums(assay(sca) > 0))
zlm_fit <- zlm(~ condition + CDR + (1 | sample), sca, method = "glmer",
                ebayes = FALSE, strictConvergence = FALSE)
summary_dt <- summary(zlm_fit, doLRT = "conditiontreated")
```

---

## 10. Trajectory & RNA Velocity Analysis

### 10.1 PAGA (Partition-based Graph Abstraction)

```python
sc.tl.paga(adata, groups="leiden")
sc.pl.paga(adata, threshold=0.03, node_size_scale=1.5, edge_width_scale=0.5)
sc.tl.umap(adata, init_pos="paga")  # PAGA-initialized UMAP
```

### 10.2 RNA Velocity (scVelo)

**Math (steady-state model)**:

```
At steady state for gene j:
  ds/dt = β * u - γ * s = 0  →  s = (β/γ) * u

Velocity for cell i, gene j:
  v[i,j] = β_j * u[i,j] - γ_j * s[i,j]

  v > 0 → gene being upregulated
  v < 0 → gene being downregulated
```

**Math (dynamical model)**:

```
Full ODE system per gene j:
  du/dt = α(t) - β * u       (transcription - splicing)
  ds/dt = β * u - γ * s      (splicing - degradation)

Parameters (α, β, γ, latent time τ, switching times) fit per gene via EM.
```

#### Complete scVelo Pipeline

```python
import scvelo as scv

# 1. Load velocity data (loom from velocyto or STARsolo/alevin-fry layers)
ldata = scv.read("velocyto_output.loom", cache=True)
adata = scv.utils.merge(adata, ldata)

# 2. Preprocessing
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

# 3a. Steady-state velocity (fast)
scv.tl.velocity(adata, mode="stochastic")

# 3b. Dynamical model (recommended)
scv.tl.recover_dynamics(adata, n_jobs=8)
scv.tl.velocity(adata, mode="dynamical")

# 4. Velocity graph
scv.tl.velocity_graph(adata)

# 5. Visualization
scv.pl.velocity_embedding_stream(adata, basis="umap", color="cell_type")

# 6. Latent time (dynamical mode only)
scv.tl.latent_time(adata)
scv.pl.scatter(adata, color="latent_time", cmap="viridis", basis="umap")

# 7. Phase portraits
scv.pl.velocity(adata, var_names=["TOP2A", "FOXP3", "CD8A"], basis="umap")

# 8. Velocity confidence
scv.tl.velocity_confidence(adata)
scv.pl.scatter(adata, color=["velocity_confidence", "velocity_length"], basis="umap")
```

### 10.3 CellRank (Fate Probabilities)

**Math**:

```
1. Transition matrix from velocity:
   T_vel[i,j] = softmax(cosine_similarity(v_i, x_j - x_i)) for j ∈ neighbors(i)

2. Combined transition matrix: T = w * T_vel + (1-w) * T_knn

3. Identify macrostates via GPCCA / Schur decomposition.

4. Absorption probabilities: for each cell, probability of reaching each
   terminal state (solved from absorbing Markov chain).
```

```python
import cellrank as cr

vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
combined_kernel = 0.8 * vk + 0.2 * ck

estimator = cr.estimators.GPCCA(combined_kernel)
estimator.compute_schur(n_components=20)
estimator.compute_macrostates(n_states=5, cluster_key="cell_type")
estimator.set_terminal_states()
estimator.compute_fate_probabilities()
estimator.plot_fate_probabilities(basis="umap")

drivers = estimator.compute_lineage_drivers(lineages=["Macrophage"])
drivers.head(20)
```

### 10.4 Monocle 3 (R)

```r
library(monocle3)
cds <- as.cell_data_set(seurat_obj)
cds <- preprocess_cds(cds, num_dim = 30)
cds <- reduce_dimension(cds, reduction_method = "UMAP")
cds <- cluster_cells(cds)
cds <- learn_graph(cds)
cds <- order_cells(cds, root_cells = root_cell_ids)
plot_cells(cds, color_cells_by = "pseudotime", label_cell_groups = FALSE)

# Genes changing over pseudotime
pr_test_res <- graph_test(cds, neighbor_graph = "principal_graph", cores = 4)
```

### 10.5 Slingshot (R)

```r
library(slingshot)
sce <- slingshot(sce, clusterLabels = "cell_type", reducedDim = "PCA", start.clus = "HSC")
pseudotime <- slingPseudotime(sce)
```

---

## 11. Gene Regulatory Network Inference

### 11.1 pySCENIC

```python
from arboreto.algo import grnboost2
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell

# Step 1: GRN inference with GRNBoost2
expr_matrix = adata.to_df()
tf_names = [line.strip() for line in open("allTFs_hg38.txt")]
adjacencies = grnboost2(expression_data=expr_matrix, tf_names=tf_names, verbose=True)

# Step 2: cisTarget pruning
dbs = [RankingDatabase(fname=db_path, name=db_name) for db_path, db_name in db_files]
modules = modules_from_adjacencies(adjacencies, expr_matrix)
df = prune2df(dbs, modules, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")
regulons = df2regulons(df)

# Step 3: AUCell scoring
auc_matrix = aucell(expr_matrix, regulons, num_workers=4)
adata.obsm["AUCell"] = auc_matrix.values
```

**AUCell math**:

```
For regulon R with genes G_R, cell i:
1. Rank all genes in cell i by expression (descending)
2. Recovery curve: fraction of G_R found as we descend the ranking
3. AUC = area under recovery curve up to top 5% threshold
```

#### CLI Pipeline

```bash
pyscenic grn expr_matrix.loom allTFs_hg38.txt -o adjacencies.tsv --num_workers 16
pyscenic ctx adjacencies.tsv rankings.feather \
    --annotations_fname motifs.tbl \
    --expression_mtx_fname expr_matrix.loom \
    --output regulons.csv --num_workers 16
pyscenic aucell expr_matrix.loom regulons.csv --output aucell.loom --num_workers 16
```

### 11.2 Decoupler (TF/Pathway Activity)

```python
import decoupler as dc

dc.run_mlm(
    mat=adata,
    net=dc.get_dorothea(organism="human", levels=["A", "B", "C"]),
    source="source", target="target", weight="weight",
    verbose=True, use_raw=False
)
# Results: adata.obsm["mlm_estimate"], adata.obsm["mlm_pvals"]

# Pathway activity (PROGENy)
dc.run_mlm(mat=adata, net=dc.get_progeny(organism="human", top=500),
            source="source", target="target", weight="weight")

acts = dc.get_acts(adata, obsm_key="mlm_estimate")
sc.pl.matrixplot(acts, var_names=acts.var_names[:20], groupby="cell_type",
                  standard_scale="var", cmap="RdBu_r")
```

---

## 12. Cell-Cell Communication

### 12.1 LIANA (Meta-Framework)

```python
import liana as li

li.mt.rank_aggregate(
    adata, groupby="cell_type", resource_name="consensus",
    expr_prop=0.1, verbose=True, use_raw=False
)

li.pl.dotplot(
    adata=adata, colour="magnitude_rank", size="specificity_rank",
    inverse_colour=True, inverse_size=True,
    source_labels=["Macrophage", "Fibroblast"],
    target_labels=["T_cell", "Endothelial"],
    top_n=20, figure_size=(12, 8)
)
```

### 12.2 CellChat (R)

```r
library(CellChat)
cellchat <- createCellChat(object = seurat_obj, group.by = "cell_type")
cellchat@DB <- CellChatDB.human
cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)
cellchat <- computeCommunProb(cellchat, type = "triMean")
cellchat <- filterCommunication(cellchat, min.cells = 10)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)
netVisual_circle(cellchat@net$count, vertex.weight = table(cellchat@idents))
```

### 12.3 NicheNet (R)

```r
library(nichenetr)
ligand_target_matrix <- readRDS("ligand_target_matrix.rds")
lr_network <- readRDS("lr_network.rds")
ligand_activities <- predict_ligand_activities(
    geneset = geneset_oi,
    background_expressed_genes = expressed_genes_receiver,
    ligand_target_matrix = ligand_target_matrix,
    potential_ligands = potential_ligands
)
top_ligands <- ligand_activities %>% top_n(20, auroc) %>% pull(test_ligand)
```

---

## 13. Spatial Transcriptomics — Platform-Specific Processing

### 13.1 10x Visium

```bash
spaceranger count \
    --id=visium_sample \
    --transcriptome=/path/to/refdata-gex-GRCh38-2024-A \
    --fastqs=/path/to/fastqs \
    --sample=VisiumSample \
    --image=/path/to/tissue_image.tif \
    --slide=V19L29-096 \
    --area=B1 \
    --localcores=16 --localmem=64
```

```python
import scanpy as sc
import squidpy as sq

adata = sc.read_visium("path/to/spaceranger/outs/")
adata.var_names_make_unique()

# adata.obsm["spatial"]  → (x, y) coordinates per spot
# adata.uns["spatial"]   → image data, scale factors

sc.pl.spatial(adata, color="total_counts", img_key="hires", spot_size=1.0)

# Standard preprocessing
sc.pp.filter_genes(adata, min_cells=10)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", layer="counts")
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8)

sc.pl.spatial(adata, color=["leiden", "CD3D", "CD68", "COL1A1"], ncols=2, spot_size=1.3)

# Build spatial neighbor graph
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
```

```r
seurat_obj <- Load10X_Spatial(data.dir = "path/to/spaceranger/outs/",
                               filename = "filtered_feature_bc_matrix.h5")
SpatialFeaturePlot(seurat_obj, features = c("nCount_Spatial", "CD3D"))
seurat_obj <- SCTransform(seurat_obj, assay = "Spatial", verbose = TRUE)
```

### 13.2 10x Xenium

```python
import squidpy as sq
import pandas as pd

adata = sc.read_10x_h5("path/to/xenium/cell_feature_matrix.h5")
cell_info = pd.read_csv("path/to/xenium/cells.csv.gz")
adata.obsm["spatial"] = cell_info[["x_centroid", "y_centroid"]].values

# Or via SpatialData
from spatialdata_io import xenium as xenium_io
sdata = xenium_io("path/to/xenium_output/")

sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=10)
sq.gr.nhood_enrichment(adata, cluster_key="leiden")
sq.pl.nhood_enrichment(adata, cluster_key="leiden")
```

### 13.3 MERFISH / MERSCOPE

```python
from spatialdata_io import merscope as merscope_io
sdata = merscope_io("path/to/merscope_output/", vpt_outputs="path/to/cellpose_segmentation/")
adata = sdata.tables["table"]
transcripts = sdata.points["transcripts"]  # x, y, z, gene per transcript
```

### 13.4 Segmentation with Cellpose

```python
from cellpose import models
import numpy as np
from skimage.io import imread

dapi = imread("dapi_image.tif")
model = models.Cellpose(model_type="nuclei", gpu=True)
masks, flows, styles, diams = model.eval(
    dapi, diameter=30, channels=[0, 0],
    flow_threshold=0.4, cellprob_threshold=0.0
)
# masks: integer-labeled segmentation (0 = background)

# For membrane staining (cytoplasm model):
model_cyto = models.Cellpose(model_type="cyto2", gpu=True)
masks_cyto, _, _, _ = model_cyto.eval(
    np.stack([membrane_image, dapi], axis=-1),
    diameter=40, channels=[1, 2]
)
```

---

## 14. Spatial Analysis Methods

### 14.1 Moran's I (Spatial Autocorrelation)

**Math**:

```
I_j = (N / W) * (Σ_i Σ_k w_ik (x_ij - x̄_j)(x_kj - x̄_j)) / (Σ_i (x_ij - x̄_j)²)

where:
  N = number of spatial units
  w_ik = spatial weight (1 if neighbors, 0 otherwise)
  W = Σ w_ik (total weight)

I ≈ +1 → clustered;  I ≈ 0 → random;  I ≈ -1 → dispersed
```

```python
sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=4)
svg_results = adata.uns["moranI"].sort_values("I", ascending=False)
print(svg_results.head(20))
```

### 14.2 SpatialDE

**Math**: Gaussian process regression testing spatial covariance.

```
H₀: y ~ N(μ, σ²_noise * I)
H₁: y ~ N(μ, σ²_spatial * K + σ²_noise * I)

K_ik = exp(-||s_i - s_k||² / (2l²))  (squared exponential kernel)
```

```python
import SpatialDE

counts = adata.to_df()
coords = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"], index=adata.obs_names)
norm_expr, sample_info = SpatialDE.stabilize(counts.T)
sample_info["x"] = coords["x"].values
sample_info["y"] = coords["y"].values
results = SpatialDE.run(sample_info, norm_expr).sort_values("qval")
```

### 14.3 Spatial Deconvolution — Cell2location

**Math**:

```
X[s,g] ~ NB(μ_sg, α_g)
μ_sg = Σ_f (w_sf * g_fg) * m_g

w_sf = abundance of cell type f at spot s (estimated)
g_fg = reference expression of gene g in cell type f
```

```python
import cell2location

# Train reference model
cell2location.models.RegressionModel.setup_anndata(adata_ref, batch_key="sample", labels_key="cell_type")
ref_model = cell2location.models.RegressionModel(adata_ref)
ref_model.train(max_epochs=250)
adata_ref = ref_model.export_posterior(adata_ref, sample_kwargs={"num_samples": 1000})
inf_aver = adata_ref.varm["means_per_cluster_mu_fg"]

# Train spatial model
cell2location.models.Cell2location.setup_anndata(adata_spatial, batch_key="sample")
mod = cell2location.models.Cell2location(
    adata_spatial, cell_state_df=inf_aver,
    N_cells_per_location=30, detection_alpha=20
)
mod.train(max_epochs=30000, batch_size=None, train_size=1)
adata_spatial = mod.export_posterior(adata_spatial, sample_kwargs={"num_samples": 1000})
# Results: adata_spatial.obsm["q05_cell_abundance_w_sf"]
```

#### RCTD (R)

```r
library(spacexr)
reference <- Reference(counts = GetAssayData(seurat_ref, slot = "counts"),
                        cell_types = seurat_ref$cell_type, nUMI = seurat_ref$nCount_RNA)
spatial_rna <- SpatialRNA(coords = GetTissueCoordinates(seurat_spatial),
                           counts = GetAssayData(seurat_spatial, slot = "counts"),
                           nUMI = seurat_spatial$nCount_Spatial)
myRCTD <- create.RCTD(spatial_rna, reference, max_cores = 8)
myRCTD <- run_RCTD(myRCTD, doublet_mode = "full")
```

#### Tangram

```python
import tangram as tg
markers = tg.gen_marker_sc(adata_sc, n_markers=100)
tg.pp_adatas(adata_sc, adata_spatial, genes=markers)
ad_map = tg.map_cells_to_space(
    adata_sc, adata_spatial, mode="clusters", cluster_label="cell_type",
    density_prior="rna_count_based", num_epochs=500, device="cuda:0"
)
tg.project_cell_annotations(ad_map, adata_spatial, annotation="cell_type")
ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)  # Gene imputation
```

### 14.4 Spatial Domains — BayesSpace (R)

**Math**: Bayesian clustering with spatial Potts prior:

```
P(z_i = k | neighbors, X) ∝ P(X_i | z_i = k) * exp(γ * Σ_{j ∈ neighbors(i)} 1[z_j = k])
```

```r
library(BayesSpace)
sce <- spatialPreprocess(sce, platform = "Visium", n.PCs = 15, n.HVGs = 2000)
sce <- qTune(sce, qs = seq(2, 10))
sce <- spatialCluster(sce, q = 7, platform = "Visium", nrep = 10000, gamma = 3)
clusterPlot(sce)
```

### 14.5 Spatial Neighborhood Analysis (Squidpy)

```python
import squidpy as sq

sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)

# Neighborhood enrichment
sq.gr.nhood_enrichment(adata, cluster_key="cell_type")
sq.pl.nhood_enrichment(adata, cluster_key="cell_type", figsize=(8, 8))

# Co-occurrence across distances
sq.gr.co_occurrence(adata, cluster_key="cell_type", spatial_key="spatial")
sq.pl.co_occurrence(adata, cluster_key="cell_type")

# Ripley's L function
sq.gr.ripley(adata, cluster_key="cell_type", mode="L")
sq.pl.ripley(adata, cluster_key="cell_type", mode="L")

# Spatial L-R interaction analysis
sq.gr.ligrec(adata, n_perms=1000, cluster_key="cell_type", use_raw=False,
              transmitter_params={"categories": "ligand"},
              receiver_params={"categories": "receptor"})
sq.pl.ligrec(adata, cluster_key="cell_type",
              source_groups=["Macrophage"], target_groups=["T_cell"])
```

**Ripley's L function math**:

```
K(r) = (A / n²) * Σ_i Σ_{j≠i} 1[d(i,j) ≤ r] / w_ij
L(r) = sqrt(K(r) / π) - r

L(r) > 0 → clustering;  L(r) < 0 → dispersion;  L(r) ≈ 0 → random
```

---

## 15. Multi-Modal & Multi-Omics Integration

### 15.1 CITE-seq (RNA + Protein)

```python
import muon as mu

mdata = mu.read_10x_h5("filtered_feature_bc_matrix.h5")

# CLR normalization for ADTs
import numpy as np
prot = mdata.mod["prot"]
X = prot.X.toarray() if hasattr(prot.X, 'toarray') else prot.X
geom_mean = np.exp(np.mean(np.log(X + 1), axis=1, keepdims=True))
prot.X = np.log(X / geom_mean + 1)

# Joint analysis with totalVI
import scvi
scvi.model.TOTALVI.setup_mudata(mdata, rna_layer=None, protein_layer=None, batch_key="sample")
model = scvi.model.TOTALVI(mdata)
model.train(max_epochs=300)
mdata.obsm["X_totalVI"] = model.get_latent_representation()
```

```r
# Seurat WNN (Weighted Nearest Neighbor)
seurat_obj <- NormalizeData(seurat_obj, assay = "ADT", normalization.method = "CLR", margin = 2)
seurat_obj <- FindMultiModalNeighbors(seurat_obj,
    reduction.list = list("pca", "apca"), dims.list = list(1:30, 1:18))
seurat_obj <- RunUMAP(seurat_obj, nn.name = "weighted.nn", reduction.name = "wnn.umap")
```

### 15.2 Multiome (RNA + ATAC)

```r
library(Signac)
library(Seurat)

# ATAC processing
DefaultAssay(seurat_obj) <- "ATAC"
seurat_obj <- NucleosomeSignal(seurat_obj)
seurat_obj <- TSSEnrichment(seurat_obj, fast = FALSE)
seurat_obj <- FindTopFeatures(seurat_obj, min.cutoff = "q0")
seurat_obj <- RunTFIDF(seurat_obj)
seurat_obj <- RunSVD(seurat_obj)

# Link peaks to genes
seurat_obj <- LinkPeaks(seurat_obj, peak.assay = "ATAC", expression.assay = "RNA")

# WNN integration
seurat_obj <- FindMultiModalNeighbors(seurat_obj,
    reduction.list = list("pca", "lsi"),
    dims.list = list(1:30, 2:30))  # Skip LSI component 1 (depth-correlated)
```

---

## 16. Perturbation & CRISPR Screen Analysis

### 16.1 Processing Perturb-seq Data

```python
import pertpy as pt
import scanpy as sc

adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5", gex_only=False)
gex = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
guides = adata[:, adata.var["feature_types"] == "CRISPR Guide Capture"].copy()

# Assign guides to cells
guide_counts = guides.X.toarray()
max_guide_idx = guide_counts.argmax(axis=1)
gex.obs["guide"] = guides.var_names[max_guide_idx]
gex.obs["guide_count"] = guide_counts.max(axis=1)
gex = gex[gex.obs["guide_count"] >= 5].copy()

# Map guide → target gene
guide_to_gene = {"guide_TP53_1": "TP53", "non-targeting_1": "control"}
gex.obs["perturbation"] = gex.obs["guide"].map(guide_to_gene)
```

### 16.2 Mixscape (Seurat)

```r
seurat_obj <- CalcPerturbSig(seurat_obj, assay = "RNA", slot = "data",
                              gd.class = "perturbation", nt.cell.class = "control")
seurat_obj <- RunMixscape(seurat_obj, assay = "PRTB", slot = "scale.data",
                           labels = "perturbation", nt.class.name = "control",
                           new.class.name = "mixscape_class", fine.mode = TRUE)
```

### 16.3 Pertpy Analysis

```python
import pertpy as pt

# E-distance test for distributional shift
etest = pt.tl.DistanceTest("edistance", n_perms=1000)
etest_result = etest(adata, groupby="perturbation", contrast="control")
```

---

## 17. Scalability, Formats & Infrastructure

### 17.1 AnnData Structure

```python
import anndata as ad
import scipy.sparse as sp

# AnnData anatomy:
# adata.X            — primary matrix (n_cells × n_genes), scipy.sparse.csr_matrix
# adata.obs          — cell metadata (DataFrame)
# adata.var          — gene metadata (DataFrame)
# adata.obsm         — cell embeddings: "X_pca", "X_umap", "spatial"
# adata.varm         — gene embeddings: "PCs"
# adata.obsp         — cell pairwise: "distances", "connectivities"
# adata.layers       — alternative matrices: "counts", "spliced", "unspliced"
# adata.uns          — unstructured: clustering params, color palettes, spatial images

# Save / Load
adata.write_h5ad("data.h5ad", compression="gzip")
adata = ad.read_h5ad("data.h5ad")

# Backed mode for huge datasets
adata = ad.read_h5ad("huge_data.h5ad", backed="r")
```

### 17.2 Working with Sparse Matrices Efficiently

```python
import scipy.sparse as sp
import numpy as np

# Check sparsity
print(f"Sparsity: {1 - adata.X.nnz / np.prod(adata.X.shape):.1%}")  # Typically >90%

# Efficient operations on sparse matrix:
total_counts = np.array(adata.X.sum(axis=1)).flatten()
gene_means = np.array(adata.X.mean(axis=0)).flatten()
fraction_expressed = np.array((adata.X > 0).mean(axis=0)).flatten()

# DON'T do this for large datasets:
# df = adata.to_df()        # Sparse → dense → OOM
# dense = adata.X.toarray() # Same problem

# For column access, use CSC format:
X_csc = sp.csc_matrix(adata.X)
gene_expr = X_csc[:, gene_idx].toarray().flatten()
```

### 17.3 GPU-Accelerated Analysis

```python
import rapids_singlecell as rsc

rsc.get.anndata_to_GPU(adata)
rsc.pp.normalize_total(adata, target_sum=1e4)
rsc.pp.log1p(adata)
rsc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
rsc.pp.scale(adata)
rsc.pp.pca(adata, n_comps=50)
rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
rsc.tl.umap(adata)
rsc.tl.leiden(adata, resolution=0.8)
rsc.get.anndata_to_CPU(adata)
```

### 17.4 Format Conversion

```r
# R → Python
library(zellkonverter)
writeH5AD(as.SingleCellExperiment(seurat_obj), file = "data.h5ad")

# Python → R
library(SeuratDisk)
Convert("data.h5ad", dest = "h5seurat", overwrite = TRUE)
seurat_obj <- LoadH5Seurat("data.h5seurat")
```

---

## 18. Foundation Models & Deep Learning

### 18.1 scGPT

```python
import scgpt

model = scgpt.load_pretrained("scGPT_human")
embeddings = scgpt.embed(adata, model=model, gene_col="gene_name",
                           batch_size=64, device="cuda")
adata.obsm["X_scGPT"] = embeddings

# Fine-tune for annotation
scgpt.finetune(adata_train, adata_test, model=model, task="annotation",
                label_key="cell_type", batch_key="sample", epochs=10, lr=1e-4)
```

### 18.2 Geneformer

```python
from geneformer import TranscriptomeTokenizer, EmbExtractor

tokenizer = TranscriptomeTokenizer(custom_attr_name_dict={"cell_type": "cell_type"})
tokenizer.tokenize_data("path/to/h5ad/", "tokenized/", "dataset")

embex = EmbExtractor(model_type="Pretrained", emb_layer=-1, emb_mode="cell")
embeddings = embex.extract_embs("path/to/geneformer/", "tokenized/dataset.dataset", "embeddings/")
```

### 18.3 scVI for Generative Tasks

```python
import scvi

scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="sample")
model = scvi.model.SCVI(adata, n_latent=10, n_layers=2, gene_likelihood="nb")
model.train(max_epochs=400, early_stopping=True)

z = model.get_latent_representation()
adata.layers["scvi_norm"] = model.get_normalized_expression(library_size=1e4)

# Bayesian DE
de = model.differential_expression(groupby="condition", group1="treated", group2="control")
# de columns: proba_de, lfc_mean, bayes_factor, is_de_fdr_0.05

model.save("scvi_model/")
model = scvi.model.SCVI.load("scvi_model/", adata=adata)
```

---

## 19. Reproducibility & Reporting

### 19.1 Essential Metadata to Report

- **Species, tissue, condition, sample size** (biological replicates).
- **Platform and chemistry** (e.g., 10x Chromium 3' v3.1).
- **Dissociation protocol** (enzymatic vs. mechanical; fresh vs. frozen vs. FFPE).
- **Quantification pipeline and version** (e.g., Cell Ranger 7.2.0).
- **Reference genome and annotation** (e.g., GRCh38, GENCODE v44).
- **QC thresholds** and justification.
- **Normalization method**, number of HVGs, PCs.
- **Integration method** and parameters.
- **Clustering algorithm and resolution**.
- **Annotation strategy**.
- **DE method** and whether pseudobulking was used.
- **All software versions**.

### 19.2 Environment Management

```bash
pip freeze > requirements.txt
conda env export > environment.yml
```

```python
import scanpy as sc
sc.logging.print_header()
# scanpy==1.10.0 anndata==0.10.5 umap==0.5.5 numpy==1.26.4 ...
```

---

## 20. Quick-Reference Decision Trees

### 20.1 Complete Standard Pipeline (Python)

```python
import scanpy as sc
import scrublet as scr
import numpy as np

# 1. LOAD
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()

# 2. RAW COUNTS BACKUP
adata.layers["counts"] = adata.X.copy()

# 3. QC
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=True)

# 4. FILTER
sc.pp.filter_cells(adata, min_genes=200)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
adata = adata[adata.obs["n_genes_by_counts"] < 6000].copy()

# 5. DOUBLETS
scrub = scr.Scrublet(adata.X)
adata.obs["doublet_score"], adata.obs["is_doublet"] = scrub.scrub_doublets()
adata = adata[~adata.obs["is_doublet"]].copy()

# 6. NORMALIZE
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 7. HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")

# 8. SCALE + PCA
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)

# 9. NEIGHBORS + CLUSTERING
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.leiden(adata, resolution=0.8)

# 10. VISUALIZATION
sc.tl.umap(adata)
sc.pl.umap(adata, color=["leiden", "n_genes_by_counts", "pct_counts_mt"])

# 11. MARKERS
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", pts=True)
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, standard_scale="var")

# 12. SAVE
adata.write_h5ad("processed.h5ad")
```

### 20.2 Quantification Decision

```
10x Chromium? → Cell Ranger / STARsolo / Alevin-fry
Need velocity? → STARsolo --soloFeatures Velocyto / Alevin-fry splici / velocyto CLI
Single-nucleus? → --include-introns / splici index
Smart-seq2? → STAR + featureCounts / RSEM / Salmon
```

### 20.3 Spatial Decision

```
Visium (55 µm)? → Deconvolution essential (Cell2location/RCTD/Tangram)
Xenium/MERFISH? → Check segmentation (Cellpose/Baysor) → Squidpy spatial analysis
Visium HD? → 8 µm bins → rapids-singlecell for scale
```

---

## Appendix A: Installation

```bash
# Python Core
pip install scanpy anndata scvi-tools squidpy cellrank
pip install scrublet scvelo harmonypy bbknn
pip install liana decoupler pertpy pydeseq2
pip install cellbender spatialdata spatialdata-io
pip install rapids-singlecell  # GPU
pip install kb-python velocyto cellpose

# R Core
# install.packages("Seurat")
# remotes::install_github("immunogenomics/harmony")
# BiocManager::install(c("scran", "scater", "scDblFinder", "muscat", "SingleR", "BayesSpace"))
# install.packages("SoupX")
# remotes::install_github("sqjin/CellChat")
# remotes::install_github("dmcable/spacexr")
```

---

## Appendix B: Common Pitfalls

| Pitfall | Fix |
|---|---|
| No introns for snRNA-seq | `--include-introns` or splici index |
| Fixed QC thresholds | MAD-based adaptive thresholds |
| Cell-level DE (pseudo-replication) | Pseudobulk + DESeq2/edgeR |
| Over-integration | Verify biology preserved post-integration |
| Ignoring ambient RNA | SoupX or CellBender |
| Skipping doublets | Scrublet / scDblFinder (per sample) |
| UMAP distances as quantitative | Use PCA/latent for quantitative analysis |
| Log-normalizing before DE | Raw counts with proper models |
| Bad spatial segmentation | Cellpose/Baysor + visual inspection |
| `adata.to_df()` on large data | Operate on sparse `adata.X` directly |

---

## Appendix C: Math Notation

| Symbol | Meaning |
|---|---|
| `X[i,j]` | Raw UMI count, cell `i`, gene `j` |
| `s_i` | Size factor for cell `i` |
| `μ_ij` | Expected expression (NB mean) |
| `r_j`, `θ_j` | Inverse dispersion (NB model) |
| `z_i` | Latent representation of cell `i` |
| `u_j`, `s_j` | Unspliced/spliced mRNA counts |
| `α`, `β`, `γ` | Transcription/splicing/degradation rates |
| `v[i,j]` | RNA velocity |
| `I` | Moran's I (spatial autocorrelation) |
| `Q` | Modularity (clustering objective) |
| `T` | Transition matrix (Markov chain) |
| `K` | Spatial covariance kernel |

---

*Single-cell and spatial transcriptomics is a rapidly evolving field. Always check tool documentation for the latest versions. When in doubt, consult original methods papers and community benchmarks (sc-best-practices.org).*
