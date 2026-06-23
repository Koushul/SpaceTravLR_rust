#!/usr/bin/env python3
"""Construct two h5ad files for SpaceTravLR validation against SPAC-seq perturbations.

baseline_ntc.h5ad
-----------------
Cells: all unambiguous-sgNTC cells (true non-perturb baseline; ~1.2k cells).
Used to TRAIN SpaceTravLR's GRN and as the substrate for in-silico KO.

perturbed_pool.h5ad
-------------------
Cells: NTC cells + cells from the top expanded perturbations (>=600 cells each).
Used to compute observed perturbation log2FC = mean(sgGene) - mean(NTC), per
cell type, per gene.

The two h5ads share the same gene panel, cluster_id encoding, and spatial frame
so predicted and observed effects can be compared on a level playing field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent
SOURCE_H5AD = REPO / "analysis/mc38_visiumhd/subQ-1/processed/mc38_subq1_cells_annotated.h5ad"
GUIDE_PARQUET = ROOT / "data/cell_guide_assignments.parquet"


# Paper-relevant marker / pathway genes that we always want trained, even if
# not in the HVG top-N. Mouse symbols.
ALWAYS_KEEP_GENES = [
    # SPAC-seq paper headline genes
    "Icam1", "Cd44", "Spp1", "Itgal", "Itgb2",
    # Top "expanded" perturbations in subQ-1 (have many cells -> good test targets)
    "Bcam", "Cks1b", "Ptk6", "Cd83", "Il4ra", "Cd74", "Bbs2", "App",
    "Tff3", "Piezo1", "Ehf", "Cldn15", "Nfib", "Fkrp", "Thbs1",
    # Immune effector
    "Cd8a", "Cd8b1", "Cd3d", "Cd3e", "Cd4", "Foxp3", "Gzmb", "Gzma", "Prf1",
    "Ifng", "Tnf", "Pdcd1", "Lag3", "Tigit", "Havcr2",
    # Macrophage M1 / inflammatory
    "Tnf", "Nos2", "Il1b", "Il6", "Cxcl9", "Cxcl10", "Cxcl11", "Stat1",
    "H2-Aa", "H2-Ab1",
    # Macrophage M2 / suppressive
    "Arg1", "Mrc1", "Cd163", "Il10", "Trem2", "Vegfa", "Tgfb1", "Chil3",
    "Retnla", "Mertk",
    # Tumor / EMT / MHC-I
    "Epcam", "Krt8", "Krt18", "Vim", "Cdh1", "Cdh2", "Snai1", "Snai2",
    "H2-K1", "H2-D1", "B2m", "Tap1", "Tap2", "Nlrc5",
    # Chemokine receptors / TFs (paper TF–chemokine axis)
    "Ccr2", "Ccr5", "Ccr7", "Cxcr3", "Cxcr4", "Cxcr5", "Cx3cr1",
    # Stress / proliferation
    "Mki67", "Top2a", "Pcna", "Hspa1a", "Atf4",
]


def select_target_genes(adata: sc.AnnData, n_hvg: int) -> list[str]:
    """Top-n HVG (Seurat flavor) on log1p-normalized COPY; plus ALWAYS_KEEP_GENES."""
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat")
    hvg = tmp.var.index[tmp.var["highly_variable"]].tolist()
    extra = [g for g in ALWAYS_KEEP_GENES if g in adata.var_names and g not in hvg]
    return list(dict.fromkeys(hvg + extra))


def encode_clusters(adata: sc.AnnData) -> tuple[sc.AnnData, dict[str, int]]:
    cat_order = sorted(adata.obs["cell_type"].astype(str).unique())
    code = {ct: i for i, ct in enumerate(cat_order)}
    adata.obs["cluster_id"] = adata.obs["cell_type"].astype(str).map(code).astype("int32")
    return adata, code


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-h5ad", type=Path, default=SOURCE_H5AD)
    p.add_argument("--guide-parquet", type=Path, default=GUIDE_PARQUET)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data")
    p.add_argument("--n-hvg", type=int, default=600,
                   help="Top variable genes to include as SpaceTravLR target panel.")
    p.add_argument("--perturb-genes", nargs="*", default=[
        "Bcam", "Cks1b", "Ptk6", "Cd83", "Il4ra", "Cd74",
    ], help="Perturbation target genes whose cells will populate perturbed_pool.h5ad.")
    p.add_argument("--max-cells-per-pert", type=int, default=1500,
                   help="Cap the number of cells per perturbation cohort.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading source AnnData (annotations) ...")
    adata = sc.read_h5ad(args.source_h5ad)
    print(f"  annotated source: {adata.shape}")

    # The annotated adata stores log1p-normalized values. SpaceTravLR auto-prep
    # expects raw counts. Reload counts from the cell-level Space Ranger h5 and
    # graft cell_type / spatial onto it.
    raw_h5 = args.source_h5ad.parent.parent / "segmentation/extracted/segmentation/filtered_feature_cell_matrix.h5"
    print(f"  reloading raw counts from {raw_h5}")
    raw = sc.read_10x_h5(raw_h5)
    raw.var_names_make_unique()
    keep = raw.obs_names.isin(adata.obs_names)
    raw = raw[keep, :].copy()
    raw = raw[:, raw.var_names.isin(adata.var_names)].copy()
    # match var order to annotated adata so panel masking works the same
    raw = raw[:, adata.var_names].copy()
    # transfer obs + spatial
    common_obs = raw.obs_names
    raw.obs = adata.obs.loc[common_obs, ["graphclust", "cell_type"]].copy()
    raw.obsm["spatial"] = adata[common_obs].obsm["spatial"]
    adata = raw  # adata now has raw counts
    print(f"  raw-counts adata: {adata.shape}, X dtype={adata.X.dtype}, nnz={adata.X.nnz if sparse.issparse(adata.X) else 'dense'}")
    ca = pd.read_parquet(args.guide_parquet)
    unamb = ca[ca.is_unambiguous].copy()
    print(f"  unambiguous-guide cells: {len(unamb)}")

    cell_to_target = dict(zip(unamb.cell_barcode, unamb.target_gene))
    ntc_set = set(unamb.loc[unamb.is_ntc, "cell_barcode"])

    adata.obs["target_gene"] = [cell_to_target.get(b, "no_guide") for b in adata.obs_names]
    adata.obs["is_ntc"] = adata.obs_names.isin(ntc_set)

    # ----- gene panel (training target list) -----
    panel = select_target_genes(adata, n_hvg=args.n_hvg)
    panel_idx = adata.var_names.isin(panel)
    print(f"  selected gene panel: {panel_idx.sum()} / {adata.n_vars} (incl. {len(ALWAYS_KEEP_GENES)} forced markers)")
    # Keep the FULL transcriptome in the h5ad so SpaceTravLR's QC (min_genes)
    # filter uses real per-cell complexity, not the panel-only count. The
    # target gene list is written to data/target_genes.txt and passed via
    # [training].genes in the config so only those genes get trained.

    # ----- baseline (NTC) h5ad -----
    ntc_mask = adata.obs["is_ntc"].values
    baseline = adata[ntc_mask, :].copy()
    baseline, code = encode_clusters(baseline)
    baseline.uns = {}  # drop log1p flag - SpaceTravLR will redo
    if sparse.issparse(baseline.X):
        baseline.X = baseline.X.astype(np.float32)
    baseline.obs["target_gene"] = "non-targeting"
    baseline.obs["sample_role"] = "baseline_ntc"
    print(f"  baseline_ntc: {baseline.shape}  cluster_id codes: {code}")
    baseline.write_h5ad(args.out_dir / "baseline_ntc.h5ad")

    # ----- perturbed_pool h5ad -----
    rng = np.random.default_rng(42)
    keep_cells = []
    cohort_table = []
    # NTC always included
    keep_cells.extend(adata.obs_names[ntc_mask].tolist())
    cohort_table.append({"target_gene": "non-targeting", "n_cells": int(ntc_mask.sum()), "role": "control"})
    for g in args.perturb_genes:
        m = (adata.obs["target_gene"] == g).values
        idx = adata.obs_names[m]
        if len(idx) > args.max_cells_per_pert:
            idx = pd.Index(rng.choice(idx, size=args.max_cells_per_pert, replace=False))
        keep_cells.extend(idx.tolist())
        cohort_table.append({"target_gene": g, "n_cells": int(len(idx)), "role": "perturbation"})
    keep_idx = pd.Index(dict.fromkeys(keep_cells).keys())  # dedupe preserving order
    pert = adata[keep_idx, :].copy()
    pert, code2 = encode_clusters(pert)
    assert code == code2, "cluster encoding mismatch"
    pert.uns = {}
    pert.obs["sample_role"] = np.where(pert.obs["is_ntc"], "baseline_ntc", "perturbed")
    pert.write_h5ad(args.out_dir / "perturbed_pool.h5ad")
    print(f"  perturbed_pool: {pert.shape}")

    # ----- summary -----
    summary = {
        "n_genes_panel": int(len(panel)),
        "panel_first_30": panel[:30],
        "always_kept_present": [g for g in ALWAYS_KEEP_GENES if g in panel],
        "baseline_ntc": {
            "n_cells": int(baseline.n_obs),
            "cluster_id_codes": code,
            "cell_type_counts": baseline.obs["cell_type"].value_counts().to_dict(),
        },
        "perturbed_pool_cohorts": cohort_table,
    }
    (args.out_dir / "gene_panel_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # write panel as one-line CSV for the CLI --genes flag if needed
    (args.out_dir / "target_genes.txt").write_text("\n".join(panel) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
