#!/usr/bin/env python3
"""QC + SpaceTravLR CCC potential assessment for Stereo-seq host/microbe H5ADs."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

BASE = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome")
RAW = BASE / "raw"
PROC = BASE / "processed"
NOTES = BASE / "notes"
PROC.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

CELLCHAT = Path("/ix1/ylee/kor11/tools/SpaceTravLR/data/cellchat_mouse.csv")
GRN = Path("/ix1/ylee/kor11/tools/SpaceTravLR/data/mouse_base_grn.parquet")
PIX_PER_UM = 2.0  # 0.5 µm / pixel


def clean_celltype(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"^q05cell_abundance_w_sf_", "", regex=True)
        .replace({"nan": np.nan, "None": np.nan})
    )


def aggregate_host(host_path: Path, min_umi: int = 30) -> ad.AnnData:
    print(f"[agg] loading {host_path.name}")
    h = ad.read_h5ad(host_path)
    if "cell" not in h.obs:
        raise ValueError(f"no cell column in {host_path}")
    keep = h.obs["cell"].notna().values
    gene_keep = h.var["total"].fillna(0).values > 0 if "total" in h.var else np.ones(h.n_vars, bool)
    h = h[keep].copy()
    h = h[:, gene_keep].copy()
    h.obs["cell_id"] = h.obs["cell"].astype(str)
    ct_col = "celltype" if "celltype" in h.obs else None
    if ct_col:
        h.obs["cell_type_raw"] = clean_celltype(h.obs[ct_col])
    else:
        h.obs["cell_type_raw"] = "unknown"

    cell_ids = h.obs["cell_id"].astype("category")
    codes = cell_ids.cat.codes.to_numpy()
    n_cells = cell_ids.cat.categories.size
    X = h.X.tocsr() if sparse.issparse(h.X) else sparse.csr_matrix(h.X)
    ind = sparse.csr_matrix(
        (np.ones(h.n_obs), (codes, np.arange(h.n_obs))), shape=(n_cells, h.n_obs)
    )
    Xcell = ind @ X
    coords = np.asarray(h.obsm["spatial"], dtype=float)
    coord_mean = (ind @ coords) / np.asarray(ind.sum(1)).ravel()[:, None]
    umi = np.asarray(Xcell.sum(1)).ravel()
    n_bins = np.asarray(ind.sum(1)).ravel()
    n_genes = np.asarray((Xcell > 0).sum(1)).ravel()

    ct = (
        h.obs.groupby("cell_id", observed=True)["cell_type_raw"]
        .agg(lambda s: s.mode(dropna=True).iloc[0] if s.notna().any() else np.nan)
        .reindex(cell_ids.cat.categories.astype(str))
    )
    tissue_frac = None
    if "mask_tissue" in h.obs:
        tissue_frac = np.asarray(
            ind @ h.obs["mask_tissue"].astype(float).to_numpy()
        ).ravel() / n_bins
    lumen_frac = None
    if "mask_lumen" in h.obs:
        lumen_frac = np.asarray(
            ind @ h.obs["mask_lumen"].astype(float).to_numpy()
        ).ravel() / n_bins

    obs = pd.DataFrame(
        {
            "cell_id": cell_ids.cat.categories.astype(str),
            "n_bins": n_bins,
            "n_counts": umi,
            "n_genes": n_genes,
            "cell_type": ct.values,
        }
    ).set_index("cell_id")
    if tissue_frac is not None:
        obs["tissue_frac"] = tissue_frac
    if lumen_frac is not None:
        obs["lumen_frac"] = lumen_frac

    qc = (obs["n_counts"] >= min_umi) & obs["cell_type"].notna()
    adata = ad.AnnData(X=Xcell[qc.values], obs=obs.loc[qc].copy(), var=h.var.copy())
    adata.obsm["spatial_px"] = coord_mean[qc.values]
    adata.obsm["spatial"] = coord_mean[qc.values] / PIX_PER_UM
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.var_names = adata.var_names.astype(str)
    # normalize gene symbols to CellChat-like Title case where needed
    adata.var["symbol_lower"] = adata.var_names.str.lower()
    print(f"[agg] cells={adata.n_obs} genes={adata.n_vars} medianUMI={adata.obs.n_counts.median():.0f}")
    return adata


def attach_microbes(adata: ad.AnnData, unmap_path: Path, radius_um: float = 50.0) -> dict:
    print(f"[micro] loading {unmap_path.name}")
    m = ad.read_h5ad(unmap_path)
    mv = m.var
    keep = (
        (mv["superkingdom"] == "Bacteria")
        & mv["genus"].notna()
        & (~mv["genus"].isin(["Mus", "Homo"]))
    )
    m = m[:, keep].copy()
    genera = m.var["genus"].astype(str)
    uniq = pd.Index(sorted(genera.unique()))
    gmap = {g: i for i, g in enumerate(uniq)}
    cols = np.array([gmap[g] for g in genera])
    G = sparse.csr_matrix(
        (np.ones(m.n_vars), (np.arange(m.n_vars), cols)), shape=(m.n_vars, len(uniq))
    )
    M = m.X.tocsr() @ G
    m_coords = np.asarray(m.obsm["spatial"], dtype=float)
    use = np.asarray(M.sum(1)).ravel() >= 1
    m_coords = m_coords[use]
    M = M[use]
    radius_px = radius_um * PIX_PER_UM
    nn = NearestNeighbors(radius=radius_px, algorithm="kd_tree").fit(m_coords)
    neigh = nn.radius_neighbors(adata.obsm["spatial_px"], return_distance=False)
    micro = np.zeros((adata.n_obs, len(uniq)), dtype=np.float32)
    for i, idx in enumerate(neigh):
        if len(idx):
            micro[i] = np.asarray(M[idx].sum(0)).ravel()
    adata.obsm["micro_genus_near50um"] = micro
    adata.uns["micro_genus_names"] = list(uniq)
    adata.obs["micro_near50um_total"] = micro.sum(1)
    top_idx = np.argsort(micro.sum(0))[-15:][::-1]
    top = {uniq[j]: float(micro[:, j].sum()) for j in top_idx}
    for j in top_idx[:8]:
        adata.obs[f"micro_{uniq[j]}"] = micro[:, j]
    summary = {
        "n_genera": int(len(uniq)),
        "frac_cells_with_micro": float((adata.obs.micro_near50um_total > 0).mean()),
        "median_micro_umi_near_cell": float(adata.obs.micro_near50um_total.median()),
        "top_genera_total_near_cells": top,
        "radius_um": radius_um,
    }
    return summary


def gene_lookup(adata: ad.AnnData) -> dict[str, str]:
    """Map lowercase symbol -> actual var_name."""
    return {g.lower(): g for g in adata.var_names.astype(str)}


def split_complex(name: str) -> list[str]:
    return [p for p in str(name).split("_") if p]


def coverage_metrics(adata: ad.AnnData) -> dict:
    gmap = gene_lookup(adata)
    lr = pd.read_csv(CELLCHAT)
    # expressed = detected in >=1% cells OR mean > 0.01 counts
    X = adata.X.tocsr()
    n = adata.n_obs
    detected_frac = np.asarray((X > 0).sum(0)).ravel() / n
    mean_counts = np.asarray(X.mean(0)).ravel()
    det = {g.lower(): (detected_frac[i], mean_counts[i]) for i, g in enumerate(adata.var_names)}

    def present(sym: str, min_frac: float = 0.01) -> bool:
        key = sym.lower()
        if key not in det:
            return False
        return det[key][0] >= min_frac or det[key][1] >= 0.02

    ligands = set()
    receptors = set()
    pairs_present = []
    pairs_partial = []
    pathway_hits = {}
    for _, row in lr.iterrows():
        ligs = split_complex(row["ligand"])
        recs = split_complex(row["receptor"])
        lig_ok = all(present(x) for x in ligs)
        rec_ok = all(present(x) for x in recs)
        for x in ligs:
            if present(x):
                ligands.add(x)
        for x in recs:
            if present(x):
                receptors.add(x)
        pair = f"{row['ligand']}${row['receptor']}"
        if lig_ok and rec_ok:
            pairs_present.append(
                {"pair": pair, "pathway": row["pathway"], "signaling": row["signaling"]}
            )
            pathway_hits[row["pathway"]] = pathway_hits.get(row["pathway"], 0) + 1
        elif any(present(x) for x in ligs + recs):
            pairs_partial.append(pair)

    # TF coverage from GRN column names (skip peak_id, gene_short_name)
    grn = pd.read_parquet(GRN, columns=["gene_short_name"])
    # read only header for TF names
    import pyarrow.parquet as pq

    schema = pq.read_schema(GRN)
    tf_names = [c for c in schema.names if c not in ("peak_id", "gene_short_name")]
    tfs_present = [tf for tf in tf_names if present(tf, min_frac=0.005)]
    targets = set(grn["gene_short_name"].astype(str))
    targets_present = [t for t in targets if present(t, min_frac=0.01)]

    # top expressed LR genes
    lr_genes = sorted({*ligands, *receptors})
    lr_expr = []
    for g in lr_genes:
        key = g.lower()
        if key in gmap:
            i = list(adata.var_names).index(gmap[key]) if False else None
        # use det
        if key in det:
            lr_expr.append({"gene": g, "frac": float(det[key][0]), "mean": float(det[key][1])})
    lr_expr = sorted(lr_expr, key=lambda d: d["frac"], reverse=True)[:40]

    return {
        "n_cellchat_pairs_db": int(len(lr)),
        "n_pairs_both_present": int(len(pairs_present)),
        "n_pairs_partial": int(len(pairs_partial)),
        "n_ligands_present": int(len(ligands)),
        "n_receptors_present": int(len(receptors)),
        "pathway_hits": dict(sorted(pathway_hits.items(), key=lambda kv: -kv[1])[:25]),
        "top_pairs": pairs_present[:40],
        "n_tfs_in_grn": int(len(tf_names)),
        "n_tfs_present": int(len(tfs_present)),
        "tfs_present_top": sorted(tfs_present)[:60],
        "n_grn_targets_present": int(len(targets_present)),
        "n_grn_targets_db": int(len(targets)),
        "top_lr_genes_by_detection": lr_expr,
        "detection_threshold": "frac>=1% or mean>=0.02 counts",
    }


def spatial_ccc_potential(adata: ad.AnnData, radius_um: float = 100.0) -> dict:
    """Estimate neighbor LR co-occurrence potential for top pathways."""
    gmap = gene_lookup(adata)
    lr = pd.read_csv(CELLCHAT)
    X = adata.X.tocsr()
    # binary detection
    det_mat = (X > 0).astype(np.float32)
    coords = adata.obsm["spatial"]
    nn = NearestNeighbors(radius=radius_um, algorithm="kd_tree").fit(coords)
    neigh = nn.radius_neighbors(coords, return_distance=False)

    def gene_vec(sym: str):
        key = sym.lower()
        if key not in gmap:
            return None
        idx = adata.var_names.get_loc(gmap[key])
        return np.asarray(det_mat[:, idx].todense()).ravel()

    # score top candidate pairs that are present
    cov = []
    for _, row in lr.iterrows():
        ligs = split_complex(row["ligand"])
        recs = split_complex(row["receptor"])
        lvecs = [gene_vec(x) for x in ligs]
        rvecs = [gene_vec(x) for x in recs]
        if any(v is None for v in lvecs + rvecs):
            continue
        lig = np.prod(np.vstack(lvecs), axis=0)  # AND for complexes
        rec = np.prod(np.vstack(rvecs), axis=0)
        if lig.sum() < 5 or rec.sum() < 5:
            continue
        cov.append((float(lig.sum() + rec.sum()), row, lig, rec))
    cov = sorted(cov, key=lambda t: -t[0])[:80]

    results = []
    rng = np.random.default_rng(0)
    # sample cells for speed
    sample_idx = rng.choice(adata.n_obs, size=min(4000, adata.n_obs), replace=False)
    for _, row, lig, rec in cov[:30]:
        hits = 0
        trials = 0
        for i in sample_idx:
            if lig[i] <= 0:
                continue
            nbrs = neigh[i]
            if len(nbrs) <= 1:
                continue
            trials += 1
            if rec[nbrs].sum() > 0:
                hits += 1
        # also receptor-centered
        hits_r = 0
        trials_r = 0
        for i in sample_idx:
            if rec[i] <= 0:
                continue
            nbrs = neigh[i]
            if len(nbrs) <= 1:
                continue
            trials_r += 1
            if lig[nbrs].sum() > 0:
                hits_r += 1
        if trials < 10 and trials_r < 10:
            continue
        results.append(
            {
                "pair": f"{row['ligand']}${row['receptor']}",
                "pathway": row["pathway"],
                "signaling": row["signaling"],
                "n_lig_pos": int(lig.sum()),
                "n_rec_pos": int(rec.sum()),
                "frac_lig_with_rec_neighbor": float(hits / trials) if trials else None,
                "frac_rec_with_lig_neighbor": float(hits_r / trials_r) if trials_r else None,
                "trials_lig": int(trials),
                "trials_rec": int(trials_r),
            }
        )
    results = sorted(
        results,
        key=lambda d: -(
            (d["frac_lig_with_rec_neighbor"] or 0) + (d["frac_rec_with_lig_neighbor"] or 0)
        ),
    )
    # cell-type neighbor mix
    cts = adata.obs["cell_type"].astype(str).to_numpy()
    uniq_ct = sorted(adata.obs["cell_type"].astype(str).unique())
    ct_index = {c: i for i, c in enumerate(uniq_ct)}
    mix = np.zeros((len(uniq_ct), len(uniq_ct)), dtype=float)
    for i in sample_idx:
        nbrs = neigh[i]
        if len(nbrs) <= 1:
            continue
        a = ct_index[cts[i]]
        for j in nbrs:
            if j == i:
                continue
            mix[a, ct_index[cts[j]]] += 1
    # top edges
    edges = []
    for i, a in enumerate(uniq_ct):
        for j, b in enumerate(uniq_ct):
            if mix[i, j] > 0:
                edges.append({"source": a, "target": b, "count": float(mix[i, j])})
    edges = sorted(edges, key=lambda d: -d["count"])[:40]
    return {
        "radius_um": radius_um,
        "pair_neighbor_scores": results[:20],
        "celltype_neighbor_edges": edges,
        "n_celltypes": len(uniq_ct),
    }


def qc_summary(adata: ad.AnnData, sample: str, micro: dict) -> dict:
    obs = adata.obs
    umi = obs["n_counts"]
    genes = obs["n_genes"]
    ct_counts = obs["cell_type"].astype(str).value_counts().to_dict()
    # spatial extent
    xy = adata.obsm["spatial"]
    extent = {
        "x_um": [float(xy[:, 0].min()), float(xy[:, 0].max())],
        "y_um": [float(xy[:, 1].min()), float(xy[:, 1].max())],
        "span_x_um": float(xy[:, 0].max() - xy[:, 0].min()),
        "span_y_um": float(xy[:, 1].max() - xy[:, 1].min()),
    }
    # UMI histogram bins
    bins = [0, 30, 50, 80, 120, 200, 500, 2000]
    umi_hist = {
        f"{bins[i]}-{bins[i+1]}": int(((umi >= bins[i]) & (umi < bins[i + 1])).sum())
        for i in range(len(bins) - 1)
    }
    # gene detection tiers
    X = adata.X.tocsr()
    frac = np.asarray((X > 0).sum(0)).ravel() / adata.n_obs
    gene_tiers = {
        "detected_any": int((frac > 0).sum()),
        "detected_ge_1pct": int((frac >= 0.01).sum()),
        "detected_ge_5pct": int((frac >= 0.05).sum()),
        "detected_ge_10pct": int((frac >= 0.10).sum()),
    }
    tissue = {}
    if "tissue_frac" in obs:
        tissue["median_tissue_frac"] = float(obs.tissue_frac.median())
        tissue["frac_cells_mostly_tissue"] = float((obs.tissue_frac >= 0.5).mean())
    if "lumen_frac" in obs:
        tissue["median_lumen_frac"] = float(obs.lumen_frac.median())
        tissue["frac_cells_mostly_lumen"] = float((obs.lumen_frac >= 0.5).mean())

    # micro vs cell type
    micro_by_ct = {}
    if "micro_near50um_total" in obs:
        g = obs.groupby("cell_type", observed=True)["micro_near50um_total"]
        micro_by_ct = {
            str(k): {"median": float(v.median()), "mean": float(v.mean()), "n": int(v.size)}
            for k, v in g
        }

    return {
        "sample": sample,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "umi": {
            "median": float(umi.median()),
            "mean": float(umi.mean()),
            "p10": float(umi.quantile(0.1)),
            "p25": float(umi.quantile(0.25)),
            "p75": float(umi.quantile(0.75)),
            "p90": float(umi.quantile(0.9)),
            "max": float(umi.max()),
            "hist": umi_hist,
        },
        "n_genes_per_cell": {
            "median": float(genes.median()),
            "mean": float(genes.mean()),
            "p10": float(genes.quantile(0.1)),
            "p90": float(genes.quantile(0.9)),
        },
        "gene_detection_tiers": gene_tiers,
        "celltype_counts": {str(k): int(v) for k, v in sorted(ct_counts.items(), key=lambda kv: -kv[1])},
        "n_celltypes": int(obs["cell_type"].nunique()),
        "spatial_um": extent,
        "tissue_masks": tissue,
        "microbe_neighborhood": micro,
        "micro_by_celltype": dict(
            sorted(micro_by_ct.items(), key=lambda kv: -kv[1]["median"])[:15]
        ),
    }


def spacetravlr_verdict(qc: dict, cov: dict, ccc: dict) -> dict:
    median_umi = qc["umi"]["median"]
    pairs = cov["n_pairs_both_present"]
    tfs = cov["n_tfs_present"]
    neighbor_hits = [
        p
        for p in ccc["pair_neighbor_scores"]
        if (p.get("frac_lig_with_rec_neighbor") or 0) >= 0.2
        or (p.get("frac_rec_with_lig_neighbor") or 0) >= 0.2
    ]
    flags = []
    if median_umi < 100:
        flags.append("low_host_umi_depth")
    if pairs < 50:
        flags.append("sparse_lr_pair_coverage")
    if tfs < 50:
        flags.append("sparse_tf_coverage")
    if qc["n_cells"] < 2000:
        flags.append("few_cells")
    if not neighbor_hits:
        flags.append("weak_spatial_lr_cooccurrence")

    if median_umi >= 80 and pairs >= 80 and tfs >= 40 and qc["n_cells"] >= 5000:
        grade = "usable_with_caveats"
    elif median_umi >= 50 and pairs >= 40:
        grade = "pilot_restricted_panel"
    else:
        grade = "not_ready_without_imputation"

    recs = []
    if "low_host_umi_depth" in flags:
        recs.append(
            "Restrict SpaceTravLR targets to detected LR/TF/antimicrobial genes; use knn/MAGIC-style imputation before training."
        )
    if qc["microbe_neighborhood"]["frac_cells_with_micro"] > 0.5:
        recs.append(
            "Add top nearby genera (or micro_near50um_total) as extra_modulators / niche covariates rather than CellChat pairs."
        )
    if neighbor_hits:
        recs.append(
            f"Prioritize spatially co-localized pairs ({len(neighbor_hits)} with ≥20% neighbor co-occurrence in 100 µm)."
        )
    recs.append(
        "Train on mucosa/tumour-boundary cells; exclude deep tissue or lumen-contaminated bins."
    )
    return {
        "grade": grade,
        "flags": flags,
        "n_spatial_lr_candidates": len(neighbor_hits),
        "recommendations": recs,
        "headline_pairs": neighbor_hits[:10],
    }


def process_sample(name: str, host: Path, unmap: Path, out_h5ad: Path) -> dict:
    adata = aggregate_host(host, min_umi=30)
    micro = attach_microbes(adata, unmap, radius_um=50.0)
    qc = qc_summary(adata, name, micro)
    print(f"[cov] CellChat/TF coverage for {name}")
    cov = coverage_metrics(adata)
    print(f"[ccc] spatial CCC potential for {name}")
    ccc = spatial_ccc_potential(adata, radius_um=100.0)
    verdict = spacetravlr_verdict(qc, cov, ccc)
    adata.uns["qc_summary"] = qc
    adata.uns["ccc_coverage"] = {
        k: cov[k]
        for k in [
            "n_pairs_both_present",
            "n_ligands_present",
            "n_receptors_present",
            "n_tfs_present",
            "pathway_hits",
        ]
    }
    adata.write_h5ad(out_h5ad, compression="gzip")
    print(f"[write] {out_h5ad} ({out_h5ad.stat().st_size/1e6:.1f} MB)")
    return {
        "qc": qc,
        "coverage": cov,
        "spatial_ccc": ccc,
        "verdict": verdict,
        "h5ad": str(out_h5ad),
    }


def main():
    samples = {
        "ileum_pap": {
            "host": RAW / "stereoseq_ileum/GSM9247063_G511_adata_host.h5ad",
            "unmap": RAW / "stereoseq_ileum/GSM9247063_G511_adata_unmap.h5ad",
            "out": PROC / "GSM9247063_ileum_cells_spacetravlr_ready.h5ad",
        },
        "tumor_pap": {
            "host": RAW / "stereoseq_tumor/GSM9456850_A612_host.h5ad",
            "unmap": RAW / "stereoseq_tumor/GSM9456850_A612_unmap.h5ad",
            "out": PROC / "GSM9456850_tumor_cells_spacetravlr_ready.h5ad",
        },
    }
    report = {
        "paper": "https://www.nature.com/articles/s41564-026-02286-7",
        "platform": "Stereo-seq + in situ polyadenylation",
        "pixel_size_um": 0.5,
        "samples": {},
    }
    for name, cfg in samples.items():
        report["samples"][name] = process_sample(name, cfg["host"], cfg["unmap"], cfg["out"])

    # comparative summary
    report["comparison"] = {
        "median_umi": {
            k: report["samples"][k]["qc"]["umi"]["median"] for k in report["samples"]
        },
        "n_cells": {k: report["samples"][k]["qc"]["n_cells"] for k in report["samples"]},
        "lr_pairs_present": {
            k: report["samples"][k]["coverage"]["n_pairs_both_present"]
            for k in report["samples"]
        },
        "tfs_present": {
            k: report["samples"][k]["coverage"]["n_tfs_present"] for k in report["samples"]
        },
        "grades": {k: report["samples"][k]["verdict"]["grade"] for k in report["samples"]},
    }
    out_json = NOTES / "qc_ccc_assessment.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
