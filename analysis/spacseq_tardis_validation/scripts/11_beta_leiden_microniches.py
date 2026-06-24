#!/usr/bin/env python3
"""Beta + Leiden functional microniches vs SPAC-seq perturbation concordance.

SpaceTravLR betas are cluster-level, but each cell's *effective regulatory state*
varies with modulator expression. We build a per-cell beta-weighted GRN score matrix
(305 trained targets × 4,915 NTC cells), reduce with PCA, and cluster spatially
within each (slice, cell_type) using a joint beta+spatial embedding and Leiden.

Steps:
  1. Per-cell beta scores from betadata × imputed modulator expression
  2. Leiden microniches within slice × cell_type (beta PCA + spatial coords)
  3. Functional distinctness: silhouette, pathway Kruskal–Wallis across niches
  4. SPAC-seq concordance: predicted vs observed Δ per microniche (like script 09)
  5. Compare vs Space Ranger graphclust niches
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import ensure_boot

ensure_boot()

import argparse
import importlib.util
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("fig05", HERE / "05_final_report_figures.py")
_fig05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fig05)

compute_metrics = _fig05.compute_metrics
dense = _fig05.dense
GENE_SETS = _fig05.GENE_SETS

_spec09 = importlib.util.spec_from_file_location("sp09", HERE / "09_spatial_validation.py")
_sp09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_sp09)

niche_vectors = _sp09.niche_vectors
attach_graphclust = _sp09.attach_graphclust
load_baseline = _sp09.load_baseline
load_pool = _sp09.load_pool

DEFAULT_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]
DEFAULT_PERTS = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b"]
PX_PER_UM = 1.0 / 0.273
CELLTYPE_TO_CLUSTER = {"fibroblast": 0, "immune": 1, "myeloid": 2, "tumor": 3}


def ensure_cluster_id(ad: sc.AnnData) -> None:
    if "cluster_id" in ad.obs.columns:
        return
    ad.obs["cluster_id"] = ad.obs["cell_type"].astype(str).map(CELLTYPE_TO_CLUSTER).astype(int)


def prep_barcode(pool_barcode: str, slice_id: str) -> str:
    return f"{slice_id}__{pool_barcode}@{slice_id}"


def matched_modulators(betadata_dir: Path, var_names: set[str]) -> list[str]:
    mods: set[str] = set()
    for p in betadata_dir.glob("*_betadata.feather"):
        for c in pd.read_feather(p, columns=None).columns:
            if c.startswith("beta_") and c != "beta0":
                m = c.replace("beta_", "")
                if m in var_names:
                    mods.add(m)
    return sorted(mods)


def build_beta_score_matrix(
    prep: sc.AnnData,
    betadata_dir: Path,
    modulators: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if "imputed_count" in prep.layers:
        prep.X = prep.layers["imputed_count"]
    var = set(prep.var_names)
    if modulators is None:
        modulators = matched_modulators(betadata_dir, var)
    if not modulators:
        raise SystemExit("No betadata modulators overlap prep var_names")
    mod_expr = sc.get.obs_df(prep, keys=modulators).to_numpy(dtype=np.float64)
    cid = prep.obs["cluster_id"].astype(int).to_numpy()

    scores: list[np.ndarray] = []
    genes: list[str] = []
    for p in sorted(betadata_dir.glob("*_betadata.feather")):
        gene = p.stem.replace("_betadata", "")
        bd = pd.read_feather(p)
        rows = {int(r.Cluster): r for _, r in bd.iterrows()}
        b0 = np.array([rows[k]["beta0"] for k in range(4)], dtype=np.float64)[cid]
        acc = b0.copy()
        for c in bd.columns:
            if not c.startswith("beta_") or c == "beta0":
                continue
            m = c.replace("beta_", "")
            if m not in var:
                continue
            mi = modulators.index(m)
            beta_v = np.array([rows[k][c] for k in range(4)], dtype=np.float64)[cid]
            acc += beta_v * mod_expr[:, mi]
        scores.append(acc)
        genes.append(gene)
    return np.column_stack(scores), genes


def leiden_microniches(
    ad: sc.AnnData,
    beta_scores: np.ndarray,
    n_pcs: int = 15,
    n_neighbors: int = 12,
    resolution: float = 0.6,
    spatial_weight: float = 0.35,
    min_cells: int = 40,
) -> pd.Series:
    if ad.n_obs < min_cells:
        return pd.Series("0", index=ad.obs_names, name="beta_leiden")

    scores = np.nan_to_num(beta_scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    col_std = scores.std(axis=0)
    keep = col_std > 1e-8
    if keep.sum() < 5:
        return pd.Series("0", index=ad.obs_names, name="beta_leiden")
    scores = scores[:, keep]

    n_pcs = min(n_pcs, scores.shape[1], ad.n_obs - 1)
    tmp = sc.AnnData(X=scores)
    sc.pp.scale(tmp, max_value=10)
    tmp.X = np.nan_to_num(tmp.X, nan=0.0)
    sc.tl.pca(tmp, n_comps=n_pcs, svd_solver="arpack")
    beta_pca = tmp.obsm["X_pca"]

    xy = ad.obsm["spatial"].astype(np.float64)
    xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-8)
    beta_n = beta_pca / (np.std(beta_pca, axis=0, keepdims=True) + 1e-8)
    sw = spatial_weight
    joint = np.hstack([beta_n * (1 - sw), xy * sw])

    tmp2 = sc.AnnData(X=joint)
    sc.pp.neighbors(tmp2, n_neighbors=min(n_neighbors, ad.n_obs - 1), use_rep="X")
    sc.tl.leiden(tmp2, resolution=resolution, key_added="beta_leiden", flavor="igraph", n_iterations=2, directed=False)
    return pd.Series(tmp2.obs["beta_leiden"].astype(str).values, index=ad.obs_names, name="beta_leiden")


def assign_compartment_microniches(
    ad: sc.AnnData,
    beta_matrix: np.ndarray,
    slices: list[str],
    cell_types: list[str],
    prefix_labels: bool = True,
    **leiden_kw,
) -> pd.Series:
    labels = pd.Series(index=ad.obs_names, dtype=str)
    for sl in slices:
        for ct in cell_types:
            mask = (ad.obs["slice_id"].astype(str) == sl) & (ad.obs["cell_type"].astype(str) == ct)
            idx = np.where(mask.values)[0]
            if len(idx) == 0:
                continue
            sub = ad[mask].copy()
            sub_scores = beta_matrix[idx]
            sub_labels = leiden_microniches(sub, sub_scores, **leiden_kw)
            if prefix_labels:
                sub_labels = sub_labels.astype(str).radd(f"{sl}|{ct}|")
            labels.loc[sub.obs_names] = sub_labels.values
    labels = labels.fillna("unassigned")
    labels.name = "beta_leiden"
    return labels


def knn_assign_perturbed(
    pool: sc.AnnData,
    labels: pd.Series,
    cell_type: str,
) -> pd.Series:
    """Assign sgP cells to the beta-Leiden niche of their nearest NTC neighbor."""
    out = labels.copy()
    ct_mask = pool.obs["cell_type"].astype(str) == cell_type
    ntc_mask = ct_mask & (pool.obs["target_gene"].astype(str) == "non-targeting")
    pert_mask = ct_mask & (pool.obs["target_gene"].astype(str) != "non-targeting")
    ntc = pool[ntc_mask]
    if ntc.n_obs == 0 or pert_mask.sum() == 0:
        return out
    labeled = labels.loc[ntc.obs_names].astype(str)
    valid_ntc_list = list(ntc.obs_names[labeled.ne("unassigned") & labeled.ne("nan")])
    if len(valid_ntc_list) < 5:
        return out
    from sklearn.neighbors import NearestNeighbors

    xy_ntc = pool.obsm["spatial"][pool.obs_names.isin(valid_ntc_list)]
    xy_pert = pool.obsm["spatial"][pert_mask]
    nn = NearestNeighbors(n_neighbors=1).fit(xy_ntc)
    _, idx = nn.kneighbors(xy_pert)
    pert_names = pool.obs_names[pert_mask]
    for i, pn in enumerate(pert_names):
        ref = valid_ntc_list[idx[i, 0]]
        out[pn] = labels[ref]
    return out


def baseline_labels_from_pool(
    baseline: sc.AnnData,
    pool: sc.AnnData,
    pool_labels: pd.Series,
) -> pd.Series:
    sl = str(pool.obs["slice_id"].iloc[0])
    mapped = pd.Series(index=baseline.obs_names, dtype=str)
    ntc = pool[pool.obs["target_gene"].astype(str) == "non-targeting"]
    for bc in ntc.obs_names:
        key = prep_barcode(bc, sl)
        if key in baseline.obs_names:
            mapped[key] = pool_labels[bc]
    return mapped.fillna("unassigned")


def assign_all_microniches(
    prep: sc.AnnData,
    beta_matrix: np.ndarray,
    slices: list[str],
    cell_types: list[str],
    **leiden_kw,
) -> pd.Series:
    return assign_compartment_microniches(prep, beta_matrix, slices, cell_types, **leiden_kw)


def propagate_to_pool(pool: sc.AnnData, prep_labels: pd.Series) -> pd.Series:
    sl = str(pool.obs["slice_id"].iloc[0]) if "slice_id" in pool.obs.columns else ""
    mapped = pd.Series(index=pool.obs_names, dtype=str)
    for bc in pool.obs_names:
        key = prep_barcode(bc, sl) if sl else bc
        mapped[bc] = prep_labels.get(key, "unassigned")
    mapped.name = "beta_leiden"
    return mapped


def pathway_distinctness(pool: sc.AnnData, niche_key: str = "beta_leiden") -> list[dict]:
    rows = []
    if niche_key not in pool.obs.columns:
        return rows
    ntc = pool[pool.obs["target_gene"].astype(str) == "non-targeting"].copy()
    if ntc.n_obs < 30:
        return rows
    niches = ntc.obs[niche_key].astype(str)
    valid = niches[(niches != "unassigned") & (niches != "nan")]
    if valid.nunique() < 2:
        return rows
    for pathway, genes in GENE_SETS.items():
        keep = [g for g in genes if g in ntc.var_names]
        if len(keep) < 3:
            continue
        expr = dense(ntc, keep).mean(axis=1)
        groups = [expr[niches == n].values for n in sorted(valid.unique()) if (niches == n).sum() >= 5]
        if len(groups) < 2:
            continue
        try:
            h, p = stats.kruskal(*groups)
        except ValueError:
            continue
        rows.append({
            "pathway": pathway,
            "n_genes": len(keep),
            "n_niches": len(groups),
            "kruskal_H": float(h),
            "kruskal_p": float(p),
        })
    return rows


def silhouette_by_compartment(
    prep: sc.AnnData,
    beta_matrix: np.ndarray,
    labels: pd.Series,
    slices: list[str],
    cell_types: list[str],
) -> list[dict]:
    rows = []
    for sl in slices:
        for ct in cell_types:
            mask = (
                (prep.obs["slice_id"].astype(str) == sl)
                & (prep.obs["cell_type"].astype(str) == ct)
                & labels.astype(str).ne("unassigned")
            )
            if mask.sum() < 30:
                continue
            idx = np.where(mask.values)[0]
            y = labels[mask].astype(str)
            if y.nunique() < 2:
                continue
            X = beta_matrix[idx]
            sc.pp.scale(sc.AnnData(X=X), max_value=10)
            try:
                sil = float(silhouette_score(X, y, metric="euclidean"))
            except ValueError:
                sil = float("nan")
            rows.append({"slice": sl, "cell_type": ct, "n_cells": int(mask.sum()), "n_niches": int(y.nunique()), "silhouette": sil})
    return rows


def plot_spatial_niches(
    slice_name: str,
    cell_type: str,
    pool: sc.AnnData,
    fig_dir: Path,
    tag: str,
) -> None:
    ct = pool[(pool.obs["cell_type"].astype(str) == cell_type)].copy()
    if "beta_leiden" not in ct.obs.columns or ct.n_obs < 20:
        return
    niches = ct.obs["beta_leiden"].astype(str)
    uniq = sorted(n for n in niches.unique() if n not in ("unassigned", "nan"))
    if len(uniq) < 2:
        return
    cmap = plt.colormaps["tab10"].resampled(max(len(uniq), 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(
        axes,
        ["beta_leiden", "graphclust"],
        ["SpaceTravLR β-Leiden microniches", "Space Ranger graphclust"],
    ):
        if key not in ct.obs.columns:
            ax.axis("off")
            continue
        labs = ct.obs[key].astype(str)
        for i, lab in enumerate(sorted(labs.unique())):
            m = labs == lab
            if lab in ("unassigned", "nan"):
                c, s, a = "#dddddd", 0.3, 0.3
            else:
                c, s, a = cmap(i % 10), 1.0, 0.75
            ax.scatter(
                ct.obsm["spatial"][m, 0],
                ct.obsm["spatial"][m, 1],
                c=[c],
                s=s,
                alpha=a,
                rasterized=True,
                label=lab if key == "beta_leiden" else None,
            )
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"{slice_name} | {cell_type} — functional microniches ({tag})", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"spatial_beta_leiden_{slice_name}_{cell_type}_{tag}.png", dpi=180)
    plt.close(fig)


def compare_methods(df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    if df.empty:
        return
    cmp = (
        df.groupby(["niche_type", "perturbation", "cell_type"])
        .pearson_r.median()
        .reset_index()
    )
    pivot = cmp.pivot_table(index=["perturbation", "cell_type"], columns="niche_type", values="pearson_r")
    if pivot.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(0.55 * len(pivot) + 3, 0.45 * len(pivot.index) + 1.5))
    x = np.arange(len(pivot))
    w = 0.35
    cols = list(pivot.columns)
    for j, col in enumerate(cols):
        vals = pivot[col].values.astype(float)
        ax.bar(x + (j - 0.5) * w, vals, width=w, label=col)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"sg{p}|{c}" for p, c in pivot.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Median Pearson r (pred vs obs Δ)")
    ax.legend()
    ax.set_title(f"Microniche concordance: β-Leiden vs graphclust ({tag})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_compare_niche_methods_{tag}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--perturbations", nargs="+", default=DEFAULT_PERTS)
    ap.add_argument("--cell-types", nargs="+", default=["immune", "myeloid", "fibroblast"])
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path, required=True)
    ap.add_argument("--betadata-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/beta_leiden")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/beta_leiden")
    ap.add_argument("--tag", default="pooled")
    ap.add_argument("--min-ntc", type=int, default=6)
    ap.add_argument("--min-pert", type=int, default=10)
    ap.add_argument("--n-pcs", type=int, default=15)
    ap.add_argument("--leiden-resolution", type=float, default=0.6)
    ap.add_argument("--spatial-weight", type=float, default=0.35)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline(args.baseline_h5ad)
    genes = sorted(set(baseline.var_names))

    baseline = load_baseline(args.baseline_h5ad)
    genes = sorted(set(baseline.var_names))

    print("Building per-cell beta score matrix (baseline NTC)…")
    beta_matrix, beta_genes = build_beta_score_matrix(baseline, args.betadata_dir)
    print(f"  {beta_matrix.shape[1]} genes × {beta_matrix.shape[0]} cells")

    baseline = baseline.copy()
    baseline.obs["beta_leiden"] = "unassigned"

    all_niche_rows: list[dict] = []
    pathway_rows: list[dict] = []
    sil_rows: list[dict] = []

    for sl in args.slices:
        pool_path = args.data_root / "slices" / sl / "perturbed_pool.h5ad"
        if not pool_path.exists():
            continue
        pool = load_pool(sl, args.data_root)
        pool.obs["slice_id"] = sl
        ensure_cluster_id(pool)

        print(f"  {sl}: beta scores + Leiden on pool NTC…")
        pool_beta, _ = build_beta_score_matrix(pool, args.betadata_dir)
        ntc = pool[pool.obs["target_gene"].astype(str) == "non-targeting"].copy()
        ntc_beta = pool_beta[pool.obs["target_gene"].astype(str) == "non-targeting"]
        pool_labels = assign_compartment_microniches(
            ntc, ntc_beta, [sl], args.cell_types,
            n_pcs=args.n_pcs, resolution=args.leiden_resolution, spatial_weight=args.spatial_weight,
        )
        full_labels = pd.Series("unassigned", index=pool.obs_names, dtype=str)
        full_labels.loc[ntc.obs_names] = pool_labels.loc[ntc.obs_names]
        for ct in args.cell_types:
            full_labels = knn_assign_perturbed(pool, full_labels, ct)
        pool.obs["beta_leiden"] = full_labels.values

        base_l = baseline_labels_from_pool(baseline, pool, full_labels)
        baseline.obs.loc[base_l.index, "beta_leiden"] = base_l.values

        sil_rows.extend(
            silhouette_by_compartment(ntc, ntc_beta, pool_labels, [sl], args.cell_types)
        )

        baseline, pool = attach_graphclust(baseline, pool, args.data_root)

        pathway_rows.extend({**r, "niche_type": "beta_leiden"} for r in pathway_distinctness(pool, "beta_leiden"))
        pathway_rows.extend({**r, "niche_type": "graphclust"} for r in pathway_distinctness(pool, "graphclust"))

        for pert in args.perturbations:
            pred_path = args.pred_dir / f"predicted_KO_{pert}.feather"
            if not pred_path.exists():
                continue
            pred = pd.read_feather(pred_path).set_index("CellID")
            if (pool.obs["target_gene"].astype(str) == pert).sum() < args.min_pert:
                continue

            for ct in args.cell_types:
                for niche_key in ("beta_leiden", "graphclust"):
                    rows = niche_vectors(
                        baseline, pool, pred, pert, ct, niche_key, genes,
                        args.min_ntc, args.min_pert,
                    )
                    if rows is None:
                        continue
                    for r in rows:
                        r.update({
                            "slice": sl,
                            "perturbation": pert,
                            "cell_type": ct,
                            "niche_type": niche_key,
                        })
                    all_niche_rows.extend(rows)

        for ct in ["immune", "fibroblast"]:
            plot_spatial_niches(sl, ct, pool, args.fig_dir, args.tag)

    pd.DataFrame(sil_rows).to_csv(args.out_dir / f"silhouette_{args.tag}.csv", index=False)

    df = pd.DataFrame(all_niche_rows)
    if not df.empty:
        df.to_csv(args.out_dir / f"niche_corr_{args.tag}.csv", index=False)

    pdf = pd.DataFrame(pathway_rows)
    if not pdf.empty:
        pdf.to_csv(args.out_dir / f"pathway_distinctness_{args.tag}.csv", index=False)

    summary_rows = []
    if not df.empty:
        for (niche_type, pert, ct), grp in df.groupby(["niche_type", "perturbation", "cell_type"]):
            summary_rows.append({
                "niche_type": niche_type,
                "perturbation": pert,
                "cell_type": ct,
                "n_niche_rows": len(grp),
                "n_slices": grp.slice.nunique(),
                "median_pearson_r": float(grp.pearson_r.median()),
                "mean_pearson_r": float(grp.pearson_r.mean()),
                "frac_pos_r": float((grp.pearson_r > 0).mean()),
                "frac_perm_p05": float((grp.pearson_perm_p < 0.05).mean()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(["niche_type", "perturbation", "cell_type"])
    summary.to_csv(args.out_dir / f"summary_{args.tag}.csv", index=False)

    compare_methods(df, args.fig_dir, args.tag)

    if not df.empty:
        bl = df[df.niche_type == "beta_leiden"]
        if not bl.empty:
            pivot = bl.groupby(["perturbation", "cell_type"]).pearson_r.median().unstack()
            fig, ax = plt.subplots(figsize=(0.8 * pivot.shape[1] + 2, 0.45 * pivot.shape[0] + 1.5))
            vals = pivot.values.astype(float)
            vmax = max(0.2, float(np.nanmax(np.abs(vals))))
            im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels([f"sg{p}" for p in pivot.index])
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = vals[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"Median β-Leiden microniche Pearson r ({args.tag})")
            fig.colorbar(im, ax=ax, label="Pearson r")
            fig.tight_layout()
            fig.savefig(args.fig_dir / f"fig_beta_leiden_heatmap_{args.tag}.png", dpi=180)
            plt.close(fig)

    overall = {
        "tag": args.tag,
        "n_beta_genes": len(beta_genes),
        "n_niche_tests": int(len(df)),
        "silhouette_median": float(pd.DataFrame(sil_rows)["silhouette"].median()) if sil_rows else None,
        "by_niche_type": summary.groupby("niche_type").median_pearson_r.median().to_dict() if not summary.empty else {},
        "best_beta_leiden": summary[summary.niche_type == "beta_leiden"].nlargest(5, "median_pearson_r").to_dict("records") if not summary.empty else [],
        "pathway_sig_frac": float((pdf.kruskal_p < 0.05).mean()) if not pdf.empty else None,
    }
    (args.out_dir / f"overall_{args.tag}.json").write_text(json.dumps(overall, indent=2, default=str))
    print(json.dumps(overall, indent=2, default=str))


if __name__ == "__main__":
    main()
