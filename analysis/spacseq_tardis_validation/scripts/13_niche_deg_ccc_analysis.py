#!/usr/bin/env python3
"""Spatial neighbor DEG + cell-cell communication validation vs SpaceTravLR.

Uses spatial kNN niches (bystander cells near sgP vs NTC sources) and β-Leiden
microniches to compare SPAC-seq Wilcoxon DEGs against SpaceTravLR predicted
pseudobulk deltas. Includes T cell state and antigen-presentation / CCC pathway
scores following SPAC-seq paper themes (local functional genomics, immune niche
effects).

Outputs:
  results/niche_deg/spatial_neighbor_stats.csv
  results/niche_deg/beta_leiden_deg_stats.csv
  results/niche_deg/ccc_state_scores.csv
  figures/niche_deg/fig6_spatial_neighbor_grid.png
  figures/niche_deg/fig7_beta_leiden_deg_grid.png
  figures/niche_deg/fig8_ccc_tcell_state.png
  figures/niche_deg/fig9_pathway_concordance_heatmap.png
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
import seaborn as sns
from scipy import sparse, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import niche_deg_utils as ndu

_spec09 = importlib.util.spec_from_file_location("sp09", HERE / "09_spatial_validation.py")
_sp09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_sp09)
load_baseline = _sp09.load_baseline
load_pool = _sp09.load_pool

_spec11 = importlib.util.spec_from_file_location("bl11", HERE / "11_beta_leiden_microniches.py")
_bl11 = importlib.util.module_from_spec(_spec11)
_spec11.loader.exec_module(_bl11)

_spec05 = importlib.util.spec_from_file_location("fig05", HERE / "05_final_report_figures.py")
_fig05 = importlib.util.module_from_spec(_spec05)
_spec05.loader.exec_module(_fig05)
dense = _fig05.dense
GENE_SETS = _fig05.GENE_SETS

DEFAULT_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]
PX_PER_UM = 1.0 / 0.273

SPATIAL_CASES = [
    # (perturb, source_ct, neighbor_ct, k_neighbors, max_dist_um)
    ("Il4ra", "immune", "immune", 25, 80),
    ("Il4ra", "immune", "myeloid", 25, 100),
    ("Cd83", "immune", "immune", 25, 80),
    ("Cd83", "immune", "myeloid", 25, 100),
    ("Cd74", "myeloid", "immune", 25, 100),
    ("Cd74", "myeloid", "myeloid", 25, 80),
    ("Cks1b", "immune", "myeloid", 25, 100),
]

BETA_CASES = [
    ("Il4ra", "immune"),
    ("Cd83", "immune"),
    ("Cd83", "fibroblast"),
    ("Cd74", "immune"),
    ("Cd74", "myeloid"),
    ("Cks1b", "myeloid"),
]

HIGHLIGHT = {
    "Il4ra": ["H2-Aa", "H2-Ab1", "Cd74", "Cxcl9", "Cxcl10", "Stat1", "Ifng", "Il4ra"],
    "Cd83": ["Cd83", "H2-Aa", "H2-Ab1", "Cd74", "Cd86", "Icosl", "Cxcl10"],
    "Cd74": ["Cd74", "H2-Aa", "H2-Ab1", "Ciita", "Cxcl9", "Ifng"],
    "Cks1b": ["Cks1b", "Myc", "Ccnb1", "Pcna"],
}


def max_dist_px(um: float) -> float:
    return um * PX_PER_UM


def spatial_experimental_de(
    pool: sc.AnnData,
    perturb: str,
    neighbor_ct: str,
    k_neighbors: int = 25,
    source_cell_type: str | None = None,
    niche_mode: str = "ntc_near_far",
) -> pd.DataFrame:
    if niche_mode == "ntc_near_far":
        return ndu.spatial_ntc_niche_pseudobulk(
            pool, perturb, k_neighbors=k_neighbors, cell_type=neighbor_ct,
            source_cell_type=source_cell_type,
        )
    return ndu.spatial_neighbor_pseudobulk(
        pool, perturb, k_neighbors=k_neighbors, cell_type=neighbor_ct,
        source_cell_type=source_cell_type, restrict_to_ntc=(niche_mode == "ntc_source_knn"),
    )


def spatial_predicted_de(
    pool: sc.AnnData,
    baseline: sc.AnnData,
    pred: pd.DataFrame,
    perturb: str,
    neighbor_ct: str,
    slice_id: str,
    k_neighbors: int = 25,
    source_cell_type: str | None = None,
    niche_mode: str = "ntc_near_far",
    pred_mode: str = "autonomous",
) -> pd.DataFrame:
    """Predicted Δ matched to experimental spatial contrast.

    pred_mode='autonomous' (default): global in-silico KO Δ on NTC neighbor cells.
    SpaceTravLR assigns the same prediction to every NTC cell of a type, so a
    near−far predicted contrast is ~0 by construction. We instead compare the
    observed spatial near−far vector to the autonomous predicted program.

    pred_mode='niche_contrast': legacy near−far predicted contrast (usually ~0).
    """
    genes = sorted(set(pool.var_names) & set(pred.columns))
    if pred_mode == "niche_contrast":
        try:
            if niche_mode == "ntc_near_far":
                idx_p, idx_c = ndu.spatial_ntc_niche_indices(
                    pool, perturb, k_neighbors=k_neighbors, cell_type=neighbor_ct,
                    source_cell_type=source_cell_type,
                )
            else:
                idx_p, idx_c = ndu.spatial_neighbor_indices(
                    pool, perturb, k_neighbors=k_neighbors, cell_type=neighbor_ct,
                    source_cell_type=source_cell_type,
                    restrict_to_ntc=(niche_mode == "ntc_source_knn"),
                )
        except ValueError:
            return pd.DataFrame(columns=["gene", "log2fc"])
        if len(idx_p) < 3 or len(idx_c) < 3:
            return pd.DataFrame(columns=["gene", "log2fc"])
        sub_p, sub_c = pool[idx_p], pool[idx_c]

        def niche_pred_delta(sub_pool: sc.AnnData) -> pd.Series:
            common = [g for g in genes if g in sub_pool.var_names and g in pred.columns]
            if len(common) < 10:
                return pd.Series(dtype=float)
            pred_aligned, ok = ndu.align_pool_pred(sub_pool, pred, slice_id, genes=common)
            if ok.sum() < 3:
                return pd.Series(dtype=float)
            sub = sub_pool[ok.values]
            expr = dense(sub, common)
            pr = pred_aligned.loc[sub.obs_names, common]
            return ndu.log1p_pred_minus_expr(pr, expr).mean(0)

        d_p, d_c = niche_pred_delta(sub_p), niche_pred_delta(sub_c)
        if d_p.empty or d_c.empty:
            return pd.DataFrame(columns=["gene", "log2fc"])
        common = d_p.index.intersection(d_c.index)
        pred_d = pd.DataFrame({"gene": common, "log2fc": (d_p.loc[common] - d_c.loc[common]).values})
        pred_d["abs_log2fc"] = pred_d.log2fc.abs()
        return pred_d

    ntc = pool[(pool.obs["target_gene"].astype(str) == "non-targeting") &
               (pool.obs["cell_type"].astype(str) == neighbor_ct)]
    if ntc.n_obs < 10:
        return pd.DataFrame(columns=["gene", "log2fc"])
    common = [g for g in genes if g in ntc.var_names and g in pred.columns]
    if len(common) < 10:
        return pd.DataFrame(columns=["gene", "log2fc"])
    pred_aligned, ok = ndu.align_pool_pred(ntc, pred, slice_id, genes=common)
    if ok.sum() < 10:
        return pd.DataFrame(columns=["gene", "log2fc"])
    sub = ntc[ok.values]
    expr = dense(sub, common)
    pr = pred_aligned.loc[sub.obs_names, common]
    delta = ndu.log1p_pred_minus_expr(pr, expr).mean(0)
    pred_d = pd.DataFrame({"gene": common, "log2fc": delta.values, "abs_log2fc": np.abs(delta.values)})
    return pred_d


def pseudobulk_from_pool(a: sc.AnnData, b: sc.AnnData, genes: list[str]) -> pd.DataFrame:
    return ndu.pseudobulk_delta_df(a, b, genes)


def predicted_delta_cells(sub: sc.AnnData, pred: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    sub = sub[sub.obs_names.isin(pred.index)]
    common = [g for g in genes if g in sub.var_names and g in pred.columns]
    expr = dense(sub, common)
    pred_sub = pred.loc[sub.obs_names, common]
    delta = (pred_sub - expr).mean(0)
    return pd.DataFrame({"gene": common, "log2fc": delta.values, "abs_log2fc": np.abs(delta.values)})


def beta_leiden_experimental_de(
    pool: sc.AnnData,
    perturb: str,
    cell_type: str,
    niche: str,
) -> pd.DataFrame:
    sub = pool[(pool.obs["cell_type"].astype(str) == cell_type) &
               (pool.obs["beta_leiden"].astype(str) == niche)].copy()
    sub.obs["_cond"] = np.where(
        sub.obs["target_gene"].astype(str) == perturb, "pert",
        np.where(sub.obs["target_gene"].astype(str) == "non-targeting", "ntc", "other"),
    )
    sub = sub[sub.obs["_cond"].isin(["pert", "ntc"])].copy()
    if (sub.obs["_cond"] == "pert").sum() < 10 or (sub.obs["_cond"] == "ntc").sum() < 6:
        raise ValueError("Insufficient cells in niche")
    sc.tl.rank_genes_groups(sub, groupby="_cond", groups=["pert"], reference="ntc", method="wilcoxon")
    df = sc.get.rank_genes_groups_df(sub, group="pert")
    df = df.rename(columns={"names": "gene", "logfoldchanges": "log2fc", "pvals": "pval", "pvals_adj": "pval_adj"})
    df["abs_log2fc"] = df.log2fc.abs()
    return df


def beta_leiden_predicted_de(
    baseline: sc.AnnData,
    pred: pd.DataFrame,
    cell_type: str,
    niche: str,
    genes: list[str],
) -> pd.DataFrame:
    sub = baseline[(baseline.obs["cell_type"].astype(str) == cell_type) &
                   (baseline.obs["beta_leiden"].astype(str) == niche)]
    sub = sub[sub.obs_names.isin(pred.index)]
    if sub.n_obs < 6:
        return pd.DataFrame(columns=["gene", "log2fc"])
    common = [g for g in genes if g in sub.var_names and g in pred.columns]
    if not common:
        return pd.DataFrame(columns=["gene", "log2fc"])
    expr_raw = dense(sub, common)
    expr_log = np.log1p(np.maximum(expr_raw.to_numpy(dtype=float), 0.0))
    expr_df = pd.DataFrame(expr_log, index=sub.obs_names, columns=common)
    pred_sub = pred.loc[sub.obs_names, common]
    delta = ndu.log1p_pred_minus_expr(pred_sub, expr_df).mean(0)
    return pd.DataFrame({"gene": common, "log2fc": delta.values, "abs_log2fc": np.abs(delta.values)})


def ccc_state_analysis(
    pool: sc.AnnData,
    baseline: sc.AnnData,
    pred: pd.DataFrame,
    perturb: str,
    neighbor_ct: str,
    k_neighbors: int = 25,
    source_cell_type: str | None = None,
) -> list[dict]:
    """Compare CCC/T-cell pathway scores: NTC near sgP vs far NTC (obs + pred)."""
    rows = []
    sl = str(pool.obs["slice_id"].iloc[0])
    src = source_cell_type or neighbor_ct
    try:
        near_idx, far_idx = ndu.spatial_ntc_niche_indices(
            pool, perturb, k_neighbors=k_neighbors, cell_type=neighbor_ct,
            source_cell_type=src,
        )
    except ValueError:
        return rows
    if len(near_idx) < 5 or len(far_idx) < 5:
        return rows

    pool_near, pool_far = pool[near_idx], pool[far_idx]
    all_pathways = {**ndu.CCC_PATHWAYS, **ndu.TCELL_STATE}

    for pname, genes in all_pathways.items():
        obs_delta = ndu.module_score_delta(pool_near, pool_far, genes)
        if not np.isfinite(obs_delta):
            continue
        near_sc = ndu.module_score(pool_near, genes, pname)
        far_sc = ndu.module_score(pool_far, genes, pname)
        try:
            _, pval = stats.mannwhitneyu(
                near_sc[~np.isnan(near_sc)], far_sc[~np.isnan(far_sc)], alternative="two-sided",
            )
        except ValueError:
            pval = float("nan")

        pred_delta = ndu.predicted_module_score_autonomous(pool, neighbor_ct, pred, sl, genes)

        rows.append({
            "pathway": pname,
            "perturbation": perturb,
            "neighbor_cell_type": neighbor_ct,
            "source_cell_type": src,
            "obs_neighbor_delta": obs_delta,
            "pred_delta_on_neighbors": pred_delta,
            "obs_pval": float(pval),
            "n_near_ntc": int(len(near_idx)),
            "n_far_ntc": int(len(far_idx)),
        })
    return rows


def run_spatial_analysis(
    slices: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    pred_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    tag: str,
) -> pd.DataFrame:
    stats_rows = []
    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    axes = axes.flatten()
    plot_i = 0

    for perturb, source_ct, neighbor_ct, k, max_um in SPATIAL_CASES:
        exp_dfs, pred_dfs = [], []
        for sl in slices:
            pool = load_pool(sl, data_root)
            pool.obs["slice_id"] = sl
            sc.pp.normalize_total(pool, target_sum=10000)
            sc.pp.log1p(pool)
            pred = ndu.load_pred_feather(pred_dir / f"predicted_KO_{perturb}.feather")
            try:
                exp = spatial_experimental_de(
                    pool, perturb, neighbor_ct, k_neighbors=k, source_cell_type=source_ct,
                )
                pred_df = spatial_predicted_de(
                    pool, baseline, pred, perturb, neighbor_ct, sl,
                    k_neighbors=k, source_cell_type=source_ct,
                )
                exp_dfs.append(exp)
                pred_dfs.append(pred_df)
            except (ValueError, KeyError) as e:
                print(f"  skip {sl} {perturb}/{neighbor_ct}: {e}")
                continue

        if not exp_dfs:
            continue
        exp_all = pd.concat(exp_dfs).groupby("gene", as_index=False).agg(
            log2fc=("log2fc", "mean"), abs_log2fc=("abs_log2fc", "mean"),
        )
        if "pval_adj" in pd.concat(exp_dfs).columns:
            pvals = pd.concat(exp_dfs).groupby("gene", as_index=False).agg(pval_adj=("pval_adj", "min"))
            exp_all = exp_all.merge(pvals, on="gene", how="left")
        pred_all = pd.concat(pred_dfs).groupby("gene", as_index=False).agg(log2fc=("log2fc", "mean"))

        save = fig_dir / f"spatial_{perturb}_{source_ct}_to_{neighbor_ct}_{tag}.png"
        ax = axes[plot_i] if plot_i < len(axes) else None
        st, _ = ndu.plot_gene_comparison_advanced(
            exp_all, pred_all,
            label1="SPAC-seq neighbors", label2="SpaceTravLR",
            highlight_genes=HIGHLIGHT.get(perturb, []),
            top_n_labels=0, target_ko=perturb,
            neighbor_ct=neighbor_ct, source_ct=f"sg{perturb} {source_ct}",
            axis_lim=1.5, save_path=str(save) if ax is None else None,
            show=False, ax=ax, use_size=False,
            title_suffix=f"pooled {len(slices)} slices",
        )
        st.update({
            "perturbation": perturb, "source_cell_type": source_ct,
            "neighbor_cell_type": neighbor_ct, "analysis": "spatial_knn",
            "tag": tag, "n_slices": len(exp_dfs),
        })
        stats_rows.append(st)
        plot_i += 1

    for j in range(plot_i, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "Spatial neighbor DEG concordance: SPAC-seq vs SpaceTravLR\n"
        "(bystander cells near sgP vs NTC sources; pooled across slices)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig6_spatial_neighbor_grid_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def run_beta_leiden_deg(
    slices: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    betadata_dir: Path,
    pred_dir: Path,
    fig_dir: Path,
    tag: str,
    leiden_resolution: float = 0.6,
    leiden_spatial_weight: float = 0.35,
) -> pd.DataFrame:
    stats_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    plot_i = 0

    for perturb, cell_type in BETA_CASES:
        exp_agg, pred_agg = [], []
        for sl in slices:
            pool = assign_labeled_pool(
                sl, data_root, betadata_dir, [cell_type],
                resolution=leiden_resolution, spatial_weight=leiden_spatial_weight,
            )
            pred = ndu.load_pred_feather(pred_dir / f"predicted_KO_{perturb}.feather")
            base_sl = baseline[baseline.obs["slice_id"].astype(str) == sl].copy()
            bl = _bl11.baseline_labels_from_pool(base_sl, pool, pool.obs["beta_leiden"])
            base_sl.obs["beta_leiden"] = base_sl.obs_names.map(bl).astype(str)

            niches = pool.obs["beta_leiden"].astype(str)
            valid = niches[(niches.str.contains(sl)) & (niches != "unassigned")].unique()
            for niche in valid:
                n_pert = ((pool.obs["beta_leiden"].astype(str) == niche) &
                          (pool.obs["target_gene"].astype(str) == perturb)).sum()
                n_ntc = ((pool.obs["beta_leiden"].astype(str) == niche) &
                         (pool.obs["target_gene"].astype(str) == "non-targeting")).sum()
                if n_pert < 10 or n_ntc < 6:
                    continue
                try:
                    exp = beta_leiden_experimental_de(pool, perturb, cell_type, niche)
                    genes = sorted(set(base_sl.var_names) & set(pred.columns))
                    pr = beta_leiden_predicted_de(base_sl, pred, cell_type, niche, genes)
                    exp["niche"] = niche
                    pr["niche"] = niche
                    exp_agg.append(exp)
                    pred_agg.append(pr)
                except ValueError:
                    continue

        if not exp_agg:
            continue
        exp_all = pd.concat(exp_agg).groupby("gene", as_index=False).agg(
            log2fc=("log2fc", "mean"), pval_adj=("pval_adj", "min"),
        )
        pred_all = pd.concat(pred_agg).groupby("gene", as_index=False).agg(log2fc=("log2fc", "mean"))

        ax = axes[plot_i] if plot_i < len(axes) else None
        st, _ = ndu.plot_gene_comparison_advanced(
            exp_all, pred_all,
            label1="SPAC-seq β-Leiden", label2="SpaceTravLR",
            highlight_genes=HIGHLIGHT.get(perturb, []),
            top_n_labels=12, target_ko=perturb,
            neighbor_ct=f"β-Leiden {cell_type}", source_ct=f"sg{perturb}",
            axis_lim=1.5, ax=ax, show=False, use_size=True,
            title_suffix=f"{len(exp_agg)} niches pooled",
        )
        st.update({"perturbation": perturb, "cell_type": cell_type, "analysis": "beta_leiden", "n_niches": len(exp_agg)})
        stats_rows.append(st)
        plot_i += 1

    for j in range(plot_i, len(axes)):
        axes[j].axis("off")
    fig.suptitle("β-Leiden microniche DEG concordance (SPAC-seq vs SpaceTravLR)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig7_beta_leiden_deg_grid_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def assign_labeled_pool(
    sl: str,
    data_root: Path,
    betadata_dir: Path,
    cell_types: list[str],
    n_pcs: int = 15,
    resolution: float = 0.6,
    spatial_weight: float = 0.35,
) -> sc.AnnData:
    pool = load_pool(sl, data_root)
    pool.obs["slice_id"] = sl
    sc.pp.normalize_total(pool, target_sum=10000)
    sc.pp.log1p(pool)
    _bl11.ensure_cluster_id(pool)
    pool_beta, _ = _bl11.build_beta_score_matrix(pool, betadata_dir)
    ntc = pool[pool.obs["target_gene"].astype(str) == "non-targeting"]
    ntc_beta = pool_beta[pool.obs["target_gene"].astype(str) == "non-targeting"]
    labels = _bl11.assign_compartment_microniches(
        ntc, ntc_beta, [sl], cell_types, n_pcs=n_pcs, resolution=resolution, spatial_weight=spatial_weight,
    )
    full = pd.Series("unassigned", index=pool.obs_names, dtype=str)
    full.loc[ntc.obs_names] = labels.loc[ntc.obs_names]
    for ct in cell_types:
        full = _bl11.knn_assign_perturbed(pool, full, ct)
    pool.obs["beta_leiden"] = full.values
    return pool


def run_ccc_analysis(
    slices: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    pred_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    tag: str,
) -> pd.DataFrame:
    rows = []
    ccc_cases = [
        ("Il4ra", "immune", "immune"),
        ("Il4ra", "immune", "myeloid"),
        ("Cd83", "immune", "immune"),
        ("Cd74", "myeloid", "immune"),
    ]
    for sl in slices:
        pool = load_pool(sl, data_root)
        pool.obs["slice_id"] = sl
        sc.pp.normalize_total(pool, target_sum=10000)
        sc.pp.log1p(pool)
        base_sl = baseline[baseline.obs["slice_id"].astype(str) == sl]
        for perturb, src, neighbor in ccc_cases:
            pred = ndu.load_pred_feather(pred_dir / f"predicted_KO_{perturb}.feather")
            for r in ccc_state_analysis(pool, base_sl, pred, perturb, neighbor, source_cell_type=src):
                r.update({"slice": sl, "source_cell_type": src})
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"ccc_state_scores_{tag}.csv", index=False)

    if df.empty:
        return df

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    show = df.groupby(["perturbation", "pathway", "neighbor_cell_type"], as_index=False).agg(
        obs_neighbor_delta=("obs_neighbor_delta", "mean"),
        pred_delta_on_neighbors=("pred_delta_on_neighbors", "mean"),
        obs_pval=("obs_pval", "median"),
    )
    show["pathway_short"] = show.pathway.str.replace(" / ", "\n")
    pivot_obs = show.pivot_table(index="pathway_short", columns=["perturbation", "neighbor_cell_type"],
                                  values="obs_neighbor_delta")
    sns.heatmap(pivot_obs, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Observed Δ score\n(pert vs ctrl neighbors)"})
    ax.set_title("A  Observed CCC / T-cell state shift\nin spatial neighbors of sgP sources", fontweight="bold", loc="left")
    ax.tick_params(axis="both", labelsize=7)

    ax = axes[1]
    for perturb, grp in show.groupby("perturbation"):
        ax.scatter(grp.obs_neighbor_delta, grp.pred_delta_on_neighbors, s=50, alpha=0.7, label=f"sg{perturb}")
    lim = max(show[["obs_neighbor_delta", "pred_delta_on_neighbors"]].abs().max().max(), 0.05)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    if len(show) >= 3:
        valid = show.dropna(subset=["obs_neighbor_delta", "pred_delta_on_neighbors"])
        if len(valid) >= 3:
            r, p = stats.pearsonr(valid.obs_neighbor_delta, valid.pred_delta_on_neighbors)
            ax.text(0.05, 0.95, f"Pearson r = {r:+.2f}\np = {p:.3f}\nn = {len(valid)}", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("Observed pathway Δ (spatial neighbors)")
    ax.set_ylabel("SpaceTravLR predicted Δ\n(on matched NTC neighbors)")
    ax.set_title("B  CCC / immune-state concordance", fontweight="bold", loc="left")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Cell-cell communication & T-cell state influence in spatial CRISPR niches\n"
        "(SPAC-seq paper theme: local functional effects on bystander immune cells)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig8_ccc_tcell_state_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    mhc = show[show.pathway.str.contains("MHC|T cell|checkpoint|Il4", case=False, regex=True)]
    if not mhc.empty:
        x = np.arange(len(mhc))
        w = 0.35
        ax.bar(x - w / 2, mhc.obs_neighbor_delta, width=w, label="Observed (SPAC-seq)", color="#2166ac")
        ax.bar(x + w / 2, mhc.pred_delta_on_neighbors, width=w, label="Predicted (SpaceTravLR)", color="#67a9cf")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"sg{r.perturbation}\n{r.pathway[:22]}\n→{r.neighbor_cell_type}" for _, r in mhc.iterrows()],
            fontsize=7, rotation=30, ha="right",
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Pathway module score Δ")
        ax.legend(fontsize=8)
        ax.set_title("MHC / T-cell / Il4 pathway shifts in spatial neighbors", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig9_pathway_concordance_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_prep/baseline_ntc_0c6fbac5e6cd947c_fullprep.h5ad")
    ap.add_argument("--betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_seed")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_pooled")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/niche_deg")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/niche_deg")
    ap.add_argument("--tag", default="pooled")
    ap.add_argument("--skip-beta-leiden", action="store_true")
    ap.add_argument("--leiden-resolution", type=float, default=0.6)
    ap.add_argument("--leiden-spatial-weight", type=float, default=0.35)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    baseline = load_baseline(args.baseline_h5ad)

    print("Spatial neighbor DEG analysis…")
    spatial_stats = run_spatial_analysis(
        args.slices, args.data_root, baseline, args.pred_dir, args.out_dir, args.fig_dir, args.tag,
    )
    spatial_stats.to_csv(args.out_dir / f"spatial_neighbor_stats_{args.tag}.csv", index=False)

    print("β-Leiden niche DEG analysis…")
    if args.skip_beta_leiden:
        beta_stats = pd.DataFrame()
    else:
        beta_stats = run_beta_leiden_deg(
            args.slices, args.data_root, baseline, args.betadata_dir, args.pred_dir, args.fig_dir, args.tag,
            leiden_resolution=args.leiden_resolution, leiden_spatial_weight=args.leiden_spatial_weight,
        )
        beta_stats.to_csv(args.out_dir / f"beta_leiden_deg_stats_{args.tag}.csv", index=False)

    print("CCC / T-cell state analysis…")
    ccc_df = run_ccc_analysis(
        args.slices, args.data_root, baseline, args.pred_dir, args.out_dir, args.fig_dir, args.tag,
    )

    overall = {
        "tag": args.tag,
        "spatial_cases": len(spatial_stats),
        "spatial_median_pearson": float(spatial_stats.pearson_r.median()) if len(spatial_stats) else None,
        "beta_leiden_cases": len(beta_stats),
        "beta_median_pearson": float(beta_stats.pearson_r.median()) if len(beta_stats) else None,
        "ccc_pathways": int(len(ccc_df)),
        "ccc_median_pearson": float(
            ccc_df.dropna(subset=["obs_neighbor_delta", "pred_delta_on_neighbors"])
            .groupby(["perturbation", "pathway", "neighbor_cell_type"], as_index=False)
            .agg(obs=("obs_neighbor_delta", "mean"), pred=("pred_delta_on_neighbors", "mean"))
            .pipe(lambda d: stats.pearsonr(d.obs, d.pred)[0] if len(d) >= 3 else float("nan"))
        ) if not ccc_df.empty else None,
        "leiden_resolution": args.leiden_resolution,
        "leiden_spatial_weight": args.leiden_spatial_weight,
        "best_spatial": spatial_stats.nlargest(3, "pearson_r").to_dict("records") if len(spatial_stats) else [],
        "best_beta": beta_stats.nlargest(3, "pearson_r").to_dict("records") if len(beta_stats) else [],
    }
    (args.out_dir / f"overall_{args.tag}.json").write_text(json.dumps(overall, indent=2, default=str))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
