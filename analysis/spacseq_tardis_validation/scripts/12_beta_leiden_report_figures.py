#!/usr/bin/env python3
"""Publication-quality figures for beta-Leiden microniche validation.

Panels cover:
  1. Method comparison (β-Leiden vs graphclust concordance)
  2. Functional distinctness (pathway separation across niches)
  3. Spotlight: sgIl4ra / immune — spatial niches + pred vs obs scatter
  4. Concordance grid for headline perturbation × cell-type pairs
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec05 = importlib.util.spec_from_file_location("fig05", HERE / "05_final_report_figures.py")
_fig05 = importlib.util.module_from_spec(_spec05)
_spec05.loader.exec_module(_fig05)
dense = _fig05.dense
GENE_SETS = _fig05.GENE_SETS

_spec09 = importlib.util.spec_from_file_location("sp09", HERE / "09_spatial_validation.py")
_sp09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_sp09)
load_baseline = _sp09.load_baseline
load_pool = _sp09.load_pool
attach_graphclust = _sp09.attach_graphclust

_spec11 = importlib.util.spec_from_file_location("bl11", HERE / "11_beta_leiden_microniches.py")
_bl11 = importlib.util.module_from_spec(_spec11)
_spec11.loader.exec_module(_bl11)

HEADLINE = [
    ("Il4ra", "immune"),
    ("Cd83", "immune"),
    ("Cd83", "fibroblast"),
    ("Cks1b", "immune"),
    ("Cks1b", "myeloid"),
    ("Cd74", "immune"),
]
NICHE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a",
]


def niche_deltas(
    baseline: sc.AnnData,
    pool: sc.AnnData,
    pred: pd.DataFrame,
    perturb: str,
    cell_type: str,
    niche: str,
    niche_key: str = "beta_leiden",
) -> tuple[pd.Series, pd.Series] | None:
    genes = sorted(set(baseline.var_names) & set(pool.var_names) & set(pred.columns))
    pool_ct = pool[pool.obs["cell_type"].astype(str) == cell_type]
    common = [g for g in genes if g in pool_ct.var_names]
    if len(common) < 30:
        return None
    pool_expr = dense(pool_ct, common)
    ntc = pool_ct.obs["target_gene"].astype(str) == "non-targeting"
    pert = pool_ct.obs["target_gene"].astype(str) == perturb
    nic = pool_ct.obs[niche_key].astype(str) == niche
    if (ntc & nic).sum() < 6 or (pert & nic).sum() < 10:
        return None

    obs_d = pool_expr.loc[pert & nic].mean(0) - pool_expr.loc[ntc & nic].mean(0)

    base_sub = baseline[baseline.obs["cell_type"].astype(str) == cell_type]
    if "slice_id" in base_sub.obs.columns and "slice_id" in pool.obs.columns:
        sl = str(pool.obs["slice_id"].iloc[0])
        base_sub = base_sub[base_sub.obs["slice_id"].astype(str) == sl]
    base_sub = base_sub[base_sub.obs_names.isin(pred.index)].copy()
    if base_sub.n_obs == 0:
        return None
    if niche_key not in base_sub.obs.columns:
        bl = _bl11.baseline_labels_from_pool(baseline, pool, pool.obs[niche_key])
        base_sub.obs[niche_key] = base_sub.obs_names.map(bl).astype(str)
    pred_sub = pred.loc[base_sub.obs_names, common]
    base_expr = dense(base_sub, common)
    pred_delta = pred_sub - base_expr
    base_n = base_sub.obs[niche_key].astype(str) == niche
    if base_n.sum() < 6:
        return None
    pred_d = pred_delta.loc[base_n].mean(0)
    return pred_d, obs_d


def assign_pool_labels(
    slice_name: str,
    data_root: Path,
    betadata_dir: Path,
    cell_types: list[str],
) -> sc.AnnData:
    pool = load_pool(slice_name, data_root)
    pool.obs["slice_id"] = slice_name
    _bl11.ensure_cluster_id(pool)
    pool_beta, _ = _bl11.build_beta_score_matrix(pool, betadata_dir)
    ntc_mask = pool.obs["target_gene"].astype(str) == "non-targeting"
    ntc = pool[ntc_mask]
    ntc_beta = pool_beta[ntc_mask.values]
    labels = _bl11.assign_compartment_microniches(
        ntc, ntc_beta, [slice_name], cell_types,
        n_pcs=15, resolution=0.6, spatial_weight=0.35,
    )
    full = pd.Series("unassigned", index=pool.obs_names, dtype=str)
    full.loc[ntc.obs_names] = labels.loc[ntc.obs_names]
    for ct in cell_types:
        full = _bl11.knn_assign_perturbed(pool, full, ct)
    pool.obs["beta_leiden"] = full.values
    _, pool = attach_graphclust(pool, pool, data_root)
    return pool


def fig_main_overview(
    summary: pd.DataFrame,
    niche_corr: pd.DataFrame,
    pathway: pd.DataFrame,
    fig_dir: Path,
    tag: str,
) -> None:
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    cmp = summary.pivot_table(
        index=["perturbation", "cell_type"], columns="niche_type", values="median_pearson_r"
    ).dropna(how="any")
    cmp["delta"] = cmp["beta_leiden"] - cmp["graphclust"]
    cmp = cmp.sort_values("delta", ascending=True)
    labels = [f"sg{p} | {c}" for p, c in cmp.index]

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(cmp))
    ax.barh(y - 0.18, cmp["graphclust"], height=0.35, color="#bdbdbd", label="graphclust")
    ax.barh(y + 0.18, cmp["beta_leiden"], height=0.35, color="#2166ac", label="β-Leiden")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Median Pearson r (pred vs obs Δ)")
    ax.set_title("A  Microniche concordance by method", loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

    ax = fig.add_subplot(gs[0, 1])
    bl = niche_corr[niche_corr.niche_type == "beta_leiden"]
    gc = niche_corr[niche_corr.niche_type == "graphclust"]
    parts = ax.violinplot(
        [gc.pearson_r.values, bl.pearson_r.values],
        positions=[0, 1],
        showmeans=True,
        showextrema=False,
    )
    for i, c in enumerate(["#bdbdbd", "#2166ac"]):
        parts["bodies"][i].set_facecolor(c)
        parts["bodies"][i].set_alpha(0.75)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["graphclust\n(n={})".format(len(gc)), "β-Leiden\n(n={})".format(len(bl))])
    ax.set_ylabel("Per-niche Pearson r")
    med_gc, med_bl = gc.pearson_r.median(), bl.pearson_r.median()
    ax.set_title(
        f"B  Distribution of niche-level r\n(median {med_gc:+.3f} → {med_bl:+.3f})",
        loc="left", fontweight="bold",
    )

    ax = fig.add_subplot(gs[0, 2])
    pw = pathway.copy()
    pw["niche_type"] = pw["niche_type"].fillna("beta_leiden")
    agg = pw.groupby(["pathway", "niche_type"]).kruskal_p.median().unstack()
    if "beta_leiden" in agg.columns and "graphclust" in agg.columns:
        plot_df = -np.log10(agg[["beta_leiden", "graphclust"]].clip(lower=1e-50))
        plot_df.plot(kind="barh", ax=ax, color=["#2166ac", "#bdbdbd"], width=0.75)
        ax.set_xlabel("−log₁₀ Kruskal–Wallis p (pathway score vs niche)")
        ax.set_title("C  Functional niche separation", loc="left", fontweight="bold")
        ax.legend(["β-Leiden", "graphclust"], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

    ax = fig.add_subplot(gs[1, 0])
    show = summary[summary.niche_type == "beta_leiden"].copy()
    show["label"] = show.apply(lambda r: f"sg{r.perturbation}\n{r.cell_type}", axis=1)
    colors = {"immune": "#d62728", "myeloid": "#ff7f0e", "fibroblast": "#2ca02c"}
    for _, row in show.iterrows():
        ax.scatter(row.frac_pos_r, row.median_pearson_r, s=60 + row.n_niche_rows * 3,
                   c=colors.get(row.cell_type, "#333"), edgecolors="k", lw=0.4, alpha=0.85)
        if row.perturbation in ("Il4ra", "Cd83") and row.cell_type in ("immune", "fibroblast"):
            ax.annotate(f"sg{row.perturbation}/{row.cell_type}", (row.frac_pos_r, row.median_pearson_r),
                        fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Fraction niches with r > 0")
    ax.set_ylabel("Median Pearson r")
    ax.set_title("D  β-Leiden reliability", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 1:])
    pivot = summary.pivot_table(
        index="perturbation", columns=["niche_type", "cell_type"], values="median_pearson_r"
    )
    cols = [(nt, ct) for nt in ("beta_leiden", "graphclust") for ct in ("immune", "myeloid", "fibroblast")]
    cols = [c for c in cols if c in pivot.columns]
    mat = pivot[cols].values.astype(float)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.05, vmax=0.18, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{nt[:4]}\n{ct[:4]}" for nt, ct in cols], fontsize=7, rotation=0)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"sg{p}" for p in pivot.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.09 else "black")
    ax.set_title("E  Concordance heatmap (β-Leiden vs graphclust × compartment)", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Median r")

    fig.suptitle(f"SpaceTravLR β-Leiden microniches vs SPAC-seq CRISPR outcomes ({tag})", fontsize=12, y=1.01)
    fig.savefig(fig_dir / f"fig1_main_overview_{tag}.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig1_main_overview_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_spotlight_il4ra(
    pool: sc.AnnData,
    baseline: sc.AnnData,
    pred_dir: Path,
    data_root: Path,
    betadata_dir: Path,
    niche_corr: pd.DataFrame,
    fig_dir: Path,
    tag: str,
    slice_name: str = "subQ-1",
) -> None:
    perturb = "Il4ra"
    cell_type = "immune"
    pred = pd.read_feather(pred_dir / f"predicted_KO_{perturb}.feather").set_index("CellID")

    ct = pool[(pool.obs["cell_type"].astype(str) == cell_type)].copy()
    ntc = ct[ct.obs["target_gene"].astype(str) == "non-targeting"]
    pert = ct[ct.obs["target_gene"].astype(str) == perturb]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    niches = ct.obs["beta_leiden"].astype(str)
    uniq = sorted(n for n in niches.unique() if n not in ("unassigned", "nan"))
    for i, lab in enumerate(uniq):
        m = niches == lab
        short = lab.split("|")[-1]
        ax.scatter(
            ct.obsm["spatial"][m, 0], ct.obsm["spatial"][m, 1],
            c=NICHE_COLORS[i % len(NICHE_COLORS)], s=1.5, alpha=0.7, rasterized=True, label=f"N{short}",
        )
    ax.scatter(pert.obsm["spatial"][:, 0], pert.obsm["spatial"][:, 1],
               facecolors="none", edgecolors="#111", s=8, lw=0.3, alpha=0.5, label="sgIl4ra")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"A  β-Leiden niches + sgIl4ra cells\n{slice_name} immune", loc="left", fontweight="bold")
    ax.legend(fontsize=5, loc="upper right", markerscale=2, framealpha=0.8)

    ax = fig.add_subplot(gs[0, 1])
    gc = ct.obs["graphclust"].astype(str)
    for i, lab in enumerate(sorted(gc.unique())):
        m = gc == lab
        ax.scatter(ct.obsm["spatial"][m, 0], ct.obsm["spatial"][m, 1],
                   c=NICHE_COLORS[i % len(NICHE_COLORS)], s=1.5, alpha=0.7, rasterized=True)
    ax.scatter(pert.obsm["spatial"][:, 0], pert.obsm["spatial"][:, 1],
               facecolors="none", edgecolors="#111", s=8, lw=0.3, alpha=0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("B  graphclust (morphology) reference", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[0, 2])
    show_g = "H2-Ab1" if "H2-Ab1" in pred.columns else "Cd74"
    ref = float(dense(ntc, [show_g]).mean().iloc[0]) if show_g in ntc.var_names else 0.0
    obs_delta = dense(pert, [show_g])[show_g] - ref if show_g in pert.var_names else pd.Series(0, index=pert.obs_names)
    sl_base = baseline[(baseline.obs["slice_id"].astype(str) == slice_name) &
                       (baseline.obs["cell_type"].astype(str) == cell_type)]
    sl_base = sl_base[sl_base.obs_names.isin(pred.index)]
    pred_delta = pred.loc[sl_base.obs_names, show_g] - dense(sl_base, [show_g])[show_g].values
    vmax = np.nanpercentile(np.abs(np.concatenate([obs_delta.values, pred_delta.values])), 95)
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    ax.scatter(sl_base.obsm["spatial"][:, 0], sl_base.obsm["spatial"][:, 1],
               c=pred_delta, s=2, cmap="RdBu_r", norm=norm, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"C  Predicted Δ {show_g} (in-silico KO)", loc="left", fontweight="bold")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    fig.colorbar(sm, ax=ax, shrink=0.6, label=f"Δ {show_g}")

    ax = fig.add_subplot(gs[1, 0])
    sub = niche_corr[
        (niche_corr.niche_type == "beta_leiden") &
        (niche_corr.perturbation == perturb) &
        (niche_corr.cell_type == cell_type)
    ].sort_values("pearson_r", ascending=True)
    y = np.arange(len(sub))
    colors = ["#2166ac" if p < 0.05 else "#92c5de" for p in sub.pearson_perm_p]
    ax.barh(y, sub.pearson_r, color=colors, edgecolor="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r.slice} N{r.niche.split('|')[-1]} (n={r.n_pert})" for _, r in sub.iterrows()], fontsize=7
    )
    ax.set_xlabel("Pearson r (pred vs obs Δ)")
    ax.set_title("D  Per-niche concordance (4 slices)", loc="left", fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#2166ac", label="perm p < 0.05"),
        mpatches.Patch(color="#92c5de", label="n.s."),
    ], fontsize=6)

    best = sub.iloc[-1]
    pair = niche_deltas(baseline, pool, pred, perturb, cell_type, best.niche)
    ax = fig.add_subplot(gs[1, 1])
    if pair:
        pred_d, obs_d = pair
        panel = [g for g in pred_d.index if g != perturb and np.isfinite(pred_d[g]) and np.isfinite(obs_d[g])]
        pv, ov = pred_d.loc[panel].values, obs_d.loc[panel].values
        ax.scatter(pv, ov, s=8, alpha=0.35, c="#333", rasterized=True)
        lim = max(np.abs(np.concatenate([pv, ov])).max(), 0.1)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
        r, p = stats.pearsonr(pv, ov)
        top = np.argsort(np.abs(pv))[-12:]
        for i in top:
            ax.annotate(panel[i], (pv[i], ov[i]), fontsize=5, alpha=0.8,
                        xytext=(2, 2), textcoords="offset points")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Predicted Δ (in-silico KO)")
        ax.set_ylabel("Observed Δ (sgIl4ra − NTC)")
        ax.set_title(
            f"E  Best niche: {best.slice} N{best.niche.split('|')[-1]}\nr = {r:+.3f}, perm p = {best.pearson_perm_p:.3f}",
            loc="left", fontweight="bold",
        )

    ax = fig.add_subplot(gs[1, 2])
    all_pv, all_ov = [], []
    pool_cache: dict[str, sc.AnnData] = {}
    for _, row in sub.iterrows():
        if row.slice not in pool_cache:
            pool_cache[row.slice] = assign_pool_labels(row.slice, data_root, betadata_dir, ["immune"])
        sl_pool = pool_cache[row.slice]
        base_sl = baseline[baseline.obs["slice_id"].astype(str) == row.slice]
        pair = niche_deltas(base_sl, sl_pool, pred, perturb, cell_type, row.niche)
        if pair is None:
            continue
        p, o = pair
        panel = [g for g in p.index if g != perturb]
        all_pv.extend(p.loc[panel].values)
        all_ov.extend(o.loc[panel].values)
    if all_pv:
        all_pv, all_ov = np.array(all_pv), np.array(all_ov)
        m = np.isfinite(all_pv) & np.isfinite(all_ov)
        ax.hexbin(all_pv[m], all_ov[m], gridsize=35, cmap="YlOrRd", mincnt=1, linewidths=0.2)
        r, _ = stats.pearsonr(all_pv[m], all_ov[m])
        lim = np.percentile(np.abs(np.concatenate([all_pv[m], all_ov[m]])), 98)
        ax.plot([-lim, lim], [-lim, lim], "w--", lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Predicted Δ")
        ax.set_ylabel("Observed Δ")
        ax.set_title(f"F  Pooled across all β-Leiden niches\nr = {r:+.3f} (n={m.sum():,} gene×niche)", loc="left", fontweight="bold")

    fig.suptitle(f"Spotlight: sgIl4ra in immune microniches ({tag})", fontsize=12, y=1.01)
    fig.savefig(fig_dir / f"fig2_spotlight_Il4ra_immune_{tag}.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig2_spotlight_Il4ra_immune_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_concordance_grid(
    baseline: sc.AnnData,
    pred_dir: Path,
    data_root: Path,
    betadata_dir: Path,
    niche_corr: pd.DataFrame,
    fig_dir: Path,
    tag: str,
) -> None:
    n = len(HEADLINE)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (pert, ct) in zip(axes, HEADLINE):
        sub = niche_corr[
            (niche_corr.niche_type == "beta_leiden") &
            (niche_corr.perturbation == pert) &
            (niche_corr.cell_type == ct)
        ]
        if sub.empty:
            ax.axis("off")
            continue
        pred = pd.read_feather(pred_dir / f"predicted_KO_{pert}.feather").set_index("CellID")
        all_pv, all_ov = [], []
        for sl in sub.slice.unique():
            pool = assign_pool_labels(sl, data_root, betadata_dir, [ct])
            pool.obs["slice_id"] = sl
            base_sl = baseline[baseline.obs["slice_id"].astype(str) == sl]
            for niche in sub[sub.slice == sl].niche:
                pair = niche_deltas(base_sl, pool, pred, pert, ct, niche)
                if pair is None:
                    continue
                p, o = pair
                panel = [g for g in p.index if g != pert]
                all_pv.extend(p.loc[panel].values)
                all_ov.extend(o.loc[panel].values)
        if not all_pv:
            ax.axis("off")
            continue
        pv, ov = np.array(all_pv), np.array(all_ov)
        m = np.isfinite(pv) & np.isfinite(ov)
        ax.scatter(pv[m], ov[m], s=6, alpha=0.2, c="#2166ac", rasterized=True)
        lim = np.percentile(np.abs(np.concatenate([pv[m], ov[m]])), 97)
        lim = max(lim, 0.08)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7, alpha=0.4)
        r = stats.pearsonr(pv[m], ov[m])[0] if m.sum() > 10 else float("nan")
        med = sub.pearson_r.median()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"sg{pert} | {ct}\nmedian niche r = {med:+.3f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted Δ", fontsize=8)
        ax.set_ylabel("Observed Δ", fontsize=8)
        ax.text(0.05, 0.92, f"meta r = {r:+.3f}", transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(f"β-Leiden microniche concordance: predicted vs observed Δ ({tag})", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig3_concordance_grid_{tag}.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig3_concordance_grid_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_slice_facet(
    data_root: Path,
    betadata_dir: Path,
    niche_corr: pd.DataFrame,
    fig_dir: Path,
    tag: str,
    perturb: str = "Il4ra",
    cell_type: str = "immune",
) -> None:
    slices = sorted(niche_corr.slice.unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    for ax, sl in zip(axes, slices):
        pool = assign_pool_labels(sl, data_root, betadata_dir, [cell_type])
        ct = pool[pool.obs["cell_type"].astype(str) == cell_type]
        pert = ct[ct.obs["target_gene"].astype(str) == perturb]
        niches = ct.obs["beta_leiden"].astype(str)
        uniq = sorted(n for n in niches.unique() if "unassigned" not in n)
        for i, lab in enumerate(uniq):
            m = niches == lab
            ax.scatter(ct.obsm["spatial"][m, 0], ct.obsm["spatial"][m, 1],
                       c=NICHE_COLORS[i % len(NICHE_COLORS)], s=1.2, alpha=0.65, rasterized=True)
        ax.scatter(pert.obsm["spatial"][:, 0], pert.obsm["spatial"][:, 1],
                   facecolors="none", edgecolors="#111", s=6, lw=0.25, alpha=0.6)
        sub = niche_corr[
            (niche_corr.slice == sl) & (niche_corr.perturbation == perturb) &
            (niche_corr.cell_type == cell_type) & (niche_corr.niche_type == "beta_leiden")
        ]
        med_r = sub.pearson_r.median() if len(sub) else float("nan")
        ax.set_title(f"{sl}  (median r = {med_r:+.3f})", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"β-Leiden immune niches + sg{perturb} across 4 tissue sections ({tag})", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig4_spatial_facet_{perturb}_{cell_type}_{tag}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_on_target_niche(
    niche_corr: pd.DataFrame,
    fig_dir: Path,
    tag: str,
) -> None:
    bl = niche_corr[niche_corr.niche_type == "beta_leiden"].copy()
    perts = ["Il4ra", "Cd83", "Cd74", "Cks1b"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()
    for ax, pert in zip(axes, perts):
        sub = bl[
            (bl.perturbation == pert) &
            bl.obs_on_target.notna() &
            bl.pred_on_target.notna()
        ]
        if len(sub) < 5:
            ax.axis("off")
            continue
        colors = {"immune": "#d62728", "myeloid": "#ff7f0e", "fibroblast": "#2ca02c"}
        for ct, grp in sub.groupby("cell_type"):
            ax.scatter(
                grp.pred_on_target, grp.obs_on_target,
                s=25 + grp.n_pert * 0.5, c=colors.get(ct, "#333"),
                alpha=0.75, edgecolors="k", lw=0.3, label=ct,
            )
        lim = np.percentile(np.abs(np.concatenate([sub.pred_on_target, sub.obs_on_target])), 95)
        lim = max(lim, 0.05)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7, alpha=0.4)
        ax.axhline(0, color="gray", lw=0.4)
        ax.axvline(0, color="gray", lw=0.4)
        r, p = stats.pearsonr(sub.pred_on_target, sub.obs_on_target)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"Predicted Δ {pert}")
        ax.set_ylabel(f"Observed Δ {pert}")
        ax.set_title(f"sg{pert} on-target (niche pseudobulk)\nr = {r:+.3f}, p = {p:.3f}, n = {len(sub)}", fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle(f"On-target concordance across β-Leiden microniches ({tag})", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig5_on_target_niche_{tag}.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig5_on_target_niche_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results/beta_leiden")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/beta_leiden")
    ap.add_argument("--baseline-h5ad", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_prep/baseline_ntc_0c6fbac5e6cd947c_fullprep.h5ad")
    ap.add_argument("--betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_seed")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_pooled")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--tag", default="pooled")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)

    summary = pd.read_csv(args.results_dir / f"summary_{args.tag}.csv")
    niche_corr = pd.read_csv(args.results_dir / f"niche_corr_{args.tag}.csv")
    pathway = pd.read_csv(args.results_dir / f"pathway_distinctness_{args.tag}.csv")

    print("Figure 1: main overview…")
    fig_main_overview(summary, niche_corr, pathway, args.fig_dir, args.tag)

    baseline = load_baseline(args.baseline_h5ad)
    print("Figure 2: Il4ra spotlight (subQ-1)…")
    pool_sq1 = assign_pool_labels("subQ-1", args.data_root, args.betadata_dir, ["immune", "fibroblast", "myeloid"])
    fig_spotlight_il4ra(pool_sq1, baseline, args.pred_dir, args.data_root, args.betadata_dir,
                      niche_corr, args.fig_dir, args.tag)

    print("Figure 3: concordance grid…")
    fig_concordance_grid(baseline, args.pred_dir, args.data_root, args.betadata_dir, niche_corr, args.fig_dir, args.tag)

    print("Figure 4: spatial facet…")
    fig_slice_facet(args.data_root, args.betadata_dir, niche_corr, args.fig_dir, args.tag)

    print("Figure 5: on-target niche validation…")
    fig_on_target_niche(niche_corr, args.fig_dir, args.tag)

    print(f"Done → {args.fig_dir}")


if __name__ == "__main__":
    main()
