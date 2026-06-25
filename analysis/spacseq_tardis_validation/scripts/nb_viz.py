"""Notebook-friendly plotting helpers (return Figure objects, no save/close)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _pos_color(v: float, pos: str = "#2166ac", neg: str = "#b2182b") -> str:
    return pos if v >= 0 else neg


def plot_meta_analysis(
    meta: pd.DataFrame,
    *,
    top_n: int = 12,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Cross-slice Stouffer meta-analysis bar chart (script 08 fig2)."""
    fig, ax = plt.subplots(figsize=figsize or (10, max(4, 0.35 * min(top_n, len(meta) or 1))))
    if meta.empty:
        ax.text(0.5, 0.5, "No meta-analysis rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    show = meta.sort_values("stouffer_meta_p").head(top_n)
    y = np.arange(len(show))
    colors = [_pos_color(r, "#2ca02c", "#d62728") for r in show.median_pearson_r]
    ax.barh(y, show.median_pearson_r, color=colors, alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"sg{r.perturbation} | {r.cell_type} (n={int(r.n_slices)})" for _, r in show.iterrows()],
        fontsize=8,
    )
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Median Pearson r across slices")
    ax.set_title(title or "Cross-slice meta-analysis")
    for i, (_, r) in enumerate(show.iterrows()):
        ax.text(
            r.median_pearson_r + (0.01 if r.median_pearson_r >= 0 else -0.01),
            i,
            f" meta-p={r.stouffer_meta_p:.2g}, {r.frac_slices_pos_r:.0%} slices r>0",
            va="center",
            ha="left" if r.median_pearson_r >= 0 else "right",
            fontsize=7,
        )
    fig.tight_layout()
    return fig, ax


def plot_slice_heatmap(
    combined: pd.DataFrame,
    slices: Sequence[str] | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Per-slice Pearson r heatmap (script 08 fig1)."""
    fig, ax = plt.subplots(figsize=figsize or (12, 5))
    if combined.empty:
        ax.text(0.5, 0.5, "No combined rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    df = combined.copy()
    df["pair"] = df.apply(lambda r: f"{r.perturbation}|{r.cell_type}", axis=1)
    pivot = df.pivot_table(index="slice", columns="pair", values="pearson_r", aggfunc="first")
    pairs = sorted(pivot.columns)
    slices_ord = [s for s in (slices or pivot.index.tolist()) if s in pivot.index]
    vals = pivot.loc[slices_ord, pairs].values.astype(float)
    vmax = max(0.25, float(np.nanmax(np.abs(vals))))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("|", "\n") for p in pairs], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(slices_ord)))
    ax.set_yticklabels(slices_ord)
    for i in range(len(slices_ord)):
        for j in range(len(pairs)):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(v) > vmax * 0.55 else "black",
                )
    ax.set_title(title or "Per-slice Pearson r: predicted vs observed Δ")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    return fig, ax


def plot_celltype_boxplot(
    combined: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    if combined.empty:
        ax.text(0.5, 0.5, "No combined rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    ct_order = sorted(combined.cell_type.unique())
    data = [combined.loc[combined.cell_type == c, "pearson_r"].values for c in ct_order]
    bp = ax.boxplot(data, tick_labels=ct_order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Pearson r (per slice × perturbation)")
    ax.set_title(title or f"Validation correlation by cell type (n={len(combined)} rows)")
    fig.tight_layout()
    return fig, ax


def plot_prediction_scorecard(
    scorecard: pd.DataFrame,
    *,
    levels: Sequence[str] = ("cell_type", "graphclust", "spatial_grid"),
    figsize: tuple[float, float] | None = None,
    model_colors: dict[str, str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Sharpened scorecard bar chart (script 10)."""
    if scorecard.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Empty scorecard", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    plot_df = scorecard[scorecard.level.isin(levels)].copy()
    plot_df["label"] = plot_df["level"] + "/" + plot_df["compartment"]
    colors = model_colors or {"seed": "#1f77b4", "pooled": "#2ca02c", "tuned": "#9467bd", "extra": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=figsize or (10, max(4, 0.35 * len(plot_df))))
    y = np.arange(len(plot_df))
    for i, (_, row) in enumerate(plot_df.iterrows()):
        c = colors.get(row.model, "#888888")
        ax.barh(i, row.median_r, color=c, alpha=0.85, height=0.7)
        ax.text(
            row.median_r + (0.008 if row.median_r >= 0 else -0.008),
            i,
            f"{row.model} r={row.median_r:+.3f} (n={int(row.n_tests)})",
            va="center",
            ha="left" if row.median_r >= 0 else "right",
            fontsize=7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.label, fontsize=8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Median Pearson r (predicted vs observed Δ)")
    ax.set_title("SpaceTravLR prediction quality scorecard")
    fig.tight_layout()
    return fig, ax


def plot_validation_dashboard(
    metrics: pd.DataFrame,
    direct_deg: pd.DataFrame,
    *,
    tag: str = "tuned",
    figsize: tuple[float, float] = (14, 10),
    support_threshold: float = 0.6,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Multi-panel dashboard (script 21)."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    axes: list[plt.Axes] = []

    ax0 = fig.add_subplot(gs[0, 0])
    axes.append(ax0)
    m = metrics[metrics.metric == "median_pearson_r"].copy()
    if not m.empty:
        ax0.barh(m.category, m.value, color=[_pos_color(v) for v in m.value], edgecolor="k", linewidth=0.3)
        ax0.axvline(0, color="k", lw=0.8)
        ax0.set_xlabel("Pearson r")
        ax0.set_title("Concordance metrics", fontweight="bold")
    else:
        ax0.text(0.5, 0.5, "No Pearson metrics", ha="center", va="center", transform=ax0.transAxes)
        ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    axes.append(ax1)
    m2 = metrics[metrics.metric == "frac_support"].copy()
    if not m2.empty:
        ax1.barh(
            m2.category, m2.value,
            color=["#4daf4a" if v >= support_threshold else "#984ea3" for v in m2.value],
            edgecolor="k", linewidth=0.3,
        )
        ax1.axvline(support_threshold, color="#b2182b", ls="--", lw=1)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Fraction modules ≥ threshold sign match")
        ax1.set_title("Paper biology recapitulation", fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No support metrics", ha="center", va="center", transform=ax1.transAxes)
        ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, :])
    axes.append(ax2)
    if not direct_deg.empty and "pearson_r" in direct_deg.columns:
        show = direct_deg.sort_values("pearson_r", ascending=True)
        ax2.barh(
            [f"{r.perturbation}|{r.cell_type}" for _, r in show.iterrows()],
            show.pearson_r,
            color=[_pos_color(v) for v in show.pearson_r],
            edgecolor="k", linewidth=0.3,
        )
        ax2.axvline(0, color="k", lw=0.8)
        ax2.set_xlabel("Pearson r (obs vs pred Δ on sgP cells)")
        ax2.set_title("Direct perturbed-cell DEG concordance", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Direct DEG stats not found", ha="center", va="center", transform=ax2.transAxes)
        ax2.axis("off")

    fig.suptitle(
        f"SpaceTravLR × SPAC-seq validation dashboard ({tag})\n"
        "Zhang et al. Cell 2026 — subQ pooled + lung M001 observed",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig, axes


def plot_paper_scorecard(
    mod_df: pd.DataFrame,
    *,
    support_threshold: float = 0.6,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Paper finding recapitulation scorecard (script 19)."""
    if mod_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No module rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, [ax]

    show = mod_df.copy()
    show["label"] = show.apply(
        lambda r: f"{r['finding_id'][:12]}\n{r['perturbation']}|{r['cell_type']}\n{r['module'][:22]}",
        axis=1,
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize or (16, max(6, 0.35 * len(show))))
    cols = ["obs_sign_match_rate", "pred_sign_match_rate", "obs_pred_concordance_rate"]
    titles = ["SPAC-seq matches paper", "SpaceTravLR matches paper", "Obs ↔ Pred concordance"]
    for ax, col, title in zip(axes, cols, titles):
        vals = show[col].fillna(0).values
        colors = [
            "#2166ac" if v >= support_threshold else "#67a9cf" if v >= 0.5 else "#d1e5f0"
            for v in vals
        ]
        ax.barh(range(len(show)), vals, color=colors, edgecolor="k", linewidth=0.3)
        ax.axvline(support_threshold, color="#b2182b", ls="--", lw=1, label=f"{support_threshold:.0%} threshold")
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels(show.label, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction genes with expected sign")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Paper finding recapitulation scorecard (Zhang et al. Cell 2026)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig, list(axes)


def plot_paper_module_heatmap(
    mod_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    if mod_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No module rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, [ax]

    pivot_obs = mod_df.pivot_table(
        index=["finding_id", "module"], columns="cell_type", values="obs_sign_match_rate", aggfunc="first",
    )
    pivot_pred = mod_df.pivot_table(
        index=["finding_id", "module"], columns="cell_type", values="pred_sign_match_rate", aggfunc="first",
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize or (12, max(4, 0.35 * len(pivot_obs))))
    for ax, pivot, title in zip(axes, [pivot_obs, pivot_pred], ["Observed (SPAC-seq)", "Predicted (SpaceTravLR)"]):
        if pivot.empty:
            ax.axis("off")
            continue
        vals = pivot.fillna(0).values.astype(float)
        im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([f"{a}|{b}" for a, b in pivot.index], fontsize=7)
        ax.set_title(title, fontweight="bold")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.6, label="Sign match rate")
    fig.suptitle("Module-level paper hypothesis support", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, list(axes)


def plot_lung_module_bars(
    mod_df: pd.DataFrame,
    *,
    title: str,
    support_threshold: float = 0.6,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if mod_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No lung module rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    show = mod_df.sort_values("obs_sign_match_rate", ascending=True)
    fig, ax = plt.subplots(figsize=figsize or (10, max(4, 0.4 * len(show))))
    colors = ["#2166ac" if v >= support_threshold else "#67a9cf" for v in show.obs_sign_match_rate]
    ax.barh(range(len(show)), show.obs_sign_match_rate, color=colors, edgecolor="k", linewidth=0.3)
    ax.axvline(support_threshold, color="#b2182b", ls="--", lw=1)
    labels = [f"{r.cell_type} | {r.module[:28]}" for _, r in show.iterrows()]
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction genes matching paper direction (observed)")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig, ax


def plot_cnn_enrichment_scatter(
    enrich_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    *,
    top_n: int = 6,
    tag: str = "cnn",
    point_color: str | None = None,
    figsize: tuple[float, float] | None = None,
    label_niches: bool = True,
    color_by_niche: bool = True,
    show_regression: bool = True,
    point_size: float = 70,
    slice_filter: str | None = None,
    perturb_filter: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if enrich_df.empty or corr_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No enrichment data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, np.array([[ax]])

    pairs = corr_df.dropna(subset=["pearson_r"]).sort_values("pearson_r", ascending=False)
    if slice_filter:
        pairs = pairs[pairs["slice"] == slice_filter]
    if perturb_filter:
        pairs = pairs[pairs["perturbation"] == perturb_filter]
    pairs = pairs.head(top_n)
    if pairs.empty:
        pairs = corr_df.head(top_n)
    n = len(pairs)
    cols = min(3, max(n, 1))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4.8 * cols, 4.2 * rows), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), pairs.iterrows()):
        sub = enrich_df[
            (enrich_df["slice"] == row["slice"]) & (enrich_df["perturbation"] == row["perturbation"])
        ].copy()
        if sub.empty:
            ax.axis("off")
            continue
        if color_by_niche:
            colors = _microniche_color_map(sub["niche"].astype(str))
            cvals = [colors.get(n, "#2563eb") for n in sub["niche"]]
        else:
            cvals = point_color or "#2563eb"
        ax.scatter(
            sub["pred_enrichment_score"], sub["obs_log2_enrichment"],
            s=point_size, alpha=0.9, c=cvals, edgecolors="k", linewidths=0.4,
        )
        if show_regression and len(sub) >= 3:
            x, y = sub["pred_enrichment_score"].to_numpy(), sub["obs_log2_enrichment"].to_numpy()
            if np.std(x) > 1e-8:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, m * xs + b, color="#111827", lw=1.2, ls="--", alpha=0.8)
        if label_niches:
            for _, pt in sub.iterrows():
                ax.annotate(
                    str(pt["niche"]).split("|")[-1],
                    (pt["pred_enrichment_score"], pt["obs_log2_enrichment"]),
                    fontsize=7, ha="center", va="bottom",
                )
        ax.axhline(0, color="k", lw=0.4, alpha=0.35)
        ax.axvline(0, color="k", lw=0.4, alpha=0.35)
        ax.grid(True, alpha=0.25)
        r = row.get("pearson_r", float("nan"))
        p = row.get("p_pearson", float("nan"))
        ax.set_title(
            f"sg{row['perturbation']} | {row['slice']}\nr={r:+.2f}, p={p:.2g}, n={int(row['n_niches'])} niches",
            fontsize=9,
        )
        ax.set_xlabel("Predicted enrichment score")
        ax.set_ylabel("Observed log₂ OR (sgP vs NTC)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"CNN β-microniche guide enrichment ({tag})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_cnn_enrichment_scatter_with_histology(
    enrich_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    pool_by_slice: dict,
    *,
    top_n: int = 6,
    tag: str = "cnn",
    mc38_dir=None,
    slice_filter: str | None = None,
    perturb_filter: str | None = None,
    spot_size: float = 1.4,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Side-by-side enrichment scatter and sc.pl.spatial microniche maps on H&E."""
    import scanpy as sc
    from pathlib import Path

    import spatial_histology as sh

    if enrich_df.empty or corr_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No enrichment data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, np.array([[ax]])

    mc38 = Path(mc38_dir) if mc38_dir is not None else sh.DEFAULT_MC38
    pairs = corr_df.dropna(subset=["pearson_r"]).sort_values("pearson_r", ascending=False)
    if slice_filter:
        pairs = pairs[pairs["slice"] == slice_filter]
    if perturb_filter:
        pairs = pairs[pairs["perturbation"] == perturb_filter]
    pairs = pairs.head(top_n)
    if pairs.empty:
        pairs = corr_df.head(top_n)

    n = len(pairs)
    fig, axes = plt.subplots(
        n, 2,
        figsize=figsize or (11.5, 3.8 * n),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.05, 1.35], "wspace": 0.28, "hspace": 0.35},
    )

    for i, (_, row) in enumerate(pairs.iterrows()):
        ax_sc, ax_sp = axes[i]
        sl = str(row["slice"])
        pert = str(row["perturbation"])
        sub = enrich_df[(enrich_df["slice"] == sl) & (enrich_df["perturbation"] == pert)].copy()
        if sub.empty:
            ax_sc.axis("off")
            ax_sp.axis("off")
            continue

        colors = _microniche_color_map(sub["niche"].astype(str))
        cvals = [colors.get(n, "#2563eb") for n in sub["niche"]]
        ax_sc.scatter(
            sub["pred_enrichment_score"], sub["obs_log2_enrichment"],
            s=90, alpha=0.92, c=cvals, edgecolors="k", linewidths=0.45, zorder=3,
        )
        if len(sub) >= 3:
            x, y = sub["pred_enrichment_score"].to_numpy(), sub["obs_log2_enrichment"].to_numpy()
            if np.std(x) > 1e-8:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax_sc.plot(xs, m * xs + b, color="#111827", lw=1.2, ls="--", alpha=0.8)
        for _, pt in sub.iterrows():
            ax_sc.annotate(
                str(pt["niche"]).split("|")[-1],
                (pt["pred_enrichment_score"], pt["obs_log2_enrichment"]),
                fontsize=8, ha="center", va="bottom",
            )
        ax_sc.axhline(0, color="k", lw=0.4, alpha=0.35)
        ax_sc.axvline(0, color="k", lw=0.4, alpha=0.35)
        ax_sc.grid(True, alpha=0.25)
        r = float(row.get("pearson_r", float("nan")))
        p = float(row.get("p_pearson", float("nan")))
        ax_sc.set_title(
            f"sg{pert} | {sl}\nr={r:+.2f}, p={p:.2g}, n={int(row['n_niches'])} niches",
            fontsize=10, fontweight="bold",
        )
        ax_sc.set_xlabel("Predicted enrichment score")
        ax_sc.set_ylabel("Observed log₂ OR (sgP vs NTC)")

        pool = pool_by_slice.get(sl)
        if pool is None or "cnn_leiden" not in pool.obs.columns:
            ax_sp.text(0.5, 0.5, "No spatial pool", ha="center", va="center", transform=ax_sp.transAxes)
            ax_sp.axis("off")
            continue

        if "cell_type" in pool.obs.columns:
            ct = pool[pool.obs["cell_type"].astype(str) == "tumor"].copy()
        else:
            ct = pool.copy()
        ct.obs["microniche"] = ct.obs["cnn_leiden"].astype(str).str.split("|").str[-1]
        order = [str(x).split("|")[-1] for x in sub["niche"].astype(str)]
        extra = sorted(set(ct.obs["microniche"].astype(str)) - set(order))
        ct.obs["microniche"] = pd.Categorical(ct.obs["microniche"], categories=order + extra, ordered=True)
        palette = {lab.split("|")[-1]: colors.get(lab, "#888888") for lab in colors}
        for lab in ct.obs["microniche"].astype(str).unique():
            palette.setdefault(lab, "#888888")

        try:
            sh.attach_histology(ct, sl, mc38)
            sc.pl.spatial(
                ct,
                color="microniche",
                library_id=sl,
                img_key="hires",
                ax=ax_sp,
                show=False,
                size=spot_size,
                alpha=0.88,
                palette=palette,
                title="",
                legend_loc="right margin",
                frameon=False,
            )
            ax_sp.set_title(f"Tumor CNN β-microniches on H&E", fontsize=10, fontweight="bold")
        except Exception as exc:
            ax_sp.text(0.5, 0.5, f"Histology unavailable\n{exc}", ha="center", va="center", transform=ax_sp.transAxes)
            ax_sp.axis("off")

    title = f"CNN β-microniche guide enrichment + tissue histology ({tag})"
    if slice_filter:
        title = f"{slice_filter} — guide enrichment + H&E microniches ({tag})"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.94, hspace=0.42, wspace=0.25)
    return fig, axes


def plot_cnn_enrichment_heatmap(
    corr_df: pd.DataFrame,
    *,
    tag: str = "cnn",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if corr_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    pivot = corr_df.pivot_table(index="perturbation", columns="slice", values="pearson_r", aggfunc="first")
    fig, ax = plt.subplots(figsize=figsize or (0.7 * len(pivot.columns) + 2, 0.5 * len(pivot.index) + 1.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Obs vs pred niche enrichment correlation ({tag})")
    fig.tight_layout()
    return fig, ax


def _microniche_color_map(labels: pd.Series) -> dict[str, tuple]:
    uniq = sorted(labels.unique())
    cmap = plt.colormaps.get_cmap("tab20")
    return {lab: cmap(i % 20) for i, lab in enumerate(uniq)}


def plot_microniche_spatial(
    spatial_df: pd.DataFrame,
    *,
    slice_id: str = "",
    perturb: str | None = None,
    panel: str = "all",
    point_size: float = 4.0,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
    """Spatial scatter of CNN β-Leiden tumor microniches on tissue coordinates.

    panel: 'all' | 'ntc' | 'pert' | 'triple' (NTC / sgP / all side-by-side)
    """
    if spatial_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No spatial tumor data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    df = spatial_df.copy()
    if slice_id:
        df = df[df["slice"].astype(str) == slice_id]
    if perturb and panel in ("pert", "triple"):
        pass

    colors = _microniche_color_map(df["cnn_leiden"].astype(str))

    def _draw(ax: plt.Axes, sub: pd.DataFrame, subtitle: str) -> None:
        for lab in sorted(sub["cnn_leiden"].astype(str).unique()):
            m = sub["cnn_leiden"].astype(str) == lab
            c = "#dddddd" if lab in ("unassigned", "nan") else colors.get(lab, "#888888")
            ax.scatter(sub.loc[m, "x"], sub.loc[m, "y"], c=[c], s=point_size, alpha=0.78, rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(subtitle, fontsize=9)

    if panel == "triple" and perturb:
        fig, axes = plt.subplots(1, 3, figsize=figsize or (14, 4.5))
        ntc = df[df["target_gene"].astype(str) == "non-targeting"]
        pert = df[df["target_gene"].astype(str) == perturb]
        _draw(axes[0], ntc, f"NTC tumor (n={len(ntc)})")
        _draw(axes[1], pert, f"sg{perturb} tumor (n={len(pert)})")
        _draw(axes[2], df, f"All tumor (n={len(df)})")
        n_n = df["cnn_leiden"].astype(str).nunique()
        fig.suptitle(
            title or f"{slice_id or df['slice'].iloc[0]} — {n_n} CNN β-microniches on tissue",
            fontweight="bold",
        )
        fig.tight_layout()
        return fig, list(axes)

    if panel == "ntc":
        df = df[df["target_gene"].astype(str) == "non-targeting"]
    elif panel == "pert" and perturb:
        df = df[df["target_gene"].astype(str) == perturb]

    fig, ax = plt.subplots(figsize=figsize or (7, 6))
    _draw(ax, df, f"n={len(df)} cells")
    n_n = spatial_df["cnn_leiden"].astype(str).nunique()
    ax.set_title(title or f"{slice_id or ''} tumor microniches (n={n_n} niches)", fontweight="bold")
    fig.tight_layout()
    return fig, ax


def plot_gene_expression_spatial(
    spatial_df: pd.DataFrame,
    gene: str,
    *,
    slice_id: str = "",
    cell_filter: str = "all",
    perturb: str | None = None,
    point_size: float = 3.5,
    figsize: tuple[float, float] = (6, 5),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Spatial tumor map colored by log-normalized expression of one gene."""
    col = f"expr_{gene}"
    df = spatial_df.copy()
    if slice_id:
        df = df[df["slice"].astype(str) == slice_id]
    if cell_filter == "ntc":
        df = df[df["target_gene"].astype(str) == "non-targeting"]
    elif cell_filter == "pert" and perturb:
        df = df[df["target_gene"].astype(str) == perturb]
    if df.empty or col not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for {gene}", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax
    vals = df[col].to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(vals - np.nanmedian(vals)), 95)) if np.isfinite(vals).any() else 1.0
    vmax = max(vmax, 0.05)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=np.nanmedian(vals), vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize)
    sc_plot = ax.scatter(df["x"], df["y"], c=vals, s=point_size, cmap="magma", norm=norm, alpha=0.85, rasterized=True)
    fig.colorbar(sc_plot, ax=ax, shrink=0.75, label=f"{gene} expression")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"{gene} — tumor ST ({cell_filter})", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_niche_field_spatial(
    spatial_df: pd.DataFrame,
    enrich_df: pd.DataFrame,
    *,
    slice_id: str,
    perturb: str,
    value_col: str = "obs_log2_enrichment",
    figsize: tuple[float, float] = (6, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Map per-niche enrichment score onto tumor cells by microniche label."""
    df = spatial_df[spatial_df["slice"].astype(str) == slice_id].copy()
    sub = enrich_df[
        (enrich_df["slice"] == slice_id) & (enrich_df["perturbation"] == perturb)
    ][["niche", value_col]]
    if df.empty or sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No niche field data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax
    df = df.merge(sub, left_on="cnn_leiden", right_on="niche", how="left")
    vals = df[value_col].fillna(0).to_numpy()
    vmax = max(0.5, float(np.nanpercentile(np.abs(vals), 95)))
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize)
    sc_plot = ax.scatter(df["x"], df["y"], c=vals, s=4, cmap="RdBu_r", norm=norm, alpha=0.9, rasterized=True)
    fig.colorbar(sc_plot, ax=ax, shrink=0.75, label=value_col)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"sg{perturb} observed niche enrichment on tissue", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_lung_m001_composite(
    enrich_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
    *,
    perturb: str = "Icam1",
    slice_id: str = "Lung_Metastasis_M001",
    tag: str = "cnn",
    paper_genes: tuple[str, ...] = ("Icam1", "Spp1", "Cxcl9"),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Pair enrichment scatter with lung tumor ST + microniche maps (paper figure layout)."""
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(16, 12), layout="constrained")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1], hspace=0.32, wspace=0.28)

    ax_sc = fig.add_subplot(gs[0, :])
    sub_e = enrich_df[(enrich_df["slice"] == slice_id) & (enrich_df["perturbation"] == perturb)]
    sub_c = corr_df[(corr_df["slice"] == slice_id) & (corr_df["perturbation"] == perturb)]
    if not sub_e.empty:
        colors = _microniche_color_map(sub_e["niche"].astype(str))
        cvals = [colors.get(n, "#2563eb") for n in sub_e["niche"]]
        ax_sc.scatter(
            sub_e["pred_enrichment_score"], sub_e["obs_log2_enrichment"],
            s=120, c=cvals, edgecolors="k", linewidths=0.5, zorder=3,
        )
        for _, pt in sub_e.iterrows():
            ax_sc.annotate(str(pt["niche"]).split("|")[-1], (pt["pred_enrichment_score"], pt["obs_log2_enrichment"]), fontsize=9)
        if len(sub_e) >= 3:
            x, y = sub_e["pred_enrichment_score"].to_numpy(), sub_e["obs_log2_enrichment"].to_numpy()
            if np.std(x) > 1e-8:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax_sc.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.7)
    ax_sc.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_sc.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax_sc.grid(True, alpha=0.3)
    r = float(sub_c["pearson_r"].iloc[0]) if not sub_c.empty else float("nan")
    p = float(sub_c["p_pearson"].iloc[0]) if not sub_c.empty and "p_pearson" in sub_c.columns else float("nan")
    ax_sc.set_xlabel("Predicted enrichment score (exclusion × escape × CNN β)")
    ax_sc.set_ylabel("Observed log₂ OR (sgP vs NTC fraction per niche)")
    ax_sc.set_title(
        f"Lung M001 sg{perturb} — microniche guide enrichment\n"
        f"Zhang et al. Cell 2026 — {perturb} spatial niche composition  |  r={r:+.2f}, p={p:.2g}, n={len(sub_e)} niches",
        fontweight="bold",
    )

    for j, gene in enumerate(paper_genes):
        ax = fig.add_subplot(gs[1, j])
        col = f"expr_{gene}"
        df = spatial_df[spatial_df["slice"].astype(str) == slice_id]
        if df.empty or col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(float)
        vmax = float(np.nanpercentile(np.abs(vals - np.nanmedian(vals)), 95)) if np.isfinite(vals).any() else 1.0
        vmax = max(vmax, 0.05)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=np.nanmedian(vals), vmax=vmax)
        sc_p = ax.scatter(df["x"], df["y"], c=vals, s=2.5, cmap="magma", norm=norm, rasterized=True)
        fig.colorbar(sc_p, ax=ax, shrink=0.7, label=gene)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{gene} expression (tumor)", fontsize=9, fontweight="bold")

    ax_n = fig.add_subplot(gs[2, 0])
    df = spatial_df[spatial_df["slice"].astype(str) == slice_id]
    colors = _microniche_color_map(df["cnn_leiden"].astype(str))
    for lab in sorted(df["cnn_leiden"].astype(str).unique()):
        m = df["cnn_leiden"].astype(str) == lab
        ax_n.scatter(df.loc[m, "x"], df.loc[m, "y"], c=[colors.get(lab, "#888")], s=3, rasterized=True)
    ax_n.set_aspect("equal")
    ax_n.axis("off")
    ax_n.set_title(f"CNN β-microniches (n={df['cnn_leiden'].nunique()})", fontweight="bold", fontsize=9)

    ax_o = fig.add_subplot(gs[2, 1])
    if not sub_e.empty:
        df2 = df.merge(sub_e[["niche", "obs_log2_enrichment"]], left_on="cnn_leiden", right_on="niche", how="left")
        vals = df2["obs_log2_enrichment"].fillna(0)
        vmax = max(0.5, float(vals.abs().quantile(0.95)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        sc_o = ax_o.scatter(df2["x"], df2["y"], c=vals, s=3, cmap="RdBu_r", norm=norm, rasterized=True)
        fig.colorbar(sc_o, ax=ax_o, shrink=0.7, label="log₂ OR")
    ax_o.set_aspect("equal")
    ax_o.axis("off")
    ax_o.set_title(f"Observed sg{perturb} enrichment field", fontweight="bold", fontsize=9)

    ax_p = fig.add_subplot(gs[2, 2])
    ntc = df[df["target_gene"].astype(str) == "non-targeting"]
    pert_cells = df[df["target_gene"].astype(str) == perturb]
    ax_p.scatter(ntc["x"], ntc["y"], c="#d1d5db", s=2, rasterized=True, label="NTC")
    ax_p.scatter(pert_cells["x"], pert_cells["y"], c="#dc2626", s=3, alpha=0.7, rasterized=True, label=f"sg{perturb}")
    ax_p.set_aspect("equal")
    ax_p.axis("off")
    ax_p.legend(fontsize=7, loc="upper right")
    ax_p.set_title("Guide occupancy (tumor)", fontweight="bold", fontsize=9)

    fig.suptitle(
        f"{slice_id} — spatial transcriptomics × CNN microniches ({tag})",
        fontsize=13,
        fontweight="bold",
    )
    return fig, [ax_sc, ax_n, ax_o, ax_p]


def plot_direct_deg_bars(
    direct_deg: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "Direct perturbed-cell DEG concordance",
) -> tuple[plt.Figure, plt.Axes]:
    if direct_deg.empty or "pearson_r" not in direct_deg.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No direct DEG stats", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    show = direct_deg.sort_values("pearson_r", ascending=True)

    def _label(row: pd.Series) -> str:
        pert = row.get("perturbation", "")
        if "cell_type" in row.index and pd.notna(row.get("cell_type")):
            return f"{pert}|{row['cell_type']}"
        if "source_cell_type" in row.index:
            return f"{pert}|{row['source_cell_type']}→{row['neighbor_cell_type']}"
        return str(pert)

    labels = [_label(r) for _, r in show.iterrows()]
    fig, ax = plt.subplots(figsize=figsize or (10, max(4, 0.35 * len(show))))
    ax.barh(
        labels,
        show.pearson_r,
        color=[_pos_color(v) for v in show.pearson_r],
        edgecolor="k", linewidth=0.3,
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pearson r (obs vs pred Δ on sgP cells)")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig, ax


def plot_spatial_niche_corr(
    niche_corr: pd.DataFrame,
    *,
    niche_type: str | None = None,
    cell_types: Sequence[str] = ("immune", "myeloid", "fibroblast"),
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if niche_corr.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No spatial niche correlation data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    df = niche_corr.copy()
    if niche_type and "niche_type" in df.columns:
        df = df[df.niche_type == niche_type]
    if "cell_type" in df.columns:
        df = df[df.cell_type.isin(cell_types)]
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No rows after filter", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    df["label"] = df["perturbation"].astype(str) + " | " + df["cell_type"].astype(str)
    show = df.sort_values("pearson_r", ascending=True)
    fig, ax = plt.subplots(figsize=figsize or (10, max(4, 0.35 * len(show))))
    ax.barh(show.label, show.pearson_r, color=[_pos_color(v) for v in show.pearson_r], edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Pearson r")
    ax.set_title(f"Spatial niche concordance ({niche_type or 'all niches'})")
    fig.tight_layout()
    return fig, ax
