#!/usr/bin/env python3
"""Paper-style spatial enrichment figure (panels D/E analog) for CNN v2 microniches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import ensure_boot

ensure_boot()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import spatial_histology as sh

SUBQ_PERTS = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b", "Ptk6"]
HEADLINE_CASES = [
    ("subQ-2", "Cd83"),
    ("subQ-4", "Ptk6"),
    ("Lung_Metastasis_M001", "Icam1"),
]
POLAR_SLICE = "subQ-2"


def niche_short(n: str) -> str:
    return str(n).split("|")[-1]


def niche_palette(labels: list[str]) -> dict[str, tuple]:
    cmap = plt.colormaps.get_cmap("tab20")
    uniq = sorted(set(labels))
    return {lab: cmap(i % 20) for i, lab in enumerate(uniq)}


def categorize_niche(exclusion: float, lo: float, hi: float) -> str:
    if exclusion >= hi:
        return "Immune exclusion"
    if exclusion <= lo:
        return "Immune infiltration"
    return "Mixed / stromal"


def load_spatial(results_dir: Path, slice_id: str, tag: str) -> pd.DataFrame:
    path = results_dir / f"spatial_tumor_{slice_id}_{tag}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def kde_contourf(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    cmap: str = "Blues",
    levels: int = 10,
    alpha: float = 0.88,
    zorder: int = 3,
) -> None:
    if len(x) < 15:
        return
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    pad = 0.06 * max(xmax - xmin, ymax - ymin, 1.0)
    xi = np.linspace(xmin - pad, xmax + pad, 100)
    yi = np.linspace(ymin - pad, ymax + pad, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    kde = gaussian_kde(np.vstack([x, y]))
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, alpha=alpha, zorder=zorder, antialiased=True)
    ax.contour(Xi, Yi, Zi, levels=levels, colors="white", linewidths=0.25, alpha=0.35, zorder=zorder + 1)


def field_contourf(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    *,
    cmap: str = "RdBu_r",
    levels: int = 12,
    alpha: float = 0.92,
    zorder: int = 3,
) -> matplotlib.cm.ScalarMappable | None:
    sub = df.dropna(subset=["x", "y", value_col])
    if len(sub) < 20:
        return None
    x = sub["x"].to_numpy(float)
    y = sub["y"].to_numpy(float)
    v = sub[value_col].to_numpy(float)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    pad = 0.04 * max(xmax - xmin, ymax - ymin, 1.0)
    xi = np.linspace(xmin - pad, xmax + pad, 110)
    yi = np.linspace(ymin - pad, ymax + pad, 110)
    Xi, Yi = np.meshgrid(xi, yi)
    kde_w = gaussian_kde(np.vstack([x, y]), weights=np.clip(v - v.min() + 0.05, 0.01, None))
    Zi = kde_w(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    vmax = float(np.nanpercentile(np.abs(v), 95)) if np.isfinite(v).any() else 1.0
    vmax = max(vmax, 0.35)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, norm=norm, alpha=alpha, zorder=zorder)
    return cf


def draw_niche_background(ax: plt.Axes, df: pd.DataFrame, colors: dict[str, tuple]) -> None:
    for lab in sorted(df["cnn_leiden"].astype(str).unique()):
        m = df["cnn_leiden"].astype(str) == lab
        c = colors.get(lab, "#cccccc")
        ax.scatter(
            df.loc[m, "x"], df.loc[m, "y"], c=[c], s=4.5, alpha=0.82,
            rasterized=True, zorder=1, linewidths=0,
        )
    ax.set_aspect("equal")
    ax.axis("off")


def field_tricontourf(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    *,
    cmap: str = "RdYlBu_r",
    levels: int = 14,
    alpha: float = 0.9,
) -> matplotlib.cm.ScalarMappable | None:
    sub = df.dropna(subset=["x", "y", value_col])
    if len(sub) < 30:
        return None
    x = sub["x"].to_numpy(float)
    y = sub["y"].to_numpy(float)
    v = sub[value_col].to_numpy(float)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    pad = 0.03 * max(xmax - xmin, ymax - ymin, 1.0)
    xi = np.linspace(xmin - pad, xmax + pad, 120)
    yi = np.linspace(ymin - pad, ymax + pad, 120)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), v, (Xi, Yi), method="linear")
    mask = np.isnan(Zi)
    if mask.all():
        return None
    Zi[mask] = np.nanmean(v)
    vmax = max(0.5, float(np.nanpercentile(np.abs(v), 95)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, norm=norm, alpha=alpha, zorder=2)
    ax.contour(Xi, Yi, Zi, levels=levels, colors="white", linewidths=0.2, alpha=0.35, zorder=3)
    return cf


def plot_spatial_panel_d(
    ax_main: plt.Axes,
    ax_inset: plt.Axes,
    spatial_df: pd.DataFrame,
    enrich_sub: pd.DataFrame,
    perturb: str,
    *,
    title: str,
    mode: str,
) -> None:
    df = spatial_df.copy()
    colors = niche_palette(df["cnn_leiden"].astype(str).tolist())
    draw_niche_background(ax_main, df, colors)

    if mode == "observed":
        sgp = df[df["target_gene"].astype(str) == perturb]
        scored = df.merge(
            enrich_sub[["niche", "obs_log2_enrichment"]],
            left_on="cnn_leiden",
            right_on="niche",
            how="left",
        )
        sm = field_tricontourf(ax_main, scored, "obs_log2_enrichment", cmap="RdBu_r")
        if sm is not None:
            cb = plt.colorbar(sm, ax=ax_main, fraction=0.035, pad=0.01, shrink=0.55)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("Observed log₂ OR", fontsize=7)
        kde_contourf(ax_inset, sgp["x"].to_numpy(), sgp["y"].to_numpy(), cmap="Blues", alpha=0.95, levels=8)
        ax_inset.set_title("sgRNA density", fontsize=7, pad=2)
    else:
        scored = df.merge(
            enrich_sub[["niche", "pred_enrichment_score", "obs_log2_enrichment"]],
            left_on="cnn_leiden",
            right_on="niche",
            how="left",
        )
        draw_niche_background(ax_main, df, colors)
        sm = field_tricontourf(ax_main, scored, "pred_enrichment_score", cmap="RdYlBu_r")
        if sm is not None:
            cb = plt.colorbar(sm, ax=ax_main, fraction=0.035, pad=0.01, shrink=0.55)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("Predicted score", fontsize=7)
        obs_sgp = scored[scored["target_gene"].astype(str) == perturb]
        if len(obs_sgp) >= 15:
            kde_contourf(
                ax_inset, obs_sgp["x"].to_numpy(), obs_sgp["y"].to_numpy(),
                cmap="Blues", alpha=0.95, levels=8,
            )
            ax_inset.set_title("sgRNA density", fontsize=7, pad=2)
        else:
            ax_inset.axis("off")

    ax_main.set_title(title, fontsize=9, fontweight="bold", loc="left")
    for spine in ax_inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")


def plot_polar_enrichment(
    ax: plt.Axes,
    enrich_sub: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    perts: list[str],
) -> None:
    niches = sorted(enrich_sub["niche"].astype(str).unique(), key=lambda n: niche_short(n))
    if not niches or not perts:
        ax.set_axis_off()
        return

    vals = enrich_sub[value_col].to_numpy(float)
    vmax = max(0.6, float(np.nanpercentile(np.abs(vals), 92))) if np.isfinite(vals).any() else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.colormaps["RdBu_r"]

    n_rings = len(niches)
    n_wedges = len(perts)
    theta = np.linspace(0, 2 * np.pi, n_wedges, endpoint=False)
    width = 2 * np.pi / n_wedges

    for i, niche in enumerate(niches):
        r0 = i + 0.3
        for j, pert in enumerate(perts):
            row = enrich_sub[(enrich_sub["niche"].astype(str) == niche) & (enrich_sub["perturbation"] == pert)]
            v = float(row[value_col].iloc[0]) if not row.empty else np.nan
            color = "#f0f0f0" if not np.isfinite(v) else cmap(norm(v))
            ax.bar(
                theta[j], 0.85, width=width * 0.92, bottom=r0,
                color=color, edgecolor="white", linewidth=0.4, align="edge",
            )

    ax.set_ylim(0, n_rings + 0.9)
    ax.set_yticks(np.arange(0.7, n_rings + 0.7, 1.0))
    ax.set_yticklabels([f"N{niche_short(n)}" for n in niches], fontsize=7)
    ax.set_xticks(theta + width / 2)
    ax.set_xticklabels([f"sg{p}" for p in perts], fontsize=6)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=12)
    ax.grid(False)


def build_figure(
    enrich_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    results_dir: Path,
    tag: str,
) -> plt.Figure:
    sh.apply_publication_style()
    fig = plt.figure(figsize=(16, 13))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.15, 0.85], hspace=0.22)

    gs_d = gridspec.GridSpecFromSubplotSpec(
        len(HEADLINE_CASES), 2, subplot_spec=outer[0], wspace=0.12, hspace=0.28,
    )
    fig.text(0.02, 0.96, "D", fontsize=16, fontweight="bold", va="top")
    fig.text(
        0.05, 0.955,
        "CNN β-microniches on tumor tissue with observed sgRNA density (left) and SpaceTravLR predicted enrichment field (right)",
        fontsize=10, va="top",
    )

    for row, (sl, pert) in enumerate(HEADLINE_CASES):
        spatial = load_spatial(results_dir, sl, tag)
        enrich_sub = enrich_df[(enrich_df["slice"] == sl) & (enrich_df["perturbation"] == pert)].copy()
        sub_c = corr_df[(corr_df["slice"] == sl) & (corr_df["perturbation"] == pert)]
        r = float(sub_c["pearson_r"].iloc[0]) if not sub_c.empty else float("nan")

        ax_main = fig.add_subplot(gs_d[row, 0])
        ax_inset = ax_main.inset_axes([0.68, 0.68, 0.30, 0.30])
        plot_spatial_panel_d(
            ax_main, ax_inset, spatial, enrich_sub, pert,
            title=f"{sl}  |  sg{pert} microniches  |  r={r:+.2f}",
            mode="observed",
        )

        ax_pred = fig.add_subplot(gs_d[row, 1])
        ax_pred_dummy = ax_pred.inset_axes([0.68, 0.68, 0.30, 0.30])
        plot_spatial_panel_d(
            ax_pred, ax_pred_dummy, spatial, enrich_sub, pert,
            title=f"SpaceTravLR predicted enrichment  |  sg{pert}",
            mode="predicted",
        )

    gs_e = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1.1, 1.1, 0.9], wspace=0.25)
    fig.text(0.02, 0.46, "E", fontsize=16, fontweight="bold", va="top")
    fig.text(
        0.05, 0.445,
        "sgRNA enrichment across CNN β-microniches vs non-targeting control (observed log₂ OR vs SpaceTravLR predicted score)",
        fontsize=10, va="top",
    )

    polar_enrich = enrich_df[enrich_df["slice"] == POLAR_SLICE].copy()
    excl = polar_enrich.groupby("niche")["pred_exclusion_index"].mean()
    lo, hi = excl.quantile(0.33), excl.quantile(0.67)
    niche_cats = {n: categorize_niche(v, lo, hi) for n, v in excl.items()}
    cat_colors = {
        "Immune exclusion": "#7b3294",
        "Immune infiltration": "#fdb863",
        "Mixed / stromal": "#c51b7d",
    }
    legend_patches = [mpatches.Patch(color=c, label=k) for k, c in cat_colors.items()]

    ax_obs = fig.add_subplot(gs_e[0], projection="polar")
    plot_polar_enrichment(
        ax_obs, polar_enrich, value_col="obs_log2_enrichment",
        title=f"Observed ({POLAR_SLICE})", perts=SUBQ_PERTS,
    )
    ax_pred = fig.add_subplot(gs_e[1], projection="polar")
    plot_polar_enrichment(
        ax_pred, polar_enrich, value_col="pred_enrichment_score",
        title=f"SpaceTravLR predicted ({POLAR_SLICE})", perts=SUBQ_PERTS,
    )

    ax_sc = fig.add_subplot(gs_e[2])
    if not enrich_df.empty:
        x = enrich_df["pred_enrichment_score"].to_numpy(float)
        y = enrich_df["obs_log2_enrichment"].to_numpy(float)
        cats = enrich_df["niche"].map(niche_cats).fillna("Mixed / stromal")
        for cat, color in cat_colors.items():
            m = cats == cat
            if m.any():
                ax_sc.scatter(x[m], y[m], c=color, s=28, alpha=0.75, edgecolors="k", linewidths=0.2, label=cat)
        if len(x) >= 3 and np.std(x) > 1e-8:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax_sc.plot(xs, coef[0] * xs + coef[1], "k--", lw=1.0, alpha=0.6)
        med_r = float(corr_df["pearson_r"].median()) if not corr_df.empty else float("nan")
        ax_sc.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax_sc.axvline(0, color="k", lw=0.4, alpha=0.4)
        ax_sc.set_xlabel("SpaceTravLR predicted enrichment")
        ax_sc.set_ylabel("Observed log₂ enrichment (sgP vs NTC)")
        ax_sc.set_title(f"Cohort concordance\nmedian r = {med_r:+.2f}", fontweight="bold", fontsize=9)
        ax_sc.legend(handles=legend_patches, fontsize=7, loc="lower right", frameon=False)

    sm = plt.cm.ScalarMappable(cmap=plt.colormaps["RdBu_r"], norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5))
    sm.set_array([])
    cax = fig.add_axes([0.38, 0.06, 0.24, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Enrichment score (red = enriched, blue = depleted)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "SpaceTravLR predicts spatial sgRNA enrichment across functional tumor microniches",
        fontsize=13, fontweight="bold", y=0.995,
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_v2")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "cnn_enrichment")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "cnn_microniche_v2_improved")
    args = ap.parse_args()

    enrich_path = args.results_dir / f"niche_enrichment_{args.tag}.csv"
    corr_path = args.results_dir / f"enrichment_corr_{args.tag}.csv"
    enrich_df = pd.read_csv(enrich_path)
    corr_df = pd.read_csv(corr_path)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(enrich_df, corr_df, args.results_dir, args.tag)
    out_stem = args.fig_dir / f"fig29_paper_spatial_enrichment_{args.tag}"
    sh.save_figure_png_svg(fig, out_stem.with_suffix(".png"), dpi=300, transparent_png=True)
    plt.close(fig)
    print(f"Wrote {out_stem}.svg and {out_stem}.png")


if __name__ == "__main__":
    main()
