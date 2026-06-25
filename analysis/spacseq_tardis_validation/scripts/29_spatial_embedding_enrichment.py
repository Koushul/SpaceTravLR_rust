#!/usr/bin/env python3
"""Side-by-side spatial embedding maps: microniche | predicted | observed ground truth.

Analysis logic
--------------
* Microniches: CNN β-Leiden clusters on tumor NTC cells (functional zones).
* Predicted enrichment: niche-level SpaceTravLR composite score mapped to every
  tumor cell in that niche (computed from NTC exclusion + KO escape + CNN β).
* Ground truth (primary): observed log₂ OR of sgP vs NTC tumor fraction per
  niche, mapped to cells — matches the Pearson correlation metric.
* Ground truth (cell-level): local sgP fraction among spatial kNN neighbors
  (4th column) — continuous “where guides landed” field.

Concordance is clear when predicted and observed niche colors align on tissue.
"""

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
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import cnn_microniche_utils as cmu
import spatial_histology as sh

HEADLINE_CASES = [
    ("subQ-2", "Cd83"),
    ("subQ-4", "Ptk6"),
    ("Lung_Metastasis_M001", "Icam1"),
]


def niche_palette(labels: list[str]) -> dict[str, tuple]:
    cmap = plt.colormaps.get_cmap("tab20")
    uniq = sorted(set(labels))
    return {lab: cmap(i % 20) for i, lab in enumerate(uniq)}


def load_tumor_embedding_adata(
    results_dir: Path,
    slice_id: str,
    tag: str,
    enrich_sub: pd.DataFrame,
    perturb: str,
    *,
    k_neighbors: int = 50,
) -> sc.AnnData:
    import scanpy as sc

    pq = results_dir / f"spatial_tumor_{slice_id}_{tag}.parquet"
    adata = sh.tumor_adata_from_parquet(pq, slice_id)
    if enrich_sub.empty:
        return adata
    scored = enrich_sub[
        ["niche", "pred_enrichment_score", "obs_log2_enrichment"]
    ].drop_duplicates("niche").set_index("niche")
    adata.obs["pred_enrichment_score"] = adata.obs["cnn_leiden"].astype(str).map(scored["pred_enrichment_score"])
    adata.obs["obs_log2_enrichment"] = adata.obs["cnn_leiden"].astype(str).map(scored["obs_log2_enrichment"])
    adata.obs["microniche"] = adata.obs["cnn_leiden"].astype(str).map(cmu.niche_short_label)
    is_sgp = (adata.obs["target_gene"].astype(str) == perturb).to_numpy(dtype=float)
    xy = adata.obsm["spatial"].astype(np.float64)
    k = min(k_neighbors, adata.n_obs - 1)
    if k >= 3:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
        _, idx = nn.kneighbors(xy)
        adata.obs["local_sgp_frac"] = is_sgp[idx[:, 1:]].mean(axis=1)
    else:
        adata.obs["local_sgp_frac"] = is_sgp
    return adata


def enrichment_limits(adata, keys: tuple[str, ...]) -> tuple[float, float]:
    vals = []
    for k in keys:
        if k in adata.obs.columns:
            vals.extend(pd.to_numeric(adata.obs[k], errors="coerce").dropna().tolist())
    if not vals:
        return -1.0, 1.0
    lim = max(0.35, float(np.percentile(np.abs(vals), 95)))
    return -lim, lim


def _microniche_colors(enrich_sub: pd.DataFrame) -> list:
    palette = niche_palette(enrich_sub["niche"].astype(str).map(cmu.niche_short_label).tolist())
    return [palette.get(cmu.niche_short_label(n), "#2563eb") for n in enrich_sub["niche"]]


def plot_triplet_row(
    adata,
    slice_id: str,
    perturb: str,
    enrich_sub: pd.DataFrame,
    pearson_r: float,
    axes: list,
    *,
    mc38_dir: Path,
    show_local_sgp: bool = True,
) -> None:
    if adata.n_obs == 0:
        for ax in axes:
            ax.axis("off")
        return

    vmin, vmax = enrichment_limits(adata, ("pred_enrichment_score", "obs_log2_enrichment"))
    palette = niche_palette(adata.obs["microniche"].astype(str).tolist())
    use_he = sh.histology_ready(mc38_dir, slice_id)
    tumor = adata
    if use_he:
        tumor = sh.attach_histology(tumor.copy(), slice_id, mc38_dir, skip_download=True)

    col_keys = ["microniche", "pred_enrichment_score", "obs_log2_enrichment"]
    col_titles = [
        "CNN β-microniche",
        "SpaceTravLR predicted",
        "Observed (ground truth)",
    ]
    for ax, key, title in zip(axes[:3], col_keys, col_titles):
        if key == "microniche":
            if use_he and "spatial" in tumor.uns:
                sh.plot_microniche_on_he(
                    tumor, key, ax, slice_id, palette, title=title, legend=False,
                )
            else:
                sh.plot_embedding_spatial(
                    tumor, key, ax, categorical=True, palette=palette,
                    title=title, size=6, colorbar=False,
                )
        elif use_he and "spatial" in tumor.uns:
            sh.plot_continuous_on_he(
                tumor, key, ax, slice_id, vmin=vmin, vmax=vmax, vcenter=0.0,
                title=title,
                colorbar_label="Predicted score" if "pred" in key else "log₂ OR (sgP vs NTC)",
            )
        else:
            sh.plot_embedding_spatial(
                tumor, key, ax, vmin=vmin, vmax=vmax, vcenter=0.0,
                title=title,
                colorbar_label="Predicted score" if "pred" in key else "log₂ OR",
                size=6,
            )

    j = 3
    if show_local_sgp and len(axes) > j:
        ax_l = axes[j]
        vmax_l = float(adata.obs["local_sgp_frac"].quantile(0.98)) if adata.obs["local_sgp_frac"].notna().any() else 0.2
        vmax_l = max(vmax_l, 0.05)
        if use_he and "spatial" in tumor.uns:
            sh.plot_continuous_on_he(
                tumor, "local_sgp_frac", ax_l, slice_id,
                cmap="YlOrRd", vmin=0, vmax=vmax_l, vcenter=None,
                title="Local sgP fraction (kNN)", colorbar_label="sgP frac",
            )
        else:
            sh.plot_embedding_spatial(
                tumor, "local_sgp_frac", ax_l, cmap="YlOrRd", vmin=0, vmax=vmax_l,
                vcenter=None, title="Local sgP fraction (kNN)", colorbar_label="sgP frac", size=6,
            )
        j += 1

    if len(axes) > j:
        ax_sc = axes[j]
        if not enrich_sub.empty:
            cols = _microniche_colors(enrich_sub)
            ax_sc.scatter(
                enrich_sub["pred_enrichment_score"], enrich_sub["obs_log2_enrichment"],
                c=cols, s=80, edgecolors="k", linewidths=0.4, zorder=3,
            )
            for _, pt in enrich_sub.iterrows():
                ax_sc.annotate(
                    cmu.niche_short_label(pt["niche"]),
                    (pt["pred_enrichment_score"], pt["obs_log2_enrichment"]),
                    fontsize=8, ha="center", va="bottom",
                )
            if len(enrich_sub) >= 3:
                x = enrich_sub["pred_enrichment_score"].to_numpy()
                y = enrich_sub["obs_log2_enrichment"].to_numpy()
                if np.std(x) > 1e-8:
                    m, b = np.polyfit(x, y, 1)
                    xs = np.linspace(x.min(), x.max(), 40)
                    ax_sc.plot(xs, m * xs + b, "k--", lw=1, alpha=0.7)
        ax_sc.axhline(0, color="k", lw=0.4, alpha=0.35)
        ax_sc.axvline(0, color="k", lw=0.4, alpha=0.35)
        ax_sc.set_xlabel("Predicted enrichment")
        ax_sc.set_ylabel("Observed log₂ OR")
        ax_sc.set_title(f"sg{perturb} | {slice_id}\nr = {pearson_r:+.2f}", fontweight="bold", fontsize=9)
        ax_sc.grid(True, alpha=0.25)


def build_figure(
    enrich_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    results_dir: Path,
    mc38_dir: Path,
    tag: str,
    *,
    cases: list[tuple[str, str]] | None = None,
    show_local_sgp: bool = True,
) -> plt.Figure:
    sh.apply_publication_style()
    cases = cases or HEADLINE_CASES
    ncols = 5 if show_local_sgp else 4
    fig, axes = plt.subplots(
        len(cases), ncols,
        figsize=(3.15 * ncols, 3.5 * len(cases)),
        squeeze=False,
        gridspec_kw={"wspace": 0.3, "hspace": 0.35},
    )

    header = [
        "CNN β-microniche",
        "SpaceTravLR predicted",
        "Observed (ground truth)",
    ]
    if show_local_sgp:
        header.append("Local sgP density (kNN)")
    header.append("Niche concordance")

    for row, (sl, pert) in enumerate(cases):
        enrich_sub = enrich_df[(enrich_df["slice"] == sl) & (enrich_df["perturbation"] == pert)].copy()
        sub_c = corr_df[(corr_df["slice"] == sl) & (corr_df["perturbation"] == pert)]
        r = float(sub_c["pearson_r"].iloc[0]) if not sub_c.empty else float("nan")

        try:
            adata = load_tumor_embedding_adata(results_dir, sl, tag, enrich_sub, pert)
        except FileNotFoundError as e:
            print(f"Skipping {sl}/{pert}: {e}")
            for ax in axes[row]:
                ax.axis("off")
            continue

        plot_triplet_row(
            adata, sl, pert, enrich_sub, r, list(axes[row]),
            mc38_dir=mc38_dir, show_local_sgp=show_local_sgp,
        )

    for j, title in enumerate(header[:ncols]):
        axes[0, j].text(
            0.5, 1.1, title, transform=axes[0, j].transAxes,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    fig.suptitle(
        "Spatial embedding: SpaceTravLR predicted vs observed niche enrichment",
        fontsize=13, fontweight="bold", y=1.02,
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_v2")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "cnn_enrichment")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "cnn_microniche_v2_improved")
    ap.add_argument("--mc38-dir", type=Path, default=(ROOT.parent / "mc38_visiumhd").resolve())
    ap.add_argument("--no-local-sgp", action="store_true")
    args = ap.parse_args()

    enrich_df = pd.read_csv(args.results_dir / f"niche_enrichment_{args.tag}.csv")
    corr_df = pd.read_csv(args.results_dir / f"enrichment_corr_{args.tag}.csv")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(
        enrich_df, corr_df, args.results_dir, args.mc38_dir, args.tag,
        show_local_sgp=not args.no_local_sgp,
    )
    out = args.fig_dir / f"fig30_spatial_embedding_enrichment_{args.tag}"
    sh.save_figure_png_svg(fig, out.with_suffix(".png"), dpi=300, transparent_png=True)
    plt.close(fig)
    print(f"Wrote {out}.svg")


if __name__ == "__main__":
    main()
