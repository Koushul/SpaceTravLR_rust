#!/usr/bin/env python3
"""Side-by-side spatial embedding maps: microniche | predicted | observed ground truth."""

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
from matplotlib.colors import TwoSlopeNorm
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

SLICE_ORDER = ["subQ-1", "subQ-2", "subQ-3", "subQ-4", "Lung_Metastasis_M001"]
PERT_ORDER = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b", "Ptk6", "Icam1"]

COL_HEADERS = [
    "CNN β-microniche",
    "SpaceTravLR predicted",
    "Observed (ground truth)",
    "Local sgP density (kNN)",
    "Niche concordance",
]


def all_cases_from_corr(corr_df: pd.DataFrame) -> list[tuple[str, str]]:
    sl_rank = {s: i for i, s in enumerate(SLICE_ORDER)}
    pert_rank = {p: i for i, p in enumerate(PERT_ORDER)}
    rows = corr_df[["slice", "perturbation"]].drop_duplicates()
    pairs = [(str(r.slice), str(r.perturbation)) for r in rows.itertuples(index=False)]
    pairs.sort(key=lambda t: (sl_rank.get(t[0], 99), pert_rank.get(t[1], 99), t[0], t[1]))
    return pairs


def niche_palette(labels: list[str]) -> dict[str, tuple]:
    cmap = plt.colormaps.get_cmap("tab20")
    uniq = sorted(set(labels))
    return {lab: cmap(i % 20) for i, lab in enumerate(uniq)}


def global_enrichment_limits(enrich_df: pd.DataFrame) -> tuple[float, float]:
    vals = pd.concat([
        enrich_df["pred_enrichment_score"],
        enrich_df["obs_log2_enrichment"],
    ], ignore_index=True)
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if vals.empty:
        return -1.0, 1.0
    lim = max(0.5, float(np.percentile(np.abs(vals), 96)))
    return -lim, lim


def load_tumor_embedding_adata(
    results_dir: Path,
    slice_id: str,
    tag: str,
    enrich_sub: pd.DataFrame,
    perturb: str,
    *,
    k_neighbors: int = 50,
    base_adata=None,
):
    if base_adata is None:
        pq = results_dir / f"spatial_tumor_{slice_id}_{tag}.parquet"
        adata = sh.tumor_adata_from_parquet(pq, slice_id)
    else:
        adata = base_adata.copy()
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


def _microniche_colors(enrich_sub: pd.DataFrame) -> list:
    palette = niche_palette(enrich_sub["niche"].astype(str).map(cmu.niche_short_label).tolist())
    return [palette.get(cmu.niche_short_label(n), "#2563eb") for n in enrich_sub["niche"]]


def _plot_scatter_panel(ax, enrich_sub: pd.DataFrame, pearson_r: float, *, compact: bool) -> None:
    if enrich_sub.empty:
        ax.axis("off")
        return
    cols = _microniche_colors(enrich_sub)
    ax.scatter(
        enrich_sub["pred_enrichment_score"], enrich_sub["obs_log2_enrichment"],
        c=cols, s=42 if compact else 70, edgecolors="k", linewidths=0.35, zorder=3,
    )
    if not compact and len(enrich_sub) <= 5:
        for _, pt in enrich_sub.iterrows():
            ax.annotate(
                cmu.niche_short_label(pt["niche"]),
                (pt["pred_enrichment_score"], pt["obs_log2_enrichment"]),
                fontsize=7, ha="center", va="bottom",
            )
    if len(enrich_sub) >= 3:
        x = enrich_sub["pred_enrichment_score"].to_numpy()
        y = enrich_sub["obs_log2_enrichment"].to_numpy()
        if np.std(x) > 1e-8:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 40)
            ax.plot(xs, m * xs + b, color="#374151", lw=0.9, ls="--", alpha=0.75)
    ax.axhline(0, color="#9ca3af", lw=0.45)
    ax.axvline(0, color="#9ca3af", lw=0.45)
    ax.set_xlabel("Predicted", fontsize=7 if compact else 8)
    ax.set_ylabel("Observed log₂ OR", fontsize=7 if compact else 8)
    ax.set_title(f"r = {pearson_r:+.2f}", fontsize=8 if compact else 9, fontweight="bold", pad=2)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2, linewidth=0.4)


def plot_row_panels(
    adata,
    slice_id: str,
    perturb: str,
    enrich_sub: pd.DataFrame,
    pearson_r: float,
    axes: list,
    *,
    mc38_dir: Path,
    tumor_he=None,
    vmin: float,
    vmax: float,
    show_microniche: bool = True,
    show_colorbar: bool = False,
    compact: bool = True,
    rasterize: bool = True,
) -> None:
    if adata.n_obs == 0:
        for ax in axes:
            ax.axis("off")
        return

    palette = niche_palette(adata.obs["microniche"].astype(str).tolist())
    use_he = tumor_he is not None or sh.histology_ready(mc38_dir, slice_id)
    tumor = tumor_he.copy() if tumor_he is not None else adata
    if use_he and tumor_he is None:
        tumor = sh.attach_histology(adata.copy(), slice_id, mc38_dir, skip_download=True)
    elif tumor_he is not None:
        for col in ("pred_enrichment_score", "obs_log2_enrichment", "local_sgp_frac", "microniche"):
            if col in adata.obs.columns:
                tumor.obs[col] = adata.obs[col].values

    ax_n, ax_p, ax_o, ax_l, ax_sc = axes

    if show_microniche:
        if use_he and "spatial" in tumor.uns:
            sh.plot_microniche_on_he(
                tumor, "microniche", ax_n, slice_id, palette,
                title="", legend=False, rasterize=rasterize,
            )
        else:
            sh.plot_embedding_spatial(
                tumor, "microniche", ax_n, categorical=True, palette=palette,
                title="", size=5 if compact else 6, colorbar=False, rasterize=rasterize,
            )
    else:
        ax_n.axis("off")
        ax_n.text(
            0.5, 0.5, "same tissue\n(see above)", ha="center", va="center",
            transform=ax_n.transAxes, fontsize=7, color="#6b7280", style="italic",
        )

    for ax, key in ((ax_p, "pred_enrichment_score"), (ax_o, "obs_log2_enrichment")):
        if use_he and "spatial" in tumor.uns:
            sh.plot_continuous_on_he(
                tumor, key, ax, slice_id, vmin=vmin, vmax=vmax, vcenter=0.0,
                title="", colorbar=show_colorbar, rasterize=rasterize,
            )
        else:
            sh.plot_embedding_spatial(
                tumor, key, ax, vmin=vmin, vmax=vmax, vcenter=0.0,
                title="", colorbar=show_colorbar, size=5 if compact else 6, rasterize=rasterize,
            )

    vmax_l = float(adata.obs["local_sgp_frac"].quantile(0.98)) if adata.obs["local_sgp_frac"].notna().any() else 0.2
    vmax_l = max(vmax_l, 0.05)
    if use_he and "spatial" in tumor.uns:
        sh.plot_continuous_on_he(
            tumor, "local_sgp_frac", ax_l, slice_id,
            cmap="YlOrRd", vmin=0, vmax=vmax_l, vcenter=None,
            title="", colorbar=show_colorbar, rasterize=rasterize,
        )
    else:
        sh.plot_embedding_spatial(
            tumor, "local_sgp_frac", ax_l, cmap="YlOrRd", vmin=0, vmax=vmax_l,
            vcenter=None, title="", colorbar=show_colorbar, size=5 if compact else 6, rasterize=rasterize,
        )

    _plot_scatter_panel(ax_sc, enrich_sub, pearson_r, compact=compact)


def _slice_base_cache(results_dir: Path, tag: str) -> dict[str, object]:
    cache: dict[str, object] = {}
    for sl in SLICE_ORDER:
        pq = results_dir / f"spatial_tumor_{sl}_{tag}.parquet"
        if pq.exists():
            cache[sl] = sh.tumor_adata_from_parquet(pq, sl)
    return cache


def build_figure(
    enrich_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    results_dir: Path,
    mc38_dir: Path,
    tag: str,
    *,
    cases: list[tuple[str, str]] | None = None,
    compact: bool = True,
    row_height: float = 2.5,
) -> plt.Figure:
    sh.apply_publication_style()
    cases = cases or HEADLINE_CASES
    nrows = len(cases)
    ncols = 5
    compact = compact or nrows > 6
    rasterize = nrows > 3
    vmin, vmax = global_enrichment_limits(enrich_df)

    fig = plt.figure(figsize=(14.5, max(4.0, row_height * nrows + 1.2)))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        left=0.14, right=0.97, top=0.91, bottom=0.02,
        wspace=0.24, hspace=0.48 if compact else 0.35,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)])

    slice_cache = _slice_base_cache(results_dir, tag)
    he_cache: dict[str, object] = {}
    prev_slice: str | None = None
    slice_row_starts: list[int] = []

    for row, (sl, pert) in enumerate(cases):
        enrich_sub = enrich_df[(enrich_df["slice"] == sl) & (enrich_df["perturbation"] == pert)].copy()
        sub_c = corr_df[(corr_df["slice"] == sl) & (corr_df["perturbation"] == pert)]
        r = float(sub_c["pearson_r"].iloc[0]) if not sub_c.empty else float("nan")

        base = slice_cache.get(sl)
        if base is None:
            for ax in axes[row]:
                ax.axis("off")
            continue

        adata = load_tumor_embedding_adata(
            results_dir, sl, tag, enrich_sub, pert, base_adata=base,
        )
        if sl not in he_cache and sh.histology_ready(mc38_dir, sl):
            he_cache[sl] = sh.attach_histology(base.copy(), sl, mc38_dir, skip_download=True)

        show_microniche = sl != prev_slice
        if show_microniche:
            slice_row_starts.append(row)
        prev_slice = sl

        plot_row_panels(
            adata, sl, pert, enrich_sub, r, list(axes[row]),
            mc38_dir=mc38_dir,
            tumor_he=he_cache.get(sl),
            vmin=vmin, vmax=vmax,
            show_microniche=show_microniche,
            show_colorbar=(row == 0),
            compact=compact,
            rasterize=rasterize,
        )

        pos = axes[row, 0].get_position()
        fig.text(
            0.012, pos.y0 + pos.height / 2,
            f"{sl}\nsg{pert}", ha="left", va="center", fontsize=7, fontweight="bold",
        )

    for row in slice_row_starts[1:]:
        y = axes[row, 0].get_position().y1 + 0.012
        fig.add_artist(plt.Line2D([0.11, 0.98], [y, y], transform=fig.transFigure, color="#d1d5db", lw=0.6))

    header_y = min(ax.get_position().y1 for ax in axes[0]) + 0.018
    for j, title in enumerate(COL_HEADERS):
        pos = axes[0, j].get_position()
        fig.text(pos.x0 + pos.width / 2, header_y, title, ha="center", va="bottom", fontsize=9, fontweight="bold")

    cbar_y = header_y + 0.028
    cbar_w = axes[0, 1].get_position().width
    cbar_x = axes[0, 1].get_position().x0
    cax_enrich = fig.add_axes([cbar_x, cbar_y, cbar_w * 2.05, 0.008])
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax_enrich, orientation="horizontal")
    cb.set_label("Predicted score / observed log₂ OR", fontsize=8)
    cb.ax.tick_params(labelsize=6)

    med_r = float(corr_df["pearson_r"].median()) if not corr_df.empty else float("nan")
    fig.suptitle(
        f"SpaceTravLR predicted vs observed niche enrichment  "
        f"(n={nrows}, median r={med_r:+.2f})",
        fontsize=12, fontweight="bold", y=0.97,
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_v2")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "cnn_enrichment")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "cnn_microniche_v2_improved")
    ap.add_argument("--mc38-dir", type=Path, default=(ROOT.parent / "mc38_visiumhd").resolve())
    ap.add_argument("--headline-only", action="store_true")
    ap.add_argument("--row-height", type=float, default=None)
    args = ap.parse_args()

    enrich_df = pd.read_csv(args.results_dir / f"niche_enrichment_{args.tag}.csv")
    corr_df = pd.read_csv(args.results_dir / f"enrichment_corr_{args.tag}.csv")

    if args.headline_only:
        cases = HEADLINE_CASES
        row_height = args.row_height or 3.2
        compact = False
        out_stem = f"fig30_spatial_embedding_enrichment_{args.tag}_headline"
    else:
        cases = all_cases_from_corr(corr_df)
        row_height = args.row_height or 2.55
        compact = True
        out_stem = f"fig30_spatial_embedding_enrichment_{args.tag}"

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(
        enrich_df, corr_df, args.results_dir, args.mc38_dir, args.tag,
        cases=cases, compact=compact, row_height=row_height,
    )
    out = args.fig_dir / out_stem
    dpi = 180 if len(cases) > 6 else 250
    sh.save_figure_png_svg(fig, out.with_suffix(".png"), dpi=dpi, transparent_png=False)
    plt.close(fig)
    svg_mb = out.with_suffix(".svg").stat().st_size / 1e6 if out.with_suffix(".svg").exists() else 0
    print(f"Wrote {out}.svg ({len(cases)} panels, {svg_mb:.1f} MB)")


if __name__ == "__main__":
    main()
