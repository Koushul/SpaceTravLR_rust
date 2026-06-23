#!/usr/bin/env python3
"""Spatial validation: graphclust niches, grid bins, and tissue maps.

Tests whether SpaceTravLR predicted KO effects match observed SPAC-seq effects
*within spatial microniches* — not only after pooling across the whole section.

For each (slice, perturbation, cell type, niche):
  obs_niche_delta[g]  = mean(sgP in niche) - mean(NTC in niche)
  pred_niche_delta[g] = mean(pred_KO NTC in niche) - mean(baseline NTC in niche)

Pearson r across genes quantifies local concordance. With a pooled model,
predicted fields cover NTC cells from all slices (matched by CellID).

Also stratifies by distance-to-nearest-immune-cell to test whether concordance
peaks at immune–tumor interfaces (where Il4ra / Cd83 biology is active).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("fig05", HERE / "05_final_report_figures.py")
_fig05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fig05)

compute_metrics = _fig05.compute_metrics
dense = _fig05.dense

DEFAULT_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]
DEFAULT_PERTS = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b"]
PX_PER_UM = 1.0 / 0.273
GRID_UM = 120.0


def load_baseline(baseline_path: Path) -> sc.AnnData:
    if baseline_path.is_dir():
        baseline_path = sorted(baseline_path.glob("*.h5ad"))[0]
    ad = sc.read_h5ad(baseline_path)
    if "imputed_count" in ad.layers:
        ad.X = ad.layers["imputed_count"]
    return ad


def load_pool(slice_name: str, data_root: Path) -> sc.AnnData:
    pool = sc.read_h5ad(data_root / "slices" / slice_name / "perturbed_pool.h5ad")
    sc.pp.normalize_total(pool, target_sum=10000)
    sc.pp.log1p(pool)
    return pool


def niche_vectors(
    baseline: sc.AnnData,
    pool: sc.AnnData,
    pred: pd.DataFrame,
    perturb: str,
    cell_type: str,
    niche_key: str,
    genes: list[str],
    min_ntc: int,
    min_pert: int,
    fallback_pred_delta: pd.Series | None = None,
) -> list[dict]:
    """Return per-niche obs/pred delta vectors and Pearson r."""
    common = [g for g in genes if g in pool.var_names and g in pred.columns]
    if fallback_pred_delta is not None:
        common = [g for g in common if g in fallback_pred_delta.index]
    else:
        common = [g for g in common if g in baseline.var_names]
    if len(common) < 30:
        return []

    pool_ct = pool[pool.obs["cell_type"].astype(str) == cell_type].copy()
    if niche_key not in pool_ct.obs.columns:
        return []
    pool_expr = dense(pool_ct, common)
    ntc_mask = pool_ct.obs["target_gene"].astype(str) == "non-targeting"
    pert_mask = pool_ct.obs["target_gene"].astype(str) == perturb

    base_sub = None
    pred_delta_cells = None
    if fallback_pred_delta is None:
        base_sub = baseline[baseline.obs["cell_type"].astype(str) == cell_type].copy()
        if "slice_id" in base_sub.obs.columns and "slice_id" in pool.obs.columns:
            sl = str(pool.obs["slice_id"].iloc[0])
            base_sub = base_sub[base_sub.obs["slice_id"].astype(str) == sl]
        base_sub = base_sub[base_sub.obs_names.isin(pred.index)]
        if base_sub.n_obs == 0:
            return None  # signal caller to use fallback
        pred_sub = pred.loc[base_sub.obs_names, common]
        base_expr = dense(base_sub, common)
        pred_delta_cells = pred_sub - base_expr

    rows = []
    for niche in sorted(pool_ct.obs[niche_key].astype(str).unique()):
        if niche == "nan" or niche == "":
            continue
        ntc_n = pool_ct.obs[niche_key].astype(str).eq(niche) & ntc_mask
        per_n = pool_ct.obs[niche_key].astype(str).eq(niche) & pert_mask
        if ntc_n.sum() < min_ntc or per_n.sum() < min_pert:
            continue

        obs_d = pool_expr.loc[per_n].mean(0) - pool_expr.loc[ntc_n].mean(0)

        if fallback_pred_delta is not None:
            pred_d = fallback_pred_delta.loc[common]
        else:
            base_n = base_sub.obs[niche_key].astype(str).eq(niche)
            if base_n.sum() < min_ntc:
                continue
            pred_d = pred_delta_cells.loc[base_n].mean(0)

        panel = [g for g in common if g != perturb]
        m = compute_metrics(pred_d.loc[panel], obs_d.loc[panel], n_perm=500, seed=hash(niche) % 2**31)
        if m is None:
            continue
        rows.append({
            "niche": niche,
            "n_ntc": int(ntc_n.sum()),
            "n_pert": int(per_n.sum()),
            "n_baseline_ntc": int(base_n.sum()) if fallback_pred_delta is None else 0,
            "pred_mode": "niche" if fallback_pred_delta is None else "celltype_fallback",
            **m,
            "obs_on_target": float(obs_d[perturb]) if perturb in obs_d.index else np.nan,
            "pred_on_target": float(pred_d[perturb]) if perturb in pred_d.index else np.nan,
        })
    return rows


def spatial_grid_labels(xy: np.ndarray, grid_um: float) -> np.ndarray:
    step = grid_um * PX_PER_UM
    gx = np.floor(xy[:, 0] / step).astype(int)
    gy = np.floor(xy[:, 1] / step).astype(int)
    return np.char.add(np.char.add(gx.astype(str), "_"), gy.astype(str))


def add_grid_niche(obs: pd.DataFrame, xy: np.ndarray, grid_um: float) -> pd.Series:
    return pd.Series(spatial_grid_labels(xy, grid_um), index=obs.index, name="spatial_grid")


def immune_distance_bins(pool: sc.AnnData, n_bins: int = 4) -> pd.Series:
    xy = pool.obsm["spatial"]
    immune = pool.obs["cell_type"].astype(str).eq("immune").values
    if immune.sum() < 20:
        return pd.Series("all", index=pool.obs_names)
    nn = NearestNeighbors(n_neighbors=1).fit(xy[immune])
    dist, _ = nn.kneighbors(xy)
    dist_um = dist[:, 0] / PX_PER_UM
    try:
        bins = pd.qcut(dist_um, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(dist_um, bins=n_bins)
    return pd.Series(bins, index=pool.obs_names, name="immune_dist_bin")


def pool_to_pred_index(pool: sc.AnnData, pred_index: pd.Index) -> pd.Index:
    """Map pool cell barcodes to prediction index (handles @slice suffix from pooled prep)."""
    if pred_index.str.contains("@", regex=False).any():
        sl = pool.obs["slice_id"].astype(str) if "slice_id" in pool.obs.columns else pd.Series("", index=pool.obs_names)
        mapped = pd.Index([f"{b}@{s}" for b, s in zip(pool.obs_names, sl)])
        return mapped
    return pool.obs_names


def attach_graphclust(baseline: sc.AnnData, pool: sc.AnnData, data_root: Path) -> tuple[sc.AnnData, sc.AnnData]:
    sl = str(pool.obs["slice_id"].iloc[0])
    ann_path = ROOT.parent / "mc38_visiumhd" / sl / "processed" / f"{sl}_cells_annotated.h5ad"
    if not ann_path.exists():
        return baseline, pool
    ann = sc.read_h5ad(ann_path, backed="r")
    gc = ann.obs["graphclust"].astype(str)
    suffix = f"@{sl}" if baseline.obs_names.str.contains("@", regex=False).any() else ""
    gc_pool = pool.obs_names.map(gc).astype(str)
    if suffix:
        gc_base = pd.Series(
            {f"{b}{suffix}": gc.get(b, np.nan) for b in gc.index},
            dtype=str,
        )
        if "graphclust" not in pool.obs.columns:
            pool.obs["graphclust"] = gc_pool.values
        if "graphclust" not in baseline.obs.columns:
            baseline = baseline.copy()
            baseline.obs["graphclust"] = baseline.obs_names.map(gc_base).astype(str)
    else:
        if "graphclust" not in pool.obs.columns:
            pool.obs["graphclust"] = gc_pool.values
        if "graphclust" not in baseline.obs.columns:
            baseline = baseline.copy()
            baseline.obs["graphclust"] = baseline.obs_names.map(gc).astype(str)
    return baseline, pool


def celltype_pred_delta(baseline, pred, perturb, cell_type, genes) -> pd.Series | None:
    common = [g for g in genes if g in baseline.var_names and g in pred.columns]
    base_sub = baseline[baseline.obs["cell_type"].astype(str) == cell_type]
    base_sub = base_sub[base_sub.obs_names.isin(pred.index)]
    if base_sub.n_obs < 10:
        return None
    pred_sub = pred.loc[base_sub.obs_names, common]
    base_expr = dense(base_sub, common)
    return (pred_sub.mean(0) - base_expr.mean(0))


def niche_rows_by_key(
    baseline, pool, pred, perturb, cell_type, niche_key, genes, min_ntc, min_pert,
) -> list[dict]:
    pool = pool.copy()
    fallback = None
    if niche_key == "spatial_grid":
        pool.obs["spatial_grid"] = add_grid_niche(pool.obs, pool.obsm["spatial"], GRID_UM)
        base_key = "spatial_grid"
        sl = str(pool.obs["slice_id"].iloc[0]) if "slice_id" in pool.obs.columns else ""
        if "slice_id" in baseline.obs.columns and sl:
            b = baseline[baseline.obs["slice_id"].astype(str) == sl].copy()
        else:
            b = baseline.copy()
        if b.n_obs == 0:
            return []
        b.obs["spatial_grid"] = add_grid_niche(b.obs, b.obsm["spatial"], GRID_UM)
        baseline = b
    elif niche_key == "immune_dist_bin":
        base_key = "immune_dist_bin"
        fallback = celltype_pred_delta(baseline, pred, perturb, cell_type, genes)
        out = niche_vectors(
            baseline, pool, pred, perturb, cell_type, base_key, genes, min_ntc, min_pert,
            fallback_pred_delta=fallback,
        )
        for r in out:
            r["niche_type"] = niche_key
        return out
    else:
        base_key = niche_key
        baseline, pool = attach_graphclust(baseline, pool, ROOT / "data")

    out = niche_vectors(
        baseline, pool, pred, perturb, cell_type, base_key, genes, min_ntc, min_pert,
    )
    if out is None:
        fallback = celltype_pred_delta(baseline, pred, perturb, cell_type, genes)
        if fallback is None:
            return []
        out = niche_vectors(
            baseline, pool, pred, perturb, cell_type, base_key, genes, min_ntc, min_pert,
            fallback_pred_delta=fallback,
        )
    for r in out:
        r["niche_type"] = niche_key
    return out


def plot_spatial_maps(
    slice_name: str,
    perturb: str,
    cell_type: str,
    baseline: sc.AnnData,
    pool: sc.AnnData,
    pred: pd.DataFrame,
    genes: list[str],
    fig_dir: Path,
    tag: str,
) -> None:
    on_g = perturb
    off_genes = [g for g in ["H2-Aa", "H2-Ab1", "Cd74", "Apoe", "B2m"] if g in genes and g != perturb][:1]
    show_g = off_genes[0] if off_genes else on_g

    sl_base = baseline
    if "slice_id" in baseline.obs.columns:
        sl_base = baseline[baseline.obs["slice_id"].astype(str) == slice_name]
    sl_base = sl_base[sl_base.obs["cell_type"].astype(str) == cell_type]
    sl_base = sl_base[sl_base.obs_names.isin(pred.index)]
    if sl_base.n_obs < 10:
        return

    pool_ct = pool[(pool.obs["cell_type"].astype(str) == cell_type)]
    ntc = pool_ct[pool_ct.obs["target_gene"].astype(str) == "non-targeting"]
    pert = pool_ct[pool_ct.obs["target_gene"].astype(str) == perturb]
    if len(pert) < 20:
        return

    ref = float(dense(ntc, [show_g]).mean().iloc[0]) if show_g in ntc.var_names else 0.0
    pert_expr = dense(pert, [show_g])[show_g] - ref if show_g in pert.var_names else pd.Series(0, index=pert.obs_names)

    pred_expr = (pred.loc[sl_base.obs_names, show_g] - dense(sl_base, [show_g])[show_g].values
                 if show_g in pred.columns else pd.Series(0, index=sl_base.obs_names))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    vmax = np.nanpercentile(np.abs(np.concatenate([pert_expr.values, pred_expr.values])), 95)
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    ax = axes[0]
    ax.scatter(
        pert.obsm["spatial"][:, 0], pert.obsm["spatial"][:, 1],
        c=pert_expr, s=1.2, cmap="RdBu_r", norm=norm, rasterized=True,
    )
    ax.set_title(f"Observed Δ {show_g}\n(sg{perturb} − NTC ref)", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")

    ax = axes[1]
    ax.scatter(
        sl_base.obsm["spatial"][:, 0], sl_base.obsm["spatial"][:, 1],
        c=pred_expr, s=1.2, cmap="RdBu_r", norm=norm, rasterized=True,
    )
    ax.set_title(f"Predicted Δ {show_g}\n(in-silico KO)", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")

    ax = axes[2]
    ax.scatter(ntc.obsm["spatial"][:, 0], ntc.obsm["spatial"][:, 1], c="#cccccc", s=0.3, rasterized=True)
    ax.scatter(pert.obsm["spatial"][:, 0], pert.obsm["spatial"][:, 1], c="#d62728", s=0.5, rasterized=True, alpha=0.6)
    ax.set_title(f"sg{perturb} cell locations\n({cell_type})", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle(f"{slice_name} | sg{perturb} | {cell_type} — spatial effect field ({tag})", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axes[:2], shrink=0.7, label=f"Δ {show_g}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"spatial_map_{slice_name}_{perturb}_{cell_type}_{tag}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--perturbations", nargs="+", default=DEFAULT_PERTS)
    ap.add_argument("--cell-types", nargs="+", default=["immune", "myeloid", "fibroblast"])
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/spatial")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/spatial")
    ap.add_argument("--tag", default="seed")
    ap.add_argument("--min-ntc", type=int, default=6)
    ap.add_argument("--min-pert", type=int, default=10)
    ap.add_argument("--make-maps", action="store_true")
    ap.add_argument("--map-pairs", nargs="*", default=[
        "subQ-1:Il4ra:immune", "subQ-1:Cd83:immune", "subQ-1:Cd83:fibroblast",
        "subQ-3:Il4ra:immune", "subQ-2:Bcam:myeloid",
    ])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline(args.baseline_h5ad)
    genes = sorted(set(baseline.var_names))

    all_rows = []
    immune_rows = []

    for sl in args.slices:
        pool_path = args.data_root / "slices" / sl / "perturbed_pool.h5ad"
        if not pool_path.exists():
            continue
        pool = load_pool(sl, args.data_root)
        pool.obs["slice_id"] = sl
        pool.obs["immune_dist_bin"] = immune_distance_bins(pool)

        for pert in args.perturbations:
            pred_path = args.pred_dir / f"predicted_KO_{pert}.feather"
            if not pred_path.exists():
                continue
            pred = pd.read_feather(pred_path).set_index("CellID")
            if (pool.obs["target_gene"].astype(str) == pert).sum() < args.min_pert:
                continue

            for ct in args.cell_types:
                for niche_key in ("graphclust", "spatial_grid"):
                    rows = niche_rows_by_key(
                        baseline, pool, pred, pert, ct, niche_key, genes,
                        args.min_ntc, args.min_pert,
                    )
                    for r in rows:
                        r.update({"slice": sl, "perturbation": pert, "cell_type": ct})
                    all_rows.extend(rows)

                rows = niche_rows_by_key(
                    baseline, pool, pred, pert, ct, "immune_dist_bin", genes,
                    max(5, args.min_ntc // 2), max(8, args.min_pert // 2),
                )
                for r in rows:
                    r.update({"slice": sl, "perturbation": pert, "cell_type": ct})
                immune_rows.extend(rows)

                if args.make_maps:
                    for spec in args.map_pairs:
                        psl, ppert, pct = spec.split(":")
                        if psl == sl and ppert == pert and pct == ct:
                            plot_spatial_maps(sl, pert, ct, baseline, pool, pred, genes, args.fig_dir, args.tag)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(args.out_dir / f"niche_corr_{args.tag}.csv", index=False)

    idf = pd.DataFrame(immune_rows)
    if not idf.empty:
        idf.to_csv(args.out_dir / f"immune_proximity_niche_corr_{args.tag}.csv", index=False)

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
    summary.to_csv(args.out_dir / f"spatial_summary_{args.tag}.csv", index=False)

    overall = {
        "tag": args.tag,
        "n_niche_tests": int(len(df)),
        "n_immune_dist_tests": int(len(idf)),
        "by_niche_type": summary.groupby("niche_type").median_pearson_r.median().to_dict() if not summary.empty else {},
        "best_graphclust": summary[summary.niche_type == "graphclust"].nlargest(5, "median_pearson_r").to_dict("records") if not summary.empty else [],
        "best_spatial_grid": summary[summary.niche_type == "spatial_grid"].nlargest(5, "median_pearson_r").to_dict("records") if not summary.empty else [],
    }
    (args.out_dir / f"overall_spatial_{args.tag}.json").write_text(json.dumps(overall, indent=2, default=str))
    print(json.dumps(overall, indent=2, default=str))

    if not df.empty:
        gc = df[df.niche_type == "graphclust"]
        if not gc.empty:
            pivot = gc.groupby(["perturbation", "cell_type"]).pearson_r.median().unstack()
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
            ax.set_title(f"Median graphclust-niche Pearson r (predicted vs observed Δ)\n{args.tag}")
            fig.colorbar(im, ax=ax, label="Pearson r")
            fig.tight_layout()
            fig.savefig(args.fig_dir / f"fig1_graphclust_niche_heatmap_{args.tag}.png", dpi=180)
            plt.close(fig)

        sg = df[df.niche_type == "spatial_grid"]
        if not sg.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for ct in sg.cell_type.unique():
                sub = sg[sg.cell_type == ct]
                ax.scatter(sub.pearson_r, sub.pearson_perm_p.clip(1e-6), alpha=0.4, s=12, label=ct)
            ax.set_xlabel("Pearson r (80 µm grid niche)")
            ax.set_ylabel("Permutation p")
            ax.set_yscale("log")
            ax.axvline(0, color="k", lw=0.5)
            ax.legend(fontsize=8)
            ax.set_title(f"Spatial grid niche tests ({args.tag})")
            fig.tight_layout()
            fig.savefig(args.fig_dir / f"fig2_grid_niche_scatter_{args.tag}.png", dpi=180)
            plt.close(fig)

    if not idf.empty:
        prox = idf.groupby(["perturbation", "cell_type", "niche"]).pearson_r.median().reset_index()
        prox = prox.rename(columns={"niche": "immune_dist_bin"})
        show = prox[prox.cell_type.isin(["immune", "myeloid"])].pivot_table(
            index=["perturbation", "cell_type"], columns="immune_dist_bin", values="pearson_r", aggfunc="first"
        )
        if not show.empty:
            fig, ax = plt.subplots(figsize=(max(6, show.shape[1] * 1.2), 0.4 * len(show) + 2))
            vals = show.values.astype(float)
            vmax = max(0.15, float(np.nanmax(np.abs(vals))))
            im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(show.shape[1]))
            ax.set_xticklabels([str(c) for c in show.columns], rotation=30, ha="right", fontsize=7)
            ax.set_yticks(range(len(show)))
            ax.set_yticklabels([f"sg{p}|{c}" for p, c in show.index], fontsize=8)
            ax.set_title(f"Pearson r by distance-to-immune quartile ({args.tag})")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(args.fig_dir / f"fig3_immune_proximity_{args.tag}.png", dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    main()
