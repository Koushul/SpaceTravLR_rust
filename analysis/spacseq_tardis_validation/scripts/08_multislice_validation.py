#!/usr/bin/env python3
"""Cross-slice validation: compare SpaceTravLR predictions vs SPAC-seq per slice.

Uses a single trained model's predicted Δ vectors (from subQ-1 NTC substrate)
and compares them against slice-specific observed Δ vectors computed from each
subQ section's perturbed_pool.h5ad. Aggregates per-slice Pearson r with
Stouffer's Z meta-analysis.

Also builds a pooled NTC h5ad (optional input for retraining) by concatenating
all slices with a spatial X-offset so cells from different sections are not
spatial neighbours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Re-use metric helpers from 05
import importlib.util

_spec = importlib.util.spec_from_file_location("fig05", HERE / "05_final_report_figures.py")
_fig05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fig05)

compute_metrics = _fig05.compute_metrics
dense = _fig05.dense
per_celltype_predicted_delta = _fig05.per_celltype_predicted_delta
stratified_observed_delta = _fig05.stratified_observed_delta
GENE_SETS = _fig05.GENE_SETS

DEFAULT_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]
SPATIAL_PAD = 5000.0


def stouffer_combine(p_values: list[float], signs: list[int]) -> float:
    """One-sided Stouffer meta p (alternative: positive effect)."""
    zs = []
    for p, s in zip(p_values, signs):
        if not np.isfinite(p) or p <= 0 or p >= 1:
            continue
        z = stats.norm.ppf(1 - p / 2)
        zs.append(z if s >= 0 else -z)
    if len(zs) < 2:
        return float("nan")
    z_meta = sum(zs) / np.sqrt(len(zs))
    return float(stats.norm.sf(z_meta))


def build_pooled_h5ad(slices: list[str], data_root: Path, out_path: Path) -> sc.AnnData:
    """Concatenate baseline NTC h5ads with spatial X-offset per slice."""
    parts = []
    x_cursor = 0.0
    for i, sl in enumerate(slices):
        p = data_root / "slices" / sl / "baseline_ntc.h5ad"
        if not p.exists():
            print(f"skip pooled: missing {p}")
            continue
        ad = sc.read_h5ad(p)
        xy = ad.obsm["spatial"].copy()
        if i > 0:
            xy[:, 0] += x_cursor
        ad.obsm["spatial"] = xy
        ad.obs["slice_id"] = sl
        ad.obs_names = [f"{b}@{sl}" for b in ad.obs_names]
        parts.append(ad)
        x_cursor = float(xy[:, 0].max()) + SPATIAL_PAD
    if not parts:
        raise SystemExit("No baseline h5ads found for pooling")
    for i, ad in enumerate(parts):
        ad.obs_names = [f"{ad.obs['slice_id'].iloc[0]}__{b}" for b in ad.obs_names]
    pooled = sc.concat(parts, join="outer", merge="same")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.write_h5ad(out_path)
    print(f"Pooled NTC: {pooled.n_obs:,} cells from {len(parts)} slices -> {out_path}")
    return pooled


def validate_slice(
    slice_name: str,
    data_root: Path,
    baseline,
    pred_dir: Path,
    eligible: list[str],
    common_genes: list[str],
    n_perm: int,
) -> tuple[pd.DataFrame, dict]:
    pool_path = data_root / "slices" / slice_name / "perturbed_pool.h5ad"
    if not pool_path.exists():
        return pd.DataFrame(), {}
    pool = sc.read_h5ad(pool_path)
    sc.pp.normalize_total(pool, target_sum=10000)
    sc.pp.log1p(pool)

    pred_store = {}
    obs_store = {}
    rows = []
    for gene in eligible:
        pred_path = pred_dir / f"predicted_KO_{gene}.feather"
        if not pred_path.exists():
            continue
        pred_ct = per_celltype_predicted_delta(baseline, pred_path, common_genes)
        obs_ct = stratified_observed_delta(pool, gene, common_genes)
        for c, p_delta in pred_ct.items():
            if c not in obs_ct:
                continue
            panel = [g for g in p_delta.index if g != gene]
            metrics = compute_metrics(p_delta.loc[panel], obs_ct[c].loc[panel], n_perm=n_perm)
            if metrics is None:
                continue
            metrics.update({
                "slice": slice_name,
                "perturbation": gene,
                "cell_type": c,
                "n_pert_in_pool": int((pool.obs.target_gene == gene).sum()),
                "n_ntc_in_pool": int((pool.obs.target_gene == "non-targeting").sum()),
            })
            rows.append(metrics)
            pred_store[(gene, c)] = p_delta
            obs_store[(gene, c)] = obs_ct[c]
    return pd.DataFrame(rows), {"pred": pred_store, "obs": obs_store}


def meta_analyze(per_slice: pd.DataFrame) -> pd.DataFrame:
    """Stouffer combine permutation p across slices for each (perturbation, cell_type)."""
    meta_rows = []
    for (pert, ct), grp in per_slice.groupby(["perturbation", "cell_type"]):
        if grp.slice.nunique() < 2:
            continue
        signs = [1 if r >= 0 else -1 for r in grp.pearson_r]
        p_meta = stouffer_combine(grp.pearson_perm_p.tolist(), signs)
        meta_rows.append({
            "perturbation": pert,
            "cell_type": ct,
            "n_slices": int(grp.slice.nunique()),
            "median_pearson_r": float(grp.pearson_r.median()),
            "mean_pearson_r": float(grp.pearson_r.mean()),
            "min_pearson_r": float(grp.pearson_r.min()),
            "max_pearson_r": float(grp.pearson_r.max()),
            "frac_slices_pos_r": float((grp.pearson_r > 0).mean()),
            "frac_slices_perm_p05": float((grp.pearson_perm_p < 0.05).mean()),
            "stouffer_meta_p": p_meta,
        })
    return pd.DataFrame(meta_rows).sort_values(["perturbation", "cell_type"]) if meta_rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path, required=True,
                    help="Trained model prep h5ad (subQ-1 NTC substrate)")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/multislice")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/multislice")
    ap.add_argument("--tag", default="multislice")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--build-pooled", action="store_true",
                    help="Write data/pooled/baseline_ntc.h5ad from all slice NTC cells")
    ap.add_argument("--min-pert-cells", type=int, default=100,
                    help="Min cells in slice perturbed_pool to include perturbation")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.build_pooled:
        build_pooled_h5ad(
            args.slices,
            args.data_root,
            args.data_root / "pooled" / "baseline_ntc.h5ad",
        )

    if args.baseline_h5ad.is_dir():
        baseline_path = sorted(args.baseline_h5ad.glob("*.h5ad"))[0]
    else:
        baseline_path = args.baseline_h5ad
    baseline = sc.read_h5ad(baseline_path)
    if "imputed_count" in baseline.layers:
        baseline.X = baseline.layers["imputed_count"]

    pred_files = sorted(args.pred_dir.glob("predicted_KO_*.feather"))
    pert_genes = [p.stem.replace("predicted_KO_", "") for p in pred_files]

    # Eligible = has predictions AND enough cells in at least one slice
    eligible = []
    for g in pert_genes:
        for sl in args.slices:
            pool_path = args.data_root / "slices" / sl / "perturbed_pool.h5ad"
            if not pool_path.exists():
                continue
            pool = sc.read_h5ad(pool_path)
            if (pool.obs.target_gene == g).sum() >= args.min_pert_cells:
                eligible.append(g)
                break
    eligible = sorted(set(eligible))
    print(f"eligible perturbations: {eligible}")

    common_genes = sorted(set(baseline.var_names))
    all_rows = []
    slice_summaries = {}
    for sl in args.slices:
        df, stores = validate_slice(sl, args.data_root, baseline, args.pred_dir,
                                     eligible, common_genes, args.n_perm)
        if not df.empty:
            df.to_csv(args.out_dir / f"per_celltype_corr_{sl}_{args.tag}.csv", index=False)
            all_rows.append(df)
            slice_summaries[sl] = {
                "n_rows": int(len(df)),
                "median_pearson_r": float(df.pearson_r.median()),
                "n_perm_p05": int((df.pearson_perm_p < 0.05).sum()),
                "by_cell_type": df.groupby("cell_type").pearson_r.median().round(4).to_dict(),
            }
            print(f"{sl}: median r={slice_summaries[sl]['median_pearson_r']:+.3f}, "
                  f"sig={slice_summaries[sl]['n_perm_p05']}/{len(df)}")

    if not all_rows:
        raise SystemExit("No validation rows produced; run 07_multislice_prepare.py first")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(args.out_dir / f"per_celltype_corr_all_slices_{args.tag}.csv", index=False)

    meta = meta_analyze(combined)
    meta.to_csv(args.out_dir / f"meta_analysis_{args.tag}.csv", index=False)

    overall = {
        "tag": args.tag,
        "slices": args.slices,
        "n_slices_with_data": len(slice_summaries),
        "eligible_perturbations": eligible,
        "per_slice": slice_summaries,
        "combined_median_pearson_r": float(combined.pearson_r.median()),
        "combined_n_perm_p05": int((combined.pearson_perm_p < 0.05).sum()),
        "combined_n_rows": int(len(combined)),
        "meta_n_stouffer_p05": int((meta.stouffer_meta_p < 0.05).sum()) if not meta.empty else 0,
        "meta_best_5": meta.nsmallest(5, "stouffer_meta_p")[
            ["perturbation", "cell_type", "n_slices", "median_pearson_r",
             "frac_slices_pos_r", "stouffer_meta_p"]
        ].to_dict("records") if not meta.empty else [],
    }
    (args.out_dir / f"overall_summary_{args.tag}.json").write_text(
        json.dumps(overall, indent=2, default=str)
    )
    print(json.dumps(overall, indent=2, default=str))

    # ---- FIGURES ----
    # Fig 1: heatmap slice × (perturbation|cell_type) median r
    combined["pair"] = combined.apply(
        lambda r: f"{r.perturbation}|{r.cell_type}", axis=1
    )
    pivot = combined.pivot_table(index="slice", columns="pair", values="pearson_r", aggfunc="first")
    pairs = sorted(pivot.columns)
    slices_ord = [s for s in args.slices if s in pivot.index]
    fig, ax = plt.subplots(figsize=(0.35 * len(pairs) + 3, 0.5 * len(slices_ord) + 1.5))
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
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > vmax * 0.55 else "black")
    ax.set_title(f"Per-slice Pearson r: SpaceTravLR predicted vs SPAC-seq observed\n(same model; slice-specific observed Δ)")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.fig_dir / f"fig1_slice_heatmap_{args.tag}.png", dpi=180)
    plt.close(fig)

    # Fig 2: meta-analysis bar chart — median r with slice consistency
    if not meta.empty:
        show = meta.sort_values("stouffer_meta_p").head(12)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(show))))
        y = np.arange(len(show))
        colors = ["#2ca02c" if r > 0 else "#d62728" for r in show.median_pearson_r]
        ax.barh(y, show.median_pearson_r, color=colors, alpha=0.85, edgecolor="k", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"sg{r.perturbation} | {r.cell_type} (n={int(r.n_slices)})"
                            for _, r in show.iterrows()], fontsize=8)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Median Pearson r across slices")
        ax.set_title("Cross-slice meta-analysis (Stouffer p in labels)")
        for i, (_, r) in enumerate(show.iterrows()):
            ax.text(r.median_pearson_r + (0.01 if r.median_pearson_r >= 0 else -0.01),
                    i, f" meta-p={r.stouffer_meta_p:.2g}, {r.frac_slices_pos_r:.0%} slices r>0",
                    va="center", ha="left" if r.median_pearson_r >= 0 else "right", fontsize=7)
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"fig2_meta_analysis_{args.tag}.png", dpi=180)
        plt.close(fig)

    # Fig 3: per cell type — boxplot of r across slices (pooled pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ct_order = sorted(combined.cell_type.unique())
    data = [combined.loc[combined.cell_type == c, "pearson_r"].values for c in ct_order]
    bp = ax.boxplot(data, tick_labels=ct_order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Pearson r (per slice × perturbation)")
    ax.set_title(f"Cross-slice distribution of validation correlation by cell type\n(n={len(combined)} slice×perturbation×celltype rows)")
    fig.tight_layout()
    fig.savefig(args.fig_dir / f"fig3_celltype_boxplot_{args.tag}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
