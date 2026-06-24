#!/usr/bin/env python3
"""Perturbation-cell and niche DEG concordance + Spp1 biology recovery.

Three comparison modes vs SpaceTravLR predictions (tuned model default):
  1. Direct perturbed cells — sgP vs NTC within cell type (cells receiving CRISPR)
  2. Spatial neighbor niche — bystanders near sgP vs NTC sources (kNN)
  3. β-Leiden microniche — sgP vs NTC within functional spatial clusters

Spp1 analysis (subQ-1…4 has no sgSpp1 cells; Bcam/Cd44 axis is the proxy):
  - Track observed vs predicted Spp1 log2FC across perturbations
  - Spp1-axis module concordance (Spp1, Cd44, Itgav, Mmp9, …)
  - Optional in-silico Spp1 KO prediction + downstream program heatmap

Outputs:
  results/niche_spp1/direct_cell_deg_stats_{tag}.csv
  results/niche_spp1/spatial_neighbor_stats_{tag}.csv
  results/niche_spp1/spp1_tracking_{tag}.csv
  results/niche_spp1/spp1_module_{tag}.csv
  figures/niche_spp1/fig10_direct_cell_deg_grid_{tag}.png
  figures/niche_spp1/fig11_spatial_niche_deg_{tag}.png
  figures/niche_spp1/fig12_spp1_recovery_{tag}.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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
import scanpy as sc
import seaborn as sns
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import niche_deg_utils as ndu

_spec09 = importlib.util.spec_from_file_location("sp09", HERE / "09_spatial_validation.py")
_sp09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_sp09)
load_baseline = _sp09.load_baseline
load_pool = _sp09.load_pool

_spec05 = importlib.util.spec_from_file_location("fig05", HERE / "05_final_report_figures.py")
_fig05 = importlib.util.module_from_spec(_spec05)
_spec05.loader.exec_module(_fig05)
dense = _fig05.dense

DEFAULT_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]
PERTURBATIONS = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b", "Ptk6"]
DIRECT_CASES = [
    ("Il4ra", "immune"), ("Il4ra", "myeloid"),
    ("Cd83", "immune"), ("Cd83", "fibroblast"),
    ("Cd74", "immune"), ("Cd74", "myeloid"),
    ("Bcam", "fibroblast"), ("Bcam", "myeloid"),
    ("Cks1b", "immune"), ("Ptk6", "immune"),
]
SPATIAL_CASES = [
    ("Il4ra", "immune", "immune", 25),
    ("Il4ra", "immune", "myeloid", 25),
    ("Cd83", "immune", "immune", 25),
    ("Cd74", "myeloid", "immune", 25),
    ("Bcam", "fibroblast", "fibroblast", 25),
    ("Bcam", "fibroblast", "myeloid", 25),
]
SPP1_HIGHLIGHT = ["Spp1", "Cd44", "Itgav", "Itgb1", "Fn1", "Mmp9", "Postn", "Col1a2", "Bcam"]


def direct_predicted_de(
    baseline: sc.AnnData,
    pred: pd.DataFrame,
    cell_type: str,
    slice_id: str,
) -> pd.DataFrame:
    """Predicted pseudobulk Δ on NTC substrate (prep scale)."""
    sub = baseline[(baseline.obs["cell_type"].astype(str) == cell_type)]
    if "slice_id" in baseline.obs.columns:
        sub = sub[sub.obs["slice_id"].astype(str) == slice_id]
    sub = sub[sub.obs_names.isin(pred.index)]
    if sub.n_obs < 10:
        return pd.DataFrame(columns=["gene", "log2fc"])
    genes = [g for g in pred.columns if g in sub.var_names]
    expr = dense(sub, genes)
    pr = pred.loc[sub.obs_names, genes]
    delta = (pr - expr).mean(0)
    return pd.DataFrame({"gene": genes, "log2fc": delta.values, "abs_log2fc": np.abs(delta.values)})


def run_direct_cell_deg(
    slices: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    pred_dir: Path,
    fig_dir: Path,
    tag: str,
) -> pd.DataFrame:
    stats_rows = []
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    plot_i = 0

    for perturb, cell_type in DIRECT_CASES:
        exp_dfs, pred_dfs = [], []
        for sl in slices:
            pool = load_pool(sl, data_root)
            pool.obs["slice_id"] = sl
            sc.pp.normalize_total(pool, target_sum=10000)
            sc.pp.log1p(pool)
            pred = ndu.load_pred_feather(pred_dir / f"predicted_KO_{perturb}.feather")
            try:
                exp = ndu.direct_cell_pseudobulk(pool, perturb, cell_type=cell_type)
                pr = direct_predicted_de(baseline, pred, cell_type, sl)
                exp_dfs.append(exp)
                pred_dfs.append(pr)
            except ValueError as e:
                print(f"  skip direct {sl} {perturb}/{cell_type}: {e}")
                continue

        if not exp_dfs:
            continue
        exp_all = pd.concat(exp_dfs).groupby("gene", as_index=False).agg(log2fc=("log2fc", "mean"))
        pred_all = pd.concat(pred_dfs).groupby("gene", as_index=False).agg(log2fc=("log2fc", "mean"))

        hl = SPP1_HIGHLIGHT if perturb == "Bcam" else (
            ["Il4ra", "Stat6", "Arg1"] if perturb == "Il4ra" else
            ["Cd83", "H2-Aa", "H2-Ab1"] if perturb == "Cd83" else
            ["Cd74", "H2-Aa", "H2-Ab1"] if perturb == "Cd74" else []
        )
        ax = axes[plot_i] if plot_i < len(axes) else None
        st, _ = ndu.plot_gene_comparison_advanced(
            exp_all, pred_all,
            label1="SPAC-seq sgP", label2="SpaceTravLR",
            highlight_genes=hl, top_n_labels=0, target_ko=perturb,
            neighbor_ct=cell_type, source_ct="direct cells",
            axis_lim=1.5, ax=ax, show=False, use_size=True,
            title_suffix=f"{len(exp_dfs)} slices",
        )
        st.update({"perturbation": perturb, "cell_type": cell_type, "analysis": "direct_cell", "tag": tag})
        stats_rows.append(st)
        plot_i += 1

    for j in range(plot_i, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "Direct perturbed-cell DEG concordance\n"
        "(Wilcoxon sgP vs NTC within cell type; pred = NTC substrate Δ)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig10_direct_cell_deg_grid_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def run_spatial_niche(
    slices: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    pred_dir: Path,
    fig_dir: Path,
    tag: str,
) -> pd.DataFrame:
    import importlib.util
    spec13 = importlib.util.spec_from_file_location("s13", HERE / "13_niche_deg_ccc_analysis.py")
    s13 = importlib.util.module_from_spec(spec13)
    spec13.loader.exec_module(s13)

    stats_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for i, (perturb, source_ct, neighbor_ct, k) in enumerate(SPATIAL_CASES):
        exp_dfs, pred_dfs = [], []
        for sl in slices:
            pool = load_pool(sl, data_root)
            pool.obs["slice_id"] = sl
            sc.pp.normalize_total(pool, target_sum=10000)
            sc.pp.log1p(pool)
            pred = ndu.load_pred_feather(pred_dir / f"predicted_KO_{perturb}.feather")
            try:
                exp = ndu.spatial_ntc_niche_pseudobulk(
                    pool, perturb, k_neighbors=k, cell_type=neighbor_ct,
                    source_cell_type=source_ct,
                )
                pr = s13.spatial_predicted_de(
                    pool, baseline, pred, perturb, neighbor_ct, sl,
                    k_neighbors=k, source_cell_type=source_ct,
                )
                exp_dfs.append(exp)
                pred_dfs.append(pr)
            except ValueError as e:
                print(f"  skip spatial {sl} {perturb}/{neighbor_ct}: {e}")
                continue
        if not exp_dfs:
            continue
        exp_all = pd.concat(exp_dfs).groupby("gene", as_index=False).agg(log2fc=("log2fc", "mean"))
        pred_all = pd.concat(pred_dfs).groupby("gene", as_index=False).agg(log2fc=("log2fc", "mean"))
        hl = SPP1_HIGHLIGHT if perturb == "Bcam" else []
        st, _ = ndu.plot_gene_comparison_advanced(
            exp_all, pred_all,
            label1="SPAC-seq neighbors", label2="SpaceTravLR",
            highlight_genes=hl, top_n_labels=0, target_ko=perturb,
            neighbor_ct=neighbor_ct, source_ct=f"sg{perturb} niche",
            axis_lim=1.2, ax=axes[i], show=False, use_size=True,
            title_suffix=f"{len(exp_dfs)} slices",
        )
        st.update({"perturbation": perturb, "neighbor_cell_type": neighbor_ct, "analysis": "spatial_knn", "tag": tag})
        stats_rows.append(st)

    fig.suptitle(
        "Spatial niche DEG concordance (NTC near sgP vs NTC far from sgP)\n"
        "Observed log1p Δ; predicted imputed_count Δ on NTC substrate",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig11_spatial_niche_deg_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def spp1_pred_delta(baseline: sc.AnnData, pred: pd.DataFrame, slice_id: str, cell_type: str) -> float:
    sub = baseline[baseline.obs["cell_type"].astype(str) == cell_type]
    if "slice_id" in baseline.obs.columns:
        sub = sub[sub.obs["slice_id"].astype(str) == slice_id]
    sub = sub[sub.obs_names.isin(pred.index)]
    if sub.n_obs < 10 or "Spp1" not in sub.var_names or "Spp1" not in pred.columns:
        return float("nan")
    expr = dense(sub, ["Spp1"]).iloc[:, 0]
    pr = pred.loc[sub.obs_names, "Spp1"]
    return float((pr - expr).mean())


def spp1_tracking(
    slices: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    pred_dir: Path,
) -> pd.DataFrame:
    rows = []
    for sl in slices:
        pool = load_pool(sl, data_root)
        pool.obs["slice_id"] = sl
        sc.pp.normalize_total(pool, target_sum=10000)
        sc.pp.log1p(pool)
        for perturb in PERTURBATIONS + ["Spp1"]:
            pred_path = pred_dir / f"predicted_KO_{perturb}.feather"
            if not pred_path.exists():
                continue
            pred = ndu.load_pred_feather(pred_path)
            for ct in ["immune", "myeloid", "fibroblast", "tumor"]:
                ntc = pool[(pool.obs["target_gene"] == "non-targeting") & (pool.obs["cell_type"] == ct)]
                per = pool[(pool.obs["target_gene"] == perturb) & (pool.obs["cell_type"] == ct)]
                if ntc.n_obs < 10 or per.n_obs < 5 or "Spp1" not in pool.var_names:
                    continue
                obs_spp1 = float(dense(per, ["Spp1"]).mean().iloc[0] - dense(ntc, ["Spp1"]).mean().iloc[0])
                pred_row = direct_predicted_de(baseline, pred, ct, sl)
                pred_spp1 = float(pred_row.loc[pred_row.gene == "Spp1", "log2fc"].iloc[0]) if "Spp1" in pred_row.gene.values else float("nan")
                rows.append({
                    "slice": sl, "perturbation": perturb, "cell_type": ct,
                    "obs_spp1_log2fc": obs_spp1, "pred_spp1_delta": pred_spp1,
                    "n_pert": int(per.n_obs), "n_ntc": int(ntc.n_obs),
                })
    return pd.DataFrame(rows)


def spp1_module_scores(
    slices: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    pred_dir: Path,
) -> pd.DataFrame:
    rows = []
    for sl in slices:
        pool = load_pool(sl, data_root)
        pool.obs["slice_id"] = sl
        sc.pp.normalize_total(pool, target_sum=10000)
        sc.pp.log1p(pool)
        for perturb in ["Bcam", "Il4ra", "Cd83", "Cd74"]:
            pred_path = pred_dir / f"predicted_KO_{perturb}.feather"
            if not pred_path.exists():
                continue
            pred = ndu.load_pred_feather(pred_path)
            for mod_name, genes in ndu.SPP1_AXIS.items():
                for ct in ["fibroblast", "myeloid", "immune"]:
                    ntc = pool[(pool.obs["target_gene"] == "non-targeting") & (pool.obs["cell_type"] == ct)]
                    per = pool[(pool.obs["target_gene"] == perturb) & (pool.obs["cell_type"] == ct)]
                    if ntc.n_obs < 10 or per.n_obs < 5:
                        continue
                    obs_p = ndu.module_score(per, genes, mod_name)
                    obs_c = ndu.module_score(ntc, genes, mod_name)
                    obs_d = float(np.nanmean(obs_p) - np.nanmean(obs_c))
                    pred_d = float("nan")
                    sub = baseline[(baseline.obs["cell_type"].astype(str) == ct)]
                    if "slice_id" in baseline.obs.columns:
                        sub = sub[sub.obs["slice_id"].astype(str) == sl]
                    sub = sub[sub.obs_names.isin(pred.index)]
                    if sub.n_obs >= 10:
                        common = [g for g in genes if g in pred.columns and g in sub.var_names]
                        if len(common) >= 2:
                            expr = dense(sub, common)
                            pr = pred.loc[sub.obs_names, common]
                            pred_d = float((pr - expr).mean().mean())
                    rows.append({
                        "slice": sl, "perturbation": perturb, "module": mod_name,
                        "cell_type": ct, "obs_module_delta": obs_d, "pred_module_delta": pred_d,
                    })
    return pd.DataFrame(rows)


def plot_spp1_recovery(
    track: pd.DataFrame,
    module: pd.DataFrame,
    fig_dir: Path,
    tag: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    show = track.groupby(["perturbation", "cell_type"], as_index=False).agg(
        obs_spp1_log2fc=("obs_spp1_log2fc", "mean"),
        pred_spp1_delta=("pred_spp1_delta", "mean"),
    )
    show = show[show.perturbation != "Spp1"]
    for ct, grp in show.groupby("cell_type"):
        ax.scatter(grp.obs_spp1_log2fc, grp.pred_spp1_delta, label=ct, s=55, alpha=0.8)
    lim = max(show[["obs_spp1_log2fc", "pred_spp1_delta"]].abs().max().max(), 0.05)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.4)
    if len(show) >= 3:
        r, p = stats.pearsonr(show.obs_spp1_log2fc, show.pred_spp1_delta)
        ax.text(0.05, 0.95, f"Spp1 Pearson r = {r:+.2f}\np = {p:.3f}", transform=ax.transAxes, va="top")
    ax.set_xlabel("Observed Spp1 Δ (sgP − NTC, log1p)")
    ax.set_ylabel("Predicted Spp1 Δ (NTC substrate)")
    ax.set_title("A  Spp1 concordance by perturbation", fontweight="bold", loc="left")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    bcam = show[show.perturbation == "Bcam"].sort_values("cell_type")
    if not bcam.empty:
        x = np.arange(len(bcam))
        w = 0.35
        ax.bar(x - w / 2, bcam.obs_spp1_log2fc, width=w, label="Observed", color="#2166ac")
        ax.bar(x + w / 2, bcam.pred_spp1_delta, width=w, label="Predicted", color="#67a9cf")
        ax.set_xticks(x)
        ax.set_xticklabels(bcam.cell_type, rotation=30, ha="right")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Spp1 Δ")
        ax.legend(fontsize=8)
        ax.set_title("B  sgBcam → Spp1 (Cd44 axis proxy)", fontweight="bold", loc="left")

    ax = axes[1, 0]
    mod = module.groupby(["perturbation", "module"], as_index=False).agg(
        obs_module_delta=("obs_module_delta", "mean"),
        pred_module_delta=("pred_module_delta", "mean"),
    )
    pivot = mod.pivot_table(index="module", columns="perturbation", values="obs_module_delta")
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Observed module Δ"})
    ax.set_title("C  Spp1-axis module shifts (observed)", fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)

    ax = axes[1, 1]
    if len(mod) >= 3:
        ax.scatter(mod.obs_module_delta, mod.pred_module_delta, c=mod.perturbation.astype("category").cat.codes,
                   cmap="tab10", s=45, alpha=0.75)
        r, p = stats.pearsonr(mod.obs_module_delta, mod.pred_module_delta)
        ax.text(0.05, 0.95, f"Module Pearson r = {r:+.2f}", transform=ax.transAxes, va="top")
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Observed module Δ")
    ax.set_ylabel("Predicted module Δ")
    ax.set_title("D  Spp1/CD44/ECM module concordance", fontweight="bold", loc="left")

    fig.suptitle(
        "Spp1 biology recovery (subQ-1…4: no sgSpp1; Bcam/Cd44/ECM axis)\n"
        "SpaceTravLR predicts niche-local Spp1 and module shifts",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig12_spp1_recovery_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def ensure_spp1_prediction(run_toml: Path, pred_dir: Path, beta: float, n_prop: int) -> None:
    out = pred_dir / "predicted_KO_Spp1.feather"
    if out.exists():
        return
    subprocess.run([
        "spacetravlr-perturb",
        "--run-toml", str(run_toml),
        "--gene", "Spp1",
        "--desired-expr", "0.0",
        "--n-propagation", str(n_prop),
        "--beta-scale-factor", str(beta),
        "--out", str(out),
    ], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_prep/baseline_ntc_0c6fbac5e6cd947c_fullprep.h5ad")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--run-toml", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_run_repro.toml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/niche_spp1")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/niche_spp1")
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--beta", type=float, default=50.0)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--skip-spp1-perturb", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    if not args.skip_spp1_perturb:
        print("Ensuring Spp1 KO prediction…")
        ensure_spp1_prediction(args.run_toml, args.pred_dir, args.beta, args.n_prop)

    baseline = load_baseline(args.baseline_h5ad)

    print("Direct perturbed-cell DEG…")
    direct = run_direct_cell_deg(args.slices, args.data_root, baseline, args.pred_dir, args.fig_dir, args.tag)
    direct.to_csv(args.out_dir / f"direct_cell_deg_stats_{args.tag}.csv", index=False)

    print("Spatial neighbor niche DEG…")
    spatial = run_spatial_niche(args.slices, args.data_root, baseline, args.pred_dir, args.fig_dir, args.tag)
    spatial.to_csv(args.out_dir / f"spatial_neighbor_stats_{args.tag}.csv", index=False)

    print("Spp1 tracking…")
    track = spp1_tracking(args.slices, args.data_root, baseline, args.pred_dir)
    track.to_csv(args.out_dir / f"spp1_tracking_{args.tag}.csv", index=False)

    print("Spp1 module analysis…")
    module = spp1_module_scores(args.slices, args.data_root, baseline, args.pred_dir)
    module.to_csv(args.out_dir / f"spp1_module_{args.tag}.csv", index=False)

    plot_spp1_recovery(track, module, args.fig_dir, args.tag)

    overall = {
        "tag": args.tag,
        "note": "subQ-1…4 has zero sgSpp1 cells; Spp1 assessed via downstream Δ and sgBcam proxy",
        "direct_cases": len(direct),
        "direct_median_pearson": float(direct.pearson_r.median()) if len(direct) else None,
        "spatial_cases": len(spatial),
        "spatial_median_pearson": float(spatial.pearson_r.median()) if len(spatial) else None,
        "spp1_rows": len(track),
        "best_direct": direct.nlargest(3, "pearson_r").to_dict("records") if len(direct) else [],
        "best_spatial": spatial.nlargest(3, "pearson_r").to_dict("records") if len(spatial) else [],
        "bcam_spp1": track[track.perturbation == "Bcam"].groupby("cell_type").agg(
            obs=("obs_spp1_log2fc", "mean"), pred=("pred_spp1_delta", "mean"),
        ).reset_index().to_dict("records") if len(track) else [],
    }
    (args.out_dir / f"overall_{args.tag}.json").write_text(json.dumps(overall, indent=2, default=str))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
