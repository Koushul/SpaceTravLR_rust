#!/usr/bin/env python3
"""Spatial + CCC-aware hyperparameter sweep for SpaceTravLR perturbations.

Extends script 17 by scoring β × n_propagation on a composite objective:
  40% cell-type pseudobulk (immune/myeloid/fibroblast)
  35% spatial kNN DEG (NTC near-vs-far contrast)
  25% CCC / immune-state pathway concordance

Outputs:
  results/spatial_ccc_tune/sweep_results.csv
  results/spatial_ccc_tune/best_params.json
  results/predictions_spatial_tuned/   (best predictions copied here)
  figures/spatial_ccc_tune/fig_spatial_ccc_sweep.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
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
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec17 = importlib.util.spec_from_file_location("tune17", HERE / "17_iterative_tune.py")
_t17 = importlib.util.module_from_spec(_spec17)
_spec17.loader.exec_module(_t17)
run_perturb = _t17.run_perturb
score_run = _t17.score_run

_spec13 = importlib.util.spec_from_file_location("niche13", HERE / "13_niche_deg_ccc_analysis.py")
_s13 = importlib.util.module_from_spec(_spec13)
_spec13.loader.exec_module(_s13)
spatial_experimental_de = _s13.spatial_experimental_de
spatial_predicted_de = _s13.spatial_predicted_de
ccc_state_analysis = _s13.ccc_state_analysis

_spec09 = importlib.util.spec_from_file_location("sp09", HERE / "09_spatial_validation.py")
_sp09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_sp09)
load_baseline = _sp09.load_baseline
load_pool = _sp09.load_pool

import niche_deg_utils as ndu

VAL_GENES = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b", "Ptk6"]
SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]

SPATIAL_PROBE = [
    ("Il4ra", "immune", "immune", 25),
    ("Il4ra", "immune", "myeloid", 25),
    ("Cd83", "immune", "immune", 25),
    ("Cd74", "myeloid", "immune", 25),
]

CCC_PROBE = [
    ("Il4ra", "immune", "immune"),
    ("Il4ra", "immune", "myeloid"),
    ("Cd83", "immune", "immune"),
]


def score_spatial(pred_dir: Path, baseline: sc.AnnData, data_root: Path) -> dict:
    rs = []
    for perturb, src, neighbor, k in SPATIAL_PROBE:
        exp_parts, pred_parts = [], []
        for sl in SLICES:
            pool = load_pool(sl, data_root)
            pool.obs["slice_id"] = sl
            sc.pp.normalize_total(pool, target_sum=10000)
            sc.pp.log1p(pool)
            pred = ndu.load_pred_feather(pred_dir / f"predicted_KO_{perturb}.feather")
            try:
                exp = spatial_experimental_de(pool, perturb, neighbor, k_neighbors=k, source_cell_type=src)
                pr = spatial_predicted_de(pool, baseline, pred, perturb, neighbor, sl, k_neighbors=k, source_cell_type=src)
                exp_parts.append(exp)
                pred_parts.append(pr)
            except (ValueError, KeyError):
                continue
        if not exp_parts or not pred_parts:
            continue
        exp_all = pd.concat(exp_parts).groupby("gene", as_index=False).agg(log2fc=("log2fc", "mean"))
        pred_all = pd.concat(pred_parts).groupby("gene", as_index=False).agg(log2fc=("log2fc", "mean"))
        merged = exp_all.merge(pred_all, on="gene", suffixes=("_obs", "_pred")).dropna()
        if len(merged) >= 30:
            r, _ = stats.pearsonr(merged.log2fc_obs, merged.log2fc_pred)
            rs.append(float(r))
    return {"spatial_median_r": float(np.median(rs)) if rs else float("nan"), "spatial_n_cases": len(rs)}


def score_ccc(pred_dir: Path, baseline: sc.AnnData, data_root: Path) -> dict:
    rows = []
    for sl in SLICES:
        pool = load_pool(sl, data_root)
        pool.obs["slice_id"] = sl
        sc.pp.normalize_total(pool, target_sum=10000)
        sc.pp.log1p(pool)
        base_sl = baseline[baseline.obs["slice_id"].astype(str) == sl] if "slice_id" in baseline.obs.columns else baseline
        for perturb, src, neighbor in CCC_PROBE:
            pred = ndu.load_pred_feather(pred_dir / f"predicted_KO_{perturb}.feather")
            rows.extend(ccc_state_analysis(pool, base_sl, pred, perturb, neighbor, source_cell_type=src))
    if not rows:
        return {"ccc_pearson_r": float("nan"), "ccc_n_pathways": 0}
    df = pd.DataFrame(rows)
    agg = df.groupby(["perturbation", "pathway", "neighbor_cell_type"], as_index=False).agg(
        obs=("obs_neighbor_delta", "mean"),
        pred=("pred_delta_on_neighbors", "mean"),
    )
    valid = agg.dropna(subset=["obs", "pred"])
    if len(valid) < 3:
        return {"ccc_pearson_r": float("nan"), "ccc_n_pathways": len(valid)}
    r, _ = stats.pearsonr(valid.obs, valid.pred)
    return {"ccc_pearson_r": float(r), "ccc_n_pathways": len(valid)}


def composite_score(m: dict, w_focus: float = 0.4, w_spatial: float = 0.35, w_ccc: float = 0.25) -> float:
    parts, weights = [], []
    for key, w in [("median_r_focus", w_focus), ("spatial_median_r", w_spatial), ("ccc_pearson_r", w_ccc)]:
        v = m.get(key, np.nan)
        if np.isfinite(v):
            parts.append(v * w)
            weights.append(w)
    return float(sum(parts) / sum(weights)) if weights else float("nan")


def sweep(
    run_toml: Path,
    baseline_h5ad: Path,
    data_root: Path,
    betas: list[float],
    n_props: list[int],
    sweep_root: Path,
    baseline_pred_dir: Path | None,
) -> pd.DataFrame:
    baseline = load_baseline(baseline_h5ad)
    rows = []

    if baseline_pred_dir and baseline_pred_dir.exists():
        base_ct = score_run(baseline_pred_dir, baseline_h5ad, data_root, VAL_GENES)
        base_sp = score_spatial(baseline_pred_dir, baseline, data_root)
        base_cc = score_ccc(baseline_pred_dir, baseline, data_root)
        base = {**base_ct, **base_sp, **base_cc}
        base["composite_score"] = composite_score(base)
        rows.append({"beta_scale_factor": "baseline", "n_propagation": "baseline", "pred_dir": str(baseline_pred_dir), **base})
        print(f"baseline composite={base['composite_score']:+.3f}  focus={base.get('median_r_focus', np.nan):+.3f}  "
              f"spatial={base.get('spatial_median_r', np.nan):+.3f}  ccc={base.get('ccc_pearson_r', np.nan):+.3f}")

    for beta in betas:
        for n_prop in n_props:
            tag = f"beta{int(beta)}_np{n_prop}"
            pred_dir = sweep_root / tag
            print(f"\n=== {tag} ===")
            for gene in VAL_GENES:
                out = pred_dir / f"predicted_KO_{gene}.feather"
                if not out.exists():
                    run_perturb(run_toml, gene, out, beta, n_prop)
            ct = score_run(pred_dir, baseline_h5ad, data_root, VAL_GENES)
            sp = score_spatial(pred_dir, baseline, data_root)
            cc = score_ccc(pred_dir, baseline, data_root)
            m = {**ct, **sp, **cc}
            m["composite_score"] = composite_score(m)
            rows.append({"beta_scale_factor": beta, "n_propagation": n_prop, "pred_dir": str(pred_dir), **m})
            print(f"  composite={m['composite_score']:+.3f}  focus={m.get('median_r_focus', np.nan):+.3f}  "
                  f"spatial={m.get('spatial_median_r', np.nan):+.3f}  ccc={m.get('ccc_pearson_r', np.nan):+.3f}")
    return pd.DataFrame(rows)


def pick_best(sweep: pd.DataFrame) -> dict:
    sub = sweep[sweep.beta_scale_factor != "baseline"].copy()
    if sub.empty:
        return {"beta_scale_factor": 50.0, "n_propagation": 3}
    sub = sub.assign(
        beta_scale_factor=sub.beta_scale_factor.astype(float),
        n_propagation=sub.n_propagation.astype(int),
    )
    idx = sub.composite_score.idxmax()
    row = sub.loc[idx]
    return {
        "beta_scale_factor": float(row.beta_scale_factor),
        "n_propagation": int(row.n_propagation),
        "pred_dir": row.pred_dir,
        "composite_score": float(row.composite_score),
        "median_r_focus": float(row.median_r_focus),
        "spatial_median_r": float(row.spatial_median_r),
        "ccc_pearson_r": float(row.ccc_pearson_r),
    }


def copy_preds(best: dict, out_dir: Path, genes: list[str]) -> None:
    src = Path(best["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for gene in genes:
        shutil.copy2(src / f"predicted_KO_{gene}.feather", out_dir / f"predicted_KO_{gene}.feather")
    meta = {k: v for k, v in best.items() if k != "pred_dir"}
    meta["source_pred_dir"] = best["pred_dir"]
    (out_dir / "tuning_meta.json").write_text(json.dumps(meta, indent=2))


def plot_sweep(sweep: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sub = sweep[sweep.beta_scale_factor != "baseline"].copy()
    if sub.empty:
        return
    sub = sub.assign(
        beta_scale_factor=sub.beta_scale_factor.astype(float),
        n_propagation=sub.n_propagation.astype(int),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col, title in zip(
        axes,
        ["median_r_focus", "spatial_median_r", "ccc_pearson_r"],
        ["Cell-type focus r", "Spatial kNN r", "CCC pathway r"],
    ):
        pivot = sub.pivot_table(index="n_propagation", columns="beta_scale_factor", values=col)
        vals = pivot.values.astype(float)
        vmax = max(0.15, float(np.nanmax(np.abs(vals))))
        im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("beta_scale_factor")
        ax.set_ylabel("n_propagation")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Spatial + CCC-aware hyperparameter sweep", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_spatial_ccc_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rerun_niche_validation(baseline_h5ad: Path, betadata_dir: Path, pred_dir: Path, tag: str) -> None:
    cmd = [
        sys.executable, str(HERE / "13_niche_deg_ccc_analysis.py"),
        "--pred-dir", str(pred_dir),
        "--baseline-h5ad", str(baseline_h5ad),
        "--betadata-dir", str(betadata_dir),
        "--tag", tag,
        "--skip-beta-leiden",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-toml", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_run_repro.toml")
    ap.add_argument("--baseline-h5ad", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_prep/baseline_ntc_0c6fbac5e6cd947c_fullprep.h5ad")
    ap.add_argument("--betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_seed")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--betas", nargs="+", type=float, default=[30, 50, 75, 100, 125])
    ap.add_argument("--n-props", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--sweep-root", type=Path, default=ROOT / "results/spatial_ccc_tune/sweep")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/spatial_ccc_tune")
    ap.add_argument("--tuned-dir", type=Path, default=ROOT / "results/predictions_spatial_tuned")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/spatial_ccc_tune")
    ap.add_argument("--tag", default="spatial_tuned")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.out_dir / "sweep_results.csv"

    if args.skip_sweep and sweep_path.exists():
        sweep_df = pd.read_csv(sweep_path)
    else:
        sweep_df = sweep(args.run_toml, args.baseline_h5ad, args.data_root, args.betas, args.n_props,
                         args.sweep_root, args.baseline_pred_dir)
        sweep_df.to_csv(sweep_path, index=False)

    best = pick_best(sweep_df)
    (args.out_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    print("\nBEST:", json.dumps(best, indent=2))

    copy_preds(best, args.tuned_dir, VAL_GENES)
    plot_sweep(sweep_df, args.fig_dir)

    if not args.skip_validation:
        rerun_niche_validation(args.baseline_h5ad, args.betadata_dir, args.tuned_dir, args.tag)


if __name__ == "__main__":
    main()
