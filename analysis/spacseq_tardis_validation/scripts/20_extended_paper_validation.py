#!/usr/bin/env python3
"""Extended paper-finding validation: lung metastasis (observed) + subQ (obs+pred).

Runs headline biology tests on:
  - Lung_Metastasis_M001 (4578 sgIcam1 cells; paper immune-escape cohort)
  - subQ-1…4 pooled (SpaceTravLR β-tuned predictions)

Generates cross-cohort comparison figures when predictions unavailable on lung.

Outputs:
  results/extended_paper/lung_icam1_modules_{tag}.csv
  results/extended_paper/subq_vs_lung_icam1_{tag}.csv
  results/extended_paper/in_silico_spp1_cd44_{tag}.csv
  figures/extended_paper/fig15_lung_icam1_observed_{tag}.png
  figures/extended_paper/fig16_subq_lung_comparison_{tag}.png
  figures/extended_paper/fig17_in_silico_headline_ko_{tag}.png
  figures/extended_paper/fig18_lung_bcam_observed_{tag}.png
  figures/extended_paper/fig19_cohort_validation_summary_{tag}.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec19 = importlib.util.spec_from_file_location("paper_findings_s19", HERE / "19_paper_findings_validation.py")
_s19 = importlib.util.module_from_spec(_spec19)
sys.modules["paper_findings_s19"] = _s19
_spec19.loader.exec_module(_s19)

PAPER_FINDINGS = _s19.PAPER_FINDINGS
load_baseline = _s19.load_baseline
load_pool = _s19.load_pool
load_pool_for_finding = _s19.load_pool_for_finding
observed_delta = _s19.observed_delta
predicted_delta = _s19.predicted_delta
predicted_delta_pooled = _s19.predicted_delta_pooled
score_gene = _s19.score_gene
evaluate_finding_obs_only = _s19.evaluate_finding_obs_only

ICAM1_MODULES = next(f for f in PAPER_FINDINGS if f.finding_id == "icam1_immune_escape").modules
BCAM_FINDING = next(f for f in PAPER_FINDINGS if f.finding_id == "cd44_spp1_axis")

HEADLINE_KO = ["Icam1", "Cd44", "Spp1", "Il4ra", "Bcam"]
HEADLINE_DOWNSTREAM = {
    "Icam1": ["Cxcl9", "Cxcl10", "Itgal", "Itgb2", "Spp1", "Cd163", "Cd8a"],
    "Cd44": ["Spp1", "Mmp9", "Pdcd1", "Lag3", "Fn1", "Postn"],
    "Spp1": ["Cd44", "Itgav", "Mmp9", "Arg1", "Mrc1", "Col1a2"],
    "Il4ra": ["H2-Aa", "H2-Ab1", "Cd74", "Stat6"],
    "Bcam": ["Spp1", "Cd44", "Mmp9", "Postn"],
}


def module_scores_obs(pool: sc.AnnData, perturb: str, cell_type: str, modules) -> pd.DataFrame:
    rows = []
    for mod in modules:
        o = observed_delta(pool, perturb, cell_type, mod.genes)
        for g in mod.genes:
            if g not in o.index:
                continue
            sc_g = score_gene(float(o[g]), 0.0, mod.expected_sign)
            rows.append({
                "gene": g, "module": mod.name, "expected_sign": mod.expected_sign,
                "obs_delta": float(o[g]), "obs_sign_ok": sc_g["obs_sign_ok"],
            })
        if mod.genes:
            sub = [r for r in rows if r["module"] == mod.name]
            if sub:
                rows.append({
                    "gene": "_module_mean", "module": mod.name,
                    "expected_sign": mod.expected_sign,
                    "obs_delta": float(np.mean([r["obs_delta"] for r in sub if r["gene"] != "_module_mean"])),
                    "obs_sign_ok": float(np.mean([r["obs_sign_ok"] for r in sub if r["gene"] != "_module_mean"])),
                })
    df = pd.DataFrame([r for r in rows if r["gene"] != "_module_mean"])
    mod_df = df.groupby("module", as_index=False).agg(
        obs_sign_match_rate=("obs_sign_ok", "mean"),
        obs_mean_delta=("obs_delta", "mean"),
        n_genes=("gene", "count"),
    )
    mod_df["perturbation"] = perturb
    mod_df["cell_type"] = cell_type
    return mod_df


def in_silico_headline(baseline: sc.AnnData, pred_dir: Path, tag: str) -> pd.DataFrame:
    rows = []
    for gene in HEADLINE_KO:
        pred_path = pred_dir / f"predicted_KO_{gene}.feather"
        if not pred_path.exists():
            continue
        pred = pd.read_feather(pred_path).set_index("CellID")
        for ct in ["immune", "myeloid", "fibroblast", "tumor"]:
            sub = baseline[baseline.obs["cell_type"].astype(str) == ct]
            sub = sub[sub.obs_names.isin(pred.index)]
            if sub.n_obs < 10:
                continue
            downstream = HEADLINE_DOWNSTREAM.get(gene, [])
            common = [g for g in downstream if g in sub.var_names and g in pred.columns]
            if not common:
                continue
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location("fig05", HERE / "05_final_report_figures.py")
            fig05 = module_from_spec(spec)
            spec.loader.exec_module(fig05)
            expr = fig05.dense(sub, common)
            pr = pred.loc[sub.obs_names, common]
            delta = (pr - expr).mean(0)
            for g in common:
                rows.append({
                    "ko_gene": gene, "cell_type": ct, "downstream_gene": g,
                    "pred_delta": float(delta[g]), "tag": tag,
                })
    return pd.DataFrame(rows)


def plot_lung_icam1(mod_df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    if mod_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(mod_df))))
    mod_df = mod_df.sort_values("obs_sign_match_rate", ascending=True)
    colors = ["#2166ac" if v >= 0.6 else "#67a9cf" for v in mod_df.obs_sign_match_rate]
    ax.barh(range(len(mod_df)), mod_df.obs_sign_match_rate, color=colors, edgecolor="k", linewidth=0.3)
    ax.axvline(0.6, color="#b2182b", ls="--", lw=1)
    labels = [f"{r.cell_type} | {r.module[:28]}" for _, r in mod_df.iterrows()]
    ax.set_yticks(range(len(mod_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction genes matching paper direction (observed)")
    ax.set_title(
        "Lung M001 sgIcam1 — paper immune-escape modules (n=4578 sgIcam1 cells)\n"
        "Zhang et al. Cell 2026 headline cohort",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig15_lung_icam1_observed_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(lung_df: pd.DataFrame, subq_df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    lung = lung_df.copy()
    lung["cohort"] = "Lung M001 (obs)"
    subq = subq_df[subq_df.finding_id == "icam1_immune_escape"].copy()
    subq["cohort"] = "subQ pooled (obs)"
    subq_pred = subq_df[subq_df.finding_id == "icam1_immune_escape"].copy()
    subq_pred["cohort"] = "subQ pooled (pred)"
    show = pd.concat([
        lung.assign(metric=lung.obs_sign_match_rate, label="obs"),
        subq.assign(metric=subq.obs_sign_match_rate, label="obs"),
        subq_pred.assign(metric=subq_pred.pred_sign_match_rate, label="pred"),
    ])
    show["key"] = show.module + " | " + show.cell_type.astype(str)
    pivot = show.pivot_table(index="key", columns="cohort", values="metric", aggfunc="first")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(pivot))))
    vals = pivot.fillna(0).values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v) and v > 0:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7)
    ax.set_title("Icam1 immune-escape modules: lung vs subQ", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Sign match rate")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig16_subq_lung_comparison_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lung_bcam(mod_df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    if mod_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(mod_df))))
    mod_df = mod_df.sort_values("obs_sign_match_rate", ascending=True)
    colors = ["#2166ac" if v >= 0.6 else "#67a9cf" for v in mod_df.obs_sign_match_rate]
    ax.barh(range(len(mod_df)), mod_df.obs_sign_match_rate, color=colors, edgecolor="k", linewidth=0.3)
    ax.axvline(0.6, color="#b2182b", ls="--", lw=1)
    labels = [f"{r.cell_type} | {r.module[:28]}" for _, r in mod_df.iterrows()]
    ax.set_yticks(range(len(mod_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction genes matching paper direction (observed)")
    ax.set_title(
        "Lung M001 sgBcam — Cd44/Spp1 axis modules (n≈1283 sgBcam cells)\n"
        "Paper proxy for Cd44–Spp1 macrophage crosstalk",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig18_lung_bcam_observed_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_summary(
    lung_icam1: pd.DataFrame,
    lung_bcam: pd.DataFrame,
    subq_mod: pd.DataFrame,
    fig_dir: Path,
    tag: str,
) -> None:
    rows = []
    if not lung_icam1.empty:
        for _, r in lung_icam1.iterrows():
            rows.append({
                "cohort": "Lung M001 (obs)",
                "finding": "Icam1",
                "key": f"{r.cell_type}|{r.module[:20]}",
                "obs": r.obs_sign_match_rate,
                "pred": np.nan,
            })
    if not lung_bcam.empty:
        for _, r in lung_bcam.iterrows():
            rows.append({
                "cohort": "Lung M001 (obs)",
                "finding": "Bcam",
                "key": f"{r.cell_type}|{r.module[:20]}",
                "obs": r.obs_sign_match_rate,
                "pred": np.nan,
            })
    if not subq_mod.empty:
        subq = subq_mod[subq_mod.finding_id.isin(["icam1_immune_escape", "cd44_spp1_axis"])]
        for _, r in subq.iterrows():
            rows.append({
                "cohort": "subQ pooled (obs)",
                "finding": r.perturbation,
                "key": f"{r.cell_type}|{r.module[:20]}",
                "obs": r.obs_sign_match_rate,
                "pred": np.nan,
            })
            rows.append({
                "cohort": "subQ pooled (pred)",
                "finding": r.perturbation,
                "key": f"{r.cell_type}|{r.module[:20]}",
                "obs": np.nan,
                "pred": r.pred_sign_match_rate,
            })
    if not rows:
        return
    show = pd.DataFrame(rows)
    pivot_obs = show.pivot_table(index="key", columns="cohort", values="obs", aggfunc="first")
    pivot_pred = show[show.cohort == "subQ pooled (pred)"].pivot_table(
        index="key", columns="finding", values="pred", aggfunc="first",
    )
    cols = [c for c in pivot_obs.columns if "obs" in c] + (["subQ pred"] if not pivot_pred.empty else [])
    if pivot_obs.empty:
        return
    mat = pivot_obs.fillna(0).values.astype(float)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(pivot_obs))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot_obs.columns)))
    ax.set_xticklabels(pivot_obs.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot_obs)))
    ax.set_yticklabels(pivot_obs.index, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6)
    ax.set_title(
        "Cross-cohort paper biology validation (Icam1 + Bcam modules)\n"
        "Lung = observed SPAC-seq; subQ = observed + SpaceTravLR β-tuned",
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.5, label="Sign match rate")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig19_cohort_validation_summary_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_in_silico(df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    if df.empty:
        return
    focus = df[df.ko_gene.isin(["Icam1", "Cd44", "Spp1", "Bcam"])].copy()
    if focus.empty:
        return
    pivot = focus.pivot_table(
        index="downstream_gene", columns=["ko_gene", "cell_type"],
        values="pred_delta", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(14, max(5, 0.25 * len(pivot))))
    vals = pivot.fillna(0).values.astype(float)
    vmax = max(0.5, float(np.nanpercentile(np.abs(vals), 95)))
    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    cols = [f"{a}|{b}" for a, b in pivot.columns]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_title(
        "In-silico headline KO downstream programs (subQ NTC substrate, β-tuned)\n"
        "Icam1 / Cd44 / Spp1 / Bcam predicted Δ on downstream genes",
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.5, label="Predicted Δ (imputed_count)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig17_in_silico_headline_ko_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subq-slices", nargs="+", default=["subQ-1", "subQ-2", "subQ-3", "subQ-4"])
    ap.add_argument("--lung-slice", default="Lung_Metastasis_M001")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_prep/baseline_ntc_0c6fbac5e6cd947c_fullprep.h5ad")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/extended_paper")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/extended_paper")
    ap.add_argument("--tag", default="tuned")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    # --- Lung Icam1 observed ---
    lung_pool = load_pool(args.lung_slice, args.data_root)
    sc.pp.normalize_total(lung_pool, target_sum=10000)
    sc.pp.log1p(lung_pool)
    lung_rows = []
    icam1_finding = next(f for f in PAPER_FINDINGS if f.finding_id == "icam1_immune_escape")
    for ct in icam1_finding.cell_types:
        mdf = module_scores_obs(lung_pool, "Icam1", ct, icam1_finding.modules)
        lung_rows.append(mdf)
    lung_mod = pd.concat(lung_rows, ignore_index=True) if lung_rows else pd.DataFrame()
    lung_mod.to_csv(args.out_dir / f"lung_icam1_modules_{args.tag}.csv", index=False)

    # --- subQ Icam1 (sparse, pooled guides) ---
    baseline = load_baseline(args.baseline_h5ad)
    _, subq_icam1_mod = _s19.evaluate_finding(
        icam1_finding, args.subq_slices, args.data_root, baseline, args.pred_dir, args.tag,
    )
    if not subq_icam1_mod.empty:
        subq_icam1_mod.to_csv(args.out_dir / f"subq_icam1_modules_{args.tag}.csv", index=False)

    cmp = pd.concat([
        lung_mod.assign(cohort="lung", finding_id="icam1_immune_escape"),
        subq_icam1_mod.assign(cohort="subq", finding_id="icam1_immune_escape") if not subq_icam1_mod.empty else pd.DataFrame(),
    ], ignore_index=True)
    cmp.to_csv(args.out_dir / f"subq_vs_lung_icam1_{args.tag}.csv", index=False)

    # --- In-silico headline KO downstream ---
    insilico = in_silico_headline(baseline, args.pred_dir, args.tag)
    insilico.to_csv(args.out_dir / f"in_silico_spp1_cd44_{args.tag}.csv", index=False)

    # --- Lung Bcam observed (Cd44/Spp1 axis; n≈1283 cells) ---
    lung_bcam_rows = []
    for ct in BCAM_FINDING.cell_types:
        mdf = module_scores_obs(lung_pool, "Bcam", ct, BCAM_FINDING.modules)
        lung_bcam_rows.append(mdf)
    lung_bcam_mod = pd.concat(lung_bcam_rows, ignore_index=True) if lung_bcam_rows else pd.DataFrame()
    lung_bcam_mod.to_csv(args.out_dir / f"lung_bcam_modules_{args.tag}.csv", index=False)

    # --- Lung paper findings (observed-only; subQ-trained preds not on lung CellIDs) ---
    lung_findings_rows = []
    lung_gene_rows = []
    for finding in PAPER_FINDINGS:
        if finding.perturbation not in ("Icam1", "Bcam"):
            continue
        lung_finding = replace(finding, sparse_from_guides=False, pool_slices=True)
        gdf, mdf = evaluate_finding_obs_only(
            lung_finding, [args.lung_slice], args.data_root, args.tag,
        )
        if not mdf.empty:
            mdf["cohort"] = "lung_M001"
            lung_findings_rows.append(mdf)
        if not gdf.empty:
            gdf["cohort"] = "lung_M001"
            lung_gene_rows.append(gdf)
    lung_all = pd.concat(lung_findings_rows, ignore_index=True) if lung_findings_rows else pd.DataFrame()
    lung_genes = pd.concat(lung_gene_rows, ignore_index=True) if lung_gene_rows else pd.DataFrame()
    if not lung_all.empty:
        lung_all.to_csv(args.out_dir / f"lung_paper_findings_{args.tag}.csv", index=False)
    if not lung_genes.empty:
        lung_genes.to_csv(args.out_dir / f"lung_paper_findings_genes_{args.tag}.csv", index=False)

    subq_headline = pd.DataFrame()
    hyp_path = ROOT / "results" / "paper_findings" / f"hypothesis_scores_{args.tag}.csv"
    if hyp_path.exists():
        subq_headline = pd.read_csv(hyp_path)

    plot_lung_icam1(lung_mod, args.fig_dir, args.tag)
    plot_lung_bcam(lung_bcam_mod, args.fig_dir, args.tag)
    if not subq_icam1_mod.empty:
        plot_comparison(lung_mod, subq_icam1_mod, args.fig_dir, args.tag)
    plot_in_silico(insilico, args.fig_dir, args.tag)
    plot_cohort_summary(lung_mod, lung_bcam_mod, subq_headline, args.fig_dir, args.tag)

    summary = {
        "tag": args.tag,
        "lung_icam1_cells": int((lung_pool.obs["target_gene"].astype(str) == "Icam1").sum()),
        "lung_bcam_cells": int((lung_pool.obs["target_gene"].astype(str) == "Bcam").sum()),
        "lung_ntc_cells": int((lung_pool.obs["target_gene"].astype(str) == "non-targeting").sum()),
        "lung_icam1_modules_obs_support": int((lung_mod.obs_sign_match_rate >= 0.6).sum()) if not lung_mod.empty else 0,
        "lung_bcam_modules_obs_support": int((lung_bcam_mod.obs_sign_match_rate >= 0.6).sum()) if not lung_bcam_mod.empty else 0,
        "lung_icam1_n_modules": int(len(lung_mod)),
        "lung_bcam_n_modules": int(len(lung_bcam_mod)),
        "subq_icam1_n_modules": int(len(subq_icam1_mod)),
        "best_lung_icam1": lung_mod.nlargest(5, "obs_sign_match_rate").to_dict("records") if not lung_mod.empty else [],
        "best_lung_bcam": lung_bcam_mod.nlargest(5, "obs_sign_match_rate").to_dict("records") if not lung_bcam_mod.empty else [],
        "lung_all_findings_rows": int(len(lung_all)),
        "lung_all_modules_obs_support": int((lung_all.obs_sign_match_rate >= 0.6).sum()) if not lung_all.empty else 0,
    }
    (args.out_dir / f"overall_{args.tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
