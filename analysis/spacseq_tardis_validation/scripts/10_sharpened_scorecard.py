#!/usr/bin/env python3
"""Sharpened prediction-quality scorecard: subQ-1 vs pooled model, all slices.

Combines cell-type pseudobulk (08), spatial niche (09), and sign-agreement
metrics into one comparison table and summary figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import ensure_boot

ensure_boot()

import argparse
import importlib.util
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]


def load_multislice(path: Path, model: str) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if model == "seed":
        alt = path.parent / "per_celltype_corr_all_slices_multislice.csv"
        if alt.exists():
            return pd.read_csv(alt)
    return pd.DataFrame()


def fisher_z_meta(rs: pd.Series, ns: pd.Series) -> tuple[float, float]:
    """Random-effects friendly: mean Fisher-z weighted by n-3."""
    mask = np.isfinite(rs) & (ns > 3)
    if mask.sum() == 0:
        return float("nan"), 0
    z = np.arctanh(rs[mask].clip(-0.999, 0.999))
    w = (ns[mask] - 3).clip(lower=1)
    z_meta = float(np.average(z, weights=w))
    return float(np.tanh(z_meta)), int(mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["seed", "pooled"],
                    help="Tags matching multislice/spatial result dirs.")
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/scorecard")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/scorecard")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    scorecard = []
    for model in args.models:
        ms = load_multislice(args.results_root / "multislice" / f"per_celltype_corr_all_slices_{model}.csv", model)
        sp = load_multislice(args.results_root / "spatial" / f"niche_corr_{model}.csv", model)
        if ms.empty and model != "seed":
            print(f"skip {model}: no multislice results")
            continue
        if ms.empty:
            ms_path = args.results_root / "multislice" / "per_celltype_corr_all_slices_multislice.csv"
            ms = load_multislice(ms_path)

        for ct in ["immune", "myeloid", "fibroblast", "tumor"]:
            sub = ms[ms.cell_type == ct] if not ms.empty else pd.DataFrame()
            if sub.empty:
                continue
            r_meta, n = fisher_z_meta(sub.pearson_r, sub.n_genes)
            scorecard.append({
                "model": model,
                "level": "cell_type",
                "compartment": ct,
                "n_tests": len(sub),
                "median_r": float(sub.pearson_r.median()),
                "fisher_z_r": r_meta,
                "frac_perm_p05": float((sub.pearson_perm_p < 0.05).mean()),
                "frac_pos_r": float((sub.pearson_r > 0).mean()),
            })

        if not sp.empty:
            for niche_type in sp.niche_type.unique():
                for ct in ["immune", "myeloid", "fibroblast"]:
                    sub = sp[(sp.niche_type == niche_type) & (sp.cell_type == ct)]
                    if sub.empty:
                        continue
                    r_meta, n = fisher_z_meta(sub.pearson_r, sub.n_genes)
                    scorecard.append({
                        "model": model,
                        "level": niche_type,
                        "compartment": ct,
                        "n_tests": len(sub),
                        "median_r": float(sub.pearson_r.median()),
                        "fisher_z_r": r_meta,
                        "frac_perm_p05": float((sub.pearson_perm_p < 0.05).mean()),
                        "frac_pos_r": float((sub.pearson_r > 0).mean()),
                    })

        if not ms.empty:
            for pert in ms.perturbation.unique():
                for ct in ["immune", "myeloid"]:
                    sub = ms[(ms.perturbation == pert) & (ms.cell_type == ct)]
                    if len(sub) < 2:
                        continue
                    r_meta, _ = fisher_z_meta(sub.pearson_r, sub.n_genes)
                    scorecard.append({
                        "model": model,
                        "level": "meta_pert",
                        "compartment": f"{pert}|{ct}",
                        "n_tests": len(sub),
                        "median_r": float(sub.pearson_r.median()),
                        "fisher_z_r": r_meta,
                        "frac_perm_p05": float((sub.pearson_perm_p < 0.05).mean()),
                        "frac_pos_r": float((sub.pearson_r > 0).mean()),
                    })

    sc_df = pd.DataFrame(scorecard)
    sc_df.to_csv(args.out_dir / "prediction_scorecard.csv", index=False)

    headline = []
    if not sc_df.empty:
        ct_imm = sc_df[(sc_df.level == "cell_type") & (sc_df.compartment == "immune")]
        for _, r in ct_imm.iterrows():
            headline.append(f"{r['model']}: immune median r={r['median_r']:+.3f}, Fisher-z={r['fisher_z_r']:+.3f}")

        gc = sc_df[(sc_df.level == "graphclust") & (sc_df.compartment == "immune")]
        for _, r in gc.iterrows():
            headline.append(f"{r['model']}: graphclust/immune median r={r['median_r']:+.3f}")

    overall = {
        "headlines": headline,
        "n_scorecard_rows": len(sc_df),
        "models": args.models,
    }
    (args.out_dir / "scorecard_summary.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps(overall, indent=2))

    if sc_df.empty:
        return

    plot_df = sc_df[sc_df.level.isin(["cell_type", "graphclust", "spatial_grid"])].copy()
    plot_df["label"] = plot_df["level"] + "/" + plot_df["compartment"]
    models = plot_df.model.unique()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    y = np.arange(len(plot_df))
    colors = {"seed": "#1f77b4", "pooled": "#2ca02c", "multislice": "#1f77b4"}
    for i, (_, row) in enumerate(plot_df.iterrows()):
        c = colors.get(row.model, "#888888")
        ax.barh(i, row.median_r, color=c, alpha=0.85, height=0.7)
        ax.text(row.median_r + (0.008 if row.median_r >= 0 else -0.008), i,
                f"{row.model} r={row.median_r:+.3f} (n={int(row.n_tests)})",
                va="center", ha="left" if row.median_r >= 0 else "right", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.label, fontsize=8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Median Pearson r (predicted vs observed Δ)")
    ax.set_title("SpaceTravLR prediction quality scorecard\n(cell-type vs spatial niche resolution)")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "fig_scorecard.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
