#!/usr/bin/env python3
"""Stacked BR (and plain LR) ranking bars.

BR stacks = fraction of each bacterial ligand (signal) expressed by each genus.
Host cells carry the receptors; bacteria express the ligands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

ROOT = Path(
    os.environ.get(
        "SPACETRAVLR_MICROBIOME_ROOT",
        Path(__file__).resolve().parents[3].parent / "spacetravlr_microbiome",
    )
)
RUN = ROOT / "runs/tumor_br_r2x"
SITE = ROOT / "site_br_report/assets/r2x"
OUT = RUN / "figures"
SITE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

TOP_N_GENUS = 8
PALETTE = [
    "#e6a84a",
    "#3db89a",
    "#6b8fd6",
    "#d4736b",
    "#c4a35a",
    "#8e7cc3",
    "#5dade2",
    "#58d68d",
    "#aab7b8",
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#0c1814",
            "figure.facecolor": "#07110e",
            "savefig.facecolor": "#07110e",
            "text.color": "#eef4f0",
            "axes.labelcolor": "#eef4f0",
            "xtick.color": "#9bb0a6",
            "ytick.color": "#9bb0a6",
            "axes.edgecolor": "#9bb0a6",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    for dest in (SITE, OUT):
        fig.savefig(dest / name, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", name)


def genus_ligand_fractions(bact: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Rows = signals, cols = genus (+ other). Values = fraction of total ligand amount."""
    totals = bact.groupby("bact_label")[signals].sum()
    # keep top genera by total ligand mass across all signals
    genus_mass = totals.sum(axis=1).sort_values(ascending=False)
    top = genus_mass.head(TOP_N_GENUS).index.tolist()
    keep = totals.loc[top]
    other = totals.drop(index=top, errors="ignore").sum(axis=0)
    mat = keep.T.copy()
    mat["other"] = other
    mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return mat


def stacked_ranking(
    labels: list[str],
    heights: np.ndarray,
    frac: pd.DataFrame,
    title: str,
    xlabel: str,
    outfile: str,
    notes: list[str] | None = None,
) -> None:
    """Horizontal stacked bars: height * genus fraction for each row in frac (indexed like labels keys)."""
    genera = list(frac.columns)
    colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(genera)}
    fig, ax = plt.subplots(figsize=(10.2, max(3.8, 0.55 * len(labels) + 1.2)))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for g in genera:
        widths = heights * frac[g].to_numpy()
        ax.barh(y, widths, left=left, color=colors[g], height=0.72, label=g, linewidth=0)
        left += widths
    ax.set_yticks(y)
    if notes:
        ax.set_yticklabels([f"{lab}   {note}" for lab, note in zip(labels, notes)], fontsize=9)
    else:
        ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=8,
        ncol=2,
        title="Ligand expressed by genus",
        title_fontsize=8,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, outfile)
    plt.close(fig)


def composition_bars(frac: pd.DataFrame, labels: list[str], title: str, outfile: str) -> None:
    genera = list(frac.columns)
    colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(genera)}
    fig, ax = plt.subplots(figsize=(10.2, max(3.6, 0.5 * len(labels) + 1.0)))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for g in genera:
        w = frac[g].to_numpy()
        ax.barh(y, w, left=left, color=colors[g], height=0.72, label=g, linewidth=0)
        left += w
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Fraction of bacterial ligand expression")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, outfile)
    plt.close(fig)


def plain_lr_bars() -> None:
    import json

    data = json.loads((SITE / "figure_meta.json").read_text())
    lr = pd.DataFrame(data["lr_top"]).sort_values("sum_abs", ascending=True)
    labels = [r.replace("beta_", "").replace("$", " → ") for r in lr["interaction"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.barh(np.arange(len(lr)), lr["sum_abs"].to_numpy(), color="#3db89a", height=0.72, linewidth=0)
    ax.set_yticks(np.arange(len(lr)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Σ|β| across target genes")
    ax.set_title("Top host–host LR terms (ligands & receptors on host cells)", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "lr_ranking.png")
    plt.close(fig)


def main() -> None:
    style()
    br = pd.read_csv(RUN / "top_br_terms.csv")
    bact = pd.read_parquet(ROOT / "processed/GSM9456850_bact_senders_colony25um_scfa_merged.parquet")

    signals = br["signal"].tolist()
    missing = [s for s in signals if s not in bact.columns]
    if missing:
        raise SystemExit(f"signals missing from senders parquet: {missing}")

    frac_all = genus_ligand_fractions(bact, signals)
    # align rows to BR table order (already ranked by sum_abs desc in csv)
    br = br.sort_values("sum_abs", ascending=False).reset_index(drop=True)
    frac = frac_all.loc[br["signal"]].copy()
    frac.index = br["signal"]

    labels = []
    notes = []
    for row in br.itertuples():
        name = getattr(row, "signal_name", row.signal)
        labels.append(f"{name} → {row.receptor}")
        notes.append(f"({row.pathway})")

    heights = br["sum_abs"].to_numpy(dtype=float)
    frac_plot = frac.copy()
    frac_plot.index = labels

    stacked_ranking(
        labels=labels,
        heights=heights,
        frac=frac_plot,
        title="Top BR terms — stacked by genus expressing each bacterial ligand",
        xlabel="Σ|β| (stack width ∝ genus share of ligand expression)",
        outfile="br_ranking_stacked_genus.png",
        notes=notes,
    )
    stacked_ranking(
        labels=labels,
        heights=heights,
        frac=frac_plot,
        title="Top BR terms — stacked by genus expressing each bacterial ligand",
        xlabel="Σ|β| (stack width ∝ genus share of ligand expression)",
        outfile="br_ranking_annotated.png",
        notes=notes,
    )
    composition_bars(
        frac_plot,
        labels,
        "Which genus expresses each BR ligand the most?",
        "br_ligand_genus_fractions.png",
    )

    out_csv = frac.copy()
    out_csv.insert(0, "signal", br["signal"].to_numpy())
    out_csv.insert(1, "receptor", br["receptor"].to_numpy())
    out_csv.insert(2, "sum_abs_beta", heights)
    out_csv.to_csv(SITE / "br_ligand_genus_fractions.csv", index=False)
    out_csv.to_csv(OUT / "br_ligand_genus_fractions.csv", index=False)

    # absolute ligand mass by genus (for reference)
    mass = bact.groupby("bact_label")[signals].sum()
    mass.to_csv(SITE / "br_ligand_genus_mass.csv")

    plain_lr_bars()
    # remove misleading LR stacked assets usage — keep composition file note
    print("genus fractions (top signals):")
    print(frac.head().to_string())


if __name__ == "__main__":
    main()
