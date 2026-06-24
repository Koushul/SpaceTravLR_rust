#!/usr/bin/env python3
"""Validate SpaceTravLR against headline biological findings from Zhang et al. Cell 2026.

The SPAC-seq paper reports three mechanistic themes:
  1. Icam1 loss → immune exclusion, IFN↓, LFA-1 axis↓, T cell↓, M2/Spp1↑
  2. Cd44–Spp1 macrophage–T cell crosstalk → exhaustion / spatial coupling
  3. Immune checkpoint / antigen presentation rewiring (Il4ra, Cd83, Cd74 in subQ)

For each theme we define gene-module hypotheses with expected direction under KO
and score observed (SPAC-seq sgP − NTC) vs predicted (SpaceTravLR in-silico KO)
sign concordance per gene and per module.

subQ-1…4 lacks sgSpp1/sgCd44 at scale; Icam1 is sparse but poolable (~95 cells);
sgBcam proxies the Cd44/ECM axis; Il4ra/Cd83/Cd74 have expanded cohorts.

Outputs:
  results/paper_findings/hypothesis_scores_{tag}.csv
  results/paper_findings/gene_level_{tag}.csv
  results/paper_findings/overall_{tag}.json
  figures/paper_findings/fig13_paper_findings_scorecard_{tag}.png
  figures/paper_findings/fig14_paper_modules_heatmap_{tag}.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
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
from scipy import sparse, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent
MC38 = REPO / "analysis" / "mc38_visiumhd"

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


@dataclass
class ModuleHypothesis:
    name: str
    genes: list[str]
    expected_sign: int  # +1 up, -1 down under KO vs NTC


@dataclass
class PaperFinding:
    finding_id: str
    title: str
    paper_context: str
    perturbation: str
    cell_types: list[str]
    modules: list[ModuleHypothesis]
    min_pert: int = 20
    min_ntc: int = 10
    pool_slices: bool = False
    sparse_from_guides: bool = False


def load_sparse_perturbation_pool(
    slice_id: str,
    perturb: str,
    data_root: Path,
    max_pert: int = 500,
) -> sc.AnnData | None:
    """Build NTC + sparse sgP pool from guide assignments (not in perturbed_pool.h5ad)."""
    guide_path = data_root / "slices" / slice_id / "cell_guide_assignments.parquet"
    ann_path = MC38 / slice_id / "processed" / f"{slice_id}_cells_annotated.h5ad"
    if not guide_path.exists() or not ann_path.exists():
        return None
    ann = sc.read_h5ad(ann_path)
    sc.pp.normalize_total(ann, target_sum=10000)
    sc.pp.log1p(ann)
    guides = pd.read_parquet(guide_path)
    unamb = guides[guides.is_unambiguous].copy()
    cell_to_target = dict(zip(unamb.cell_barcode, unamb.target_gene))
    ntc_bc = set(unamb.loc[unamb.is_ntc, "cell_barcode"])
    ann = ann[ann.obs_names.isin(unamb.cell_barcode)].copy()
    ann.obs["target_gene"] = [cell_to_target.get(b, "no_guide") for b in ann.obs_names]
    ntc_mask = ann.obs_names.isin(ntc_bc)
    pert_mask = ann.obs["target_gene"].astype(str) == perturb
    if pert_mask.sum() < 5:
        return None
    pert_idx = ann.obs_names[pert_mask]
    if len(pert_idx) > max_pert:
        pert_idx = pd.Index(np.random.default_rng(42).choice(pert_idx, size=max_pert, replace=False))
    keep = ann.obs_names[ntc_mask].tolist() + pert_idx.tolist()
    pool = ann[keep].copy()
    pool.obs.loc[pool.obs_names.isin(ntc_bc), "target_gene"] = "non-targeting"
    pool.obs["slice_id"] = slice_id
    return pool


def load_pool_for_finding(
    slice_id: str,
    perturb: str,
    data_root: Path,
    sparse_from_guides: bool,
) -> sc.AnnData | None:
    if sparse_from_guides:
        pool = load_sparse_perturbation_pool(slice_id, perturb, data_root)
        if pool is not None:
            return pool
    pool_path = data_root / "slices" / slice_id / "perturbed_pool.h5ad"
    if not pool_path.exists():
        return None
    pool = load_pool(slice_id, data_root)
    pool.obs["slice_id"] = slice_id
    sc.pp.normalize_total(pool, target_sum=10000)
    sc.pp.log1p(pool)
    return pool


PAPER_FINDINGS: list[PaperFinding] = [
    PaperFinding(
        finding_id="icam1_immune_escape",
        title="Icam1 KO → immune exclusion program",
        paper_context="Lung metastasis: Icam1 loss enriches immune-excluded niches; IFN and "
        "ICAM1–LFA-1 synapse disrupted; T cells down; M2/Spp1 macrophage polarization up.",
        perturbation="Icam1",
        cell_types=["tumor", "myeloid", "immune"],
        min_pert=10,
        pool_slices=True,
        sparse_from_guides=True,
        modules=[
            ModuleHypothesis("On-target Icam1", ["Icam1"], -1),
            ModuleHypothesis("IFN / inflammatory chemokines", ["Cxcl9", "Cxcl10", "Stat1", "Ifit3", "Isg15"], -1),
            ModuleHypothesis("LFA-1 immune synapse", ["Itgal", "Itgb2"], -1),
            ModuleHypothesis("T cell effector", ["Cd8a", "Cd3e", "Gzmb", "Prf1"], -1),
            ModuleHypothesis("M2 / suppressive macrophage", ["Spp1", "Cd163", "Mrc1", "Arg1"], +1),
        ],
    ),
    PaperFinding(
        finding_id="cd44_spp1_axis",
        title="Cd44–Spp1 macrophage crosstalk (sgBcam proxy)",
        paper_context="Paper: Cd44 on T cells couples to macrophage Spp1; axis drives exhaustion. "
        "subQ uses sgBcam (basigin / Cd44-axis partner) as proxy — no sgCd44/sgSpp1 cohort.",
        perturbation="Bcam",
        cell_types=["fibroblast", "myeloid", "immune"],
        modules=[
            ModuleHypothesis("Spp1 / osteopontin", ["Spp1"], +1),
            ModuleHypothesis("Cd44–integrin axis", ["Cd44", "Itgav", "Itgb1", "Fn1"], +1),
            ModuleHypothesis("ECM remodeling", ["Mmp9", "Postn", "Col1a2", "Mmp2"], +1),
            ModuleHypothesis("T cell exhaustion", ["Pdcd1", "Lag3", "Tigit", "Havcr2"], +1),
            ModuleHypothesis("On-target Bcam", ["Bcam"], -1),
        ],
    ),
    PaperFinding(
        finding_id="il4ra_mhc2",
        title="Il4ra KO → MHC-II / antigen presentation down",
        paper_context="subQ expanded cohort: Il4ra is immune cytokine receptor; IL-4 signaling "
        "promotes MHC-II on myeloid cells — KO should reduce antigen presentation.",
        perturbation="Il4ra",
        cell_types=["immune", "myeloid"],
        modules=[
            ModuleHypothesis("On-target Il4ra", ["Il4ra"], -1),
            ModuleHypothesis("MHC-II presentation", ["H2-Aa", "H2-Ab1", "Cd74"], -1),
            ModuleHypothesis("Il4 / Th2 macrophage", ["Stat6", "Arg1", "Mrc1"], -1),
            ModuleHypothesis("MHC-I / cross-presentation", ["H2-K1", "B2m", "Tap1"], -1),
        ],
    ),
    PaperFinding(
        finding_id="cd83_costimulation",
        title="Cd83 KO → DC activation / MHC-II down",
        paper_context="Cd83 is B-cell/DC activation marker; KO reduces costimulation and "
        "antigen presentation in immune niches.",
        perturbation="Cd83",
        cell_types=["immune", "myeloid"],
        modules=[
            ModuleHypothesis("On-target Cd83", ["Cd83"], -1),
            ModuleHypothesis("MHC-II presentation", ["H2-Aa", "H2-Ab1", "Cd74"], -1),
            ModuleHypothesis("T cell costimulation", ["Cd80", "Cd86", "Icosl"], -1),
            ModuleHypothesis("Immune checkpoint ligands", ["Cd274", "Pdcd1lg2"], -1),
        ],
    ),
    PaperFinding(
        finding_id="cd74_mhc2",
        title="Cd74 (invariant chain) KO → MHC-II down",
        paper_context="Cd74 chaperones MHC-II; KO disrupts antigen presentation in myeloid/immune.",
        perturbation="Cd74",
        cell_types=["myeloid", "immune"],
        modules=[
            ModuleHypothesis("On-target Cd74", ["Cd74"], -1),
            ModuleHypothesis("MHC-II presentation", ["H2-Aa", "H2-Ab1", "Ciita"], -1),
            ModuleHypothesis("Antigen processing", ["H2-K1", "B2m", "Tap1"], -1),
        ],
    ),
    PaperFinding(
        finding_id="tf_chemokine_axis",
        title="TF–chemokine receptor axis (paper model)",
        paper_context="Paper couples cell states to chemotaxis via TF–chemokine receptor programs. "
        "Test chemokine receptor module shifts under immune KOs.",
        perturbation="Il4ra",
        cell_types=["immune", "myeloid"],
        modules=[
            ModuleHypothesis("Chemokine receptors", ["Ccr2", "Ccr5", "Ccr7", "Cxcr3", "Cxcr4"], -1),
            ModuleHypothesis("Chemokine ligands", ["Ccl2", "Ccl5", "Cxcl9", "Cxcl10"], -1),
        ],
    ),
]


def observed_delta(pool: sc.AnnData, perturb: str, cell_type: str, genes: list[str]) -> pd.Series:
    common = [g for g in genes if g in pool.var_names]
    if not common:
        return pd.Series(dtype=float)
    expr = dense(pool, common)
    ntc = (pool.obs["cell_type"].astype(str) == cell_type) & (pool.obs["target_gene"].astype(str) == "non-targeting")
    per = (pool.obs["cell_type"].astype(str) == cell_type) & (pool.obs["target_gene"].astype(str) == perturb)
    if ntc.sum() < 1 or per.sum() < 1:
        return pd.Series(dtype=float)
    return expr.loc[per.values].mean(0) - expr.loc[ntc.values].mean(0)


def predicted_delta(
    baseline: sc.AnnData,
    pred: pd.DataFrame,
    cell_type: str,
    slice_id: str,
    genes: list[str],
) -> pd.Series:
    sub = baseline[baseline.obs["cell_type"].astype(str) == cell_type]
    if "slice_id" in baseline.obs.columns:
        sub = sub[sub.obs["slice_id"].astype(str) == slice_id]
    sub = sub[sub.obs_names.isin(pred.index)]
    common = [g for g in genes if g in sub.var_names and g in pred.columns]
    if sub.n_obs < 5 or not common:
        return pd.Series(dtype=float)
    expr = dense(sub, common)
    pr = pred.loc[sub.obs_names, common]
    return (pr - expr).mean(0)


def score_gene(obs: float, pred: float, expected: int) -> dict:
    if not np.isfinite(obs) or not np.isfinite(pred):
        return {"obs_sign_ok": np.nan, "pred_sign_ok": np.nan, "concordant": np.nan}
    obs_ok = int(np.sign(obs) == expected or (expected == -1 and obs <= 0) or (expected == +1 and obs >= 0))
    pred_ok = int(np.sign(pred) == expected or (expected == -1 and pred <= 0) or (expected == +1 and pred >= 0))
    conc = int(np.sign(obs) == np.sign(pred)) if obs != 0 and pred != 0 else np.nan
    return {"obs_sign_ok": obs_ok, "pred_sign_ok": pred_ok, "concordant": conc}


def predicted_delta_pooled(
    baseline: sc.AnnData,
    pred: pd.DataFrame,
    cell_type: str,
    slices: list[str],
    genes: list[str],
) -> pd.Series:
    parts_obs, parts_pred = [], []
    for sl in slices:
        sub = baseline[baseline.obs["cell_type"].astype(str) == cell_type]
        if "slice_id" in baseline.obs.columns:
            sub = sub[sub.obs["slice_id"].astype(str) == sl]
        sub = sub[sub.obs_names.isin(pred.index)]
        common = [g for g in genes if g in sub.var_names and g in pred.columns]
        if sub.n_obs < 3 or not common:
            continue
        parts_obs.append(dense(sub, common))
        parts_pred.append(pred.loc[sub.obs_names, common])
    if not parts_obs:
        return pd.Series(dtype=float)
    obs = pd.concat(parts_obs).mean(0)
    pr = pd.concat(parts_pred).mean(0)
    return pr - obs


def evaluate_finding(
    finding: PaperFinding,
    slices: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    pred_dir: Path,
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = pred_dir / f"predicted_KO_{finding.perturbation}.feather"
    if not pred_path.exists():
        print(f"  skip {finding.finding_id}: no prediction for {finding.perturbation}")
        return pd.DataFrame(), pd.DataFrame()

    pred = pd.read_feather(pred_path).set_index("CellID")
    gene_rows = []
    module_rows = []

    for cell_type in finding.cell_types:
        if finding.pool_slices:
            pools = []
            for sl in slices:
                pool = load_pool_for_finding(sl, finding.perturbation, data_root, finding.sparse_from_guides)
                if pool is None:
                    continue
                pools.append(pool)
            if not pools:
                continue
            pool_all = sc.concat(pools, join="outer")
            ntc_n = int(((pool_all.obs["cell_type"].astype(str) == cell_type) & (pool_all.obs["target_gene"].astype(str) == "non-targeting")).sum())
            pert_n = int(((pool_all.obs["cell_type"].astype(str) == cell_type) & (pool_all.obs["target_gene"].astype(str) == finding.perturbation)).sum())
            if ntc_n < finding.min_ntc or pert_n < finding.min_pert:
                continue
            for mod in finding.modules:
                o = observed_delta(pool_all, finding.perturbation, cell_type, mod.genes)
                p = predicted_delta_pooled(baseline, pred, cell_type, slices, mod.genes)
                for g in mod.genes:
                    if g not in o.index or g not in p.index:
                        continue
                    sc_g = score_gene(float(o[g]), float(p[g]), mod.expected_sign)
                    gene_rows.append({
                        "finding_id": finding.finding_id,
                        "title": finding.title,
                        "perturbation": finding.perturbation,
                        "cell_type": cell_type,
                        "slice": "pooled",
                        "module": mod.name,
                        "expected_sign": mod.expected_sign,
                        "gene": g,
                        "obs_delta": float(o[g]),
                        "pred_delta": float(p[g]),
                        **sc_g,
                        "tag": tag,
                    })
            all_genes = [r for r in gene_rows if r["finding_id"] == finding.finding_id and r["cell_type"] == cell_type]
            if not all_genes:
                continue
            gdf = pd.DataFrame(all_genes)
            for mod_name, sub in gdf.groupby("module"):
                exp = int(sub.expected_sign.iloc[0])
                n = len(sub)
                module_rows.append({
                    "finding_id": finding.finding_id,
                    "title": finding.title,
                    "perturbation": finding.perturbation,
                    "cell_type": cell_type,
                    "module": mod_name,
                    "expected_sign": exp,
                    "n_genes_scored": n,
                    "obs_sign_match_rate": float(sub.obs_sign_ok.mean()),
                    "pred_sign_match_rate": float(sub.pred_sign_ok.mean()),
                    "obs_pred_concordance_rate": float(sub.concordant.mean()),
                    "obs_binom_p": float(stats.binomtest(int(sub.obs_sign_ok.sum()), n, 0.5, alternative="greater").pvalue),
                    "pred_binom_p": float(stats.binomtest(int(sub.pred_sign_ok.sum()), n, 0.5, alternative="greater").pvalue),
                    "obs_mean_delta": float(sub.obs_delta.mean()),
                    "pred_mean_delta": float(sub.pred_delta.mean()),
                    "paper_recapitulated_obs": float(sub.obs_sign_ok.mean()) >= 0.6,
                    "paper_recapitulated_pred": float(sub.pred_sign_ok.mean()) >= 0.6,
                    "tag": tag,
                })
            continue

        for sl in slices:
            pool = load_pool_for_finding(sl, finding.perturbation, data_root, finding.sparse_from_guides)
            if pool is None:
                continue
            ntc_n = int(((pool.obs["cell_type"].astype(str) == cell_type) & (pool.obs["target_gene"].astype(str) == "non-targeting")).sum())
            pert_n = int(((pool.obs["cell_type"].astype(str) == cell_type) & (pool.obs["target_gene"].astype(str) == finding.perturbation)).sum())
            if ntc_n < finding.min_ntc or pert_n < max(5, finding.min_pert // len(slices)):
                continue
            for mod in finding.modules:
                o = observed_delta(pool, finding.perturbation, cell_type, mod.genes)
                p = predicted_delta(baseline, pred, cell_type, sl, mod.genes)
                for g in mod.genes:
                    if g not in o.index or g not in p.index:
                        continue
                    sc_g = score_gene(float(o[g]), float(p[g]), mod.expected_sign)
                    gene_rows.append({
                        "finding_id": finding.finding_id,
                        "title": finding.title,
                        "perturbation": finding.perturbation,
                        "cell_type": cell_type,
                        "slice": sl,
                        "module": mod.name,
                        "expected_sign": mod.expected_sign,
                        "gene": g,
                        "obs_delta": float(o[g]),
                        "pred_delta": float(p[g]),
                        **sc_g,
                        "tag": tag,
                    })

        all_genes = [r for r in gene_rows if r["finding_id"] == finding.finding_id and r["cell_type"] == cell_type]
        if not all_genes:
            continue
        gdf = pd.DataFrame(all_genes)
        for mod_name, sub in gdf.groupby("module"):
            exp = int(sub.expected_sign.iloc[0])
            obs_rate = float(sub.obs_sign_ok.mean())
            pred_rate = float(sub.pred_sign_ok.mean())
            conc_rate = float(sub.concordant.mean())
            n = len(sub)
            obs_p = float(stats.binomtest(int(sub.obs_sign_ok.sum()), n, 0.5, alternative="greater").pvalue) if n else np.nan
            pred_p = float(stats.binomtest(int(sub.pred_sign_ok.sum()), n, 0.5, alternative="greater").pvalue) if n else np.nan
            module_rows.append({
                "finding_id": finding.finding_id,
                "title": finding.title,
                "perturbation": finding.perturbation,
                "cell_type": cell_type,
                "module": mod_name,
                "expected_sign": exp,
                "n_genes_scored": n,
                "obs_sign_match_rate": obs_rate,
                "pred_sign_match_rate": pred_rate,
                "obs_pred_concordance_rate": conc_rate,
                "obs_binom_p": obs_p,
                "pred_binom_p": pred_p,
                "obs_mean_delta": float(sub.obs_delta.mean()),
                "pred_mean_delta": float(sub.pred_delta.mean()),
                "paper_recapitulated_obs": obs_rate >= 0.6,
                "paper_recapitulated_pred": pred_rate >= 0.6,
                "tag": tag,
            })

    return pd.DataFrame(gene_rows), pd.DataFrame(module_rows)


def evaluate_finding_obs_only(
    finding: PaperFinding,
    slices: list[str],
    data_root: Path,
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score observed SPAC-seq only (no SpaceTravLR predictions required)."""
    gene_rows: list[dict] = []
    module_rows: list[dict] = []

    for cell_type in finding.cell_types:
        pools = []
        for sl in slices:
            pool = load_pool_for_finding(sl, finding.perturbation, data_root, finding.sparse_from_guides)
            if pool is None:
                continue
            pools.append(pool)
        if not pools:
            continue
        pool_all = sc.concat(pools, join="outer") if finding.pool_slices or len(pools) > 1 else pools[0]
        ntc_n = int(((pool_all.obs["cell_type"].astype(str) == cell_type) & (pool_all.obs["target_gene"].astype(str) == "non-targeting")).sum())
        pert_n = int(((pool_all.obs["cell_type"].astype(str) == cell_type) & (pool_all.obs["target_gene"].astype(str) == finding.perturbation)).sum())
        if ntc_n < finding.min_ntc or pert_n < finding.min_pert:
            continue
        for mod in finding.modules:
            o = observed_delta(pool_all, finding.perturbation, cell_type, mod.genes)
            for g in mod.genes:
                if g not in o.index:
                    continue
                obs = float(o[g])
                obs_ok = int(
                    np.sign(obs) == mod.expected_sign
                    or (mod.expected_sign == -1 and obs <= 0)
                    or (mod.expected_sign == +1 and obs >= 0)
                )
                gene_rows.append({
                    "finding_id": finding.finding_id,
                    "title": finding.title,
                    "perturbation": finding.perturbation,
                    "cell_type": cell_type,
                    "slice": "pooled" if finding.pool_slices or len(slices) > 1 else slices[0],
                    "module": mod.name,
                    "expected_sign": mod.expected_sign,
                    "gene": g,
                    "obs_delta": obs,
                    "pred_delta": np.nan,
                    "obs_sign_ok": obs_ok,
                    "pred_sign_ok": np.nan,
                    "concordant": np.nan,
                    "tag": tag,
                })
        ct_genes = [r for r in gene_rows if r["finding_id"] == finding.finding_id and r["cell_type"] == cell_type]
        if not ct_genes:
            continue
        gdf = pd.DataFrame(ct_genes)
        for mod_name, sub in gdf.groupby("module"):
            n = len(sub)
            obs_rate = float(sub.obs_sign_ok.mean())
            module_rows.append({
                "finding_id": finding.finding_id,
                "title": finding.title,
                "perturbation": finding.perturbation,
                "cell_type": cell_type,
                "module": mod_name,
                "expected_sign": int(sub.expected_sign.iloc[0]),
                "n_genes_scored": n,
                "obs_sign_match_rate": obs_rate,
                "pred_sign_match_rate": np.nan,
                "obs_pred_concordance_rate": np.nan,
                "obs_binom_p": float(stats.binomtest(int(sub.obs_sign_ok.sum()), n, 0.5, alternative="greater").pvalue),
                "pred_binom_p": np.nan,
                "obs_mean_delta": float(sub.obs_delta.mean()),
                "pred_mean_delta": np.nan,
                "paper_recapitulated_obs": obs_rate >= 0.6,
                "paper_recapitulated_pred": np.nan,
                "tag": tag,
            })

    return pd.DataFrame(gene_rows), pd.DataFrame(module_rows)


def plot_scorecard(mod_df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    if mod_df.empty:
        return
    show = mod_df.copy()
    show["label"] = show.apply(
        lambda r: f"{r['finding_id'][:12]}\n{r['perturbation']}|{r['cell_type']}\n{r['module'][:22]}",
        axis=1,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, 0.35 * len(show))))
    for ax, col, title in zip(
        axes,
        ["obs_sign_match_rate", "pred_sign_match_rate", "obs_pred_concordance_rate"],
        ["SPAC-seq matches paper", "SpaceTravLR matches paper", "Obs ↔ Pred concordance"],
    ):
        vals = show[col].fillna(0).values
        colors = ["#2166ac" if v >= 0.6 else "#67a9cf" if v >= 0.5 else "#d1e5f0" for v in vals]
        ax.barh(range(len(show)), vals, color=colors, edgecolor="k", linewidth=0.3)
        ax.axvline(0.6, color="#b2182b", ls="--", lw=1, label="60% threshold")
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels(show.label, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction genes with expected sign")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Paper finding recapitulation scorecard (Zhang et al. Cell 2026)\n"
        "Blue ≥60% = hypothesis supported at module level",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig13_paper_findings_scorecard_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_module_heatmap(mod_df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    if mod_df.empty:
        return
    pivot_obs = mod_df.pivot_table(
        index=["finding_id", "module"], columns="cell_type",
        values="obs_sign_match_rate", aggfunc="first",
    )
    pivot_pred = mod_df.pivot_table(
        index=["finding_id", "module"], columns="cell_type",
        values="pred_sign_match_rate", aggfunc="first",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * len(pivot_obs))))
    for ax, pivot, title in zip(axes, [pivot_obs, pivot_pred], ["Observed (SPAC-seq)", "Predicted (SpaceTravLR)"]):
        if pivot.empty:
            ax.axis("off")
            continue
        vals = pivot.fillna(0).values.astype(float)
        im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([f"{a}|{b}" for a, b in pivot.index], fontsize=7)
        ax.set_title(title, fontweight="bold")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.6, label="Sign match rate")
    fig.suptitle("Module-level paper hypothesis support", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig14_paper_modules_heatmap_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_prep/baseline_ntc_0c6fbac5e6cd947c_fullprep.h5ad")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/paper_findings")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/paper_findings")
    ap.add_argument("--tag", default="tuned")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    baseline = load_baseline(args.baseline_h5ad)
    all_genes, all_mods = [], []
    for finding in PAPER_FINDINGS:
        print(f"Evaluating: {finding.title} …")
        gdf, mdf = evaluate_finding(finding, args.slices, args.data_root, baseline, args.pred_dir, args.tag)
        all_genes.append(gdf)
        all_mods.append(mdf)

    gene_df = pd.concat(all_genes, ignore_index=True) if all_genes else pd.DataFrame()
    mod_df = pd.concat(all_mods, ignore_index=True) if all_mods else pd.DataFrame()
    gene_df.to_csv(args.out_dir / f"gene_level_{args.tag}.csv", index=False)
    mod_df.to_csv(args.out_dir / f"hypothesis_scores_{args.tag}.csv", index=False)

    n_obs_hit = int(mod_df.paper_recapitulated_obs.sum()) if not mod_df.empty else 0
    n_pred_hit = int(mod_df.paper_recapitulated_pred.sum()) if not mod_df.empty else 0
    n_mod = len(mod_df)
    overall = {
        "tag": args.tag,
        "paper": "Zhang et al. Cell 2026 SPAC-seq",
        "n_findings": len(PAPER_FINDINGS),
        "n_module_tests": n_mod,
        "n_modules_obs_supports_paper": n_obs_hit,
        "n_modules_pred_supports_paper": n_pred_hit,
        "frac_obs_support": float(n_obs_hit / n_mod) if n_mod else np.nan,
        "frac_pred_support": float(n_pred_hit / n_mod) if n_mod else np.nan,
        "best_pred_modules": mod_df.nlargest(8, "pred_sign_match_rate").to_dict("records") if not mod_df.empty else [],
        "obs_only_modules": mod_df[mod_df.paper_recapitulated_obs & ~mod_df.paper_recapitulated_pred].to_dict("records") if not mod_df.empty else [],
        "pred_only_modules": mod_df[~mod_df.paper_recapitulated_obs & mod_df.paper_recapitulated_pred].to_dict("records") if not mod_df.empty else [],
        "both_support": mod_df[mod_df.paper_recapitulated_obs & mod_df.paper_recapitulated_pred].to_dict("records") if not mod_df.empty else [],
    }
    (args.out_dir / f"overall_{args.tag}.json").write_text(json.dumps(overall, indent=2, default=str))
    print(json.dumps(overall, indent=2, default=str))

    plot_scorecard(mod_df, args.fig_dir, args.tag)
    plot_module_heatmap(mod_df, args.fig_dir, args.tag)


if __name__ == "__main__":
    main()
