"""CNN per-cell beta microniches and guide-enrichment prediction utilities."""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors

CELLTYPE_TO_CLUSTER = {"fibroblast": 0, "immune": 1, "myeloid": 2, "tumor": 3}

IMMUNE_EXCLUSION_UP = ["Spp1", "Cd163", "Mrc1", "Tgfb1", "Vegfa", "Arg1"]
IMMUNE_EXCLUSION_DN = ["Cxcl9", "Cxcl10", "Cd8a", "Ifit3", "Itgal", "Gzmb", "Prf1"]
IMMUNE_INFILTRATION_UP = ["Cxcl9", "Cxcl10", "Cd8a", "Cd3e", "Gzmb", "Ifit3", "Ccl5"]
IMMUNE_INFILTRATION_DN = ["Spp1", "Cd163", "Foxp3", "Il10"]

PERT_ENRICHMENT_PROFILE = {
    "Icam1": {
        "exclusion_sign": -1.0,
        "cnn_weight": -0.5,
        "escape_up": IMMUNE_EXCLUSION_UP,
        "escape_dn": IMMUNE_EXCLUSION_DN,
    },
    "Bcam": {"exclusion_sign": +0.7, "escape_up": ["Spp1", "Mmp9", "Postn", "Col1a2"], "escape_dn": ["Cd8a", "Cxcl9"]},
    "Il4ra": {"exclusion_sign": -0.6, "escape_up": IMMUNE_INFILTRATION_UP, "escape_dn": IMMUNE_INFILTRATION_DN},
    "Cd83": {"exclusion_sign": -0.5, "escape_up": IMMUNE_INFILTRATION_UP, "escape_dn": IMMUNE_INFILTRATION_DN},
    "Cd74": {"exclusion_sign": -0.4, "escape_up": ["H2-Aa", "H2-Ab1", "Cd74", "Cd8a"], "escape_dn": IMMUNE_EXCLUSION_UP},
    "Cks1b": {"exclusion_sign": 0.0, "escape_up": ["Pcna", "Mki67"], "escape_dn": []},
    "Ptk6": {"exclusion_sign": 0.0, "escape_up": ["Vim", "Snai1"], "escape_dn": ["Epcam", "Cdh1"]},
}

PAPER_LUNG_GENES = [
    "Icam1", "Spp1", "Cxcl9", "Cxcl10", "Cd8a", "Stat1", "Ifit3",
    "Cd163", "H2-Aa", "Bcam", "Itgal", "Itgb2",
]

MICRONICHE_PATHWAY_EXTRA = {
    "Immune exclusion index": IMMUNE_EXCLUSION_UP + IMMUNE_EXCLUSION_DN,
    "Immune infiltration index": IMMUNE_INFILTRATION_UP + IMMUNE_INFILTRATION_DN,
    "Spp1 / osteopontin": ["Spp1", "Cd44", "Itgav", "Itgb1", "Fn1", "Mmp9"],
    "LFA-1 immune synapse": ["Itgal", "Itgb2", "Icam1"],
    "Proliferation": ["Pcna", "Mki67", "Top2a"],
    "EMT / mesenchymal": ["Vim", "Snai1", "Zeb1", "Cdh1", "Epcam"],
}

PERT_PATHWAY_EXPECTED_SIGN: dict[tuple[str, str], int] = {
    ("Icam1", "Interferon response"): -1,
    ("Icam1", "Immune exclusion index"): +1,
    ("Icam1", "M2/suppressive macrophage"): +1,
    ("Icam1", "T-cell effector"): -1,
    ("Icam1", "LFA-1 immune synapse"): -1,
    ("Icam1", "Spp1 / osteopontin"): +1,
    ("Bcam", "Spp1 / osteopontin"): +1,
    ("Bcam", "ECM/fibroblast"): +1,
    ("Bcam", "T-cell exhaustion / Treg"): +1,
    ("Il4ra", "Antigen presentation (MHC-II)"): -1,
    ("Il4ra", "Immune infiltration index"): -1,
    ("Il4ra", "M1/inflam macrophage"): -1,
    ("Cd83", "Antigen presentation (MHC-II)"): -1,
    ("Cd83", "Immune infiltration index"): -1,
    ("Cd74", "Antigen presentation (MHC-II)"): -1,
    ("Cks1b", "Proliferation"): -1,
    ("Ptk6", "EMT / mesenchymal"): +1,
}

PATHWAY_KIND_PRED_WEIGHTS: dict[str, dict[str, float]] = {
    "immune": {"exclusion": 0.65, "escape": 0.35},
    "antigen": {"pathway": 0.45, "escape": 0.30, "cnn": 0.25},
    "prolif": {"pathway": 0.40, "cnn": 0.40, "escape": 0.20},
    "emt": {"pathway": 0.50, "escape": 0.35, "cnn": 0.15},
    "spp1_ecm": {"pathway": 0.45, "exclusion": 0.30, "escape": 0.25},
    "tcell_exh": {"exclusion": 0.50, "pathway": 0.35, "escape": 0.15},
    "default": {"pathway": 0.50, "escape": 0.30, "cnn": 0.20},
}

PERT_PATHWAY_PRED_WEIGHTS: dict[tuple[str, str], dict[str, float]] = {
    ("Il4ra", "Antigen presentation (MHC-II)"): {"pathway": 0.70, "escape": 0.15, "cnn": 0.15},
    ("Cd83", "Antigen presentation (MHC-II)"): {"pathway": 0.70, "escape": 0.15, "cnn": 0.15},
    ("Cd74", "Antigen presentation (MHC-II)"): {"pathway": 0.50, "escape": 0.30, "cnn": 0.20},
    ("Il4ra", "Immune infiltration index"): {"exclusion": 0.70, "escape": 0.30},
    ("Cd83", "Immune infiltration index"): {"exclusion": 0.70, "escape": 0.30},
    ("Il4ra", "M1/inflam macrophage"): {"exclusion": 0.50, "escape": 0.30, "pathway": 0.20},
    ("Icam1", "Immune exclusion index"): {"exclusion": 0.70, "escape": 0.30},
    ("Icam1", "Interferon response"): {"exclusion": 0.45, "escape": 0.35, "pathway": 0.20},
    ("Icam1", "Spp1 / osteopontin"): {"exclusion": 0.40, "escape": 0.35, "pathway": 0.25},
    ("Cks1b", "Proliferation"): {"exclusion": 0.45, "cnn": 0.45, "escape": 0.10},
    ("Ptk6", "EMT / mesenchymal"): {"pathway": 0.50, "escape": 0.35, "cnn": 0.15},
    ("Bcam", "Spp1 / osteopontin"): {"pathway": 0.45, "exclusion": 0.30, "escape": 0.25},
    ("Bcam", "ECM/fibroblast"): {"pathway": 0.50, "exclusion": 0.25, "escape": 0.25},
    ("Bcam", "T-cell exhaustion / Treg"): {"exclusion": 0.65, "escape": 0.35},
}


def _pathway_kind_key(pathway: str) -> str:
    if "Immune infiltration" in pathway or "Immune exclusion" in pathway:
        return "immune"
    if any(k in pathway for k in ("Antigen", "M1/", "M2/", "Interferon", "T-cell effector")):
        return "antigen"
    if "Proliferation" in pathway:
        return "prolif"
    if "EMT" in pathway:
        return "emt"
    if "Spp1" in pathway or "ECM" in pathway:
        return "spp1_ecm"
    if "T-cell exhaustion" in pathway or "Treg" in pathway:
        return "tcell_exh"
    return "default"


def _pathway_pred_weights(perturb: str, pathway: str) -> dict[str, float]:
    return PERT_PATHWAY_PRED_WEIGHTS.get(
        (perturb, pathway), PATHWAY_KIND_PRED_WEIGHTS[_pathway_kind_key(pathway)]
    )


def _zscore_niche_signal(values: np.ndarray) -> np.ndarray | None:
    ok = np.isfinite(values)
    if ok.sum() < 2 or np.nanstd(values[ok]) < 1e-8:
        return None
    out = np.full_like(values, np.nan, dtype=np.float64)
    v = values[ok]
    out[ok] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def _compose_pathway_niche_pred(
    df: pd.DataFrame,
    esc_df: pd.DataFrame,
    perturb: str,
    pathway: str,
    *,
    gshift: float = 0.0,
) -> pd.DataFrame:
    """Blend cell-level pathway Δ with enrichment niche signals (pathway-specific weights)."""
    if df.empty:
        return df
    merged = df.merge(esc_df, on="niche", how="left", suffixes=("", "_enr"))
    weights = _pathway_pred_weights(perturb, pathway)
    signal_cols = {
        "pathway": "pred_pathway_raw",
        "escape": "pred_escape_gain",
        "exclusion": "pred_exclusion_index",
        "cnn": "pred_cnn_target_score",
    }
    comp = np.zeros(len(merged), dtype=np.float64)
    total_w = 0.0
    for key, col in signal_cols.items():
        wt = weights.get(key, 0.0)
        if wt <= 0 or col not in merged.columns:
            continue
        z = _zscore_niche_signal(merged[col].to_numpy(dtype=float))
        if z is None:
            continue
        comp += wt * np.where(np.isfinite(z), z, 0.0)
        total_w += wt
    out = df.copy()
    if total_w < 1e-8:
        return out
    comp /= total_w
    scale = max(abs(gshift), float(np.nanstd(out["pred_pathway_raw"].to_numpy(dtype=float))), 0.35)
    if np.nanstd(comp) > 1e-8:
        comp = comp / np.nanstd(comp) * scale
    out["pred_pathway_delta"] = comp
    return out


def microniche_pathway_gene_sets(base_sets: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    out = dict(base_sets or {})
    for name, genes in MICRONICHE_PATHWAY_EXTRA.items():
        out[name] = list(dict.fromkeys(genes))
    return out


def pathway_module_score(ad: sc.AnnData, genes: list[str], *, signed_index: bool = False) -> np.ndarray:
    """Mean log-expression of pathway genes; optional up-minus-down for index pathways."""
    if signed_index:
        pair = _signed_index_genes(genes)
        if pair is not None:
            up, dn = pair
            return niche_expression_index(ad, up, dn)
    keep = [g for g in genes if g in ad.var_names]
    if len(keep) < 2:
        return np.full(ad.n_obs, np.nan)
    expr = _dense(ad, keep)
    return expr.mean(axis=1).to_numpy(dtype=np.float64)


def _signed_index_genes(genes: list[str]) -> tuple[list[str], list[str]] | None:
    s = set(genes)
    if s >= set(IMMUNE_EXCLUSION_UP) | set(IMMUNE_EXCLUSION_DN):
        return IMMUNE_EXCLUSION_UP, IMMUNE_EXCLUSION_DN
    if s >= set(IMMUNE_INFILTRATION_UP) | set(IMMUNE_INFILTRATION_DN):
        return IMMUNE_INFILTRATION_UP, IMMUNE_INFILTRATION_DN
    return None


def pathway_distinctness_ntc(
    pool: sc.AnnData,
    pathways: dict[str, list[str]],
    niche_key: str = "cnn_leiden",
    cell_type: str = "tumor",
    min_cells: int = 5,
) -> list[dict]:
    rows: list[dict] = []
    ct = pool[(pool.obs["cell_type"].astype(str) == cell_type) & (pool.obs["target_gene"].astype(str) == "non-targeting")]
    if niche_key not in ct.obs.columns or ct.n_obs < 30:
        return rows
    niches = ct.obs[niche_key].astype(str)
    valid = sorted(n for n in niches.unique() if n not in ("unassigned", "nan", ""))
    if len(valid) < 2:
        return rows
    signed = {"Immune exclusion index", "Immune infiltration index"}
    for pathway, genes in pathways.items():
        scores = pathway_module_score(ct, genes, signed_index=pathway in signed)
        groups = [scores[niches.to_numpy() == n] for n in valid if int((niches == n).sum()) >= min_cells]
        groups = [g[np.isfinite(g)] for g in groups if len(g[np.isfinite(g)]) >= min_cells]
        if len(groups) < 2:
            continue
        try:
            h, p = stats.kruskal(*groups)
        except ValueError:
            continue
        rows.append({
            "pathway": pathway,
            "n_genes": len([g for g in genes if g in ct.var_names]),
            "n_niches": len(groups),
            "kruskal_H": float(h),
            "kruskal_p": float(p),
            "significant": bool(p < 0.05),
        })
    return rows


def _pathway_prediction_genes(perturb: str, genes: list[str]) -> list[str]:
    profile = PERT_ENRICHMENT_PROFILE.get(perturb, {})
    pri = set(profile.get("escape_up", [])) | set(profile.get("escape_dn", [])) | {perturb}
    pri_genes = [g for g in genes if g in pri]
    return pri_genes if len(pri_genes) >= 2 else genes


def global_pathway_shift(
    pred: pd.DataFrame,
    baseline: sc.AnnData,
    genes: list[str],
    *,
    signed: bool = False,
    cell_type: str = "tumor",
) -> float:
    """Global in-silico KO shift for a pathway gene set (pooled pred vs baseline)."""
    base_ct = baseline[baseline.obs["cell_type"].astype(str) == cell_type]
    common = [g for g in genes if g in pred.columns and g in base_ct.var_names]
    if len(common) < 2:
        return 0.0
    common_idx = pred.index.intersection(base_ct.obs_names)
    if len(common_idx) < 10:
        common_idx = pred.index
    base = _dense(base_ct, common).reindex(common_idx)
    pr = pred.reindex(common_idx)[common]
    if signed:
        pair = _signed_index_genes(genes)
        if pair is not None:
            up = [g for g in pair[0] if g in common]
            dn = [g for g in pair[1] if g in common]
            pred_ix = (pr[up].mean(axis=1).mean() if up else 0.0) - (pr[dn].mean(axis=1).mean() if dn else 0.0)
            base_ix = (base[up].mean(axis=1).mean() if up else 0.0) - (base[dn].mean(axis=1).mean() if dn else 0.0)
            return float(pred_ix - base_ix)
    return float(pr[common].mean().mean() - base[common].mean().mean())


def _niche_cnn_scores(
    pool_ct: sc.AnnData,
    ntc_mask: pd.Series,
    niche_key: str,
    prep_names: pd.Index,
    slice_id: str,
    cell_cnn_scores: pd.Series | None,
) -> dict[str, float]:
    if cell_cnn_scores is None:
        return {}
    out: dict[str, float] = {}
    for niche, sub in pool_ct[ntc_mask].obs.groupby(niche_key, observed=True):
        if str(niche) in ("unassigned", "nan", ""):
            continue
        prep_ids = pd.Index([
            map_pool_to_prep(slice_id, b, prep_names) or prep_barcode(slice_id, b)
            for b in sub.index
        ])
        by_prep = cell_cnn_scores.reindex(prep_ids)
        by_pool = cell_cnn_scores.reindex(sub.index)
        vals = by_prep if by_prep.notna().sum() >= by_pool.notna().sum() else by_pool
        if vals.notna().any():
            out[str(niche)] = float(vals.mean())
    return out


def _apply_pathway_cnn_modulation(
    df: pd.DataFrame,
    cnn_by_niche: dict[str, float],
    *,
    cnn_weight: float = 0.35,
) -> pd.DataFrame:
    if df.empty or not cnn_by_niche:
        return df
    cnn = df["niche"].astype(str).map(cnn_by_niche)
    if cnn.notna().sum() < 2:
        return df
    z_cnn = pd.Series(_zscore(cnn.to_numpy(dtype=float)), index=df.index)
    raw = df["pred_pathway_delta"].to_numpy(dtype=float)
    scale = float(np.nanstd(raw[np.isfinite(raw)])) if np.isfinite(raw).sum() >= 2 else 1.0
    if scale < 1e-8:
        scale = 1.0
    df = df.copy()
    ok = np.isfinite(raw)
    df.loc[ok, "pred_pathway_delta"] = raw[ok] + cnn_weight * scale * z_cnn.loc[ok].to_numpy()
    return df


def _pathway_enrichment_proxy(
    esc_df: pd.DataFrame,
    pathway: str,
    perturb: str,
    profile: dict,
) -> np.ndarray:
    """Map enrichment niche z-scores to a pathway-specific predicted Δ."""
    sign = float(profile.get("exclusion_sign", 0.0))
    exp_sign = PERT_PATHWAY_EXPECTED_SIGN.get((perturb, pathway))
    orient = float(exp_sign if exp_sign is not None else sign if sign != 0 else 1.0)
    z_exc = esc_df["z_pred_exclusion_index"].to_numpy(dtype=float)
    z_esc = esc_df["z_pred_escape_gain"].to_numpy(dtype=float)
    z_cnn = esc_df["z_pred_cnn_target_score"].to_numpy(dtype=float)
    if pathway in {"Immune exclusion index", "Immune infiltration index"}:
        return orient * z_exc + z_esc
    if "Antigen" in pathway or "M1" in pathway or "M2" in pathway or "T-cell" in pathway or "Interferon" in pathway:
        return z_esc + 0.35 * z_cnn
    if "Spp1" in pathway or "ECM" in pathway:
        return z_esc + 0.5 * z_exc
    if "Proliferation" in pathway:
        return z_esc + 0.35 * z_cnn
    if "EMT" in pathway:
        return z_esc - 0.35 * z_cnn
    return esc_df["pred_enrichment_score"].to_numpy(dtype=float)


def pathway_niche_deltas(
    prep: sc.AnnData,
    pool: sc.AnnData,
    pred: pd.DataFrame,
    perturb: str,
    pathway: str,
    genes: list[str],
    niche_key: str,
    slice_id: str,
    cell_type: str = "tumor",
    min_ntc: int = 2,
    min_pert: int = 2,
    global_baseline: sc.AnnData | None = None,
    cell_cnn_scores: pd.Series | None = None,
    score_genes: list[str] | None = None,
) -> pd.DataFrame:
    """Per-niche observed and predicted pathway Δ for one perturbation."""
    signed = pathway in {"Immune exclusion index", "Immune infiltration index"}
    pred_genes = _pathway_prediction_genes(perturb, genes)
    pool_ct = pool[pool.obs["cell_type"].astype(str) == cell_type].copy()
    if niche_key not in pool_ct.obs.columns or pool_ct.n_obs == 0:
        return pd.DataFrame()

    ntc_mask = pool_ct.obs["target_gene"].astype(str) == "non-targeting"
    pert_mask = pool_ct.obs["target_gene"].astype(str) == perturb
    pool_scores = pathway_module_score(pool_ct, genes, signed_index=signed)

    prep_names = prep.obs_names
    pred_ids = pd.Index([
        map_pool_to_prep(slice_id, b, prep_names) or prep_barcode(slice_id, b)
        for b in pool_ct.obs_names[ntc_mask]
    ])
    ref_base = global_baseline if global_baseline is not None else prep
    base_sub = ref_base[ref_base.obs["cell_type"].astype(str) == cell_type].copy()
    if "slice_id" in base_sub.obs.columns and slice_id:
        if slice_id.startswith("subQ"):
            base_sub = base_sub[base_sub.obs["slice_id"].astype(str) == slice_id]
    aligned = pred_ids.intersection(pred.index).intersection(base_sub.obs_names)
    use_global_pred = len(aligned) < max(10, 0.2 * len(pred_ids))
    pred_delta_cells: pd.Series | None = None
    if not use_global_pred and len(aligned) >= 10:
        common = [g for g in pred_genes if g in pred.columns and g in base_sub.var_names]
        if len(common) >= 2:
            base_expr = _dense(base_sub, common).reindex(aligned)
            pred_expr = pred.reindex(aligned)[common]
            if signed:
                up = [g for g in (IMMUNE_EXCLUSION_UP if pathway == "Immune exclusion index" else IMMUNE_INFILTRATION_UP) if g in common]
                dn = [g for g in (IMMUNE_EXCLUSION_DN if pathway == "Immune exclusion index" else IMMUNE_INFILTRATION_DN) if g in common]
                pred_delta_cells = pd.Series(
                    (pred_expr[up].mean(axis=1) if up else 0)
                    - (pred_expr[dn].mean(axis=1) if dn else 0)
                    - ((base_expr[up].mean(axis=1) if up else 0) - (base_expr[dn].mean(axis=1) if dn else 0)),
                    index=aligned,
                )
            else:
                pred_delta_cells = pred_expr.mean(axis=1) - base_expr.mean(axis=1)

    enrich = observed_log_enrichment(pool, perturb, cell_type, niche_key, min_ntc=min_ntc, min_pert=min_pert)
    enrich_map = enrich.set_index("niche")["obs_log2_enrichment"].to_dict() if not enrich.empty else {}
    cnn_by_niche = _niche_cnn_scores(pool_ct, ntc_mask, niche_key, prep_names, slice_id, cell_cnn_scores)

    rows = []
    niches = sorted(pool_ct.obs[niche_key].astype(str).unique())
    for niche in niches:
        if niche in ("unassigned", "nan", ""):
            continue
        n_ntc = ntc_mask & pool_ct.obs[niche_key].astype(str).eq(niche)
        n_pert = pert_mask & pool_ct.obs[niche_key].astype(str).eq(niche)
        if int(n_ntc.sum()) < min_ntc or int(n_pert.sum()) < min_pert:
            continue
        obs_delta = float(np.nanmean(pool_scores[n_pert.to_numpy()]) - np.nanmean(pool_scores[(ntc_mask & n_ntc).to_numpy()]))
        ntc_baseline = float(np.nanmean(pool_scores[(ntc_mask & n_ntc).to_numpy()]))

        pred_delta = float("nan")
        if pred_delta_cells is not None:
            prep_ntc_ids = pd.Index([
                map_pool_to_prep(slice_id, b, prep_names) or prep_barcode(slice_id, b)
                for b in pool_ct.obs_names[ntc_mask & n_ntc]
            ])
            vals = pred_delta_cells.reindex(prep_ntc_ids).dropna()
            if len(vals) >= min_ntc:
                pred_delta = float(vals.mean())

        rows.append({
            "niche": niche,
            "n_ntc": int(n_ntc.sum()),
            "n_pert": int(n_pert.sum()),
            "obs_pathway_delta": obs_delta,
            "pred_pathway_delta": pred_delta,
            "ntc_pathway_score": ntc_baseline,
            "obs_log2_enrichment": float(enrich_map.get(niche, np.nan)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["pred_pathway_raw"] = df["pred_pathway_delta"]
    profile = PERT_ENRICHMENT_PROFILE.get(perturb, {})
    gshift = global_pathway_shift(pred, ref_base, pred_genes, signed=signed, cell_type=cell_type)

    if not use_global_pred and df["pred_pathway_raw"].isna().all():
        slice_mean = float(df["ntc_pathway_score"].mean())
        cnn_w = float(profile.get("cnn_weight", 0.35))
        z_cnn = _zscore(df["niche"].astype(str).map(cnn_by_niche).to_numpy(dtype=float))
        for i, row in df.iterrows():
            scale = float(row["ntc_pathway_score"] / slice_mean) if abs(slice_mean) > 1e-8 else 1.0
            df.at[i, "pred_pathway_raw"] = gshift * scale + cnn_w * z_cnn[i] * max(abs(gshift), 0.25)

    if score_genes:
        esc_df = predicted_niche_scores(
            prep, pool, pred, perturb, cell_type, niche_key, score_genes, profile,
            cell_cnn_scores, global_baseline=ref_base,
            min_ntc_per_niche=min_ntc, min_pert_per_niche=min_pert,
        )
        if not esc_df.empty:
            df = _compose_pathway_niche_pred(df, esc_df, perturb, pathway, gshift=gshift)
        elif use_global_pred and df["pred_pathway_delta"].isna().all():
            slice_mean = float(df["ntc_pathway_score"].mean())
            cnn_w = float(profile.get("cnn_weight", 0.35))
            z_cnn = _zscore(df["niche"].astype(str).map(cnn_by_niche).to_numpy(dtype=float))
            for i, row in df.iterrows():
                scale = float(row["ntc_pathway_score"] / slice_mean) if abs(slice_mean) > 1e-8 else 1.0
                df.at[i, "pred_pathway_delta"] = gshift * scale + cnn_w * z_cnn[i] * max(abs(gshift), 0.25)

    return df.drop(columns=["pred_pathway_raw"], errors="ignore")


def pathway_concordance_stats(delta_df: pd.DataFrame, *, min_niches: int = 4) -> dict:
    n = len(delta_df)
    if delta_df.empty or n < min_niches:
        return {"n_niches": n, "pearson_r": float("nan"), "spearman_r": float("nan"), "p_pearson": float("nan")}
    obs = delta_df["obs_pathway_delta"].to_numpy(dtype=float)
    pred = delta_df["pred_pathway_delta"].to_numpy(dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    if ok.sum() < min_niches or obs[ok].std() < 1e-8 or pred[ok].std() < 1e-8:
        return {"n_niches": int(ok.sum()), "pearson_r": float("nan"), "spearman_r": float("nan"), "p_pearson": float("nan")}
    r, p = stats.pearsonr(obs[ok], pred[ok])
    rs, _ = stats.spearmanr(obs[ok], pred[ok])
    return {"n_niches": int(ok.sum()), "pearson_r": float(r), "spearman_r": float(rs), "p_pearson": float(p)}


def pathway_enrichment_tie_stats(delta_df: pd.DataFrame) -> dict:
    """Correlate niche sgP enrichment with NTC pathway score and obs pathway Δ."""
    out: dict = {}
    if delta_df.empty or len(delta_df) < 3:
        return out
    y = delta_df["obs_log2_enrichment"].to_numpy(dtype=float)
    for col, key in (("ntc_pathway_score", "or_vs_ntc_pathway"), ("obs_pathway_delta", "or_vs_obs_pathway")):
        x = delta_df[col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or x[ok].std() < 1e-8 or y[ok].std() < 1e-8:
            out[f"{key}_r"] = float("nan")
            out[f"{key}_p"] = float("nan")
            continue
        r, p = stats.pearsonr(x[ok], y[ok])
        out[f"{key}_r"] = float(r)
        out[f"{key}_p"] = float(p)
    return out

# Subset for β-score microniche clustering (perturbation targets + niche modules).
MICRONICHE_CLUSTER_GENES = {
    "Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b", "Ptk6", "Icam1", "Cd44", "Spp1",
    "Cxcl9", "Cxcl10", "Stat1", "Ifit3", "Itgal", "Itgb2", "Cd8a", "Cd163",
    "Mrc1", "Arg1", "H2-Aa", "H2-Ab1", "Col1a2", "Postn", "Vim", "Epcam",
}


def niche_short_label(niche: str) -> str:
    return str(niche).split("|")[-1]


def local_sgp_fraction(
    pool: sc.AnnData,
    perturb: str,
    cell_type: str = "tumor",
    k_neighbors: int = 50,
) -> pd.Series:
    """Per-cell fraction of sgP tumor cells among spatial kNN (continuous ground truth)."""
    ct_mask = pool.obs["cell_type"].astype(str) == cell_type
    ct = pool[ct_mask]
    if ct.n_obs < 5:
        return pd.Series(np.nan, index=pool.obs_names)
    is_sgp = (ct.obs["target_gene"].astype(str) == perturb).to_numpy(dtype=float)
    xy = ct.obsm["spatial"].astype(np.float64)
    k = min(k_neighbors, ct.n_obs - 1)
    if k < 3:
        out = pd.Series(np.nan, index=pool.obs_names)
        out.loc[ct.obs_names] = is_sgp
        return out
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nn.kneighbors(xy)
    local = is_sgp[idx[:, 1:]].mean(axis=1)
    out = pd.Series(np.nan, index=pool.obs_names)
    out.loc[ct.obs_names] = local
    return out


def attach_niche_enrichment_to_cells(
    pool: sc.AnnData,
    enrich_sub: pd.DataFrame,
    perturb: str,
    *,
    niche_key: str = "cnn_leiden",
    k_neighbors: int = 50,
) -> sc.AnnData:
    """Map niche-level predicted/observed enrichment and local sgP fraction onto tumor cells."""
    ad = pool.copy()
    if enrich_sub.empty or niche_key not in ad.obs.columns:
        return ad
    scored = enrich_sub[
        ["niche", "pred_enrichment_score", "obs_log2_enrichment"]
    ].drop_duplicates("niche")
    ad.obs = ad.obs.join(
        scored.set_index("niche"),
        on=ad.obs[niche_key].astype(str),
        how="left",
    )
    ad.obs["microniche"] = ad.obs[niche_key].astype(str).map(niche_short_label)
    ad.obs["local_sgp_frac"] = local_sgp_fraction(ad, perturb, k_neighbors=k_neighbors)
    return ad


def map_pool_to_prep(slice_id: str, pool_barcode: str, prep_names: pd.Index) -> str | None:
    for key in (prep_barcode(slice_id, pool_barcode), pool_barcode):
        if key in prep_names:
            return key
    return None


def prep_barcode(slice_id: str, pool_barcode: str) -> str:
    return f"{slice_id}__{pool_barcode}@{slice_id}"


def ensure_cluster_id(ad: sc.AnnData) -> None:
    if "cluster_id" not in ad.obs.columns:
        ad.obs["cluster_id"] = ad.obs["cell_type"].astype(str).map(CELLTYPE_TO_CLUSTER).astype(int)


def betadata_is_per_cell(betadata_dir: Path) -> bool:
    for p in sorted(betadata_dir.glob("*_betadata.feather")):
        cols = pd.read_feather(p, columns=None).columns
        if "CellID" in cols:
            return True
        if "Cluster" in cols:
            return False
    return False


def betadata_ready(betadata_dir: Path, min_genes: int = 10) -> bool:
    files = list(betadata_dir.glob("*_betadata.feather"))
    return len(files) >= min_genes and betadata_is_per_cell(betadata_dir)


def matched_modulators(betadata_dir: Path, var_names: set[str]) -> list[str]:
    mods: set[str] = set()
    for p in betadata_dir.glob("*_betadata.feather"):
        for c in pd.read_feather(p, columns=None).columns:
            if c.startswith("beta_") and c != "beta0":
                m = c.replace("beta_", "")
                if m in var_names:
                    mods.add(m)
    return sorted(mods)


def _dense(ad: sc.AnnData, genes: list[str]) -> pd.DataFrame:
    keep = [g for g in genes if g in ad.var_names]
    sub = ad[:, keep]
    arr = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
    return pd.DataFrame(arr, index=ad.obs_names, columns=keep)


def build_beta_score_matrix(
    prep: sc.AnnData,
    betadata_dir: Path,
    modulators: list[str] | None = None,
    gene_filter: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Per-cell regulatory scores from betadata (CNN CellID or seed Cluster)."""
    if "imputed_count" in prep.layers:
        prep.X = prep.layers["imputed_count"]
    var = set(prep.var_names)
    if modulators is None:
        modulators = matched_modulators(betadata_dir, var)
    if not modulators:
        raise SystemExit(f"No betadata modulators overlap var_names in {betadata_dir}")
    mod_expr = sc.get.obs_df(prep, keys=modulators).to_numpy(dtype=np.float64)
    per_cell = betadata_is_per_cell(betadata_dir)
    cid = prep.obs["cluster_id"].astype(int).to_numpy() if not per_cell else None

    scores: list[np.ndarray] = []
    genes: list[str] = []
    for p in sorted(betadata_dir.glob("*_betadata.feather")):
        gene = p.stem.replace("_betadata", "")
        if gene_filter and gene not in gene_filter:
            continue
        bd = pd.read_feather(p)
        if per_cell:
            id_col = "CellID" if "CellID" in bd.columns else bd.columns[0]
            bd = bd.set_index(bd[id_col].astype(str))
            b0 = bd.reindex(prep.obs_names.astype(str))["beta0"].to_numpy(dtype=np.float64)
            b0 = np.nan_to_num(b0, nan=0.0)
            acc = b0.copy()
            for c in bd.columns:
                if not c.startswith("beta_") or c == "beta0":
                    continue
                m = c.replace("beta_", "")
                if m not in var:
                    continue
                mi = modulators.index(m)
                beta_v = bd.reindex(prep.obs_names.astype(str))[c].to_numpy(dtype=np.float64)
                beta_v = np.nan_to_num(beta_v, nan=0.0)
                acc += beta_v * mod_expr[:, mi]
        else:
            rows = {int(r.Cluster): r for _, r in bd.iterrows()}
            b0 = np.array([rows[k]["beta0"] for k in range(4)], dtype=np.float64)[cid]
            acc = b0.copy()
            for c in bd.columns:
                if not c.startswith("beta_") or c == "beta0":
                    continue
                m = c.replace("beta_", "")
                if m not in var:
                    continue
                mi = modulators.index(m)
                beta_v = np.array([rows[k][c] for k in range(4)], dtype=np.float64)[cid]
                acc += beta_v * mod_expr[:, mi]
        scores.append(acc)
        genes.append(gene)
    if not scores:
        raise SystemExit(f"No betadata genes matched in {betadata_dir}")
    return np.column_stack(scores), genes


DEFAULT_LEIDEN_KW = {
    "n_pcs": 18,
    "n_neighbors": 12,
    "resolution": 0.75,
    "spatial_weight": 0.55,
    "min_cells": 20,
}

PER_SLICE_LEIDEN_PATH = Path(__file__).resolve().parent.parent / "results/cnn_enrichment/tune/per_slice_leiden.json"
PER_SLICE_LEIDEN_V3_PATH = Path(__file__).resolve().parent.parent / "results/cnn_enrichment/tune/per_slice_leiden_v3.json"

TAG_LEIDEN_PATHS = {
    "cnn_v2": PER_SLICE_LEIDEN_PATH,
    "cnn_v3": PER_SLICE_LEIDEN_V3_PATH,
}


def leiden_config_path_for_tag(tag: str) -> Path:
    return TAG_LEIDEN_PATHS.get(tag, PER_SLICE_LEIDEN_PATH)


def load_per_slice_leiden_config(path: Path | None = None) -> dict[str, dict]:
    p = path or PER_SLICE_LEIDEN_PATH
    if not p.exists():
        return {}
    import json

    data = json.loads(p.read_text())
    return {str(k): dict(v) for k, v in data.get("slices", {}).items()}


def resolve_leiden_kw(
    slice_id: str,
    leiden_kw: dict | None = None,
    per_slice_config: dict[str, dict] | None = None,
) -> dict:
    kw = dict(DEFAULT_LEIDEN_KW)
    cfg = per_slice_config if per_slice_config is not None else load_per_slice_leiden_config()
    if slice_id in cfg:
        kw.update(cfg[slice_id])
    if leiden_kw:
        kw.update(leiden_kw)
    return kw


def leiden_microniches(
    ad: sc.AnnData,
    beta_scores: np.ndarray,
    n_pcs: int = DEFAULT_LEIDEN_KW["n_pcs"],
    n_neighbors: int = DEFAULT_LEIDEN_KW["n_neighbors"],
    resolution: float = DEFAULT_LEIDEN_KW["resolution"],
    spatial_weight: float = DEFAULT_LEIDEN_KW["spatial_weight"],
    min_cells: int = DEFAULT_LEIDEN_KW["min_cells"],
    key: str = "cnn_leiden",
) -> pd.Series:
    if ad.n_obs < min_cells:
        return pd.Series("0", index=ad.obs_names, name=key)

    scores = np.nan_to_num(beta_scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    col_std = scores.std(axis=0)
    keep = col_std > 1e-8
    if keep.sum() < 5:
        return pd.Series("0", index=ad.obs_names, name=key)
    scores = scores[:, keep]

    n_pcs = min(n_pcs, max(1, scores.shape[1] - 1), max(1, ad.n_obs - 1))
    tmp = sc.AnnData(X=scores)
    sc.pp.scale(tmp, max_value=10)
    tmp.X = np.nan_to_num(tmp.X, nan=0.0)
    sc.tl.pca(tmp, n_comps=n_pcs, svd_solver="arpack")
    beta_pca = tmp.obsm["X_pca"]

    xy = ad.obsm["spatial"].astype(np.float64)
    xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-8)
    beta_n = beta_pca / (np.std(beta_pca, axis=0, keepdims=True) + 1e-8)
    sw = spatial_weight
    joint = np.hstack([beta_n * (1 - sw), xy * sw])

    tmp2 = sc.AnnData(X=joint)
    sc.pp.neighbors(tmp2, n_neighbors=min(n_neighbors, ad.n_obs - 1), use_rep="X")
    sc.tl.leiden(tmp2, resolution=resolution, key_added=key, flavor="igraph", n_iterations=2, directed=False)
    return pd.Series(tmp2.obs[key].astype(str).values, index=ad.obs_names, name=key)


def shuffle_microniche_labels(labels: pd.Series, seed: int = 42) -> pd.Series:
    """Permute niche labels among cells (preserves niche sizes; weak null)."""
    rng = np.random.default_rng(seed)
    out = labels.copy().astype(str)
    mask = out.notna() & ~out.isin(["unassigned", "nan", ""])
    if int(mask.sum()) < 2:
        return out
    shuffled = out.loc[mask].values.copy()
    rng.shuffle(shuffled)
    out.loc[mask] = shuffled
    return out


def random_uniform_niche_labels(labels: pd.Series, seed: int = 42) -> pd.Series:
    """Assign cells to K niches i.i.d. uniformly (breaks size structure)."""
    rng = np.random.default_rng(seed)
    out = labels.copy().astype(str)
    mask = out.notna() & ~out.isin(["unassigned", "nan", ""])
    if int(mask.sum()) < 2:
        return out
    uniq = sorted(set(out.loc[mask].astype(str)))
    k = len(uniq)
    if k < 2:
        return out
    out.loc[mask] = [uniq[i] for i in rng.integers(0, k, size=int(mask.sum()))]
    return out


def shuffle_sgp_assignment(
    pool: sc.AnnData,
    perturb: str,
    cell_type: str = "tumor",
    seed: int = 42,
) -> sc.AnnData:
    """Permute sgP vs NTC labels among tumor cells (niches fixed; destroys spatial sgP signal)."""
    ad = pool.copy()
    ct = ad.obs["cell_type"].astype(str) == cell_type
    tg = ad.obs["target_gene"].astype(str)
    mask = ct & tg.isin(["non-targeting", perturb])
    if int(mask.sum()) < 2:
        return ad
    labels = tg.loc[mask].to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    ad.obs.loc[mask, "target_gene"] = labels
    return ad


def permute_predicted_enrichment(merged: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Shuffle predicted niche scores across niches (observed fixed; predictor-only null)."""
    m = merged.copy()
    if len(m) < 3 or "pred_enrichment_score" not in m.columns:
        return m
    rng = np.random.default_rng(seed)
    m["pred_enrichment_score"] = rng.permutation(m["pred_enrichment_score"].to_numpy(dtype=float))
    if "pred_exclusion_x_escape" in m.columns:
        m["pred_exclusion_x_escape"] = rng.permutation(m["pred_exclusion_x_escape"].to_numpy(dtype=float))
    return m


def _perturb_seed(base_seed: int, slice_id: str, perturb: str) -> int:
    return int(base_seed) + sum(ord(c) for c in f"{slice_id}|{perturb}") % 100_003


def _expression_clustering_matrix(ad: sc.AnnData, n_top_genes: int = 1500) -> sc.AnnData:
    if "imputed_count" in ad.layers:
        x = ad.layers["imputed_count"]
    else:
        x = ad.X
    mat = x.toarray() if sparse.issparse(x) else np.asarray(x)
    mat = np.nan_to_num(mat.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.log1p(np.clip(mat, 0.0, None))

    tmp = sc.AnnData(X=mat, obs=ad.obs.copy(), var=ad.var.copy())
    tmp.obsm["spatial"] = ad.obsm["spatial"].copy()
    sc.pp.filter_genes(tmp, min_cells=3)
    gene_var = tmp.X.var(axis=0)
    keep = gene_var > 1e-8
    if int(keep.sum()) < 50:
        keep = np.ones(tmp.n_vars, dtype=bool)
    top_k = min(n_top_genes, int(keep.sum()))
    top_idx = np.argsort(gene_var[keep])[-top_k:]
    gene_idx = np.where(keep)[0][top_idx]
    tmp = tmp[:, gene_idx].copy()
    sc.pp.scale(tmp, max_value=10)
    tmp.X = np.nan_to_num(tmp.X, nan=0.0)
    return tmp


def leiden_expression_clusters(
    ad: sc.AnnData,
    n_pcs: int = 15,
    n_neighbors: int = 12,
    resolution: float = 0.9,
    spatial_weight: float = 0.4,
    min_cells: int = 20,
    n_top_genes: int = 1500,
    key: str = "expr_leiden",
) -> pd.Series:
    """Leiden clusters from gene-expression PCA + spatial (control vs β-microniches)."""
    if ad.n_obs < min_cells:
        return pd.Series("0", index=ad.obs_names, name=key)

    tmp = _expression_clustering_matrix(ad, n_top_genes=n_top_genes)
    n_pcs = min(n_pcs, tmp.n_vars - 1, ad.n_obs - 1)
    sc.tl.pca(tmp, n_comps=n_pcs, svd_solver="arpack")
    expr_pca = tmp.obsm["X_pca"]

    xy = ad.obsm["spatial"].astype(np.float64)
    xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-8)
    expr_n = expr_pca / (np.std(expr_pca, axis=0, keepdims=True) + 1e-8)
    sw = spatial_weight
    joint = np.hstack([expr_n * (1 - sw), xy * sw])

    tmp2 = sc.AnnData(X=joint)
    sc.pp.neighbors(tmp2, n_neighbors=min(n_neighbors, ad.n_obs - 1), use_rep="X")
    sc.tl.leiden(tmp2, resolution=resolution, key_added=key, flavor="igraph", n_iterations=2, directed=False)
    return pd.Series(tmp2.obs[key].astype(str).values, index=ad.obs_names, name=key)


def assign_slice_expression_clusters(
    prep: sc.AnnData,
    slice_id: str,
    cell_type: str,
    prefix: bool = True,
    **leiden_kw,
) -> pd.Series:
    mask = (prep.obs["slice_id"].astype(str) == slice_id) & (prep.obs["cell_type"].astype(str) == cell_type)
    labels = pd.Series(index=prep.obs_names, dtype=str)
    sub = prep[mask].copy()
    if sub.n_obs == 0:
        return labels
    sub_labels = leiden_expression_clusters(sub, **leiden_kw)
    if prefix:
        sub_labels = sub_labels.astype(str).radd(f"{slice_id}|{cell_type}|expr|")
    labels.loc[sub.obs_names] = sub_labels.values
    return labels.fillna("unassigned")


def banksy_clusters(
    ad: sc.AnnData,
    resolution: float = 0.9,
    num_neighbours: int = 15,
    lambda_param: float = 0.2,
    min_cells: int = 20,
    n_top_genes: int = 1500,
    key: str = "banksy",
    tmp_dir: Path | None = None,
) -> pd.Series:
    """BANKSY spatial clusters (expression + neighborhood-augmented features)."""
    if ad.n_obs < min_cells:
        return pd.Series("0", index=ad.obs_names, name=key)

    try:
        from banksy.initialize_banksy import initialize_banksy
        from banksy.run_banksy import run_banksy_multiparam
    except ImportError as exc:
        raise SystemExit(
            "pybanksy is required for BANKSY clustering. Install with: pip install pybanksy"
        ) from exc

    tmp = _expression_clustering_matrix(ad, n_top_genes=n_top_genes)
    xy = tmp.obsm["spatial"]
    tmp.obs["x"] = xy[:, 0]
    tmp.obs["y"] = xy[:, 1]

    with contextlib.redirect_stdout(io.StringIO()):
        banksy_dict = initialize_banksy(
            tmp,
            coord_keys=("x", "y", "spatial"),
            num_neighbours=num_neighbours,
            nbr_weight_decay="scaled_gaussian",
            plt_edge_hist=False,
            plt_nbr_weights=False,
            plt_theta=False,
        )
        out_dir = Path(tmp_dir) if tmp_dir is not None else Path(tempfile.mkdtemp(prefix="banksy_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        color_list = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        ]
        results = run_banksy_multiparam(
            tmp,
            banksy_dict,
            lambda_list=[lambda_param],
            resolutions=[resolution],
            color_list=color_list,
            max_m=1,
            filepath=str(out_dir),
            key=("x", "y", "spatial"),
            annotation_key=None,
            pca_dims=[20],
            savefig=False,
            add_nonspatial=False,
            cluster_algorithm="leiden",
        )

    row = results.iloc[0]
    labels = row["labels"]
    if hasattr(labels, "dense"):
        labels = labels.dense
    return pd.Series(labels.astype(str), index=ad.obs_names, name=key)


def assign_slice_banksy_clusters(
    prep: sc.AnnData,
    slice_id: str,
    cell_type: str,
    prefix: bool = True,
    tmp_dir: Path | None = None,
    **banksy_kw,
) -> pd.Series:
    mask = (prep.obs["slice_id"].astype(str) == slice_id) & (prep.obs["cell_type"].astype(str) == cell_type)
    labels = pd.Series(index=prep.obs_names, dtype=str)
    sub = prep[mask].copy()
    if sub.n_obs == 0:
        return labels
    sub_labels = banksy_clusters(sub, tmp_dir=tmp_dir, **banksy_kw)
    if prefix:
        sub_labels = sub_labels.astype(str).radd(f"{slice_id}|{cell_type}|banksy|")
    labels.loc[sub.obs_names] = sub_labels.values
    return labels.fillna("unassigned")


def assign_slice_microniches(
    prep: sc.AnnData,
    beta_matrix: np.ndarray,
    slice_id: str,
    cell_type: str,
    prefix: bool = True,
    **leiden_kw,
) -> pd.Series:
    mask = (prep.obs["slice_id"].astype(str) == slice_id) & (prep.obs["cell_type"].astype(str) == cell_type)
    labels = pd.Series(index=prep.obs_names, dtype=str)
    sub = prep[mask].copy()
    if sub.n_obs == 0:
        return labels
    sub_scores = beta_matrix[np.where(mask.values)[0]]
    kw = {**DEFAULT_LEIDEN_KW, **leiden_kw}
    sub_labels = leiden_microniches(sub, sub_scores, **kw)
    if prefix:
        sub_labels = sub_labels.astype(str).radd(f"{slice_id}|{cell_type}|")
    labels.loc[sub.obs_names] = sub_labels.values
    return labels.fillna("unassigned")


def knn_assign_perturbed(pool: sc.AnnData, labels: pd.Series, cell_type: str) -> pd.Series:
    out = labels.copy()
    ct_mask = pool.obs["cell_type"].astype(str) == cell_type
    ntc_mask = ct_mask & (pool.obs["target_gene"].astype(str) == "non-targeting")
    pert_mask = ct_mask & (pool.obs["target_gene"].astype(str) != "non-targeting")
    if ntc_mask.sum() == 0 or pert_mask.sum() == 0:
        return out
    labeled_ntc = pool.obs_names[ntc_mask & labels.ne("unassigned") & labels.ne("nan")]
    if len(labeled_ntc) < 5:
        return out
    xy_ntc = pool.obsm["spatial"][pool.obs_names.isin(labeled_ntc)]
    xy_pert = pool.obsm["spatial"][pert_mask]
    nn = NearestNeighbors(n_neighbors=1).fit(xy_ntc)
    _, idx = nn.kneighbors(xy_pert)
    valid = list(labeled_ntc)
    for i, pn in enumerate(pool.obs_names[pert_mask]):
        out[pn] = labels[valid[idx[i, 0]]]
    return out


def niche_expression_index(ad: sc.AnnData, up_genes: list[str], dn_genes: list[str]) -> np.ndarray:
    up = [g for g in up_genes if g in ad.var_names]
    dn = [g for g in dn_genes if g in ad.var_names]
    if not up and not dn:
        return np.zeros(ad.n_obs)
    expr = _dense(ad, up + dn)
    up_m = expr[up].mean(axis=1).to_numpy() if up else np.zeros(ad.n_obs)
    dn_m = expr[dn].mean(axis=1).to_numpy() if dn else np.zeros(ad.n_obs)
    return up_m - dn_m


def observed_log_enrichment(
    pool: sc.AnnData,
    perturb: str,
    cell_type: str,
    niche_key: str,
    pseudocount: float = 0.5,
    min_ntc: int = 2,
    min_pert: int = 2,
) -> pd.DataFrame:
    ct = pool[pool.obs["cell_type"].astype(str) == cell_type].copy()
    ntc = ct[ct.obs["target_gene"].astype(str) == "non-targeting"]
    pert = ct[ct.obs["target_gene"].astype(str) == perturb]
    if ntc.n_obs < min_ntc or pert.n_obs < min_pert:
        return pd.DataFrame()

    niches = ct.obs[niche_key].astype(str)
    valid = sorted(n for n in niches.unique() if n not in ("unassigned", "nan"))
    rows = []
    for n in valid:
        n_ntc = int((ntc.obs[niche_key].astype(str) == n).sum())
        n_pert = int((pert.obs[niche_key].astype(str) == n).sum())
        if n_ntc < min_ntc or n_pert < min_pert:
            continue
        frac_ntc = n_ntc / ntc.n_obs
        frac_pert = n_pert / pert.n_obs
        log2_or = float(np.log2((frac_pert + pseudocount / ntc.n_obs) / (frac_ntc + pseudocount / ntc.n_obs)))
        rows.append({
            "niche": n,
            "n_ntc": n_ntc,
            "n_pert": n_pert,
            "frac_ntc": frac_ntc,
            "frac_pert": frac_pert,
            "obs_log2_enrichment": log2_or,
        })
    return pd.DataFrame(rows)


def global_escape_score(pred: pd.DataFrame, baseline: sc.AnnData, profile: dict) -> float:
    up = [g for g in profile.get("escape_up", []) if g in pred.columns and g in baseline.var_names]
    dn = [g for g in profile.get("escape_dn", []) if g in pred.columns and g in baseline.var_names]
    if not up and not dn:
        return 0.0
    common_idx = pred.index.intersection(baseline.obs_names)
    if len(common_idx) < 10:
        common_idx = pred.index
    base = _dense(baseline, up + dn).reindex(common_idx)
    pr = pred.reindex(common_idx)[up + dn]
    esc = 0.0
    if up:
        esc += float((pr[up].mean() - base[up].mean()).mean())
    if dn:
        esc -= float((pr[dn].mean() - base[dn].mean()).mean())
    return esc


def predicted_niche_scores(
    prep: sc.AnnData,
    pool: sc.AnnData,
    pred: pd.DataFrame,
    perturb: str,
    cell_type: str,
    niche_key: str,
    score_genes: list[str],
    profile: dict,
    cell_cnn_scores: pd.Series | None = None,
    global_baseline: sc.AnnData | None = None,
    min_ntc_per_niche: int = 2,
    min_pert_per_niche: int = 2,
) -> pd.DataFrame:
    ct_mask = pool.obs["cell_type"].astype(str) == cell_type
    ntc_pool = pool[ct_mask & (pool.obs["target_gene"].astype(str) == "non-targeting")].copy()
    if ntc_pool.n_obs == 0 or niche_key not in ntc_pool.obs.columns:
        return pd.DataFrame()

    slice_id = str(pool.obs["slice_id"].iloc[0]) if "slice_id" in pool.obs.columns else ""
    prep_names = prep.obs_names
    pred_ids = pd.Index([
        map_pool_to_prep(slice_id, b, prep_names) or prep_barcode(slice_id, b)
        for b in ntc_pool.obs_names
    ])
    common_genes = [g for g in score_genes if g in pred.columns and g in prep.var_names]
    aligned = pred_ids.intersection(pred.index)
    use_global = len(aligned) < max(10, 0.2 * len(pred_ids))
    ref_base = global_baseline if global_baseline is not None else prep

    if use_global:
        pred_escape_scalar = global_escape_score(pred, ref_base, profile)
        pred_escape = np.full(len(ntc_pool), pred_escape_scalar)
    else:
        if len(common_genes) < 3:
            return pd.DataFrame()
        base_expr = _dense(prep, common_genes).reindex(pred_ids)
        pred_expr = pred.reindex(pred_ids)[common_genes]
        up = [g for g in profile.get("escape_up", []) if g in common_genes]
        dn = [g for g in profile.get("escape_dn", []) if g in common_genes]
        pred_escape = np.zeros(len(ntc_pool))
        if up:
            pred_escape += pred_expr[up].mean(axis=1).to_numpy() - base_expr[up].mean(axis=1).to_numpy()
        if dn:
            pred_escape -= pred_expr[dn].mean(axis=1).to_numpy() - base_expr[dn].mean(axis=1).to_numpy()

    exclusion = niche_expression_index(ntc_pool, IMMUNE_EXCLUSION_UP, IMMUNE_EXCLUSION_DN)
    if cell_cnn_scores is not None:
        by_prep = cell_cnn_scores.reindex(pred_ids)
        by_pool = cell_cnn_scores.reindex(ntc_pool.obs_names)
        if by_prep.notna().sum() >= by_pool.notna().sum():
            cnn_target = by_prep.fillna(0.0).to_numpy()
        else:
            cnn_target = by_pool.fillna(0.0).to_numpy()
    else:
        cnn_target = np.zeros(len(ntc_pool))

    ntc_pool.obs["_pred_escape"] = pred_escape
    ntc_pool.obs["_exclusion"] = exclusion
    ntc_pool.obs["_cnn_target"] = cnn_target

    rows = []
    for n, sub in ntc_pool.obs.groupby(niche_key, observed=True):
        if str(n) in ("unassigned", "nan") or len(sub) < min_ntc_per_niche:
            continue
        rows.append({
            "niche": str(n),
            "pred_exclusion_index": float(sub["_exclusion"].mean()),
            "pred_escape_gain": float(sub["_pred_escape"].mean()),
            "pred_cnn_target_score": float(sub["_cnn_target"].mean()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sign = float(profile.get("exclusion_sign", 0.0))
    for col in ["pred_exclusion_index", "pred_escape_gain", "pred_cnn_target_score"]:
        df[f"z_{col}"] = _zscore(df[col].to_numpy())
    cnn_w = float(profile.get("cnn_weight", 0.5 if use_global else 0.35))
    if use_global:
        esc = float(pred_escape_scalar)
        df["pred_enrichment_score"] = (
            sign * df["z_pred_exclusion_index"] * (1.0 + np.sign(esc) * min(abs(esc), 3.0))
            + cnn_w * df["z_pred_cnn_target_score"]
        )
        df["pred_exclusion_x_escape"] = sign * df["z_pred_exclusion_index"] * np.sign(esc)
    else:
        df["pred_enrichment_score"] = (
            sign * df["z_pred_exclusion_index"]
            + df["z_pred_escape_gain"]
            + cnn_w * df["z_pred_cnn_target_score"]
        )
        df["pred_exclusion_x_escape"] = sign * df["z_pred_exclusion_index"] * df["z_pred_escape_gain"]
    return df


def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 2:
        return np.zeros_like(v)
    s = v.std()
    if s < 1e-8:
        return np.zeros_like(v)
    return (v - v.mean()) / s


def merge_obs_pred_enrichment(obs_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    if obs_df.empty or pred_df.empty:
        return pd.DataFrame()
    m = obs_df.merge(pred_df, on="niche", how="inner")
    return m


def enrichment_correlation(merged: pd.DataFrame) -> dict:
    if merged.empty or len(merged) < 3:
        return {"n_niches": len(merged), "pearson_r": float("nan"), "spearman_r": float("nan"), "p_pearson": float("nan")}
    obs = merged["obs_log2_enrichment"].to_numpy()
    pred = merged["pred_enrichment_score"].to_numpy()
    ix = merged["pred_exclusion_x_escape"].to_numpy()
    r_p, p_p = stats.pearsonr(obs, pred) if obs.std() > 1e-8 and pred.std() > 1e-8 else (float("nan"), float("nan"))
    r_s, p_s = stats.spearmanr(obs, pred) if len(obs) >= 3 else (float("nan"), float("nan"))
    r_ix, p_ix = stats.pearsonr(obs, ix) if obs.std() > 1e-8 and ix.std() > 1e-8 else (float("nan"), float("nan"))
    return {
        "n_niches": int(len(merged)),
        "pearson_r": float(r_p),
        "p_pearson": float(p_p),
        "spearman_r": float(r_s),
        "p_spearman": float(p_s),
        "pearson_exclusion_x_escape": float(r_ix),
        "p_exclusion_x_escape": float(p_ix),
    }
