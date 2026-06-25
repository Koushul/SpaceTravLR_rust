"""CNN per-cell beta microniches and guide-enrichment prediction utilities."""

from __future__ import annotations

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


def niche_short_label(niche: str) -> str:
    return str(niche).split("|")[-1]


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
    "n_pcs": 12,
    "n_neighbors": 12,
    "resolution": 0.9,
    "spatial_weight": 0.4,
    "min_cells": 20,
}


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

    n_pcs = min(n_pcs, scores.shape[1], ad.n_obs - 1)
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
    """Permute niche labels among cells (negative control; destroys spatial coherence)."""
    rng = np.random.default_rng(seed)
    out = labels.copy().astype(str)
    mask = out.notna() & ~out.isin(["unassigned", "nan", ""])
    if int(mask.sum()) < 2:
        return out
    shuffled = out.loc[mask].values.copy()
    rng.shuffle(shuffled)
    out.loc[mask] = shuffled
    return out


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

    if "imputed_count" in ad.layers:
        x = ad.layers["imputed_count"]
    else:
        x = ad.X
    mat = x.toarray() if sparse.issparse(x) else np.asarray(x)
    mat = np.nan_to_num(mat.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.log1p(np.clip(mat, 0.0, None))

    gene_var = mat.var(axis=0)
    keep = gene_var > 1e-8
    if int(keep.sum()) < 50:
        keep = np.ones(mat.shape[1], dtype=bool)
    top_k = min(n_top_genes, int(keep.sum()))
    top_idx = np.argsort(gene_var[keep])[-top_k:]
    gene_idx = np.where(keep)[0][top_idx]
    mat = mat[:, gene_idx]

    tmp = sc.AnnData(X=mat)
    sc.pp.scale(tmp, max_value=10)
    tmp.X = np.nan_to_num(tmp.X, nan=0.0)
    n_pcs = min(n_pcs, mat.shape[1] - 1, ad.n_obs - 1)
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
    sub_labels = leiden_microniches(sub, sub_scores, **leiden_kw)
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
