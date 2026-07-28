#!/usr/bin/env python3
"""Moran's I screen + group-Shapley attribution of TF / LR / BR to CV R².

Default: top 100 genes by Moran's I among genes detected in ≥2% of subsample cells.
"""

from __future__ import annotations

import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome")
HOST = BASE / "processed/GSM9456850_tumor_cells_spacetravlr_ready.h5ad"
FEAT = BASE / "lasso_pilot/br_features_subsample.npz"
OUT = BASE / "lasso_pilot"
OUT.mkdir(parents=True, exist_ok=True)

MORAN_K = 30
MIN_POS_FRAC = 0.02
MAX_GENES = 100
N_JOBS = 8
RESUME = True

TF_CANDIDATES = [
    "Spi1", "Irf8", "Irf7", "Irf3", "Stat1", "Stat3", "Nfkb1", "Rela", "Relb",
    "Hif1a", "Atf3", "Jun", "Fos", "Egr1", "Klf4", "Klf5", "Cdx2", "Hnf4a",
    "Spdef", "Atoh1", "Gata4", "Gata6", "Pparg", "Nr1h3", "Myc", "Trp53",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def gene_map(adata: ad.AnnData) -> dict[str, str]:
    return {g.lower(): g for g in adata.var_names.astype(str)}


def resolve(gmap: dict[str, str], names: list[str]) -> list[str]:
    out, seen = [], set()
    for n in names:
        key = n.lower()
        if key in gmap and gmap[key] not in seen:
            out.append(gmap[key])
            seen.add(gmap[key])
    return out


def densify_cols(X, idx: list[int]) -> np.ndarray:
    if not idx:
        return np.zeros((X.shape[0], 0), dtype=np.float64)
    sub = X[:, idx]
    if sparse.issparse(sub):
        return np.asarray(sub.todense(), dtype=np.float64)
    return np.asarray(sub, dtype=np.float64)


def spatial_knn_impute(expr: np.ndarray, xy: np.ndarray, k: int = 15) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k + 1, expr.shape[0]), algorithm="kd_tree").fit(xy)
    ind = nn.kneighbors(xy, return_distance=False)[:, 1:]
    out = expr.copy()
    for j in range(expr.shape[1]):
        col = expr[:, j]
        zero = col <= 0
        if not zero.any() or zero.all():
            continue
        neigh = col[ind]
        pos = neigh > 0
        counts = pos.sum(1)
        ok = counts > 0
        fill = np.zeros_like(col)
        fill[ok] = (neigh * pos).sum(1)[ok] / counts[ok]
        out[zero, j] = fill[zero]
    return out


def knn_row_normalize(xy: np.ndarray, k: int) -> sparse.csr_matrix:
    nn = NearestNeighbors(n_neighbors=min(k + 1, xy.shape[0]), algorithm="kd_tree").fit(xy)
    ind = nn.kneighbors(xy, return_distance=False)[:, 1:]
    n = xy.shape[0]
    rows = np.repeat(np.arange(n), ind.shape[1])
    cols = ind.ravel()
    data = np.ones(cols.shape[0], dtype=np.float64)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    rs = np.asarray(W.sum(1)).ravel()
    rs[rs == 0] = 1.0
    return sparse.diags(1.0 / rs) @ W


def morans_i(x: np.ndarray, W: sparse.csr_matrix) -> float:
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    den = float(np.dot(xc, xc))
    if den <= 1e-18:
        return 0.0
    Wx = W @ xc
    n = x.shape[0]
    w_sum = float(W.sum())
    return float((n / w_sum) * (np.dot(xc, Wx) / den))


def fit_lasso_r2(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> float:
    if X.shape[1] == 0 or float(y.std()) < 1e-12:
        return 0.0
    Xs = StandardScaler().fit_transform(X)
    keep = Xs.std(0) > 1e-8
    Xs = Xs[:, keep]
    if Xs.shape[1] == 0:
        return 0.0
    model = LassoCV(
        cv=KFold(3, shuffle=True, random_state=random_state),
        alphas=10,
        max_iter=2000,
        n_jobs=N_JOBS,
        random_state=random_state,
        tol=1e-3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xs, y)
    mse = float(model.mse_path_[model.alphas_ == model.alpha_].mean())
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - (mse * len(y)) / sst) if sst > 0 else 0.0


def coalition_key(groups: tuple[str, ...]) -> str:
    if not groups:
        return "empty"
    return "_".join(groups)


def group_shapley(v: dict[str, float], players: list[str]) -> dict[str, float]:
    n = len(players)
    phi = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                s_key = coalition_key(tuple(sorted(S)))
                sp_key = coalition_key(tuple(sorted(S + (p,))))
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                phi[p] += weight * (v.get(sp_key, 0.0) - v.get(s_key, 0.0))
    return phi


def save_checkpoint(shap_rows, coal_rows, details) -> None:
    shap_df = pd.DataFrame(shap_rows).sort_values("moran_I", ascending=False)
    coal_df = pd.DataFrame(coal_rows)
    shap_df.to_csv(OUT / "shapley_group_summary.csv", index=False)
    coal_df.to_csv(OUT / "shapley_coalitions.csv", index=False)
    (OUT / "shapley_results.json").write_text(json.dumps(details, indent=2))


def main():
    log("[load] features + host…")
    feat = np.load(FEAT, allow_pickle=True)
    sel = feat["sel"]
    xy = feat["xy"]
    LR = feat["LR"]
    BR = feat["BR"]
    lr_names = [str(x) for x in feat["lr_names"]]
    br_names = [str(x) for x in feat["br_names"]]

    adata = ad.read_h5ad(HOST)
    gmap = gene_map(adata)
    tfs = resolve(gmap, TF_CANDIDATES)
    Xraw = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    Xsub = Xraw[sel]

    frac = np.asarray((Xsub > 0).mean(0)).ravel()
    cand_idx = np.where(frac >= MIN_POS_FRAC)[0]
    candidates = [str(adata.var_names[i]) for i in cand_idx]
    log(f"[moran] screening {len(candidates)} genes with pos_frac≥{MIN_POS_FRAC} (k={MORAN_K})…")

    W = knn_row_normalize(xy, MORAN_K)
    # impute in chunks to limit peak memory
    moran_rows = []
    chunk = 50
    for start in range(0, len(cand_idx), chunk):
        idx = cand_idx[start : start + chunk].tolist()
        names = candidates[start : start + chunk]
        expr = spatial_knn_impute(densify_cols(Xsub, idx), xy, k=15)
        for j, g in enumerate(names):
            x = expr[:, j]
            pos = float((x > 0).mean())
            if pos < MIN_POS_FRAC:
                continue
            moran_rows.append(
                {
                    "gene": g,
                    "moran_I": morans_i(x, W),
                    "pos_frac": pos,
                    "mean": float(x.mean()),
                    "var": float(x.var()),
                }
            )
        log(f"[moran] {min(start + chunk, len(cand_idx))}/{len(cand_idx)}")

    moran_df = pd.DataFrame(moran_rows).sort_values("moran_I", ascending=False)
    moran_df.to_csv(OUT / "moran_gene_screen.csv", index=False)
    log("[moran] top 20:")
    log(moran_df.head(20).to_string(index=False))

    selected = moran_df.head(MAX_GENES).copy()
    genes = selected["gene"].tolist()
    log(f"[select] top {len(genes)} by Moran I (range {selected['moran_I'].min():.3f}–{selected['moran_I'].max():.3f})")

    log("[impute] TF + selected targets…")
    tf_idx = [adata.var_names.get_loc(g) for g in tfs]
    tgt_idx = [adata.var_names.get_loc(g) for g in genes]
    tf_expr = spatial_knn_impute(densify_cols(Xsub, tf_idx), xy, k=15)
    tgt_expr = spatial_knn_impute(densify_cols(Xsub, tgt_idx), xy, k=15)
    mi_map = dict(zip(selected["gene"], selected["moran_I"]))

    players = ["br", "lr", "tf"]
    shap_rows: list[dict] = []
    coal_rows: list[dict] = []
    details = {
        "selection": "top_n_by_moran_I",
        "max_genes": MAX_GENES,
        "moran_k": MORAN_K,
        "min_pos_frac": MIN_POS_FRAC,
        "n_cells": int(len(sel)),
        "genes": {},
    }

    done = set()
    if RESUME and (OUT / "shapley_group_summary.csv").exists():
        prev = pd.read_csv(OUT / "shapley_group_summary.csv")
        if (OUT / "shapley_coalitions.csv").exists():
            prev_coal = pd.read_csv(OUT / "shapley_coalitions.csv")
        else:
            prev_coal = pd.DataFrame(columns=["gene", "coalition", "r2_cv"])
        prev_genes = set(prev["gene"]) & set(genes)
        if prev_genes:
            shap_rows = prev[prev["gene"].isin(prev_genes)].to_dict("records")
            coal_rows = prev_coal[prev_coal["gene"].isin(prev_genes)].to_dict("records")
            done = set(prev_genes)
            if (OUT / "shapley_results.json").exists():
                details = json.loads((OUT / "shapley_results.json").read_text())
                details.setdefault("genes", {})
            log(f"[resume] {len(done)} / {len(genes)} genes already done")

    for ti, gene in enumerate(genes):
        if gene in done:
            continue
        y = tgt_expr[:, ti]
        if float((y > 0).mean()) < MIN_POS_FRAC or float(y.std()) < 1e-12:
            log(f"[skip] {gene}: too sparse after impute")
            continue
        tf_mask = [g != gene for g in tfs]
        tf_X = tf_expr[:, tf_mask]
        lr_mask = [
            not (n.endswith(f"${gene}") or n.startswith(f"{gene}$")) for n in lr_names
        ]
        br_mask = [not n.endswith(f"${gene}") for n in br_names]
        lr_X = LR[:, lr_mask] if LR.shape[1] else LR
        br_X = BR[:, br_mask] if BR.shape[1] else BR

        group_X = {"tf": tf_X, "lr": lr_X, "br": br_X}
        v = {"empty": 0.0}
        for r in range(1, 4):
            for S in combinations(players, r):
                key = coalition_key(S)
                mats = [group_X[p] for p in S if group_X[p].shape[1] > 0]
                if not mats:
                    v[key] = 0.0
                else:
                    X = np.hstack(mats)
                    v[key] = fit_lasso_r2(X, y, random_state=0)
                coal_rows.append({"gene": gene, "coalition": key, "r2_cv": v[key]})

        phi = group_shapley(v, players)
        full = v.get("br_lr_tf", 0.0)
        mi = float(mi_map[gene])
        row = {
            "gene": gene,
            "moran_I": mi,
            "r2_full": full,
            "shap_tf": phi["tf"],
            "shap_lr": phi["lr"],
            "shap_br": phi["br"],
            "shap_sum": phi["tf"] + phi["lr"] + phi["br"],
            "frac_tf": phi["tf"] / full if full > 1e-8 else 0.0,
            "frac_lr": phi["lr"] / full if full > 1e-8 else 0.0,
            "frac_br": phi["br"] / full if full > 1e-8 else 0.0,
        }
        shap_rows.append(row)
        details["genes"][gene] = {"moran_I": mi, "coalitions": v, "shapley": phi}
        n_done = len(shap_rows)
        log(
            f"[shap] {n_done:3d}/{len(genes)} {gene:12s} I={mi:.3f} R2={full:.3f}  "
            f"φTF={phi['tf']:+.4f} φLR={phi['lr']:+.4f} φBR={phi['br']:+.4f}"
        )
        save_checkpoint(shap_rows, coal_rows, details)

    shap_df = pd.DataFrame(shap_rows).sort_values("moran_I", ascending=False)
    save_checkpoint(shap_rows, coal_rows, details)

    if len(shap_df):
        mean_phi = shap_df[["shap_tf", "shap_lr", "shap_br"]].mean()
        mean_frac = shap_df[["frac_tf", "frac_lr", "frac_br"]].mean()
        med_frac = shap_df[["frac_tf", "frac_lr", "frac_br"]].median()
        log("\n=== GROUP SHAPLEY top-100 Moran I ===")
        log(f"n={len(shap_df)}  Moran I {shap_df['moran_I'].min():.3f}–{shap_df['moran_I'].max():.3f}")
        log("Mean φ: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_phi.items()))
        log("Mean frac: " + ", ".join(f"{k}={v:.3f}" for k, v in mean_frac.items()))
        log("Median frac: " + ", ".join(f"{k}={v:.3f}" for k, v in med_frac.items()))
        log("\nTop 15 by Moran I:")
        log(shap_df.head(15).to_string(index=False))
        log("\nTop 10 by φ_BR:")
        log(
            shap_df.sort_values("shap_br", ascending=False)
            .head(10)[["gene", "moran_I", "r2_full", "shap_tf", "shap_lr", "shap_br", "frac_br"]]
            .to_string(index=False)
        )

    (OUT / "SHAPLEY_README.md").write_text(
        f"""# Group Shapley (TF / LR / BR) — top {MAX_GENES} Moran I

Value function: 5-fold LassoCV R² on coalitions of feature groups.

Players: **TF**, **LR** (CellChat host–host), **BR** (microbial secretion×receptor).

Selection: all genes with detection ≥ {MIN_POS_FRAC} on the 8k-cell subsample, ranked by Moran's I (knn k={MORAN_K}), top {MAX_GENES}.

Outputs: `moran_gene_screen.csv`, `shapley_group_summary.csv`, `shapley_coalitions.csv`, `shapley_results.json`.
"""
    )
    log(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
