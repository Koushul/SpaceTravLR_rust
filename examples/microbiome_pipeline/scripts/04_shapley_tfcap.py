#!/usr/bin/env python3
"""Group-Shapley TF/LR/BR with LR count capped to n_TF per gene.

Selection: top MAX_GENES by Moran's I. For each target, keep at most
len(TF features) LR columns (top by CellChat pair score), matching TF budget.
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
LR_META = BASE / "lasso_pilot/host_lr_pairs_used.csv"
MORAN_CSV = BASE / "lasso_pilot/moran_gene_screen.csv"
OUT = BASE / "lasso_pilot"
OUT.mkdir(parents=True, exist_ok=True)

TAG = "tfcap50"
MORAN_K = 30
MIN_POS_FRAC = 0.02
MAX_GENES = 50
N_JOBS = 8

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


def cap_lr_to_tf(
    lr_X: np.ndarray,
    lr_names: list[str],
    n_tf: int,
    score_rank: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Keep at most n_tf LR features, preferring higher CellChat pair scores."""
    if lr_X.shape[1] == 0 or n_tf <= 0:
        return lr_X[:, :0], []
    order = sorted(
        range(len(lr_names)),
        key=lambda i: score_rank.get(lr_names[i], -1.0),
        reverse=True,
    )[:n_tf]
    order = sorted(order)  # stable column order
    return lr_X[:, order], [lr_names[i] for i in order]


def main():
    log("[load] features + host…")
    feat = np.load(FEAT, allow_pickle=True)
    sel = feat["sel"]
    xy = feat["xy"]
    LR = feat["LR"]
    BR = feat["BR"]
    lr_names_all = [str(x) for x in feat["lr_names"]]
    br_names = [str(x) for x in feat["br_names"]]

    lr_meta = pd.read_csv(LR_META)
    score_rank = dict(zip(lr_meta["pair"].astype(str), lr_meta["score"].astype(float)))
    # pairs missing from meta get score 0
    for n in lr_names_all:
        score_rank.setdefault(n, 0.0)

    adata = ad.read_h5ad(HOST)
    gmap = gene_map(adata)
    tfs = resolve(gmap, TF_CANDIDATES)
    Xraw = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    Xsub = Xraw[sel]

    moran_df = pd.read_csv(MORAN_CSV).sort_values("moran_I", ascending=False)
    selected = moran_df.head(MAX_GENES).copy()
    genes = selected["gene"].tolist()
    log(
        f"[select] top {len(genes)} Moran I "
        f"({selected['moran_I'].min():.3f}–{selected['moran_I'].max():.3f})"
    )
    log(f"[cap] LR ≤ n_TF per gene (n_TF pool={len(tfs)}; full LR pool={len(lr_names_all)})")

    log("[impute] TF + targets…")
    tf_idx = [adata.var_names.get_loc(g) for g in tfs]
    tgt_idx = [adata.var_names.get_loc(g) for g in genes]
    tf_expr = spatial_knn_impute(densify_cols(Xsub, tf_idx), xy, k=15)
    tgt_expr = spatial_knn_impute(densify_cols(Xsub, tgt_idx), xy, k=15)
    mi_map = dict(zip(selected["gene"], selected["moran_I"]))

    players = ["br", "lr", "tf"]
    shap_rows = []
    coal_rows = []
    details = {
        "tag": TAG,
        "selection": "top_n_by_moran_I",
        "max_genes": MAX_GENES,
        "lr_cap": "n_tf_per_gene",
        "lr_cap_rule": "top CellChat pair score among non-self LR",
        "n_tf_pool": len(tfs),
        "n_lr_pool": len(lr_names_all),
        "n_br_pool": len(br_names),
        "moran_k": MORAN_K,
        "n_cells": int(len(sel)),
        "genes": {},
    }

    summary_path = OUT / f"shapley_{TAG}_summary.csv"
    coal_path = OUT / f"shapley_{TAG}_coalitions.csv"
    json_path = OUT / f"shapley_{TAG}_results.json"

    for ti, gene in enumerate(genes):
        y = tgt_expr[:, ti]
        if float((y > 0).mean()) < MIN_POS_FRAC or float(y.std()) < 1e-12:
            log(f"[skip] {gene}: too sparse")
            continue

        tf_mask = [g != gene for g in tfs]
        tf_X = tf_expr[:, tf_mask]
        n_tf = int(tf_X.shape[1])

        lr_mask = [
            not (n.endswith(f"${gene}") or n.startswith(f"{gene}$")) for n in lr_names_all
        ]
        br_mask = [not n.endswith(f"${gene}") for n in br_names]
        lr_X_full = LR[:, lr_mask] if LR.shape[1] else LR
        lr_names_g = [n for n, k in zip(lr_names_all, lr_mask) if k]
        br_X = BR[:, br_mask] if BR.shape[1] else BR
        br_names_g = [n for n, k in zip(br_names, br_mask) if k]

        lr_X, lr_names_g = cap_lr_to_tf(lr_X_full, lr_names_g, n_tf, score_rank)

        group_X = {"tf": tf_X, "lr": lr_X, "br": br_X}
        v = {"empty": 0.0}
        for r in range(1, 4):
            for S in combinations(players, r):
                key = coalition_key(S)
                mats = [group_X[p] for p in S if group_X[p].shape[1] > 0]
                v[key] = 0.0 if not mats else fit_lasso_r2(np.hstack(mats), y, random_state=0)
                coal_rows.append({"gene": gene, "coalition": key, "r2_cv": v[key]})

        phi = group_shapley(v, players)
        full = v.get("br_lr_tf", 0.0)
        mi = float(mi_map[gene])
        row = {
            "gene": gene,
            "moran_I": mi,
            "n_tf": n_tf,
            "n_lr": len(lr_names_g),
            "n_br": len(br_names_g),
            "r2_full": full,
            "shap_tf": phi["tf"],
            "shap_lr": phi["lr"],
            "shap_br": phi["br"],
            "shap_sum": phi["tf"] + phi["lr"] + phi["br"],
            "frac_tf": phi["tf"] / full if full > 1e-8 else 0.0,
            "frac_lr": phi["lr"] / full if full > 1e-8 else 0.0,
            "frac_br": phi["br"] / full if full > 1e-8 else 0.0,
            "lr_kept": ",".join(lr_names_g),
        }
        shap_rows.append(row)
        details["genes"][gene] = {
            "moran_I": mi,
            "n_tf": n_tf,
            "n_lr": len(lr_names_g),
            "n_br": len(br_names_g),
            "lr_kept": lr_names_g,
            "coalitions": v,
            "shapley": phi,
        }
        log(
            f"[shap] {len(shap_rows):2d}/{len(genes)} {gene:12s} "
            f"nTF={n_tf} nLR={len(lr_names_g)} nBR={len(br_names_g)}  "
            f"I={mi:.3f} R2={full:.3f}  "
            f"φTF={phi['tf']:+.4f} φLR={phi['lr']:+.4f} φBR={phi['br']:+.4f}"
        )
        pd.DataFrame(shap_rows).sort_values("moran_I", ascending=False).to_csv(
            summary_path, index=False
        )
        pd.DataFrame(coal_rows).to_csv(coal_path, index=False)
        json_path.write_text(json.dumps(details, indent=2))

    shap_df = pd.DataFrame(shap_rows).sort_values("moran_I", ascending=False)
    shap_df.to_csv(summary_path, index=False)
    pd.DataFrame(coal_rows).to_csv(coal_path, index=False)
    json_path.write_text(json.dumps(details, indent=2))

    if len(shap_df):
        mean_phi = shap_df[["shap_tf", "shap_lr", "shap_br"]].mean()
        mean_frac = shap_df[["frac_tf", "frac_lr", "frac_br"]].mean()
        med_frac = shap_df[["frac_tf", "frac_lr", "frac_br"]].median()
        log(f"\n=== GROUP SHAPLEY {TAG}: top-{MAX_GENES}, LR capped to n_TF ===")
        log(f"n={len(shap_df)}  Moran {shap_df['moran_I'].min():.3f}–{shap_df['moran_I'].max():.3f}")
        log(f"n_TF={shap_df['n_tf'].iloc[0]}  n_LR_capped={shap_df['n_lr'].median():.0f}  n_BR={shap_df['n_br'].iloc[0]}")
        log("Mean φ: " + ", ".join(f"{k}={v:.4f}" for k, v in mean_phi.items()))
        log("Mean frac: " + ", ".join(f"{k}={v:.3f}" for k, v in mean_frac.items()))
        log("Median frac: " + ", ".join(f"{k}={v:.3f}" for k, v in med_frac.items()))
        log("\nTop 10 by Moran I:")
        log(
            shap_df.head(10)[
                ["gene", "moran_I", "n_tf", "n_lr", "r2_full", "shap_tf", "shap_lr", "shap_br", "frac_tf", "frac_lr", "frac_br"]
            ].to_string(index=False)
        )
        log("\nTop 10 by φ_BR:")
        log(
            shap_df.sort_values("shap_br", ascending=False)
            .head(10)[["gene", "moran_I", "r2_full", "shap_tf", "shap_lr", "shap_br", "frac_br"]]
            .to_string(index=False)
        )

    (OUT / f"SHAPLEY_{TAG}_README.md").write_text(
        f"""# Group Shapley — top {MAX_GENES}, LR capped to n_TF

Per gene: `#LR ≤ #TF` (TF self-dropped). LR kept = top by CellChat pair score among non-self pairs.

| Group | role |
|-------|------|
| TF | cell-intrinsic TFs (~{len(tfs)}) |
| LR | host–host received(L)×R, capped to n_TF |
| BR | microbial received(S)×R (uncapped, 13) |

Outputs: `shapley_{TAG}_summary.csv`, `shapley_{TAG}_coalitions.csv`, `shapley_{TAG}_results.json`.
"""
    )
    log(f"[wrote] {OUT}/shapley_{TAG}_*")


if __name__ == "__main__":
    main()
