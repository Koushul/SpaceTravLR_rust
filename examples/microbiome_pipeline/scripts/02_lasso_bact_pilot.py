#!/usr/bin/env python3
"""Lasso-only pilot: host–host LR + bacterial secretion (BR) modulators.

Ablations: TF / LR / BR and combinations via LassoCV on microbe-proximal tumor cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome")
RUST = Path("/ix1/ylee/kor11/tools/SpaceTravLR_rust")
HOST = BASE / "processed/GSM9456850_tumor_cells_spacetravlr_ready.h5ad"
UNMAP = BASE / "raw/stereoseq_tumor/GSM9456850_A612_unmap.h5ad"
INTER = RUST / "data/microbial/bact_host_interactions.v0.csv"
PRIORS = RUST / "data/microbial/taxon_signal_priors.v0.csv"
CELLCHAT = Path("/ix1/ylee/kor11/tools/SpaceTravLR/data/cellchat_mouse.csv")
OUT = BASE / "lasso_pilot"
OUT.mkdir(parents=True, exist_ok=True)

PIX_PER_UM = 2.0
N_CELL_SUBSAMPLE = 8000
GRID_UM = 25.0
# Name avoids CodeQL clear-text-"secret" false positive on *SECRETION*.
PARACRINE_RADIUS_UM = 300.0
CONTACT_RADIUS_UM = 30.0
MAX_LR_PAIRS = 120
MIN_GENE_FRAC = 0.005
TARGETS = [
    "Lyz1",
    "Muc2",
    "Defa24",
    "Reg3g",
    "Nos2",
    "Cxcl1",
    "Duox2",
    "Nfkb1",
    "Rela",
    "Cd74",
]
TF_CANDIDATES = [
    "Spi1",
    "Irf8",
    "Irf7",
    "Irf3",
    "Stat1",
    "Stat3",
    "Nfkb1",
    "Rela",
    "Relb",
    "Hif1a",
    "Atf3",
    "Jun",
    "Fos",
    "Egr1",
    "Klf4",
    "Klf5",
    "Cdx2",
    "Hnf4a",
    "Spdef",
    "Atoh1",
    "Gata4",
    "Gata6",
    "Pparg",
    "Nr1h3",
    "Myc",
    "Trp53",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def gene_map(adata: ad.AnnData) -> dict[str, str]:
    return {g.lower(): g for g in adata.var_names.astype(str)}


def resolve(gmap: dict[str, str], names: list[str]) -> list[str]:
    return [gmap[n.lower()] for n in names if n.lower() in gmap]


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


def expand_cellchat(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        ligands = [p for p in str(row["ligand"]).split("_") if p]
        receptors = [p for p in str(row["receptor"]).split("_") if p]
        for lig in ligands:
            for rec in receptors:
                rows.append(
                    {
                        "ligand": lig,
                        "receptor": rec,
                        "pathway": row["pathway"],
                        "signaling": row["signaling"],
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["ligand", "receptor", "signaling"])


def select_host_lr_pairs(
    adata: ad.AnnData,
    gmap: dict[str, str],
    max_pairs: int = MAX_LR_PAIRS,
    min_frac: float = MIN_GENE_FRAC,
) -> pd.DataFrame:
    raw = pd.read_csv(CELLCHAT)
    lr = expand_cellchat(raw)
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    frac = np.asarray((X > 0).sum(0)).ravel() / adata.n_obs
    mean = np.asarray(X.mean(0)).ravel()
    frac_map = {g.lower(): float(frac[i]) for i, g in enumerate(adata.var_names)}
    mean_map = {g.lower(): float(mean[i]) for i, g in enumerate(adata.var_names)}

    def ok(sym: str) -> bool:
        key = sym.lower()
        return key in gmap and (frac_map.get(key, 0.0) >= min_frac or mean_map.get(key, 0.0) >= 0.02)

    keep = []
    for _, row in lr.iterrows():
        if not (ok(row["ligand"]) and ok(row["receptor"])):
            continue
        lig = gmap[row["ligand"].lower()]
        rec = gmap[row["receptor"].lower()]
        score = float(np.sqrt(max(mean_map[lig.lower()], 0.0) * max(mean_map[rec.lower()], 0.0)))
        radius = (
            PARACRINE_RADIUS_UM
            if row["signaling"] == "Secreted Signaling"
            else CONTACT_RADIUS_UM
        )
        keep.append(
            {
                "ligand": lig,
                "receptor": rec,
                "pathway": row["pathway"],
                "signaling": row["signaling"],
                "radius_um": radius,
                "score": score,
                "pair": f"{lig}${rec}",
            }
        )
    out = pd.DataFrame(keep)
    if out.empty:
        return out
    out = out.sort_values("score", ascending=False).drop_duplicates("pair").head(max_pairs)
    return out.reset_index(drop=True)


def build_senders_colony(
    unmap_path: Path,
    priors: pd.DataFrame,
    interactions: pd.DataFrame,
    grid_um: float = 25.0,
    min_umi: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    log("[senders] loading unmap…")
    m = ad.read_h5ad(unmap_path)
    mv = m.var
    keep = (
        (mv["superkingdom"] == "Bacteria")
        & mv["genus"].notna()
        & (~mv["genus"].isin(["Mus", "Homo"]))
    )
    m = m[:, keep].copy()
    genera = m.var["genus"].astype(str)
    uniq = pd.Index(sorted(genera.unique()))
    gmap = {g: i for i, g in enumerate(uniq)}
    cols = np.array([gmap[g] for g in genera])
    G = sparse.csr_matrix(
        (np.ones(m.n_vars), (np.arange(m.n_vars), cols)), shape=(m.n_vars, len(uniq))
    )
    M = m.X.tocsr() @ G
    xy_um = np.asarray(m.obsm["spatial"], dtype=np.float64) / PIX_PER_UM
    umi = np.asarray(M.sum(1)).ravel()
    use = umi >= min_umi
    xy_um = xy_um[use]
    M = M[use]
    log(f"[senders] bins kept={int(use.sum())} genera={len(uniq)}")

    signals = sorted(interactions["signal_id"].unique())
    E = np.zeros((len(uniq), len(signals)), dtype=np.float64)
    sig_index = {s: i for i, s in enumerate(signals)}
    genus_phylum = (
        m.var.dropna(subset=["genus"])
        .drop_duplicates("genus")
        .set_index("genus")["phylum"]
        .to_dict()
    )
    gram_of = {}
    for g, phy in genus_phylum.items():
        if phy in ("Bacillota", "Actinomycetota"):
            gram_of[g] = "Gram_positive"
        elif phy in ("Bacteroidota", "Pseudomonadota", "Fusobacteriota"):
            gram_of[g] = "Gram_negative"

    def add_prior(level, taxon, signal_id, w):
        if signal_id not in sig_index:
            return
        j = sig_index[signal_id]
        if level == "genus" and taxon in gmap:
            E[gmap[taxon], j] = max(E[gmap[taxon], j], w)
        elif level == "phylum":
            for g, i in gmap.items():
                if genus_phylum.get(g) == taxon:
                    E[i, j] = max(E[i, j], w)
        elif level == "gram":
            for g, i in gmap.items():
                if gram_of.get(g) == taxon:
                    E[i, j] = max(E[i, j], w)

    for _, row in priors.iterrows():
        add_prior(row["taxon_level"], row["taxon"], row["signal_id"], float(row["emission_weight"]))

    A_bin = np.asarray(M @ E, dtype=np.float64)
    gx = np.floor(xy_um[:, 0] / grid_um).astype(np.int64)
    gy = np.floor(xy_um[:, 1] / grid_um).astype(np.int64)
    keys = gx * 10_000_000 + gy
    df = pd.DataFrame({"key": keys, "x": xy_um[:, 0], "y": xy_um[:, 1]})
    for j, s in enumerate(signals):
        df[s] = A_bin[:, j]
    agg = df.groupby("key", sort=False).agg(
        x=("x", "mean"),
        y=("y", "mean"),
        **{s: (s, "sum") for s in signals},
    )
    tot = agg[signals].sum(1)
    agg = agg.loc[tot > 0].copy()
    sender_xy = agg[["x", "y"]].to_numpy(dtype=np.float64)
    A = agg[signals].to_numpy(dtype=np.float64)
    log(f"[senders] colonies={len(agg)} grid={grid_um}µm channels={len(signals)}")
    return sender_xy, A, signals


def received_signals_fast(
    receiver_xy: np.ndarray,
    sender_xy: np.ndarray,
    A: np.ndarray,
    radius_um: np.ndarray,
    scale: float = 1.0,
    dmax_factor: float = 3.0,
    label: str = "field",
) -> np.ndarray:
    """Received field: one neighbor graph, CSR weights @ sender amounts."""
    n, k = receiver_xy.shape[0], A.shape[1]
    n_send = sender_xy.shape[0]
    out = np.zeros((n, k), dtype=np.float64)
    r_max = float(np.max(radius_um) * dmax_factor)
    log(f"[{label}] building neighbor graph (r_max={r_max:.0f}µm)…")
    nn = NearestNeighbors(radius=r_max, algorithm="kd_tree").fit(sender_xy)
    dist_list, ind_list = nn.radius_neighbors(receiver_xy, return_distance=True)
    lengths = np.fromiter((d.size for d in dist_list), dtype=np.int64, count=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    nnz = int(indptr[-1])
    log(f"[{label}] edges={nnz} mean_nbrs={nnz / max(n, 1):.1f}")
    indices = np.empty(nnz, dtype=np.int64)
    distances = np.empty(nnz, dtype=np.float64)
    for i in range(n):
        a, b = indptr[i], indptr[i + 1]
        indices[a:b] = ind_list[i]
        distances[a:b] = dist_list[i]
    for j in range(k):
        r = float(radius_um[j])
        if r <= 0 or A[:, j].sum() <= 0:
            continue
        dmax = dmax_factor * r
        inv_r2 = 1.0 / (r * r)
        w = scale * np.exp(-0.5 * (distances * distances) * inv_r2)
        w[distances > dmax] = 0.0
        W = sparse.csr_matrix((w, indices, indptr), shape=(n, n_send))
        out[:, j] = np.asarray(W @ A[:, j]).ravel()
        if (j + 1) % max(1, k // 8) == 0 or j + 1 == k:
            log(f"[{label}] channel {j + 1}/{k} done")
    return out


def median_normalize_positive(S: np.ndarray) -> np.ndarray:
    out = S.copy()
    for j in range(out.shape[1]):
        pos = out[:, j] > 0
        if pos.any():
            out[:, j] /= np.median(out[pos, j])
    return out


def fit_lasso(X: np.ndarray, y: np.ndarray, names: list[str], random_state: int = 0) -> dict:
    if X.shape[1] == 0 or float(y.std()) < 1e-12:
        return {"r2_cv": 0.0, "r2_train": 0.0, "n_nonzero": 0, "top_coefs": [], "alpha": None}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    keep = Xs.std(0) > 1e-8
    Xs = Xs[:, keep]
    kept_names = [n for n, k in zip(names, keep) if k]
    if Xs.shape[1] == 0:
        return {"r2_cv": 0.0, "r2_train": 0.0, "n_nonzero": 0, "top_coefs": [], "alpha": None}
    model = LassoCV(
        cv=KFold(5, shuffle=True, random_state=random_state),
        alphas=30,
        max_iter=5000,
        n_jobs=4,
        random_state=random_state,
    )
    model.fit(Xs, y)
    r2_train = float(model.score(Xs, y))
    mse = float(model.mse_path_[model.alphas_ == model.alpha_].mean())
    sst = float(((y - y.mean()) ** 2).sum())
    r2_cv = float(1.0 - (mse * len(y)) / sst) if sst > 0 else 0.0
    coefs = {
        kept_names[i]: float(c)
        for i, c in enumerate(model.coef_)
        if abs(c) > 1e-8
    }
    top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:15]
    return {
        "r2_cv": r2_cv,
        "r2_train": r2_train,
        "n_nonzero": len(coefs),
        "alpha": float(model.alpha_),
        "top_coefs": [{"feature": a, "beta": b} for a, b in top],
    }


def feature_family(name: str, tf_names: list[str], lr_names: list[str], br_names: list[str]) -> str:
    if name in br_names:
        return "BR"
    if name in lr_names:
        return "LR"
    if name in tf_names:
        return "TF"
    return "OTHER"


def main():
    interactions = pd.read_csv(INTER)
    priors = pd.read_csv(PRIORS)

    log("[host] loading cells…")
    adata = ad.read_h5ad(HOST)
    gmap = gene_map(adata)
    targets = resolve(gmap, TARGETS)
    tfs = resolve(gmap, TF_CANDIDATES)
    log(f"[host] cells={adata.n_obs} targets={len(targets)} tfs={len(tfs)}")

    host_lr = select_host_lr_pairs(adata, gmap)
    log(f"[db] host–host LR pairs usable: {len(host_lr)} (cap={MAX_LR_PAIRS})")
    if len(host_lr):
        log(host_lr[["pair", "pathway", "signaling", "radius_um", "score"]].head(12).to_string(index=False))

    inter = interactions[interactions["receptor"].map(lambda r: r.lower() in gmap)].copy()
    inter["receptor_resolved"] = inter["receptor"].map(lambda r: gmap[r.lower()])
    log(f"[db] BR pairs usable: {len(inter)}")

    sender_xy, A, signals = build_senders_colony(UNMAP, priors, inter, grid_um=GRID_UM)

    rad = []
    for s in signals:
        sub = inter.loc[inter["signal_id"] == s, "default_radius_um"]
        rad.append(float(sub.iloc[0]) if len(sub) else 40.0)
    rad = np.asarray(rad, dtype=np.float64)

    receiver_xy_all = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    nn0 = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(sender_xy)
    d0, _ = nn0.kneighbors(receiver_xy_all)
    d0 = d0.ravel()
    close = np.where(d0 <= np.median(d0))[0]
    rng = np.random.default_rng(0)
    if len(close) > N_CELL_SUBSAMPLE:
        sel = rng.choice(close, size=N_CELL_SUBSAMPLE, replace=False)
    else:
        sel = close
    sel = np.sort(sel)
    log(f"[subset] training cells={len(sel)} (nearest-microbiome half, capped)")

    receiver_xy = receiver_xy_all[sel]
    log("[field] computing received microbial signals…")
    S = median_normalize_positive(
        received_signals_fast(receiver_xy, sender_xy, A, rad, scale=1.0, label="bact")
    )

    Xraw = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    tf_idx = [adata.var_names.get_loc(g) for g in tfs]
    tgt_idx = [adata.var_names.get_loc(g) for g in targets]
    br_rec_genes = sorted(inter["receptor_resolved"].unique())
    lr_lig_genes = sorted(host_lr["ligand"].unique()) if len(host_lr) else []
    lr_rec_genes = sorted(host_lr["receptor"].unique()) if len(host_lr) else []
    host_rec_genes = sorted(set(br_rec_genes) | set(lr_rec_genes))

    log("[impute] spatial knn impute (TFs/targets/receptors on subset; ligands on all cells)…")
    tf_expr = spatial_knn_impute(densify_cols(Xraw[sel], tf_idx), receiver_xy, k=15)
    host_rec_idx = [adata.var_names.get_loc(g) for g in host_rec_genes]
    host_rec_expr = spatial_knn_impute(densify_cols(Xraw[sel], host_rec_idx), receiver_xy, k=15)
    tgt_expr = spatial_knn_impute(densify_cols(Xraw[sel], tgt_idx), receiver_xy, k=15)
    host_rec_index = {g: i for i, g in enumerate(host_rec_genes)}

    # Host–host received ligands: all cells as senders
    LR = np.zeros((len(sel), 0), dtype=np.float64)
    lr_names: list[str] = []
    if len(host_lr) and lr_lig_genes:
        lig_idx = [adata.var_names.get_loc(g) for g in lr_lig_genes]
        lig_expr_all = spatial_knn_impute(
            densify_cols(Xraw, lig_idx), receiver_xy_all, k=15
        )
        lig_index = {g: i for i, g in enumerate(lr_lig_genes)}
        # group by radius so we can reuse neighbor graphs
        for radius, sub in host_lr.groupby("radius_um", sort=False):
            ligs = sorted(sub["ligand"].unique())
            A_lig = np.column_stack([lig_expr_all[:, lig_index[g]] for g in ligs])
            rad_vec = np.full(len(ligs), float(radius), dtype=np.float64)
            # truncated Gaussian; factor 2 keeps edges tractable vs SpaceTravLR's dense kernel
            dmax = 2.0 if float(radius) >= 100 else 3.0
            rec_L = median_normalize_positive(
                received_signals_fast(
                    receiver_xy,
                    receiver_xy_all,
                    A_lig,
                    rad_vec,
                    scale=1.0,
                    dmax_factor=dmax,
                    label=f"hostLR@{radius:.0f}",
                )
            )
            lig_pos = {g: i for i, g in enumerate(ligs)}
            for _, row in sub.iterrows():
                lig, rec = row["ligand"], row["receptor"]
                if rec not in host_rec_index:
                    continue
                name = f"{lig}${rec}"
                lr_names.append(name)
                col = rec_L[:, lig_pos[lig]] * host_rec_expr[:, host_rec_index[rec]]
                if LR.shape[1] == 0:
                    LR = col.reshape(-1, 1)
                else:
                    LR = np.column_stack([LR, col])

    br_names = []
    br_cols = []
    sig_index = {s: i for i, s in enumerate(signals)}
    for _, row in inter.iterrows():
        s, r = row["signal_id"], row["receptor_resolved"]
        if s not in sig_index or r not in host_rec_index:
            continue
        br_names.append(f"{s}${r}")
        br_cols.append(S[:, sig_index[s]] * host_rec_expr[:, host_rec_index[r]])
    BR = np.column_stack(br_cols) if br_cols else np.zeros((len(sel), 0))
    log(f"[features] TF={tf_expr.shape[1]} LR={LR.shape[1]} BR={BR.shape[1]} cells={len(sel)}")

    results = {
        "n_cells": int(len(sel)),
        "n_tf": len(tfs),
        "n_lr": len(lr_names),
        "n_br": len(br_names),
        "n_colonies": int(sender_xy.shape[0]),
        "paracrine_radius_um": PARACRINE_RADIUS_UM,
        "contact_radius_um": CONTACT_RADIUS_UM,
        "targets": {},
    }
    coef_rows = []
    modes_order = ["tf", "lr", "br", "tf_lr", "tf_br", "lr_br", "tf_lr_br"]

    for ti, gene in enumerate(targets):
        y = tgt_expr[:, ti]
        if (y > 0).mean() < 0.01:
            log(f"[skip] {gene}: too sparse")
            continue
        tf_mask = [g != gene for g in tfs]
        tf_X = tf_expr[:, tf_mask]
        tf_names = [g for g, keep in zip(tfs, tf_mask) if keep]
        # drop pairs that multiply by the target itself (e.g. App$Cd74 → Cd74)
        lr_mask = [not (n.endswith(f"${gene}") or n.startswith(f"{gene}$")) for n in lr_names]
        br_mask = [not n.endswith(f"${gene}") for n in br_names]
        lr_X = LR[:, lr_mask] if LR.shape[1] else LR
        br_X = BR[:, br_mask] if BR.shape[1] else BR
        lr_names_g = [n for n, k in zip(lr_names, lr_mask) if k]
        br_names_g = [n for n, k in zip(br_names, br_mask) if k]
        blocks = {
            "tf": (tf_X, tf_names),
            "lr": (lr_X, lr_names_g),
            "br": (br_X, br_names_g),
            "tf_lr": (np.hstack([tf_X, lr_X]), list(tf_names) + list(lr_names_g)),
            "tf_br": (np.hstack([tf_X, br_X]), list(tf_names) + list(br_names_g)),
            "lr_br": (np.hstack([lr_X, br_X]), list(lr_names_g) + list(br_names_g)),
            "tf_lr_br": (
                np.hstack([tf_X, lr_X, br_X]),
                list(tf_names) + list(lr_names_g) + list(br_names_g),
            ),
        }
        gene_res = {}
        for mode in modes_order:
            X, names = blocks[mode]
            fit = fit_lasso(X, y, names, random_state=0)
            gene_res[mode] = fit
            for c in fit["top_coefs"]:
                coef_rows.append(
                    {
                        "target": gene,
                        "mode": mode,
                        "feature": c["feature"],
                        "beta": c["beta"],
                        "family": feature_family(
                            c["feature"], tf_names, lr_names_g, br_names_g
                        ),
                    }
                )
            log(
                f"[lasso] {gene:8s} {mode:8s}  R2_cv={fit['r2_cv']:+.3f}  "
                f"R2_tr={fit['r2_train']:+.3f}  nz={fit['n_nonzero']}"
            )
        gene_res["delta_r2_cv_tf_to_tflr"] = gene_res["tf_lr"]["r2_cv"] - gene_res["tf"]["r2_cv"]
        gene_res["delta_r2_cv_tf_to_tfbr"] = gene_res["tf_br"]["r2_cv"] - gene_res["tf"]["r2_cv"]
        gene_res["delta_r2_cv_tflr_to_tflrbr"] = (
            gene_res["tf_lr_br"]["r2_cv"] - gene_res["tf_lr"]["r2_cv"]
        )
        results["targets"][gene] = gene_res

    rows = []
    for gene, gr in results["targets"].items():
        def hits(mode, family_names):
            return [
                c["feature"]
                for c in gr[mode]["top_coefs"]
                if c["feature"] in family_names
            ]

        lr_ok = [
            n
            for n in lr_names
            if not (n.endswith(f"${gene}") or n.startswith(f"{gene}$"))
        ]
        br_ok = [n for n in br_names if not n.endswith(f"${gene}")]
        rows.append(
            {
                "target": gene,
                "r2_tf": gr["tf"]["r2_cv"],
                "r2_lr": gr["lr"]["r2_cv"],
                "r2_br": gr["br"]["r2_cv"],
                "r2_tf_lr": gr["tf_lr"]["r2_cv"],
                "r2_tf_br": gr["tf_br"]["r2_cv"],
                "r2_lr_br": gr["lr_br"]["r2_cv"],
                "r2_tf_lr_br": gr["tf_lr_br"]["r2_cv"],
                "delta_tf_to_tflr": gr["delta_r2_cv_tf_to_tflr"],
                "delta_tf_to_tfbr": gr["delta_r2_cv_tf_to_tfbr"],
                "delta_tflr_to_tflrbr": gr["delta_r2_cv_tflr_to_tflrbr"],
                "top_lr_in_tflrbr": ", ".join(hits("tf_lr_br", lr_ok)[:6]),
                "top_br_in_tflrbr": ", ".join(hits("tf_lr_br", br_ok)[:6]),
            }
        )
    summary = pd.DataFrame(rows).sort_values("delta_tflr_to_tflrbr", ascending=False)
    summary.to_csv(OUT / "lasso_pilot_summary.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(OUT / "lasso_pilot_coefs.csv", index=False)
    (OUT / "lasso_pilot_results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        OUT / "br_features_subsample.npz",
        BR=BR,
        LR=LR,
        S=S,
        br_names=np.array(br_names),
        lr_names=np.array(lr_names),
        signals=np.array(signals),
        sel=sel,
        xy=receiver_xy,
    )
    host_lr.to_csv(OUT / "host_lr_pairs_used.csv", index=False)
    log("\n=== SUMMARY (sorted by ΔR² TF+LR → TF+LR+BR) ===")
    log(summary.to_string(index=False))
    log(f"[wrote] {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        raise
