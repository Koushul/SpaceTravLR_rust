#!/usr/bin/env python3
"""Functional microniches driven by microbiome composition and secretions.

Goal: find tissue niches where bacterial community / PAMP secretion profiles
coincide with distinct host functional programs (sensing βs, gene modules,
cell-type mix).
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

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

PAMP_COLS = [
    "Cpg_dna",
    "Flagellin",
    "Fmlp",
    "Ie_dap",
    "Lipoprotein",
    "Lps",
    "Lta",
    "Mdp",
    "Scfa_butyrate",
    "Scfa_ffar2",
]
BR_TERMS = [
    "beta_Ie_dap$Nod1",
    "beta_Lta$Tlr2",
    "beta_Mdp$Nod2",
    "beta_Flagellin$Tlr5",
    "beta_Lipoprotein$Tlr2",
    "beta_Cpg_dna$Tlr9",
    "beta_Fmlp$Fpr1",
    "beta_Lps$Tlr4",
    "beta_Scfa_ffar2$Ffar2",
]
BR_SHORT = {
    "beta_Ie_dap$Nod1": "iE-DAP→Nod1",
    "beta_Lta$Tlr2": "LTA→Tlr2",
    "beta_Mdp$Nod2": "MDP→Nod2",
    "beta_Flagellin$Tlr5": "Flagellin→Tlr5",
    "beta_Lipoprotein$Tlr2": "Lipoprotein→Tlr2",
    "beta_Cpg_dna$Tlr9": "CpG→Tlr9",
    "beta_Fmlp$Fpr1": "fMLP→Fpr1",
    "beta_Lps$Tlr4": "LPS→Tlr4",
    "beta_Scfa_ffar2$Ffar2": "SCFA→Ffar2",
}
GENUS_COLS = [
    "micro_Duncaniella",
    "micro_Turicibacter",
    "micro_Clostridium",
    "micro_Lactobacillus",
    "micro_Faecalibaculum",
    "micro_Muribaculum",
    "micro_Staphylococcus",
    "micro_Vagococcus",
]

# Host functional gene modules (curated for gut + BR biology)
MODULES: dict[str, list[str]] = {
    "antimicrobial": ["Defa17", "Defa22", "Defa24", "Defa26", "Reg3b", "Ang4", "Lyz2", "Itln1", "Camp"],
    "chemokine_inflam": ["Ccl20", "Ccl25", "Cxcl10", "Cxcl9", "Nfkb1", "Rela", "Stat1", "Stat3", "Jun", "Fos"],
    "barrier_mucus": ["Muc2", "Muc4", "Tff3", "Pigr", "Cldn7", "Ocln"],
    "iga_plasma": ["Igha", "Ighm", "Jchain", "Cd74", "Pigr"],
    "enterocyte_metab": ["Fabp1", "Fabp2", "Apoa1", "Apoa4", "Alpi", "Sis", "Lct", "Ace2"],
    "stem_crypt": ["Lgr5", "Ascl2", "Myc", "Axin2", "Mki67"],
    "oxidative_burst": ["Duox2", "Duoxa2", "Nos2"],
}

PALETTE = [
    "#e6a84a",
    "#3db89a",
    "#6b8fd6",
    "#d4736b",
    "#c4a35a",
    "#8e7cc3",
    "#5dade2",
    "#58d68d",
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
        fig.savefig(dest / name, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", name)


def cluster_k(X: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    Xs = StandardScaler().fit_transform(X)
    n = min(10, Xs.shape[1], max(2, Xs.shape[0] - 1))
    Z = PCA(n_components=n, random_state=seed).fit_transform(Xs)
    lab = KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(Z)
    return lab, Z


def local_pamp(xy: np.ndarray, bact: pd.DataFrame, k: int = 25) -> np.ndarray:
    tree = cKDTree(bact[["x", "y"]].to_numpy())
    d, idx = tree.query(xy, k=min(k, len(bact)))
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(d, 1.0)
    w /= w.sum(axis=1, keepdims=True)
    pam = bact[PAMP_COLS].to_numpy(dtype=np.float64)
    out = np.zeros((len(xy), len(PAMP_COLS)))
    for j in range(idx.shape[1]):
        out += w[:, j : j + 1] * np.log1p(pam[idx[:, j]])
    return out


def load_br_abs(cell_ids: np.ndarray) -> pd.DataFrame:
    feathers = sorted(RUN.glob("*_betadata.feather"))
    sums = {t: None for t in BR_TERMS}
    counts = {t: 0 for t in BR_TERMS}
    order = None
    for path in feathers:
        df = feather.read_feather(path)
        ids = df["CellID"].astype(str).to_numpy()
        if order is None:
            order = ids
        for t in BR_TERMS:
            if t not in df.columns:
                continue
            v = np.abs(df[t].to_numpy(dtype=np.float64))
            if sums[t] is None:
                sums[t] = np.zeros(len(order))
            sums[t] += v
            counts[t] += 1
    out = pd.DataFrame(
        {t: sums[t] / counts[t] if counts[t] else np.zeros(len(order)) for t in BR_TERMS},
        index=pd.Index(order, name="CellID"),
    )
    return out.reindex(cell_ids).fillna(0.0)


def module_scores(adata: ad.AnnData) -> pd.DataFrame:
    # use imputed_count log1p, z-score genes, mean per module
    X = adata.layers["imputed_count"]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    var = pd.Index(adata.var_names.astype(str))
    # log1p
    X = np.log1p(X)
    # standardize genes present
    scores = {}
    for name, genes in MODULES.items():
        present = [g for g in genes if g in var]
        if not present:
            scores[name] = np.zeros(adata.n_obs)
            continue
        idx = var.get_indexer(present)
        block = X[:, idx]
        mu = block.mean(0, keepdims=True)
        sd = block.std(0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        scores[name] = ((block - mu) / sd).mean(1)
    return pd.DataFrame(scores, index=adata.obs_names.astype(str))


def cca1(X: np.ndarray, Y: np.ndarray) -> float:
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    n = min(1, Xs.shape[1], Ys.shape[1], Xs.shape[0] - 1)
    if n < 1:
        return 0.0
    a, b = CCA(n_components=1, max_iter=500).fit_transform(Xs, Ys)
    return float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])


def main() -> None:
    style()
    adata = ad.read_h5ad(ROOT / "processed/GSM9456850_tumor_cells_imputed.h5ad")
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    cell_ids = adata.obs_names.astype(str).to_numpy()
    genus = adata.obs[GENUS_COLS].astype(np.float64)
    genus_log = np.log1p(genus.to_numpy())
    genus_frac = genus.div(genus.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    bact = pd.read_parquet(ROOT / "processed/GSM9456850_bact_senders_colony25um_scfa_merged.parquet")
    pamp = local_pamp(xy, bact, k=25)
    br = load_br_abs(cell_ids)
    br_keep = [c for c in br.columns if float(br[c].sum()) > 0]
    br = br[br_keep]
    print("scoring modules…")
    mods = module_scores(adata)
    mod_cols = list(mods.columns)

    # --- 1) Secretion niches on colonies, mapped to host ---
    k_sec = 5
    bact_feat = np.hstack(
        [
            np.log1p(bact[PAMP_COLS].to_numpy()),
            bact[PAMP_COLS].to_numpy()
            / np.maximum(bact[PAMP_COLS].to_numpy().sum(1, keepdims=True), 1e-9),
        ]
    )
    sec_lab_col, _ = cluster_k(bact_feat, k=k_sec, seed=1)
    tree = cKDTree(bact[["x", "y"]].to_numpy())
    _, nn = tree.query(xy, k=1)
    sec_lab = sec_lab_col[nn]

    # --- 2) Joint microbe niches on host: genus + local PAMP ---
    k_micro = 5
    micro_feat = np.hstack([genus_log, pamp])
    micro_lab, Z_micro = cluster_k(micro_feat, k=k_micro, seed=2)

    # --- 3) Functional niches on host: modules + BR |β| ---
    k_fun = 5
    fun_feat = np.hstack([mods.to_numpy(), br.to_numpy()])
    fun_lab, Z_fun = cluster_k(fun_feat, k=k_fun, seed=3)

    # --- 4) Joint NMF: microbe secretions/genus ↔ host function ---
    # Build non-negative joint matrix for host cells
    joint = np.hstack(
        [
            MaxAbsScaler().fit_transform(np.log1p(genus.to_numpy())),
            MaxAbsScaler().fit_transform(pamp),
            MaxAbsScaler().fit_transform(np.clip(mods.to_numpy() - mods.to_numpy().min(0), 0, None)),
            MaxAbsScaler().fit_transform(br.to_numpy()),
        ]
    )
    n_comp = 4
    nmf = NMF(n_components=n_comp, init="nndsvda", max_iter=800, random_state=0)
    W = nmf.fit_transform(joint + 1e-9)
    H = nmf.components_
    nmf_lab = W.argmax(1)

    # feature names for H interpretation
    feat_names = (
        [c.replace("micro_", "g:") for c in GENUS_COLS]
        + [f"p:{c}" for c in PAMP_COLS]
        + [f"f:{c}" for c in mod_cols]
        + [f"br:{BR_SHORT.get(c, c)}" for c in br_keep]
    )
    H_df = pd.DataFrame(H, columns=feat_names, index=[f"factor{i}" for i in range(n_comp)])
    H_df.to_csv(SITE / "functional_microniche_nmf_loadings.csv")

    # CCA: secretions/genus vs function
    cca_pamp_mod = cca1(pamp, mods.to_numpy())
    cca_pamp_br = cca1(pamp, br.to_numpy())
    cca_genus_mod = cca1(genus_log, mods.to_numpy())
    cca_micro_fun = cca1(micro_feat, fun_feat)

    # Per secretion niche: mean modules, BR, cell types, genus
    cell_type = adata.obs["cell_type"].astype(str).to_numpy()
    rows = []
    for i in range(k_sec):
        m = sec_lab == i
        row = {
            "secretion_niche": i,
            "n_host_cells": int(m.sum()),
            "n_colonies": int((sec_lab_col == i).sum()),
        }
        for j, p in enumerate(PAMP_COLS):
            row[f"pamp_{p}"] = float(pamp[m, j].mean()) if m.any() else 0.0
        for c in mod_cols:
            row[f"mod_{c}"] = float(mods.loc[cell_ids[m], c].mean()) if m.any() else 0.0
        for c in br_keep:
            row[f"br_{BR_SHORT.get(c, c)}"] = float(br.loc[cell_ids[m], c].mean()) if m.any() else 0.0
        # top cell type
        if m.any():
            vc = pd.Series(cell_type[m]).value_counts(normalize=True)
            row["top_celltype"] = vc.index[0]
            row["top_celltype_frac"] = float(vc.iloc[0])
            # dominant genus among nearby
            gmean = genus_frac.to_numpy()[m].mean(0)
            row["top_genus"] = GENUS_COLS[int(gmean.argmax())].replace("micro_", "")
            row["top_genus_frac"] = float(gmean.max())
            # top functional module
            mod_m = mods.to_numpy()[m].mean(0)
            row["top_module"] = mod_cols[int(mod_m.argmax())]
            row["top_module_score"] = float(mod_m.max())
        rows.append(row)
    niche_summary = pd.DataFrame(rows)
    niche_summary.to_csv(SITE / "functional_secretion_niche_summary.csv", index=False)

    # Kruskal: modules differ by secretion niche?
    kw = []
    for c in mod_cols:
        groups = [mods.to_numpy()[sec_lab == i, mod_cols.index(c)] for i in range(k_sec) if (sec_lab == i).sum() > 50]
        if len(groups) < 2:
            continue
        Hstat, p = kruskal(*groups)
        kw.append({"feature": c, "class": "module", "H": float(Hstat), "p": float(p)})
    for c in br_keep:
        groups = [br.to_numpy()[sec_lab == i, br_keep.index(c)] for i in range(k_sec) if (sec_lab == i).sum() > 50]
        if len(groups) < 2:
            continue
        Hstat, p = kruskal(*groups)
        kw.append({"feature": BR_SHORT.get(c, c), "class": "BR", "H": float(Hstat), "p": float(p)})
    kw_df = pd.DataFrame(kw).sort_values("p")
    kw_df.to_csv(SITE / "functional_microniche_kruskal.csv", index=False)

    # Microbe→function: which PAMPs correlate with which modules (Spearman)
    corr_rows = []
    for j, p in enumerate(PAMP_COLS):
        for c in mod_cols:
            r, pv = spearmanr(pamp[:, j], mods[c].to_numpy())
            corr_rows.append({"pamp": p, "module": c, "spearman_r": float(r), "p": float(pv)})
        for c in br_keep:
            r, pv = spearmanr(pamp[:, j], br[c].to_numpy())
            corr_rows.append(
                {"pamp": p, "module": BR_SHORT.get(c, c), "spearman_r": float(r), "p": float(pv)}
            )
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_r", key=np.abs, ascending=False)
    corr_df.to_csv(SITE / "functional_pamp_module_correlations.csv", index=False)

    # Cell-level table (subset cols)
    tab = pd.DataFrame(
        {
            "CellID": cell_ids,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "secretion_niche": sec_lab,
            "microbe_niche": micro_lab,
            "function_niche": fun_lab,
            "nmf_factor": nmf_lab,
            "cell_type": cell_type,
        }
    )
    for c in mod_cols:
        tab[c] = mods[c].to_numpy()
    tab.to_csv(SITE / "functional_microniche_cells.csv", index=False)

    metrics = {
        "cca_pamp_vs_modules": cca_pamp_mod,
        "cca_pamp_vs_BR": cca_pamp_br,
        "cca_genus_vs_modules": cca_genus_mod,
        "cca_microbe_vs_function": cca_micro_fun,
        "n_modules_KW_p_lt_0.01": int(((kw_df["class"] == "module") & (kw_df["p"] < 0.01)).sum()),
        "n_BR_KW_p_lt_0.01": int(((kw_df["class"] == "BR") & (kw_df["p"] < 0.01)).sum()),
        "k_secretion": k_sec,
        "k_microbe": k_micro,
        "k_function": k_fun,
        "nmf_components": n_comp,
    }
    pd.Series(metrics).to_json(SITE / "functional_microniche_metrics.json")
    print(json.dumps(metrics, indent=2))
    print(niche_summary[["secretion_niche", "n_host_cells", "top_genus", "top_module", "top_celltype"]].to_string(index=False))

    # ========== FIGURES ==========
    cmap5 = ListedColormap(PALETTE[:5])
    cmap4 = ListedColormap(PALETTE[:4])

    # Fig A: spatial quartet — secretion / microbe / function / NMF
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
    panels = [
        (axes[0, 0], sec_lab, cmap5, "Secretion microniches\n(colony PAMP profiles)"),
        (axes[0, 1], micro_lab, cmap5, "Microbe microniches\n(local genus + PAMP exposure)"),
        (axes[1, 0], fun_lab, cmap5, "Functional microniches\n(host modules + BR |β|)"),
        (axes[1, 1], nmf_lab, cmap4, "Joint NMF factors\n(microbe ↔ host function)"),
    ]
    for ax, lab, cmap, title in panels:
        ax.scatter(xy[:, 0], xy[:, 1], c=lab, s=0.7, cmap=cmap, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(
        f"Functional microniches driven by microbiome / secretions\n"
        f"CCA(microbe, function)={cca_micro_fun:.2f} · CCA(PAMP, modules)={cca_pamp_mod:.2f} · "
        f"CCA(PAMP, BR)={cca_pamp_br:.2f}",
        fontsize=12,
    )
    save(fig, "functional_microniche_spatial.png")
    plt.close(fig)

    # Fig B: secretion niche fingerprints — PAMPs, modules, BR, cell types
    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # PAMP profile per secretion niche
    ax0 = fig.add_subplot(gs[0, 0])
    pamp_by = np.vstack([pamp[sec_lab == i].mean(0) for i in range(k_sec)])
    pamp_n = pamp_by / np.maximum(pamp_by.max(axis=1, keepdims=True), 1e-12)
    im0 = ax0.imshow(pamp_n, aspect="auto", cmap="YlOrRd")
    ax0.set_yticks(range(k_sec))
    ax0.set_yticklabels([f"Sec{i}" for i in range(k_sec)])
    ax0.set_xticks(range(len(PAMP_COLS)))
    ax0.set_xticklabels(PAMP_COLS, rotation=40, ha="right", fontsize=8)
    ax0.set_title("Secretion niche · local PAMP exposure")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[0, 1])
    mod_by = np.vstack([mods.to_numpy()[sec_lab == i].mean(0) for i in range(k_sec)])
    im1 = ax1.imshow(mod_by, aspect="auto", cmap="coolwarm", vmin=-np.percentile(np.abs(mod_by), 95), vmax=np.percentile(np.abs(mod_by), 95))
    ax1.set_yticks(range(k_sec))
    ax1.set_yticklabels([f"Sec{i}" for i in range(k_sec)])
    ax1.set_xticks(range(len(mod_cols)))
    ax1.set_xticklabels(mod_cols, rotation=35, ha="right", fontsize=8)
    ax1.set_title("Host functional modules by secretion niche")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[1, 0])
    br_by = np.vstack([br.to_numpy()[sec_lab == i].mean(0) for i in range(k_sec)])
    br_n = br_by / np.maximum(br_by.max(axis=1, keepdims=True), 1e-12)
    im2 = ax2.imshow(br_n, aspect="auto", cmap="magma")
    ax2.set_yticks(range(k_sec))
    ax2.set_yticklabels([f"Sec{i}" for i in range(k_sec)])
    ax2.set_xticks(range(len(br_keep)))
    ax2.set_xticklabels([BR_SHORT.get(c, c) for c in br_keep], rotation=35, ha="right", fontsize=8)
    ax2.set_title("BR |β| sensing fingerprint by secretion niche")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[1, 1])
    # cell type composition for top types
    top_types = pd.Series(cell_type).value_counts().head(8).index.tolist()
    ct = np.zeros((k_sec, len(top_types)))
    for i in range(k_sec):
        vc = pd.Series(cell_type[sec_lab == i]).value_counts(normalize=True)
        for j, t in enumerate(top_types):
            ct[i, j] = float(vc.get(t, 0.0))
    im3 = ax3.imshow(ct, aspect="auto", cmap="Greens")
    ax3.set_yticks(range(k_sec))
    ax3.set_yticklabels([f"Sec{i}" for i in range(k_sec)])
    ax3.set_xticks(range(len(top_types)))
    ax3.set_xticklabels(top_types, rotation=35, ha="right", fontsize=7)
    ax3.set_title("Host cell-type mix by secretion niche")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    fig.suptitle("Secretion-driven niches → host function", fontsize=13)
    save(fig, "functional_secretion_niche_fingerprints.png")
    plt.close(fig)

    # Fig C: PAMP↔module correlation heatmap (modules only)
    pivot = corr_df[corr_df["module"].isin(mod_cols)].pivot(index="pamp", columns="module", values="spearman_r")
    pivot = pivot.reindex(index=PAMP_COLS, columns=mod_cols)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
    im = axes[0].imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_yticks(range(len(PAMP_COLS)))
    axes[0].set_yticklabels(PAMP_COLS, fontsize=8)
    axes[0].set_xticks(range(len(mod_cols)))
    axes[0].set_xticklabels(mod_cols, rotation=35, ha="right", fontsize=8)
    axes[0].set_title("Spearman: local PAMP vs host modules")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # NMF loadings: top features per factor
    ax = axes[1]
    # show function + microbe loadings as heatmap (selected)
    show_cols = [c for c in H_df.columns if c.startswith(("g:", "p:", "f:", "br:"))]
    Hshow = H_df[show_cols]
    # keep top-variance features for readability
    keep_f = Hshow.var(axis=0).sort_values(ascending=False).head(22).index.tolist()
    Hs = Hshow[keep_f]
    Hs_n = Hs.div(Hs.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    im2 = ax.imshow(Hs_n.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels(Hs_n.index)
    ax.set_xticks(range(len(keep_f)))
    ax.set_xticklabels(keep_f, rotation=55, ha="right", fontsize=7)
    ax.set_title("NMF loadings (microbe + function factors)")
    fig.colorbar(im2, ax=ax, fraction=0.046)
    fig.suptitle("Secretions covary with host programs", fontsize=12)
    save(fig, "functional_pamp_module_links.png")
    plt.close(fig)

    # Fig D: callout cards — label each secretion niche functionally
    fig, axes = plt.subplots(1, k_sec, figsize=(14.5, 3.8), constrained_layout=True)
    for i, ax in enumerate(axes):
        m = sec_lab == i
        ax.scatter(xy[~m, 0], xy[~m, 1], c="#1a2822", s=0.25, linewidths=0, rasterized=True)
        ax.scatter(xy[m, 0], xy[m, 1], c=PALETTE[i], s=1.2, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        r = niche_summary.iloc[i]
        ax.set_title(
            f"Sec{i}\n{r.top_genus} · {r.top_module}\n{r.top_celltype}",
            fontsize=8,
        )
    fig.suptitle("Per-niche identity: dominant genus · top host module · top cell type", fontsize=12)
    save(fig, "functional_secretion_niche_callouts.png")
    plt.close(fig)

    # Fig E: metrics bar
    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    names = ["CCA\nPAMP↔modules", "CCA\nPAMP↔BR", "CCA\ngenus↔modules", "CCA\nmicrobe↔function", "Modules\nKW sig frac", "BR\nKW sig frac"]
    vals = [
        cca_pamp_mod,
        cca_pamp_br,
        cca_genus_mod,
        cca_micro_fun,
        metrics["n_modules_KW_p_lt_0.01"] / max(len(mod_cols), 1),
        metrics["n_BR_KW_p_lt_0.01"] / max(len(br_keep), 1),
    ]
    ax.bar(np.arange(len(vals)), vals, color=["#e6a84a", "#e6a84a", "#3db89a", "#3db89a", "#6b8fd6", "#6b8fd6"])
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Evidence that microbiome/secretions organize host functional microniches")
    save(fig, "functional_microniche_metrics.png")
    plt.close(fig)

    # human-readable niche blurbs
    blurbs = []
    for i, r in niche_summary.iterrows():
        # strongest PAMPs
        pcols = [c for c in niche_summary.columns if c.startswith("pamp_")]
        top_p = niche_summary.loc[i, pcols].astype(float).sort_values(ascending=False).head(2)
        blurbs.append(
            {
                "secretion_niche": int(r.secretion_niche),
                "n_host_cells": int(r.n_host_cells),
                "label": f"{r.top_genus}-enriched · {r.top_module}",
                "top_celltype": r.top_celltype,
                "top_pamps": ", ".join([c.replace("pamp_", "") for c in top_p.index]),
                "summary": (
                    f"Sec{int(r.secretion_niche)}: {r.top_genus}-associated secretion niche; "
                    f"host program {r.top_module}; enriched {r.top_celltype}; "
                    f"elevated local {', '.join(c.replace('pamp_','') for c in top_p.index)}."
                ),
            }
        )
    pd.DataFrame(blurbs).to_csv(SITE / "functional_secretion_niche_blurbs.csv", index=False)
    (SITE / "functional_secretion_niche_blurbs.json").write_text(json.dumps(blurbs, indent=2))
    print("blurbs:")
    for b in blurbs:
        print("-", b["summary"])


if __name__ == "__main__":
    main()
