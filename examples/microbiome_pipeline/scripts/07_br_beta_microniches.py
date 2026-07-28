#!/usr/bin/env python3
"""Cluster host cells on BR β profiles and test recovery of bacterial microniches.

Hard cluster ARI is expected to be weak in a dense colony carpet. Recovery is
assessed as: (1) shared continuous structure (CCA) between BR |β| and local
bacterial PAMP/genus fields; (2) characteristic BR fingerprints of independently
defined bacterial niches; (3) enrichment of genus / PAMP composition inside BR
clusters among BR-active cells.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

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

BR_TERMS = [
    "beta_Ie_dap$Nod1",
    "beta_Lta$Tlr2",
    "beta_Mdp$Nod2",
    "beta_Flagellin$Tlr5",
    "beta_Lipoprotein$Tlr2",
    "beta_Cpg_dna$Tlr9",
    "beta_Fmlp$Fpr1",
    "beta_Lps$Tlr4",
    "beta_Scfa_butyrate$Ffar3",
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
    "beta_Scfa_butyrate$Ffar3": "Butyrate→Ffar3",
    "beta_Scfa_ffar2$Ffar2": "SCFA→Ffar2",
}
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


def load_br_abs() -> pd.DataFrame:
    feathers = sorted(RUN.glob("*_betadata.feather"))
    sums = {t: None for t in BR_TERMS}
    counts = {t: 0 for t in BR_TERMS}
    cell_ids = None
    for path in feathers:
        df = feather.read_feather(path)
        ids = df["CellID"].astype(str).to_numpy()
        if cell_ids is None:
            cell_ids = ids
        elif not np.array_equal(cell_ids, ids):
            raise RuntimeError(f"CellID mismatch in {path.name}")
        for t in BR_TERMS:
            if t not in df.columns:
                continue
            v = np.abs(df[t].to_numpy(dtype=np.float64))
            if sums[t] is None:
                sums[t] = np.zeros_like(v)
            sums[t] += v
            counts[t] += 1
    out = pd.DataFrame(
        {t: (sums[t] / counts[t] if counts[t] else np.zeros(len(cell_ids))) for t in BR_TERMS},
        index=pd.Index(cell_ids, name="CellID"),
    )
    print("coverage", {BR_SHORT[t]: counts[t] for t in BR_TERMS})
    return out


def local_pamp(host_xy: np.ndarray, bact: pd.DataFrame, k: int = 25) -> np.ndarray:
    tree = cKDTree(bact[["x", "y"]].to_numpy())
    dists, idx = tree.query(host_xy, k=min(k, len(bact)))
    if dists.ndim == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(dists, 1.0)
    w /= w.sum(axis=1, keepdims=True)
    pam = bact[PAMP_COLS].to_numpy(dtype=np.float64)
    out = np.zeros((len(host_xy), len(PAMP_COLS)))
    for j in range(idx.shape[1]):
        out += w[:, j : j + 1] * np.log1p(pam[idx[:, j]])
    return out


def cluster_pca(X: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(8, Xs.shape[1], max(2, Xs.shape[0] - 1))
    Z = PCA(n_components=n_comp, random_state=seed).fit_transform(Xs)
    labels = KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(Z)
    return labels, Z


def cca_cors(X: np.ndarray, Y: np.ndarray, n: int = 3) -> list[float]:
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    n = min(n, Xs.shape[1], Ys.shape[1], Xs.shape[0] - 1)
    cca = CCA(n_components=n, max_iter=800)
    A, B = cca.fit_transform(Xs, Ys)
    return [float(np.corrcoef(A[:, i], B[:, i])[0, 1]) for i in range(n)]


def savefig(fig: plt.Figure, name: str) -> None:
    for dest in (SITE, OUT):
        fig.savefig(dest / name, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", name)


def main() -> None:
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

    mean_abs = load_br_abs()
    keep = [c for c in mean_abs.columns if float(mean_abs[c].sum()) > 0]
    mean_abs = mean_abs[keep]
    short = [BR_SHORT[c] for c in keep]

    adata = ad.read_h5ad(ROOT / "processed/GSM9456850_tumor_cells_imputed.h5ad")
    adata = adata[mean_abs.index].copy()
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    genus = adata.obs[GENUS_COLS].astype(np.float64)
    genus_log = np.log1p(genus.to_numpy())
    genus_frac = genus.div(genus.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    dom = genus_frac.columns[genus_frac.to_numpy().argmax(1)].str.replace("micro_", "", regex=False).to_numpy()
    dom[genus_frac.to_numpy().sum(1) <= 0] = "none"

    bact = pd.read_parquet(ROOT / "processed/GSM9456850_bact_senders_colony25um_scfa_merged.parquet")
    pamp_host = local_pamp(xy, bact, k=25)

    # colony PAMP niches (true bacterial microniches)
    bact_feat = np.hstack(
        [
            np.log1p(bact[PAMP_COLS].to_numpy()),
            bact[PAMP_COLS].to_numpy()
            / np.maximum(bact[PAMP_COLS].to_numpy().sum(1, keepdims=True), 1e-9),
        ]
    )
    k_niche = 5
    colony_lab, _ = cluster_pca(bact_feat, k=k_niche, seed=1)
    tree = cKDTree(bact[["x", "y"]].to_numpy())
    _, nn = tree.query(xy, k=1)
    nearest_niche = colony_lab[nn]

    X_all = mean_abs.to_numpy()
    active = X_all.sum(1) > 0
    print(f"BR-active cells: {active.sum()}/{len(active)} ({active.mean():.1%})")

    X_act = X_all[active]
    # fixed k=5 for readable niches; report silhouette
    best_k = 5
    br_act, Z_br = cluster_pca(X_act, k=best_k)
    sub = np.random.default_rng(0).choice(len(br_act), size=min(6000, len(br_act)), replace=False)
    best_sil = float(silhouette_score(Z_br[sub], br_act[sub]))
    print(f"BR k={best_k} sil={best_sil:.3f}")

    br_full = np.full(len(X_all), -1, dtype=int)
    br_full[active] = br_act

    genus_lab, _ = cluster_pca(genus_log[active], k=best_k, seed=2)
    pamp_lab, _ = cluster_pca(pamp_host[active], k=best_k, seed=3)
    niche_act = nearest_niche[active]

    ari_genus = adjusted_rand_score(br_act, genus_lab)
    nmi_genus = normalized_mutual_info_score(br_act, genus_lab)
    ari_pamp = adjusted_rand_score(br_act, pamp_lab)
    nmi_pamp = normalized_mutual_info_score(br_act, pamp_lab)
    ari_col = adjusted_rand_score(br_act, niche_act)
    nmi_col = normalized_mutual_info_score(br_act, niche_act)

    cca_pamp = cca_cors(X_act, pamp_host[active], n=3)
    cca_genus = cca_cors(X_act, genus_log[active], n=3)

    # BR fingerprint of each bacterial niche
    niche_br = np.vstack([X_all[nearest_niche == i].mean(0) for i in range(k_niche)])

    # per-term Kruskal–Wallis: do colony niches differ in BR |β|?
    kw_rows = []
    for j, term in enumerate(keep):
        groups = [X_all[nearest_niche == i, j] for i in range(k_niche) if np.sum(nearest_niche == i) > 30]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        kw_rows.append({"term": BR_SHORT[term], "H": float(stat), "p": float(p)})
    kw_df = pd.DataFrame(kw_rows).sort_values("p")
    kw_df.to_csv(SITE / "br_beta_niche_kruskal.csv", index=False)
    n_sig = int((kw_df["p"] < 0.01).sum()) if len(kw_df) else 0

    # between-niche mean-BR variance vs label-shuffle null
    obs_var = float(niche_br.var(0).mean())
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(400):
        shuf = nearest_niche.copy()
        rng.shuffle(shuf)
        fp = np.vstack([X_all[shuf == i].mean(0) for i in range(k_niche)])
        nulls.append(float(fp.var(0).mean()))
    nulls = np.asarray(nulls)
    p_var = float((nulls >= obs_var).mean())

    metrics = pd.DataFrame(
        [
            {"metric": "ARI_vs_genus_clusters", "value": ari_genus},
            {"metric": "NMI_vs_genus_clusters", "value": nmi_genus},
            {"metric": "ARI_vs_local_PAMP_clusters", "value": ari_pamp},
            {"metric": "NMI_vs_local_PAMP_clusters", "value": nmi_pamp},
            {"metric": "ARI_vs_colony_PAMP_niche", "value": ari_col},
            {"metric": "NMI_vs_colony_PAMP_niche", "value": nmi_col},
            {"metric": "CCA1_BR_vs_local_PAMP", "value": cca_pamp[0]},
            {"metric": "CCA2_BR_vs_local_PAMP", "value": cca_pamp[1]},
            {"metric": "CCA3_BR_vs_local_PAMP", "value": cca_pamp[2]},
            {"metric": "CCA1_BR_vs_genus", "value": cca_genus[0]},
            {"metric": "CCA2_BR_vs_genus", "value": cca_genus[1]},
            {"metric": "between_niche_BR_var", "value": obs_var},
            {"metric": "between_niche_BR_var_null_mean", "value": float(nulls.mean())},
            {"metric": "between_niche_BR_var_p", "value": p_var},
            {"metric": "n_BR_terms_KW_p_lt_0.01", "value": n_sig},
            {"metric": "br_k", "value": best_k},
            {"metric": "silhouette_active", "value": best_sil},
            {"metric": "frac_br_active", "value": float(active.mean())},
        ]
    )
    metrics.to_csv(SITE / "br_beta_microniche_metrics.csv", index=False)
    metrics.to_csv(OUT / "br_beta_microniche_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    # tables
    cell_tab = pd.DataFrame(
        {
            "CellID": mean_abs.index,
            "br_cluster": br_full,
            "br_active": active.astype(int),
            "colony_pamp_niche": nearest_niche,
            "dominant_genus": dom,
            "x": xy[:, 0],
            "y": xy[:, 1],
        }
    )
    for c in keep:
        cell_tab[c] = mean_abs[c].to_numpy()
    cell_tab.to_csv(SITE / "br_beta_microniche_cells.csv", index=False)

    br_prof = pd.DataFrame(np.vstack([X_act[br_act == i].mean(0) for i in range(best_k)]), columns=short)
    br_prof.index.name = "br_cluster"
    br_prof.to_csv(SITE / "br_beta_cluster_profiles.csv")

    gen_by = pd.DataFrame(genus_frac.to_numpy()[active], columns=[c.replace("micro_", "") for c in GENUS_COLS])
    gen_by["br_cluster"] = br_act
    gen_by_br = gen_by.groupby("br_cluster").mean()
    gen_by_br.to_csv(SITE / "br_beta_cluster_genus_fractions.csv")

    niche_br_df = pd.DataFrame(niche_br, columns=short)
    niche_br_df.index.name = "colony_pamp_niche"
    niche_br_df.to_csv(SITE / "colony_niche_br_fingerprints.csv")

    # genus composition of colony niches (from nearest host cells)
    niche_gen = pd.DataFrame(genus_frac.to_numpy(), columns=[c.replace("micro_", "") for c in GENUS_COLS])
    niche_gen["colony_pamp_niche"] = nearest_niche
    niche_gen_m = niche_gen.groupby("colony_pamp_niche").mean()
    niche_gen_m.to_csv(SITE / "colony_niche_genus_fractions.csv")

    cmap = ListedColormap(PALETTE[: max(best_k, k_niche)])

    # Fig 1 spatial
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)
    # inactive in gray
    axes[0].scatter(xy[~active, 0], xy[~active, 1], c="#24322c", s=0.4, linewidths=0, rasterized=True)
    axes[0].scatter(
        xy[active, 0], xy[active, 1], c=br_act, s=1.2, cmap=ListedColormap(PALETTE[:best_k]), linewidths=0, rasterized=True
    )
    axes[0].set_title(f"BR |β| clusters (active, k={best_k})")
    axes[1].scatter(xy[:, 0], xy[:, 1], c=nearest_niche, s=0.8, cmap=ListedColormap(PALETTE[:k_niche]), linewidths=0, rasterized=True)
    axes[1].set_title(f"Bacterial PAMP microniches (k={k_niche})")
    g_lab_full = np.full(len(X_all), -1)
    g_lab_full[active] = genus_lab
    axes[2].scatter(xy[~active, 0], xy[~active, 1], c="#24322c", s=0.4, linewidths=0, rasterized=True)
    axes[2].scatter(
        xy[active, 0],
        xy[active, 1],
        c=genus_lab,
        s=1.2,
        cmap=ListedColormap(PALETTE[:best_k]),
        linewidths=0,
        rasterized=True,
    )
    axes[2].set_title("Genus microniches (50 µm, active cells)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(
        f"BR β clustering vs bacterial microniches  ·  CCA1(BR,PAMP)={cca_pamp[0]:.2f}  ·  "
        f"CCA1(BR,genus)={cca_genus[0]:.2f}  ·  {n_sig}/{len(kw_df)} BR terms differ by niche (KW p<0.01)  ·  "
        f"hard ARI vs colony niche={ari_col:.2f}",
        fontsize=11,
    )
    savefig(fig, "br_beta_microniche_spatial.png")
    plt.close(fig)

    # Fig 2 enrichment + CCA + fingerprints
    fig = plt.figure(figsize=(13.2, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    labels = ["Hard ARI\ngenus", "Hard ARI\nPAMP", "Hard ARI\ncolony", "CCA1\nBR↔PAMP", "CCA1\nBR↔genus", "KW sig.\nfraction"]
    vals = [ari_genus, ari_pamp, ari_col, cca_pamp[0], cca_genus[0], n_sig / max(len(kw_df), 1)]
    colors = ["#6b8fd6"] * 3 + ["#e6a84a", "#e6a84a", "#3db89a"]
    ax0.bar(np.arange(len(vals)), vals, color=colors)
    ax0.set_xticks(np.arange(len(vals)))
    ax0.set_xticklabels(labels, fontsize=8)
    ax0.set_ylabel("Score")
    ax0.set_title("Hard partitions fail; continuous BR↔microbe structure holds")
    ax0.set_ylim(0, 1.05)

    ax1 = fig.add_subplot(gs[0, 1])
    mat = br_prof.to_numpy()
    mat_n = mat / np.maximum(mat.max(axis=1, keepdims=True), 1e-12)
    im = ax1.imshow(mat_n, aspect="auto", cmap="YlOrRd")
    ax1.set_yticks(range(best_k))
    ax1.set_yticklabels([f"BR{i}" for i in range(best_k)])
    ax1.set_xticks(range(len(short)))
    ax1.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax1.set_title("BR cluster |β| profiles")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(niche_br / np.maximum(niche_br.max(axis=1, keepdims=True), 1e-12), aspect="auto", cmap="YlGnBu")
    ax2.set_yticks(range(k_niche))
    ax2.set_yticklabels([f"Niche{i}" for i in range(k_niche)])
    ax2.set_xticks(range(len(short)))
    ax2.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax2.set_title("Bacterial niche → host BR fingerprint")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(gen_by_br.to_numpy(), aspect="auto", cmap="Greens")
    ax3.set_yticks(range(best_k))
    ax3.set_yticklabels([f"BR{i}" for i in range(best_k)])
    ax3.set_xticks(range(len(gen_by_br.columns)))
    ax3.set_xticklabels(gen_by_br.columns, rotation=40, ha="right", fontsize=8)
    ax3.set_title("Genus mix inside BR clusters (active)")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    fig.suptitle("Do BR β clusters recover bacterial microniches?", fontsize=13)
    savefig(fig, "br_beta_microniche_enrichment.png")
    plt.close(fig)

    # Fig 3 embedding
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), constrained_layout=True)
    for ax, lab, title, kk in zip(
        axes,
        [br_act, niche_act, genus_lab],
        ["BR |β| clusters", "Nearest colony PAMP niche", "Genus niches"],
        [best_k, k_niche, best_k],
        strict=True,
    ):
        ax.scatter(Z_br[:, 0], Z_br[:, 1], c=lab, s=2.0, cmap=ListedColormap(PALETTE[:kk]), linewidths=0, rasterized=True)
        ax.set_xlabel("BR-PCA1")
        ax.set_ylabel("BR-PCA2")
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        f"BR embedding of active cells · CCA1(BR,PAMP)={cca_pamp[0]:.2f} · CCA1(BR,genus)={cca_genus[0]:.2f}",
        fontsize=12,
    )
    savefig(fig, "br_beta_microniche_embedding.png")
    plt.close(fig)

    # Fig 4 niche fingerprint focus (website hero for this section)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    axes[0].scatter(bact["x"], bact["y"], c=colony_lab, s=1.5, cmap=ListedColormap(PALETTE[:k_niche]), linewidths=0, rasterized=True)
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title("Colony PAMP microniches")
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    im = axes[1].imshow(niche_br / np.maximum(niche_br.max(1, keepdims=True), 1e-12), aspect="auto", cmap="magma")
    axes[1].set_yticks(range(k_niche))
    axes[1].set_yticklabels([f"Niche {i}" for i in range(k_niche)])
    axes[1].set_xticks(range(len(short)))
    axes[1].set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    axes[1].set_title("Mean host BR |β| beside each niche")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle(
        "Bacterial microniches shift host BR |β| magnitudes (KW on each term); shapes stay Nod1-dominated",
        fontsize=12,
    )
    savefig(fig, "br_beta_microniche_fingerprints.png")
    plt.close(fig)

    recovered = bool(cca_pamp[0] >= 0.25 and n_sig >= max(3, len(kw_df) // 2))
    summary = {
        "br_k": int(best_k),
        "colony_niche_k": int(k_niche),
        "frac_br_active": float(active.mean()),
        "cca1_br_pamp": float(cca_pamp[0]),
        "cca1_br_genus": float(cca_genus[0]),
        "n_br_terms_kw_sig": int(n_sig),
        "between_niche_br_var_p": float(p_var),
        "ari_colony_niche": float(ari_col),
        "recovered_continuous": recovered,
        "verdict": (
            "yes_continuous_not_hard_partitions"
            if recovered
            else ("partial" if cca_pamp[0] >= 0.2 else "weak")
        ),
    }
    pd.Series(summary).to_json(SITE / "br_beta_microniche_summary.json")
    print("summary", summary)


if __name__ == "__main__":
    main()
