#!/usr/bin/env python3
"""Robust pathway / contact / cycling analysis for get-microniches outputs.

Runs for GC B cells and Tfh:
  - spatial neighbor enrichment (Tfh/FDC/GC/…)
  - independent program / cell-cycle module scores
  - niche DE → Enrichr (GO BP, Reactome, Hallmark)
  - decoupler ORA on Hallmark when available
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("HOME", "/tmp/fakehome")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

H5AD = Path("/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil_processed.h5ad")
OUT = Path(
    os.environ.get(
        "MICRONICHE_PATHWAY_OUT",
        Path(__file__).resolve().parent / "public" / "tonsil_niche_benchmark" / "pathway_analysis",
    )
)
SEED = 0
K_SPATIAL = 15
MIN_NICHE = 10

PROGRAMS = {
    "CSR_proliferation": ["AICDA", "CXCR4", "FOXO1", "BCL6", "TOP2A"],
    "selection_activation": ["CD83", "CXCR5", "CD40", "LMO2", "BATF"],
    "plasma_exit": ["IRF4", "PRDM1"],
    "BAFF_BCR_axis": ["TNFRSF13B", "TNFRSF13C", "BANK1", "CD86", "CR2"],
    "FDC_cue": ["FCER2", "CR2", "FDCSP", "CXCL13"],
    "Tfh_receptor_axis": ["ICOS", "PDCD1", "CXCR5", "CD40"],
    "cell_cycle": ["TOP2A", "HMGB2", "CENPF", "RRM2"],
    "B_activation": ["CD69", "NFKBIA", "FOS", "CD86"],
}

TFH_PROGRAMS = {
    "Tfh_canonical": ["CXCL13", "PDCD1", "ICOS", "CXCR5", "BCL6", "IL21", "MAF", "TOX2", "SH2D1A"],
    "cytokine_help": ["IL21", "CD40LG", "CXCL13", "IFNG"],
    "exhaustion_checkpoint": ["PDCD1", "TIGIT", "LAG3", "HAVCR2", "CTLA4"],
    "cell_cycle": ["TOP2A", "HMGB2", "CENPF", "RRM2"],
    "TCR_activation": ["CD69", "FOS", "NFKBIA", "NR4A1", "NR4A2", "NR4A3"],
}

ENRICHR_LIBS = [
    "GO_Biological_Process_2025",
    "Reactome_Pathways_2024",
    "MSigDB_Hallmark_2020",
]


def eta_squared_oneway(y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    cats = pd.Categorical(groups)
    if cats.categories.size < 2:
        return 0.0, 1.0
    frames = [y[cats.codes == i] for i in range(cats.categories.size)]
    frames = [f for f in frames if len(f) > 1]
    if len(frames) < 2:
        return 0.0, 1.0
    F, p = stats.f_oneway(*frames)
    ss_between = sum(len(f) * (f.mean() - y.mean()) ** 2 for f in frames)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    eta = float(ss_between / ss_tot) if ss_tot > 0 else 0.0
    return eta, float(p)


def module_score(adata: AnnData, genes: list[str], key: str) -> np.ndarray:
    present = [g for g in genes if g in adata.var_names]
    if len(present) < 1:
        adata.obs[key] = 0.0
        return np.zeros(adata.n_obs, dtype=np.float64)
    sc.tl.score_genes(adata, gene_list=present, score_name=key, use_raw=False, random_state=SEED)
    return np.asarray(adata.obs[key], dtype=np.float64)


def neighbor_fractions(
    query_xy: np.ndarray,
    full_xy: np.ndarray,
    full_types: np.ndarray,
    types_of_interest: list[str],
    k: int = K_SPATIAL,
) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(full_xy)
    idx = nn.kneighbors(query_xy, return_distance=False)
    # drop self when query is a subset of full — approximate by skipping nearest if identical coords
    rows = {t: np.zeros(len(query_xy)) for t in types_of_interest}
    for i in range(len(query_xy)):
        neigh = idx[i]
        # exclude exact self match if present
        keep = []
        for j in neigh:
            if np.allclose(full_xy[j], query_xy[i]) and len(keep) == 0:
                # likely self; skip first exact hit
                continue
            keep.append(j)
            if len(keep) >= k:
                break
        if len(keep) < k:
            keep = [j for j in neigh if not (np.allclose(full_xy[j], query_xy[i]))][:k]
        labs = full_types[keep]
        for t in types_of_interest:
            rows[t][i] = float((labs == t).mean()) if len(keep) else 0.0
    return pd.DataFrame(rows)


def niche_deg(adata: AnnData, labels: np.ndarray, top_n: int = 75) -> dict[str, list[str]]:
    ad = adata.copy()
    ad.obs["niche"] = pd.Categorical(labels.astype(str))
    vc = ad.obs["niche"].value_counts()
    keep = vc[vc >= MIN_NICHE].index.astype(str).tolist()
    ad = ad[ad.obs["niche"].isin(keep)].copy()
    if ad.n_obs < 30 or len(keep) < 2:
        return {}
    sc.tl.rank_genes_groups(ad, groupby="niche", method="wilcoxon", use_raw=False)
    out: dict[str, list[str]] = {}
    for g in keep:
        df = sc.get.rank_genes_groups_df(ad, group=g)
        sig = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0.25)]
        if len(sig) >= 5:
            out[g] = sig["names"].head(top_n).astype(str).tolist()
            continue
        pos = df[df["logfoldchanges"] > 0].sort_values("scores", ascending=False)
        if len(pos) < 5:
            pos = df.sort_values("scores", ascending=False)
        out[g] = pos["names"].head(top_n).astype(str).tolist()
    return out


def enrich_gene_lists(gene_lists: dict[str, list[str]]) -> pd.DataFrame:
    import gseapy as gp

    rows = []
    for niche, genes in gene_lists.items():
        genes = [g for g in genes if isinstance(g, str) and g]
        if len(genes) < 5:
            continue
        for lib in ENRICHR_LIBS:
            try:
                time.sleep(0.25)
                enr = gp.enrichr(
                    gene_list=genes, gene_sets=lib, organism="human", outdir=None, verbose=False
                )
                df = enr.results
            except Exception as exc:
                rows.append(
                    {
                        "niche": niche,
                        "library": lib,
                        "term": f"ERROR: {exc}",
                        "adj_p": np.nan,
                        "odds": np.nan,
                        "overlap": "",
                        "genes": "",
                    }
                )
                continue
            if df is None or df.empty:
                continue
            df = df.sort_values("Adjusted P-value").head(10)
            for _, r in df.iterrows():
                rows.append(
                    {
                        "niche": niche,
                        "library": lib,
                        "term": r.get("Term", ""),
                        "adj_p": float(r.get("Adjusted P-value", np.nan)),
                        "odds": float(r.get("Odds Ratio", np.nan))
                        if pd.notna(r.get("Odds Ratio", np.nan))
                        else np.nan,
                        "overlap": r.get("Overlap", ""),
                        "genes": r.get("Genes", ""),
                    }
                )
    return pd.DataFrame(rows)


def decoupler_hallmark_ora(gene_lists: dict[str, list[str]], background: list[str]) -> pd.DataFrame:
    try:
        import decoupler as dc
    except Exception:
        return pd.DataFrame()
    try:
        msig = dc.op.resource("MSigDB")
        # prefer hallmark
        if "collection" in msig.columns:
            hall = msig[msig["collection"].astype(str).str.contains("hallmark", case=False, na=False)]
        elif "geneset" in msig.columns:
            hall = msig[msig["geneset"].astype(str).str.startswith("HALLMARK")]
        else:
            hall = msig
        if hall.empty:
            return pd.DataFrame()
        net = hall.rename(columns={"genesymbol": "target", "geneset": "source"})[
            ["source", "target"]
        ].drop_duplicates()
    except Exception:
        return pd.DataFrame()

    rows = []
    bg = set(background)
    for niche, genes in gene_lists.items():
        hits = [g for g in genes if g in bg]
        if len(hits) < 5:
            continue
        try:
            enr = dc.mt.ora(features=hits, net=net, tmin=3)
            if enr is None or len(enr) == 0:
                continue
            df = enr.copy()
            # normalize column names across decoupler versions
            pcol = "padj" if "padj" in df.columns else ("pvals_adj" if "pvals_adj" in df.columns else None)
            termcol = "source" if "source" in df.columns else ("Term" if "Term" in df.columns else df.columns[0])
            if pcol is None:
                continue
            df = df.sort_values(pcol).head(12)
            for _, r in df.iterrows():
                rows.append(
                    {
                        "niche": niche,
                        "library": "decoupler_Hallmark_ORA",
                        "term": str(r[termcol]),
                        "adj_p": float(r[pcol]),
                        "odds": np.nan,
                        "overlap": "",
                        "genes": "",
                    }
                )
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_expr_subset(full: AnnData, mask: np.ndarray) -> AnnData:
    ad = full[mask].copy()
    if "raw_count" in ad.layers:
        ad.X = ad.layers["raw_count"].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    return ad


def analyze_cohort(
    name: str,
    labels_csv: Path,
    summary_json: Path,
    full: AnnData,
    programs: dict[str, list[str]],
    neighbor_types: list[str],
    contact_focus: str,
) -> dict:
    out_dir = OUT / name
    img = out_dir / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(labels_csv)
    summary = json.loads(summary_json.read_text()) if summary_json.is_file() else {}
    # align to AnnData
    lab["cell_id"] = lab["cell_id"].astype(str)
    full_ids = full.obs_names.astype(str)
    idx = full_ids.get_indexer(lab["cell_id"].to_numpy())
    if (idx < 0).any():
        missing = int((idx < 0).sum())
        raise RuntimeError(f"{name}: {missing} label cell_ids not in AnnData")
    mask = np.zeros(full.n_obs, dtype=bool)
    mask[idx] = True
    ad = load_expr_subset(full, mask)
    # reorder labels to ad.obs_names
    lab = lab.set_index("cell_id").loc[ad.obs_names.astype(str)]
    labels = lab["microniche"].astype(str).to_numpy()
    spatial = np.asarray(ad.obsm["spatial"], dtype=np.float64)
    xy_full = np.asarray(full.obsm["spatial"], dtype=np.float64)
    types_full = full.obs["cell_type"].astype(str).to_numpy()

    # neighbor contact
    nhood = neighbor_fractions(spatial, xy_full, types_full, neighbor_types, k=K_SPATIAL)
    nhood["niche"] = labels
    nhood_mean = nhood.groupby("niche")[neighbor_types].mean()
    nhood_mean.to_csv(out_dir / "neighbor_means_by_niche.csv")
    contact_eta = {}
    for t in neighbor_types:
        eta, p = eta_squared_oneway(nhood[t].to_numpy(), labels)
        contact_eta[t] = {"eta2": eta, "anova_p": p}
    pd.DataFrame(contact_eta).T.to_csv(out_dir / "neighbor_eta2.csv")

    # programs / cycling
    prog_scores = {}
    prog_rows = []
    for pname, genes in programs.items():
        present = [g for g in genes if g in ad.var_names]
        score = module_score(ad, genes, key=f"prog_{pname}")
        prog_scores[pname] = score
        eta, p = eta_squared_oneway(score, labels)
        # Kruskal as robust alternative
        cats = pd.Categorical(labels)
        frames = [score[cats.codes == i] for i in range(cats.categories.size) if (cats.codes == i).sum() > 1]
        kruskal_p = float(stats.kruskal(*frames).pvalue) if len(frames) >= 2 else 1.0
        prog_rows.append(
            {
                "program": pname,
                "n_genes_present": len(present),
                "genes_present": ",".join(present),
                "eta2_vs_niche": eta,
                "anova_p": p,
                "kruskal_p": kruskal_p,
                "mean": float(score.mean()),
            }
        )
    prog_df = pd.DataFrame(prog_rows).sort_values("eta2_vs_niche", ascending=False)
    # BH on kruskal
    if len(prog_df):
        prog_df["kruskal_q"] = multipletests(prog_df["kruskal_p"], method="fdr_bh")[1]
    prog_df.to_csv(out_dir / "program_eta2.csv", index=False)
    prog_by_niche = pd.DataFrame(prog_scores)
    prog_by_niche["niche"] = labels
    mean_prog = prog_by_niche.groupby("niche").mean(numeric_only=True)
    mean_prog.to_csv(out_dir / "program_means_by_niche.csv")

    # cycling classification: top quartile of cell_cycle score within cohort
    cycle = prog_scores.get("cell_cycle", np.zeros(ad.n_obs))
    thr = float(np.quantile(cycle, 0.75)) if len(cycle) else 0.0
    cycling = cycle >= thr
    cycle_tbl = (
        pd.DataFrame({"niche": labels, "cycling": cycling.astype(int), "cycle_score": cycle})
        .groupby("niche")
        .agg(n=("cycling", "size"), n_cycling=("cycling", "sum"), mean_cycle=("cycle_score", "mean"))
    )
    cycle_tbl["frac_cycling"] = cycle_tbl["n_cycling"] / cycle_tbl["n"]
    cycle_tbl.to_csv(out_dir / "cycling_by_niche.csv")

    # DE + pathways
    markers = niche_deg(ad, labels, top_n=75)
    pd.DataFrame(
        [{"niche": k, "n_markers": len(v), "markers": ",".join(v)} for k, v in markers.items()]
    ).to_csv(out_dir / "niche_markers.csv", index=False)
    enr = enrich_gene_lists(markers)
    ora = decoupler_hallmark_ora(markers, background=list(ad.var_names.astype(str)))
    enr_all = pd.concat([enr, ora], ignore_index=True) if len(ora) else enr
    enr_all.to_csv(out_dir / "pathway_enrichment.csv", index=False)

    # ---- plots ----
    cats = sorted(set(labels), key=lambda x: (len(x), x))
    rng = np.random.default_rng(0)
    cols = plt.cm.tab20(np.linspace(0, 1, max(len(cats), 20)))[: len(cats)]
    cmap = {c: cols[i] for i, c in enumerate(cats)}

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    for c in cats:
        m = labels == c
        ax.scatter(spatial[m, 0], spatial[m, 1], s=8, c=[cmap[c]], linewidths=0, label=c, alpha=0.9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{name}: microniches (k={len(cats)})")
    ax.legend(title="niche", markerscale=2, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(img / "spatial_niches.png", bbox_inches="tight", facecolor="white")
    plt.close()

    # contact focus ordered bar
    if contact_focus in nhood_mean.columns:
        ord_ = nhood_mean[contact_focus].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        ax.bar(np.arange(len(ord_)), ord_.to_numpy(), color="#0b6e63")
        ax.set_xticks(np.arange(len(ord_)))
        ax.set_xticklabels(ord_.index.astype(str), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"mean {contact_focus} neighbor frac")
        eta = contact_eta.get(contact_focus, {}).get("eta2", np.nan)
        ax.set_title(f"{name}: {contact_focus} contact by niche (η²={eta:.3f})")
        fig.tight_layout()
        fig.savefig(img / "contact_by_niche.png", bbox_inches="tight", facecolor="white")
        plt.close()

        fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=150)
        show_cols = [c for c in [contact_focus, "FDC", "B_germinal_center", "T_CD4", "plasma"] if c in nhood_mean.columns]
        mat = nhood_mean[show_cols].loc[ord_.index]
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(show_cols)))
        ax.set_xticklabels(show_cols, rotation=30, ha="right")
        ax.set_yticks(range(len(mat)))
        ax.set_yticklabels(mat.index.astype(str), fontsize=8)
        ax.set_title(f"{name}: neighbor composition")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(img / "neighbor_heatmap.png", bbox_inches="tight", facecolor="white")
        plt.close()

    # cycling vs contact scatter at niche level
    if contact_focus in nhood_mean.columns and "cell_cycle" in mean_prog.columns:
        fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=150)
        x = nhood_mean.loc[mean_prog.index, contact_focus]
        y = mean_prog["cell_cycle"]
        ax.scatter(x, y, s=cycle_tbl.loc[mean_prog.index, "n"] * 0.4, c="#0b6e63", alpha=0.85)
        for ni in mean_prog.index:
            ax.text(x.loc[ni], y.loc[ni], str(ni), fontsize=7, ha="center", va="bottom")
        ax.set_xlabel(f"mean {contact_focus} neighbor fraction")
        ax.set_ylabel("mean cell-cycle module score")
        ax.set_title(f"{name}: cycling vs contact")
        fig.tight_layout()
        fig.savefig(img / "cycling_vs_contact.png", bbox_inches="tight", facecolor="white")
        plt.close()

    # program heatmap
    if len(mean_prog.columns):
        mean_z = (mean_prog - mean_prog.mean()) / mean_prog.std(ddof=0).replace(0, np.nan)
        mean_z = mean_z.fillna(0)
        order_key = contact_focus if contact_focus in nhood_mean.columns else mean_prog.columns[0]
        if contact_focus in nhood_mean.columns:
            order = nhood_mean[contact_focus].sort_values(ascending=False).index
            mean_z = mean_z.reindex(order)
        fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * len(mean_z))), dpi=150)
        im = ax.imshow(mean_z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(mean_z.shape[1]))
        ax.set_xticklabels(mean_z.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(mean_z.shape[0]))
        ax.set_yticklabels(mean_z.index.astype(str), fontsize=8)
        ax.set_title(f"{name}: program specialization (z)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(img / "program_by_niche.png", bbox_inches="tight", facecolor="white")
        plt.close()

    # enrichr top
    if len(enr_all) and enr_all["adj_p"].notna().any():
        top = (
            enr_all.dropna(subset=["adj_p"])
            .sort_values("adj_p")
            .groupby("niche", as_index=False)
            .head(2)
            .head(24)
        )
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.32 * len(top))), dpi=150)
        y = np.arange(len(top))
        ax.barh(y, -np.log10(top["adj_p"].clip(lower=1e-20)), color="#264653")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"n{r.niche}: {str(r.term)[:58]}" for r in top.itertuples()], fontsize=7
        )
        ax.invert_yaxis()
        ax.set_xlabel("-log10(adj p)")
        ax.set_title(f"{name}: pathway enrichment from niche DEGs")
        fig.tight_layout()
        fig.savefig(img / "pathways_top.png", bbox_inches="tight", facecolor="white")
        plt.close()

    # cycling spatial
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    sca = ax.scatter(spatial[:, 0], spatial[:, 1], c=cycle, s=8, cmap="magma", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name}: cell-cycle module")
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(img / "cycle_spatial.png", bbox_inches="tight", facecolor="white")
    plt.close()

    result = {
        "name": name,
        "n_cells": int(len(labels)),
        "n_niches": int(len(cats)),
        "cli_summary": summary,
        "contact_focus": contact_focus,
        "contact_eta2": contact_eta.get(contact_focus, {}),
        "top_programs": prog_df.head(8).to_dict(orient="records"),
        "n_enrichment_rows": int(len(enr_all)),
        "cycling_threshold_q75": thr,
        "mean_frac_cycling": float(cycle_tbl["frac_cycling"].mean()) if len(cycle_tbl) else 0.0,
        "niches_by_contact": nhood_mean[contact_focus].sort_values(ascending=False).to_dict()
        if contact_focus in nhood_mean.columns
        else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    full = sc.read_h5ad(H5AD)

    gc_labels = Path(os.environ.get("GC_LABELS", "/tmp/tonsil_gc_microniches_full/microniche_labels.csv"))
    tfh_labels = Path(
        os.environ.get("TFH_LABELS", "/tmp/tonsil_tfh_microniches_full/microniche_labels.csv")
    )
    gc_sum = gc_labels.parent / "summary.json"
    tfh_sum = tfh_labels.parent / "summary.json"

    neighbor_types = [
        "T_follicular_helper",
        "FDC",
        "B_germinal_center",
        "T_CD4",
        "T_CD8",
        "plasma",
        "B_naive",
        "B_memory",
        "mDC",
    ]

    results = {}
    if gc_labels.is_file():
        print("analyzing GC…", flush=True)
        results["gc"] = analyze_cohort(
            "gc",
            gc_labels,
            gc_sum,
            full,
            PROGRAMS,
            neighbor_types,
            contact_focus="T_follicular_helper",
        )
    else:
        print("missing GC labels", gc_labels)

    if tfh_labels.is_file():
        print("analyzing Tfh…", flush=True)
        results["tfh"] = analyze_cohort(
            "tfh",
            tfh_labels,
            tfh_sum,
            full,
            TFH_PROGRAMS,
            neighbor_types,
            contact_focus="B_germinal_center",
        )
    else:
        print("missing Tfh labels", tfh_labels)

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "top_programs"} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
