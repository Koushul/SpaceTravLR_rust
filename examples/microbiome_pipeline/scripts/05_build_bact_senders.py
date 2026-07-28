#!/usr/bin/env python3
"""Build bacterial colony sender parquet for SpaceTravLR [microbial].sender_table.

Prefer `06_add_imputed_layer.py`, which imputes bacterial genus abundances
within dominant-genus labels before writing the same parquet path. This script
keeps a raw (no-impute) rebuild path for debugging.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

BASE = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome")
RUST = Path("/ix1/ylee/kor11/tools/SpaceTravLR_rust")
UNMAP = BASE / "raw/stereoseq_tumor/GSM9456850_A612_unmap.h5ad"
INTER = RUST / "data/microbial/bact_host_interactions.v0.csv"
PRIORS = RUST / "data/microbial/taxon_signal_priors.v0.csv"
OUT = BASE / "processed/GSM9456850_bact_senders_colony25um.parquet"
PIX_PER_UM = 2.0
GRID_UM = 25.0


def main():
    interactions = pd.read_csv(INTER)
    priors = pd.read_csv(PRIORS)
    signals = sorted(interactions["signal_id"].unique())

    print("[load]", UNMAP)
    m = ad.read_h5ad(UNMAP)
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
    use = umi >= 2.0
    xy_um, M = xy_um[use], M[use]

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
    gx = np.floor(xy_um[:, 0] / GRID_UM).astype(np.int64)
    gy = np.floor(xy_um[:, 1] / GRID_UM).astype(np.int64)
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
    agg.insert(0, "sender_id", [f"c{i}" for i in range(len(agg))])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)
    print(f"[wrote] {OUT} colonies={len(agg)} signals={len(signals)}")


if __name__ == "__main__":
    main()
