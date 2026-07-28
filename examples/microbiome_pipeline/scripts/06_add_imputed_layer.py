#!/usr/bin/env python3
"""Independent host / bacterial spatial imputation for microbiome SpaceTravLR runs.

Host and bacteria are never mixed:
  - Host: knn-fill zeros within each host cell_type (cluster label).
  - Bacteria: aggregate unmap bins → colony×genus, label each colony by dominant
    genus, then knn-fill genus abundances within each bacterial label only.
    Signal amounts are rebuilt from imputed genera × taxon priors.

Writes:
  processed/GSM9456850_tumor_cells_imputed.h5ad  (layer imputed_count)
  processed/GSM9456850_bact_senders_colony25um.parquet
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

BASE = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome")
RUST = Path("/ix1/ylee/kor11/tools/SpaceTravLR_rust")
HOST = BASE / "processed/GSM9456850_tumor_cells_spacetravlr_ready.h5ad"
UNMAP = BASE / "raw/stereoseq_tumor/GSM9456850_A612_unmap.h5ad"
INTER = RUST / "data/microbial/bact_host_interactions.v0.csv"
PRIORS = RUST / "data/microbial/taxon_signal_priors.v0.csv"
HOST_OUT = BASE / "processed/GSM9456850_tumor_cells_imputed.h5ad"
BACT_OUT = BASE / "processed/GSM9456850_bact_senders_colony25um.parquet"

K_HOST = 15
K_BACT = 10
PIX_PER_UM = 2.0
GRID_UM = 25.0
MIN_GENUS_TOTAL = 50.0
MIN_BIN_UMI = 2.0


def neighbor_index_by_label(xy: np.ndarray, labels: np.ndarray, k: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each label → (row indices into full array, local knn index matrix)."""
    labs = np.asarray(labels).astype(str)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lab in pd.unique(labs):
        idx = np.flatnonzero(labs == lab)
        n = idx.size
        kk = min(k, max(n - 1, 0))
        if kk < 1:
            out[lab] = (idx, np.zeros((n, 0), dtype=np.int64))
            continue
        nn = NearestNeighbors(n_neighbors=kk + 1, algorithm="kd_tree").fit(xy[idx])
        ind = nn.kneighbors(xy[idx], return_distance=False)[:, 1:]
        # neighbors must remain inside the label
        assert np.all(labs[idx][ind] == lab)
        out[lab] = (idx, ind)
    return out


def fill_zeros_with_precomputed(
    X: np.ndarray,
    by_label: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, int]:
    out = np.asarray(X, dtype=np.float32).copy()
    fills = 0
    for idx, ind in by_label.values():
        if ind.shape[1] == 0 or idx.size == 0:
            continue
        block = out[idx]
        for j in range(block.shape[1]):
            col = block[:, j]
            zero = col <= 0
            if not zero.any() or zero.all():
                continue
            neigh = col[ind]
            pos = neigh > 0
            counts = pos.sum(1)
            ok = counts > 0
            if not ok.any():
                continue
            fill = np.zeros_like(col)
            fill[ok] = (neigh * pos).sum(1)[ok] / counts[ok]
            take = zero & ok
            if not take.any():
                continue
            col = col.copy()
            col[take] = fill[take]
            block[:, j] = col
            fills += int(take.sum())
        out[idx] = block
    return out, fills


def impute_host() -> None:
    print("[host] load", HOST, flush=True)
    adata = ad.read_h5ad(HOST)
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    present = np.asarray(X.sum(0)).ravel() > 0
    gene_idx = np.where(present)[0]
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    print(
        f"[host] cells={adata.n_obs} genes={len(gene_idx)} "
        f"labels={len(np.unique(labels))} (within cell_type only)",
        flush=True,
    )
    by_label = neighbor_index_by_label(xy, labels, K_HOST)
    for lab, (idx, ind) in by_label.items():
        print(f"  [host] {lab!r}: n={idx.size} k={ind.shape[1]}", flush=True)

    # Build imputed sparse matrix in chunks (avoid giant dense + slow lil column writes).
    chunk = 500
    total_fills = 0
    blocks: list[sparse.csr_matrix] = []
    for start in range(0, len(gene_idx), chunk):
        cols = gene_idx[start : start + chunk]
        block = np.asarray(X[:, cols].todense(), dtype=np.float32)
        filled, nfill = fill_zeros_with_precomputed(block, by_label)
        blocks.append(sparse.csr_matrix(filled))
        total_fills += nfill
        print(
            f"[host] genes {min(start + chunk, len(gene_idx))}/{len(gene_idx)} fills={nfill}",
            flush=True,
        )
        del block, filled

    imp_present = sparse.hstack(blocks, format="csr")
    del blocks
    # Scatter imputed present-gene columns back into full gene space.
    Ximp = X.tocsc(copy=True).astype(np.float32)
    Ximp[:, gene_idx] = imp_present
    adata.layers["imputed_count"] = Ximp.tocsr()
    del Ximp, imp_present
    adata.obs["cluster"] = adata.obs["cell_type"].astype("category").cat.codes.astype(str)
    adata.uns["imputation"] = {
        "mode": "host_within_cell_type",
        "k": K_HOST,
        "label_col": "cell_type",
        "independent_of_bacteria": True,
        "zero_fills": int(total_fills),
    }
    HOST_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = HOST_OUT.with_suffix(".tmp.h5ad")
    adata.write_h5ad(tmp, compression="gzip")
    tmp.replace(HOST_OUT)
    print("[host] wrote", HOST_OUT, "total_fills", total_fills, flush=True)


def impute_bacteria() -> None:
    print("[bact] load", UNMAP, flush=True)
    interactions = pd.read_csv(INTER)
    priors = pd.read_csv(PRIORS)
    signals = sorted(interactions["signal_id"].unique())

    m = ad.read_h5ad(UNMAP)
    mv = m.var
    keep = (
        (mv["superkingdom"] == "Bacteria")
        & mv["genus"].notna()
        & (~mv["genus"].isin(["Mus", "Homo"]))
    )
    m = m[:, keep].copy()
    genera_per_feat = m.var["genus"].astype(str)
    uniq = pd.Index(sorted(genera_per_feat.unique()))
    gmap = {g: i for i, g in enumerate(uniq)}
    feat_to_genus = np.array([gmap[g] for g in genera_per_feat])
    G = sparse.csr_matrix(
        (np.ones(m.n_vars), (np.arange(m.n_vars), feat_to_genus)),
        shape=(m.n_vars, len(uniq)),
    )
    M = m.X.tocsr() @ G
    xy_um = np.asarray(m.obsm["spatial"], dtype=np.float64) / PIX_PER_UM
    umi = np.asarray(M.sum(1)).ravel()
    use = umi >= MIN_BIN_UMI
    xy_um, M = xy_um[use], M[use]
    print(f"[bact] bins_used={M.shape[0]} genera_raw={len(uniq)}", flush=True)

    gtot = np.asarray(M.sum(0)).ravel()
    keep_g = gtot >= MIN_GENUS_TOTAL
    M = M[:, keep_g]
    uniq = uniq[keep_g]
    gmap = {g: i for i, g in enumerate(uniq)}
    print(f"[bact] genera_kept (>={MIN_GENUS_TOTAL} UMI): {len(uniq)}", flush=True)

    gx = np.floor(xy_um[:, 0] / GRID_UM).astype(np.int64)
    gy = np.floor(xy_um[:, 1] / GRID_UM).astype(np.int64)
    keys = gx * 10_000_000 + gy
    key_codes, _ = pd.factorize(keys, sort=False)
    n_col = int(key_codes.max()) + 1

    coo = M.tocoo()
    colony_genus = sparse.coo_matrix(
        (coo.data, (key_codes[coo.row], coo.col)), shape=(n_col, M.shape[1])
    ).tocsr()

    sum_xy = np.zeros((n_col, 2), dtype=np.float64)
    cnt = np.zeros(n_col, dtype=np.float64)
    np.add.at(sum_xy[:, 0], key_codes, xy_um[:, 0])
    np.add.at(sum_xy[:, 1], key_codes, xy_um[:, 1])
    np.add.at(cnt, key_codes, 1.0)
    nonempty = cnt > 0
    xy_c = np.zeros((n_col, 2), dtype=np.float64)
    xy_c[nonempty] = sum_xy[nonempty] / cnt[nonempty, None]

    tot = np.asarray(colony_genus.sum(1)).ravel()
    keep_c = tot > 0
    colony_genus = colony_genus[keep_c]
    xy_c = xy_c[keep_c]

    dens = np.asarray(colony_genus.todense(), dtype=np.float32)
    dom_i = dens.argmax(1)
    bact_labels = uniq.to_numpy()[dom_i]
    print(
        f"[bact] colonies={dens.shape[0]} bacterial_labels={len(np.unique(bact_labels))}",
        flush=True,
    )
    print("[bact] top labels", pd.Series(bact_labels).value_counts().head(10).to_dict(), flush=True)

    by_label = neighbor_index_by_label(xy_c, bact_labels, K_BACT)
    for lab, (idx, ind) in list(by_label.items())[:12]:
        print(f"  [bact] {lab!r}: n={idx.size} k={ind.shape[1]}", flush=True)

    dens_imp, nfill = fill_zeros_with_precomputed(dens, by_label)
    print(f"[bact] genus zero-fills={nfill}", flush=True)

    # Mass per bacterial label can only increase inside that label's colonies
    for lab in pd.Series(bact_labels).value_counts().head(5).index:
        mask = bact_labels == lab
        before = float(dens[mask].sum())
        after = float(dens_imp[mask].sum())
        other_before = float(dens[~mask].sum())
        other_after = float(dens_imp[~mask].sum())
        assert after + 1e-3 >= before
        print(
            f"[bact] label={lab!r} sum {before:.1f}->{after:.1f}; "
            f"other_labels {other_before:.1f}->{other_after:.1f}",
            flush=True,
        )

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

    E = np.zeros((len(uniq), len(signals)), dtype=np.float64)
    sig_index = {s: i for i, s in enumerate(signals)}

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

    A = dens_imp.astype(np.float64) @ E
    df = pd.DataFrame(
        {
            "sender_id": [f"c{i}" for i in range(len(A))],
            "x": xy_c[:, 0],
            "y": xy_c[:, 1],
            "bact_label": bact_labels,
            "dominant_genus_umi": dens_imp[np.arange(len(dens_imp)), dom_i],
        }
    )
    for j, s in enumerate(signals):
        df[s] = A[:, j]
    df = df.loc[df[signals].sum(1) > 0].copy()
    BACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BACT_OUT, index=False)
    print(
        f"[bact] wrote {BACT_OUT} colonies={len(df)} signals={len(signals)} "
        f"(independent_of_host=True, label=dominant_genus)",
        flush=True,
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--bact-only", action="store_true")
    args = ap.parse_args()
    if args.host_only and args.bact_only:
        raise SystemExit("pick at most one of --host-only / --bact-only")
    if not args.bact_only:
        impute_host()
    if not args.host_only:
        impute_bacteria()
    print("[done] host and bacterial imputation finished independently", flush=True)


if __name__ == "__main__":
    main()
