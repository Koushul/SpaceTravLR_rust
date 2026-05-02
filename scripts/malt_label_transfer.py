"""
Marker-Aware Label Transfer (MALT)

Optimized label transfer from reference scRNA-seq to query that minimizes
the difference in marker/DEG expression between reference and predicted labels.

Loss = alpha_p * L_profile + alpha_c * L_cell + alpha_k * L_knn + alpha_e * L_entropy
        + alpha_m * L_manifold

- L_profile: MSE between per-type marker profiles in ref vs query
- L_cell: per-cell negative log-likelihood under type-specific marker distributions
- L_knn: KL divergence from initial KNN (embedding structure prior)
- L_entropy: encourages confident single-type assignments
- L_manifold: weighted match of inter-type distances in ref vs soft query centroids
  (``X_umap`` on both sides when dims match, else aligned PCA used for KNN)

CLI:

  spacetravlr --map-labels --reference ref.h5ad --query q.h5ad --map-labels-outdir ./malt_out

  Or: python scripts/malt_label_transfer.py --reference ref.h5ad --query q.h5ad --outdir ./malt_out

  Query path may omit ``.h5ad`` (``-q /tmp/e14s`` resolves ``/tmp/e14s.h5ad``).
  If no ``--groupby`` / ``-g`` is passed, the first matching column among common names is used
  (e.g. ``cell_type``, ``final_annotation``).   Pass ``-g`` once with comma-separated names (e.g. ``-g cell_type,cell_type_fine``) and/or
  ``-g`` multiple times to run MALT once per column; with multiple groupings, obs columns
  and ``malt_labels.csv`` use suffixes derived
  from the column name (e.g. ``malt_label_cell_type``, ``malt_label_annotation``). A single
  explicit or inferred grouping keeps the legacy names ``malt_label``, ``malt_confidence``,
  ``knn_label``. Legacy h5ad files whose ``var`` is a 1D dataset of gene names are repacked
  on read for anndata 0.12+.

  Writes ``malt_labels.csv`` with ``obs_name`` (``adata.obs_names``) as the row index and
  all MALT output columns for every grouping in one file.

  Optional expression handling (see --expression-mode, --counts-layer, --prefer-raw-counts).

  After MALT, by default runs adaptive Leiden on the query (``obs['leiden']``) guided by
  ``malt_label`` purity, then ``obs['leiden_celltype']`` from optimizing cluster→type mapping
  vs reference dotplots. Pass ``--no-leiden-map`` to skip.

  AnnData read quirks: ``uns['log1p']`` entries with legacy ``encoding-type=null`` are stripped
  when needed. If reference ``var_names`` are numeric placeholders (no overlap with the query),
  pass ``--reference-gene-list`` (one symbol per line, length = reference ``n_vars``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import tempfile
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


_GROUPBY_FALLBACKS = (
    "cell_type",
    "final_annotation",
    "subcluster_annotation",
    "annotation",
    "predicted_cell_type",
    "celltype",
)


def _resolve_h5ad_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    alt = f"{path}.h5ad" if not path.endswith(".h5ad") else path
    if os.path.isfile(alt):
        return alt
    tried = repr(path) if alt == path else f"{path!r} and {alt!r}"
    raise FileNotFoundError(f"No such h5ad file (tried {tried})")


def _h5ad_attr_str(val) -> str:
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _h5ad_needs_compat_patch(path: str) -> tuple[bool, bool]:
    with h5py.File(path, "r") as f:
        legacy_var = "var" in f and isinstance(f["var"], h5py.Dataset)
        bad_log1p = False
        if "uns" in f and "log1p" in f["uns"]:
            lg = f["uns"]["log1p"]
            if isinstance(lg, h5py.Group):
                for key in lg.keys():
                    ds = lg[key]
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    et = _h5ad_attr_str(ds.attrs.get("encoding-type", "")).lower()
                    if et == "null":
                        bad_log1p = True
                        break
        return legacy_var, bad_log1p


def read_h5ad_compat(path: str) -> sc.AnnData:
    path = os.path.abspath(path)
    legacy_var, bad_log1p = _h5ad_needs_compat_patch(path)
    if not legacy_var and not bad_log1p:
        return sc.read_h5ad(path)

    if legacy_var:
        print(
            f"  [read_h5ad_compat] legacy var layout in {os.path.basename(path)}; "
            "repacked var as dataframe (var/_index) for anndata reader."
        )
    if bad_log1p:
        print(
            f"  [read_h5ad_compat] stripping uns['log1p'] with null-encoded fields in "
            f"{os.path.basename(path)} (incompatible with this anndata reader)."
        )

    str_dt = h5py.string_dtype(encoding="utf-8")
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(path, tmp_path)
    try:
        with h5py.File(tmp_path, "r+") as f:
            if bad_log1p and "uns" in f and "log1p" in f["uns"]:
                del f["uns"]["log1p"]
            if legacy_var:
                raw = f["var"][()]
                decoded = np.array(
                    [
                        x.decode("utf-8")
                        if isinstance(x, (bytes, bytearray))
                        else str(x)
                        for x in raw
                    ],
                    dtype=object,
                )
                del f["var"]
                vg = f.create_group("var")
                idx = vg.create_dataset("_index", data=np.asarray(decoded, dtype=str_dt))
                vg.attrs.create("column-order", data=np.array([], dtype=str_dt))
                vg.attrs["_index"] = "_index"
                vg.attrs["encoding-type"] = "dataframe"
                vg.attrs["encoding-version"] = "0.2.0"
                idx.attrs["encoding-type"] = "string-array"
                idx.attrs["encoding-version"] = "0.2.0"
        return sc.read_h5ad(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _placeholder_var_names_ratio(var_names) -> float:
    n = int(len(var_names))
    if n == 0:
        return 0.0
    k = sum(1 for x in var_names if str(x).isdigit())
    return float(k) / float(n)


def _read_gene_symbols_one_per_line(list_path: str) -> list[str]:
    list_path = os.path.abspath(list_path)
    lines: list[str] = []
    with open(list_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            lines.append(t.split()[0])
    return lines


def _apply_reference_gene_list_inplace(adata: sc.AnnData, list_path: str) -> None:
    lines = _read_gene_symbols_one_per_line(list_path)
    n_var = int(adata.shape[1])
    if len(lines) != n_var:
        raise ValueError(
            f"--reference-gene-list {list_path!r}: expected {n_var} lines, got {len(lines)}"
        )
    if len(set(lines)) != len(lines):
        raise ValueError(
            f"--reference-gene-list {list_path!r}: duplicate symbols are not allowed "
            f"({len(lines)} lines, {len(set(lines))} unique)"
        )
    adata.var_names = lines
    print(
        f"  [--reference-gene-list] set reference var_names from {os.path.basename(list_path)!r}"
    )


def _resolve_groupby(ref: sc.AnnData, groupby: str | None) -> str:
    if groupby is not None:
        if groupby not in ref.obs.columns:
            raise KeyError(
                f"groupby column {groupby!r} not in ref.obs; available: {list(ref.obs.columns)}"
            )
        return groupby
    for cand in _GROUPBY_FALLBACKS:
        if cand in ref.obs.columns:
            print(f"  [--groupby] inferred {cand!r} (first match in fallback list)")
            return cand
    raise KeyError(
        "No --groupby given and no default column found in ref.obs. "
        f"Pass -g <column>. Available: {list(ref.obs.columns)}"
    )


def _slug_for_suffix(col: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col)).strip("_")
    if not s:
        return "group"
    if s[0].isdigit():
        return f"g_{s}"
    return s


def _dedupe_preserve_groupby(columns: list[str | None]) -> list[str | None]:
    seen: set[str] = set()
    out: list[str | None] = []
    for c in columns:
        key = "__none__" if c is None else c
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _dotplot_set_title(dp, title: str) -> None:
    fig = getattr(dp, "fig", None)
    if fig is not None:
        fig.suptitle(title, y=1.02, fontsize=12)
    else:
        plt.gcf().suptitle(title, y=1.02, fontsize=12)


def _is_categorical_series(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype)


def _leiden_label_sort_key(x: str) -> tuple:
    xs = str(x)
    if "_u" in xs:
        base, rest = xs.split("_u", 1)
        try:
            return (int(base), int(rest))
        except ValueError:
            return (0, 0)
    try:
        return (int(xs), 0)
    except ValueError:
        return (10**9, hash(xs) % (10**6))


def clean_leiden(
    adata: sc.AnnData,
    old_col: str = "leiden_R",
    new_col: str = "leiden",
    key: str = "leiden",
) -> None:
    if old_col in adata.obs.columns:
        adata.obs[new_col] = adata.obs[old_col].copy()
        adata.obs.drop(columns=[old_col], inplace=True)
    if not _is_categorical_series(adata.obs[key]):
        adata.obs[key] = adata.obs[key].astype(str)
    cats = sorted(adata.obs[key].astype(str).unique(), key=_leiden_label_sort_key)
    adata.obs[key] = pd.Categorical(adata.obs[key].astype(str), categories=cats)
    new_cats = [str(i) for i in range(len(cats))]
    adata.obs[key] = adata.obs[key].cat.rename_categories(
        dict(zip(cats, new_cats))
    ).astype("category")
    for color_key in (f"{key}_colors",):
        adata.uns.pop(color_key, None)


def _leiden_n_unique(adata: sc.AnnData, col: str) -> int:
    return int(adata.obs[col].astype(str).nunique())


def _leiden_at_resolution(
    adata: sc.AnnData, resolution: float, key_added: str
) -> int:
    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
        )
    except Exception:
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added)
    return _leiden_n_unique(adata, key_added)


def _binary_search_leiden_resolution(
    adata: sc.AnnData,
    target_clusters: int,
    key_tmp: str = "leiden_R",
    lo: float = 0.05,
    hi: float = 8.0,
    max_iter: int = 22,
) -> float:
    target_clusters = max(2, int(target_clusters))
    best_r, best_dist = 0.5, 10**9
    lo, hi = float(lo), float(hi)
    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        n = _leiden_at_resolution(adata, mid, key_tmp)
        dist = abs(n - target_clusters)
        if dist < best_dist:
            best_dist, best_r = dist, mid
        if n < target_clusters:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    _leiden_at_resolution(adata, best_r, key_tmp)
    return best_r


def _strip_diffmap_before_fresh_neighbors(adata: sc.AnnData) -> None:
    if "X_diffmap" in adata.obsm:
        adata.obsm.pop("X_diffmap", None)
    if "X_diffmap0" in adata.obs.columns:
        adata.obs.drop(columns=["X_diffmap0"], inplace=True)
    adata.uns.pop("diffmap_evals", None)
    adata.uns.pop("diffmap_params", None)


def _ensure_query_embedding_neighbors(
    adata: sc.AnnData,
    *,
    n_top_genes: int = 800,
    n_pcs: int = 40,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> None:
    _strip_diffmap_before_fresh_neighbors(adata)
    n_var = int(adata.shape[1])
    sc.pp.highly_variable_genes(
        adata, n_top_genes=min(n_top_genes, n_var), subset=False
    )
    nhvg = int(adata.var["highly_variable"].sum())
    if nhvg < 3:
        nhvg = min(n_var, n_top_genes)
    n_pcs_use = min(n_pcs, nhvg - 1, adata.n_obs - 1)
    n_pcs_use = max(2, n_pcs_use)
    sc.tl.pca(adata, n_comps=n_pcs_use, svd_solver="arpack", random_state=random_state)
    sc.pp.neighbors(
        adata,
        n_neighbors=min(n_neighbors, max(5, adata.n_obs - 1)),
        n_pcs=n_pcs_use,
        random_state=random_state,
    )


def _malt_types_above_min(
    malt: np.ndarray, cell_types: list[str], min_cells: int
) -> list[str]:
    vc = pd.Series(malt).astype(str).value_counts()
    out = [c for c in cell_types if vc.get(str(c), 0) >= min_cells]
    return out


def _impure_clusters(
    leiden: np.ndarray,
    malt: np.ndarray,
    *,
    purity_threshold: float,
    min_cells: int,
) -> list[str]:
    leid_s = np.asarray(leiden).astype(str)
    malt_s = np.asarray(malt).astype(str)
    impure: list[str] = []
    for cid in np.unique(leid_s):
        m = leid_s == cid
        if m.sum() < min_cells * 2:
            continue
        sub = malt_s[m]
        vc = pd.Series(sub).value_counts()
        major = int(vc.iloc[0]) if len(vc) else 0
        purity = float(major) / float(m.sum()) if m.sum() else 0.0
        n_sig = int((vc >= min_cells).sum())
        if purity < purity_threshold and n_sig >= 2:
            impure.append(str(cid))
    return impure


def adaptive_leiden_clustering(
    adata: sc.AnnData,
    malt_col: str,
    *,
    leiden_key: str = "leiden",
    work_col: str = "leiden_R",
    purity_threshold: float = 0.65,
    min_cells: int = 10,
    max_rounds: int = 3,
    n_neighbors: int = 15,
    n_pcs: int = 40,
    random_state: int = 42,
) -> dict:
    malt = adata.obs[malt_col].astype(str).values
    all_types = sorted(np.unique(malt).tolist())
    types_ok = _malt_types_above_min(malt, all_types, min_cells)
    n_target = max(2, len(types_ok))
    target_coarse = max(2, n_target // 2)

    print(f"  [leiden] MALT types with n>={min_cells}: {n_target} -> target coarse clusters ~{target_coarse}")

    _ensure_query_embedding_neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=random_state,
    )

    _binary_search_leiden_resolution(adata, target_coarse, key_tmp=work_col)

    def _to_u0(x: str) -> str:
        xs = str(x)
        if "_u" in xs:
            return xs
        try:
            return f"{int(float(xs))}_u0"
        except ValueError:
            return f"{xs}_u0"

    adata.obs[work_col] = adata.obs[work_col].astype(str).map(_to_u0)
    clean_leiden(adata, old_col=work_col, new_col=leiden_key, key=leiden_key)

    round_info: list[dict] = []
    for rnd in range(max_rounds):
        leid = adata.obs[leiden_key].astype(str).values
        impure = _impure_clusters(
            leid, malt, purity_threshold=purity_threshold, min_cells=min_cells
        )
        round_info.append(
            {
                "round": rnd,
                "n_clusters": int(np.unique(leid).size),
                "impure": list(impure),
            }
        )
        print(
            f"  [leiden] round {rnd}: {_leiden_n_unique(adata, leiden_key)} clusters, "
            f"impure={len(impure)}"
        )
        if not impure:
            break

        labels = np.asarray(leid, dtype=object).copy()
        split_any = False
        for cid in sorted(impure, key=_leiden_label_sort_key):
            mask = labels == cid
            n_sub = int(mask.sum())
            if n_sub < 2 * min_cells:
                continue
            sub_malt = malt[mask]
            vc = pd.Series(sub_malt).astype(str).value_counts()
            n_types = int((vc >= min_cells).sum())
            if n_types < 2:
                continue
            k_aim = min(n_types, max(2, n_sub // max(8, min_cells)))
            sub = adata[mask].copy()
            if sub.n_obs < 10:
                continue
            _ensure_query_embedding_neighbors(
                sub,
                n_neighbors=min(n_neighbors, max(5, sub.n_obs - 1)),
                n_pcs=min(n_pcs, sub.n_obs - 2),
                random_state=random_state + rnd + hash(cid) % 997,
            )
            _binary_search_leiden_resolution(
                sub, k_aim, key_tmp=work_col, lo=0.05, hi=6.0
            )
            local = sub.obs[work_col].astype(str).values
            uniq_local = sorted(np.unique(local).tolist(), key=_leiden_label_sort_key)
            if len(uniq_local) < 2:
                continue
            parent_base = cid.split("_u")[0] if "_u" in cid else cid
            new_labels = np.asarray(labels, dtype=object)
            idxs = np.flatnonzero(mask)
            for li, lv in enumerate(local):
                gidx = idxs[li]
                new_labels[gidx] = f"{parent_base}_u{uniq_local.index(lv)}"
            labels = new_labels.astype(str)
            split_any = True

        if not split_any:
            print("  [leiden] no successful subclusters; stopping.")
            break

        adata.obs[work_col] = pd.Series(labels, index=adata.obs_names).astype(str)
        clean_leiden(adata, old_col=work_col, new_col=leiden_key, key=leiden_key)

    return {
        "n_target_malt_types": n_target,
        "target_coarse": target_coarse,
        "rounds": round_info,
        "final_n_clusters": _leiden_n_unique(adata, leiden_key),
    }


def optimize_cluster_mapping(
    adata: sc.AnnData,
    leiden_key: str,
    malt_col: str,
    cell_types: list[str],
    all_mk: list[str],
    mk_idx: np.ndarray,
    ref_rel: np.ndarray,
    mk_mask: np.ndarray,
    *,
    n_epochs: int = 300,
    lr: float = 0.08,
    alpha_dp: float = 15.0,
    alpha_malt: float = 0.5,
    alpha_sparse: float = 0.25,
    random_state: int = 42,
) -> dict:
    dev = torch.device("cpu")
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    leid = adata.obs[leiden_key].astype(str).values
    malt = adata.obs[malt_col].astype(str).values
    q_mk = np.asarray(adata.layers["ln"][:, mk_idx], dtype=np.float32)
    if sp.issparse(q_mk):
        q_mk = q_mk.toarray()

    cluster_ids = sorted(np.unique(leid).tolist(), key=_leiden_label_sort_key)
    n_c = len(cluster_ids)
    n_ct = len(cell_types)
    c2i = {c: i for i, c in enumerate(cell_types)}
    k2i = {k: i for i, k in enumerate(cluster_ids)}

    counts = np.zeros(n_c, dtype=np.float32)
    cluster_mean = np.zeros((n_c, q_mk.shape[1]), dtype=np.float32)
    emp = np.zeros((n_c, n_ct), dtype=np.float32) + 1e-6

    for k, cid in enumerate(cluster_ids):
        m = leid == cid
        counts[k] = float(m.sum())
        if m.sum() == 0:
            continue
        cluster_mean[k] = q_mk[m].mean(0)
        for t, val in zip(*np.unique(malt[m], return_counts=True)):
            ti = c2i.get(str(t))
            if ti is not None:
                emp[k, ti] += float(val)
        emp[k] /= emp[k].sum()

    cluster_mean_t = torch.tensor(cluster_mean, dtype=torch.float32, device=dev)
    counts_t = torch.tensor(counts, dtype=torch.float32, device=dev).unsqueeze(1)
    emp_t = torch.tensor(emp, dtype=torch.float32, device=dev)
    ref_rel_t = torch.tensor(ref_rel, dtype=torch.float32, device=dev)
    mmask_t = torch.tensor(mk_mask, dtype=torch.float32, device=dev)

    q_global_mean_t = torch.tensor(q_mk.mean(0), dtype=torch.float32, device=dev)
    q_global_std_t = torch.tensor(q_mk.std(0) + 1e-6, dtype=torch.float32, device=dev)
    n_mp = float(mmask_t.sum().item())
    n_mp = max(n_mp, 1.0)

    W = torch.log(emp_t.clamp_min(1e-6)).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    best_loss, best_W = float("inf"), None

    for ep in range(n_epochs):
        opt.zero_grad()
        P = F.softmax(W, dim=1)
        log_p = F.log_softmax(W, dim=1)
        Wc = P * counts_t
        den = Wc.sum(dim=0, keepdim=True).clamp_min(1e-6)
        qprof_raw = (Wc.T @ cluster_mean_t) / den.T
        qprof_rel = (qprof_raw - q_global_mean_t) / q_global_std_t
        Ldp = (((qprof_rel - ref_rel_t) ** 2) * mmask_t).sum() / n_mp
        Lmalt = (emp_t * (emp_t.clamp_min(1e-6).log() - log_p)).sum(dim=1).mean()
        row_ent = -(P * (P.clamp_min(1e-8).log())).sum(dim=1).mean()
        loss = alpha_dp * Ldp + alpha_malt * Lmalt + alpha_sparse * row_ent
        loss.backward()
        opt.step()
        lv = loss.item()
        if lv < best_loss:
            best_loss, best_W = lv, W.detach().clone()
        if ep % 80 == 0 or ep == n_epochs - 1:
            print(
                f"  [leiden-map] E{ep:3d} L={lv:.4f} dp={Ldp.item():.4f} "
                f"malt={Lmalt.item():.4f} ent={row_ent.item():.4f}"
            )

    P_final = F.softmax(best_W, dim=1).cpu().numpy()
    hard = P_final.argmax(axis=1)
    cluster_to_type = {cluster_ids[i]: cell_types[hard[i]] for i in range(n_c)}
    leiden_celltype = np.array([cluster_to_type[str(x)] for x in leid], dtype=object)
    adata.obs["leiden_celltype"] = pd.Series(
        leiden_celltype, index=adata.obs_names, dtype=str
    ).astype("category")

    rows = []
    for i, cid in enumerate(cluster_ids):
        rows.append(
            {
                "leiden_cluster": cid,
                "cell_type": cell_types[hard[i]],
                "n_cells": int(counts[i]),
                "soft_probs": {cell_types[t]: float(P_final[i, t]) for t in range(n_ct)},
            }
        )

    return {
        "best_loss": float(best_loss),
        "cluster_to_type": cluster_to_type,
        "mapping_rows": rows,
        "P_final": P_final,
    }


_PREFERRED_COUNT_LAYERS = (
    "raw_counts",
    "counts",
    "counts_raw",
    "X_counts",
    "umi",
    "umis",
)


def _as_matrix(X):
    if sp.issparse(X):
        return X.copy()
    return np.asarray(X)


def _dense_sample(X: np.ndarray | sp.spmatrix, rng: np.random.Generator, n_cells=400, n_genes=200):
    n_obs, n_var = X.shape
    nc = min(n_cells, n_obs)
    ng = min(n_genes, n_var)
    ri = rng.choice(n_obs, size=nc, replace=False)
    ci = rng.choice(n_var, size=ng, replace=False)
    sub = X[ri][:, ci]
    if sp.issparse(sub):
        sub = sub.toarray()
    return sub.astype(np.float64)


def _looks_log_normalized(X: np.ndarray | sp.spmatrix) -> bool:
    rng = np.random.default_rng(0)
    sub = _dense_sample(X, rng)
    flat = sub.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return False
    mx = float(flat.max())
    pos = flat[flat > 0]
    med_nz = float(np.median(pos)) if pos.size else 0.0
    if mx > 50:
        return False
    if mx <= 20 and med_nz < 10:
        return True
    int_like = np.mean(np.isclose(flat, np.round(flat), rtol=0, atol=1e-5))
    if mx > 200 and int_like > 0.85:
        return False
    if mx > 30 and med_nz > 15 and int_like > 0.8:
        return False
    if mx < 40 and med_nz < 10:
        return True
    log1p_10k = float(np.log1p(10_000.0))
    frac_in_log_range = float(np.mean(flat <= log1p_10k + 1e-6))
    if mx < 45 and med_nz < 12 and frac_in_log_range > 0.88:
        return True
    return mx < 35


def _raw_counts_aligned(adata: sc.AnnData):
    if adata.raw is None:
        return None
    want = list(adata.var_names)
    raw_names = adata.raw.var_names
    idx = raw_names.get_indexer(want)
    if (idx < 0).any():
        return None
    X = adata.raw.X[:, idx]
    return _as_matrix(X)


def _pick_counts_matrix(
    adata: sc.AnnData,
    counts_layer: str | None,
    prefer_raw_counts: bool,
) -> tuple[np.ndarray | sp.spmatrix | None, str | None]:
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise KeyError(
                f"{counts_layer!r} not in adata.layers; available: {list(adata.layers.keys())}"
            )
        return _as_matrix(adata.layers[counts_layer]), f"layers[{counts_layer!r}]"

    for k in _PREFERRED_COUNT_LAYERS:
        if k in adata.layers:
            return _as_matrix(adata.layers[k]), f"layers[{k!r}]"

    if prefer_raw_counts:
        Xr = _raw_counts_aligned(adata)
        if Xr is not None:
            return Xr, "raw.X (genes aligned to adata.var_names)"

    return None, None


def _strip_scanpy_log1p_uns(adata: sc.AnnData) -> None:
    """Remove Scanpy's log1p flag so pp.log1p does not warn after replacing X with counts."""
    adata.uns.pop("log1p", None)


def _normalize_from_counts(adata: sc.AnnData, counts_mat, name: str) -> None:
    """Replace .X with counts → normalize_total(1e4) → log1p, store in layers['ln']."""
    adata.layers["malt_counts_input"] = counts_mat.copy()
    adata.X = counts_mat.copy()
    _strip_scanpy_log1p_uns(adata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["ln"] = adata.X.copy()


def prepare_expression_inplace(
    adata: sc.AnnData,
    name: str,
    *,
    expression_mode: str = "auto",
    counts_layer: str | None = None,
    prefer_raw_counts: bool = False,
) -> dict:
    """
    Ensures adata.layers['ln'] contains log1p(normalize_total(counts, 1e4)).

    Tries hard to find raw counts even when .X looks log-normalized, so that
    both reference and query always go through the same normalization pipeline.

    expression_mode: 'auto' | 'counts' | 'lognorm'
    """
    mode = expression_mode.lower().strip()
    if mode not in ("auto", "counts", "lognorm"):
        raise ValueError(
            f"expression_mode must be auto|counts|lognorm; got {expression_mode!r}"
        )

    meta: dict = {"object": name, "expression_mode_requested": mode}

    if mode == "lognorm":
        meta["preprocess"] = "skip (forced log-normalized .X)"
        meta["counts_source"] = None
        if sp.issparse(adata.X):
            adata.layers["ln"] = adata.X.copy()
        else:
            adata.layers["ln"] = np.asarray(adata.X, dtype=np.float32).copy()
        print(
            f"  [{name}] expression_mode=lognorm: using .X as-is (no renormalization)."
        )
        return meta

    # --- Try to find counts (layers → raw → .X) ---
    counts_mat, counts_src = _pick_counts_matrix(
        adata, counts_layer, prefer_raw_counts
    )
    X0 = adata.X

    # In auto mode, also try .raw when .X looks log-normalized
    if counts_mat is None and mode in ("auto", "counts") and adata.raw is not None:
        Xr = _raw_counts_aligned(adata)
        if Xr is not None and not _looks_log_normalized(Xr):
            counts_mat, counts_src = Xr, "raw.X (aligned to var_names)"

    if counts_mat is not None:
        meta["counts_source"] = counts_src
        _normalize_from_counts(adata, counts_mat, name)
        meta["preprocess"] = "normalize_total(1e4) + log1p from counts"
        print(f"  [{name}] counts from {counts_src}; normalized + log1p.")
        return meta

    if mode == "counts":
        raise ValueError(
            f"[{name}] expression_mode=counts but no count matrix found. "
            f"Pass --counts-layer, add a layer in {_PREFERRED_COUNT_LAYERS}, "
            f"or use --prefer-raw-counts with .raw containing all var_names."
        )

    # --- auto: no counts found — check if .X is already log-normalized ---
    already_log = "log1p" in adata.uns or _looks_log_normalized(X0)
    reason = (
        "adata.uns['log1p'] present"
        if "log1p" in adata.uns
        else "heuristic (max/median)"
    )

    if already_log:
        meta["preprocess"] = f"skip ({reason} — .X treated as log-normalized)"
        meta["counts_source"] = None
        if sp.issparse(X0):
            adata.layers["ln"] = X0.copy()
        else:
            adata.layers["ln"] = np.asarray(X0, dtype=np.float32).copy()
        print(
            f"  [{name}] auto: .X looks log-normalized ({reason}); skipping normalization. "
            f"Pass --counts-layer to force re-normalization."
        )
        return meta

    # --- auto: .X looks like raw counts, no counts layer found ---
    meta["counts_source"] = ".X (auto: treated as counts)"
    counts_x = _as_matrix(X0)
    _normalize_from_counts(adata, counts_x, name)
    meta["preprocess"] = "normalize_total(1e4) + log1p from .X"
    print(
        f"  [{name}] auto: .X treated as raw counts; normalized + log1p."
    )
    return meta


def _malt_manifold_loss_tensors(
    ref_i: sc.AnnData,
    query: sc.AnnData,
    ref_li: np.ndarray,
    n_ct: int,
    rp: np.ndarray,
    qp: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, str]:
    if n_ct < 2:
        return None, None, None, "skip"
    ru = ref_i.obsm.get("X_umap")
    qu = query.obsm.get("X_umap")
    src = "pca"
    U_ref = np.asarray(rp, dtype=np.float32)
    U_q = np.asarray(qp, dtype=np.float32)
    if ru is not None and qu is not None:
        ru = np.asarray(ru, dtype=np.float32)
        qu = np.asarray(qu, dtype=np.float32)
        if (
            ru.shape[0] == ref_i.n_obs
            and qu.shape[0] == query.n_obs
            and ru.shape[1] == qu.shape[1]
            and ru.shape[1] >= 2
        ):
            U_ref, U_q = ru, qu
            src = "umap"
    d = int(U_ref.shape[1])
    C_ref = np.zeros((n_ct, d), dtype=np.float64)
    cnt = np.zeros(n_ct, dtype=np.float64)
    for i in range(int(ref_li.shape[0])):
        t = int(ref_li[i])
        if 0 <= t < n_ct:
            C_ref[t] += U_ref[i]
            cnt[t] += 1.0
    cnt = np.maximum(cnt, 1.0)
    C_ref = (C_ref / cnt[:, None]).astype(np.float32)
    diff = C_ref[:, np.newaxis, :] - C_ref[np.newaxis, :, :]
    dmat = np.sqrt(np.maximum((diff**2).sum(-1), 0.0) + 1e-16).astype(np.float32)
    tri_i, tri_j = np.triu_indices(n_ct, k=1)
    sub = dmat[tri_i, tri_j]
    if sub.size == 0:
        return None, None, None, "skip"
    sig = float(np.median(sub)) + 1e-4
    W = np.exp(-(dmat**2) / (2.0 * sig * sig)).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    return (
        torch.tensor(C_ref, dtype=torch.float32, device=device),
        torch.tensor(W, dtype=torch.float32, device=device),
        torch.tensor(U_q, dtype=torch.float32, device=device),
        src,
    )


def _safe_zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sig = np.nanstd(X, axis=0, keepdims=True)
    sig = np.where(sig < 1e-6, 1.0, sig)
    return np.nan_to_num((X - mu) / sig, copy=False).astype(np.float32)


def _resolve_spatial_xy(adata: sc.AnnData) -> tuple[str | None, np.ndarray | None]:
    for key in ("spatial", "X_spatial", "spatial_loc", "unscaled_spatial"):
        if key not in adata.obsm:
            continue
        arr = np.asarray(adata.obsm[key], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
            return key, arr[:, :2].astype(np.float32)
    return None, None


def select_dotplot_training_markers(
    ref_i: sc.AnnData,
    groupby_col: str,
    cell_types: list[str],
    shared: list[str],
    *,
    genes_per_type: int = 6,
) -> tuple[dict[str, list[str]], list[str], dict[str, list[dict]]]:
    ln = ref_i.layers["ln"]
    if sp.issparse(ln):
        ln = ln.toarray()
    ln = np.asarray(ln, dtype=np.float32)
    labels = ref_i.obs[groupby_col].astype(str).values
    global_mu = ln.mean(0)
    global_sd = ln.std(0) + 1e-6
    gene_to_i = {str(g): i for i, g in enumerate(ref_i.var_names)}
    ranked: dict[str, list[str]] = {}
    details: dict[str, list[dict]] = {}
    for ct in cell_types:
        m = labels == str(ct)
        if not m.any():
            ranked[ct], details[ct] = [], []
            continue
        rest = ~m
        in_mu = ln[m].mean(0)
        out_mu = ln[rest].mean(0) if rest.any() else global_mu
        pct_in = (ln[m] > 0).mean(0)
        pct_out = (ln[rest] > 0).mean(0) if rest.any() else np.zeros_like(pct_in)
        specificity = (in_mu - out_mu) / global_sd
        prevalence = np.sqrt(np.clip(pct_in, 0, 1)) * np.clip(pct_in - pct_out, 0, 1)
        score = specificity * (0.25 + prevalence) * (0.25 + np.log1p(np.maximum(in_mu, 0)))
        try:
            df = sc.get.rank_genes_groups_df(ref_i, group=ct)
            for _, r in df.head(160).iterrows():
                gi = gene_to_i.get(str(r["names"]))
                if gi is not None:
                    score[gi] += 0.3 * max(0.0, float(r.get("logfoldchanges", 0.0)))
        except Exception:
            pass
        chosen: list[str] = []
        chosen_vecs: list[np.ndarray] = []
        rows: list[dict] = []
        for gi in np.argsort(-score):
            g = str(ref_i.var_names[int(gi)])
            if g not in shared or not np.isfinite(score[int(gi)]) or score[int(gi)] <= 0:
                continue
            v = ln[:, int(gi)]
            redundant = False
            for prev in chosen_vecs:
                if np.std(v) > 1e-6 and np.std(prev) > 1e-6:
                    if abs(float(np.corrcoef(v, prev)[0, 1])) > 0.92:
                        redundant = True
                        break
            if redundant:
                continue
            chosen.append(g)
            chosen_vecs.append(v)
            rows.append({
                "gene": g,
                "score": float(score[int(gi)]),
                "specificity_z": float(specificity[int(gi)]),
                "pct_in": float(pct_in[int(gi)]),
                "pct_out": float(pct_out[int(gi)]),
            })
            if len(chosen) >= genes_per_type:
                break
        ranked[ct] = chosen
        details[ct] = rows
    flat: list[str] = []
    seen: set[str] = set()
    for ct in cell_types:
        for g in ranked.get(ct, []):
            if g not in seen:
                flat.append(g)
                seen.add(g)
    return ranked, flat, details


def _find_betadata_file(betadata_dir: str | None, gene: str) -> str | None:
    if not betadata_dir or not os.path.isdir(betadata_dir):
        return None
    exact = os.path.join(betadata_dir, f"{gene}_betadata.feather")
    if os.path.isfile(exact):
        return exact
    hits = glob.glob(os.path.join(betadata_dir, "**", f"{gene}_betadata.feather"), recursive=True)
    if hits:
        return sorted(hits)[0]
    safe = re.escape(gene)
    for path in glob.glob(os.path.join(betadata_dir, "**", "*_betadata.feather"), recursive=True):
        if re.match(rf"^{safe}_betadata\.feather$", os.path.basename(path), flags=re.IGNORECASE):
            return path
    return None


def _available_betadata_genes(betadata_dir: str | None, shared: set[str]) -> list[str]:
    if not betadata_dir or not os.path.isdir(betadata_dir):
        return []
    genes: list[str] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(os.path.join(betadata_dir, "**", "*_betadata.feather"), recursive=True)):
        base = os.path.basename(path)
        gene = base[: -len("_betadata.feather")]
        if gene in shared and gene not in seen:
            genes.append(gene)
            seen.add(gene)
    return genes


def _read_betadata_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_feather(path)
    except Exception:
        return feather.read_feather(path)


def _betadata_query_keys(id_col: str | None, obs_names: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    if id_col in ("CellID", "obs_names", "cell_id"):
        return obs_names
    return cluster_labels


def load_betadata_embedding(
    betadata_dir: str | None,
    genes: list[str],
    *,
    obs_names: list[str],
    cluster_labels: np.ndarray,
    max_beta_features: int = 8,
) -> tuple[np.ndarray | None, list[str], dict]:
    if not betadata_dir:
        return None, [], {"enabled": False, "reason": "no betadata dir"}
    feats: list[np.ndarray] = []
    names: list[str] = []
    used: list[dict] = []
    obs = np.asarray([str(x) for x in obs_names])
    fallback_labels = np.asarray([str(x) for x in cluster_labels])
    for gene in genes:
        path = _find_betadata_file(betadata_dir, gene)
        if path is None:
            continue
        df = _read_betadata_table(path)
        id_col = next((c for c in ("CellID", "obs_names", "cell_id", "Cluster") if c in df.columns), None)
        if id_col is None:
            row_keys = np.asarray([str(i) for i in range(df.shape[0])])
            value_df = df
        else:
            row_keys = df[id_col].astype(str).to_numpy()
            value_df = df.drop(columns=[id_col])
        numeric = value_df.select_dtypes(include=[np.number])
        cols = [c for c in numeric.columns if c not in ("beta0", "beta_0")]
        if not cols:
            continue
        cols = list(numeric[cols].abs().mean(axis=0).sort_values(ascending=False).head(max_beta_features).index)
        mat = numeric[cols].to_numpy(dtype=np.float32)
        fallback = np.nanmean(mat, axis=0, keepdims=True)
        fallback = np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)
        key_to_row = {str(k): i for i, k in enumerate(row_keys)}
        keys = _betadata_query_keys(id_col, obs, fallback_labels)
        idx = np.array([key_to_row.get(str(k), -1) for k in keys], dtype=int)
        aligned = np.repeat(fallback, len(keys), axis=0)
        ok = idx >= 0
        if np.any(ok):
            aligned[ok] = mat[idx[ok]]
        feats.append(aligned.astype(np.float32))
        names.extend([f"beta:{gene}:{c}" for c in cols])
        used.append({"gene": gene, "path": path, "id_col": id_col, "features": cols, "n_rows": int(df.shape[0])})
    if not feats:
        return None, [], {"enabled": True, "used": [], "reason": "no matching beta feathers"}
    emb = _safe_zscore(np.concatenate(feats, axis=1))
    return emb, names, {"enabled": True, "used": used, "n_features": len(names), "n_genes": len(used)}


def _feature_knn_probs(
    ref_features: np.ndarray,
    query_features: np.ndarray,
    ref_li: np.ndarray,
    n_ct: int,
    *,
    n_neighbors: int = 50,
) -> tuple[np.ndarray, dict]:
    k = min(n_neighbors, ref_features.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    nn.fit(ref_features)
    dists, idxs = nn.kneighbors(query_features)
    w = 1.0 / (dists + 1e-6)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-8)
    p = np.zeros((query_features.shape[0], n_ct), dtype=np.float32)
    for i in range(query_features.shape[0]):
        for j in range(k):
            p[i, ref_li[idxs[i, j]]] += w[i, j]
    p /= np.maximum(p.sum(1, keepdims=True), 1e-8)
    return p, {"n_neighbors": int(k), "metric": "cosine"}


def _self_knn_probs(
    features: np.ndarray,
    labels: np.ndarray,
    n_ct: int,
    *,
    n_neighbors: int = 30,
) -> np.ndarray:
    k = min(n_neighbors + 1, features.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    nn.fit(features)
    dists, idxs = nn.kneighbors(features)
    if k > 1:
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]
    w = 1.0 / (dists + 1e-6)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-8)
    p = np.zeros((features.shape[0], n_ct), dtype=np.float32)
    for i in range(features.shape[0]):
        for j in range(idxs.shape[1]):
            p[i, labels[idxs[i, j]]] += w[i, j]
    p /= np.maximum(p.sum(1, keepdims=True), 1e-8)
    return p


def spatial_coordinate_prior(
    ref_i: sc.AnnData,
    query: sc.AnnData,
    ref_labels: np.ndarray,
    cell_types: list[str],
) -> tuple[np.ndarray | None, dict]:
    _, rxy = _resolve_spatial_xy(ref_i)
    qkey, qxy = _resolve_spatial_xy(query)
    if rxy is None or qxy is None or len(cell_types) < 2:
        return None, {"enabled": False, "reason": "missing spatial coordinates"}
    rz = _safe_zscore(rxy)
    qz = _safe_zscore(qxy)
    labels = np.asarray([str(x) for x in ref_labels])
    centers = []
    for ct in cell_types:
        m = labels == str(ct)
        centers.append(rz[m].mean(0) if m.any() else np.zeros(2, dtype=np.float32))
    centers = np.asarray(centers, dtype=np.float32)
    d = np.sqrt(np.maximum(((qz[:, None, :] - centers[None, :, :]) ** 2).sum(-1), 0.0))
    p = np.exp(-d / max(float(np.median(d)), 0.25))
    p /= np.maximum(p.sum(1, keepdims=True), 1e-8)
    return p.astype(np.float32), {"enabled": True, "query_obsm": qkey}


def _spatial_neighbor_label_prior(
    query: sc.AnnData,
    base_p: np.ndarray,
    *,
    k: int = 8,
    blend: float = 0.35,
) -> tuple[np.ndarray, dict]:
    key, xy = _resolve_spatial_xy(query)
    if xy is None or query.n_obs < 4:
        return base_p, {"enabled": False, "reason": "no spatial coordinates"}
    nn = NearestNeighbors(n_neighbors=min(k + 1, query.n_obs), metric="euclidean")
    nn.fit(xy)
    dist, idx = nn.kneighbors(xy)
    neigh = idx[:, 1:]
    dist = dist[:, 1:]
    if neigh.shape[1] == 0:
        return base_p, {"enabled": False, "reason": "not enough spatial neighbors"}
    sig = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
    w = np.exp(-(dist**2) / (2.0 * max(sig, 1e-6) ** 2)).astype(np.float32)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
    sm = np.einsum("ij,ijk->ik", w, base_p[neigh])
    out = (1.0 - blend) * base_p + blend * sm
    out /= np.maximum(out.sum(1, keepdims=True), 1e-8)
    return out.astype(np.float32), {"enabled": True, "obsm": key, "k": int(neigh.shape[1]), "blend": float(blend)}


def _prob_accuracy(p: np.ndarray, labels: np.ndarray) -> float:
    if p is None or p.size == 0:
        return 0.0
    return float(np.mean(p.argmax(1) == labels))


def _weighted_prob_blend(parts: list[tuple[str, np.ndarray, float]]) -> tuple[np.ndarray, dict]:
    usable = [(name, p, max(0.0, float(w))) for name, p, w in parts if p is not None and p.size and w > 0]
    if not usable:
        raise ValueError("no probability parts to blend")
    total = sum(w for _, _, w in usable)
    out = sum(w * p for _, p, w in usable) / max(total, 1e-8)
    out /= np.maximum(out.sum(1, keepdims=True), 1e-8)
    return out.astype(np.float32), {name: float(w / max(total, 1e-8)) for name, _, w in usable}


def _evaluate_transfer(
    labels: np.ndarray,
    truth: np.ndarray | None,
    query: sc.AnnData,
    ref_ln: np.ndarray,
    ref_labels: np.ndarray,
    mgenes: dict[str, list[str]],
    amk: list[str],
    midx: np.ndarray,
) -> dict:
    out: dict = {}
    if truth is not None:
        truth_s = np.asarray([str(x) for x in truth])
        pred_s = np.asarray([str(x) for x in labels])
        out["accuracy"] = float(np.mean(truth_s == pred_s))
        out["balanced_accuracy"] = float(balanced_accuracy_score(truth_s, pred_s))
        out["ari"] = float(adjusted_rand_score(truth_s, pred_s))
    ref_lab_s = np.array([str(x) for x in ref_labels])
    pred = np.asarray([str(x) for x in labels])
    vals = []
    per_type = {}
    for ct in mgenes:
        mr = ref_lab_s == str(ct)
        mq = pred == str(ct)
        if mq.sum() < 3:
            continue
        ci = [amk.index(g) for g in mgenes[ct] if g in amk]
        if len(ci) < 2:
            continue
        rv = ref_ln[mr][:, midx[ci]].mean(0)
        qv = query.layers["ln"][mq][:, midx[ci]].mean(0)
        if np.std(rv) < 1e-6 or np.std(qv) < 1e-6:
            continue
        r, _ = pearsonr(rv, qv)
        r2 = float(r**2)
        vals.append(r2)
        per_type[str(ct)] = {"r2": r2, "n": int(mq.sum())}
    out["dotplot_mean_r2"] = float(np.mean(vals)) if vals else 0.0
    out["dotplot_per_type"] = per_type
    return out


def _label_distribution_features(
    adata: sc.AnnData,
    *,
    pca_features: np.ndarray,
    beta_features: np.ndarray | None = None,
    spatial_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    if pca_features is not None and pca_features.size:
        d = min(30, pca_features.shape[1])
        blocks.append(_safe_zscore(pca_features[:, :d]))
        names.append(f"pca:{d}")
    umap = adata.obsm.get("X_umap")
    if umap is not None:
        umap = np.asarray(umap, dtype=np.float32)
        if umap.ndim == 2 and umap.shape[0] == adata.n_obs and umap.shape[1] >= 2:
            d = min(umap.shape[1], 3)
            blocks.append(_safe_zscore(umap[:, :d]))
            names.append(f"umap:{d}")
    _, xy = _resolve_spatial_xy(adata)
    if xy is not None:
        blocks.append(_safe_zscore(xy))
        names.append("spatial_xy")
    if beta_features is not None and beta_features.size:
        d = min(beta_features.shape[1], 128)
        blocks.append(_safe_zscore(beta_features[:, :d]))
        names.append(f"betadata:{d}")
    if spatial_prior is not None:
        blocks.append(_safe_zscore(spatial_prior))
        names.append("spatial_prior")
    if not blocks:
        return np.zeros((adata.n_obs, 1), dtype=np.float32), ["bias"]
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def label_distribution_learning_probs(
    ref_i: sc.AnnData,
    query: sc.AnnData,
    ref_li: np.ndarray,
    cell_types: list[str],
    *,
    ref_pca: np.ndarray,
    query_pca: np.ndarray,
    ref_beta: np.ndarray | None,
    query_beta: np.ndarray | None,
    spatial_prior: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    ref_beta_use = query_beta_use = None
    if ref_beta is not None and query_beta is not None:
        d = min(ref_beta.shape[1], query_beta.shape[1])
        ref_beta_use = ref_beta[:, :d]
        query_beta_use = query_beta[:, :d]

    components: list[tuple[str, float, np.ndarray, dict]] = []

    d_pca = min(40, ref_pca.shape[1], query_pca.shape[1])
    pca_p, pca_meta = _feature_knn_probs(
        _safe_zscore(ref_pca[:, :d_pca]),
        _safe_zscore(query_pca[:, :d_pca]),
        ref_li,
        len(cell_types),
    )
    components.append(("pca_knn", 0.40, pca_p, {"n_features": int(d_pca), **pca_meta}))

    ru = ref_i.obsm.get("X_umap")
    qu = query.obsm.get("X_umap")
    if ru is not None and qu is not None:
        ru = np.asarray(ru, dtype=np.float32)
        qu = np.asarray(qu, dtype=np.float32)
        if ru.ndim == 2 and qu.ndim == 2 and ru.shape[0] == ref_i.n_obs and qu.shape[0] == query.n_obs:
            d_umap = min(ru.shape[1], qu.shape[1], 3)
            if d_umap >= 2:
                umap_p, umap_meta = _feature_knn_probs(
                    _safe_zscore(ru[:, :d_umap]),
                    _safe_zscore(qu[:, :d_umap]),
                    ref_li,
                    len(cell_types),
                    n_neighbors=35,
                )
                components.append(("umap_knn", 0.15, umap_p, {"n_features": int(d_umap), **umap_meta}))

    if ref_beta_use is not None and query_beta_use is not None and ref_beta_use.size and query_beta_use.size:
        beta_p, beta_meta = _feature_knn_probs(
            _safe_zscore(ref_beta_use),
            _safe_zscore(query_beta_use),
            ref_li,
            len(cell_types),
            n_neighbors=35,
        )
        components.append(("betadata_knn", 0.25, beta_p, {"n_features": int(ref_beta_use.shape[1]), **beta_meta}))

    if spatial_prior is not None and spatial_prior.shape == (query.n_obs, len(cell_types)):
        components.append(("spatial_prior", 0.10, spatial_prior.astype(np.float32), {"n_features": int(spatial_prior.shape[1])}))

    ref_prior = np.zeros((ref_i.n_obs, len(cell_types)), dtype=np.float32)
    ref_prior[np.arange(ref_i.n_obs), ref_li.astype(int)] = 1.0
    X_ref, blocks_ref = _label_distribution_features(
        ref_i,
        pca_features=ref_pca,
        beta_features=ref_beta_use,
        spatial_prior=ref_prior,
    )
    X_query, blocks_query = _label_distribution_features(
        query,
        pca_features=query_pca,
        beta_features=query_beta_use,
        spatial_prior=spatial_prior,
    )
    if X_ref.shape[1] != X_query.shape[1]:
        d = min(X_ref.shape[1], X_query.shape[1])
        X_ref = X_ref[:, :d]
        X_query = X_query[:, :d]
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.75,
        solver="lbfgs",
    )
    clf.fit(X_ref, ref_li)
    raw = clf.predict_proba(X_query).astype(np.float32)
    logit_p = np.full((query.n_obs, len(cell_types)), 1e-6, dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        ci = int(cls)
        if 0 <= ci < logit_p.shape[1]:
            logit_p[:, ci] = raw[:, j]
    logit_p /= np.maximum(logit_p.sum(1, keepdims=True), 1e-8)
    train_pred = clf.predict(X_ref)
    train_acc = float(np.mean(train_pred.astype(int) == ref_li.astype(int)))
    logistic_weight = 0.10 if train_acc >= 0.65 else 0.03
    components.append(("logistic", logistic_weight, logit_p, {"train_accuracy": train_acc}))

    total_w = sum(w for _, w, _, _ in components)
    probs = sum(w * p for _, w, p, _ in components) / max(total_w, 1e-8)
    probs /= np.maximum(probs.sum(1, keepdims=True), 1e-8)
    return probs, {
        "enabled": True,
        "model": "calibrated_label_distribution_ensemble",
        "feature_blocks_ref": blocks_ref,
        "feature_blocks_query": blocks_query,
        "n_features": int(X_ref.shape[1]),
        "classes": [cell_types[int(i)] for i in clf.classes_],
        "components": {
            name: {"weight": float(w), **meta}
            for name, w, _, meta in components
        },
    }


def run_malt(
    reference_path: str,
    query_path: str,
    groupby_columns: list[str | None] | None,
    outdir: str,
    output_query_name: str = "query_labeled.h5ad",
    extra_dotplot_markers: list[str] | None = None,
    expression_mode: str = "auto",
    counts_layer: str | None = None,
    prefer_raw_counts: bool = False,
    leiden_map: bool = True,
    reference_gene_list: str | None = None,
    spatial: bool = False,
    reference_betadata_dir: str | None = None,
    query_betadata_dir: str | None = None,
    benchmark_truth: str | None = None,
    spatial_genes_per_type: int = 6,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    extra_dotplot_markers = extra_dotplot_markers or []

    def _json_sanitize(obj):
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        if isinstance(obj, dict):
            return {str(k): _json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_sanitize(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return _json_sanitize(obj.tolist())
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    if groupby_columns is None or len(groupby_columns) == 0:
        gb_loop = _dedupe_preserve_groupby([None])
    else:
        gb_loop = _dedupe_preserve_groupby(list(groupby_columns))

    print("=" * 60)
    print("STEP 1: Load & preprocess")
    print("=" * 60)

    ref_path = _resolve_h5ad_path(reference_path)
    query_path_res = _resolve_h5ad_path(query_path)
    ref = read_h5ad_compat(ref_path)
    query = read_h5ad_compat(query_path_res)

    if reference_gene_list:
        _apply_reference_gene_list_inplace(ref, reference_gene_list)

    shared = sorted(set(ref.var_names) & set(query.var_names))
    if not shared:
        ph = _placeholder_var_names_ratio(ref.var_names)
        msg = "No overlapping genes between reference and query."
        if ph >= 0.9:
            msg += (
                "\n  Reference var_names look like numeric placeholders (e.g. '0'..'n'); "
                "they must match query gene symbols, or pass --reference-gene-list PATH "
                f"with exactly {ref.shape[1]} lines (one symbol per line, same order as ref.X columns)."
            )
        raise ValueError(msg)

    ref = ref[:, shared].copy()
    query = query[:, shared].copy()

    for gb_spec in gb_loop:
        if gb_spec is not None and gb_spec not in ref.obs.columns:
            raise KeyError(
                f"groupby column {gb_spec!r} not in ref.obs; available: {list(ref.obs.columns)}"
            )

    ref_meta = prepare_expression_inplace(
        ref,
        "reference",
        expression_mode=expression_mode,
        counts_layer=counts_layer,
        prefer_raw_counts=prefer_raw_counts,
    )
    query_meta = prepare_expression_inplace(
        query,
        "query",
        expression_mode=expression_mode,
        counts_layer=counts_layer,
        prefer_raw_counts=prefer_raw_counts,
    )

    ref.raw = None
    query.raw = None

    ref_pp = ref_meta.get("preprocess", "")
    query_pp = query_meta.get("preprocess", "")
    ref_skipped = "skip" in ref_pp
    query_skipped = "skip" in query_pp
    if ref_skipped != query_skipped:
        print(
            f"\n  ⚠ WARNING: Preprocessing mismatch!\n"
            f"    Reference: {ref_pp}\n"
            f"    Query:     {query_pp}\n"
            f"    This may cause scale differences in dotplots.\n"
            f"    Fix: pass --counts-layer <layer_name> so both use the same normalization,\n"
            f"    or --expression-mode lognorm if both are already consistently normalized.\n"
        )

    for ad in (ref, query):
        ln = ad.layers["ln"]
        if sp.issparse(ln):
            ln = ln.toarray()
        ln = np.asarray(ln, dtype=np.float32)
        ad.layers["ln"] = ln
        ad.X = ln.copy()

    n_q = query.shape[0]
    g2i = {g: i for i, g in enumerate(shared)}
    print(f"  Genes: {len(shared)} | Ref: {ref.shape[0]} | Query: {n_q}")
    print(f"  Groupby runs ({len(gb_loop)}): {[_resolve_groupby(ref, g) for g in gb_loop]}")

    def calc_r2(adata, col, ref_ln_arr, ref_lab, mgenes, amk, midx):
        res, vals = {}, []
        pred = adata.obs[col].astype(str).values
        ref_lab_s = np.array([str(x) for x in ref_lab])
        for ct in mgenes:
            mr = ref_lab_s == str(ct)
            mq = pred == str(ct)
            if mq.sum() < 5:
                continue
            ci = [amk.index(g) for g in mgenes[ct]]
            if len(ci) < 3:
                continue
            rv = ref_ln_arr[mr][:, midx[ci]].mean(0)
            qv = adata.layers["ln"][mq][:, midx[ci]].mean(0)
            if np.std(rv) < 1e-6 or np.std(qv) < 1e-6:
                continue
            r, _ = pearsonr(rv, qv)
            r2 = r**2
            vals.append(r2)
            res[ct] = {"r2": r2, "n": int(mq.sum()), "nm": len(ci)}
        return res, np.mean(vals) if vals else 0.0

    all_markers_by_group: dict[str, dict] = {}
    per_group_metrics: list[dict] = []
    csv_label_columns: list[str] = []
    last_snapshot: dict | None = None

    for run_i, gb_spec in enumerate(gb_loop):
        groupby_col = _resolve_groupby(ref, gb_spec)
        multi = len(gb_loop) > 1
        slug = _slug_for_suffix(groupby_col) if multi else ""
        plot_suffix = f"_{slug}" if multi else ""
        if multi:
            malt_c = f"malt_label_{slug}"
            conf_c = f"malt_confidence_{slug}"
            knn_c = f"knn_label_{slug}"
        else:
            malt_c, conf_c, knn_c = "malt_label", "malt_confidence", "knn_label"

        print("\n" + "=" * 60)
        print(f"MALT run {run_i + 1}/{len(gb_loop)}  groupby={groupby_col!r}")
        print("=" * 60)

        ref_i = ref.copy()
        ref_i.obs[groupby_col] = ref_i.obs[groupby_col].astype(str).astype("category")

        cell_types = sorted(ref_i.obs[groupby_col].cat.categories.tolist())
        ct2i = {c: i for i, c in enumerate(cell_types)}
        n_ct = len(cell_types)
        ref_labels = ref_i.obs[groupby_col].values
        ref_li = np.array([ct2i[str(l)] for l in ref_labels])

        print(f"  Types: {n_ct}")

        print("\n" + "=" * 60)
        print("STEP 2: DEGs & marker profiles")
        print("=" * 60)

        sc.tl.rank_genes_groups(
            ref_i, groupby=groupby_col, method="wilcoxon", n_genes=200, use_raw=False
        )

        n_top = 25
        mk_per_ct: dict[str, list] = {}
        all_mk: set[str] = set()

        for ct in cell_types:
            df = sc.get.rank_genes_groups_df(ref_i, group=ct)
            sig = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0.5)]
            top = sig.head(n_top)["names"].tolist()
            mk_per_ct[ct] = top
            all_mk.update(top)
            print(f"  {ct:18s}: {len(top):2d} markers | {top[:3]}")

        all_mk = sorted(all_mk)
        mk_idx = np.array([g2i[g] for g in all_mk])

        ref_ln = ref_i.layers["ln"]
        ref_mu = np.zeros((n_ct, len(all_mk)), dtype=np.float32)
        ref_sig = np.zeros_like(ref_mu)
        for ci, ct in enumerate(cell_types):
            m = ref_labels == ct
            e = ref_ln[m][:, mk_idx]
            ref_mu[ci] = e.mean(0)
            ref_sig[ci] = e.std(0) + 0.1

        mk_mask = np.zeros((n_ct, len(all_mk)), dtype=np.float32)
        for ci, ct in enumerate(cell_types):
            for g in mk_per_ct[ct]:
                mk_mask[ci, all_mk.index(g)] = 1.0

        print(f"\n  Unique markers: {len(all_mk)}, mask pairs: {int(mk_mask.sum())}")

        print("\n" + "=" * 60)
        print("STEP 3: PCA + KNN")
        print("=" * 60)

        sc.pp.highly_variable_genes(
            ref_i, n_top_genes=min(800, len(shared)), subset=False
        )
        hvg_i = [g2i[g] for g in ref_i.var_names[ref_i.var.highly_variable]]

        pca = PCA(n_components=50, random_state=42)
        rp = pca.fit_transform(ref_ln[:, hvg_i])
        qp = pca.transform(query.layers["ln"][:, hvg_i])
        print(f"  PCA explained: {pca.explained_variance_ratio_.sum():.3f}")

        nn = NearestNeighbors(n_neighbors=50, metric="cosine", n_jobs=-1)
        nn.fit(rp)
        dists, idxs = nn.kneighbors(qp)

        w = 1.0 / (dists + 1e-6)
        w /= w.sum(1, keepdims=True)

        knn_p = np.zeros((n_q, n_ct), dtype=np.float32)
        for i in range(n_q):
            for j in range(50):
                knn_p[i, ref_li[idxs[i, j]]] += w[i, j]
        knn_p /= knn_p.sum(1, keepdims=True)

        knn_labels = np.array(cell_types)[knn_p.argmax(1)]
        print("  KNN distribution:")
        for c, n in sorted(
            zip(*np.unique(knn_labels, return_counts=True)), key=lambda x: -x[1]
        ):
            print(f"    {c}: {n}")

        print("\n" + "=" * 60)
        print("STEP 4: Per-cell marker scoring")
        print("=" * 60)

        q_mk = query.layers["ln"][:, mk_idx]

        ref_mk_all = ref_ln[:, mk_idx]
        ref_global_mean = ref_mk_all.mean(0)
        ref_global_std = ref_mk_all.std(0) + 1e-6

        ref_rel = np.zeros_like(ref_mu)
        for ci, ct in enumerate(cell_types):
            ref_rel[ci] = (ref_mu[ci] - ref_global_mean) / ref_global_std

        cell_ll = np.full((n_q, n_ct), -50.0, dtype=np.float32)
        for ci, ct in enumerate(cell_types):
            mi = [all_mk.index(g) for g in mk_per_ct[ct]]
            if not mi:
                continue
            ref_pat = ref_rel[ci, mi]
            q_pat = q_mk[:, mi]
            q_centered = q_pat - q_pat.mean(axis=1, keepdims=True)
            ref_centered = ref_pat - ref_pat.mean()
            ref_norm = np.linalg.norm(ref_centered) + 1e-8
            q_norms = np.linalg.norm(q_centered, axis=1) + 1e-8
            cos_sim = (q_centered @ ref_centered) / (q_norms * ref_norm)
            cell_ll[:, ci] = cos_sim

        tau = 0.3
        mk_p = np.exp(cell_ll / tau)
        mk_p /= mk_p.sum(1, keepdims=True)

        init_p = 0.6 * knn_p + 0.4 * mk_p
        init_p /= init_p.sum(1, keepdims=True)

        init_lbl = np.array(cell_types)[init_p.argmax(1)]
        print("  Blended init:")
        for c, n in sorted(
            zip(*np.unique(init_lbl, return_counts=True)), key=lambda x: -x[1]
        ):
            print(f"    {c}: {n}")

        print("\n" + "=" * 60)
        print("STEP 5: Optimization")
        print("=" * 60)

        dev = torch.device("cpu")
        q_mk_t = torch.tensor(q_mk, dtype=torch.float32, device=dev)
        ref_rel_t = torch.tensor(ref_rel, dtype=torch.float32, device=dev)
        mmask_t = torch.tensor(mk_mask, dtype=torch.float32, device=dev)
        knn_t = torch.tensor(knn_p, dtype=torch.float32, device=dev)
        cll_t = torch.tensor(cell_ll, dtype=torch.float32, device=dev)
        anchor_np = 0.75 * knn_p + 0.25 * mk_p
        anchor_np /= np.maximum(anchor_np.sum(1, keepdims=True), 1e-8)
        anchor_order = np.sort(anchor_np, axis=1)
        anchor_margin = anchor_order[:, -1] - anchor_order[:, -2] if n_ct > 1 else anchor_order[:, -1]
        anchor_w_np = np.clip((anchor_margin - 0.05) / 0.35, 0.0, 1.0).astype(np.float32)
        anchor_t = torch.tensor(anchor_np, dtype=torch.float32, device=dev)
        anchor_w_t = torch.tensor(anchor_w_np, dtype=torch.float32, device=dev)

        logits = torch.tensor(
            np.log(init_p + 1e-8), dtype=torch.float32, device=dev, requires_grad=True
        )

        alpha_p = 15.0
        alpha_c = 3.0
        alpha_k = 0.8
        alpha_e = 0.5
        alpha_m = 0.12
        alpha_a = 3.0

        C_ref_m, W_m, U_q_m, manifold_src = _malt_manifold_loss_tensors(
            ref_i, query, ref_li, n_ct, rp, qp, dev
        )
        use_manifold = C_ref_m is not None and W_m is not None and U_q_m is not None
        if use_manifold:
            tri_m = torch.triu(torch.ones(n_ct, n_ct, device=dev), diagonal=1)
        else:
            tri_m = None

        opt = torch.optim.Adam([logits], lr=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=250, T_mult=2, eta_min=0.001
        )

        n_mp = float(mmask_t.sum().item())
        n_mp = max(n_mp, 1.0)
        hist = {
            k: []
            for k in ["total", "profile", "cell", "knn", "anchor", "entropy", "manifold"]
        }
        best_l, best_lg = float("inf"), None
        pat, pat_ctr = 60, 0

        q_global_mean_t = q_mk_t.mean(0)
        q_global_std_t = q_mk_t.std(0) + 1e-6

        if use_manifold:
            print(
                f"  Weights: profile={alpha_p}, cell={alpha_c}, knn={alpha_k}, "
                f"anchor={alpha_a}, entropy={alpha_e}, manifold={alpha_m} ({manifold_src})\n"
            )
        else:
            print(
                f"  Weights: profile={alpha_p}, cell={alpha_c}, knn={alpha_k}, "
                f"anchor={alpha_a}, entropy={alpha_e} (manifold skipped)\n"
            )

        for ep in range(800):
            opt.zero_grad()
            p = F.softmax(logits, dim=1)
            lp = F.log_softmax(logits, dim=1)

            tw = p.sum(0).clamp(min=1.0)
            qprof_raw = (p.T @ q_mk_t) / tw.unsqueeze(1)
            qprof_rel = (qprof_raw - q_global_mean_t) / q_global_std_t

            Lp = (((qprof_rel - ref_rel_t) ** 2) * mmask_t).sum() / n_mp

            Lc = -(p * cll_t).sum() / n_q

            Lk = F.kl_div(lp, knn_t, reduction="batchmean")

            La = -((anchor_t * lp).sum(dim=1) * anchor_w_t).sum() / anchor_w_t.sum().clamp_min(1.0)

            Le = -(p * lp).sum(1).mean()

            if use_manifold and tri_m is not None:
                den_b = p.sum(0).clamp(min=1e-6).unsqueeze(1)
                C_q = (p.T @ U_q_m) / den_b
                D_ref = torch.cdist(C_ref_m, C_ref_m, p=2).clamp_min(1e-8)
                D_q = torch.cdist(C_q, C_q, p=2).clamp_min(1e-8)
                Wmt = W_m * tri_m
                wsum = Wmt.sum() + 1e-8
                wr = (Wmt * D_ref).sum() / wsum
                wq = (Wmt * D_q).sum() / wsum
                N_ref = D_ref / wr.clamp_min(1e-8)
                N_q = D_q / wq.clamp_min(1e-8)
                Lm = ((Wmt * (N_ref - N_q) ** 2)).sum() / wsum
            else:
                Lm = torch.zeros((), device=dev)

            loss = alpha_p * Lp + alpha_c * Lc + alpha_k * Lk + alpha_a * La + alpha_e * Le
            if use_manifold:
                loss = loss + alpha_m * Lm
            loss.backward()
            opt.step()
            sched.step()

            lv = loss.item()
            hist["total"].append(lv)
            hist["profile"].append(Lp.item())
            hist["cell"].append(Lc.item())
            hist["knn"].append(Lk.item())
            hist["anchor"].append(La.item())
            hist["entropy"].append(Le.item())
            hist["manifold"].append(float(Lm.item()))

            if lv < best_l - 1e-5:
                best_l, best_lg, pat_ctr = lv, logits.detach().clone(), 0
            else:
                pat_ctr += 1

            if ep % 100 == 0 or ep == 799:
                hard = np.array(cell_types)[p.detach().numpy().argmax(1)]
                uq, uqn = np.unique(hard, return_counts=True)
                tstr = ", ".join(
                    f"{u}:{n}" for u, n in sorted(zip(uq, uqn), key=lambda x: -x[1])[:6]
                )
                print(
                    f"  E{ep:4d} | L={lv:7.2f} | prof={Lp.item():.3f} cell={Lc.item():.3f} "
                    f"knn={Lk.item():.3f} anchor={La.item():.3f} ent={Le.item():.3f} "
                    f"man={Lm.item():.3f} | {tstr}"
                )

            if pat_ctr >= pat and ep > 200:
                print(f"\n  Early stopping at epoch {ep}")
                break

        print(f"\n  Best loss: {best_l:.4f}")

        print("\n" + "=" * 60)
        print("STEP 6: Extract labels & validate")
        print("=" * 60)

        fp = F.softmax(best_lg, dim=1).numpy()
        opt_lbl = np.array(cell_types)[fp.argmax(1)]
        conf = fp.max(1)

        min_c = 5
        lc = dict(zip(*np.unique(opt_lbl, return_counts=True)))
        rare = {c for c, n in lc.items() if n < min_c}
        if rare:
            print(f"  Reassigning rare types (<{min_c}): {rare}")
            for i in range(len(opt_lbl)):
                if opt_lbl[i] in rare:
                    pi = fp[i].copy()
                    for ci, ct in enumerate(cell_types):
                        if ct in rare:
                            pi[ci] = 0
                    if pi.sum() > 0:
                        pi /= pi.sum()
                        opt_lbl[i] = cell_types[pi.argmax()]

        query.obs[malt_c] = opt_lbl
        query.obs[malt_c] = query.obs[malt_c].astype(str).astype("category")
        query.obs[conf_c] = conf
        query.obs[knn_c] = knn_labels
        query.obs[knn_c] = query.obs[knn_c].astype(str).astype("category")

        print(f"\n  MALT labels ({groupby_col}):")
        for c, n in sorted(
            query.obs[malt_c].value_counts().items(), key=lambda x: -x[1]
        ):
            print(f"    {c}: {n}")
        print(f"\n  KNN labels ({groupby_col}):")
        for c, n in sorted(
            query.obs[knn_c].value_counts().items(), key=lambda x: -x[1]
        ):
            print(f"    {c}: {n}")
        print(f"\n  Agreement: {(opt_lbl == knn_labels).mean():.3f}")

        malt_r, malt_r2 = calc_r2(
            query, malt_c, ref_ln, ref_labels, mk_per_ct, all_mk, mk_idx
        )
        knn_r, knn_r2 = calc_r2(
            query, knn_c, ref_ln, ref_labels, mk_per_ct, all_mk, mk_idx
        )

        print(f"\n  {'Type':<18} {'MALT R²':>8} {'KNN R²':>8} {'MALT n':>7} {'KNN n':>7}")
        print("  " + "-" * 50)
        for ct in sorted(set(list(malt_r) + list(knn_r))):
            mr = malt_r.get(ct, {}).get("r2", float("nan"))
            kr = knn_r.get(ct, {}).get("r2", float("nan"))
            mn = malt_r.get(ct, {}).get("n", 0)
            kn = knn_r.get(ct, {}).get("n", 0)
            print(f"  {ct:<18} {mr:>8.3f} {kr:>8.3f} {mn:>7d} {kn:>7d}")
        print(f"\n  {'MEAN':<18} {malt_r2:>8.3f} {knn_r2:>8.3f}")
        print(f"  Improvement: {malt_r2 - knn_r2:+.3f}")

        spatial_section = None
        if spatial:
            print("\n" + "=" * 60)
            print("STEP 6b: Spatial MALT beta benchmark")
            print("=" * 60)
            spatial_mk_per_ct, spatial_genes, spatial_marker_details = select_dotplot_training_markers(
                ref_i,
                groupby_col,
                cell_types,
                shared,
                genes_per_type=spatial_genes_per_type,
            )
            if not spatial_genes:
                spatial_genes = list(all_mk)
                spatial_mk_per_ct = mk_per_ct
            spatial_midx = np.array([g2i[g] for g in spatial_genes])
            train_gene_path = os.path.join(outdir, f"spatial_malt_training_genes{plot_suffix}.txt")
            with open(train_gene_path, "w") as f:
                for g in spatial_genes:
                    f.write(f"{g}\n")
            with open(os.path.join(outdir, f"spatial_malt_marker_scores{plot_suffix}.json"), "w") as f:
                json.dump(_json_sanitize(spatial_marker_details), f, indent=2)

            ref_beta, ref_beta_cols, ref_beta_meta = load_betadata_embedding(
                reference_betadata_dir,
                spatial_genes,
                obs_names=list(ref_i.obs_names.astype(str)),
                cluster_labels=ref_i.obs[groupby_col].astype(str).values,
            )
            query_beta, query_beta_cols, query_beta_meta = load_betadata_embedding(
                query_betadata_dir or reference_betadata_dir,
                spatial_genes,
                obs_names=list(query.obs_names.astype(str)),
                cluster_labels=knn_labels,
            )

            beta_knn_p = None
            beta_meta = {"enabled": False, "reference": ref_beta_meta, "query": query_beta_meta}
            beta_feature_scale = 1.0
            if ref_beta is not None and query_beta is not None:
                d_beta = min(ref_beta.shape[1], query_beta.shape[1])
                beta_feature_scale = max(1.0, np.sqrt(max(1, len(hvg_i)) / max(1, d_beta))) * 2.0
                ref_aug = np.concatenate(
                    [_safe_zscore(ref_ln[:, hvg_i]), _safe_zscore(ref_beta[:, :d_beta]) * beta_feature_scale],
                    axis=1,
                )
                query_aug = np.concatenate(
                    [
                        _safe_zscore(query.layers["ln"][:, hvg_i]),
                        _safe_zscore(query_beta[:, :d_beta]) * beta_feature_scale,
                    ],
                    axis=1,
                )
                beta_knn_p, beta_knn_meta = _feature_knn_probs(ref_aug, query_aug, ref_li, n_ct)
                beta_meta = {
                    "enabled": True,
                    "reference": ref_beta_meta,
                    "query": query_beta_meta,
                    "knn": beta_knn_meta,
                    "n_features": int(d_beta),
                    "feature_scale": float(beta_feature_scale),
                    "columns": {"reference": ref_beta_cols[:50], "query": query_beta_cols[:50]},
                }
                print(f"  Betadata KNN features: {d_beta} beta columns from {ref_beta_meta.get('n_genes', 0)} genes")
            else:
                print(f"  Betadata skipped: ref={ref_beta_meta.get('reason')} query={query_beta_meta.get('reason')}")

            spatial_prior, spatial_prior_meta = spatial_coordinate_prior(ref_i, query, ref_labels, cell_types)
            if spatial_prior is None:
                spatial_prior = np.full_like(knn_p, 1.0 / max(n_ct, 1))
            ldl_p, ldl_meta = label_distribution_learning_probs(
                ref_i,
                query,
                ref_li,
                cell_types,
                ref_pca=rp,
                query_pca=qp,
                ref_beta=ref_beta,
                query_beta=query_beta,
                spatial_prior=spatial_prior,
            )
            ldl_label = np.array(cell_types)[ldl_p.argmax(1)]
            ldl_conf = ldl_p.max(1)
            beta_prior = beta_knn_p if beta_knn_p is not None else knn_p
            prior_weights = (
                {
                    "malt": 0.25,
                    "ldl": 0.35,
                    "knn": 0.10,
                    "marker": 0.05,
                    "betadata": 0.15,
                    "spatial": 0.10,
                }
                if beta_knn_p is not None
                else {
                    "malt": 0.35,
                    "ldl": 0.30,
                    "knn": 0.20,
                    "marker": 0.05,
                    "betadata": 0.0,
                    "spatial": 0.10,
                }
            )
            spatial_seed = (
                prior_weights["malt"] * fp
                + prior_weights["ldl"] * ldl_p
                + prior_weights["knn"] * knn_p
                + prior_weights["marker"] * mk_p
                + prior_weights["betadata"] * beta_prior
                + prior_weights["spatial"] * spatial_prior
            )
            spatial_seed /= np.maximum(spatial_seed.sum(1, keepdims=True), 1e-8)
            spatial_smoothed, spatial_neighbor_meta = _spatial_neighbor_label_prior(query, spatial_seed)
            spatial_label = np.array(cell_types)[spatial_smoothed.argmax(1)]
            spatial_conf = spatial_smoothed.max(1)

            spatial_c = f"spatial_malt_label_{slug}" if multi else "spatial_malt_label"
            spatial_conf_c = f"spatial_malt_confidence_{slug}" if multi else "spatial_malt_confidence"
            beta_c = f"beta_knn_label_{slug}" if multi else "beta_knn_label"
            ldl_c = f"ldl_label_{slug}" if multi else "ldl_label"
            ldl_conf_c = f"ldl_confidence_{slug}" if multi else "ldl_confidence"
            if beta_knn_p is not None:
                query.obs[beta_c] = np.array(cell_types)[beta_knn_p.argmax(1)]
                query.obs[beta_c] = query.obs[beta_c].astype(str).astype("category")
                csv_label_columns.append(beta_c)
            query.obs[ldl_c] = ldl_label
            query.obs[ldl_c] = query.obs[ldl_c].astype(str).astype("category")
            query.obs[ldl_conf_c] = ldl_conf
            csv_label_columns.extend([ldl_c, ldl_conf_c])
            query.obs[spatial_c] = spatial_label
            query.obs[spatial_c] = query.obs[spatial_c].astype(str).astype("category")
            query.obs[spatial_conf_c] = spatial_conf
            csv_label_columns.extend([spatial_c, spatial_conf_c])

            truth = query.obs[benchmark_truth].astype(str).values if benchmark_truth and benchmark_truth in query.obs else None
            bench = {
                "knn": _evaluate_transfer(knn_labels, truth, query, ref_ln, ref_labels, spatial_mk_per_ct, spatial_genes, spatial_midx),
                "malt": _evaluate_transfer(opt_lbl, truth, query, ref_ln, ref_labels, spatial_mk_per_ct, spatial_genes, spatial_midx),
                "ldl": _evaluate_transfer(ldl_label, truth, query, ref_ln, ref_labels, spatial_mk_per_ct, spatial_genes, spatial_midx),
                "spatial_malt": _evaluate_transfer(spatial_label, truth, query, ref_ln, ref_labels, spatial_mk_per_ct, spatial_genes, spatial_midx),
            }
            if beta_knn_p is not None:
                bench["beta_knn"] = _evaluate_transfer(
                    np.array(cell_types)[beta_knn_p.argmax(1)],
                    truth,
                    query,
                    ref_ln,
                    ref_labels,
                    spatial_mk_per_ct,
                    spatial_genes,
                    spatial_midx,
                )
            spatial_section = {
                "enabled": True,
                "label_column": spatial_c,
                "confidence_column": spatial_conf_c,
                "training_genes": spatial_genes,
                "training_genes_file": train_gene_path,
                "markers_by_type": spatial_mk_per_ct,
                "reference_betadata_dir": reference_betadata_dir,
                "query_betadata_dir": query_betadata_dir,
                "betadata": beta_meta,
                "ldl": ldl_meta,
                "ldl_label_column": ldl_c,
                "ldl_confidence_column": ldl_conf_c,
                "prior_weights": prior_weights,
                "spatial_prior": spatial_prior_meta,
                "spatial_neighbors": spatial_neighbor_meta,
                "benchmark_truth": benchmark_truth,
                "benchmark": bench,
            }
            print("  Spatial benchmark dotplot R2:")
            for name, info in bench.items():
                acc = info.get("accuracy")
                acc_s = f", acc={acc:.3f}" if acc is not None else ""
                print(f"    {name}: {info.get('dotplot_mean_r2', 0.0):.3f}{acc_s}")

        print("\n" + "=" * 60)
        print("STEP 7: Dotplot comparison")
        print("=" * 60)

        vc = query.obs[malt_c].value_counts()
        malt_cts = [ct for ct in cell_types if ct in vc.index and vc[ct] >= 5]

        flat_mk, seen_g = [], set()
        for ct in malt_cts:
            for g in mk_per_ct.get(ct, [])[:4]:
                if g not in seen_g:
                    flat_mk.append(g)
                    seen_g.add(g)

        for g in extra_dotplot_markers:
            g = g.strip()
            if g and g in shared and g not in seen_g:
                flat_mk.append(g)
                seen_g.add(g)

        print(f"  Markers: {len(flat_mk)} across {len(malt_cts)} types")
        print(f"  Active types: {malt_cts}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(hist["total"], lw=1.5)
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Total Loss")
        axes[0].grid(alpha=0.3)
        for k in ["profile", "cell", "knn", "anchor", "entropy", "manifold"]:
            axes[1].plot(hist[k], lw=1.2, label=k)
        axes[1].set(xlabel="Epoch", ylabel="Value", title="Loss Components")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"loss_curve{plot_suffix}.png"), dpi=150
        )
        plt.close()
        print(f"  Loss curve saved (loss_curve{plot_suffix}.png)")

        dp_kw = dict(
            var_names=flat_mk,
            standard_scale="var",
            show=False,
            return_fig=True,
        )

        ref_sub = ref_i[ref_i.obs[groupby_col].isin(malt_cts)].copy()
        ref_sub.obs[groupby_col] = ref_sub.obs[groupby_col].cat.remove_unused_categories()
        dp = sc.pl.dotplot(ref_sub, groupby=groupby_col, **dp_kw)
        _dotplot_set_title(dp, "reference")
        dp.savefig(
            os.path.join(outdir, f"dotplot_reference{plot_suffix}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close("all")
        print(f"  Reference dotplot saved (dotplot_reference{plot_suffix}.png)")

        q_sub = query[query.obs[malt_c].isin(malt_cts)].copy()
        q_sub.obs[malt_c] = q_sub.obs[malt_c].cat.remove_unused_categories()
        dp = sc.pl.dotplot(q_sub, groupby=malt_c, **dp_kw)
        _dotplot_set_title(dp, "query")
        dp.savefig(
            os.path.join(outdir, f"dotplot_malt{plot_suffix}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close("all")
        print(f"  MALT dotplot saved (dotplot_malt{plot_suffix}.png)")

        kvc = query.obs[knn_c].value_counts()
        knn_cts = [ct for ct in cell_types if ct in kvc.index and kvc[ct] >= 5]
        if knn_cts:
            qk = query[query.obs[knn_c].isin(knn_cts)].copy()
            qk.obs[knn_c] = qk.obs[knn_c].cat.remove_unused_categories()
            dp = sc.pl.dotplot(qk, groupby=knn_c, **dp_kw)
            _dotplot_set_title(dp, "query")
            dp.savefig(
                os.path.join(outdir, f"dotplot_knn{plot_suffix}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")
            print(f"  KNN dotplot saved (dotplot_knn{plot_suffix}.png)")

        all_markers_by_group[groupby_col] = mk_per_ct
        metric_row = {
            "groupby": groupby_col,
            "malt_label_column": malt_c,
            "malt_mean_r2": float(malt_r2),
            "knn_mean_r2": float(knn_r2),
        }
        if spatial_section is not None:
            metric_row["spatial_malt_label_column"] = spatial_section["label_column"]
            metric_row["spatial_malt_mean_r2"] = float(
                spatial_section["benchmark"]["spatial_malt"].get("dotplot_mean_r2", 0.0)
            )
            metric_row["spatial_malt"] = spatial_section
        per_group_metrics.append(metric_row)
        csv_label_columns.extend([malt_c, conf_c, knn_c])

        _rl = ref_ln
        if sp.issparse(_rl):
            _rl = _rl.toarray()
        last_snapshot = {
            "groupby_col": groupby_col,
            "malt_c": malt_c,
            "cell_types": list(cell_types),
            "mk_per_ct": mk_per_ct,
            "all_mk": list(all_mk),
            "mk_idx": np.asarray(mk_idx).copy(),
            "ref_rel": np.asarray(ref_rel, dtype=np.float32).copy(),
            "mk_mask": np.asarray(mk_mask, dtype=np.float32).copy(),
            "ref_ln": np.asarray(_rl, dtype=np.float32),
            "ref_labels": np.asarray(ref_i.obs[groupby_col].astype(str)),
            "flat_mk": list(flat_mk),
            "malt_cts": list(malt_cts),
            "plot_suffix": plot_suffix,
        }

    leiden_section: dict | None = None
    if leiden_map and last_snapshot is not None:
        print("\n" + "=" * 60)
        print("STEP 8: MALT-guided adaptive Leiden")
        print("=" * 60)
        snap = last_snapshot
        leiden_info = adaptive_leiden_clustering(
            query,
            snap["malt_c"],
            leiden_key="leiden",
            work_col="leiden_R",
        )
        print("\n" + "=" * 60)
        print("STEP 9: Leiden cluster → cell type (dotplot loss)")
        print("=" * 60)
        map_info = optimize_cluster_mapping(
            query,
            "leiden",
            snap["malt_c"],
            snap["cell_types"],
            snap["all_mk"],
            snap["mk_idx"],
            snap["ref_rel"],
            snap["mk_mask"],
        )
        leiden_section = {
            "adaptive_leiden": leiden_info,
            "mapping": {
                "best_loss": map_info["best_loss"],
                "cluster_to_type": map_info["cluster_to_type"],
            },
            "leiden_mapping_json": os.path.join(outdir, "leiden_mapping.json"),
        }
        with open(os.path.join(outdir, "leiden_mapping.json"), "w") as f:
            json.dump(_json_sanitize(map_info["mapping_rows"]), f, indent=2)
        print(f"  leiden_mapping.json written ({len(map_info['mapping_rows'])} clusters)")

        dp_kw = dict(
            var_names=snap["flat_mk"],
            standard_scale="var",
            show=False,
            return_fig=True,
        )
        ref_i_vis = ref.copy()
        ref_i_vis.obs[snap["groupby_col"]] = ref_i_vis.obs[snap["groupby_col"]].astype(
            str
        )
        ref_sub = ref_i_vis[
            ref_i_vis.obs[snap["groupby_col"]].isin(snap["malt_cts"])
        ].copy()
        ref_sub.obs[snap["groupby_col"]] = ref_sub.obs[
            snap["groupby_col"]
        ].astype("category")
        ref_sub.obs[snap["groupby_col"]] = ref_sub.obs[
            snap["groupby_col"]
        ].cat.remove_unused_categories()
        dp = sc.pl.dotplot(ref_sub, groupby=snap["groupby_col"], **dp_kw)
        _dotplot_set_title(dp, "reference")
        dp.savefig(
            os.path.join(outdir, f"dotplot_reference_leiden{snap['plot_suffix']}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close("all")

        lc = query.obs["leiden_celltype"].astype(str).value_counts()
        leiden_cts = [c for c in snap["cell_types"] if c in lc.index and lc[c] >= 5]
        if leiden_cts:
            q_ld = query[query.obs["leiden_celltype"].isin(leiden_cts)].copy()
            q_ld.obs["leiden_celltype"] = q_ld.obs[
                "leiden_celltype"
            ].cat.remove_unused_categories()
            dp = sc.pl.dotplot(q_ld, groupby="leiden_celltype", **dp_kw)
            _dotplot_set_title(dp, "query leiden_celltype")
            dp.savefig(
                os.path.join(outdir, f"dotplot_leiden{snap['plot_suffix']}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")
            print(
                f"  dotplot_leiden{snap['plot_suffix']}.png saved "
                f"({len(leiden_cts)} types)"
            )
        csv_label_columns.extend(["leiden", "leiden_celltype"])

    out_h5ad = os.path.join(outdir, output_query_name)
    query.write_h5ad(out_h5ad)

    labels_csv = os.path.join(outdir, "malt_labels.csv")
    labels_df = query.obs.loc[:, csv_label_columns].copy()
    labels_df.index = pd.Index(query.obs_names.astype(str), name="obs_name")
    for c in labels_df.columns:
        if (
            c.startswith("malt_label")
            or c.startswith("knn_label")
            or c in ("leiden", "leiden_celltype")
        ):
            labels_df[c] = labels_df[c].astype(str)
    labels_df.to_csv(labels_csv, index=True)
    print(f"  Labels CSV saved ({labels_csv})")

    with open(os.path.join(outdir, "marker_genes.json"), "w") as f:
        json.dump(_json_sanitize(all_markers_by_group), f, indent=2)

    resolved = [m["groupby"] for m in per_group_metrics]
    meta = {
        "reference_path": reference_path,
        "query_path": query_path,
        "groupby_columns_resolved": resolved,
        "per_group": per_group_metrics,
        "output_query": out_h5ad,
        "labels_csv": labels_csv,
        "expression_mode": expression_mode,
        "counts_layer": counts_layer,
        "prefer_raw_counts": prefer_raw_counts,
        "reference_gene_list": reference_gene_list,
        "reference_expression": ref_meta,
        "query_expression": query_meta,
        "spatial": {
            "enabled": bool(spatial),
            "reference_betadata_dir": reference_betadata_dir,
            "query_betadata_dir": query_betadata_dir,
            "benchmark_truth": benchmark_truth,
            "genes_per_type": spatial_genes_per_type,
        },
        "leiden_map": bool(leiden_map),
        "leiden": leiden_section,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(_json_sanitize(meta), f, indent=2)

    print(f"\n{'=' * 60}")
    if per_group_metrics:
        parts = [
            f"{m['groupby']}: MALT R²={m['malt_mean_r2']:.3f} vs KNN R²={m['knn_mean_r2']:.3f}"
            for m in per_group_metrics
        ]
        print("DONE!  " + " | ".join(parts))
    print(f"Results: {outdir}/")
    print(f"{'=' * 60}")


def _flatten_groupby_cli(gb: list[str] | None) -> list[str] | None:
    if not gb:
        return None
    out: list[str] = []
    for item in gb:
        for part in str(item).split(","):
            t = part.strip()
            if t:
                out.append(t)
    return out or None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Marker-aware label transfer (MALT) from reference to query AnnData."
    )
    p.add_argument(
        "--reference",
        "-r",
        required=True,
        help="Path to reference .h5ad (must have labels in obs).",
    )
    p.add_argument(
        "--query",
        "-q",
        required=True,
        help="Path to query .h5ad (labels written to obs).",
    )
    p.add_argument(
        "--groupby",
        "-g",
        action="append",
        default=None,
        help="Reference obs column(s) for labels. Use a comma-separated list in one -g "
        "(e.g. -g cell_type,cell_type_fine) and/or pass -g multiple times. Each column gets "
        "an independent MALT run (obs + malt_labels.csv get suffixed columns when >1). "
        "If omitted entirely, uses the first present among: "
        + ", ".join(_GROUPBY_FALLBACKS)
        + ".",
    )
    p.add_argument(
        "--outdir",
        "-o",
        default="/tmp/malt_results",
        help="Directory for figures, marker_genes.json, and labeled query (default: /tmp/malt_results).",
    )
    p.add_argument(
        "--output-query",
        default="query_labeled.h5ad",
        help="Filename under outdir for the labeled query (default: query_labeled.h5ad).",
    )
    p.add_argument(
        "--extra-markers",
        default="",
        help="Comma-separated gene symbols to append to dotplots if present in data (optional).",
    )
    p.add_argument(
        "--expression-mode",
        choices=("auto", "counts", "lognorm"),
        default="auto",
        help="auto: infer counts vs log-normalized .X (respects adata.uns['log1p'] if set); "
        "counts: require counts from layer/raw/.X; lognorm: treat .X as log-normalized "
        "and skip normalize_total+log1p.",
    )
    p.add_argument(
        "--counts-layer",
        default=None,
        help="Use this adata.layers key as UMI/count matrix (then normalize_total+log1p). "
        "Takes precedence over other layer names and, unless --prefer-raw-counts helps resolve, .raw.",
    )
    p.add_argument(
        "--prefer-raw-counts",
        action="store_true",
        help="In auto/counts mode, try AnnData.raw.X (genes aligned to var_names) after standard layer names.",
    )
    p.add_argument(
        "--no-leiden-map",
        action="store_true",
        help="Skip Steps 8–9 (adaptive Leiden + cluster→type mapping). Default: run after MALT.",
    )
    p.add_argument(
        "--reference-gene-list",
        default=None,
        metavar="PATH",
        help="Text file: one gene symbol per line (and/or # comments, blank lines skipped). "
        "Line count must equal reference n_vars. Replaces reference var_names before intersecting "
        "with the query (for .h5ad files whose var_names are numeric placeholders).",
    )
    p.add_argument(
        "--spatial",
        action="store_true",
        help="Enable spatial MALT: dotplot-selected genes, SpaceTravLR betadata features, and spatial smoothing.",
    )
    p.add_argument(
        "--reference-betadata-dir",
        default=None,
        help="Reference SpaceTravLR output directory containing seed *_betadata.feather files.",
    )
    p.add_argument(
        "--query-betadata-dir",
        default=None,
        help="Query SpaceTravLR output directory containing seed *_betadata.feather files.",
    )
    p.add_argument(
        "--benchmark-truth",
        default=None,
        help="Optional query obs column with true labels for benchmarking KNN/MALT/spatial MALT.",
    )
    p.add_argument(
        "--spatial-genes-per-type",
        type=int,
        default=6,
        help="Number of dotplot-optimized genes to select per reference cell type.",
    )
    args = p.parse_args()

    extra = [x for x in args.extra_markers.split(",") if x.strip()]

    run_malt(
        reference_path=args.reference,
        query_path=args.query,
        groupby_columns=_flatten_groupby_cli(args.groupby),
        outdir=args.outdir,
        output_query_name=args.output_query,
        extra_dotplot_markers=extra,
        expression_mode=args.expression_mode,
        counts_layer=args.counts_layer,
        prefer_raw_counts=args.prefer_raw_counts,
        leiden_map=not args.no_leiden_map,
        reference_gene_list=args.reference_gene_list,
        spatial=args.spatial,
        reference_betadata_dir=args.reference_betadata_dir,
        query_betadata_dir=args.query_betadata_dir,
        benchmark_truth=args.benchmark_truth,
        spatial_genes_per_type=args.spatial_genes_per_type,
    )


if __name__ == "__main__":
    main()
