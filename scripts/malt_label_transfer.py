"""
Marker-Aware Label Transfer (MALT)

Optimized label transfer from reference scRNA-seq to query that minimizes
the difference in marker/DEG expression between reference and predicted labels.

Loss = alpha_p * L_profile + alpha_c * L_cell + alpha_k * L_knn + alpha_e * L_entropy

- L_profile: MSE between per-type marker profiles in ref vs query
- L_cell: per-cell negative log-likelihood under type-specific marker distributions
- L_knn: KL divergence from initial KNN (embedding structure prior)
- L_entropy: encourages confident single-type assignments

CLI:

  spacetravlr --map-labels --reference ref.h5ad --query q.h5ad --map-labels-outdir ./malt_out

  Or: python scripts/malt_label_transfer.py --reference ref.h5ad --query q.h5ad --outdir ./malt_out

  Query path may omit ``.h5ad`` (``-q /tmp/e14s`` resolves ``/tmp/e14s.h5ad``).
  If ``--groupby`` is omitted, the first matching column among common names is used
  (e.g. ``cell_type``, ``final_annotation``). Legacy h5ad files whose ``var`` is a
  1D dataset of gene names are repacked on read for anndata 0.12+.

  Optional expression handling (see --expression-mode, --counts-layer, --prefer-raw-counts).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
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


def read_h5ad_compat(path: str) -> sc.AnnData:
    path = os.path.abspath(path)
    with h5py.File(path, "r") as f:
        legacy_var = "var" in f and isinstance(f["var"], h5py.Dataset)
    if not legacy_var:
        return sc.read_h5ad(path)
    print(
        f"  [read_h5ad_compat] legacy var layout in {os.path.basename(path)}; "
        "repacked var as dataframe (var/_index) for anndata reader."
    )
    str_dt = h5py.string_dtype(encoding="utf-8")
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(path, tmp_path)
    try:
        with h5py.File(tmp_path, "r+") as f:
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


def run_malt(
    reference_path: str,
    query_path: str,
    groupby: str | None,
    outdir: str,
    output_query_name: str = "query_labeled.h5ad",
    extra_dotplot_markers: list[str] | None = None,
    expression_mode: str = "auto",
    counts_layer: str | None = None,
    prefer_raw_counts: bool = False,
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

    print("=" * 60)
    print("STEP 1: Load & preprocess")
    print("=" * 60)

    ref_path = _resolve_h5ad_path(reference_path)
    query_path_res = _resolve_h5ad_path(query_path)
    ref = read_h5ad_compat(ref_path)
    query = read_h5ad_compat(query_path_res)

    groupby = _resolve_groupby(ref, groupby)

    ref.obs[groupby] = ref.obs[groupby].astype(str).astype("category")

    shared = sorted(set(ref.var_names) & set(query.var_names))
    if not shared:
        raise ValueError("No overlapping genes between reference and query.")

    ref = ref[:, shared].copy()
    query = query[:, shared].copy()

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

    cell_types = sorted(ref.obs[groupby].cat.categories.tolist())
    ct2i = {c: i for i, c in enumerate(cell_types)}
    n_ct = len(cell_types)
    ref_labels = ref.obs[groupby].values
    ref_li = np.array([ct2i[str(l)] for l in ref_labels])
    n_q = query.shape[0]
    g2i = {g: i for i, g in enumerate(shared)}

    print(f"  Genes: {len(shared)} | Ref: {ref.shape[0]} | Query: {n_q} | Types: {n_ct}")

    print("\n" + "=" * 60)
    print("STEP 2: DEGs & marker profiles")
    print("=" * 60)

    sc.tl.rank_genes_groups(
        ref, groupby=groupby, method="wilcoxon", n_genes=200, use_raw=False
    )

    n_top = 25
    mk_per_ct: dict[str, list] = {}
    all_mk: set[str] = set()

    for ct in cell_types:
        df = sc.get.rank_genes_groups_df(ref, group=ct)
        sig = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0.5)]
        top = sig.head(n_top)["names"].tolist()
        mk_per_ct[ct] = top
        all_mk.update(top)
        print(f"  {ct:18s}: {len(top):2d} markers | {top[:3]}")

    all_mk = sorted(all_mk)
    mk_idx = np.array([g2i[g] for g in all_mk])

    ref_ln = ref.layers["ln"]
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

    sc.pp.highly_variable_genes(ref, n_top_genes=min(800, len(shared)), subset=False)
    hvg_i = [g2i[g] for g in ref.var_names[ref.var.highly_variable]]

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

    q_mk = query.layers["ln"][:, mk_idx]  # (n_q, n_markers)

    # For the profile loss we compare the RELATIVE ordering of markers within
    # each cell type (correlation-based), not their absolute expression.
    # This makes it robust to cross-platform level shifts (scRNA-seq vs spatial).
    #
    # For each type c, compute the reference's per-type mean profile as a
    # z-score relative to the overall reference mean, so we capture which
    # markers are ENRICHED vs depleted for that type.  In the loss, the
    # query profile for type c is similarly z-scored within the query.
    ref_mk_all = ref_ln[:, mk_idx]
    ref_global_mean = ref_mk_all.mean(0)
    ref_global_std = ref_mk_all.std(0) + 1e-6

    # Reference relative profile: how each type deviates from the global mean
    ref_rel = np.zeros_like(ref_mu)  # (n_ct, n_markers)
    for ci, ct in enumerate(cell_types):
        ref_rel[ci] = (ref_mu[ci] - ref_global_mean) / ref_global_std

    # Per-cell scoring: use rank-correlation with each type's RELATIVE profile
    # This is cross-platform robust (measures pattern, not level)
    cell_ll = np.full((n_q, n_ct), -50.0, dtype=np.float32)
    for ci, ct in enumerate(cell_types):
        mi = [all_mk.index(g) for g in mk_per_ct[ct]]
        if not mi:
            continue
        ref_pat = ref_rel[ci, mi]
        q_pat = q_mk[:, mi]
        # center each cell's marker vector (remove cell-level mean)
        q_centered = q_pat - q_pat.mean(axis=1, keepdims=True)
        ref_centered = ref_pat - ref_pat.mean()
        ref_norm = np.linalg.norm(ref_centered) + 1e-8
        q_norms = np.linalg.norm(q_centered, axis=1) + 1e-8
        cos_sim = (q_centered @ ref_centered) / (q_norms * ref_norm)
        cell_ll[:, ci] = cos_sim  # [-1, 1] range

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

    logits = torch.tensor(
        np.log(init_p + 1e-8), dtype=torch.float32, device=dev, requires_grad=True
    )

    alpha_p = 15.0
    alpha_c = 3.0
    alpha_k = 0.3
    alpha_e = 0.5

    opt = torch.optim.Adam([logits], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=250, T_mult=2, eta_min=0.001
    )

    n_mp = mmask_t.sum().item()
    hist = {k: [] for k in ["total", "profile", "cell", "knn", "entropy"]}
    best_l, best_lg = float("inf"), None
    pat, pat_ctr = 60, 0

    # Pre-compute query global stats for online z-scoring in the loss
    q_global_mean_t = q_mk_t.mean(0)
    q_global_std_t = q_mk_t.std(0) + 1e-6

    print(
        f"  Weights: profile={alpha_p}, cell={alpha_c}, knn={alpha_k}, entropy={alpha_e}\n"
    )

    for ep in range(800):
        opt.zero_grad()
        p = F.softmax(logits, dim=1)
        lp = F.log_softmax(logits, dim=1)

        # L_profile: relative profile matching
        # Compute query per-type profiles, then z-score them using the
        # query's global mean/std (analogous to ref_rel using ref global stats).
        # The loss penalizes differences in the RELATIVE enrichment pattern.
        tw = p.sum(0).clamp(min=1.0)
        qprof_raw = (p.T @ q_mk_t) / tw.unsqueeze(1)  # (n_ct, n_markers)
        qprof_rel = (qprof_raw - q_global_mean_t) / q_global_std_t

        Lp = (((qprof_rel - ref_rel_t) ** 2) * mmask_t).sum() / n_mp

        Lc = -(p * cll_t).sum() / n_q

        Lk = F.kl_div(lp, knn_t, reduction="batchmean")

        Le = -(p * lp).sum(1).mean()

        loss = alpha_p * Lp + alpha_c * Lc + alpha_k * Lk + alpha_e * Le
        loss.backward()
        opt.step()
        sched.step()

        lv = loss.item()
        hist["total"].append(lv)
        hist["profile"].append(Lp.item())
        hist["cell"].append(Lc.item())
        hist["knn"].append(Lk.item())
        hist["entropy"].append(Le.item())

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
                f"knn={Lk.item():.3f} ent={Le.item():.3f} | {tstr}"
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

    query.obs["malt_label"] = opt_lbl
    query.obs["malt_label"] = query.obs["malt_label"].astype("category")
    query.obs["malt_confidence"] = conf
    query.obs["knn_label"] = knn_labels
    query.obs["knn_label"] = query.obs["knn_label"].astype("category")

    print("\n  MALT labels:")
    for c, n in sorted(
        query.obs["malt_label"].value_counts().items(), key=lambda x: -x[1]
    ):
        print(f"    {c}: {n}")
    print("\n  KNN labels:")
    for c, n in sorted(
        query.obs["knn_label"].value_counts().items(), key=lambda x: -x[1]
    ):
        print(f"    {c}: {n}")
    print(f"\n  Agreement: {(opt_lbl == knn_labels).mean():.3f}")

    def calc_r2(adata, col, ref_ln_arr, ref_lab, mgenes, amk, midx):
        res, vals = {}, []
        for ct in mgenes:
            mr, mq = ref_lab == ct, adata.obs[col].values == ct
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

    malt_r, malt_r2 = calc_r2(
        query, "malt_label", ref_ln, ref_labels, mk_per_ct, all_mk, mk_idx
    )
    knn_r, knn_r2 = calc_r2(
        query, "knn_label", ref_ln, ref_labels, mk_per_ct, all_mk, mk_idx
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

    print("\n" + "=" * 60)
    print("STEP 7: Dotplot comparison")
    print("=" * 60)

    vc = query.obs["malt_label"].value_counts()
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
    for k in ["profile", "cell", "knn", "entropy"]:
        axes[1].plot(hist[k], lw=1.2, label=k)
    axes[1].set(xlabel="Epoch", ylabel="Value", title="Loss Components")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss_curve.png"), dpi=150)
    plt.close()
    print("  Loss curve saved")

    dp_kw = dict(
        var_names=flat_mk,
        standard_scale="var",
        show=False,
        return_fig=True,
    )

    ref_sub = ref[ref.obs[groupby].isin(malt_cts)].copy()
    ref_sub.obs[groupby] = ref_sub.obs[groupby].cat.remove_unused_categories()
    dp = sc.pl.dotplot(ref_sub, groupby=groupby, **dp_kw)
    dp.savefig(os.path.join(outdir, "dotplot_reference.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  Reference dotplot saved")

    q_sub = query[query.obs["malt_label"].isin(malt_cts)].copy()
    q_sub.obs["malt_label"] = q_sub.obs["malt_label"].cat.remove_unused_categories()
    dp = sc.pl.dotplot(q_sub, groupby="malt_label", **dp_kw)
    dp.savefig(os.path.join(outdir, "dotplot_malt.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  MALT dotplot saved")

    kvc = query.obs["knn_label"].value_counts()
    knn_cts = [ct for ct in cell_types if ct in kvc.index and kvc[ct] >= 5]
    if knn_cts:
        qk = query[query.obs["knn_label"].isin(knn_cts)].copy()
        qk.obs["knn_label"] = qk.obs["knn_label"].cat.remove_unused_categories()
        dp = sc.pl.dotplot(qk, groupby="knn_label", **dp_kw)
        dp.savefig(os.path.join(outdir, "dotplot_knn.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
        print("  KNN dotplot saved")

    out_h5ad = os.path.join(outdir, output_query_name)
    query.write_h5ad(out_h5ad)
    with open(os.path.join(outdir, "marker_genes.json"), "w") as f:
        json.dump(_json_sanitize(mk_per_ct), f, indent=2)
    meta = {
        "reference_path": reference_path,
        "query_path": query_path,
        "groupby": groupby,
        "output_query": out_h5ad,
        "malt_mean_r2": float(malt_r2),
        "knn_mean_r2": float(knn_r2),
        "expression_mode": expression_mode,
        "counts_layer": counts_layer,
        "prefer_raw_counts": prefer_raw_counts,
        "reference_expression": ref_meta,
        "query_expression": query_meta,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(_json_sanitize(meta), f, indent=2)

    print(f"\n{'=' * 60}")
    print(
        f"DONE!  MALT R² = {malt_r2:.3f}  vs  KNN R² = {knn_r2:.3f}  (Δ={malt_r2 - knn_r2:+.3f})"
    )
    print(f"Results: {outdir}/")
    print(f"{'=' * 60}")


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
        default=None,
        help="Reference obs column with cell type labels. "
        "If omitted, uses the first present among: "
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
    args = p.parse_args()

    extra = [x for x in args.extra_markers.split(",") if x.strip()]

    run_malt(
        reference_path=args.reference,
        query_path=args.query,
        groupby=args.groupby,
        outdir=args.outdir,
        output_query_name=args.output_query,
        extra_dotplot_markers=extra,
        expression_mode=args.expression_mode,
        counts_layer=args.counts_layer,
        prefer_raw_counts=args.prefer_raw_counts,
    )


if __name__ == "__main__":
    main()
