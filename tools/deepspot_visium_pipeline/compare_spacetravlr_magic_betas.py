#!/usr/bin/env python3
"""
Paired H&E-imputed vs measured-expression SpaceTravLR seed-beta benchmark.

The script builds two matched 10-gene AnnData files from the paired DeepSpot
benchmark, applies the same clusterwise MAGIC smoothing to both expression
sources, trains SpaceTravLR seed models, and correlates matched beta
coefficients.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr


ID_COLUMNS = ("CellID", "obs_names", "cell_id", "Cluster")
INTERCEPT_COLUMNS = {"beta0", "beta_0"}

MAGIC_UV_PY = r"""
import sys

import anndata as ad
import magic
import numpy as np
import scipy.sparse as sp

src, dst, cluster_obs, batch_obs = sys.argv[1:5]
knn_def = int(sys.argv[5])
knn_max_cap = int(sys.argv[6])
decay = int(sys.argv[7])
t_magic = int(sys.argv[8])
n_pca_cap = int(sys.argv[9])

a = ad.read_h5ad(src)
if "normalized_count" not in a.layers:
    sys.exit("expected layers['normalized_count']")
if cluster_obs not in a.obs.columns:
    sys.exit("cluster obs column not found: %r" % (cluster_obs,))
if batch_obs != "-" and batch_obs not in a.obs.columns:
    sys.exit("magic batch obs column not found: %r" % (batch_obs,))

nc = a.layers["normalized_count"]
if sp.issparse(nc):
    x = nc.toarray().astype(np.float64)
else:
    x = np.asarray(nc, dtype=np.float64)
x[~np.isfinite(x)] = 0.0

labels = np.asarray([str(v) for v in a.obs[cluster_obs].to_numpy()], dtype=object)
batch = None if batch_obs == "-" else np.asarray([str(v) for v in a.obs[batch_obs].to_numpy()], dtype=object)
out = x.copy()


def magic_op(n_sub, n_genes):
    knn = min(knn_def, max(1, n_sub - 1))
    knn_max = max(knn, min(knn_max_cap, n_sub - 1))
    pca_bound = min(int(n_sub), int(n_genes))
    n_pca = min(n_pca_cap, max(1, pca_bound - 1))
    return magic.MAGIC(knn=knn, knn_max=knn_max, decay=decay, t=t_magic, n_pca=n_pca, verbose=0)


if batch is None:
    keys = [(lab, None) for lab in np.unique(labels)]
else:
    keys = sorted({(labels[i], batch[i]) for i in range(labels.size)})

for lab, bt in keys:
    mask = labels == lab if bt is None else (labels == lab) & (batch == bt)
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        continue
    sub = np.asarray(x[idx, :], dtype=np.float64)
    active = np.sum(sub, axis=0) > 0.0
    if int(active.sum()) == 0:
        continue
    op = magic_op(int(idx.size), int(active.sum()))
    if bool(np.all(active)):
        out[idx, :] = np.asarray(op.fit_transform(sub, genes="all_genes"), dtype=np.float64)
    else:
        imputed = sub.copy()
        imputed[:, active] = np.asarray(op.fit_transform(sub[:, active], genes="all_genes"), dtype=np.float64)
        out[idx, :] = imputed

out[~np.isfinite(out)] = 0.0
a.X = out.astype(np.float32)
a.layers["normalized_count"] = x.astype(np.float32)
a.layers["imputed_count"] = out.astype(np.float32)
a.write_h5ad(dst)
"""


@dataclass(frozen=True)
class CorrStats:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    cosine: float
    mae: float
    rmse: float
    sign_concordance: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paired_h5ad() -> Path:
    example = Path(__file__).resolve().parent / "example_run"
    for name in ("zen38_paired_uni_official.h5ad", "zen38_paired_measured_vs_deepspot.h5ad"):
        p = example / name
        if p.exists():
            return p
    return example / "zen38_paired_measured_vs_deepspot.h5ad"


def dense_matrix(a: ad.AnnData, layer: str) -> np.ndarray:
    m = a.X if layer == "X" else a.layers[layer]
    if sp.issparse(m):
        m = m.toarray()
    out = np.asarray(m, dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan, math.nan
    xv = np.asarray(x[mask], dtype=np.float64)
    yv = np.asarray(y[mask], dtype=np.float64)
    if float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return math.nan, math.nan
    r, p = pearsonr(xv, yv)
    return float(r), float(p)


def parse_gene_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    genes = [g.strip() for g in raw.replace("\n", ",").split(",") if g.strip()]
    return genes or None


def choose_genes(
    a: ad.AnnData,
    measured_layer: str,
    he_layer: str,
    n_genes: int,
    explicit_genes: list[str] | None,
) -> pd.DataFrame:
    var_names = list(map(str, a.var_names))
    var_index = {g: i for i, g in enumerate(var_names)}
    if explicit_genes is not None:
        missing = [g for g in explicit_genes if g not in var_index]
        if missing:
            raise ValueError(f"Explicit genes missing from AnnData var_names: {missing}")
        genes = explicit_genes
    else:
        measured = dense_matrix(a, measured_layer)
        he = dense_matrix(a, he_layer)
        rows = []
        for i, gene in enumerate(var_names):
            r, p = safe_pearson(measured[:, i], he[:, i])
            rows.append(
                {
                    "gene": gene,
                    "source_pearson_r": r,
                    "source_pearson_p": p,
                    "measured_std": float(np.std(measured[:, i])),
                    "he_std": float(np.std(he[:, i])),
                    "measured_mean": float(np.mean(measured[:, i])),
                    "he_mean": float(np.mean(he[:, i])),
                }
            )
        ranked = pd.DataFrame(rows)
        ranked = ranked[
            ranked["source_pearson_r"].notna()
            & (ranked["measured_std"] > 1e-12)
            & (ranked["he_std"] > 1e-12)
        ].sort_values("source_pearson_r", ascending=False)
        if ranked.shape[0] < n_genes:
            raise ValueError(f"Only {ranked.shape[0]} variable paired genes available; need {n_genes}.")
        return ranked.head(n_genes).reset_index(drop=True)

    measured = dense_matrix(a[:, genes], measured_layer)
    he = dense_matrix(a[:, genes], he_layer)
    rows = []
    for i, gene in enumerate(genes):
        r, p = safe_pearson(measured[:, i], he[:, i])
        rows.append(
            {
                "gene": gene,
                "source_pearson_r": r,
                "source_pearson_p": p,
                "measured_std": float(np.std(measured[:, i])),
                "he_std": float(np.std(he[:, i])),
                "measured_mean": float(np.mean(measured[:, i])),
                "he_mean": float(np.mean(he[:, i])),
            }
        )
    return pd.DataFrame(rows)


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(values.shape[0], dtype=np.int64)
    bins[order] = np.minimum(np.arange(values.shape[0]) * n_bins // values.shape[0], n_bins - 1)
    return bins


def add_common_clusters(
    a: ad.AnnData,
    obs: pd.DataFrame,
    cluster_obs: str,
    force_spatial_bins: bool,
    grid: int,
    min_cells: int,
) -> tuple[pd.DataFrame, str]:
    if not force_spatial_bins and cluster_obs in obs.columns:
        obs[cluster_obs] = obs[cluster_obs].astype(str)
        return obs, f"existing obs[{cluster_obs!r}]"
    if not force_spatial_bins and "leiden" in obs.columns:
        obs[cluster_obs] = obs["leiden"].astype(str)
        return obs, "existing obs['leiden'] copied"
    if "spatial" not in a.obsm:
        obs[cluster_obs] = "spot"
        return obs, "single cluster; no obsm['spatial']"

    xy = np.asarray(a.obsm["spatial"], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        obs[cluster_obs] = "spot"
        return obs, "single cluster; invalid obsm['spatial']"

    g = max(1, int(grid))
    labels = None
    while g >= 1:
        bx = quantile_bins(xy[:, 0], g)
        by = quantile_bins(xy[:, 1], g)
        labels = np.array([f"spatial_bin_{x}_{y}" for x, y in zip(bx, by)], dtype=object)
        counts = pd.Series(labels).value_counts()
        if g == 1 or int(counts.min()) >= min_cells:
            break
        g -= 1
    obs[cluster_obs] = labels
    return obs, f"{g}x{g} spatial quantile bins"


def magic_op(n_sub: int, n_genes: int, args: argparse.Namespace, magic_module: object) -> object:
    knn = min(args.magic_knn, max(1, n_sub - 1))
    knn_max = max(knn, min(args.magic_knn_max, n_sub - 1))
    pca_bound = min(int(n_sub), int(n_genes))
    n_pca = min(args.magic_n_pca, max(1, pca_bound - 1))
    return magic_module.MAGIC(
        knn=knn,
        knn_max=knn_max,
        decay=args.magic_decay,
        t=args.magic_t,
        n_pca=n_pca,
        verbose=0,
    )


def magic_impute_clusterwise(
    x: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray | None,
    args: argparse.Namespace,
    magic_module: object,
) -> np.ndarray:
    out = x.copy()
    if batch is None:
        keys = [(lab, None) for lab in np.unique(labels)]
    else:
        keys = sorted({(labels[i], batch[i]) for i in range(labels.size)})

    for lab, bt in keys:
        mask = labels == lab if bt is None else (labels == lab) & (batch == bt)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue
        sub = np.asarray(x[idx, :], dtype=np.float64)
        active = np.sum(sub, axis=0) > 0.0
        if int(active.sum()) == 0:
            continue
        op = magic_op(int(idx.size), int(active.sum()), args, magic_module)
        if bool(np.all(active)):
            out[idx, :] = np.asarray(op.fit_transform(sub, genes="all_genes"), dtype=np.float64)
        else:
            imputed = sub.copy()
            imputed[:, active] = np.asarray(
                op.fit_transform(sub[:, active], genes="all_genes"),
                dtype=np.float64,
            )
            out[idx, :] = imputed
    out[~np.isfinite(out)] = 0.0
    return out


def run_magic_uv(source_path: Path, out_path: Path, cluster_obs: str, args: argparse.Namespace) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("MAGIC backend 'uv' requires the `uv` command. Install uv or pass --magic-backend python.")
    cmd = [
        uv,
        "--no-cache",
        "run",
        "--isolated",
        "--quiet",
        "--no-cache-dir",
        "--with",
        "numpy<2",
        "--with",
        "anndata>=0.11",
        "--with",
        "scipy",
        "--with",
        "magic-impute>=3,<4",
        "python",
        "-c",
        MAGIC_UV_PY,
        str(source_path),
        str(out_path),
        cluster_obs,
        args.magic_batch_obs or "-",
        str(args.magic_knn),
        str(args.magic_knn_max),
        str(args.magic_decay),
        str(args.magic_t),
        str(args.magic_n_pca),
    ]
    log_path = out_path.with_suffix(".magic.log")
    with log_path.open("w") as log:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env.setdefault(key, "1")
        proc = subprocess.run(
            cmd,
            cwd=repo_root(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"MAGIC imputation failed with exit code {proc.returncode}; see {log_path}")


def run_magic_python(
    source: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    try:
        import magic as magic_module
    except ImportError as exc:
        raise RuntimeError(
            "MAGIC backend 'python' requires `pip install magic-impute`; "
            "the default --magic-backend uv avoids this dependency conflict."
        ) from exc
    return magic_impute_clusterwise(source, labels, batch, args, magic_module)


def make_training_h5ad(
    paired: ad.AnnData,
    genes: list[str],
    source_layer: str,
    label: str,
    cluster_obs: str,
    args: argparse.Namespace,
    out_path: Path,
) -> dict[str, object]:
    sub = paired[:, genes].copy()
    source = dense_matrix(sub, source_layer)
    source[source < 0.0] = 0.0
    obs = paired.obs.copy()
    obs, cluster_note = add_common_clusters(
        paired,
        obs,
        cluster_obs,
        args.force_spatial_bins,
        args.cluster_grid,
        args.min_cells_per_cluster,
    )
    labels = obs[cluster_obs].astype(str).to_numpy()
    batch = None
    if args.magic_batch_obs:
        if args.magic_batch_obs not in obs.columns:
            raise ValueError(f"--magic-batch-obs {args.magic_batch_obs!r} not found in obs")
        batch = obs[args.magic_batch_obs].astype(str).to_numpy()

    out = ad.AnnData(
        source.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    out.layers["normalized_count"] = source.astype(np.float32)
    if "spatial" in paired.obsm:
        out.obsm["spatial"] = np.asarray(paired.obsm["spatial"])
    out.uns["spacetravlr_magic_beta_benchmark"] = {
        "source_label": label,
        "source_layer": source_layer,
        "cluster_obs": cluster_obs,
        "cluster_note": cluster_note,
        "magic": {
            "knn": args.magic_knn,
            "knn_max": args.magic_knn_max,
            "decay": args.magic_decay,
            "t": args.magic_t,
            "n_pca": args.magic_n_pca,
            "batch_obs": args.magic_batch_obs,
            "backend": args.magic_backend,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.magic_backend == "python":
        imputed = run_magic_python(source, labels, batch, args)
        out.X = imputed.astype(np.float32)
        out.layers["imputed_count"] = imputed.astype(np.float32)
        out.write_h5ad(out_path)
    else:
        source_path = out_path.with_name(f"{out_path.stem}_source.h5ad")
        out.write_h5ad(source_path)
        run_magic_uv(source_path, out_path, cluster_obs, args)
    return {
        "path": str(out_path),
        "source_layer": source_layer,
        "cluster_note": cluster_note,
        "n_clusters": int(pd.Series(labels).nunique()),
    }


def toml_quote(value: str | Path) -> str:
    return json.dumps(str(value))


def write_config(
    path: Path,
    h5ad_path: Path,
    run_dir: Path,
    genes: list[str],
    cluster_obs: str,
    args: argparse.Namespace,
) -> None:
    data_dir = args.network_data_dir or (repo_root() / "data")
    genes_toml = "[" + ", ".join(toml_quote(g) for g in genes) + "]"
    text = f"""[data]
adata_path = {toml_quote(h5ad_path)}
layer = "imputed_count"
cluster_annot = {toml_quote(cluster_obs)}

[spatial]
radius = {args.radius}
spatial_dim = {args.spatial_dim}
contact_distance = {args.contact_distance}
weighted_ligand_scale_factor = {args.weighted_ligand_scale_factor}

[grn]
network_data_dir = {toml_quote(data_dir)}
tf_ligand_cutoff = 0.1
max_lr = 0
use_tf_modulators = false
use_lr_modulators = true
use_tfl_modulators = false
extra_modulators = {genes_toml}

[lasso]
l1_reg = {args.l1_reg}
group_reg = {args.group_reg}
n_iter = {args.n_iter}
tol = {args.tol}
scale_modulators = true
unscale_betas_on_export = false
parallel_lasso_clusters = false
gram_override = true

[training]
mode = "seed"
epochs = 1
score_threshold = -1000000000.0
genes = {genes_toml}

[execution]
n_parallel = {args.parallel}
output_dir = {toml_quote(run_dir)}
random_seed = {args.random_seed}
stale_lock_secs = 3600
write_minimal_repro_h5ad = false

[perturbation]
beta_scale_factor = 1.0
n_propagation = 4

[model_export]
save_cnn_weights = false
compressed_npz = true
output_subdir = "CNN_weights"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def spacetravlr_command(args: argparse.Namespace) -> list[str]:
    if args.spacetravlr_cmd:
        return shlex.split(args.spacetravlr_cmd)
    exe = shutil.which("spacetravlr")
    if exe:
        return [exe]
    return ["cargo", "run", "--quiet", "--bin", "spacetravlr", "--"]


def run_spacetravlr(label: str, config: Path, run_dir: Path, args: argparse.Namespace) -> None:
    cmd = spacetravlr_command(args) + [
        "--plain",
        "--skip-auto-adata-prep",
        "--config",
        str(config),
        "--training-mode",
        "seed",
        "--parallel",
        str(args.parallel),
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "spacetravlr_train.log"
    env = os.environ.copy()
    env.setdefault("SPACETRAVLR_FORCE_CPU", "1")
    print(f"[{label}] running: {' '.join(shlex.quote(x) for x in cmd)}")
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=repo_root(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"SpaceTravLR {label} run failed with exit code {proc.returncode}; see {log_path}")


def reset_run_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_beta_long(run_dir: Path, genes: list[str], source: str, include_intercept: bool) -> pd.DataFrame:
    rows = []
    for gene in genes:
        path = run_dir / f"{gene}_betadata.feather"
        if not path.exists():
            raise FileNotFoundError(f"Missing betadata for {gene}: {path}")
        df = pd.read_feather(path)
        id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
        if id_col is None:
            df = df.copy()
            id_col = "_row"
            df[id_col] = np.arange(df.shape[0]).astype(str)
        data_cols = [c for c in df.columns if c != id_col]
        if not include_intercept:
            data_cols = [c for c in data_cols if c not in INTERCEPT_COLUMNS]
        for _, row in df.iterrows():
            row_id = str(row[id_col])
            for col in data_cols:
                value = row[col]
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "source": source,
                        "gene": gene,
                        "row_id": row_id,
                        "beta_column": col,
                        "beta": value_f,
                    }
                )
    return pd.DataFrame(rows)


def compute_stats(x: np.ndarray, y: np.ndarray, zero_eps: float) -> CorrStats:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    n = int(x.size)
    if n < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return CorrStats(n, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    pr, pp = pearsonr(x, y)
    sr, spv = spearmanr(x, y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    cosine = float(np.dot(x, y) / denom) if denom > 0 else math.nan
    delta = x - y
    nonzero = (np.abs(x) > zero_eps) | (np.abs(y) > zero_eps)
    sign = float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero]))) if np.any(nonzero) else math.nan
    return CorrStats(
        n=n,
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_r=float(sr),
        spearman_p=float(spv),
        cosine=cosine,
        mae=float(np.mean(np.abs(delta))),
        rmse=float(np.sqrt(np.mean(delta**2))),
        sign_concordance=sign,
    )


def bootstrap_ci(
    pairs: pd.DataFrame,
    value_col_x: str,
    value_col_y: str,
    n_bootstrap: int,
    seed: int,
    group_col: str | None,
) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    rs = []
    if group_col is None:
        x = pairs[value_col_x].to_numpy(dtype=np.float64)
        y = pairs[value_col_y].to_numpy(dtype=np.float64)
        n = len(pairs)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            r, _ = safe_pearson(x[idx], y[idx])
            if np.isfinite(r):
                rs.append(r)
    else:
        groups = pairs[group_col].drop_duplicates().to_numpy()
        by_group = {g: frame.index.to_numpy() for g, frame in pairs.groupby(group_col)}
        for _ in range(n_bootstrap):
            sampled = rng.choice(groups, size=groups.size, replace=True)
            idx = np.concatenate([by_group[g] for g in sampled])
            r, _ = safe_pearson(
                pairs.loc[idx, value_col_x].to_numpy(dtype=np.float64),
                pairs.loc[idx, value_col_y].to_numpy(dtype=np.float64),
            )
            if np.isfinite(r):
                rs.append(r)
    if not rs:
        return {"n": 0, "lo": math.nan, "hi": math.nan}
    arr = np.asarray(rs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
    }


def permutation_pvalue(x: np.ndarray, y: np.ndarray, observed: float, n_perm: int, seed: int) -> float:
    if n_perm <= 0 or not np.isfinite(observed):
        return math.nan
    rng = np.random.default_rng(seed)
    count = 0
    valid = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r, _ = safe_pearson(x, yp)
        if not np.isfinite(r):
            continue
        valid += 1
        count += int(abs(r) >= abs(observed))
    if valid == 0:
        return math.nan
    return float((count + 1) / (valid + 1))


def compare_betas(
    measured_run: Path,
    he_run: Path,
    genes: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, object]:
    measured = read_beta_long(measured_run, genes, "measured", args.include_intercept)
    he = read_beta_long(he_run, genes, "he", args.include_intercept)
    pairs = measured.merge(
        he,
        on=["gene", "row_id", "beta_column"],
        suffixes=("_measured", "_he"),
        validate="one_to_one",
    )
    pairs = pairs[np.isfinite(pairs["beta_measured"]) & np.isfinite(pairs["beta_he"])].copy()
    if pairs.empty:
        raise RuntimeError("No matched finite beta coefficients found between runs.")

    x = pairs["beta_measured"].to_numpy(dtype=np.float64)
    y = pairs["beta_he"].to_numpy(dtype=np.float64)
    overall = compute_stats(x, y, args.zero_eps)
    per_gene_rows = []
    for gene, frame in pairs.groupby("gene"):
        s = compute_stats(
            frame["beta_measured"].to_numpy(dtype=np.float64),
            frame["beta_he"].to_numpy(dtype=np.float64),
            args.zero_eps,
        )
        per_gene_rows.append({"gene": gene, **s.__dict__})
    per_gene = pd.DataFrame(per_gene_rows).sort_values("pearson_r", ascending=False, na_position="last")

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "matched_beta_pairs.csv", index=False)
    per_gene.to_csv(out_dir / "per_gene_beta_correlations.csv", index=False)

    summary = {
        "overall": overall.__dict__,
        "bootstrap_pairs_pearson_ci95": bootstrap_ci(
            pairs,
            "beta_measured",
            "beta_he",
            args.bootstrap,
            args.random_seed + 101,
            None,
        ),
        "bootstrap_gene_pearson_ci95": bootstrap_ci(
            pairs,
            "beta_measured",
            "beta_he",
            args.bootstrap,
            args.random_seed + 202,
            "gene",
        ),
        "permutation_two_sided_p": permutation_pvalue(x, y, overall.pearson_r, args.permutations, args.random_seed + 303),
        "n_genes": len(genes),
        "genes": genes,
        "include_intercept": args.include_intercept,
    }
    (out_dir / "beta_correlation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train matched SpaceTravLR seed models on MAGIC-smoothed measured and H&E-imputed "
            "10-gene panels, then correlate matched beta coefficients."
        )
    )
    p.add_argument("--paired-h5ad", type=Path, default=default_paired_h5ad())
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "example_run" / "spacetravlr_magic_beta_benchmark")
    p.add_argument("--measured-layer", default="measured_log1p")
    p.add_argument("--he-layer", default="imputed_count")
    p.add_argument("--n-genes", type=int, default=10)
    p.add_argument("--genes", default=None, help="Comma-separated gene list; defaults to top paired source correlations.")
    p.add_argument("--cluster-obs", default="cell_type")
    p.add_argument("--force-spatial-bins", action="store_true", help="Ignore existing cluster obs and use common spatial quantile bins.")
    p.add_argument("--cluster-grid", type=int, default=3)
    p.add_argument("--min-cells-per-cluster", type=int, default=8)
    p.add_argument("--magic-batch-obs", default=None)
    p.add_argument("--magic-knn", type=int, default=5)
    p.add_argument("--magic-knn-max", type=int, default=10)
    p.add_argument("--magic-decay", type=int, default=1)
    p.add_argument("--magic-t", type=int, default=3)
    p.add_argument("--magic-n-pca", type=int, default=100)
    p.add_argument("--magic-backend", choices=("uv", "python"), default="uv")
    p.add_argument("--spacetravlr-cmd", default=None, help='Command prefix, e.g. "cargo run --release --bin spacetravlr --".')
    p.add_argument("--prepare-only", action="store_true", help="Write MAGIC-smoothed inputs and configs without training or comparing betas.")
    p.add_argument("--skip-train", action="store_true", help="Prepare inputs and compare existing run directories without rerunning SpaceTravLR.")
    p.add_argument("--measured-run-dir", type=Path, default=None)
    p.add_argument("--he-run-dir", type=Path, default=None)
    p.add_argument("--network-data-dir", type=Path, default=None)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--radius", type=float, default=300.0)
    p.add_argument("--spatial-dim", type=int, default=64)
    p.add_argument("--contact-distance", type=float, default=30.0)
    p.add_argument("--weighted-ligand-scale-factor", type=float, default=1.0)
    p.add_argument("--l1-reg", type=float, default=1e-4)
    p.add_argument("--group-reg", type=float, default=1e-5)
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--include-intercept", action="store_true")
    p.add_argument("--zero-eps", type=float, default=1e-9)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=2000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paired_path = args.paired_h5ad.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not paired_path.exists():
        raise FileNotFoundError(f"Paired h5ad not found: {paired_path}")

    paired = ad.read_h5ad(paired_path)
    for layer in (args.measured_layer, args.he_layer):
        if layer != "X" and layer not in paired.layers:
            raise KeyError(f"Layer {layer!r} missing from {paired_path}")

    explicit_genes = parse_gene_list(args.genes)
    genes_df = choose_genes(paired, args.measured_layer, args.he_layer, args.n_genes, explicit_genes)
    genes = genes_df["gene"].astype(str).tolist()
    if len(genes) != args.n_genes and explicit_genes is None:
        raise ValueError(f"Expected {args.n_genes} selected genes; got {len(genes)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    genes_df.to_csv(out_dir / "selected_genes.csv", index=False)
    (out_dir / "selected_genes.txt").write_text("\n".join(genes) + "\n")

    inputs_dir = out_dir / "inputs"
    measured_h5ad = inputs_dir / "measured_magic_input.h5ad"
    he_h5ad = inputs_dir / "he_magic_input.h5ad"
    input_meta = {
        "paired_h5ad": str(paired_path),
        "measured": make_training_h5ad(
            paired,
            genes,
            args.measured_layer,
            "measured",
            args.cluster_obs,
            args,
            measured_h5ad,
        ),
        "he": make_training_h5ad(
            paired,
            genes,
            args.he_layer,
            "he",
            args.cluster_obs,
            args,
            he_h5ad,
        ),
    }
    (out_dir / "input_metadata.json").write_text(json.dumps(input_meta, indent=2, sort_keys=True))

    measured_run = args.measured_run_dir or (out_dir / "spacetravlr_measured_seed")
    he_run = args.he_run_dir or (out_dir / "spacetravlr_he_seed")
    measured_cfg = out_dir / "configs" / "measured_seed.toml"
    he_cfg = out_dir / "configs" / "he_seed.toml"
    write_config(measured_cfg, measured_h5ad, measured_run, genes, args.cluster_obs, args)
    write_config(he_cfg, he_h5ad, he_run, genes, args.cluster_obs, args)

    if args.prepare_only:
        print(f"Wrote prepared inputs and configs under {out_dir}")
        return 0

    if not args.skip_train:
        reset_run_dir(measured_run)
        reset_run_dir(he_run)
        run_spacetravlr("measured", measured_cfg, measured_run, args)
        run_spacetravlr("he", he_cfg, he_run, args)

    summary = compare_betas(measured_run, he_run, genes, args, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote beta benchmark outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
