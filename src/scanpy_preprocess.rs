//! Run a standard Scanpy QC + embedding pipeline in a **fresh uv-isolated** env each time.
//!
//! Spawns **`uv run --isolated`** with explicit `--with` packages, then **`python -`** reading this
//! module’s embedded source from **stdin** (no `.py` on disk, no shell scripts).
//! Requires [**uv**](https://docs.astral.sh/uv/) on `PATH`, or **`UV_BIN`**.
//! Child processes clear **`PYTHONPATH`** and set **`PYTHONNOUSERSITE=1`** so Conda/base site-packages
//! do not leak into the isolated env.
//! **`--no-cache-dir`** is passed to **`uv run`** unless **`cfg(test)`** is enabled or **`SPACETRAVLR_UV_ALLOW_CACHE`**
//! is set to **`1`**, **`true`**, or **`yes`** (trimmed, ASCII case-insensitive). Unit tests and opt-in callers can reuse wheels.
//! On failure, the same invocation is retried once with global **`uv --no-cache`** (disables uv’s cache entirely) to recover from bad cache state.
//! **`uv`** children also cap **`OPENBLAS_NUM_THREADS`**, **`OMP_NUM_THREADS`**, **`MKL_NUM_THREADS`**,
//! **`NUMEXPR_NUM_THREADS`**, and **`VECLIB_MAXIMUM_THREADS`** (default **`1`**) so inherited huge
//! thread counts do not trigger OpenBLAS “NUM_THREADS exceeded” / bad unallocation; override with
//! **`SPACETRAVLR_UV_BLAS_THREADS`** (1–64) or **`SPACETRAVLR_PRESERVE_BLAS_THREADS=1`** to skip.
//!
//! **Input `X`:** Before normalizing, the embedded Scanpy script infers whether **`X` is already
//! `log1p`-transformed** (Scanpy **`uns['log1p']`**, plus value heuristics) or **raw / linear
//! counts**. Raw path: `normalize_total` → **`layers["normalized_count"]`** → `log1p`. If **`X`**
//! is classified as already log-normalized, **`X` is not expm1’d or re-logged**: a dense copy is
//! stored in **`layers["normalized_count"]`** and **`X`** is left as-is for **`scale` / HVG /
//! PCA** (avoids double transform and Scanpy **`normalize_total`** warnings on log-like data).
//!
//! When **`layers["normalized_count"]`** and **`layers["imputed_count"]`** are already present,
//! the embed skips QC filters, spatial microns, normalization, scaling, HVG, PCA, neighbors,
//! UMAP, and MAGIC (layers are left unchanged aside from optional CSR conversion at write). If
//! **`obs["cell_type"]`** is missing, it runs **`sc.tl.leiden`** only (reusing **`obsp`** graphs
//! when **`connectivities`** exists, otherwise a compact **`sc.pp.neighbors`** on **`X`** or
//! **`obsm["X_pca"]`**), then copies Leiden labels into **`obs["cell_type"]`**.
//!
//! **Otherwise**, after QC and embedding, **clusterwise Markov imputation** on `normalized_count` uses
//! isolated **`uv`** with [**magic-impute**](https://pypi.org/project/magic-impute/) (one
//! **`MAGIC.fit_transform`** per `cell_type` or `leiden` label; optional **batch** column for
//! (cluster × batch) groups). The same **`uv`** step **CSR-sparsifies `X` and every `layers` matrix**
//! and writes the final file.
//!
//! Before writing, embedded Python and [`strip_heavy_training_artifacts_from_h5ad`] drop any
//! **`obsm` / `layers` / `obsp` / `uns`** entries that look like precomputed received or weighted
//! ligand tensors (and other heavy training caches), so processed outputs stay lean.
//!
//! Output is `<stem>_processed.h5ad` beside the input (pathlib `stem` rule, same as [`processed_h5ad_path`]).
//!
//! **Library entry points**
//!
//! | Step | Function | Output |
//! |------|----------|--------|
//! | Full QC → UMAP/Leiden → magic-impute | [`full_preprocess`] / [`full_preprocess_maybe_log`] | explicit **`--process-h5ad`** (or other callers); training **`FullPreprocess`** uses [`crate::rust_preprocess::rust_preprocess_h5ad_with_steps`] instead |
//! | Imputation + CSR only | [`imputed_count_from_normalized`] / [`magic_impute_and_attach_batch`] | `<stem>_imputed.h5ad` |
//! | `leiden` → `cell_type` (path helper) | [`cell_type_patch_h5ad_path`] | sibling filename |
//! | `leiden` → `cell_type` (write) | [`write_cell_type_from_leiden`] | caller-chosen path |
//! | MAGIC batch column resolution | [`resolve_magic_batch_obs_column`] | CLI / config wiring |
//! | Training auto-prep | [`ensure_training_adata_ready`] (pass **`[data].condition`** as MAGIC batch when set) | updates **`adata_path`** to the input when no prep is needed, else a stable file under **`output_dir/spacetravlr_prep/`** |
//!
//! **CLI:** **`spacetravlr --process-h5ad --h5ad …`** → [`full_preprocess_maybe_log`] (or reuse when output is fresh).
//! **`spacetravlr --impute --h5ad …`** → [`imputed_count_from_normalized`] (same imputation+CSR as the full pipeline).
//! With **`--condition`**, the same obs column name is used as the MAGIC batch axis unless **`--magic-batch-obs`** overrides.
//!
//! **Training:** [`ensure_training_adata_ready`] uses [`plan_training_prep`] to pick the minimal
//! fix; **`FullPreprocess`** / **`LayersLeidenAnnotate`** run the Rust pipeline ([`crate::rust_preprocess`]),
//! which **subsets to HVG (or all non-MT genes when `n_vars ≤ n_top_hvg`) before PCA and MAGIC** so prep outputs are HVG-wide for training.
//! Except **`FullPreprocess`** still uses Scanpy when a non-empty MAGIC **batch** obs column is set (Rust MAGIC is not batch-aware yet).
//! Patch / impute-only branches still use small **`uv`** Python steps. Derived `.h5ad` files go under **`spacetravlr_prep/`** (content-keyed from the source path + mtime). When **`[data].condition`** is set, imputation uses it as the MAGIC batch column. Opt out with **`--skip-auto-adata-prep`**.
//!
//! **Spatial coordinates:** After cell/gene filtering, when **`obsm['unscaled_spatial']`** is absent
//! and a 2D array exists under **`spatial`** / **`X_spatial`** / **`spatial_loc`**, the embedded
//! Scanpy script (constant **`SCANPY_BASIC_PREPROCESS_PY`** in this module) scales coordinates to
//! **microns** (median *k*-NN distance vs a species prior).
//! It stores the raw 2D matrix in **`obsm['unscaled_spatial']`** and **`obsm['spatial']`** in µm.
//! Then set **[`spatial.radius`](crate::config::SpatialConfig::radius)**,
//! **`contact_distance`**, and **[`cnn.spatial_feature_radius`](crate::config::CnnConfig::spatial_feature_radius)**
//! in **µm** to match.

use crate::config::expand_user_path;
use crate::rust_preprocess::{
    rust_preprocess_h5ad_with_steps, RustPreprocessParams, RustPreprocessSteps,
};
use anndata::{AnnData, AnnDataOp, AxisArraysOp, Backend, ElemCollectionOp};
use anndata_hdf5::H5;
use anyhow::{Context, bail};
use serde_json::json;
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const UV_WITH_ANNDATA: &[&str] = &["numpy<2", "anndata>=0.11"];
const UV_WITH_ATTACH: &[&str] = &["numpy<2", "anndata>=0.11", "scipy"];
const UV_WITH_MAGIC_IMPUTE: &[&str] = &["numpy<2", "anndata>=0.11", "scipy", "magic-impute>=3,<4"];
const UV_WITH_SCANPY: &[&str] = &[
    "numpy<2",
    "anndata>=0.11",
    "scipy",
    "scanpy",
    "h5py",
    "leidenalg",
    "igraph",
    "magic-impute>=3,<4",
];

// #region agent log
const DEBUG_AGENT_LOG_PATH: &str =
    "/Users/koush/Projects/SpaceTravLR_rust/.cursor/debug-f9143e.log";
const DEBUG_AGENT_SESSION: &str = "f9143e";

pub fn agent_debug_ndjson(
    hypothesis_id: &str,
    location: &str,
    message: &str,
    run_id: &str,
    data: serde_json::Value,
) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let line = json!({
        "sessionId": DEBUG_AGENT_SESSION,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": ts,
        "runId": run_id,
    });
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(DEBUG_AGENT_LOG_PATH)
    {
        let _ = writeln!(f, "{line}");
    }
}
// #endregion

/// Options for heuristic **`obsm['spatial']` → microns** in the Scanpy embed ([`full_preprocess_maybe_log`]).
#[derive(Clone, Debug, Default)]
pub struct SpatialMicronsOptions {
    pub skip: bool,
    pub species: String,
    pub target_median_nn_um: Option<f64>,
}

fn read_h5ad_var_names_for_infer(path: &Path) -> anyhow::Result<Vec<String>> {
    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(adata.var_names().into_vec())
}

/// When **`spatial_microns.species`** is empty and microns are not skipped, set it from
/// [`crate::network::infer_species`] on this `.h5ad`’s `var_names`.
pub fn resolve_spatial_microns_species_for_h5ad(
    mut opt: SpatialMicronsOptions,
    adata_path: &Path,
) -> anyhow::Result<SpatialMicronsOptions> {
    if opt.skip || !opt.species.trim().is_empty() {
        return Ok(opt);
    }
    let names = read_h5ad_var_names_for_infer(adata_path)?;
    let s = crate::network::infer_species(&names).ok_or_else(|| {
        anyhow::anyhow!(
            "could not infer human vs mouse from var_names in {}; set [data].spatial_species or --spatial-species (human|mouse)",
            adata_path.display()
        )
    })?;
    opt.species = s.to_string();
    Ok(opt)
}

fn uv_executable() -> OsString {
    env::var_os("UV_BIN")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "uv".into())
}

fn apply_blas_thread_caps_for_uv_child(cmd: &mut Command) {
    if env::var("SPACETRAVLR_PRESERVE_BLAS_THREADS")
        .ok()
        .as_deref()
        == Some("1")
    {
        return;
    }
    let n = env::var("SPACETRAVLR_UV_BLAS_THREADS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&k| (1..=64).contains(&k))
        .map(|k| k.to_string())
        .unwrap_or_else(|| "1".into());
    for k in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ] {
        cmd.env(k, &n);
    }
}

fn uv_command_base() -> Command {
    let mut c = Command::new(uv_executable());
    c.env_remove("PYTHONPATH");
    c.env("PYTHONNOUSERSITE", "1");
    apply_blas_thread_caps_for_uv_child(&mut c);
    c
}

fn uv_allow_wheel_cache() -> bool {
    if cfg!(test) {
        return true;
    }
    env::var("SPACETRAVLR_UV_ALLOW_CACHE")
        .map(|s| {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "yes")
        })
        .unwrap_or(false)
}

/// `uv run --isolated` + `python -` reading **`stdin_script`**, with **`--with`** deps, then **`argv`** after `python -`.
pub(crate) fn uv_python_stdin(
    with_packages: &[&str],
    stdin_script: &str,
    argv_after_dash: &[&str],
    capture_output: bool,
    spawn_hint: &str,
) -> anyhow::Result<String> {
    match uv_python_stdin_once(
        false,
        with_packages,
        stdin_script,
        argv_after_dash,
        capture_output,
        spawn_hint,
    ) {
        Ok(s) => Ok(s),
        Err(first) => uv_python_stdin_once(
            true,
            with_packages,
            stdin_script,
            argv_after_dash,
            capture_output,
            spawn_hint,
        )
        .map_err(|second| {
            anyhow::anyhow!(
                "uv run failed ({spawn_hint}); retry with `uv --no-cache` also failed.\n--- first attempt ---\n{first}\n--- retry ---\n{second}"
            )
        }),
    }
}

fn uv_python_stdin_once(
    global_no_cache: bool,
    with_packages: &[&str],
    stdin_script: &str,
    argv_after_dash: &[&str],
    capture_output: bool,
    spawn_hint: &str,
) -> anyhow::Result<String> {
    let (stdout, stderr) = if capture_output {
        (Stdio::piped(), Stdio::piped())
    } else {
        (Stdio::inherit(), Stdio::inherit())
    };

    let mut cmd = uv_command_base();
    #[cfg(test)]
    cmd.env("SPACETRAVLR_TEST_FAST_UV", "1");
    if global_no_cache {
        cmd.arg("--no-cache");
    }
    cmd.arg("run").arg("--isolated");
    if !uv_allow_wheel_cache() {
        cmd.arg("--no-cache-dir");
    }
    for w in with_packages {
        cmd.args(["--with", w]);
    }
    cmd.arg("python").arg("-");
    for a in argv_after_dash {
        cmd.arg(a);
    }
    cmd.stdin(Stdio::piped()).stdout(stdout).stderr(stderr);

    let mut child = cmd.spawn().with_context(|| {
        format!(
            "failed to spawn `{}` ({spawn_hint}; https://docs.astral.sh/uv/)",
            uv_executable().to_string_lossy()
        )
    })?;

    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(stdin_script.as_bytes())
        .with_context(|| format!("write embedded Python ({spawn_hint})"))?;

    if capture_output {
        let out = child
            .wait_with_output()
            .with_context(|| format!("wait for uv ({spawn_hint})"))?;
        let mut combined = String::new();
        combined.push_str(&String::from_utf8_lossy(&out.stdout));
        combined.push_str(&String::from_utf8_lossy(&out.stderr));
        if !out.status.success() {
            bail!(
                "uv run failed ({spawn_hint}): {}\n{}",
                out.status,
                combined.trim_end()
            );
        }
        Ok(combined)
    } else {
        let status = child
            .wait()
            .with_context(|| format!("wait for uv ({spawn_hint})"))?;
        if !status.success() {
            bail!("uv run failed ({spawn_hint}): {status}");
        }
        Ok(String::new())
    }
}

const MAGIC_CLUSTERWISE_IMPUTE_CSR_PY: &str = r#"
import os
import sys

import anndata as ad
import magic
import numpy as np
import scipy.sparse as sp

src, dst = sys.argv[1], sys.argv[2]
batch_col = sys.argv[3] if len(sys.argv) > 3 else None

a = ad.read_h5ad(src)
if "normalized_count" not in a.layers:
    sys.exit("expected layers['normalized_count']")
if "cell_type" in a.obs.columns:
    annot = "cell_type"
elif "leiden" in a.obs.columns:
    annot = "leiden"
else:
    sys.exit("clusterwise MAGIC needs obs column 'cell_type' or 'leiden'")
if batch_col is not None and batch_col not in a.obs.columns:
    sys.exit("magic batch obs column not found: %r" % (batch_col,))

nc = a.layers["normalized_count"]
if sp.issparse(nc):
    col_sum = np.asarray(nc.sum(axis=0)).ravel()
else:
    col_sum = np.sum(np.asarray(nc, dtype=np.float64), axis=0)
expressed = col_sum > 0.0
if not np.any(expressed):
    sys.exit("normalized_count has no genes with positive total expression")
if not np.all(expressed):
    a = a[:, np.flatnonzero(expressed)].copy()
    nc = a.layers["normalized_count"]
if sp.issparse(nc):
    X = nc.toarray().astype(np.float64)
else:
    X = np.asarray(nc, dtype=np.float64)

labels = np.array([str(x) for x in a.obs[annot].to_numpy()], dtype=object)
out = X.copy()
_fast_uv = os.environ.get("SPACETRAVLR_TEST_FAST_UV") == "1"
knn_def, knn_max_cap, t_magic, n_pca_cap = (
    (3, 6, 2, 12) if _fast_uv else (5, 10, 3, 100)
)


def magic_op_for_subset(n_sub, n_genes):
    knn = min(knn_def, max(1, n_sub - 1))
    knn_max = max(knn, min(knn_max_cap, n_sub - 1))
    # graphtools/sklearn PCA requires n_components < min(n_samples, n_features)
    pca_bound = min(int(n_sub), int(n_genes))
    n_pca_eff = min(n_pca_cap, max(1, pca_bound - 1))
    return magic.MAGIC(
        knn=knn,
        knn_max=knn_max,
        decay=1,
        t=t_magic,
        n_pca=n_pca_eff,
        verbose=0,
    )


def magic_impute_rows(sub, n_sub):
    gene_active = sub.sum(axis=0) > 0.0
    n_g = int(gene_active.sum())
    if n_g == 0:
        return np.asarray(sub, dtype=np.float64)
    op = magic_op_for_subset(n_sub, n_g)
    if n_g == sub.shape[1]:
        return np.asarray(op.fit_transform(sub, genes="all_genes"), dtype=np.float64)
    imp = np.asarray(sub, dtype=np.float64)
    sub_f = sub[:, gene_active]
    imp_f = np.asarray(op.fit_transform(sub_f, genes="all_genes"), dtype=np.float64)
    imp[:, gene_active] = imp_f
    return imp


if batch_col is None:
    for lab in np.unique(labels):
        m = labels == lab
        idx = np.flatnonzero(m)
        if idx.size < 2:
            continue
        sub = X[m]
        out[m] = magic_impute_rows(sub, int(idx.size))
else:
    batch_vals = np.array([str(x) for x in a.obs[batch_col].to_numpy()], dtype=object)
    keys = list({(labels[i], batch_vals[i]) for i in range(labels.size)})
    for ct, bt in keys:
        m = (labels == ct) & (batch_vals == bt)
        if not np.any(m):
            continue
        idx = np.flatnonzero(m)
        if idx.size < 2:
            continue
        sub = X[m]
        out[m] = magic_impute_rows(sub, int(idx.size))

a.layers["imputed_count"] = out


def _as_csr(m):
    if sp.issparse(m):
        return m.tocsr()
    return sp.csr_matrix(np.asarray(m, dtype=np.float64))


def _spacetravlr_strip_heavy_training_artifacts(a):
    def _lig_key(k):
        lk = str(k).lower()
        return "weighted_ligand" in lk or "received_ligand" in lk

    for k in list(a.obsm.keys()):
        if k in ("spacetravlr_spatial_features", "spacetravlr_spatial_maps_flat") or _lig_key(k):
            del a.obsm[k]
    for k in list(a.layers.keys()):
        if _lig_key(k):
            del a.layers[k]
    for k in list(a.obsp.keys()):
        if _lig_key(k):
            del a.obsp[k]
    for k in list(a.uns.keys()):
        if str(k).startswith("spacetravlr_cache_") or _lig_key(k):
            del a.uns[k]


_spacetravlr_strip_heavy_training_artifacts(a)
a.X = _as_csr(a.X)
for k in list(a.layers.keys()):
    a.layers[k] = _as_csr(a.layers[k])
a.write_h5ad(dst)
print("wrote", dst)
"#;

const CSR_LAYERS_PY: &str = r#"
import sys
import anndata as ad
import numpy as np
import scipy.sparse as sp

src, dst = sys.argv[1], sys.argv[2]
a = ad.read_h5ad(src)


def _spacetravlr_strip_heavy_training_artifacts(adata):
    def _lig_key(k):
        lk = str(k).lower()
        return "weighted_ligand" in lk or "received_ligand" in lk

    for k in list(adata.obsm.keys()):
        if k in ("spacetravlr_spatial_features", "spacetravlr_spatial_maps_flat") or _lig_key(k):
            del adata.obsm[k]
    for k in list(adata.layers.keys()):
        if _lig_key(k):
            del adata.layers[k]
    for k in list(adata.obsp.keys()):
        if _lig_key(k):
            del adata.obsp[k]
    for k in list(adata.uns.keys()):
        if str(k).startswith("spacetravlr_cache_") or _lig_key(k):
            del adata.uns[k]


_spacetravlr_strip_heavy_training_artifacts(a)


def _as_csr(m):
    if sp.issparse(m):
        return m.tocsr()
    return sp.csr_matrix(np.asarray(m, dtype=np.float64))


a.X = _as_csr(a.X)
for k in list(a.layers.keys()):
    a.layers[k] = _as_csr(a.layers[k])
a.write_h5ad(dst)
print("wrote", dst)
"#;

/// Embedded Scanpy + QC pipeline (stdin to `uv run python -`). Includes **spatial → microns**:
/// median distance to the 4th nearest neighbor (`knn_k = 4`) over finite 2D points; scale
/// `s = target_median_nn_um / d_raw` unless `d_raw` lies in `[0.5×, 2×] target` (“already micron-like”,
/// then `s = 1`). Default targets: human **13** µm, mouse **10.5** µm. Provenance:
/// `uns['spacetravlr_spatial_microns']`.
const SCANPY_BASIC_PREPROCESS_PY: &str = r#"
import os, shutil, sys, tempfile
from pathlib import Path

import h5py
import magic
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata._io.specs.registry import IORegistryError
from datetime import datetime, timezone
from scipy.spatial import cKDTree


def _strip_uns_log1p(p: str) -> None:
    with h5py.File(p, "a") as f:
        if "uns" in f and "log1p" in f["uns"]:
            del f["uns"]["log1p"]


def _read_h5ad(path: str):
    try:
        return sc.read_h5ad(path)
    except IORegistryError:
        pass
    fd, tmp = tempfile.mkstemp(suffix=".h5ad")
    os.close(fd)
    try:
        shutil.copy2(path, tmp)
        _strip_uns_log1p(tmp)
        return sc.read_h5ad(tmp)
    finally:
        os.unlink(tmp)


def _infer_x_is_log1p(adata) -> bool:
    if "log1p" in adata.uns:
        return True
    X = adata.X
    if sp.issparse(X):
        d = X.data.astype(np.float64)
        if d.size == 0:
            return True
        rng = np.random.default_rng(0)
        sample = d if d.size <= 100000 else rng.choice(d, 100000, replace=False)
    else:
        flat = np.asarray(X, dtype=np.float64).ravel()
        if flat.size == 0:
            return True
        rng = np.random.default_rng(0)
        sample = flat if flat.size <= 100000 else rng.choice(flat, 100000, replace=False)
    mx = float(np.max(sample))
    med = float(np.median(sample))
    frac_int = float(np.mean(np.abs(sample - np.round(sample)) < 1e-5))
    if mx > 30:
        return False
    if frac_int > 0.72 and mx > 12:
        return False
    if mx <= 12 and med <= 3.5 and frac_int < 0.55:
        return True
    return False


def _spacetravlr_strip_heavy_training_artifacts(adata):
    def _lig_key(k):
        lk = str(k).lower()
        return "weighted_ligand" in lk or "received_ligand" in lk

    for k in list(adata.obsm.keys()):
        if k in ("spacetravlr_spatial_features", "spacetravlr_spatial_maps_flat") or _lig_key(k):
            del adata.obsm[k]
    for k in list(adata.layers.keys()):
        if _lig_key(k):
            del adata.layers[k]
    for k in list(adata.obsp.keys()):
        if _lig_key(k):
            del adata.obsp[k]
    for k in list(adata.uns.keys()):
        if str(k).startswith("spacetravlr_cache_") or _lig_key(k):
            del adata.uns[k]


def _as_csr_magic(m):
    if sp.issparse(m):
        return m.tocsr()
    return sp.csr_matrix(np.asarray(m, dtype=np.float64))


src = Path(sys.argv[1])
dest_final = Path(sys.argv[2])
if src.suffix.lower() != ".h5ad":
    sys.exit("expected path ending in .h5ad")
_skip_microns = len(sys.argv) > 3 and sys.argv[3] == "1"
_species_m = (sys.argv[4] if len(sys.argv) > 4 else "human").lower().strip()
_target_um_s = sys.argv[5] if len(sys.argv) > 5 else ""
_batch_magic_arg = sys.argv[6] if len(sys.argv) > 6 else "-"
batch_col = (
    None
    if _batch_magic_arg.strip() in ("", "-")
    else _batch_magic_arg.strip()
)
adata = _read_h5ad(str(src))
_fast_uv = os.environ.get("SPACETRAVLR_TEST_FAST_UV") == "1"
_layers_ready = (
    "normalized_count" in adata.layers and "imputed_count" in adata.layers
)
if _layers_ready:
    print(
        "spacetravlr_preprocess: layers['normalized_count'] and layers['imputed_count'] present; skipping filters, spatial microns, normalize/HVG/PCA/UMAP/MAGIC",
        file=sys.stderr,
    )
    if "cell_type" not in adata.obs.columns:
        if "leiden" not in adata.obs.columns:
            if "connectivities" in adata.obsp:
                try:
                    import igraph  # noqa: F401
                    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
                except ImportError:
                    sc.tl.leiden(adata)
            elif "X_pca" in adata.obsm:
                no = int(adata.n_obs) - 1
                n_nb = min(10, max(2, no)) if _fast_uv else min(15, max(2, no))
                sc.pp.neighbors(adata, n_neighbors=n_nb, use_rep="X_pca")
                try:
                    import igraph  # noqa: F401
                    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
                except ImportError:
                    sc.tl.leiden(adata)
            else:
                no = int(adata.n_obs) - 1
                n_nb = min(10, max(2, no)) if _fast_uv else min(15, max(2, no))
                sc.pp.neighbors(adata, n_neighbors=n_nb)
                try:
                    import igraph  # noqa: F401
                    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
                except ImportError:
                    sc.tl.leiden(adata)
        adata.obs["cell_type"] = adata.obs["leiden"].astype(str)
    a = adata
    _spacetravlr_strip_heavy_training_artifacts(a)
    a.X = _as_csr_magic(a.X)
    for k in list(a.layers.keys()):
        a.layers[k] = _as_csr_magic(a.layers[k])
    a.write_h5ad(dest_final)
    print("wrote_processed", dest_final)
    raise SystemExit(0)

sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)


def _spacetravlr_default_target_nn_um(species):
    if species == "mouse":
        return 10.5
    if species == "human":
        return 13.0
    raise ValueError("spatial species must be human or mouse")


def _spacetravlr_resolve_spatial_xy(a):
    for key in ("spatial", "X_spatial", "spatial_loc"):
        if key not in a.obsm:
            continue
        arr = np.asarray(a.obsm[key], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 2:
            return key, arr[:, :2].copy()
    return None, None


def _spacetravlr_median_knn_raw(xy, knn_k=4):
    mask = np.all(np.isfinite(xy), axis=1)
    pts = np.asarray(xy[mask], dtype=np.float64)
    n = pts.shape[0]
    if n < knn_k + 1:
        return float("nan")
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=knn_k + 1)
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
    return float(np.median(dists[:, knn_k]))


def _spacetravlr_maybe_obsm_spatial_microns(a):
    if _skip_microns:
        return
    if "unscaled_spatial" in a.obsm:
        return
    sk, xy = _spacetravlr_resolve_spatial_xy(a)
    if sk is None:
        return
    if _species_m not in ("human", "mouse"):
        print(
            "spacetravlr_preprocess spatial_microns skip bad species",
            _species_m,
            file=sys.stderr,
        )
        return
    try:
        if _target_um_s.strip():
            target = float(_target_um_s)
        else:
            target = _spacetravlr_default_target_nn_um(_species_m)
    except ValueError as e:
        print("spacetravlr_preprocess spatial_microns skip", e, file=sys.stderr)
        return
    d_raw = _spacetravlr_median_knn_raw(xy, 4)
    if not np.isfinite(d_raw) or d_raw <= 0:
        print(
            "spacetravlr_preprocess spatial_microns skip insufficient cells for kNN",
            file=sys.stderr,
        )
        return
    lo, hi = 0.5 * target, 2.0 * target
    if lo <= d_raw <= hi:
        scale = 1.0
        already = True
    else:
        scale = target / d_raw
        already = False
    scaled = xy * scale
    a.obsm["unscaled_spatial"] = np.asarray(xy, dtype=np.float64)
    a.obsm["spatial"] = np.asarray(scaled, dtype=np.float64)
    a.uns["spacetravlr_spatial_microns"] = {
        "applied": True,
        "scale": float(scale),
        "species": _species_m,
        "target_median_nn_um": float(target),
        "knn_k": 4,
        "median_knn_raw": float(d_raw),
        "source_key": sk,
        "already_micron_like": bool(already),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(
        "spacetravlr_preprocess spatial_microns scale",
        scale,
        "species",
        _species_m,
        file=sys.stderr,
    )


_spacetravlr_maybe_obsm_spatial_microns(adata)
adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))

x_is_log1p = _infer_x_is_log1p(adata)
print("spacetravlr_preprocess x_space", "log1p" if x_is_log1p else "counts", file=sys.stderr)

if x_is_log1p:
    print(
        "spacetravlr_preprocess: X classified as log1p-space; copying X -> layers['normalized_count'] (no expm1/normalize_total/log1p)",
        file=sys.stderr,
    )
    nc = adata.X.copy()
    if hasattr(nc, "toarray"):
        adata.layers["normalized_count"] = nc.toarray()
    else:
        adata.layers["normalized_count"] = np.asarray(nc, dtype=np.float64)
else:
    adata.layers["raw_count"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    nc = adata.X.copy()
    if hasattr(nc, "toarray"):
        adata.layers["normalized_count"] = nc.toarray()
    else:
        adata.layers["normalized_count"] = np.asarray(nc, dtype=np.float64)
    sc.pp.log1p(adata)

sc.pp.scale(adata, max_value=10)
nv = int(adata.n_vars) - 1
no = int(adata.n_obs) - 1
if _fast_uv:
    n_hvg = min(400, max(3, nv))
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    n_pca = min(20, max(2, nv), max(2, no))
    sc.pp.pca(adata, n_comps=n_pca)
    n_nb = min(10, max(2, no))
    sc.pp.neighbors(adata, n_neighbors=n_nb)
else:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
sc.tl.umap(adata)
try:
    import igraph  # noqa: F401
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
except ImportError:
    sc.tl.leiden(adata)
if "cell_type" not in adata.obs.columns:
    adata.obs["cell_type"] = adata.obs["leiden"].astype(str)

a = adata
if "normalized_count" not in a.layers:
    sys.exit("expected layers['normalized_count']")
if "cell_type" in a.obs.columns:
    annot = "cell_type"
elif "leiden" in a.obs.columns:
    annot = "leiden"
else:
    sys.exit("clusterwise MAGIC needs obs column 'cell_type' or 'leiden'")
if batch_col is not None and batch_col not in a.obs.columns:
    sys.exit("magic batch obs column not found: %r" % (batch_col,))

nc = a.layers["normalized_count"]
if sp.issparse(nc):
    col_sum = np.asarray(nc.sum(axis=0)).ravel()
else:
    col_sum = np.sum(np.asarray(nc, dtype=np.float64), axis=0)
expressed = col_sum > 0.0
if not np.any(expressed):
    sys.exit("normalized_count has no genes with positive total expression")
if not np.all(expressed):
    a = a[:, np.flatnonzero(expressed)].copy()
    nc = a.layers["normalized_count"]
if sp.issparse(nc):
    X = nc.toarray().astype(np.float64)
else:
    X = np.asarray(nc, dtype=np.float64)

labels = np.array([str(x) for x in a.obs[annot].to_numpy()], dtype=object)
out = X.copy()
_fast_uv = os.environ.get("SPACETRAVLR_TEST_FAST_UV") == "1"
knn_def, knn_max_cap, t_magic, n_pca_cap = (
    (3, 6, 2, 12) if _fast_uv else (5, 10, 3, 100)
)


def magic_op_for_subset(n_sub, n_genes):
    knn = min(knn_def, max(1, n_sub - 1))
    knn_max = max(knn, min(knn_max_cap, n_sub - 1))
    pca_bound = min(int(n_sub), int(n_genes))
    n_pca_eff = min(n_pca_cap, max(1, pca_bound - 1))
    return magic.MAGIC(
        knn=knn,
        knn_max=knn_max,
        decay=1,
        t=t_magic,
        n_pca=n_pca_eff,
        verbose=0,
    )


def magic_impute_rows(sub, n_sub):
    gene_active = sub.sum(axis=0) > 0.0
    n_g = int(gene_active.sum())
    if n_g == 0:
        return np.asarray(sub, dtype=np.float64)
    op = magic_op_for_subset(n_sub, n_g)
    if n_g == sub.shape[1]:
        return np.asarray(op.fit_transform(sub, genes="all_genes"), dtype=np.float64)
    imp = np.asarray(sub, dtype=np.float64)
    sub_f = sub[:, gene_active]
    imp_f = np.asarray(op.fit_transform(sub_f, genes="all_genes"), dtype=np.float64)
    imp[:, gene_active] = imp_f
    return imp


if batch_col is None:
    for lab in np.unique(labels):
        m = labels == lab
        idx = np.flatnonzero(m)
        if idx.size < 2:
            continue
        sub = X[m]
        out[m] = magic_impute_rows(sub, int(idx.size))
else:
    batch_vals = np.array([str(x) for x in a.obs[batch_col].to_numpy()], dtype=object)
    keys = list({(labels[i], batch_vals[i]) for i in range(labels.size)})
    for ct, bt in keys:
        m = (labels == ct) & (batch_vals == bt)
        if not np.any(m):
            continue
        idx = np.flatnonzero(m)
        if idx.size < 2:
            continue
        sub = X[m]
        out[m] = magic_impute_rows(sub, int(idx.size))

a.layers["imputed_count"] = out


_spacetravlr_strip_heavy_training_artifacts(a)
a.X = _as_csr_magic(a.X)
for k in list(a.layers.keys()):
    a.layers[k] = _as_csr_magic(a.layers[k])
a.write_h5ad(dest_final)
print("wrote_processed", dest_final)
"#;

const ADD_CELL_TYPE_FROM_LEIDEN_PY: &str = r#"
import sys
import anndata as ad

inp, outp = sys.argv[1], sys.argv[2]
a = ad.read_h5ad(inp)
if "cell_type" in a.obs.columns:
    pass
elif "leiden" in a.obs.columns:
    a.obs["cell_type"] = a.obs["leiden"].astype(str)
else:
    sys.exit("cannot set cell_type: obs has no leiden column")


def _spacetravlr_strip_heavy_training_artifacts(adata):
    def _lig_key(k):
        lk = str(k).lower()
        return "weighted_ligand" in lk or "received_ligand" in lk

    for k in list(adata.obsm.keys()):
        if k in ("spacetravlr_spatial_features", "spacetravlr_spatial_maps_flat") or _lig_key(k):
            del adata.obsm[k]
    for k in list(adata.layers.keys()):
        if _lig_key(k):
            del adata.layers[k]
    for k in list(adata.obsp.keys()):
        if _lig_key(k):
            del adata.obsp[k]
    for k in list(adata.uns.keys()):
        if str(k).startswith("spacetravlr_cache_") or _lig_key(k):
            del adata.uns[k]


_spacetravlr_strip_heavy_training_artifacts(a)
a.write_h5ad(outp)
print("wrote", outp)
"#;

/// Obs/layer flags for training auto-prep ([`ensure_training_adata_ready`]).
#[derive(Clone, Debug)]
pub struct AdataTrainingReadiness {
    pub has_cell_type: bool,
    pub has_leiden: bool,
    pub has_normalized_count: bool,
    pub has_imputed_count: bool,
}

/// Inspect `.h5ad` for columns and layers used before training.
pub fn probe_adata_training_readiness(path: &Path) -> anyhow::Result<AdataTrainingReadiness> {
    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    let obs = adata.read_obs().map_err(|e| anyhow::anyhow!("{}", e))?;
    let has_cell_type = obs.column("cell_type").is_ok();
    let has_leiden = obs.column("leiden").is_ok();
    let has_normalized_count = adata.layers().get("normalized_count").is_some();
    let has_imputed_count = adata.layers().get("imputed_count").is_some();
    adata.close()?;
    Ok(AdataTrainingReadiness {
        has_cell_type,
        has_leiden,
        has_normalized_count,
        has_imputed_count,
    })
}

/// Training auto-prep is unnecessary when **`cell_type`**, **`layers["imputed_count"]`**, and
/// **`layers["normalized_count"]`** are all present.
pub fn training_h5ad_is_fully_prepared(path: &Path) -> anyhow::Result<bool> {
    let r = probe_adata_training_readiness(path)?;
    Ok(r.has_cell_type && r.has_imputed_count && r.has_normalized_count)
}

/// **`candidate`** can replace a full preprocess when it exists, is at least as new as **`source`**
/// (mtime), and [`training_h5ad_is_fully_prepared`] is true for **`candidate`**.
pub fn prepared_training_output_is_reusable(
    source: &Path,
    candidate: &Path,
) -> anyhow::Result<bool> {
    if !candidate.is_file() {
        return Ok(false);
    }
    let src_mtime = source.metadata()?.modified()?;
    let out_mtime = candidate.metadata()?.modified()?;
    if out_mtime < src_mtime {
        return Ok(false);
    }
    training_h5ad_is_fully_prepared(candidate)
}

fn sibling_h5ad_with_suffix(adata: &Path, suffix_before_ext: &str) -> anyhow::Result<PathBuf> {
    let name = adata
        .file_name()
        .and_then(|n| n.to_str())
        .context("AnnData path has no file name")?;
    if !name.to_lowercase().ends_with(".h5ad") {
        bail!("expected .h5ad");
    }
    let stem = &name[..name.len() - 5];
    let mut out = adata.to_path_buf();
    out.set_file_name(format!("{stem}{suffix_before_ext}.h5ad"));
    Ok(out)
}

/// `<stem>_imputed.h5ad` beside `adata_in` (impute-only pipeline output).
pub fn imputed_only_output_path(adata_in: &Path) -> anyhow::Result<PathBuf> {
    sibling_h5ad_with_suffix(adata_in, "_imputed")
}

/// `<stem>_with_cell_type.h5ad` beside **`adata_in`** (Leiden → `cell_type` patch).
pub fn cell_type_patch_h5ad_path(adata_in: &Path) -> anyhow::Result<PathBuf> {
    sibling_h5ad_with_suffix(adata_in, "_with_cell_type")
}

fn write_cell_type_from_leiden_impl(
    in_h5ad: &Path,
    out_h5ad: &Path,
    capture_output: bool,
) -> anyhow::Result<String> {
    let i = in_h5ad
        .to_str()
        .with_context(|| format!("input path must be UTF-8: {}", in_h5ad.display()))?;
    let o = out_h5ad
        .to_str()
        .with_context(|| format!("output path must be UTF-8: {}", out_h5ad.display()))?;
    uv_python_stdin(
        UV_WITH_ANNDATA,
        ADD_CELL_TYPE_FROM_LEIDEN_PY,
        &[i, o],
        capture_output,
        "add cell_type from leiden",
    )
}

/// Copy **`obs['leiden']`** into **`obs['cell_type']`** when the latter is missing; writes **`out_h5ad`**.
pub fn write_cell_type_from_leiden(in_h5ad: &Path, out_h5ad: &Path) -> anyhow::Result<()> {
    write_cell_type_from_leiden_impl(in_h5ad, out_h5ad, false).map(|_| ())
}

/// Like [`write_cell_type_from_leiden`], but captures uv output and prints it to **stderr** (stdout stays clean).
pub fn write_cell_type_from_leiden_echo_uv_to_stderr(
    in_h5ad: &Path,
    out_h5ad: &Path,
) -> anyhow::Result<()> {
    let log = write_cell_type_from_leiden_impl(in_h5ad, out_h5ad, true)?;
    eprint!("{log}");
    Ok(())
}

/// `{stem}_processed.h5ad` under **`output_dir`** (stem from [`crate::config::canonical_training_prep_stem`]).
///
/// Used for explicit CLI outputs (e.g. **`--process-h5ad`**). Training auto-prep uses
/// [`training_prep_h5ad_path`] instead so the input file is not mirrored at the run root.
pub fn training_processed_h5ad_path(output_dir: &Path, stem: &str) -> PathBuf {
    output_dir.join(format!("{stem}_processed.h5ad"))
}

pub const TRAINING_PREP_SUBDIR: &str = "spacetravlr_prep";

pub fn training_prep_subdir(output_dir: &Path) -> PathBuf {
    output_dir.join(TRAINING_PREP_SUBDIR)
}

fn stem_sanitized_for_filename(stem: &str) -> String {
    stem.chars()
        .map(|c| match c {
            '/' | '\\' | ':' => '_',
            c if c.is_ascii_alphanumeric() || c == '-' || c == '_' => c,
            _ => '_',
        })
        .collect()
}

pub fn prep_cache_key(source: &Path) -> anyhow::Result<u64> {
    let canon = source
        .canonicalize()
        .unwrap_or_else(|_| source.to_path_buf());
    let nanos = std::fs::metadata(&canon)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut h = DefaultHasher::new();
    canon.to_string_lossy().hash(&mut h);
    nanos.hash(&mut h);
    Ok(h.finish())
}

pub fn training_prep_h5ad_path(
    output_dir: &Path,
    source: &Path,
    stem: &str,
    role: &str,
) -> anyhow::Result<PathBuf> {
    let dir = training_prep_subdir(output_dir);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("mkdir training prep {}", dir.display()))?;
    let key = prep_cache_key(source)?;
    let stem_tok = stem_sanitized_for_filename(stem);
    Ok(dir.join(format!("{stem_tok}_{key:016x}_{role}.h5ad")))
}

/// Removes precomputed CNN spatial tensors, received-/weighted-ligand matrices, and `spacetravlr_cache_*` `uns`
/// keys from an on-disk `.h5ad`. Keeps coordinate `obsm` (`spatial`, `X_spatial`, `spatial_loc`, `unscaled_spatial`)
/// and standard expression layers. No-op if the path is missing or not `.h5ad`.
pub fn strip_heavy_training_artifacts_from_h5ad(path: &Path) -> anyhow::Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let is_h5ad = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("h5ad"))
        .unwrap_or(false);
    if !is_h5ad {
        return Ok(());
    }

    let adata = AnnData::<H5>::open(H5::open_rw(path)?)
        .with_context(|| format!("open .h5ad rw for artifact strip: {}", path.display()))?;

    fn ligand_artifact_key(k: &str) -> bool {
        let kl = k.to_ascii_lowercase();
        kl.contains("weighted_ligand") || kl.contains("received_ligand")
    }

    for key in adata.obsm().keys() {
        if key == "spacetravlr_spatial_features"
            || key == "spacetravlr_spatial_maps_flat"
            || ligand_artifact_key(&key)
        {
            adata.obsm().remove(&key)?;
        }
    }
    for key in adata.layers().keys() {
        if ligand_artifact_key(&key) {
            adata.layers().remove(&key)?;
        }
    }
    for key in adata.obsp().keys() {
        if ligand_artifact_key(&key) {
            adata.obsp().remove(&key)?;
        }
    }
    for key in adata.uns().keys() {
        if key.starts_with("spacetravlr_cache_") || ligand_artifact_key(&key) {
            adata.uns().remove(&key)?;
        }
    }

    adata.close()?;
    Ok(())
}

/// `{stem}_imputed.h5ad` under **`output_dir`**.
pub fn training_imputed_h5ad_path(output_dir: &Path, stem: &str) -> PathBuf {
    output_dir.join(format!("{stem}_imputed.h5ad"))
}

/// `{stem}_with_cell_type.h5ad` under **`output_dir`**.
pub fn training_cell_type_patch_h5ad_path(output_dir: &Path, stem: &str) -> PathBuf {
    output_dir.join(format!("{stem}_with_cell_type.h5ad"))
}

/// CSR-sparsify **`X`** and all **`layers`** via isolated **`uv`** (same stack as attach step).
pub fn write_h5ad_csr_layers_uv(
    from_h5ad: &Path,
    to_h5ad: &Path,
    capture_output: bool,
) -> anyhow::Result<Option<String>> {
    let s = from_h5ad
        .to_str()
        .with_context(|| format!("source .h5ad must be UTF-8: {}", from_h5ad.display()))?;
    let d = to_h5ad
        .to_str()
        .with_context(|| format!("dest .h5ad must be UTF-8: {}", to_h5ad.display()))?;
    let log = uv_python_stdin(
        UV_WITH_ATTACH,
        CSR_LAYERS_PY,
        &[s, d],
        capture_output,
        "csr layers",
    )?;
    Ok(if capture_output { Some(log) } else { None })
}

/// True when **`X`** and every present **`layers[...]`** matrix is CSR/CSC (sparse), not dense.
///
/// Uses the HDF5 `encoding-type` attribute (metadata only) — does **not** read matrix data.
pub fn adata_x_and_layers_are_csr(path: &Path) -> anyhow::Result<bool> {
    use anndata::ArrayElemOp;
    use anndata::backend::DataType;

    fn dtype_is_sparse(dt: Option<DataType>) -> bool {
        matches!(dt, Some(DataType::CsrMatrix(_) | DataType::CscMatrix(_)))
    }

    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    if !dtype_is_sparse(adata.x().dtype()) {
        adata.close()?;
        return Ok(false);
    }
    for name in adata.layers().keys() {
        let Some(elem) = adata.layers().get(name.as_str()) else {
            continue;
        };
        if !dtype_is_sparse(elem.dtype()) {
            adata.close()?;
            return Ok(false);
        }
    }
    adata.close()?;
    Ok(true)
}

/// Writes **`to_h5ad`** with CSR **`X`/layers; if **`from_h5ad == to_h5ad`**, uses a temp file and rename.
pub fn ensure_h5ad_csr_layers_on_path(
    from_h5ad: &Path,
    to_h5ad: &Path,
    capture_output: bool,
) -> anyhow::Result<()> {
    if from_h5ad == to_h5ad {
        if adata_x_and_layers_are_csr(from_h5ad)? {
            return Ok(());
        }
        let tmp = from_h5ad.with_extension("h5ad.csr_work");
        let _ = std::fs::remove_file(&tmp);
        write_h5ad_csr_layers_uv(from_h5ad, &tmp, capture_output)?;
        std::fs::rename(&tmp, from_h5ad)
            .with_context(|| format!("rename {} -> {}", tmp.display(), from_h5ad.display()))?;
        return Ok(());
    }
    let _ = std::fs::remove_file(to_h5ad);
    write_h5ad_csr_layers_uv(from_h5ad, to_h5ad, capture_output)?;
    if !to_h5ad.is_file() {
        bail!("csr materialize did not create {}", to_h5ad.display());
    }
    Ok(())
}

fn ensure_magic_batch_obs_column_exists(source_h5ad: &Path, batch_col: &str) -> anyhow::Result<()> {
    let adata =
        AnnData::<H5>::open(H5::open(source_h5ad)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    let obs = adata.read_obs().map_err(|e| anyhow::anyhow!("{}", e))?;
    if obs.column(batch_col).is_err() {
        bail!("magic batch obs column not found: {:?}", batch_col);
    }
    adata.close().map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(())
}

/// Writes **`dest_h5ad`**: clusterwise [**magic-impute**](https://pypi.org/project/magic-impute/) on
/// **`layers["normalized_count"]`** (one MAGIC fit per `cell_type` or `leiden` label), then CSR-sparsify **`X`** and all **`layers`** (isolated **`uv`**).
///
/// When **`obs_batch_column`** is set, imputation matches the batch-clusterwise pattern: one MAGIC fit
/// per distinct **(annotation × batch)** pair in **`adata.obs`**, then rows are written back in obs order.
///
/// **`strip_heavy_after`**: when **true**, run [`strip_heavy_training_artifacts_from_h5ad`] on **`dest_h5ad`**
/// after the uv write (extra HDF5 pass; auto-prep callers pass **false** to keep training fast).
pub fn magic_impute_and_attach_batch(
    source_h5ad: &Path,
    dest_h5ad: &Path,
    obs_batch_column: Option<&str>,
    capture_output: bool,
    strip_heavy_after: bool,
) -> anyhow::Result<String> {
    let p = source_h5ad
        .to_str()
        .with_context(|| format!("source .h5ad must be UTF-8: {}", source_h5ad.display()))?;
    let f = dest_h5ad
        .to_str()
        .with_context(|| format!("dest .h5ad must be UTF-8: {}", dest_h5ad.display()))?;
    let batch_owned = obs_batch_column
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    if let Some(b) = &batch_owned {
        ensure_magic_batch_obs_column_exists(source_h5ad, b)?;
    }
    let log = match &batch_owned {
        None => uv_python_stdin(
            UV_WITH_MAGIC_IMPUTE,
            MAGIC_CLUSTERWISE_IMPUTE_CSR_PY,
            &[p, f],
            capture_output,
            "magic impute clusterwise + csr",
        )?,
        Some(b) => uv_python_stdin(
            UV_WITH_MAGIC_IMPUTE,
            MAGIC_CLUSTERWISE_IMPUTE_CSR_PY,
            &[p, f, b.as_str()],
            capture_output,
            "magic impute clusterwise + csr",
        )?,
    };

    if !dest_h5ad.is_file() {
        bail!(
            "expected output .h5ad missing after magic impute: {}",
            dest_h5ad.display()
        );
    }

    if strip_heavy_after {
        strip_heavy_training_artifacts_from_h5ad(dest_h5ad)?;
    }

    Ok(log)
}

/// Like [`magic_impute_and_attach_batch`] with no batch column (cluster annotation only).
pub fn magic_impute_and_attach(
    source_h5ad: &Path,
    dest_h5ad: &Path,
    capture_output: bool,
) -> anyhow::Result<String> {
    magic_impute_and_attach_batch(source_h5ad, dest_h5ad, None, capture_output, true)
}

/// Clusterwise **magic-impute** on **`layers["normalized_count"]`** + CSR write → **`<stem>_imputed.h5ad`**.
pub fn imputed_count_from_normalized(adata_in: &Path) -> anyhow::Result<PathBuf> {
    let out = imputed_only_output_path(adata_in)?;
    magic_impute_and_attach(adata_in, &out, false)?;
    Ok(out)
}

/// Alias for [`imputed_count_from_normalized`].
pub fn run_imputed_layer_only_pipeline(adata_in: &Path) -> anyhow::Result<PathBuf> {
    imputed_count_from_normalized(adata_in)
}

/// Resolves the `adata.obs` column name for **batch-aware** clusterwise MAGIC.
///
/// **`magic_batch_obs`** wins when non-empty after trim; otherwise **`condition_column`** (e.g. CLI
/// `--condition` or config **`[data].condition`**) is used.
pub fn resolve_magic_batch_obs_column(
    magic_batch_obs: Option<&str>,
    condition_column: Option<&str>,
) -> Option<String> {
    let trimmed = |s: &str| {
        let t = s.trim();
        (!t.is_empty()).then(|| t.to_string())
    };
    magic_batch_obs
        .and_then(trimmed)
        .or_else(|| condition_column.and_then(trimmed))
}

/// Minimal action for [`ensure_training_adata_ready`], derived from [`probe_adata_training_readiness`].
#[derive(Debug, PartialEq, Eq)]
pub enum TrainingPrepPlan {
    Noop,
    PatchCellType {
        out: PathBuf,
    },
    ImputeOnly {
        out: PathBuf,
    },
    PatchThenImpute {
        patched: PathBuf,
        out: PathBuf,
    },
    /// Both expression layers exist; add **`obs['cell_type']`** via Leiden only (no MAGIC/UMAP).
    LayersLeidenAnnotate {
        out: PathBuf,
    },
    FullPreprocess {
        out: PathBuf,
    },
}

pub fn plan_training_prep(
    r: &AdataTrainingReadiness,
    output_dir: &Path,
    source: &Path,
    stem: &str,
) -> anyhow::Result<TrainingPrepPlan> {
    if r.has_cell_type && r.has_imputed_count && r.has_normalized_count {
        return Ok(TrainingPrepPlan::Noop);
    }
    let can_impute_only = r.has_normalized_count && (r.has_cell_type || r.has_leiden);
    if !r.has_cell_type && r.has_leiden && r.has_normalized_count && r.has_imputed_count {
        return Ok(TrainingPrepPlan::PatchCellType {
            out: training_prep_h5ad_path(output_dir, source, stem, "celltype")?,
        });
    }
    if r.has_imputed_count && r.has_normalized_count && !r.has_cell_type && !r.has_leiden {
        return Ok(TrainingPrepPlan::LayersLeidenAnnotate {
            out: training_prep_h5ad_path(output_dir, source, stem, "layers_leiden")?,
        });
    }
    if can_impute_only && r.has_cell_type && !r.has_imputed_count {
        return Ok(TrainingPrepPlan::ImputeOnly {
            out: training_prep_h5ad_path(output_dir, source, stem, "imputed")?,
        });
    }
    if can_impute_only && !r.has_cell_type && !r.has_imputed_count && r.has_leiden {
        let patched = training_prep_h5ad_path(output_dir, source, stem, "celltype_patch")?;
        let out = training_prep_h5ad_path(output_dir, source, stem, "imputed")?;
        return Ok(TrainingPrepPlan::PatchThenImpute { patched, out });
    }
    Ok(TrainingPrepPlan::FullPreprocess {
        out: training_prep_h5ad_path(output_dir, source, stem, "fullprep")?,
    })
}

/// When **`data.adata_path`** points at a `.h5ad`, ensure **`obs["cell_type"]`** and
/// **`layers["imputed_count"]`** exist (and **`normalized_count`** + Leiden when needed).
///
/// **`magic_batch_obs`**: when `Some`, MAGIC imputation is batch-clusterwise on this **`adata.obs`**
/// column (use the same name as **`[data].condition`** when training is split by that column).
///
/// **`spatial_microns`**: used only by **`--process-h5ad`** / [`full_preprocess_maybe_log`]
/// (heuristic **`obsm['spatial']` → µm`**). Rust training auto-prep does not rescale spatial coordinates.
///
/// Prepared files are written under **`output_dir/spacetravlr_prep/`** (see [`training_prep_h5ad_path`]).
/// Use [`crate::config::canonical_training_prep_stem`] on the pre-prep path for **`original_input_for_stem`**
/// so filename stems match the user's dataset.
///
/// **The user's input `.h5ad` is never modified.** When auto-prep runs, derived files live under
/// `output_dir/spacetravlr_prep/` (see [`training_prep_h5ad_path`]). Optional post-write
/// [`strip_heavy_training_artifacts_from_h5ad`] / extra HDF5 passes are skipped here for speed.
pub fn ensure_training_adata_ready(
    adata_path: &mut String,
    output_dir: &Path,
    original_input_for_stem: &Path,
    magic_batch_obs: Option<&str>,
    spatial_microns: SpatialMicronsOptions,
) -> anyhow::Result<()> {
    let expanded = expand_user_path(adata_path.trim());
    let p = PathBuf::from(&expanded);
    if !p.is_file() {
        bail!("AnnData not found at {}.", p.display());
    }
    if !p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("h5ad"))
        .unwrap_or(false)
    {
        *adata_path = expanded;
        return Ok(());
    }

    let stem = crate::config::canonical_training_prep_stem(original_input_for_stem);
    let r = probe_adata_training_readiness(&p)?;
    let plan = plan_training_prep(&r, output_dir, &p, &stem)?;
    // #region agent log
    agent_debug_ndjson(
        "F",
        "scanpy_preprocess.rs:ensure_training_adata_ready",
        "training auto-prep plan; spatial_microns only for explicit Scanpy full_preprocess callers",
        "preprocess",
        json!({
            "adata_in": p.to_string_lossy(),
            "plan": format!("{plan:?}"),
            "has_cell_type": r.has_cell_type,
            "has_imputed_count": r.has_imputed_count,
            "has_normalized_count": r.has_normalized_count,
            "has_leiden": r.has_leiden,
            "spatial_microns_skip": spatial_microns.skip,
            "spatial_microns_species": spatial_microns.species,
            "spatial_microns_target_um": spatial_microns.target_median_nn_um,
        }),
    );
    // #endregion
    let reuse_out: Option<PathBuf> = match &plan {
        TrainingPrepPlan::Noop => None,
        TrainingPrepPlan::PatchCellType { out } => Some(out.clone()),
        TrainingPrepPlan::ImputeOnly { out } => Some(out.clone()),
        TrainingPrepPlan::PatchThenImpute { out, .. } => Some(out.clone()),
        TrainingPrepPlan::LayersLeidenAnnotate { out } => Some(out.clone()),
        TrainingPrepPlan::FullPreprocess { out } => Some(out.clone()),
    };
    if let Some(out) = reuse_out {
        if prepared_training_output_is_reusable(&p, &out)? {
            eprintln!(
                "spacetravlr: reusing existing training-prep output {} (>= mtime of {})",
                out.display(),
                p.display()
            );
            *adata_path = expand_user_path(out.to_string_lossy().as_ref());
            return Ok(());
        }
    }
    match plan {
        TrainingPrepPlan::Noop => {
            *adata_path = expanded;
        }
        TrainingPrepPlan::PatchCellType { out } => {
            eprintln!(
                "spacetravlr: obs has no cell_type; writing Leiden → cell_type → {}",
                out.display()
            );
            let _ = std::fs::remove_file(&out);
            write_cell_type_from_leiden(&p, &out)?;
            *adata_path = expand_user_path(out.to_string_lossy().as_ref());
        }
        TrainingPrepPlan::ImputeOnly { out } => {
            eprintln!(
                "spacetravlr: adding layers[\"imputed_count\"] (clusterwise) → {}",
                out.display()
            );
            magic_impute_and_attach_batch(&p, &out, magic_batch_obs, false, false)?;
            *adata_path = expand_user_path(out.to_string_lossy().as_ref());
        }
        TrainingPrepPlan::PatchThenImpute { patched, out } => {
            eprintln!(
                "spacetravlr: obs has no cell_type; Leiden → cell_type, then imputation → {}",
                out.display()
            );
            let _ = std::fs::remove_file(&patched);
            write_cell_type_from_leiden(&p, &patched)?;
            magic_impute_and_attach_batch(&patched, &out, magic_batch_obs, false, false)?;
            let _ = std::fs::remove_file(&patched);
            *adata_path = expand_user_path(out.to_string_lossy().as_ref());
        }
        TrainingPrepPlan::LayersLeidenAnnotate { out } => {
            eprintln!(
                "spacetravlr: expression layers present; Rust UMAP+Leiden → cell_type (no re-impute) → {}",
                out.display()
            );
            let _ = std::fs::remove_file(&out);
            let params = RustPreprocessParams::default();
            rust_preprocess_h5ad_with_steps(
                &p,
                Some(out.as_path()),
                &params,
                &RustPreprocessSteps::TRAINING_LAYERS_LEIDEN_ANNOTATE,
            )?;
            *adata_path = expand_user_path(out.to_string_lossy().as_ref());
        }
        TrainingPrepPlan::FullPreprocess { out } => {
            if let Some(batch_label) = magic_batch_obs.map(str::trim).filter(|s| !s.is_empty()) {
                eprintln!(
                    "spacetravlr: running full Scanpy preprocess (batch-aware MAGIC on `{batch_label}`) → {}",
                    out.display()
                );
                let (written, _) = full_preprocess_maybe_log(
                    &p,
                    &out,
                    false,
                    magic_batch_obs,
                    spatial_microns,
                    false,
                )?;
                debug_assert_eq!(written, out);
                *adata_path = expand_user_path(written.to_string_lossy().as_ref());
            } else {
                eprintln!(
                    "spacetravlr: running Rust preprocess (QC → log-norm → HVG → PCA → UMAP → Leiden → MAGIC) → {}",
                    out.display()
                );
                let _ = std::fs::remove_file(&out);
                let params = RustPreprocessParams::default();
                rust_preprocess_h5ad_with_steps(
                    &p,
                    Some(out.as_path()),
                    &params,
                    &RustPreprocessSteps::FULL,
                )?;
                *adata_path = expand_user_path(out.to_string_lossy().as_ref());
            }
        }
    }
    Ok(())
}

/// `<stem>_processed.h5ad` beside `adata_in` (last5 chars treated as `.h5ad` suffix, any case).
pub fn processed_h5ad_path(adata_in: &Path) -> anyhow::Result<PathBuf> {
    let name = adata_in
        .file_name()
        .and_then(|n| n.to_str())
        .context("AnnData path has no file name")?;
    if !name.to_lowercase().ends_with(".h5ad") {
        bail!("expected input path ending in .h5ad");
    }
    let stem = &name[..name.len() - 5];
    let mut out = adata_in.to_path_buf();
    out.set_file_name(format!("{stem}_processed.h5ad"));
    Ok(out)
}

fn run_uv_full_preprocess_one_write(
    work_out: &Path,
    adata_in: &Path,
    capture_output: bool,
    spatial_microns: &SpatialMicronsOptions,
    magic_batch_obs: Option<&str>,
) -> anyhow::Result<String> {
    let adata_str = adata_in
        .to_str()
        .with_context(|| format!("AnnData path must be UTF-8: {}", adata_in.display()))?;
    let work_str = work_out
        .to_str()
        .with_context(|| format!("work .h5ad path must be UTF-8: {}", work_out.display()))?;
    let skip = if spatial_microns.skip { "1" } else { "0" };
    let species_trim = spatial_microns.species.trim();
    let species_arg = if spatial_microns.skip {
        if species_trim.is_empty() {
            "human"
        } else {
            species_trim
        }
    } else if species_trim.is_empty() {
        bail!(
            "internal: spatial microns species empty for {}; call resolve_spatial_microns_species_for_h5ad first",
            adata_in.display()
        );
    } else {
        species_trim
    };
    let target_um = spatial_microns
        .target_median_nn_um
        .map(|x| x.to_string())
        .unwrap_or_default();
    let batch_token = magic_batch_obs
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("-");
    // #region agent log
    agent_debug_ndjson(
        "B",
        "scanpy_preprocess.rs:run_uv_full_preprocess_one_write",
        "embedded scanpy+magic argv: skip_microns species target_um batch",
        "preprocess",
        json!({
            "adata_in": adata_in.to_string_lossy(),
            "work_out": work_out.to_string_lossy(),
            "skip_microns": skip,
            "species_arg": species_arg,
            "target_um_arg": target_um,
            "batch_token": batch_token,
        }),
    );
    // #endregion
    uv_python_stdin(
        UV_WITH_SCANPY,
        SCANPY_BASIC_PREPROCESS_PY,
        &[
            adata_str,
            work_str,
            skip,
            species_arg,
            target_um.as_str(),
            batch_token,
        ],
        capture_output,
        "scanpy preprocess + magic impute + csr",
    )
}

/// Scanpy QC → UMAP/Leiden → **magic-impute** (one embedded uv run) → **`<stem>_processed.h5ad`** beside the input.
pub fn full_preprocess(adata_in: &Path) -> anyhow::Result<PathBuf> {
    let dest = processed_h5ad_path(adata_in)?;
    let (path, _) = full_preprocess_maybe_log(
        adata_in,
        &dest,
        false,
        None,
        SpatialMicronsOptions::default(),
        true,
    )?;
    Ok(path)
}

/// Full preprocess; when **`capture_output`** is true, uv stdout/stderr are returned for echoing (e.g. to stderr) while keeping stdout clean for paths.
///
/// **`magic_batch_obs`**: optional `adata.obs` column name; when set, MAGIC runs per **(cell_type or leiden) × batch** group (see [`magic_impute_and_attach_batch`]).
///
/// **`spatial_microns`**: passed to the Scanpy embed (see [`SpatialMicronsOptions`]).
///
/// **`strip_heavy_after`**: when **false**, skip [`strip_heavy_training_artifacts_from_h5ad`] after the
/// preprocess write (saves a slow HDF5 rewrite; use **true** for CLI `--process-h5ad` hygiene).
pub fn full_preprocess_maybe_log(
    adata_in: &Path,
    dest_processed: &Path,
    capture_output: bool,
    magic_batch_obs: Option<&str>,
    spatial_microns: SpatialMicronsOptions,
    strip_heavy_after: bool,
) -> anyhow::Result<(PathBuf, Option<String>)> {
    let adata_str = adata_in
        .to_str()
        .with_context(|| format!("AnnData path must be UTF-8: {}", adata_in.display()))?;
    if !adata_str.to_lowercase().ends_with(".h5ad") {
        bail!("expected input path ending in .h5ad");
    }
    let spatial_microns = resolve_spatial_microns_species_for_h5ad(spatial_microns, adata_in)?;
    let expected_out = dest_processed.to_path_buf();
    if let Some(parent) = expected_out.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create output directory for processed h5ad: {}",
                parent.display()
            )
        })?;
    }
    let parent = expected_out
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    if let Some(b) = magic_batch_obs.map(str::trim).filter(|s| !s.is_empty()) {
        ensure_magic_batch_obs_column_exists(adata_in, b)?;
    }

    let work_path = parent.join(format!(
        ".spacetravlr_preprocess_work_{}.h5ad",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&work_path);

    let log_scanpy = run_uv_full_preprocess_one_write(
        &work_path,
        adata_in,
        capture_output,
        &spatial_microns,
        magic_batch_obs,
    )?;
    if !work_path.is_file() {
        bail!(
            "preprocess+magic work output missing: {}",
            work_path.display()
        );
    }

    let same_in_out = match (
        std::fs::canonicalize(adata_in),
        std::fs::canonicalize(dest_processed),
    ) {
        (Ok(a), Ok(d)) => a == d,
        _ => false,
    };
    if dest_processed.exists() && !same_in_out {
        std::fs::remove_file(dest_processed).with_context(|| {
            format!(
                "remove existing output before replace: {}",
                dest_processed.display()
            )
        })?;
    }
    std::fs::rename(&work_path, dest_processed).with_context(|| {
        format!(
            "rename {} -> {}",
            work_path.display(),
            dest_processed.display()
        )
    })?;

    if !dest_processed.is_file() {
        bail!(
            "expected output file missing after pipeline: {}",
            dest_processed.display()
        );
    }

    if strip_heavy_after {
        strip_heavy_training_artifacts_from_h5ad(dest_processed)?;
    }

    let log = if capture_output {
        Some(log_scanpy)
    } else {
        None
    };
    Ok((expected_out, log))
}

/// Same as [`full_preprocess`] (kept for older call sites).
pub fn run_uv_isolated_scanpy_basic_preprocess(adata_in: &Path) -> anyhow::Result<PathBuf> {
    full_preprocess(adata_in)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, AxisArraysOp, Backend};
    use anndata_hdf5::H5;
    use anyhow::Context;
    use ndarray::Array2;
    use std::path::Path;
    use std::process::{Command, Stdio};

    fn uv_run_status_retry_no_cache(
        mut build: impl FnMut(bool) -> Command,
    ) -> anyhow::Result<std::process::ExitStatus> {
        let s = build(false).status().context("spawn uv (first attempt)")?;
        if s.success() {
            return Ok(s);
        }
        build(true).status().context("spawn uv (--no-cache retry)")
    }

    #[test]
    fn processed_h5ad_path_stem() {
        let p = PathBuf::from("/tmp/foo.h5ad");
        assert_eq!(
            processed_h5ad_path(&p).unwrap(),
            PathBuf::from("/tmp/foo_processed.h5ad")
        );
    }

    #[test]
    fn training_processed_path_under_output_dir() {
        let d = PathBuf::from("/tmp/run");
        assert_eq!(
            training_processed_h5ad_path(&d, "foo"),
            PathBuf::from("/tmp/run/foo_processed.h5ad")
        );
        assert_eq!(
            training_imputed_h5ad_path(&d, "foo"),
            PathBuf::from("/tmp/run/foo_imputed.h5ad")
        );
        assert_eq!(
            training_cell_type_patch_h5ad_path(&d, "foo"),
            PathBuf::from("/tmp/run/foo_with_cell_type.h5ad")
        );
    }

    fn uv_available() -> bool {
        Command::new(uv_executable())
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn write_minimal_h5ad_via_uv(path: &Path) -> anyhow::Result<()> {
        let path_str = path.to_str().context("toy path utf-8")?;
        let status = uv_run_status_retry_no_cache(|no_cache| {
            let mut c = Command::new(uv_executable());
            c.env_remove("PYTHONPATH").env("PYTHONNOUSERSITE", "1");
            if no_cache {
                c.arg("--no-cache");
            }
            c.arg("run")
                .arg("--isolated")
                .args(["--with", "numpy<2"])
                .args(["--with", "anndata>=0.11"])
                .arg("python")
                .arg("-c")
                .arg(
                    r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

p = Path(sys.argv[1])
n_obs, n_var = 80, 800
rng = np.random.default_rng(0)
x = np.full((n_obs, n_var), 25.0, dtype=np.float32)
x += rng.normal(0.0, 2.0, size=x.shape).astype(np.float32)
x = np.clip(x, 0.0, None)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"GEN{i}" for i in range(n_var)]
a.write_h5ad(p)
"#,
                )
                .arg(path_str);
            c
        })
        .context("uv to write toy h5ad")?;
        anyhow::ensure!(status.success(), "uv toy h5ad write failed: {status}");
        Ok(())
    }

    fn write_spatial_grid_h5ad_via_uv(path: &Path) -> anyhow::Result<()> {
        let path_str = path.to_str().context("toy path utf-8")?;
        let status = uv_run_status_retry_no_cache(|no_cache| {
            let mut c = Command::new(uv_executable());
            c.env_remove("PYTHONPATH").env("PYTHONNOUSERSITE", "1");
            if no_cache {
                c.arg("--no-cache");
            }
            c.arg("run")
                .arg("--isolated")
                .args(["--with", "numpy<2"])
                .args(["--with", "anndata>=0.11"])
                .arg("python")
                .arg("-c")
                .arg(
                    r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

p = Path(sys.argv[1])
n_side = 10
n_obs = n_side * n_side
n_var = 800
rng = np.random.default_rng(0)
pitch = 1.0
idx = np.arange(n_obs, dtype=np.int64)
rows = idx // n_side
cols = idx % n_side
spatial = np.column_stack([rows.astype(np.float64) * pitch, cols.astype(np.float64) * pitch])
x = np.full((n_obs, n_var), 25.0, dtype=np.float32)
x += rng.normal(0.0, 2.0, size=x.shape).astype(np.float32)
x = np.clip(x, 0.0, None)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"GEN{i}" for i in range(n_var)]
a.obsm["spatial"] = spatial
a.write_h5ad(p)
"#,
                )
                .arg(path_str);
            c
        })
        .context("uv to write spatial grid h5ad")?;
        anyhow::ensure!(
            status.success(),
            "uv spatial grid h5ad write failed: {status}"
        );
        Ok(())
    }

    #[test]
    fn uv_isolated_scanpy_basic_preprocess_writes_sibling_processed() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_scanpy_uv_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let in_path = dir.join("toy.h5ad");
        write_minimal_h5ad_via_uv(&in_path).expect("toy h5ad");

        let expected = processed_h5ad_path(&in_path).unwrap();
        let (out, log) = full_preprocess_maybe_log(
            &in_path,
            &expected,
            true,
            None,
            SpatialMicronsOptions::default(),
            true,
        )
        .expect("preprocess");
        let log = log.expect("captured");

        assert_eq!(out, expected);
        assert!(out.is_file(), "{}", out.display());
        assert!(
            log.contains("wrote"),
            "expected 'wrote' in uv output, got:\n{log}"
        );
        assert!(
            log.contains("x_space") && log.contains("counts"),
            "toy float matrix should classify as counts:\n{log}"
        );

        let processed =
            AnnData::<H5>::open(H5::open(&out).expect("open processed")).expect("ann data read");
        assert!(
            processed.layers().get("normalized_count").is_some(),
            "processed h5ad should have layers[\"normalized_count\"]"
        );
        assert!(
            processed.layers().get("imputed_count").is_some(),
            "processed h5ad should retain layers[\"imputed_count\"]"
        );

        let xdata = processed
            .x()
            .get::<ArrayData>()
            .expect("read X dtype")
            .expect("X present");
        assert!(
            matches!(
                xdata,
                ArrayData::CsrMatrix(_) | ArrayData::CsrNonCanonical(_) | ArrayData::CscMatrix(_)
            ),
            "final X should be sparse"
        );
        for key in ["raw_count", "normalized_count", "imputed_count"] {
            let Some(elem) = processed.layers().get(key) else {
                continue;
            };
            let d = elem
                .get::<ArrayData>()
                .expect("layer dtype")
                .expect("layer matrix");
            assert!(
                matches!(
                    d,
                    ArrayData::CsrMatrix(_)
                        | ArrayData::CsrNonCanonical(_)
                        | ArrayData::CscMatrix(_)
                ),
                "layer {key} should be sparse"
            );
        }

        let obs = processed.read_obs().expect("obs");
        assert!(
            obs.column("cell_type").is_ok(),
            "processed h5ad should have obs cell_type (from Leiden)"
        );
        assert!(
            obs.column("leiden").is_ok(),
            "processed h5ad should have leiden"
        );

        processed.close().expect("close");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uv_preprocess_scales_obsm_spatial_microns() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_spatial_um_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let in_path = dir.join("grid.h5ad");
        write_spatial_grid_h5ad_via_uv(&in_path).expect("spatial grid h5ad");

        let expected = processed_h5ad_path(&in_path).unwrap();
        let (out, log) = full_preprocess_maybe_log(
            &in_path,
            &expected,
            true,
            None,
            SpatialMicronsOptions::default(),
            true,
        )
        .expect("preprocess");
        let log = log.expect("captured");
        assert!(
            log.contains("spatial_microns"),
            "expected spatial_microns in log:\n{log}"
        );

        assert_eq!(out, expected);
        let processed =
            AnnData::<H5>::open(H5::open(&out).expect("open processed")).expect("ann data read");
        let sp = processed
            .obsm()
            .get_item::<Array2<f64>>("spatial")
            .expect("read spatial")
            .expect("spatial key");
        let un = processed
            .obsm()
            .get_item::<Array2<f64>>("unscaled_spatial")
            .expect("read unscaled")
            .expect("unscaled_spatial key");
        assert_eq!(sp.nrows(), un.nrows());
        let scale = 13.0 / 1.0;
        let mut ratios = Vec::new();
        for i in 0..sp.nrows() {
            let ux = un[[i, 0]];
            let uy = un[[i, 1]];
            if ux.abs() > 0.2 && uy.abs() < 1e-6 {
                ratios.push(sp[[i, 0]] / ux);
            }
        }
        assert!(
            ratios.len() >= 3,
            "expected several lattice points on x-axis after QC, got {}",
            ratios.len()
        );
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ratios[ratios.len() / 2];
        assert!(
            (med - scale).abs() < 0.35,
            "median spatial/unscaled x ratio got {med} expected ~{scale}"
        );
        processed.close().expect("close");

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_log1p_like_h5ad_via_uv(path: &Path) -> anyhow::Result<()> {
        let path_str = path.to_str().context("toy path utf-8")?;
        let status = uv_run_status_retry_no_cache(|no_cache| {
            let mut c = Command::new(uv_executable());
            c.env_remove("PYTHONPATH").env("PYTHONNOUSERSITE", "1");
            if no_cache {
                c.arg("--no-cache");
            }
            c.arg("run")
                .arg("--isolated")
                .args(["--with", "numpy<2"])
                .args(["--with", "anndata>=0.11"])
                .arg("python")
                .arg("-c")
                .arg(
                    r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad

p = Path(sys.argv[1])
rng = np.random.default_rng(1)
# Values typical of log1p(counts): bounded, mostly non-integer
x = rng.uniform(0.0, 3.5, size=(60, 300)).astype(np.float32)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(60)]
a.var_names = [f"G{i}" for i in range(300)]
a.write_h5ad(p)
"#,
                )
                .arg(path_str);
            c
        })
        .context("uv to write log1p-like h5ad")?;
        anyhow::ensure!(status.success(), "uv log1p toy write failed: {status}");
        Ok(())
    }

    #[test]
    fn uv_preprocess_classifies_log1p_like_input() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_scanpy_log_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let in_path = dir.join("logish.h5ad");
        write_log1p_like_h5ad_via_uv(&in_path).expect("log1p-like h5ad");

        let dest = processed_h5ad_path(&in_path).unwrap();
        let (out, log) = full_preprocess_maybe_log(
            &in_path,
            &dest,
            true,
            None,
            SpatialMicronsOptions {
                skip: false,
                species: "human".into(),
                target_median_nn_um: None,
            },
            true,
        )
        .expect("preprocess");
        let log = log.expect("captured");
        assert!(
            log.contains("x_space") && log.contains("log1p"),
            "low-range float matrix should classify as log1p:\n{log}"
        );

        let ad = AnnData::<H5>::open(H5::open(&out).expect("open out")).expect("read");
        assert!(
            log.contains("copying X -> layers['normalized_count']"),
            "log1p-classified path should log direct copy:\n{log}"
        );
        assert!(
            ad.layers().get("normalized_count").is_some(),
            "log1p-classified input should get layers[\"normalized_count\"] from X"
        );
        assert!(
            ad.layers().get("imputed_count").is_some(),
            "processed output should include imputed_count"
        );
        ad.close().expect("close");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_magic_batch_obs_column_prefers_explicit_batch() {
        assert_eq!(
            resolve_magic_batch_obs_column(Some("  batch1  "), Some("cond")),
            Some("batch1".to_string())
        );
        assert_eq!(
            resolve_magic_batch_obs_column(None, Some("  slice  ")),
            Some("slice".to_string())
        );
        assert_eq!(
            resolve_magic_batch_obs_column(Some(""), Some("c")),
            Some("c".to_string())
        );
        assert_eq!(resolve_magic_batch_obs_column(Some("  "), None), None);
        assert_eq!(resolve_magic_batch_obs_column(None, None), None);
        assert_eq!(
            resolve_magic_batch_obs_column(Some("x"), Some("y")),
            Some("x".to_string())
        );
    }

    #[test]
    fn plan_training_prep_all_branches() {
        let tmp =
            std::env::temp_dir().join(format!("spacetravlr_plan_branches_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("dummy_source.h5ad");
        std::fs::write(&src, b"x").unwrap();
        let out = tmp.join("run_out");
        std::fs::create_dir_all(&out).unwrap();
        let stem = "stem";
        let r = |has_cell_type, has_leiden, has_normalized_count, has_imputed_count| {
            AdataTrainingReadiness {
                has_cell_type,
                has_leiden,
                has_normalized_count,
                has_imputed_count,
            }
        };

        assert_eq!(
            plan_training_prep(&r(true, false, true, true), &out, &src, stem).unwrap(),
            TrainingPrepPlan::Noop
        );
        assert_eq!(
            plan_training_prep(&r(true, true, true, true), &out, &src, stem).unwrap(),
            TrainingPrepPlan::Noop
        );

        assert_eq!(
            plan_training_prep(&r(false, true, true, true), &out, &src, stem).unwrap(),
            TrainingPrepPlan::PatchCellType {
                out: training_prep_h5ad_path(&out, &src, stem, "celltype").unwrap(),
            }
        );

        assert_eq!(
            plan_training_prep(&r(true, false, true, false), &out, &src, stem).unwrap(),
            TrainingPrepPlan::ImputeOnly {
                out: training_prep_h5ad_path(&out, &src, stem, "imputed").unwrap(),
            }
        );
        assert_eq!(
            plan_training_prep(&r(true, true, true, false), &out, &src, stem).unwrap(),
            TrainingPrepPlan::ImputeOnly {
                out: training_prep_h5ad_path(&out, &src, stem, "imputed").unwrap(),
            }
        );

        assert_eq!(
            plan_training_prep(&r(false, true, true, false), &out, &src, stem).unwrap(),
            TrainingPrepPlan::PatchThenImpute {
                patched: training_prep_h5ad_path(&out, &src, stem, "celltype_patch").unwrap(),
                out: training_prep_h5ad_path(&out, &src, stem, "imputed").unwrap(),
            }
        );

        assert_eq!(
            plan_training_prep(&r(false, false, true, true), &out, &src, stem).unwrap(),
            TrainingPrepPlan::LayersLeidenAnnotate {
                out: training_prep_h5ad_path(&out, &src, stem, "layers_leiden").unwrap(),
            }
        );

        assert_eq!(
            plan_training_prep(&r(false, false, true, false), &out, &src, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_prep_h5ad_path(&out, &src, stem, "fullprep").unwrap(),
            }
        );
        assert_eq!(
            plan_training_prep(&r(true, false, false, false), &out, &src, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_prep_h5ad_path(&out, &src, stem, "fullprep").unwrap(),
            }
        );
        assert_eq!(
            plan_training_prep(&r(false, true, false, false), &out, &src, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_prep_h5ad_path(&out, &src, stem, "fullprep").unwrap(),
            }
        );
        assert_eq!(
            plan_training_prep(&r(false, false, false, true), &out, &src, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_prep_h5ad_path(&out, &src, stem, "fullprep").unwrap(),
            }
        );
        assert_eq!(
            plan_training_prep(&r(true, false, false, true), &out, &src, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_prep_h5ad_path(&out, &src, stem, "fullprep").unwrap(),
            }
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn write_h5ad_probe_fixture(path: &Path, py_body: &str) -> anyhow::Result<()> {
        let path_str = path.to_str().context("fixture path utf-8")?;
        let script = format!(
            r#"
import sys
from pathlib import Path
import numpy as np
import anndata as ad
import scipy.sparse as sp

out = Path(sys.argv[1])
{py_body}
a.write_h5ad(out)
"#,
            py_body = py_body
        );
        let status = uv_run_status_retry_no_cache(|no_cache| {
            let mut c = Command::new(uv_executable());
            c.env_remove("PYTHONPATH").env("PYTHONNOUSERSITE", "1");
            if no_cache {
                c.arg("--no-cache");
            }
            c.arg("run")
                .arg("--isolated")
                .args(["--with", "numpy<2"])
                .args(["--with", "anndata>=0.11"])
                .args(["--with", "scipy"])
                .arg("python")
                .arg("-c")
                .arg(&script)
                .arg(path_str);
            c
        })
        .context("uv fixture h5ad")?;
        anyhow::ensure!(status.success(), "uv fixture h5ad failed: {status}");
        Ok(())
    }

    #[test]
    fn uv_probe_and_plan_leiden_only_normalized_not_imputed() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_probe_leiden_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.h5ad");
        write_h5ad_probe_fixture(
            &path,
            r#"
n_obs, n_var = 36, 90
rng = np.random.default_rng(42)
X = np.abs(rng.normal(2.0, 0.5, size=(n_obs, n_var))).astype(np.float32)
a = ad.AnnData(X=sp.csr_matrix(X))
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"G{i}" for i in range(n_var)]
a.layers["normalized_count"] = X.copy()
a.obs["leiden"] = np.array([str(i % 3) for i in range(n_obs)], dtype=object)
"#,
        )
        .expect("fixture");

        let r = probe_adata_training_readiness(&path).expect("probe");
        assert!(!r.has_cell_type);
        assert!(r.has_leiden);
        assert!(r.has_normalized_count);
        assert!(!r.has_imputed_count);

        let plan = plan_training_prep(&r, &dir, &path, "stem").expect("plan");
        assert_eq!(
            plan,
            TrainingPrepPlan::PatchThenImpute {
                patched: training_prep_h5ad_path(&dir, &path, "stem", "celltype_patch").unwrap(),
                out: training_prep_h5ad_path(&dir, &path, "stem", "imputed").unwrap(),
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uv_probe_cell_type_normalized_not_imputed_is_impute_only() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!("spacetravlr_probe_ct_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("y.h5ad");
        write_h5ad_probe_fixture(
            &path,
            r#"
n_obs, n_var = 24, 60
rng = np.random.default_rng(43)
X = np.abs(rng.normal(2.0, 0.5, size=(n_obs, n_var))).astype(np.float32)
a = ad.AnnData(X=sp.csr_matrix(X))
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"G{i}" for i in range(n_var)]
a.layers["normalized_count"] = X.copy()
a.obs["cell_type"] = np.array([str(i % 2) for i in range(n_obs)], dtype=object)
"#,
        )
        .expect("fixture");

        let r = probe_adata_training_readiness(&path).expect("probe");
        assert!(r.has_cell_type);
        assert!(!r.has_imputed_count);
        assert!(r.has_normalized_count);

        let plan = plan_training_prep(&r, &dir, &path, "s").expect("plan");
        assert_eq!(
            plan,
            TrainingPrepPlan::ImputeOnly {
                out: training_prep_h5ad_path(&dir, &path, "s", "imputed").unwrap(),
            }
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uv_probe_ready_for_training_is_noop() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_probe_noop_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("z.h5ad");
        write_h5ad_probe_fixture(
            &path,
            r#"
n_obs, n_var = 16, 40
rng = np.random.default_rng(44)
X = np.abs(rng.normal(2.0, 0.5, size=(n_obs, n_var))).astype(np.float32)
a = ad.AnnData(X=sp.csr_matrix(X))
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"G{i}" for i in range(n_var)]
a.layers["normalized_count"] = X.copy()
a.layers["imputed_count"] = (X * 1.01).astype(np.float32)
a.obs["cell_type"] = np.array([str(i % 2) for i in range(n_obs)], dtype=object)
"#,
        )
        .expect("fixture");

        let r = probe_adata_training_readiness(&path).expect("probe");
        assert!(r.has_cell_type);
        assert!(r.has_normalized_count);
        assert!(r.has_imputed_count);

        let plan = plan_training_prep(&r, &dir, &path, "s").expect("plan");
        assert_eq!(plan, TrainingPrepPlan::Noop);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uv_magic_impute_batch_obs_column_produces_imputed_layer() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_magic_batch_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.h5ad");
        write_h5ad_probe_fixture(
            &src,
            r#"
n_obs, n_var = 24, 48
rng = np.random.default_rng(45)
X = np.abs(rng.normal(2.0, 0.5, size=(n_obs, n_var))).astype(np.float32)
a = ad.AnnData(X=sp.csr_matrix(X))
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"G{i}" for i in range(n_var)]
a.layers["normalized_count"] = X.copy()
a.obs["cell_type"] = np.array([str(i % 3) for i in range(n_obs)], dtype=object)
a.obs["sample"] = np.where(np.arange(n_obs) % 2 == 0, "A", "B")
"#,
        )
        .expect("fixture");

        let dst = dir.join("out.h5ad");
        magic_impute_and_attach_batch(&src, &dst, Some("sample"), false, true)
            .expect("magic batch");

        let ad = AnnData::<H5>::open(H5::open(&dst).expect("open")).expect("read");
        assert!(ad.layers().get("imputed_count").is_some());
        let obs = ad.read_obs().expect("obs");
        assert!(obs.column("cell_type").is_ok());
        ad.close().expect("close");

        let dst2 = dir.join("out_nobatch.h5ad");
        magic_impute_and_attach_batch(&src, &dst2, None, false, true).expect("magic no batch");
        let ad2 = AnnData::<H5>::open(H5::open(&dst2).expect("open2")).expect("read2");
        assert!(ad2.layers().get("imputed_count").is_some());
        ad2.close().expect("close2");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uv_magic_impute_unknown_batch_obs_column_fails() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("spacetravlr_magic_badbatch_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.h5ad");
        write_h5ad_probe_fixture(
            &src,
            r#"
n_obs, n_var = 16, 40
rng = np.random.default_rng(46)
X = np.abs(rng.normal(2.0, 0.5, size=(n_obs, n_var))).astype(np.float32)
a = ad.AnnData(X=sp.csr_matrix(X))
a.obs_names = [f"c{i}" for i in range(n_obs)]
a.var_names = [f"G{i}" for i in range(n_var)]
a.layers["normalized_count"] = X.copy()
a.obs["cell_type"] = np.array([str(i % 2) for i in range(n_obs)], dtype=object)
"#,
        )
        .expect("fixture");

        let dst = dir.join("bad.h5ad");
        let err = magic_impute_and_attach_batch(&src, &dst, Some("no_such_column"), true, true)
            .expect_err("expected uv/python failure");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no_such_column") || msg.contains("not found"),
            "unexpected error: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
