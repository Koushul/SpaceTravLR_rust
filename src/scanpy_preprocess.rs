//! Run a standard Scanpy QC + embedding pipeline in a **fresh uv-isolated** env each time.
//!
//! Spawns **`uv run --isolated`** with explicit `--with` packages, then **`python -`** reading this
//! module’s embedded source from **stdin** (no `.py` on disk, no shell scripts).
//! Requires [**uv**](https://docs.astral.sh/uv/) on `PATH`, or **`UV_BIN`**.
//! Child processes clear **`PYTHONPATH`** and set **`PYTHONNOUSERSITE=1`** so Conda/base site-packages
//! do not leak into the isolated env.
//!
//! **Input `X`:** Before normalizing, the embedded Scanpy script infers whether **`X` is already
//! `log1p`-transformed** (Scanpy **`uns['log1p']`**, plus value heuristics) or **raw / linear
//! counts**. Raw path: `normalize_total` → **`layers["normalized_count"]`** → `log1p`. Log path:
//! preserve incoming expression in **`layers["log1p_incoming"]`**, **`expm1(X)`** as a linear
//! approximation, `normalize_total`, store **`normalized_count`**, then `log1p` again for HVG/PCA.
//!
//! Scanpy then runs through Leiden. **Clusterwise Markov imputation** on `normalized_count` uses
//! isolated **`uv`** with [**magic-impute**](https://pypi.org/project/magic-impute/) (one
//! **`MAGIC.fit_transform`** per `cell_type` or `leiden` label; optional **batch** column for
//! (cluster × batch) groups). The same **`uv`** step **CSR-sparsifies `X` and every `layers` matrix**
//! and writes the final file.
//!
//! Output is `<stem>_processed.h5ad` beside the input (pathlib `stem` rule, same as [`processed_h5ad_path`]).
//!
//! **Library entry points**
//!
//! | Step | Function | Output |
//! |------|----------|--------|
//! | Full QC → UMAP/Leiden → magic-impute | [`full_preprocess`] / [`full_preprocess_maybe_log`] | `<stem>_processed.h5ad` |
//! | Imputation + CSR only | [`imputed_count_from_normalized`] / [`magic_impute_and_attach_batch`] | `<stem>_imputed.h5ad` |
//! | `leiden` → `cell_type` (path helper) | [`cell_type_patch_h5ad_path`] | sibling filename |
//! | `leiden` → `cell_type` (write) | [`write_cell_type_from_leiden`] | caller-chosen path |
//! | MAGIC batch column resolution | [`resolve_magic_batch_obs_column`] | CLI / config wiring |
//! | Training auto-prep | [`ensure_training_adata_ready`] (pass **`[data].condition`** as MAGIC batch when set) | updates path in place |
//!
//! **CLI:** **`spacetravlr --process-h5ad --h5ad …`** → [`full_preprocess`];
//! **`spacetravlr --impute --h5ad …`** → [`imputed_count_from_normalized`] (same imputation+CSR as the full pipeline).
//! With **`--condition`**, the same obs column name is used as the MAGIC batch axis unless **`--magic-batch-obs`** overrides.
//!
//! **Training:** [`ensure_training_adata_ready`] uses [`plan_training_prep`] to pick the minimal
//! fix; when **`[data].condition`** is set, imputation uses it as the MAGIC batch column. Opt out with **`--skip-auto-adata-prep`**.

use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp, ArrayElemOp, AxisArraysOp, Backend};
use anndata_hdf5::H5;
use anyhow::{Context, bail};
use crate::config::expand_user_path;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::env;

const UV_WITH_ANNDATA: &[&str] = &["numpy<2", "anndata>=0.11"];
const UV_WITH_ATTACH: &[&str] = &["numpy<2", "anndata>=0.11", "scipy"];
const UV_WITH_MAGIC_IMPUTE: &[&str] = &[
    "numpy<2",
    "anndata>=0.11",
    "scipy",
    "magic-impute>=3,<4",
];
const UV_WITH_SCANPY: &[&str] = &[
    "numpy<2",
    "anndata>=0.11",
    "scanpy",
    "h5py",
    "leidenalg",
    "igraph",
];

fn uv_executable() -> OsString {
    env::var_os("UV_BIN")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "uv".into())
}

fn uv_command_base() -> Command {
    let mut c = Command::new(uv_executable());
    c.env_remove("PYTHONPATH");
    c.env("PYTHONNOUSERSITE", "1");
    c
}

/// `uv run --isolated` + `python -` reading **`stdin_script`**, with **`--with`** deps, then **`argv`** after `python -`.
fn uv_python_stdin(
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
    cmd.arg("run").arg("--isolated");
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
        let status = child.wait().with_context(|| format!("wait for uv ({spawn_hint})"))?;
        if !status.success() {
            bail!("uv run failed ({spawn_hint}): {status}");
        }
        Ok(String::new())
    }
}

const MAGIC_CLUSTERWISE_IMPUTE_CSR_PY: &str = r#"
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
    X = nc.toarray().astype(np.float64)
else:
    X = np.asarray(nc, dtype=np.float64)

labels = np.array([str(x) for x in a.obs[annot].to_numpy()], dtype=object)
out = X.copy()
n_vars = X.shape[1]
knn_def, knn_max_cap, t_magic, n_pca_cap = 5, 10, 3, 100


def magic_op_for_subset(n_sub):
    knn = min(knn_def, max(1, n_sub - 1))
    knn_max = max(knn, min(knn_max_cap, n_sub - 1))
    n_pca_eff = min(n_pca_cap, n_sub, max(1, n_vars))
    return magic.MAGIC(
        knn=knn,
        knn_max=knn_max,
        decay=1,
        t=t_magic,
        n_pca=n_pca_eff,
        verbose=0,
    )


if batch_col is None:
    for lab in np.unique(labels):
        m = labels == lab
        idx = np.flatnonzero(m)
        if idx.size < 2:
            continue
        op = magic_op_for_subset(int(idx.size))
        sub = X[m]
        imp = np.asarray(op.fit_transform(sub, genes="all_genes"), dtype=np.float64)
        out[m] = imp
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
        op = magic_op_for_subset(int(idx.size))
        sub = X[m]
        imp = np.asarray(op.fit_transform(sub, genes="all_genes"), dtype=np.float64)
        out[m] = imp

a.layers["imputed_count"] = out


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

const CSR_LAYERS_PY: &str = r#"
import sys
import anndata as ad
import numpy as np
import scipy.sparse as sp

src, dst = sys.argv[1], sys.argv[2]
a = ad.read_h5ad(src)


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

const SCANPY_BASIC_PREPROCESS_PY: &str = r#"
import os, shutil, sys, tempfile
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata._io.specs.registry import IORegistryError


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


def _expm1_x(X):
    if sp.issparse(X):
        out = X.tocsr().copy()
        out.data = np.expm1(out.data)
        return out
    return np.expm1(np.asarray(X, dtype=np.float64))


src = Path(sys.argv[1])
out_partial = Path(sys.argv[2])
if src.suffix.lower() != ".h5ad":
    sys.exit("expected path ending in .h5ad")
adata = _read_h5ad(str(src))
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))

x_is_log1p = _infer_x_is_log1p(adata)
print("spacetravlr_preprocess x_space", "log1p" if x_is_log1p else "counts", file=sys.stderr)

if x_is_log1p:
    adata.layers["log1p_incoming"] = adata.X.copy()
    adata.X = _expm1_x(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    nc = adata.X.copy()
    if hasattr(nc, "toarray"):
        adata.layers["normalized_count"] = nc.toarray()
    else:
        adata.layers["normalized_count"] = np.asarray(nc, dtype=np.float64)
    sc.pp.log1p(adata)
else:
    adata.layers["raw_count"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    nc = adata.X.copy()
    if hasattr(nc, "toarray"):
        adata.layers["normalized_count"] = nc.toarray()
    else:
        adata.layers["normalized_count"] = np.asarray(nc, dtype=np.float64)
    sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.scale(adata, max_value=10)
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
adata.write_h5ad(out_partial)
print("wrote_partial", out_partial)
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

/// `{stem}_processed.h5ad` under **`output_dir`** (stem from [`crate::config::canonical_adata_stem`]).
pub fn training_processed_h5ad_path(output_dir: &Path, stem: &str) -> PathBuf {
    output_dir.join(format!("{stem}_processed.h5ad"))
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
pub fn adata_x_and_layers_are_csr(path: &Path) -> anyhow::Result<bool> {
    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    let x_ok = match adata.x().get::<ArrayData>()? {
        None => false,
        Some(ArrayData::CsrMatrix(_))
        | Some(ArrayData::CsrNonCanonical(_))
        | Some(ArrayData::CscMatrix(_)) => true,
        Some(ArrayData::Array(_)) | Some(ArrayData::DataFrame(_)) => false,
    };
    if !x_ok {
        adata.close()?;
        return Ok(false);
    }
    for name in adata.layers().keys() {
        let Some(elem) = adata.layers().get(name.as_str()) else {
            continue;
        };
        let ok = match elem.get::<ArrayData>()? {
            None => false,
            Some(ArrayData::CsrMatrix(_))
            | Some(ArrayData::CsrNonCanonical(_))
            | Some(ArrayData::CscMatrix(_)) => true,
            Some(ArrayData::Array(_)) | Some(ArrayData::DataFrame(_)) => false,
        };
        if !ok {
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
        std::fs::rename(&tmp, from_h5ad).with_context(|| {
            format!(
                "rename {} -> {}",
                tmp.display(),
                from_h5ad.display()
            )
        })?;
        return Ok(());
    }
    let _ = std::fs::remove_file(to_h5ad);
    write_h5ad_csr_layers_uv(from_h5ad, to_h5ad, capture_output)?;
    if !to_h5ad.is_file() {
        bail!("csr materialize did not create {}", to_h5ad.display());
    }
    Ok(())
}

/// Writes **`dest_h5ad`**: clusterwise [**magic-impute**](https://pypi.org/project/magic-impute/) on
/// **`layers["normalized_count"]`** (one MAGIC fit per `cell_type` or `leiden` label), then CSR-sparsify **`X`** and all **`layers`** (isolated **`uv`**).
///
/// When **`obs_batch_column`** is set, imputation matches the batch-clusterwise pattern: one MAGIC fit
/// per distinct **(annotation × batch)** pair in **`adata.obs`**, then rows are written back in obs order.
pub fn magic_impute_and_attach_batch(
    source_h5ad: &Path,
    dest_h5ad: &Path,
    obs_batch_column: Option<&str>,
    capture_output: bool,
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

    Ok(log)
}

/// Like [`magic_impute_and_attach_batch`] with no batch column (cluster annotation only).
pub fn magic_impute_and_attach(
    source_h5ad: &Path,
    dest_h5ad: &Path,
    capture_output: bool,
) -> anyhow::Result<String> {
    magic_impute_and_attach_batch(source_h5ad, dest_h5ad, None, capture_output)
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
    PatchCellType { out: PathBuf },
    ImputeOnly { out: PathBuf },
    PatchThenImpute { patched: PathBuf, out: PathBuf },
    FullPreprocess { out: PathBuf },
}

pub fn plan_training_prep(
    r: &AdataTrainingReadiness,
    output_dir: &Path,
    stem: &str,
) -> anyhow::Result<TrainingPrepPlan> {
    if r.has_cell_type && r.has_imputed_count && r.has_normalized_count {
        return Ok(TrainingPrepPlan::Noop);
    }
    let can_impute_only = r.has_normalized_count && (r.has_cell_type || r.has_leiden);
    if !r.has_cell_type && r.has_leiden && r.has_normalized_count && r.has_imputed_count {
        return Ok(TrainingPrepPlan::PatchCellType {
            out: training_cell_type_patch_h5ad_path(output_dir, stem),
        });
    }
    if can_impute_only && r.has_cell_type && !r.has_imputed_count {
        return Ok(TrainingPrepPlan::ImputeOnly {
            out: training_processed_h5ad_path(output_dir, stem),
        });
    }
    if can_impute_only && !r.has_cell_type && !r.has_imputed_count && r.has_leiden {
        let patched = training_cell_type_patch_h5ad_path(output_dir, stem);
        let out = training_processed_h5ad_path(output_dir, stem);
        return Ok(TrainingPrepPlan::PatchThenImpute { patched, out });
    }
    Ok(TrainingPrepPlan::FullPreprocess {
        out: training_processed_h5ad_path(output_dir, stem),
    })
}

/// When **`data.adata_path`** points at a `.h5ad`, ensure **`obs["cell_type"]`** and
/// **`layers["imputed_count"]`** exist (and **`normalized_count`** + Leiden when needed).
///
/// **`magic_batch_obs`**: when `Some`, MAGIC imputation is batch-clusterwise on this **`adata.obs`**
/// column (use the same name as **`[data].condition`** when training is split by that column).
///
/// Prepared files are written under **`output_dir`**. Use [`crate::config::canonical_adata_stem`] on
/// the pre-prep path for **`original_input_for_stem`** so filenames match the user's dataset stem.
pub fn ensure_training_adata_ready(
    adata_path: &mut String,
    output_dir: &Path,
    original_input_for_stem: &Path,
    magic_batch_obs: Option<&str>,
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

    let stem = crate::config::canonical_adata_stem(original_input_for_stem);
    let r = probe_adata_training_readiness(&p)?;
    match plan_training_prep(&r, output_dir, &stem)? {
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
            magic_impute_and_attach_batch(&p, &out, magic_batch_obs, false)?;
            *adata_path = expand_user_path(out.to_string_lossy().as_ref());
        }
        TrainingPrepPlan::PatchThenImpute { patched, out } => {
            eprintln!(
                "spacetravlr: obs has no cell_type; Leiden → cell_type, then imputation → {}",
                out.display()
            );
            let _ = std::fs::remove_file(&patched);
            write_cell_type_from_leiden(&p, &patched)?;
            magic_impute_and_attach_batch(&patched, &out, magic_batch_obs, false)?;
            let _ = std::fs::remove_file(&patched);
            *adata_path = expand_user_path(out.to_string_lossy().as_ref());
        }
        TrainingPrepPlan::FullPreprocess { out } => {
            eprintln!(
                "spacetravlr: running full Scanpy preprocess (UMAP, Leiden, cell_type, imputation) → {}",
                out.display()
            );
            let (written, _) = full_preprocess_maybe_log(&p, &out, false, magic_batch_obs)?;
            debug_assert_eq!(written, out);
            *adata_path = expand_user_path(written.to_string_lossy().as_ref());
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

fn run_uv_scanpy_to_scratch(
    scratch_out: &Path,
    adata_in: &Path,
    capture_output: bool,
) -> anyhow::Result<String> {
    let adata_str = adata_in
        .to_str()
        .with_context(|| format!("AnnData path must be UTF-8: {}", adata_in.display()))?;
    let scratch_str = scratch_out
        .to_str()
        .with_context(|| format!("scratch path must be UTF-8: {}", scratch_out.display()))?;
    uv_python_stdin(
        UV_WITH_SCANPY,
        SCANPY_BASIC_PREPROCESS_PY,
        &[adata_str, scratch_str],
        capture_output,
        "scanpy preprocess",
    )
}

/// Scanpy QC → UMAP/Leiden (scratch `.h5ad`) → **magic-impute** → **`<stem>_processed.h5ad`** beside the input.
pub fn full_preprocess(adata_in: &Path) -> anyhow::Result<PathBuf> {
    let dest = processed_h5ad_path(adata_in)?;
    let (path, _) = full_preprocess_maybe_log(adata_in, &dest, false, None)?;
    Ok(path)
}

/// Full preprocess; when **`capture_output`** is true, uv stdout/stderr are returned for echoing (e.g. to stderr) while keeping stdout clean for paths.
///
/// **`magic_batch_obs`**: optional `adata.obs` column name; when set, MAGIC runs per **(cell_type or leiden) × batch** group (see [`magic_impute_and_attach_batch`]).
pub fn full_preprocess_maybe_log(
    adata_in: &Path,
    dest_processed: &Path,
    capture_output: bool,
    magic_batch_obs: Option<&str>,
) -> anyhow::Result<(PathBuf, Option<String>)> {
    let adata_str = adata_in
        .to_str()
        .with_context(|| format!("AnnData path must be UTF-8: {}", adata_in.display()))?;
    if !adata_str.to_lowercase().ends_with(".h5ad") {
        bail!("expected input path ending in .h5ad");
    }
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
    let scratch = parent.join(format!(
        ".spacetravlr_scanpy_scratch_{}.h5ad",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&scratch);

    let mut log_scanpy = run_uv_scanpy_to_scratch(&scratch, adata_in, capture_output)?;
    if !scratch.is_file() {
        bail!("Scanpy scratch output missing: {}", scratch.display());
    }

    let log_attach = match magic_impute_and_attach_batch(
        &scratch,
        dest_processed,
        magic_batch_obs,
        capture_output,
    ) {
        Ok(s) => s,
        Err(e) => {
            let _ = std::fs::remove_file(&scratch);
            return Err(e);
        }
    };

    let _ = std::fs::remove_file(&scratch);

    if !dest_processed.is_file() {
        bail!(
            "expected output file missing after pipeline: {}",
            dest_processed.display()
        );
    }

    let log = if capture_output {
        log_scanpy.push_str(&log_attach);
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
    use std::path::Path;
    use std::process::{Command, Stdio};

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
        let status = Command::new(uv_executable())
            .env_remove("PYTHONPATH")
            .env("PYTHONNOUSERSITE", "1")
            .arg("run")
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
n_obs, n_var = 100, 2500
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
            .arg(path_str)
            .status()
            .context("spawn uv to write toy h5ad")?;
        anyhow::ensure!(status.success(), "uv toy h5ad write failed: {status}");
        Ok(())
    }

    #[test]
    fn uv_isolated_scanpy_basic_preprocess_writes_sibling_processed() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_scanpy_uv_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let in_path = dir.join("toy.h5ad");
        write_minimal_h5ad_via_uv(&in_path).expect("toy h5ad");

        let expected = processed_h5ad_path(&in_path).unwrap();
        let (out, log) =
            full_preprocess_maybe_log(&in_path, &expected, true, None).expect("preprocess");
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
                ArrayData::CsrMatrix(_)
                    | ArrayData::CsrNonCanonical(_)
                    | ArrayData::CscMatrix(_)
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
        assert!(obs.column("leiden").is_ok(), "processed h5ad should have leiden");

        processed.close().expect("close");

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_log1p_like_h5ad_via_uv(path: &Path) -> anyhow::Result<()> {
        let path_str = path.to_str().context("toy path utf-8")?;
        let status = Command::new(uv_executable())
            .env_remove("PYTHONPATH")
            .env("PYTHONNOUSERSITE", "1")
            .arg("run")
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
x = rng.uniform(0.0, 3.5, size=(80, 600)).astype(np.float32)
a = ad.AnnData(X=x)
a.obs_names = [f"c{i}" for i in range(80)]
a.var_names = [f"G{i}" for i in range(600)]
a.write_h5ad(p)
"#,
            )
            .arg(path_str)
            .status()
            .context("spawn uv to write log1p-like h5ad")?;
        anyhow::ensure!(status.success(), "uv log1p toy write failed: {status}");
        Ok(())
    }

    #[test]
    fn uv_preprocess_classifies_log1p_like_input() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_scanpy_log_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let in_path = dir.join("logish.h5ad");
        write_log1p_like_h5ad_via_uv(&in_path).expect("log1p-like h5ad");

        let dest = processed_h5ad_path(&in_path).unwrap();
        let (out, log) = full_preprocess_maybe_log(&in_path, &dest, true, None).expect("preprocess");
        let log = log.expect("captured");
        assert!(
            log.contains("x_space") && log.contains("log1p"),
            "low-range float matrix should classify as log1p:\n{log}"
        );

        let ad = AnnData::<H5>::open(H5::open(&out).expect("open out")).expect("read");
        assert!(
            ad.layers().get("log1p_incoming").is_some(),
            "log1p path should store layers[\"log1p_incoming\"]"
        );
        assert!(
            ad.layers().get("normalized_count").is_some(),
            "log1p-classified input should still get layers[\"normalized_count\"]"
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
        assert_eq!(
            resolve_magic_batch_obs_column(Some("  "), None),
            None
        );
        assert_eq!(resolve_magic_batch_obs_column(None, None), None);
        assert_eq!(
            resolve_magic_batch_obs_column(Some("x"), Some("y")),
            Some("x".to_string())
        );
    }

    #[test]
    fn plan_training_prep_all_branches() {
        let out = Path::new("/tmp/spacetravlr_plan_test_out");
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
            plan_training_prep(&r(true, false, true, true), out, stem).unwrap(),
            TrainingPrepPlan::Noop
        );
        assert_eq!(
            plan_training_prep(&r(true, true, true, true), out, stem).unwrap(),
            TrainingPrepPlan::Noop
        );

        assert_eq!(
            plan_training_prep(&r(false, true, true, true), out, stem).unwrap(),
            TrainingPrepPlan::PatchCellType {
                out: training_cell_type_patch_h5ad_path(out, stem),
            }
        );

        assert_eq!(
            plan_training_prep(&r(true, false, true, false), out, stem).unwrap(),
            TrainingPrepPlan::ImputeOnly {
                out: training_processed_h5ad_path(out, stem),
            }
        );
        assert_eq!(
            plan_training_prep(&r(true, true, true, false), out, stem).unwrap(),
            TrainingPrepPlan::ImputeOnly {
                out: training_processed_h5ad_path(out, stem),
            }
        );

        assert_eq!(
            plan_training_prep(&r(false, true, true, false), out, stem).unwrap(),
            TrainingPrepPlan::PatchThenImpute {
                patched: training_cell_type_patch_h5ad_path(out, stem),
                out: training_processed_h5ad_path(out, stem),
            }
        );

        assert_eq!(
            plan_training_prep(&r(false, false, true, false), out, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_processed_h5ad_path(out, stem),
            }
        );
        assert_eq!(
            plan_training_prep(&r(true, false, false, false), out, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_processed_h5ad_path(out, stem),
            }
        );
        assert_eq!(
            plan_training_prep(&r(false, true, false, false), out, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_processed_h5ad_path(out, stem),
            }
        );
        assert_eq!(
            plan_training_prep(&r(false, false, false, true), out, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_processed_h5ad_path(out, stem),
            }
        );
        assert_eq!(
            plan_training_prep(&r(true, false, false, true), out, stem).unwrap(),
            TrainingPrepPlan::FullPreprocess {
                out: training_processed_h5ad_path(out, stem),
            }
        );
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
        let status = Command::new(uv_executable())
            .env_remove("PYTHONPATH")
            .env("PYTHONNOUSERSITE", "1")
            .arg("run")
            .arg("--isolated")
            .args(["--with", "numpy<2"])
            .args(["--with", "anndata>=0.11"])
            .args(["--with", "scipy"])
            .arg("python")
            .arg("-c")
            .arg(&script)
            .arg(path_str)
            .status()
            .context("spawn uv fixture h5ad")?;
        anyhow::ensure!(status.success(), "uv fixture h5ad failed: {status}");
        Ok(())
    }

    #[test]
    fn uv_probe_and_plan_leiden_only_normalized_not_imputed() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_probe_leiden_{}",
            std::process::id()
        ));
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

        let plan = plan_training_prep(&r, &dir, "stem").expect("plan");
        assert_eq!(
            plan,
            TrainingPrepPlan::PatchThenImpute {
                patched: training_cell_type_patch_h5ad_path(&dir, "stem"),
                out: training_processed_h5ad_path(&dir, "stem"),
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
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_probe_ct_{}",
            std::process::id()
        ));
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

        let plan = plan_training_prep(&r, &dir, "s").expect("plan");
        assert_eq!(
            plan,
            TrainingPrepPlan::ImputeOnly {
                out: training_processed_h5ad_path(&dir, "s"),
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
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_probe_noop_{}",
            std::process::id()
        ));
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

        let plan = plan_training_prep(&r, &dir, "s").expect("plan");
        assert_eq!(plan, TrainingPrepPlan::Noop);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uv_magic_impute_batch_obs_column_produces_imputed_layer() {
        if !uv_available() {
            eprintln!("skip: uv not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_magic_batch_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.h5ad");
        write_h5ad_probe_fixture(
            &src,
            r#"
n_obs, n_var = 32, 80
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
        magic_impute_and_attach_batch(&src, &dst, Some("sample"), false).expect("magic batch");

        let ad = AnnData::<H5>::open(H5::open(&dst).expect("open")).expect("read");
        assert!(ad.layers().get("imputed_count").is_some());
        let obs = ad.read_obs().expect("obs");
        assert!(obs.column("cell_type").is_ok());
        ad.close().expect("close");

        let dst2 = dir.join("out_nobatch.h5ad");
        magic_impute_and_attach_batch(&src, &dst2, None, false).expect("magic no batch");
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
        let dir = std::env::temp_dir().join(format!(
            "spacetravlr_magic_badbatch_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.h5ad");
        write_h5ad_probe_fixture(
            &src,
            r#"
n_obs, n_var = 20, 50
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
        let err = magic_impute_and_attach_batch(&src, &dst, Some("no_such_column"), true)
            .expect_err("expected uv/python failure");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no_such_column") || msg.contains("not found"),
            "unexpected error: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
