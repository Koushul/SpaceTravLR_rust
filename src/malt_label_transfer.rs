//! Marker-aware label transfer (MALT): runs [`scripts/malt_label_transfer.py`] in an isolated **`uv`** env.
//! Prefers on-disk `malt_label_transfer.py` from [`crate::config::resolve_malt_label_transfer_py_path`]
//! (e.g. `data/` next to the binary after `install.sh`); falls back to the copy embedded at build time.

use crate::config::resolve_malt_label_transfer_py_path;
use crate::scanpy_preprocess::uv_python_stdin;
use anyhow::Context;
use std::fs;
use std::path::Path;

const MALT_SCRIPT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/scripts/malt_label_transfer.py"
));

pub const UV_WITH_MAP_LABELS: &[&str] = &[
    "numpy<2",
    "pandas",
    "pyarrow",
    "anndata>=0.11",
    "scipy",
    "scanpy",
    "h5py",
    "matplotlib",
    "scikit-learn",
    "torch",
    "leidenalg",
    "igraph",
];

pub struct MapLabelsParams<'a> {
    pub reference: &'a Path,
    pub query: &'a Path,
    pub outdir: &'a Path,
    pub groupby: &'a [String],
    pub output_query: &'a str,
    pub extra_markers: Option<&'a str>,
    pub expression_mode: &'a str,
    pub counts_layer: Option<&'a str>,
    pub prefer_raw_counts: bool,
    pub ref_betadata_dir: Option<&'a Path>,
    pub query_betadata_dir: Option<&'a Path>,
    pub query_grn_cluster_obs: Option<&'a str>,
    pub grn_loss_weight: f64,
}

pub fn run_map_labels(params: MapLabelsParams<'_>) -> anyhow::Result<()> {
    let r = params.reference.to_str().with_context(|| {
        format!(
            "reference path must be UTF-8: {}",
            params.reference.display()
        )
    })?;
    let q = params
        .query
        .to_str()
        .with_context(|| format!("query path must be UTF-8: {}", params.query.display()))?;
    let o = params
        .outdir
        .to_str()
        .with_context(|| format!("outdir path must be UTF-8: {}", params.outdir.display()))?;

    let mut argv: Vec<String> = vec![
        "--reference".into(),
        r.into(),
        "--query".into(),
        q.into(),
        "--outdir".into(),
        o.into(),
        "--output-query".into(),
        params.output_query.into(),
        "--expression-mode".into(),
        params.expression_mode.into(),
    ];
    for g in params.groupby {
        argv.push("--groupby".into());
        argv.push(g.clone());
    }
    if let Some(ex) = params.extra_markers {
        if !ex.trim().is_empty() {
            argv.push("--extra-markers".into());
            argv.push(ex.into());
        }
    }
    if let Some(layer) = params.counts_layer {
        argv.push("--counts-layer".into());
        argv.push(layer.into());
    }
    if params.prefer_raw_counts {
        argv.push("--prefer-raw-counts".into());
    }
    if let Some(p) = params.ref_betadata_dir {
        let s = p.to_str().with_context(|| format!("ref betadata path UTF-8: {}", p.display()))?;
        argv.push("--ref-betadata-dir".into());
        argv.push(s.into());
    }
    if let Some(p) = params.query_betadata_dir {
        let s = p.to_str().with_context(|| format!("query betadata path UTF-8: {}", p.display()))?;
        argv.push("--query-betadata-dir".into());
        argv.push(s.into());
    }
    if params.ref_betadata_dir.is_some() && params.query_betadata_dir.is_some() {
        argv.push("--grn-loss-weight".into());
        argv.push(params.grn_loss_weight.to_string());
    }
    if let Some(col) = params.query_grn_cluster_obs {
        if !col.trim().is_empty() {
            argv.push("--query-grn-cluster-obs".into());
            argv.push(col.into());
        }
    }

    let argv_refs: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
    let script = if let Some(p) = resolve_malt_label_transfer_py_path() {
        fs::read_to_string(&p).with_context(|| format!("read MALT script {}", p.display()))?
    } else {
        MALT_SCRIPT.to_string()
    };
    uv_python_stdin(
        UV_WITH_MAP_LABELS,
        &script,
        &argv_refs,
        false,
        "map-labels (MALT)",
    )?;
    Ok(())
}
