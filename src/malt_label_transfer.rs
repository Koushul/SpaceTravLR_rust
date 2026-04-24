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
    pub groupby: Option<&'a str>,
    pub output_query: &'a str,
    pub extra_markers: Option<&'a str>,
    pub expression_mode: &'a str,
    pub counts_layer: Option<&'a str>,
    pub prefer_raw_counts: bool,
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
    if let Some(g) = params.groupby {
        argv.push("--groupby".into());
        argv.push(g.into());
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
