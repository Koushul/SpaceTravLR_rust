//! BANKSY spatial clustering: runs [`scripts/banksy_cluster.py`] in an isolated **`uv`** env.

use crate::config::resolve_banksy_cluster_py_path;
use crate::scanpy_preprocess::uv_python_stdin;
use anyhow::Context;
use std::fs;
use std::path::Path;

const BANKSY_SCRIPT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/scripts/banksy_cluster.py"
));

pub const UV_WITH_BANKSY: &[&str] = &[
    "numpy<2",
    "pandas",
    "anndata>=0.10",
    "scipy",
    "scanpy",
    "h5py",
    "scikit-learn",
    "leidenalg",
    "igraph",
    "python-igraph",
    "pybanksy>=1.3.5",
];

pub struct BanksyParams<'a> {
    pub h5ad: &'a Path,
    pub output: Option<&'a Path>,
    pub lambda: f64,
    pub num_neighbours: u32,
    pub nbr_weight_decay: &'a str,
    pub max_m: u32,
    pub resolution: f64,
    pub num_nn: u32,
    pub pca_dims: u32,
    pub partition_seed: u32,
    pub num_iterations: i32,
    pub coord_key: &'a str,
    pub x_col: Option<&'a str>,
    pub y_col: Option<&'a str>,
    pub cluster_key: &'a str,
    pub preprocess: bool,
    pub verbose: bool,
}

pub fn run_banksy(params: BanksyParams<'_>) -> anyhow::Result<()> {
    let h5ad = params
        .h5ad
        .to_str()
        .with_context(|| format!("h5ad path must be UTF-8: {}", params.h5ad.display()))?;

    let mut argv: Vec<String> = vec![
        "--h5ad".into(),
        h5ad.into(),
        "--lambda".into(),
        params.lambda.to_string(),
        "--num-neighbours".into(),
        params.num_neighbours.to_string(),
        "--nbr-weight-decay".into(),
        params.nbr_weight_decay.into(),
        "--max-m".into(),
        params.max_m.to_string(),
        "--resolution".into(),
        params.resolution.to_string(),
        "--num-nn".into(),
        params.num_nn.to_string(),
        "--pca-dims".into(),
        params.pca_dims.to_string(),
        "--partition-seed".into(),
        params.partition_seed.to_string(),
        "--num-iterations".into(),
        params.num_iterations.to_string(),
        "--coord-key".into(),
        params.coord_key.into(),
        "--cluster-key".into(),
        params.cluster_key.into(),
    ];

    if let Some(out) = params.output {
        let s = out
            .to_str()
            .with_context(|| format!("output path must be UTF-8: {}", out.display()))?;
        argv.push("-o".into());
        argv.push(s.into());
    }
    if let Some(x) = params.x_col {
        if !x.trim().is_empty() {
            argv.push("--x-col".into());
            argv.push(x.into());
        }
    }
    if let Some(y) = params.y_col {
        if !y.trim().is_empty() {
            argv.push("--y-col".into());
            argv.push(y.into());
        }
    }
    if !params.preprocess {
        argv.push("--no-preprocess".into());
    }
    if params.verbose {
        argv.push("-v".into());
    }

    let argv_refs: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
    let script = if let Some(p) = resolve_banksy_cluster_py_path() {
        fs::read_to_string(&p).with_context(|| format!("read BANKSY script {}", p.display()))?
    } else {
        BANKSY_SCRIPT.to_string()
    };
    uv_python_stdin(
        UV_WITH_BANKSY,
        &script,
        &argv_refs,
        false,
        "banksy clustering",
    )?;
    Ok(())
}
