use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use serde::Deserialize;

use crate::ref_adata;

#[derive(Debug, Deserialize)]
struct RdsMeta {
    format: String,
    kind: String,
    nrows: usize,
    ncols: usize,
}

const BRIDGE_R: &str = include_str!("../resources/export_rds_bridge.R");

fn read_lines(path: &Path) -> Result<Vec<String>> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    Ok(s.lines().map(|l| l.to_string()).collect())
}

fn read_matrix_bin(path: &Path, nrows: usize, ncols: usize) -> Result<Array2<f64>> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let byte_len = nrows
        .checked_mul(ncols)
        .and_then(|n| n.checked_mul(8))
        .context("matrix dimensions overflow")?;
    let mut buf = vec![0u8; byte_len];
    f.read_exact(&mut buf)
        .with_context(|| format!("read {} bytes from {}", byte_len, path.display()))?;
    let mut v = Vec::with_capacity(nrows * ncols);
    for chunk in buf.chunks_exact(8) {
        v.push(f64::from_le_bytes(
            chunk.try_into().expect("8-byte chunk"),
        ));
    }
    Array2::from_shape_vec((nrows, ncols), v).map_err(|e| anyhow::anyhow!("{e}"))
}

fn run_r_bridge(rds: &Path, out_dir: &Path, bridge_arg: &str) -> Result<()> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let script_path = out_dir.join("_export_rds_bridge.R");
    std::fs::write(&script_path, BRIDGE_R).context("write temp R script")?;
    let st = Command::new("Rscript")
        .arg(script_path.as_os_str())
        .arg(rds.as_os_str())
        .arg(out_dir.as_os_str())
        .arg(bridge_arg)
        .status()
        .context(
            "failed to run `Rscript` (install R from https://www.r-project.org/ and ensure `Rscript` is on PATH)",
        )?;
    let _ = std::fs::remove_file(&script_path);
    if !st.success() {
        bail!("Rscript export_rds_bridge.R failed: {st}");
    }
    Ok(())
}

fn read_meta(dir: &Path) -> Result<RdsMeta> {
    let p = dir.join("meta.json");
    let s = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    serde_json::from_str(&s).with_context(|| format!("parse {}", p.display()))
}

/// Spatial counts as **pixels × genes** (same as AnnData `X`), gene names (var), and obs / spot names (row index).
pub fn load_spatial_rds(path: &Path) -> Result<(Array2<f64>, Vec<String>, Vec<String>)> {
    let work = tempfile::tempdir().context("tempdir for R bridge")?;
    run_r_bridge(path, work.path(), "spatial")?;
    let meta = read_meta(work.path())?;
    if meta.format != "rctd_rds_bridge_v1" || meta.kind != "spatial_rna" {
        bail!(
            "unexpected bridge meta format={:?} kind={:?}",
            meta.format,
            meta.kind
        );
    }
    let genes = read_lines(&work.path().join("genes.txt"))?;
    let obs = read_lines(&work.path().join("obs.txt"))?;
    if genes.len() != meta.ncols {
        bail!(
            "genes.txt length {} != meta.ncols {}",
            genes.len(),
            meta.ncols
        );
    }
    if obs.len() != meta.nrows {
        bail!(
            "obs.txt length {} != meta.nrows {}",
            obs.len(),
            meta.nrows
        );
    }
    let m = read_matrix_bin(&work.path().join("matrix.bin"), meta.nrows, meta.ncols)?;
    Ok((m, genes, obs))
}

/// Single-cell reference: spacexr `Reference` → same **K×G** profiles as `single_cell_reference_profiles` + gene names.
pub fn load_reference_sc_rds(
    path: &Path,
    cell_min: usize,
    min_umi: f64,
    n_max_cells: usize,
) -> Result<(Array2<f64>, Vec<String>, Vec<String>)> {
    let work = tempfile::tempdir().context("tempdir for R bridge")?;
    run_r_bridge(path, work.path(), "ref_sc")?;
    let meta = read_meta(work.path())?;
    if meta.format != "rctd_rds_bridge_v1" || meta.kind != "reference_sc" {
        bail!(
            "unexpected bridge meta format={:?} kind={:?}",
            meta.format,
            meta.kind
        );
    }
    let genes = read_lines(&work.path().join("genes.txt"))?;
    let obs = read_lines(&work.path().join("obs.txt"))?;
    let cell_types = read_lines(&work.path().join("cell_types.txt"))?;
    if genes.len() != meta.ncols {
        bail!(
            "genes.txt length {} != meta.ncols {}",
            genes.len(),
            meta.ncols
        );
    }
    if obs.len() != meta.nrows || cell_types.len() != meta.nrows {
        bail!(
            "obs / cell_types length mismatch: nrows {} obs {} ct {}",
            meta.nrows,
            obs.len(),
            cell_types.len()
        );
    }
    let x = read_matrix_bin(&work.path().join("matrix.bin"), meta.nrows, meta.ncols)?;
    let (profiles, types) = ref_adata::single_cell_reference_profiles_from_arrays(
        &x,
        &cell_types,
        cell_min,
        min_umi,
        n_max_cells,
    )?;
    Ok((profiles, types, genes))
}

/// **K×G** type profile matrix (rows = types, cols = genes), same as `--ref-rows-are-types` h5ad layout.
pub fn load_reference_profiles_rds(path: &Path) -> Result<(Array2<f64>, Vec<String>, Vec<String>)> {
    let work = tempfile::tempdir().context("tempdir for R bridge")?;
    run_r_bridge(path, work.path(), "ref_profiles")?;
    let meta = read_meta(work.path())?;
    if meta.format != "rctd_rds_bridge_v1" || meta.kind != "reference_profiles" {
        bail!(
            "unexpected bridge meta format={:?} kind={:?}",
            meta.format,
            meta.kind
        );
    }
    let genes = read_lines(&work.path().join("genes.txt"))?;
    let types = read_lines(&work.path().join("obs.txt"))?;
    if genes.len() != meta.ncols || types.len() != meta.nrows {
        bail!("sidecar length mismatch with meta");
    }
    let profiles = read_matrix_bin(&work.path().join("matrix.bin"), meta.nrows, meta.ncols)?;
    Ok((profiles, types, genes))
}
