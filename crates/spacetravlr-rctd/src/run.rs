use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::{bail, ensure, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2};
use rctd_core::io_npz::load_q_matrices_npz;
use rctd_core::likelihood_tables::compute_spline_coefficients;
use rctd_core::{
    device_cpu, sync_device, BatchProgress, DeconvMode, DeconvolutionOutput, PreparedData,
    RctdConfig, RctdDevice, run_deconvolution,
};

#[cfg(feature = "wgpu")]
use rctd_core::{default_device, init_wgpu};

use crate::ref_adata;
use crate::ref_rds;

const Q_MATRICES_URL: &str =
    "https://github.com/p-gueguen/rctd-py/releases/download/v0.1.1/q_matrices.npz";

#[derive(Debug, Clone)]
pub struct RctdCliArgs {
    pub spatial: PathBuf,
    pub reference: PathBuf,
    pub cell_type_col: String,
    pub ref_rows_are_types: bool,
    pub ref_cell_min: usize,
    pub ref_min_umi: u32,
    pub ref_max_cells_per_type: usize,
    pub q_matrices: Option<PathBuf>,
    pub sigma: i32,
    pub mode: DeconvMode,
    pub batch_size: usize,
    pub output_prefix: Option<PathBuf>,
    pub gpu: bool,
}

fn open_h5ad(path: &Path) -> Result<AnnData<H5>> {
    let store = H5::open(path).with_context(|| format!("open {}", path.display()))?;
    AnnData::open(store).with_context(|| format!("read AnnData {}", path.display()))
}

fn input_is_rds(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("rds"))
}

fn align_genes_spatial_ref(
    spatial_genes: &[String],
    ref_genes: &[String],
    counts: &Array2<f64>,
    profiles_kg: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let map_ref: HashMap<&str, usize> = ref_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
    let mut si = Vec::new();
    let mut ri = Vec::new();
    for (i, g) in spatial_genes.iter().enumerate() {
        if let Some(&j) = map_ref.get(g.as_str()) {
            si.push(i);
            ri.push(j);
        }
    }
    if si.is_empty() {
        bail!("no overlapping gene names between spatial and reference");
    }
    let n = si.len();
    let k = profiles_kg.nrows();
    let mut c = Array2::<f64>::zeros((counts.nrows(), n));
    let mut p = Array2::<f64>::zeros((k, n));
    for (new_g, (is, ir)) in si.iter().zip(ri.iter()).enumerate() {
        c.column_mut(new_g).assign(&counts.column(*is));
        p.column_mut(new_g).assign(&profiles_kg.column(*ir));
    }
    let profiles_gk = p.t().to_owned();
    Ok((c, profiles_gk))
}

pub fn resolve_q_matrices_path(arg: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = arg {
        if !p.exists() {
            bail!(
                "q_matrices.npz not found at {}",
                p.display()
            );
        }
        return Ok(p);
    }

    let cache_path = dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("rctd")
        .join("q_matrices.npz");

    if cache_path.exists() {
        return Ok(cache_path);
    }

    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }

    eprintln!(
        "Downloading Q-matrices from {} to {} ...",
        Q_MATRICES_URL,
        cache_path.display()
    );
    let response = ureq::get(Q_MATRICES_URL)
        .call()
        .map_err(|e| anyhow::Error::new(io::Error::new(io::ErrorKind::Other, format!("{e}"))))
        .context("download q_matrices.npz")?;

    let mut reader = response.into_reader();
    let mut file = fs::File::create(&cache_path)?;
    io::copy(&mut reader, &mut file)?;
    file.flush()?;
    eprintln!("Saved Q-matrices to {}", cache_path.display());

    Ok(cache_path)
}

fn resolve_rctd_device(gpu: bool) -> Result<RctdDevice> {
    if gpu {
        #[cfg(feature = "wgpu")]
        {
            let d = default_device();
            init_wgpu(&d);
            return Ok(d);
        }
        #[cfg(not(feature = "wgpu"))]
        {
            bail!(
                "RCTD GPU was requested (--gpu) but this build does not include wgpu; rebuild with:\n  cargo build -p spacetravlr --features rctd,rctd-wgpu"
            );
        }
    }
    Ok(device_cpu())
}

fn heuristic_progress_len(n_pixels: usize, mode: DeconvMode, batch_size: usize) -> u64 {
    let bs = batch_size.max(1);
    let full_batches = (n_pixels + bs - 1) / bs;
    match mode {
        DeconvMode::Full => full_batches as u64,
        DeconvMode::Doublet => {
            let triple_est = (n_pixels * 3 + bs - 1) / bs;
            let single_est = (n_pixels * 2 + bs - 1) / bs;
            (full_batches + triple_est + single_est + full_batches * 2) as u64
        }
        DeconvMode::Multi => ((full_batches * 8) as u64).max(n_pixels as u64),
    }
}

fn csv_header_field(name: &str) -> String {
    if name.contains(',')
        || name.contains('"')
        || name.contains('\r')
        || name.contains('\n')
    {
        format!("\"{}\"", name.replace('"', "\"\""))
    } else {
        name.to_owned()
    }
}

fn write_weights_csv(
    path: &Path,
    w: &Array2<f64>,
    row_names: &[String],
    col_names: &[String],
) -> Result<()> {
    ensure!(
        w.nrows() == row_names.len(),
        "weights have {} rows but {} spot / obs names",
        w.nrows(),
        row_names.len()
    );
    ensure!(
        w.ncols() == col_names.len(),
        "weights have {} columns but {} cell type names",
        w.ncols(),
        col_names.len()
    );
    let mut f = std::io::BufWriter::new(fs::File::create(path)?);
    let mut header_fields: Vec<String> = vec![csv_header_field("obs")];
    header_fields.extend(col_names.iter().map(|n| csv_header_field(n)));
    writeln!(f, "{}", header_fields.join(","))?;
    for (i, row) in w.rows().into_iter().enumerate() {
        let mut fields: Vec<String> = vec![csv_header_field(&row_names[i])];
        fields.extend(
            row.iter()
                .map(|x| format!("{:.8e}", x))
                .collect::<Vec<_>>(),
        );
        writeln!(f, "{}", fields.join(","))?;
    }
    Ok(())
}

pub fn run_rctd(args: RctdCliArgs) -> Result<()> {
    let dev = resolve_rctd_device(args.gpu)?;

    let (counts, spatial_genes, spatial_obs_names) = if input_is_rds(&args.spatial) {
        ref_rds::load_spatial_rds(args.spatial.as_path())
            .context("spatial .rds (spacexr::SpatialRNA; needs Rscript)")?
    } else {
        let spatial_ad = open_h5ad(&args.spatial)?;
        let spatial_genes = spatial_ad.var_names().into_vec();
        let counts = ref_adata::x_to_dense_f64(&spatial_ad)?;
        if counts.ncols() != spatial_genes.len() || counts.nrows() != spatial_ad.n_obs() {
            bail!(
                "spatial X shape {:?} does not match n_obs × n_vars",
                counts.dim()
            );
        }
        let spatial_obs_names = spatial_ad.obs_names().into_vec();
        if spatial_obs_names.len() != counts.nrows() {
            bail!("spatial obs index length does not match n_obs");
        }
        (counts, spatial_genes, spatial_obs_names)
    };

    let (profiles_kg, cell_type_names, ref_genes) = if input_is_rds(&args.reference) {
        if args.ref_rows_are_types {
            ref_rds::load_reference_profiles_rds(args.reference.as_path()).context(
                "reference .rds type profiles (matrix rows = types, cols = genes)",
            )?
        } else {
            ref_rds::load_reference_sc_rds(
                args.reference.as_path(),
                args.ref_cell_min,
                f64::from(args.ref_min_umi),
                args.ref_max_cells_per_type,
            )
            .context("reference .rds (spacexr::Reference)")?
        }
    } else {
        let ref_ad = open_h5ad(&args.reference)?;
        let ref_genes = ref_ad.var_names().into_vec();
        let pair = if args.ref_rows_are_types {
            let p = ref_adata::x_to_dense_f64(&ref_ad)?;
            if p.ncols() != ref_genes.len() || p.nrows() != ref_ad.n_obs() {
                bail!(
                    "reference X shape {:?} does not match n_obs × n_vars ({} × {})",
                    p.dim(),
                    ref_ad.n_obs(),
                    ref_genes.len()
                );
            }
            let names = ref_ad.obs_names().into_vec();
            if names.len() != p.nrows() {
                bail!("obs index length does not match reference n_obs");
            }
            (p, names)
        } else {
            let obs = ref_ad.read_obs().context("read reference obs")?;
            if obs.get_column_index(&args.cell_type_col).is_none() {
                bail!(
                    "obs has no column {:?}; use --ref-rows-are-types if reference X is K×G",
                    args.cell_type_col
                );
            }
            ref_adata::single_cell_reference_profiles(
                &ref_ad,
                &args.cell_type_col,
                args.ref_cell_min,
                f64::from(args.ref_min_umi),
                args.ref_max_cells_per_type,
            )
            .with_context(|| "single-cell reference profiles")?
        };
        (pair.0, pair.1, ref_genes)
    };

    let (counts, profiles_gk_raw) =
        align_genes_spatial_ref(&spatial_genes, &ref_genes, &counts, &profiles_kg)?;
    let norm_profiles = ref_adata::normalize_columns(&profiles_gk_raw);
    if cell_type_names.len() != norm_profiles.ncols() {
        bail!("cell type count does not match norm_profiles columns");
    }
    let numi: Array1<f64> = counts.sum_axis(ndarray::Axis(1));
    let n_pixels = counts.nrows();

    let q_path = resolve_q_matrices_path(args.q_matrices)?;
    let (q_map, x_vals) = load_q_matrices_npz(q_path.as_path())?;
    let q_prefixed = format!("Q_{}", args.sigma);
    let q_mat = q_map
        .get(&q_prefixed)
        .or_else(|| q_map.get(&args.sigma.to_string()))
        .with_context(|| {
            format!(
                "sigma {} not in q_matrices.npz (try keys like Q_{})",
                args.sigma, args.sigma
            )
        })?
        .clone();
    let sq_mat = compute_spline_coefficients(&q_mat, &x_vals);

    let config = RctdConfig::default();
    let data = PreparedData {
        spatial_counts: counts,
        spatial_numi: numi,
        norm_profiles,
        cell_type_names,
        q_mat,
        sq_mat,
        x_vals,
    };

    let len = heuristic_progress_len(n_pixels, args.mode, args.batch_size).max(1);
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} RCTD work units {msg}",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message("deconvolution");

    let pb_arc = Arc::new(pb);
    let progress: Option<BatchProgress> = Some(Arc::new({
        let pb = pb_arc.clone();
        move |units: usize| {
            pb.inc(units as u64);
        }
    }));

    let out = run_deconvolution(
        &data,
        &config,
        args.mode,
        args.batch_size,
        &dev,
        progress,
    );
    sync_device(&dev);
    pb_arc.finish_with_message("RCTD finished");

    if let Some(prefix) = args.output_prefix {
        let row_names = spatial_obs_names.as_slice();
        let col_names = data.cell_type_names.as_slice();
        match out {
            DeconvolutionOutput::Full(r) => {
                write_weights_csv(
                    &prefix.with_extension("weights.csv"),
                    &r.weights,
                    row_names,
                    col_names,
                )?;
            }
            DeconvolutionOutput::Doublet(r) => {
                write_weights_csv(
                    &prefix.with_extension("weights.csv"),
                    &r.weights,
                    row_names,
                    col_names,
                )?;
            }
            DeconvolutionOutput::Multi(r) => {
                write_weights_csv(
                    &prefix.with_extension("weights.csv"),
                    &r.weights,
                    row_names,
                    col_names,
                )?;
            }
        }
        eprintln!("wrote {}", prefix.with_extension("weights.csv").display());
    }

    Ok(())
}
