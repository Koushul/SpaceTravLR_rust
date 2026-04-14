use anyhow::Context;
use clap::Parser;
use std::path::{Path, PathBuf};

use spacetravlr::celloracle::{
    build_coef_matrix, filter_links_p_max, infer_grn_per_cluster, infer_grn_whole, scale_gem_no_center,
    write_coef_matrix_exports, write_links_csv, write_links_parquet,
};
use spacetravlr::config::{canonical_adata_stem, expand_user_path};
use spacetravlr::network::{GeneNetwork, infer_species};
use spacetravlr::scanpy_preprocess::{
    SpatialMicronsOptions, ensure_training_adata_ready, resolve_magic_batch_obs_column,
};
use spacetravlr::{
    read_h5ad_expression_dense_f64, read_h5ad_obs_column_str, read_h5ad_var_names,
};

#[derive(Parser, Debug)]
#[command(name = "spacetravlr-celloracle")]
#[command(about = "CellOracle-style TF GRN (Bayesian ridge) with SpaceTravLR GRN priors")]
struct Cli {
    #[arg(long)]
    h5ad: PathBuf,

    #[arg(long, help = "GRN species (human|mouse); omit to infer from var gene symbols")]
    species: Option<String>,

    #[arg(long)]
    network_data_dir: Option<String>,

    #[arg(long, default_value = "imputed_count")]
    layer: String,

    #[arg(long)]
    skip_preprocess: bool,

    #[arg(long)]
    output_dir: Option<PathBuf>,

    #[arg(long)]
    magic_batch_obs: Option<String>,

    #[arg(long)]
    spatial_species: Option<String>,

    #[arg(long)]
    spatial_median_nn_target_um: Option<f64>,

    #[arg(long, default_value_t = false)]
    skip_spatial_microns: bool,

    #[arg(long)]
    obs_key: Option<String>,

    #[arg(long, default_value_t = false)]
    per_cluster: bool,

    #[arg(long)]
    links_out: Option<PathBuf>,

    #[arg(
        long,
        help = "Path prefix: writes {prefix}_genes.json and {prefix}_coef_matrix.npy"
    )]
    coef_out: Option<PathBuf>,

    #[arg(long)]
    p_max: Option<f64>,

    #[arg(long, help = "Rayon thread count (omit for default)")]
    threads: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let h5ad_expanded = expand_user_path(cli.h5ad.to_string_lossy().as_ref());
    let h5ad_path = PathBuf::from(&h5ad_expanded);

    let output_dir = cli.output_dir.clone().unwrap_or_else(|| {
        h5ad_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    });
    std::fs::create_dir_all(&output_dir).with_context(|| format!("mkdir {:?}", output_dir))?;

    let mut adata_path = h5ad_expanded.clone();

    let h5ad_for_read = PathBuf::from(expand_user_path(adata_path.trim()));
    if !h5ad_for_read.is_file() {
        anyhow::bail!("AnnData not found at {}", h5ad_for_read.display());
    }
    let var_names_infer = read_h5ad_var_names(&h5ad_for_read).context("read var_names")?;

    let grn_species: String = match cli.species.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => s.to_lowercase(),
        None => infer_species(&var_names_infer)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "could not infer GRN species from var_names; pass --species human or mouse"
                )
            })?
            .to_string(),
    };

    if !cli.skip_preprocess {
        let magic_batch = resolve_magic_batch_obs_column(cli.magic_batch_obs.as_deref(), None);
        let spatial_species_cli = cli
            .spatial_species
            .as_deref()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty());
        let spatial_microns = SpatialMicronsOptions {
            skip: cli.skip_spatial_microns,
            species: spatial_species_cli.unwrap_or_else(|| grn_species.clone()),
            target_median_nn_um: cli.spatial_median_nn_target_um,
        };
        ensure_training_adata_ready(
            &mut adata_path,
            &output_dir,
            Path::new(&h5ad_expanded),
            magic_batch.as_deref(),
            spatial_microns,
        )?;
    }

    let adata_in = PathBuf::from(expand_user_path(adata_path.trim()));
    if !adata_in.is_file() {
        anyhow::bail!("AnnData not found at {}", adata_in.display());
    }

    let var_names = read_h5ad_var_names(&adata_in).context("read var_names")?;
    let gem = read_h5ad_expression_dense_f64(&adata_in, cli.layer.trim())
        .with_context(|| format!("read layer {:?}", cli.layer))?;
    anyhow::ensure!(
        gem.ncols() == var_names.len(),
        "expression shape {:?} vs len(var_names) {}",
        gem.dim(),
        var_names.len()
    );

    let network = GeneNetwork::new(
        grn_species.as_str(),
        &var_names,
        cli.network_data_dir.as_deref(),
    )?;
    let tf_by_target = network.grn_regulators_by_target()?;

    let gem_scaled = scale_gem_no_center(&gem);

    let run_infer = || {
        if cli.per_cluster {
            let key = cli.obs_key.as_deref().unwrap_or("cell_type");
            let obs = read_h5ad_obs_column_str(&adata_in, key)
                .with_context(|| format!("read obs[{key}]"))?;
            infer_grn_per_cluster(&gem, &gem_scaled, &var_names, &tf_by_target, &obs, true)
        } else {
            infer_grn_whole(&gem, &gem_scaled, &var_names, &tf_by_target, true)
        }
    };

    let mut links = if let Some(n) = cli.threads.filter(|n| *n > 0) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .context("rayon ThreadPool")?;
        pool.install(run_infer)?
    } else {
        run_infer()?
    };

    if let Some(pm) = cli.p_max {
        links = filter_links_p_max(links, pm);
    }

    if let Some(ref p) = cli.links_out {
        let csv = p
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("csv"));
        if csv {
            write_links_csv(p, &links).with_context(|| format!("write {:?}", p))?;
        } else {
            write_links_parquet(p, &links).with_context(|| format!("write {:?}", p))?;
        }
    }

    if let Some(prefix) = cli.coef_out {
        let pre = if prefix.as_os_str().is_empty() {
            let stem = canonical_adata_stem(&adata_in);
            output_dir.join(format!("{stem}_celloracle_coef"))
        } else {
            prefix
        };
        let mat = build_coef_matrix(&var_names, &links);
        write_coef_matrix_exports(&pre, &var_names, &mat)
            .with_context(|| format!("write coef exports under {:?}", pre))?;
    }

    Ok(())
}
