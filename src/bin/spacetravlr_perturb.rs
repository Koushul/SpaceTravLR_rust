use clap::Parser;
use space_trav_lr_rust::betadata::write_betadata_feather;
use space_trav_lr_rust::perturb::{PerturbTarget, perturb_with_targets};
use space_trav_lr_rust::perturb_mode::{
    PerturbRuntime, interactive_run_toml_prompt, run_interactive,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "spacetravlr-perturb",
    version,
    about = "SpaceTravLR perturbation UI mode."
)]
struct Cli {
    #[arg(
        long = "run-toml",
        value_name = "PATH",
        help = "Path to spacetravlr_run_repro.toml. If omitted, an interactive prompt is shown (unless --export is used)."
    )]
    run_toml: Option<PathBuf>,

    #[arg(
        long = "export",
        value_name = "PATH",
        help = "Write simulated expression as feather (rows = cells, columns = CellID + genes); exit. Requires --run-toml and --gene."
    )]
    export: Option<PathBuf>,

    #[arg(long = "gene", help = "Gene to perturb (use with --export)")]
    gene: Option<String>,

    #[arg(long = "desired-expr", default_value_t = 0.0)]
    desired_expr: f64,

    #[arg(
        long = "n-propagation",
        help = "Override [perturbation].n_propagation from the TOML (use with --export)"
    )]
    n_propagation: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let run_toml = if cli.export.is_some() {
        cli
            .run_toml
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--run-toml is required with --export"))?
    } else {
        match &cli.run_toml {
            Some(p) => p.clone(),
            None => interactive_run_toml_prompt()?,
        }
    };

    if let Some(export_path) = &cli.export {
        let gene = cli
            .gene
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--gene is required with --export"))?;
        let mut runtime = PerturbRuntime::from_run_toml(run_toml.as_path())?;
        if let Some(n) = cli.n_propagation {
            runtime.perturb_cfg.n_propagation = n;
        }
        if !runtime.gene_names.iter().any(|g| g == gene) {
            anyhow::bail!("Gene '{}' is not present in AnnData var_names.", gene);
        }
        let targets = vec![PerturbTarget {
            gene: gene.to_string(),
            desired_expr: cli.desired_expr,
            cell_indices: None,
        }];
        let result = perturb_with_targets(
            &runtime.bb,
            &runtime.gene_mtx,
            &runtime.gene_names,
            &runtime.xy,
            &runtime.rw_ligands_init,
            &runtime.rw_tfligands_init,
            &targets,
            &runtime.perturb_cfg,
            &runtime.lr_radii,
            None,
            None,
            None,
        )
        .map_err(|_| anyhow::anyhow!("perturbation failed"))?;
        let p = export_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("export path must be UTF-8"))?;
        write_betadata_feather(p, "CellID", &runtime.obs_names, &runtime.gene_names, &result.simulated)?;
        eprintln!(
            "Wrote {} ({} cells × {} genes, n_propagation={})",
            export_path.display(),
            runtime.obs_names.len(),
            runtime.gene_names.len(),
            runtime.perturb_cfg.n_propagation
        );
        return Ok(());
    }

    let runtime = PerturbRuntime::from_run_toml(run_toml.as_path())?;
    run_interactive(runtime)
}
