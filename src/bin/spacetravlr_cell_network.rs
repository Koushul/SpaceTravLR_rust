use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use spacetravlr::cell_comm_network::{CellCommNetworkParams, aggregate_lr_tfl_communication_edges};
use spacetravlr::perturb_mode::PerturbRuntime;

#[derive(Parser, Debug)]
#[command(
    name = "spacetravlr-cell-network",
    version,
    about = "Cell–cell communication graph from a finished run (repro TOML + betadata): edges aggregate |∂(trained targets)/∂(sender ligand expression)| over LR/TFL channels and Gaussian proximity."
)]
struct Cli {
    #[arg(
        long = "run-toml",
        value_name = "PATH",
        help = "Path to spacetravlr_run_repro.toml (same as spacetravlr-perturb)."
    )]
    run_toml: PathBuf,

    #[arg(
        long = "edges-out",
        value_name = "PATH",
        help = "Write edge list CSV: source (sender cell obs), target (receiver), weight (L1 aggregate of |∂ŷ/∂ligand| over trained targets and LR/TFL terms)."
    )]
    edges_out: PathBuf,

    #[arg(
        long = "nodes-out",
        value_name = "PATH",
        help = "Optional CSV of cell_id, x, y for the same row order as the run."
    )]
    nodes_out: Option<PathBuf>,

    #[arg(
        long = "threshold",
        default_value_t = 0.0,
        help = "Drop edges with weight ≤ this value (after summing absolute contributions)."
    )]
    threshold: f64,

    #[arg(
        long = "max-edges",
        value_name = "N",
        help = "Keep only the N strongest edges by weight (after threshold). Omit for all edges."
    )]
    max_edges: Option<usize>,

    #[arg(
        long = "include-self",
        help = "Include i→i edges (normally excluded)."
    )]
    include_self: bool,

    #[arg(
        long = "no-contact-cutoff",
        help = "Ignore [spatial].contact_distance when building the ligand Jacobian (full Gaussian support)."
    )]
    no_contact_cutoff: bool,

    #[arg(
        long = "beta-scale",
        value_name = "FLOAT",
        help = "Override [perturbation].beta_scale_factor for the Jacobian (default: from run TOML)."
    )]
    beta_scale: Option<f32>,

    #[arg(
        long = "min-expression",
        default_value_t = 1e-9,
        help = "Mask expression at or below this when applying receptor/TF factors (matches perturbation default)."
    )]
    min_expression: f64,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    eprintln!("Loading run from {} …", cli.run_toml.display());
    let rt = PerturbRuntime::from_run_toml(&cli.run_toml)?;

    let params = CellCommNetworkParams {
        beta_scale_factor: cli
            .beta_scale
            .unwrap_or(rt.perturb_cfg.beta_scale_factor as f32),
        min_expression: cli.min_expression,
        edge_threshold_abs: cli.threshold,
        include_self_loops: cli.include_self,
        ignore_contact_distance: cli.no_contact_cutoff,
    };

    eprintln!(
        "Aggregating LR/TFL communication for {} cells, {} target models …",
        rt.obs_names.len(),
        rt.bb.data.len()
    );
    let (w, _obs_to_idx) = aggregate_lr_tfl_communication_edges(&rt, &params);

    let n = w.nrows();
    let mut edges: Vec<(String, String, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let v = w[[i, j]];
            if v > 0.0 {
                edges.push((rt.obs_names[j].clone(), rt.obs_names[i].clone(), v));
            }
        }
    }
    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(cap) = cli.max_edges {
        edges.truncate(cap.min(edges.len()));
    }

    {
        let f = File::create(&cli.edges_out)
            .map_err(|e| anyhow::anyhow!("create {}: {e}", cli.edges_out.display()))?;
        let mut wtr = BufWriter::new(f);
        writeln!(wtr, "source,target,weight")?;
        for (s, t, wt) in &edges {
            writeln!(wtr, "{},{},{}", escape_csv(s), escape_csv(t), wt)?;
        }
    }
    eprintln!(
        "Wrote {} edges to {}",
        edges.len(),
        cli.edges_out.display()
    );

    if let Some(path) = cli.nodes_out {
        let f = File::create(&path).map_err(|e| anyhow::anyhow!("create {}: {e}", path.display()))?;
        let mut wtr = BufWriter::new(f);
        writeln!(wtr, "cell_id,x,y")?;
        for (i, id) in rt.obs_names.iter().enumerate() {
            let x = rt.xy[[i, 0]];
            let y = if rt.xy.ncols() > 1 { rt.xy[[i, 1]] } else { 0.0 };
            writeln!(wtr, "{},{},{}", escape_csv(id), x, y)?;
        }
        eprintln!("Wrote {} nodes to {}", rt.obs_names.len(), path.display());
    }

    Ok(())
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
