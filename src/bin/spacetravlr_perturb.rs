use clap::Parser;
use space_trav_lr_rust::perturb_mode::{PerturbRuntime, interactive_run_toml_prompt, run_interactive};
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
        help = "Path to spacetravlr_run_repro.toml. If omitted, an interactive prompt is shown."
    )]
    run_toml: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let run_toml = match cli.run_toml {
        Some(p) => p,
        None => interactive_run_toml_prompt()?,
    };
    let runtime = PerturbRuntime::from_run_toml(&run_toml)?;
    run_interactive(runtime)
}
