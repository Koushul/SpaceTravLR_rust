use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::Context;
use clap::Parser;
use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use ndarray::Array2;
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use spacetravlr::spatial_estimator::{
    read_h5ad_expression_dense_f64, read_h5ad_obs_column_str, read_h5ad_obsm_dense_f64,
    read_h5ad_var_names,
};
use spacetravlr::transition_umap::{
    SignatureUmapParams, TransitionUmapParams, compute_umap_transition_grid,
    grid_cosine_alignment_to_signature_gradient, mann_whitney_u_normal_two_sided,
    per_cell_alignment_from_grid, umap_grid_points,
};

#[derive(Parser, Debug)]
#[command(
    name = "spacetravlr-alignment",
    about = "VirtualTissue-style UMAP alignment (Rust): transition quiver vs pseudotime-gradient reference + random null. Default pseudotime matches embeds2perturb.ipynb (Palantir branched) via scripts/compute_branched_pseudotime_notebook.py."
)]
struct Cli {
    #[arg(long, help = "AnnData .h5ad (same cells/order as training)")]
    h5ad: PathBuf,
    #[arg(long, default_value = "imputed_count")]
    layer: String,
    #[arg(long, default_value = "X_umap")]
    umap_key: String,
    #[arg(
        long,
        default_value = "cell_type_2",
        help = "obs column for per-cell-type summaries (embeds2perturb uses cell_type_2)"
    )]
    annot: String,
    #[arg(
        long,
        help = "obs column for pseudotime (smoothed on UMAP before ref gradient). Mutually exclusive with --pseudotime-csv / --notebook-branched-pseudotime."
    )]
    pseudotime_key: Option<String>,
    #[arg(
        long,
        help = "CSV with columns obs_name,pseudotime (e.g. from scripts/compute_branched_pseudotime_notebook.py). Rows must cover AnnData obs order."
    )]
    pseudotime_csv: Option<PathBuf>,
    #[arg(
        long,
        action = clap::ArgAction::SetTrue,
        help = "Use UMAP x as pseudotime (no Palantir). Default is embeds2perturb-style Palantir branched pseudotime."
    )]
    pseudotime_umap_x: bool,
    #[arg(
        long,
        action = clap::ArgAction::SetTrue,
        help = "If --branched-pseudotime-csv exists, load it instead of re-running Palantir."
    )]
    reuse_branched_pseudotime: bool,
    #[arg(long, default_value = "python3")]
    palantir_python: String,
    #[arg(
        long,
        help = "Write / reuse branched pseudotime CSV when using --notebook-branched-pseudotime (default: <temp>/spacetravlr_branched_pseudotime.csv)"
    )]
    branched_pseudotime_csv: Option<PathBuf>,
    #[arg(long, default_value = "cell_type_2")]
    branched_annot: String,
    #[arg(long, default_value = "Naive CD4 T")]
    branched_source_cell_type: String,
    #[arg(
        long = "branched-pair",
        value_name = "A|B",
        default_values = [
            "Naive CD4 T|T_follicular_helper",
            "Naive CD4 T|Th1",
            "Naive CD4 T|Th2",
        ],
        help = "Palantir subgraph pair A|B (repeat flag for more; defaults match embeds2perturb.ipynb)."
    )]
    branched_pairs: Vec<String>,
    #[arg(long, default_value_t = 1usize)]
    branched_n_source_cells: usize,
    #[arg(long, default_value_t = 10usize)]
    palantir_knn: usize,
    #[arg(long, default_value_t = 5usize)]
    palantir_n_components: usize,
    #[arg(long, default_value_t = 300usize)]
    smooth_k: usize,
    #[arg(long, default_value = "manifest.csv")]
    manifest: PathBuf,
    #[arg(long, default_value = "alignment_rust.csv")]
    out_csv: PathBuf,
    #[arg(long, default_value_t = 200usize)]
    n_neighbors: usize,
    #[arg(long, default_value_t = 0.05)]
    temperature: f64,
    #[arg(long, default_value_t = true)]
    remove_null: bool,
    #[arg(long, default_value_t = 1.0)]
    grid_scale: f64,
    #[arg(long, default_value_t = 4.0)]
    vector_scale: f64,
    #[arg(long, default_value_t = 1.0)]
    delta_rescale: f64,
    #[arg(long, default_value_t = 0.0)]
    magnitude_threshold: f64,
    #[arg(long, default_value_t = 100usize)]
    signature_grid_knn: usize,
    #[arg(long, default_value_t = 1.0)]
    signature_vector_scale: f64,
    #[arg(long, default_value_t = 0.0)]
    signature_magnitude_threshold: f64,
    #[arg(long, default_value_t = 2.0)]
    signature_gradient_gain: f64,
    #[arg(long, default_value_t = 1u64)]
    random_seed: u64,
    #[arg(long, default_value_t = -55.0)]
    random_low: f64,
    #[arg(long, default_value_t = 55.0)]
    random_high: f64,
}

fn obs_names_in_order(path: &std::path::Path) -> anyhow::Result<Vec<String>> {
    let adata = AnnData::<H5>::open(H5::open(path)?).map_err(|e| anyhow::anyhow!("{}", e))?;
    let v: Vec<String> = adata.obs_names().into_iter().collect();
    adata.close()?;
    Ok(v)
}

fn read_perturb_matrix(
    path: &std::path::Path,
    gene_names: &[String],
    obs_order: &[String],
) -> anyhow::Result<Array2<f64>> {
    let df = LazyFrame::scan_ipc(
        polars_utils::plpath::PlPath::from_string(path.display().to_string()),
        ScanArgsIpc::default(),
    )
    .with_context(|| format!("scan feather {:?}", path))?
    .collect()
    .with_context(|| format!("read feather {:?}", path))?;
    let cell_col = if df.column("CellID").is_ok() {
        "CellID"
    } else {
        anyhow::bail!("{:?}: expected CellID column", path);
    };
    let obs_index: HashMap<&str, usize> = obs_order
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut mat = Array2::<f64>::zeros((obs_order.len(), gene_names.len()));
    let id_series = df.column(cell_col)?.cast(&DataType::String)?;
    let ids = id_series.str()?;
    for row in 0..df.height() {
        let Some(id) = ids.get(row) else {
            continue;
        };
        let id = id.trim();
        let Some(&ri) = obs_index.get(id) else {
            continue;
        };
        for (j, g) in gene_names.iter().enumerate() {
            if let Ok(c) = df.column(g.as_str()) {
                let f = c.cast(&DataType::Float64)?;
                let ca = f.f64()?;
                if let Some(v) = ca.get(row) {
                    mat[[ri, j]] = v;
                }
            }
        }
    }
    Ok(mat)
}

fn notebook_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/compute_branched_pseudotime_notebook.py")
}

fn load_pseudotime_csv(path: &Path, obs_order: &[String]) -> anyhow::Result<Vec<f64>> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("open pseudotime csv {:?}", path))?;
    let mut map: HashMap<String, f64> = HashMap::new();
    for rec in rdr.records() {
        let rec = rec.with_context(|| format!("parse row in {:?}", path))?;
        if rec.len() < 2 {
            continue;
        }
        let name = rec[0].trim().to_string();
        let v: f64 = rec[1].trim().parse().unwrap_or(f64::NAN);
        map.insert(name, v);
    }
    let mut out = Vec::with_capacity(obs_order.len());
    let mut missing = 0usize;
    for id in obs_order {
        if let Some(&v) = map.get(id) {
            out.push(v);
        } else {
            missing += 1;
            out.push(f64::NAN);
        }
    }
    if missing > 0 {
        eprintln!(
            "warning: {} obs_names missing from pseudotime CSV {:?}",
            missing,
            path
        );
    }
    Ok(out)
}

fn run_notebook_branched_pseudotime(cli: &Cli, csv_out: &Path) -> anyhow::Result<()> {
    let script = notebook_script_path();
    anyhow::ensure!(
        script.is_file(),
        "missing script {:?} (clone full repo)",
        script
    );
    let mut cmd = Command::new(&cli.palantir_python);
    cmd.arg(&script)
        .arg("--h5ad")
        .arg(&cli.h5ad)
        .arg("--out-csv")
        .arg(csv_out)
        .arg("--annot")
        .arg(&cli.branched_annot)
        .arg("--source-cell-type")
        .arg(&cli.branched_source_cell_type)
        .arg("--n-source-cells")
        .arg(cli.branched_n_source_cells.to_string())
        .arg("--palantir-knn")
        .arg(cli.palantir_knn.to_string())
        .arg("--palantir-n-components")
        .arg(cli.palantir_n_components.to_string());
    for p in &cli.branched_pairs {
        cmd.arg("--pairs").arg(p);
    }
    let st = cmd.status().with_context(|| {
        format!(
            "failed to run {:?} {:?} (need Python + scanpy.external / Palantir)",
            cli.palantir_python, script
        )
    })?;
    anyhow::ensure!(st.success(), "Palantir script exited {:?}", st);
    Ok(())
}

fn smooth_on_umap(values: &[f64], umap: &[[f64; 2]], k: usize) -> Vec<f64> {
    let n = umap.len();
    if n == 0 {
        return vec![];
    }
    let tree_points: Vec<[f64; 2]> = umap.to_vec();
    let tree = ImmutableKdTree::<f64, 2>::new_from_slice(&tree_points);
    let k_take = k.max(1).min(n);
    let kq = std::num::NonZero::new(k_take).unwrap();
    (0..n)
        .map(|i| {
            let nns = tree.nearest_n::<SquaredEuclidean>(&umap[i], kq);
            let mut s = 0.0_f64;
            let mut c = 0_usize;
            for nn in nns {
                let j = nn.item as usize;
                if j < values.len() {
                    s += values[j];
                    c += 1;
                }
            }
            if c > 0 {
                s / c as f64
            } else {
                0.0
            }
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let n_pt_sources = (cli.pseudotime_key.is_some() as usize)
        + (cli.pseudotime_csv.is_some() as usize)
        + (cli.pseudotime_umap_x as usize);
    anyhow::ensure!(
        n_pt_sources <= 1,
        "choose at most one of --pseudotime-key, --pseudotime-csv, --pseudotime-umap-x"
    );
    let gene_names = read_h5ad_var_names(&cli.h5ad)?;
    let obs_order = obs_names_in_order(&cli.h5ad)?;
    let baseline = read_h5ad_expression_dense_f64(&cli.h5ad, &cli.layer)?;
    anyhow::ensure!(
        baseline.nrows() == obs_order.len(),
        "nrows mismatch baseline vs obs"
    );
    anyhow::ensure!(
        baseline.ncols() == gene_names.len(),
        "ncols mismatch baseline vs var"
    );

    let umap_arr = read_h5ad_obsm_dense_f64(&cli.h5ad, &cli.umap_key)
        .with_context(|| format!("read UMAP from {:?}", cli.h5ad))?;
    anyhow::ensure!(umap_arr.nrows() == obs_order.len(), "umap nrows");
    anyhow::ensure!(umap_arr.ncols() >= 2, "umap needs ≥2 columns");
    let umap: Vec<[f64; 2]> = (0..umap_arr.nrows())
        .map(|i| [umap_arr[[i, 0]], umap_arr[[i, 1]]])
        .collect();

    let pseudo_raw: Vec<f64> = if let Some(ref key) = cli.pseudotime_key {
        let s = read_h5ad_obs_column_str(&cli.h5ad, key)?;
        s.iter()
            .map(|t| t.parse::<f64>().unwrap_or(f64::NAN))
            .collect()
    } else if let Some(ref csv) = cli.pseudotime_csv {
        load_pseudotime_csv(csv, &obs_order)?
    } else if cli.pseudotime_umap_x {
        umap.iter().map(|p| p[0]).collect()
    } else {
        let csv_out = cli
            .branched_pseudotime_csv
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("spacetravlr_branched_pseudotime.csv"));
        if !(cli.reuse_branched_pseudotime && csv_out.is_file()) {
            run_notebook_branched_pseudotime(&cli, &csv_out)?;
        }
        load_pseudotime_csv(&csv_out, &obs_order)?
    };
    let pseudo_smoothed = smooth_on_umap(&pseudo_raw, &umap, cli.smooth_k);

    let cell_types = read_h5ad_obs_column_str(&cli.h5ad, &cli.annot)?;

    let tparams = TransitionUmapParams {
        n_neighbors: cli.n_neighbors,
        temperature: cli.temperature,
        remove_null: cli.remove_null,
        unit_directions: false,
        grid_scale: cli.grid_scale,
        vector_scale: cli.vector_scale,
        delta_rescale: cli.delta_rescale,
        magnitude_threshold: cli.magnitude_threshold,
        use_full_graph: false,
        full_graph_max_cells: 4096,
    };

    let sig_params = SignatureUmapParams {
        n_knn: cli.signature_grid_knn,
        grid_scale: cli.grid_scale,
        vector_scale: cli.signature_vector_scale,
        magnitude_threshold: cli.signature_magnitude_threshold,
        gradient_gain: cli.signature_gradient_gain,
    };

    let manifest = std::fs::read_to_string(&cli.manifest)
        .with_context(|| format!("read manifest {:?}", cli.manifest))?;
    let mut wtr = csv::Writer::from_path(&cli.out_csv)
        .with_context(|| format!("write {:?}", cli.out_csv))?;
    wtr.write_record([
        "label",
        "feather_path",
        "cell_type",
        "mean_alignment",
        "mean_alignment_rand",
        "delta_mean_vs_rand",
        "p_wilcox_normal_two_sided",
    ])?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(cli.random_seed);
    let rand_delta: Array2<f64> = Array2::from_shape_fn(baseline.dim(), |_| {
        rng.gen_range(cli.random_low..=cli.random_high)
    });
    let rand_grid = compute_umap_transition_grid(&baseline, &rand_delta, &umap, &tparams);
    let grid_pts = umap_grid_points(&umap, cli.grid_scale);
    let cos_rand_field = grid_cosine_alignment_to_signature_gradient(
        &umap,
        &rand_grid.vectors,
        &pseudo_smoothed,
        &rand_grid.vectors,
        100,
        &sig_params,
    );
    let cos_rand_cell = per_cell_alignment_from_grid(&umap, &grid_pts, &cos_rand_field);

    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() < 2 {
            continue;
        }
        let label = parts[0].trim();
        let feather_path = parts[1].trim();
        if label.eq_ignore_ascii_case("label") && feather_path.eq_ignore_ascii_case("feather_path") {
            continue;
        }
        let path = PathBuf::from(feather_path);
        let pert = read_perturb_matrix(&path, &gene_names, &obs_order)?;
        let delta = &pert - &baseline;
        let grid = compute_umap_transition_grid(&baseline, &delta, &umap, &tparams);
        let cos_sig = grid_cosine_alignment_to_signature_gradient(
            &umap,
            &grid.vectors,
            &pseudo_smoothed,
            &grid.vectors,
            100,
            &sig_params,
        );
        let cos_cell = per_cell_alignment_from_grid(&umap, &grid_pts, &cos_sig);

        let mut by_ct: HashMap<String, Vec<f64>> = HashMap::new();
        let mut by_ct_rand: HashMap<String, Vec<f64>> = HashMap::new();
        for i in 0..obs_order.len() {
            let ct = cell_types
                .get(i)
                .cloned()
                .unwrap_or_else(|| "NA".into());
            by_ct.entry(ct.clone()).or_default().push(cos_cell[i]);
            by_ct_rand.entry(ct).or_default().push(cos_rand_cell[i]);
        }
        for ct in by_ct.keys() {
            let a = by_ct.get(ct).map(|v| v.as_slice()).unwrap_or(&[]);
            let b = by_ct_rand.get(ct).map(|v| v.as_slice()).unwrap_or(&[]);
            let ma = if a.is_empty() {
                f64::NAN
            } else {
                a.iter().sum::<f64>() / a.len() as f64
            };
            let mb = if b.is_empty() {
                f64::NAN
            } else {
                b.iter().sum::<f64>() / b.len() as f64
            };
            let p = mann_whitney_u_normal_two_sided(a, b);
            wtr.write_record([
                label,
                feather_path,
                ct.as_str(),
                &format!("{:.8}", ma),
                &format!("{:.8}", mb),
                &format!("{:.8}", ma - mb),
                &format!("{:.6e}", p),
            ])?;
        }
    }
    wtr.flush()?;
    eprintln!("Wrote {}", cli.out_csv.display());
    Ok(())
}
