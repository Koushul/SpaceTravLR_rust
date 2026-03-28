use crate::betadata::{write_betadata_feather, Betabase, GeneMatrix};
use crate::config::{expand_user_path, SpaceshipConfig};
use crate::ligand::{calculate_weighted_ligands, calculate_weighted_ligands_grid};
use crate::perturb::{perturb_with_targets, PerturbConfig, PerturbTarget};
use anndata::data::{ArrayConvert, SelectInfoElem};
use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, AxisArraysOp, Backend};
use anndata_hdf5::H5;
use anyhow::Context;
use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub struct PerturbRuntime {
    pub run_toml_path: PathBuf,
    pub run_dir: PathBuf,
    pub cfg: SpaceshipConfig,
    pub gene_mtx: Array2<f64>,
    pub gene_names: Vec<String>,
    pub obs_names: Vec<String>,
    pub cell_types: Vec<usize>,
    pub bb: Betabase,
    pub xy: Array2<f64>,
    pub rw_ligands_init: GeneMatrix,
    pub rw_tfligands_init: GeneMatrix,
    pub lr_radii: HashMap<String, f64>,
    pub perturb_cfg: PerturbConfig,
}

fn array_data_to_dense_f64(data: ArrayData) -> anyhow::Result<Array2<f64>> {
    match data {
        ArrayData::Array(d) => d.try_convert(),
        ArrayData::CsrMatrix(csr) => {
            let csr_f64: nalgebra_sparse::csr::CsrMatrix<f64> = csr.try_convert()?;
            let mut out = Array2::<f64>::zeros((csr_f64.nrows(), csr_f64.ncols()));
            for (r, c, v) in csr_f64.triplet_iter() {
                out[[r, c]] = *v;
            }
            Ok(out)
        }
        ArrayData::CscMatrix(csc) => {
            let csc_f64: nalgebra_sparse::csc::CscMatrix<f64> = csc.try_convert()?;
            let mut out = Array2::<f64>::zeros((csc_f64.nrows(), csc_f64.ncols()));
            for (r, c, v) in csc_f64.triplet_iter() {
                out[[r, c]] = *v;
            }
            Ok(out)
        }
        ArrayData::CsrNonCanonical(non) => match non.canonicalize() {
            Ok(csr) => {
                let csr_f64: nalgebra_sparse::csr::CsrMatrix<f64> = csr.try_convert()?;
                let mut out = Array2::<f64>::zeros((csr_f64.nrows(), csr_f64.ncols()));
                for (r, c, v) in csr_f64.triplet_iter() {
                    out[[r, c]] = *v;
                }
                Ok(out)
            }
            Err(_) => anyhow::bail!("Failed to canonicalize non-canonical CSR matrix."),
        },
        ArrayData::DataFrame(_) => anyhow::bail!("Expected matrix array data, found dataframe."),
    }
}

fn read_expression_matrix_dense_f64<AnB: Backend>(
    adata: &AnnData<AnB>,
    layer: &str,
    slice: &[SelectInfoElem],
) -> anyhow::Result<Array2<f64>> {
    let data: ArrayData = if layer != "X" && !layer.is_empty() {
        if let Some(layer_elem) = adata.layers().get(layer) {
            layer_elem
                .slice(slice)?
                .ok_or_else(|| anyhow::anyhow!("Failed to slice layer {}", layer))?
        } else {
            let x_elem = adata.x();
            if x_elem.is_none() {
                anyhow::bail!("Layer '{}' not found and X is empty", layer);
            }
            x_elem
                .slice(slice)?
                .ok_or_else(|| anyhow::anyhow!("Failed to slice X"))?
        }
    } else {
        let x_elem = adata.x();
        if x_elem.is_none() {
            anyhow::bail!("X is empty");
        }
        x_elem
            .slice(slice)?
            .ok_or_else(|| anyhow::anyhow!("Failed to slice X"))?
    };
    array_data_to_dense_f64(data)
}

fn load_spatial_coords_f64<AnB: Backend>(adata: &AnnData<AnB>) -> anyhow::Result<Array2<f64>> {
    const KEYS: [&str; 3] = ["spatial", "X_spatial", "spatial_loc"];
    for key in KEYS {
        if let Some(arr) = adata.obsm().get_item::<Array2<f64>>(key)? {
            if arr.nrows() > 0 && arr.ncols() >= 2 {
                return Ok(arr);
            }
        }
        if let Some(arr) = adata.obsm().get_item::<Array2<f32>>(key)? {
            if arr.nrows() > 0 && arr.ncols() >= 2 {
                return Ok(arr.mapv(|v| v as f64));
            }
        }
    }
    anyhow::bail!("No usable 2D spatial coordinates in obsm.");
}

fn read_clusters_as_usize<AnB: Backend>(
    adata: &AnnData<AnB>,
    cluster_annot: &str,
) -> anyhow::Result<Vec<usize>> {
    let obs = adata.read_obs()?;
    let col = obs
        .column(cluster_annot)
        .with_context(|| format!("Missing obs column '{}'", cluster_annot))?;
    let as_f64 = col
        .as_materialized_series()
        .cast(&polars::prelude::DataType::Float64)?;
    let out = as_f64
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(0.0) as usize)
        .collect::<Vec<_>>();
    Ok(out)
}

fn sanitize_float(v: f64) -> String {
    format!("{:.6}", v).replace('-', "m").replace('.', "p")
}

fn request_output_dir(run_dir: &Path, selected: &[String], value: f64, n_propagation: usize) -> PathBuf {
    let mut sorted = selected.to_vec();
    sorted.sort();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    sorted.hash(&mut hasher);
    n_propagation.hash(&mut hasher);
    let hash = hasher.finish();
    run_dir
        .join("perturbations")
        .join(format!("genes_{hash:016x}_value_{}", sanitize_float(value)))
}

fn compute_initial_wl(
    gene_mtx: &Array2<f64>,
    gene_names: &[String],
    ligand_names: &[String],
    xy: &Array2<f64>,
    lr_radii: &HashMap<String, f64>,
    min_expression: f64,
    grid_factor: Option<f64>,
) -> GeneMatrix {
    let n_cells = gene_mtx.nrows();
    let gene_to_idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    let mut seen = HashSet::new();
    let unique: Vec<&String> = ligand_names
        .iter()
        .filter(|l| seen.insert(l.as_str()))
        .collect();

    let mut lig_names = Vec::new();
    let mut lig_data_cols = Vec::new();
    for &lig in &unique {
        if let Some(&gi) = gene_to_idx.get(lig.as_str()) {
            lig_names.push(lig.clone());
            let col: Vec<f64> = (0..n_cells)
                .map(|i| {
                    let v = gene_mtx[[i, gi]];
                    if v > min_expression { v } else { 0.0 }
                })
                .collect();
            lig_data_cols.push(col);
        }
    }

    if lig_names.is_empty() {
        return GeneMatrix::new(Array2::zeros((n_cells, 0)), Vec::new());
    }

    let n_lig = lig_names.len();
    let mut lig_data = Array2::zeros((n_cells, n_lig));
    for (j, col) in lig_data_cols.iter().enumerate() {
        for i in 0..n_cells {
            lig_data[[i, j]] = col[i];
        }
    }

    let mut radius_groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (j, name) in lig_names.iter().enumerate() {
        if let Some(&r) = lr_radii.get(name) {
            radius_groups.entry(r.to_bits()).or_default().push(j);
        }
    }

    let mut result = Array2::zeros((n_cells, n_lig));
    for (rbits, group) in &radius_groups {
        let radius = f64::from_bits(*rbits);
        let mut sub = Array2::zeros((n_cells, group.len()));
        for (k, &j) in group.iter().enumerate() {
            sub.column_mut(k).assign(&lig_data.column(j));
        }
        let weighted = match grid_factor {
            Some(gf) if gf.is_finite() && gf > 0.0 => {
                calculate_weighted_ligands_grid(xy, &sub, radius, 1.0, gf)
            }
            _ => calculate_weighted_ligands(xy, &sub, radius, 1.0),
        };
        for (k, &j) in group.iter().enumerate() {
            result.column_mut(j).assign(&weighted.column(k));
        }
    }

    GeneMatrix::new(result, lig_names)
}

impl PerturbRuntime {
    pub fn from_run_toml(run_toml: &Path) -> anyhow::Result<Self> {
        let run_toml_path = run_toml.to_path_buf();
        let run_dir = run_toml
            .parent()
            .ok_or_else(|| anyhow::anyhow!("run TOML has no parent directory"))?
            .to_path_buf();
        let cfg = SpaceshipConfig::from_file(&run_toml_path)?;

        let adata_path = expand_user_path(cfg.resolve_adata_path().as_str());
        if adata_path.is_empty() {
            anyhow::bail!("data.adata_path is empty in run TOML");
        }
        let adata = AnnData::<H5>::open(H5::open(adata_path.as_str())?)?;
        let gene_names = adata.var_names().into_vec();
        let obs_names = adata.obs_names().into_vec();
        let clusters = read_clusters_as_usize(&adata, cfg.data.cluster_annot.as_str())?;
        let xy = load_spatial_coords_f64(&adata)?;
        let slice = [SelectInfoElem::full(), SelectInfoElem::full()];
        let gene_mtx = read_expression_matrix_dense_f64(&adata, cfg.data.layer.as_str(), &slice)?;

        let gene2index: HashMap<String, usize> = gene_names
            .iter()
            .enumerate()
            .map(|(i, g)| (g.clone(), i))
            .collect();

        let betadata_dir = run_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("run directory is not valid UTF-8"))?;
        let bb = Betabase::from_directory(betadata_dir, &obs_names, &clusters, Some(&gene2index))
            .with_context(|| {
                format!(
                    "Failed to load *_betadata.feather from run dir {}",
                    run_dir.display()
                )
            })?;

        let mut lr_radii: HashMap<String, f64> = HashMap::new();
        for lig in bb.ligands_set.iter().chain(bb.tfl_ligands_set.iter()) {
            lr_radii.insert(lig.clone(), cfg.spatial.radius);
        }

        let min_expression = 1e-9;
        let grid = cfg.perturbation.ligand_grid_factor;
        let lr_ligands: Vec<String> = bb.ligands_set.iter().cloned().collect();
        let tfl_ligands: Vec<String> = bb.tfl_ligands_set.iter().cloned().collect();
        let rw_ligands_init = compute_initial_wl(
            &gene_mtx,
            &gene_names,
            &lr_ligands,
            &xy,
            &lr_radii,
            min_expression,
            grid,
        );
        let rw_tfligands_init = compute_initial_wl(
            &gene_mtx,
            &gene_names,
            &tfl_ligands,
            &xy,
            &lr_radii,
            min_expression,
            grid,
        );

        let perturb_cfg = PerturbConfig {
            n_propagation: cfg.perturbation.n_propagation,
            scale_factor: 1.0,
            beta_scale_factor: cfg.perturbation.beta_scale_factor,
            beta_cap: cfg.perturbation.beta_cap,
            min_expression,
            ligand_grid_factor: cfg.perturbation.ligand_grid_factor,
        };

        Ok(Self {
            run_toml_path,
            run_dir,
            cfg,
            gene_mtx,
            gene_names,
            obs_names,
            cell_types: clusters,
            bb,
            xy,
            rw_ligands_init,
            rw_tfligands_init,
            lr_radii,
            perturb_cfg,
        })
    }
}

#[derive(Serialize)]
struct PerturbRunSummary {
    run_toml_path: String,
    selected_genes: Vec<String>,
    target_value: f64,
    output_dir: String,
    n_propagation: usize,
    beta_scale_factor: f64,
    beta_cap: Option<f64>,
    ligand_grid_factor: Option<f64>,
    outputs: Vec<String>,
    selected_cell_types_per_gene: HashMap<String, Option<Vec<usize>>>,
}

type GeneCellTypeScopes = HashMap<String, Option<HashSet<usize>>>;

pub fn execute_marked_perturbations(
    runtime: &PerturbRuntime,
    selected_genes: &[String],
    selected_cell_types_per_gene: &GeneCellTypeScopes,
    value: f64,
) -> anyhow::Result<PathBuf> {
    if selected_genes.is_empty() {
        anyhow::bail!("No selected genes to perturb.");
    }
    let selected: Vec<String> = selected_genes.to_vec();
    for g in &selected {
        if !runtime.gene_names.iter().any(|x| x == g) {
            anyhow::bail!("Gene '{}' is not present in AnnData var_names.", g);
        }
    }

    let out_dir = request_output_dir(
        runtime.run_dir.as_path(),
        &selected,
        value,
        runtime.perturb_cfg.n_propagation,
    );
    std::fs::create_dir_all(&out_dir)?;

    let outputs = selected
        .par_iter()
        .map(|gene| -> anyhow::Result<PathBuf> {
            let selected_cells: Option<Vec<usize>> = selected_cell_types_per_gene
                .get(gene)
                .and_then(|scope| scope.as_ref())
                .map(|cell_types| {
                    runtime
                        .cell_types
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, ct)| {
                            if cell_types.contains(ct) {
                                Some(idx)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                });
            let targets = vec![PerturbTarget {
                gene: gene.clone(),
                desired_expr: value,
                cell_indices: selected_cells,
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
            );
            let out_path = out_dir.join(format!("{}_perturb_expr.feather", gene));
            write_betadata_feather(
                out_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("non-utf8 output path"))?,
                "CellID",
                &runtime.obs_names,
                &runtime.gene_names,
                &result.simulated,
            )?;
            Ok(out_path)
        })
        .collect::<Vec<_>>();

    let mut output_paths = Vec::with_capacity(outputs.len());
    for path in outputs {
        output_paths.push(path?);
    }

    let summary = PerturbRunSummary {
        run_toml_path: runtime.run_toml_path.display().to_string(),
        selected_genes: selected.clone(),
        target_value: value,
        output_dir: out_dir.display().to_string(),
        n_propagation: runtime.perturb_cfg.n_propagation,
        beta_scale_factor: runtime.perturb_cfg.beta_scale_factor,
        beta_cap: runtime.perturb_cfg.beta_cap,
        ligand_grid_factor: runtime.perturb_cfg.ligand_grid_factor,
        outputs: output_paths
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
            .collect(),
        selected_cell_types_per_gene: selected
            .iter()
            .map(|g| {
                let scope = selected_cell_types_per_gene
                    .get(g)
                    .and_then(|s| s.as_ref())
                    .map(|set| {
                        let mut v = set.iter().copied().collect::<Vec<_>>();
                        v.sort_unstable();
                        v
                    });
                (g.clone(), scope)
            })
            .collect(),
    };

    let summary_path = out_dir.join("perturbation_run_summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    Ok(out_dir)
}

fn prompt_line(prompt: &str) -> anyhow::Result<String> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    Ok(s.trim().to_string())
}

pub fn interactive_run_toml_prompt() -> anyhow::Result<PathBuf> {
    loop {
        let raw = prompt_line("Path to spacetravlr_run_repro.toml: ")?;
        let expanded = expand_user_path(raw.as_str());
        let p = PathBuf::from(expanded);
        if p.is_file() {
            return Ok(p);
        }
        eprintln!("Not found: {}", p.display());
    }
}

pub fn run_interactive(runtime: PerturbRuntime) -> anyhow::Result<()> {
    let mut selected: HashSet<String> = HashSet::new();
    let mut selected_cell_types_per_gene: GeneCellTypeScopes = HashMap::new();
    let mut all_cell_types = runtime.cell_types.iter().copied().collect::<Vec<_>>();
    all_cell_types.sort_unstable();
    all_cell_types.dedup();
    println!("Perturbation mode loaded from {}", runtime.run_toml_path.display());
    println!("Run directory: {}", runtime.run_dir.display());
    println!("Loaded {} genes and {} cells.", runtime.gene_names.len(), runtime.obs_names.len());
    println!(
        "Commands: list [N], search <query>, mark <gene> [all|ct1,ct2], scope <gene> <all|ct1,ct2>, unmark <gene>, show, run <value>, quit"
    );
    println!("Available cell_type_int values: {:?}", all_cell_types);

    fn parse_cell_type_scope(raw: &str) -> anyhow::Result<Option<HashSet<usize>>> {
        let cleaned = raw.trim();
        if cleaned.is_empty() || cleaned.eq_ignore_ascii_case("all") {
            return Ok(None);
        }
        let mut out = HashSet::new();
        for part in cleaned.split(',') {
            let v = part
                .trim()
                .parse::<usize>()
                .with_context(|| format!("Invalid cell_type '{}'", part.trim()))?;
            out.insert(v);
        }
        Ok(Some(out))
    }

    loop {
        let cmd = prompt_line("perturb> ")?;
        let mut parts = cmd.split_whitespace();
        let Some(head) = parts.next() else {
            continue;
        };

        match head {
            "list" => {
                let n = parts
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(40);
                for g in runtime.gene_names.iter().take(n) {
                    let mark = if selected.contains(g) { "[x]" } else { "[ ]" };
                    println!("{mark} {g}");
                }
                if runtime.gene_names.len() > n {
                    println!("... {} more", runtime.gene_names.len() - n);
                }
            }
            "search" => {
                let q = parts.collect::<Vec<_>>().join(" ");
                if q.is_empty() {
                    continue;
                }
                let q_lower = q.to_ascii_lowercase();
                let mut shown = 0usize;
                for g in &runtime.gene_names {
                    if g.to_ascii_lowercase().contains(&q_lower) {
                        let mark = if selected.contains(g) { "[x]" } else { "[ ]" };
                        println!("{mark} {g}");
                        shown += 1;
                        if shown >= 100 {
                            break;
                        }
                    }
                }
                if shown == 0 {
                    println!("No genes matched '{q}'.");
                }
            }
            "mark" => {
                let remaining = parts.collect::<Vec<_>>();
                if remaining.is_empty() {
                    println!("Usage: mark <gene> [all|ct1,ct2]");
                    continue;
                }
                let gene = remaining[0].to_string();
                if runtime.gene_names.iter().any(|g| g == &gene) {
                    selected.insert(gene.clone());
                    let scope = if remaining.len() > 1 {
                        parse_cell_type_scope(remaining[1])?
                    } else {
                        None
                    };
                    selected_cell_types_per_gene.insert(gene, scope);
                } else {
                    println!("Unknown gene.");
                }
            }
            "scope" => {
                let remaining = parts.collect::<Vec<_>>();
                if remaining.len() < 2 {
                    println!("Usage: scope <gene> <all|ct1,ct2>");
                    continue;
                }
                let gene = remaining[0].to_string();
                if !selected.contains(&gene) {
                    println!("Gene is not marked.");
                    continue;
                }
                let scope = parse_cell_type_scope(remaining[1])?;
                selected_cell_types_per_gene.insert(gene, scope);
            }
            "unmark" => {
                let gene = parts.collect::<Vec<_>>().join(" ");
                selected.remove(gene.as_str());
                selected_cell_types_per_gene.remove(gene.as_str());
            }
            "show" => {
                if selected.is_empty() {
                    println!("No genes selected.");
                } else {
                    let mut v = selected.iter().cloned().collect::<Vec<_>>();
                    v.sort();
                    println!("Selected {} genes:", v.len());
                    for g in v {
                        let scope = selected_cell_types_per_gene
                            .get(&g)
                            .and_then(|s| s.as_ref())
                            .map(|set| {
                                let mut vv = set.iter().copied().collect::<Vec<_>>();
                                vv.sort_unstable();
                                vv
                            });
                        match scope {
                            Some(vv) => println!("- {g} (cell_types={:?})", vv),
                            None => println!("- {g} (cell_types=all)"),
                        }
                    }
                }
            }
            "run" => {
                let value = parts
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("Usage: run <value>"))?
                    .parse::<f64>()
                    .with_context(|| "run value must be a floating number")?;
                let mut genes = selected.iter().cloned().collect::<Vec<_>>();
                genes.sort();
                let out = execute_marked_perturbations(
                    &runtime,
                    &genes,
                    &selected_cell_types_per_gene,
                    value,
                )?;
                println!("Finished. Outputs written under {}", out.display());
            }
            "quit" | "exit" => return Ok(()),
            _ => println!("Unknown command."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use polars::prelude::SerReader;

    #[test]
    fn output_dir_is_deterministic() {
        let run_dir = PathBuf::from("/tmp/example");
        let genes = vec!["GZMB".to_string(), "CD74".to_string()];
        let a = request_output_dir(&run_dir, &genes, 0.0, 4);
        let b = request_output_dir(&run_dir, &genes, 0.0, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn write_feather_shape_matches_matrix() {
        let temp = std::env::temp_dir().join(format!(
            "spacetravlr_perturb_test_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&temp);
        let out = temp.join("matrix.feather");
        let obs = vec!["c1".to_string(), "c2".to_string()];
        let genes = vec!["g1".to_string(), "g2".to_string(), "g3".to_string()];
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        write_betadata_feather(
            out.to_str().unwrap(),
            "CellID",
            &obs,
            &genes,
            &data,
        )
        .unwrap();
        let f = std::fs::File::open(&out).unwrap();
        let df = polars::prelude::IpcReader::new(f).finish().unwrap();
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 4);
    }
}
