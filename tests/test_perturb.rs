use ndarray::Array2;
use space_trav_lr_rust::betadata::{Betabase, GeneMatrix};
use space_trav_lr_rust::ligand::calculate_weighted_ligands;
use space_trav_lr_rust::perturb::{perturb, PerturbConfig};
use std::collections::{HashMap, HashSet};
use std::io::Write;

/// Build all inputs needed for perturb: betas, gene expression, coordinates,
/// received ligands, gene-to-index map.
fn build_perturb_inputs(
    betas_dir: &str,
    n_cells: usize,
    n_clusters: usize,
) -> (
    Betabase,
    Array2<f64>,
    Vec<String>,
    Array2<f64>,
    GeneMatrix,
    GeneMatrix,
    HashMap<String, f64>,
) {
    let obs_names: Vec<String> = (0..n_cells).map(|i| format!("cell_{}", i)).collect();
    let clusters: Vec<usize> = (0..n_cells).map(|i| i % n_clusters).collect();

    // First pass: load betas to discover gene names
    let bb_temp = Betabase::from_directory(betas_dir, &obs_names, &clusters, None).unwrap();

    // Collect all genes (trained + modulators)
    let mut all_genes_set: HashSet<String> = HashSet::new();
    for (gene_name, bf) in &bb_temp.data {
        all_genes_set.insert(gene_name.clone());
        all_genes_set.extend(bf.tfs.iter().cloned());
        all_genes_set.extend(bf.ligands.iter().cloned());
        all_genes_set.extend(bf.receptors.iter().cloned());
        all_genes_set.extend(bf.tfl_ligands.iter().cloned());
        all_genes_set.extend(bf.tfl_regulators.iter().cloned());
    }
    let mut gene_names: Vec<String> = all_genes_set.into_iter().collect();
    gene_names.sort();

    let gene2index: HashMap<String, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    // Reload with gene2index so modulator_gene_indices are set
    let bb = Betabase::from_directory(betas_dir, &obs_names, &clusters, Some(&gene2index)).unwrap();

    let n_genes = gene_names.len();

    // Deterministic gene expression: varies by gene and cell
    let gene_mtx = Array2::from_shape_fn((n_cells, n_genes), |(cell, gene)| {
        1.0 + 0.01 * gene as f64 + 0.001 * (cell % 7) as f64
    });

    // Grid coordinates
    let grid_w = (n_cells as f64).sqrt().ceil() as usize;
    let xy = Array2::from_shape_fn((n_cells, 2), |(i, dim)| {
        if dim == 0 {
            (i % grid_w) as f64 * 10.0
        } else {
            (i / grid_w) as f64 * 10.0
        }
    });

    // LR radii: 200 for all ligands
    let mut lr_radii: HashMap<String, f64> = HashMap::new();
    for l in bb.ligands_set.iter().chain(bb.tfl_ligands_set.iter()) {
        lr_radii.insert(l.clone(), 200.0);
    }

    // Compute initial received ligands
    let lr_ligands: Vec<String> = bb.ligands_set.iter().cloned().collect();
    let tfl_ligands: Vec<String> = bb.tfl_ligands_set.iter().cloned().collect();

    let rw_ligands = compute_initial_wl(&gene_mtx, &gene_names, &lr_ligands, &xy, &lr_radii, 1.0);
    let rw_tfligands =
        compute_initial_wl(&gene_mtx, &gene_names, &tfl_ligands, &xy, &lr_radii, 1.0);

    (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii)
}

fn compute_initial_wl(
    gene_mtx: &Array2<f64>,
    gene_names: &[String],
    ligand_names: &[String],
    xy: &Array2<f64>,
    lr_radii: &HashMap<String, f64>,
    scale_factor: f64,
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
                    if v > 1e-9 { v } else { 0.0 }
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

    // Group by radius
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
        let weighted = calculate_weighted_ligands(xy, &sub, radius, scale_factor);
        for (k, &j) in group.iter().enumerate() {
            result.column_mut(j).assign(&weighted.column(k));
        }
    }

    GeneMatrix::new(result, lig_names)
}

/// Save inputs to /tmp/perturb_compare/ so the Python script can load them.
fn save_inputs_for_python(
    gene_mtx: &Array2<f64>,
    gene_names: &[String],
    xy: &Array2<f64>,
    rw_ligands: &GeneMatrix,
    rw_tfligands: &GeneMatrix,
    lr_radii: &HashMap<String, f64>,
    target: &str,
    gene_expr: f64,
    n_propagation: usize,
    out_dir: &str,
) {
    std::fs::create_dir_all(out_dir).unwrap();

    // gene_names
    let mut f = std::fs::File::create(format!("{}/gene_names.csv", out_dir)).unwrap();
    for g in gene_names {
        writeln!(f, "{}", g).unwrap();
    }

    // gene_mtx
    let mut f = std::fs::File::create(format!("{}/gene_mtx.csv", out_dir)).unwrap();
    let (nr, nc) = (gene_mtx.nrows(), gene_mtx.ncols());
    for i in 0..nr {
        for j in 0..nc {
            if j > 0 { write!(f, ",").unwrap(); }
            write!(f, "{}", gene_mtx[[i, j]]).unwrap();
        }
        writeln!(f).unwrap();
    }

    // xy
    let mut f = std::fs::File::create(format!("{}/xy.csv", out_dir)).unwrap();
    for i in 0..xy.nrows() {
        writeln!(f, "{},{}", xy[[i, 0]], xy[[i, 1]]).unwrap();
    }

    // rw_ligands
    save_gene_matrix(rw_ligands, &format!("{}/rw_ligands.csv", out_dir));
    save_gene_matrix(rw_tfligands, &format!("{}/rw_tfligands.csv", out_dir));

    // lr_radii
    let mut f = std::fs::File::create(format!("{}/lr_radii.csv", out_dir)).unwrap();
    for (name, radius) in lr_radii {
        writeln!(f, "{},{}", name, radius).unwrap();
    }

    // config
    let mut f = std::fs::File::create(format!("{}/config.txt", out_dir)).unwrap();
    writeln!(f, "target={}", target).unwrap();
    writeln!(f, "gene_expr={}", gene_expr).unwrap();
    writeln!(f, "n_propagation={}", n_propagation).unwrap();
}

fn save_gene_matrix(gm: &GeneMatrix, path: &str) {
    let mut f = std::fs::File::create(path).unwrap();
    // Header
    for (j, name) in gm.col_names.iter().enumerate() {
        if j > 0 { write!(f, ",").unwrap(); }
        write!(f, "{}", name).unwrap();
    }
    writeln!(f).unwrap();
    // Data
    let (nr, nc) = (gm.data.nrows(), gm.data.ncols());
    for i in 0..nr {
        for j in 0..nc {
            if j > 0 { write!(f, ",").unwrap(); }
            write!(f, "{}", gm.data[[i, j]]).unwrap();
        }
        writeln!(f).unwrap();
    }
}

#[test]
fn test_perturb_from_tmp_betas() {
    let betas_dir = "/tmp/betas";
    if !std::path::Path::new(betas_dir).exists() {
        eprintln!("Skipping: /tmp/betas not found");
        return;
    }

    let n_cells = 200;
    let n_clusters = 13;
    let n_propagation = 3;
    let target_gene_expr = 0.0; // knockout

    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        build_perturb_inputs(betas_dir, n_cells, n_clusters);

    // Pick a target gene that IS a modulator of other trained genes (ensures propagation)
    let mut modulator_counts: HashMap<String, usize> = HashMap::new();
    for bf in bb.data.values() {
        for mg in &bf.modulator_genes {
            let plain = mg.strip_prefix("beta_").unwrap_or(mg).to_string();
            *modulator_counts.entry(plain).or_default() += 1;
        }
    }
    let target = modulator_counts
        .iter()
        .filter(|(g, _)| gene_names.contains(g))
        .max_by_key(|&(_, count)| *count)
        .map(|(g, _)| g.clone())
        .unwrap_or_else(|| bb.data.keys().next().unwrap().clone());

    eprintln!(
        "  {} is a modulator in {} gene models",
        target,
        modulator_counts.get(&target).unwrap_or(&0)
    );

    eprintln!(
        "\nPerturbing {} -> {} ({} cells, {} genes, {} trained, {} propagation steps)",
        target,
        target_gene_expr,
        n_cells,
        gene_names.len(),
        bb.data.len(),
        n_propagation
    );

    let out_dir = "/tmp/perturb_compare";
    save_inputs_for_python(
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &lr_radii,
        &target,
        target_gene_expr,
        n_propagation,
        out_dir,
    );

    let config = PerturbConfig {
        n_propagation,
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[(target.clone(), target_gene_expr)],
        &config,
        &lr_radii,
    );
    let elapsed = t0.elapsed();

    // Save result
    let mut f = std::fs::File::create(format!("{}/result_rust.csv", out_dir)).unwrap();
    for (j, name) in gene_names.iter().enumerate() {
        if j > 0 { write!(f, ",").unwrap(); }
        write!(f, "{}", name).unwrap();
    }
    writeln!(f).unwrap();
    for i in 0..n_cells {
        for j in 0..gene_names.len() {
            if j > 0 { write!(f, ",").unwrap(); }
            write!(f, "{:.15e}", result.simulated[[i, j]]).unwrap();
        }
        writeln!(f).unwrap();
    }

    // Summary
    let delta = &result.simulated - &gene_mtx;
    let nonzero = delta.iter().filter(|&&v| v.abs() > 1e-15).count();
    let max_delta = delta.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let mean_delta: f64 = delta.iter().map(|v| v.abs()).sum::<f64>() / delta.len() as f64;

    println!();
    println!("Perturb result:");
    println!("  target: {} -> {}", target, target_gene_expr);
    println!("  shape: {} x {}", result.simulated.nrows(), result.simulated.ncols());
    println!("  nonzero delta elements: {} / {}", nonzero, delta.len());
    println!("  max |delta|: {:.6e}", max_delta);
    println!("  mean |delta|: {:.6e}", mean_delta);
    println!("  elapsed: {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  inputs saved to {}", out_dir);
    println!();
}

#[test]
fn bench_perturb() {
    use std::time::Instant;

    let betas_dir = "/tmp/betas";
    if !std::path::Path::new(betas_dir).exists() {
        eprintln!("Skipping: /tmp/betas not found");
        return;
    }

    let n_clusters = 13;
    let n_propagation = 3;
    let grid_factor = 0.5;

    println!();
    println!("  {:>5}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}",
        "cells", "exact(ms)", "grid(ms)", "speedup", "max_err", "mean_err");
    println!("  {}",  "-".repeat(70));

    for &n_cells in &[200, 500, 1_000, 2_000, 5_000, 10_000] {
        let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
            build_perturb_inputs(betas_dir, n_cells, n_clusters);

        let mut modulator_counts: HashMap<String, usize> = HashMap::new();
        for bf in bb.data.values() {
            for mg in &bf.modulator_genes {
                let plain = mg.strip_prefix("beta_").unwrap_or(mg).to_string();
                *modulator_counts.entry(plain).or_default() += 1;
            }
        }
        let target = modulator_counts
            .iter()
            .filter(|(g, _)| gene_names.contains(g))
            .max_by_key(|&(_, count)| *count)
            .map(|(g, _)| g.clone())
            .unwrap_or_else(|| bb.data.keys().next().unwrap().clone());

        let config_exact = PerturbConfig {
            n_propagation,
            scale_factor: 1.0,
            beta_scale_factor: 1.0,
            beta_cap: None,
            min_expression: 1e-9,
            ligand_grid_factor: None,
        };

        let config_grid = PerturbConfig {
            n_propagation,
            scale_factor: 1.0,
            beta_scale_factor: 1.0,
            beta_cap: None,
            min_expression: 1e-9,
            ligand_grid_factor: Some(grid_factor),
        };

        let targets = vec![(target.clone(), 0.0)];

        // Warmup both
        let _ = perturb(&bb, &gene_mtx, &gene_names, &xy, &rw_ligands, &rw_tfligands, &targets, &config_exact, &lr_radii);
        let _ = perturb(&bb, &gene_mtx, &gene_names, &xy, &rw_ligands, &rw_tfligands, &targets, &config_grid, &lr_radii);

        // Exact
        let t0 = Instant::now();
        let result_exact = perturb(&bb, &gene_mtx, &gene_names, &xy, &rw_ligands, &rw_tfligands, &targets, &config_exact, &lr_radii);
        let exact_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Grid
        let t0 = Instant::now();
        let result_grid = perturb(&bb, &gene_mtx, &gene_names, &xy, &rw_ligands, &rw_tfligands, &targets, &config_grid, &lr_radii);
        let grid_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Accuracy: compare simulated outputs
        let diff = &result_exact.simulated - &result_grid.simulated;
        let max_err = diff.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let mean_err = diff.iter().map(|v| v.abs()).sum::<f64>() / diff.len() as f64;

        let speedup = exact_ms / grid_ms;

        println!(
            "  {:>5}  {:>10.1}  {:>10.1}  {:>7.1}x  {:>10.2e}  {:>10.2e}",
            n_cells, exact_ms, grid_ms, speedup, max_err, mean_err,
        );
    }
    println!();
}
