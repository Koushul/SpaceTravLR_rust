use ndarray::{Array2, Zip, array};
use rayon::prelude::*;
use space_trav_lr_rust::betadata::{BetaFrame, Betabase, GeneMatrix};
use space_trav_lr_rust::ligand::calculate_weighted_ligands;
use space_trav_lr_rust::perturb::{PerturbConfig, PerturbTarget, perturb, perturb_with_targets};
use space_trav_lr_rust::perturb_mode::compute_initial_weighted_ligands;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;

// ── Self-contained perturb tests (no /tmp/betas needed) ─────────────────────

fn make_synthetic_betabase(n_cells: usize) -> (Betabase, Vec<String>, HashMap<String, usize>) {
    // Gene universe: A (TF), B (ligand), C (receptor), D (target1), E (target2), Z (no edges —
    // not a target row in Betabase, not a ligand/TF in the graph; negative control for KO tests).
    let gene_names: Vec<String> = vec!["A", "B", "C", "D", "E", "Z"]
        .into_iter()
        .map(String::from)
        .collect();
    let gene2index: HashMap<String, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    let rows = vec!["0".to_string()]; // single cluster

    // D = f(A_tf, B_lr_C) → A is TF, B$C is LR pair
    let mut bf_d = BetaFrame::from_parts(
        "D".into(),
        rows.clone(),
        array![0.0],
        array![[0.5]],
        vec!["A".into()], // TF: A with beta=0.5
        array![[0.3]],
        vec!["B".into()],
        vec!["C".into()], // LR: B$C with beta=0.3
        ndarray::Array2::zeros((1, 0)),
        vec![],
        vec![], // no TFL
    );

    // E = f(A_tf, D_tf) → both A and D are TFs for E (creates cascade)
    let mut bf_e = BetaFrame::from_parts(
        "E".into(),
        rows.clone(),
        array![0.0],
        array![[0.4, 0.6]],
        vec!["A".into(), "D".into()],
        ndarray::Array2::zeros((1, 0)),
        vec![],
        vec![],
        ndarray::Array2::zeros((1, 0)),
        vec![],
        vec![],
    );

    // Expand to cells (all cells = cluster 0)
    let obs: Vec<String> = (0..n_cells).map(|i| format!("cell_{}", i)).collect();
    let clusters: Vec<usize> = vec![0; n_cells];
    let cluster_keys: Vec<String> = clusters.iter().map(|c| c.to_string()).collect();
    let mapping = Arc::new(BetaFrame::compute_cell_mapping(&rows, &obs, &cluster_keys));
    let obs_arc = Arc::new(obs);

    bf_d.expand_to_cells(obs_arc.clone(), mapping.clone());
    bf_e.expand_to_cells(obs_arc, mapping);

    // Resolve modulator indices
    bf_d.modulator_gene_indices = Some(
        bf_d.modulator_genes
            .iter()
            .map(|g| {
                let plain = g.strip_prefix("beta_").unwrap_or(g);
                *gene2index.get(plain).unwrap()
            })
            .collect(),
    );
    bf_e.modulator_gene_indices = Some(
        bf_e.modulator_genes
            .iter()
            .map(|g| {
                let plain = g.strip_prefix("beta_").unwrap_or(g);
                *gene2index.get(plain).unwrap()
            })
            .collect(),
    );

    let mut data = HashMap::new();
    data.insert("D".to_string(), bf_d);
    data.insert("E".to_string(), bf_e);

    let bb = Betabase {
        data,
        ligands_set: ["B".to_string()].into_iter().collect(),
        receptors_set: ["C".to_string()].into_iter().collect(),
        tfl_ligands_set: HashSet::new(),
        tfs_set: ["A".to_string(), "D".to_string()].into_iter().collect(),
    };

    (bb, gene_names, gene2index)
}

fn make_synthetic_inputs(
    n_cells: usize,
) -> (
    Betabase,
    Array2<f64>,
    Vec<String>,
    Array2<f64>,
    GeneMatrix,
    GeneMatrix,
    HashMap<String, f64>,
) {
    let (bb, gene_names, _) = make_synthetic_betabase(n_cells);

    // Gene expression: all genes at 1.0
    let gene_mtx = Array2::from_elem((n_cells, gene_names.len()), 1.0);

    // Grid coordinates
    let grid_w = (n_cells as f64).sqrt().ceil() as usize;
    let xy = Array2::from_shape_fn((n_cells, 2), |(i, d)| {
        if d == 0 {
            (i % grid_w) as f64 * 10.0
        } else {
            (i / grid_w) as f64 * 10.0
        }
    });

    let mut lr_radii = HashMap::new();
    lr_radii.insert("B".to_string(), 50.0);

    // Compute initial received ligands (B is the only ligand)
    let lig_vals = gene_mtx.column(1).to_owned().insert_axis(ndarray::Axis(1));
    let rw_data = calculate_weighted_ligands(&xy, &lig_vals, 50.0, 1.0);
    let rw_ligands = GeneMatrix::new(rw_data.mapv(|v| v as f32), vec!["B".to_string()]);
    let rw_tfligands = GeneMatrix::new(Array2::<f32>::zeros((n_cells, 0)), vec![]);

    (
        bb,
        gene_mtx,
        gene_names,
        xy,
        rw_ligands,
        rw_tfligands,
        lr_radii,
    )
}

#[test]
fn test_perturb_knockout_propagates() {
    let n_cells = 25;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    // Knock out gene A (TF for both D and E)
    let config = PerturbConfig {
        n_propagation: 2,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("A".to_string(), 0.0)],
        &config,
        &lr_radii,
    );

    // A should be 0
    for i in 0..n_cells {
        assert_eq!(result.simulated[[i, 0]], 0.0, "A should be knocked out");
    }

    // D should be affected (A is a TF for D)
    let delta_d: f64 = (0..n_cells)
        .map(|i| (result.simulated[[i, 3]] - gene_mtx[[i, 3]]).abs())
        .sum::<f64>();
    assert!(delta_d > 0.0, "D should change when A is knocked out");

    // E should be affected (A is a TF for E, and D cascades to E)
    let delta_e: f64 = (0..n_cells)
        .map(|i| (result.simulated[[i, 4]] - gene_mtx[[i, 4]]).abs())
        .sum::<f64>();
    assert!(delta_e > 0.0, "E should change via cascade from A → D → E");
}

#[test]
fn test_perturb_knockout_negative_control_unwired_gene_unchanged() {
    let n_cells = 25;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);
    let z_idx = gene_names
        .iter()
        .position(|g| g == "Z")
        .expect("Z in gene list");

    let config = PerturbConfig {
        n_propagation: 4,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("A".to_string(), 0.0)],
        &config,
        &lr_radii,
    );

    let max_abs_z = (0..n_cells)
        .map(|i| result.delta[[i, z_idx]].abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs_z < 1e-9,
        "Z has no trained target row and is not a ligand/TF in the graph; KO(A) must not move Z (max|Δ|={:.3e})",
        max_abs_z
    );

    let delta_d: f64 = (0..n_cells)
        .map(|i| result.delta[[i, 3]].abs())
        .sum::<f64>();
    let delta_e: f64 = (0..n_cells)
        .map(|i| result.delta[[i, 4]].abs())
        .sum::<f64>();
    assert!(
        delta_d > 0.05 * n_cells as f64,
        "D should move materially when A is KO (connected TF); got sum|Δ|={}",
        delta_d
    );
    assert!(
        delta_e > 0.05 * n_cells as f64,
        "E should move materially (A→D→E); got sum|Δ|={}",
        delta_e
    );
}

#[test]
fn test_perturb_knockout_isolated_gene_only_self_changes() {
    let n_cells = 20;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);
    let z_idx = gene_names
        .iter()
        .position(|g| g == "Z")
        .expect("Z in gene list");
    let n_genes = gene_names.len();

    let config = PerturbConfig {
        n_propagation: 4,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("Z".to_string(), 0.0)],
        &config,
        &lr_radii,
    );

    for g in 0..n_genes {
        if g == z_idx {
            continue;
        }
        let mx = (0..n_cells)
            .map(|i| result.delta[[i, g]].abs())
            .fold(0.0f64, f64::max);
        assert!(
            mx < 1e-9,
            "KO(Z): gene {} must be unaffected (no model uses Z); max|Δ|={:.3e}",
            gene_names[g],
            mx
        );
    }
    for i in 0..n_cells {
        assert!(
            (result.simulated[[i, z_idx]] - 0.0).abs() < 1e-9,
            "Z should be knocked out in all cells"
        );
    }
}

#[test]
fn test_perturb_no_change_when_target_at_original() {
    let n_cells = 16;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    // "Perturb" A to its original value (1.0) — should produce no change
    let config = PerturbConfig {
        n_propagation: 2,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("A".to_string(), 1.0)],
        &config,
        &lr_radii,
    );

    let max_delta = result.delta.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        max_delta < 1e-10,
        "no perturbation should produce no change, got max_delta={:.4e}",
        max_delta
    );
}

#[test]
fn test_perturb_result_clipped_nonnegative() {
    let n_cells = 25;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    let config = PerturbConfig {
        n_propagation: 3,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("A".to_string(), 0.0)],
        &config,
        &lr_radii,
    );

    for &v in result.simulated.iter() {
        assert!(
            v >= 0.0,
            "simulated expression should be non-negative, got {}",
            v
        );
    }
}

#[test]
fn test_perturb_grid_vs_exact_consistency() {
    let n_cells = 100;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    let targets = vec![("A".to_string(), 0.0)];

    let config_exact = PerturbConfig {
        n_propagation: 2,
        ligand_grid_factor: None,
        ..Default::default()
    };
    let config_grid = PerturbConfig {
        n_propagation: 2,
        ligand_grid_factor: Some(0.3),
        ..Default::default()
    };

    let exact = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &targets,
        &config_exact,
        &lr_radii,
    );
    let grid = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &targets,
        &config_grid,
        &lr_radii,
    );

    let max_diff = exact
        .simulated
        .iter()
        .zip(grid.simulated.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    assert!(
        max_diff < 0.5,
        "grid approx should be close to exact, got max_diff={:.4e}",
        max_diff
    );
}

#[test]
fn test_perturb_overexpression() {
    let n_cells = 25;
    let (bb, _, gene_names, xy, _, _, lr_radii) = make_synthetic_inputs(n_cells);

    // Varying expression so max_per_gene > baseline, giving room for increase
    let gene_mtx = Array2::from_shape_fn((n_cells, gene_names.len()), |(cell, gene)| {
        0.5 + 0.1 * (cell % 5) as f64 + 0.05 * gene as f64
    });

    let lig_vals = gene_mtx.column(1).to_owned().insert_axis(ndarray::Axis(1));
    let rw_data = calculate_weighted_ligands(&xy, &lig_vals, 50.0, 1.0);
    let rw_ligands = GeneMatrix::new(rw_data.mapv(|v| v as f32), vec!["B".to_string()]);
    let rw_tfligands = GeneMatrix::new(Array2::<f32>::zeros((n_cells, 0)), vec![]);

    // Overexpress A to 5.0
    let config = PerturbConfig {
        n_propagation: 2,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("A".to_string(), 5.0)],
        &config,
        &lr_radii,
    );

    for i in 0..n_cells {
        assert!((result.simulated[[i, 0]] - 5.0).abs() < 1e-10);
    }

    let delta_d: f64 = (0..n_cells)
        .map(|i| (result.simulated[[i, 3]] - gene_mtx[[i, 3]]).abs())
        .sum::<f64>();
    assert!(delta_d > 0.0, "D should change with A overexpression");
}

#[test]
fn test_perturb_shape_preserved() {
    let n_cells = 16;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    let config = PerturbConfig {
        n_propagation: 1,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("A".to_string(), 0.0)],
        &config,
        &lr_radii,
    );

    assert_eq!(result.simulated.shape(), gene_mtx.shape());
    assert_eq!(result.delta.shape(), gene_mtx.shape());
}

#[test]
fn test_perturb_with_target_cell_subset() {
    let n_cells = 12;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    let config = PerturbConfig {
        n_propagation: 1,
        ..Default::default()
    };
    let target_cells = vec![0usize, 1usize, 2usize];
    let mut no_timings: Option<space_trav_lr_rust::perturb::PerturbTimings> = None;
    let result = perturb_with_targets(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[PerturbTarget {
            gene: "A".to_string(),
            desired_expr: 0.0,
            cell_indices: Some(target_cells.clone()),
        }],
        &config,
        &lr_radii,
        None,
        None,
        None,
        None,
        &mut no_timings,
    )
    .unwrap();

    for cell in target_cells {
        assert_eq!(result.simulated[[cell, 0]], 0.0);
    }
    for cell in 3..n_cells {
        assert!(result.simulated[[cell, 0]] > 0.0);
    }
}

#[test]
fn test_synthetic_tf_lr_spatial_propagation_known_effects() {
    let gene_names: Vec<String> = vec!["TF1", "LIG", "REC", "TARGET"]
        .into_iter()
        .map(String::from)
        .collect();
    let gene2index: HashMap<String, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    let row_labels = vec!["0".to_string()];
    let mut bf_target = BetaFrame::from_parts(
        "TARGET".to_string(),
        row_labels.clone(),
        array![0.0],
        array![[0.1]], // TF1 -> TARGET
        vec!["TF1".to_string()],
        array![[0.05]], // (LIG$REC) -> TARGET
        vec!["LIG".to_string()],
        vec!["REC".to_string()],
        ndarray::Array2::zeros((1, 0)),
        vec![],
        vec![],
    );

    let obs_names = vec!["cell_0".to_string(), "cell_1".to_string()];
    let cluster_keys = vec!["0".to_string(), "0".to_string()];
    let mapping = Arc::new(BetaFrame::compute_cell_mapping(
        &row_labels,
        &obs_names,
        &cluster_keys,
    ));
    bf_target.expand_to_cells(Arc::new(obs_names.clone()), mapping);
    bf_target.modulator_gene_indices = Some(
        bf_target
            .modulator_genes
            .iter()
            .map(|g| {
                let plain = g.strip_prefix("beta_").unwrap_or(g);
                *gene2index.get(plain).unwrap()
            })
            .collect(),
    );

    let mut bb_data = HashMap::new();
    bb_data.insert("TARGET".to_string(), bf_target);
    let bb = Betabase {
        data: bb_data,
        ligands_set: ["LIG".to_string()].into_iter().collect(),
        receptors_set: ["REC".to_string()].into_iter().collect(),
        tfl_ligands_set: HashSet::new(),
        tfs_set: ["TF1".to_string()].into_iter().collect(),
    };

    // cell_0 has strong ligand and TF; cell_1 receives ligand through space
    let gene_mtx = array![
        [1.0, 10.0, 1.0, 1.0], // cell_0
        [1.0, 0.0, 1.0, 1.0],  // cell_1
    ];
    let xy = array![[0.0, 0.0], [1.0, 0.0],];

    let lig_vals = gene_mtx.column(1).to_owned().insert_axis(ndarray::Axis(1));
    let rw_data = calculate_weighted_ligands(&xy, &lig_vals, 1.0, 1.0);
    let rw_ligands = GeneMatrix::new(rw_data.mapv(|v| v as f32), vec!["LIG".to_string()]);
    let rw_tfligands = GeneMatrix::new(Array2::<f32>::zeros((2, 0)), vec![]);

    let mut lr_radii = HashMap::new();
    lr_radii.insert("LIG".to_string(), 1.0);

    let config = PerturbConfig {
        n_propagation: 1,
        ..Default::default()
    };
    let mut no_timings: Option<space_trav_lr_rust::perturb::PerturbTimings> = None;
    let result = perturb_with_targets(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[
            PerturbTarget {
                gene: "TF1".to_string(),
                desired_expr: 0.0,
                cell_indices: Some(vec![0]),
            },
            PerturbTarget {
                gene: "LIG".to_string(),
                desired_expr: 0.0,
                cell_indices: Some(vec![0]),
            },
        ],
        &config,
        &lr_radii,
        None,
        None,
        None,
        None,
        &mut no_timings,
    )
    .unwrap();

    let target_idx = *gene2index.get("TARGET").unwrap();
    let tf_idx = *gene2index.get("TF1").unwrap();
    let lig_idx = *gene2index.get("LIG").unwrap();

    let target_before_c0 = gene_mtx[[0, target_idx]];
    let target_before_c1 = gene_mtx[[1, target_idx]];
    let target_after_c0 = result.simulated[[0, target_idx]];
    let target_after_c1 = result.simulated[[1, target_idx]];
    let delta_target_c0 = target_after_c0 - target_before_c0;
    let delta_target_c1 = target_after_c1 - target_before_c1;

    println!("Synthetic TF+LR perturbation summary:");
    println!(
        "  cell_0: TF1 {:.3} -> {:.3}, LIG {:.3} -> {:.3}, TARGET {:.6} -> {:.6} (delta {:.6})",
        gene_mtx[[0, tf_idx]],
        result.simulated[[0, tf_idx]],
        gene_mtx[[0, lig_idx]],
        result.simulated[[0, lig_idx]],
        target_before_c0,
        target_after_c0,
        delta_target_c0
    );
    println!(
        "  cell_1: TF1 {:.3} -> {:.3}, LIG {:.3} -> {:.3}, TARGET {:.6} -> {:.6} (delta {:.6})",
        gene_mtx[[1, tf_idx]],
        result.simulated[[1, tf_idx]],
        gene_mtx[[1, lig_idx]],
        result.simulated[[1, lig_idx]],
        target_before_c1,
        target_after_c1,
        delta_target_c1
    );

    // Direct perturbations should apply only to cell_0.
    assert_eq!(result.simulated[[0, tf_idx]], 0.0);
    assert_eq!(result.simulated[[0, lig_idx]], 0.0);
    assert!(result.simulated[[1, tf_idx]] > 0.0);
    assert!(result.simulated[[1, lig_idx]] >= 0.0);

    // TARGET should decrease in both cells:
    // - cell_0: direct TF + local LR effects
    // - cell_1: spatially propagated LR effect (no direct TF perturbation)
    assert!(delta_target_c0 < 0.0, "cell_0 TARGET should decrease");
    assert!(
        delta_target_c1 < 0.0,
        "cell_1 TARGET should decrease via spatial LR propagation"
    );
    assert!(
        delta_target_c0 < delta_target_c1,
        "cell_0 decrease should be stronger than cell_1 (direct TF+LR vs propagated LR only)"
    );

    // Magnitude sanity checks for this synthetic setup.
    assert!((delta_target_c0 + 0.41).abs() < 0.15);
    assert!((delta_target_c1 + 0.19).abs() < 0.15);
}

// ── UMAP transition vector shift tests ──────────────────────────────────────
//
// These verify that perturbation deltas fed into the transition UMAP pipeline
// produce vectors with correct magnitude ordering and direction.

#[test]
fn test_transition_ko_vs_oe_opposite_direction() {
    use space_trav_lr_rust::transition_umap::{TransitionUmapParams, compute_umap_transition_grid};

    let n_cells = 25;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    let umap: Vec<[f64; 2]> = (0..n_cells)
        .map(|i| [(i % 5) as f64, (i / 5) as f64])
        .collect();

    let params = TransitionUmapParams {
        n_neighbors: 10,
        temperature: 0.05,
        remove_null: false,
        unit_directions: false,
        grid_scale: 1.0,
        vector_scale: 1.0,
        delta_rescale: 1.0,
        magnitude_threshold: 0.0,
        use_full_graph: true,
        full_graph_max_cells: 100,
    };

    let config = PerturbConfig {
        n_propagation: 2,
        ..Default::default()
    };

    let ko_result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("A".to_string(), 0.0)],
        &config,
        &lr_radii,
    );
    let ko_grid = compute_umap_transition_grid(&gene_mtx, &ko_result.delta, &umap, &params);

    // Overexpress A to 5.0 in a setup with room for increase
    let gene_mtx_varied = Array2::from_shape_fn((n_cells, gene_names.len()), |(cell, gene)| {
        0.5 + 0.1 * (cell % 5) as f64 + 0.05 * gene as f64
    });
    let lig_vals = gene_mtx_varied
        .column(1)
        .to_owned()
        .insert_axis(ndarray::Axis(1));
    let rw_oe = space_trav_lr_rust::betadata::GeneMatrix::new(
        space_trav_lr_rust::ligand::calculate_weighted_ligands(&xy, &lig_vals, 50.0, 1.0)
            .mapv(|v| v as f32),
        vec!["B".to_string()],
    );
    let oe_result = perturb(
        &bb,
        &gene_mtx_varied,
        &gene_names,
        &xy,
        &rw_oe,
        &rw_tfligands,
        &[("A".to_string(), 5.0)],
        &config,
        &lr_radii,
    );
    let oe_grid = compute_umap_transition_grid(&gene_mtx_varied, &oe_result.delta, &umap, &params);

    // Compute mean vector direction for KO and OE
    let mean_ko: [f64; 2] = {
        let mut sx = 0.0;
        let mut sy = 0.0;
        for v in &ko_grid.cell_vectors {
            sx += v[0];
            sy += v[1];
        }
        [sx / n_cells as f64, sy / n_cells as f64]
    };
    let mean_oe: [f64; 2] = {
        let mut sx = 0.0;
        let mut sy = 0.0;
        for v in &oe_grid.cell_vectors {
            sx += v[0];
            sy += v[1];
        }
        [sx / n_cells as f64, sy / n_cells as f64]
    };

    // KO and OE should produce vectors in roughly opposite directions.
    // Dot product of mean vectors should be negative (or close to zero in degenerate cases).
    let dot = mean_ko[0] * mean_oe[0] + mean_ko[1] * mean_oe[1];
    let mag_ko = (mean_ko[0].powi(2) + mean_ko[1].powi(2)).sqrt();
    let mag_oe = (mean_oe[0].powi(2) + mean_oe[1].powi(2)).sqrt();

    if mag_ko > 1e-9 && mag_oe > 1e-9 {
        let cos_angle = dot / (mag_ko * mag_oe);
        assert!(
            cos_angle < 0.5,
            "KO and OE should produce roughly opposite mean vectors; cos_angle={:.3} (expected < 0.5)\n  mean_ko={:?}\n  mean_oe={:?}",
            cos_angle,
            mean_ko,
            mean_oe,
        );
    }
}

#[test]
fn test_transition_magnitude_monotonic_with_perturbation_strength() {
    use space_trav_lr_rust::transition_umap::{TransitionUmapParams, compute_umap_transition_grid};

    let n_cells = 25;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    let umap: Vec<[f64; 2]> = (0..n_cells)
        .map(|i| [(i % 5) as f64, (i / 5) as f64])
        .collect();

    let params = TransitionUmapParams {
        n_neighbors: 10,
        temperature: 0.05,
        remove_null: false,
        unit_directions: false,
        grid_scale: 1.0,
        vector_scale: 1.0,
        delta_rescale: 1.0,
        magnitude_threshold: 0.0,
        use_full_graph: true,
        full_graph_max_cells: 100,
    };

    let perturb_levels = [0.0, 0.5, 0.9]; // baseline is 1.0 for all cells
    let mut magnitudes: Vec<f64> = Vec::new();

    for &level in &perturb_levels {
        let config = PerturbConfig {
            n_propagation: 2,
            ..Default::default()
        };
        let result = perturb(
            &bb,
            &gene_mtx,
            &gene_names,
            &xy,
            &rw_ligands,
            &rw_tfligands,
            &[("A".to_string(), level)],
            &config,
            &lr_radii,
        );
        let grid = compute_umap_transition_grid(&gene_mtx, &result.delta, &umap, &params);

        let total_mag: f64 = grid
            .cell_vectors
            .iter()
            .map(|v| (v[0].powi(2) + v[1].powi(2)).sqrt())
            .sum();
        magnitudes.push(total_mag);
    }

    // KO (level=0.0, delta=-1.0) should produce the strongest shift,
    // halving (level=0.5, delta=-0.5) should produce intermediate,
    // 10% reduction (level=0.9, delta=-0.1) the weakest.
    assert!(
        magnitudes[0] >= magnitudes[1],
        "stronger perturbation (0.0 vs 0.5) should produce >= magnitude: {:.4} vs {:.4}",
        magnitudes[0],
        magnitudes[1]
    );
    assert!(
        magnitudes[1] >= magnitudes[2],
        "stronger perturbation (0.5 vs 0.9) should produce >= magnitude: {:.4} vs {:.4}",
        magnitudes[1],
        magnitudes[2]
    );
}

#[test]
fn test_transition_no_perturbation_no_vectors() {
    use space_trav_lr_rust::transition_umap::{TransitionUmapParams, compute_umap_transition_grid};

    let n_cells = 16;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    let umap: Vec<[f64; 2]> = (0..n_cells)
        .map(|i| [(i % 4) as f64, (i / 4) as f64])
        .collect();

    let config = PerturbConfig {
        n_propagation: 2,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("A".to_string(), 1.0)], // no change (original = 1.0)
        &config,
        &lr_radii,
    );

    let params = TransitionUmapParams {
        n_neighbors: 8,
        temperature: 0.05,
        remove_null: true,
        unit_directions: false,
        grid_scale: 1.0,
        vector_scale: 1.0,
        delta_rescale: 1.0,
        magnitude_threshold: 0.0,
        use_full_graph: true,
        full_graph_max_cells: 100,
    };

    let grid = compute_umap_transition_grid(&gene_mtx, &result.delta, &umap, &params);

    let total_mag: f64 = grid
        .cell_vectors
        .iter()
        .map(|v| (v[0].powi(2) + v[1].powi(2)).sqrt())
        .sum();
    assert!(
        total_mag < 1e-6,
        "no perturbation should yield (near-)zero transition vectors, got total_mag={:.6e}",
        total_mag
    );
}

#[test]
fn test_transition_isolated_gene_no_field() {
    use space_trav_lr_rust::transition_umap::{TransitionUmapParams, compute_umap_transition_grid};

    let n_cells = 16;
    let (bb, gene_mtx, gene_names, xy, rw_ligands, rw_tfligands, lr_radii) =
        make_synthetic_inputs(n_cells);

    let umap: Vec<[f64; 2]> = (0..n_cells)
        .map(|i| [(i % 4) as f64, (i / 4) as f64])
        .collect();

    let config = PerturbConfig {
        n_propagation: 4,
        ..Default::default()
    };
    let result = perturb(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_ligands,
        &rw_tfligands,
        &[("Z".to_string(), 0.0)], // Z is unwired
        &config,
        &lr_radii,
    );

    let params = TransitionUmapParams {
        n_neighbors: 8,
        temperature: 0.05,
        remove_null: true,
        unit_directions: false,
        grid_scale: 1.0,
        vector_scale: 1.0,
        delta_rescale: 1.0,
        magnitude_threshold: 0.0,
        use_full_graph: true,
        full_graph_max_cells: 100,
    };

    let grid = compute_umap_transition_grid(&gene_mtx, &result.delta, &umap, &params);

    // Z is not connected to any gene model. KO(Z) produces deltas only in the Z
    // column. Since those deltas are uniform across cells (all cells had same
    // expression), transitions should be uniform and remove_null should zero them.
    let max_cell_mag: f64 = grid
        .cell_vectors
        .iter()
        .map(|v| (v[0].powi(2) + v[1].powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    assert!(
        max_cell_mag < 0.01,
        "KO of isolated gene Z should produce negligible transition field, got max_mag={:.6e}",
        max_cell_mag
    );
}

// ── Helpers for /tmp/betas-dependent tests ──────────────────────────────────

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
    let cluster_keys: Vec<String> = clusters.iter().map(|c| c.to_string()).collect();

    let mut bb =
        Betabase::from_directory(betas_dir, &obs_names, &cluster_keys, None, None).unwrap();

    let mut all_genes_set: HashSet<String> = HashSet::new();
    for (gene_name, bf) in &bb.data {
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

    bb.apply_modulator_gene_indices(&gene2index);

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

    (
        bb,
        gene_mtx,
        gene_names,
        xy,
        rw_ligands,
        rw_tfligands,
        lr_radii,
    )
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
        return GeneMatrix::new(Array2::<f32>::zeros((n_cells, 0)), Vec::new());
    }

    let n_lig = lig_names.len();
    let mut lig_data = Array2::<f64>::zeros((n_cells, n_lig));
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

    let mut result = Array2::<f32>::zeros((n_cells, n_lig));
    for (rbits, group) in &radius_groups {
        let radius = f64::from_bits(*rbits);
        let mut sub = Array2::<f64>::zeros((n_cells, group.len()));
        for (k, &j) in group.iter().enumerate() {
            sub.column_mut(k).assign(&lig_data.column(j));
        }
        let weighted = calculate_weighted_ligands(xy, &sub, radius, scale_factor);
        for (k, &j) in group.iter().enumerate() {
            let col = weighted.column(k);
            for i in 0..n_cells {
                result[[i, j]] = col[i] as f32;
            }
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
            if j > 0 {
                write!(f, ",").unwrap();
            }
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
        if j > 0 {
            write!(f, ",").unwrap();
        }
        write!(f, "{}", name).unwrap();
    }
    writeln!(f).unwrap();
    // Data
    let (nr, nc) = (gm.data.nrows(), gm.data.ncols());
    for i in 0..nr {
        for j in 0..nc {
            if j > 0 {
                write!(f, ",").unwrap();
            }
            write!(f, "{}", gm.data[[i, j]]).unwrap();
        }
        writeln!(f).unwrap();
    }
}

#[test]
#[ignore]
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
        if j > 0 {
            write!(f, ",").unwrap();
        }
        write!(f, "{}", name).unwrap();
    }
    writeln!(f).unwrap();
    for i in 0..n_cells {
        for j in 0..gene_names.len() {
            if j > 0 {
                write!(f, ",").unwrap();
            }
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
    println!(
        "  shape: {} x {}",
        result.simulated.nrows(),
        result.simulated.ncols()
    );
    println!("  nonzero delta elements: {} / {}", nonzero, delta.len());
    println!("  max |delta|: {:.6e}", max_delta);
    println!("  mean |delta|: {:.6e}", mean_delta);
    println!("  elapsed: {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  inputs saved to {}", out_dir);
    println!();
}

#[test]
#[ignore]
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
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}",
        "cells", "exact(ms)", "grid(ms)", "speedup", "max_err", "mean_err"
    );
    println!("  {}", "-".repeat(70));

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
            contact_distance: None,
        };

        let config_grid = PerturbConfig {
            n_propagation,
            scale_factor: 1.0,
            beta_scale_factor: 1.0,
            beta_cap: None,
            min_expression: 1e-9,
            ligand_grid_factor: Some(grid_factor),
            contact_distance: None,
        };

        let targets = vec![(target.clone(), 0.0)];

        // Warmup both
        let _ = perturb(
            &bb,
            &gene_mtx,
            &gene_names,
            &xy,
            &rw_ligands,
            &rw_tfligands,
            &targets,
            &config_exact,
            &lr_radii,
        );
        let _ = perturb(
            &bb,
            &gene_mtx,
            &gene_names,
            &xy,
            &rw_ligands,
            &rw_tfligands,
            &targets,
            &config_grid,
            &lr_radii,
        );

        // Exact
        let t0 = Instant::now();
        let result_exact = perturb(
            &bb,
            &gene_mtx,
            &gene_names,
            &xy,
            &rw_ligands,
            &rw_tfligands,
            &targets,
            &config_exact,
            &lr_radii,
        );
        let exact_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Grid
        let t0 = Instant::now();
        let result_grid = perturb(
            &bb,
            &gene_mtx,
            &gene_names,
            &xy,
            &rw_ligands,
            &rw_tfligands,
            &targets,
            &config_grid,
            &lr_radii,
        );
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

#[test]
fn test_pin_clip_parity_and_perf() {
    use std::time::Instant;

    let n_genes = 2000;
    let target_genes: Vec<usize> = vec![0, 42, 999];

    eprintln!();
    eprintln!("pin_clip parity & performance (n_genes={})", n_genes);
    eprintln!(
        "  {:>6}  {:>10}  {:>10}  {:>8}  {:>12}",
        "cells", "old(ms)", "new(ms)", "speedup", "max_diff"
    );
    eprintln!("  {}", "-".repeat(56));

    for &n_cells in &[500, 2_000, 10_000, 50_000] {
        let gene_mtx = Array2::from_shape_fn((n_cells, n_genes), |(c, g)| {
            0.5 + 0.1 * ((c * 7 + g * 13) % 17) as f64
        });
        let max_per_gene: Vec<f64> = (0..n_genes)
            .map(|j| gene_mtx.column(j).iter().cloned().fold(0.0f64, f64::max))
            .collect();

        let delta_input = {
            let mut d = Array2::zeros((n_cells, n_genes));
            for &gi in &target_genes {
                for c in 0..n_cells {
                    d[[c, gi]] = 0.0 - gene_mtx[[c, gi]];
                }
            }
            d
        };

        let delta_after_grn = Array2::from_shape_fn((n_cells, n_genes), |(c, g)| {
            -0.1 + 0.02 * ((c * 3 + g * 11) % 23) as f64
        });

        // ---- OLD implementation (full-matrix Zip + allocating clip) ----
        let mut delta_old = delta_after_grn.clone();
        let t0 = Instant::now();
        Zip::from(&mut delta_old)
            .and(&delta_input)
            .for_each(|d, &di| {
                if di != 0.0 {
                    *d = di;
                }
            });
        let gem = &gene_mtx + &delta_old;
        let gem_flat = gem.as_slice().unwrap();
        let gmtx_flat_old = gene_mtx.as_slice().unwrap();
        let delta_flat_old = delta_old.as_slice_memory_order_mut().unwrap();
        for cell in 0..n_cells {
            let base = cell * n_genes;
            for gene in 0..n_genes {
                let idx = base + gene;
                let val = gem_flat[idx].max(0.0).min(max_per_gene[gene]);
                delta_flat_old[idx] = val - gmtx_flat_old[idx];
            }
        }
        let old_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // ---- NEW implementation (column pin + zero-alloc parallel clip) ----
        let mut delta_new = delta_after_grn.clone();
        let t0 = Instant::now();
        for &gi in &target_genes {
            delta_new.column_mut(gi).assign(&delta_input.column(gi));
        }
        let delta_flat_new = delta_new.as_slice_memory_order_mut().unwrap();
        let gmtx_flat_new = gene_mtx.as_slice().unwrap();
        let max_ref = &max_per_gene;
        delta_flat_new
            .par_chunks_mut(n_genes)
            .enumerate()
            .for_each(|(cell, row)| {
                let base = cell * n_genes;
                for gene in 0..n_genes {
                    unsafe {
                        let orig = *gmtx_flat_new.get_unchecked(base + gene);
                        let val = (orig + *row.get_unchecked(gene))
                            .max(0.0)
                            .min(*max_ref.get_unchecked(gene));
                        *row.get_unchecked_mut(gene) = val - orig;
                    }
                }
            });
        let new_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // ---- Numerical parity ----
        let max_diff = delta_old
            .iter()
            .zip(delta_new.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        let speedup = old_ms / new_ms;
        eprintln!(
            "  {:>6}  {:>10.2}  {:>10.2}  {:>7.1}x  {:>12.2e}",
            n_cells, old_ms, new_ms, speedup, max_diff
        );

        assert!(
            max_diff < 1e-14,
            "pin_clip old vs new max_diff={:.2e} exceeds tolerance at n_cells={}",
            max_diff,
            n_cells
        );
    }
    eprintln!();
}

#[test]
fn perturb_obs_subset_file_maps_to_sorted_unique_indices() {
    use space_trav_lr_rust::perturb_mode::perturb_obs_indices_from_file;
    let obs: Vec<String> = (0..5).map(|i| format!("cell-{i}")).collect();
    let path = std::env::temp_dir().join(format!(
        "spacetravlr_obs_subset_{}.txt",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, "# header\ncell-3\ncell-1\ncell-3\ncell-0\n").unwrap();
    let idx = perturb_obs_indices_from_file(&path, &obs).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(idx, vec![0, 1, 3]);
}

fn two_cell_tf_lr_perturb_bundle() -> (
    Betabase,
    Array2<f64>,
    Vec<String>,
    Array2<f64>,
    GeneMatrix,
    Vec<PerturbTarget>,
) {
    let gene_names: Vec<String> = vec!["TF1", "LIG", "REC", "TARGET"]
        .into_iter()
        .map(String::from)
        .collect();
    let gene2index: HashMap<String, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    let row_labels = vec!["0".to_string()];
    let mut bf_target = BetaFrame::from_parts(
        "TARGET".to_string(),
        row_labels.clone(),
        array![0.0],
        array![[0.1]],
        vec!["TF1".to_string()],
        array![[0.05]],
        vec!["LIG".to_string()],
        vec!["REC".to_string()],
        ndarray::Array2::zeros((1, 0)),
        vec![],
        vec![],
    );

    let obs_names = vec!["cell_0".to_string(), "cell_1".to_string()];
    let cluster_keys = vec!["0".to_string(), "0".to_string()];
    let mapping = Arc::new(BetaFrame::compute_cell_mapping(
        &row_labels,
        &obs_names,
        &cluster_keys,
    ));
    bf_target.expand_to_cells(Arc::new(obs_names.clone()), mapping);
    bf_target.modulator_gene_indices = Some(
        bf_target
            .modulator_genes
            .iter()
            .map(|g| {
                let plain = g.strip_prefix("beta_").unwrap_or(g);
                *gene2index.get(plain).unwrap()
            })
            .collect(),
    );

    let mut bb_data = HashMap::new();
    bb_data.insert("TARGET".to_string(), bf_target);
    let bb = Betabase {
        data: bb_data,
        ligands_set: ["LIG".to_string()].into_iter().collect(),
        receptors_set: ["REC".to_string()].into_iter().collect(),
        tfl_ligands_set: HashSet::new(),
        tfs_set: ["TF1".to_string()].into_iter().collect(),
    };

    let gene_mtx = array![
        [1.0, 10.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
    ];
    let xy = array![[0.0, 0.0], [1.0, 0.0],];
    let rw_tfligands = GeneMatrix::new(Array2::<f32>::zeros((2, 0)), vec![]);
    let targets = vec![
        PerturbTarget {
            gene: "TF1".to_string(),
            desired_expr: 0.0,
            cell_indices: Some(vec![0]),
        },
        PerturbTarget {
            gene: "LIG".to_string(),
            desired_expr: 0.0,
            cell_indices: Some(vec![0]),
        },
    ];
    (
        bb,
        gene_mtx,
        gene_names,
        xy,
        rw_tfligands,
        targets,
    )
}

#[test]
fn perturb_spatial_knobs_change_simulated() {
    fn sim_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    let (bb, gene_mtx, gene_names, xy, rw_tfl, targets) = two_cell_tf_lr_perturb_bundle();
    let lig_names = vec!["LIG".to_string()];
    let scale = 1.0;
    let min_e = 0.0;

    let mut lr_close = HashMap::new();
    lr_close.insert("LIG".to_string(), 1.0);
    let rw_no_cut = compute_initial_weighted_ligands(
        &gene_mtx,
        &gene_names,
        &lig_names,
        &xy,
        &lr_close,
        scale,
        min_e,
        None,
        None,
    );
    let rw_cut = compute_initial_weighted_ligands(
        &gene_mtx,
        &gene_names,
        &lig_names,
        &xy,
        &lr_close,
        scale,
        min_e,
        None,
        Some(0.5),
    );
    let cfg_open = PerturbConfig {
        n_propagation: 1,
        contact_distance: None,
        ..Default::default()
    };
    let cfg_contact = PerturbConfig {
        n_propagation: 1,
        contact_distance: Some(0.5),
        ..Default::default()
    };
    let mut t0 = None;
    let s_open = perturb_with_targets(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_no_cut,
        &rw_tfl,
        &targets,
        &cfg_open,
        &lr_close,
        None,
        None,
        None,
        None,
        &mut t0,
    )
    .unwrap()
    .simulated;
    let mut t1 = None;
    let s_contact = perturb_with_targets(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_cut,
        &rw_tfl,
        &targets,
        &cfg_contact,
        &lr_close,
        None,
        None,
        None,
        None,
        &mut t1,
    )
    .unwrap()
    .simulated;
    assert!(
        sim_diff(&s_open, &s_contact) > 1e-6,
        "contact_distance should change received-ligand geometry and perturb output"
    );

    let mut lr_tight = HashMap::new();
    lr_tight.insert("LIG".to_string(), 0.08);
    let mut lr_wide = HashMap::new();
    lr_wide.insert("LIG".to_string(), 8.0);
    let rw_tight = compute_initial_weighted_ligands(
        &gene_mtx,
        &gene_names,
        &lig_names,
        &xy,
        &lr_tight,
        scale,
        min_e,
        None,
        None,
    );
    let rw_wide = compute_initial_weighted_ligands(
        &gene_mtx,
        &gene_names,
        &lig_names,
        &xy,
        &lr_wide,
        scale,
        min_e,
        None,
        None,
    );
    let cfg_r = PerturbConfig {
        n_propagation: 1,
        ..Default::default()
    };
    let mut t2 = None;
    let s_tight = perturb_with_targets(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_tight,
        &rw_tfl,
        &targets,
        &cfg_r,
        &lr_tight,
        None,
        None,
        None,
        None,
        &mut t2,
    )
    .unwrap()
    .simulated;
    let mut t3 = None;
    let s_wide = perturb_with_targets(
        &bb,
        &gene_mtx,
        &gene_names,
        &xy,
        &rw_wide,
        &rw_tfl,
        &targets,
        &cfg_r,
        &lr_wide,
        None,
        None,
        None,
        None,
        &mut t3,
    )
    .unwrap()
    .simulated;
    assert!(
        sim_diff(&s_tight, &s_wide) > 1e-6,
        "LR radius should change weighted ligands and perturb output"
    );

    let n_cells = 200;
    let (bb2, mut gene_mtx2, gene_names2, xy2, _, rw_tfl2, lr_radii2) =
        make_synthetic_inputs(n_cells);
    let b_idx = gene_names2.iter().position(|g| g == "B").unwrap();
    for i in 0..n_cells {
        gene_mtx2[[i, b_idx]] = 1.0 + (i as f64) * 0.02;
    }
    let lig_b = vec!["B".to_string()];
    let rw_shared = compute_initial_weighted_ligands(
        &gene_mtx2,
        &gene_names2,
        &lig_b,
        &xy2,
        &lr_radii2,
        scale,
        min_e,
        None,
        None,
    );
    let cfg_exact = PerturbConfig {
        n_propagation: 6,
        ligand_grid_factor: None,
        ..Default::default()
    };
    let cfg_grid = PerturbConfig {
        n_propagation: 6,
        ligand_grid_factor: Some(0.3),
        ..Default::default()
    };
    let ko = vec![("A".to_string(), 0.0)];
    let g_exact = perturb(
        &bb2,
        &gene_mtx2,
        &gene_names2,
        &xy2,
        &rw_shared,
        &rw_tfl2,
        &ko,
        &cfg_exact,
        &lr_radii2,
    );
    let g_grid = perturb(
        &bb2,
        &gene_mtx2,
        &gene_names2,
        &xy2,
        &rw_shared,
        &rw_tfl2,
        &ko,
        &cfg_grid,
        &lr_radii2,
    );
    let g_diff = sim_diff(&g_exact.simulated, &g_grid.simulated);
    assert!(
        g_diff > 1e-6,
        "ligand_grid_factor should change recomputed weighted ligands during propagation (L2 diff = {g_diff:e})"
    );
}
