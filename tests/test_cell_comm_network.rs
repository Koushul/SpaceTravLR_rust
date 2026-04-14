use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use ndarray::{Array1, Array2};
use spacetravlr::betadata::{BetaFrame, Betabase, GeneMatrix};
use spacetravlr::cell_comm_network::{CellCommNetworkParams, aggregate_lr_tfl_communication_edges};
use spacetravlr::config::SpaceshipConfig;
use spacetravlr::perturb::PerturbConfig;
use spacetravlr::perturb_mode::PerturbRuntime;

fn mock_rt_one_lr_target(
    xy: Array2<f64>,
    gene_mtx: Array2<f64>,
    gene_names: Vec<String>,
    obs_names: Vec<String>,
    lr_beta_per_row: &[f32],
    cfg_radius: f64,
    wl_scale: f64,
    beta_scale: f64,
) -> PerturbRuntime {
    assert_eq!(lr_beta_per_row.len(), obs_names.len());
    let n = obs_names.len();
    let lr_betas = Array2::from_shape_vec((n, 1), lr_beta_per_row.to_vec()).unwrap();
    let frame = BetaFrame::from_parts(
        "TGT".into(),
        obs_names.clone(),
        Array1::zeros(n),
        Array2::zeros((n, 0)),
        vec![],
        lr_betas,
        vec!["LIG".into()],
        vec!["REC".into()],
        Array2::zeros((n, 0)),
        vec![],
        vec![],
    );
    let mut data = HashMap::new();
    data.insert("TGT".into(), frame);
    let bb = Betabase {
        data,
        ligands_set: HashSet::from(["LIG".into()]),
        receptors_set: HashSet::from(["REC".into()]),
        tfl_ligands_set: HashSet::new(),
        tfs_set: HashSet::new(),
    };
    let mut cfg = SpaceshipConfig::default();
    cfg.spatial.radius = cfg_radius;
    cfg.spatial.weighted_ligand_scale_factor = wl_scale;
    cfg.spatial.contact_distance = 1.0e12;

    let mut lr_radii = HashMap::new();
    lr_radii.insert("LIG".into(), cfg_radius);

    PerturbRuntime {
        run_toml_path: PathBuf::from("test"),
        run_dir: PathBuf::from("test"),
        cfg,
        gene_mtx,
        gene_names,
        obs_names,
        betadata_cluster_key: vec!["ct".into(); n],
        cell_types: vec![0; n],
        bb,
        xy,
        rw_ligands_init: GeneMatrix::new(Array2::zeros((n, 0)), vec![]),
        rw_tfligands_init: GeneMatrix::new(Array2::zeros((n, 0)), vec![]),
        lr_radii,
        perturb_cfg: PerturbConfig {
            beta_scale_factor: beta_scale,
            ..Default::default()
        },
        baseline_splash_cache: Mutex::new(None),
    }
}

#[test]
fn cell_network_single_cell_self_loop_matches_manual() {
    let n = 1usize;
    let xy = Array2::from_shape_vec((n, 2), vec![0.0_f64, 0.0]).unwrap();
    let gene_names = vec!["LIG".into(), "REC".into()];
    let gene_mtx = Array2::from_shape_vec((n, 2), vec![5.0_f64, 4.0]).unwrap();
    let obs_names = vec!["c0".into()];
    let rt = mock_rt_one_lr_target(
        xy,
        gene_mtx,
        gene_names,
        obs_names,
        &[1.0_f32],
        100.0,
        2.0,
        0.5,
    );
    let params = CellCommNetworkParams {
        beta_scale_factor: 0.5,
        min_expression: 1e-12,
        edge_threshold_abs: 0.0,
        include_self_loops: true,
        ignore_contact_distance: true,
    };
    let (w, _) = aggregate_lr_tfl_communication_edges(&rt, &params);
    let n_inv = 1.0_f64;
    let jac_00 = 2.0 * n_inv * (0.0_f64).exp();
    let expected = (1.0_f64 * 4.0 * 0.5 * jac_00).abs();
    approx::assert_relative_eq!(w[[0, 0]], expected, epsilon = 1e-9);
}

#[test]
fn cell_network_two_cells_cross_edge_ordering() {
    let n = 2usize;
    let xy = Array2::from_shape_vec((n, 2), vec![0.0_f64, 0.0, 1.0, 0.0]).unwrap();
    let gene_names = vec!["LIG".into(), "REC".into()];
    let gene_mtx = Array2::from_shape_vec((n, 2), vec![1.0_f64, 10.0, 1.0, 10.0]).unwrap();
    let obs_names = vec!["c0".into(), "c1".into()];
    let rt = mock_rt_one_lr_target(
        xy,
        gene_mtx,
        gene_names,
        obs_names,
        &[1.0_f32, 1.0_f32],
        50.0,
        1.0,
        1.0,
    );
    let params = CellCommNetworkParams {
        beta_scale_factor: 1.0,
        min_expression: 1e-12,
        edge_threshold_abs: 0.0,
        include_self_loops: false,
        ignore_contact_distance: true,
    };
    let (w, _) = aggregate_lr_tfl_communication_edges(&rt, &params);
    let r = 50.0_f64;
    let scale = 1.0_f64;
    let n_inv = 1.0 / n as f64;
    let d01 = 1.0_f64;
    let jac_01 = scale * n_inv * (-d01 * d01 / (2.0 * r * r)).exp();
    let expected_w01 = (1.0_f64 * 10.0 * 1.0 * jac_01).abs();
    approx::assert_relative_eq!(w[[0, 1]], expected_w01, epsilon = 1e-9);
    approx::assert_relative_eq!(w[[1, 0]], expected_w01, epsilon = 1e-9);
    assert_eq!(w[[0, 0]], 0.0);
    assert_eq!(w[[1, 1]], 0.0);
}

#[test]
fn cell_network_zero_receptor_skips_row() {
    let n = 2usize;
    let xy = Array2::from_shape_vec((n, 2), vec![0.0_f64, 0.0, 10.0, 0.0]).unwrap();
    let gene_names = vec!["LIG".into(), "REC".into()];
    let gene_mtx = Array2::from_shape_vec((n, 2), vec![1.0_f64, 5.0, 1.0, 0.0]).unwrap();
    let obs_names = vec!["c0".into(), "c1".into()];
    let rt = mock_rt_one_lr_target(
        xy,
        gene_mtx,
        gene_names,
        obs_names,
        &[1.0_f32, 1.0_f32],
        50.0,
        1.0,
        1.0,
    );
    let params = CellCommNetworkParams {
        beta_scale_factor: 1.0,
        min_expression: 1e-9,
        edge_threshold_abs: 0.0,
        include_self_loops: true,
        ignore_contact_distance: true,
    };
    let (w, _) = aggregate_lr_tfl_communication_edges(&rt, &params);
    assert_eq!(w[[1, 0]], 0.0);
    assert_eq!(w[[1, 1]], 0.0);
    assert!(w[[0, 0]] > 0.0);
    assert!(w[[0, 1]] > 0.0);
}
