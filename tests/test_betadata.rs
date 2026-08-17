use ndarray::{Array2, array};
use spacetravlr::betadata::{
    BetaFrame, BetaFrameFromParts, Betabase, GeneMatrix, betadata_feather_per_cell_column,
    betadata_pair_lr_parallel, write_betadata_feather,
};
use std::collections::HashSet;
use std::sync::Arc;

fn cluster_keys_from_usize(cl: &[usize]) -> Vec<String> {
    cl.iter().map(|c| c.to_string()).collect()
}

#[test]
fn resolve_betadata_key_column_matches_cluster_annot() {
    use polars::prelude::*;
    let df = DataFrame::new(vec![Series::new("cell_type".into(), &["A", "B"]).into()]).unwrap();
    assert_eq!(
        spacetravlr::betadata::resolve_betadata_cluster_key_column(&df, "cell_type"),
        "cell_type"
    );
    assert_eq!(
        spacetravlr::betadata::resolve_betadata_cluster_key_column(&df, "leiden"),
        "leiden"
    );
}

#[test]
fn cluster_betadata_row_keys_match_string_cell_type_labels() {
    use ndarray::Array1;
    use polars::prelude::*;
    use spacetravlr::betadata::build_cluster_id_to_betadata_cluster_key_map;
    let df = DataFrame::new(vec![
        Series::new("cell_type".into(), &["Tfh", "Tfh", "B cell"]).into(),
    ])
    .unwrap();
    let clusters = Array1::from_vec(vec![1usize, 1, 0]);
    let m = build_cluster_id_to_betadata_cluster_key_map(&df, "cell_type", &clusters).unwrap();
    assert_eq!(m.get(&0), Some(&"B cell".to_string()));
    assert_eq!(m.get(&1), Some(&"Tfh".to_string()));
}

fn make_test_betaframe() -> BetaFrame {
    BetaFrame::from_parts(BetaFrameFromParts {
        gene_name: "target1".to_string(),
        row_labels: vec!["0".to_string(), "1".to_string()],
        intercepts: array![1.0, 2.0],
        tf_betas: array![[0.5, -0.3], [0.8, 0.1]],
        tfs: vec!["A".to_string(), "B".to_string()],
        lr_betas: array![[0.2], [-0.1]],
        ligands: vec!["C".to_string()],
        receptors: vec!["D".to_string()],
        tfl_betas: array![[0.1], [0.4]],
        tfl_ligands: vec!["C".to_string()],
        tfl_regulators: vec!["A".to_string()],
    })
}

fn make_test_matrices() -> (GeneMatrix, GeneMatrix, GeneMatrix) {
    let rw_ligands = GeneMatrix::new(array![[3.0], [1.0]], vec!["C".to_string()]);
    let rw_ligands_tfl = GeneMatrix::new(array![[2.0], [0.5]], vec!["C".to_string()]);
    let gex_df = GeneMatrix::new(
        array![[1.5, 0.8, 3.0, 2.0], [0.0, 1.2, 1.0, 0.0]],
        vec!["A".into(), "B".into(), "C".into(), "D".into()],
    );
    (rw_ligands, rw_ligands_tfl, gex_df)
}

#[test]
fn test_betaframe_construction() {
    let bf = make_test_betaframe();
    assert_eq!(bf.gene_name, "target1");
    assert_eq!(bf.n_beta_rows, 2);
    assert_eq!(bf.n_cells, 2);
    assert_eq!(bf.tfs, vec!["A", "B"]);
    assert_eq!(bf.ligands, vec!["C"]);
    assert_eq!(bf.receptors, vec!["D"]);
    assert_eq!(bf.tfl_ligands, vec!["C"]);
    assert_eq!(bf.tfl_regulators, vec!["A"]);
    assert_eq!(
        bf.modulator_genes,
        vec!["beta_A", "beta_B", "beta_C", "beta_D"]
    );
}

#[test]
fn test_splash_identity_mapping() {
    let bf = make_test_betaframe();
    let (rw_lig, rw_tfl, gex) = make_test_matrices();

    let result = bf.splash(&rw_lig, &rw_tfl, &gex, 1.0, None);

    assert_eq!(
        result.col_names,
        vec!["beta_A", "beta_B", "beta_C", "beta_D"]
    );
    assert_eq!(result.n_rows(), 2);

    let eps = 1e-10;

    let a = result.col("beta_A").unwrap();
    assert!((a[0] - 0.7).abs() < eps, "beta_A[0]={} expected 0.7", a[0]);
    assert!((a[1] - 1.0).abs() < eps, "beta_A[1]={} expected 1.0", a[1]);

    let b = result.col("beta_B").unwrap();
    assert!((b[0] - (-0.3)).abs() < eps);
    assert!((b[1] - 0.1).abs() < eps);

    let c = result.col("beta_C").unwrap();
    assert!((c[0] - 0.55).abs() < eps);
    assert!((c[1] - 0.0).abs() < eps);

    let d = result.col("beta_D").unwrap();
    assert!((d[0] - 0.6).abs() < eps);
    assert!((d[1] - 0.0).abs() < eps);
}

#[test]
fn test_splash_expanded_cells() {
    let mut bf = make_test_betaframe();

    let obs: Vec<String> = vec![
        "cell0".into(),
        "cell1".into(),
        "cell2".into(),
        "cell3".into(),
    ];
    let cluster_keys = cluster_keys_from_usize(&[0usize, 0, 1, 1]);
    let mapping = Arc::new(BetaFrame::compute_cell_mapping(&bf.row_labels, &obs, &cluster_keys).0);
    bf.expand_to_cells(Arc::new(obs), mapping);
    assert_eq!(bf.n_cells, 4);
    assert_eq!(*bf.cell_to_beta_row, vec![0, 0, 1, 1]);

    let rw_lig = GeneMatrix::new(array![[3.0], [2.0], [1.0], [0.5]], vec!["C".to_string()]);
    let rw_tfl = GeneMatrix::new(array![[2.0], [1.0], [0.5], [0.3]], vec!["C".to_string()]);
    let gex = GeneMatrix::new(
        array![
            [1.5, 0.8, 3.0, 2.0],
            [1.0, 0.5, 2.0, 1.0],
            [0.0, 1.2, 1.0, 0.0],
            [0.5, 0.3, 0.8, 0.5]
        ],
        vec!["A".into(), "B".into(), "C".into(), "D".into()],
    );

    let result = bf.splash(&rw_lig, &rw_tfl, &gex, 1.0, None);
    assert_eq!(result.n_rows(), 4);

    let eps = 1e-10;
    let a = result.col("beta_A").unwrap();
    assert!((a[0] - 0.7).abs() < eps, "cell0 beta_A={}", a[0]);
    assert!((a[1] - 0.6).abs() < eps, "cell1 beta_A={}", a[1]);
    assert!((a[2] - 1.0).abs() < eps, "cell2 beta_A={}", a[2]);
    assert!((a[3] - 0.92).abs() < eps, "cell3 beta_A={}", a[3]);

    let d = result.col("beta_D").unwrap();
    assert!((d[0] - 0.6).abs() < eps, "cell0 beta_D={}", d[0]);
    assert!((d[1] - 0.4).abs() < eps, "cell1 beta_D={}", d[1]);
    assert!((d[2] - 0.0).abs() < eps);
    assert!((d[3] - (-0.05)).abs() < eps, "cell3 beta_D={}", d[3]);

    let c = result.col("beta_C").unwrap();
    assert!((c[3] - 0.15).abs() < eps, "cell3 beta_C={}", c[3]);
}

#[test]
fn test_splash_with_scale_and_cap() {
    let bf = make_test_betaframe();
    let (rw_lig, rw_tfl, gex) = make_test_matrices();

    let result = bf.splash(&rw_lig, &rw_tfl, &gex, 2.0, Some(0.5));
    let eps = 1e-10;

    // TF + unscaled TFL-reg: 0.5 + 0.1*2 = 0.7 → cap 0.5
    let a = result.col("beta_A").unwrap();
    assert!((a[0] - 0.5).abs() < eps);
    // 0.8 + 0.4*0.5 = 1.0 → cap 0.5
    assert!((a[1] - 0.5).abs() < eps);

    let b = result.col("beta_B").unwrap();
    assert!((b[0] - (-0.3)).abs() < eps);
    assert!((b[1] - 0.1).abs() < eps);
}

#[test]
fn splash_beta_scale_affects_ligands_not_tfs() {
    let bf = make_test_betaframe();
    let (rw_lig, rw_tfl, gex) = make_test_matrices();
    let s1 = bf.splash(&rw_lig, &rw_tfl, &gex, 1.0, None);
    let s2 = bf.splash(&rw_lig, &rw_tfl, &gex, 2.0, None);
    let eps = 1e-10;

    let b1 = s1.col("beta_B").unwrap();
    let b2 = s2.col("beta_B").unwrap();
    assert!((b1[0] - b2[0]).abs() < eps);
    assert!((b1[1] - b2[1]).abs() < eps);

    let a1 = s1.col("beta_A").unwrap();
    let a2 = s2.col("beta_A").unwrap();
    assert!((a1[0] - a2[0]).abs() < eps);
    assert!((a1[1] - a2[1]).abs() < eps);

    let c1 = s1.col("beta_C").unwrap()[0];
    let c2 = s2.col("beta_C").unwrap()[0];
    assert!(
        c2 > c1 + eps,
        "ligand channel should grow with beta_scale_factor"
    );
}

#[test]
fn test_feather_roundtrip_seed_only() {
    let dir = std::env::temp_dir().join("betadata_test_feather_seed");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("target1_betadata.feather");

    let ids = vec!["0".to_string(), "1".to_string()];
    let cols = vec![
        "beta0".into(),
        "A".into(),
        "B".into(),
        "C$D".into(),
        "C#A".into(),
    ];
    let mut mat = Array2::zeros((2, 5));
    mat[[0, 0]] = 1.0;
    mat[[0, 1]] = 0.5;
    mat[[0, 2]] = -0.3;
    mat[[0, 3]] = 0.2;
    mat[[0, 4]] = 0.1;
    mat[[1, 0]] = 2.0;
    mat[[1, 1]] = 0.8;
    mat[[1, 2]] = 0.1;
    mat[[1, 3]] = -0.1;
    mat[[1, 4]] = 0.4;

    write_betadata_feather(path.to_str().unwrap(), "Cluster", &ids, &cols, &mat).unwrap();

    let bf = BetaFrame::from_feather(path.to_str().unwrap()).unwrap();
    assert_eq!(bf.gene_name, "target1");
    assert_eq!(bf.n_beta_rows, 2);
    assert_eq!(bf.tfs, vec!["A", "B"]);
    assert_eq!(bf.intercepts, array![1.0, 2.0]);
    assert_eq!(bf.tf_betas, array![[0.5, -0.3], [0.8, 0.1]]);

    let (rw_lig, rw_tfl, gex) = make_test_matrices();
    let result = bf.splash(&rw_lig, &rw_tfl, &gex, 1.0, None);
    let a = result.col("beta_A").unwrap();
    assert!((a[0] - 0.7).abs() < 1e-10);
    assert!((a[1] - 1.0).abs() < 1e-10);

    let bf2 = BetaFrame::from_path(path.to_str().unwrap()).unwrap();
    assert_eq!(bf2.gene_name, "target1");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_betabase_from_directory_cnn_cellid() {
    let dir = std::env::temp_dir().join("betadata_test_dir_cnn_feather");
    std::fs::create_dir_all(&dir).unwrap();

    let ids = vec!["c0".into(), "c1".into()];
    let cols_g1 = vec!["beta0".into(), "X".into()];
    let m1 = array![[1.0f64, 0.5], [2.0, 0.25]];
    write_betadata_feather(
        dir.join("G1_betadata.feather").to_str().unwrap(),
        "CellID",
        &ids,
        &cols_g1,
        &m1,
    )
    .unwrap();
    let cols_g2 = vec!["beta0".into(), "Y".into()];
    let m2 = array![[0.0f64, 1.0], [0.0, 2.0]];
    write_betadata_feather(
        dir.join("G2_betadata.feather").to_str().unwrap(),
        "CellID",
        &ids,
        &cols_g2,
        &m2,
    )
    .unwrap();

    let obs = vec!["c0".into(), "c1".into()];
    let clusters = [99usize, 99];
    let ck = cluster_keys_from_usize(&clusters);
    let bb = Betabase::from_directory(dir.to_str().unwrap(), &obs, &ck, None, None).unwrap();

    assert_eq!(bb.data.len(), 2);
    let f1 = &bb.data["G1"];
    assert_eq!(f1.n_cells, 2);
    assert_eq!(*f1.cell_to_beta_row, vec![0, 1]);
    assert!(f1.join_by_obs_name);

    std::fs::remove_dir_all(&dir).ok();
}

/// When CellID equals obs name but cluster keys repeat (e.g. all `"0"`), mapping must use obs
/// names — not cluster keys — or every cell hits the same feather row and spatial β nuance is lost.
#[test]
fn betadata_per_cell_column_cellid_ignores_cluster_key_collisions() {
    let dir = std::env::temp_dir().join("betadata_cellid_map_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("T_betadata.feather");
    let cols = vec!["beta0".into(), "coef".into()];
    let m = array![[0.0f64, 10.0], [0.0, 20.0], [0.0, 30.0]];
    write_betadata_feather(
        path.to_str().unwrap(),
        "CellID",
        &["0".into(), "1".into(), "2".into()],
        &cols,
        &m,
    )
    .unwrap();

    let obs = vec!["0".into(), "1".into(), "2".into()];
    let cluster_keys = vec!["0".into(), "0".into(), "0".into()];
    let v = betadata_feather_per_cell_column(path.to_str().unwrap(), "coef", &obs, &cluster_keys)
        .unwrap();
    assert!((v[0] - 10.0).abs() < 1e-5);
    assert!((v[1] - 20.0).abs() < 1e-5);
    assert!((v[2] - 30.0).abs() < 1e-5);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn splash_empty_modulators_does_not_panic() {
    let bf = BetaFrame::from_parts(BetaFrameFromParts {
        gene_name: "intercept_only".to_string(),
        row_labels: vec!["c0".to_string()],
        intercepts: array![0.0],
        tf_betas: Array2::zeros((1, 0)),
        tfs: vec![],
        lr_betas: Array2::zeros((1, 0)),
        ligands: vec![],
        receptors: vec![],
        tfl_betas: Array2::zeros((1, 0)),
        tfl_ligands: vec![],
        tfl_regulators: vec![],
    });
    let empty = GeneMatrix::new(Array2::zeros((1, 0)), vec![]);
    let gex = GeneMatrix::new(array![[1.0]], vec!["X".to_string()]);
    let result = bf.splash(&empty, &empty, &gex, 1.0, None);
    assert_eq!(result.n_cols(), 0);
    assert_eq!(result.n_rows(), 1);
}

#[test]
fn test_betaframe_tf_only() {
    let bf = BetaFrame::from_parts(BetaFrameFromParts {
        gene_name: "simple".to_string(),
        row_labels: vec!["c0".to_string(), "c1".to_string()],
        intercepts: array![0.0, 0.0],
        tf_betas: array![[1.0, 2.0], [3.0, 4.0]],
        tfs: vec!["X".to_string(), "Y".to_string()],
        lr_betas: Array2::zeros((2, 0)),
        ligands: vec![],
        receptors: vec![],
        tfl_betas: Array2::zeros((2, 0)),
        tfl_ligands: vec![],
        tfl_regulators: vec![],
    });

    let empty = GeneMatrix::new(Array2::zeros((2, 0)), vec![]);
    let gex = GeneMatrix::new(
        array![[1.0, 2.0], [3.0, 4.0]],
        vec!["X".to_string(), "Y".to_string()],
    );

    let result = bf.splash(&empty, &empty, &gex, 1.0, None);

    let x = result.col("beta_X").unwrap();
    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 3.0).abs() < 1e-10);
    let y = result.col("beta_Y").unwrap();
    assert!((y[0] - 2.0).abs() < 1e-10);
    assert!((y[1] - 4.0).abs() < 1e-10);
}

#[test]
fn test_betabase_from_directory() {
    let dir = std::env::temp_dir().join("betadata_test_dir_feather");
    std::fs::create_dir_all(&dir).unwrap();

    let g1_cols = vec!["beta0".into(), "TF1".into(), "TF2".into()];
    let g1_mat = array![[1.0f64, 0.5, 0.3], [2.0, 0.1, 0.2]];
    write_betadata_feather(
        dir.join("gene1_betadata.feather").to_str().unwrap(),
        "Cluster",
        &["0".into(), "1".into()],
        &g1_cols,
        &g1_mat,
    )
    .unwrap();

    let g2_cols = vec!["beta0".into(), "TF1".into(), "L1$R1".into()];
    let g2_mat = array![[0.5f64, 0.1, 0.4], [0.6, 0.2, 0.3]];
    write_betadata_feather(
        dir.join("gene2_betadata.feather").to_str().unwrap(),
        "Cluster",
        &["0".into(), "1".into()],
        &g2_cols,
        &g2_mat,
    )
    .unwrap();

    std::fs::write(dir.join("other.csv"), "x,y\n1,2").unwrap();

    let obs: Vec<String> = vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()];
    let clusters = vec![0, 0, 1, 1, 0];

    let ck = cluster_keys_from_usize(&clusters);
    let bb = Betabase::from_directory(dir.to_str().unwrap(), &obs, &ck, None, None).unwrap();

    assert_eq!(bb.data.len(), 2);
    assert!(bb.data.contains_key("gene1"));
    assert!(bb.data.contains_key("gene2"));
    assert!(bb.tfs_set.contains("TF1"));
    assert!(bb.ligands_set.contains("L1"));

    for frame in bb.data.values() {
        assert_eq!(frame.n_cells, 5);
        assert_eq!(*frame.cell_to_beta_row, vec![0, 0, 1, 1, 0]);
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// One bad cluster id must not force obs-name fallback for every cell (seed-only / Cluster rows).
#[test]
fn compute_cell_mapping_partial_cluster_still_maps_others() {
    let row_labels = vec!["0".to_string(), "1".to_string()];
    let obs = vec!["a".into(), "b".into(), "c".into()];
    let keys = vec!["0".into(), "1".into(), "999".into()];
    let (m, _) = BetaFrame::compute_cell_mapping(&row_labels, &obs, &keys);
    assert_eq!(
        m,
        vec![0, 1, BetaFrame::missing_beta_row_index(row_labels.len())]
    );
}

#[test]
fn compute_cell_mapping_matches_category_name_not_code() {
    let row_labels = vec!["0".into(), "1".into(), "10".into()];
    let obs = vec!["a".into(), "b".into(), "c".into()];
    let keys = vec!["0".into(), "1".into(), "10".into()];
    let (m, _) = BetaFrame::compute_cell_mapping(&row_labels, &obs, &keys);
    assert_eq!(m, vec![0, 1, 2]);
    let wrong = vec!["0".into(), "1".into(), "2".into()];
    let (m2, _) = BetaFrame::compute_cell_mapping(&row_labels, &obs, &wrong);
    assert_eq!(
        m2,
        vec![0, 1, BetaFrame::missing_beta_row_index(row_labels.len())]
    );
}

#[test]
fn splash_unmapped_cluster_uses_zero_betas_not_row_zero() {
    let mut bf = make_test_betaframe();
    let obs: Vec<String> = vec!["cell0".into(), "cell1".into()];
    let keys = vec!["0".into(), "missing_type".into()];
    let mapping = Arc::new(BetaFrame::compute_cell_mapping(&bf.row_labels, &obs, &keys).0);
    bf.expand_to_cells(Arc::new(obs), mapping);

    let rw_lig = GeneMatrix::new(array![[3.0], [2.0]], vec!["C".to_string()]);
    let rw_tfl = GeneMatrix::new(array![[2.0], [1.0]], vec!["C".to_string()]);
    let gex = GeneMatrix::new(
        array![[1.5, 0.8], [1.0, 0.5]],
        vec!["C".to_string(), "A".to_string()],
    );
    let splash = bf.splash(&rw_lig, &rw_tfl, &gex, 1.0, None);
    let a0 = splash.col("beta_A").unwrap()[0];
    let a1 = splash.col("beta_A").unwrap()[1];
    assert!(
        (a0 - 0.7).abs() < 1e-6,
        "mapped cell should use cluster 0 β"
    );
    assert!(a1.abs() < 1e-6, "unmapped cell must not inherit row 0 β");
}

#[test]
fn test_feather_cnn_cellid_columns() {
    let dir = std::env::temp_dir().join("betadata_test_feather_cnn");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("target2_betadata.feather");

    let ids = vec!["cell0".into(), "cell1".into()];
    let cols = vec![
        "beta0".into(),
        "beta_A".into(),
        "beta_B".into(),
        "beta_C$D".into(),
    ];
    let mat = array![[1.0f64, 0.5, -0.3, 0.2], [2.0, 0.8, 0.1, -0.1]];
    write_betadata_feather(path.to_str().unwrap(), "CellID", &ids, &cols, &mat).unwrap();

    let bf = BetaFrame::from_feather(path.to_str().unwrap()).unwrap();
    assert_eq!(bf.gene_name, "target2");
    assert_eq!(bf.tfs, vec!["A", "B"]);
    assert_eq!(bf.ligands, vec!["C"]);
    assert_eq!(bf.receptors, vec!["D"]);

    let obs: Vec<String> = vec!["cell0".into(), "cell1".into()];
    let cluster_keys = cluster_keys_from_usize(&[0usize, 0]);
    let (mapping, _) = BetaFrame::compute_cell_mapping(&bf.row_labels, &obs, &cluster_keys);
    assert_eq!(mapping, vec![0, 1]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_mapping_shared_via_arc() {
    let dir = std::env::temp_dir().join("betadata_test_arc_feather");
    std::fs::create_dir_all(&dir).unwrap();

    let c = vec!["beta0".into(), "TF1".into()];
    let m1 = array![[1.0f64, 0.5], [2.0, 0.1]];
    write_betadata_feather(
        dir.join("g1_betadata.feather").to_str().unwrap(),
        "Cluster",
        &["0".into(), "1".into()],
        &c,
        &m1,
    )
    .unwrap();
    let c2 = vec!["beta0".into(), "TF2".into()];
    let m2 = array![[0.5f64, 0.3], [0.6, 0.2]];
    write_betadata_feather(
        dir.join("g2_betadata.feather").to_str().unwrap(),
        "Cluster",
        &["0".into(), "1".into()],
        &c2,
        &m2,
    )
    .unwrap();

    let obs: Vec<String> = (0..1000).map(|i| format!("cell{}", i)).collect();
    let clusters: Vec<usize> = (0..1000).map(|i| i % 2).collect();
    let ck = cluster_keys_from_usize(&clusters);

    let bb = Betabase::from_directory(dir.to_str().unwrap(), &obs, &ck, None, None).unwrap();

    let g1 = &bb.data["g1"];
    let g2 = &bb.data["g2"];
    assert!(Arc::ptr_eq(&g1.cell_to_beta_row, &g2.cell_to_beta_row));
    assert!(Arc::ptr_eq(&g1.cell_labels, &g2.cell_labels));

    std::fs::remove_dir_all(&dir).ok();
}

fn build_splash_inputs(
    betas_dir: &str,
    n_cells: usize,
    n_clusters: usize,
) -> (Betabase, GeneMatrix, GeneMatrix, GeneMatrix, Vec<String>) {
    let obs_names: Vec<String> = (0..n_cells).map(|i| format!("cell_{}", i)).collect();
    let clusters: Vec<usize> = (0..n_cells).map(|i| i % n_clusters).collect();
    let ck = cluster_keys_from_usize(&clusters);

    let bb = Betabase::from_directory(betas_dir, &obs_names, &ck, None, None).unwrap();

    let mut all_genes: HashSet<String> = HashSet::new();
    let mut lig_set: HashSet<String> = HashSet::new();
    for frame in bb.data.values() {
        all_genes.extend(frame.tfs.iter().cloned());
        all_genes.extend(frame.ligands.iter().cloned());
        all_genes.extend(frame.receptors.iter().cloned());
        all_genes.extend(frame.tfl_ligands.iter().cloned());
        all_genes.extend(frame.tfl_regulators.iter().cloned());
        lig_set.extend(frame.ligands.iter().cloned());
        lig_set.extend(frame.tfl_ligands.iter().cloned());
    }
    let all_genes_vec: Vec<String> = {
        let mut v: Vec<_> = all_genes.into_iter().collect();
        v.sort();
        v
    };
    let lig_genes: Vec<String> = {
        let mut v: Vec<_> = lig_set.into_iter().collect();
        v.sort();
        v
    };

    let gex_df = GeneMatrix::new(
        Array2::from_elem((n_cells, all_genes_vec.len()), 1.0),
        all_genes_vec,
    );
    let rw_data = Array2::from_elem((n_cells, lig_genes.len()), 1.0);
    let rw_ligands = GeneMatrix::new(rw_data.clone(), lig_genes.clone());
    let rw_ligands_tfl = GeneMatrix::new(rw_data, lig_genes);

    let mut gene_names: Vec<String> = bb.data.keys().cloned().collect();
    gene_names.sort();

    (bb, rw_ligands, rw_ligands_tfl, gex_df, gene_names)
}

#[test]
#[ignore]
fn test_splash_from_tmp_betas() {
    let betas_dir = "/tmp/betas";
    if !std::path::Path::new(betas_dir).exists() {
        eprintln!("Skipping: /tmp/betas not found");
        return;
    }

    let out_dir = "/tmp/splash_compare";
    std::fs::create_dir_all(out_dir).unwrap();

    let (bb, rw_ligands, rw_ligands_tfl, gex_df, gene_names) =
        build_splash_inputs(betas_dir, 20, 2);

    for gene_name in &gene_names {
        let frame = &bb.data[gene_name];
        let result = frame.splash(&rw_ligands, &rw_ligands_tfl, &gex_df, 1.0, None);

        let csv_path = format!("{}/{}_splash_rs.csv", out_dir, gene_name);
        let mut wtr = std::fs::File::create(&csv_path).unwrap();
        use std::io::Write;
        write!(wtr, "cell").unwrap();
        for col in &result.col_names {
            write!(wtr, ",{}", col).unwrap();
        }
        writeln!(wtr).unwrap();
        for i in 0..result.n_rows() {
            write!(wtr, "cell_{}", i).unwrap();
            for j in 0..result.col_names.len() {
                write!(wtr, ",{}", result.data[[i, j]]).unwrap();
            }
            writeln!(wtr).unwrap();
        }
    }

    println!("Rust splash CSVs saved to {}", out_dir);
}

#[test]
#[ignore]
fn bench_splash() {
    use std::time::Instant;

    let betas_dir = "/tmp/betas";
    if !std::path::Path::new(betas_dir).exists() {
        eprintln!("Skipping: /tmp/betas not found");
        return;
    }

    let n_clusters = 13;
    let n_warmup = 1;
    let n_iters = 5;

    println!();
    for &n_cells in &[100, 1_000, 10_000, 50_000, 100_000] {
        let (bb, rw_ligands, rw_ligands_tfl, gex_df, gene_names) =
            build_splash_inputs(betas_dir, n_cells, n_clusters);

        // Warmup
        for _ in 0..n_warmup {
            for gn in &gene_names {
                let _ = bb.data[gn].splash(&rw_ligands, &rw_ligands_tfl, &gex_df, 1.0, None);
            }
        }

        // Timed runs
        let mut times_ms = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            for gn in &gene_names {
                let _ = bb.data[gn].splash(&rw_ligands, &rw_ligands_tfl, &gex_df, 1.0, None);
            }
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let mean: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let std: f64 = (times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / times_ms.len() as f64)
            .sqrt();
        println!(
            "  {:>6} cells x {} genes: {:>8.1} ms  (std {:.1} ms)",
            n_cells,
            gene_names.len(),
            mean,
            std
        );
    }
    println!();
}

#[test]
fn betadata_pair_lr_parallel_seed_clusters() {
    let dir = std::env::temp_dir().join(format!(
        "stlr_pair_lr_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("EGFR_betadata.feather");
    let ids = vec!["0".to_string(), "1".to_string()];
    let cols = vec!["beta_TGFA$EGFR".to_string()];
    let data = ndarray::array![[0.7_f64], [0.1_f64]];
    write_betadata_feather(path.to_str().unwrap(), "Cluster", &ids, &cols, &data).unwrap();

    let obs: Vec<String> = (0..10).map(|i| format!("o{}", i)).collect();
    let clusters: Vec<usize> = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let ck = cluster_keys_from_usize(&clusters);
    let rows = betadata_pair_lr_parallel(
        dir.to_str().unwrap(),
        &[String::from("EGFR")],
        &obs,
        &ck,
        0,
        5,
    )
    .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].target_gene, "EGFR");
    assert_eq!(rows[0].interaction, "beta_TGFA$EGFR");
    assert!((rows[0].beta_cell_a - 0.7).abs() < 1e-9);
    assert!((rows[0].beta_cell_b - 0.1).abs() < 1e-9);
    let _ = std::fs::remove_dir_all(&dir);
}
