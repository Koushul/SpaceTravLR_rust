//! Betas are fit only for `[data].train_cell_types`, while received ligand still
//! includes senders of every other type.

use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp};
use anndata_hdf5::H5;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn_autodiff::Autodiff;
use ndarray::{Array2, Axis};
use polars::prelude::{DataFrame, IpcReader, NamedFrom, ParquetWriter, SerReader, Series};
use spacetravlr::config::{CnnTrainingMode, SpaceshipConfig};
use spacetravlr::ligand::calculate_weighted_ligands;
use spacetravlr::spatial_estimator::{SpatialCellularProgramsEstimator, dense_to_csr_f64};
use std::path::{Path, PathBuf};

const RADIUS: f64 = 1.5;
const N_A: usize = 6;
const N_B: usize = 2;

fn write_grn(dir: &Path) -> anyhow::Result<()> {
    let mut df = DataFrame::new(vec![
        Series::new("source".into(), vec!["Reg1", "Lig1", "Lig1"]).into(),
        Series::new("target".into(), vec!["Tgt1", "Rec1", "Reg1"]).into(),
        Series::new("edge_type".into(), vec!["grn", "lr", "nichenet"]).into(),
        Series::new("weight".into(), vec![1.0_f64, 1.0, 1.0]).into(),
    ])?;
    let f = std::fs::File::create(dir.join("mouse_network.parquet"))?;
    ParquetWriter::new(f).finish(&mut df)?;
    Ok(())
}

fn slide_xy_and_expr() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let n = N_A + N_B;
    let mut xy = Array2::<f64>::zeros((n, 2));
    let mut lig = Array2::<f64>::zeros((n, 1));
    let mut mat = Array2::<f64>::zeros((n, 4));
    for i in 0..N_A {
        xy[[i, 0]] = i as f64;
        mat[[i, 0]] = 0.25 + 0.2 * i as f64;
    }
    xy[[N_A, 0]] = 0.0;
    xy[[N_A + 1, 0]] = 0.2;
    lig[[N_A, 0]] = 12.0;
    lig[[N_A + 1, 0]] = 8.0;
    for i in 0..n {
        mat[[i, 2]] = lig[[i, 0]];
    }
    let received = calculate_weighted_ligands(&xy, &lig, RADIUS, 1.0);
    for i in 0..n {
        mat[[i, 1]] = mat[[i, 0]] + received[[i, 0]];
    }
    (xy, mat, lig)
}

fn write_slide(
    path: &Path,
    xy: &Array2<f64>,
    mat: &Array2<f64>,
    sample: Option<&str>,
) -> anyhow::Result<()> {
    let n = xy.nrows();
    let a = AnnData::<H5>::new(path)?;
    let obs_names: Vec<String> = (0..n)
        .map(|i| match sample {
            Some(s) => format!("{s}_c{i}"),
            None => format!("c{i}"),
        })
        .collect();
    a.set_obs_names(obs_names.into())?;
    a.set_var_names(vec!["Reg1".into(), "Tgt1".into(), "Lig1".into(), "Rec1".into()].into())?;
    let cell_types: Vec<String> = (0..n)
        .map(|i| if i < N_A { "ct_a" } else { "ct_b" }.to_string())
        .collect();
    let mut cols = vec![Series::new("cell_type".into(), cell_types).into()];
    if let Some(s) = sample {
        cols.push(Series::new("sample".into(), vec![s.to_string(); n]).into());
    }
    a.set_obs(DataFrame::new(cols)?)?;
    a.set_var(DataFrame::new(vec![
        Series::new("gene_ids".into(), vec!["r", "t", "l", "c"]).into(),
    ])?)?;
    a.set_x(ArrayData::from(dense_to_csr_f64(mat)?))?;
    a.set_obsm([("spatial".to_string(), ArrayData::from(xy.clone()))])?;
    a.close()?;
    Ok(())
}

fn write_two_slides(path: &Path, xy: &Array2<f64>, mat: &Array2<f64>) -> anyhow::Result<()> {
    let n = xy.nrows();
    let n_obs = n * 2;
    let a = AnnData::<H5>::new(path)?;
    let mut names = Vec::with_capacity(n_obs);
    let mut types = Vec::with_capacity(n_obs);
    let mut samples = Vec::with_capacity(n_obs);
    let mut stacked = Array2::<f64>::zeros((n_obs, mat.ncols()));
    let mut xy_all = Array2::<f64>::zeros((n_obs, 2));
    for s in 0..2 {
        let label = if s == 0 { "s1" } else { "s2" };
        for i in 0..n {
            names.push(format!("{label}_c{i}"));
            types.push(if i < N_A { "ct_a" } else { "ct_b" }.to_string());
            samples.push(label.to_string());
            let row = s * n + i;
            stacked.row_mut(row).assign(&mat.row(i));
            xy_all.row_mut(row).assign(&xy.row(i));
        }
    }
    a.set_obs_names(names.into())?;
    a.set_var_names(vec!["Reg1".into(), "Tgt1".into(), "Lig1".into(), "Rec1".into()].into())?;
    a.set_obs(DataFrame::new(vec![
        Series::new("cell_type".into(), types).into(),
        Series::new("sample".into(), samples).into(),
    ])?)?;
    a.set_var(DataFrame::new(vec![
        Series::new("gene_ids".into(), vec!["r", "t", "l", "c"]).into(),
    ])?)?;
    a.set_x(ArrayData::from(dense_to_csr_f64(&stacked)?))?;
    a.set_obsm([("spatial".to_string(), ArrayData::from(xy_all))])?;
    a.close()?;
    Ok(())
}

fn temp_dir(suffix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_train_ct_{}_{}",
        std::process::id(),
        suffix
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run_fit(dir: &Path, mode: CnnTrainingMode, pool: bool) {
    let h5ad = dir.join("mock_train.h5ad");
    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = h5ad.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.data.train_cell_types = vec!["ct_a".into()];
    if pool {
        cfg.data.sample = Some("sample".into());
        cfg.training.pool_lasso = true;
    }
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_lr_modulators = false;
    cfg.grn.use_tfl_modulators = true;
    cfg.training.score_threshold = -1.0;
    cfg.training.mode = Some(mode);
    cfg.training.epochs = 1;
    cfg.training.genes = Some(vec!["Tgt1".into()]);
    cfg.execution.output_dir = dir.to_string_lossy().into_owned();
    cfg.execution.n_parallel = 1;
    cfg.lasso.l1_reg = 0.0;
    cfg.lasso.group_reg = 0.0;
    cfg.lasso.n_iter = 400;
    cfg.spatial.radius = RADIUS;
    cfg.spatial.spatial_dim = 8;
    if pool {
        cfg.validate_pool_lasso_sample().unwrap();
    }

    let device = NdArrayDevice::Cpu;
    SpatialCellularProgramsEstimator::<Autodiff<NdArray<f32, i32>>, H5>::fit_all_genes(
        cfg.data.adata_path.as_str(),
        None,
        cfg.spatial.radius,
        cfg.spatial.spatial_dim,
        cfg.spatial.contact_distance,
        cfg.grn.tf_ligand_cutoff,
        cfg.grn.max_ligands,
        cfg.grn.use_tf_modulators,
        cfg.grn.use_lr_modulators,
        cfg.grn.use_tfl_modulators,
        cfg.data.layer.as_str(),
        cfg.data.cluster_annot.as_str(),
        &cfg.cnn,
        cfg.training.epochs,
        cfg.training.learning_rate,
        cfg.training.score_threshold,
        cfg.lasso.l1_reg,
        cfg.lasso.group_reg,
        cfg.lasso.n_iter,
        cfg.lasso.tol,
        cfg.resolved_cnn_mode(),
        cfg.training.genes.clone(),
        None,
        cfg.execution.n_parallel,
        cfg.execution.output_dir.as_str(),
        &cfg.model_export,
        None,
        cfg.grn.network_data_dir.as_deref(),
        None,
        false,
        &cfg,
        None,
        false,
        false,
        None,
        None,
        &device,
    )
    .expect("fit_all_genes");
}

fn feather_labels(path: &Path, col: &str) -> Vec<String> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let df = IpcReader::new(f).finish().unwrap();
    let s = df.column(col).unwrap().str().unwrap();
    (0..df.height())
        .map(|i| s.get(i).unwrap().to_string())
        .collect()
}

fn assert_ligand_beta_nonzero(path: &Path) {
    let f = std::fs::File::open(path).unwrap();
    let df = IpcReader::new(f).finish().unwrap();
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let col = names
        .iter()
        .find(|n| n.contains("Lig1"))
        .unwrap_or_else(|| {
            panic!(
                "expected a Lig1 beta column in {names:?} ({})",
                path.display()
            )
        });
    let beta = df.column(col).unwrap().f64().unwrap();
    let max = (0..df.height())
        .filter_map(|i| beta.get(i))
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        max > 1e-4,
        "{col} in {} should be nonzero because ct_b still sends ligand, max={max}",
        path.display()
    );
}

#[test]
fn dropping_senders_zeros_received_ligand_on_receivers() {
    let (xy, _, lig) = slide_xy_and_expr();
    let full = calculate_weighted_ligands(&xy, &lig, RADIUS, 1.0);
    let receiver_rows: Vec<usize> = (0..N_A).collect();
    let xy_a = xy.select(Axis(0), &receiver_rows);
    let lig_a = lig.select(Axis(0), &receiver_rows);
    let dropped = calculate_weighted_ligands(&xy_a, &lig_a, RADIUS, 1.0);
    assert!(
        full[[0, 0]] > 0.05,
        "receiver next to the sender should see ligand, got {}",
        full[[0, 0]]
    );
    assert!(
        full[[N_A - 1, 0]] < full[[0, 0]],
        "farther receiver should see less ligand"
    );
    assert!(
        dropped.iter().all(|v| v.abs() < 1e-12),
        "without ct_b the field on ct_a is zero: {dropped}"
    );
}

#[test]
fn seed_feather_omits_untrained_type_but_keeps_its_ligand() {
    let dir = temp_dir("seed");
    write_grn(&dir).unwrap();
    let (xy, mat, _) = slide_xy_and_expr();
    write_slide(&dir.join("mock_train.h5ad"), &xy, &mat, None).unwrap();
    run_fit(&dir, CnnTrainingMode::Seed, false);

    let feather = dir.join("Tgt1_betadata.feather");
    let labels = feather_labels(&feather, "Cluster");
    assert_eq!(labels, vec!["ct_a".to_string()]);
    assert_ligand_beta_nonzero(&feather);

    let obs: Vec<String> = (0..N_A + N_B).map(|i| format!("c{i}")).collect();
    let parsed =
        spacetravlr::perturb_mode::parse_obs_columns_csv(&dir.join("cells.csv"), &obs).unwrap();
    assert_eq!(parsed.indices_for_column("ct_a").unwrap().len(), N_A);
    assert_eq!(parsed.indices_for_column("ct_b").unwrap().len(), N_B);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn omitted_allow_list_still_writes_every_cluster() {
    let dir = temp_dir("all");
    write_grn(&dir).unwrap();
    let (xy, mat, _) = slide_xy_and_expr();
    write_slide(&dir.join("mock_train.h5ad"), &xy, &mat, None).unwrap();
    let h5ad = dir.join("mock_train.h5ad");
    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = h5ad.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_lr_modulators = false;
    cfg.training.score_threshold = -1.0;
    cfg.training.mode = Some(CnnTrainingMode::Seed);
    cfg.training.genes = Some(vec!["Tgt1".into()]);
    cfg.execution.output_dir = dir.to_string_lossy().into_owned();
    cfg.execution.n_parallel = 1;
    cfg.lasso.l1_reg = 0.0;
    cfg.lasso.group_reg = 0.0;
    cfg.lasso.n_iter = 200;
    cfg.spatial.radius = RADIUS;
    let device = NdArrayDevice::Cpu;
    SpatialCellularProgramsEstimator::<Autodiff<NdArray<f32, i32>>, H5>::fit_all_genes(
        cfg.data.adata_path.as_str(),
        None,
        cfg.spatial.radius,
        cfg.spatial.spatial_dim,
        cfg.spatial.contact_distance,
        cfg.grn.tf_ligand_cutoff,
        cfg.grn.max_ligands,
        cfg.grn.use_tf_modulators,
        cfg.grn.use_lr_modulators,
        cfg.grn.use_tfl_modulators,
        cfg.data.layer.as_str(),
        cfg.data.cluster_annot.as_str(),
        &cfg.cnn,
        cfg.training.epochs,
        cfg.training.learning_rate,
        cfg.training.score_threshold,
        cfg.lasso.l1_reg,
        cfg.lasso.group_reg,
        cfg.lasso.n_iter,
        cfg.lasso.tol,
        cfg.resolved_cnn_mode(),
        cfg.training.genes.clone(),
        None,
        1,
        cfg.execution.output_dir.as_str(),
        &cfg.model_export,
        None,
        cfg.grn.network_data_dir.as_deref(),
        None,
        false,
        &cfg,
        None,
        false,
        false,
        None,
        None,
        &device,
    )
    .expect("fit all clusters");
    let mut labels = feather_labels(&dir.join("Tgt1_betadata.feather"), "Cluster");
    labels.sort();
    assert_eq!(labels, vec!["ct_a".to_string(), "ct_b".to_string()]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn full_cnn_feather_lists_only_trained_cells() {
    let dir = temp_dir("full");
    write_grn(&dir).unwrap();
    let (xy, mat, _) = slide_xy_and_expr();
    write_slide(&dir.join("mock_train.h5ad"), &xy, &mat, None).unwrap();
    run_fit(&dir, CnnTrainingMode::Full, false);
    let mut ids = feather_labels(&dir.join("Tgt1_betadata.feather"), "CellID");
    ids.sort();
    let expected: Vec<String> = (0..N_A).map(|i| format!("c{i}")).collect();
    assert_eq!(ids, expected);
    assert_ligand_beta_nonzero(&dir.join("Tgt1_betadata.feather"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pool_lasso_each_sample_omits_untrained_type() {
    let dir = temp_dir("pool");
    write_grn(&dir).unwrap();
    let (xy, mat, _) = slide_xy_and_expr();
    write_two_slides(&dir.join("mock_train.h5ad"), &xy, &mat).unwrap();
    run_fit(&dir, CnnTrainingMode::Seed, true);
    for sample in ["s1", "s2"] {
        let feather = dir
            .join("conditions")
            .join(sample)
            .join("Tgt1_betadata.feather");
        assert_eq!(
            feather_labels(&feather, "Cluster"),
            vec!["ct_a".to_string()]
        );
        assert_ligand_beta_nonzero(&feather);
    }
    let obs: Vec<String> = ["s1", "s2"]
        .into_iter()
        .flat_map(|s| (0..N_A + N_B).map(move |i| format!("{s}_c{i}")))
        .collect();
    let parsed =
        spacetravlr::perturb_mode::parse_obs_columns_csv(&dir.join("cells.csv"), &obs).unwrap();
    assert_eq!(parsed.indices_for_column("ct_b").unwrap().len(), N_B * 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unknown_train_cell_type_errors() {
    let dir = temp_dir("missing");
    write_grn(&dir).unwrap();
    let (xy, mat, _) = slide_xy_and_expr();
    write_slide(&dir.join("mock_train.h5ad"), &xy, &mat, None).unwrap();
    let h5ad = dir.join("mock_train.h5ad");
    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = h5ad.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.data.train_cell_types = vec!["ct_a".into(), "not_a_type".into()];
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_lr_modulators = false;
    cfg.training.genes = Some(vec!["Tgt1".into()]);
    cfg.execution.output_dir = dir.to_string_lossy().into_owned();
    cfg.execution.n_parallel = 1;
    cfg.spatial.radius = RADIUS;
    let device = NdArrayDevice::Cpu;
    let err = SpatialCellularProgramsEstimator::<Autodiff<NdArray<f32, i32>>, H5>::fit_all_genes(
        cfg.data.adata_path.as_str(),
        None,
        cfg.spatial.radius,
        cfg.spatial.spatial_dim,
        cfg.spatial.contact_distance,
        cfg.grn.tf_ligand_cutoff,
        cfg.grn.max_ligands,
        false,
        false,
        false,
        cfg.data.layer.as_str(),
        cfg.data.cluster_annot.as_str(),
        &cfg.cnn,
        1,
        cfg.training.learning_rate,
        cfg.training.score_threshold,
        cfg.lasso.l1_reg,
        cfg.lasso.group_reg,
        20,
        cfg.lasso.tol,
        CnnTrainingMode::Seed,
        cfg.training.genes.clone(),
        None,
        1,
        cfg.execution.output_dir.as_str(),
        &cfg.model_export,
        None,
        cfg.grn.network_data_dir.as_deref(),
        None,
        false,
        &cfg,
        None,
        false,
        false,
        None,
        None,
        &device,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("not_a_type"), "{err}");
    assert!(err.contains("ct_b"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pool_sample_with_no_trained_cells_errors() {
    let dir = temp_dir("pool_empty");
    write_grn(&dir).unwrap();
    let (xy, mat, _) = slide_xy_and_expr();
    let n = xy.nrows();
    let path = dir.join("mock_train.h5ad");
    let a = AnnData::<H5>::new(&path).unwrap();
    let mut names = Vec::new();
    let mut types = Vec::new();
    let mut samples = Vec::new();
    let mut stacked = Array2::<f64>::zeros((n + N_B, mat.ncols()));
    let mut xy_all = Array2::<f64>::zeros((n + N_B, 2));
    for i in 0..n {
        names.push(format!("s1_c{i}"));
        types.push(if i < N_A { "ct_a" } else { "ct_b" }.to_string());
        samples.push("s1".to_string());
        stacked.row_mut(i).assign(&mat.row(i));
        xy_all.row_mut(i).assign(&xy.row(i));
    }
    for i in 0..N_B {
        names.push(format!("s2_c{i}"));
        types.push("ct_b".to_string());
        samples.push("s2".to_string());
        stacked.row_mut(n + i).assign(&mat.row(N_A));
        xy_all.row_mut(n + i).assign(&xy.row(N_A));
    }
    a.set_obs_names(names.into()).unwrap();
    a.set_var_names(vec!["Reg1".into(), "Tgt1".into(), "Lig1".into(), "Rec1".into()].into())
        .unwrap();
    a.set_obs(
        DataFrame::new(vec![
            Series::new("cell_type".into(), types).into(),
            Series::new("sample".into(), samples).into(),
        ])
        .unwrap(),
    )
    .unwrap();
    a.set_var(
        DataFrame::new(vec![
            Series::new("gene_ids".into(), vec!["r", "t", "l", "c"]).into(),
        ])
        .unwrap(),
    )
    .unwrap();
    a.set_x(ArrayData::from(dense_to_csr_f64(&stacked).unwrap()))
        .unwrap();
    a.set_obsm([("spatial".to_string(), ArrayData::from(xy_all))])
        .unwrap();
    a.close().unwrap();

    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = path.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.data.sample = Some("sample".into());
    cfg.data.train_cell_types = vec!["ct_a".into()];
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_lr_modulators = false;
    cfg.training.pool_lasso = true;
    cfg.training.mode = Some(CnnTrainingMode::Seed);
    cfg.training.genes = Some(vec!["Tgt1".into()]);
    cfg.training.score_threshold = -1.0;
    cfg.execution.output_dir = dir.to_string_lossy().into_owned();
    cfg.execution.n_parallel = 1;
    cfg.lasso.n_iter = 30;
    cfg.spatial.radius = RADIUS;
    cfg.validate_pool_lasso_sample().unwrap();
    let device = NdArrayDevice::Cpu;
    let err = SpatialCellularProgramsEstimator::<Autodiff<NdArray<f32, i32>>, H5>::fit_all_genes(
        cfg.data.adata_path.as_str(),
        None,
        cfg.spatial.radius,
        cfg.spatial.spatial_dim,
        cfg.spatial.contact_distance,
        cfg.grn.tf_ligand_cutoff,
        cfg.grn.max_ligands,
        cfg.grn.use_tf_modulators,
        false,
        cfg.grn.use_tfl_modulators,
        cfg.data.layer.as_str(),
        cfg.data.cluster_annot.as_str(),
        &cfg.cnn,
        1,
        cfg.training.learning_rate,
        cfg.training.score_threshold,
        0.0,
        0.0,
        cfg.lasso.n_iter,
        cfg.lasso.tol,
        CnnTrainingMode::Seed,
        cfg.training.genes.clone(),
        None,
        1,
        cfg.execution.output_dir.as_str(),
        &cfg.model_export,
        None,
        cfg.grn.network_data_dir.as_deref(),
        None,
        false,
        &cfg,
        None,
        false,
        false,
        None,
        None,
        &device,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("s2") || err.contains("sample"),
        "expected an empty-sample error, got {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
