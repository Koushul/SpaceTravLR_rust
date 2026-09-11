//! Seed-mode training writes `beta_Cd9&Cd81` only when `[grn].tetraspanin_pairs` is set.

use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp};
use anndata_hdf5::H5;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn_autodiff::Autodiff;
use ndarray::Array2;
use polars::prelude::{DataFrame, IpcReader, NamedFrom, ParquetWriter, SerReader, Series};
use spacetravlr::config::{CnnTrainingMode, SpaceshipConfig};
use spacetravlr::spatial_estimator::{SpatialCellularProgramsEstimator, dense_to_csr_f64};
use std::path::{Path, PathBuf};

fn write_minimal_mouse_grn_parquet(dir: &Path) -> anyhow::Result<()> {
    let mut df = DataFrame::new(vec![
        Series::new("source".into(), vec!["Cd9", "Tgt1"]).into(),
        Series::new("target".into(), vec!["Tgt1", "Cd9"]).into(),
        Series::new("edge_type".into(), vec!["grn", "grn"]).into(),
        Series::new("weight".into(), vec![1.0_f64, 1.0]).into(),
    ])?;
    let path = dir.join("mouse_network.parquet");
    let f = std::fs::File::create(&path)?;
    ParquetWriter::new(f).finish(&mut df)?;
    Ok(())
}

fn write_mock_h5ad(path: &Path) -> anyhow::Result<()> {
    let a = AnnData::<H5>::new(path)?;
    let n_obs = 12usize;
    let obs_names: Vec<String> = (0..n_obs).map(|i| format!("c{i}")).collect();
    a.set_obs_names(obs_names.into())?;
    a.set_var_names(vec!["Cd9".into(), "Cd81".into(), "Tgt1".into()].into())?;

    let cell_types: Vec<String> = (0..n_obs)
        .map(|i| if i < n_obs / 2 { "ct_a" } else { "ct_b" }.to_string())
        .collect();
    let obs = DataFrame::new(vec![Series::new("cell_type".into(), cell_types).into()])?;
    a.set_obs(obs)?;

    let var = DataFrame::new(vec![
        Series::new("gene_ids".into(), vec!["cd9", "cd81", "tgt"]).into(),
    ])?;
    a.set_var(var)?;

    let mut mat = Array2::<f64>::zeros((n_obs, 3));
    for i in 0..n_obs {
        let cl = if i < n_obs / 2 { 0.0 } else { 1.0 };
        let cd9 = 0.4 + cl * 0.5 + (i as f64) * 0.03;
        let cd81 = 0.5 + cl * 0.4 + (i as f64) * 0.02;
        mat[[i, 0]] = cd9;
        mat[[i, 1]] = cd81;
        mat[[i, 2]] = cd9 * cd81 + 0.05 * cl;
    }
    let csr = dense_to_csr_f64(&mat)?;
    a.set_x(ArrayData::from(csr))?;

    let xy = Array2::from_shape_fn((n_obs, 2), |(i, j)| {
        if j == 0 {
            (i % 4) as f64
        } else {
            (i / 4) as f64
        }
    });
    a.set_obsm([("spatial".to_string(), ArrayData::from(xy))])?;
    a.close()?;
    Ok(())
}

fn setup_run_dir(suffix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_tspan_cis_{}_{}",
        std::process::id(),
        suffix
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_minimal_mouse_grn_parquet(&dir).unwrap();
    let h5ad = dir.join("mock_train.h5ad");
    write_mock_h5ad(&h5ad).unwrap();
    dir
}

fn run_fit(dir: &Path, tetraspanin_pairs: Vec<String>) {
    let h5ad = dir.join("mock_train.h5ad");
    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = h5ad.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_tf_modulators = false;
    cfg.grn.use_lr_modulators = false;
    cfg.grn.use_tfl_modulators = false;
    cfg.grn.tetraspanin_pairs = tetraspanin_pairs;
    cfg.training.score_threshold = -1.0;
    cfg.training.mode = Some(CnnTrainingMode::Seed);
    cfg.execution.output_dir = dir.to_string_lossy().into_owned();
    cfg.lasso.n_iter = 200;
    cfg.execution.n_parallel = 1;

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
        Some(vec!["Tgt1".into()]),
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

fn feather_colnames(dir: &Path) -> Vec<String> {
    let path = dir.join("Tgt1_betadata.feather");
    assert!(path.is_file(), "expected {}", path.display());
    let f = std::fs::File::open(&path).unwrap();
    let df = IpcReader::new(f).finish().unwrap();
    df.get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

#[test]
fn training_writes_tetraspanin_beta_column() {
    let dir = setup_run_dir("on");
    run_fit(&dir, vec!["Cd81&Cd9".into()]);
    let cols = feather_colnames(&dir);
    assert!(
        cols.iter().any(|c| c == "beta_Cd9&Cd81"),
        "columns: {cols:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn training_omits_tetraspanin_column_when_list_empty() {
    let dir = setup_run_dir("off");
    run_fit(&dir, vec![]);
    assert!(
        dir.join("Tgt1.orphan").is_file() || !dir.join("Tgt1_betadata.feather").is_file(),
        "empty tetraspanin_pairs with TF/LR/TFL off should not produce a cis feather column"
    );
    if dir.join("Tgt1_betadata.feather").is_file() {
        let cols = feather_colnames(&dir);
        assert!(
            !cols.iter().any(|c| c.contains('&')),
            "unexpected cis column in {cols:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn write_two_sample_tspan_h5ad(path: &Path) -> anyhow::Result<()> {
    let a = AnnData::<H5>::new(path)?;
    let n_per = 8usize;
    let n_obs = n_per * 2;
    let obs_names: Vec<String> = (0..n_obs)
        .map(|i| {
            if i < n_per {
                format!("s1_c{i}")
            } else {
                format!("s2_c{}", i - n_per)
            }
        })
        .collect();
    a.set_obs_names(obs_names.into())?;
    a.set_var_names(vec!["Cd9".into(), "Cd81".into(), "Tgt1".into()].into())?;

    let mut cell_types = Vec::with_capacity(n_obs);
    let mut samples = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let local = i % n_per;
        cell_types.push(if local < n_per / 2 { "ct_a" } else { "ct_b" }.to_string());
        samples.push(if i < n_per { "s1" } else { "s2" }.to_string());
    }
    let obs = DataFrame::new(vec![
        Series::new("cell_type".into(), cell_types).into(),
        Series::new("sample".into(), samples).into(),
    ])?;
    a.set_obs(obs)?;
    a.set_var(DataFrame::new(vec![
        Series::new("gene_ids".into(), vec!["cd9", "cd81", "tgt"]).into(),
    ])?)?;

    let mut mat = Array2::<f64>::zeros((n_obs, 3));
    for i in 0..n_obs {
        let local = (i % n_per) as f64;
        let cl = if (i % n_per) < n_per / 2 { 0.0 } else { 1.0 };
        let sample_off = if i < n_per { 0.0 } else { 0.12 };
        let cd9 = 0.4 + cl * 0.5 + local * 0.03;
        let cd81 = 0.5 + cl * 0.4 + local * 0.02 + sample_off;
        mat[[i, 0]] = cd9;
        mat[[i, 1]] = cd81;
        mat[[i, 2]] = cd9 * cd81 + 0.05 * cl;
    }
    a.set_x(ArrayData::from(dense_to_csr_f64(&mat)?))?;
    let xy = Array2::from_shape_fn((n_obs, 2), |(i, j)| {
        let local = i % n_per;
        if j == 0 {
            (local % 4) as f64
        } else {
            (local / 4) as f64
        }
    });
    a.set_obsm([("spatial".to_string(), ArrayData::from(xy))])?;
    a.close()?;
    Ok(())
}

fn run_pool_fit(dir: &Path, mode: CnnTrainingMode) {
    let h5ad = dir.join("mock_train.h5ad");
    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = h5ad.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.data.sample = Some("sample".into());
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_tf_modulators = false;
    cfg.grn.use_lr_modulators = false;
    cfg.grn.use_tfl_modulators = false;
    cfg.grn.tetraspanin_pairs = vec!["Cd81&Cd9".into()];
    cfg.training.score_threshold = -1.0;
    cfg.training.mode = Some(mode);
    cfg.training.pool_lasso = true;
    cfg.training.epochs = 2;
    cfg.execution.output_dir = dir.to_string_lossy().into_owned();
    cfg.lasso.n_iter = 80;
    cfg.execution.n_parallel = 1;
    cfg.spatial.spatial_dim = 8;
    cfg.validate_pool_lasso_sample().unwrap();

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
        Some(vec!["Tgt1".into()]),
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
    .expect("fit_all_genes pool_lasso tetraspanin");
}

#[test]
fn pool_lasso_full_cnn_tetraspanin_writes_feather_and_parent_log() {
    let dir = setup_run_dir("pool_full");
    write_two_sample_tspan_h5ad(&dir.join("mock_train.h5ad")).unwrap();
    run_pool_fit(&dir, CnnTrainingMode::Full);

    let parent_log = dir.join("log").join("Tgt1.log");
    assert!(
        parent_log.is_file(),
        "expected parent training log {}",
        parent_log.display()
    );
    for sample in ["s1", "s2"] {
        let sample_dir = dir.join("conditions").join(sample);
        let feather = sample_dir.join("Tgt1_betadata.feather");
        assert!(
            feather.is_file(),
            "tetraspanin pool-lasso should write {}",
            feather.display()
        );
        let df = IpcReader::new(std::fs::File::open(&feather).unwrap())
            .finish()
            .unwrap();
        let cols: Vec<String> = df
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        assert!(
            cols.iter().any(|c| c == "beta_Cd9&Cd81"),
            "{sample} columns: {cols:?}"
        );
        assert!(df.column("CellID").is_ok(), "{sample} expected CellID");
        assert!(
            !sample_dir.join("Tgt1.orphan").is_file(),
            "{sample} should not be orphan"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
