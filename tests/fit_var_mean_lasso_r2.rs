//! End-to-end: tiny `.h5ad` + minimal GRN parquet → `fit_all_genes` → `var['mean_lasso_r2']`
//! finite for every trained gene (no all-NaN regression).

use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn_autodiff::Autodiff;
use ndarray::Array2;
use polars::prelude::{DataFrame, NamedFrom, ParquetWriter, Series};
use spacetravlr::config::SpaceshipConfig;
use spacetravlr::spatial_estimator::{SpatialCellularProgramsEstimator, dense_to_csr_f64};
use std::path::Path;

fn write_minimal_mouse_grn_parquet(dir: &Path) -> anyhow::Result<()> {
    let mut df = DataFrame::new(vec![
        Series::new("source".into(), vec!["Reg1", "Tgt1", "Tgt1", "Reg2"]).into(),
        Series::new("target".into(), vec!["Tgt1", "Reg1", "Reg2", "Tgt1"]).into(),
        Series::new("edge_type".into(), vec!["grn", "grn", "grn", "grn"]).into(),
        Series::new("weight".into(), vec![1.0_f64, 1.0, 1.0, 1.0]).into(),
    ])?;
    let path = dir.join("mouse_network.parquet");
    let f = std::fs::File::create(&path)?;
    ParquetWriter::new(f).finish(&mut df)?;
    Ok(())
}

fn write_mock_training_h5ad(path: &Path) -> anyhow::Result<()> {
    let a = AnnData::<H5>::new(path)?;
    let n_obs = 12usize;
    let obs_names: Vec<String> = (0..n_obs).map(|i| format!("c{i}")).collect();
    a.set_obs_names(obs_names.into())?;
    a.set_var_names(vec!["Reg1".into(), "Tgt1".into(), "Reg2".into()].into())?;

    let cell_types: Vec<String> = (0..n_obs)
        .map(|i| if i < n_obs / 2 { "ct_a" } else { "ct_b" }.to_string())
        .collect();
    let obs = DataFrame::new(vec![Series::new("cell_type".into(), cell_types).into()])?;
    a.set_obs(obs)?;

    let var = DataFrame::new(vec![
        Series::new("gene_ids".into(), vec!["r1", "t1", "r2"]).into(),
    ])?;
    a.set_var(var)?;

    let mut mat = Array2::<f64>::zeros((n_obs, 3));
    for i in 0..n_obs {
        let cl = if i < n_obs / 2 { 0.0 } else { 1.0 };
        mat[[i, 0]] = 0.4 + cl * 0.5 + (i as f64) * 0.02;
        mat[[i, 1]] = 0.6 + cl * 0.3 + (i as f64) * 0.015;
        mat[[i, 2]] = 0.35 + cl * 0.55 + (i as f64) * 0.018;
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

#[test]
fn fit_all_genes_writes_finite_mean_lasso_r2_to_var() {
    let dir = std::env::temp_dir().join(format!("spacetravlr_fit_var_r2_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_minimal_mouse_grn_parquet(&dir).unwrap();

    let h5ad = dir.join("mock_train.h5ad");
    write_mock_training_h5ad(&h5ad).unwrap();

    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = h5ad.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_lr_modulators = false;
    cfg.grn.use_tfl_modulators = false;
    cfg.training.score_threshold = -1.0;
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
        false,
        &cfg.training.hybrid,
        cfg.min_mean_lasso_r2_for_hybrid_cnn(),
        None,
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
        &device,
    )
    .expect("fit_all_genes");

    let a = AnnData::<H5>::open(H5::open(&h5ad).unwrap()).unwrap();
    let v = a.read_var().unwrap();
    let col = v.column("mean_lasso_r2").expect("mean_lasso_r2 column");
    let r2 = col.f64().unwrap();
    assert_eq!(r2.len(), 3, "one slot per gene");
    for (i, name) in ["Reg1", "Tgt1", "Reg2"].iter().enumerate() {
        let x = r2.get(i).unwrap();
        assert!(
            x.is_finite(),
            "gene {name} mean_lasso_r2 should be finite, got {x}"
        );
        assert!(x >= 0.0, "gene {name} mean_lasso_r2 unexpected {x}");
    }
    a.close().unwrap();

    let _ = std::fs::remove_dir_all(&dir);
}
