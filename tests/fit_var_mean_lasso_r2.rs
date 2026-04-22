//! End-to-end: tiny `.h5ad` + minimal GRN parquet → `fit_all_genes` →
//! `spacetravlr_gene_performance.feather` with finite `mean_lasso_r2` per trained gene.

use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp};
use anndata_hdf5::H5;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn_autodiff::Autodiff;
use ndarray::Array2;
use polars::prelude::{
    DataFrame, IpcReader, NamedFrom, ParquetWriter, SerReader, Series,
};
use spacetravlr::config::{CnnTrainingMode, SpaceshipConfig};
use spacetravlr::spatial_estimator::{
    SpatialCellularProgramsEstimator, dense_to_csr_f64, GENE_PERFORMANCE_FEATHER_NAME,
};
use std::path::{Path, PathBuf};

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

fn setup_run_dir(suffix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_fit_var_r2_{}_{}",
        std::process::id(),
        suffix
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_minimal_mouse_grn_parquet(&dir).unwrap();
    let h5ad = dir.join("mock_train.h5ad");
    write_mock_training_h5ad(&h5ad).unwrap();
    dir
}

fn run_fit_all_genes(dir: &Path, mode: CnnTrainingMode, spatial_dim_override: Option<usize>) {
    let h5ad = dir.join("mock_train.h5ad");
    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = h5ad.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_lr_modulators = false;
    cfg.grn.use_tfl_modulators = false;
    cfg.training.score_threshold = -1.0;
    cfg.training.mode = Some(mode);
    cfg.execution.output_dir = dir.to_string_lossy().into_owned();
    cfg.lasso.n_iter = 200;
    cfg.execution.n_parallel = 1;
    if let Some(d) = spatial_dim_override {
        cfg.spatial.spatial_dim = d.max(1);
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
}

fn assert_gene_performance_feather(dir: &Path) {
    let perf_path = dir.join(GENE_PERFORMANCE_FEATHER_NAME);
    assert!(
        perf_path.is_file(),
        "expected {}",
        perf_path.display()
    );
    let f = std::fs::File::open(&perf_path).unwrap();
    let df = IpcReader::new(f).finish().unwrap();
    let genes = df.column("gene").unwrap().str().unwrap();
    let r2 = df.column("mean_lasso_r2").unwrap().f64().unwrap();
    assert_eq!(df.height(), 3, "one row per var gene");
    for name in ["Reg1", "Tgt1", "Reg2"] {
        let mut found = false;
        for i in 0..df.height() {
            if genes.get(i) == Some(name) {
                let x = r2.get(i).unwrap();
                assert!(
                    x.is_finite(),
                    "gene {name} mean_lasso_r2 should be finite, got {x}"
                );
                assert!(x >= 0.0, "gene {name} mean_lasso_r2 unexpected {x}");
                found = true;
                break;
            }
        }
        assert!(found, "missing gene {name} in feather");
    }
}

#[test]
fn fit_all_genes_writes_finite_mean_lasso_r2_to_gene_performance_feather() {
    let dir = setup_run_dir("seed");
    run_fit_all_genes(&dir, CnnTrainingMode::Seed, None);
    assert_gene_performance_feather(&dir);

    let lasso_dir = dir.join("lasso_coefs");
    assert!(
        !lasso_dir.exists(),
        "seed-only mode must NOT create {}",
        lasso_dir.display()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fit_all_genes_full_cnn_writes_lasso_coefs_per_cluster_feathers() {
    let dir = setup_run_dir("full");
    // Default `spatial_dim` is 32 (CNN grid H=W). Use a tiny grid here to keep full-CNN CPU tests fast.
    run_fit_all_genes(&dir, CnnTrainingMode::Full, Some(8));
    assert_gene_performance_feather(&dir);

    let lasso_dir = dir.join("lasso_coefs");
    assert!(
        lasso_dir.is_dir(),
        "full-CNN mode must create {}",
        lasso_dir.display()
    );

    for gene in ["Reg1", "Tgt1", "Reg2"] {
        let per_cell = dir.join(format!("{gene}_betadata.feather"));
        if !per_cell.is_file() {
            continue;
        }
        let cell_df = IpcReader::new(std::fs::File::open(&per_cell).unwrap())
            .finish()
            .unwrap();
        assert!(
            cell_df.column("CellID").is_ok(),
            "{gene}: per-cell betadata uses CellID id column"
        );

        let p = lasso_dir.join(format!("{gene}_lasso_coefs.feather"));
        assert!(p.is_file(), "expected {}", p.display());
        let df = IpcReader::new(std::fs::File::open(&p).unwrap())
            .finish()
            .unwrap();
        let label = df.column("cell_type").expect("cell_type column");
        let labels: Vec<String> = label
            .str()
            .unwrap()
            .into_iter()
            .map(|o| o.unwrap_or("").to_string())
            .collect();
        assert!(
            labels.iter().any(|s| s == "ct_a") && labels.iter().any(|s| s == "ct_b"),
            "{gene}: cell_type rows should be the cluster annotation labels, got {:?}",
            labels
        );
        assert!(df.column("beta0").is_ok(), "{gene}: beta0 column");
        let mut has_mod = false;
        for col in df.get_column_names() {
            if col.starts_with("beta_") {
                has_mod = true;
                break;
            }
        }
        assert!(has_mod, "{gene}: at least one beta_<modulator> column");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
