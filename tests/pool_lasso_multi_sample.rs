//! Pooled-lasso multi-sample training: two slides, overlapping coordinates,
//! joint Lasso, per-sample CNN / seed export under `conditions/<sample>/`.

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

fn write_two_sample_h5ad(path: &Path) -> anyhow::Result<()> {
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
    a.set_var_names(vec!["Reg1".into(), "Tgt1".into(), "Reg2".into()].into())?;

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
        Series::new("gene_ids".into(), vec!["r1", "t1", "r2"]).into(),
    ])?)?;

    let mut mat = Array2::<f64>::zeros((n_obs, 3));
    for i in 0..n_obs {
        let local = (i % n_per) as f64;
        let cl = if (i % n_per) < n_per / 2 { 0.0 } else { 1.0 };
        let sample_off = if i < n_per { 0.0 } else { 0.15 };
        mat[[i, 0]] = 0.4 + cl * 0.5 + local * 0.02;
        mat[[i, 1]] = 0.6 + cl * 0.3 + local * 0.015 + sample_off;
        mat[[i, 2]] = 0.35 + cl * 0.55 + local * 0.018;
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

fn setup_run_dir(suffix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_pool_lasso_{}_{}",
        std::process::id(),
        suffix
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_minimal_mouse_grn_parquet(&dir).unwrap();
    write_two_sample_h5ad(&dir.join("mock_train.h5ad")).unwrap();
    dir
}

fn run_fit(dir: &Path, mode: CnnTrainingMode, spatial_dim_override: Option<usize>) {
    let h5ad = dir.join("mock_train.h5ad");
    let mut cfg = SpaceshipConfig::default();
    cfg.data.adata_path = h5ad.to_string_lossy().into_owned();
    cfg.data.layer = "X".into();
    cfg.data.cluster_annot = "cell_type".into();
    cfg.data.sample = Some("sample".into());
    cfg.grn.network_data_dir = Some(dir.to_string_lossy().into_owned());
    cfg.grn.use_lr_modulators = false;
    cfg.grn.use_tfl_modulators = false;
    cfg.training.score_threshold = -1.0;
    cfg.training.mode = Some(mode);
    cfg.training.pool_lasso = true;
    cfg.execution.output_dir = dir.to_string_lossy().into_owned();
    cfg.lasso.n_iter = 200;
    cfg.execution.n_parallel = 1;
    if let Some(d) = spatial_dim_override {
        cfg.spatial.spatial_dim = d.max(1);
    }
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
    .expect("fit_all_genes pool_lasso");
}

fn sample_dir(root: &Path, name: &str) -> PathBuf {
    root.join("conditions").join(name)
}

#[test]
fn pool_lasso_seed_writes_identical_cluster_betadata_per_sample() {
    let dir = setup_run_dir("seed");
    run_fit(&dir, CnnTrainingMode::Seed, None);

    for sample in ["s1", "s2"] {
        let d = sample_dir(&dir, sample);
        assert!(d.is_dir(), "missing {}", d.display());
        let label = std::fs::read_to_string(d.join("condition_label.txt")).unwrap();
        assert_eq!(label.trim(), sample);
    }

    let mut n_ok = 0usize;
    for gene in ["Reg1", "Tgt1", "Reg2"] {
        let p1 = sample_dir(&dir, "s1").join(format!("{gene}_betadata.feather"));
        let p2 = sample_dir(&dir, "s2").join(format!("{gene}_betadata.feather"));
        if !p1.is_file() && !p2.is_file() {
            continue;
        }
        n_ok += 1;
        assert!(p1.is_file(), "{}", p1.display());
        assert!(p2.is_file(), "{}", p2.display());
        let df1 = IpcReader::new(std::fs::File::open(&p1).unwrap())
            .finish()
            .unwrap();
        let df2 = IpcReader::new(std::fs::File::open(&p2).unwrap())
            .finish()
            .unwrap();
        assert!(df1.column("Cluster").is_ok());
        assert_eq!(df1.get_column_names(), df2.get_column_names());
        assert_eq!(df1.height(), df2.height());
        for col in df1.get_column_names() {
            if col == "Cluster" {
                continue;
            }
            let a = df1.column(col).unwrap().f64().unwrap();
            let b = df2.column(col).unwrap().f64().unwrap();
            for i in 0..a.len() {
                let x = a.get(i).unwrap_or(0.0);
                let y = b.get(i).unwrap_or(0.0);
                assert!((x - y).abs() < 1e-8, "{gene} {col}[{i}] s1={x} s2={y}");
            }
        }
    }
    assert!(
        n_ok > 0,
        "expected at least one gene with pooled seed betadata"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pool_lasso_full_cnn_identical_lasso_coefs_and_cellid_betadata() {
    let dir = setup_run_dir("full");
    run_fit(&dir, CnnTrainingMode::Full, Some(8));

    for sample in ["s1", "s2"] {
        assert!(
            sample_dir(&dir, sample).is_dir(),
            "missing {}",
            sample_dir(&dir, sample).display()
        );
    }

    let mut n_cellid = 0usize;
    let mut n_lasso_pairs = 0usize;
    for gene in ["Reg1", "Tgt1", "Reg2"] {
        let mut lasso_paths = Vec::new();
        for sample in ["s1", "s2"] {
            let d = sample_dir(&dir, sample);
            let per_cell = d.join(format!("{gene}_betadata.feather"));
            if per_cell.is_file() {
                n_cellid += 1;
                let df = IpcReader::new(std::fs::File::open(&per_cell).unwrap())
                    .finish()
                    .unwrap();
                assert!(
                    df.column("CellID").is_ok(),
                    "{gene} {sample}: expected CellID"
                );
            }
            let lp = d
                .join("lasso_coefs")
                .join(format!("{gene}_lasso_coefs.feather"));
            if lp.is_file() {
                lasso_paths.push(lp);
            }
        }
        if lasso_paths.len() == 2 {
            n_lasso_pairs += 1;
            let df1 = IpcReader::new(std::fs::File::open(&lasso_paths[0]).unwrap())
                .finish()
                .unwrap();
            let df2 = IpcReader::new(std::fs::File::open(&lasso_paths[1]).unwrap())
                .finish()
                .unwrap();
            assert_eq!(df1.height(), df2.height());
            let labels1 = df1.column("cell_type").unwrap().str().unwrap();
            let labels2 = df2.column("cell_type").unwrap().str().unwrap();
            for i in 0..df1.height() {
                assert_eq!(labels1.get(i), labels2.get(i));
            }
            for col in df1.get_column_names() {
                if col == "cell_type" {
                    continue;
                }
                let a = df1.column(col).unwrap().f64().unwrap();
                let b = df2.column(col).unwrap().f64().unwrap();
                for i in 0..a.len() {
                    let x = a.get(i).unwrap_or(0.0);
                    let y = b.get(i).unwrap_or(0.0);
                    assert!(
                        (x - y).abs() < 1e-8,
                        "{gene} lasso {col}[{i}] s1={x} s2={y}"
                    );
                }
            }
        }
    }
    assert!(
        n_cellid > 0,
        "expected at least one per-cell betadata with CellID"
    );
    assert!(
        n_lasso_pairs > 0,
        "expected identical lasso_coefs feathers in both sample dirs"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
