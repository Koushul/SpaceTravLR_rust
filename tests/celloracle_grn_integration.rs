//! Minimal GRN parquet + synthetic expression → [`spacetravlr::celloracle::infer_grn_whole`] runs and yields edges.

use ndarray::Array2;
use spacetravlr::celloracle::{infer_grn_whole, scale_gem_no_center};
use polars::prelude::{DataFrame, NamedFrom, ParquetWriter, Series};
use spacetravlr::network::GeneNetwork;

fn write_minimal_mouse_grn_parquet(dir: &std::path::Path) -> anyhow::Result<()> {
    let mut df = DataFrame::new(vec![
        Series::new(
            "source".into(),
            vec!["Reg1", "Tgt1", "Tgt1", "Reg2"],
        )
        .into(),
        Series::new(
            "target".into(),
            vec!["Tgt1", "Reg1", "Reg2", "Tgt1"],
        )
        .into(),
        Series::new(
            "edge_type".into(),
            vec!["grn", "grn", "grn", "grn"],
        )
        .into(),
        Series::new("weight".into(), vec![1.0_f64, 1.0, 1.0, 1.0]).into(),
    ])?;
    let path = dir.join("mouse_network.parquet");
    let f = std::fs::File::create(&path)?;
    ParquetWriter::new(f).finish(&mut df)?;
    Ok(())
}

#[test]
fn infer_grn_whole_runs_on_synthetic_gem() {
    let dir = std::env::temp_dir().join(format!(
        "celloracle_grn_int_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_minimal_mouse_grn_parquet(&dir).unwrap();

    let var_names = vec![
        "Reg1".to_string(),
        "Tgt1".to_string(),
        "Reg2".to_string(),
    ];
    let net = GeneNetwork::new(
        "mouse",
        &var_names,
        Some(dir.to_str().unwrap()),
    )
    .expect("GeneNetwork");
    let tf_by_target = net.grn_regulators_by_target().expect("grn map");
    assert!(
        tf_by_target.contains_key("Tgt1"),
        "expected Tgt1 in GRN priors"
    );

    let n_obs = 40usize;
    let mut gem = Array2::<f64>::zeros((n_obs, 3));
    for i in 0..n_obs {
        let t = i as f64 * 0.1;
        gem[[i, 0]] = 0.5 + t.sin();
        gem[[i, 2]] = 0.3 + t.cos();
        gem[[i, 1]] = 0.2 * gem[[i, 0]] + 0.3 * gem[[i, 2]] + 0.01 * (i as f64);
    }

    let gem_scaled = scale_gem_no_center(&gem);
    let links =
        infer_grn_whole(&gem, &gem_scaled, &var_names, &tf_by_target, true, None).expect("infer");

    assert!(
        !links.is_empty(),
        "expected at least one GRN link; got {}",
        links.len()
    );

    let _ = std::fs::remove_dir_all(&dir);
}
