use std::path::PathBuf;

use ndarray::Array2;
use ndarray_npy::read_npy;
use spacetravlr::magic::{
    Csr, MagicGraphParams, build_magic_kernel_graphtools_style, diffusion_operator_from_affinity,
    impute_markov, magic_impute_from_embedding,
};

fn fixture_dir() -> PathBuf {
    std::env::var("MAGIC_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/magic_kidney")
        })
}

fn csr_frobenius_diff(a: &Csr, b: &Csr) -> f64 {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let n = a.nrows;
    let mut dense_a = vec![0.0_f64; n * n];
    let mut dense_b = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in a.indptr[i]..a.indptr[i + 1] {
            dense_a[i * n + a.indices[k]] += a.data[k];
        }
        for k in b.indptr[i]..b.indptr[i + 1] {
            dense_b[i * n + b.indices[k]] += b.data[k];
        }
    }
    dense_a
        .iter()
        .zip(dense_b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn kidney_diffusion_matches_python_export() {
    let dir = fixture_dir();
    let data_path = dir.join("data_nu.npy");
    if !data_path.exists() {
        eprintln!("skip: run Python export to create {}", data_path.display());
        return;
    }

    let data_nu: Array2<f64> = read_npy(data_path).expect("data_nu.npy");
    let params = MagicGraphParams::default();
    let k = build_magic_kernel_graphtools_style(data_nu.view(), &params).expect("kernel");
    let p_rust = diffusion_operator_from_affinity(&k);

    let p_py = Csr::from_scipy_npz(&dir.join("P_csr.npz")).expect("P_csr.npz");
    let fd = csr_frobenius_diff(&p_rust, &p_py);
    assert!(
        fd < 1e-6,
        "diffusion operator Frobenius diff {fd} (Rust vs Python export)"
    );
}

#[test]
fn kidney_imputation_matches_python_export() {
    let dir = fixture_dir();
    let golden_p = dir.join("P_csr.npz");
    let x_path = dir.join("X_input_first50cols.npy");
    let y_path = dir.join("X_magic_golden.npy");
    if !golden_p.exists() || !x_path.exists() || !y_path.exists() {
        eprintln!("skip magic imputation parity: missing fixtures in {}", dir.display());
        return;
    }

    let p = Csr::from_scipy_npz(&golden_p).expect("P");
    let x: Array2<f64> = read_npy(x_path).expect("X");
    let y_golden: Array2<f64> = read_npy(y_path).expect("Y");

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let t = meta["t"].as_u64().unwrap() as usize;

    let y = impute_markov(&p, x.view(), t);
    let diff = (&y - &y_golden)
        .mapv(f64::abs)
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    assert!(
        diff < 1e-9,
        "imputed matrix max abs diff {diff} vs Python MAGIC"
    );
}

#[test]
fn kidney_impute_from_python_embedding_matches_golden() {
    let dir = fixture_dir();
    let data_path = dir.join("data_nu.npy");
    let x_path = dir.join("X_input_first50cols.npy");
    let y_path = dir.join("X_magic_golden.npy");
    if !data_path.exists() || !x_path.exists() || !y_path.exists() {
        eprintln!(
            "skip embedding-chain parity: missing fixtures in {}",
            dir.display()
        );
        return;
    }

    let data_nu: Array2<f64> = read_npy(data_path).expect("data_nu.npy");
    let x: Array2<f64> = read_npy(x_path).expect("X");
    let y_golden: Array2<f64> = read_npy(y_path).expect("Y");

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let t = meta["t"].as_u64().unwrap() as usize;

    let y = magic_impute_from_embedding(
        data_nu.view(),
        x.view(),
        t,
        &MagicGraphParams::default(),
    )
    .expect("magic_impute_from_embedding");
    let diff = (&y - &y_golden)
        .mapv(f64::abs)
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    assert!(
        diff < 1e-6,
        "imputed matrix max abs diff {diff} vs Python MAGIC (Rust graph + Markov)"
    );
}
