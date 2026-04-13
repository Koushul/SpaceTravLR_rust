//! **R spacexr ↔ Rust RCTD numerical parity.**
//!
//! Loads binary fixtures produced by `tests/r_parity/export_r_parity_fixtures.R`,
//! re-runs RCTD full/doublet/multi in Rust on the *same* aligned data, and asserts
//! that the results agree within documented tolerances.
//!
//! ## Known algorithmic differences
//!
//! The Rust IRWLS solver uses a coordinate-descent QP (non-negative least squares)
//! whereas R spacexr uses `quadprog::solve.QP` (active-set QP). The convergence
//! criterion also differs: Rust uses L1 change, R uses L2 norm. These lead to
//! ~1-2% max absolute weight difference while preserving the same decomposition
//! (Pearson r > 0.999 per pixel).
//!
//! ## Regenerate fixtures
//!
//! Requires R + spacexr:
//! ```sh
//! Rscript crates/rctd-core/tests/r_parity/export_r_parity_fixtures.R \
//!         crates/rctd-core/tests/r_parity/fixtures
//! ```
//!
//! Run tests:
//! ```sh
//! cargo test -p rctd-core --test parity_r_spacexr -- --ignored
//! ```

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use rctd_core::likelihood_tables::compute_spline_coefficients;
use rctd_core::{device_cpu, run_doublet_mode, run_full_mode, run_multi_mode, RctdConfig};
use serde::Deserialize;

fn fixture_dir() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/r_parity/fixtures"
    ))
}

fn have_fixtures() -> bool {
    fixture_dir().join("meta.json").is_file()
}

fn missing_fixture_msg() -> String {
    format!(
        "R parity fixtures not found at {}.\n\
         Generate with:\n  \
         Rscript crates/rctd-core/tests/r_parity/export_r_parity_fixtures.R \
         crates/rctd-core/tests/r_parity/fixtures",
        fixture_dir().display()
    )
}

#[derive(Debug, Deserialize)]
struct FixtureMeta {
    n_pixels: usize,
    n_genes: usize,
    n_types: usize,
    q_nrows: usize,
    q_ncols: usize,
    n_xvals: usize,
    sigma: i32,
}

fn load_meta() -> FixtureMeta {
    let p = fixture_dir().join("meta.json");
    let s = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn read_f64_bin(path: &Path, n: usize) -> Vec<f64> {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut buf = vec![0u8; n * 8];
    f.read_exact(&mut buf)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    buf.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn load_matrix(name: &str, nrows: usize, ncols: usize) -> Array2<f64> {
    let v = read_f64_bin(&fixture_dir().join(name), nrows * ncols);
    Array2::from_shape_vec((nrows, ncols), v)
        .unwrap_or_else(|e| panic!("shape {name} ({nrows}×{ncols}): {e}"))
}

fn load_vector(name: &str, n: usize) -> Array1<f64> {
    let v = read_f64_bin(&fixture_dir().join(name), n);
    Array1::from_vec(v)
}

fn load_lines(name: &str) -> Vec<String> {
    let p = fixture_dir().join(name);
    let s = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    s.lines()
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect()
}

fn k_names(k: usize) -> Vec<String> {
    (0..k).map(|i| format!("t{i}")).collect()
}

// ---- Comparison metrics ----

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "dimension mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn mean_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    let n = a.len() as f64;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
        / n.max(1.0)
}

fn pearson_r_rows(a: &Array2<f64>, b: &Array2<f64>) -> Vec<f64> {
    assert_eq!(a.dim(), b.dim());
    let n = a.nrows();
    (0..n)
        .map(|i| {
            let ra = a.row(i);
            let rb = b.row(i);
            let ma = ra.mean().unwrap_or(0.0);
            let mb = rb.mean().unwrap_or(0.0);
            let num: f64 = ra.iter().zip(rb.iter()).map(|(x, y)| (x - ma) * (y - mb)).sum();
            let da: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
            let db: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum::<f64>().sqrt();
            if da * db < 1e-30 {
                1.0
            } else {
                num / (da * db)
            }
        })
        .collect()
}

fn max_relative_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(_, y)| y.abs() > 1e-6)
        .map(|(x, y)| (x - y).abs() / y.abs())
        .fold(0.0f64, f64::max)
}

struct PreparedFixture {
    meta: FixtureMeta,
    counts: Array2<f64>,
    numi: Array1<f64>,
    norm_profiles: Array2<f64>,
    q_mat: Array2<f64>,
    sq_mat: Array2<f64>,
    x_vals: Array1<f64>,
}

fn load_prepared() -> PreparedFixture {
    let meta = load_meta();
    let counts = load_matrix("spatial_counts.bin", meta.n_pixels, meta.n_genes);
    let numi = load_vector("numi.bin", meta.n_pixels);
    let norm_profiles = load_matrix("norm_profiles.bin", meta.n_genes, meta.n_types);
    let q_mat = load_matrix("q_mat.bin", meta.q_nrows, meta.q_ncols);
    let x_vals = load_vector("x_vals.bin", meta.n_xvals);
    let sq_mat = compute_spline_coefficients(&q_mat, &x_vals);
    PreparedFixture {
        meta,
        counts,
        numi,
        norm_profiles,
        q_mat,
        sq_mat,
        x_vals,
    }
}

// -----------------------------------------------------------------------
// X_vals grid parity
// -----------------------------------------------------------------------
#[ignore = "needs R fixtures"]
#[test]
fn x_vals_matches_r() {
    if !have_fixtures() {
        panic!("{}", missing_fixture_msg());
    }
    let meta = load_meta();
    let r_x_vals = load_vector("x_vals.bin", meta.n_xvals);
    let rust_x_vals = rctd_core::build_x_vals();

    assert_eq!(r_x_vals.len(), rust_x_vals.len());
    let max_diff: f64 = r_x_vals
        .iter()
        .zip(rust_x_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    eprintln!("X_vals: max |R-Rust| = {max_diff:.2e}  (len={})", r_x_vals.len());
    assert_abs_diff_eq!(max_diff, 0.0, epsilon = 1e-10);
}

// -----------------------------------------------------------------------
// Q-matrix parity
// -----------------------------------------------------------------------
#[ignore = "needs R fixtures"]
#[test]
fn q_matrix_matches_r_precomputed() {
    if !have_fixtures() {
        panic!("{}", missing_fixture_msg());
    }
    let meta = load_meta();
    let r_q_mat = load_matrix("q_mat.bin", meta.q_nrows, meta.q_ncols);
    let r_x_vals = load_vector("x_vals.bin", meta.n_xvals);

    let sigma = meta.sigma as f64 / 100.0;
    let k_val = (meta.q_nrows - 3) as usize;
    let rust_q = rctd_core::compute_q_matrix(sigma, &r_x_vals, k_val);

    assert_eq!(rust_q.dim(), r_q_mat.dim());
    let max_diff: f64 = rust_q
        .iter()
        .zip(r_q_mat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let mean_diff: f64 = rust_q
        .iter()
        .zip(r_q_mat.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / rust_q.len() as f64;

    eprintln!(
        "Q_mat (sigma={sigma:.2}, k_val={k_val}, dim={:?}):",
        r_q_mat.dim()
    );
    eprintln!("  max  |Rust-R| = {max_diff:.2e}");
    eprintln!("  mean |Rust-R| = {mean_diff:.2e}");
    assert_abs_diff_eq!(max_diff, 0.0, epsilon = 5e-4);
}

// -----------------------------------------------------------------------
// Full mode parity
// -----------------------------------------------------------------------
#[ignore = "needs R fixtures"]
#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity; wgpu uses f32")]
#[test]
fn full_mode_matches_r_spacexr() {
    if !have_fixtures() {
        panic!("{}", missing_fixture_msg());
    }
    let pf = load_prepared();
    let r_weights = load_matrix("r_full_weights.bin", pf.meta.n_pixels, pf.meta.n_types);

    let device = device_cpu();
    let res = run_full_mode(
        &pf.counts,
        &pf.numi,
        &pf.norm_profiles,
        &pf.q_mat,
        &pf.sq_mat,
        &pf.x_vals,
        64,
        &device,
        None,
    );

    let max_diff = max_abs_diff(&res.weights, &r_weights);
    let mean_diff = mean_abs_diff(&res.weights, &r_weights);
    let max_rel = max_relative_diff(&res.weights, &r_weights);
    let r_vec = pearson_r_rows(&res.weights, &r_weights);
    let min_r = r_vec.iter().copied().fold(f64::INFINITY, f64::min);

    eprintln!("=== Full mode R-spacexr parity ===");
    eprintln!("  max  |Rust-R|   = {max_diff:.4e}");
    eprintln!("  mean |Rust-R|   = {mean_diff:.4e}");
    eprintln!("  max  relative   = {max_rel:.4e}");
    eprintln!("  min Pearson r   = {min_r:.6}");
    eprintln!("  weights dim     = {:?}", res.weights.dim());

    for i in 0..5.min(pf.meta.n_pixels) {
        let rust_row: Vec<f64> = res.weights.row(i).to_vec();
        let r_row: Vec<f64> = r_weights.row(i).to_vec();
        let rust_sum: f64 = rust_row.iter().sum();
        let r_sum: f64 = r_row.iter().sum();
        eprintln!(
            "  px{i}: Rust={rust_row:.5?} (Σ={rust_sum:.4})  R={r_row:.5?} (Σ={r_sum:.4})"
        );
    }

    // Absolute tolerance: the QP solver difference (quadprog vs coordinate descent)
    // produces ~1-2% max abs diff. Both produce valid decompositions.
    assert!(
        max_diff < 0.02,
        "full weights max |Rust-R| = {max_diff:.4e} ≥ 0.02"
    );
    assert!(
        mean_diff < 0.01,
        "full weights mean |Rust-R| = {mean_diff:.4e} ≥ 0.01"
    );
    // Per-pixel correlation must be extremely high
    assert!(
        min_r > 0.999,
        "full weights min Pearson r = {min_r:.6} < 0.999"
    );
}

// -----------------------------------------------------------------------
// Doublet mode parity
// -----------------------------------------------------------------------
#[ignore = "needs R fixtures"]
#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity")]
#[test]
fn doublet_mode_matches_r_spacexr() {
    if !have_fixtures() {
        panic!("{}", missing_fixture_msg());
    }
    let pf = load_prepared();
    let type_names = load_lines("type_names.txt");

    let r_weights_full =
        load_matrix("r_doublet_weights_full.bin", pf.meta.n_pixels, pf.meta.n_types);
    let r_spot_class = load_lines("r_doublet_spot_class.txt");
    let r_first_type = load_lines("r_doublet_first_type.txt");
    let r_second_type = load_lines("r_doublet_second_type.txt");

    let cfg = RctdConfig::default();
    let device = device_cpu();
    let res = run_doublet_mode(
        &pf.counts,
        &pf.numi,
        &pf.norm_profiles,
        k_names(pf.meta.n_types),
        &pf.q_mat,
        &pf.sq_mat,
        &pf.x_vals,
        &cfg,
        64,
        &device,
        None,
    );

    // Full weights comparison (same as full mode since doublet starts with full)
    let max_w = max_abs_diff(&res.weights, &r_weights_full);
    let mean_w = mean_abs_diff(&res.weights, &r_weights_full);
    let r_vec = pearson_r_rows(&res.weights, &r_weights_full);
    let min_r = r_vec.iter().copied().fold(f64::INFINITY, f64::min);

    eprintln!("=== Doublet mode R-spacexr parity ===");
    eprintln!("  full weights: max |Rust-R| = {max_w:.4e}, mean = {mean_w:.4e}");
    eprintln!("  min Pearson r = {min_r:.6}");

    // Spot class comparison
    let rust_spot_class_strs: Vec<String> = res
        .spot_class
        .iter()
        .map(|&sc| match sc {
            0 => "reject".to_string(),
            1 => "singlet".to_string(),
            2 => "doublet_certain".to_string(),
            3 => "doublet_uncertain".to_string(),
            x => format!("unknown({x})"),
        })
        .collect();

    let mut spot_class_matches = 0usize;
    for (i, (rust_sc, r_sc)) in rust_spot_class_strs
        .iter()
        .zip(r_spot_class.iter())
        .enumerate()
    {
        if rust_sc == r_sc {
            spot_class_matches += 1;
        } else {
            eprintln!("  px{i}: spot_class Rust={rust_sc} vs R={r_sc}");
        }
    }
    let sc_pct = 100.0 * spot_class_matches as f64 / pf.meta.n_pixels as f64;
    eprintln!("  spot_class agreement: {spot_class_matches}/{} ({sc_pct:.1}%)", pf.meta.n_pixels);

    // Type assignment comparison
    let mut type_matches = 0usize;
    for (i, ((&ft, &st), (r_ft, r_st))) in res
        .first_type
        .iter()
        .zip(res.second_type.iter())
        .zip(r_first_type.iter().zip(r_second_type.iter()))
        .enumerate()
    {
        let rust_ft = type_names.get(ft as usize).cloned().unwrap_or_default();
        let rust_st = type_names.get(st as usize).cloned().unwrap_or_default();
        if rust_ft == *r_ft && rust_st == *r_st {
            type_matches += 1;
        } else {
            eprintln!(
                "  px{i}: types Rust=({rust_ft},{rust_st}) vs R=({r_ft},{r_st})"
            );
        }
    }
    let tp_pct = 100.0 * type_matches as f64 / pf.meta.n_pixels as f64;
    eprintln!(
        "  type assignment agreement: {type_matches}/{} ({tp_pct:.1}%)",
        pf.meta.n_pixels
    );

    assert!(max_w < 0.02, "doublet full weights max abs diff = {max_w:.4e} ≥ 0.02");
    assert!(min_r > 0.999, "doublet min Pearson r = {min_r:.6} < 0.999");
    assert!(
        sc_pct >= 80.0,
        "spot class agreement {sc_pct:.1}% < 80%"
    );
    assert!(
        tp_pct >= 70.0,
        "type assignment agreement {tp_pct:.1}% < 70%"
    );
}

// -----------------------------------------------------------------------
// Multi mode parity
// -----------------------------------------------------------------------
#[ignore = "needs R fixtures"]
#[cfg_attr(feature = "wgpu", ignore = "f64 NdArray parity")]
#[test]
fn multi_mode_matches_r_spacexr() {
    if !have_fixtures() {
        panic!("{}", missing_fixture_msg());
    }
    let pf = load_prepared();
    let r_weights_full =
        load_matrix("r_multi_weights_full.bin", pf.meta.n_pixels, pf.meta.n_types);

    let cfg = RctdConfig::default();
    let device = device_cpu();
    let res = run_multi_mode(
        &pf.counts,
        &pf.numi,
        &pf.norm_profiles,
        k_names(pf.meta.n_types),
        &pf.q_mat,
        &pf.sq_mat,
        &pf.x_vals,
        &cfg,
        64,
        &device,
        None,
    );

    let max_w = max_abs_diff(&res.weights, &r_weights_full);
    let mean_w = mean_abs_diff(&res.weights, &r_weights_full);
    let max_rel = max_relative_diff(&res.weights, &r_weights_full);
    let r_vec = pearson_r_rows(&res.weights, &r_weights_full);
    let min_r = r_vec.iter().copied().fold(f64::INFINITY, f64::min);

    eprintln!("=== Multi mode R-spacexr parity ===");
    eprintln!("  full weights: max |Rust-R| = {max_w:.4e}, mean = {mean_w:.4e}");
    eprintln!("  max relative  = {max_rel:.4e}");
    eprintln!("  min Pearson r = {min_r:.6}");

    for i in 0..5.min(pf.meta.n_pixels) {
        let rust_row: Vec<f64> = res.weights.row(i).to_vec();
        let r_row: Vec<f64> = r_weights_full.row(i).to_vec();
        let nt = res.n_types[i];
        eprintln!("  px{i} (n_types={nt}): Rust={rust_row:.4?}  R={r_row:.4?}");
    }

    assert!(max_w < 0.02, "multi full weights max abs diff = {max_w:.4e} ≥ 0.02");
    assert!(min_r > 0.999, "multi min Pearson r = {min_r:.6} < 0.999");
}
