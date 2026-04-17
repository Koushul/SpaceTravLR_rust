//! Mock “training finished” path: write a tiny `.h5ad`, patch `var` with `mean_lasso_r2`
//! (same HDF5 path as real training), then read back with **anndata-rs** and optionally **h5py**.
//!
//! Run from repo root:
//! ```text
//! cargo run --example mock_h5ad_roundtrip
//! ```
//!
//! If `python3` + `h5py` are available, the script also probes `var/_index` (usually readable)
//! and `var/mean_lasso_r2` (may fail on some arm64 `h5py` wheels when data use Blosc).

use anndata::data::ArrayData;
use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use ndarray::Array2;
use polars::prelude::{DataFrame, NamedFrom, Series};
use spacetravlr::spatial_estimator::{MeanLassoR2Accum, dense_to_csr_f64, patch_adata_var_mean_lasso_r2};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_mock_roundtrip_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let p = dir.join("mock.h5ad");

    println!("→ write {}", p.display());
    let a = AnnData::<H5>::new(&p)?;
    a.set_obs_names(vec!["c0".into(), "c1".into()].into())?;
    a.set_var_names(vec!["G0".into(), "G1".into()].into())?;
    let obs = DataFrame::new(vec![Series::new(
        "cell_type".into(),
        vec!["t".to_string(), "t".to_string()],
    )
    .into()])?;
    a.set_obs(obs)?;
    let var = DataFrame::new(vec![Series::new("gene_ids".into(), vec!["g0", "g1"]).into()])?;
    a.set_var(var)?;
    let dense = Array2::from_elem((2, 2), 0.5f64);
    let csr = dense_to_csr_f64(&dense)?;
    a.set_x(ArrayData::from(csr))?;
    a.close()?;

    println!("→ patch_adata_var_mean_lasso_r2 (adds mean_lasso_r2 column)");
    let mut m: HashMap<String, usize> = HashMap::new();
    m.insert("G0".into(), 0);
    m.insert("G1".into(), 1);
    let scores = Arc::new(vec![
        AtomicU64::new(0.25f64.to_bits()),
        AtomicU64::new(f64::NAN.to_bits()),
    ]);
    let accum = MeanLassoR2Accum {
        gene_to_idx: Arc::new(m),
        scores,
        mean_cnn_r2_scores: None,
    };
    patch_adata_var_mean_lasso_r2(&p, &accum)?;

    println!("→ read back with anndata-rs");
    let a2 = AnnData::<H5>::open(H5::open(&p)?)?;
    let v = a2.read_var()?;
    let r2 = v.column("mean_lasso_r2")?.f64()?;
    anyhow::ensure!((r2.get(0).unwrap() - 0.25).abs() < 1e-9);
    anyhow::ensure!(r2.get(1).unwrap().is_nan());
    a2.close()?;
    println!("   OK: anndata-rs read_var + mean_lasso_r2");

    try_h5py_probe(&p);

    let _ = std::fs::remove_dir_all(&dir);
    println!("→ done (temp dir removed)");
    Ok(())
}

fn try_h5py_probe(p: &Path) {
    let path = p.to_string_lossy().replace('\\', "\\\\");
    let script = format!(
        r#"
import sys
path = r"{path}"
try:
    import h5py
except ImportError:
    print("   (skip h5py: not installed)")
    sys.exit(0)
f = h5py.File(path, "r")
vg = f["var"]
names = list(vg.keys())
print("   h5py var keys:", names)
for k in names:
    obj = vg[k]
    if not hasattr(obj, "shape"):
        continue
    try:
        sl = obj[:2] if getattr(obj, "shape", (0,))[0] != 0 else obj[()]
        print(f"   OK: h5py read var/{{k}} (sample {{type(sl).__name__}})")
    except Exception as e:
        print(f"   WARN: h5py cannot read var/{{k}}:", type(e).__name__, e)
f.close()
"#
    );
    let out = Command::new("python3")
        .args(["-c", &script])
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            let e = String::from_utf8_lossy(&o.stderr);
            if !s.trim().is_empty() {
                print!("{s}");
            }
            if !e.trim().is_empty() && !o.status.success() {
                eprint!("{e}");
            }
        }
        Err(e) => println!("   (skip h5py probe: {e})"),
    }
}
