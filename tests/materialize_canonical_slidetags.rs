//! Timing / behavior checks for [`spacetravlr::spatial_estimator::materialize_canonical_training_adata`].
//!
//! Ignored by default. Uses `SPACETRAVLR_MATERIALIZE_H5AD` or `SlideTags_human_tonsil_processed.h5ad`
//! at the repo root. Fails if that file is missing or larger than 500 MiB.

use std::path::PathBuf;
use std::time::Instant;

use spacetravlr::config::{SpaceshipConfig, expand_user_path};
use spacetravlr::scanpy_preprocess::training_prep_subdir;
use spacetravlr::spatial_estimator::materialize_canonical_training_adata;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
#[ignore = "local SlideTags h5ad; set SPACETRAVLR_MATERIALIZE_H5AD or place the file at the repo root"]
fn materialize_reuse_canonical_skips_full_copy_second_pass() {
    let src = std::env::var("SPACETRAVLR_MATERIALIZE_H5AD").map_or_else(
        |_| repo_root().join("SlideTags_human_tonsil_processed.h5ad"),
        PathBuf::from,
    );
    assert!(
        src.is_file(),
        "missing {} (set SPACETRAVLR_MATERIALIZE_H5AD)",
        src.display()
    );
    let meta = std::fs::metadata(&src).unwrap_or_else(|e| panic!("stat {}: {e}", src.display()));
    assert!(
        meta.len() <= 500 * 1024 * 1024,
        "{} is >500 MiB",
        src.display()
    );

    let tmp = std::env::temp_dir().join(format!("st_materialize_slidetags_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    let out_dir = tmp.join("out");
    std::fs::create_dir_all(&out_dir).expect("mkdir out");

    let foreign = tmp.join("foreign_copy.h5ad");
    std::fs::copy(&src, &foreign).expect("seed foreign copy");

    let cfg = SpaceshipConfig::default();
    let stem_path = &src;

    let mut path1 = foreign.to_string_lossy().to_string();
    let t1 = Instant::now();
    materialize_canonical_training_adata(&mut path1, &out_dir, stem_path, &cfg, None)
        .expect("materialize first");
    let d1 = t1.elapsed();

    let resolved1 = PathBuf::from(expand_user_path(path1.trim()));
    assert!(
        resolved1 == foreign || resolved1.starts_with(training_prep_subdir(&out_dir)),
        "first pass should keep input or use prep cache under spacetravlr_prep, got {}",
        resolved1.display()
    );
    assert!(resolved1.is_file(), "missing {}", resolved1.display());

    let mut path2 = foreign.to_string_lossy().to_string();
    let t2 = Instant::now();
    materialize_canonical_training_adata(&mut path2, &out_dir, stem_path, &cfg, None)
        .expect("materialize second");
    let d2 = t2.elapsed();

    eprintln!(
        "materialize_canonical_training_adata: first={d1:?} second={d2:?} (second should be fast reuse)"
    );

    let resolved2 = PathBuf::from(expand_user_path(path2.trim()));
    assert_eq!(resolved1, resolved2);

    assert!(
        d1 < std::time::Duration::from_secs(120) && d2 < std::time::Duration::from_secs(120),
        "unexpected stall: first={d1:?} second={d2:?}"
    );
    if d1 >= std::time::Duration::from_millis(200) {
        assert!(
            d2 * 3 < d1,
            "expected second pass much faster than first (reuse), got first={d1:?} second={d2:?}"
        );
    }

    let _ = std::fs::remove_dir_all(&tmp);
}
