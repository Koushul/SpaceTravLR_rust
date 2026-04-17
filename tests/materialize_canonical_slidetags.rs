//! Timing / behavior checks for [`spacetravlr::spatial_estimator::materialize_canonical_training_adata`].
//!
//! Uses `SlideTags_human_tonsil_processed.h5ad` at the repo root when present (typical dev checkout).
//! Skips automatically when the file is missing or very large (>500 MiB) so CI stays light.

use std::path::PathBuf;
use std::time::Instant;

use spacetravlr::config::{SpaceshipConfig, expand_user_path};
use spacetravlr::spatial_estimator::materialize_canonical_training_adata;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn materialize_reuse_canonical_skips_full_copy_second_pass() {
    let src = repo_root().join("SlideTags_human_tonsil_processed.h5ad");
    if !src.is_file() {
        eprintln!("skip: {} not found", src.display());
        return;
    }
    let Ok(meta) = std::fs::metadata(&src) else {
        eprintln!("skip: could not stat {}", src.display());
        return;
    };
    if meta.len() > 500 * 1024 * 1024 {
        eprintln!("skip: {} is >500 MiB", src.display());
        return;
    }

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

    let canonical = out_dir.join("SlideTags_human_tonsil_processed.h5ad");
    assert_eq!(
        PathBuf::from(expand_user_path(path1.trim())),
        canonical,
        "first pass should end on canonical path, got {path1}"
    );
    assert!(canonical.is_file(), "missing {}", canonical.display());

    let mut path2 = foreign.to_string_lossy().to_string();
    let t2 = Instant::now();
    materialize_canonical_training_adata(&mut path2, &out_dir, stem_path, &cfg, None)
        .expect("materialize second");
    let d2 = t2.elapsed();

    eprintln!(
        "materialize_canonical_training_adata: first={d1:?} second={d2:?} (second should be fast reuse)"
    );

    let resolved = PathBuf::from(expand_user_path(path2.trim()));
    assert_eq!(resolved, canonical);

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
