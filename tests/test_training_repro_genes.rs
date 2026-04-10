//! Join-style loading of `spacetravlr_run_repro.toml` with `[training] genes` / `max_genes`.

use spacetravlr::config::{
    SpaceshipConfig, filter_training_var_names, resolve_training_target_genes,
};
use std::fs;
use std::io::Write;

#[test]
fn from_file_join_repro_with_training_genes() {
    let dir = std::env::temp_dir().join(format!(
        "spacetravlr_repro_genes_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("spacetravlr_run_repro.toml");
    let mut f = fs::File::create(&path).unwrap();
    write!(
        f,
        r#"
[data]
adata_path = "/data/dataset.h5ad"
layer = "X"
cluster_annot = "cell_type_int"

[training]
mode = "seed"
epochs = 10
learning_rate = 0.001
score_threshold = 0.1
genes = ["g1", "g2", "g3"]
max_genes = 2

[execution]
output_dir = "{}"
n_parallel = 4
write_minimal_repro_h5ad = false
stale_lock_secs = 0
"#,
        dir.display()
    )
    .unwrap();

    let cfg = SpaceshipConfig::from_file(&path).expect("parse repro TOML");
    assert_eq!(
        cfg.training.genes,
        Some(vec!["g1".into(), "g2".into(), "g3".into()])
    );
    assert_eq!(cfg.training.max_genes, Some(2));

    let all_var = vec!["z".into(), "g3".into(), "g1".into(), "g2".into()];
    let queue = resolve_training_target_genes(
        &all_var,
        cfg.training.genes.as_deref(),
        cfg.training.max_genes,
    );
    assert_eq!(queue, vec!["g3", "g1"]);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn public_helpers_match_join_queue_semantics() {
    let all_var: Vec<String> = (0..10).map(|i| format!("gene{i}")).collect();
    let allow = vec!["gene7".into(), "gene2".into(), "gene9".into()];
    let filtered = filter_training_var_names(&all_var, Some(&allow));
    assert_eq!(filtered, vec!["gene2", "gene7", "gene9"]);
    let capped = resolve_training_target_genes(&all_var, Some(&allow), Some(2));
    assert_eq!(capped, vec!["gene2", "gene7"]);
}
