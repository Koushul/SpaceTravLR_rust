use ndarray::Array2;
use spacetravlr::config::{ExecutionConfig, SpaceshipConfig, mix_execution_random_seed};
use spacetravlr::lasso::{GroupLasso, GroupLassoParams};

#[test]
fn mix_execution_random_seed_is_stable() {
    let a = mix_execution_random_seed(42, "BCL6");
    let b = mix_execution_random_seed(42, "BCL6");
    assert_eq!(a, b);
    assert_ne!(a, mix_execution_random_seed(43, "BCL6"));
    assert_ne!(a, mix_execution_random_seed(42, "BCL7"));
}

#[test]
fn default_execution_random_seed_is_nonzero_fixed() {
    assert_eq!(ExecutionConfig::default().random_seed, 42);
}

#[test]
fn toml_execution_random_seed_roundtrip() {
    let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[execution]
random_seed = 1001
"#;
    let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
    assert_eq!(cfg.execution.random_seed, 1001);
}

#[test]
fn group_lasso_twice_same_seed_same_coef() {
    let n = 30usize;
    let p = 4usize;
    let x = Array2::from_shape_fn((n, p), |(i, j)| ((i + j * 7) as f64) * 0.01);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| x[[i, 0]] * 2.0 + x[[i, 1]] * -0.5 + 0.1);
    let groups = vec![0i64, 0, 1, 1];
    let seed = mix_execution_random_seed(42, "GENE1");
    let params = GroupLassoParams {
        groups,
        l1_reg: 0.01,
        group_reg: 0.02,
        n_iter: 200,
        tol: 1e-4,
        seed,
        ..Default::default()
    };
    let mut m1 = GroupLasso::new(params.clone());
    let mut m2 = GroupLasso::new(params);
    m1.fit(&x, &y, None).unwrap();
    m2.fit(&x, &y, None).unwrap();
    let c1 = m1.fitted.as_ref().unwrap().coef.clone();
    let c2 = m2.fitted.as_ref().unwrap().coef.clone();
    assert_eq!(c1.shape(), c2.shape());
    for (a, b) in c1.iter().zip(c2.iter()) {
        assert!((a - b).abs() < 1e-12, "coef mismatch: {} vs {}", a, b);
    }
}
