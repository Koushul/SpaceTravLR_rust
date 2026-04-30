use approx::assert_abs_diff_eq;
use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn_autodiff::Autodiff;
use ndarray::{Array1, Array2, Array4};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use spacetravlr::config::{CnnConfig, CnnOutputActivation};
use spacetravlr::estimator::{train_cluster_cnn_epochs, TrainClusterCnnEpochsInput};
use spacetravlr::lasso::{
    ClusteredGroupLasso, GroupLassoParams, largest_eigenvalue_symmetric_power_iter,
};
use spacetravlr::model::CellularNicheNetworkConfig;

fn synthetic_clustered_regression(
    seed: u64,
    n: usize,
    p: usize,
    n_clusters: usize,
) -> (Array2<f64>, Array2<f64>, Array1<i64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    let mut clusters = Array1::<i64>::zeros(n);
    for i in 0..n {
        let c = (i % n_clusters.max(1)) as i64;
        clusters[i] = c;
        let off = c as f64 * 0.15;
        for j in 0..p {
            x[[i, j]] = rng.gen_range(0.0..1.0) + off * (j as f64 / p as f64).sin();
        }
    }
    let mut y = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        let c = clusters[i];
        let mut s = 0.3 + 0.02 * (c as f64);
        for j in 0..p.min(12) {
            s += x[[i, j]] * (0.05 + (j as f64) * 0.01);
        }
        y[[i, 0]] = s + rng.gen_range(0.0..1.0) * 0.08;
    }
    (x, y, clusters)
}

#[test]
fn bench_gram_coefficient_parity() {
    let p = 40usize;
    let n_clusters = 5usize;
    for &n in &[500usize] {
        let (x, y, clusters) = synthetic_clustered_regression(42, n, p, n_clusters);
        let groups: Vec<i64> = (0..p as i64).collect();

        let base = GroupLassoParams {
            groups,
            group_reg: 1e-4,
            l1_reg: 1e-4,
            n_iter: 1200,
            tol: 1e-6,
            gram_override: Some(false),
            ..Default::default()
        };

        let mut resid = ClusteredGroupLasso::new(base.clone());
        let mut gram_m = ClusteredGroupLasso::new(GroupLassoParams {
            gram_override: Some(true),
            ..base
        });

        let t0 = std::time::Instant::now();
        let _ = resid.fit(&x, &y, &clusters);
        let ms_resid = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let _ = gram_m.fit(&x, &y, &clusters);
        let ms_gram = t1.elapsed().as_secs_f64() * 1000.0;

        let mut max_coef = 0.0_f64;
        let mut max_int = 0.0_f64;
        for cid in resid.models.keys() {
            let mr = resid.models.get(cid).unwrap().fitted.as_ref().unwrap();
            let mg = gram_m.models.get(cid).unwrap().fitted.as_ref().unwrap();
            for i in 0..mr.coef.nrows() {
                for j in 0..mr.coef.ncols() {
                    max_coef = max_coef.max((mr.coef[[i, j]] - mg.coef[[i, j]]).abs());
                }
            }
            max_int = max_int.max((mr.intercept[[0, 0]] - mg.intercept[[0, 0]]).abs());
        }

        println!(
            "bench_gram_coefficient_parity n={n} p={p} ms_resid={ms_resid:.2} ms_gram={ms_gram:.2} speedup={:.2}x max_coef_diff={max_coef:.2e} max_int_diff={max_int:.2e}",
            ms_resid / ms_gram.max(1e-9)
        );

        assert!(max_coef < 5e-6, "n={n} max_coef_diff {max_coef} too large");
        assert!(max_int < 5e-6, "n={n} max_int_diff {max_int} too large");
    }
}

#[test]
#[ignore]
fn bench_gram_coefficient_parity_10k() {
    let p = 40usize;
    let n_clusters = 5usize;
    let n = 10_000usize;
    let (x, y, clusters) = synthetic_clustered_regression(42, n, p, n_clusters);
    let groups: Vec<i64> = (0..p as i64).collect();
    let base = GroupLassoParams {
        groups,
        group_reg: 1e-4,
        l1_reg: 1e-4,
        n_iter: 1200,
        tol: 1e-6,
        gram_override: Some(false),
        ..Default::default()
    };
    let mut resid = ClusteredGroupLasso::new(base.clone());
    let mut gram_m = ClusteredGroupLasso::new(GroupLassoParams {
        gram_override: Some(true),
        ..base
    });
    let _ = resid.fit(&x, &y, &clusters);
    let _ = gram_m.fit(&x, &y, &clusters);
    println!("bench n=10000 parity check done");
}

#[test]
#[ignore]
fn bench_gram_coefficient_parity_large_n() {
    let p = 200usize;
    let n_clusters = 5usize;
    for &n in &[50_000usize, 200_000usize] {
        let (x, y, clusters) = synthetic_clustered_regression(42, n, p, n_clusters);
        let groups: Vec<i64> = (0..p as i64).collect();
        let base = GroupLassoParams {
            groups,
            group_reg: 1e-4,
            l1_reg: 1e-4,
            n_iter: 300,
            tol: 1e-6,
            gram_override: Some(false),
            ..Default::default()
        };

        let mut resid = ClusteredGroupLasso::new(base.clone());
        let mut gram_m = ClusteredGroupLasso::new(GroupLassoParams {
            gram_override: Some(true),
            ..base
        });

        let t0 = std::time::Instant::now();
        let _ = resid.fit(&x, &y, &clusters);
        let ms_resid = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let _ = gram_m.fit(&x, &y, &clusters);
        let ms_gram = t1.elapsed().as_secs_f64() * 1000.0;

        let mut max_coef = 0.0_f64;
        for cid in resid.models.keys() {
            let mr = resid.models.get(cid).unwrap().fitted.as_ref().unwrap();
            let mg = gram_m.models.get(cid).unwrap().fitted.as_ref().unwrap();
            for i in 0..mr.coef.nrows() {
                for j in 0..mr.coef.ncols() {
                    max_coef = max_coef.max((mr.coef[[i, j]] - mg.coef[[i, j]]).abs());
                }
            }
        }

        println!(
            "bench_gram_large n={n} ms_resid={ms_resid:.1} ms_gram={ms_gram:.1} ratio={:.2}x max_coef_diff={max_coef:.2e}",
            ms_resid / ms_gram.max(1e-9)
        );

        assert!(max_coef < 5e-6, "max_coef_diff {max_coef}");
    }
}

#[test]
fn bench_gram_fista_scaling() {
    let p_feat = 50usize;
    let groups: Vec<i64> = (0..p_feat as i64).collect();
    let base_params = GroupLassoParams {
        groups,
        group_reg: 1e-6,
        l1_reg: 1e-6,
        n_iter: 120,
        tol: 1e-8,
        gram_override: Some(false),
        ..Default::default()
    };

    for &n in &[100usize, 500usize] {
        let x = Array2::<f64>::from_shape_fn((n, p_feat), |(i, j)| {
            (((i + 13) * (j + 7)) % 251) as f64 / 251.0
        });
        let y = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.001).sin());

        let mut m_res = spacetravlr::lasso::GroupLasso::new(base_params.clone());
        let t0 = std::time::Instant::now();
        let _ = m_res.fit(&x, &y, None);
        let ms_res = t0.elapsed().as_secs_f64() * 1000.0;

        let mut m_gram = spacetravlr::lasso::GroupLasso::new(GroupLassoParams {
            gram_override: Some(true),
            ..base_params.clone()
        });
        let t1 = std::time::Instant::now();
        let _ = m_gram.fit(&x, &y, None);
        let ms_gr = t1.elapsed().as_secs_f64() * 1000.0;

        let cr = m_res.fitted.as_ref().unwrap();
        let cg = m_gram.fitted.as_ref().unwrap();
        let diff: f64 = cr
            .coef
            .iter()
            .zip(cg.coef.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        println!(
            "bench_gram_fista_scaling n={n} ms_residual={ms_res:.3} ms_gram={ms_gr:.3} coef_l1diff={diff:.3e}"
        );

        assert_abs_diff_eq!(cr.coef[[0, 0]], cg.coef[[0, 0]], epsilon = 1e-5);
        assert!(diff < 1e-4, "coef drift n={n} sum_abs_diff={diff}");
    }
}

#[test]
#[ignore]
fn bench_gram_fista_scaling_mid_n() {
    let p_feat = 50usize;
    let groups: Vec<i64> = (0..p_feat as i64).collect();
    let base_params = GroupLassoParams {
        groups,
        group_reg: 1e-6,
        l1_reg: 1e-6,
        n_iter: 120,
        tol: 1e-8,
        gram_override: Some(false),
        ..Default::default()
    };
    let n = 5_000usize;
    let x = Array2::<f64>::from_shape_fn((n, p_feat), |(i, j)| {
        (((i + 13) * (j + 7)) % 251) as f64 / 251.0
    });
    let y = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.001).sin());
    let mut m_res = spacetravlr::lasso::GroupLasso::new(base_params.clone());
    let t0 = std::time::Instant::now();
    let _ = m_res.fit(&x, &y, None);
    let ms_res = t0.elapsed().as_secs_f64() * 1000.0;
    let mut m_gram = spacetravlr::lasso::GroupLasso::new(GroupLassoParams {
        gram_override: Some(true),
        ..base_params.clone()
    });
    let t1 = std::time::Instant::now();
    let _ = m_gram.fit(&x, &y, None);
    let ms_gr = t1.elapsed().as_secs_f64() * 1000.0;
    println!(
        "bench_gram_fista_scaling_mid_n n={n} ms_residual={ms_res:.2} ms_gram={ms_gr:.2} ratio={:.2}x",
        ms_res / ms_gr.max(1e-9)
    );
}

#[test]
#[ignore]
fn bench_gram_fista_scaling_large_n() {
    let p_feat = 200usize;
    let groups: Vec<i64> = (0..p_feat as i64).collect();
    let base_params = GroupLassoParams {
        groups,
        group_reg: 1e-6,
        l1_reg: 1e-6,
        n_iter: 80,
        tol: 1e-7,
        gram_override: Some(false),
        ..Default::default()
    };

    for &n in &[20_000usize, 100_000usize] {
        let x = Array2::<f64>::from_shape_fn((n, p_feat), |(i, j)| {
            (((i + 13) * (j + 7)) % 251) as f64 / 251.0
        });
        let y = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.0001).cos());

        let mut m_res = spacetravlr::lasso::GroupLasso::new(base_params.clone());
        let t0 = std::time::Instant::now();
        let _ = m_res.fit(&x, &y, None);
        let ms_res = t0.elapsed().as_secs_f64() * 1000.0;

        let mut m_gram = spacetravlr::lasso::GroupLasso::new(GroupLassoParams {
            gram_override: Some(true),
            ..base_params.clone()
        });
        let t1 = std::time::Instant::now();
        let _ = m_gram.fit(&x, &y, None);
        let ms_gr = t1.elapsed().as_secs_f64() * 1000.0;

        println!(
            "bench_gram_fista_scaling_large n={n} ms_residual={ms_res:.1} ms_gram={ms_gr:.1} ratio={:.2}x",
            ms_res / ms_gr.max(1e-9)
        );
    }
}

#[test]
fn eigenvalue_power_iter_positive_on_gram() {
    let g = ndarray::array![[2.0_f64, 0.5, 0.0], [0.5, 2.0, 0.1], [0.0, 0.1, 1.0]];
    let ev = largest_eigenvalue_symmetric_power_iter(&g, 1u64, 64);
    assert!(
        ev > 2.0 && ev < 3.0,
        "expected dominant eigen ~2.2+, got {ev}"
    );
}

#[test]
#[ignore]
fn bench_cnn_cap_tradeoff() {
    type B = Autodiff<NdArray<f32, i32>>;
    let device = Default::default();
    const CLUSTER_N: usize = 512;
    const H: usize = 24;
    const W: usize = 24;
    const P: usize = 20;
    const K: usize = 4;

    let sm_c = Array4::<f32>::from_elem((CLUSTER_N, 1, H, W), 0.1f32);
    let x_c = Array2::<f64>::from_elem((CLUSTER_N, P), 0.05);
    let sf_c = Array2::<f64>::from_elem((CLUSTER_N, K), 0.02);
    let mut y_c = Array1::<f64>::zeros(CLUSTER_N);
    for i in 0..CLUSTER_N {
        y_c[i] = 0.05 * (i as f64).sin();
    }

    let anchors_vec: Vec<f32> = std::iter::once(0.45f32)
        .chain(std::iter::repeat_n(0.09f32, P))
        .collect();
    let anchors_tensor = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(anchors_vec.clone(), [anchors_vec.len()]),
        &device,
    );
    let cfg_nn = CellularNicheNetworkConfig {
        n_modulators: P,
        n_clusters: K,
        vision_in_channels: 1,
    };

    println!(
        "{:-<80}",
        "bench_cnn_cap_tradeoff max_batches total_ms final_mse "
    );

    for max_b in [
        None,
        Some(64usize),
        Some(32usize),
        Some(16usize),
        Some(8usize),
    ] {
        let model = cfg_nn.init::<B>(
            &device,
            anchors_tensor.clone(),
            CnnOutputActivation::Sigmoid,
        );
        let cnn = CnnConfig {
            lasso_pred_align_weight: 0.0,
            cnn_minibatch_size: 64,
            cnn_max_batches_per_epoch: max_b,
            cnn_max_cells_per_epoch: None,
            cnn_early_stop_patience: 0,
            ..Default::default()
        };

        let t0 = std::time::Instant::now();
        let (_m, mse_ep, div) = train_cluster_cnn_epochs(TrainClusterCnnEpochsInput {
            model,
            device: &device,
            sm_c: &sm_c,
            x_c: &x_c,
            sf_c: &sf_c,
            y_c: &y_c,
            cluster_n: CLUSTER_N,
            cluster_id: 0usize,
            y_lasso_cpu: None,
            beta_prior_w: 0.01f32,
            cnn: &cnn,
            learning_rate: 1e-3,
            epochs: 24usize,
            cnn_epoch_slot: None,
            shuffle_seed: 0,
        });
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let final_mse = mse_ep.last().copied().unwrap_or(f32::NAN);
        let bs = cnn.cnn_minibatch_size.clamp(1, CLUSTER_N);
        let steps_ep = max_b.unwrap_or(CLUSTER_N.div_ceil(bs));
        let total_steps = steps_ep * 24;

        println!(
            "max_batches={max_b:?} steps_per_epoch~{steps_ep} total_steps~{total_steps} wall_ms={ms:.1} final_mse={final_mse:.6} diverged={div}"
        );

        assert!(!div);
        assert!(final_mse.is_finite());
    }
}
