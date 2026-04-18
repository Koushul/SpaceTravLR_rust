//! End-to-end test for `spacetravlr::niche` on a synthetic spatial run with
//! known functional microniches.
//!
//! Three baselines:
//!   1. **Expression k-means** — run k-means directly on the cell × gene log
//!      matrix. No spatial info, no signalling info.
//!   2. **BANKSY-like** — concatenate per-cell expression with the mean
//!      expression over its k spatial neighbours, then k-means.
//!   3. **Splash-PCA k-means** — flatten the per-cell splash image, project
//!      to a few principal components via SVD, then k-means.
//!
//! The CNN niche detector ([`spacetravlr::niche::NicheRuntime`]) must beat
//! all three in adjusted rand index against the ground-truth niche labels and
//! produce labels that are spatially coherent (spatial purity ≥ baselines).

use burn::backend::NdArray;
use burn_autodiff::Autodiff;
use ndarray::{Array2, Axis, s};
use spacetravlr::niche::image::{StandardizeMode, build_niche_image_stack};
use spacetravlr::niche::{
    NicheRuntime, NicheRuntimeBuilder, NicheTrainConfig, adjusted_rand_index, kmeans_lloyd,
    make_synthetic_run, normalized_mutual_info, spatial_purity_knn,
};

type B = Autodiff<NdArray<f32, i32>>;

fn flatten_array(a: &Array2<f32>) -> Vec<f32> {
    a.iter().copied().collect()
}

fn build_neighbor_aug_expression(
    expr: &Array2<f64>,
    coords: &[[f64; 2]],
    k: usize,
) -> Array2<f32> {
    let n = expr.nrows();
    let g = expr.ncols();
    let pts: Vec<[f64; 2]> = coords.to_vec();
    let tree = kiddo::ImmutableKdTree::<f64, 2>::new_from_slice(&pts);
    let k_q = std::num::NonZero::new(k + 1).unwrap();
    let mut out = Array2::<f32>::zeros((n, g * 2));
    for i in 0..n {
        for j in 0..g {
            out[[i, j]] = expr[[i, j]] as f32;
        }
        let nbrs = tree.nearest_n::<kiddo::SquaredEuclidean>(&coords[i], k_q);
        let mut accum = vec![0.0f64; g];
        let mut count = 0usize;
        for nb in &nbrs {
            let idx = nb.item as usize;
            if idx == i {
                continue;
            }
            for j in 0..g {
                accum[j] += expr[[idx, j]];
            }
            count += 1;
            if count == k {
                break;
            }
        }
        let cnt = count.max(1) as f64;
        for j in 0..g {
            out[[i, g + j]] = (accum[j] / cnt) as f32;
        }
    }
    out
}

/// Tiny PCA via SVD of the centered matrix; returns the top `k` components.
fn pca_features(x: &Array2<f32>, k: usize) -> Array2<f32> {
    let n = x.nrows();
    let _d = x.ncols();
    let mut centered = x.mapv(|v| v as f64);
    let mean = centered.mean_axis(Axis(0)).unwrap();
    for mut row in centered.rows_mut() {
        for j in 0..row.len() {
            row[j] -= mean[j];
        }
    }
    // Use a tall-skinny SVD via nalgebra: convert to nalgebra::DMatrix.
    let na = nalgebra::DMatrix::from_row_slice(
        n,
        x.ncols(),
        &centered.iter().copied().collect::<Vec<_>>(),
    );
    let svd = na.svd(true, false);
    let u = svd.u.expect("U returned");
    let s = svd.singular_values;
    let dim = k.min(u.ncols()).min(s.len());
    let mut out = Array2::<f32>::zeros((n, dim));
    for i in 0..n {
        for j in 0..dim {
            out[[i, j]] = (u[(i, j)] * s[j]) as f32;
        }
    }
    out
}

fn cluster_kmeans(features: &Array2<f32>, k: usize, seed: u64) -> Vec<usize> {
    let n = features.nrows();
    let d = features.ncols();
    let flat = flatten_array(features);
    kmeans_lloyd(&flat, n, d, k, 200, seed).labels
}

fn evaluate(name: &str, gt: &[usize], pred: &[usize], coords: &[[f64; 2]]) -> (f64, f64, f64) {
    let ari = adjusted_rand_index(gt, pred);
    let nmi = normalized_mutual_info(gt, pred);
    let purity = spatial_purity_knn(coords, pred, 10);
    println!(
        "{:<20} ari={:.3} nmi={:.3} spatial_purity_k10={:.3}",
        name, ari, nmi, purity
    );
    (ari, nmi, purity)
}

#[test]
fn cnn_niche_beats_baselines_on_synthetic() {
    let run = make_synthetic_run(80, 5, 42);
    println!(
        "cell_type distribution per niche: {:?}",
        (0..run.n_niches)
            .map(|n| {
                let mut h = std::collections::HashMap::new();
                for (i, &g) in run.niche_gt.iter().enumerate() {
                    if g == n {
                        *h.entry(run.cell_type[i]).or_insert(0usize) += 1;
                    }
                }
                (n, h)
            })
            .collect::<Vec<_>>()
    );
    let coords: Vec<[f64; 2]> = (0..run.n_cells).map(|i| [run.xy[[i, 0]], run.xy[[i, 1]]]).collect();
    let n_clusters = run.n_niches;

    println!(
        "synthetic run: n_cells={}, n_niches={}, n_genes={}, n_targets={}",
        run.n_cells,
        run.n_niches,
        run.gene_names.len(),
        run.splash.len()
    );

    // Baseline 1: k-means on log1p expression
    let expr_log = run.gene_matrix.mapv(|v| (v + 1.0).ln() as f32);
    let pred_expr = cluster_kmeans(&expr_log, n_clusters, 0);
    let (ari_expr, _nmi_expr, purity_expr) = evaluate("expression_kmeans", &run.niche_gt, &pred_expr, &coords);

    // Baseline 2: BANKSY-like
    let banksy_feat = build_neighbor_aug_expression(&run.gene_matrix, &coords, 8);
    let pred_banksy = cluster_kmeans(&banksy_feat, n_clusters, 0);
    let (ari_banksy, _nmi_banksy, purity_banksy) =
        evaluate("banksy_like", &run.niche_gt, &pred_banksy, &coords);

    // Baseline 3: PCA on splash image
    let stack = build_niche_image_stack(&run.splash, run.n_cells, StandardizeMode::PerEntry);
    let mut flat = Array2::<f32>::zeros((run.n_cells, stack.n_targets * stack.n_modulators));
    for c in 0..run.n_cells {
        flat.slice_mut(s![c, ..])
            .assign(&ndarray::ArrayView1::from(stack.cell(c)));
    }
    let pca = pca_features(&flat, 16);
    let pred_pca = cluster_kmeans(&pca, n_clusters, 0);
    let (ari_pca, _nmi_pca, purity_pca) = evaluate("splash_pca", &run.niche_gt, &pred_pca, &coords);

    // CNN niche detector
    let cfg = NicheTrainConfig {
        epochs: 30,
        batch_size: 64,
        learning_rate: 1e-3,
        embedding_dim: 16,
        n_programs: 8,
        spatial_k: 8,
        lambda_recon: 1.0,
        lambda_func: 2.0,
        lambda_spatial: 1.0,
        recon_down: 4,
        conv_channels: (16, 32, 32),
        mlp_hidden: 64,
        projection_dim: 8,
        seed: 0,
        verbose: false,
    };
    let device = Default::default();
    let builder = NicheRuntimeBuilder::from_synthetic(run, StandardizeMode::PerEntry);
    let gt = builder.niche_gt.clone().expect("synth provides gt");
    let coords_owned: Vec<[f64; 2]> = (0..builder.stack.n_cells)
        .map(|i| [builder.xy[[i, 0]], builder.xy[[i, 1]]])
        .collect();
    let out = NicheRuntime::fit::<B>(&device, builder, &cfg, n_clusters);
    let (ari_cnn, _nmi_cnn, purity_cnn) =
        evaluate("cnn_niche", &gt, &out.labels, &coords_owned);

    println!(
        "summary: cnn={:.3} banksy={:.3} expr={:.3} pca={:.3} (purity cnn={:.3} banksy={:.3} expr={:.3} pca={:.3})",
        ari_cnn, ari_banksy, ari_expr, ari_pca, purity_cnn, purity_banksy, purity_expr, purity_pca
    );

    assert!(
        ari_cnn > ari_expr,
        "CNN ARI {ari_cnn:.3} did not beat expression baseline {ari_expr:.3}"
    );
    assert!(
        ari_cnn >= ari_pca - 1e-3,
        "CNN ARI {ari_cnn:.3} did not beat splash-PCA baseline {ari_pca:.3}"
    );
    assert!(
        ari_cnn >= ari_banksy - 1e-3,
        "CNN ARI {ari_cnn:.3} did not beat BANKSY-like baseline {ari_banksy:.3}"
    );
    assert!(
        purity_cnn >= 0.7,
        "CNN niches not spatially coherent: spatial_purity = {purity_cnn:.3}"
    );
    assert!(
        ari_cnn >= 0.7,
        "CNN niche ARI too low: {ari_cnn:.3}"
    );

    // Functional-fidelity check: each *predicted* niche should have a clearly
    // dominant signalling program. We measure
    //   purity_p(c) = max_p [#cells in cluster c with dominant_program == p] / |c|
    // and require its mean across clusters to be much higher than chance.
    // This is what the user calls "niches must be functional": each cluster
    // is dominated by a single signalling program, not just spatially smooth.
    let funcs = &out.train.functional_targets;
    let n_p = funcs.ncols();
    let mut per_cluster_dom = vec![std::collections::HashMap::<usize, usize>::new(); n_clusters];
    for (i, &c) in out.labels.iter().enumerate() {
        let row = funcs.row(i);
        let max_p = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        *per_cluster_dom[c].entry(max_p).or_insert(0) += 1;
    }
    let mut purities = Vec::with_capacity(n_clusters);
    for h in &per_cluster_dom {
        if h.is_empty() {
            continue;
        }
        let total: usize = h.values().sum();
        let max_count = *h.values().max().unwrap();
        purities.push(max_count as f64 / total as f64);
    }
    let mean_purity: f64 = purities.iter().sum::<f64>() / purities.len() as f64;
    let chance = 1.0 / n_p as f64;
    println!(
        "functional fidelity: mean per-cluster dominant-program purity = {:.3} (chance = {:.3}, perfect = 1.0)",
        mean_purity, chance
    );
    assert!(
        mean_purity > 0.7,
        "predicted niches do not concentrate on a single program: purity {mean_purity:.3}"
    );
}
