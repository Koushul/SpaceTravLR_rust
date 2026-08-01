//! Native WebGPU smoke for the CNN WASM trainer (Metal/Vulkan/DX via wgpu).
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use burn::backend::wgpu::{init_setup, AutoGraphicsApi, RuntimeOptions, WgpuDevice};
    use burn::backend::Wgpu;
    use burn_autodiff::Autodiff;
    use spacetravlr_cnn_wasm::{
        CnnClusterPack, CnnGeneTrainPack, CnnTrainHyperparams, train::train_gene_pack_on,
    };

    let device = WgpuDevice::default();
    println!("init_setup AutoGraphicsApi…");
    let t0 = std::time::Instant::now();
    init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
    println!("init_setup ok in {:.3}s", t0.elapsed().as_secs_f64());

    let n = 64usize;
    let h = 8usize;
    let n_mod = 4usize;
    let n_clust = 3usize;
    let pack = CnnGeneTrainPack {
        gene: "SMOKE_WEBGPU".into(),
        hyperparams: CnnTrainHyperparams {
            epochs: 4,
            cnn_minibatch_size: 32,
            cnn_max_cells_per_epoch: Some(64),
            cnn_early_stop_patience: 0,
            ..Default::default()
        },
        clusters: vec![CnnClusterPack {
            cluster_id: 0,
            n_cells: n as u32,
            n_modulators: n_mod as u32,
            n_clusters: n_clust as u32,
            spatial_h: h as u32,
            spatial_w: h as u32,
            vision_in_channels: 1,
            spatial_maps: vec![0.1f32; n * 1 * h * h],
            x: vec![0.05f32; n * n_mod],
            spatial_features: vec![0.02f32; n * n_clust],
            y: vec![0.0f32; n],
            anchors: {
                let mut a = vec![0.5f32];
                a.extend(std::iter::repeat_n(0.1f32, n_mod));
                a
            },
            y_lasso: None,
            lasso_r2: 0.5,
        }],
    };

    type B = Autodiff<Wgpu<f32, i32>>;
    let t1 = std::time::Instant::now();
    let result = train_gene_pack_on::<B>(&pack, &device, "webgpu");
    println!(
        "webgpu smoke: wall_ms={} backend={} mse={:?} elapsed_s={:.3}",
        result.wall_ms,
        result.backend,
        result.clusters[0].mse_epochs,
        t1.elapsed().as_secs_f64()
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {}
