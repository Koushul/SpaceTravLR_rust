//! Train a cached `{gene}_cnn_train_data.npz` on native WGPU (Metal/Vulkan/DX).
use std::env;
use std::path::PathBuf;
use std::time::Instant;

use burn::backend::wgpu::{init_setup, AutoGraphicsApi, RuntimeOptions, WgpuDevice};
use burn::backend::Wgpu;
use burn_autodiff::Autodiff;
use spacetravlr_cnn_pack::cnn_gene_train_pack_from_npz;
use spacetravlr_cnn_wasm::train::train_gene_pack_on;

fn main() {
    let npz = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/spacetravlr_cnn_web/packs/AICDA_cnn_train_data.npz"));
    let gene = env::args().nth(2).unwrap_or_else(|| "AICDA".into());
    let epochs: u32 = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let device = WgpuDevice::default();
    println!("init_setup…");
    init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());

    let pack = cnn_gene_train_pack_from_npz(&npz, &gene, Some(epochs)).expect("load npz");
    println!(
        "loaded {} clusters from {} (epochs={})",
        pack.clusters.len(),
        npz.display(),
        epochs
    );

    type B = Autodiff<Wgpu<f32, i32>>;
    let t0 = Instant::now();
    let result = train_gene_pack_on::<B>(&pack, &device, "webgpu");
    println!(
        "done wall_ms={} backend={} elapsed_s={:.3}",
        result.wall_ms,
        result.backend,
        t0.elapsed().as_secs_f64()
    );
    for c in &result.clusters {
        println!(
            "  cluster {} n={} mse {:?} -> {:?} ({} ms)",
            c.cluster_id,
            c.n_cells,
            c.mse_epochs.first(),
            c.mse_epochs.last(),
            c.wall_ms
        );
    }
}
