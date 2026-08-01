//! SpaceTravLR CellularNicheNetwork CNN trainer compiled to WebAssembly (Burn NdArray).

pub mod model;
pub mod pack;
pub mod train;

pub use pack::{
    ClusterTrainResult, CnnClusterPack, CnnGeneTrainPack, CnnTrainHyperparams, GeneTrainResult,
    decode_pack, encode_pack,
};
pub use train::{train_gene_pack, train_gene_pack_bytes};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

/// Train CNN clusters from a bincode [`CnnGeneTrainPack`]. Returns JSON [`GeneTrainResult`].
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn train_pack(pack_bytes: &[u8]) -> Result<JsValue, JsValue> {
    let result = train_gene_pack_bytes(pack_bytes).map_err(|e| JsValue::from_str(&e))?;
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Smoke-test: tiny synthetic cluster, a few Adam steps. Returns wall time ms.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn smoke_train_ms() -> u32 {
    let n = 32usize;
    let h = 8usize;
    let n_mod = 4usize;
    let n_clust = 3usize;
    let mut pack = CnnGeneTrainPack {
        gene: "SMOKE".into(),
        hyperparams: CnnTrainHyperparams {
            epochs: 2,
            cnn_minibatch_size: 16,
            cnn_max_cells_per_epoch: Some(32),
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
    let _ = &mut pack;
    train_gene_pack(&pack).wall_ms
}
