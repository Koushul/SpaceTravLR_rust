//! Foyer hybrid cache for full GRN [`PerturbResult`](crate::perturb::PerturbResult) payloads and
//! optional UMAP transition grid blobs (spatial viewer).

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use foyer::{
    BlockEngineConfig, Compression, DeviceBuilder, FsDeviceBuilder, HybridCache,
    HybridCacheBuilder, RecoverMode,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::perturb::{PerturbConfig, PerturbResult, PerturbTarget};
use crate::transition_umap::TransitionUmapParams;

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct PerturbCacheKey {
    pub dataset_epoch: u64,
    pub fingerprint: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct UmapGridCacheKey {
    pub dataset_epoch: u64,
    pub perturb_fingerprint: [u8; 32],
    pub limit_clusters: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub highlight_cell_types: Vec<String>,
    pub transition_blob_hash: [u8; 32],
}

#[derive(Serialize, Deserialize)]
struct GrnFingerprintPayload {
    pub quick_ko: bool,
    pub adata_path: String,
    pub n_obs: usize,
    pub n_vars: usize,
    pub targets: Vec<PerturbTarget>,
    pub cfg: PerturbConfig,
}

#[derive(Serialize, Deserialize)]
struct PerturbResultBlob {
    nrows: usize,
    ncols: usize,
    delta: Vec<f64>,
    simulated: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct UmapGridBlob {
    pub nx: usize,
    pub ny: usize,
    pub grid_x: Vec<f64>,
    pub grid_y: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    #[serde(default)]
    pub cell_u: Vec<f32>,
    #[serde(default)]
    pub cell_v: Vec<f32>,
}

fn hash_grn_fingerprint(payload: &GrnFingerprintPayload) -> [u8; 32] {
    let bytes = bincode::serde::encode_to_vec(payload, bincode::config::standard())
        .expect("GrnFingerprintPayload bincode");
    *blake3::hash(&bytes).as_bytes()
}

/// Build a stable cache fingerprint for GRN perturbation (excludes UMAP / quiver-only parameters).
pub fn grn_perturb_cache_key(
    dataset_epoch: u64,
    quick_ko: bool,
    adata_path: &str,
    n_obs: usize,
    n_vars: usize,
    targets: &[PerturbTarget],
    cfg: &PerturbConfig,
) -> PerturbCacheKey {
    let payload = GrnFingerprintPayload {
        quick_ko,
        adata_path: adata_path.to_string(),
        n_obs,
        n_vars,
        targets: targets.to_vec(),
        cfg: cfg.clone(),
    };
    PerturbCacheKey {
        dataset_epoch,
        fingerprint: hash_grn_fingerprint(&payload),
    }
}

pub fn encode_perturb_result(r: &PerturbResult) -> anyhow::Result<Vec<u8>> {
    let (nrows, ncols) = r.delta.dim();
    anyhow::ensure!(
        r.simulated.dim() == (nrows, ncols),
        "simulated/delta shape mismatch"
    );
    let blob = PerturbResultBlob {
        nrows,
        ncols,
        delta: r.delta.iter().copied().collect(),
        simulated: r.simulated.iter().copied().collect(),
    };
    bincode::serde::encode_to_vec(&blob, bincode::config::standard())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

pub fn decode_perturb_result(bytes: &[u8]) -> anyhow::Result<PerturbResult> {
    let (blob, _): (PerturbResultBlob, _) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    let n = blob.nrows * blob.ncols;
    anyhow::ensure!(blob.delta.len() == n, "delta length mismatch");
    anyhow::ensure!(blob.simulated.len() == n, "simulated length mismatch");
    let delta = Array2::from_shape_vec((blob.nrows, blob.ncols), blob.delta)
        .map_err(|e| anyhow::anyhow!("delta reshape: {e}"))?;
    let simulated = Array2::from_shape_vec((blob.nrows, blob.ncols), blob.simulated)
        .map_err(|e| anyhow::anyhow!("simulated reshape: {e}"))?;
    Ok(PerturbResult { simulated, delta })
}

#[derive(Serialize)]
struct TransitionVisualFp {
    params: TransitionUmapParams,
    include_cell_vectors: bool,
}

pub fn transition_visual_fingerprint(
    params: &TransitionUmapParams,
    include_cell_vectors: bool,
) -> [u8; 32] {
    let bytes = bincode::serde::encode_to_vec(
        &TransitionVisualFp {
            params: params.clone(),
            include_cell_vectors,
        },
        bincode::config::standard(),
    )
    .expect("transition fp bincode");
    *blake3::hash(&bytes).as_bytes()
}

pub fn umap_grid_cache_key(
    dataset_epoch: u64,
    perturb_fingerprint: [u8; 32],
    limit_clusters: bool,
    highlight_cell_types: &[String],
    params: &TransitionUmapParams,
    include_cell_vectors: bool,
) -> UmapGridCacheKey {
    let mut hl: Vec<String> = highlight_cell_types.iter().cloned().collect();
    hl.sort();
    UmapGridCacheKey {
        dataset_epoch,
        perturb_fingerprint,
        limit_clusters,
        highlight_cell_types: hl,
        transition_blob_hash: transition_visual_fingerprint(params, include_cell_vectors),
    }
}

pub fn encode_umap_grid_blob(b: &UmapGridBlob) -> anyhow::Result<Vec<u8>> {
    bincode::serde::encode_to_vec(b, bincode::config::standard())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

pub fn decode_umap_grid_blob(bytes: &[u8]) -> anyhow::Result<UmapGridBlob> {
    let (b, _): (UmapGridBlob, _) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(b)
}

pub type PerturbHybridCache = HybridCache<PerturbCacheKey, Vec<u8>>;
pub type GridHybridCache = HybridCache<UmapGridCacheKey, Vec<u8>>;

pub struct FoyerPerturbCaches {
    pub grn: Arc<PerturbHybridCache>,
    pub grid: Arc<GridHybridCache>,
}

pub async fn open_foyer_perturb_caches(
    cache_dir: Option<&Path>,
) -> anyhow::Result<FoyerPerturbCaches> {
    let dir = match cache_dir {
        Some(p) => p.to_path_buf(),
        None => std::env::temp_dir().join("spacetravlr_foyer_perturb"),
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("create cache dir {}", dir.display()))?;

    let grn_path = dir.join("grn");
    std::fs::create_dir_all(&grn_path)?;
    let grn_device = FsDeviceBuilder::new(&grn_path)
        .with_capacity(512 * 1024 * 1024)
        .build()?;

    let grn: PerturbHybridCache = HybridCacheBuilder::new()
        .with_name("spacetravlr_grn_perturb")
        .memory(256 * 1024 * 1024)
        .with_weighter(|_k, v: &Vec<u8>| v.len())
        .storage()
        .with_compression(Compression::Lz4)
        .with_engine_config(BlockEngineConfig::new(grn_device))
        .with_recover_mode(RecoverMode::Quiet)
        .build()
        .await?;

    let grid_path = dir.join("umap_grid");
    std::fs::create_dir_all(&grid_path)?;
    let grid_device = FsDeviceBuilder::new(&grid_path)
        .with_capacity(128 * 1024 * 1024)
        .build()?;

    let grid: GridHybridCache = HybridCacheBuilder::new()
        .with_name("spacetravlr_umap_grid")
        .memory(64 * 1024 * 1024)
        .with_weighter(|_k, v: &Vec<u8>| v.len())
        .storage()
        .with_compression(Compression::Lz4)
        .with_engine_config(BlockEngineConfig::new(grid_device))
        .with_recover_mode(RecoverMode::Quiet)
        .build()
        .await?;

    Ok(FoyerPerturbCaches {
        grn: Arc::new(grn),
        grid: Arc::new(grid),
    })
}

pub async fn close_foyer_caches(c: &FoyerPerturbCaches) -> anyhow::Result<()> {
    c.grn.close().await?;
    c.grid.close().await?;
    Ok(())
}

#[cfg(all(test, feature = "spatial-viewer"))]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use ndarray::Array2;
    use tempfile::TempDir;

    use super::*;
    use crate::perturb::{PerturbConfig, PerturbResult, PerturbTarget};
    use crate::transition_umap::TransitionUmapParams;

    fn sample_perturb_result() -> PerturbResult {
        let delta = Array2::from_shape_vec((2, 3), (1..=6).map(|i| i as f64).collect()).unwrap();
        let simulated = Array2::from_shape_vec((2, 3), vec![0.25; 6]).unwrap();
        PerturbResult { delta, simulated }
    }

    #[test]
    fn grn_cache_key_stable_for_same_perturbation() {
        let t = vec![PerturbTarget {
            gene: "TP53".into(),
            desired_expr: 2.0,
            cell_indices: Some(vec![0, 2]),
        }];
        let cfg = PerturbConfig {
            n_propagation: 3,
            scale_factor: 1.0,
            beta_scale_factor: 1.0,
            beta_cap: None,
            min_expression: 1e-9,
            ligand_grid_factor: Some(0.5),
        };
        let a = grn_perturb_cache_key(1, false, "/data/x.h5ad", 100, 2000, &t, &cfg);
        let b = grn_perturb_cache_key(1, false, "/data/x.h5ad", 100, 2000, &t, &cfg);
        assert_eq!(a, b);
        let c = grn_perturb_cache_key(2, false, "/data/x.h5ad", 100, 2000, &t, &cfg);
        assert_ne!(a, c);
        assert_eq!(
            a.fingerprint, c.fingerprint,
            "blake3 payload excludes dataset_epoch; epoch namespaces the HybridCache key"
        );
        let d = grn_perturb_cache_key(1, false, "/other.h5ad", 100, 2000, &t, &cfg);
        assert_ne!(a.fingerprint, d.fingerprint);
    }

    #[test]
    fn grn_cache_key_differs_when_transition_only_inputs_change_elsewhere() {
        let t = vec![PerturbTarget {
            gene: "MYC".into(),
            desired_expr: 1.0,
            cell_indices: None,
        }];
        let cfg_a = PerturbConfig {
            n_propagation: 4,
            ..Default::default()
        };
        let mut cfg_b = cfg_a.clone();
        cfg_b.n_propagation = 5;
        let k_a = grn_perturb_cache_key(1, false, "p", 10, 500, &t, &cfg_a);
        let k_b = grn_perturb_cache_key(1, false, "p", 10, 500, &t, &cfg_b);
        assert_ne!(k_a.fingerprint, k_b.fingerprint);
    }

    #[test]
    fn transition_visual_fingerprint_changes_with_umap_params() {
        let mut p2 = TransitionUmapParams::default();
        p2.vector_scale = 0.99;
        assert_ne!(
            transition_visual_fingerprint(&TransitionUmapParams::default(), false),
            transition_visual_fingerprint(&p2, false)
        );
    }

    #[test]
    fn encode_decode_perturb_result_roundtrip() {
        let pr = sample_perturb_result();
        let bytes = encode_perturb_result(&pr).unwrap();
        let out = decode_perturb_result(&bytes).unwrap();
        assert_eq!(out.delta, pr.delta);
        assert_eq!(out.simulated, pr.simulated);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hybrid_grn_get_or_fetch_invokes_fetch_only_once_sequential() {
        let dir = TempDir::new().unwrap();
        let caches = open_foyer_perturb_caches(Some(dir.path())).await.unwrap();
        let key = PerturbCacheKey {
            dataset_epoch: 42,
            fingerprint: [0xAB; 32],
        };
        let fetches = Arc::new(AtomicU64::new(0));
        let pr = sample_perturb_result();
        let enc = encode_perturb_result(&pr).unwrap();

        for _ in 0..3 {
            let enc_cl = enc.clone();
            let fetches_cl = Arc::clone(&fetches);
            let entry = caches
                .grn
                .get_or_fetch(&key, move || {
                    let enc_cl = enc_cl.clone();
                    let fetches_cl = Arc::clone(&fetches_cl);
                    async move {
                        fetches_cl.fetch_add(1, Ordering::SeqCst);
                        Ok::<Vec<u8>, anyhow::Error>(enc_cl)
                    }
                })
                .await
                .unwrap();
            let decoded = decode_perturb_result(entry.value()).unwrap();
            assert_eq!(decoded.delta, pr.delta);
        }

        assert_eq!(
            fetches.load(Ordering::SeqCst),
            1,
            "HybridCache should reuse the serialized GRN blob without re-running fetch"
        );
        close_foyer_caches(&caches).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn hybrid_grn_concurrent_get_or_fetch_coalesces_to_single_fetch() {
        let dir = TempDir::new().unwrap();
        let caches = Arc::new(open_foyer_perturb_caches(Some(dir.path())).await.unwrap());
        let key = PerturbCacheKey {
            dataset_epoch: 7,
            fingerprint: [0xCD; 32],
        };
        let fetches = Arc::new(AtomicU64::new(0));
        let enc = encode_perturb_result(&sample_perturb_result()).unwrap();

        let mut handles = vec![];
        for _ in 0..12 {
            let caches = Arc::clone(&caches);
            let fetches = Arc::clone(&fetches);
            let enc = enc.clone();
            let key = key.clone();
            handles.push(tokio::spawn(async move {
                caches
                    .grn
                    .get_or_fetch(&key, move || {
                        let enc = enc.clone();
                        let fetches = Arc::clone(&fetches);
                        async move {
                            fetches.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(std::time::Duration::from_millis(8)).await;
                            Ok::<Vec<u8>, anyhow::Error>(enc)
                        }
                    })
                    .await
            }));
        }

        for h in handles {
            let entry = h.await.unwrap().unwrap();
            decode_perturb_result(entry.value()).unwrap();
        }

        assert_eq!(
            fetches.load(Ordering::SeqCst),
            1,
            "Concurrent get_or_fetch for the same key should compute once (foyer deduplication)"
        );
        close_foyer_caches(caches.as_ref()).await.unwrap();
    }
}
