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

use crate::perturb::{PerturbConfig, PerturbResult, PerturbTarget, perturb_result_from_delta};
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

/// Pre–enum on-disk layout (flat struct). Decoded after tagged [`PerturbResultBlob`] fails, for older foyer files.
#[derive(Serialize, Deserialize)]
pub(crate) struct PerturbResultBlobLegacy {
    nrows: usize,
    ncols: usize,
    delta: Vec<f64>,
    simulated: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct PerturbResultBlobV1 {
    nrows: usize,
    ncols: usize,
    delta: Vec<f64>,
    simulated: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct PerturbResultBlobV2 {
    nrows: usize,
    ncols: usize,
    delta: Vec<f64>,
}

/// On-disk / in-foyer encoding for [`PerturbResult`](crate::perturb::PerturbResult): **V1** stores
/// both `delta` and `simulated`; **V2** stores only `delta` (~half the size; reconstruct simulated via
/// [`decode_perturb_cache_entry`]).
#[derive(Serialize, Deserialize)]
enum PerturbResultBlob {
    V1(PerturbResultBlobV1),
    V2(PerturbResultBlobV2),
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
    let blob = PerturbResultBlob::V2(PerturbResultBlobV2 {
        nrows,
        ncols,
        delta: r.delta.iter().copied().collect(),
    });
    bincode::serde::encode_to_vec(&blob, bincode::config::standard())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Encode **V1** (delta + simulated) for tests and migration tooling; prefer [`encode_perturb_result`] for production.
pub fn encode_perturb_result_v1(r: &PerturbResult) -> anyhow::Result<Vec<u8>> {
    let (nrows, ncols) = r.delta.dim();
    anyhow::ensure!(
        r.simulated.dim() == (nrows, ncols),
        "simulated/delta shape mismatch"
    );
    let blob = PerturbResultBlob::V1(PerturbResultBlobV1 {
        nrows,
        ncols,
        delta: r.delta.iter().copied().collect(),
        simulated: r.simulated.iter().copied().collect(),
    });
    bincode::serde::encode_to_vec(&blob, bincode::config::standard())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn perturb_from_legacy_or_v1_parts(
    nrows: usize,
    ncols: usize,
    delta: Vec<f64>,
    simulated: Vec<f64>,
) -> anyhow::Result<PerturbResult> {
    let n = nrows * ncols;
    anyhow::ensure!(delta.len() == n, "delta length mismatch");
    anyhow::ensure!(simulated.len() == n, "simulated length mismatch");
    let delta = Array2::from_shape_vec((nrows, ncols), delta)
        .map_err(|e| anyhow::anyhow!("delta reshape: {e}"))?;
    let simulated = Array2::from_shape_vec((nrows, ncols), simulated)
        .map_err(|e| anyhow::anyhow!("simulated reshape: {e}"))?;
    Ok(PerturbResult { simulated, delta })
}

/// Decode a GRN hybrid-cache entry: **legacy flat** and **V1** return stored tensors; **V2** reconstructs
/// `simulated` with [`perturb_result_from_delta`].
pub fn decode_perturb_cache_entry(
    bytes: &[u8],
    gene_mtx: &Array2<f64>,
    gene_names: &[String],
    targets: &[PerturbTarget],
) -> anyhow::Result<PerturbResult> {
    let cfg = bincode::config::standard();
    if let Ok((blob, consumed)) =
        bincode::serde::decode_from_slice::<PerturbResultBlob, _>(bytes, cfg)
    {
        if consumed == bytes.len() {
            return match blob {
                PerturbResultBlob::V1(v) => {
                    perturb_from_legacy_or_v1_parts(v.nrows, v.ncols, v.delta, v.simulated)
                }
                PerturbResultBlob::V2(v) => {
                    let n = v.nrows * v.ncols;
                    anyhow::ensure!(v.delta.len() == n, "delta length mismatch");
                    let delta = Array2::from_shape_vec((v.nrows, v.ncols), v.delta)
                        .map_err(|e| anyhow::anyhow!("delta reshape: {e}"))?;
                    anyhow::ensure!(
                        gene_mtx.nrows() == v.nrows && gene_mtx.ncols() == v.ncols,
                        "gene_mtx shape {:?} != cached blob {}×{}",
                        gene_mtx.dim(),
                        v.nrows,
                        v.ncols
                    );
                    Ok(perturb_result_from_delta(
                        gene_mtx, delta, targets, gene_names,
                    ))
                }
            };
        }
    }

    let (leg, consumed): (PerturbResultBlobLegacy, usize) =
        bincode::serde::decode_from_slice::<PerturbResultBlobLegacy, _>(bytes, cfg)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    anyhow::ensure!(
        consumed == bytes.len(),
        "trailing bytes after legacy GRN perturb blob"
    );
    perturb_from_legacy_or_v1_parts(leg.nrows, leg.ncols, leg.delta, leg.simulated)
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
        bincode::serde::decode_from_slice::<UmapGridBlob, _>(bytes, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(b)
}

pub type PerturbHybridCache = HybridCache<PerturbCacheKey, Vec<u8>>;
pub type GridHybridCache = HybridCache<UmapGridCacheKey, Vec<u8>>;

const MB: usize = 1024 * 1024;

fn foyer_mb_from_env(key: &'static str, default_mb: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .map(|m| m.saturating_mul(MB))
        .unwrap_or(default_mb.saturating_mul(MB))
}

/// In-process (foyer memory tier) and on-disk capacity for hybrid caches.
///
/// Environment (positive integers, mebibytes; invalid or unset → defaults):
/// `SPACETRAVLR_FOYER_GRN_MEMORY_MB` (default 256), `SPACETRAVLR_FOYER_GRN_DISK_MB` (512),
/// `SPACETRAVLR_FOYER_GRID_MEMORY_MB` (64), `SPACETRAVLR_FOYER_GRID_DISK_MB` (128).
///
/// The spatial viewer skips hybrid GRN/grid caches for tiny matrices (default: estimated δ payload
/// under 1 MiB). Set `SPACETRAVLR_FOYER_GRN_MIN_PAYLOAD_MB` to change; `0` means always use foyer
/// for sizes below the large-payload cutoff in `spatial_viewer`.
///
/// Lowering the GRN **memory** cap only reduces how many serialized blobs stay hot in foyer’s
/// LRU (weight = byte length). A cache hit still **decodes** in the spatial viewer into full
/// `delta` / `simulated` arrays — that peak is separate from this tier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FoyerCacheLimits {
    pub grn_memory: usize,
    pub grn_disk: usize,
    pub grid_memory: usize,
    pub grid_disk: usize,
}

impl Default for FoyerCacheLimits {
    fn default() -> Self {
        Self {
            grn_memory: 256 * MB,
            grn_disk: 512 * MB,
            grid_memory: 64 * MB,
            grid_disk: 128 * MB,
        }
    }
}

impl FoyerCacheLimits {
    pub fn from_env() -> Self {
        Self {
            grn_memory: foyer_mb_from_env("SPACETRAVLR_FOYER_GRN_MEMORY_MB", 256),
            grn_disk: foyer_mb_from_env("SPACETRAVLR_FOYER_GRN_DISK_MB", 512),
            grid_memory: foyer_mb_from_env("SPACETRAVLR_FOYER_GRID_MEMORY_MB", 64),
            grid_disk: foyer_mb_from_env("SPACETRAVLR_FOYER_GRID_DISK_MB", 128),
        }
    }
}

pub struct FoyerPerturbCaches {
    pub grn: Arc<PerturbHybridCache>,
    pub grid: Arc<GridHybridCache>,
}

pub async fn open_foyer_perturb_caches(
    cache_dir: Option<&Path>,
) -> anyhow::Result<FoyerPerturbCaches> {
    open_foyer_perturb_caches_with_limits(cache_dir, FoyerCacheLimits::from_env()).await
}

pub async fn open_foyer_perturb_caches_with_limits(
    cache_dir: Option<&Path>,
    limits: FoyerCacheLimits,
) -> anyhow::Result<FoyerPerturbCaches> {
    let dir = match cache_dir {
        Some(p) => p.to_path_buf(),
        None => std::env::temp_dir().join("spacetravlr_foyer_perturb"),
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("create cache dir {}", dir.display()))?;

    let grn_path = dir.join("grn");
    std::fs::create_dir_all(&grn_path)?;
    let grn_device = FsDeviceBuilder::new(&grn_path)
        .with_capacity(limits.grn_disk)
        .build()?;

    let grn: PerturbHybridCache = HybridCacheBuilder::new()
        .with_name("spacetravlr_grn_perturb")
        .memory(limits.grn_memory)
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
        .with_capacity(limits.grid_disk)
        .build()?;

    let grid: GridHybridCache = HybridCacheBuilder::new()
        .with_name("spacetravlr_umap_grid")
        .memory(limits.grid_memory)
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
    use crate::perturb::{PerturbConfig, PerturbResult, PerturbTarget, perturb_result_from_delta};
    use crate::transition_umap::TransitionUmapParams;

    fn sample_gene_names() -> Vec<String> {
        (0..3).map(|i| format!("g{i}")).collect()
    }

    fn sample_perturb_result_consistent() -> (Array2<f64>, Vec<String>, PerturbResult) {
        let gene_names = sample_gene_names();
        let gene_mtx = Array2::<f64>::zeros((2, 3));
        let delta = Array2::from_shape_vec((2, 3), (1..=6).map(|i| i as f64).collect()).unwrap();
        let pr = perturb_result_from_delta(&gene_mtx, delta, &[], &gene_names);
        (gene_mtx, gene_names, pr)
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
    fn encode_decode_perturb_result_v2_roundtrip() {
        let (gene_mtx, gene_names, pr) = sample_perturb_result_consistent();
        let bytes = encode_perturb_result(&pr).unwrap();
        let out = decode_perturb_cache_entry(&bytes, &gene_mtx, &gene_names, &[]).unwrap();
        assert_eq!(out.delta, pr.delta);
        assert_eq!(out.simulated, pr.simulated);
    }

    #[test]
    fn encode_decode_perturb_result_v1_enum_roundtrip() {
        let (gene_mtx, gene_names, pr) = sample_perturb_result_consistent();
        let bytes = encode_perturb_result_v1(&pr).unwrap();
        let out = decode_perturb_cache_entry(&bytes, &gene_mtx, &gene_names, &[]).unwrap();
        assert_eq!(out.delta, pr.delta);
        assert_eq!(out.simulated, pr.simulated);
    }

    #[test]
    fn legacy_flat_grn_blob_decode() {
        let (gene_mtx, gene_names, _) = sample_perturb_result_consistent();
        let delta: Vec<f64> = (1..=6).map(|i| i as f64).collect();
        let simulated: Vec<f64> = vec![0.25; 6];
        let leg = PerturbResultBlobLegacy {
            nrows: 2,
            ncols: 3,
            delta: delta.clone(),
            simulated: simulated.clone(),
        };
        let bytes = bincode::serde::encode_to_vec(&leg, bincode::config::standard()).unwrap();
        let out = decode_perturb_cache_entry(&bytes, &gene_mtx, &gene_names, &[]).unwrap();
        assert_eq!(out.delta.as_slice().unwrap(), delta.as_slice());
        assert_eq!(out.simulated.as_slice().unwrap(), simulated.as_slice());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hybrid_grn_get_or_fetch_invokes_fetch_only_once_sequential() {
        let dir = TempDir::new().unwrap();
        let caches =
            open_foyer_perturb_caches_with_limits(Some(dir.path()), FoyerCacheLimits::default())
                .await
                .unwrap();
        let key = PerturbCacheKey {
            dataset_epoch: 42,
            fingerprint: [0xAB; 32],
        };
        let fetches = Arc::new(AtomicU64::new(0));
        let (gene_mtx, gene_names, pr) = sample_perturb_result_consistent();
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
            let decoded =
                decode_perturb_cache_entry(entry.value(), &gene_mtx, &gene_names, &[]).unwrap();
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
        let caches = Arc::new(
            open_foyer_perturb_caches_with_limits(Some(dir.path()), FoyerCacheLimits::default())
                .await
                .unwrap(),
        );
        let key = PerturbCacheKey {
            dataset_epoch: 7,
            fingerprint: [0xCD; 32],
        };
        let fetches = Arc::new(AtomicU64::new(0));
        let (gene_mtx, gene_names, pr_sample) = sample_perturb_result_consistent();
        let enc = encode_perturb_result(&pr_sample).unwrap();

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
            decode_perturb_cache_entry(entry.value(), &gene_mtx, &gene_names, &[]).unwrap();
        }

        assert_eq!(
            fetches.load(Ordering::SeqCst),
            1,
            "Concurrent get_or_fetch for the same key should compute once (foyer deduplication)"
        );
        close_foyer_caches(caches.as_ref()).await.unwrap();
    }
}
