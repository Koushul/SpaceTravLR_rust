use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const SPACESHIP_MERGE_SECTIONS: &[&str] = &[
    "data",
    "spatial",
    "grn",
    "cnn",
    "lasso",
    "training",
    "execution",
    "perturbation",
    "model_export",
];

fn merge_toml_table_maps(
    base: &mut toml::map::Map<String, toml::Value>,
    overlay: &toml::map::Map<String, toml::Value>,
) {
    for (k, v) in overlay {
        match (base.get_mut(k), v) {
            (Some(toml::Value::Table(base_sub)), toml::Value::Table(ov_sub)) => {
                merge_toml_table_maps(base_sub, ov_sub);
            }
            _ => {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

/// Merges `[data]`, `[spatial]`, … from `overlay_root` into a TOML document that will deserialize
/// as [`SpaceshipConfig`]. Unknown top-level keys in `overlay_root` are ignored.
pub fn merge_spaceship_overlay_into_toml(into: &mut toml::Value, overlay_root: &toml::Value) {
    let Some(into_t) = into.as_table_mut() else {
        return;
    };
    let Some(ov_t) = overlay_root.as_table() else {
        return;
    };
    for &sec in SPACESHIP_MERGE_SECTIONS {
        if let Some(ov_sec) = ov_t.get(sec).and_then(|x| x.as_table()) {
            let entry = into_t
                .entry(sec.to_string())
                .or_insert(toml::Value::Table(Default::default()));
            if let Some(bt) = entry.as_table_mut() {
                merge_toml_table_maps(bt, ov_sec);
            } else {
                *entry = toml::Value::Table(ov_sec.clone());
            }
        }
    }
}

/// Canonical per-run TOML in the training output directory (full `SpaceshipConfig` as executed).
pub const RUN_REPRO_TOML_FILENAME: &str = "spacetravlr_run_repro.toml";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpaceshipConfig {
    #[serde(default)]
    pub data: DataConfig,
    #[serde(default)]
    pub spatial: SpatialConfig,
    #[serde(default)]
    pub grn: GrnConfig,
    #[serde(default)]
    pub cnn: CnnConfig,
    #[serde(default)]
    pub lasso: LassoConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    #[serde(default)]
    pub execution: ExecutionConfig,
    #[serde(default)]
    pub perturbation: PerturbationConfig,
    #[serde(default)]
    pub model_export: ModelExportConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataConfig {
    pub adata_path: String,
    pub layer: String,
    pub cluster_annot: String,
    pub condition: Option<String>,
    /// Optional path (tilde-expanded like `adata_path`): one AnnData `obs_names` value per line (`#` comments, blanks skipped).
    /// When set, perturbation loads only these rows (expression, spatial, betadata alignment). Results apply to this ROI only.
    pub perturb_obs_subset_file: Option<String>,
    /// Prior for heuristic `obsm['spatial']` → micron scaling during full Scanpy preprocess (`human` or `mouse`).
    #[serde(default = "default_data_spatial_species")]
    pub spatial_species: String,
    /// Override median k-NN target distance (µm) for spatial scaling; omit to use the species default.
    pub spatial_median_nn_target_um: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpatialConfig {
    pub radius: f64,
    pub spatial_dim: usize,
    pub contact_distance: f64,
    /// Multiplier on Gaussian kernel weights in received-ligand aggregation (`calculate_weighted_ligands`).
    #[serde(default = "default_one_f64")]
    pub weighted_ligand_scale_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GrnConfig {
    /// Directory containing `{mouse|human}_network.parquet`. Overrides `SPACETRAVLR_DATA_DIR` and
    /// built-in search (manifest / cwd walk). Tilde and `~/` expanded like `data.adata_path`.
    pub network_data_dir: Option<String>,
    /// Optional Feather/IPC file containing TF priors with columns:
    /// `source` (TF), `target` (gene), `cell_type` (obs.cell_type label).
    pub tf_priors_feather: Option<String>,
    pub tf_ligand_cutoff: f64,
    /// Keep only DB L–R pairs whose **ligand** is among the top `max_ligands` by mean expression
    /// (uses `[data].layer`, e.g. `imputed_count`). Requires per-gene mean map when training.
    /// Deserialize accepts legacy TOML key `max_lr_pairs`.
    #[serde(alias = "max_lr_pairs")]
    pub max_ligands: Option<usize>,
    #[serde(default = "default_true")]
    pub use_tf_modulators: bool,
    #[serde(default = "default_true")]
    pub use_lr_modulators: bool,
    #[serde(default = "default_true")]
    pub use_tfl_modulators: bool,
    /// Genes to add as raw-expression modulators (fourth Lasso group). Excludes target and any gene
    /// already used as TF / LR / TFL (see `resolve_extra_modulators_and_lr`).
    #[serde(default)]
    pub extra_modulators: Vec<String>,
    /// Optional file: one gene per line or comma-separated; `#` comments. Appended to `extra_modulators`.
    pub extra_modulators_file: Option<String>,
    /// Extra L–R pairs as `LIG$REC` strings (or `LIG,REC` per element). Merged after database LR selection.
    #[serde(default)]
    pub extra_lr: Vec<String>,
    /// Optional file: one pair per line (`LIG$REC` or `LIG,REC`); `#` comments.
    pub extra_lr_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LassoConfig {
    pub l1_reg: f64,
    pub group_reg: f64,
    pub n_iter: usize,
    pub tol: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CnnTrainingMode {
    #[serde(alias = "minimal", alias = "seed-only")]
    Seed,
    Full,
    #[default]
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HybridCnnGatingConfig {
    /// Smallest cluster cell count must be at least this for CNN (sample-complexity gate).
    pub min_cells_per_cluster_for_cnn: usize,
    /// If modulator count exceeds this, Moran's p must pass `moran_p_value_max_when_over_modulator_cap`.
    pub max_modulators_soft_for_cnn: Option<usize>,
    pub moran_k_neighbors: usize,
    pub moran_permutations: usize,
    pub moran_p_value_max: f64,
    pub moran_p_value_max_when_over_modulator_cap: Option<f64>,
    pub require_all_clusters_lasso_converged: bool,
    /// If None, `TrainingConfig.score_threshold` is used as the minimum mean lasso R² for CNN.
    pub min_mean_lasso_r2_for_cnn: Option<f64>,
    pub min_mean_target_expression_for_cnn: Option<f64>,
    pub hybrid_modulator_spatial_weight: f64,
    pub cnn_force_genes_file: Option<String>,
    pub cnn_skip_genes_file: Option<String>,
    /// If set, phase 1 only records candidates; phase 2 runs CNN for the top-K by `rank_score`.
    pub hybrid_cnn_top_k: Option<usize>,
    /// 0 = conservative (stricter Moran p and mean R² gates → fewer CNNs). 1 = permissive.
    /// 0.5 reproduces the effective thresholds implied by `moran_p_value_max` / mean R² alone (legacy behavior).
    #[serde(default = "default_hybrid_cnn_permissiveness")]
    pub hybrid_cnn_permissiveness: f64,
}

fn default_hybrid_cnn_permissiveness() -> f64 {
    0.5
}

fn default_true() -> bool {
    true
}

fn default_one_f64() -> f64 {
    1.0
}

impl Default for HybridCnnGatingConfig {
    fn default() -> Self {
        Self {
            min_cells_per_cluster_for_cnn: 80,
            max_modulators_soft_for_cnn: Some(256),
            moran_k_neighbors: 8,
            moran_permutations: 99,
            moran_p_value_max: 0.05,
            moran_p_value_max_when_over_modulator_cap: Some(0.01),
            require_all_clusters_lasso_converged: true,
            min_mean_lasso_r2_for_cnn: None,
            min_mean_target_expression_for_cnn: None,
            hybrid_modulator_spatial_weight: 1.0,
            cnn_force_genes_file: None,
            cnn_skip_genes_file: None,
            hybrid_cnn_top_k: None,
            hybrid_cnn_permissiveness: default_hybrid_cnn_permissiveness(),
        }
    }
}

impl HybridCnnGatingConfig {
    fn permissiveness_t(&self) -> f64 {
        self.hybrid_cnn_permissiveness.clamp(0.0, 1.0)
    }

    pub fn effective_moran_p_max(&self) -> f64 {
        let t = self.permissiveness_t();
        let f = 0.3 + 1.4 * t;
        (self.moran_p_value_max * f).clamp(1e-12, 1.0)
    }

    pub fn effective_moran_p_strict(&self) -> f64 {
        let base = self
            .moran_p_value_max_when_over_modulator_cap
            .unwrap_or(self.moran_p_value_max);
        let t = self.permissiveness_t();
        let f = 0.3 + 1.4 * t;
        (base * f).clamp(1e-12, 1.0)
    }

    pub fn effective_min_mean_lasso_r2(&self, base_min_r2: f64) -> f64 {
        let t = self.permissiveness_t();
        let r2f = 1.4 - 0.8 * t;
        (base_min_r2 * r2f).max(0.0)
    }
}

fn default_training_mode_option() -> Option<CnnTrainingMode> {
    Some(CnnTrainingMode::Seed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// Kept for CLI/back-compat; hybrid runs lasso first per gene regardless.
    pub seed_only: bool,
    #[serde(default = "default_training_mode_option")]
    pub mode: Option<CnnTrainingMode>,
    pub epochs: usize,
    pub learning_rate: f64,
    pub score_threshold: f64,
    #[serde(default)]
    pub hybrid: HybridCnnGatingConfig,
    /// Subset of AnnData `var` to train (`--genes`, `[training] genes`). Persisted in
    /// `spacetravlr_run_repro.toml` for `--join-output-dir`.
    #[serde(default)]
    pub genes: Option<Vec<String>>,
    /// Cap after the `genes` filter (`--max-genes`, `[training] max_genes`). Persisted for join.
    #[serde(default)]
    pub max_genes: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CnnOutputActivation {
    Identity,
    Sigmoid,
    Tanh,
    /// `2 * sigmoid(x)`, output in (0, 2) before anchor scaling (plain sigmoid is (0, 1)).
    #[default]
    SigmoidX2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CnnConfig {
    pub adam_beta_1: f64,
    pub adam_beta_2: f64,
    pub adam_epsilon: f64,
    pub weight_decay: Option<f64>,
    pub grad_clip_norm: Option<f64>,
    pub spatial_feature_radius: f64,
    /// Applied after the final head linear (`mlp.l2`), before multiplying by lasso anchors.
    pub output_activation: CnnOutputActivation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionConfig {
    pub n_parallel: usize,
    pub output_dir: String,
    /// When true, write `spacetravlr_minimal_repro.h5ad` under the output directory (large I/O on big datasets).
    pub write_minimal_repro_h5ad: bool,
    /// If > 0, remove a gene `*.lock` file older than this many seconds before claiming the gene,
    /// and run a background sweep about every 10 minutes over the output directory (crash recovery on shared storage).
    pub stale_lock_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerturbationConfig {
    pub beta_scale_factor: f64,
    pub beta_cap: Option<f64>,
    pub n_propagation: usize,
    /// Grid spacing as a fraction of the Gaussian radius for approximate
    /// received-ligand computation.  E.g. 0.5 → spacing = radius/2.
    /// Smaller = more accurate, larger = faster.  Omit or comment out for
    /// exact O(N²) computation.
    pub ligand_grid_factor: Option<f64>,
    /// Default cells CSV path for `spacetravlr-perturb` export / TUI (relative to run TOML directory unless absolute).
    #[serde(default)]
    pub cells_csv: Option<String>,
    /// Default column in `cells_csv` (required in TOML when `cells_csv` is set).
    #[serde(default)]
    pub cells_csv_column: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelExportConfig {
    /// When true, export trained CNN weights for genes that run per-cell CNN refinement (default off).
    pub save_cnn_weights: bool,
    /// Write .npz with deflate compression (recommended).
    pub compressed_npz: bool,
    /// Subdirectory under [execution].output_dir for CNN `.npz` exports (default `CNN_weights`).
    pub output_subdir: String,
}

fn default_data_spatial_species() -> String {
    "human".into()
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            adata_path: String::new(),
            layer: "imputed_count".into(),
            cluster_annot: "cell_type".into(),
            condition: None,
            perturb_obs_subset_file: None,
            spatial_species: default_data_spatial_species(),
            spatial_median_nn_target_um: None,
        }
    }
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            radius: 200.0,
            spatial_dim: 32,
            contact_distance: 50.0,
            weighted_ligand_scale_factor: 1.0,
        }
    }
}

impl Default for GrnConfig {
    fn default() -> Self {
        Self {
            network_data_dir: None,
            tf_priors_feather: None,
            tf_ligand_cutoff: 0.2,
            max_ligands: None,
            use_tf_modulators: true,
            use_lr_modulators: true,
            use_tfl_modulators: true,
            extra_modulators: Vec::new(),
            extra_modulators_file: None,
            extra_lr: Vec::new(),
            extra_lr_file: None,
        }
    }
}

impl GrnConfig {
    /// Merge TOML `extra_modulators` / `extra_lr` with optional files. Paths are expanded (`~`);
    /// relative paths resolve against `config_file_parent` when provided.
    pub fn resolve_extra_modulators_and_lr(
        &self,
        config_file_parent: Option<&Path>,
    ) -> anyhow::Result<(Vec<String>, Vec<(String, String)>)> {
        let resolve_path = |raw: &str| -> PathBuf {
            let exp = expand_user_path(raw.trim());
            let pb = Path::new(&exp);
            if pb.is_absolute() {
                pb.to_path_buf()
            } else if let Some(parent) = config_file_parent {
                parent.join(pb)
            } else {
                pb.to_path_buf()
            }
        };

        let mut genes: Vec<String> = Vec::new();
        let mut gene_seen: HashSet<String> = HashSet::new();
        for g in &self.extra_modulators {
            let t = g.trim();
            if t.is_empty() {
                continue;
            }
            let s = t.to_string();
            if gene_seen.insert(s.clone()) {
                genes.push(s);
            }
        }
        if let Some(ref f) = self.extra_modulators_file {
            let path = resolve_path(f);
            for g in crate::grn_extra::load_extra_modulators_file(&path)? {
                if gene_seen.insert(g.clone()) {
                    genes.push(g);
                }
            }
        }

        let mut pairs: Vec<(String, String)> = Vec::new();
        let mut pair_seen: HashSet<String> = HashSet::new();
        for s in &self.extra_lr {
            if let Some(p) = crate::grn_extra::parse_extra_lr_token(s) {
                let key = format!("{}${}", p.0, p.1);
                if pair_seen.insert(key.clone()) {
                    pairs.push(p);
                }
            }
        }
        if let Some(ref f) = self.extra_lr_file {
            let path = resolve_path(f);
            for p in crate::grn_extra::load_extra_lr_file(&path)? {
                let key = format!("{}${}", p.0, p.1);
                if pair_seen.insert(key) {
                    pairs.push(p);
                }
            }
        }

        Ok((genes, pairs))
    }
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            adam_beta_1: 0.9,
            adam_beta_2: 0.999,
            adam_epsilon: 1e-5,
            weight_decay: None,
            grad_clip_norm: None,
            spatial_feature_radius: 100.0,
            output_activation: CnnOutputActivation::default(),
        }
    }
}

impl Default for LassoConfig {
    fn default() -> Self {
        Self {
            l1_reg: 1e-5,
            group_reg: 1e-5,
            n_iter: 500,
            tol: 1e-4,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            seed_only: true,
            mode: Some(CnnTrainingMode::Seed),
            epochs: 10,
            learning_rate: 1e-3,
            score_threshold: 0.2,
            hybrid: HybridCnnGatingConfig::default(),
            genes: None,
            max_genes: None,
        }
    }
}

/// Intersect AnnData `var` names with an optional allow-list, preserving `all_var_names` order.
pub fn filter_training_var_names(
    all_var_names: &[String],
    gene_filter: Option<&[String]>,
) -> Vec<String> {
    let mut v = all_var_names.to_vec();
    if let Some(filter) = gene_filter {
        v.retain(|g| filter.contains(g));
    }
    v
}

/// Filter (see [`filter_training_var_names`]) then cap length at `max_genes` when set.
pub fn resolve_training_target_genes(
    all_var_names: &[String],
    gene_filter: Option<&[String]>,
    max_genes: Option<usize>,
) -> Vec<String> {
    let mut v = filter_training_var_names(all_var_names, gene_filter);
    if let Some(n) = max_genes {
        if v.len() > n {
            v.truncate(n);
        }
    }
    v
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            n_parallel: 1,
            output_dir: String::new(),
            write_minimal_repro_h5ad: false,
            stale_lock_secs: 0,
        }
    }
}

impl Default for PerturbationConfig {
    fn default() -> Self {
        Self {
            beta_scale_factor: 1.0,
            beta_cap: None,
            n_propagation: 4,
            ligand_grid_factor: None,
            cells_csv: None,
            cells_csv_column: None,
        }
    }
}

impl Default for ModelExportConfig {
    fn default() -> Self {
        Self {
            save_cnn_weights: false,
            compressed_npz: true,
            output_subdir: "CNN_weights".into(),
        }
    }
}

/// Expand `~` / `~/` in a path string (HOME / USERPROFILE).
pub fn expand_user_path(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    if s == "~" {
        return std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| s.to_string());
    }
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(h) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return format!("{}/{}", h.trim_end_matches('/'), rest);
        }
    }
    s.to_string()
}

/// Strip a `file:` / `file://` URL prefix so pasted Finder / browser paths open correctly.
fn strip_file_url_prefix(s: &str) -> &str {
    let Some(rest) = s.strip_prefix("file:") else {
        return s;
    };
    let rest = rest.strip_prefix("//").unwrap_or(rest);
    if rest.is_empty() {
        return s;
    }
    if rest.starts_with('/') {
        return rest;
    }
    match rest.find('/') {
        Some(i) => &rest[i..],
        None => s,
    }
}

/// Normalize paths pasted into the spatial viewer (or similar UIs): trim, strip UTF-8 BOM,
/// optional wrapping quotes, `file://` URLs, then [`expand_user_path`].
pub fn normalize_ui_path(s: &str) -> String {
    let s = s.trim().trim_start_matches('\u{feff}').trim();
    let s = if s.len() >= 2 {
        let b = s.as_bytes();
        if (b[0] == b'"' && b[b.len() - 1] == b'"') || (b[0] == b'\'' && b[b.len() - 1] == b'\'') {
            s[1..s.len() - 1].trim()
        } else {
            s
        }
    } else {
        s
    };
    let s = strip_file_url_prefix(s.trim());
    expand_user_path(s)
}

#[cfg(test)]
mod normalize_ui_path_tests {
    use super::normalize_ui_path;

    #[test]
    fn file_triple_slash_unix() {
        assert_eq!(
            normalize_ui_path("file:///tmp/snrna_human_tonsil_v2.h5ad"),
            "/tmp/snrna_human_tonsil_v2.h5ad"
        );
    }

    #[test]
    fn file_localhost_unix() {
        assert_eq!(
            normalize_ui_path("file://localhost/tmp/a.h5ad"),
            "/tmp/a.h5ad"
        );
    }

    #[test]
    fn strips_wrapping_quotes() {
        assert_eq!(normalize_ui_path("  \"/tmp/x.h5ad\"  "), "/tmp/x.h5ad");
    }
}

/// Default training output directory: current working directory + `{adata_stem}_{YYYY-MM-DD}`.
/// `adata_stem` is the `.h5ad` file stem; `/` and `\\` in the stem are replaced with `_`.
pub fn default_output_dir_for_adata_path(adata_path: impl AsRef<Path>) -> anyhow::Result<String> {
    let adata_path = adata_path.as_ref();
    let stem = adata_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim();
    let stem = if stem.is_empty() {
        "spacetravlr_run"
    } else {
        stem
    };
    let stem: String = stem
        .chars()
        .map(|c| match c {
            '/' | '\\' | '\0' => '_',
            c => c,
        })
        .collect();

    let date = chrono::Local::now().format("%Y-%m-%d");
    let dir_name = format!("{}_{}", stem, date);
    let cwd =
        std::env::current_dir().context("default output_dir: could not read current directory")?;
    Ok(cwd.join(dir_name).to_string_lossy().to_string())
}

/// Sanitized `.h5ad` stem for canonical output filenames (same rules as [`default_output_dir_for_adata_path`]).
pub fn canonical_adata_stem(adata_path: &std::path::Path) -> String {
    let stem = adata_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim();
    let stem = if stem.is_empty() {
        "spacetravlr_run"
    } else {
        stem
    };
    stem.chars()
        .map(|c| match c {
            '/' | '\\' | '\0' => '_',
            c => c,
        })
        .collect()
}

impl SpaceshipConfig {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        let config: SpaceshipConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Load a run repro TOML and merge overlay fragments (`[data]`, `[perturbation]`, …) from
    /// `overlay_root`. Scalar/array keys in overlay tables replace; nested tables merge recursively.
    pub fn from_file_merged(
        run_repro_path: impl AsRef<Path>,
        overlay_root: Option<&toml::Value>,
    ) -> anyhow::Result<Self> {
        let path = run_repro_path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut root: toml::Value = toml::from_str(&text)
            .with_context(|| format!("parse SpaceshipConfig TOML {}", path.display()))?;
        if let Some(ov) = overlay_root {
            merge_spaceship_overlay_into_toml(&mut root, ov);
        }
        let merged_text = toml::to_string_pretty(&root)
            .map_err(|e| anyhow::anyhow!("serialize merged SpaceshipConfig TOML: {e}"))?;
        toml::from_str(&merged_text).with_context(|| {
            format!(
                "deserialize merged SpaceshipConfig from {} (after overlay)",
                path.display()
            )
        })
    }

    pub fn to_toml_pretty(&self) -> anyhow::Result<String> {
        toml::to_string_pretty(self).map_err(|e| anyhow::anyhow!("serialize config to TOML: {e}"))
    }

    pub fn write_run_repro_toml(&self, output_dir: &Path) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(output_dir)?;
        let text = self.to_toml_pretty()?;
        let path = output_dir.join(RUN_REPRO_TOML_FILENAME);
        std::fs::write(&path, text.as_str())?;
        let _ = std::fs::remove_file(output_dir.join("spacetravlr_run_config.toml"));
        Ok(path)
    }

    /// Write `spacetravlr_run_repro.toml` only if missing (first trainer on a shared output directory).
    pub fn write_run_repro_toml_if_missing(
        &self,
        output_dir: &Path,
    ) -> anyhow::Result<Option<PathBuf>> {
        std::fs::create_dir_all(output_dir)?;
        let path = output_dir.join(RUN_REPRO_TOML_FILENAME);
        if path.is_file() {
            return Ok(None);
        }
        let text = self.to_toml_pretty()?;
        std::fs::write(&path, text.as_str())?;
        let _ = std::fs::remove_file(output_dir.join("spacetravlr_run_config.toml"));
        Ok(Some(path))
    }

    pub fn discover_default_path() -> Option<PathBuf> {
        for name in &["spaceship_config.toml", "SpaceshipConfig.toml"] {
            let p = Path::new(name);
            if p.is_file() {
                return Some(p.to_path_buf());
            }
        }
        None
    }

    pub fn load() -> Self {
        let candidates = ["spaceship_config.toml", "SpaceshipConfig.toml"];
        for name in &candidates {
            if Path::new(name).exists() {
                match Self::from_file(name) {
                    Ok(cfg) => {
                        eprintln!("Loaded config from {}", name);
                        return cfg;
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to parse {}: {}", name, e);
                    }
                }
            }
        }
        Self::default()
    }

    pub fn resolved_cnn_mode(&self) -> CnnTrainingMode {
        self.training.mode.unwrap_or(CnnTrainingMode::Seed)
    }

    pub fn full_cnn(&self) -> bool {
        matches!(self.resolved_cnn_mode(), CnnTrainingMode::Full)
    }

    pub fn min_mean_lasso_r2_for_hybrid_cnn(&self) -> f64 {
        self.training
            .hybrid
            .min_mean_lasso_r2_for_cnn
            .unwrap_or(self.training.score_threshold)
    }

    pub fn resolve_adata_path(&self) -> String {
        self.data.adata_path.trim().to_string()
    }

    /// Training output directory (contains `*_betadata.feather`): `[execution].output_dir`.
    ///
    /// Relative `output_dir` entries are resolved against the directory that contains
    /// `run_toml_path` so a copied or symlinked repro TOML still finds feathers. If `output_dir`
    /// is empty, returns the parent of `run_toml_path` (legacy layout: TOML next to feathers).
    pub fn resolve_training_output_dir(&self, run_toml_path: &Path) -> PathBuf {
        let toml_dir = run_toml_path.parent().unwrap_or_else(|| Path::new("."));
        let raw = self.execution.output_dir.trim();
        if raw.is_empty() {
            return toml_dir.to_path_buf();
        }
        let expanded = expand_user_path(raw);
        let p = Path::new(expanded.as_str());
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            toml_dir.join(p)
        }
    }
}

#[cfg(test)]
mod resolve_training_output_dir_tests {
    use super::SpaceshipConfig;
    use std::path::Path;

    #[test]
    fn repro_toml_serializes_tf_priors_feather_when_some() {
        let mut cfg = SpaceshipConfig::default();
        cfg.grn.tf_priors_feather = Some("/data/priors.feather".into());
        let s = cfg.to_toml_pretty().unwrap();
        assert!(
            s.contains("tf_priors_feather") && s.contains("/data/priors.feather"),
            "repro TOML should record grn.tf_priors_feather for join / viewer: {s}"
        );
    }

    #[test]
    fn empty_output_dir_uses_toml_parent() {
        let mut cfg = SpaceshipConfig::default();
        cfg.execution.output_dir = String::new();
        let p = Path::new("/configs/x/spacetravlr_run_repro.toml");
        assert_eq!(cfg.resolve_training_output_dir(p), Path::new("/configs/x"));
    }

    #[test]
    fn relative_output_dir_joined_to_toml_parent() {
        let mut cfg = SpaceshipConfig::default();
        cfg.execution.output_dir = "lasso_out".into();
        let p = Path::new("/home/u/notebook.toml");
        assert_eq!(
            cfg.resolve_training_output_dir(p),
            Path::new("/home/u/lasso_out")
        );
    }

    #[test]
    fn repro_toml_roundtrip_training_genes_and_max_genes() {
        let mut cfg = SpaceshipConfig::default();
        cfg.training.genes = Some(vec!["Actb".into(), "Gapdh".into()]);
        cfg.training.max_genes = Some(128);
        let s = cfg.to_toml_pretty().unwrap();
        let back: SpaceshipConfig = toml::from_str(&s).expect("deserialize repro TOML");
        assert_eq!(
            back.training.genes,
            Some(vec!["Actb".into(), "Gapdh".into()])
        );
        assert_eq!(back.training.max_genes, Some(128));
    }

    #[test]
    fn repro_toml_deserialize_without_training_genes_defaults_none() {
        let toml = r#"
[data]
adata_path = "/tmp/x.h5ad"
layer = "X"
cluster_annot = "c"

[training]
mode = "seed"
epochs = 5
learning_rate = 0.001
score_threshold = 0.1
"#;
        let cfg: SpaceshipConfig = toml::from_str(toml).unwrap();
        assert!(cfg.training.genes.is_none());
        assert!(cfg.training.max_genes.is_none());
    }
}

#[cfg(test)]
mod training_target_genes_tests {
    use super::{filter_training_var_names, resolve_training_target_genes};

    fn vars() -> Vec<String> {
        vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()]
    }

    #[test]
    fn filter_none_keeps_order_and_len() {
        let v = vars();
        let out = filter_training_var_names(&v, None);
        assert_eq!(out, v);
    }

    #[test]
    fn filter_preserves_var_order() {
        let v = vars();
        let f = vec!["c".into(), "a".into()];
        let out = filter_training_var_names(&v, Some(&f));
        assert_eq!(out, vec!["a", "c"]);
    }

    #[test]
    fn filter_empty_list_yields_empty() {
        let v = vars();
        let f: Vec<String> = vec![];
        let out = filter_training_var_names(&v, Some(&f));
        assert!(out.is_empty());
    }

    #[test]
    fn resolve_cap_only_truncates_prefix_in_var_order() {
        let v = vars();
        let out = resolve_training_target_genes(&v, None, Some(3));
        assert_eq!(out, vec!["a", "b", "c"]);
    }

    #[test]
    fn resolve_filter_then_cap() {
        let v = vars();
        let f = vec!["e".into(), "b".into(), "a".into(), "d".into()];
        let out = resolve_training_target_genes(&v, Some(&f), Some(2));
        assert_eq!(out, vec!["a", "b"]);
    }

    #[test]
    fn resolve_cap_larger_than_filtered_no_op() {
        let v = vars();
        let f = vec!["b".into(), "c".into()];
        let out = resolve_training_target_genes(&v, Some(&f), Some(10));
        assert_eq!(out, vec!["b", "c"]);
    }

    #[test]
    fn resolve_matches_sequential_filter_and_truncate() {
        let v = vars();
        let f = vec!["d".into(), "b".into()];
        let mut manual = filter_training_var_names(&v, Some(&f));
        manual.truncate(1);
        let resolved = resolve_training_target_genes(&v, Some(&f), Some(1));
        assert_eq!(resolved, manual);
        assert_eq!(resolved, vec!["b"]);
    }
}

#[cfg(test)]
mod merge_spaceship_overlay_tests {
    use super::{SpaceshipConfig, merge_spaceship_overlay_into_toml};
    use std::path::PathBuf;

    fn tmp_run_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "stlr_cfg_merge_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn overlay_replaces_nested_perturbation_fields() {
        let base = r#"
[data]
adata_path = "/data/a.h5ad"
layer = "imputed_count"
cluster_annot = "ct"

[perturbation]
n_propagation = 2
beta_scale_factor = 1.0
"#;
        let overlay = r#"
[perturbation]
n_propagation = 9
"#;
        let mut root: toml::Value = toml::from_str(base).unwrap();
        let ov: toml::Value = toml::from_str(overlay).unwrap();
        merge_spaceship_overlay_into_toml(&mut root, &ov);
        let cfg: SpaceshipConfig =
            toml::from_str(&toml::to_string_pretty(&root).unwrap()).unwrap();
        assert_eq!(cfg.perturbation.n_propagation, 9);
        assert_eq!(cfg.perturbation.beta_scale_factor, 1.0);
        assert_eq!(cfg.data.layer, "imputed_count");
    }

    #[test]
    fn from_file_merged_matches_manual_merge() {
        let tmp = tmp_run_dir();
        let repro = tmp.join("spacetravlr_run_repro.toml");
        let body = r#"
[data]
adata_path = "/x.h5ad"
layer = "L0"
cluster_annot = "c0"

[perturbation]
n_propagation = 3
"#;
        std::fs::write(&repro, body).unwrap();
        let overlay: toml::Value = toml::from_str(
            r#"
[perturbation]
n_propagation = 7
[data]
layer = "L1"
"#,
        )
        .unwrap();
        let merged = SpaceshipConfig::from_file_merged(&repro, Some(&overlay)).unwrap();
        assert_eq!(merged.perturbation.n_propagation, 7);
        assert_eq!(merged.data.layer, "L1");
        assert_eq!(merged.data.cluster_annot, "c0");

        let round = SpaceshipConfig::from_file_merged(&repro, None).unwrap();
        assert_eq!(round.perturbation.n_propagation, 3);
        assert_eq!(round.data.layer, "L0");
    }

    #[test]
    fn resolve_training_output_dir_uses_merged_execution() {
        let tmp = tmp_run_dir();
        let repro = tmp.join("spacetravlr_run_repro.toml");
        std::fs::write(
            &repro,
            r#"
[data]
adata_path = "/d.h5ad"

[execution]
output_dir = "out_a"
"#,
        )
        .unwrap();
        let overlay: toml::Value = toml::from_str("[execution]\noutput_dir = \"out_b\"\n").unwrap();
        let cfg = SpaceshipConfig::from_file_merged(&repro, Some(&overlay)).unwrap();
        let dir = cfg.resolve_training_output_dir(repro.as_path());
        assert_eq!(dir, tmp.join("out_b"));
    }
}
