use crate::config::expand_user_path;
use anyhow::{Context, Result};
use polars::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment variable: directory containing `mouse_network.parquet` / `human_network.parquet`.
pub const SPACETRAVLR_DATA_DIR_ENV: &str = "SPACETRAVLR_DATA_DIR";

/// Subfolder under `[execution].output_dir` where GRN parquets are auto-downloaded (see `scripts/install.sh`).
pub const RUN_NETWORK_SUBDIR: &str = "network";

const DEFAULT_GITHUB_REPO: &str = "Koushul/SpaceTravLR_rust";
const GRN_PARQUET_MIN_BYTES: u64 = 100_000;
const GRN_NETWORK_PARQUETS: [&str; 2] = ["human_network.parquet", "mouse_network.parquet"];

fn push_tried(tried: &mut Vec<String>, p: &Path) {
    tried.push(p.display().to_string());
}

fn try_file_path(path: PathBuf, tried: &mut Vec<String>) -> Option<PathBuf> {
    push_tried(tried, &path);
    if path.is_file() { Some(path) } else { None }
}

fn grn_parquet_ready(path: &Path) -> bool {
    path.is_file()
        && fs::metadata(path)
            .map(|m| m.len() >= GRN_PARQUET_MIN_BYTES)
            .unwrap_or(false)
}

fn github_repo_slug() -> String {
    std::env::var("SPACETRAVLR_GITHUB_REPO")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_GITHUB_REPO.to_string())
}

fn download_grn_parquet_url(filename: &str, dest: &Path) -> Result<()> {
    let repo = github_repo_slug();
    let version_tag = format!("v{}", env!("CARGO_PKG_VERSION"));
    let part = dest.with_extension("part");
    if part.exists() {
        fs::remove_file(&part).ok();
    }
    let mut last_err: Option<anyhow::Error> = None;
    for ref_name in [version_tag.as_str(), "main"] {
        let url = format!("https://raw.githubusercontent.com/{repo}/{ref_name}/data/{filename}");
        match download_url_to_path(&url, &part) {
            Ok(()) if grn_parquet_ready(&part) => {
                fs::rename(&part, dest)
                    .with_context(|| format!("install {}", dest.display()))?;
                return Ok(());
            }
            Ok(()) => {
                fs::remove_file(&part).ok();
                last_err = Some(anyhow::anyhow!(
                    "downloaded {filename} from {url} was too small (< {GRN_PARQUET_MIN_BYTES} bytes)"
                ));
            }
            Err(e) => {
                fs::remove_file(&part).ok();
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no download attempts for {filename}")))
}

fn download_url_to_path(url: &str, path: &Path) -> Result<()> {
    let st = Command::new("curl")
        .args(["-fsSL", url, "-o"])
        .arg(path)
        .status()
        .with_context(|| format!("spawn curl to fetch {url}"))?;
    if st.success() {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "curl failed for {url} (exit {:?}); install curl or copy GRN parquets manually",
        st.code()
    ))
}

/// `{output_dir}/network` — auto-populated by [`ensure_grn_network_parquets`] when missing.
pub fn run_network_data_dir(output_dir: &Path) -> PathBuf {
    output_dir.join(RUN_NETWORK_SUBDIR)
}

/// Download `human_network.parquet` and `mouse_network.parquet` into `{output_dir}/network/`
/// (same files as `scripts/install.sh` → `data/`). Skips files that already look valid.
pub fn ensure_grn_network_parquets(output_dir: &Path) -> Result<PathBuf> {
    let dir = run_network_data_dir(output_dir);
    fs::create_dir_all(&dir)
        .with_context(|| format!("create {}", dir.display()))?;
    for filename in GRN_NETWORK_PARQUETS {
        let dest = dir.join(filename);
        if grn_parquet_ready(&dest) {
            continue;
        }
        eprintln!(
            "spacetravlr: downloading {filename} into {} …",
            dir.display()
        );
        download_grn_parquet_url(filename, &dest).with_context(|| {
            format!(
                "download GRN parquet {filename} into {}",
                dir.display()
            )
        })?;
    }
    Ok(dir)
}

/// Resolve GRN parquet search path for a training run: config/env/cwd first, then download into
/// `{output_dir}/network/` when nothing is found (mirrors `scripts/install.sh`).
pub fn resolve_or_fetch_network_data_dir(
    species: &str,
    output_dir: &Path,
    config_network_data_dir: Option<&str>,
) -> Result<Option<String>> {
    let filename = format!("{species}_network.parquet");

    if let Some(d) = config_network_data_dir.map(str::trim).filter(|s| !s.is_empty()) {
        let expanded = expand_user_path(d);
        if grn_parquet_ready(&Path::new(&expanded).join(&filename)) {
            return Ok(Some(expanded));
        }
    }

    let run_net = run_network_data_dir(output_dir);
    if grn_parquet_ready(&run_net.join(&filename)) {
        let dir_s = run_net
            .to_str()
            .with_context(|| format!("network dir path must be UTF-8: {}", run_net.display()))?
            .to_string();
        return Ok(Some(dir_s));
    }

    if resolve_species_network_parquet(species, None).is_ok() {
        return Ok(None);
    }

    eprintln!(
        "spacetravlr: GRN network not found for species {species:?}; fetching parquets into {} …",
        run_net.display()
    );
    let dir = ensure_grn_network_parquets(output_dir)?;
    let dir_s = dir
        .to_str()
        .with_context(|| format!("network dir path must be UTF-8: {}", dir.display()))?
        .to_string();
    resolve_species_network_parquet(species, Some(dir_s.as_str()))
        .with_context(|| format!("GRN parquet for {species} missing after download"))?;
    Ok(Some(dir_s))
}

/// Resolve `{species}_network.parquet` using config override, env, exe-relative paths, and cwd
/// ancestors. Prebuilt binaries do **not** embed a repo path; ship or download `data/*.parquet`
/// and set `SPACETRAVLR_DATA_DIR` / `[grn].network_data_dir` (see error text when missing).
pub fn resolve_species_network_parquet(
    species: &str,
    config_network_data_dir: Option<&str>,
) -> anyhow::Result<PathBuf> {
    let filename = format!("{}_network.parquet", species);
    let mut tried: Vec<String> = Vec::new();

    if let Some(dir) = config_network_data_dir
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let base = PathBuf::from(expand_user_path(dir));
        let candidate = base.join(&filename);
        if let Some(p) = try_file_path(candidate, &mut tried) {
            return Ok(p);
        }
    }

    if let Ok(dir) = std::env::var(SPACETRAVLR_DATA_DIR_ENV) {
        let dir = dir.trim();
        if !dir.is_empty() {
            let candidate = PathBuf::from(expand_user_path(dir)).join(&filename);
            if let Some(p) = try_file_path(candidate, &mut tried) {
                return Ok(p);
            }
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for rel in ["data", "../data"] {
                let candidate = parent.join(rel).join(&filename);
                if let Some(p) = try_file_path(candidate, &mut tried) {
                    return Ok(p);
                }
            }
        }
    }

    let mut dir = std::env::current_dir().unwrap_or_default();
    for _ in 0..10 {
        let candidate = dir.join("data").join(&filename);
        if let Some(p) = try_file_path(candidate, &mut tried) {
            return Ok(p);
        }
        if !dir.pop() {
            break;
        }
    }

    let cwd_rel = Path::new("data").join(&filename);
    if let Some(p) = try_file_path(cwd_rel, &mut tried) {
        return Ok(p);
    }

    anyhow::bail!(
        "Could not find GRN network file {:?} for species {:?}. \
Prebuilt binaries do not bundle these files; training runs download into {{output_dir}}/{RUN_NETWORK_SUBDIR}/ automatically when curl is available, or run `scripts/install.sh` (downloads into `data/` next to the binary), copy `human_network.parquet` / `mouse_network.parquet` from the SpaceTravLR_rust repo `data/`, or set [{}] / [grn].network_data_dir. \
You can also place a `data/` folder next to the executable (or walk up from the current cwd). Tried:\n  {}",
        filename,
        species,
        SPACETRAVLR_DATA_DIR_ENV,
        tried.join("\n  ")
    );
}

#[derive(Clone, Serialize)]
pub struct Modulators {
    pub regulators: Vec<String>,
    pub ligands: Vec<String>,
    pub receptors: Vec<String>,
    pub tfl_ligands: Vec<String>,
    pub tfl_regulators: Vec<String>,
    pub lr_pairs: Vec<String>,
    pub tfl_pairs: Vec<String>,
}

impl Modulators {
    pub fn apply_modulator_mask(
        mut self,
        use_tf_modulators: bool,
        use_lr_modulators: bool,
        use_tfl_modulators: bool,
    ) -> Self {
        if !use_tf_modulators {
            self.regulators.clear();
        }
        if !use_lr_modulators {
            self.ligands.clear();
            self.receptors.clear();
            self.lr_pairs.clear();
        }
        if !use_tfl_modulators {
            self.tfl_ligands.clear();
            self.tfl_regulators.clear();
            self.tfl_pairs.clear();
        }
        self
    }
}

#[derive(Clone, Default)]
pub struct TfPriors {
    by_target_cell_type: HashMap<String, HashMap<String, Vec<String>>>,
    by_target_any: HashMap<String, Vec<String>>,
}

impl TfPriors {
    pub fn from_feather(path: &str, var_names: &[String]) -> Result<Self> {
        let priors_path = expand_user_path(path);
        let df = LazyFrame::scan_ipc(
            polars_utils::plpath::PlPath::from_string(priors_path.clone()),
            ScanArgsIpc::default(),
        )
        .with_context(|| format!("scan_ipc TF priors {:?}", priors_path))?
        .collect()
        .with_context(|| format!("read TF priors {:?}", priors_path))?;

        for req in ["source", "target", "cell_type"] {
            if df.column(req).is_err() {
                anyhow::bail!(
                    "TF priors file {:?} missing required column {:?}. Expected columns: source, target, cell_type.",
                    priors_path,
                    req
                );
            }
        }

        let source_s = df.column("source")?.cast(&DataType::String)?;
        let target_s = df.column("target")?.cast(&DataType::String)?;
        let cell_type_s = df.column("cell_type")?.cast(&DataType::String)?;
        let source = source_s.str()?;
        let target = target_s.str()?;
        let cell_type = cell_type_s.str()?;

        let var_set: HashSet<&str> = var_names.iter().map(|s| s.as_str()).collect();

        let mut by_target_cell_type: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
        let mut by_target_any: HashMap<String, Vec<String>> = HashMap::new();
        let mut seen_tct: HashSet<(String, String, String)> = HashSet::new();
        let mut seen_tgt: HashSet<(String, String)> = HashSet::new();

        for i in 0..df.height() {
            let Some(src) = source.get(i).map(str::trim) else {
                continue;
            };
            let Some(tgt) = target.get(i).map(str::trim) else {
                continue;
            };
            let Some(ct) = cell_type.get(i).map(str::trim) else {
                continue;
            };
            if src.is_empty() || tgt.is_empty() || ct.is_empty() {
                continue;
            }
            if !var_set.contains(src) || !var_set.contains(tgt) {
                continue;
            }

            let src_s = src.to_string();
            let tgt_s = tgt.to_string();
            let ct_s = ct.to_string();

            if seen_tgt.insert((tgt_s.clone(), src_s.clone())) {
                by_target_any
                    .entry(tgt_s.clone())
                    .or_default()
                    .push(src_s.clone());
            }
            if seen_tct.insert((tgt_s.clone(), ct_s.clone(), src_s.clone())) {
                by_target_cell_type
                    .entry(tgt_s)
                    .or_default()
                    .entry(ct_s)
                    .or_default()
                    .push(src_s);
            }
        }

        Ok(Self {
            by_target_cell_type,
            by_target_any,
        })
    }

    pub fn tfs_for_target_any(&self, target: &str) -> Option<&Vec<String>> {
        self.by_target_any.get(target)
    }

    pub fn tfs_for_target_cell_type(&self, target: &str, cell_type: &str) -> Option<&Vec<String>> {
        self.by_target_cell_type
            .get(target)
            .and_then(|m| m.get(cell_type))
    }
}

#[derive(Clone)]
pub struct GeneNetwork {
    pub species: String,
    pub network_path: String,
    pub network_df: DataFrame,
}

/// Infer **`human`** vs **`mouse`** from `var_names` using **gene symbol capitalization**
/// (HGNC-style all caps vs MGI-style title case), **Ensembl ID prefixes**, and common **mouse
/// RIKEN / mt-** patterns. Scans **all** symbols (deterministic; no random subsampling).
///
/// Returns **`None`** when the list is empty or scores tie / no symbol matched any rule (set
/// `[data].spatial_species` / `--spatial-species` / `--celloracle-species` explicitly).
pub fn infer_species(var_names: &[String]) -> Option<&'static str> {
    if var_names.is_empty() {
        return None;
    }

    let mut mouse = 0i64;
    let mut human = 0i64;

    for gene in var_names {
        let g = gene.trim();
        if g.is_empty() {
            continue;
        }

        let upper = g.to_ascii_uppercase();
        let base = upper.split('.').next().unwrap_or(&upper);

        if base.starts_with("ENSMUSG") {
            mouse += 100;
            continue;
        }
        if base.starts_with("ENSG") {
            human += 100;
            continue;
        }

        if upper.ends_with("RIK") {
            mouse += 5;
            continue;
        }

        let gl = g.to_ascii_lowercase();
        if gl.starts_with("mt-") || gl.starts_with("mt_") {
            mouse += 3;
            continue;
        }

        let letters: Vec<char> = g.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if letters.len() < 2 {
            continue;
        }

        let all_caps = letters.iter().all(|c| c.is_ascii_uppercase());
        let title_mouse =
            letters[0].is_ascii_uppercase() && letters[1..].iter().all(|c| c.is_ascii_lowercase());

        if all_caps {
            human += 1;
        } else if title_mouse {
            mouse += 1;
        }
    }

    if mouse > human {
        Some("mouse")
    } else if human > mouse {
        Some("human")
    } else {
        None
    }
}

/// Keep only L–R rows whose ligand ranks in the top `max_ligands` by `gene_mean_expression`
/// (descending mean; tie-break lexicographic on ligand name). `max_ligands` must be `Some(k)` with `k > 0`.
pub(crate) fn apply_max_ligands_filter(
    ligands: &mut Vec<String>,
    receptors: &mut Vec<String>,
    lr_pairs: &mut Vec<String>,
    max_ligands: Option<usize>,
    gene_mean_expression: &HashMap<String, f64>,
) {
    let n = lr_pairs.len();
    if n == 0 {
        return;
    }
    let Some(k_raw) = max_ligands else {
        return;
    };
    if k_raw == 0 {
        return;
    }
    let k = k_raw.max(1);
    let mut unique: Vec<String> = ligands
        .iter()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique.sort_by(|a, b| {
        let ma = gene_mean_expression.get(a.as_str()).copied().unwrap_or(0.0);
        let mb = gene_mean_expression.get(b.as_str()).copied().unwrap_or(0.0);
        mb.partial_cmp(&ma)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    let take_n = k.min(unique.len());
    let allowed: HashSet<String> = unique.into_iter().take(take_n).collect();
    let mut triples: Vec<(String, String, String)> = Vec::new();
    for i in 0..n {
        if allowed.contains(&ligands[i]) {
            triples.push((
                ligands[i].clone(),
                receptors[i].clone(),
                lr_pairs[i].clone(),
            ));
        }
    }
    triples.sort_by(|a, b| a.2.cmp(&b.2));
    *ligands = triples.iter().map(|t| t.0.clone()).collect();
    *receptors = triples.iter().map(|t| t.1.clone()).collect();
    *lr_pairs = triples.into_iter().map(|t| t.2).collect();
}

impl GeneNetwork {
    pub fn new(
        species: &str,
        var_names: &[String],
        network_data_dir: Option<&str>,
    ) -> Result<Self> {
        let path = resolve_species_network_parquet(species, network_data_dir)
            .with_context(|| format!("load GRN for species {:?}", species))?;
        let network_path = path.to_string_lossy().into_owned();

        let full_df = LazyFrame::scan_parquet(
            polars_utils::plpath::PlPath::from_string(network_path.clone()),
            ScanArgsParquet::default(),
        )
        .with_context(|| format!("scan_parquet {:?}", network_path))?
        .collect()
        .with_context(|| format!("read GRN parquet {:?}", network_path))?;

        let mut source_keep = Vec::new();
        let var_names_set: HashSet<&str> = var_names.iter().map(|s| s.as_str()).collect();

        if let (Ok(s_col), Ok(t_col)) = (
            full_df.column("source")?.cast(&DataType::String)?.str(),
            full_df.column("target")?.cast(&DataType::String)?.str(),
        ) {
            for (s, t) in s_col.into_no_null_iter().zip(t_col.into_no_null_iter()) {
                source_keep.push(var_names_set.contains(s) && var_names_set.contains(t));
            }
        }

        let filter_chunk = BooleanChunked::new("".into(), &source_keep);
        let network_df = full_df.filter(&filter_chunk)?;

        Ok(Self {
            species: species.to_string(),
            network_path,
            network_df,
        })
    }

    pub fn get_modulators(
        &self,
        target_gene: &str,
        tf_ligand_cutoff: f64,
        max_ligands: Option<usize>,
        gene_mean_expression: Option<&HashMap<String, f64>>,
    ) -> Result<Modulators> {
        let lf = self.network_df.clone().lazy();

        // --- 1. Regulators (edge_type == "grn") ---
        let grn_df = lf
            .clone()
            .filter(
                col("edge_type")
                    .cast(DataType::String)
                    .eq(lit("grn"))
                    .and(col("target").cast(DataType::String).eq(lit(target_gene))),
            )
            .select([col("source")])
            .collect()?;

        let mut regulators = Vec::new();
        if let Ok(src) = grn_df.column("source")?.cast(&DataType::String)?.str() {
            let mut seen = HashSet::new();
            for v in src.into_no_null_iter() {
                if v != target_gene && seen.insert(v.to_string()) {
                    regulators.push(v.to_string());
                }
            }
        }

        // --- 2. LR Pairs (edge_type == "lr") ---
        let lr_df = lf
            .clone()
            .filter(col("edge_type").cast(DataType::String).eq(lit("lr")))
            .select([col("source"), col("target")])
            .collect()?;

        let mut ligands = Vec::new();
        let mut receptors = Vec::new();
        let mut lr_pairs = Vec::new();

        if let (Ok(l_col), Ok(r_col)) = (
            lr_df.column("source")?.cast(&DataType::String)?.str(),
            lr_df.column("target")?.cast(&DataType::String)?.str(),
        ) {
            let mut seen_pairs = HashSet::new();
            for (l, r) in l_col.into_no_null_iter().zip(r_col.into_no_null_iter()) {
                if l == target_gene || r == target_gene {
                    continue;
                }
                let pair = format!("{}${}", l, r);
                if seen_pairs.insert(pair.clone()) {
                    ligands.push(l.to_string());
                    receptors.push(r.to_string());
                    lr_pairs.push(pair);
                }
            }
        }

        if let Some(k) = max_ligands {
            if k > 0 && gene_mean_expression.is_none() {
                anyhow::bail!(
                    "max_ligands={k} requires per-gene mean expression (from [data].layer); gene_mean_expression is missing"
                );
            }
        }
        if let (Some(means), Some(k)) = (gene_mean_expression, max_ligands) {
            if k > 0 {
                apply_max_ligands_filter(
                    &mut ligands,
                    &mut receptors,
                    &mut lr_pairs,
                    max_ligands,
                    means,
                );
            }
        }

        let (tfl_ligands, tfl_regulators, tfl_pairs) =
            self.nichenet_tfl_pairs_for_regulators(&regulators, tf_ligand_cutoff, &ligands)?;

        Ok(Modulators {
            regulators,
            ligands,
            receptors,
            tfl_ligands,
            tfl_regulators,
            lr_pairs,
            tfl_pairs,
        })
    }

    /// NicheNet ligand–TF modulator triples for `regulators` (TFs), using `lr_ligands` as the
    /// allowed ligand pool — same rule as inside [`GeneNetwork::get_modulators`].
    pub fn nichenet_tfl_pairs_for_regulators(
        &self,
        regulators: &[String],
        tf_ligand_cutoff: f64,
        lr_ligands: &[String],
    ) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
        if regulators.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let lf = self.network_df.clone().lazy();
        let nn_df = lf
            .filter(
                col("edge_type")
                    .cast(DataType::String)
                    .eq(lit("nichenet"))
                    .and(
                        col("weight")
                            .cast(DataType::Float64)
                            .gt(lit(tf_ligand_cutoff)),
                    ),
            )
            .select([col("source"), col("target"), col("weight")])
            .collect()?;

        let mut tfl_ligands = Vec::new();
        let mut tfl_regulators = Vec::new();
        let mut tfl_pairs = Vec::new();

        if let (Ok(l_col), Ok(tf_col), Ok(w_col)) = (
            nn_df.column("source")?.cast(&DataType::String)?.str(),
            nn_df.column("target")?.cast(&DataType::String)?.str(),
            nn_df.column("weight")?.cast(&DataType::Float64)?.f64(),
        ) {
            let ligands_set: HashSet<&String> = lr_ligands.iter().collect();
            let regs_set: HashSet<&String> = regulators.iter().collect();

            let mut tf_candidates: HashMap<String, Vec<(String, f64)>> = HashMap::new();

            for i in 0..nn_df.height() {
                if let (Some(l), Some(tf), Some(w)) = (l_col.get(i), tf_col.get(i), w_col.get(i)) {
                    let l_string = l.to_string();
                    let tf_string = tf.to_string();
                    if ligands_set.contains(&l_string) && regs_set.contains(&tf_string) {
                        tf_candidates
                            .entry(tf_string)
                            .or_default()
                            .push((l_string, w));
                    }
                }
            }

            for reg in regulators.iter() {
                if let Some(mut candidates) = tf_candidates.remove(reg) {
                    candidates
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    for (l, _w) in candidates.into_iter().take(5) {
                        tfl_ligands.push(l.clone());
                        tfl_regulators.push(reg.clone());
                        tfl_pairs.push(format!("{}#{}", l, reg));
                    }
                }
            }
        }

        Ok((tfl_ligands, tfl_regulators, tfl_pairs))
    }

    /// All GRN edges (`edge_type == "grn"`) as **target → deduplicated regulator list** (TF-only priors).
    ///
    /// Self-loops are omitted. Order follows the parquet scan (stable, not sorted).
    pub fn grn_regulators_by_target(&self) -> Result<HashMap<String, Vec<String>>> {
        let lf = self
            .network_df
            .clone()
            .lazy()
            .filter(col("edge_type").cast(DataType::String).eq(lit("grn")))
            .select([col("source"), col("target")]);
        let df = lf.collect()?;

        let mut by_target: HashMap<String, Vec<String>> = HashMap::new();
        let src_series = df.column("source")?.cast(&DataType::String)?;
        let tgt_series = df.column("target")?.cast(&DataType::String)?;
        let (Ok(s_col), Ok(t_col)) = (src_series.str(), tgt_series.str()) else {
            return Ok(by_target);
        };

        for i in 0..df.height() {
            let Some(src) = s_col.get(i).map(str::trim) else {
                continue;
            };
            let Some(tgt) = t_col.get(i).map(str::trim) else {
                continue;
            };
            if src.is_empty() || tgt.is_empty() {
                continue;
            }
            if src == tgt {
                continue;
            }
            let entry = by_target.entry(tgt.to_string()).or_default();
            if !entry.iter().any(|s| s == src) {
                entry.push(src.to_string());
            }
        }

        Ok(by_target)
    }

    /// GRN edges for **CellOracle / TF-prior** inference: only **TF→TF** links derived from the
    /// species `*_network.parquet`.
    ///
    /// A gene is treated as a TF if it appears at least once as **`source`** on an `edge_type ==
    /// "grn"` row (same convention as curated GRN files). Targets of the Bayesian ridge step are
    /// restricted to that TF set, and each target’s regulators are restricted to TFs as well, so
    /// ligand–receptor / NicheNet rows never enter this map.
    pub fn grn_celloracle_tf_regulators_by_target(&self) -> Result<HashMap<String, Vec<String>>> {
        let lf = self
            .network_df
            .clone()
            .lazy()
            .filter(col("edge_type").cast(DataType::String).eq(lit("grn")))
            .select([col("source"), col("target")]);
        let df = lf.collect()?;

        let src_series = df.column("source")?.cast(&DataType::String)?;
        let tgt_series = df.column("target")?.cast(&DataType::String)?;
        let (Ok(s_col), Ok(t_col)) = (src_series.str(), tgt_series.str()) else {
            return Ok(HashMap::new());
        };

        let mut tf_set: HashSet<String> = HashSet::new();
        for i in 0..df.height() {
            if let Some(src) = s_col.get(i).map(str::trim).filter(|s| !s.is_empty()) {
                tf_set.insert(src.to_string());
            }
        }

        let mut by_target: HashMap<String, Vec<String>> = HashMap::new();
        for i in 0..df.height() {
            let Some(src) = s_col.get(i).map(str::trim) else {
                continue;
            };
            let Some(tgt) = t_col.get(i).map(str::trim) else {
                continue;
            };
            if src.is_empty() || tgt.is_empty() || src == tgt {
                continue;
            }
            if !tf_set.contains(tgt) {
                continue;
            }
            debug_assert!(tf_set.contains(src));
            let entry = by_target.entry(tgt.to_string()).or_default();
            if !entry.iter().any(|s| s == src) {
                entry.push(src.to_string());
            }
        }

        Ok(by_target)
    }

    /// Union of LR ligands on edges whose endpoints are both in `var_names`, for dataset-wide caches.
    /// Applies [`apply_max_ligands_filter`] when `max_ligands` is `Some(k)` with `k > 0`.
    pub fn dataset_union_lr_ligands(
        &self,
        var_names: &[String],
        max_ligands: Option<usize>,
        gene_mean_expression: Option<&HashMap<String, f64>>,
    ) -> Result<Vec<String>> {
        let var_set: HashSet<&str> = var_names.iter().map(|s| s.as_str()).collect();
        let lf = self.network_df.clone().lazy();
        let lr_df = lf
            .filter(col("edge_type").cast(DataType::String).eq(lit("lr")))
            .select([col("source"), col("target")])
            .collect()?;

        let mut ligands = Vec::new();
        let mut receptors = Vec::new();
        let mut lr_pairs = Vec::new();

        if let (Ok(l_col), Ok(r_col)) = (
            lr_df.column("source")?.cast(&DataType::String)?.str(),
            lr_df.column("target")?.cast(&DataType::String)?.str(),
        ) {
            let mut seen_pairs = HashSet::new();
            for (l, r) in l_col.into_no_null_iter().zip(r_col.into_no_null_iter()) {
                if !var_set.contains(l) || !var_set.contains(r) {
                    continue;
                }
                let pair = format!("{}${}", l, r);
                if seen_pairs.insert(pair.clone()) {
                    ligands.push(l.to_string());
                    receptors.push(r.to_string());
                    lr_pairs.push(pair);
                }
            }
        }

        if let Some(k) = max_ligands {
            if k > 0 && gene_mean_expression.is_none() {
                anyhow::bail!(
                    "max_ligands={k} requires per-gene mean expression (from [data].layer); gene_mean_expression is missing"
                );
            }
        }
        if let (Some(means), Some(k)) = (gene_mean_expression, max_ligands) {
            if k > 0 {
                apply_max_ligands_filter(
                    &mut ligands,
                    &mut receptors,
                    &mut lr_pairs,
                    max_ligands,
                    means,
                );
            }
        }

        Ok(ligands)
    }

    /// Curated `ligand$receptor` keys for `edge_type == "lr"` (as in betadata column stems).
    pub fn all_lr_pair_keys(&self) -> Result<HashSet<String>> {
        let lr_df = self
            .network_df
            .clone()
            .lazy()
            .filter(col("edge_type").cast(DataType::String).eq(lit("lr")))
            .select([col("source"), col("target")])
            .collect()?;
        let mut out = HashSet::new();
        let l_s = lr_df.column("source")?.cast(&DataType::String)?;
        let r_s = lr_df.column("target")?.cast(&DataType::String)?;
        let (Ok(l_col), Ok(r_col)) = (l_s.str(), r_s.str()) else {
            return Ok(out);
        };
        for (l, r) in l_col.into_no_null_iter().zip(r_col.into_no_null_iter()) {
            out.insert(format!("{}${}", l, r));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn grn_celloracle_tf_map_omits_non_tf_targets_and_keeps_tf_tf_edges() {
        use polars::prelude::{DataFrame, NamedFrom, ParquetWriter, Series};

        let dir = std::env::temp_dir().join(format!("st_net_co_tf_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut df = DataFrame::new(vec![
            Series::new("source".into(), vec!["TFa", "TFa", "TFb"]).into(),
            Series::new("target".into(), vec!["TFb", "PassiveGene", "TFa"]).into(),
            Series::new("edge_type".into(), vec!["grn", "grn", "grn"]).into(),
            Series::new("weight".into(), vec![1.0_f64, 1.0, 1.0]).into(),
        ])
        .unwrap();
        let path = dir.join("mouse_network.parquet");
        let f = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(f).finish(&mut df).unwrap();

        let var_names = vec!["TFa".into(), "TFb".into(), "PassiveGene".into()];
        let net = GeneNetwork::new("mouse", &var_names, Some(dir.to_str().unwrap())).unwrap();
        let all = net.grn_regulators_by_target().unwrap();
        assert!(
            all.contains_key("PassiveGene"),
            "full GRN includes targets that are never GRN sources"
        );

        let co = net.grn_celloracle_tf_regulators_by_target().unwrap();
        assert!(
            !co.contains_key("PassiveGene"),
            "CellOracle map should skip targets not in TF (source) set"
        );
        assert_eq!(co.get("TFb").unwrap(), &vec!["TFa".to_string()]);
        assert_eq!(co.get("TFa").unwrap(), &vec!["TFb".to_string()]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn infer_species_mouse_genes() {
        let genes: Vec<String> = vec![
            "Gapdh", "Actb", "Sox2", "Pou5f1", "Nanog", "Klf4", "Myc", "Bmp4", "Fgf2", "Wnt3a",
            "Shh", "Notch1", "Dll1", "Jag1", "Hes1",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_human_genes() {
        let genes: Vec<String> = vec![
            "GAPDH", "ACTB", "SOX2", "POU5F1", "NANOG", "KLF4", "MYC", "BMP4", "FGF2", "WNT3A",
            "SHH", "NOTCH1", "DLL1", "JAG1", "HES1",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), Some("human"));
    }

    #[test]
    fn infer_species_mixed_defaults_to_majority() {
        // Mostly mouse-style
        let genes: Vec<String> = vec![
            "Gapdh", "Actb", "Sox2", "Pou5f1", "Nanog", "Klf4", "Myc", "BRCA1", "TP53",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_empty_is_none() {
        let genes: Vec<String> = vec![];
        assert_eq!(infer_species(&genes), None);
    }

    #[test]
    fn infer_species_numeric_genes_is_none() {
        let genes: Vec<String> = vec!["123", "456", "789"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(infer_species(&genes), None);
    }

    #[test]
    fn infer_species_ensembl_prefers_mouse() {
        let genes = vec!["ENSMUSG00000000001".to_string(), "GAPDH".to_string()];
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_ensembl_prefers_human() {
        let genes = vec!["ENSG00000139618".to_string(), "Gapdh".to_string()];
        assert_eq!(infer_species(&genes), Some("human"));
    }

    #[test]
    fn infer_species_riken_mouse() {
        let genes: Vec<String> = vec!["1700001C02Rik", "A230050P20Rik", "Gapdh", "BRCA1"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_mt_prefix_mouse() {
        let genes: Vec<String> = vec!["mt-Co1", "mt-Nd1", "mt-Cytb", "ACTB", "SOX2"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_human_with_mt_uppercase() {
        let genes: Vec<String> = vec![
            "MT-CO1", "MT-ND1", "GAPDH", "ACTB", "SOX2", "TP53", "BRCA1", "EGFR", "MYC", "KRAS",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), Some("human"));
    }

    #[test]
    fn infer_species_mouse_large_set() {
        let genes: Vec<String> = vec![
            "Gapdh", "Actb", "Sox2", "Pou5f1", "Nanog", "Klf4", "Myc", "Bmp4", "Fgf2", "Wnt3a",
            "Shh", "Notch1", "Dll1", "Jag1", "Hes1", "Hes5", "Hey1", "Hey2", "Neurog2", "Ascl1",
            "Pax6", "Tbr1", "Tbr2", "Dcx", "Map2", "Tubb3", "Rbfox3", "Gfap", "Aldh1l1", "Olig2",
            "Mbp", "Pdgfra", "Cspg4",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_human_large_set() {
        let genes: Vec<String> = vec![
            "GAPDH", "ACTB", "SOX2", "POU5F1", "NANOG", "KLF4", "MYC", "BMP4", "FGF2", "WNT3A",
            "SHH", "NOTCH1", "DLL1", "JAG1", "HES1", "HES5", "HEY1", "HEY2", "NEUROG2", "ASCL1",
            "PAX6", "TBR1", "EOMES", "DCX", "MAP2", "TUBB3", "RBFOX3", "GFAP", "ALDH1L1", "OLIG2",
            "MBP", "PDGFRA", "CSPG4",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), Some("human"));
    }

    #[test]
    fn infer_species_all_ensembl_mouse() {
        let genes: Vec<String> = vec![
            "ENSMUSG00000000001",
            "ENSMUSG00000000028",
            "ENSMUSG00000000031",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_all_ensembl_human() {
        let genes: Vec<String> = vec!["ENSG00000139618", "ENSG00000141510", "ENSG00000171862"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(infer_species(&genes), Some("human"));
    }

    #[test]
    fn infer_species_mixed_ensembl_and_symbols() {
        let genes: Vec<String> = vec!["Gapdh", "Actb", "ENSMUSG00000000001", "BRCA1", "TP53"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_whitespace_and_edge_cases() {
        let genes: Vec<String> = vec!["  Gapdh  ", "", "  ", "Actb", "Sox2"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_single_gene_mouse() {
        let genes = vec!["Gapdh".to_string()];
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn infer_species_single_gene_human() {
        let genes = vec!["GAPDH".to_string()];
        assert_eq!(infer_species(&genes), Some("human"));
    }

    #[test]
    fn infer_species_stereoseq_mouse_style() {
        let genes: Vec<String> = vec![
            "Apoe",
            "Alb",
            "Serpina1a",
            "Hpx",
            "Fga",
            "Fgb",
            "Fgg",
            "Hp",
            "Itih4",
            "Orm1",
            "Orm2",
            "Saa1",
            "Saa2",
            "Lcn2",
            "Cyp2e1",
            "Cyp2f2",
            "Cyp3a11",
            "Cyp1a2",
            "Glul",
            "Oat",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(infer_species(&genes), Some("mouse"));
    }

    #[test]
    fn modulators_struct_fields() {
        let m = Modulators {
            regulators: vec!["A".into()],
            ligands: vec!["B".into()],
            receptors: vec!["C".into()],
            tfl_ligands: vec!["D".into()],
            tfl_regulators: vec!["E".into()],
            lr_pairs: vec!["B$C".into()],
            tfl_pairs: vec!["D#E".into()],
        };
        assert_eq!(m.regulators.len(), 1);
        assert_eq!(m.lr_pairs[0], "B$C");
        assert_eq!(m.tfl_pairs[0], "D#E");
    }

    #[test]
    fn apply_modulator_mask_lr_only() {
        let m = Modulators {
            regulators: vec!["A".into()],
            ligands: vec!["B".into()],
            receptors: vec!["C".into()],
            tfl_ligands: vec!["D".into()],
            tfl_regulators: vec!["E".into()],
            lr_pairs: vec!["B$C".into()],
            tfl_pairs: vec!["D#E".into()],
        };
        let m = m.apply_modulator_mask(false, true, false);
        assert!(m.regulators.is_empty());
        assert_eq!(m.lr_pairs.len(), 1);
        assert!(m.tfl_pairs.is_empty());
    }

    #[test]
    fn run_network_data_dir_is_under_output() {
        let out = Path::new("/tmp/spacetravlr_run");
        assert_eq!(
            run_network_data_dir(out),
            Path::new("/tmp/spacetravlr_run/network")
        );
    }

    #[test]
    fn resolve_or_fetch_uses_existing_run_network_dir() {
        let repo_data = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let out = std::env::temp_dir().join(format!("st_run_net_{}", std::process::id()));
        let run_net = run_network_data_dir(&out);
        let _ = fs::remove_dir_all(&out);
        fs::create_dir_all(&run_net).unwrap();
        fs::copy(
            repo_data.join("mouse_network.parquet"),
            run_net.join("mouse_network.parquet"),
        )
        .unwrap();
        let resolved =
            resolve_or_fetch_network_data_dir("mouse", &out, Some("/nonexistent/network"))
                .unwrap()
                .expect("run network dir");
        assert_eq!(Path::new(&resolved), run_net);
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn resolve_mouse_with_explicit_config_data_dir() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let p = resolve_species_network_parquet("mouse", Some(dir.to_str().unwrap())).unwrap();
        assert!(p.ends_with("mouse_network.parquet"));
        assert!(p.is_file());
    }

    #[test]
    fn resolve_human_with_explicit_config_data_dir() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let p = resolve_species_network_parquet("human", Some(dir.to_str().unwrap())).unwrap();
        assert!(p.ends_with("human_network.parquet"));
        assert!(p.is_file());
    }

    #[test]
    fn resolve_mouse_none_config_uses_search_path() {
        let p = resolve_species_network_parquet("mouse", None).unwrap();
        assert!(p.ends_with("mouse_network.parquet"));
        assert!(p.is_file());
    }

    #[test]
    fn resolve_error_lists_tried_paths() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let err = resolve_species_network_parquet(
            "definitely_missing_species_xyz",
            Some(dir.to_str().unwrap()),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("definitely_missing_species_xyz_network.parquet"));
        assert!(err.contains(SPACETRAVLR_DATA_DIR_ENV));
        assert!(err.contains("Tried:"));
    }

    #[test]
    fn gene_network_new_loads_mouse_from_manifest_data_dir() {
        let genes: Vec<String> = vec!["Gapdh".into(), "Actb".into()];
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let net = GeneNetwork::new("mouse", &genes, Some(dir.to_str().unwrap())).unwrap();
        assert!(net.network_path.ends_with("mouse_network.parquet"));
    }

    #[test]
    fn max_ligands_filter_keeps_top_ligands_by_mean() {
        let mut ligands = vec!["low".into(), "high".into(), "mid".into()];
        let mut receptors = vec!["R1".into(), "R2".into(), "R3".into()];
        let mut lr_pairs = vec!["low$R1".into(), "high$R2".into(), "mid$R3".into()];
        let mut means = HashMap::new();
        means.insert("low".into(), 1.0);
        means.insert("high".into(), 10.0);
        means.insert("mid".into(), 5.0);
        apply_max_ligands_filter(&mut ligands, &mut receptors, &mut lr_pairs, Some(2), &means);
        assert_eq!(lr_pairs.len(), 2);
        assert!(lr_pairs.contains(&"high$R2".into()));
        assert!(lr_pairs.contains(&"mid$R3".into()));
        assert_eq!(lr_pairs, vec!["high$R2".to_string(), "mid$R3".to_string()]);
    }

    #[test]
    fn max_ligands_filter_row_order_is_lexicographic_by_pair() {
        let mut means = HashMap::new();
        means.insert("a".into(), 2.0);
        means.insert("b".into(), 2.0);
        let mut ligands = vec!["b".into(), "a".into()];
        let mut receptors = vec!["R1".into(), "R1".into()];
        let mut lr_pairs = vec!["b$R1".into(), "a$R1".into()];
        apply_max_ligands_filter(&mut ligands, &mut receptors, &mut lr_pairs, Some(2), &means);
        assert_eq!(lr_pairs, vec!["a$R1".to_string(), "b$R1".to_string()]);
    }
}
