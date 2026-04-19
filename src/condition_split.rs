use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::Context;

use crate::betadata::obs_series_row_str;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ConditionDirStatus {
    pub dir_name: String,
    pub label: String,
    pub output_dir: PathBuf,
    pub n_feathers: usize,
    pub n_orphans: usize,
    pub n_tf_ablated: usize,
    pub n_locks: usize,
}

impl ConditionDirStatus {
    pub fn n_done(&self) -> usize {
        self.n_feathers + self.n_orphans + self.n_tf_ablated
    }
}

/// Scan existing `conditions/<group>/` subdirectories under `output_root` and
/// report per-group training status (feathers done, orphans, active locks).
/// Does NOT require AnnData access — purely filesystem-based.
pub fn scan_condition_status(output_root: &str) -> anyhow::Result<Vec<ConditionDirStatus>> {
    let cond_root = Path::new(output_root).join(CONDITION_RUNS_SUBDIR);
    if !cond_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<_> = fs::read_dir(&cond_root)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let dir_name = entry.file_name().to_string_lossy().into_owned();
        let dir_path = entry.path();
        let label = fs::read_to_string(dir_path.join("condition_label.txt"))
            .unwrap_or_else(|_| dir_name.clone())
            .trim()
            .to_string();

        let mut n_feathers = 0usize;
        let mut n_orphans = 0usize;
        let mut n_tf_ablated = 0usize;
        let mut n_locks = 0usize;
        if let Ok(files) = fs::read_dir(&dir_path) {
            for f in files.filter_map(|f| f.ok()) {
                let name = f.file_name();
                let name = name.to_string_lossy();
                if name.ends_with("_betadata.feather") {
                    n_feathers += 1;
                } else if name.ends_with(".orphan") {
                    n_orphans += 1;
                } else if name.ends_with(".tf_ablated") {
                    n_tf_ablated += 1;
                } else if name.ends_with(".lock") {
                    n_locks += 1;
                }
            }
        }

        out.push(ConditionDirStatus {
            dir_name,
            label,
            output_dir: dir_path,
            n_feathers,
            n_orphans,
            n_tf_ablated,
            n_locks,
        });
    }
    Ok(out)
}

/// Parent directory under the run output root for per-condition training (betadata, logs, models).
pub const CONDITION_RUNS_SUBDIR: &str = "conditions";

/// Normalizes a condition label the same way as when writing `condition_label.txt`.
pub fn normalize_condition_label(s: &str) -> String {
    s.replace(['\n', '\r'], " ").trim().to_string()
}

/// If `output_root/conditions/` already has a subfolder whose `condition_label.txt` matches
/// `label`, return that path so resume / `--join-output-dir` writes betadata beside prior runs.
pub fn find_condition_dir_matching_label(output_root: &str, label: &str) -> Option<PathBuf> {
    let cond_root = Path::new(output_root).join(CONDITION_RUNS_SUBDIR);
    if !cond_root.is_dir() {
        return None;
    }
    let want = normalize_condition_label(label);
    if want.is_empty() {
        return None;
    }
    let mut matches: Vec<PathBuf> = fs::read_dir(&cond_root)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| {
            let p = e.path();
            let txt = p.join("condition_label.txt");
            if !txt.is_file() {
                return None;
            }
            let disk = fs::read_to_string(&txt).ok()?;
            if normalize_condition_label(disk.trim()) == want {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

#[derive(Debug, Clone)]
pub struct ConditionSplitPlan {
    pub label: String,
    pub output_dir: PathBuf,
    pub obs_indices: Vec<usize>,
    pub n_obs: usize,
}

pub fn sanitize_condition_value(label: &str) -> String {
    const MAX_LEN: usize = 64;
    let mut out = String::with_capacity(label.len());
    let mut prev_sep = false;
    for ch in label.trim().chars() {
        let keep = ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.');
        let c = if keep { ch } else { '_' };
        if c == '_' {
            if !prev_sep {
                out.push('_');
            }
            prev_sep = true;
        } else {
            out.push(c);
            prev_sep = false;
        }
        if out.len() >= MAX_LEN {
            break;
        }
    }
    let out = out.trim_matches('_').trim_matches('.').to_string();
    if out.is_empty() {
        "group".to_string()
    } else {
        out
    }
}

pub fn resolve_condition_dir_names(labels: &[String]) -> Vec<String> {
    let mut used = HashSet::<String>::new();
    let mut out = Vec::with_capacity(labels.len());
    for label in labels {
        let base = sanitize_condition_value(label);
        if !used.contains(&base) {
            used.insert(base.clone());
            out.push(base);
            continue;
        }
        let mut idx = 2usize;
        loop {
            let candidate = format!("{}_{}", base, idx);
            if !used.contains(&candidate) {
                used.insert(candidate.clone());
                out.push(candidate);
                break;
            }
            idx = idx.saturating_add(1);
        }
    }
    out
}

/// When `reuse_existing_condition_dirs` is true (e.g. `--join-output-dir`), each split's output
/// directory is an existing `conditions/<subdir>/` with a matching `condition_label.txt` if one
/// exists; otherwise the canonical sanitized name is used. This keeps betadata and locks on the
/// same paths as the leader run.
pub fn prepare_condition_splits(
    adata_path: &str,
    output_root: &str,
    condition_column: &str,
    reuse_existing_condition_dirs: bool,
) -> anyhow::Result<Vec<ConditionSplitPlan>> {
    let adata = AnnData::<H5>::open(H5::open(adata_path)?)?;
    let obs = adata.read_obs()?;
    let condition_series = obs.column(condition_column).with_context(|| {
        let names: Vec<String> = obs
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .take(25)
            .collect();
        format!(
            "obs column {:?} not found (needed for --condition split). First obs columns: {:?}.",
            condition_column, names
        )
    })?;

    let series = condition_series.as_materialized_series();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for idx in 0..series.len() {
        let raw = obs_series_row_str(series, idx).unwrap_or_default();
        let label = if raw.trim().is_empty() {
            "_na".to_string()
        } else {
            raw
        };
        groups.entry(label).or_default().push(idx);
    }
    if groups.is_empty() {
        anyhow::bail!(
            "obs column {:?} has no values; cannot split training by condition.",
            condition_column
        );
    }

    fs::create_dir_all(output_root)?;
    let labels: Vec<String> = groups.keys().cloned().collect();
    let dir_names = resolve_condition_dir_names(&labels);
    let mut plans = Vec::with_capacity(groups.len());

    for ((label, indices), dir_name) in groups.into_iter().zip(dir_names.into_iter()) {
        if indices.is_empty() {
            anyhow::bail!("condition group {:?} has zero rows; cannot train.", label);
        }
        let n_obs = indices.len();
        let canonical_dir = Path::new(output_root)
            .join(CONDITION_RUNS_SUBDIR)
            .join(&dir_name);
        let split_output_dir = if reuse_existing_condition_dirs {
            find_condition_dir_matching_label(output_root, &label).unwrap_or(canonical_dir)
        } else {
            canonical_dir
        };
        fs::create_dir_all(&split_output_dir)?;
        let label_path = split_output_dir.join("condition_label.txt");
        let label_one_line = label.replace(['\n', '\r'], " ");
        if reuse_existing_condition_dirs && label_path.is_file() {
            let on_disk = fs::read_to_string(&label_path).unwrap_or_default();
            if normalize_condition_label(on_disk.trim()) != normalize_condition_label(&label) {
                anyhow::bail!(
                    "reuse dirs: {} exists but condition_label.txt ({:?}) does not match group label {:?}",
                    label_path.display(),
                    on_disk.trim(),
                    label
                );
            }
        } else {
            fs::write(&label_path, format!("{label_one_line}\n"))?;
        }
        plans.push(ConditionSplitPlan {
            label,
            output_dir: split_output_dir,
            obs_indices: indices,
            n_obs,
        });
    }

    Ok(plans)
}
