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
        let label = fs::read_to_string(dir_path.join(CONDITION_LABEL_FILENAME))
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

/// Nested per-sample dirs when `[data].condition` and `[training].pool_lasso` are both set:
/// `conditions/<condition>/samples/<sample>/`.
pub const SAMPLE_RUNS_SUBDIR: &str = "samples";

pub const CONDITION_LABEL_FILENAME: &str = "condition_label.txt";
pub const SAMPLE_LABEL_FILENAME: &str = "sample_label.txt";

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
            let txt = p.join(CONDITION_LABEL_FILENAME);
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

/// Group row indices by a string-like `obs` column. Empty / missing values become `"_na"`.
pub fn group_obs_column_indices(
    obs: &polars::prelude::DataFrame,
    column: &str,
) -> anyhow::Result<BTreeMap<String, Vec<usize>>> {
    let series_col = obs.column(column).with_context(|| {
        let names: Vec<String> = obs
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .take(25)
            .collect();
        format!(
            "obs column {:?} not found. First obs columns: {:?}.",
            column, names
        )
    })?;
    let series = series_col.as_materialized_series();
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
        anyhow::bail!("obs column {:?} has no values.", column);
    }
    Ok(groups)
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
    prepare_obs_splits(
        adata_path,
        output_root,
        condition_column,
        reuse_existing_condition_dirs,
        CONDITION_RUNS_SUBDIR,
        CONDITION_LABEL_FILENAME,
        "condition",
    )
}

/// Sample splits under `output_root/conditions/<sample>/` (standalone pool-lasso) or
/// `output_root/samples/<sample>/` (nested under a condition directory).
pub fn prepare_sample_splits(
    adata_path: &str,
    output_root: &str,
    sample_column: &str,
    reuse_existing: bool,
    nested_under_condition: bool,
) -> anyhow::Result<Vec<ConditionSplitPlan>> {
    if nested_under_condition {
        prepare_obs_splits(
            adata_path,
            output_root,
            sample_column,
            reuse_existing,
            SAMPLE_RUNS_SUBDIR,
            SAMPLE_LABEL_FILENAME,
            "sample",
        )
    } else {
        prepare_obs_splits(
            adata_path,
            output_root,
            sample_column,
            reuse_existing,
            CONDITION_RUNS_SUBDIR,
            CONDITION_LABEL_FILENAME,
            "sample",
        )
    }
}

/// Build split dirs from an already-subsetted `obs` table (local row indices).
pub fn prepare_sample_splits_from_obs(
    obs: &polars::prelude::DataFrame,
    output_root: &str,
    sample_column: &str,
    reuse_existing: bool,
    nested_under_condition: bool,
) -> anyhow::Result<Vec<ConditionSplitPlan>> {
    let groups = group_obs_column_indices(obs, sample_column)?;
    let (subdir, label_file) = if nested_under_condition {
        (SAMPLE_RUNS_SUBDIR, SAMPLE_LABEL_FILENAME)
    } else {
        (CONDITION_RUNS_SUBDIR, CONDITION_LABEL_FILENAME)
    };
    write_split_plans(
        groups,
        output_root,
        reuse_existing,
        subdir,
        label_file,
        "sample",
    )
}

fn prepare_obs_splits(
    adata_path: &str,
    output_root: &str,
    column: &str,
    reuse_existing: bool,
    runs_subdir: &str,
    label_filename: &str,
    kind: &str,
) -> anyhow::Result<Vec<ConditionSplitPlan>> {
    let adata = AnnData::<H5>::open(H5::open(adata_path)?)?;
    let obs = adata.read_obs()?;
    let groups = group_obs_column_indices(&obs, column)
        .with_context(|| format!("needed for {kind} split on obs column {column:?}"))?;
    write_split_plans(
        groups,
        output_root,
        reuse_existing,
        runs_subdir,
        label_filename,
        kind,
    )
}

/// Read-only sample (and optional condition) plan for `collect-interactions` on a pool-lasso run.
#[derive(Debug, Clone)]
pub struct CollectSamplePlan {
    pub sample: String,
    pub condition: Option<String>,
    pub output_dir: PathBuf,
    pub obs_indices: Vec<usize>,
}

fn group_aligned_labels(labels: &[String]) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, raw) in labels.iter().enumerate() {
        let label = if raw.trim().is_empty() {
            "_na".to_string()
        } else {
            raw.clone()
        };
        groups.entry(label).or_default().push(idx);
    }
    groups
}

fn count_betadata_feathers(dir: &Path) -> usize {
    let Ok(rd) = fs::read_dir(dir) else {
        return 0;
    };
    rd.filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.ends_with("_betadata.feather"))
        })
        .count()
}

/// Locate existing pool-lasso sample directories without creating files.
///
/// `obs_sample_labels` / `obs_condition_labels` must be aligned with collect-interactions obs
/// (including `perturb_obs_subset_file`). When `obs_condition_labels` is `Some`, dirs are
/// `conditions/<condition>/samples/<sample>/`; otherwise `conditions/<sample>/`.
pub fn discover_pool_lasso_collect_plans(
    output_root: &Path,
    obs_sample_labels: &[String],
    obs_condition_labels: Option<&[String]>,
) -> anyhow::Result<Vec<CollectSamplePlan>> {
    anyhow::ensure!(
        !obs_sample_labels.is_empty(),
        "no cells to assign to pool-lasso samples"
    );
    if let Some(cond) = obs_condition_labels {
        anyhow::ensure!(
            cond.len() == obs_sample_labels.len(),
            "obs_condition_labels len {} != obs_sample_labels len {}",
            cond.len(),
            obs_sample_labels.len()
        );
    }
    let root_s = output_root
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("training output directory path must be UTF-8"))?;

    let mut plans = Vec::new();
    if let Some(cond_labels) = obs_condition_labels {
        let mut pairs: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
        for i in 0..obs_sample_labels.len() {
            let c = if cond_labels[i].trim().is_empty() {
                "_na".to_string()
            } else {
                cond_labels[i].clone()
            };
            let s = if obs_sample_labels[i].trim().is_empty() {
                "_na".to_string()
            } else {
                obs_sample_labels[i].clone()
            };
            pairs.entry((c, s)).or_default().push(i);
        }
        for ((condition, sample), obs_indices) in pairs {
            let Some(cond_dir) = find_split_dir_matching_label(
                root_s,
                CONDITION_RUNS_SUBDIR,
                CONDITION_LABEL_FILENAME,
                &condition,
            ) else {
                eprintln!(
                    "Warning: no conditions/ directory for condition {condition:?}; skipping sample {sample:?}"
                );
                continue;
            };
            let cond_s = cond_dir
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("condition directory path must be UTF-8"))?;
            let Some(sample_dir) = find_split_dir_matching_label(
                cond_s,
                SAMPLE_RUNS_SUBDIR,
                SAMPLE_LABEL_FILENAME,
                &sample,
            ) else {
                eprintln!(
                    "Warning: no samples/ directory for sample {sample:?} under condition {condition:?}"
                );
                continue;
            };
            plans.push(CollectSamplePlan {
                sample,
                condition: Some(condition),
                output_dir: sample_dir,
                obs_indices,
            });
        }
    } else {
        for (sample, obs_indices) in group_aligned_labels(obs_sample_labels) {
            let Some(sample_dir) = find_split_dir_matching_label(
                root_s,
                CONDITION_RUNS_SUBDIR,
                CONDITION_LABEL_FILENAME,
                &sample,
            ) else {
                eprintln!("Warning: no conditions/ directory matching sample label {sample:?}");
                continue;
            };
            plans.push(CollectSamplePlan {
                sample,
                condition: None,
                output_dir: sample_dir,
                obs_indices,
            });
        }
    }

    let n_feathers: usize = plans
        .iter()
        .map(|p| count_betadata_feathers(&p.output_dir))
        .sum();
    anyhow::ensure!(
        n_feathers > 0,
        "pool-lasso collect-interactions: no *_betadata.feather files under sample directories of {}",
        output_root.display()
    );
    Ok(plans)
}

fn find_split_dir_matching_label(
    output_root: &str,
    runs_subdir: &str,
    label_filename: &str,
    label: &str,
) -> Option<PathBuf> {
    let root = Path::new(output_root).join(runs_subdir);
    if !root.is_dir() {
        return None;
    }
    let want = normalize_condition_label(label);
    if want.is_empty() {
        return None;
    }
    let mut matches: Vec<PathBuf> = fs::read_dir(&root)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| {
            let p = e.path();
            let txt = p.join(label_filename);
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

fn write_split_plans(
    groups: BTreeMap<String, Vec<usize>>,
    output_root: &str,
    reuse_existing: bool,
    runs_subdir: &str,
    label_filename: &str,
    kind: &str,
) -> anyhow::Result<Vec<ConditionSplitPlan>> {
    fs::create_dir_all(output_root)?;
    let labels: Vec<String> = groups.keys().cloned().collect();
    let dir_names = resolve_condition_dir_names(&labels);
    let mut plans = Vec::with_capacity(groups.len());

    for ((label, indices), dir_name) in groups.into_iter().zip(dir_names.into_iter()) {
        if indices.is_empty() {
            anyhow::bail!("{kind} group {label:?} has zero rows; cannot train.");
        }
        let n_obs = indices.len();
        let canonical_dir = Path::new(output_root).join(runs_subdir).join(&dir_name);
        let split_output_dir = if reuse_existing {
            find_split_dir_matching_label(output_root, runs_subdir, label_filename, &label)
                .unwrap_or(canonical_dir)
        } else {
            canonical_dir
        };
        fs::create_dir_all(&split_output_dir)?;
        let label_path = split_output_dir.join(label_filename);
        let label_one_line = label.replace(['\n', '\r'], " ");
        if reuse_existing && label_path.is_file() {
            let on_disk = fs::read_to_string(&label_path).unwrap_or_default();
            if normalize_condition_label(on_disk.trim()) != normalize_condition_label(&label) {
                anyhow::bail!(
                    "reuse dirs: {} exists but {} ({:?}) does not match group label {:?}",
                    label_path.display(),
                    label_filename,
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

#[cfg(test)]
mod collect_sample_plan_tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "st_collect_plans_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn discover_uses_label_file_not_folder_name() {
        let root = tmp("label");
        let dir = root.join(CONDITION_RUNS_SUBDIR).join("slide_a");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(CONDITION_LABEL_FILENAME), "Slide A!\n").unwrap();
        fs::write(dir.join("GENE_betadata.feather"), b"x").unwrap();
        let labels = vec!["Slide A!".into(), "Slide A!".into()];
        let plans = discover_pool_lasso_collect_plans(&root, &labels, None).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].sample, "Slide A!");
        assert_eq!(plans[0].obs_indices, vec![0, 1]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn discover_nested_condition_sample() {
        let root = tmp("nested");
        let cdir = root.join(CONDITION_RUNS_SUBDIR).join("cA");
        let sdir = cdir.join(SAMPLE_RUNS_SUBDIR).join("s1");
        fs::create_dir_all(&sdir).unwrap();
        fs::write(cdir.join(CONDITION_LABEL_FILENAME), "condA\n").unwrap();
        fs::write(sdir.join(SAMPLE_LABEL_FILENAME), "s1\n").unwrap();
        fs::write(sdir.join("G_betadata.feather"), b"x").unwrap();
        let samples = vec!["s1".into(), "s1".into()];
        let conds = vec!["condA".into(), "condA".into()];
        let plans = discover_pool_lasso_collect_plans(&root, &samples, Some(&conds)).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].sample, "s1");
        assert_eq!(plans[0].condition.as_deref(), Some("condA"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn discover_errors_when_no_feathers() {
        let root = tmp("empty");
        let dir = root.join(CONDITION_RUNS_SUBDIR).join("s1");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(CONDITION_LABEL_FILENAME), "s1\n").unwrap();
        let err = discover_pool_lasso_collect_plans(&root, &["s1".into()], None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no *_betadata.feather"), "{err}");
        let _ = fs::remove_dir_all(&root);
    }
}
