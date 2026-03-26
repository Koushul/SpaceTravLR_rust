use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use anyhow::Context;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Parent directory under the run output root for per-condition training (betadata, logs, models).
pub const CONDITION_RUNS_SUBDIR: &str = "conditions";

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

pub fn prepare_condition_splits(
    adata_path: &str,
    output_root: &str,
    condition_column: &str,
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

    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, v) in condition_series.as_materialized_series().iter().enumerate() {
        let raw = v.to_string();
        let label = if raw == "null" || raw.trim().is_empty() {
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
        let split_output_dir = Path::new(output_root)
            .join(CONDITION_RUNS_SUBDIR)
            .join(&dir_name);
        fs::create_dir_all(&split_output_dir)?;
        let label_one_line = label.replace(['\n', '\r'], " ");
        fs::write(
            split_output_dir.join("condition_label.txt"),
            format!("{label_one_line}\n"),
        )?;
        plans.push(ConditionSplitPlan {
            label,
            output_dir: split_output_dir,
            obs_indices: indices,
            n_obs,
        });
    }

    Ok(plans)
}
