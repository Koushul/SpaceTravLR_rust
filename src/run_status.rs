use crate::condition_split::{
    CONDITION_LABEL_FILENAME, CONDITION_RUNS_SUBDIR, SAMPLE_LABEL_FILENAME, SAMPLE_RUNS_SUBDIR,
    scan_condition_status,
};
use crate::config::RUN_REPRO_TOML_FILENAME;
use crate::run_setup_lock::{SETUP_LOCK_FILENAME, SETUP_READY_FILENAME};
use anyhow::Context;
use colored::Colorize;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

pub const TARGET_GENES_FILENAME: &str = "spacetravlr_target_genes.txt";

const RATE_WINDOW: Duration = Duration::from_secs(30 * 60);
const RATE_FALLBACK_N: usize = 20;
const STALL_SECS: u64 = 15 * 60;
const BAR_WIDTH: usize = 20;
const LABEL_W: usize = 12;

pub fn gene_lock_payload(n_parallel: usize) -> String {
    format!(
        "spacetravlr_gene 1\nhost={}\npid={}\nn_parallel={}\nstarted_unix={}\n",
        host_label(),
        std::process::id(),
        n_parallel.max(1),
        unix_secs(SystemTime::now())
    )
}

pub fn host_label() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".into())
}

fn unix_secs(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Clone, Default)]
pub struct GeneLockPayload {
    pub host: Option<String>,
    pub pid: Option<u32>,
    pub n_parallel: Option<usize>,
    pub started_unix: Option<u64>,
}

pub fn parse_gene_lock_payload(text: &str) -> GeneLockPayload {
    let mut p = GeneLockPayload::default();
    for line in text.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("host=") {
            let v = v.trim();
            if !v.is_empty() {
                p.host = Some(v.to_string());
            }
        } else if let Some(v) = line.strip_prefix("pid=") {
            p.pid = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("n_parallel=") {
            p.n_parallel = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("started_unix=") {
            p.started_unix = v.trim().parse().ok();
        }
    }
    p
}

pub fn write_target_genes_if_missing(dir: &Path, genes: &[String]) -> std::io::Result<bool> {
    fs::create_dir_all(dir)?;
    let path = dir.join(TARGET_GENES_FILENAME);
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut f) => {
            for g in genes {
                writeln!(f, "{g}")?;
            }
            f.flush()?;
            Ok(true)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(e) => Err(e),
    }
}

pub fn read_target_genes_file(dir: &Path) -> Option<Vec<String>> {
    let text = fs::read_to_string(dir.join(TARGET_GENES_FILENAME)).ok()?;
    let genes: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    if genes.is_empty() { None } else { Some(genes) }
}

pub fn resolve_status_dir(path: &Path) -> anyhow::Result<PathBuf> {
    if path.is_file() {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == RUN_REPRO_TOML_FILENAME {
            return path
                .parent()
                .map(Path::to_path_buf)
                .ok_or_else(|| anyhow::anyhow!("--status: {} has no parent", path.display()));
        }
        anyhow::bail!(
            "--status: expected a training output directory or {RUN_REPRO_TOML_FILENAME}, got file {}",
            path.display()
        );
    }
    if path.is_dir() {
        if looks_like_split_subdir(path) {
            return Ok(find_run_root(path));
        }
        return Ok(path.to_path_buf());
    }
    anyhow::bail!("--status: not found: {}", path.display())
}

fn looks_like_split_subdir(dir: &Path) -> bool {
    if dir.join(SAMPLE_LABEL_FILENAME).is_file() || dir.join(CONDITION_LABEL_FILENAME).is_file() {
        return true;
    }
    dir.parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .is_some_and(|n| n == CONDITION_RUNS_SUBDIR || n == SAMPLE_RUNS_SUBDIR)
}

fn find_run_root(start: &Path) -> PathBuf {
    let mut cur = start.to_path_buf();
    for _ in 0..4 {
        if cur.join(RUN_REPRO_TOML_FILENAME).is_file() {
            return cur;
        }
        match cur.parent() {
            Some(p) if p != cur => cur = p.to_path_buf(),
            _ => break,
        }
    }
    start.to_path_buf()
}

#[derive(Debug, Clone)]
struct FileStamp {
    name: String,
    gene: String,
    mtime: SystemTime,
    len: u64,
}

#[derive(Debug, Clone)]
struct LockStamp {
    name: String,
    gene: String,
    mtime: SystemTime,
    payload: GeneLockPayload,
}

#[derive(Debug, Clone, Default)]
struct DirArtifacts {
    feathers: Vec<FileStamp>,
    locks: Vec<LockStamp>,
    orphan_genes: HashSet<String>,
    tf_ablated_genes: HashSet<String>,
    done_genes: HashSet<String>,
    done_stamps: Vec<FileStamp>,
}

impl DirArtifacts {
    fn completion_genes(&self) -> HashSet<String> {
        let mut s = self.done_genes.clone();
        s.extend(self.orphan_genes.iter().cloned());
        s.extend(self.tf_ablated_genes.iter().cloned());
        s.extend(self.feathers.iter().map(|f| f.gene.clone()));
        s
    }
}

#[derive(Debug, Clone)]
pub struct TrainerRow {
    pub host: String,
    pub pid: u32,
    pub n_parallel: usize,
    pub n_locks: usize,
}

#[derive(Debug, Clone)]
pub struct ConditionRow {
    pub label: String,
    pub n_done: usize,
    pub n_feathers: usize,
    pub n_locks: usize,
    pub planned: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RunStatus {
    pub dir: PathBuf,
    pub planned: Option<usize>,
    pub n_done: usize,
    pub n_in_flight: usize,
    pub n_remaining: Option<usize>,
    pub n_feathers: usize,
    pub n_orphans: usize,
    pub n_tf_ablated: usize,
    pub n_locks: usize,
    pub n_stale_locks: usize,
    pub n_logs: usize,
    pub feather_bytes: u64,
    pub dir_bytes: u64,
    pub dir_files: usize,
    pub last_feather: Option<(String, Duration)>,
    pub last_lock: Option<(String, Duration)>,
    pub genes_per_hour: Option<f64>,
    pub throughput_n: usize,
    pub throughput_window: Option<Duration>,
    pub eta: Option<Duration>,
    pub stalled: bool,
    pub trainers: Vec<TrainerRow>,
    pub worker_slots: usize,
    pub n_unlabelled_locks: usize,
    pub setup_ready: bool,
    pub setup_flock: bool,
    pub has_repro: bool,
    pub training_mode: Option<String>,
    pub epochs: Option<usize>,
    pub repro_n_parallel: Option<usize>,
    pub pool_lasso: bool,
    pub condition_col: Option<String>,
    pub sample_col: Option<String>,
    pub adata_path: Option<String>,
    pub stale_lock_secs: u64,
    pub conditions: Vec<ConditionRow>,
    pub samples: Vec<ConditionRow>,
}

#[derive(Debug, Default)]
struct ReproHints {
    genes: Option<Vec<String>>,
    max_genes: Option<usize>,
    mode: Option<String>,
    epochs: Option<usize>,
    n_parallel: Option<usize>,
    pool_lasso: bool,
    condition: Option<String>,
    sample: Option<String>,
    adata_path: Option<String>,
    stale_lock_secs: u64,
}

fn load_repro_hints(dir: &Path) -> ReproHints {
    let path = dir.join(RUN_REPRO_TOML_FILENAME);
    let Ok(text) = fs::read_to_string(&path) else {
        return ReproHints::default();
    };
    let Ok(root) = text.parse::<toml::Value>() else {
        return ReproHints::default();
    };
    let mut h = ReproHints::default();
    if let Some(t) = root.get("training") {
        h.genes = t.get("genes").and_then(|v| {
            v.as_array().map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
        });
        h.max_genes = t.get("max_genes").and_then(|v| {
            v.as_integer()
                .and_then(|n| usize::try_from(n).ok())
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        });
        h.mode = t.get("mode").and_then(|v| v.as_str()).map(str::to_string);
        h.epochs = t
            .get("epochs")
            .and_then(|v| v.as_integer().and_then(|n| usize::try_from(n).ok()));
        h.pool_lasso = t
            .get("pool_lasso")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    }
    if let Some(e) = root.get("execution") {
        h.n_parallel = e
            .get("n_parallel")
            .and_then(|v| v.as_integer().and_then(|n| usize::try_from(n).ok()));
        h.stale_lock_secs = e
            .get("stale_lock_secs")
            .and_then(|v| v.as_integer())
            .and_then(|n| u64::try_from(n).ok())
            .unwrap_or(0);
    }
    if let Some(d) = root.get("data") {
        h.condition = d
            .get("condition")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty());
        h.sample = d
            .get("sample")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty());
        h.adata_path = d
            .get("adata_path")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty());
    }
    h
}

fn planned_from_genes_and_max(genes: Option<&[String]>, max_genes: Option<usize>) -> Option<usize> {
    match (genes, max_genes) {
        (Some(g), Some(m)) => Some(g.len().min(m)),
        (Some(g), None) => Some(g.len()),
        (None, Some(m)) => Some(m),
        (None, None) => None,
    }
}

fn scan_flat_dir(dir: &Path) -> DirArtifacts {
    let mut out = DirArtifacts::default();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let path = e.path();
        let Ok(meta) = e.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if let Some(gene) = name.strip_suffix("_betadata.feather") {
            out.feathers.push(FileStamp {
                name: name.to_string(),
                gene: gene.to_string(),
                mtime,
                len: meta.len(),
            });
        } else if let Some(gene) = name.strip_suffix(".lock") {
            let payload = fs::read_to_string(&path)
                .map(|t| parse_gene_lock_payload(&t))
                .unwrap_or_default();
            out.locks.push(LockStamp {
                name: name.to_string(),
                gene: gene.to_string(),
                mtime,
                payload,
            });
        } else if let Some(gene) = name.strip_suffix(".orphan") {
            out.orphan_genes.insert(gene.to_string());
        } else if let Some(gene) = name.strip_suffix(".tf_ablated") {
            out.tf_ablated_genes.insert(gene.to_string());
        } else if let Some(gene) = name.strip_suffix(".done") {
            out.done_genes.insert(gene.to_string());
            out.done_stamps.push(FileStamp {
                name: name.to_string(),
                gene: gene.to_string(),
                mtime,
                len: meta.len(),
            });
        }
    }
    out
}

fn walk_tree_stats(root: &Path, bytes: &mut u64, files: &mut usize, logs: &mut usize) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        let Ok(meta) = e.metadata() else {
            continue;
        };
        if meta.is_dir() {
            walk_tree_stats(&path, bytes, files, logs);
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        *bytes += meta.len();
        *files += 1;
        if path.extension().and_then(|x| x.to_str()) == Some("log") {
            *logs += 1;
        }
    }
}

fn collect_nested_training_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for sub in [CONDITION_RUNS_SUBDIR, SAMPLE_RUNS_SUBDIR] {
        let p = root.join(sub);
        if !p.is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(&p) else {
            continue;
        };
        for e in entries.flatten() {
            let child = e.path();
            if !child.is_dir() {
                continue;
            }
            dirs.push(child.clone());
            let nested = child.join(SAMPLE_RUNS_SUBDIR);
            if nested.is_dir() {
                if let Ok(inner) = fs::read_dir(&nested) {
                    for ie in inner.flatten() {
                        let ip = ie.path();
                        if ip.is_dir() {
                            dirs.push(ip);
                        }
                    }
                }
            }
        }
    }
    dirs
}

fn sample_leaf_dirs(root: &Path) -> Vec<PathBuf> {
    collect_nested_training_dirs(root)
        .into_iter()
        .filter(|p| !p.join(SAMPLE_RUNS_SUBDIR).is_dir())
        .collect()
}

fn split_dir_label(dir: &Path) -> String {
    for fname in [SAMPLE_LABEL_FILENAME, CONDITION_LABEL_FILENAME] {
        if let Ok(s) = fs::read_to_string(dir.join(fname)) {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    dir.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.display().to_string())
}

fn sample_display_label(dir: &Path) -> String {
    let own = split_dir_label(dir);
    let Some(parent) = dir.parent() else {
        return own;
    };
    if parent.file_name().and_then(|n| n.to_str()) != Some(SAMPLE_RUNS_SUBDIR) {
        return own;
    }
    let Some(cond_dir) = parent.parent() else {
        return own;
    };
    format!("{}/{}", split_dir_label(cond_dir), own)
}

fn terminal_genes(s: &DirArtifacts) -> HashSet<String> {
    let mut g = s.orphan_genes.clone();
    g.extend(s.tf_ablated_genes.iter().cloned());
    g.extend(s.feathers.iter().map(|f| f.gene.clone()));
    g
}

fn pool_complete_genes(
    root: &DirArtifacts,
    samples: &[(PathBuf, DirArtifacts)],
) -> HashSet<String> {
    let mut done = HashSet::new();
    done.extend(root.done_genes.iter().cloned());
    done.extend(root.orphan_genes.iter().cloned());
    done.extend(root.tf_ablated_genes.iter().cloned());
    if samples.is_empty() {
        return done;
    }
    let mut candidates = HashSet::new();
    for (_, s) in samples {
        candidates.extend(terminal_genes(s));
    }
    for gene in candidates {
        if samples
            .iter()
            .all(|(_, s)| terminal_genes(s).contains(&gene))
        {
            done.insert(gene);
        }
    }
    done
}

fn pool_completion_mtimes(
    root: &DirArtifacts,
    samples: &[(PathBuf, DirArtifacts)],
    complete: &HashSet<String>,
) -> Vec<SystemTime> {
    let mut times = Vec::new();
    for gene in complete {
        if let Some(t) = root
            .done_stamps
            .iter()
            .find(|s| s.gene == *gene)
            .map(|s| s.mtime)
        {
            times.push(t);
            continue;
        }
        let mut latest: Option<SystemTime> = None;
        for (_, s) in samples {
            for f in s.feathers.iter().filter(|f| f.gene == *gene) {
                latest = Some(match latest {
                    Some(prev) if prev >= f.mtime => prev,
                    _ => f.mtime,
                });
            }
        }
        if let Some(t) = latest {
            times.push(t);
        }
    }
    times
}

pub fn feather_throughput(
    mtimes: &[SystemTime],
    now: SystemTime,
) -> Option<(f64, usize, Duration)> {
    if mtimes.len() < 2 {
        return None;
    }
    let mut times = mtimes.to_vec();
    times.sort();
    let in_window: Vec<SystemTime> = times
        .iter()
        .copied()
        .filter(|&t| now.duration_since(t).unwrap_or(Duration::MAX) <= RATE_WINDOW)
        .collect();
    let used: Vec<SystemTime> = if in_window.len() >= 2 {
        in_window
    } else {
        let start = times.len().saturating_sub(RATE_FALLBACK_N);
        times[start..].to_vec()
    };
    if used.len() < 2 {
        return None;
    }
    let first = *used.first()?;
    let last = *used.last()?;
    let elapsed = last.duration_since(first).ok()?;
    if elapsed.as_secs_f64() < 1.0 {
        return None;
    }
    let genes_per_h = used.len() as f64 / elapsed.as_secs_f64() * 3600.0;
    Some((genes_per_h, used.len(), elapsed))
}

fn newest_stamp(items: &[FileStamp]) -> Option<(String, Duration)> {
    let now = SystemTime::now();
    items.iter().max_by_key(|f| f.mtime).map(|f| {
        (
            f.name.clone(),
            now.duration_since(f.mtime).unwrap_or_default(),
        )
    })
}

fn newest_lock(items: &[LockStamp]) -> Option<(String, Duration)> {
    let now = SystemTime::now();
    items.iter().max_by_key(|f| f.mtime).map(|f| {
        (
            f.name.clone(),
            now.duration_since(f.mtime).unwrap_or_default(),
        )
    })
}

fn group_trainers(locks: &[LockStamp]) -> (Vec<TrainerRow>, usize, usize) {
    let mut by_key: BTreeMap<(String, u32), (usize, usize)> = BTreeMap::new();
    let mut unlabelled = 0usize;
    for lock in locks {
        match (lock.payload.host.as_deref(), lock.payload.pid) {
            (Some(host), Some(pid)) => {
                let entry = by_key.entry((host.to_string(), pid)).or_insert((0, 0));
                entry.1 += 1;
                if let Some(n) = lock.payload.n_parallel {
                    if n > entry.0 {
                        entry.0 = n;
                    }
                }
            }
            _ => unlabelled += 1,
        }
    }
    let trainers: Vec<TrainerRow> = by_key
        .into_iter()
        .map(|((host, pid), (n_parallel, n_locks))| TrainerRow {
            host,
            pid,
            n_parallel: n_parallel.max(1),
            n_locks,
        })
        .collect();
    let slots: usize = trainers.iter().map(|t| t.n_parallel).sum();
    let worker_slots = if slots == 0 {
        unlabelled.max(locks.len())
    } else {
        slots
    };
    (trainers, worker_slots, unlabelled)
}

pub fn collect_run_status(dir: &Path) -> anyhow::Result<RunStatus> {
    let dir = dir.to_path_buf();
    let repro = load_repro_hints(&dir);
    let root_scan = scan_flat_dir(&dir);
    let nested_dirs = collect_nested_training_dirs(&dir);
    let nested_scans: Vec<(PathBuf, DirArtifacts)> = nested_dirs
        .iter()
        .map(|p| (p.clone(), scan_flat_dir(p)))
        .collect();

    let sample_dirs = sample_leaf_dirs(&dir);
    let sample_scans: Vec<(PathBuf, DirArtifacts)> = sample_dirs
        .iter()
        .map(|p| (p.clone(), scan_flat_dir(p)))
        .collect();

    let pool_style = repro.pool_lasso
        || !root_scan.done_genes.is_empty()
        || (!root_scan.locks.is_empty()
            && sample_scans.iter().any(|(_, s)| !s.feathers.is_empty()));

    let mut all_feathers: Vec<FileStamp> = root_scan.feathers.clone();
    for (_, s) in &nested_scans {
        all_feathers.extend(s.feathers.iter().cloned());
    }

    let all_locks: Vec<LockStamp> = if pool_style {
        root_scan.locks.clone()
    } else {
        let mut locks = root_scan.locks.clone();
        for (_, s) in &nested_scans {
            locks.extend(s.locks.iter().cloned());
        }
        locks
    };

    let (n_orphans, n_tf_ablated) = if pool_style {
        (
            root_scan.orphan_genes.len(),
            root_scan.tf_ablated_genes.len(),
        )
    } else {
        let mut n_orphans = root_scan.orphan_genes.len();
        let mut n_tf_ablated = root_scan.tf_ablated_genes.len();
        for (_, s) in &nested_scans {
            n_orphans += s.orphan_genes.len();
            n_tf_ablated += s.tf_ablated_genes.len();
        }
        (n_orphans, n_tf_ablated)
    };

    let done_genes_all = if pool_style {
        pool_complete_genes(&root_scan, &sample_scans)
    } else if !nested_scans.is_empty()
        && root_scan.locks.is_empty()
        && root_scan.feathers.is_empty()
    {
        let mut genes = HashSet::new();
        for (_, s) in &nested_scans {
            genes.extend(s.completion_genes());
        }
        genes
    } else {
        let mut genes = root_scan.completion_genes();
        for (_, s) in &nested_scans {
            genes.extend(s.completion_genes());
        }
        genes
    };
    let n_done = if pool_style {
        done_genes_all.len()
    } else if !nested_scans.is_empty()
        && root_scan.locks.is_empty()
        && root_scan.feathers.is_empty()
    {
        nested_scans
            .iter()
            .map(|(_, s)| s.completion_genes().len())
            .sum()
    } else {
        done_genes_all.len()
    };

    let n_in_flight = if pool_style {
        all_locks
            .iter()
            .filter(|l| !done_genes_all.contains(&l.gene))
            .count()
    } else if !nested_scans.is_empty()
        && root_scan.locks.is_empty()
        && root_scan.feathers.is_empty()
    {
        nested_scans
            .iter()
            .map(|(_, s)| {
                let done = s.completion_genes();
                s.locks.iter().filter(|l| !done.contains(&l.gene)).count()
            })
            .sum()
    } else {
        all_locks
            .iter()
            .filter(|l| !done_genes_all.contains(&l.gene))
            .count()
    };

    let root_targets = read_target_genes_file(&dir);
    let mut planned = planned_from_genes_and_max(
        root_targets.as_deref().or(repro.genes.as_deref()),
        if root_targets.is_some() {
            None
        } else {
            repro.max_genes
        },
    );
    if planned.is_none() {
        let nested_n: usize = nested_dirs
            .iter()
            .filter_map(|p| read_target_genes_file(p).map(|g| g.len()))
            .sum();
        if nested_n > 0 {
            planned = Some(nested_n);
        }
    } else if !pool_style
        && !nested_scans.is_empty()
        && root_scan.locks.is_empty()
        && root_targets.is_none()
    {
        if let Some(n) = planned {
            let n_groups = nested_scans
                .iter()
                .filter(|(_, s)| {
                    !s.feathers.is_empty()
                        || !s.locks.is_empty()
                        || !s.orphan_genes.is_empty()
                        || !s.done_genes.is_empty()
                        || !s.tf_ablated_genes.is_empty()
                })
                .count()
                .max(nested_scans.len());
            if n_groups > 1 {
                planned = Some(n.saturating_mul(n_groups));
            }
        }
    }

    let n_remaining = planned.map(|p| p.saturating_sub(n_done));

    let feather_bytes: u64 = all_feathers.iter().map(|f| f.len).sum();
    let mut dir_bytes = 0u64;
    let mut dir_files = 0usize;
    let mut tree_logs = 0usize;
    walk_tree_stats(&dir, &mut dir_bytes, &mut dir_files, &mut tree_logs);

    let now = SystemTime::now();
    let mtimes: Vec<SystemTime> = if pool_style {
        pool_completion_mtimes(&root_scan, &sample_scans, &done_genes_all)
    } else {
        all_feathers.iter().map(|f| f.mtime).collect()
    };
    let (genes_per_hour, throughput_n, throughput_window) = match feather_throughput(&mtimes, now) {
        Some((r, n, w)) => (Some(r), n, Some(w)),
        None => (None, 0, None),
    };
    let eta = match (n_remaining, genes_per_hour) {
        (Some(rem), Some(rate)) if rate > 0.0 && rem > 0 => {
            Some(Duration::from_secs_f64((rem as f64 / rate) * 3600.0))
        }
        _ => None,
    };

    let last_feather = newest_stamp(&all_feathers);
    let last_lock = newest_lock(&all_locks);
    let stalled = n_in_flight > 0
        && last_lock
            .as_ref()
            .map(|(_, age)| age.as_secs() >= STALL_SECS)
            .unwrap_or(false)
        && last_feather
            .as_ref()
            .map(|(_, age)| age.as_secs() >= STALL_SECS)
            .unwrap_or(true);

    let (trainers, worker_slots, n_unlabelled_locks) = group_trainers(&all_locks);
    let n_stale_locks = if repro.stale_lock_secs > 0 {
        all_locks
            .iter()
            .filter(|l| {
                now.duration_since(l.mtime)
                    .map(|d| d.as_secs() >= repro.stale_lock_secs)
                    .unwrap_or(false)
            })
            .count()
    } else {
        0
    };

    let cond_status = scan_condition_status(dir.to_str().unwrap_or("")).unwrap_or_default();
    let conditions: Vec<ConditionRow> = if pool_style {
        Vec::new()
    } else {
        cond_status
            .into_iter()
            .map(|c| {
                let planned_c = read_target_genes_file(&c.output_dir).map(|g| g.len());
                let n_done = c.n_done();
                ConditionRow {
                    label: c.label,
                    n_done,
                    n_feathers: c.n_feathers,
                    n_locks: c.n_locks,
                    planned: planned_c,
                }
            })
            .collect()
    };
    let samples: Vec<ConditionRow> = if pool_style {
        sample_scans
            .iter()
            .map(|(p, s)| ConditionRow {
                label: sample_display_label(p),
                n_done: terminal_genes(s).len(),
                n_feathers: s.feathers.len(),
                n_locks: s.locks.len(),
                planned,
            })
            .collect()
    } else {
        Vec::new()
    };

    let setup_ready = dir.join(SETUP_READY_FILENAME).is_file();
    let setup_flock = dir.join(SETUP_LOCK_FILENAME).is_file();
    let has_repro = dir.join(RUN_REPRO_TOML_FILENAME).is_file();

    Ok(RunStatus {
        dir,
        planned,
        n_done,
        n_in_flight,
        n_remaining,
        n_feathers: all_feathers.len(),
        n_orphans,
        n_tf_ablated,
        n_locks: all_locks.len(),
        n_stale_locks,
        n_logs: tree_logs,
        feather_bytes,
        dir_bytes,
        dir_files,
        last_feather,
        last_lock,
        genes_per_hour,
        throughput_n,
        throughput_window,
        eta,
        stalled,
        trainers,
        worker_slots,
        n_unlabelled_locks,
        setup_ready,
        setup_flock,
        has_repro,
        training_mode: repro.mode,
        epochs: repro.epochs,
        repro_n_parallel: repro.n_parallel,
        pool_lasso: repro.pool_lasso || pool_style,
        condition_col: repro.condition,
        sample_col: repro.sample,
        adata_path: repro.adata_path,
        stale_lock_secs: repro.stale_lock_secs,
        conditions,
        samples,
    })
}

fn format_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

fn format_age(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s ago")
    } else if s < 3600 {
        format!("{}m ago", s / 60)
    } else if s < 86400 {
        let h = s / 3600;
        let m = (s % 3600) / 60;
        if m == 0 {
            format!("{h}h ago")
        } else {
            format!("{h}h {m}m ago")
        }
    } else {
        let days = s / 86400;
        let h = (s % 86400) / 3600;
        if h == 0 {
            format!("{days}d ago")
        } else {
            format!("{days}d {h}h ago")
        }
    }
}

fn format_eta(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("~{s}s")
    } else if s < 3600 {
        format!("~{}m", (s + 30) / 60)
    } else if s < 86400 {
        let h = s / 3600;
        let m = (s % 3600) / 60;
        if m == 0 {
            format!("~{h}h")
        } else {
            format!("~{h}h {m}m")
        }
    } else {
        let days = s / 86400;
        let h = (s % 86400) / 3600;
        if h == 0 {
            format!("~{days}d")
        } else {
            format!("~{days}d {h}h")
        }
    }
}

fn progress_bar(done: usize, total: usize) -> String {
    if total == 0 {
        return String::new();
    }
    let frac = (done as f64 / total as f64).clamp(0.0, 1.0);
    let filled = (frac * BAR_WIDTH as f64).round() as usize;
    let filled = filled.min(BAR_WIDTH);
    format!("{}{}", "█".repeat(filled), "░".repeat(BAR_WIDTH - filled))
}

fn color_on() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn paint(s: &str, rgb: (u8, u8, u8), on: bool) -> String {
    if on {
        s.truecolor(rgb.0, rgb.1, rgb.2).to_string()
    } else {
        s.to_string()
    }
}

fn lbl(name: &str, on: bool) -> String {
    let pad = format!("{name:<LABEL_W$}");
    paint(&pad, (146, 131, 116), on)
}

pub fn format_run_status(st: &RunStatus, color: bool) -> String {
    let mut out = String::new();
    let path_s = st.dir.display().to_string();
    out.push_str(&format!(
        "{}  {}\n",
        paint("SpaceTravLR", (184, 187, 38), color),
        paint(&path_s, (250, 189, 47), color)
    ));

    let progress_val = if let Some(total) = st.planned {
        let pct = if total == 0 {
            0.0
        } else {
            100.0 * st.n_done as f64 / total as f64
        };
        format!(
            "{}/{}  ({:.1}%)  {}",
            st.n_done,
            total,
            pct,
            progress_bar(st.n_done, total)
        )
    } else {
        format!("{} done", st.n_done)
    };
    out.push_str(&format!(
        "{}  {}\n",
        lbl("Progress", color),
        paint(&progress_val, (142, 192, 124), color)
    ));
    out.push_str(&format!(
        "{}  {} genes\n",
        lbl("In flight", color),
        st.n_in_flight
    ));
    match st.n_remaining {
        Some(r) => out.push_str(&format!("{}  {}\n", lbl("Remaining", color), r)),
        None => out.push_str(&format!(
            "{}  unknown (no {} or gene list in repro)\n",
            lbl("Remaining", color),
            TARGET_GENES_FILENAME
        )),
    }
    out.push('\n');

    match (st.genes_per_hour, st.throughput_window, st.throughput_n) {
        (Some(rate), Some(win), n) if n > 0 => {
            out.push_str(&format!(
                "{}  {:.1} genes/h  ({} feathers in {})\n",
                lbl("Throughput", color),
                rate,
                n,
                format_eta(win).trim_start_matches('~')
            ));
        }
        _ => {
            out.push_str(&format!("{}  —\n", lbl("Throughput", color)));
        }
    }
    match st.eta {
        Some(d) => out.push_str(&format!("{}  {}\n", lbl("ETA", color), format_eta(d))),
        None => out.push_str(&format!("{}  —\n", lbl("ETA", color))),
    }
    if st.stalled {
        out.push_str(&format!(
            "{}  {}\n",
            lbl("Note", color),
            paint(
                "locks present but no recent feather/lock activity (≥15m)",
                (251, 73, 52),
                color
            )
        ));
    }
    out.push('\n');

    match &st.last_feather {
        Some((name, age)) => out.push_str(&format!(
            "{}  {}    {}\n",
            lbl("Last feather", color),
            name,
            format_age(*age)
        )),
        None => out.push_str(&format!("{}  —\n", lbl("Last feather", color))),
    }
    match &st.last_lock {
        Some((name, age)) => out.push_str(&format!(
            "{}  {}    {}\n",
            lbl("Last lock", color),
            name,
            format_age(*age)
        )),
        None => out.push_str(&format!("{}  —\n", lbl("Last lock", color))),
    }
    out.push('\n');

    let trainer_n = st.trainers.len();
    let slots = if st.worker_slots == 0 && st.n_locks > 0 {
        st.n_locks
    } else {
        st.worker_slots
    };
    out.push_str(&format!(
        "{}  {} process{}  ·  {} worker slots\n",
        lbl("Trainers", color),
        trainer_n.max(if st.n_unlabelled_locks > 0 && trainer_n == 0 {
            0
        } else {
            trainer_n
        }),
        if trainer_n == 1 { "" } else { "es" },
        slots
    ));
    if st.n_unlabelled_locks > 0 {
        out.push_str(&format!(
            "               {} unlabelled lock{} (empty or legacy .lock files)\n",
            st.n_unlabelled_locks,
            if st.n_unlabelled_locks == 1 { "" } else { "s" }
        ));
    }
    for t in &st.trainers {
        out.push_str(&format!(
            "               {}  pid {}  n_parallel={}  {} lock{}\n",
            t.host,
            t.pid,
            t.n_parallel,
            t.n_locks,
            if t.n_locks == 1 { "" } else { "s" }
        ));
    }
    if st.n_stale_locks > 0 {
        out.push_str(&format!(
            "{}  {} lock{} older than stale_lock_secs={}\n",
            lbl("Stale locks", color),
            st.n_stale_locks,
            if st.n_stale_locks == 1 { "" } else { "s" },
            st.stale_lock_secs
        ));
    }
    out.push('\n');

    let mean_feather = if st.n_feathers > 0 {
        format!(
            "  (mean {})",
            format_bytes(st.feather_bytes / st.n_feathers as u64)
        )
    } else {
        String::new()
    };
    out.push_str(&format!(
        "{}  {} files   {}{}\n",
        lbl("Feathers", color),
        st.n_feathers,
        format_bytes(st.feather_bytes),
        mean_feather
    ));
    out.push_str(&format!(
        "{}  {}     {} files\n",
        lbl("Directory", color),
        format_bytes(st.dir_bytes),
        st.dir_files
    ));
    if st.n_orphans > 0 || st.n_tf_ablated > 0 || st.n_logs > 0 {
        out.push_str(&format!(
            "{}  orphans {}  ·  tf_ablated {}  ·  logs {}\n",
            lbl("Other", color),
            st.n_orphans,
            st.n_tf_ablated,
            st.n_logs
        ));
    }

    if st.setup_flock && !st.setup_ready {
        out.push_str(&format!(
            "\n{}  {}\n",
            lbl("Setup", color),
            paint(
                "still preparing (spacetravlr_setup.flock present)",
                (250, 189, 47),
                color
            )
        ));
    } else if st.setup_ready {
        out.push_str(&format!("{}  ready\n", lbl("Setup", color)));
    }

    let mut run_bits = Vec::new();
    if let Some(m) = &st.training_mode {
        run_bits.push(format!("mode {m}"));
    }
    if let Some(e) = st.epochs {
        run_bits.push(format!("{e} epochs"));
    }
    if let Some(n) = st.repro_n_parallel {
        run_bits.push(format!("n_parallel {n}"));
    }
    if st.pool_lasso {
        run_bits.push("pool-lasso".into());
    }
    if let Some(c) = &st.condition_col {
        run_bits.push(format!("condition {c}"));
    }
    if let Some(s) = &st.sample_col {
        run_bits.push(format!("sample {s}"));
    }
    if !run_bits.is_empty() {
        out.push_str(&format!(
            "{}  {}\n",
            lbl("Run", color),
            run_bits.join("  ·  ")
        ));
    }
    if let Some(p) = &st.adata_path {
        out.push_str(&format!("{}  {}\n", lbl("AnnData", color), p));
    }

    if !st.samples.is_empty() {
        out.push_str(&format!(
            "\n{}  {} slide{}\n",
            lbl("Samples", color),
            st.samples.len(),
            if st.samples.len() == 1 { "" } else { "s" }
        ));
        for s in &st.samples {
            let prog = match s.planned {
                Some(p) => format!("{}/{}", s.n_done, p),
                None => format!("{} genes", s.n_done),
            };
            out.push_str(&format!(
                "               {:<24}  {prog} genes with artifacts  {} feathers\n",
                s.label, s.n_feathers
            ));
        }
    } else if !st.conditions.is_empty() {
        out.push_str(&format!(
            "\n{}  {} groups\n",
            lbl("Conditions", color),
            st.conditions.len()
        ));
        for c in &st.conditions {
            let prog = match c.planned {
                Some(p) => format!("{}/{}", c.n_done, p),
                None => format!("{} done", c.n_done),
            };
            out.push_str(&format!(
                "               {:<24}  {prog}  {} feathers  {} locks\n",
                c.label, c.n_feathers, c.n_locks
            ));
        }
    }

    if !st.has_repro && st.n_feathers == 0 && st.n_locks == 0 {
        out.push_str(&format!(
            "\n{}  no repro TOML, feathers, or locks — this may not be a training output directory\n",
            lbl("Warning", color)
        ));
    }

    out
}

pub fn print_run_status(path: &Path) -> anyhow::Result<()> {
    let dir = resolve_status_dir(path).with_context(|| format!("resolve {}", path.display()))?;
    let st = collect_run_status(&dir)?;
    print!("{}", format_run_status(&st, color_on()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, SystemTime};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn tmp() -> PathBuf {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let d =
            std::env::temp_dir().join(format!("spacetravlr_run_status_{}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn cleanup(d: &Path) {
        let _ = fs::remove_dir_all(d);
    }

    #[test]
    fn parse_lock_payload_fields() {
        let p = parse_gene_lock_payload(
            "spacetravlr_gene 1\nhost=node01\npid=4321\nn_parallel=8\nstarted_unix=10\n",
        );
        assert_eq!(p.host.as_deref(), Some("node01"));
        assert_eq!(p.pid, Some(4321));
        assert_eq!(p.n_parallel, Some(8));
        assert_eq!(p.started_unix, Some(10));
    }

    #[test]
    fn parse_empty_lock() {
        let p = parse_gene_lock_payload("");
        assert!(p.host.is_none());
        assert!(p.pid.is_none());
        assert!(p.n_parallel.is_none());
    }

    #[test]
    fn target_genes_write_once() {
        let d = tmp();
        let genes = vec!["A".into(), "B".into(), "C".into()];
        assert!(write_target_genes_if_missing(&d, &genes).unwrap());
        assert!(!write_target_genes_if_missing(&d, &["Z".into()]).unwrap());
        let read = read_target_genes_file(&d).unwrap();
        assert_eq!(read, genes);
        cleanup(&d);
    }

    #[test]
    fn status_counts_feathers_locks_orphans() {
        let d = tmp();
        fs::write(
            d.join("spacetravlr_target_genes.txt"),
            "SOX2\nACTB\nMALAT1\nGAPDH\n",
        )
        .unwrap();
        fs::write(d.join("SOX2_betadata.feather"), vec![0u8; 1000]).unwrap();
        fs::write(d.join("ACTB_betadata.feather"), vec![0u8; 2000]).unwrap();
        fs::write(d.join("GAPDH.orphan"), b"").unwrap();
        fs::write(
            d.join("MALAT1.lock"),
            "spacetravlr_gene 1\nhost=hostA\npid=99\nn_parallel=4\nstarted_unix=1\n",
        )
        .unwrap();
        fs::write(d.join("LEGACY.lock"), b"").unwrap();

        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.planned, Some(4));
        assert_eq!(st.n_done, 3);
        assert_eq!(st.n_remaining, Some(1));
        assert_eq!(st.n_in_flight, 2);
        assert_eq!(st.n_feathers, 2);
        assert_eq!(st.n_orphans, 1);
        assert_eq!(st.feather_bytes, 3000);
        assert_eq!(st.trainers.len(), 1);
        assert_eq!(st.trainers[0].host, "hostA");
        assert_eq!(st.trainers[0].n_parallel, 4);
        assert_eq!(st.n_unlabelled_locks, 1);
        assert!(st.dir_bytes >= 3000);
        let text = format_run_status(&st, false);
        assert!(text.contains("3/4"));
        assert!(text.contains("hostA"));
        assert!(text.contains("n_parallel=4"));
        cleanup(&d);
    }

    #[test]
    fn status_nested_conditions() {
        let d = tmp();
        fs::write(
            d.join("spacetravlr_run_repro.toml"),
            "[training]\ngenes = [\"G1\", \"G2\"]\nmode = \"full\"\nepochs = 12\n\n[data]\ncondition = \"batch\"\n",
        )
        .unwrap();
        let a = d.join("conditions/groupA");
        let b = d.join("conditions/groupB");
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        fs::write(a.join("condition_label.txt"), "A").unwrap();
        fs::write(b.join("condition_label.txt"), "B").unwrap();
        fs::write(a.join("G1_betadata.feather"), b"xx").unwrap();
        fs::write(
            a.join("G2.lock"),
            b"spacetravlr_gene 1\nhost=h\npid=1\nn_parallel=2\n",
        )
        .unwrap();
        fs::write(b.join("G1_betadata.feather"), b"yy").unwrap();
        fs::write(a.join("spacetravlr_target_genes.txt"), "G1\nG2\n").unwrap();
        fs::write(b.join("spacetravlr_target_genes.txt"), "G1\nG2\n").unwrap();

        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.n_feathers, 2);
        assert_eq!(st.n_locks, 1);
        assert_eq!(st.conditions.len(), 2);
        assert_eq!(st.n_done, 2);
        assert_eq!(st.planned, Some(4));
        assert_eq!(st.training_mode.as_deref(), Some("full"));
        assert_eq!(st.epochs, Some(12));
        cleanup(&d);
    }

    #[test]
    fn throughput_from_spaced_mtimes() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(10_000);
        let times = vec![
            now - Duration::from_secs(1800),
            now - Duration::from_secs(900),
            now - Duration::from_secs(1),
        ];
        let (rate, n, win) = feather_throughput(&times, now).unwrap();
        assert_eq!(n, 3);
        assert!(win.as_secs() >= 1700);
        assert!(rate > 5.0 && rate < 10.0);
    }

    #[test]
    fn throughput_none_for_single() {
        let now = SystemTime::now();
        assert!(feather_throughput(&[now], now).is_none());
    }

    #[test]
    fn resolve_repro_toml_parent() {
        let d = tmp();
        let repro = d.join(RUN_REPRO_TOML_FILENAME);
        fs::write(&repro, "[training]\n").unwrap();
        let got = resolve_status_dir(&repro).unwrap();
        assert_eq!(got, d);
        cleanup(&d);
    }

    #[test]
    fn missing_planned_list() {
        let d = tmp();
        fs::write(d.join("SOX2_betadata.feather"), b"abc").unwrap();
        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.planned, None);
        assert_eq!(st.n_done, 1);
        assert!(st.n_remaining.is_none());
        let text = format_run_status(&st, false);
        assert!(text.contains("unknown"));
        cleanup(&d);
    }

    fn lock_body(host: &str, pid: u32, n_parallel: usize) -> String {
        format!(
            "spacetravlr_gene 1\nhost={host}\npid={pid}\nn_parallel={n_parallel}\nstarted_unix=1\n"
        )
    }

    fn set_mtime(path: &Path, age: Duration) {
        let f = fs::File::options().write(true).open(path).unwrap();
        f.set_modified(SystemTime::now() - age).unwrap();
    }

    #[test]
    fn target_genes_skips_comments_and_blanks() {
        let d = tmp();
        fs::write(
            d.join(TARGET_GENES_FILENAME),
            "# header\nSOX2\n\n  ACTB  \n# skip\n",
        )
        .unwrap();
        assert_eq!(
            read_target_genes_file(&d).unwrap(),
            vec!["SOX2".to_string(), "ACTB".to_string()]
        );
        cleanup(&d);
    }

    #[test]
    fn planned_from_repro_max_genes() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[training]\nmax_genes = 10\nmode = \"seed\"\n",
        )
        .unwrap();
        fs::write(d.join("G1_betadata.feather"), b"x").unwrap();
        fs::write(d.join("G2_betadata.feather"), b"y").unwrap();
        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.planned, Some(10));
        assert_eq!(st.n_done, 2);
        assert_eq!(st.n_remaining, Some(8));
        assert_eq!(st.training_mode.as_deref(), Some("seed"));
        cleanup(&d);
    }

    #[test]
    fn planned_genes_capped_by_max_genes() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[training]\ngenes = [\"A\", \"B\", \"C\", \"D\"]\nmax_genes = 2\n",
        )
        .unwrap();
        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.planned, Some(2));
        cleanup(&d);
    }

    #[test]
    fn complete_run_zero_in_flight() {
        let d = tmp();
        fs::write(d.join(TARGET_GENES_FILENAME), "A\nB\n").unwrap();
        fs::write(d.join("A_betadata.feather"), b"aa").unwrap();
        fs::write(d.join("B.tf_ablated"), b"").unwrap();
        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.n_done, 2);
        assert_eq!(st.n_in_flight, 0);
        assert_eq!(st.n_remaining, Some(0));
        assert_eq!(st.n_tf_ablated, 1);
        assert!(!st.stalled);
        let text = format_run_status(&st, false);
        assert!(text.contains("2/2"));
        cleanup(&d);
    }

    #[test]
    fn remaining_saturates_when_done_exceeds_planned() {
        let d = tmp();
        fs::write(d.join(TARGET_GENES_FILENAME), "A\n").unwrap();
        fs::write(d.join("A_betadata.feather"), b"a").unwrap();
        fs::write(d.join("B_betadata.feather"), b"b").unwrap();
        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.planned, Some(1));
        assert_eq!(st.n_done, 2);
        assert_eq!(st.n_remaining, Some(0));
        cleanup(&d);
    }

    #[test]
    fn two_trainers_sum_worker_slots() {
        let d = tmp();
        fs::write(d.join("G1.lock"), lock_body("nodeA", 11, 8)).unwrap();
        fs::write(d.join("G2.lock"), lock_body("nodeA", 11, 8)).unwrap();
        fs::write(d.join("G3.lock"), lock_body("nodeB", 22, 4)).unwrap();
        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.trainers.len(), 2);
        assert_eq!(st.worker_slots, 12);
        assert_eq!(st.n_locks, 3);
        assert_eq!(st.n_in_flight, 3);
        assert_eq!(st.n_unlabelled_locks, 0);
        let text = format_run_status(&st, false);
        assert!(text.contains("nodeA"));
        assert!(text.contains("nodeB"));
        assert!(text.contains("12 worker slots"));
        cleanup(&d);
    }

    #[test]
    fn pool_lasso_done_marker_not_double_counted() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[training]\npool_lasso = true\ngenes = [\"G1\", \"G2\"]\n",
        )
        .unwrap();
        fs::write(d.join(TARGET_GENES_FILENAME), "G1\nG2\n").unwrap();
        fs::write(d.join("G1.done"), b"").unwrap();
        fs::write(d.join("G2.lock"), lock_body("poolhost", 7, 2)).unwrap();
        let s1 = d.join("conditions/s1");
        let s2 = d.join("conditions/s2");
        fs::create_dir_all(&s1).unwrap();
        fs::create_dir_all(&s2).unwrap();
        fs::write(s1.join("condition_label.txt"), "s1").unwrap();
        fs::write(s2.join("condition_label.txt"), "s2").unwrap();
        fs::write(s1.join("G1_betadata.feather"), b"1111").unwrap();
        fs::write(s2.join("G1_betadata.feather"), b"2222").unwrap();

        let st = collect_run_status(&d).unwrap();
        assert!(st.pool_lasso);
        assert_eq!(st.planned, Some(2));
        assert_eq!(
            st.n_done, 1,
            "parent .done is one gene, not per-sample feathers"
        );
        assert_eq!(st.n_feathers, 2);
        assert_eq!(st.n_in_flight, 1);
        assert_eq!(st.n_remaining, Some(1));
        let text = format_run_status(&st, false);
        assert!(text.contains("Samples"));
        assert!(text.contains("s1"));
        assert!(text.contains("s2"));
        assert!(!text.contains("Conditions"));
        cleanup(&d);
    }

    #[test]
    fn pool_lasso_partial_sample_is_not_done() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[training]\npool_lasso = true\ngenes = [\"G1\", \"G2\"]\n\n[data]\nsample = \"slide\"\n",
        )
        .unwrap();
        fs::write(d.join(TARGET_GENES_FILENAME), "G1\nG2\n").unwrap();
        let s1 = d.join("conditions/s1");
        let s2 = d.join("conditions/s2");
        fs::create_dir_all(&s1).unwrap();
        fs::create_dir_all(&s2).unwrap();
        fs::write(s1.join("condition_label.txt"), "s1").unwrap();
        fs::write(s2.join("condition_label.txt"), "s2").unwrap();
        fs::write(s1.join("G1_betadata.feather"), b"only-one-sample").unwrap();
        fs::write(d.join("G1.lock"), lock_body("poolhost", 1, 2)).unwrap();

        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.n_done, 0, "one sample feather is not a finished gene");
        assert_eq!(st.n_in_flight, 1);
        assert_eq!(st.n_feathers, 1);
        assert_eq!(st.n_remaining, Some(2));
        assert_eq!(st.sample_col.as_deref(), Some("slide"));
        cleanup(&d);
    }

    #[test]
    fn pool_lasso_complete_when_every_sample_has_artifact() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[training]\npool_lasso = true\ngenes = [\"G1\"]\n",
        )
        .unwrap();
        let s1 = d.join("conditions/s1");
        let s2 = d.join("conditions/s2");
        fs::create_dir_all(&s1).unwrap();
        fs::create_dir_all(&s2).unwrap();
        fs::write(s1.join("condition_label.txt"), "s1").unwrap();
        fs::write(s2.join("condition_label.txt"), "s2").unwrap();
        fs::write(s1.join("G1_betadata.feather"), b"a").unwrap();
        fs::write(s2.join("G1.orphan"), b"").unwrap();

        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.n_done, 1);
        assert_eq!(st.n_in_flight, 0);
        assert_eq!(st.n_feathers, 1);
        cleanup(&d);
    }

    #[test]
    fn pool_lasso_nested_condition_samples_layout() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[training]\npool_lasso = true\ngenes = [\"G1\", \"G2\"]\n\n[data]\ncondition = \"batch\"\nsample = \"slide\"\n",
        )
        .unwrap();
        fs::write(d.join(TARGET_GENES_FILENAME), "G1\nG2\n").unwrap();
        let a1 = d.join("conditions/tumor/samples/s1");
        let a2 = d.join("conditions/tumor/samples/s2");
        fs::create_dir_all(&a1).unwrap();
        fs::create_dir_all(&a2).unwrap();
        fs::write(d.join("conditions/tumor/condition_label.txt"), "tumor").unwrap();
        fs::write(a1.join("sample_label.txt"), "s1").unwrap();
        fs::write(a2.join("sample_label.txt"), "s2").unwrap();
        fs::write(a1.join("G1_betadata.feather"), b"x").unwrap();
        fs::write(a2.join("G1_betadata.feather"), b"y").unwrap();
        fs::write(d.join("G1.done"), b"").unwrap();
        fs::write(d.join("G2.lock"), lock_body("h", 9, 4)).unwrap();

        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.n_done, 1);
        assert_eq!(st.n_in_flight, 1);
        assert_eq!(st.samples.len(), 2);
        assert!(st.samples.iter().any(|s| s.label.contains("tumor/s1")));
        assert_eq!(st.n_remaining, Some(1));
        let text = format_run_status(&st, false);
        assert!(text.contains("Samples"));
        assert!(text.contains("sample slide"));
        assert!(text.contains("condition batch"));
        cleanup(&d);
    }

    #[test]
    fn pool_lasso_walks_up_from_sample_dir() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[training]\npool_lasso = true\ngenes = [\"G1\"]\n",
        )
        .unwrap();
        fs::write(d.join(TARGET_GENES_FILENAME), "G1\n").unwrap();
        let s1 = d.join("conditions/s1");
        fs::create_dir_all(&s1).unwrap();
        fs::write(s1.join("condition_label.txt"), "s1").unwrap();
        fs::write(s1.join("G1_betadata.feather"), b"x").unwrap();
        fs::write(d.join("G1.done"), b"").unwrap();

        let got = resolve_status_dir(&s1).unwrap();
        assert_eq!(got, d);
        let st = collect_run_status(&got).unwrap();
        assert_eq!(st.n_done, 1);
        cleanup(&d);
    }

    #[test]
    fn pool_lasso_throughput_uses_gene_completions() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[training]\npool_lasso = true\ngenes = [\"G1\", \"G2\"]\n",
        )
        .unwrap();
        let s1 = d.join("conditions/s1");
        let s2 = d.join("conditions/s2");
        fs::create_dir_all(&s1).unwrap();
        fs::create_dir_all(&s2).unwrap();
        fs::write(s1.join("condition_label.txt"), "s1").unwrap();
        fs::write(s2.join("condition_label.txt"), "s2").unwrap();
        fs::write(s1.join("G1_betadata.feather"), b"a").unwrap();
        fs::write(s2.join("G1_betadata.feather"), b"b").unwrap();
        fs::write(s1.join("G2_betadata.feather"), b"c").unwrap();
        fs::write(s2.join("G2_betadata.feather"), b"d").unwrap();
        let d1 = d.join("G1.done");
        let d2 = d.join("G2.done");
        fs::write(&d1, b"").unwrap();
        fs::write(&d2, b"").unwrap();
        set_mtime(&d1, Duration::from_secs(1200));
        set_mtime(&d2, Duration::from_secs(1));

        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.n_done, 2);
        assert_eq!(st.n_feathers, 4);
        assert_eq!(
            st.throughput_n, 2,
            "rate from 2 gene .done files, not 4 feathers"
        );
        cleanup(&d);
    }

    #[test]
    fn setup_flock_without_ready() {
        let d = tmp();
        fs::write(d.join("spacetravlr_setup.flock"), b"spacetravlr_setup 1\n").unwrap();
        let st = collect_run_status(&d).unwrap();
        assert!(st.setup_flock);
        assert!(!st.setup_ready);
        let text = format_run_status(&st, false);
        assert!(text.contains("still preparing"));
        cleanup(&d);
    }

    #[test]
    fn setup_ready_and_repro_annodata() {
        let d = tmp();
        fs::write(d.join("spacetravlr_setup.ready"), b"ok\n").unwrap();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[data]\nadata_path = \"/data/slide.h5ad\"\n\n[execution]\nn_parallel = 16\n\n[training]\nmode = \"full\"\nepochs = 40\n",
        )
        .unwrap();
        let st = collect_run_status(&d).unwrap();
        assert!(st.setup_ready);
        assert!(!st.setup_flock);
        assert_eq!(st.repro_n_parallel, Some(16));
        assert_eq!(st.adata_path.as_deref(), Some("/data/slide.h5ad"));
        let text = format_run_status(&st, false);
        assert!(text.contains("Setup"));
        assert!(text.contains("ready"));
        assert!(text.contains("/data/slide.h5ad"));
        assert!(text.contains("n_parallel 16"));
        cleanup(&d);
    }

    #[test]
    fn empty_dir_warns() {
        let d = tmp();
        let st = collect_run_status(&d).unwrap();
        let text = format_run_status(&st, false);
        assert!(text.contains("may not be a training output directory"));
        cleanup(&d);
    }

    #[test]
    fn resolve_status_rejects_plain_file() {
        let d = tmp();
        let f = d.join("notes.txt");
        fs::write(&f, "hi").unwrap();
        let err = resolve_status_dir(&f).unwrap_err().to_string();
        assert!(err.contains("expected a training output directory"));
        cleanup(&d);
    }

    #[test]
    fn stale_locks_flagged_from_repro() {
        let d = tmp();
        fs::write(
            d.join(RUN_REPRO_TOML_FILENAME),
            "[execution]\nstale_lock_secs = 60\n",
        )
        .unwrap();
        let lock = d.join("OLD.lock");
        fs::write(&lock, lock_body("h", 1, 1)).unwrap();
        set_mtime(&lock, Duration::from_secs(120));
        let fresh = d.join("NEW.lock");
        fs::write(&fresh, lock_body("h", 1, 1)).unwrap();
        let st = collect_run_status(&d).unwrap();
        assert_eq!(st.stale_lock_secs, 60);
        assert_eq!(st.n_stale_locks, 1);
        let text = format_run_status(&st, false);
        assert!(text.contains("Stale locks"));
        cleanup(&d);
    }

    #[test]
    fn stalled_when_lock_and_feather_are_old() {
        let d = tmp();
        fs::write(d.join(TARGET_GENES_FILENAME), "A\nB\n").unwrap();
        let feather = d.join("A_betadata.feather");
        fs::write(&feather, b"aa").unwrap();
        let lock = d.join("B.lock");
        fs::write(&lock, lock_body("h", 3, 1)).unwrap();
        set_mtime(&feather, Duration::from_secs(20 * 60));
        set_mtime(&lock, Duration::from_secs(20 * 60));
        let st = collect_run_status(&d).unwrap();
        assert!(st.stalled);
        assert_eq!(st.n_in_flight, 1);
        let text = format_run_status(&st, false);
        assert!(text.contains("no recent feather/lock activity"));
        cleanup(&d);
    }

    #[test]
    fn not_stalled_when_lock_is_fresh() {
        let d = tmp();
        fs::write(d.join("A_betadata.feather"), b"aa").unwrap();
        fs::write(d.join("B.lock"), lock_body("h", 3, 1)).unwrap();
        let st = collect_run_status(&d).unwrap();
        assert!(!st.stalled);
        cleanup(&d);
    }

    #[test]
    fn eta_from_remaining_and_throughput() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(10_000);
        let times = vec![
            now - Duration::from_secs(1800),
            now - Duration::from_secs(900),
            now,
        ];
        let (rate, n, win) = feather_throughput(&times, now).unwrap();
        assert_eq!(n, 3);
        assert_eq!(win, Duration::from_secs(1800));
        assert!((rate - 6.0).abs() < 0.01);
        let eta = Duration::from_secs_f64((9.0 / rate) * 3600.0);
        assert_eq!(format_eta(eta), "~1h 30m");
    }

    #[test]
    fn progress_bar_full_and_empty() {
        assert_eq!(progress_bar(0, 10).chars().filter(|c| *c == '█').count(), 0);
        assert_eq!(
            progress_bar(10, 10).chars().filter(|c| *c == '█').count(),
            20
        );
        assert!(progress_bar(1, 0).is_empty());
    }
}
